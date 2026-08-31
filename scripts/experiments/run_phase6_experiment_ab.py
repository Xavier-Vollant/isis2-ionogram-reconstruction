#!/usr/bin/env python3
"""Run an inference-only A/B test for marker leakage and contrast handling.

The production checkpoint is read-only. Results are written to a separate
experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram
from isis_research.models import image_features
from scripts.evaluation.benchmark_phase6_512_image_baselines import metric
from scripts.evaluation.evaluate_phase6_512_image_model import load_model
from scripts.training.train_phase6_512_image_model import load_sample, rows_for

DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_GROUPS = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/evaluation/phase6_continual_models/hybrid_unet/best_model.pt"
)
DEFAULT_BENCHMARK = ROOT / "outputs/evaluation/phase6_all_models_report/report.json"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_experiments"
MARKER_STRENGTH = 0.75


def read_group_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["pair_name"]: row for row in csv.DictReader(handle)}


def load_experiment_sample(corpus, targets, row, target_rows):
    signal, target, loss_mask = load_sample(corpus, targets, row, target_rows)
    scan = ionogram.read_validated(corpus / row["csa_artifact"])
    return signal, target, loss_mask, np.asarray(scan.valid_mask, dtype=bool)


def suppress_markers(
    signal: np.ndarray, valid: np.ndarray, strength: float
) -> np.ndarray:
    """Remove persistent column offsets while preserving the 2-D trace shape."""
    values = np.where(valid, signal, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        global_median = float(np.nanmedian(values))
        column_medians = np.nanmedian(values, axis=0)
    column_medians = np.nan_to_num(column_medians, nan=global_median)
    offsets = column_medians - global_median
    result = signal - float(strength) * offsets[None, :]
    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    result[~valid] = 0.0
    return result


def predict(model, signal: np.ndarray, input_channels: int, torch) -> np.ndarray:
    with torch.inference_mode():
        output = torch.sigmoid(
            model(torch.from_numpy(image_features(signal, input_channels)[None, ...]))
        )
    return output.cpu().numpy()[0].astype(np.float32)


def update_stats(stats, values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    stats["sum"] += float(values.sum())
    stats["sum_sq"] += float(np.square(values).sum())
    stats["count"] += int(values.size)


def finish_stats(stats):
    mean = stats["sum"] / max(stats["count"], 1)
    variance = stats["sum_sq"] / max(stats["count"], 1) - mean * mean
    return float(mean), float(np.sqrt(max(variance, 0.0)))


def fit_calibration(
    model, rows, corpus, targets, target_rows, input_channels, torch, strength
):
    """Fit prediction -> NASA affine calibration on training-only scans."""
    raw = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    suppressed = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    target = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    for row in rows:
        signal, target_image, mask, input_valid = load_experiment_sample(
            corpus, targets, row, target_rows
        )
        clean_signal = suppress_markers(signal, input_valid, strength)
        raw_prediction = predict(model, signal, input_channels, torch)
        clean_prediction = predict(model, clean_signal, input_channels, torch)
        update_stats(raw, raw_prediction[mask])
        update_stats(suppressed, clean_prediction[mask])
        update_stats(target, target_image[mask])

    target_mean, target_std = finish_stats(target)
    result = {"target_mean": target_mean, "target_std": target_std, "scans": len(rows)}
    for name, stats in (("raw", raw), ("marker_suppressed", suppressed)):
        prediction_mean, prediction_std = finish_stats(stats)
        scale = target_std / max(prediction_std, 1e-9)
        result[name] = {
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "scale": float(scale),
            "bias": float(target_mean - scale * prediction_mean),
        }
    return result


def calibrate(prediction: np.ndarray, parameters: dict) -> np.ndarray:
    return np.clip(
        float(parameters["scale"]) * prediction + float(parameters["bias"]), 0.0, 1.0
    ).astype(np.float32)


def record_metrics(prediction, target, mask):
    result = metric(prediction, target, mask)
    result["prediction_mean"] = float(np.mean(prediction[mask]))
    result["prediction_std"] = float(np.std(prediction[mask]))
    result["target_mean"] = float(np.mean(target[mask]))
    result["target_std"] = float(np.std(target[mask]))
    result["prediction_std_ratio"] = float(
        result["prediction_std"] / max(result["target_std"], 1e-9)
    )
    result["bias"] = float(np.mean(prediction[mask] - target[mask]))
    return result


def summarize(records):
    valid = [item for item in records if item["mae"] is not None]
    if not valid:
        return {"scans": 0}
    return {
        "scans": len(valid),
        "macro_mae": float(np.mean([item["mae"] for item in valid])),
        "macro_rmse": float(np.mean([item["rmse"] for item in valid])),
        "macro_correlation": float(np.mean([item["correlation"] for item in valid])),
        "median_mae": float(np.median([item["mae"] for item in valid])),
        "p90_mae": float(np.percentile([item["mae"] for item in valid], 90)),
        "mean_prediction_std_ratio": float(
            np.mean([item["prediction_std_ratio"] for item in valid])
        ),
        "mean_bias": float(np.mean([item["bias"] for item in valid])),
        "mae_over_0_20": int(sum(item["mae"] > 0.20 for item in valid)),
        "correlation_below_0_20": int(
            sum(item["correlation"] < 0.20 for item in valid)
        ),
    }


def stratified(records, key):
    grouped = {}
    for record in records:
        grouped.setdefault(record[key], []).append(record["metrics"])
    return {name: summarize(items) for name, items in sorted(grouped.items())}


def worst(records, count=10):
    return [
        {
            "pair_name": item["pair_name"],
            "reel": item["reel"],
            "station": item["station"],
            "metrics": item["metrics"],
        }
        for item in sorted(
            records, key=lambda value: value["metrics"]["mae"], reverse=True
        )[:count]
    ]


def main():
    """Parse CLI options and run the inference-only A/B comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--marker-strength", type=float, default=MARKER_STRENGTH)
    args = parser.parse_args()
    if not 0.0 <= args.marker_strength <= 1.0:
        raise SystemExit("--marker-strength must be between 0 and 1")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"output is not empty: {args.output}; remove it to rerun this isolated experiment"
        )

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus_rows = rows_for(args.corpus)
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    groups = read_group_rows(args.groups)
    train_rows = [row for row in corpus_rows if row["split"] == "train"]
    held_out_by_name = {
        row["pair_name"]: row for row in corpus_rows if row["split"] == "held_out"
    }
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    selected_names = benchmark["model_benchmark"]["pair_names"]
    held_out = [held_out_by_name[name] for name in selected_names]
    train_reels = {groups[row["pair_name"]]["reel"] for row in train_rows}
    held_out_reels = {groups[row["pair_name"]]["reel"] for row in held_out}
    if train_reels & held_out_reels:
        raise SystemExit("training and held-out reels overlap")

    calibration_rows = {}
    for row in sorted(train_rows, key=lambda value: value["pair_name"]):
        calibration_rows.setdefault(groups[row["pair_name"]]["reel"], row)
    calibration_rows = list(calibration_rows.values())
    model, checkpoint = load_model(args.checkpoint, torch)
    input_channels = int(checkpoint.get("input_channels", 1))
    model.eval()
    print(f"fitting calibration on {len(calibration_rows)} training reels", flush=True)
    calibration = fit_calibration(
        model,
        calibration_rows,
        args.corpus,
        args.targets,
        target_rows,
        input_channels,
        torch,
        args.marker_strength,
    )

    variant_names = (
        "baseline_hybrid",
        "marker_suppressed_input",
        "contrast_adjusted",
        "marker_suppressed_contrast",
    )
    records = {name: [] for name in variant_names}
    per_pair = {}
    for index, row in enumerate(held_out, 1):
        signal, target, mask, input_valid = load_experiment_sample(
            args.corpus, args.targets, row, target_rows
        )
        clean_signal = suppress_markers(signal, input_valid, args.marker_strength)
        baseline = predict(model, signal, input_channels, torch)
        clean = predict(model, clean_signal, input_channels, torch)
        candidates = {
            "baseline_hybrid": baseline,
            "marker_suppressed_input": clean,
            "contrast_adjusted": calibrate(baseline, calibration["raw"]),
            "marker_suppressed_contrast": calibrate(
                clean, calibration["marker_suppressed"]
            ),
        }
        metadata = {
            "pair_name": row["pair_name"],
            "reel": groups[row["pair_name"]]["reel"],
            "station": row.get("station") or "<blank>",
            "target_coverage": float(mask.mean()),
        }
        per_pair[row["pair_name"]] = {
            "reel": metadata["reel"],
            "station": metadata["station"],
            "variants": {},
        }
        for name, prediction in candidates.items():
            item = {**metadata, "metrics": record_metrics(prediction, target, mask)}
            records[name].append(item)
            per_pair[row["pair_name"]]["variants"][name] = item["metrics"]
        if index == 1 or index % 16 == 0 or index == len(held_out):
            print(f"evaluated {index}/{len(held_out)}", flush=True)

    summaries = {
        name: summarize([item["metrics"] for item in items])
        for name, items in records.items()
    }
    baseline_mae = np.asarray(
        [item["metrics"]["mae"] for item in records["baseline_hybrid"]]
    )
    comparison = {}
    for name in variant_names[1:]:
        values = np.asarray([item["metrics"]["mae"] for item in records[name]])
        comparison[name] = {
            "delta_macro_mae": float(values.mean() - baseline_mae.mean()),
            "mae_improvement": float(np.mean(baseline_mae - values)),
            "mae_win_rate": float(np.mean(values < baseline_mae)),
            "delta_macro_correlation": float(
                np.mean([item["metrics"]["correlation"] for item in records[name]])
                - np.mean(
                    [
                        item["metrics"]["correlation"]
                        for item in records["baseline_hybrid"]
                    ]
                )
            ),
            "delta_mean_prediction_std_ratio": float(
                summaries[name]["mean_prediction_std_ratio"]
                - summaries["baseline_hybrid"]["mean_prediction_std_ratio"]
            ),
        }

    report = {
        "schema": "isis.phase6_inference_ab_experiment.v1",
        "experiment": {
            "type": "inference_only_reversible_ab",
            "checkpoint": str(args.checkpoint),
            "checkpoint_model": checkpoint.get("model"),
            "marker_strength": args.marker_strength,
            "marker_method": "subtract per-column median offset relative to global median",
            "contrast_method": "training-only affine calibration to NASA target mean and standard deviation",
            "calibration_pair_count": len(calibration_rows),
            "variants": list(variant_names),
        },
        "split": {
            "scans": len(held_out),
            "reels": len(held_out_reels),
            "held_out_reel_disjoint_from_training": True,
            "selection_source": str(args.benchmark),
            "pair_names": selected_names,
        },
        "calibration": calibration,
        "summaries": summaries,
        "comparison_to_baseline": comparison,
        "by_station": {
            name: stratified(items, "station") for name, items in records.items()
        },
        "by_reel": {name: stratified(items, "reel") for name, items in records.items()},
        "worst_cases": {name: worst(items) for name, items in records.items()},
        "per_pair": per_pair,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Inference-only A/B comparison",
        "",
        f"This inference-only test compares the existing Hybrid U-Net with marker suppression and training-only contrast calibration on `{len(held_out)}` fixed held-out scans covering `{len(held_out_reels)}` reels.",
        "Production checkpoints and the main report were not modified.",
        "",
        "## Results",
        "",
        "| Variant | MAE | RMSE | Correlation | Std ratio | Mean bias | MAE > 0.20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in variant_names:
        summary = summaries[name]
        lines.append(
            f"| {name} | {summary['macro_mae']:.4f} | {summary['macro_rmse']:.4f} | {summary['macro_correlation']:.4f} | {summary['mean_prediction_std_ratio']:.3f} | {summary['mean_bias']:+.4f} | {summary['mae_over_0_20']} |"
        )
    lines.extend(
        [
            "",
            "## Change from baseline",
            "",
            "| Variant | MAE change | MAE win rate | Correlation change | Std-ratio change |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in comparison.items():
        lines.append(
            f"| {name} | {values['delta_macro_mae']:+.4f} | {values['mae_win_rate']:.1%} | {values['delta_macro_correlation']:+.4f} | {values['delta_mean_prediction_std_ratio']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "Lower MAE/RMSE and higher correlation are better. A standard-deviation ratio near 1.0 means the output has similar contrast to the NASA target. The contrast variant is deliberately a test-time calibration, not a retrained model; a positive result justifies a later training run with a contrast-aware loss.",
            "",
            f"Marker suppression strength: `{args.marker_strength}`. Calibration used one training scan per reel (`{len(calibration_rows)}` scans) and no held-out targets.",
            "",
            "Use the static gallery renderer for visual comparison when experiment assets are available.",
            "",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summaries": summaries,
                "comparison": comparison,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
