#!/usr/bin/env python3
"""Test training-only contrast calibration on every stored Phase 6 model.

This is inference-only.  Checkpoints are never overwritten; the output is a
separate report containing before/after metrics for the fixed 128-scan,
reel-balanced held-out benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.evaluation.evaluate_phase6_512_image_model import load_model  # noqa: E402
from scripts.experiments.run_phase6_experiment_ab import (  # noqa: E402
    load_experiment_sample,
    predict,
    record_metrics,
)
from scripts.training.train_phase6_512_image_model import rows_for  # noqa: E402


DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_GROUPS = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_CHECKPOINTS = ROOT / "outputs/evaluation/phase6_continual_models"
DEFAULT_BENCHMARK = ROOT / "outputs/evaluation/phase6_all_models_report/report.json"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_all_models_contrast_test"

MODEL_LABELS = {
    "cnn_2d": "ScanCNN",
    "coord_unet": "Coordinate U-Net",
    "dilated_cnn": "Dilated CNN",
    "hybrid_unet": "Hybrid U-Net",
    "norm_residual_unet": "Normalized Residual U-Net",
    "residual_unet": "Residual U-Net",
    "unet": "Tiny U-Net",
    "wide_unet": "Wide U-Net",
}


def read_group_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["pair_name"]: row for row in csv.DictReader(handle)}


def update_stats(stats, values):
    values = np.asarray(values, dtype=np.float64)
    stats["sum"] += float(values.sum())
    stats["sum_sq"] += float(np.square(values).sum())
    stats["count"] += int(values.size)


def finish_stats(stats):
    mean = stats["sum"] / max(stats["count"], 1)
    variance = stats["sum_sq"] / max(stats["count"], 1) - mean * mean
    return float(mean), float(np.sqrt(max(variance, 0.0)))


def summarize(metrics):
    values = [item for item in metrics if item["mae"] is not None]
    if not values:
        return {"scans": 0}
    return {
        "scans": len(values),
        "macro_mae": float(np.mean([item["mae"] for item in values])),
        "macro_rmse": float(np.mean([item["rmse"] for item in values])),
        "macro_correlation": float(np.mean([item["correlation"] for item in values])),
        "median_mae": float(np.median([item["mae"] for item in values])),
        "p90_mae": float(np.percentile([item["mae"] for item in values], 90)),
        "mean_prediction_std_ratio": float(
            np.mean([item["prediction_std_ratio"] for item in values])
        ),
        "mean_bias": float(np.mean([item["bias"] for item in values])),
        "mae_over_0_20": int(sum(item["mae"] > 0.20 for item in values)),
    }


def grouped(records, key):
    result = {}
    for record in records:
        result.setdefault(record[key], []).append(record["metrics"])
    return {name: summarize(values) for name, values in sorted(result.items())}


def worst(records, count=5):
    return [
        {
            "pair_name": record["pair_name"],
            "reel": record["reel"],
            "station": record["station"],
            "metrics": record["metrics"],
        }
        for record in sorted(
            records, key=lambda item: item["metrics"]["mae"], reverse=True
        )[:count]
    ]


def affine_calibration(model, input_channels, samples, torch):
    prediction = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    target = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    for signal, target_image, mask, _ in samples:
        output = predict(model, signal, input_channels, torch)
        update_stats(prediction, output[mask])
        update_stats(target, target_image[mask])
    prediction_mean, prediction_std = finish_stats(prediction)
    target_mean, target_std = finish_stats(target)
    scale = target_std / max(prediction_std, 1e-9)
    return {
        "scans": len(samples),
        "target_mean": target_mean,
        "target_std": target_std,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "scale": float(scale),
        "bias": float(target_mean - scale * prediction_mean),
    }


def calibrated(output, calibration):
    return np.clip(
        float(calibration["scale"]) * output + float(calibration["bias"]), 0.0, 1.0
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}; remove it to rerun the test")

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus_rows = rows_for(args.corpus)
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    groups = read_group_rows(args.groups)
    train_rows = [row for row in corpus_rows if row["split"] == "train"]
    held_out_rows = {row["pair_name"]: row for row in corpus_rows if row["split"] == "held_out"}
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    selected_names = benchmark["model_benchmark"]["pair_names"]
    benchmark_rows = [held_out_rows[name] for name in selected_names]
    train_reels = {groups[row["pair_name"]]["reel"] for row in train_rows}
    held_out_reels = {groups[row["pair_name"]]["reel"] for row in benchmark_rows}
    if train_reels & held_out_reels:
        raise SystemExit("training and held-out reels overlap")

    calibration_rows = {}
    for row in sorted(train_rows, key=lambda item: item["pair_name"]):
        calibration_rows.setdefault(groups[row["pair_name"]]["reel"], row)
    calibration_rows = list(calibration_rows.values())
    calibration_samples = [
        load_experiment_sample(args.corpus, args.targets, row, target_rows)
        for row in calibration_rows
    ]

    checkpoint_paths = sorted(args.checkpoints.glob("*/best_model.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"no checkpoints found under {args.checkpoints}")
    models = {}
    for path in checkpoint_paths:
        model, checkpoint = load_model(path, torch)
        name = path.parent.name
        models[name] = {
            "model": model,
            "checkpoint": checkpoint,
            "path": str(path),
            "label": MODEL_LABELS.get(name, name),
            "input_channels": int(checkpoint.get("input_channels", 1)),
        }

    calibrations = {}
    for index, (name, item) in enumerate(models.items(), 1):
        print(f"calibrating {index}/{len(models)}: {name}", flush=True)
        calibrations[name] = affine_calibration(
            item["model"], item["input_channels"], calibration_samples, torch
        )

    records = {name: [] for name in models}
    per_pair = {
        row["pair_name"]: {
            "reel": groups[row["pair_name"]]["reel"],
            "station": row.get("station") or "<blank>",
            "models": {},
        }
        for row in benchmark_rows
    }
    for index, row in enumerate(benchmark_rows, 1):
        signal, target, mask, _ = load_experiment_sample(
            args.corpus, args.targets, row, target_rows
        )
        metadata = {
            "pair_name": row["pair_name"],
            "reel": groups[row["pair_name"]]["reel"],
            "station": row.get("station") or "<blank>",
            "target_coverage": float(mask.mean()),
        }
        for name, item in models.items():
            before = predict(item["model"], signal, item["input_channels"], torch)
            after = calibrated(before, calibrations[name])
            before_metrics = record_metrics(before, target, mask)
            after_metrics = record_metrics(after, target, mask)
            records[name].append({**metadata, "metrics": before_metrics})
            per_pair[row["pair_name"]]["models"][name] = {
                "before": before_metrics,
                "after": after_metrics,
            }
        if index == 1 or index % 16 == 0 or index == len(benchmark_rows):
            print(f"evaluated {index}/{len(benchmark_rows)}", flush=True)

    model_reports = {}
    for name, item in models.items():
        before_records = records[name]
        before_metrics = [record["metrics"] for record in before_records]
        after_metrics = [per_pair[row["pair_name"]]["models"][name]["after"] for row in benchmark_rows]
        before = summarize(before_metrics)
        after = summarize(after_metrics)
        before_mae = np.asarray([metric_item["mae"] for metric_item in before_metrics])
        after_mae = np.asarray([metric_item["mae"] for metric_item in after_metrics])
        model_reports[name] = {
            "label": item["label"],
            "model": item["checkpoint"].get("model"),
            "checkpoint": item["path"],
            "input_channels": item["input_channels"],
            "calibration": calibrations[name],
            "before": before,
            "after": after,
            "comparison": {
                "delta_macro_mae": float(after["macro_mae"] - before["macro_mae"]),
                "mae_improvement": float(np.mean(before_mae - after_mae)),
                "mae_win_rate": float(np.mean(after_mae < before_mae)),
                "delta_macro_correlation": float(
                    after["macro_correlation"] - before["macro_correlation"]
                ),
                "delta_std_ratio": float(
                    after["mean_prediction_std_ratio"]
                    - before["mean_prediction_std_ratio"]
                ),
            },
            "before_by_station": grouped(before_records, "station"),
            "before_by_reel": grouped(before_records, "reel"),
            "after_by_station": {
                station: summarize(
                    [
                        per_pair[row["pair_name"]]["models"][name]["after"]
                        for row in benchmark_rows
                        if per_pair[row["pair_name"]]["station"] == station
                    ]
                )
                for station in sorted({record["station"] for record in before_records})
            },
            "worst_before": worst(before_records),
            "worst_after": worst(
                [
                    {
                        "pair_name": row["pair_name"],
                        "reel": per_pair[row["pair_name"]]["reel"],
                        "station": per_pair[row["pair_name"]]["station"],
                        "metrics": per_pair[row["pair_name"]]["models"][name]["after"],
                    }
                    for row in benchmark_rows
                ]
            ),
        }

    report = {
        "schema": "isis.phase6_all_models_contrast_test.v1",
        "experiment": {
            "type": "inference_only_reversible",
            "contrast_method": "model-specific affine calibration fitted on one training scan per reel",
            "calibration_scans": len(calibration_rows),
            "models_changed": False,
        },
        "split": {
            "scans": len(benchmark_rows),
            "reels": len(held_out_reels),
            "train_reels": len(train_reels),
            "reel_disjoint": True,
            "selection_source": str(args.benchmark),
            "pair_names": selected_names,
        },
        "models": model_reports,
        "per_pair": per_pair,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Phase 6 all-model contrast test",
        "",
        f"Inference-only comparison of all `{len(models)}` stored checkpoints on `{len(benchmark_rows)}` held-out scans covering `{len(held_out_reels)}` reels.",
        "No checkpoint or production model was modified. Contrast calibration was fitted separately for each model using one training scan per reel.",
        "",
        "| Model | Before MAE | After MAE | Δ MAE | Before corr. | After corr. | Before std | After std | MAE win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in sorted(model_reports.items(), key=lambda item: item[1]["after"]["macro_mae"]):
        before, after, comparison = result["before"], result["after"], result["comparison"]
        lines.append(
            f"| {result['label']} | {before['macro_mae']:.4f} | {after['macro_mae']:.4f} | {comparison['delta_macro_mae']:+.4f} | {before['macro_correlation']:.4f} | {after['macro_correlation']:.4f} | {before['mean_prediction_std_ratio']:.3f} | {after['mean_prediction_std_ratio']:.3f} | {comparison['mae_win_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Lower MAE is better, higher correlation is better, and a standard-deviation ratio near 1.0 means the output contrast is closer to NASA. This is post-processing only: it does not create new weights and is not a retrained model.",
            "",
            "Use the static gallery renderer for visual comparison when experiment assets are available.",
            "",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": list(model_reports), "report": str(args.output / "REPORT.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
