#!/usr/bin/env python3
"""Evaluate final Phase 6 candidates on a short, reel-balanced held-out set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research.models import image_features  # noqa: E402
from scripts.evaluation.evaluate_phase6_all_models import select_model_rows  # noqa: E402
from scripts.evaluation.evaluate_phase6_512_image_model import load_model  # noqa: E402
from scripts.experiments.run_phase6_experiment_ab import (  # noqa: E402
    load_experiment_sample,
    record_metrics,
)
from scripts.training.train_phase6_512_image_model import rows_for  # noqa: E402


DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_GROUPS = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_final_candidates_balanced_256"
DEFAULT_LIMIT = 256
DEFAULT_CHECKPOINTS = {
    "contrast_aware_norm_residual": ROOT / "outputs/evaluation/phase6_contrast_aware_norm_residual/best_model.pt",
    "norm_residual_unet": ROOT / "outputs/evaluation/phase6_continual_models/norm_residual_unet/best_model.pt",
    "hybrid_unet": ROOT / "outputs/evaluation/phase6_continual_models/hybrid_unet/best_model.pt",
    "residual_unet": ROOT / "outputs/evaluation/phase6_continual_models/residual_unet/best_model.pt",
}
MODEL_LABELS = {
    "contrast_aware_norm_residual": "Contrast-aware Normalized Residual U-Net",
    "norm_residual_unet": "Normalized Residual U-Net",
    "hybrid_unet": "Hybrid U-Net",
    "residual_unet": "Residual U-Net",
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


def predict_batch(model, signals, input_channels, torch):
    features = np.stack([image_features(signal, input_channels) for signal in signals])
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(features))).cpu().numpy().astype(np.float32)


def fit_calibration(model, input_channels, samples, batch_size, torch):
    prediction = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    target = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        outputs = predict_batch(model, [sample[0] for sample in batch], input_channels, torch)
        for output, (_, target_image, mask, _) in zip(outputs, batch):
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


def summarize(metrics):
    values = [item for item in metrics if item.get("mae") is not None]
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
        "correlation_below_0_20": int(sum(item["correlation"] < 0.20 for item in values)),
    }


def confidence_summary(per_pair, model_name, confidence_bin):
    values = [
        item["models"][model_name]["after"]
        for item in per_pair.values()
        if item["confidence"]["bin"] == confidence_bin
    ]
    return summarize(values)


def save_partial(path, per_pair):
    path.write_text(
        json.dumps({"schema": "isis.phase6_final_candidates_partial.v1", "per_pair": per_pair}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", action="append", metavar="NAME=PATH")
    parser.add_argument("--only", action="append", help="evaluate only these checkpoint names")
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1 or args.batch_size < 1:
        raise SystemExit("checkpoint and batch sizes must be positive")

    checkpoints = dict(DEFAULT_CHECKPOINTS)
    for item in args.checkpoint or []:
        if "=" not in item:
            raise SystemExit("--checkpoint must be NAME=PATH")
        name, path = item.split("=", 1)
        checkpoints[name] = Path(path)
    if args.only:
        missing = [name for name in args.only if name not in checkpoints]
        if missing:
            raise SystemExit(f"unknown --only model(s): {', '.join(missing)}")
        checkpoints = {name: checkpoints[name] for name in args.only}
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise SystemExit(f"output is not empty: {args.output}; use --resume or remove it")

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus_rows = rows_for(args.corpus)
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    group_rows = read_group_rows(args.groups)
    train_rows = [row for row in corpus_rows if row["split"] == "train"]
    held_out = [row for row in corpus_rows if row["split"] == "held_out"]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        held_out = select_model_rows(held_out, group_rows, args.limit, args.seed)
    train_reels = {group_rows[row["pair_name"]]["reel"] for row in train_rows}
    held_out_reels = {group_rows[row["pair_name"]]["reel"] for row in held_out}
    if train_reels & held_out_reels:
        raise SystemExit("training and held-out reels overlap")

    calibration_rows = {}
    for row in sorted(train_rows, key=lambda item: item["pair_name"]):
        calibration_rows.setdefault(group_rows[row["pair_name"]]["reel"], row)
    calibration_rows = list(calibration_rows.values())
    calibration_samples = [
        load_experiment_sample(args.corpus, args.targets, row, target_rows)
        for row in calibration_rows
    ]

    models = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise SystemExit(f"missing checkpoint for {name}: {path}")
        model, checkpoint = load_model(path, torch)
        models[name] = {
            "model": model,
            "checkpoint": checkpoint,
            "path": str(path),
            "input_channels": int(checkpoint.get("input_channels", 1)),
            "label": MODEL_LABELS.get(name, name),
        }
    calibrations = {
        name: fit_calibration(
            item["model"], item["input_channels"], calibration_samples, args.batch_size, torch
        )
        for name, item in models.items()
    }

    partial_path = args.output / "partial.json"
    per_pair = {}
    if args.resume and partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        per_pair = partial.get("per_pair", {})
        print(f"resuming {len(per_pair)}/{len(held_out)} held-out scans", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)

    pending = [row for row in held_out if row["pair_name"] not in per_pair]
    for start in range(0, len(pending), args.batch_size):
        batch_rows = pending[start : start + args.batch_size]
        samples = [
            load_experiment_sample(args.corpus, args.targets, row, target_rows)
            for row in batch_rows
        ]
        before_by_model = {}
        outputs_by_model = {}
        for model_name, item in models.items():
            before_batch = predict_batch(
                item["model"], [sample[0] for sample in samples], item["input_channels"], torch
            )
            before_by_model[model_name] = before_batch
            outputs_by_model[model_name] = [
                calibrated(before, calibrations[model_name]) for before in before_batch
            ]
        for local, row in enumerate(batch_rows):
            name = row["pair_name"]
            _, target, loss_mask, input_valid = samples[local]
            outputs = {model_name: values[local] for model_name, values in outputs_by_model.items()}
            metrics = {}
            for model_name in models:
                after = outputs[model_name]
                before = before_by_model[model_name][local]
                metrics[model_name] = {
                    "before": record_metrics(before, target, loss_mask),
                    "after": record_metrics(after, target, loss_mask),
                }
            ensemble = np.stack(list(outputs.values()))
            valid_disagreement = np.std(ensemble, axis=0)[input_valid]
            disagreement = float(np.mean(valid_disagreement)) if valid_disagreement.size else 1.0
            per_pair[name] = {
                "reel": group_rows[name]["reel"],
                "station": row.get("station") or "<blank>",
                "input_coverage": float(input_valid.mean()),
                "models": metrics,
                "confidence": {
                    "disagreement": disagreement,
                    "input_coverage": float(input_valid.mean()),
                },
            }
        completed = len(per_pair)
        if completed == len(held_out) or completed % args.checkpoint_every < args.batch_size:
            save_partial(partial_path, per_pair)
            print(f"evaluated {completed}/{len(held_out)}", flush=True)

    train_disagreements = []
    for start in range(0, len(calibration_samples), args.batch_size):
        batch = calibration_samples[start : start + args.batch_size]
        outputs_by_model = {}
        for model_name, item in models.items():
            before_batch = predict_batch(
                item["model"], [sample[0] for sample in batch], item["input_channels"], torch
            )
            outputs_by_model[model_name] = [
                calibrated(before, calibrations[model_name]) for before in before_batch
            ]
        for local, (_, _, _, input_valid) in enumerate(batch):
            outputs = np.stack(
                [values[local] for values in outputs_by_model.values()]
            )
            train_disagreements.append(float(np.mean(np.std(outputs, axis=0)[input_valid])))
    disagreement_p90 = float(np.percentile(train_disagreements, 90))
    disagreement_median = float(np.median(train_disagreements))
    disagreement_scale = max(disagreement_p90, 1e-6)
    for item in per_pair.values():
        disagreement = item["confidence"]["disagreement"]
        coverage = item["confidence"]["input_coverage"]
        score = float(np.exp(-disagreement / disagreement_scale))
        if coverage < 0.9:
            score *= float(coverage / 0.9)
        flag = disagreement > disagreement_p90 or coverage < 0.9 or score < 0.5
        item["confidence"].update(
            {
                "score": score,
                "bin": "high" if score >= 0.55 else "medium" if score >= 0.45 else "low",
                "flag_for_review": bool(flag),
                "flag_reasons": [
                    reason
                    for reason, condition in (
                        ("model_disagreement", disagreement > disagreement_p90),
                        ("low_input_coverage", coverage < 0.9),
                        ("low_confidence_score", score < 0.5),
                    )
                    if condition
                ],
            }
        )

    model_reports = {}
    for name, item in models.items():
        before = summarize([value["models"][name]["before"] for value in per_pair.values()])
        after = summarize([value["models"][name]["after"] for value in per_pair.values()])
        before_values = np.asarray([value["models"][name]["before"]["mae"] for value in per_pair.values()])
        after_values = np.asarray([value["models"][name]["after"]["mae"] for value in per_pair.values()])
        model_reports[name] = {
            "label": item["label"],
            "checkpoint": item["path"],
            "model": item["checkpoint"].get("model"),
            "calibration": calibrations[name],
            "before": before,
            "after": after,
            "comparison": {
                "delta_macro_mae": float(after["macro_mae"] - before["macro_mae"]),
                "mae_improvement": float(np.mean(before_values - after_values)),
                "mae_win_rate": float(np.mean(after_values < before_values)),
                "delta_macro_correlation": float(after["macro_correlation"] - before["macro_correlation"]),
                "delta_std_ratio": float(after["mean_prediction_std_ratio"] - before["mean_prediction_std_ratio"]),
            },
            "confidence_bins": {
                confidence_bin: confidence_summary(per_pair, name, confidence_bin)
                for confidence_bin in ("high", "medium", "low")
            },
        }

    best_model = min(model_reports, key=lambda name: model_reports[name]["after"]["macro_mae"])
    flagged = sorted(
        (
            {
                "pair_name": name,
                "reel": item["reel"],
                "station": item["station"],
                "confidence": item["confidence"],
                "best_model_after": item["models"][best_model]["after"],
            }
            for name, item in per_pair.items()
            if item["confidence"]["flag_for_review"]
        ),
        key=lambda item: item["confidence"]["score"],
    )
    report = {
        "schema": "isis.phase6_final_candidates_balanced.v1",
        "experiment": {
            "type": "isolated_final_candidate_evaluation",
            "models_changed": False,
            "confidence_method": "ensemble pixel disagreement plus input coverage; no NASA target used",
            "confidence_ensemble": list(models),
            "confidence_thresholds_fit_on_training_scans": len(calibration_samples),
            "disagreement_training_median": disagreement_median,
            "disagreement_training_p90": disagreement_p90,
        },
        "split": {
            "train_scans": len(train_rows),
            "held_out_scans": len(held_out),
            "train_reels": len(train_reels),
            "held_out_reels": len(held_out_reels),
            "reel_disjoint": True,
            "selection": "deterministic round-robin by reel, then stable pair hash",
            "seed": args.seed,
        },
        "best_model_by_after_mae": best_model,
        "models": model_reports,
        "confidence": {
            "flagged_scans": len(flagged),
            "flag_rate": float(len(flagged) / max(len(per_pair), 1)),
            "flagged_cases": flagged[:50],
        },
        "per_pair": per_pair,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Phase 6 final candidate balanced held-out evaluation",
        "",
        f"Evaluated `{len(models)}` final candidates on all `{len(held_out)}` held-out scans across `{len(held_out_reels)}` reels. No model weights were modified.",
        "",
        "| Model | Before MAE | After MAE | Δ MAE | After correlation | After std ratio | MAE win rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in sorted(model_reports.items(), key=lambda pair: pair[1]["after"]["macro_mae"]):
        lines.append(
            f"| {item['label']} | {item['before']['macro_mae']:.4f} | {item['after']['macro_mae']:.4f} | {item['comparison']['delta_macro_mae']:+.4f} | {item['after']['macro_correlation']:.4f} | {item['after']['mean_prediction_std_ratio']:.3f} | {item['comparison']['mae_win_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Best candidate by after-calibration MAE: **{model_reports[best_model]['label']}**.",
            "",
            "## Confidence flags",
            "",
            f"`{len(flagged)}` of `{len(per_pair)}` scans ({len(flagged) / max(len(per_pair), 1):.1%}) were flagged using model disagreement, input coverage, and a training-derived confidence threshold. NASA targets were used only for evaluation, not for confidence scoring.",
            "",
            "High-confidence scans should be prioritized for automation; flagged scans should be reviewed or excluded until the model improves on those failure modes.",
            "",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best_model": best_model, "flagged_scans": len(flagged), "models": list(model_reports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
