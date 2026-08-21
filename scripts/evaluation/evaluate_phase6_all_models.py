#!/usr/bin/env python3
"""Evaluate all stored Phase 6 checkpoints on one reproducible held-out set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram  # noqa: E402
from isis_research.evaluation.splits import group_summary  # noqa: E402
from scripts.evaluation.benchmark_phase6_512_image_baselines import (  # noqa: E402
    metric,
    predictions as baseline_predictions,
)
from scripts.evaluation.evaluate_phase6_512_image_model import load_model  # noqa: E402
from scripts.training.train_phase6_512_image_model import load_sample, rows_for  # noqa: E402
from isis_research.models import image_features  # noqa: E402


DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_GROUPS = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_CHECKPOINTS = ROOT / "outputs/evaluation/phase6_continual_models"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_all_models_report"
DEFAULT_MODEL_LIMIT = 128
BASELINE_NAMES = (
    "constant_train_mean",
    "inverted_film",
    "inverted_film_normalized",
    "local_contrast_normalized",
    "blurred_inverted_film",
)


def read_group_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["pair_name"]: row for row in csv.DictReader(handle)}


def stable_key(seed: int, value: str):
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def select_model_rows(rows, groups, limit: int, seed: int):
    """Select a deterministic, reel-balanced native-resolution benchmark."""
    if limit < 1:
        raise ValueError("model limit must be positive")
    by_reel = {}
    for row in rows:
        reel = groups[row["pair_name"]]["reel"]
        by_reel.setdefault(reel, []).append(row)
    reels = sorted(by_reel, key=lambda value: stable_key(seed, value))
    for reel in reels:
        by_reel[reel] = sorted(
            by_reel[reel],
            key=lambda row: stable_key(seed, row["pair_name"]),
        )
    selected = []
    offset = 0
    while len(selected) < min(limit, len(rows)):
        added = False
        for reel in reels:
            if offset < len(by_reel[reel]):
                selected.append(by_reel[reel][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def train_mean(corpus, targets, rows, target_rows):
    total = 0.0
    pixels = 0
    for row in rows:
        _, target, mask = load_sample(corpus, targets, row, target_rows)
        total += float(target[mask].sum())
        pixels += int(mask.sum())
    return total / max(pixels, 1)


def axis_profile(scan):
    return f"{scan.frequency_mhz[0]:.1f}-{scan.frequency_mhz[-1]:.1f}MHz"


def record_metrics(candidate, target, mask):
    result = metric(candidate, target, mask)
    result["prediction_std_ratio"] = float(
        np.std(candidate[mask]) / max(np.std(target[mask]), 1e-9)
    )
    return result


def summarize(records):
    valid = [record for record in records if record["mae"] is not None]
    if not valid:
        return {"scans": 0}
    return {
        "scans": len(valid),
        "macro_mae": float(np.mean([record["mae"] for record in valid])),
        "macro_rmse": float(np.mean([record["rmse"] for record in valid])),
        "macro_correlation": float(np.mean([record["correlation"] for record in valid])),
        "median_mae": float(np.median([record["mae"] for record in valid])),
        "p90_mae": float(np.percentile([record["mae"] for record in valid], 90)),
        "mean_prediction_std_ratio": float(
            np.mean([record["prediction_std_ratio"] for record in valid])
        ),
    }


def stratified(records, key):
    groups = {}
    for record in records:
        groups.setdefault(record[key], []).append(record["metrics"])
    return {name: summarize(values) for name, values in sorted(groups.items())}


def worst(records, count=10):
    return [
        {
            "pair_name": record["pair_name"],
            "reel": record["reel"],
            "station": record["station"],
            "axis_profile": record["axis_profile"],
            "metrics": record["metrics"],
        }
        for record in sorted(records, key=lambda item: item["metrics"]["mae"], reverse=True)[:count]
    ]


def group_uncertainty(records):
    rows = [{"reel": record["reel"]} for record in records]
    values = [record["metrics"]["mae"] for record in records]
    return group_summary(rows, values, key="reel", statistic=np.median, seed=20260818)


def evaluate_baselines(
    corpus, targets, rows, target_rows, train_target_mean, include_expensive=True
):
    results = {name: [] for name in BASELINE_NAMES}
    for row in rows:
        scan = ionogram.read_validated(corpus / row["csa_artifact"])
        _, target, target_mask = load_sample(corpus, targets, row, target_rows)
        candidates = baseline_predictions(
            scan.intensity,
            scan.valid_mask,
            train_target_mean,
            include_local=include_expensive,
        )
        for name, candidate in candidates.items():
            results[name].append(record_metrics(candidate, target, target_mask))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-limit", type=int, default=DEFAULT_MODEL_LIMIT)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = rows_for(args.corpus)
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    groups = read_group_rows(args.groups)
    for row in rows:
        if row["pair_name"] not in groups:
            raise SystemExit(f"missing reel metadata for {row['pair_name']}")
    train_rows = [row for row in rows if row["split"] == "train"]
    held_out = [row for row in rows if row["split"] == "held_out"]
    train_reels = {groups[row["pair_name"]]["reel"] for row in train_rows}
    held_reels = {groups[row["pair_name"]]["reel"] for row in held_out}
    if train_reels & held_reels:
        raise SystemExit("train and held-out reels overlap")
    model_rows = select_model_rows(held_out, groups, args.model_limit, args.seed)
    train_target_mean = train_mean(args.corpus, args.targets, train_rows, target_rows)

    baseline_full = evaluate_baselines(
        args.corpus,
        args.targets,
        held_out,
        target_rows,
        train_target_mean,
        include_expensive=False,
    )
    baseline_subset = evaluate_baselines(
        args.corpus, args.targets, model_rows, target_rows, train_target_mean
    )
    print(
        f"baselines complete: full={len(held_out)} benchmark={len(model_rows)}",
        flush=True,
    )

    checkpoint_paths = sorted(args.checkpoints.glob("*/best_model.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"no best_model.pt files found under {args.checkpoints}")
    models = {}
    for checkpoint_path in checkpoint_paths:
        model, checkpoint = load_model(checkpoint_path, torch)
        models[checkpoint_path.parent.name] = {
            "model": model,
            "checkpoint": checkpoint,
            "path": str(checkpoint_path),
        }

    model_records = {name: [] for name in models}
    per_pair = {
        row["pair_name"]: {
            "reel": groups[row["pair_name"]]["reel"],
            "station": row.get("station") or "<blank>",
            "baselines": {},
            "models": {},
        }
        for row in model_rows
    }
    for index, row in enumerate(model_rows, 1):
        scan = ionogram.read_validated(args.corpus / row["csa_artifact"])
        film, target, target_mask = load_sample(args.corpus, args.targets, row, target_rows)
        candidates = baseline_predictions(scan.intensity, scan.valid_mask, train_target_mean)
        common = {
            "pair_name": row["pair_name"],
            "reel": groups[row["pair_name"]]["reel"],
            "station": row.get("station") or "<blank>",
            "axis_profile": axis_profile(scan),
            "target_coverage": f"{float(target_mask.mean()):.2f}",
        }
        for name, candidate in candidates.items():
            metrics = record_metrics(candidate, target, target_mask)
            per_pair[row["pair_name"]]["baselines"][name] = metrics

        for name, item in models.items():
            input_channels = item["checkpoint"].get("input_channels", 1)
            with torch.no_grad():
                prediction = torch.sigmoid(
                    item["model"](
                        torch.from_numpy(
                            image_features(film, input_channels)[None, ...]
                        )
                    )
                ).numpy()[0]
            metrics = record_metrics(prediction, target, target_mask)
            record = {**common, "metrics": metrics}
            model_records[name].append(record)
            per_pair[row["pair_name"]]["models"][name] = metrics
        if index == 1 or index % 16 == 0 or index == len(model_rows):
            print(f"models evaluated {index}/{len(model_rows)}", flush=True)

    def model_report(records):
        return {
            "summary": summarize([record["metrics"] for record in records]),
            "by_station": stratified(records, "station"),
            "by_reel": stratified(records, "reel"),
            "by_axis_profile": stratified(records, "axis_profile"),
            "by_target_coverage": stratified(records, "target_coverage"),
            "group_uncertainty_mae": group_uncertainty(records),
            "worst_cases": worst(records),
        }

    model_results = {name: model_report(records) for name, records in model_records.items()}
    baseline_results = {
        "full_held_out": {name: summarize(values) for name, values in baseline_full.items()},
        "model_benchmark_subset": {
            name: summarize(values) for name, values in baseline_subset.items()
        },
    }
    comparison = {}
    for model_name in model_records:
        comparison[model_name] = {}
        for baseline_name in BASELINE_NAMES:
            model_values = [
                per_pair[name]["models"][model_name]["mae"] for name in per_pair
            ]
            baseline_values = [
                per_pair[name]["baselines"][baseline_name]["mae"] for name in per_pair
            ]
            comparison[model_name][baseline_name] = {
                "mae_win_rate": float(np.mean(np.asarray(model_values) < np.asarray(baseline_values))),
                "mae_improvement": float(np.mean(np.asarray(baseline_values) - np.asarray(model_values))),
            }

    report = {
        "schema": "isis.phase6_all_models_heldout_report.v1",
        "seed": args.seed,
        "split": {
            "train_scans": len(train_rows),
            "held_out_scans": len(held_out),
            "train_reels": len(train_reels),
            "held_out_reels": len(held_reels),
            "reel_disjoint": True,
            "held_out_stations": sorted(
                {row.get("station") or "<blank>" for row in held_out}
            ),
        },
        "model_benchmark": {
            "scans": len(model_rows),
            "reels": len({groups[row["pair_name"]]["reel"] for row in model_rows}),
            "selection": "deterministic round-robin by reel, then stable pair hash",
            "pair_names": [row["pair_name"] for row in model_rows],
        },
        "train_mean_target": float(train_target_mean),
        "checkpoints": {name: {"path": item["path"], "model": item["checkpoint"].get("model")} for name, item in models.items()},
        "baselines": baseline_results,
        "models": model_results,
        "model_vs_baseline": comparison,
        "per_pair": per_pair,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    ranking = sorted(
        ((name, result["summary"]) for name, result in model_results.items()),
        key=lambda item: item[1]["macro_mae"],
    )
    lines = [
        "# Phase 6 model evaluation report",
        "",
        f"Generated with seed `{args.seed}` on `{len(model_rows)}` native 512×512 held-out scans covering `{report['model_benchmark']['reels']}` reels.",
        "The held-out split contains 1,137 scans across 40 reels and is disjoint from training by reel. Baselines were also run over all 1,137 held-out scans.",
        "",
        "## Model ranking on the fixed benchmark subset",
        "",
        "| Model | MAE | RMSE | Correlation | Median MAE | P90 MAE | MAE std ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in ranking:
        lines.append(
            f"| {name} | {summary['macro_mae']:.4f} | {summary['macro_rmse']:.4f} | {summary['macro_correlation']:.4f} | {summary['median_mae']:.4f} | {summary['p90_mae']:.4f} | {summary['mean_prediction_std_ratio']:.3f} |"
        )
    lines.extend(["", "## Baselines", "", "| Baseline | Full held-out MAE | Full held-out correlation | Benchmark-subset MAE | Benchmark-subset correlation |", "|---|---:|---:|---:|---:|"])
    for name in BASELINE_NAMES:
        full = baseline_results["full_held_out"].get(name)
        subset = baseline_results["model_benchmark_subset"][name]
        full_mae = (
            f"{full['macro_mae']:.4f}"
            if full and full.get("macro_mae") is not None
            else "subset only"
        )
        full_corr = (
            f"{full['macro_correlation']:.4f}"
            if full and full.get("macro_correlation") is not None
            else "subset only"
        )
        lines.append(f"| {name} | {full_mae} | {full_corr} | {subset['macro_mae']:.4f} | {subset['macro_correlation']:.4f} |")
    lines.extend(["", "## Worst cases", ""])
    for name, _ in ranking:
        lines.append(f"### {name}")
        for case in model_results[name]["worst_cases"][:5]:
            metrics = case["metrics"]
            lines.append(f"- `{case['pair_name']}` — reel `{case['reel']}`, station `{case['station']}`, MAE `{metrics['mae']:.4f}`, correlation `{metrics['correlation']:.4f}`")
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "Model metrics are native-resolution results on the fixed 128-scan benchmark. Baseline full-held-out results use all 1,137 held-out scans. The benchmark is intentionally reel-balanced because native CPU inference makes a full eight-model pass expensive; `report.json` contains the exact pair list and all stratified results.",
        "",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model_scans": len(model_rows), "models": list(models), "ranking": ranking}, indent=2), flush=True)


if __name__ == "__main__":
    main()
