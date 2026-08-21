#!/usr/bin/env python3
"""Evaluate a native CSA-to-NASA image translator on held-out usable scans.

The direct target is the matched NASA image.  Trace scores are a secondary
diagnostic: the same ridge/continuity extractor is run on the NASA target and
on each candidate image after a common 64x96 evaluation resampling.  NASA is
therefore a held-out reference, never an inference input.
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

from isis_research import ionogram  # noqa: E402
from isis_research.extraction import echo  # noqa: E402
from scripts.evaluation.benchmark_phase6_512_image_baselines import (  # noqa: E402
    metric as image_metric,
    predictions as baseline_predictions,
)
from isis_research.grids import (  # noqa: E402
    TARGET_FREQUENCY,
    TARGET_HEIGHT,
    resample_grid,
)
from isis_research.models import image_features, model_constructor  # noqa: E402
from scripts.training.train_phase6_512_image_model import load_sample, rows_for  # noqa: E402


DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_CHECKPOINT = ROOT / "outputs/evaluation/phase6_512_image_model_full_v2/model.pt"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_512_image_model_full_v2_evaluation"
MAX_HEIGHT_KM = 2500.0
TRACE_THRESHOLD = 5.0
TRACES = 2


def trace_reference(image, valid):
    """Extract ordered NASA reference paths on the common evaluation grid."""
    keep = TARGET_HEIGHT <= MAX_HEIGHT_KM
    target = np.where(valid[keep], image[keep], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        found = echo.extract(
            target.T,
            traces=TRACES,
            **echo.parameters_for(TARGET_HEIGHT[keep]),
        )
    return found


def trace_candidate(image):
    keep = TARGET_HEIGHT <= MAX_HEIGHT_KM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return echo.extract(
            image[keep].T,
            traces=TRACES,
            **echo.parameters_for(TARGET_HEIGHT[keep]),
        )


def path_metric(candidate, reference, height):
    candidate_path, candidate_score = candidate
    reference_path, reference_score = reference
    reference_detected = np.isfinite(reference_score) & (reference_score >= TRACE_THRESHOLD)
    candidate_detected = np.isfinite(candidate_score)
    common = reference_detected & candidate_detected
    if not reference_detected.any():
        return {"comparable": False, "reference_points": 0}
    if not common.any():
        return {
            "comparable": False,
            "reference_points": int(reference_detected.sum()),
            "candidate_coverage": 0.0,
        }
    columns = np.flatnonzero(common)
    error = height[candidate_path[columns]] - height[reference_path[columns]]
    absolute = np.abs(error)
    return {
        "comparable": True,
        "reference_points": int(reference_detected.sum()),
        "candidate_points": int(candidate_detected.sum()),
        "candidate_coverage": float(common.sum() / reference_detected.sum()),
        "median_abs_error_km": float(np.median(absolute)),
        "p90_abs_error_km": float(np.percentile(absolute, 90.0)),
        "within_60_fraction": float(np.mean(absolute <= 60.0)),
        "within_150_fraction": float(np.mean(absolute <= 150.0)),
        "median_step_km": float(np.median(np.abs(np.diff(height[candidate_path])))),
    }


def summarize(items, field):
    values = [item[field] for item in items if item.get("comparable") and field in item]
    if not values:
        return {"scans": 0}
    values = np.asarray(values, dtype=float)
    return {
        "scans": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
    }


def trace_summary(items):
    fields = (
        "candidate_coverage",
        "median_abs_error_km",
        "p90_abs_error_km",
        "within_60_fraction",
        "within_150_fraction",
        "median_step_km",
    )
    return {field: summarize(items, field) for field in fields}


def image_summary(items):
    result = {
        "scans": len(items),
        "macro_mae": float(np.mean([item["mae"] for item in items])),
        "macro_rmse": float(np.mean([item["rmse"] for item in items])),
        "macro_correlation": float(np.mean([item["correlation"] for item in items])),
        "pixels": int(sum(item["pixels"] for item in items)),
    }
    if any("prediction_std_ratio" in item for item in items):
        result["mean_prediction_std_ratio"] = float(
            np.mean([item["prediction_std_ratio"] for item in items])
        )
    return result


def coverage_bin(value):
    if value < 0.9:
        return "<0.90"
    if value < 0.99:
        return "0.90-0.99"
    return ">=0.99"


def stratified_image_summary(records, field):
    groups = {}
    for record in records:
        groups.setdefault(record[field], []).append(record["metrics"])
    return {key: image_summary(values) for key, values in sorted(groups.items())}


def stratified_trace_summary(records, field):
    groups = {}
    for record in records:
        groups.setdefault(record[field], []).append(record["metrics"])
    return {key: trace_summary(values) for key, values in sorted(groups.items())}


def load_model(checkpoint_path, torch):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model")
    model = model_constructor(model_name)(checkpoint.get("input_channels", 1))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-held-out", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")

    import torch

    rows = rows_for(args.corpus)
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    held_out = [row for row in rows if row["split"] == "held_out"]
    if args.limit_held_out is not None:
        held_out = held_out[: args.limit_held_out]
    train_rows = [row for row in rows if row["split"] == "train"]
    model, checkpoint = load_model(args.checkpoint, torch)

    # The scalar baseline is computed from training targets only.
    train_total = 0.0
    train_pixels = 0
    for row in train_rows:
        _, target, mask = load_sample(args.corpus, args.targets, row, target_rows)
        train_total += float(target[mask].sum())
        train_pixels += int(mask.sum())
    train_mean = train_total / max(train_pixels, 1)

    image_results = {}
    trace_results = {name: {f"trace_{i + 1}": [] for i in range(TRACES)} for name in (
        "model",
        "inverted_film",
        "blurred_inverted_film",
    )}
    all_image_metrics = {name: [] for name in trace_results}
    image_records = {name: [] for name in trace_results}
    trace_records = {
        name: {f"trace_{i + 1}": [] for i in range(TRACES)} for name in trace_results
    }
    all_names = []

    for start in range(0, len(held_out), args.batch_size):
        batch_rows = held_out[start : start + args.batch_size]
        samples = [load_sample(args.corpus, args.targets, row, target_rows) for row in batch_rows]
        films, targets, masks = zip(*samples)
        with torch.no_grad():
            model_output = torch.sigmoid(
                model(
                    torch.from_numpy(
                        np.stack([image_features(film, checkpoint.get("input_channels", 1)) for film in films])
                    )
                )
            ).numpy()
        for row, (film, target, mask), prediction in zip(batch_rows, samples, model_output):
            scan = ionogram.read_validated(args.corpus / row["csa_artifact"])
            candidates = baseline_predictions(scan.intensity, scan.valid_mask, train_mean)
            candidates["model"] = prediction.astype(np.float32)
            coverage = float(mask.mean())
            strata = {
                "station": row.get("station") or "<blank>",
                "target_coverage": coverage_bin(coverage),
                "axis_profile": f"{scan.frequency_mhz[0]:.1f}-{scan.frequency_mhz[-1]:.1f}MHz",
            }
            # Image metrics are measured on native 512x512 values.
            for name, candidate in candidates.items():
                metrics = image_metric(candidate, target, mask)
                if name == "model":
                    metrics["prediction_std_ratio"] = float(
                        np.std(candidate[mask]) / max(np.std(target[mask]), 1e-9)
                    )
                all_image_metrics.setdefault(name, []).append(metrics)
                image_records.setdefault(name, []).append({**strata, "metrics": metrics})

            # Trace metrics use one common, inexpensive 64x96 representation.
            target_small = resample_grid(
                target, scan.virtual_height_km, scan.frequency_mhz, fill_value=np.nan
            ).astype(np.float32)
            target_valid_small = (
                resample_grid(
                    mask.astype(float), scan.virtual_height_km, scan.frequency_mhz, fill_value=0.0
                )
                > 0.5
            )
            reference = trace_reference(target_small, target_valid_small)
            for name in trace_results:
                candidate_small = resample_grid(
                    candidates[name],
                    scan.virtual_height_km,
                    scan.frequency_mhz,
                    fill_value=0.0,
                ).astype(np.float32)
                found = trace_candidate(candidate_small)
                for index in range(TRACES):
                    metrics = path_metric(
                        found[index],
                        reference[index],
                        TARGET_HEIGHT[TARGET_HEIGHT <= MAX_HEIGHT_KM],
                    )
                    trace_name = f"trace_{index + 1}"
                    trace_results[name][trace_name].append(metrics)
                    trace_records[name][trace_name].append({**strata, "metrics": metrics})
            all_names.append(row["pair_name"])
        count = min(start + args.batch_size, len(held_out))
        if count == args.batch_size or count % 100 == 0 or count == len(held_out):
            print(f"evaluated {count}/{len(held_out)}: {batch_rows[-1]['pair_name']}", flush=True)

    for name, metrics in all_image_metrics.items():
        image_results[name] = image_summary(metrics)

    report = {
        "schema": "isis.phase6_native_512_image_model_heldout_evaluation.v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_model": checkpoint.get("model"),
        "held_out_scans": len(held_out),
        "train_scans_for_constant": len(train_rows),
        "train_mean_target": float(train_mean),
        "image_grid": {"height": 512, "frequency": 512},
        "trace_grid": {"height": 64, "frequency": 96, "max_height_km": MAX_HEIGHT_KM},
        "trace_reference": "same ridge/continuity extractor on held-out NASA target",
        "trace_threshold_sigma": TRACE_THRESHOLD,
        "image_metrics": image_results,
        "stratified_image_metrics": {
            name: {
                "station": stratified_image_summary(records, "station"),
                "target_coverage": stratified_image_summary(records, "target_coverage"),
                "axis_profile": stratified_image_summary(records, "axis_profile"),
            }
            for name, records in image_records.items()
        },
        "trace_metrics": {
            name: {trace: trace_summary(items) for trace, items in traces.items()}
            for name, traces in trace_results.items()
        },
        "stratified_trace_metrics": {
            name: {
                "station": {
                    trace: stratified_trace_summary(records, "station")
                    for trace, records in traces.items()
                },
                "target_coverage": {
                    trace: stratified_trace_summary(records, "target_coverage")
                    for trace, records in traces.items()
                },
                "axis_profile": {
                    trace: stratified_trace_summary(records, "axis_profile")
                    for trace, records in traces.items()
                },
            }
            for name, traces in trace_records.items()
        },
        "model_beats_blurred_baseline_correlation": image_results["model"]["macro_correlation"]
        > image_results["blurred_inverted_film"]["macro_correlation"],
        "model_beats_blurred_baseline_mae": image_results["model"]["macro_mae"]
        < image_results["blurred_inverted_film"]["macro_mae"],
        "pair_names": all_names,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
