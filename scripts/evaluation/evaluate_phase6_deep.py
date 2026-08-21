#!/usr/bin/env python3
"""Deep full-held-out evaluation for the strongest Phase 6 models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram  # noqa: E402
from isis_research.evaluation.splits import group_summary  # noqa: E402
from isis_research.grids import TARGET_HEIGHT, resample_grid  # noqa: E402
from scripts.evaluation.benchmark_phase6_512_image_baselines import (  # noqa: E402
    metric as image_metric,
    predictions as baseline_predictions,
)
from scripts.evaluation.evaluate_phase6_512_image_model import (  # noqa: E402
    MAX_HEIGHT_KM,
    path_metric,
    trace_candidate,
    trace_reference,
)
from scripts.evaluation.evaluate_phase6_512_image_model import load_model  # noqa: E402
from scripts.training.train_phase6_512_image_model import load_sample, rows_for  # noqa: E402
from isis_research.models import image_features  # noqa: E402


DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_GROUPS = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_deep_report"
DEFAULT_CHECKPOINTS = {
    "hybrid_unet": ROOT / "outputs/evaluation/phase6_continual_models/hybrid_unet/best_model.pt",
    "norm_residual_unet": ROOT / "outputs/evaluation/phase6_continual_models/norm_residual_unet/best_model.pt",
}
TRACE_NAMES = ("trace_1", "trace_2")
BASELINE_NAMES = ("inverted_film", "blurred_inverted_film")


def rows_for_path(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def correlation(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def calibration(prediction, target, mask):
    left = prediction[mask].astype(float)
    right = target[mask].astype(float)
    if len(left) < 2:
        return {"prediction_mean": None, "target_mean": None, "prediction_std_ratio": None}
    pred_mean = float(left.mean())
    target_mean = float(right.mean())
    pred_std = float(left.std())
    target_std = float(right.std())
    variance = float(np.var(left))
    slope = float(np.cov(left, right, bias=True)[0, 1] / variance) if variance > 1e-12 else 0.0
    intercept = target_mean - slope * pred_mean
    calibrated = slope * left + intercept
    return {
        "prediction_mean": pred_mean,
        "target_mean": target_mean,
        "prediction_std_ratio": pred_std / max(target_std, 1e-9),
        "mean_bias": pred_mean - target_mean,
        "correlation": correlation(left, right),
        "oracle_affine_slope": slope,
        "oracle_affine_intercept": float(intercept),
        "oracle_affine_mae": float(np.mean(np.abs(calibrated - right))),
    }


def metric_record(prediction, target, mask):
    return {
        **image_metric(prediction, target, mask),
        **calibration(prediction, target, mask),
    }


def summarize(items):
    items = [item for item in items if item.get("mae") is not None]
    if not items:
        return {"scans": 0}
    fields = (
        "mae",
        "rmse",
        "correlation",
        "prediction_std_ratio",
        "mean_bias",
        "oracle_affine_mae",
    )
    result = {"scans": len(items)}
    for field in fields:
        values = [item[field] for item in items if item.get(field) is not None]
        if values:
            result[f"mean_{field}"] = float(np.mean(values))
            result[f"median_{field}"] = float(np.median(values))
    result["p90_mae"] = float(np.percentile([item["mae"] for item in items], 90))
    result["mae_gt_0.15"] = int(sum(item["mae"] > 0.15 for item in items))
    result["mae_gt_0.20"] = int(sum(item["mae"] > 0.20 for item in items))
    result["correlation_lt_0.20"] = int(sum(item["correlation"] < 0.20 for item in items))
    return result


def trace_summarize(items):
    fields = (
        "candidate_coverage",
        "median_abs_error_km",
        "p90_abs_error_km",
        "within_60_fraction",
        "within_150_fraction",
        "median_step_km",
    )
    result = {}
    for field in fields:
        values = [item[field] for item in items if item.get("comparable") and field in item]
        result[field] = {
            "scans": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p90": float(np.percentile(values, 90)) if values else None,
        }
    result["reference_detected_scans"] = int(sum(item.get("reference_points", 0) > 0 for item in items))
    result["candidate_comparable_scans"] = int(sum(item.get("comparable") for item in items))
    return result


def stratify(records, key, metric_key="image"):
    groups = {}
    for record in records:
        groups.setdefault(record[key], []).append(record[metric_key])
    summary = summarize if metric_key == "image" else trace_summarize
    return {name: summary(values) for name, values in sorted(groups.items())}


def worst_cases(records, count=15):
    return [
        {
            "pair_name": item["pair_name"],
            "reel": item["reel"],
            "station": item["station"],
            "axis_profile": item["axis_profile"],
            "image": item["image"],
            "trace_1": item["trace_1"],
        }
        for item in sorted(records, key=lambda item: item["image"]["mae"], reverse=True)[:count]
    ]


def group_uncertainty(records):
    rows = [{"reel": record["reel"]} for record in records]
    values = [record["image"]["mae"] for record in records]
    return group_summary(rows, values, key="reel", statistic=np.median, seed=20260818)


def save_partial(path, records, completed):
    path.write_text(
        json.dumps({"schema": "isis.phase6_deep_partial.v1", "completed": completed, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", action="append", metavar="NAME=PATH")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=16)
    args = parser.parse_args()

    checkpoints = dict(DEFAULT_CHECKPOINTS)
    for item in args.checkpoint or []:
        if "=" not in item:
            raise SystemExit("--checkpoint must be NAME=PATH")
        name, path = item.split("=", 1)
        checkpoints[name] = Path(path)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus_rows = rows_for(args.corpus)
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    group_rows = {row["pair_name"]: row for row in rows_for_path(args.groups)}
    held_out = [row for row in corpus_rows if row["split"] == "held_out"]
    if args.limit is not None:
        held_out = held_out[: args.limit]
    models = {}
    for name, path in checkpoints.items():
        model, checkpoint = load_model(path, torch)
        models[name] = {"model": model, "checkpoint": checkpoint, "path": str(path)}

    args.output.mkdir(parents=True, exist_ok=True)
    partial_path = args.output / "partial.json"
    records = {}
    if args.resume and partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        records = partial.get("records", {})
        print(f"resuming {len(records)}/{len(held_out)} completed scans", flush=True)

    for index, row in enumerate(held_out, 1):
        pair_name = row["pair_name"]
        if pair_name in records:
            continue
        scan = ionogram.read_validated(args.corpus / row["csa_artifact"])
        film, target, target_mask = load_sample(args.corpus, args.targets, row, target_rows)
        common = {
            "pair_name": pair_name,
            "reel": row.get("reel") or group_rows.get(pair_name, {}).get("reel", "<unknown>"),
            "station": row.get("station") or "<blank>",
            "axis_profile": f"{scan.frequency_mhz[0]:.1f}-{scan.frequency_mhz[-1]:.1f}MHz",
            "target_coverage": float(target_mask.mean()),
        }
        candidates = baseline_predictions(
            scan.intensity,
            scan.valid_mask,
            0.0,
            include_local=False,
        )
        # The constant mean is not needed for trace or deep image diagnostics.
        candidates = {name: candidates[name] for name in BASELINE_NAMES}
        image = {}
        trace = {}
        for name, candidate in candidates.items():
            image[name] = metric_record(candidate, target, target_mask)
        for name, item in models.items():
            channels = item["checkpoint"].get("input_channels", 1)
            with torch.no_grad():
                prediction = torch.sigmoid(
                    item["model"](
                        torch.from_numpy(image_features(film, channels)[None, ...])
                    )
                ).numpy()[0]
            candidates[name] = prediction.astype(np.float32)
            image[name] = metric_record(candidates[name], target, target_mask)

        target_small = resample_grid(
            target, scan.virtual_height_km, scan.frequency_mhz, fill_value=np.nan
        ).astype(np.float32)
        target_valid_small = (
            resample_grid(
                target_mask.astype(float),
                scan.virtual_height_km,
                scan.frequency_mhz,
                fill_value=0.0,
            )
            > 0.5
        )
        reference = trace_reference(target_small, target_valid_small)
        for name in (*BASELINE_NAMES, *models):
            candidate_small = resample_grid(
                candidates[name],
                scan.virtual_height_km,
                scan.frequency_mhz,
                fill_value=0.0,
            ).astype(np.float32)
            found = trace_candidate(candidate_small)
            trace[name] = {}
            for trace_index, trace_name in enumerate(TRACE_NAMES):
                trace[name][trace_name] = path_metric(
                    found[trace_index],
                    reference[trace_index],
                    TARGET_HEIGHT[TARGET_HEIGHT <= MAX_HEIGHT_KM],
                )
        records[pair_name] = {**common, "image": image, "trace": trace}
        if index == 1 or index % args.checkpoint_every == 0 or index == len(held_out):
            save_partial(partial_path, records, index)
            print(f"deep evaluated {len(records)}/{len(held_out)}: {pair_name}", flush=True)

    candidates = (*BASELINE_NAMES, *models)
    report_models = {}
    for name in candidates:
        model_records = []
        for record in records.values():
            model_records.append(
                {
                    **record,
                    "image": record["image"][name],
                    "trace_1": record["trace"][name]["trace_1"],
                    "trace_2": record["trace"][name]["trace_2"],
                }
            )
        report_models[name] = {
            "image": {
                "summary": summarize([record["image"] for record in model_records]),
                "by_station": stratify(model_records, "station"),
                "by_axis_profile": stratify(model_records, "axis_profile"),
                "group_uncertainty_mae": group_uncertainty(model_records),
                "worst_cases": worst_cases(model_records),
            },
            "trace": {
                trace_name: trace_summarize([record[trace_name] for record in model_records])
                for trace_name in TRACE_NAMES
            },
        }

    report = {
        "schema": "isis.phase6_deep_heldout_evaluation.v1",
        "held_out_scans": len(held_out),
        "models": {name: {"checkpoint": item["path"], "model": item["checkpoint"].get("model")} for name, item in models.items()},
        "baselines": list(BASELINE_NAMES),
        "trace_grid": {"height": 64, "frequency": 96, "max_height_km": MAX_HEIGHT_KM},
        "trace_method": "existing ridge-and-continuity extractor, identical target/candidate settings",
        "reports": report_models,
        "records": records,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Deep Phase 6 held-out evaluation",
        "",
        f"Full held-out set: **{len(held_out)} scans**. The final test split was not changed.",
        "",
        "## Native image metrics",
        "",
        "| Candidate | MAE | RMSE | Correlation | Std ratio | P90 MAE | MAE > 0.20 | Corr < 0.20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in candidates:
        summary = report_models[name]["image"]["summary"]
        lines.append(
            f"| {name} | {summary['mean_mae']:.4f} | {summary['mean_rmse']:.4f} | {summary['mean_correlation']:.4f} | {summary['mean_prediction_std_ratio']:.3f} | {summary['p90_mae']:.4f} | {summary['mae_gt_0.20']} | {summary['correlation_lt_0.20']} |"
        )
    lines.extend(["", "Std ratio below 1 means the output has less contrast than the NASA target. Oracle affine MAE is a diagnostic only; it fits a correction separately on each test scan and is not a valid deployment score.", "", "## Trace metrics", "", "| Candidate | Trace | Comparable scans | Median height error (km) | P90 height error (km) | Within 60 km | Candidate coverage |", "|---|---|---:|---:|---:|---:|---:|"])
    for name in candidates:
        for trace_name in TRACE_NAMES:
            summary = report_models[name]["trace"][trace_name]
            lines.append(
                f"| {name} | {trace_name} | {summary['candidate_comparable_scans']} | {summary['median_abs_error_km']['median'] if summary['median_abs_error_km']['median'] is not None else 'n/a'} | {summary['p90_abs_error_km']['median'] if summary['p90_abs_error_km']['median'] is not None else 'n/a'} | {summary['within_60_fraction']['mean'] if summary['within_60_fraction']['mean'] is not None else 'n/a'} | {summary['candidate_coverage']['mean'] if summary['candidate_coverage']['mean'] is not None else 'n/a'} |"
            )
    lines.extend(["", "## Worst cases", ""])
    for name in models:
        lines.append(f"### {name}")
        for case in report_models[name]["image"]["worst_cases"][:10]:
            lines.append(
                f"- `{case['pair_name']}` — reel `{case['reel']}`, station `{case['station']}`, MAE `{case['image']['mae']:.4f}`, correlation `{case['image']['correlation']:.4f}`"
            )
        lines.append("")
    lines.extend(["## Group uncertainty", ""])
    for name in models:
        overall = report_models[name]["image"]["group_uncertainty_mae"]["overall"]
        low = f"{overall['ci_low']:.4f}" if overall["ci_low"] is not None else "n/a"
        high = f"{overall['ci_high']:.4f}" if overall["ci_high"] is not None else "n/a"
        lines.append(f"- {name}: reel-median MAE `{overall['value']:.4f}` with bootstrap interval `[{low}, {high}]`")
    lines.extend(["", "## Scope", "", "This report evaluates image reconstruction, dynamic range, failure rates, reel/station stratification, and trace geometry. Trace extraction is a secondary diagnostic and does not establish physical correctness by itself.", ""])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
