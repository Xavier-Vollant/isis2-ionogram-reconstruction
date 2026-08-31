#!/usr/bin/env python3
"""Benchmark non-learned CSA-to-NASA image baselines on held-out pairs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram

DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_512_image_baselines.json"


def rows_for(corpus):
    with (Path(corpus) / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_for(targets, row, target_rows):
    target_row = target_rows[row["pair_name"]]
    with np.load(
        Path(targets) / target_row["target_artifact"], allow_pickle=False
    ) as data:
        return np.asarray(data["amplitude"], dtype=np.float32), np.asarray(
            data["valid_mask"], dtype=bool
        )


def normalize(values, valid):
    finite = valid & np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(values[finite], [2.0, 98.0])
    return np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0).astype(
        np.float32
    )


def predictions(film, valid, train_mean, include_local=True):
    """Build non-learned image predictions for one film scan."""
    signal = np.where(valid, 1.0 - film, 0.0).astype(np.float32)
    candidates = {
        "constant_train_mean": np.full_like(signal, train_mean),
        "inverted_film": signal,
        "inverted_film_normalized": normalize(signal, valid),
        "blurred_inverted_film": gaussian_filter(signal, sigma=1.2).astype(np.float32),
    }
    if include_local:
        local = signal - median_filter(signal, size=(9, 9), mode="nearest")
        candidates["local_contrast_normalized"] = normalize(local, valid)
    return candidates


def metric(prediction, target, valid):
    """Compare one prediction with its NASA target on jointly valid pixels."""
    mask = valid & np.isfinite(prediction) & np.isfinite(target)
    left = prediction[mask].astype(float)
    right = target[mask].astype(float)
    if len(left) < 2:
        return {"pixels": len(left), "mae": None, "rmse": None, "correlation": None}
    centred_left = left - left.mean()
    centred_right = right - right.mean()
    denominator = np.linalg.norm(centred_left) * np.linalg.norm(centred_right)
    return {
        "pixels": len(left),
        "mae": float(np.mean(np.abs(left - right))),
        "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
        "correlation": float(np.dot(centred_left, centred_right) / denominator)
        if denominator > 0
        else 0.0,
    }


def main():
    """Parse CLI options and benchmark non-learned held-out baselines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-held-out", type=int, default=None)
    args = parser.parse_args()

    rows = rows_for(args.corpus)
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    train_rows = [row for row in rows if row["split"] == "train"]
    held_out = [row for row in rows if row["split"] == "held_out"]
    if args.limit_held_out is not None:
        held_out = held_out[: args.limit_held_out]

    train_sum = 0.0
    train_count = 0
    for row in train_rows:
        target, valid = target_for(args.targets, row, target_rows)
        values = target[valid]
        train_sum += float(values.sum())
        train_count += int(values.size)
    train_mean = train_sum / max(train_count, 1)

    results = {}
    for count, row in enumerate(held_out, 1):
        scan = ionogram.read_validated(args.corpus / row["csa_artifact"])
        target, target_valid = target_for(args.targets, row, target_rows)
        valid = scan.valid_mask & target_valid
        for name, prediction in predictions(
            scan.intensity, scan.valid_mask, train_mean
        ).items():
            results.setdefault(name, []).append(metric(prediction, target, valid))
        if count == 1 or count % 100 == 0 or count == len(held_out):
            print(
                f"benchmarked {count}/{len(held_out)}: {row['pair_name']}", flush=True
            )

    summary = {
        "schema": "isis.phase6_native_512_image_baselines.v1",
        "held_out_scans": len(held_out),
        "train_scans_for_constant": len(train_rows),
        "train_mean_target": train_mean,
        "metrics": {},
    }
    for name, values in results.items():
        summary["metrics"][name] = {
            "macro_mae": float(
                np.mean([item["mae"] for item in values if item["mae"] is not None])
            ),
            "macro_rmse": float(
                np.mean([item["rmse"] for item in values if item["rmse"] is not None])
            ),
            "macro_correlation": float(
                np.mean(
                    [
                        item["correlation"]
                        for item in values
                        if item["correlation"] is not None
                    ]
                )
            ),
            "pixels": int(sum(item["pixels"] for item in values)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
