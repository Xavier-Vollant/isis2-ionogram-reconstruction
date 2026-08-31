#!/usr/bin/env python3
"""Prepare NASA image targets for the usable CSA corpus.

Each target is the matched NASA `ampl` image resampled onto the CSA artifact's
axes. Amplitude is normalized to [0, 1], with `valid_mask` marking real CDF
coverage.
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

from isis_research import ionogram
from isis_research.nasa.cdf import cdf_amplitude

DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
GRID_SHAPE = (512, 512)


def read_rows(corpus):
    """Read the usable-corpus manifest."""
    with (Path(corpus) / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_for(row, corpus):
    """Resample one matched NASA CDF onto the CSA artifact's exact grid."""
    scan = ionogram.read_validated(corpus / row["csa_artifact"])
    cdf_path = Path(row["nasa_cdf"])
    if not cdf_path.is_absolute():
        cdf_path = ROOT / cdf_path
    target, valid = cdf_amplitude(cdf_path, scan.frequency_mhz, scan.virtual_height_km)
    if target.shape != GRID_SHAPE or valid.shape != GRID_SHAPE:
        raise ValueError(
            f"{row['pair_name']}: target is {target.shape}, expected {GRID_SHAPE}"
        )
    if not valid.any():
        raise ValueError(f"{row['pair_name']}: NASA target has no valid overlap")
    # CDF amplitudes are stored as 8-bit-like values.  Keep the target finite
    # outside its support, because the mask—not NaN arithmetic—controls loss.
    normalized = np.clip(np.nan_to_num(target / 255.0, nan=0.0), 0.0, 1.0).astype(
        np.float32
    )
    return scan, normalized, valid.astype(bool)


def main():
    """Parse CLI options and prepare NASA image targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = read_rows(args.corpus)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("target corpus manifest contains no rows")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise SystemExit(f"output is not empty: {args.output}")

    records = []
    for index, row in enumerate(rows, 1):
        target_path = (
            args.output / "targets" / f"{Path(row['csa_artifact']).stem}_target.npz"
        )
        if args.resume and target_path.is_file():
            with np.load(target_path, allow_pickle=False) as data:
                target = np.asarray(data["amplitude"], dtype=np.float32)
                valid = np.asarray(data["valid_mask"], dtype=bool)
                frequency = np.asarray(data["frequency_mhz"], dtype=float)
                height = np.asarray(data["virtual_height_km"], dtype=float)
        else:
            scan, target, valid = target_for(row, args.corpus)
            frequency = scan.frequency_mhz
            height = scan.virtual_height_km
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target_path,
                amplitude=target,
                valid_mask=valid,
                frequency_mhz=frequency,
                virtual_height_km=height,
            )
        if target.shape != GRID_SHAPE or valid.shape != GRID_SHAPE:
            raise ValueError(f"{row['pair_name']}: stored target shape mismatch")
        known = target[valid]
        records.append(
            {
                "pair_name": row["pair_name"],
                "split": row["split"],
                "csa_artifact": row["csa_artifact"],
                "nasa_cdf": row["nasa_cdf"],
                "target_artifact": str(target_path.relative_to(args.output)),
                "target_shape": "512x512",
                "target_valid_fraction": f"{float(valid.mean()):.6f}",
                "target_mean": f"{float(np.mean(known)):.6f}",
                "target_std": f"{float(np.std(known)):.6f}",
            }
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(
                f"prepared target {index}/{len(rows)}: {row['pair_name']}", flush=True
            )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema": "isis.phase6_native_512_direct_nasa_targets.v1",
        "records": len(records),
        "grid_shape": list(GRID_SHAPE),
        "target": "NASA_ampl_normalized_to_[0,1]",
        "mask_policy": "only_CDF_supported_pixels_are_loss_valid",
        "trace_labels_used": False,
        "train": sum(row["split"] == "train" for row in records),
        "held_out": sum(row["split"] == "held_out" for row in records),
        "mean_valid_fraction": float(
            np.mean([float(row["target_valid_fraction"]) for row in records])
        ),
        "mean_target_std": float(
            np.mean([float(row["target_std"]) for row in records])
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
