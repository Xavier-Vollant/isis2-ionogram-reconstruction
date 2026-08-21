#!/usr/bin/env python3
"""Prepare the strict usable film-only Phase 6 corpus at native 512x512.

The Phase 6 arrays already use the desired physical grid and exclude the
detected film-region margins.  This command only migrates their legacy NPZ
layout into the validated ionogram contract and verifies that each matched
NASA CDF can be placed on the exact same axes.
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

from isis_research import ionogram  # noqa: E402
from isis_research.nasa.cdf import cdf_amplitude  # noqa: E402


DEFAULT_DATASET = ROOT / "outputs/datasets/amplitude_usable_6400_batch"
DEFAULT_PHASE1 = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
GRID_SHAPE = (512, 512)


def read_rows(dataset_dir, phase1_path):
    with (Path(dataset_dir) / "dataset_index.csv").open(newline="", encoding="utf-8") as handle:
        dataset_rows = list(csv.DictReader(handle))
    with Path(phase1_path).open(newline="", encoding="utf-8") as handle:
        phase1 = {row["pair_name"]: row for row in csv.DictReader(handle)}
    rows = []
    for row in dataset_rows:
        if row["selected_route"] != "film_only":
            continue
        if row["pair_name"] not in phase1:
            raise ValueError(f"missing Phase 1 row for {row['pair_name']}")
        csa = Path(dataset_dir) / row["csa_warped"]
        cdf = Path(dataset_dir) / row["nasa_cdf"]
        if not csa.is_file() or not cdf.is_file():
            raise ValueError(f"missing CSA/CDF source for {row['pair_name']}")
        rows.append({**row, "phase1": phase1[row["pair_name"]], "csa_path": csa, "cdf_path": cdf})
    return rows


def choose_rows(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("limit must be positive")
    rng = np.random.default_rng(seed)
    return [rows[index] for index in np.sort(rng.choice(len(rows), limit, replace=False))]


def validate_pair(row, scan):
    if scan.intensity.shape != GRID_SHAPE:
        raise ValueError(f"{row['pair_name']}: expected {GRID_SHAPE}, got {scan.intensity.shape}")
    if scan.valid_mask.shape != GRID_SHAPE or not scan.valid_mask.any():
        raise ValueError(f"{row['pair_name']}: invalid CSA mask")
    target, target_valid = cdf_amplitude(
        row["cdf_path"], scan.frequency_mhz, scan.virtual_height_km
    )
    if target.shape != GRID_SHAPE or target_valid.shape != GRID_SHAPE:
        raise ValueError(f"{row['pair_name']}: NASA target is not {GRID_SHAPE}")
    if not target_valid.any():
        raise ValueError(f"{row['pair_name']}: NASA has no overlap with the CSA grid")
    return target_valid


def migrate(row, scan, output):
    path = output / "usable" / f"{int(row['pair_number']):04d}__{row['pair_name']}.npz"
    ionogram.write(
        path,
        scan.intensity,
        scan.valid_mask,
        scan.frequency_mhz,
        scan.virtual_height_km,
        status="usable",
        route="film_only",
        confidence=1.0,
        source={
            "pair_name": row["pair_name"],
            "csa_warp": str(row["csa_path"].resolve()),
            "nasa_cdf": portable(row["cdf_path"]),
            "raw_csa": row["phase1"]["csa_source"],
        },
        provenance={
            "producer": "scripts/training/prepare_phase6_usable_512.py",
            "source_route": "phase6_film_only",
            "grid_policy": "preserve_native_512x512_film_region",
        },
    )
    return path


def portable(path):
    """Store NASA paths relative to ROOT so the corpus survives being moved.

    ponytail: only rewrites paths that actually sit under ROOT; a dataset
    symlinked in from another checkout still records an absolute path.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def record(row, scan, target_valid, path, output):
    return {
        "pair_number": row["pair_number"],
        "pair_name": row["pair_name"],
        "split": row["split"],
        "reel": row["phase1"]["reel"],
        "station": row["station"],
        "format_class": row["phase1"].get("format_class", ""),
        "sweep_class": row["phase1"].get("sweep_class", ""),
        "raw_csa": row["phase1"]["csa_source"],
        "csa_artifact": str(path.relative_to(output)),
        "nasa_cdf": portable(row["cdf_path"]),
        "height_max_km": f"{scan.virtual_height_km[-1]:.6f}",
        "csa_valid_fraction": f"{scan.coverage:.6f}",
        "nasa_valid_fraction": f"{float(target_valid.mean()):.6f}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = choose_rows(read_rows(args.dataset, args.phase1), args.limit, args.seed)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise SystemExit(f"output is not empty: {args.output}")

    records = []
    for index, row in enumerate(rows, 1):
        scan = ionogram.read(row["csa_path"])
        target_valid = validate_pair(row, scan)
        path = args.output / "usable" / f"{int(row['pair_number']):04d}__{row['pair_name']}.npz"
        if not (args.resume and path.is_file()):
            path = migrate(row, scan, args.output)
        records.append(record(row, scan, target_valid, path, args.output))
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"prepared {index}/{len(rows)}: {row['pair_name']}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "schema": "isis.phase6_usable_film_only_512.v1",
        "records": len(records),
        "grid_shape": list(GRID_SHAPE),
        "status_policy": "usable_only",
        "route_policy": "film_only_only",
        "cdf_policy": "label_and_alignment_reference_only",
        "height_policy": "preserve_phase6_detected_film_region",
        "train": sum(row["split"] == "train" for row in records),
        "held_out": sum(row["split"] == "held_out" for row in records),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
