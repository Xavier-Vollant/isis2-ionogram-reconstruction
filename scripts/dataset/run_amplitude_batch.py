#!/usr/bin/env python3
"""Run one small CSA/NASA dataset batch."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_RECORDS = ROOT / "outputs/calibration/phase1_pairs_6400/phase1_records.csv"
DEFAULT_PROFILE = ROOT / "configs/film_calibration_profile.json"


def run(command, env):
    """Run one batch stage from the repository root and fail on errors."""
    print("$ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def next_batch(root):
    """Return the next unused numeric batch identifier under ``root``."""
    numbers = []
    for path in root.glob("batch_*"):
        try:
            numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return max(numbers, default=0) + 1


def main():
    """Parse CLI options and run one amplitude-data batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-id", type=int, default=None)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--base-records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "outputs/amplitude_batches"
    )
    parser.add_argument("--candidates", type=Path, action="append", default=None)
    args = parser.parse_args()
    if args.size < 1:
        raise SystemExit("--size must be positive")

    args.out_root.mkdir(parents=True, exist_ok=True)
    batch_id = args.batch_id or next_batch(args.out_root)
    batch = args.out_root / f"batch_{batch_id:04d}"
    if batch.exists():
        raise SystemExit(f"batch already exists: {batch}; choose another --batch-id")

    pairs = batch / "phase1_pairs"
    landmarks = batch / "landmarks"
    structure = batch / "phase3_structure"
    frequency = batch / "phase4_frequency_axis"
    height = batch / "phase5_height_axis"
    warp = batch / "phase6_warped_scans"
    quality = batch / "phase7_quality_gate"
    dataset = batch / "dataset"
    candidates = args.candidates or sorted(
        (ROOT / "data/processed").glob("review_ranked*.csv")
    )
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise SystemExit("no candidate CSVs found")

    used = [
        path
        for path in args.out_root.glob("batch_*/phase1_pairs/manifest.csv")
        if path.parent.parent != batch
    ]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/private/tmp/isis_mpl"
    python = sys.executable
    prepare = [
        python,
        str(ROOT / "scripts/dataset/prepare_phase1_pairs_target.py"),
        "--existing",
        str(args.base_manifest),
        "--records",
        str(args.base_records),
        "--landmarks",
        str(ROOT / "outputs/landmarks/batch6400"),
        "--target",
        str(args.size),
        "--seed",
        str(args.seed or batch_id),
        "--randomize",
        "--fresh-only",
        "--out",
        str(pairs),
    ]
    for candidate in candidates:
        prepare.extend(("--candidates", str(candidate)))
    for manifest in used:
        prepare.extend(("--used-manifest", str(manifest)))
    run(prepare, env)

    reuse = [ROOT / "outputs/landmarks/batch6400", ROOT / "outputs/landmarks/batch3200"]
    align = [
        python,
        str(ROOT / "scripts/dataset/align_phase1_pairs.py"),
        "--pairs",
        str(pairs / "manifest.csv"),
        "--out",
        str(landmarks),
        "--workers",
        "8",
    ]
    for path in reuse:
        align.extend(("--reuse", str(path)))
    run(align, env)
    run(
        [
            python,
            str(ROOT / "scripts/pipeline/extract_scan_structure.py"),
            "--manifest",
            str(pairs / "manifest.csv"),
            "--out-dir",
            str(structure),
            "--no-plots",
        ],
        env,
    )
    run(
        [
            python,
            str(ROOT / "scripts/pipeline/fit_frequency_axis.py"),
            "--batch",
            "--phase1-manifest",
            str(pairs / "manifest.csv"),
            "--phase1-records",
            str(pairs / "phase1_records.csv"),
            "--structure-dir",
            str(structure / "json"),
            "--landmark-dir",
            str(landmarks),
            "--profile",
            str(args.profile),
            "--out-dir",
            str(frequency),
        ],
        env,
    )
    run(
        [
            python,
            str(ROOT / "scripts/pipeline/fit_height_axis.py"),
            "--phase1-manifest",
            str(pairs / "manifest.csv"),
            "--structure-dir",
            str(structure / "json"),
            "--frequency-dir",
            str(frequency),
            "--landmark-dir",
            str(landmarks),
            "--profile",
            str(args.profile),
            "--out-dir",
            str(height),
            "--route",
            "both",
        ],
        env,
    )
    run(
        [
            python,
            str(ROOT / "scripts/pipeline/warp_calibrated_scan.py"),
            "--manifest",
            str(pairs / "manifest.csv"),
            "--structure-dir",
            str(structure / "json"),
            "--frequency-dir",
            str(frequency),
            "--height-dir",
            str(height),
            "--out-dir",
            str(warp),
            "--route",
            "both",
            "--no-plots",
        ],
        env,
    )
    run(
        [
            python,
            str(ROOT / "scripts/dataset/quality_gate.py"),
            "--phase1-manifest",
            str(pairs / "manifest.csv"),
            "--structure-dir",
            str(structure / "json"),
            "--frequency-dir",
            str(frequency),
            "--height-dir",
            str(height),
            "--warp-dir",
            str(warp),
            "--comparison-dir",
            str(warp / "comparisons"),
            "--out-dir",
            str(quality),
        ],
        env,
    )
    run(
        [
            python,
            str(ROOT / "scripts/dataset/package_amplitude_dataset.py"),
            "--final",
            str(quality / "final_routing.csv"),
            "--pairs",
            str(pairs / "manifest.csv"),
            "--warp",
            str(warp),
            "--out",
            str(dataset),
            "--min-usable-pairs",
            "1",
        ],
        env,
    )
    print(f"completed batch {batch_id:04d}: {dataset}")


if __name__ == "__main__":
    main()
