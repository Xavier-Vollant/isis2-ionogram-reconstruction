#!/usr/bin/env python3
"""Run the CDF landmark aligner over a Phase 1 pair manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALIGN = ROOT / "scripts/dataset/align_landmarks.py"
DEFAULT_PAIRS = ROOT / "outputs/calibration/phase1_pairs_2000/manifest.csv"
DEFAULT_OUT = ROOT / "outputs/landmarks/batch2000"


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--reuse",
        type=Path,
        action="append",
        default=[ROOT / "outputs/landmarks/batch1500", ROOT / "outputs/landmarks/batch1000", ROOT / "outputs/landmarks/batch100"],
        help="landmark directories to reuse before running the aligner",
    )
    args = parser.parse_args()
    rows = read_rows(args.pairs)
    args.out.mkdir(parents=True, exist_ok=True)

    def process(row):
        pair_dir = args.out / row["pair_name"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        figure = pair_dir / f"{row['pair_name']}_landmarks.png"
        result_path = figure.with_suffix(".json")
        if result_path.exists():
            return {"pair_name": row["pair_name"], "status": "existing"}
        for reuse_dir in args.reuse:
            matches = list(Path(reuse_dir).rglob(f"{row['pair_name']}_landmarks.json"))
            if matches:
                result_path.symlink_to(os.path.relpath(matches[0], result_path.parent))
                return {"pair_name": row["pair_name"], "status": "reused"}
        film = (args.pairs.parent / row["csa_link"]).resolve()
        cdf = (args.pairs.parent / row["cdf_link"]).resolve()
        command = [
            sys.executable,
            str(ALIGN),
            "--film",
            str(film),
            "--cdf",
            str(cdf),
            "--out",
            str(figure),
            "--fast",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        record = {"pair_name": row["pair_name"], "status": "ok" if completed.returncode == 0 else "failed"}
        if completed.returncode != 0:
            record["error"] = (completed.stderr or completed.stdout).strip().splitlines()[-1]
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process, rows))
    fields = sorted({key for result in results for key in result})
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    counts = {status: sum(item["status"] == status for item in results) for status in sorted({item["status"] for item in results})}
    print(f"processed {len(results)} pairs: {counts}")


if __name__ == "__main__":
    main()
