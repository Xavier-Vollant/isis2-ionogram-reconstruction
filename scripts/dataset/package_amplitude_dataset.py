#!/usr/bin/env python3
"""Package quality-gated CSA tensors with resampled NASA targets."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cdflib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from isis_research import ionogram
from isis_research.grids import FREQUENCY, HEIGHT
from isis_research.nasa.cdf import _resample_cdf

DEFAULT_FINAL = ROOT / "outputs/calibration/phase7_quality_gate_800/final_routing.csv"
DEFAULT_PAIRS = ROOT / "outputs/calibration/phase1_pairs_800/manifest.csv"
DEFAULT_WARP = ROOT / "outputs/calibration/phase6_warped_scans_800"
DEFAULT_OUT = ROOT / "outputs/datasets/amplitude_usable_500"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    """Parse CLI options and package selected calibrated pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--warp", type=Path, default=DEFAULT_WARP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-target-valid-fraction", type=float, default=0.50)
    parser.add_argument("--min-usable-pairs", type=int, default=500)
    parser.add_argument(
        "--status",
        default="usable",
        choices=("usable", "review", "not_usable"),
        help=(
            "which quality band to package; defaults to usable. "
            "Use review to build a separate comparison corpus."
        ),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"destination already exists: {args.out}")

    final_rows = [row for row in read_csv(args.final) if row["status"] == args.status]
    pair_rows = {row["pair_name"]: row for row in read_csv(args.pairs)}
    if not final_rows:
        raise SystemExit(f"quality gate produced no {args.status} pairs")
    if len(final_rows) < args.min_usable_pairs:
        raise SystemExit(
            f"quality gate produced only {len(final_rows)} {args.status} pairs"
        )

    csa_dir = args.out / "csa_warped"
    cdf_dir = args.out / "nasa_cdf"
    target_dir = args.out / "nasa_targets"
    for directory in (csa_dir, cdf_dir, target_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    for final in final_rows:
        pair = pair_rows.get(final["pair_name"])
        if pair is None:
            failures.append((final["pair_name"], "missing_pair_manifest_row"))
            continue
        stem = f"{int(final['pair_number']):04d}__{final['pair_name']}"
        graph_value = final.get("selected_warp_graph", "")
        graph_path = Path(graph_value) if graph_value else Path()
        if graph_value and not graph_path.is_absolute():
            graph_path = ROOT / graph_path
        csa_source = (
            graph_path.with_name(graph_path.name.replace("_ionogram.png", "_warp.npz"))
            if graph_value
            else Path()
        )
        if not csa_source.is_file():
            csa_source = args.warp / final["selected_route"] / f"{stem}_warp.npz"
        cdf_source = (ROOT / pair["cdf_source"]).resolve()
        if not csa_source.is_file() or not cdf_source.is_file():
            failures.append((final["pair_name"], "missing_csa_or_cdf"))
            continue

        try:
            csa = ionogram.read_validated(csa_source)
            csa_shape = tuple(csa.intensity.shape)
            csa_valid = csa.coverage
            cdf = cdflib.CDF(str(cdf_source))
            target, target_mask = _resample_cdf(
                {
                    "nasa_amplitude": cdf.varget("ampl"),
                    "freq": cdf.varget("freq"),
                    "v_height": cdf.varget("v_height"),
                }
            )
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            failures.append((final["pair_name"], str(error)))
            continue
        csa_link = csa_dir / f"{stem}_warp.npz"
        cdf_link = cdf_dir / f"{final['pair_name']}.cdf"
        csa_link.symlink_to(os.path.relpath(csa_source, csa_link.parent))
        cdf_link.symlink_to(os.path.relpath(cdf_source, cdf_link.parent))
        target_valid_fraction = float(target_mask.mean())
        if target_valid_fraction < args.min_target_valid_fraction:
            failures.append(
                (
                    final["pair_name"],
                    (
                        f"NASA target coverage {target_valid_fraction:.3f} is below "
                        f"{args.min_target_valid_fraction:.3f}"
                    ),
                )
            )
            csa_link.unlink(missing_ok=True)
            cdf_link.unlink(missing_ok=True)
            continue

        target_path = target_dir / f"{stem}_target.npz"
        np.savez_compressed(
            target_path,
            amplitude=target,
            valid_mask=target_mask,
            frequency_mhz=FREQUENCY.astype(np.float32),
            virtual_height_km=HEIGHT.astype(np.float32),
        )
        records.append(
            {
                "pair_number": final["pair_number"],
                "pair_name": final["pair_name"],
                "split": final["split"],
                "selected_route": final["selected_route"],
                "quality_score": final["selected_score"],
                "station": pair.get("station", ""),
                "csa_warped": str(csa_link.relative_to(args.out)),
                "nasa_cdf": str(cdf_link.relative_to(args.out)),
                "nasa_target": str(target_path.relative_to(args.out)),
                "csa_shape": "x".join(str(value) for value in csa_shape),
                "target_shape": "x".join(str(value) for value in target.shape),
                "csa_valid_fraction": f"{csa_valid:.6f}",
                "nasa_valid_fraction": f"{target_valid_fraction:.6f}",
            }
        )

    if not records:
        detail = failures[0][1] if failures else "no packageable rows"
        raise SystemExit(f"could package no {args.status} pairs; first={detail}")
    if failures and len(records) < args.min_usable_pairs:
        raise SystemExit(
            f"could package only {len(records)} pairs; {len(failures)} failed, first={failures[0]}"
        )
    fields = list(records[0])
    with (args.out / "dataset_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    if failures:
        with (args.out / "excluded.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["pair_name", "reason"])
            writer.writerows(failures)
    (args.out / "README.md").write_text(
        "# Usable CSA/NASA amplitude dataset\n\n"
        f"Packaged {len(records)} quality-gated matched pairs. Every row has a selected "
        "512x512 CSA warp, its exact NASA CDF, and a NASA amplitude target "
        f"resampled to the benchmark grid ({len(HEIGHT)} heights x {len(FREQUENCY)} frequencies) "
        f"with at least {args.min_target_valid_fraction:.0%} valid target coverage.\n\n"
        "- `dataset_index.csv` is the training/evaluation index.\n"
        "- `csa_warped/` and `nasa_cdf/` are symlinks to the existing artifacts.\n"
        "- `nasa_targets/` stores normalized `amplitude`, `valid_mask`, `frequency_mhz`, "
        "and `virtual_height_km` arrays.\n"
        f"- Excluded pairs: {len(failures)}; their NASA CDF could not be resampled "
        "safely. See `excluded.csv`.\n"
        "- Splits are reel-level and preserve the quality-gate train/held-out assignment.\n"
        "- The target is NASA's measured `ampl` array; no hand-labeled echo trace is used.\n",
        encoding="utf-8",
    )
    print(f"packaged {len(records)} usable matched pairs under {args.out}")
    print(f"excluded from paired package: {len(failures)}")
    print(f"train={sum(row['split'] == 'train' for row in records)}")
    print(f"held_out={sum(row['split'] == 'held_out' for row in records)}")


if __name__ == "__main__":
    main()
