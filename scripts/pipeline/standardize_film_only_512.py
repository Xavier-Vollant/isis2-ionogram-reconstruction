#!/usr/bin/env python3
"""Standardize raw CSA film onto the usable native 512x512 film-only grid.

NASA is deliberately absent from this command.  The frequency and height
profile is loaded from the same calibration artifact that produced the usable
Phase 6 batch, while each input image supplies its own detected film
boundaries and landmarks.  Files are
written only for scans classified ``usable`` so the later model cannot
silently consume a review or metadata-band artifact.
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
from isis_research.image_io import load_image  # noqa: E402
from scripts.pipeline.extract_scan_structure import extract_structure  # noqa: E402
from scripts.pipeline.fit_frequency_axis import fit_from_profile, load_json  # noqa: E402
from scripts.pipeline.fit_height_axis import fit_from_profile as fit_height_from_profile  # noqa: E402
from scripts.pipeline.warp_calibrated_scan import warp_one  # noqa: E402


DEFAULT_FILM_DIR = ROOT / "data/raw/matches/csa_png"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_usable_film_only_512_inference"
GRID_SHAPE = (512, 512)


DEFAULT_PROFILE = ROOT / "configs/film_calibration_profile.json"


def load_profile():
    """Load the same film-only profile used to create the Phase 6 corpus."""
    return load_json(DEFAULT_PROFILE)


def collapse_duplicate_fallback(frequency):
    """Keep a profile fit usable when its only ambiguity is a duplicate fallback.

    With no metadata, Phase 4 presents the selected format profile and its
    format fallback as two candidates.  For the calibrated wide and narrow
    groups those candidates have the same marker positions; treating that
    bookkeeping duplicate as a real ambiguity downgrades an otherwise exact
    Phase 6 fit to ``review``.
    """
    selection = frequency.get("profile_selection", {})
    selected = selection.get("selected", {})
    runner = selected.get("runner_up") or {}
    if (
        frequency.get("status") == "review"
        and runner.get("profile", "").endswith("__fallback")
        and selected.get("marker_count", 0) >= 4
        and selected.get("marker_rms_px", 99.0) <= 1.5
        and selected.get("marker_coverage", 0.0) >= 0.60
        and selected.get("reference_start") == runner.get("reference_start")
        and abs(
            float(selected.get("marker_rms_px", 99.0))
            - float(runner.get("marker_rms_px", -99.0))
        )
        <= 1e-6
    ):
        frequency = dict(frequency)
        frequency["status"] = "usable"
        selection = dict(selection)
        selection["status"] = "selected"
        selection["confidence"] = "high"
        selection["reason"] = "duplicate_format_fallback_collapsed"
        frequency["profile_selection"] = selection
    return frequency


def input_paths(args):
    if args.film:
        return [args.film]
    if args.film_list:
        return [Path(line.strip()) for line in args.film_list.read_text().splitlines() if line.strip()]
    if args.film_dir:
        return sorted(args.film_dir.glob("*.png"))
    raise SystemExit("provide --film, --film-list, or --film-dir")


def process(path, profile, output):
    image = load_image(path)
    structure = extract_structure(image)
    observed = [item["x"] for item in structure["vertical_markers"]["candidates"]]
    frequency = collapse_duplicate_fallback(
        fit_from_profile(observed, image.shape, profile, metadata={})
    )
    height = fit_height_from_profile(structure, profile, frequency)
    if frequency.get("status") == "not_usable" or height.get("status") == "not_usable":
        result = {
            "status": "not_usable",
            "reason": "film-only frequency or height calibration is not usable",
        }
    else:
        result, arrays = warp_one(
            image,
            frequency,
            height,
            structure,
            frequency_bins=GRID_SHAPE[1],
            height_bins=GRID_SHAPE[0],
        )
        if result.get("status") == "usable" and arrays is not None:
            result["warped"] = arrays["warped"]
            result["valid"] = arrays["valid"]
            result["frequency"] = arrays["frequency"]
            result["height"] = arrays["height"]
    row = {
        "film_file": str(path),
        "status": result["status"],
        "reason": result.get("reason", ""),
        "grid_shape": "512x512",
        "frequency_min_mhz": "",
        "frequency_max_mhz": "",
        "height_min_km": "",
        "height_max_km": "",
        "coherence": "",
        "detected": "",
        "artifact": "",
        "frequency_status": frequency.get("status", ""),
        "height_status": height.get("status", ""),
        "structure_status": structure.get("status", ""),
    }
    if result["status"] != "usable":
        return row
    if result["warped"].shape != GRID_SHAPE:
        raise ValueError(f"{path}: standardizer returned {result['warped'].shape}")

    # Phase 6's warp_array already returns the canonical (height, frequency)
    # orientation.  Do not transpose here: the square 512x512 shape would
    # hide that mistake while swapping the physical axes.
    brightness = np.asarray(result["warped"], dtype=np.float32)
    valid = np.asarray(result["valid"], dtype=bool)
    target = output / "usable" / f"{path.stem}.npz"
    ionogram.write(
        target,
        brightness,
        valid,
        result["frequency"],
        result["height"],
        status="usable",
        route="film_only",
        confidence=float(result["confidence"]),
        source={"raw_csa": path.name},
        provenance={
            "producer": "scripts/pipeline/standardize_film_only_512.py",
            "grid_policy": "native_512x512_film_only",
            "cdf_used": False,
        },
    )
    row.update(
        {
            "artifact": str(target.relative_to(output)),
            "frequency_min_mhz": float(result["frequency"][0]),
            "frequency_max_mhz": float(result["frequency"][-1]),
            "height_min_km": float(result["height"][0]),
            "height_max_km": float(result["height"][-1]),
        }
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film", type=Path)
    parser.add_argument("--film-list", type=Path)
    parser.add_argument("--film-dir", type=Path, default=DEFAULT_FILM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.film is not None and args.film_list is not None:
        raise SystemExit("choose only one input source")
    paths = input_paths(args)
    if not paths:
        raise SystemExit("no PNG inputs found")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    profile = load_profile()
    rows = []
    for index, path in enumerate(paths, 1):
        try:
            row = process(path, profile, args.output)
        except Exception as error:  # one bad scan must not hide the others
            row = {
                "film_file": str(path),
                "status": "not_usable",
                "reason": str(error),
                "grid_shape": "512x512",
                "frequency_min_mhz": "",
                "frequency_max_mhz": "",
                "height_min_km": "",
                "height_max_km": "",
                "coherence": "",
                "detected": "",
                "artifact": "",
                "frequency_status": "",
                "height_status": "",
                "structure_status": "",
            }
        rows.append(row)
        print(f"processed {index}/{len(paths)}: {path.name} [{row['status']}]", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "isis.film_only_native_512_inference.v1",
        "records": len(rows),
        "usable": sum(row["status"] == "usable" for row in rows),
        "review": sum(row["status"] == "review" for row in rows),
        "not_usable": sum(row["status"] == "not_usable" for row in rows),
        "grid_shape": list(GRID_SHAPE),
        "profile": str(DEFAULT_PROFILE),
        "cdf_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
