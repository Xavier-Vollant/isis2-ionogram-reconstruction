#!/usr/bin/env python3
"""Detect structure in raw CSA scans.

This command finds film boundaries and candidates. It does not assign physical
axes or read a CDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isis_research.registration import film, landmarks

DEFAULT_MANIFEST = ROOT / "outputs/calibration/phase1_pairs/manifest.csv"
DEFAULT_OUT = ROOT / "outputs/calibration/phase3_structure"


def normalize(image):
    """Scale an image to display range and report its robust contrast."""
    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image), {"p02": None, "p98": None, "contrast": 0.0}
    low, high = np.nanpercentile(image[finite], [2, 98])
    contrast = float(high - low)
    return np.clip((np.nan_to_num(image) - low) / max(contrast, 1e-9), 0.0, 1.0), {
        "p02": float(low),
        "p98": float(high),
        "contrast": contrast,
    }


def marker_details(image, top, bottom, candidates):
    profile = np.asarray(image[top:bottom], dtype=float).mean(axis=0)
    highpass = profile - film.rolling_median(profile, 61)
    scale = max(float(np.std(highpass)), 1e-9)
    result = []
    for column in np.asarray(candidates, dtype=float):
        index = int(np.clip(round(column), 0, len(highpass) - 1))
        result.append(
            {
                "x": float(column),
                "strength_sigma": float(-highpass[index] / scale),
            }
        )
    return result


def _warnings(image, top, bottom, markers, candidates, lattice, normalization):
    height, width = image.shape
    warnings = []
    if normalization["contrast"] < 8.0:
        warnings.append("low_image_contrast")
    if top <= 0 or bottom >= height - 1:
        warnings.append("film_region_touches_image_edge")
    if len(markers) < 4:
        warnings.append("too_few_vertical_marker_candidates")
    elif len(markers) < 8:
        warnings.append("sparse_vertical_marker_candidates")
    if lattice.get("status") != "regular_lattice" or lattice.get("count", 0) < 3:
        warnings.append("insufficient_horizontal_ruling_lattice")
    if candidates:
        support = [item["support_fraction"] for item in candidates]
        if float(np.median(support)) < 0.5:
            warnings.append("weak_horizontal_support")
    if markers and (
        min(item["x"] for item in markers) < 2
        or max(item["x"] for item in markers) > width - 3
    ):
        warnings.append("marker_candidate_near_image_edge")
    rows = np.asarray(lattice.get("rows", []), dtype=float)
    if len(rows) >= 4:
        spacing = np.diff(rows)
        median = float(np.median(spacing))
        if median and float(np.median(np.abs(spacing - median)) / median) > 0.15:
            warnings.append("irregular_ruling_spacing")
    return sorted(set(warnings))


def extract_structure(image, marker_sigma=2.0, ruling_sigma=2.0):
    """Return the structure record for one grayscale image."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not image.size:
        raise ValueError("expected a non-empty 2-D grayscale image")
    normalized, normalization = normalize(image)
    observed = landmarks.detect_film_features(image, marker_sigma=marker_sigma)
    top = int(observed["top_row"])
    bottom = int(observed["bottom_exclusive"])
    marker_candidates = marker_details(
        image, top, bottom, observed["marker_candidates"]
    )
    marker_x = [item["x"] for item in marker_candidates]
    if len(marker_x) >= 2:
        x_start = int(np.clip(np.floor(min(marker_x)), 0, image.shape[1] - 1))
        x_end = int(np.clip(np.ceil(max(marker_x)), x_start, image.shape[1] - 1))
    else:
        x_start, x_end = 0, image.shape[1] - 1
    horizontal_candidates = landmarks.detect_csa_horizontal_candidates(
        image,
        observed["top_row"],
        observed["bottom_exclusive"],
        x_start,
        x_end,
        sigma=ruling_sigma,
    )
    lattice = landmarks.fit_csa_ruling_lattice(
        horizontal_candidates, observed["top_row"], observed["bottom_row"]
    )
    warnings = _warnings(
        image,
        observed["top_row"],
        observed["bottom_row"],
        marker_candidates,
        horizontal_candidates,
        lattice,
        normalization,
    )
    if normalization["contrast"] < 8.0:
        status = "not_usable"
    elif len(marker_candidates) >= 4 and lattice.get("count", 0) >= 3:
        status = "structured" if not warnings else "review"
    else:
        status = "review"
    return {
        "schema": "isis.csa_scan_structure.v1",
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "normalization": normalization,
        "film_region": {
            "top_row": float(observed["top_row"]),
            "bottom_row": float(observed["bottom_row"]),
            "bottom_exclusive": int(observed["bottom_exclusive"]),
            "analysis_x_start": int(x_start),
            "analysis_x_end": int(x_end),
        },
        "vertical_markers": {
            "count": len(marker_candidates),
            "candidates": marker_candidates,
        },
        "horizontal_rulings": {
            "candidate_count": len(horizontal_candidates),
            "candidates": horizontal_candidates,
            "lattice": lattice,
        },
        "warnings": warnings,
        "status": status,
        "phase_boundary": "candidate landmarks only; no frequency or height labels",
        "normalized_image": normalized,
    }


def write_overlay(path, image, structure, title):
    normalized = structure.pop("normalized_image", None)
    if normalized is None:
        normalized, _ = normalize(image)
    figure, axis = plt.subplots(figsize=(10, 6), dpi=120)
    axis.imshow(normalized, cmap="gray", aspect="auto", interpolation="nearest")
    region = structure["film_region"]
    axis.axhline(region["top_row"], color="#00d084", linewidth=1.4)
    axis.axhline(region["bottom_row"], color="#ff8c00", linewidth=1.4)
    for item in structure["vertical_markers"]["candidates"]:
        axis.axvline(item["x"], color="#2589bd", linewidth=0.8, alpha=0.8)
    for item in structure["horizontal_rulings"]["candidates"]:
        axis.axhline(item["csa_row"], color="#999999", linewidth=0.45, alpha=0.45)
    for row in structure["horizontal_rulings"]["lattice"].get("rows", []):
        axis.axhline(row, color="#d81b60", linewidth=1.2, linestyle="--")
    warnings = ", ".join(structure["warnings"]) or "none"
    axis.set_title(
        f"{title} — {structure['status']} — warnings: {warnings}", fontsize=8
    )
    axis.set_xlabel("film column (candidate frequency markers only)")
    axis.set_ylabel("film row (candidate rulings only)")
    axis.set_xlim(0, image.shape[1] - 1)
    axis.set_ylim(image.shape[0] - 1, 0)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def process_one(film_path, json_path, plot_path, marker_sigma, ruling_sigma, title):
    """Extract one scan, save its JSON record, and write an overlay plot."""
    image = np.asarray(Image.open(film_path).convert("L"), dtype=float)
    structure = extract_structure(image, marker_sigma, ruling_sigma)
    plot_structure = dict(structure)
    plot_structure["film_region"] = dict(structure["film_region"])
    write_overlay(plot_path, image, plot_structure, title)
    structure.pop("normalized_image", None)
    json_path.write_text(json.dumps(structure, indent=2), encoding="utf-8")
    return structure


def process_manifest(
    manifest_path,
    out_dir,
    marker_sigma=2.0,
    ruling_sigma=2.0,
    limit=None,
    write_plots=True,
):
    """Run structure extraction for scans listed in a pair manifest."""
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    structure_dir = out_dir / "json"
    plot_dir = out_dir / "plots"
    structure_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    if limit is not None:
        rows = rows[:limit]
    records = []
    for index, row in enumerate(rows, start=1):
        film_path = (manifest_path.parent / row["csa_link"]).resolve()
        stem = f"{int(row['pair_number']):04d}__{row['pair_name']}"
        if write_plots:
            structure = process_one(
                film_path,
                structure_dir / f"{stem}_structure.json",
                plot_dir / f"{stem}_structure.png",
                marker_sigma,
                ruling_sigma,
                stem,
            )
        else:
            image = np.asarray(Image.open(film_path).convert("L"), dtype=float)
            structure = extract_structure(image, marker_sigma, ruling_sigma)
            structure.pop("normalized_image", None)
            (structure_dir / f"{stem}_structure.json").write_text(
                json.dumps(structure, indent=2), encoding="utf-8"
            )
        records.append(
            {
                "pair_number": row["pair_number"],
                "pair_name": row["pair_name"],
                "split": row["split"],
                "format_class": row["format_class"],
                "status": structure["status"],
                "warnings": ";".join(structure["warnings"]),
                "marker_count": structure["vertical_markers"]["count"],
                "horizontal_candidate_count": structure["horizontal_rulings"][
                    "candidate_count"
                ],
                "lattice_count": structure["horizontal_rulings"]["lattice"].get(
                    "count", 0
                ),
                "lattice_spacing_px": structure["horizontal_rulings"]["lattice"].get(
                    "spacing_px"
                ),
            }
        )
        print(
            f"{index:4d}/{len(rows)} {structure['status']:>10} "
            f"markers={records[-1]['marker_count']:2d} "
            f"rulings={records[-1]['lattice_count']:2d} {row['pair_name']}"
        )
    fields = list(records[0]) if records else []
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (out_dir / "README.md").write_text(
        "# Scan structure\n\n"
        f"Structure extraction for {len(records)} raw CSA scans.\n\n"
        "The JSON contains candidate film boundaries, vertical marker lines, "
        "horizontal ruling candidates, a ruling lattice, and warnings. No "
        "frequency or virtual-height labels are assigned in this phase.\n\n"
        "`plots/` contains diagnostic overlays; `manifest.csv` summarizes the batch.\n",
        encoding="utf-8",
    )
    return records


def main():
    """Parse CLI options and extract candidate structure from scan images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--marker-sigma", type=float, default=2.0)
    parser.add_argument("--ruling-sigma", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.film:
        output = args.out_dir
        output.mkdir(parents=True, exist_ok=True)
        structure = process_one(
            args.film,
            output.with_suffix(".json"),
            output.with_suffix(".png"),
            args.marker_sigma,
            args.ruling_sigma,
            args.film.name,
        )
        print(
            f"status={structure['status']} markers={structure['vertical_markers']['count']} "
            f"rulings={structure['horizontal_rulings']['lattice'].get('count', 0)}"
        )
    else:
        records = process_manifest(
            args.manifest,
            args.out_dir,
            args.marker_sigma,
            args.ruling_sigma,
            args.limit,
            not args.no_plots,
        )
        counts = {}
        for record in records:
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        print(
            "status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
        print(f"wrote {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
