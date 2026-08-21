#!/usr/bin/env python3
"""Fit Phase 5's film-row to virtual-height mapping.

CDF-assisted results reuse the paired CDF landmark geometry.  Film-only
results use the Phase 1 ruling scale and zero-height offset profile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "configs/film_calibration_profile.json"
DEFAULT_PHASE1_MANIFEST = ROOT / "outputs/calibration/phase1_pairs/manifest.csv"
DEFAULT_STRUCTURE_DIR = ROOT / "outputs/calibration/phase3_structure/json"
DEFAULT_FREQUENCY_DIR = ROOT / "outputs/calibration/phase4_frequency_axis"
DEFAULT_LANDMARK_DIR = ROOT / "outputs/landmarks/batch1500"
DEFAULT_OUT = ROOT / "outputs/calibration/phase5_height_axis"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _median(statistics):
    value = (statistics or {}).get("median")
    return float(value) if value is not None else None


def _map_rows(rows, anchor_rows, anchor_heights):
    """Interpolate/extrapolate monotonically from film rows to heights."""
    rows = np.asarray(rows, dtype=float)
    anchor_rows = np.asarray(anchor_rows, dtype=float)
    anchor_heights = np.asarray(anchor_heights, dtype=float)
    if len(anchor_rows) < 2 or np.any(np.diff(anchor_rows) <= 0):
        raise ValueError("height anchors must have increasing film rows")
    if np.any(np.diff(anchor_heights) <= 0):
        raise ValueError("height anchors must have increasing heights")
    result = np.interp(rows, anchor_rows, anchor_heights)
    low = rows < anchor_rows[0]
    high = rows > anchor_rows[-1]
    low_slope = (anchor_heights[1] - anchor_heights[0]) / (anchor_rows[1] - anchor_rows[0])
    high_slope = (anchor_heights[-1] - anchor_heights[-2]) / (anchor_rows[-1] - anchor_rows[-2])
    result[low] = anchor_heights[0] + (rows[low] - anchor_rows[0]) * low_slope
    result[high] = anchor_heights[-1] + (rows[high] - anchor_rows[-1]) * high_slope
    return result


def _ruling_breakpoints(structure, heights):
    rows = np.asarray(
        structure.get("horizontal_rulings", {}).get("lattice", {}).get("rows", []),
        dtype=float,
    )
    values = _map_rows(rows, heights["anchor_rows"], heights["anchor_heights"])
    return [
        {
            "ruling_index": int(index),
            "film_row": float(row),
            "virtual_height_km": float(value),
        }
        for index, (row, value) in enumerate(zip(rows, values))
    ]


def _profile_mapping_anchors(structure, zero_row, px_per_km, km_per_ruling):
    """Build a restrained piecewise map from the observed ruling lattice.

    The profile supplies the absolute height of the ruling lattice.  The
    detected ruling rows then provide local row positions, allowing the warp
    to follow small nonuniform vertical stretches instead of assuming that
    the whole film has one exact pixel scale.
    """
    if km_per_ruling is None or km_per_ruling <= 0:
        return None
    rows = np.asarray(
        structure.get("horizontal_rulings", {}).get("lattice", {}).get("rows", []),
        dtype=float,
    )
    if len(rows) < 5:
        return None
    step_px = float(px_per_km) * float(km_per_ruling)
    indices = np.rint((rows - float(zero_row)) / step_px).astype(int)
    if np.any(np.diff(indices) <= 0):
        return None
    residuals = rows - (float(zero_row) + indices * step_px)
    if float(np.median(np.abs(residuals))) > max(2.0, 0.30 * step_px):
        return None

    image_bottom = float(structure["image_shape"][0] - 1)
    bottom_height = (image_bottom - float(zero_row)) / float(px_per_km)
    anchor_rows = [float(zero_row)]
    anchor_heights = [0.0]
    for row, index in zip(rows, indices):
        height = float(index * km_per_ruling)
        if row > zero_row and 0.0 < height < bottom_height:
            anchor_rows.append(float(row))
            anchor_heights.append(height)
    anchor_rows.append(image_bottom)
    anchor_heights.append(float(bottom_height))
    if len(anchor_rows) < 3 or np.any(np.diff(anchor_rows) <= 0):
        return None
    return {
        "anchor_rows": np.asarray(anchor_rows, dtype=float),
        "anchor_heights": np.asarray(anchor_heights, dtype=float),
        "residuals_px": residuals.tolist(),
    }


def _cdf_mapping_anchors(structure, baseline_rows, baseline_heights, matches):
    """Add only scored interior CDF matches to the absolute affine map."""
    baseline_rows = np.asarray(baseline_rows, dtype=float)
    baseline_heights = np.asarray(baseline_heights, dtype=float)
    candidates = []
    for item in matches:
        score = item.get("match_score")
        row = item.get("csa_row")
        height = item.get("virtual_height_km")
        if score is None or float(score) < 0.60 or row is None or height is None:
            continue
        row = float(row)
        height = float(height)
        if not (
            baseline_heights[0] < height < baseline_heights[-1]
            and baseline_rows[0] < row < baseline_rows[-1]
        ):
            continue
        candidates.append((height, row, float(score)))
    if not candidates:
        return None

    # Keep the strongest match if two records refer to the same CDF height.
    selected = {}
    for height, row, score in candidates:
        key = round(height, 6)
        if key not in selected or score > selected[key][2]:
            selected[key] = (height, row, score)
    selected = sorted(selected.values())
    rows = [float(baseline_rows[0])] + [item[1] for item in selected] + [float(baseline_rows[-1])]
    heights = [float(baseline_heights[0])] + [item[0] for item in selected] + [float(baseline_heights[-1])]
    if len(rows) < 3 or np.any(np.diff(rows) <= 0) or np.any(np.diff(heights) <= 0):
        return None
    return {
        "anchor_rows": np.asarray(rows, dtype=float),
        "anchor_heights": np.asarray(heights, dtype=float),
        "match_scores": [item[2] for item in selected],
    }


def _base_result(source, frequency_result):
    return {
        "schema": "isis.csa_height_axis.v1",
        "source": source,
        "frequency_status": frequency_result.get("status", "unknown"),
        "warnings": [],
    }


def fit_from_profile(structure, profile, frequency_result):
    """Fit a profile-based absolute height axis for one scan."""
    result = _base_result("film_only_profile", frequency_result)
    selected = frequency_result.get("profile")
    if not selected:
        selected = (
            frequency_result.get("profile_selection", {})
            .get("selected", {})
            .get("profile")
        )
    if not selected:
        result.update(
            {
                "status": "not_usable",
                "confidence": "low",
                "reason": "frequency_route_has_no_selected_height_profile",
            }
        )
        return result

    if selected.endswith("__fallback"):
        group = profile.get("format_fallbacks", {}).get(selected.removesuffix("__fallback"))
    else:
        group = profile.get("profiles", {}).get(selected)
    if not group:
        result.update(
            {
                "status": "not_usable",
                "confidence": "low",
                "profile": selected,
                "reason": "selected_profile_has_no_height_statistics",
            }
        )
        return result

    height = group.get("height", {})
    px_per_km = _median(height.get("px_per_km"))
    top_offset = _median(height.get("top_offset_px"))
    expected_spacing = _median(height.get("ruling_spacing_px"))
    km_per_ruling = _median(height.get("km_per_ruling"))
    lattice = structure.get("horizontal_rulings", {}).get("lattice", {})
    rows = np.asarray(lattice.get("rows", []), dtype=float)
    top_row = float(structure.get("film_region", {}).get("top_row", 0.0))
    result.update(
        {
            "profile": selected,
            "profile_sample_count": int(group.get("sample_count", 0)),
            "mapping": "affine_row_to_height",
            "ruling_count": int(len(rows)),
            "film_top_row": top_row,
            "film_bottom_row": float(
                structure.get("film_region", {}).get("bottom_row", 0.0)
            ),
        }
    )
    if (
        px_per_km is None
        or px_per_km <= 0
        or top_offset is None
        or len(rows) < 3
        or lattice.get("status") != "regular_lattice"
    ):
        result.update(
            {
                "status": "not_usable",
                "confidence": "low",
                "reason": "insufficient_height_profile_or_ruling_lattice",
            }
        )
        return result

    zero_row = top_row + top_offset
    actual_spacing = float(lattice.get("spacing_px")) if lattice.get("spacing_px") is not None else None
    spacing_error = (
        abs(actual_spacing - expected_spacing)
        if actual_spacing is not None and expected_spacing is not None
        else None
    )
    warnings = []
    if frequency_result.get("status") == "not_usable":
        warnings.append("frequency_axis_not_usable")
    elif frequency_result.get("status") == "review":
        warnings.append("frequency_axis_needs_review")
    if group.get("sample_count", 0) < 25:
        warnings.append("low_height_profile_sample_count")
    if spacing_error is not None and expected_spacing is not None:
        if spacing_error > max(3.0, 0.15 * expected_spacing):
            warnings.append("ruling_spacing_mismatch")
    baseline_heights = {
        "anchor_rows": [0.0, float(structure["image_shape"][0] - 1)],
        "anchor_heights": [
            (0.0 - zero_row) / px_per_km,
            (float(structure["image_shape"][0] - 1) - zero_row) / px_per_km,
        ],
    }
    local = _profile_mapping_anchors(
        structure, zero_row, px_per_km, km_per_ruling
    )
    heights = (
        {
            "anchor_rows": local["anchor_rows"],
            "anchor_heights": local["anchor_heights"],
        }
        if local is not None
        else baseline_heights
    )
    breakpoints = _ruling_breakpoints(structure, heights)
    result.update(
        {
            "status": "usable" if not warnings else "review",
            "confidence": "high" if not warnings else "medium",
            "mapping": (
                "piecewise_ruling_lattice"
                if local is not None
                else "affine_row_to_height"
            ),
            "mapping_anchors": [
                {
                    "film_row": float(row),
                    "virtual_height_km": float(height),
                    "source": "observed_ruling_lattice",
                }
                for row, height in zip(
                    heights["anchor_rows"], heights["anchor_heights"]
                )
            ]
            if local is not None
            else None,
            "mapping_anchor_count": int(len(heights["anchor_rows"])),
            "zero_row_px": float(zero_row),
            "px_per_km": float(px_per_km),
            "km_per_ruling": float(km_per_ruling) if km_per_ruling is not None else None,
            "profile_ruling_spacing_px": expected_spacing,
            "observed_ruling_spacing_px": actual_spacing,
            "ruling_spacing_error_px": spacing_error,
            "height_min_km": float(baseline_heights["anchor_heights"][0]),
            "height_max_km": float(baseline_heights["anchor_heights"][1]),
            "breakpoints": breakpoints,
            "warnings": sorted(set(warnings)),
        }
    )
    return result


def fit_from_landmark_reference(structure, frequency_result, landmark_document):
    """Fit a CDF-assisted height axis from stored CDF/film vertical anchors."""
    result = _base_result("cdf_landmark_reference", frequency_result)
    # The CDF geometry's zero row is an actual calibrated zero-height
    # intercept.  The optional vertical_warp starts at the film boundary,
    # which is not necessarily zero height, so use it only as a fallback.
    geometry = landmark_document.get("geometry") or {}
    anchor_rows = np.asarray(geometry.get("vertical_rows", []), dtype=float)
    anchor_heights = np.asarray(geometry.get("vertical_heights", []), dtype=float)
    if len(anchor_rows) < 2:
        zero_row = geometry.get("zero_row")
        px_per_km = geometry.get("px_per_km")
        if zero_row is not None and px_per_km and px_per_km > 0:
            anchor_rows = np.asarray(
                [zero_row, structure["image_shape"][0] - 1], dtype=float
            )
            anchor_heights = np.asarray(
                [0.0, (anchor_rows[1] - zero_row) / px_per_km], dtype=float
            )
    if len(anchor_rows) < 2:
        warp = landmark_document.get("vertical_warp") or {}
        anchor_rows = np.asarray(warp.get("rows", []), dtype=float)
        anchor_heights = np.asarray(warp.get("heights", []), dtype=float)
    result.update(
        {
            "mapping": "affine_row_to_height",
            "cdf_height_anchor_count": int(len(anchor_rows)),
        }
    )
    if (
        len(anchor_rows) < 2
        or len(anchor_rows) != len(anchor_heights)
        or np.any(np.diff(anchor_rows) <= 0)
        or np.any(np.diff(anchor_heights) <= 0)
    ):
        result.update(
            {
                "status": "not_usable",
                "confidence": "low",
                "reason": "missing_or_nonmonotonic_cdf_height_anchors",
            }
        )
        return result

    matches = [
        item
        for item in landmark_document.get("horizontal_matches", [])
        if item.get("status") == "matched_csa_candidate"
    ]
    scores = [float(item["match_score"]) for item in matches if item.get("match_score") is not None]
    warnings = []
    if frequency_result.get("status") == "not_usable":
        warnings.append("frequency_axis_not_usable")
    elif frequency_result.get("status") == "review":
        warnings.append("frequency_axis_needs_review")
    if not matches:
        warnings.append("no_cdf_horizontal_ruling_match")
    elif max(scores, default=0.0) < 0.60:
        warnings.append("weak_cdf_horizontal_ruling_match")
    baseline_heights = {"anchor_rows": anchor_rows, "anchor_heights": anchor_heights}
    local = _cdf_mapping_anchors(
        structure, anchor_rows, anchor_heights, matches
    )
    heights = (
        {
            "anchor_rows": local["anchor_rows"],
            "anchor_heights": local["anchor_heights"],
        }
        if local is not None
        else baseline_heights
    )
    breakpoints = _ruling_breakpoints(structure, heights)
    mapping_anchors = [
        {
            "film_row": float(row),
            "virtual_height_km": float(height),
            "source": "matched_cdf_horizontal_ruling",
        }
        for row, height in zip(heights["anchor_rows"], heights["anchor_heights"])
    ] if local is not None else None
    result.update(
        {
            "status": "usable" if not warnings else "review",
            "confidence": "high" if not warnings else "medium",
            "cdf_horizontal_match_count": int(len(matches)),
            "cdf_horizontal_match_score_max": max(scores) if scores else None,
            "anchor_rows": heights["anchor_rows"].tolist(),
            "anchor_heights_km": heights["anchor_heights"].tolist(),
            "mapping": "piecewise_cdf_anchor" if local is not None else "affine_row_to_height",
            "mapping_anchors": mapping_anchors,
            "mapping_anchor_count": int(len(heights["anchor_rows"])),
            "zero_row_px": float(anchor_rows[0])
            if len(anchor_heights) >= 1 and anchor_heights[0] == 0.0
            else None,
            "px_per_km": float(
                (anchor_rows[-1] - anchor_rows[0])
                / (anchor_heights[-1] - anchor_heights[0])
            ),
            "film_top_row": float(structure["film_region"]["top_row"]),
            "film_bottom_row": float(structure["film_region"]["bottom_row"]),
            "height_min_km": float(_map_rows([0], heights["anchor_rows"], heights["anchor_heights"])[0]),
            "height_max_km": float(
                _map_rows(
                    [structure["image_shape"][0] - 1],
                    heights["anchor_rows"],
                    heights["anchor_heights"],
                )[0]
            ),
            "breakpoints": breakpoints,
            "warnings": sorted(set(warnings)),
        }
    )
    return result


def fit_structure(structure, profile, frequency_result, landmark_document=None):
    if landmark_document is not None:
        return fit_from_landmark_reference(structure, frequency_result, landmark_document)
    return fit_from_profile(structure, profile, frequency_result)


def _landmark_paths(directory):
    return {
        path.name.removesuffix("_landmarks.json"): path
        for path in Path(directory).rglob("*_landmarks.json")
        if not path.name.endswith("_ml_labels.json")
    }


def process_manifest(
    phase1_manifest=DEFAULT_PHASE1_MANIFEST,
    structure_dir=DEFAULT_STRUCTURE_DIR,
    frequency_dir=DEFAULT_FREQUENCY_DIR,
    landmark_dir=DEFAULT_LANDMARK_DIR,
    profile_path=DEFAULT_PROFILE,
    out_dir=DEFAULT_OUT,
    route="both",
    limit=None,
):
    phase1_manifest = Path(phase1_manifest)
    structure_dir, frequency_dir, landmark_dir, out_dir = map(
        Path, (structure_dir, frequency_dir, landmark_dir, out_dir)
    )
    profile = load_json(profile_path)
    landmark_paths = _landmark_paths(landmark_dir)
    rows = list(csv.DictReader(phase1_manifest.open(newline="", encoding="utf-8")))
    if limit is not None:
        rows = rows[:limit]
    routes = ["cdf_assisted", "film_only"] if route == "both" else [route]
    records = []
    for route_name in routes:
        route_dir = out_dir / route_name
        route_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            stem = f"{int(row['pair_number']):04d}__{row['pair_name']}"
            structure = load_json(structure_dir / f"{stem}_structure.json")
            frequency = load_json(frequency_dir / route_name / f"{stem}_frequency.json")
            landmark = (
                load_json(landmark_paths[row["pair_name"]])
                if route_name == "cdf_assisted" and row["pair_name"] in landmark_paths
                else None
            )
            result = fit_structure(structure, profile, frequency, landmark)
            output = route_dir / f"{stem}_height.json"
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            records.append(
                {
                    "route": route_name,
                    "pair_number": row["pair_number"],
                    "pair_name": row["pair_name"],
                    "split": row["split"],
                    "status": result["status"],
                    "source": result["source"],
                    "profile": result.get("profile"),
                    "ruling_count": result.get("ruling_count", len(result.get("breakpoints", []))),
                    "height_anchor_count": result.get("cdf_height_anchor_count", 2),
                    "zero_row_px": result.get("zero_row_px"),
                    "px_per_km": result.get("px_per_km"),
                    "km_per_ruling": result.get("km_per_ruling"),
                    "warnings": ";".join(result.get("warnings", [])),
                    "output": str(output.relative_to(out_dir)),
                }
            )
            print(
                f"{route_name:>12} {int(row['pair_number']):4d}/{len(rows)} "
                f"{result['status']:>10} rulings={len(result.get('breakpoints', [])):2d} "
                f"{row['pair_name']}"
            )
    manifest_out = out_dir / "manifest.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with manifest_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (out_dir / "README.md").write_text(
        "# Phase 5 virtual-height axes\n\n"
        f"Height mappings for {len(rows)} scans and route(s): {', '.join(routes)}.\n\n"
        "CDF-assisted outputs use the absolute CDF/film geometry plus trusted interior CDF horizontal matches when available. Film-only outputs use the Phase 1 profile's ruling scale and observed ruling lattice when it is consistent. Each JSON maps film rows to estimated virtual height in kilometres and records confidence/warnings.\n\n"
        "A `review` or `not_usable` result must not be presented as a precise absolute height calibration.\n",
        encoding="utf-8",
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", type=Path, default=DEFAULT_PHASE1_MANIFEST)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--frequency-dir", type=Path, default=DEFAULT_FREQUENCY_DIR)
    parser.add_argument("--landmark-dir", type=Path, default=DEFAULT_LANDMARK_DIR)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--route", choices=["cdf_assisted", "film_only", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = process_manifest(
        args.phase1_manifest,
        args.structure_dir,
        args.frequency_dir,
        args.landmark_dir,
        args.profile,
        args.out_dir,
        args.route,
        args.limit,
    )
    print(f"wrote {len(records)} height-axis results to {args.out_dir}")


if __name__ == "__main__":
    main()
