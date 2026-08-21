#!/usr/bin/env python3
"""Quality-gate Phase 6 calibrations and select the best route per scan.

Phase 7 does not recalibrate or rerender images.  It audits the existing
Phase 3--6 sidecars, assigns a route-level status and score, and selects the
strongest available route for each paired scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isis_research import ionogram  # noqa: E402
DEFAULT_PHASE1_MANIFEST = ROOT / "outputs/calibration/phase1_pairs/manifest.csv"
DEFAULT_STRUCTURE_DIR = ROOT / "outputs/calibration/phase3_structure/json"
DEFAULT_FREQUENCY_DIR = ROOT / "outputs/calibration/phase4_frequency_axis"
DEFAULT_HEIGHT_DIR = ROOT / "outputs/calibration/phase5_height_axis"
DEFAULT_WARP_DIR = ROOT / "outputs/calibration/phase6_warped"
DEFAULT_COMPARISON_DIR = ROOT / "outputs/calibration/phase6_warped/comparisons"
DEFAULT_OUT = ROOT / "outputs/calibration/phase7_quality_gate"

MIN_WARP_COVERAGE = 0.80
MIN_REVIEW_MARKERS = 4
MIN_USABLE_MARKERS = 8
MIN_USABLE_COVERAGE = 0.60
MAX_REVIEW_RMS_PX = 3.0
MAX_REVIEW_MAX_ERROR_PX = 3.0

STATUS_RANK = {"not_usable": 0, "review": 1, "usable": 2}
ROUTE_PREFERENCE = {"cdf_assisted": 1, "film_only": 0}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded(value, low=0.0, high=1.0):
    value = _number(value)
    if value is None:
        return 0.0
    return max(low, min(high, value))


def _append_unique(items, value):
    if value not in items:
        items.append(value)


def _mapping_points(document):
    """Return a mapping's source and target points in stored order."""
    points = document.get("mapping_anchors") or document.get("breakpoints") or []
    result = []
    for item in points:
        row = _number(item.get("film_row"))
        height = _number(item.get("virtual_height_km"))
        if row is None or height is None:
            continue
        result.append((row, height))
    return result


def _axis_points(document, x_key, y_key):
    """Return finite, ordered points for either calibration axis."""
    points = document.get("mapping_anchors") or document.get("breakpoints") or []
    result = []
    for item in points:
        x = _number(item.get(x_key))
        y = _number(item.get(y_key))
        if x is not None and y is not None:
            result.append((x, y))
    result.sort()
    return result


def _inverse_axis(target, points):
    """Invert a strictly increasing axis mapping with endpoint extrapolation."""
    if len(points) < 2:
        raise ValueError("axis mapping has too few points")
    source = np.asarray([point[0] for point in points], dtype=float)
    mapped = np.asarray([point[1] for point in points], dtype=float)
    if np.any(np.diff(source) <= 0) or np.any(np.diff(mapped) <= 0):
        raise ValueError("axis mapping is not strictly increasing")
    target = np.asarray(target, dtype=float)
    result = np.interp(target, mapped, source)
    low = target < mapped[0]
    high = target > mapped[-1]
    result[low] = source[0] + (target[low] - mapped[0]) * (
        (source[1] - source[0]) / (mapped[1] - mapped[0])
    )
    result[high] = source[-1] + (target[high] - mapped[-1]) * (
        (source[-1] - source[-2]) / (mapped[-1] - mapped[-2])
    )
    return result


def _support(values, candidates, tolerance):
    values = np.asarray(values, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if not len(values) or not len(candidates):
        return {"count": int(len(values)), "supported": 0, "fraction": 0.0, "max_distance": None}
    distances = np.min(np.abs(values[:, None] - candidates[None, :]), axis=1)
    return {
        "count": int(len(values)),
        "supported": int(np.sum(distances <= tolerance)),
        "fraction": float(np.mean(distances <= tolerance)),
        "max_distance": float(np.max(distances)),
    }


def _monotonic_mapping(points):
    if len(points) < 2:
        return False
    return all(
        left[0] < right[0] and left[1] < right[1]
        for left, right in zip(points, points[1:])
    )


def _path_from_sidecar(value):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def frequency_quality(document):
    """Score frequency-axis evidence and return independent gate findings."""
    warnings = []
    errors = []
    status = document.get("status")
    if status == "not_usable":
        errors.append("frequency_axis_not_usable")
    elif status == "review":
        warnings.append("frequency_axis_needs_review")

    points = document.get("breakpoints") or []
    if len(points) < 2:
        errors.append("frequency_mapping_has_too_few_breakpoints")
    else:
        columns = [_number(item.get("film_column")) for item in points]
        frequencies = [_number(item.get("frequency_mhz")) for item in points]
        if any(value is None for value in columns + frequencies):
            errors.append("frequency_mapping_contains_nonfinite_values")
        elif not _monotonic_mapping(list(zip(columns, frequencies))):
            errors.append("frequency_mapping_is_not_monotonic")

    coverage = _number(document.get("marker_coverage"))
    rms = _number(document.get("marker_rms_px"))
    maximum = _number(document.get("marker_max_error_px"))
    count = _number(document.get("matched_marker_count"))
    if coverage is None or rms is None or count is None:
        errors.append("frequency_quality_metrics_missing")
    else:
        if coverage < MIN_USABLE_COVERAGE or count < MIN_REVIEW_MARKERS:
            errors.append("frequency_marker_support_is_insufficient")
        elif rms > MAX_REVIEW_RMS_PX:
            errors.append("frequency_fit_residual_is_too_large")
        elif maximum is not None and maximum > MAX_REVIEW_MAX_ERROR_PX:
            warnings.append("frequency_max_residual_needs_review")
        if document.get("warnings"):
            for warning in document["warnings"]:
                if warning in {
                    "partial_frequency_marker_coverage",
                    "large_marker_fit_residual",
                    "few_matched_frequency_markers",
                }:
                    _append_unique(warnings, warning)

    score = (
        0.45 * _bounded(coverage)
        + 0.35 * _bounded(1.0 - (rms or MAX_REVIEW_RMS_PX) / MAX_REVIEW_RMS_PX)
        + 0.20 * _bounded((count or 0.0) / MIN_USABLE_MARKERS)
    )
    if status == "not_usable":
        score *= 0.25
    elif status == "review":
        score *= 0.85
    return {
        "score": round(100.0 * score, 3),
        "marker_count": int(count) if count is not None else None,
        "marker_coverage": coverage,
        "marker_rms_px": rms,
        "marker_max_error_px": maximum,
        "warnings": warnings,
        "errors": errors,
    }


def height_quality(document, structure, route):
    """Check height mapping monotonicity, ruling support, and fallback use."""
    warnings = []
    errors = []
    status = document.get("status")
    if status == "not_usable":
        errors.append("height_axis_not_usable")
    elif status == "review":
        warnings.append("height_axis_needs_review")

    points = _mapping_points(document)
    if len(points) < 2:
        errors.append("height_mapping_has_too_few_anchors")
    elif not _monotonic_mapping(points):
        errors.append("height_mapping_is_not_monotonic")

    mapping = document.get("mapping")
    if route == "cdf_assisted" and mapping != "piecewise_cdf_anchor":
        warnings.append("cdf_height_mapping_fallback")
    if route == "film_only" and mapping != "piecewise_ruling_lattice":
        warnings.append("film_height_mapping_fallback")

    lattice = structure.get("horizontal_rulings", {}).get("lattice", {})
    if lattice.get("status") != "regular_lattice":
        warnings.append("ruling_lattice_needs_review")
    if len(lattice.get("rows", [])) < 3:
        warnings.append("few_horizontal_rulings")
    if structure.get("status") != "structured":
        warnings.append("structure_needs_review")

    image_shape = structure.get("image_shape", [])
    film_region = structure.get("film_region", {})
    if len(image_shape) != 2:
        errors.append("image_shape_is_missing")
    else:
        top = _number(film_region.get("top_row"))
        bottom = _number(film_region.get("bottom_row"))
        if top is None or bottom is None or not (0 <= top < bottom <= image_shape[0] - 1):
            errors.append("film_region_is_invalid")

    height_min = _number(document.get("height_min_km"))
    height_max = _number(document.get("height_max_km"))
    if height_min is not None and height_max is not None and height_max <= height_min:
        errors.append("height_extent_is_not_positive")

    if document.get("warnings"):
        for warning in document["warnings"]:
            if warning in {
                "low_height_profile_sample_count",
                "ruling_spacing_mismatch",
                "no_cdf_horizontal_ruling_match",
                "weak_cdf_horizontal_ruling_match",
            }:
                _append_unique(warnings, warning)

    if not points:
        mapping_score = 0.0
    elif mapping == "piecewise_cdf_anchor":
        mapping_score = 1.0
    elif mapping == "piecewise_ruling_lattice":
        mapping_score = 0.95
    elif mapping == "affine_row_to_height":
        mapping_score = 0.75
    else:
        mapping_score = 0.50
    if structure.get("status") != "structured":
        mapping_score *= 0.90
    if status == "not_usable":
        mapping_score = 0.0
    elif status == "review":
        mapping_score *= 0.85
    return {
        "score": round(100.0 * mapping_score, 3),
        "mapping": mapping,
        "mapping_anchor_count": len(points),
        "warnings": warnings,
        "errors": errors,
    }


def mapping_evidence_quality(structure, frequency, height):
    """Check that fitted anchors land on evidence detected in the CSA image."""
    warnings = []
    errors = []
    tolerance_px = 4.0

    image_shape = structure.get("image_shape", [])
    width = _number(image_shape[1]) if len(image_shape) == 2 else None
    film_region = structure.get("film_region", {})
    film_top = _number(film_region.get("top_row"))
    film_bottom = _number(film_region.get("bottom_row"))

    vertical_candidates = [
        _number(item.get("x"))
        for item in structure.get("vertical_markers", {}).get("candidates", [])
    ]
    vertical_candidates = [item for item in vertical_candidates if item is not None]
    frequency_points = _axis_points(frequency, "film_column", "frequency_mhz")
    frequency_columns = [item[0] for item in frequency_points]
    frequency_support = _support(
        frequency_columns, vertical_candidates, tolerance_px
    )
    if frequency_points and not vertical_candidates:
        errors.append("frequency_mapping_has_no_csa_marker_evidence")
    elif frequency_points:
        if frequency_support["fraction"] < 0.80:
            warnings.append("frequency_mapping_has_weak_csa_marker_support")
        if (frequency_support["max_distance"] or 0.0) > tolerance_px:
            warnings.append("frequency_mapping_has_distant_csa_marker")
    if width is not None and any(column < 0 or column > width - 1 for column in frequency_columns):
        errors.append("frequency_mapping_leaves_csa_image")

    coverage = _number(frequency.get("marker_coverage"))
    if coverage is not None and coverage < 0.70:
        warnings.append("frequency_axis_has_low_confidence_coverage")

    horizontal_candidates = [
        _number(item.get("csa_row"))
        for item in structure.get("horizontal_rulings", {}).get("candidates", [])
    ]
    horizontal_candidates = [item for item in horizontal_candidates if item is not None]
    height_points = _axis_points(height, "film_row", "virtual_height_km")
    interior_rows = [
        row
        for row, _ in height_points
        if film_top is not None
        and film_bottom is not None
        and film_top + tolerance_px < row < film_bottom - tolerance_px
    ]
    height_support = _support(interior_rows, horizontal_candidates, tolerance_px)
    if interior_rows and not horizontal_candidates:
        errors.append("height_mapping_has_no_csa_ruling_evidence")
    elif interior_rows:
        if height_support["fraction"] < 0.80:
            warnings.append("height_mapping_has_weak_csa_ruling_support")
        if (height_support["max_distance"] or 0.0) > tolerance_px:
            warnings.append("height_mapping_has_distant_csa_ruling")

    spacing_error = _number(height.get("ruling_spacing_error_px"))
    if spacing_error is not None and spacing_error > 1.0:
        warnings.append("height_mapping_ruling_spacing_is_inconsistent")

    score = 100.0
    score -= 35.0 * (1.0 - frequency_support["fraction"])
    if interior_rows:
        score -= 35.0 * (1.0 - height_support["fraction"])
    if coverage is not None:
        score -= 20.0 * max(0.0, 0.70 - coverage) / 0.70
    score -= 15.0 * len(warnings)
    if errors:
        score = 0.0
    return {
        "score": round(max(0.0, score), 3),
        "frequency_marker_support": frequency_support,
        "height_ruling_support": height_support,
        "frequency_coverage": coverage,
        "warnings": warnings,
        "errors": errors,
    }


def warp_artifact_quality(warp, frequency, height, structure):
    """Verify that the stored grid and validity mask match the mappings."""
    warnings = []
    errors = []
    path = _path_from_sidecar(warp.get("npz_sidecar"))
    if path is None or not path.is_file() or warp.get("status") == "not_usable":
        return {"score": 0.0 if warp.get("status") == "not_usable" else 100.0, "warnings": warnings, "errors": errors}

    try:
        artifact = ionogram.read_validated(path)
    except (OSError, KeyError, TypeError, ValueError):
        errors.append("warp_npz_artifact_contract_invalid")
        return {"score": 0.0, "warnings": warnings, "errors": errors}
    warped = artifact.intensity
    valid = artifact.valid_mask
    frequency_axis = artifact.frequency_mhz
    height_axis = artifact.virtual_height_km

    if warped.ndim != 2 or valid.shape != warped.shape:
        errors.append("warp_npz_grid_shape_is_inconsistent")
    if warped.ndim == 2 and (len(height_axis) != warped.shape[0] or len(frequency_axis) != warped.shape[1]):
        errors.append("warp_npz_axes_do_not_match_grid")
    if len(frequency_axis) >= 2 and np.any(np.diff(frequency_axis) <= 0):
        errors.append("warp_frequency_grid_is_not_increasing")
    if len(height_axis) >= 2 and np.any(np.diff(height_axis) <= 0):
        errors.append("warp_height_grid_is_not_increasing")

    finite = np.isfinite(warped)
    if np.any(valid & ~finite):
        errors.append("warp_valid_pixels_are_not_finite")
    if np.any(~valid & finite):
        warnings.append("warp_invalid_pixels_are_finite")
    actual_coverage = float(valid.mean()) if valid.size else 0.0
    stored_coverage = _number(warp.get("valid_coverage"))
    if stored_coverage is not None and abs(actual_coverage - stored_coverage) > 1e-3:
        errors.append("warp_coverage_does_not_match_valid_mask")
    if finite.any() and float(np.nanstd(warped[finite])) < 1e-4:
        warnings.append("warp_output_has_no_signal_variation")

    frequency_points = _axis_points(frequency, "film_column", "frequency_mhz")
    height_points = _axis_points(height, "film_row", "virtual_height_km")
    image_shape = structure.get("image_shape", [])
    if len(image_shape) == 2 and len(frequency_axis) >= 2 and len(height_axis) >= 2:
        try:
            source_x = _inverse_axis(frequency_axis, frequency_points)
            source_y = _inverse_axis(height_axis, height_points)
            grid_y, grid_x = np.meshgrid(source_y, source_x, indexing="ij")
            expected_valid = (
                (grid_x >= 0)
                & (grid_x <= image_shape[1] - 1)
                & (grid_y >= 0)
                & (grid_y <= image_shape[0] - 1)
            )
            mismatch = float(np.mean(expected_valid != valid))
            if mismatch > 1e-3:
                errors.append("warp_valid_mask_disagrees_with_axis_mappings")
        except (ValueError, IndexError, TypeError):
            errors.append("warp_grid_cannot_be_reconstructed_from_mappings")

    score = 100.0 - 20.0 * len(warnings) - 50.0 * len(errors)
    return {
        "score": round(max(0.0, score), 3),
        "actual_valid_coverage": actual_coverage,
        "warnings": warnings,
        "errors": errors,
    }


def warp_quality(document):
    """Check the regular-grid warp and its stored artifacts."""
    warnings = []
    errors = []
    status = document.get("status")
    if status == "not_usable":
        errors.append("warp_not_usable")
    elif status == "review":
        warnings.append("warp_needs_review")

    coverage = _number(document.get("valid_coverage"))
    if coverage is None:
        if status != "not_usable":
            errors.append("warp_coverage_is_missing")
    elif coverage < MIN_WARP_COVERAGE:
        errors.append("warp_valid_coverage_below_80_percent")

    frequency_min = _number(document.get("frequency_min_mhz"))
    frequency_max = _number(document.get("frequency_max_mhz"))
    height_min = _number(document.get("height_min_km"))
    height_max = _number(document.get("height_max_km"))
    if status != "not_usable":
        if frequency_min is None or frequency_max is None or frequency_max <= frequency_min:
            errors.append("warp_frequency_extent_is_invalid")
        if height_min is None or height_max is None or height_max <= height_min:
            errors.append("warp_height_extent_is_invalid")
        if height_min is not None and height_min < -1e-6:
            warnings.append("warp_height_starts_below_zero")

        for key in ("npz_sidecar",):
            if _path_from_sidecar(document.get(key)) is None or not _path_from_sidecar(document.get(key)).is_file():
                errors.append(f"warp_{key}_artifact_missing")

    score = _bounded(coverage)
    if status == "not_usable":
        score *= 0.25
    elif status == "review":
        score *= 0.85
    return {
        "score": round(100.0 * score, 3),
        "valid_coverage": coverage,
        "frequency_extent_mhz": [frequency_min, frequency_max],
        "height_extent_km": [height_min, height_max],
        "warnings": warnings,
        "errors": errors,
    }


def evaluate_route(route, structure, frequency, height, warp):
    """Return a route-level Phase 7 report."""
    frequency_check = frequency_quality(frequency)
    height_check = height_quality(height, structure, route)
    warp_check = warp_quality(warp)
    evidence_check = mapping_evidence_quality(structure, frequency, height)
    artifact_check = warp_artifact_quality(warp, frequency, height, structure)
    warnings = []
    errors = []
    for check in (frequency_check, height_check, warp_check, evidence_check, artifact_check):
        for warning in check["warnings"]:
            _append_unique(warnings, warning)
        for error in check["errors"]:
            _append_unique(errors, error)

    # These warnings describe known edge-marker limitations but do not by
    # themselves invalidate a well-supported interior mapping.
    informational = sorted(
        warning
        for warning in set(frequency.get("warnings", []))
        if warning in {"high_frequency_markers_missing", "low_frequency_markers_missing"}
    )

    if errors:
        status = "not_usable"
    elif warnings:
        status = "review"
    else:
        status = "usable"

    score = (
        0.40 * frequency_check["score"]
        + 0.30 * height_check["score"]
        + 0.15 * warp_check["score"]
        + 0.10 * evidence_check["score"]
        + 0.05 * artifact_check["score"]
    )
    critical_warning_count = len(
        [
            item
            for item in warnings
            if item
            not in {"high_frequency_markers_missing", "low_frequency_markers_missing"}
        ]
    )
    score *= max(0.0, 1.0 - 0.05 * critical_warning_count)
    if status == "not_usable":
        score *= 0.25
    elif status == "review":
        score *= 0.85

    return {
        "schema": "isis.csa_quality_gate.v1",
        "route": route,
        "status": status,
        "quality_score": round(score, 3),
        "warnings": sorted(set(warnings)),
        "informational_warnings": informational,
        "errors": sorted(set(errors)),
        "checks": {
            "frequency": frequency_check,
            "height": height_check,
            "warp": warp_check,
            "mapping_evidence": evidence_check,
            "warp_artifact": artifact_check,
            "structure_status": structure.get("status"),
            "structure_warnings": structure.get("warnings", []),
        },
        "source_status": {
            "frequency": frequency.get("status"),
            "height": height.get("status"),
            "warp": warp.get("status"),
        },
    }


def select_route(reports):
    """Select a route, preferring usable quality and CDF on a tie."""
    candidates = [item for item in reports if item["status"] != "not_usable"]
    if not candidates:
        return {
            "status": "not_usable",
            "selected_route": None,
            "reason": "no_route_passed_quality_gate",
            "selected_score": 0.0,
        }

    selected = sorted(
        candidates,
        key=lambda item: (
            STATUS_RANK[item["status"]],
            item["quality_score"],
            ROUTE_PREFERENCE.get(item["route"], 0),
        ),
        reverse=True,
    )[0]
    alternatives = [item for item in reports if item is not selected]
    if selected["status"] == "usable":
        reason = "highest_quality_usable_route"
    else:
        reason = "best_available_route_requires_review"
    if alternatives and all(item["status"] != "not_usable" for item in alternatives):
        if abs(selected["quality_score"] - alternatives[0]["quality_score"]) < 5.0:
            reason = "routes_are_close;_selected_by_quality_and_cdf_tiebreak"
    return {
        "status": selected["status"],
        "selected_route": selected["route"],
        "reason": reason,
        "selected_score": selected["quality_score"],
    }


def _route_record(pair, report, warp, comparison_dir, out_dir):
    stem = f"{int(pair['pair_number']):04d}__{pair['pair_name']}"
    comparison = comparison_dir / report["route"] / f"{stem}_comparison.png"
    return {
        "route": report["route"],
        "pair_number": pair["pair_number"],
        "pair_name": pair["pair_name"],
        "split": pair["split"],
        "status": report["status"],
        "quality_score": report["quality_score"],
        "frequency_status": warp.get("frequency_status"),
        "height_status": warp.get("height_status"),
        "warp_status": warp.get("status"),
        "valid_coverage": warp.get("valid_coverage"),
        "height_mapping": warp.get("height_mapping"),
        "warnings": ";".join(report["warnings"]),
        "informational_warnings": ";".join(report.get("informational_warnings", [])),
        "errors": ";".join(report["errors"]),
        "quality_report": str((out_dir / report["route"] / f"{stem}_quality.json").relative_to(out_dir)),
        "warp_graph": warp.get("graph", ""),
        "comparison_graph": str(comparison) if comparison.is_file() else "",
    }


def process_manifest(
    phase1_manifest=DEFAULT_PHASE1_MANIFEST,
    structure_dir=DEFAULT_STRUCTURE_DIR,
    frequency_dir=DEFAULT_FREQUENCY_DIR,
    height_dir=DEFAULT_HEIGHT_DIR,
    warp_dir=DEFAULT_WARP_DIR,
    comparison_dir=DEFAULT_COMPARISON_DIR,
    out_dir=DEFAULT_OUT,
    limit=None,
):
    phase1_manifest = Path(phase1_manifest)
    structure_dir, frequency_dir, height_dir, warp_dir = map(
        Path, (structure_dir, frequency_dir, height_dir, warp_dir)
    )
    comparison_dir = Path(comparison_dir)
    out_dir = Path(out_dir)
    rows = list(csv.DictReader(phase1_manifest.open(newline="", encoding="utf-8")))
    if limit is not None:
        rows = rows[:limit]

    route_records = []
    final_records = []
    routes = ("cdf_assisted", "film_only")
    for pair in rows:
        stem = f"{int(pair['pair_number']):04d}__{pair['pair_name']}"
        structure = load_json(structure_dir / f"{stem}_structure.json")
        reports = []
        route_warps = {}
        for route in routes:
            frequency = load_json(frequency_dir / route / f"{stem}_frequency.json")
            height = load_json(height_dir / route / f"{stem}_height.json")
            warp = load_json(warp_dir / route / f"{stem}_warp.json")
            report = evaluate_route(route, structure, frequency, height, warp)
            route_dir = out_dir / route
            route_dir.mkdir(parents=True, exist_ok=True)
            quality_path = route_dir / f"{stem}_quality.json"
            quality_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            route_records.append(_route_record(pair, report, warp, comparison_dir, out_dir))
            reports.append(report)
            route_warps[route] = warp

        selected = select_route(reports)
        selected_route = selected["selected_route"]
        selected_warp = route_warps.get(selected_route, {})
        selected_comparison = (
            comparison_dir / selected_route / f"{stem}_comparison.png"
            if selected_route
            else None
        )
        final_report = {
            "schema": "isis.csa_final_route.v1",
            "pair_number": int(pair["pair_number"]),
            "pair_name": pair["pair_name"],
            "split": pair["split"],
            **selected,
            "routes": {
                report["route"]: {
                    "status": report["status"],
                    "quality_score": report["quality_score"],
                    "warnings": report["warnings"],
                    "errors": report["errors"],
                }
                for report in reports
            },
            "selected_warp_json": selected_warp.get("json_sidecar"),
            "selected_warp_graph": selected_warp.get("graph"),
            "selected_comparison_graph": str(selected_comparison)
            if selected_comparison and selected_comparison.is_file()
            else None,
        }
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_path = final_dir / f"{stem}_routing.json"
        final_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        final_records.append(
            {
                "pair_number": pair["pair_number"],
                "pair_name": pair["pair_name"],
                "split": pair["split"],
                "status": selected["status"],
                "selected_route": selected_route or "",
                "selected_score": selected["selected_score"],
                "reason": selected["reason"],
                "cdf_status": reports[0]["status"],
                "cdf_score": reports[0]["quality_score"],
                "film_status": reports[1]["status"],
                "film_score": reports[1]["quality_score"],
                "selected_warp_graph": selected_warp.get("graph", ""),
                "selected_comparison_graph": str(selected_comparison)
                if selected_comparison and selected_comparison.is_file()
                else "",
                "routing_report": str(final_path.relative_to(out_dir)),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    route_fields = list(route_records[0]) if route_records else []
    with (out_dir / "route_quality_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=route_fields)
        writer.writeheader()
        writer.writerows(route_records)
    final_fields = list(final_records[0]) if final_records else []
    with (out_dir / "final_routing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(final_records)
    review_records = [
        record
        for record in final_records
        if record["status"] != "usable"
    ]
    with (out_dir / "review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(review_records)

    (out_dir / "README.md").write_text(
        "# Phase 7 quality gate and routing\n\n"
        f"Audited {len(rows)} scans across the CDF-assisted and film-only routes. "
        "Phase 7 reads existing Phase 3--6 JSON sidecars and does not recalibrate or rerender images.\n\n"
        "A route is `not_usable` when an upstream stage fails, a required mapping is missing or non-monotonic, an axis extent is invalid, the stored warp artifact disagrees with its mappings, or valid warp coverage is below 80%. A route is `review` when it has structural warnings, weak calibration coverage, fitted anchors that do not land on detected CSA lines/rulings, an upstream review status, large marker residuals, or a fallback height mapping. Missing edge markers are retained as information when the interior mapping remains supported.\n\n"
        "`route_quality_manifest.csv` contains one record per route. `final_routing.csv` selects the best available route for each scan, preferring usable over review and then the quality score; CDF-assisted is the deterministic tie-break. `review_queue.csv` contains final scans that still require inspection. The `final/` directory contains one routing JSON per scan.\n",
        encoding="utf-8",
    )
    return route_records, final_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", type=Path, default=DEFAULT_PHASE1_MANIFEST)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--frequency-dir", type=Path, default=DEFAULT_FREQUENCY_DIR)
    parser.add_argument("--height-dir", type=Path, default=DEFAULT_HEIGHT_DIR)
    parser.add_argument("--warp-dir", type=Path, default=DEFAULT_WARP_DIR)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    route_records, final_records = process_manifest(
        args.phase1_manifest,
        args.structure_dir,
        args.frequency_dir,
        args.height_dir,
        args.warp_dir,
        args.comparison_dir,
        args.out_dir,
        args.limit,
    )
    print(
        f"wrote {len(route_records)} route quality reports and "
        f"{len(final_records)} final routing reports to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
