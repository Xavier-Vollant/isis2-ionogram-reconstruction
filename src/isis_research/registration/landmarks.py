"""Detect physical landmarks shared by a NASA CDF and CSA scan.

The CDF supplies the coordinates of frequency markers, the sweep start, and
the sampled height bounds. The scan supplies the matching printed lines and
the exposed ionogram edges. This module does not warp either image.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter

from . import film


def _finite(values):
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values) & (values > -1e30)]


def resolve_sweep_start(frequency, stored_start, sample_tolerance=0.02):
    """Return the first sample of the swept frequency, not the fixed prefix.

    NASA sometimes stores ``swept_start`` as the first column of the record
    (or a later 1 MHz column).  The frequency vector is the more direct
    evidence: the sweep starts at its first low-frequency sample, normally
    0.1 MHz.  The stored value remains in the result so bad archive metadata
    is visible to the batch quality gate.
    """
    frequency = np.asarray(frequency, dtype=float).ravel()
    valid = np.isfinite(frequency) & (frequency > 0) & (frequency < 1e6)
    stored = int(np.clip(int(np.asarray(stored_start)), 0, max(len(frequency) - 1, 0)))
    if not np.any(valid):
        return {
            "stored": stored,
            "inferred": None,
            "resolved": stored,
            "source": "stored_swept_start_no_frequency_axis",
            "delta_columns": None,
            "frequency_mhz": None,
            "fixed_prefix_median_mhz": None,
        }

    valid_indices = np.flatnonzero(valid)
    minimum = float(np.min(frequency[valid_indices]))
    # A real sweep begins near 0.1 MHz.  Some sweeps then descend through
    # zero, so the global minimum is a later sample and is not the boundary.
    candidates = valid_indices[
        (frequency[valid_indices] >= 0.02)
        & (frequency[valid_indices] <= 0.10 + max(float(sample_tolerance), 0.02))
    ]
    if not len(candidates):
        tolerance = max(float(sample_tolerance), minimum * 0.12)
        candidates = valid_indices[frequency[valid_indices] <= minimum + tolerance]
    inferred = int(candidates[0]) if len(candidates) else None
    if inferred is None or minimum > 0.30:
        return {
            "stored": stored,
            "inferred": None,
            "resolved": stored,
            "source": "stored_swept_start_frequency_axis_ambiguous",
            "delta_columns": None,
            "frequency_mhz": float(frequency[stored]) if valid[stored] else None,
            "fixed_prefix_median_mhz": None,
        }

    prefix = frequency[valid_indices[valid_indices < inferred]]
    return {
        "stored": stored,
        "inferred": inferred,
        "resolved": inferred,
        "source": "frequency_axis_first_low_frequency_sample",
        "delta_columns": int(stored - inferred),
        "frequency_mhz": float(frequency[inferred]),
        "fixed_prefix_median_mhz": float(np.median(prefix)) if len(prefix) else None,
    }


def nearest_columns(epoch, times):
    """Return the CDF scan-line nearest to each NASA marker time."""
    epoch = np.asarray(epoch, dtype=float).ravel()
    # NASA marker times are relative to the first epoch while CDF Epoch is an
    # absolute TT2000 value.
    epoch = epoch - epoch[0]
    return np.asarray(
        [int(np.argmin(np.abs(epoch - time))) for time in _finite(times)], dtype=int
    )


def nasa_features(amplitude, epoch, time_mark, freq_mark, swept_start, frequency=None):
    """Return only coordinates explicitly present in the CDF."""
    amplitude = np.asarray(amplitude)
    epoch = np.asarray(epoch, dtype=float).ravel()
    time_mark = np.asarray(time_mark, dtype=float).ravel()
    freq_mark = np.asarray(freq_mark, dtype=float).ravel()
    valid = np.isfinite(time_mark) & (time_mark > -1e30)
    valid &= np.isfinite(freq_mark) & (freq_mark > -1e30)
    time_mark, freq_mark = time_mark[valid], freq_mark[valid]
    marker_columns = nearest_columns(epoch, time_mark)
    sweep = (
        resolve_sweep_start(frequency, swept_start)
        if frequency is not None
        else {
            "stored": int(swept_start),
            "inferred": None,
            "resolved": int(swept_start),
            "source": "stored_swept_start_no_frequency_axis",
            "delta_columns": None,
            "frequency_mhz": None,
            "fixed_prefix_median_mhz": None,
        }
    )
    result = {
        "marker_columns": marker_columns,
        "marker_times": time_mark,
        "marker_frequencies": freq_mark,
        "sweep_start": int(np.clip(sweep["resolved"], 0, len(epoch) - 1)),
        "sweep_start_stored": int(np.clip(sweep["stored"], 0, len(epoch) - 1)),
        "sweep_start_inferred": sweep["inferred"],
        "sweep_start_source": sweep["source"],
        "sweep_start_delta_columns": sweep["delta_columns"],
        "sweep_start_frequency_mhz": sweep["frequency_mhz"],
        "fixed_prefix_median_mhz": sweep["fixed_prefix_median_mhz"],
        "top_row": 0.0,
        "bottom_row": float(amplitude.shape[1] - 1),
    }
    if frequency is not None:
        result["frequency"] = np.asarray(frequency, dtype=float).ravel()
    return result


def detect_film_features(image, marker_sigma=2.0):
    """Detect CSA marker lines and the exposed ionogram edges."""
    image = np.asarray(image, dtype=float)
    top, bottom_exclusive = film.film_regions(image)
    markers = film.find_dark_lines(
        image[top:bottom_exclusive].mean(axis=0), 61, marker_sigma, 4
    )

    return {
        "top_row": float(top),
        "bottom_row": float(max(top, bottom_exclusive - 1)),
        "bottom_exclusive": int(bottom_exclusive),
        "marker_candidates": markers,
    }


def _profile_candidates(response, valid, sigma, merge):
    """Return strong, grouped one-dimensional profile features."""
    valid = np.asarray(valid, dtype=bool)
    scale = float(np.std(response[valid])) if np.any(valid) else 0.0
    scale = max(scale, 1e-9)
    hits = np.flatnonzero(valid & (response >= sigma * scale))
    groups = []
    for index in hits:
        if groups and index - groups[-1][-1] <= merge:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])

    candidates = []
    for group in groups:
        weights = response[group]
        peak = group[int(np.argmax(weights))]
        candidates.append(
            {
                "row": float(np.average(group, weights=weights)),
                "peak_row": int(peak),
                "strength_sigma": float(response[peak] / scale),
            }
        )
    return candidates


def detect_nasa_horizontal_rows(
    amplitude, v_height, min_height=500.0, max_height=2500.0, sigma=4.0, max_rows=5
):
    """Find strong horizontal rows that are actually visible in the CDF."""
    profile = np.mean(np.asarray(amplitude, dtype=float), axis=0)
    highpass = profile - median_filter(profile, size=31, mode="nearest")
    valid = np.isfinite(v_height) & (v_height >= min_height) & (v_height <= max_height)
    candidates = _profile_candidates(np.abs(highpass), valid, sigma, 2)
    candidates.sort(key=lambda item: item["strength_sigma"], reverse=True)
    rows = []
    for candidate in candidates:
        row = int(candidate["peak_row"])
        if any(abs(row - int(item["cdf_row"])) < 8 for item in rows):
            continue
        rows.append(
            {
                "cdf_row": row,
                "virtual_height_km": float(v_height[row]),
                "strength": float(abs(highpass[row])),
                "strength_sigma": candidate["strength_sigma"],
                "source": "NASA CDF interior horizontal marking",
            }
        )
        if len(rows) == max_rows:
            break
    return sorted(rows, key=lambda item: item["cdf_row"])


def detect_nasa_middle_marking(amplitude, v_height):
    """Return the strongest interior horizontal NASA row feature."""
    rows = detect_nasa_horizontal_rows(amplitude, v_height, max_rows=5)
    if rows:
        return max(rows, key=lambda item: item["strength_sigma"])

    profile = np.mean(np.asarray(amplitude, dtype=float), axis=0)
    highpass = profile - median_filter(profile, size=31, mode="nearest")
    valid = (v_height >= 500.0) & (v_height <= 2500.0)
    row = int(np.argmax(np.where(valid, np.abs(highpass), -np.inf)))
    return {
        "cdf_row": row,
        "virtual_height_km": float(v_height[row]),
        "strength": float(abs(highpass[row])),
        "strength_sigma": 0.0,
        "source": "NASA CDF interior horizontal marking",
    }


def detect_csa_horizontal_candidates(image, top, bottom, x_start, x_end, sigma=2.0):
    """Find possible CSA horizontal lines in the frequency-marker interval."""
    image = np.asarray(image, dtype=float)
    top, bottom = int(top), int(bottom)
    x_start = int(np.clip(x_start, 0, image.shape[1] - 1))
    x_end = int(np.clip(x_end, x_start, image.shape[1] - 1))
    profile = image[top:bottom, x_start : x_end + 1].mean(axis=1)
    highpass = profile - median_filter(profile, size=41, mode="nearest")
    valid = np.ones(len(profile), dtype=bool)
    candidates = _profile_candidates(-highpass, valid, sigma, 2)
    segments = np.linspace(x_start, x_end + 1, 9, dtype=int)
    result = []
    for candidate in candidates:
        row = int(candidate["peak_row"])
        segment_scores = []
        for left, right in zip(segments[:-1], segments[1:]):
            segment_profile = image[top:bottom, left:right].mean(axis=1)
            segment_highpass = segment_profile - median_filter(
                segment_profile, size=41, mode="nearest"
            )
            scale = max(float(np.std(segment_highpass)), 1e-9)
            segment_scores.append(float(-segment_highpass[row] / scale))
        support = float(np.mean(np.asarray(segment_scores) >= 1.5))
        result.append(
            {
                "csa_row": float(top + row),
                "strength_sigma": candidate["strength_sigma"],
                "support_fraction": support,
                "segment_strength_sigma": float(np.median(segment_scores)),
                "source": "CSA horizontal candidate",
            }
        )
    return result


def fit_csa_ruling_lattice(candidates, top, bottom):
    """Summarize the regularly spaced CSA ruling candidates."""
    rows = sorted(
        float(candidate["csa_row"])
        for candidate in candidates
        if top + 12 <= candidate["csa_row"] <= bottom - 12
    )
    if len(rows) < 3:
        return {"status": "too_few_candidates", "spacing_px": None, "rows": rows}

    spacing = float(np.median(np.diff(rows)))
    regular = []
    for row in rows:
        if regular and row - regular[-1] < 0.55 * spacing:
            previous = next(
                item for item in candidates if item["csa_row"] == regular[-1]
            )
            current = next(item for item in candidates if item["csa_row"] == row)
            if current["strength_sigma"] > previous["strength_sigma"]:
                regular[-1] = row
            continue
        regular.append(row)
    return {
        "status": "regular_lattice",
        "spacing_px": spacing,
        "rows": regular,
        "count": len(regular),
    }


def label_csa_rulings(
    candidates, lattice, nasa_rows, zero_row, px_per_km, height_step_km
):
    """Attach NASA heights to CSA rulings only when all checks agree."""
    lattice_rows = np.asarray(lattice.get("rows", []), dtype=float)
    nasa_rows = list(nasa_rows)
    tolerance = max(2.0 * float(height_step_km), 30.0)
    labels = []
    for candidate in candidates:
        csa_row = float(candidate["csa_row"])
        estimated_height = (csa_row - float(zero_row)) / float(px_per_km)
        if nasa_rows:
            nearest = min(
                nasa_rows,
                key=lambda item: abs(item["virtual_height_km"] - estimated_height),
            )
            height_error = abs(nearest["virtual_height_km"] - estimated_height)
            nasa_agreement = height_error <= tolerance
            nasa_height = float(nearest["virtual_height_km"])
        else:
            height_error = None
            nasa_agreement = False
            nasa_height = None
        spacing_error = (
            float(np.min(np.abs(lattice_rows - csa_row))) if len(lattice_rows) else None
        )
        regular_spacing = spacing_error is not None and spacing_error <= 2.5
        lattice_index = (
            int(np.argmin(np.abs(lattice_rows - csa_row))) if regular_spacing else None
        )
        continuity = bool(
            candidate.get("support_fraction", 0.0) >= 0.75
            and candidate.get("segment_strength_sigma", 0.0) >= 2.5
        )
        reasons = []
        if not continuity:
            reasons.append("weak_continuity")
        if not regular_spacing:
            reasons.append("irregular_spacing")
        if not nasa_agreement:
            reasons.append("no_nasa_height_agreement")
        if not reasons:
            status = "verified_height_ruling"
        elif continuity and regular_spacing:
            status = "relative_ruling_candidate"
        else:
            status = "rejected_csa_candidate"
        labels.append(
            {
                "csa_row": csa_row,
                "estimated_virtual_height_km": float(estimated_height),
                "nearest_nasa_height_km": nasa_height,
                "nasa_height_error_km": (
                    float(height_error) if height_error is not None else None
                ),
                "continuity_score": float(candidate.get("support_fraction", 0.0)),
                "line_strength_sigma": float(
                    candidate.get("segment_strength_sigma", 0.0)
                ),
                "spacing_error_px": spacing_error,
                "lattice_index": lattice_index,
                "continuity_ok": continuity,
                "regular_spacing_ok": regular_spacing,
                "nasa_agreement_ok": nasa_agreement,
                "status": status,
                "weak_label": status == "relative_ruling_candidate",
                "rejection_reasons": reasons,
            }
        )
    verified = [item for item in labels if item["status"] == "verified_height_ruling"]
    verified_index = verified[0]["lattice_index"] if verified else None
    for item in labels:
        item["relative_to_verified_index"] = (
            item["lattice_index"] - verified_index
            if item["lattice_index"] is not None and verified_index is not None
            else None
        )
    return labels


def select_ruling_review_targets(labels, count=3):
    """Choose weak candidates near low, middle, and high lattice positions."""
    candidates = [
        item for item in labels if item["status"] == "relative_ruling_candidate"
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda item: item["csa_row"])
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)))
    selected = []
    used = set()
    for position in positions:
        index = int(round(float(position)))
        item = candidates[index]
        if item["csa_row"] in used:
            continue
        target = dict(item)
        target["review_priority"] = (
            "low"
            if position == positions[0]
            else "high" if position == positions[-1] else "middle"
        )
        selected.append(target)
        used.add(item["csa_row"])
    return selected


ML_CLASSES = [
    {"class_id": 0, "class_name": "background"},
    {"class_id": 1, "class_name": "csa_horizontal_ruling"},
    {"class_id": 2, "class_name": "frequency_marker"},
    {"class_id": 3, "class_name": "start_of_frequency_sweep"},
    {"class_id": 4, "class_name": "top_of_ionogram"},
    {"class_id": 5, "class_name": "bottom_of_ionogram"},
]
_COLUMN_CLASSES = {"frequency_marker": 2, "start_of_frequency_sweep": 3}
_ROW_CLASSES = {"top_of_ionogram": 4, "bottom_of_ionogram": 5}


def _band(center, halfwidth, limit):
    center = int(round(float(center)))
    return max(0, center - halfwidth), min(int(limit) - 1, center + halfwidth)


def build_ml_labels(result, image_shape, halfwidth=1):
    """-> every located CSA landmark in one class list, plus ignore regions.

    The rulings fix the vertical axis, but only the frequency markers carry a
    frequency, so a warp onto a standard grid needs both and the rulings alone
    cannot supply the horizontal half.

    A landmark the detector saw but could not identify is neither a positive
    nor background: a marker predicted off the observed run, a dark line the
    lattice rejected, and the film margin outside the exposed ionogram are all
    places where the absence of a label means unknown, not empty.  Training
    them as negatives teaches the detector to suppress the very lines it
    exists to find.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    labels = list(result.get("csa_consensus_ruling_labels", []))
    ignore = []

    for item in result.get("features", []):
        name = item.get("name")
        if name in _COLUMN_CLASSES:
            position = item.get("csa_x")
            if position is None or not 0.0 <= float(position) <= width - 1.0:
                continue
            start, end = _band(position, halfwidth, width)
            labels.append(
                {
                    "class_id": _COLUMN_CLASSES[name],
                    "class_name": name,
                    "annotation_type": "column_landmark",
                    "label_quality": "verified_frequency",
                    "weak_label": False,
                    "source": "matched_cdf_marker",
                    "csa_x": float(position),
                    "col_start": start,
                    "col_end": end,
                    "frequency_mhz": item.get("frequency_mhz"),
                    "cdf_column": item.get("cdf_column"),
                    "residual_px": item.get("residual_px"),
                    "source_status": item.get("status", "matched_frequency_marker"),
                }
            )
        elif name in _ROW_CLASSES:
            position = item.get("csa_y")
            if position is None or not 0.0 <= float(position) <= height - 1.0:
                continue
            start, end = _band(position, halfwidth, height)
            labels.append(
                {
                    "class_id": _ROW_CLASSES[name],
                    "class_name": name,
                    "annotation_type": "row_landmark",
                    # The edge is where the vertical fit was anchored, so it is
                    # exact by construction and not independently confirmed.
                    "label_quality": "film_boundary",
                    "weak_label": False,
                    "source": "film_data_bounds",
                    "csa_row": float(position),
                    "row_start": start,
                    "row_end": end,
                    "cdf_row": item.get("cdf_row"),
                    "residual_px": item.get("residual_px"),
                }
            )
        elif name == "frequency_marker_unmatched":
            position = item.get("predicted_csa_x")
            if position is None or not 0.0 <= float(position) <= width - 1.0:
                continue
            start, end = _band(position, halfwidth, width)
            ignore.append(
                {
                    "annotation_type": "column_region",
                    "col_start": start,
                    "col_end": end,
                    "reason": "marker_predicted_but_not_observed",
                    "frequency_mhz": item.get("frequency_mhz"),
                    "source_status": item.get("status"),
                }
            )

    lattice_rows = np.asarray(
        (result.get("csa_ruling_lattice") or {}).get("rows", []), dtype=float
    )
    for candidate in result.get("csa_horizontal_candidates", []):
        row = float(candidate["csa_row"])
        if len(lattice_rows) and np.min(np.abs(lattice_rows - row)) <= 2.5:
            continue
        start, end = _band(row, halfwidth, height)
        ignore.append(
            {
                "annotation_type": "row_region",
                "row_start": start,
                "row_end": end,
                "reason": "horizontal_line_outside_lattice",
                "strength_sigma": candidate.get("strength_sigma"),
            }
        )

    film_bounds = result.get("film") or {}
    top = float(film_bounds.get("top_row", 0.0))
    bottom = float(film_bounds.get("bottom_row", height - 1))
    for start, end in ((0, int(top) - 1), (int(math.ceil(bottom)) + 1, height - 1)):
        if start <= end:
            ignore.append(
                {
                    "annotation_type": "row_region",
                    "row_start": start,
                    "row_end": end,
                    "reason": "outside_film_data_bounds",
                }
            )

    return {
        "schema": "isis.csa_landmarks.v2",
        "image_shape": [height, width],
        "background_class_id": 0,
        "classes": ML_CLASSES,
        "labels": labels,
        "ignore": ignore,
    }


def label_consensus_rulings(
    lattice,
    middle_csa_row,
    existing_labels,
    consensus,
    zero_row,
    px_per_km,
    image_height,
    row_halfwidth=1,
):
    """Project stable cross-scan ruling indices onto one CSA scan."""
    rows = np.asarray(lattice.get("rows", []), dtype=float)
    if not len(rows) or middle_csa_row is None:
        return []
    middle_index = int(np.argmin(np.abs(rows - float(middle_csa_row))))
    existing = list(existing_labels)
    labels = []
    for group in consensus.get("rulings", []):
        if group.get("status") != "consensus_relative_ruling":
            continue
        relative_index = int(group["relative_index"])
        lattice_index = middle_index + relative_index
        if not 0 <= lattice_index < len(rows):
            continue
        row = float(rows[lattice_index])
        match = min(existing, key=lambda item: abs(item["csa_row"] - row), default={})
        if not match or abs(float(match["csa_row"]) - row) > 2.5:
            match = {}
        exact = match.get("status") == "verified_height_ruling"
        center = int(round(row))
        labels.append(
            {
                "class_id": 1,
                "class_name": "csa_horizontal_ruling",
                "annotation_type": "row_landmark",
                "label_quality": "verified_height" if exact else "consensus_weak",
                "weak_label": not exact,
                "source": "cross_scan_consensus",
                "relative_index": relative_index,
                "lattice_index": lattice_index,
                "csa_row": row,
                "row_start": max(0, center - int(row_halfwidth)),
                "row_end": min(int(image_height) - 1, center + int(row_halfwidth)),
                "estimated_virtual_height_km": float(
                    (row - float(zero_row)) / float(px_per_km)
                ),
                "estimated_height_is_exact": exact,
                "consensus_scan_count": int(group["scan_count"]),
                "consensus_mad_rulings": float(group["mad_offset_rulings"]),
                "verified_seed_count": int(group.get("verified_seed_count", 0)),
                "source_status": match.get("status", "unmatched_lattice_row"),
            }
        )
    return labels


def score_csa_ruling_local_similarity(
    image,
    amplitude,
    v_height,
    csa_row,
    virtual_height_km,
    csa_x_coefficients,
    exclude_columns=(),
    segments=8,
):
    """Compare local NASA/CSA row-band texture after frequency alignment."""
    image = np.asarray(image, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    v_height = np.asarray(v_height, dtype=float).ravel()
    cdf_row = int(np.argmin(np.abs(v_height - virtual_height_km)))
    cdf_x = np.arange(amplitude.shape[0], dtype=float)
    csa_x = np.polyval(csa_x_coefficients, cdf_x)
    valid = np.isfinite(csa_x) & (csa_x >= 0) & (csa_x <= image.shape[1] - 1)
    for column in exclude_columns:
        valid &= np.abs(cdf_x - column) > 4
    row = int(round(csa_row))
    nasa_band = np.nanmean(
        amplitude[:, max(0, cdf_row - 2) : min(amplitude.shape[1], cdf_row + 3)],
        axis=1,
    )
    csa_band = np.nanmean(
        255.0 - image[max(0, row - 2) : min(image.shape[0], row + 3)],
        axis=0,
    )
    csa_band = np.interp(csa_x, np.arange(image.shape[1]), csa_band)
    scores = []
    edges = np.linspace(0, len(cdf_x), segments + 1, dtype=int)
    for start, end in zip(edges[:-1], edges[1:]):
        keep = valid[start:end]
        if np.sum(keep) < 12:
            continue
        nasa = nasa_band[start:end][keep]
        csa = csa_band[start:end][keep]
        nasa = nasa - median_filter(nasa, size=9, mode="nearest")
        csa = csa - median_filter(csa, size=9, mode="nearest")
        if np.std(nasa) < 1e-9 or np.std(csa) < 1e-9:
            continue
        scores.append(float(np.corrcoef(nasa, csa)[0, 1]))
    return {
        "cdf_row": cdf_row,
        "score": float(np.median(scores)) if scores else None,
        "positive_segment_fraction": (
            float(np.mean(np.asarray(scores) > 0)) if scores else 0.0
        ),
        "segments_used": len(scores),
    }


def match_nasa_rows_to_csa(
    nasa_rows,
    csa_rows,
    y_coefficients,
    max_distance=24.0,
    min_score=0.60,
    min_margin=0.08,
):
    """Keep only CSA rows that are a strong, unambiguous NASA-row match."""
    matches = []
    for nasa in nasa_rows:
        predicted = float(np.polyval(y_coefficients, nasa["virtual_height_km"]))
        scored = []
        for csa in csa_rows:
            distance = abs(float(csa["csa_row"]) - predicted)
            if distance > max_distance:
                continue
            distance_score = 1.0 - distance / max_distance
            strength_score = np.clip((csa["strength_sigma"] - 2.0) / 3.0, 0.0, 1.0)
            support_score = csa.get("support_fraction", 1.0)
            # A continuous, strong CSA line is better evidence than proximity
            # alone. ponytail: replace this heuristic with labelled matches if
            # a larger validated training set becomes available.
            score = 0.25 * distance_score + 0.45 * strength_score + 0.30 * support_score
            scored.append((score, distance, csa))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0] if scored else None
        second_score = scored[1][0] if len(scored) > 1 else -np.inf
        accepted = bool(
            best
            and best[0] >= min_score
            and (len(scored) == 1 or best[0] - second_score >= min_margin)
        )
        match = dict(nasa)
        match.update(
            {
                "predicted_csa_row": predicted,
                "csa_row": float(best[2]["csa_row"]) if best else None,
                "distance_px": float(best[1]) if best else None,
                "match_score": float(best[0]) if best else 0.0,
                "status": (
                    "matched_csa_candidate" if accepted else "no_confident_csa_match"
                ),
            }
        )
        matches.append(match)
    return matches


def align(
    image,
    amplitude,
    epoch,
    time_mark,
    freq_mark,
    swept_start,
    marker_sigma=2.0,
    v_height=None,
    frequency=None,
):
    """Align CDF landmarks to a CSA scan and return geometry plus diagnostics."""
    image = np.asarray(image, dtype=float)
    amplitude = np.asarray(amplitude)
    epoch = np.asarray(epoch, dtype=float).ravel()
    nasa = nasa_features(amplitude, epoch, time_mark, freq_mark, swept_start, frequency)
    observed = detect_film_features(image, marker_sigma)
    x_fit = film.fit_marker_axis(observed["marker_candidates"], nasa["marker_columns"])

    if v_height is None:
        v_height = np.arange(amplitude.shape[1], dtype=float)
    v_height = np.asarray(v_height, dtype=float).ravel()
    reference_heights = np.array([v_height[0], v_height[-1]])
    observed_rows = np.array([observed["top_row"], observed["bottom_row"]])
    y_coefficients = np.polyfit(reference_heights, observed_rows, 1)
    y_residual = observed_rows - np.polyval(y_coefficients, reference_heights)

    marker_reference = nasa["marker_columns"][
        x_fit["reference_start"] : x_fit["reference_start"] + x_fit["count"]
    ]
    marker_observed = observed["marker_candidates"][x_fit["observed_indices"]]
    sweep_expected = float(np.polyval(x_fit["coefficients"], nasa["sweep_start"]))
    nearest = int(np.argmin(np.abs(observed["marker_candidates"] - sweep_expected)))
    sweep_observed = float(observed["marker_candidates"][nearest])
    sweep_residual = sweep_observed - sweep_expected
    sweep_marker_distance = abs(sweep_residual)

    marker_features = []
    for column, frequency, csa_x, residual in zip(
        marker_reference,
        nasa["marker_frequencies"][
            x_fit["reference_start"] : x_fit["reference_start"] + x_fit["count"]
        ],
        marker_observed,
        x_fit["residual"],
    ):
        marker_features.append(
            {
                "name": "frequency_marker",
                "frequency_mhz": float(frequency),
                "cdf_column": float(column),
                "csa_x": float(csa_x),
                "residual_px": float(residual),
            }
        )

    matched_reference = set(marker_reference.astype(int))
    unmatched_marker_features = []
    for column, frequency in zip(nasa["marker_columns"], nasa["marker_frequencies"]):
        if int(column) in matched_reference:
            continue
        predicted = float(np.polyval(x_fit["coefficients"], column))
        unmatched_marker_features.append(
            {
                "name": "frequency_marker_unmatched",
                "frequency_mhz": float(frequency),
                "cdf_column": float(column),
                "predicted_csa_x": predicted,
                "status": "outside_detected_marker_run",
            }
        )

    features = (
        marker_features
        + unmatched_marker_features
        + [
            {
                "name": "start_of_frequency_sweep",
                "cdf_column": float(nasa["sweep_start"]),
                # The nearest film candidate is only a genuine position when
                # it is close enough to trust; otherwise it is frame noise
                # or an unrelated line, and reporting it as "csa_x" would let
                # a consumer draw or use it as if it were a real match, the
                # same way a matched frequency marker's position is used.
                # residual_px is computed from it regardless, since that
                # distance is exactly what identifies the "not found" case.
                "csa_x": (sweep_observed if sweep_marker_distance <= 20.0 else None),
                "predicted_csa_x": sweep_expected,
                "residual_px": float(sweep_residual),
                "distance_to_observed_marker_px": float(sweep_marker_distance),
                "status": (
                    "matched_nearby_marker"
                    if sweep_marker_distance <= 20.0
                    else "no_nearby_marker"
                ),
            },
            {
                "name": "top_of_ionogram",
                "cdf_row": float(nasa["top_row"]),
                "csa_y": float(observed["top_row"]),
                "residual_px": float(y_residual[0]),
            },
            {
                "name": "bottom_of_ionogram",
                "cdf_row": float(nasa["bottom_row"]),
                "csa_y": float(observed["bottom_row"]),
                "residual_px": float(y_residual[1]),
            },
        ]
    )

    line_time = epoch - epoch[0]
    marker_times = nasa["marker_times"][
        x_fit["reference_start"] : x_fit["reference_start"] + x_fit["count"]
    ]
    x_coefficients = np.polyfit(
        line_time[
            nasa["marker_columns"][
                x_fit["reference_start"] : x_fit["reference_start"] + x_fit["count"]
            ]
        ],
        marker_observed,
        1,
    )
    nasa_rows = detect_nasa_horizontal_rows(amplitude, v_height)
    if not nasa_rows:
        nasa_rows = [detect_nasa_middle_marking(amplitude, v_height)]
    csa_rows = detect_csa_horizontal_candidates(
        image,
        observed["top_row"],
        observed["bottom_exclusive"],
        marker_observed[0],
        marker_observed[-1],
    )
    horizontal_matches = match_nasa_rows_to_csa(nasa_rows, csa_rows, y_coefficients)
    csa_lattice = fit_csa_ruling_lattice(
        csa_rows, observed["top_row"], observed["bottom_row"]
    )
    accepted_matches = [
        item for item in horizontal_matches if item["status"] == "matched_csa_candidate"
    ]
    anchor_heights = [float(v_height[0])]
    anchor_rows = [float(observed["top_row"])]
    for item in accepted_matches:
        anchor_heights.append(float(item["virtual_height_km"]))
        anchor_rows.append(float(item["csa_row"]))
    anchor_heights.append(float(v_height[-1]))
    anchor_rows.append(float(observed["bottom_row"]))
    order = np.argsort(anchor_heights)
    anchor_heights = np.asarray(anchor_heights)[order]
    anchor_rows = np.asarray(anchor_rows)[order]
    if (
        len(accepted_matches) >= 1
        and np.all(np.diff(anchor_heights) > 0)
        and np.all(np.diff(anchor_rows) > 0)
    ):
        vertical_warp = {
            "status": "piecewise_linear",
            "anchor_count": len(anchor_heights),
            "heights": anchor_heights,
            "rows": anchor_rows,
        }
    else:
        vertical_warp = {
            "status": "linear_insufficient_verified_rows",
            "anchor_count": 2,
            "heights": np.asarray([v_height[0], v_height[-1]], dtype=float),
            "rows": np.asarray(
                [observed["top_row"], observed["bottom_row"]], dtype=float
            ),
        }
    nasa_middle = max(nasa_rows, key=lambda item: item["strength_sigma"])
    nasa_middle = next(
        item for item in horizontal_matches if item["cdf_row"] == nasa_middle["cdf_row"]
    )
    return {
        "nasa": nasa,
        "film": observed,
        "x_fit": x_fit,
        "y_coefficients": y_coefficients,
        "y_rms_px": float(np.sqrt(np.mean(y_residual**2))),
        "y_residual": y_residual,
        "geometry": {
            "coefficients": x_coefficients,
            "zero_row": float(y_coefficients[1]),
            "px_per_km": float(y_coefficients[0]),
            "vertical_heights": vertical_warp["heights"],
            "vertical_rows": vertical_warp["rows"],
        },
        "features": features,
        "marker_reference": marker_reference,
        "marker_observed": marker_observed,
        "marker_times": marker_times,
        "nasa_horizontal_rows": nasa_rows,
        "csa_horizontal_candidates": csa_rows,
        "csa_ruling_lattice": csa_lattice,
        "horizontal_matches": horizontal_matches,
        "vertical_warp": vertical_warp,
        "nasa_middle_marking": nasa_middle,
    }
