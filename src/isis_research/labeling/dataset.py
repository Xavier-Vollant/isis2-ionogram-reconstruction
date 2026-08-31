"""Build training arrays for a film-only landmark model.

The alignment pipeline writes one `*_ml_labels.json` file per scan. This module
turns those labels into profiles, targets, weights, and masks. `ignore` regions
remain unknown rather than becoming negatives, and weak ruling labels are
down-weighted and excluded from scoring. Verified height labels are the only
independent ruling labels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from isis_research.image_io import load_image

# Landmark positions are resampled onto fixed-length axes. Scan width varies
# 1.58x within a single reel (~835 px against ~1321 px for the same film frame,
# at identical height), so a column measured in pixels is not comparable between
# two scans of the same reel. Resampling puts both on the film's own axis.
ROW_LEN = 400
COL_LEN = 1024

# Highpass windows, in resampled samples. Rulings sit ~22 px apart and markers
# are thinner and denser, which is why the two axes do not share a window.
ROW_MEDIAN = 41
COL_MEDIAN = 61

LABEL_SCHEMA = "isis.csa_landmarks.v2"

ROW_CLASSES = ("csa_horizontal_ruling", "top_of_ionogram", "bottom_of_ionogram")
COL_CLASSES = ("frequency_marker", "start_of_frequency_sweep")

# Marker fit, not the usable gate. The usable gate scores echo agreement between
# the archives; a landmark model needs scans whose marker geometry was confirmed.
MIN_MARKERS = 8
MAX_MARKER_RMS_PX = 3.0


def load_manifest(batch_dir):
    """Load alignment metrics keyed by scan name."""
    path = Path(batch_dir) / "quality_manifest.json"
    records = json.loads(path.read_text())
    return {record["name"]: record.get("metrics", {}) for record in records}


def passes_marker_gate(metrics):
    """Return whether marker count and residual pass the training gate."""
    count = metrics.get("marker_count") or 0
    rms = metrics.get("marker_rms_px")
    return count >= MIN_MARKERS and rms is not None and rms <= MAX_MARKER_RMS_PX


def load_pair_rows(review_csvs):
    """Load review rows grouped by scan name."""
    rows = {}
    for path in review_csvs:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = Path(row["nasa_png_file"]).stem.split("_", 1)[1]
                rows.setdefault(name, []).append(row)
    return rows


def _pair_for_scan(candidates, name, shape):
    """Find the review row that produced a scan's labels."""
    if "__" in name:
        film_stem = name.split("__", 1)[1]
        for row in candidates:
            if Path(row["csa_file"]).stem == film_stem:
                return row
    height, width = shape
    for row in candidates:
        if row.get("csa_size") == f"{width}x{height}":
            return row
    return None


def index_scans(batch_dirs, review_csvs, film_dir):
    """Return one indexed record per scan that passes the training gates."""
    film_dir = Path(film_dir)
    pairs = load_pair_rows(review_csvs)
    scans = {}
    for batch_dir in batch_dirs:
        batch_dir = Path(batch_dir)
        metrics = load_manifest(batch_dir)
        for labels_path in sorted(batch_dir.glob("*/*/*_ml_labels.json")):
            name = labels_path.name[: -len("_landmarks_ml_labels.json")]
            if name in scans or not passes_marker_gate(metrics.get(name, {})):
                continue
            document = json.loads(labels_path.read_text())
            if document.get("schema") != LABEL_SCHEMA:
                continue
            nasa_id = name.split("__", 1)[0]
            pair = _pair_for_scan(pairs.get(nasa_id, []), name, document["image_shape"])
            if pair is None:
                continue
            film = film_dir / pair["csa_file"]
            if not film.exists():
                continue
            scans[name] = {
                "name": name,
                "labels_path": labels_path,
                "film_path": film,
                "reel": pair["csa_film_subdir"],
                "station": pair["csa_station"],
                "year": pair["csa_timestamp_utc"][:4],
            }
    return [scans[name] for name in sorted(scans)]


def resample(profile, length):
    """Resample a one-dimensional profile to a fixed length."""
    source = np.linspace(0.0, 1.0, len(profile))
    return np.interp(np.linspace(0.0, 1.0, length), source, profile)


def to_axis(position, native_length, length):
    """Map a native pixel coordinate to a resampled axis."""
    return float(position) * (length - 1) / max(native_length - 1, 1)


def to_native(position, native_length, length):
    """Map a resampled coordinate back to native pixels."""
    return np.asarray(position, dtype=float) * max(native_length - 1, 1) / (length - 1)


def highpass(profile, size):
    """Return local darkness after removing broad film variation."""
    return median_filter(profile, size=size, mode="nearest") - profile


def robust_scale(values):
    """Return a nonzero scale based on median absolute deviation."""
    return max(1.4826 * float(np.median(np.abs(values - np.median(values)))), 1e-9)


def row_profile(image):
    """Return the mean value for each image row."""
    return image.mean(axis=1)


def column_profile(image, top, bottom):
    """Return column means inside the exposed data area."""
    top = int(np.clip(top, 0, image.shape[0] - 1))
    bottom = int(np.clip(bottom, top + 1, image.shape[0]))
    return image[top:bottom].mean(axis=0)


def _positions(labels, class_name, key, native_length, length):
    return [
        to_axis(item[key], native_length, length)
        for item in labels
        if item["class_name"] == class_name and item.get(key) is not None
    ]


def _mark(target, weight, centre, band, value):
    lo = int(np.floor(centre - band))
    hi = int(np.ceil(centre + band))
    index = np.arange(max(lo, 0), min(hi, len(target) - 1) + 1)
    if len(index):
        target[index] = 1.0
        weight[index] = np.maximum(weight[index], value)


def build_targets(
    labels,
    ignore,
    classes,
    native_length,
    length,
    key,
    region_type,
    weak_weight=0.3,
    band=1.0,
    halo=2.0,
):
    """Build class targets, fitting weights, and a loss mask."""
    targets = {name: np.zeros(length) for name in classes}
    weights = {name: np.zeros(length) for name in classes}
    mask = np.ones(length)

    for item in labels:
        name = item["class_name"]
        if name not in classes or item.get(key) is None:
            continue
        centre = to_axis(item[key], native_length, length)
        if not 0.0 <= centre <= length - 1:
            continue
        value = weak_weight if item.get("weak_label") else 1.0
        _mark(targets[name], weights[name], centre, band, value)
        halo_index = np.arange(
            max(int(np.floor(centre - halo)), 0),
            min(int(np.ceil(centre + halo)), length - 1) + 1,
        )
        mask[halo_index] = np.minimum(mask[halo_index], 0.0)

    for name in classes:
        mask[targets[name] > 0] = 1.0

    start_key, end_key = (
        ("row_start", "row_end") if key == "csa_row" else ("col_start", "col_end")
    )
    for item in ignore:
        if item.get("annotation_type") != region_type:
            continue
        lo = to_axis(item[start_key], native_length, length)
        hi = to_axis(item[end_key], native_length, length)
        index = np.arange(
            max(int(np.floor(lo)), 0), min(int(np.ceil(hi)), length - 1) + 1
        )
        if len(index):
            mask[index] = np.where(
                np.max([targets[name][index] for name in classes], axis=0) > 0, 1.0, 0.0
            )

    for name in classes:
        weights[name] = np.where(targets[name] > 0, weights[name], 1.0)
    return targets, weights, mask


def window_features(profile, size, half):
    """Build local profile features for each sample."""
    response = highpass(profile, size)
    response = response / robust_scale(response)
    length = len(response)
    padded = np.pad(response, half, mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    centred = (profile - np.median(profile)) / robust_scale(profile)
    position = np.linspace(0.0, 1.0, length)
    return np.column_stack([window, centred, position])


def feature_names(half):
    """Return names for columns produced by `window_features`."""
    return [f"hp{offset:+d}" for offset in range(-half, half + 1)] + [
        "darkness",
        "position",
    ]


def scan_arrays(scan, half=8, weak_weight=0.3, bounds=None):
    """Build profiles, targets, weights, and masks for one scan."""
    document = json.loads(Path(scan["labels_path"]).read_text())
    height, width = document["image_shape"]
    labels, ignore = document["labels"], document["ignore"]
    image = load_image(scan["film_path"])
    if image.shape != (height, width):
        raise ValueError(
            f"{scan['name']}: film is {image.shape}, labels say {(height, width)}"
        )

    if bounds is None:
        top = min(
            _positions(labels, "top_of_ionogram", "csa_row", height, height) or [0]
        )
        bottom = max(
            _positions(labels, "bottom_of_ionogram", "csa_row", height, height)
            or [height - 1]
        )
    else:
        top, bottom = bounds

    rows = resample(row_profile(image), ROW_LEN)
    columns = resample(column_profile(image, top, bottom + 1), COL_LEN)

    row_targets, row_weights, row_mask = build_targets(
        labels,
        ignore,
        ROW_CLASSES,
        height,
        ROW_LEN,
        "csa_row",
        "row_region",
        weak_weight,
    )
    col_targets, col_weights, col_mask = build_targets(
        labels,
        ignore,
        COL_CLASSES,
        width,
        COL_LEN,
        "csa_x",
        "column_region",
        weak_weight,
    )
    return {
        "name": scan["name"],
        "reel": scan["reel"],
        "station": scan["station"],
        "shape": (height, width),
        "row": {
            "profile": rows,
            "features": window_features(rows, ROW_MEDIAN, half),
            "targets": row_targets,
            "weights": row_weights,
            "mask": row_mask,
        },
        "column": {
            "profile": columns,
            "features": window_features(columns, COL_MEDIAN, half),
            "targets": col_targets,
            "weights": col_weights,
            "mask": col_mask,
        },
        "truth": {
            "csa_horizontal_ruling": sorted(
                item["csa_row"]
                for item in labels
                if item["class_name"] == "csa_horizontal_ruling"
            ),
            "verified_ruling": sorted(
                item["csa_row"]
                for item in labels
                if item["class_name"] == "csa_horizontal_ruling"
                and not item.get("weak_label")
            ),
            "frequency_marker": sorted(
                item["csa_x"]
                for item in labels
                if item["class_name"] == "frequency_marker"
            ),
            "start_of_frequency_sweep": sorted(
                item["csa_x"]
                for item in labels
                if item["class_name"] == "start_of_frequency_sweep"
            ),
            "top_of_ionogram": top,
            "bottom_of_ionogram": bottom,
            "nasa_marker_columns": nasa_marker_columns(scan["labels_path"]),
        },
    }


def nasa_marker_columns(labels_path):
    """Return stored NASA marker columns, if available."""
    path = Path(labels_path).with_name(
        Path(labels_path).name.replace("_ml_labels.json", ".json")
    )
    if not path.exists():
        return None
    columns = (json.loads(path.read_text()).get("nasa") or {}).get("marker_columns")
    return np.asarray(columns, dtype=float) if columns else None


def stack(arrays, axis, class_name):
    """Pool features, targets, weights, and masks over scans."""
    features = np.concatenate([item[axis]["features"] for item in arrays])
    target = np.concatenate([item[axis]["targets"][class_name] for item in arrays])
    weight = np.concatenate([item[axis]["weights"][class_name] for item in arrays])
    mask = np.concatenate([item[axis]["mask"] for item in arrays])
    return features, target, weight, mask
