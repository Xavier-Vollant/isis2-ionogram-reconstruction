#!/usr/bin/env python3
"""Build and validate the first offline CSA calibration profile.

The input is the existing CDF-assisted landmark batch.  CDF-derived labels are
used only to create the profile and score held-out reels; the learned profile
itself contains scan-format summaries that can later be used without a CDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_BATCH = ROOT / "outputs/landmarks/batch1500"
DEFAULT_PAIRS = ROOT / "data/processed/review_ranked_top1500_reel.csv"
DEFAULT_OUT = ROOT / "configs/film_calibration_profile.json"
DEFAULT_RECORDS_OUT = ROOT / "outputs/calibration/phase1_records.csv"
DEFAULT_REPORT_OUT = ROOT / "outputs/calibration/phase1_report.json"
WIDTH_SPLIT_PX = 1000
MIN_PROFILE_SAMPLES = 25
MIN_FALLBACK_SAMPLES = 3
MIN_MARKER_SAMPLES = 20


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_or_none(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def width_class(width):
    return "narrow" if width < WIDTH_SPLIT_PX else "wide"


def format_class(width, height):
    if height >= 450:
        return "tall_ultrawide" if width >= 1450 else "tall"
    return width_class(width)


def sweep_class(row):
    value = (row.get("nasa_swept_freq_range") or "").strip()
    if value == "0.1 - 10 MHz":
        return "sweep_10mhz"
    if value == "0.1 - 20 MHz":
        return "sweep_20mhz"
    return "sweep_unknown"


def format_key(scan):
    return f"{scan['format_class']}__{scan['sweep_class']}"


def _nasa_id(row):
    value = row.get("nasa_png_file", "")
    return Path(value).stem.split("_", 1)[-1]


def _image_shape(document):
    shape = (document.get("csa_ml_labels") or {}).get("image_shape")
    if shape and len(shape) == 2:
        return int(shape[1]), int(shape[0])
    shape = document.get("image_shape")
    if shape and len(shape) == 2:
        return int(shape[1]), int(shape[0])
    return None


def _record_reliability(document, quality, summary_item=None):
    """Return the explicit gate used for profile training."""
    vertical = document.get("vertical_calibration") or {}
    within_one = float_or_none(
        (quality.get("metrics") or {}).get("trace", {}).get("within_1bin")
    )
    if within_one is None:
        within_one = float_or_none(document.get("selected_trace_within_1bin"))
    if within_one is None:
        within_one = float_or_none((summary_item or {}).get("selected_trace_within_1bin"))
    marker_count = int((quality.get("metrics") or {}).get("marker_count", 0))
    if not marker_count:
        marker_count = int(document.get("x_fit", {}).get("count", 0))
    marker_rms = float_or_none(
        (quality.get("metrics") or {}).get("marker_rms_px")
    )
    if marker_rms is None:
        marker_rms = float_or_none(document.get("marker_rms_px"))
    checks = {
        "quality_status_usable": quality.get("status") == "usable",
        "marker_count_at_least_8": marker_count >= 8,
        "marker_rms_at_most_1_5px": marker_rms is not None and marker_rms <= 1.5,
        "vertical_fit_accepted": vertical.get("status") == "accepted_affine",
        "trace_within_one_bin_at_least_35pct": within_one is not None
        and within_one >= 0.35,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "marker_count": marker_count,
        "marker_rms_px": marker_rms,
        "trace_within_1bin": within_one,
    }


def collect_scans(batch, pairs_csv):
    """Join landmark JSON, quality manifest, summary metadata, and pair CSV."""
    pair_rows = {_nasa_id(row): row for row in read_csv(pairs_csv)}
    summary_path = Path(batch) / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = {
            Path(item["cdf_file"]).stem: item
            for item in json.loads(summary_path.read_text(encoding="utf-8"))
            if item.get("status") == "ok"
        }
    quality_path = Path(batch) / "quality_manifest.json"
    quality = {}
    if quality_path.exists():
        quality = {
            item["name"]: item
            for item in json.loads(quality_path.read_text(encoding="utf-8"))
        }

    scans = []
    for path in sorted(Path(batch).rglob("*_landmarks.json")):
        name = path.name.removesuffix("_landmarks.json")
        document = json.loads(path.read_text(encoding="utf-8"))
        summary_item = summary.get(name, {})
        row = pair_rows.get(name)
        if row is None:
            continue
        shape = _image_shape(document)
        if shape is None:
            size = (row.get("csa_size") or "").split("x")
            if len(size) != 2:
                continue
            shape = (int(size[0]), int(size[1]))
        quality_item = quality.get(name, {})
        if not quality_item:
            quality_item = {"status": path.parent.name}
        reliability = _record_reliability(document, quality_item, summary_item)
        scan = {
            "name": name,
            "landmark_path": str(path),
            "csa_file": summary_item.get("csa_file", row.get("csa_file", "")),
            "nasa_file": name + ".cdf",
            "reel": row.get("csa_film_subdir", "unknown"),
            "station": row.get("csa_station", "unknown"),
            "width": shape[0],
            "height": shape[1],
            "width_class": width_class(shape[0]),
            "format_class": format_class(shape[0], shape[1]),
            "sweep_class": sweep_class(row),
            "document": document,
            "reliability": reliability,
        }
        scans.append(scan)
    return scans


def split_reels(scans, fraction=0.20, seed=0):
    """Split by reel so scans from one physical film never leak across sets."""
    by_reel = defaultdict(list)
    for scan in scans:
        by_reel[scan["reel"]].append(scan)
    reels = sorted(by_reel)
    np.random.default_rng(seed).shuffle(reels)
    target = max(1, int(round(len(scans) * fraction)))
    held_reels, count = set(), 0
    for reel in reels:
        if count >= target and held_reels:
            break
        held_reels.add(reel)
        count += len(by_reel[reel])
    held = [scan for scan in scans if scan["reel"] in held_reels]
    train = [scan for scan in scans if scan["reel"] not in held_reels]
    return train, held, sorted(held_reels)


def robust(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"median": None, "mad": None, "p10": None, "p90": None}
    median = float(np.median(values))
    return {
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def marker_points(document):
    return [
        (float(item["csa_x"]), float(item["frequency_mhz"]))
        for item in document.get("features", [])
        if item.get("name") == "frequency_marker"
        and item.get("csa_x") is not None
        and item.get("frequency_mhz") is not None
    ]


def build_group(scans):
    by_frequency = defaultdict(list)
    ruling_spacing = []
    km_per_ruling = []
    px_per_km_values = []
    top_offsets = []
    zero_rows = []
    for scan in scans:
        document = scan["document"]
        for column, frequency in marker_points(document):
            by_frequency[frequency].append(column / scan["width"])
        lattice = document.get("csa_ruling_lattice") or {}
        geometry = document.get("geometry") or {}
        spacing = float_or_none(lattice.get("spacing_px"))
        px_per_km = float_or_none(geometry.get("px_per_km"))
        zero_row = float_or_none(geometry.get("zero_row"))
        top_row = float_or_none((document.get("film") or {}).get("top_row"))
        if spacing is not None:
            ruling_spacing.append(spacing)
        if spacing is not None and px_per_km and px_per_km > 0:
            km_per_ruling.append(spacing / px_per_km)
            px_per_km_values.append(px_per_km)
        if zero_row is not None:
            zero_rows.append(zero_row)
        if zero_row is not None and top_row is not None:
            top_offsets.append(zero_row - top_row)

    frequencies = sorted(
        frequency
        for frequency, positions in by_frequency.items()
        if len(positions) >= MIN_MARKER_SAMPLES
    )
    positions = [robust(by_frequency[frequency]) for frequency in frequencies]
    return {
        "sample_count": len(scans),
        "reels": sorted({scan["reel"] for scan in scans}),
        "frequency": {
            "frequencies_mhz": frequencies,
            "position_fraction": [item["median"] for item in positions],
            "position_mad_fraction": [item["mad"] for item in positions],
            "sample_count": [len(by_frequency[frequency]) for frequency in frequencies],
        },
        "height": {
            "km_per_ruling": robust(km_per_ruling),
            "px_per_km": robust(px_per_km_values),
            "ruling_spacing_px": robust(ruling_spacing),
            "top_offset_px": robust(top_offsets),
            "zero_row_px": robust(zero_rows),
        },
    }


def build_profile(train, all_scans):
    groups = defaultdict(list)
    format_groups = defaultdict(list)
    for scan in train:
        groups[format_key(scan)].append(scan)
        format_groups[scan["format_class"]].append(scan)
    profiles = {key: build_group(value) for key, value in sorted(groups.items())}
    fallbacks = {key: build_group(value) for key, value in sorted(format_groups.items())}
    return {
        "schema": "isis.csa_calibration_profile.v1",
        "source": {
            "training_scans": len(train),
            "training_reels": len({scan["reel"] for scan in train}),
            "all_eligible_scans": len(all_scans),
            "width_split_px": WIDTH_SPLIT_PX,
            "min_profile_samples": MIN_PROFILE_SAMPLES,
            "min_fallback_samples": MIN_FALLBACK_SAMPLES,
            "min_marker_samples": MIN_MARKER_SAMPLES,
        },
        "profiles": profiles,
        "format_fallbacks": fallbacks,
    }


def choose_group(profile, scan):
    exact = profile["profiles"].get(format_key(scan))
    if exact and exact["sample_count"] >= MIN_PROFILE_SAMPLES:
        return exact, format_key(scan)
    fallback = profile["format_fallbacks"].get(scan["format_class"])
    if fallback and fallback["sample_count"] >= MIN_FALLBACK_SAMPLES:
        return fallback, scan["format_class"] + "__fallback"
    return None, None


def validate_frequency(scan, group):
    frequencies = np.asarray(group["frequency"]["frequencies_mhz"], dtype=float)
    positions = np.asarray(group["frequency"]["position_fraction"], dtype=float)
    observed = np.asarray(scan["document"].get("film", {}).get("marker_candidates", []))
    if len(frequencies) < 4 or len(observed) < 4:
        return {"fitted": False}
    from isis_research.registration.landmarks import fit_marker_axis

    try:
        fit = fit_marker_axis(observed, positions * scan["width"])
    except Exception:
        return {"fitted": False}
    actual = marker_points(scan["document"])
    actual_by_frequency = {frequency: column for column, frequency in actual}
    correct = 0
    checked = 0
    position_errors = []
    matched_frequencies = frequencies[
        fit["reference_start"] : fit["reference_start"] + fit["count"]
    ]
    matched_columns = observed[fit["observed_indices"]]
    for column, frequency in zip(matched_columns, matched_frequencies):
        if frequency not in actual_by_frequency:
            continue
        checked += 1
        error = abs(float(column) - actual_by_frequency[frequency])
        position_errors.append(error)
        if error <= 5.0:
            correct += 1
    return {
        "fitted": True,
        "count": int(fit["count"]),
        "coverage": float(fit["count"] / len(frequencies)),
        "rms_px": float(fit["rms_px"]),
        "checked": checked,
        "assignment_accuracy": float(correct / checked) if checked else None,
        "position_error_median_px": (
            float(np.median(position_errors)) if position_errors else None
        ),
    }


def validate_height(scan, group):
    document = scan["document"]
    height = group["height"]
    px_per_km = height["px_per_km"]["median"]
    top_offset = height["top_offset_px"]["median"]
    lattice = document.get("csa_ruling_lattice") or {}
    geometry = document.get("geometry") or {}
    spacing = float_or_none(lattice.get("spacing_px"))
    actual_zero = float_or_none(geometry.get("zero_row"))
    actual_px_per_km = float_or_none(geometry.get("px_per_km"))
    top_row = float_or_none((document.get("film") or {}).get("top_row"))
    rows = np.asarray(lattice.get("rows", []), dtype=float)
    if (
        px_per_km is None
        or top_offset is None
        or spacing is None
        or actual_zero is None
        or actual_px_per_km is None
        or top_row is None
        or len(rows) < 3
    ):
        return {"fitted": False}
    predicted_zero = top_row + top_offset
    predicted_px_per_km = px_per_km
    predicted = (rows - predicted_zero) / predicted_px_per_km
    actual = (rows - actual_zero) / actual_px_per_km
    errors = np.abs(predicted - actual)
    return {
        "fitted": True,
        "ruling_count": int(len(rows)),
        "predicted_zero_row_px": float(predicted_zero),
        "predicted_px_per_km": float(predicted_px_per_km),
        "actual_zero_error_px": float(abs(predicted_zero - actual_zero)),
        "height_error_median_km": float(np.median(errors)),
        "height_error_p90_km": float(np.percentile(errors, 90)),
    }


def validate(profile, held_out):
    rows = []
    for scan in held_out:
        group, selected = choose_group(profile, scan)
        if group is None:
            rows.append({"name": scan["name"], "profile": None})
            continue
        rows.append(
            {
                "name": scan["name"],
                "profile": selected,
                "frequency": validate_frequency(scan, group),
                "height": validate_height(scan, group),
            }
        )
    frequency = [row["frequency"] for row in rows if row.get("frequency", {}).get("fitted")]
    height = [row["height"] for row in rows if row.get("height", {}).get("fitted")]
    height_medians = [row["height_error_median_km"] for row in height]
    height_p90s = [row["height_error_p90_km"] for row in height]
    return {
        "scans": len(held_out),
        "profile_selected": sum(row.get("profile") is not None for row in rows),
        "frequency": {
            "fitted": len(frequency),
            "fit_fraction": len(frequency) / len(held_out) if held_out else 0.0,
            "median_rms_px": float(np.median([row["rms_px"] for row in frequency]))
            if frequency
            else None,
            "median_assignment_accuracy": float(
                np.median(
                    [row["assignment_accuracy"] for row in frequency if row["assignment_accuracy"] is not None]
                )
            )
            if any(row["assignment_accuracy"] is not None for row in frequency)
            else None,
            "median_coverage": float(np.median([row["coverage"] for row in frequency]))
            if frequency
            else None,
        },
        "height": {
            "fitted": len(height),
            "fit_fraction": len(height) / len(held_out) if held_out else 0.0,
            "median_error_km": float(np.median(height_medians)) if height else None,
            "p90_error_km": float(np.percentile(height_medians, 90))
            if height
            else None,
            "median_scan_p90_error_km": float(np.median(height_p90s))
            if height
            else None,
        },
        "per_scan": rows,
    }


def write_records(path, train, held_out):
    held_names = {scan["name"] for scan in held_out}
    fields = [
        "name",
        "split",
        "reel",
        "station",
        "width",
        "height",
        "width_class",
        "format_class",
        "sweep_class",
        "marker_count",
        "marker_rms_px",
        "trace_within_1bin",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scan in train + held_out:
            reliability = scan["reliability"]
            writer.writerow(
                {
                    "name": scan["name"],
                    "split": "held_out" if scan["name"] in held_names else "train",
                    "reel": scan["reel"],
                    "station": scan["station"],
                    "width": scan["width"],
                    "height": scan["height"],
                    "width_class": scan["width_class"],
                    "format_class": scan["format_class"],
                    "sweep_class": scan["sweep_class"],
                    "marker_count": reliability["marker_count"],
                    "marker_rms_px": reliability["marker_rms_px"],
                    "trace_within_1bin": reliability["trace_within_1bin"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--held-out-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scans = collect_scans(args.batch, args.pairs)
    eligible = [scan for scan in scans if scan["reliability"]["eligible"]]
    train, held_out, held_reels = split_reels(
        eligible, fraction=args.held_out_fraction, seed=args.seed
    )
    if not train or not held_out:
        raise SystemExit("need at least one training and one held-out scan")
    profile = build_profile(train, eligible)
    validation = validate(profile, held_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    records_path = (
        DEFAULT_RECORDS_OUT
        if args.out == DEFAULT_OUT
        else args.out.with_name("phase1_records.csv")
    )
    records_path.parent.mkdir(parents=True, exist_ok=True)
    write_records(records_path, train, held_out)
    report = {
        "schema": "isis.csa_calibration_phase1_report.v1",
        "reliability_gate": {
            "quality_status": "usable",
            "marker_count": ">= 8",
            "marker_rms_px": "<= 1.5",
            "vertical_status": "accepted_affine",
            "trace_within_1bin": ">= 0.35",
        },
        "input_scans": len(scans),
        "eligible_scans": len(eligible),
        "training_scans": len(train),
        "held_out_scans": len(held_out),
        "training_reels": len({scan["reel"] for scan in train}),
        "held_out_reels": held_reels,
        "format_counts": {
            key: sum(scan["format_class"] == key for scan in eligible)
            for key in sorted({scan["format_class"] for scan in eligible})
        },
        "profile_groups": {
            key: value["sample_count"] for key, value in profile["profiles"].items()
        },
        "validation": validation,
        "artifacts": {
            "profile": str(args.out.relative_to(ROOT)),
            "records": str(records_path.relative_to(ROOT)),
        },
    }
    report_path = (
        DEFAULT_REPORT_OUT
        if args.out == DEFAULT_OUT
        else args.out.with_name("phase1_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"eligible={len(eligible)} train={len(train)} held_out={len(held_out)} "
        f"held_out_reels={len(held_reels)}"
    )
    print("profile groups: " + ", ".join(f"{k}={v}" for k, v in report["profile_groups"].items()))
    print(
        "frequency validation: "
        f"fit={validation['frequency']['fit_fraction']:.1%}, "
        f"median_rms={validation['frequency']['median_rms_px']:.2f}px, "
        f"median_assignment={validation['frequency']['median_assignment_accuracy']:.1%}"
    )
    print(
        "height validation: "
        f"fit={validation['height']['fit_fraction']:.1%}, "
        f"median_error={validation['height']['median_error_km']:.1f}km, "
        f"p90_error={validation['height']['p90_error_km']:.1f}km"
    )
    print(f"wrote {args.out.relative_to(ROOT)}")
    print(f"wrote {records_path.relative_to(ROOT)}")
    print(f"wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
