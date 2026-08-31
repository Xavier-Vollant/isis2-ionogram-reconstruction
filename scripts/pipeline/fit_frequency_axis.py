#!/usr/bin/env python3
"""Fit the film-column to frequency mapping.

The command supports CDF-assisted and film-only calibration. It does not render
or warp the image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from isis_research.registration import landmarks

try:
    from scripts.pipeline.route_calibration import (
        scan_descriptor,
        select_film_profile,
    )
except ModuleNotFoundError:  # direct script execution
    from scripts.pipeline.route_calibration import (
        scan_descriptor,
        select_film_profile,
    )

DEFAULT_PROFILE = ROOT / "configs/film_calibration_profile.json"
DEFAULT_PHASE1_MANIFEST = ROOT / "outputs/calibration/phase1_pairs/manifest.csv"
DEFAULT_PHASE1_RECORDS = ROOT / "outputs/calibration/phase1_records.csv"
DEFAULT_STRUCTURE_DIR = ROOT / "outputs/calibration/phase3_structure/json"
DEFAULT_LANDMARK_DIR = ROOT / "outputs/landmarks/batch1500"
DEFAULT_OUT = ROOT / "outputs/calibration/phase4_frequency_axis"


def load_json(path):
    """Read a UTF-8 JSON sidecar into a Python object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference_from_landmark_json(path):
    document = load_json(path)
    nasa = document.get("nasa", {})
    return {
        "source": "cdf_landmark_reference",
        "marker_columns": nasa.get("marker_columns", []),
        "marker_frequencies": nasa.get("marker_frequencies", []),
        "reference_file": str(path),
    }


def reference_from_cdf(path):
    """Read only marker coordinates from a CDF when cdflib is installed."""
    try:
        import cdflib
    except ModuleNotFoundError as error:
        raise RuntimeError("cdflib is required to read a raw CDF reference") from error
    cdf = cdflib.CDF(str(path))
    epoch = np.asarray(cdf.varget("Epoch"), dtype=float)
    times = np.asarray(cdf.varget("Time_mark"), dtype=float)
    frequencies = np.asarray(cdf.varget("freq_mark"), dtype=float)
    valid = np.isfinite(times) & (times > -1e30)
    valid &= np.isfinite(frequencies) & (frequencies > -1e30)
    return {
        "source": "cdf",
        "marker_columns": landmarks.nearest_columns(epoch, times[valid]).tolist(),
        "marker_frequencies": frequencies[valid].tolist(),
        "reference_file": str(path),
    }


def _clean_reference(reference):
    columns = np.asarray(reference.get("marker_columns", []), dtype=float).ravel()
    frequencies = np.asarray(
        reference.get("marker_frequencies", []), dtype=float
    ).ravel()
    length = min(len(columns), len(frequencies))
    columns, frequencies = columns[:length], frequencies[:length]
    valid = np.isfinite(columns) & np.isfinite(frequencies)
    columns, frequencies = columns[valid], frequencies[valid]
    if len(columns) < 4:
        raise landmarks.film.FitFailed("reference has fewer than four valid markers")
    order = np.argsort(columns)
    columns, frequencies = columns[order], frequencies[order]
    if np.any(np.diff(columns) <= 0):
        raise landmarks.film.FitFailed(
            "reference marker columns are not strictly increasing"
        )
    if np.any(np.diff(frequencies) <= 0):
        raise landmarks.film.FitFailed(
            "reference marker frequencies are not increasing"
        )
    return columns, frequencies


def fit_reference(observed_markers, reference, source=None, profile=None):
    """Fit ordered markers and return a piecewise film-column→MHz mapping."""
    observed = np.asarray(observed_markers, dtype=float).ravel()
    observed = observed[np.isfinite(observed)]
    reference_columns, frequencies = _clean_reference(reference)
    if len(observed) < 4:
        raise landmarks.film.FitFailed("scan has fewer than four marker candidates")
    fit = landmarks.film.fit_marker_axis(observed, reference_columns)
    start = fit["reference_start"]
    stop = start + fit["count"]
    matched_reference_columns = reference_columns[start:stop]
    matched_frequencies = frequencies[start:stop]
    matched_columns = observed[fit["observed_indices"]]
    if len(matched_columns) < 4 or np.any(np.diff(matched_columns) <= 0):
        raise landmarks.film.FitFailed("matched scan marker columns are not monotonic")
    coverage = float(fit["count"] / len(reference_columns))
    breakpoints = [
        {
            "film_column": float(column),
            "frequency_mhz": float(frequency),
            "reference_column": float(reference_column),
            "residual_px": float(residual),
        }
        for column, frequency, reference_column, residual in zip(
            matched_columns,
            matched_frequencies,
            matched_reference_columns,
            fit["residual"],
        )
    ]
    warnings = []
    if coverage < 0.60:
        warnings.append("partial_frequency_marker_coverage")
    if matched_frequencies[0] > frequencies[0]:
        warnings.append("low_frequency_markers_missing")
    if matched_frequencies[-1] < frequencies[-1]:
        warnings.append("high_frequency_markers_missing")
    if fit["max_error_px"] > 3.0:
        warnings.append("large_marker_fit_residual")
    if fit["count"] < 8:
        warnings.append("few_matched_frequency_markers")
    if fit["rms_px"] <= 1.5 and coverage >= 0.60 and fit["count"] >= 8:
        status = "usable"
    elif fit["rms_px"] <= 3.0 and fit["count"] >= 4:
        status = "review"
    else:
        status = "not_usable"
    return {
        "schema": "isis.csa_frequency_axis.v1",
        "source": source or reference.get("source", "unknown"),
        "profile": profile,
        "status": status,
        "mapping": "monotonic_piecewise_linear",
        "matched_marker_count": int(fit["count"]),
        "reference_marker_count": len(reference_columns),
        "marker_coverage": coverage,
        "marker_rms_px": float(fit["rms_px"]),
        "marker_max_error_px": float(fit["max_error_px"]),
        "reference_start": int(start),
        "reference_end": int(stop - 1),
        "frequency_min_mhz": float(matched_frequencies[0]),
        "frequency_max_mhz": float(matched_frequencies[-1]),
        "breakpoints": breakpoints,
        "warnings": sorted(set(warnings)),
    }


def fit_from_profile(observed_markers, image_shape, profile, metadata=None):
    """Fit the image-only marker profile and return a frequency-axis record."""
    metadata = metadata or {}
    descriptor = scan_descriptor(np.zeros(image_shape, dtype=float), metadata)
    selection = select_film_profile(observed_markers, descriptor, profile, metadata)
    if not selection.get("selected"):
        return {
            "schema": "isis.csa_frequency_axis.v1",
            "source": "film_only_profile",
            "status": selection["status"],
            "profile_selection": selection,
            "warnings": [selection["reason"]],
        }
    selected_name = selection["selected"]["profile"]
    group = None
    if selected_name.endswith("__fallback"):
        group = profile.get("format_fallbacks", {}).get(descriptor["format_class"])
    else:
        group = profile.get("profiles", {}).get(selected_name)
    reference = {
        "marker_columns": np.asarray(
            group["frequency"]["position_fraction"], dtype=float
        )
        * descriptor["width"],
        "marker_frequencies": group["frequency"]["frequencies_mhz"],
        "source": "film_only_profile",
    }
    result = fit_reference(
        observed_markers,
        reference,
        source="film_only_profile",
        profile=selected_name,
    )
    result["profile_selection"] = selection
    if selection["status"] == "review":
        result["status"] = "review"
    return result


def fit_structure(structure, profile, metadata=None, reference=None, cdf=None):
    """Choose the CDF or film-only route and fit one scan's frequency axis."""
    observed = [
        item["x"]
        for item in structure.get("vertical_markers", {}).get("candidates", [])
    ]
    if cdf:
        reference = reference_from_cdf(cdf)
    if reference is not None:
        try:
            return fit_reference(
                observed,
                reference,
                source=reference.get("source"),
            )
        except landmarks.film.FitFailed as error:
            return {
                "schema": "isis.csa_frequency_axis.v1",
                "source": reference.get("source", "unknown"),
                "status": "not_usable",
                "warnings": [str(error)],
            }
    return fit_from_profile(observed, structure["image_shape"], profile, metadata)


def _landmark_paths(directory):
    return {
        path.name.removesuffix("_landmarks.json"): path
        for path in Path(directory).rglob("*_landmarks.json")
    }


def process_manifest(
    phase1_manifest,
    structure_dir,
    landmark_dir,
    profile_path,
    out_dir,
    phase1_records=DEFAULT_PHASE1_RECORDS,
):
    """Fit frequency axes for a manifest and write JSON sidecars."""
    phase1_manifest = Path(phase1_manifest)
    structure_dir = Path(structure_dir)
    profile = load_json(profile_path)
    landmark_paths = _landmark_paths(landmark_dir)
    with phase1_manifest.open(newline="", encoding="utf-8") as handle:
        phase1_rows = list(csv.DictReader(handle))
    records_by_pair = {}
    if Path(phase1_records).exists():
        with Path(phase1_records).open(newline="", encoding="utf-8") as handle:
            records_by_pair = {item["name"]: item for item in csv.DictReader(handle)}
    cdf_dir = Path(out_dir) / "cdf_assisted"
    film_dir = Path(out_dir) / "film_only"
    cdf_dir.mkdir(parents=True, exist_ok=True)
    film_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in phase1_rows:
        pair_name = row["pair_name"]
        structure_path = (
            structure_dir / f"{int(row['pair_number']):04d}__{pair_name}_structure.json"
        )
        structure = load_json(structure_path)
        reference_path = landmark_paths.get(pair_name)
        reference = (
            reference_from_landmark_json(reference_path) if reference_path else None
        )
        metadata = {
            "format_class": row.get("format_class", "")
            or records_by_pair.get(pair_name, {}).get("format_class", ""),
            "sweep_class": row.get("sweep_class", "")
            or records_by_pair.get(pair_name, {}).get("sweep_class", ""),
        }
        cdf_result = fit_structure(structure, profile, metadata, reference=reference)
        film_result = fit_structure(structure, profile, metadata)
        stem = f"{int(row['pair_number']):04d}__{pair_name}"
        (cdf_dir / f"{stem}_frequency.json").write_text(
            json.dumps(cdf_result, indent=2), encoding="utf-8"
        )
        (film_dir / f"{stem}_frequency.json").write_text(
            json.dumps(film_result, indent=2), encoding="utf-8"
        )
        records.append(
            {
                "pair_number": row["pair_number"],
                "pair_name": pair_name,
                "split": row["split"],
                "cdf_status": cdf_result["status"],
                "cdf_markers": cdf_result.get("matched_marker_count", 0),
                "cdf_rms_px": cdf_result.get("marker_rms_px"),
                "cdf_frequency_coverage": cdf_result.get("marker_coverage"),
                "film_status": film_result["status"],
                "film_profile": film_result.get("profile"),
                "film_markers": film_result.get("matched_marker_count", 0),
                "film_rms_px": film_result.get("marker_rms_px"),
                "film_frequency_coverage": film_result.get("marker_coverage"),
                "film_warnings": ";".join(film_result.get("warnings", [])),
            }
        )
    with (Path(out_dir) / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (Path(out_dir) / "README.md").write_text(
        "# Frequency-axis fits\n\n"
        f"Frequency-axis fits for {len(records)} scans.\n\n"
        "`cdf_assisted/` uses matching CDF landmarks; "
        "`film_only/` uses the stored calibration profile and detected markers. Each "
        "JSON contains a monotonic piecewise film-column→MHz mapping.\n",
        encoding="utf-8",
    )
    return records


def main():
    """Parse CLI options and fit film columns to frequency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=None)
    parser.add_argument("--reference-json", type=Path, default=None)
    parser.add_argument("--cdf", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--phase1-manifest", type=Path, default=DEFAULT_PHASE1_MANIFEST)
    parser.add_argument("--phase1-records", type=Path, default=DEFAULT_PHASE1_RECORDS)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--landmark-dir", type=Path, default=DEFAULT_LANDMARK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.batch or args.structure is None:
        records = process_manifest(
            args.phase1_manifest,
            args.structure_dir,
            args.landmark_dir,
            args.profile,
            args.out_dir,
            args.phase1_records,
        )
        counts = {
            "cdf": {},
            "film": {},
        }
        for record in records:
            counts["cdf"][record["cdf_status"]] = (
                counts["cdf"].get(record["cdf_status"], 0) + 1
            )
            counts["film"][record["film_status"]] = (
                counts["film"].get(record["film_status"], 0) + 1
            )
        print(f"processed {len(records)} scans")
        print(f"cdf-assisted statuses: {counts['cdf']}")
        print(f"film-only statuses: {counts['film']}")
        print(f"wrote {args.out_dir / 'manifest.csv'}")
        return
    structure = load_json(args.structure)
    profile = load_json(args.profile)
    metadata = {}
    if args.metadata:
        metadata = load_json(args.metadata)
    reference = (
        reference_from_landmark_json(args.reference_json)
        if args.reference_json
        else None
    )
    result = fit_structure(structure, profile, metadata, reference, args.cdf)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
