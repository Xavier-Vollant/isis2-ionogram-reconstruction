#!/usr/bin/env python3
"""Choose the Phase 2 calibration route for one raw CSA scan.

This phase only chooses the calibration source.  It does not warp or render
the image; those operations belong to later phases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from isis_research.registration import landmarks  # noqa: E402

try:  # noqa: E402
    from scripts.dataset.build_calibration_profile import (
        format_class,
        format_key,
        sweep_class,
        width_class,
    )
except ModuleNotFoundError:  # direct script execution
    from scripts.dataset.build_calibration_profile import (
        format_class,
        format_key,
        sweep_class,
        width_class,
    )

DEFAULT_PROFILE = ROOT / "configs/film_calibration_profile.json"
DEFAULT_CDF_DIR = ROOT / "data/raw/matches/nasa_cdf"


def read_metadata(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path_value(value, base=ROOT):
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def resolve_cdf(explicit, metadata, cdf_dir=DEFAULT_CDF_DIR):
    """Return (path, source, warning), without opening the CDF."""
    candidates = []
    if explicit:
        candidates.append((_path_value(explicit), "explicit_argument"))
    for key in ("cdf", "cdf_path", "cdf_file"):
        value = metadata.get(key)
        if value:
            candidates.append((_path_value(value), f"metadata:{key}"))
            candidates.append((Path(cdf_dir) / Path(value).name, f"cdf_dir:{key}"))
    nasa_id = metadata.get("nasa_id") or metadata.get("nasa_record")
    if nasa_id:
        candidates.append((Path(cdf_dir) / f"{nasa_id}.cdf", "metadata:nasa_id"))
    seen = set()
    for path, source in candidates:
        if path is None or path in seen:
            continue
        seen.add(path)
        if path.is_file() and path.stat().st_size:
            return path, source, None
    if candidates:
        return None, None, "a CDF was referenced but no readable local file was found"
    return None, None, None


def _metadata_sweep(metadata):
    value = metadata.get("sweep_class")
    if value in {"sweep_10mhz", "sweep_20mhz", "sweep_unknown"}:
        return value
    value = metadata.get("nasa_swept_freq_range") or metadata.get("sweep_range")
    return sweep_class({"nasa_swept_freq_range": value or ""})


def scan_descriptor(image, metadata):
    height, width = image.shape
    return {
        "width": int(width),
        "height": int(height),
        "width_class": width_class(width),
        "format_class": metadata.get("format_class") or format_class(width, height),
        "sweep_class": _metadata_sweep(metadata),
    }


def candidate_groups(profile, descriptor, metadata):
    """Return compatible profile groups, deduplicated by profile name."""
    groups = []
    known_sweep = _metadata_sweep(metadata) != "sweep_unknown"
    if known_sweep:
        key = format_key(descriptor)
        group = profile.get("profiles", {}).get(key)
        if (
            group
            and group.get("sample_count", 0)
            >= profile["source"].get("min_profile_samples", 25)
            and len(group.get("frequency", {}).get("frequencies_mhz", [])) >= 4
        ):
            # Metadata has already resolved the format/sweep ambiguity.  Do
            # not let a pooled fallback override that exact profile merely
            # because its marker medians are a fraction closer on one scan.
            return [(key, "format_and_sweep", group)]
    else:
        prefix = descriptor["format_class"] + "__"
        for key, group in profile.get("profiles", {}).items():
            if key.startswith(prefix) and group.get("frequency", {}).get(
                "frequencies_mhz"
            ):
                if group.get("sample_count", 0) >= profile["source"].get(
                    "min_profile_samples", 25
                ):
                    groups.append((key, "format_and_sweep_candidate", group))
    fallback = profile.get("format_fallbacks", {}).get(descriptor["format_class"])
    if fallback and fallback.get("sample_count", 0) >= profile["source"].get(
        "min_fallback_samples", 3
    ):
        groups.append(
            (descriptor["format_class"] + "__fallback", "format_fallback", fallback)
        )
    result = []
    seen = set()
    for item in groups:
        if (
            item[0] not in seen
            and len(item[2].get("frequency", {}).get("frequencies_mhz", [])) >= 4
        ):
            result.append(item)
            seen.add(item[0])
    return result


def select_film_profile(observed_markers, descriptor, profile, metadata=None):
    """Fit image-only marker candidates and choose a compatible profile."""
    metadata = metadata or {}
    candidates = []
    for name, source, group in candidate_groups(profile, descriptor, metadata):
        frequencies = np.asarray(group["frequency"]["frequencies_mhz"], dtype=float)
        reference = np.asarray(group["frequency"]["position_fraction"], dtype=float)
        try:
            fit = landmarks.film.fit_marker_axis(
                observed_markers, reference * descriptor["width"]
            )
        except landmarks.film.FitFailed:
            continue
        coverage = float(fit["count"] / len(frequencies))
        score = float(fit["rms_px"] + 2.0 * (1.0 - coverage))
        candidates.append(
            {
                "profile": name,
                "profile_source": source,
                "sample_count": int(group["sample_count"]),
                "marker_count": int(fit["count"]),
                "marker_rms_px": float(fit["rms_px"]),
                "marker_coverage": coverage,
                "selection_score": score,
                "reference_start": int(fit["reference_start"]),
            }
        )
    candidates.sort(key=lambda item: item["selection_score"])
    if not candidates:
        return {
            "status": "not_usable",
            "confidence": "low",
            "reason": "no_compatible_profile_or_marker_fit",
            "candidates": [],
        }
    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    margin = (
        float(runner_up["selection_score"] - best["selection_score"])
        if runner_up
        else None
    )
    best["runner_up"] = runner_up
    best["selection_margin"] = margin
    good_fit = best["marker_rms_px"] <= 1.5 and best["marker_coverage"] >= 0.60
    usable_choice = good_fit and (margin is None or margin >= 0.25)
    review_choice = best["marker_rms_px"] <= 3.0 and best["marker_coverage"] >= 0.50
    if usable_choice:
        status, confidence = "selected", "high"
    elif review_choice:
        status, confidence = "review", "medium"
    else:
        status, confidence = "not_usable", "low"
    return {
        "status": status,
        "confidence": confidence,
        "reason": "best_compatible_profile_by_marker_fit",
        "selected": best,
        "candidates": candidates,
    }


def route_scan(
    film_path,
    profile_path=DEFAULT_PROFILE,
    cdf=None,
    metadata=None,
    cdf_dir=DEFAULT_CDF_DIR,
):
    """Choose CDF-assisted or film-only routing for one scan."""
    film_path = Path(film_path)
    metadata = metadata or {}
    cdf_path, cdf_source, cdf_warning = resolve_cdf(cdf, metadata, cdf_dir)
    result = {
        "schema": "isis.csa_calibration_route.v1",
        "film": str(film_path),
        "metadata": metadata,
        "cdf": str(cdf_path) if cdf_path else None,
    }
    if cdf_path:
        result.update(
            {
                "route": "cdf_assisted",
                "status": "selected",
                "confidence": "high",
                "reason": "matching CDF is available",
                "cdf_source": cdf_source,
                "next": "run CDF-assisted landmark calibration",
            }
        )
        return result

    image = np.asarray(Image.open(film_path).convert("L"), dtype=float)
    descriptor = scan_descriptor(image, metadata)
    observed = landmarks.detect_film_features(image)["marker_candidates"]
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    selection = select_film_profile(observed, descriptor, profile, metadata)
    result.update(
        {
            "route": "film_only",
            "descriptor": descriptor,
            "profile": selection.get("selected", {}).get("profile"),
            "profile_selection": selection,
            "status": selection["status"],
            "confidence": selection["confidence"],
            "reason": selection["reason"],
            "next": (
                "run film-only calibration"
                if selection["status"] != "not_usable"
                else "request review"
            ),
        }
    )
    if cdf_warning:
        result.setdefault("warnings", []).append(cdf_warning)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--cdf", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--cdf-dir", type=Path, default=DEFAULT_CDF_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = route_scan(
        args.film,
        args.profile,
        args.cdf,
        read_metadata(args.metadata),
        args.cdf_dir,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
