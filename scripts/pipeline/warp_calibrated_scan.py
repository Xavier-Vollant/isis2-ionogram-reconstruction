#!/usr/bin/env python3
"""Warp calibrated CSA scans onto regular frequency x virtual-height grids."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from isis_research import ionogram  # noqa: E402
from scripts.pipeline.extract_scan_structure import normalize  # noqa: E402

DEFAULT_MANIFEST = ROOT / "outputs/calibration/phase1_pairs/manifest.csv"
DEFAULT_STRUCTURE_DIR = ROOT / "outputs/calibration/phase3_structure/json"
DEFAULT_FREQUENCY_DIR = ROOT / "outputs/calibration/phase4_frequency_axis"
DEFAULT_HEIGHT_DIR = ROOT / "outputs/calibration/phase5_height_axis"
DEFAULT_OUT = ROOT / "outputs/calibration/phase6_warped"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mapping_points(document, x_key, y_key):
    points = [
        (float(item[x_key]), float(item[y_key]))
        for item in document.get("breakpoints", [])
        if item.get(x_key) is not None and item.get(y_key) is not None
    ]
    points.sort()
    if len(points) < 2:
        raise ValueError(f"mapping needs at least two {x_key} breakpoints")
    x = np.asarray([item[0] for item in points], dtype=float)
    y = np.asarray([item[1] for item in points], dtype=float)
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError(f"{x_key} mapping is not strictly monotonic")
    return x, y


def inverse_mapping(target, breakpoints, x_key, y_key):
    """Invert a monotonic x→y mapping with linear endpoint extrapolation."""
    source_x, source_y = _mapping_points(breakpoints, x_key, y_key)
    target = np.asarray(target, dtype=float)
    result = np.interp(target, source_y, source_x)
    low = target < source_y[0]
    high = target > source_y[-1]
    low_slope = (source_x[1] - source_x[0]) / (source_y[1] - source_y[0])
    high_slope = (source_x[-1] - source_x[-2]) / (source_y[-1] - source_y[-2])
    result[low] = source_x[0] + (target[low] - source_y[0]) * low_slope
    result[high] = source_x[-1] + (target[high] - source_y[-1]) * high_slope
    return result


def forward_mapping(target, breakpoints, x_key, y_key):
    """Evaluate a monotonic x→y mapping with linear endpoint extrapolation."""
    source_x, source_y = _mapping_points(breakpoints, x_key, y_key)
    target = np.asarray(target, dtype=float)
    result = np.interp(target, source_x, source_y)
    low = target < source_x[0]
    high = target > source_x[-1]
    low_slope = (source_y[1] - source_y[0]) / (source_x[1] - source_x[0])
    high_slope = (source_y[-1] - source_y[-2]) / (source_x[-1] - source_x[-2])
    result[low] = source_y[0] + (target[low] - source_x[0]) * low_slope
    result[high] = source_y[-1] + (target[high] - source_x[-1]) * high_slope
    return result


def _height_at_row(height_result, row):
    mapping_anchors = height_result.get("mapping_anchors") or []
    if len(mapping_anchors) >= 2:
        return float(
            forward_mapping(
                np.asarray([row], dtype=float),
                {"breakpoints": mapping_anchors},
                "film_row",
                "virtual_height_km",
            )[0]
        )
    if height_result.get("zero_row_px") is not None and height_result.get("px_per_km"):
        return (float(row) - float(height_result["zero_row_px"])) / float(
            height_result["px_per_km"]
        )
    anchor_rows = np.asarray(height_result.get("anchor_rows", []), dtype=float)
    anchor_heights = np.asarray(height_result.get("anchor_heights_km", []), dtype=float)
    if len(anchor_rows) >= 2:
        return float(
            forward_mapping(
                np.asarray([row], dtype=float),
                {
                    "breakpoints": [
                        {"film_row": r, "virtual_height_km": h}
                        for r, h in zip(anchor_rows, anchor_heights)
                    ]
                },
                "film_row",
                "virtual_height_km",
            )[0]
        )
    raise ValueError("height result has no usable row-to-height mapping")


def combined_status(frequency_result, height_result):
    statuses = {frequency_result.get("status"), height_result.get("status")}
    if "not_usable" in statuses:
        return "not_usable"
    if "review" in statuses:
        return "review"
    return "usable"


def target_bounds(structure, frequency_result, height_result):
    frequency = _mapping_points(frequency_result, "film_column", "frequency_mhz")[1]
    film_top = float(structure["film_region"]["top_row"])
    film_bottom = float(structure["film_region"]["bottom_row"])
    top_height = _height_at_row(height_result, film_top)
    bottom_height = _height_at_row(height_result, film_bottom)
    height_min = max(0.0, min(top_height, bottom_height))
    height_max = max(top_height, bottom_height)
    if height_max <= height_min:
        raise ValueError("film region has no positive height span")
    return float(frequency[0]), float(frequency[-1]), height_min, float(height_max)


def warp_array(image, frequency_result, height_result, bounds, frequency_bins=512, height_bins=512):
    """Return normalized (height, frequency) data and its valid mask."""
    frequency_min, frequency_max, height_min, height_max = bounds
    target_frequency = np.linspace(frequency_min, frequency_max, frequency_bins)
    target_height = np.linspace(height_min, height_max, height_bins)
    source_x = inverse_mapping(
        target_frequency, frequency_result, "film_column", "frequency_mhz"
    )
    height_mapping = {
        "breakpoints": height_result.get("mapping_anchors")
        or height_result.get("breakpoints", [])
    }
    source_y = inverse_mapping(
        target_height, height_mapping, "film_row", "virtual_height_km"
    )
    grid_y, grid_x = np.meshgrid(source_y, source_x, indexing="ij")
    normalized, _ = normalize(image)
    warped = map_coordinates(
        normalized,
        [grid_y.ravel(), grid_x.ravel()],
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(height_bins, frequency_bins)
    valid = (
        (grid_x >= 0)
        & (grid_x <= image.shape[1] - 1)
        & (grid_y >= 0)
        & (grid_y <= image.shape[0] - 1)
    )
    warped[~valid] = np.nan
    return warped.astype(np.float32), valid, target_frequency, target_height


def write_figure(path, warped, frequency, height, title, status, coverage, warnings):
    finite = np.isfinite(warped)
    image = np.nan_to_num(warped, nan=1.0)
    figure, axis = plt.subplots(figsize=(11, 8), dpi=130)
    axis.imshow(
        image,
        cmap="gray",
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[float(frequency[0]), float(frequency[-1]), float(height[0]), float(height[-1])],
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xlabel("frequency (MHz)")
    axis.set_ylabel("height (km)")
    warning_text = "; ".join(warnings) or "none"
    axis.set_title(
        f"{title}\nstatus={status} | valid coverage={coverage:.1%} | warnings: {warning_text}",
        fontsize=9,
    )
    axis.set_xlim(float(frequency[0]), float(frequency[-1]))
    # Keep the final Phase 6 convention fixed: 0 km at the top and increasing
    # virtual height downward.
    axis.set_ylim(float(height[-1]), float(height[0]))
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def warp_one(image, frequency_result, height_result, structure, frequency_bins=512, height_bins=512):
    status = combined_status(frequency_result, height_result)
    result = {
        "schema": "isis.csa_warp_result.v1",
        "status": status,
        "frequency_status": frequency_result.get("status"),
        "height_status": height_result.get("status"),
        "frequency_source": frequency_result.get("source"),
        "height_source": height_result.get("source"),
        "height_mapping": height_result.get("mapping"),
        "height_mapping_anchor_count": height_result.get("mapping_anchor_count"),
        "warnings": sorted(
            set(frequency_result.get("warnings", []))
            | set(height_result.get("warnings", []))
        ),
        "frequency_bins": int(frequency_bins),
        "height_bins": int(height_bins),
    }
    if status == "not_usable":
        result["reason"] = "one_or_both_axis_calibrations_not_usable"
        return result, None
    try:
        bounds = target_bounds(structure, frequency_result, height_result)
        warped, valid, frequency, height = warp_array(
            image,
            frequency_result,
            height_result,
            bounds,
            frequency_bins,
            height_bins,
        )
    except (ValueError, KeyError) as error:
        result.update({"status": "not_usable", "reason": str(error)})
        return result, None
    result.update(
        {
            "mapping": "regular_frequency_height_grid",
            "frequency_min_mhz": float(frequency[0]),
            "frequency_max_mhz": float(frequency[-1]),
            "height_min_km": float(height[0]),
            "height_max_km": float(height[-1]),
            "valid_coverage": float(valid.mean()),
            "confidence": float(valid.mean()),
            "confidence_metric": "valid_mask_coverage",
            "source_image_shape": [int(image.shape[0]), int(image.shape[1])],
            "grid_shape": [int(height_bins), int(frequency_bins)],
        }
    )
    if result["valid_coverage"] < 0.80:
        result["warnings"].append("low_warp_valid_coverage")
        result["warnings"] = sorted(set(result["warnings"]))
        result["status"] = "review"
    return result, {
        "warped": warped,
        "valid": valid,
        "frequency": frequency,
        "height": height,
    }


def _write_outputs(out_dir, stem, result, arrays, title, write_plot=True, route=None):
    json_path = out_dir / f"{stem}_warp.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if arrays is None:
        return json_path, None, None
    npz_path = out_dir / f"{stem}_warp.npz"
    ionogram.write(
        npz_path,
        arrays["warped"],
        arrays["valid"],
        arrays["frequency"],
        arrays["height"],
        status=result["status"],
        route=route,
        # Phase 6 has no calibrated echo-confidence measure.  Its only
        # defensible scalar confidence is geometric support: the fraction of
        # target pixels backed by the source film.
        confidence=float(result["confidence"]),
        source={"phase6_stem": stem},
        provenance={
            "producer": "scripts/pipeline/warp_calibrated_scan.py",
            "legacy_phase6_schema": result["schema"],
            "frequency_source": result.get("frequency_source"),
            "height_source": result.get("height_source"),
            "confidence_metric": result["confidence_metric"],
            "confidence_note": "geometric support, not echo confidence",
        },
    )
    png_path = out_dir / f"{stem}_ionogram.png" if write_plot else None
    if write_plot:
        write_figure(
            png_path,
            arrays["warped"],
            arrays["frequency"],
            arrays["height"],
            title,
            result["status"],
            result["valid_coverage"],
            result["warnings"],
        )
    result["json_sidecar"] = str(json_path)
    result["npz_sidecar"] = str(npz_path)
    result["graph"] = str(png_path) if png_path else ""
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json_path, npz_path, png_path


def process_manifest(
    manifest_path=DEFAULT_MANIFEST,
    structure_dir=DEFAULT_STRUCTURE_DIR,
    frequency_dir=DEFAULT_FREQUENCY_DIR,
    height_dir=DEFAULT_HEIGHT_DIR,
    out_dir=DEFAULT_OUT,
    route="both",
    frequency_bins=512,
    height_bins=512,
    limit=None,
    write_plots=True,
):
    manifest_path, structure_dir, frequency_dir, height_dir, out_dir = map(
        Path, (manifest_path, structure_dir, frequency_dir, height_dir, out_dir)
    )
    rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
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
            frequency_result = load_json(
                frequency_dir / route_name / f"{stem}_frequency.json"
            )
            height_result = load_json(height_dir / route_name / f"{stem}_height.json")
            film_path = (manifest_path.parent / row["csa_link"]).resolve()
            image = np.asarray(Image.open(film_path).convert("L"), dtype=float)
            result, arrays = warp_one(
                image,
                frequency_result,
                height_result,
                structure,
                frequency_bins,
                height_bins,
            )
            json_path, npz_path, png_path = _write_outputs(
                route_dir, stem, result, arrays, stem, write_plots, route_name
            )
            record = {
                "route": route_name,
                "pair_number": row["pair_number"],
                "pair_name": row["pair_name"],
                "split": row["split"],
                "status": result["status"],
                "frequency_status": result["frequency_status"],
                "height_status": result["height_status"],
                "valid_coverage": result.get("valid_coverage"),
                "frequency_min_mhz": result.get("frequency_min_mhz"),
                "frequency_max_mhz": result.get("frequency_max_mhz"),
                "height_min_km": result.get("height_min_km"),
                "height_max_km": result.get("height_max_km"),
                "warnings": ";".join(result.get("warnings", [])),
                "json_sidecar": str(json_path.relative_to(out_dir)),
                "npz_sidecar": str(npz_path.relative_to(out_dir)) if npz_path else "",
                "graph": str(png_path.relative_to(out_dir)) if png_path else "",
            }
            records.append(record)
            print(
                f"{route_name:>12} {int(row['pair_number']):4d}/{len(rows)} "
                f"{result['status']:>10} coverage={result.get('valid_coverage', 0) or 0:.1%} "
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
        "# Phase 6 warped ionograms\n\n"
        f"Regular-grid renders for {len(rows)} scans and route(s): {', '.join(routes)}.\n\n"
        "Each successful result contains a PNG graph, a canonical ionogram NPZ, and a JSON calibration sidecar. The NPZ stores normalized warped intensity, a valid-pixel mask, frequency coordinates, virtual-height coordinates, and embedded status/provenance metadata. Its confidence is valid-pixel coverage: geometric support, not echo confidence. Height uses a constrained piecewise mapping when trusted CDF or ruling-lattice anchors are available, otherwise it falls back to the affine scale. The graph covers only the calibrated frequency range and the film's calibrated height region.\n\n"
        "A `review` graph is usable for inspection but carries warnings. `not_usable` cases receive a JSON reason and no graph. The `comparisons/` subdirectory contains three-panel images with the pure CDF, route-specific warped CSA, and raw CSA on shared axes; 0 km is at the top and virtual height increases downward.\n",
        encoding="utf-8",
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--structure-dir", type=Path, default=DEFAULT_STRUCTURE_DIR)
    parser.add_argument("--frequency-dir", type=Path, default=DEFAULT_FREQUENCY_DIR)
    parser.add_argument("--height-dir", type=Path, default=DEFAULT_HEIGHT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--route", choices=["cdf_assisted", "film_only", "both"], default="both")
    parser.add_argument("--frequency-bins", type=int, default=512)
    parser.add_argument("--height-bins", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    records = process_manifest(
        args.manifest,
        args.structure_dir,
        args.frequency_dir,
        args.height_dir,
        args.out_dir,
        args.route,
        args.frequency_bins,
        args.height_bins,
        args.limit,
        not args.no_plots,
    )
    print(f"wrote {len(records)} warp results to {args.out_dir}")


if __name__ == "__main__":
    main()
