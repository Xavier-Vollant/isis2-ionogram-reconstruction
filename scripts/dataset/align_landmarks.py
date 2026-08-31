#!/usr/bin/env python
"""Align landmarks shared by one NASA CDF and one CSA film scan.

Pass `--film` and `--cdf` to use a different pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cdflib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from isis_research.extraction.echo import detrend
from isis_research.registration import calibrate, film, landmarks

ROOT = Path(__file__).resolve().parents[2]
TARGET = "i2_av_ksh_1972357075017_v01"
DEFAULT_FILM = ROOT / "data/raw/matches/csa_png/B1-35-25_ISIS_B_D-759_Image0243.png"
DEFAULT_CDF = ROOT / f"data/raw/matches/nasa_cdf/{TARGET}.cdf"
DEFAULT_OUT = ROOT / "outputs/landmarks"


def normalize(array):
    """Scale an array to display range without changing its shape."""
    array = np.asarray(array, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array)
    low, high = np.nanpercentile(array[finite], [2, 98])
    return np.clip((np.nan_to_num(array) - low) / max(high - low, 1e-9), 0, 1)


def add_frequency_axis(axis, frequency, sweep_start=None):
    """Show exact CDF per-scan-line frequency values on a secondary x-axis."""
    if frequency is None:
        return
    frequency = np.asarray(frequency, dtype=float)
    valid = np.flatnonzero(np.isfinite(frequency) & (frequency > 0))
    if len(valid) < 2:
        return
    positions = np.rint(np.linspace(valid[0], valid[-1], min(9, len(valid)))).astype(
        int
    )
    if sweep_start is not None and valid[0] <= sweep_start <= valid[-1]:
        positions = np.unique(np.append(positions, int(sweep_start)))
    top = axis.twiny()
    top.set_xlim(axis.get_xlim())
    top.set_xticks(positions)
    top.set_xticklabels([f"{frequency[i]:g}" for i in positions], fontsize=7)
    top.set_xlabel("CDF freq (MHz)", fontsize=8)


def draw_csa(axis, image, result):
    axis.imshow(image, cmap="gray", aspect="auto", interpolation="nearest")
    top, bottom = result["film"]["top_row"], result["film"]["bottom_row"]
    axis.axhline(top, color="#00d084", linewidth=1.4, label="top of ionogram")
    axis.axhline(bottom, color="#ff8c00", linewidth=1.4, label="bottom of ionogram")

    for marker in result["features"]:
        if marker["name"] in ("frequency_marker", "frequency_marker_unmatched"):
            if marker["name"] == "frequency_marker":
                x = marker["csa_x"]
                colour, style = "#2589bd", "-"
            else:
                x = marker["predicted_csa_x"]
                colour, style = "#777777", ":"
            if not (0 <= x <= image.shape[1] - 1):
                x = float(np.clip(x, 0, image.shape[1] - 1))
                label = f"{marker['frequency_mhz']:g} MHz not on scan"
            else:
                label = f"{marker['frequency_mhz']:g}"
            axis.axvline(x, color=colour, linestyle=style, linewidth=0.8, alpha=0.8)
            axis.text(
                x,
                max(1, top + 2),
                label,
                color=colour,
                fontsize=7,
                rotation=90,
                va="top",
                ha="right",
                clip_on=True,
            )
    sweep = next(
        item
        for item in result["features"]
        if item["name"] == "start_of_frequency_sweep"
    )
    if sweep["csa_x"] is not None:
        sweep_x, colour, style, label = (
            sweep["csa_x"],
            "#8e44ad",
            "--",
            "start of frequency sweep",
        )
    else:
        sweep_x, colour, style, label = (
            sweep["predicted_csa_x"],
            "#777777",
            ":",
            "start of frequency sweep (predicted, not confirmed)",
        )
    axis.axvline(sweep_x, color=colour, linewidth=1.8, linestyle=style)
    axis.text(
        sweep_x + 4,
        top + 4,
        label,
        color=colour,
        fontsize=8,
        rotation=90,
        va="top",
    )
    for candidate in result["csa_horizontal_candidates"]:
        axis.axhline(candidate["csa_row"], color="#9e9e9e", linewidth=0.45, alpha=0.35)
    consensus_labels = result.get("csa_consensus_ruling_labels", [])
    for index, item in enumerate(consensus_labels):
        exact = item["label_quality"] == "verified_height"
        colour = "#d81b60" if exact else "#f39c12"
        label = "NASA-height verified" if exact else "cross-scan consensus"
        axis.axhline(
            item["csa_row"],
            color=colour,
            linewidth=1.8 if exact else 1.15,
            linestyle="--" if exact else ":",
            alpha=0.95,
            label=label if index == 0 or exact else None,
        )
        axis.text(
            4,
            item["csa_row"] - 3,
            f"R{item['relative_index']:+d} · {item['label_quality']}",
            color=colour,
            fontsize=7,
            ha="left",
            va="bottom",
        )
    if not consensus_labels:
        labels = [
            item
            for item in result["csa_ruling_labels"]
            if item["status"] == "verified_height_ruling"
        ]
        for item in labels:
            axis.axhline(
                item["csa_row"], color="#d81b60", linewidth=1.8, linestyle="--"
            )
            axis.text(
                image.shape[1] - 4,
                item["csa_row"] + 10,
                f"NASA {item['nearest_nasa_height_km']:.0f} km → CSA y={item['csa_row']:.0f}",
                color="#ad1457",
                fontsize=8,
                ha="right",
                va="top",
            )
    for item in result["csa_ruling_review_targets"]:
        axis.text(
            image.shape[1] - 4,
            item["csa_row"] - 4,
            f"review {item['review_priority']} · R{item['relative_to_verified_index']}",
            color="#f39c12",
            fontsize=7,
            ha="right",
            va="bottom",
        )
    axis.set_title(
        "CSA scan — orange consensus ruling labels; magenta exact NASA match",
        fontsize=10,
    )
    axis.set_ylabel("CSA pixel row")
    axis.set_xlim(0, image.shape[1] - 1)
    axis.set_ylim(image.shape[0] - 1, 0)
    axis.legend(loc="lower left", fontsize=7)


def draw_nasa(axis, amplitude, v_height, result):
    image = axis.imshow(
        amplitude.T,
        cmap="gray_r",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=255,
        extent=[0, amplitude.shape[0] - 1, float(v_height[-1]), float(v_height[0])],
    )
    axis.figure.colorbar(
        image, ax=axis, pad=0.01, fraction=0.025, label="CDF ampl (raw 0–255)"
    )
    nasa = result["nasa"]
    axis.axhline(0, color="#00d084", linewidth=1.4, label="top of ionogram")
    axis.axhline(
        v_height[-1], color="#ff8c00", linewidth=1.4, label="bottom of ionogram"
    )
    for column, frequency in zip(nasa["marker_columns"], nasa["marker_frequencies"]):
        axis.axvline(column, color="#2589bd", linewidth=0.8, alpha=0.8)
        axis.text(
            column,
            5,
            f"{frequency:g}",
            color="#075985",
            fontsize=7,
            rotation=90,
            va="top",
            ha="right",
            clip_on=True,
        )
    axis.axvline(nasa["sweep_start"], color="#8e44ad", linewidth=1.8, linestyle="--")
    axis.text(
        nasa["sweep_start"] + 4,
        40,
        "start of frequency sweep",
        color="#8e44ad",
        fontsize=8,
        rotation=90,
        va="top",
    )
    for item in result["nasa_horizontal_rows"]:
        colour = (
            "#d81b60"
            if item["cdf_row"]
            in {
                match["cdf_row"]
                for match in result["horizontal_matches"]
                if match["status"] == "matched_csa_candidate"
            }
            else "#777777"
        )
        axis.axhline(
            item["virtual_height_km"], color=colour, linewidth=1.8, linestyle="--"
        )
        axis.text(
            amplitude.shape[0] - 4,
            item["virtual_height_km"] - 18,
            f"NASA row {item['virtual_height_km']:.0f} km",
            color=colour,
            fontsize=8,
            ha="right",
            va="bottom",
        )
    axis.set_title("NASA CDF — candidate horizontal rows", fontsize=10)
    axis.set_ylabel("virtual height (km)")
    axis.set_xlim(0, amplitude.shape[0] - 1)
    add_frequency_axis(axis, result["nasa"].get("frequency"), nasa.get("sweep_start"))
    axis.legend(loc="lower left", fontsize=7)


def draw_common_grid(axis, warped, amplitude, v_height, result):
    """Draw the calibrated CSA/NASA comparison on a shared physical grid."""
    nasa = normalize(amplitude.T)
    csa = normalize(warped.T)
    rgb = np.zeros(nasa.shape + (3,))
    rgb[..., 0] = nasa
    rgb[..., 1] = csa
    rgb[..., 2] = csa
    axis.imshow(
        1.0 - rgb,
        aspect="auto",
        interpolation="nearest",
        extent=[0, amplitude.shape[0] - 1, float(v_height[-1]), float(v_height[0])],
    )
    for column in result["nasa"]["marker_columns"]:
        axis.axvline(column, color="#2589bd", linewidth=0.6, alpha=0.55)
    axis.axvline(
        result["nasa"]["sweep_start"], color="#8e44ad", linewidth=1.3, linestyle="--"
    )
    for item in result.get("csa_consensus_ruling_labels", []):
        exact = item["label_quality"] == "verified_height"
        axis.axhline(
            item["estimated_virtual_height_km"],
            color="#d81b60" if exact else "#f39c12",
            linewidth=1.2 if exact else 0.8,
            linestyle="--" if exact else ":",
            alpha=0.85,
        )
    axis.set_title("Warped common grid — red: NASA, cyan: CSA", fontsize=10)
    axis.set_xlabel("NASA scan-line index; marker labels are MHz")
    axis.set_ylabel("virtual height (km)")
    add_frequency_axis(
        axis, result["nasa"].get("frequency"), result["nasa"].get("sweep_start")
    )


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items() if key != "profile"}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def swept_frequency_band(frequency, sweep_start):
    """-> (low, high) MHz spanning the scan's own swept range.

    A fixed absolute band assumes every pass's usable echo sits in the same
    MHz range; high-latitude passes routinely show it far outside a band
    tuned to one reference station, or sweep to a much lower top frequency,
    so the band is derived from this scan's own swept frequencies instead of
    a constant. ``sweep_start`` already excludes the fixed-frequency preamble.
    """
    if frequency is None:
        return 0.0, 1e6
    frequency = np.asarray(frequency, dtype=float)
    valid = np.isfinite(frequency) & (frequency > 0) & (frequency < 1e6)
    valid[: int(sweep_start)] = False
    if not valid.any():
        return 0.0, 1e6
    return float(frequency[valid].min()) - 0.01, float(frequency[valid].max()) + 0.01


def calibrate_vertical_if_better(image, amplitude, epoch, frequency, v_height, result):
    """Keep an affine vertical fit only when independent traces improve."""
    base = film.Geometry(
        np.asarray(result["geometry"]["coefficients"]),
        result["geometry"]["zero_row"],
        result["geometry"]["px_per_km"],
    )
    line_time = np.asarray(epoch, dtype=float) - float(epoch[0])
    nasa = detrend(np.asarray(amplitude, dtype=float))
    window = calibrate.evaluation_window(v_height, amplitude.shape)
    low, high = swept_frequency_band(frequency, result["nasa"]["sweep_start"])
    baseline_score, baseline_warp = calibrate.objective(
        image,
        base,
        base.zero_row,
        base.px_per_km,
        line_time,
        v_height,
        nasa,
        window,
    )
    baseline_trace = calibrate.trace_offset(
        baseline_warp, amplitude, frequency, v_height, low, high
    )

    coarse = calibrate.search(
        image,
        base,
        line_time,
        v_height,
        nasa,
        window,
        base.px_per_km * np.linspace(0.70, 1.60, 15),
        base.zero_row + np.arange(-40, 20.1, 3.0),
    )
    candidate_score, zero_row, px_per_km = calibrate.search(
        image,
        base,
        line_time,
        v_height,
        nasa,
        window,
        coarse[2] * np.linspace(0.96, 1.04, 9),
        coarse[1] + np.arange(-1.5, 1.51, 0.25),
    )
    candidate_geometry = base.with_vertical(zero_row, px_per_km)
    candidate_warp = film.resample(image, candidate_geometry, line_time, v_height)
    candidate_trace = calibrate.trace_offset(
        candidate_warp, amplitude, frequency, v_height, low, high
    )
    step = float(v_height[1] - v_height[0])
    improved = bool(
        candidate_trace
        and candidate_trace["n"] >= 10
        and candidate_trace.get("within_60km", 0.0) >= 0.35
        and candidate_score >= baseline_score
        and (
            not baseline_trace
            or (
                abs(candidate_trace["median_km"]) < abs(baseline_trace["median_km"])
                and candidate_trace["within_1bin"] >= baseline_trace["within_1bin"]
            )
        )
    )
    calibration = {
        "status": "accepted_affine" if improved else "kept_top_bottom",
        "method": "mutual_information_with_independent_trace_gate",
        "height_step_km": step,
        "baseline": {
            "zero_row": base.zero_row,
            "px_per_km": base.px_per_km,
            "mi": float(baseline_score),
            "trace": baseline_trace,
        },
        "candidate": {
            "zero_row": zero_row,
            "px_per_km": px_per_km,
            "mi": float(candidate_score),
            "trace": candidate_trace,
        },
    }
    if improved:
        result["geometry"].update(
            {
                "zero_row": float(zero_row),
                "px_per_km": float(px_per_km),
                "vertical_heights": np.asarray(
                    [v_height[0], v_height[-1]], dtype=float
                ),
                "vertical_rows": np.asarray(
                    [zero_row, zero_row + px_per_km * v_height[-1]], dtype=float
                ),
            }
        )
        result["nasa_middle_marking"]["calibrated_predicted_csa_row"] = float(
            zero_row + px_per_km * result["nasa_middle_marking"]["virtual_height_km"]
        )
    result["vertical_calibration"] = calibration
    return result


def main():
    """Parse CLI options and align one CSA/CDF pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--film", type=Path, default=DEFAULT_FILM)
    parser.add_argument("--cdf", type=Path, default=DEFAULT_CDF)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--consensus",
        type=Path,
        default=DEFAULT_OUT / "ruling_consensus.json",
        help="cross-scan ruling consensus JSON used for weak labels",
    )
    parser.add_argument("--marker-sigma", type=float, default=2.0)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the expensive mutual-information vertical refinement",
    )
    args = parser.parse_args()

    image = np.asarray(Image.open(args.film).convert("L"), dtype=float)
    cdf = cdflib.CDF(str(args.cdf))
    amplitude = np.asarray(cdf.varget("ampl"), dtype=float)
    epoch = np.asarray(cdf.varget("Epoch"), dtype=float)
    v_height = np.asarray(cdf.varget("v_height"), dtype=float)
    result = landmarks.align(
        image,
        amplitude,
        epoch,
        cdf.varget("Time_mark"),
        cdf.varget("freq_mark"),
        cdf.varget("swept_start"),
        args.marker_sigma,
        v_height,
        cdf.varget("freq"),
    )
    if (
        not args.fast
        and result["x_fit"]["count"] >= 6
        and result["x_fit"]["rms_px"] <= 3.0
    ):
        result = calibrate_vertical_if_better(
            image, amplitude, epoch, result["nasa"].get("frequency"), v_height, result
        )
    else:
        result["vertical_calibration"] = {
            "status": "skipped_fast_mode"
            if args.fast
            else "skipped_bad_horizontal_fit",
            "method": "fast_landmark_batch" if args.fast else "quality_gate",
            "baseline": {"trace": None},
            "candidate": {"trace": None},
        }
    result["csa_ruling_labels"] = landmarks.label_csa_rulings(
        result["csa_horizontal_candidates"],
        result["csa_ruling_lattice"],
        result["nasa_horizontal_rows"],
        result["geometry"]["zero_row"],
        result["geometry"]["px_per_km"],
        float(v_height[1] - v_height[0]),
    )
    for item in result["csa_ruling_labels"]:
        item["local_nasa_csa_similarity"] = landmarks.score_csa_ruling_local_similarity(
            image,
            amplitude,
            v_height,
            item["csa_row"],
            item["estimated_virtual_height_km"],
            result["x_fit"]["coefficients"],
            result["nasa"]["marker_columns"],
        )
    result["csa_ruling_review_targets"] = landmarks.select_ruling_review_targets(
        result["csa_ruling_labels"]
    )
    consensus_path = args.consensus
    consensus = (
        json.loads(consensus_path.read_text(encoding="utf-8"))
        if consensus_path.exists()
        else {}
    )
    result["csa_consensus_ruling_labels"] = landmarks.label_consensus_rulings(
        result["csa_ruling_lattice"],
        result["nasa_middle_marking"].get("csa_row"),
        result["csa_ruling_labels"],
        consensus,
        result["geometry"]["zero_row"],
        result["geometry"]["px_per_km"],
        image.shape[0],
    )
    result["csa_ruling_consensus_metadata"] = {
        "source": str(consensus_path) if consensus else None,
        "stable_ruling_count": len(result["csa_consensus_ruling_labels"]),
        "exact_height_claim": False,
    }
    result["csa_ml_labels"] = landmarks.build_ml_labels(result, image.shape)
    fitted = film.Geometry(
        np.asarray(result["geometry"]["coefficients"]),
        result["geometry"]["zero_row"],
        result["geometry"]["px_per_km"],
        np.asarray(result["geometry"]["vertical_heights"]),
        np.asarray(result["geometry"]["vertical_rows"]),
    )
    warped = film.resample(image, fitted, epoch - epoch[0], v_height)
    out = args.out or (DEFAULT_OUT / f"{args.cdf.stem}_landmarks.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(16, 17), dpi=130)
    draw_csa(axes[0], image, result)
    draw_nasa(axes[1], amplitude, v_height, result)
    draw_common_grid(axes[2], warped, amplitude, v_height, result)
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)

    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    npz_path = out.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        warped_film=warped.astype(np.float32),
        nasa_amplitude=amplitude.astype(np.uint8),
        freq=np.asarray(result["nasa"]["frequency"]),
        v_height=v_height,
        marker_cdf_columns=result["marker_reference"],
        marker_csa_x=result["marker_observed"],
        nasa_middle_cdf_row=np.asarray(result["nasa_middle_marking"]["cdf_row"]),
        nasa_middle_height=np.asarray(
            result["nasa_middle_marking"]["virtual_height_km"]
        ),
        nasa_middle_projected_csa_row=np.asarray(
            result["nasa_middle_marking"]["predicted_csa_row"]
        ),
        nasa_middle_matched_csa_row=np.asarray(
            result["nasa_middle_marking"]["csa_row"]
            if result["nasa_middle_marking"]["csa_row"] is not None
            else np.nan
        ),
        vertical_warp_heights=np.asarray(result["vertical_warp"]["heights"]),
        vertical_warp_rows=np.asarray(result["vertical_warp"]["rows"]),
        vertical_calibration_zero_row=np.asarray(result["geometry"]["zero_row"]),
        vertical_calibration_px_per_km=np.asarray(result["geometry"]["px_per_km"]),
        csa_ruling_rows=np.asarray(
            [item["csa_row"] for item in result["csa_ruling_labels"]]
        ),
        csa_ruling_heights=np.asarray(
            [
                item["estimated_virtual_height_km"]
                for item in result["csa_ruling_labels"]
            ]
        ),
        csa_ruling_label_mask=np.asarray(
            [
                item["status"] == "verified_height_ruling"
                for item in result["csa_ruling_labels"]
            ]
        ),
        csa_ruling_weak_mask=np.asarray(
            [
                item["status"] == "relative_ruling_candidate"
                for item in result["csa_ruling_labels"]
            ]
        ),
        csa_consensus_ruling_rows=np.asarray(
            [item["csa_row"] for item in result["csa_consensus_ruling_labels"]],
            dtype=np.float32,
        ),
        csa_consensus_ruling_relative_indices=np.asarray(
            [item["relative_index"] for item in result["csa_consensus_ruling_labels"]],
            dtype=np.int16,
        ),
        csa_consensus_ruling_mask=np.asarray(
            [
                any(
                    item["row_start"] <= row <= item["row_end"]
                    for item in result["csa_consensus_ruling_labels"]
                )
                for row in range(image.shape[0])
            ],
            dtype=np.uint8,
        ),
        csa_consensus_exact_mask=np.asarray(
            [
                any(
                    item["label_quality"] == "verified_height"
                    and item["row_start"] <= row <= item["row_end"]
                    for item in result["csa_consensus_ruling_labels"]
                )
                for row in range(image.shape[0])
            ],
            dtype=np.uint8,
        ),
        csa_consensus_weak_mask=np.asarray(
            [
                any(
                    item["weak_label"] and item["row_start"] <= row <= item["row_end"]
                    for item in result["csa_consensus_ruling_labels"]
                )
                for row in range(image.shape[0])
            ],
            dtype=np.uint8,
        ),
        csa_consensus_ruling_pixel_mask=np.broadcast_to(
            np.asarray(
                [
                    any(
                        item["row_start"] <= row <= item["row_end"]
                        for item in result["csa_consensus_ruling_labels"]
                    )
                    for row in range(image.shape[0])
                ],
                dtype=np.uint8,
            )[:, None],
            image.shape,
        ),
    )
    ml_json_path = out.with_name(f"{out.stem}_ml_labels.json")
    ml_json_path.write_text(
        json.dumps(jsonable(result["csa_ml_labels"]), indent=2),
        encoding="utf-8",
    )

    print(
        f"markers: {result['x_fit']['count']} matched, RMS {result['x_fit']['rms_px']:.2f}px"
    )
    print(f"vertical landmarks: RMS {result['y_rms_px']:.2f}px")
    middle = result["nasa_middle_marking"]
    print(
        f"NASA middle marking: CDF row={middle['cdf_row']}, "
        f"height={middle['virtual_height_km']:.0f} km -> "
        f"predicted CSA row={middle['predicted_csa_row']:.1f}; "
        f"matched CSA row={middle['csa_row']} ({middle['status']})"
    )
    lattice = result["csa_ruling_lattice"]
    print(
        f"CSA ruling lattice: {lattice.get('count', 0)} regular candidates, "
        f"spacing={lattice.get('spacing_px')} px"
    )
    print(
        f"vertical warp: {result['vertical_warp']['status']} "
        f"({result['vertical_warp']['anchor_count']} anchors)"
    )
    calibration = result["vertical_calibration"]
    print(
        f"affine vertical calibration: {calibration['status']} "
        f"(baseline median={calibration['baseline'].get('trace', {}).get('median_km') if calibration['baseline'].get('trace') else None} km, "
        f"candidate median={calibration['candidate'].get('trace', {}).get('median_km') if calibration['candidate'].get('trace') else None} km)"
    )
    print(
        f"CSA ruling labels: {sum(item['status'] == 'verified_height_ruling' for item in result['csa_ruling_labels'])} verified, "
        f"{sum(item['status'] == 'relative_ruling_candidate' for item in result['csa_ruling_labels'])} relative candidates, "
        f"{len(result['csa_ruling_review_targets'])} review targets"
    )
    ml_labels = result["csa_ml_labels"]["labels"]
    print(
        f"ML labels: {len(ml_labels)} in "
        f"{len({item['class_name'] for item in ml_labels})} classes "
        f"({sum(item['weak_label'] for item in ml_labels)} weak), "
        f"{len(result['csa_ml_labels']['ignore'])} ignore regions"
    )
    for item in result["features"]:
        if item["name"] == "frequency_marker":
            print(
                f"  marker {item['frequency_mhz']:>5g} MHz: CSA x={item['csa_x']:.1f}, residual={item['residual_px']:+.1f}px"
            )
        elif item["name"] == "frequency_marker_unmatched":
            print(
                f"  marker {item['frequency_mhz']:>5g} MHz: not observed on CSA scan (predicted x={item['predicted_csa_x']:.1f})"
            )
        elif item["name"] == "start_of_frequency_sweep":
            if item["csa_x"] is not None:
                print(
                    f"  start of frequency sweep: CDF x={item['cdf_column']:.1f}, CSA x={item['csa_x']:.1f}"
                )
            else:
                print(
                    f"  start of frequency sweep: CDF x={item['cdf_column']:.1f}, "
                    f"not observed on CSA scan (predicted x={item['predicted_csa_x']:.1f})"
                )
        else:
            print(
                f"  {item['name']}: CSA y={item['csa_y']:.1f}, residual={item['residual_px']:+.1f}px"
            )
    print(f"wrote {out}")
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(f"wrote {ml_json_path}")


if __name__ == "__main__":
    sys.exit(main())
