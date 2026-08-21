#!/usr/bin/env python3
"""Run one raw CSA scan through calibration, inference, and CDF export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram  # noqa: E402
from isis_research.image_io import load_image  # noqa: E402
from isis_research.nasa.model_cdf import (  # noqa: E402
    export_model_cdf,
    header_from_csa,
)
from scripts.pipeline.extract_scan_structure import (  # noqa: E402
    extract_structure,
    write_overlay,
)
from scripts.pipeline.fit_frequency_axis import fit_from_profile, load_json  # noqa: E402
from scripts.pipeline.fit_height_axis import fit_from_profile as fit_height_from_profile  # noqa: E402
from scripts.pipeline.infer_isis_model import (  # noqa: E402
    candidate_checkpoint,
    infer,
    load_model_candidates,
)
from scripts.pipeline.standardize_film_only_512 import process as standardize  # noqa: E402
from scripts.pipeline.warp_calibrated_scan import warp_one, write_figure  # noqa: E402


DEFAULT_PROFILE = ROOT / "configs/film_calibration_profile.json"
DEFAULT_MODEL = load_model_candidates()["default_model"]
DEFAULT_CHECKPOINT = candidate_checkpoint(DEFAULT_MODEL)


def resolve_checkpoint(checkpoint, model_name=None):
    """Use a registered model when requested, otherwise keep the checkpoint."""
    return candidate_checkpoint(model_name) if model_name else checkpoint


def _write_diagnostics(film_path, output_dir, profile):
    image = load_image(film_path)
    structure = extract_structure(image)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_structure = dict(structure)
    write_overlay(
        output_dir / "structure_overlay.png",
        image,
        overlay_structure,
        Path(film_path).name,
    )

    observed = [item["x"] for item in structure["vertical_markers"]["candidates"]]
    frequency = fit_from_profile(observed, image.shape, profile, metadata={})
    height = fit_height_from_profile(structure, profile, frequency)
    result, arrays = warp_one(image, frequency, height, structure)

    serializable_structure = dict(structure)
    serializable_structure.pop("normalized_image", None)
    (output_dir / "structure.json").write_text(
        json.dumps(serializable_structure, indent=2), encoding="utf-8"
    )
    (output_dir / "frequency.json").write_text(
        json.dumps(frequency, indent=2), encoding="utf-8"
    )
    (output_dir / "height.json").write_text(
        json.dumps(height, indent=2), encoding="utf-8"
    )

    if arrays is not None:
        write_figure(
            output_dir / "warped.png",
            arrays["warped"],
            arrays["frequency"],
            arrays["height"],
            Path(film_path).name,
            result["status"],
            result["valid_coverage"],
            result["warnings"],
        )
        mask = np.where(arrays["valid"], 255, 0).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(output_dir / "valid_mask.png")

    (output_dir / "warp.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def run_scan(
    film_path,
    output_dir,
    *,
    checkpoint=DEFAULT_CHECKPOINT,
    profile_path=DEFAULT_PROFILE,
    pair_name=None,
    station="",
    diagnostics=False,
    model_name=None,
):
    film_path = Path(film_path)
    output_dir = Path(output_dir)
    checkpoint = resolve_checkpoint(checkpoint, model_name)
    profile = load_json(profile_path)
    row = standardize(film_path, profile, output_dir)
    if row["status"] != "usable" or not row.get("artifact"):
        reason = row.get("reason") or row["status"]
        if diagnostics:
            try:
                _write_diagnostics(film_path, output_dir / "diagnostics", profile)
            except Exception as error:
                reason = f"{reason}; diagnostics failed: {error}"
        raise ValueError(f"scan was not usable: {reason}")

    artifact = output_dir / row["artifact"]
    stem = film_path.stem
    prediction = output_dir / f"{stem}_prediction.npz"
    try:
        infer(artifact, checkpoint, prediction)
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError(
                "PyTorch is not installed in the active Python environment; "
                "activate .venv and run `python -m pip install -e '.[dev,notebooks]'`"
            ) from error
        raise

    scan = ionogram.read_validated(artifact)
    name = pair_name or stem
    header = header_from_csa(name, station, scan.frequency_mhz, scan.virtual_height_km)
    cdf = output_dir / f"{stem}_model.cdf"
    values, provenance = export_model_cdf(prediction, header, cdf)

    if diagnostics:
        _write_diagnostics(film_path, output_dir / "diagnostics", profile)

    summary = {
        "schema": "isis.single_scan_run.v1",
        "film": film_path.name,
        "artifact": str(Path(row["artifact"])),
        "prediction": prediction.name,
        "cdf": cdf.name,
        "status": scan.meta["status"],
        "grid_shape": list(scan.intensity.shape),
        "frequency_mhz": [float(scan.frequency_mhz[0]), float(scan.frequency_mhz[-1])],
        "virtual_height_km": [
            float(scan.virtual_height_km[0]),
            float(scan.virtual_height_km[-1]),
        ],
        "ampl_shape": list(values["ampl"].shape),
        "provenance": provenance,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("film", type=Path, help="raw CSA PNG to process")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run")
    parser.add_argument(
        "--model",
        choices=tuple(load_model_candidates()["models"]),
        help=f"registered model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="custom checkpoint; --model takes precedence when both are supplied",
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="calibration profile JSON")
    parser.add_argument("--pair-name", help="observation name for the exported CDF header")
    parser.add_argument("--station", default="", help="CSA station code for the exported CDF header")
    parser.add_argument("--diagnostics", action="store_true", help="write static inspection products")
    args = parser.parse_args(argv)
    summary = run_scan(
        args.film,
        args.output,
        checkpoint=args.checkpoint,
        profile_path=args.profile,
        pair_name=args.pair_name,
        station=args.station,
        diagnostics=args.diagnostics,
        model_name=args.model,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
