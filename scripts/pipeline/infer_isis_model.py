#!/usr/bin/env python3
"""Run a registered CSA-to-amplitude model on one validated artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram
from isis_research.models import image_features, model_constructor

DEFAULT_MODEL_CONFIG = ROOT / "configs/model_candidates.json"


def load_model_candidates(path=DEFAULT_MODEL_CONFIG):
    """Load and validate the registered model/checkpoint configuration."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "isis.model_candidates.v1":
        raise ValueError(f"unsupported model candidate schema in {path}")
    if not document.get("models"):
        raise ValueError(f"model candidate config is empty: {path}")
    return document


def candidate_checkpoint(name, path=DEFAULT_MODEL_CONFIG):
    """Resolve a registered model name to its local checkpoint path."""
    candidates = load_model_candidates(path)["models"]
    try:
        checkpoint = Path(candidates[name]["checkpoint"])
    except KeyError as error:
        raise ValueError(f"unsupported model candidate: {name!r}") from error
    return checkpoint if checkpoint.is_absolute() else ROOT / checkpoint


def calibrate_prediction(prediction, calibration):
    """Apply the historical training-only affine contrast calibration."""
    if not calibration:
        return np.asarray(prediction, dtype=np.float32)
    return np.clip(
        float(calibration["scale"]) * np.asarray(prediction)
        + float(calibration["bias"]),
        0.0,
        1.0,
    ).astype(np.float32)


def _candidate_name_for_checkpoint(checkpoint, document):
    checkpoint = Path(checkpoint).resolve()
    for name, item in document["models"].items():
        configured = Path(item["checkpoint"])
        configured = configured if configured.is_absolute() else ROOT / configured
        if configured.resolve() == checkpoint:
            return name
    return None


def load_model(path, torch):
    """Create a checkpoint's architecture and load its saved weights."""
    checkpoint = torch.load(path, map_location="cpu")
    model_name = checkpoint.get("model")
    channels = int(checkpoint.get("input_channels", 1))
    model = model_constructor(model_name)(channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, channels, model_name


def infer(
    artifact,
    checkpoint,
    output,
    *,
    model_name=None,
    calibrated=None,
    model_config=DEFAULT_MODEL_CONFIG,
):
    """Run calibrated model inference on one validated ionogram artifact."""
    import torch

    candidates = load_model_candidates(model_config)
    candidate_name = model_name or _candidate_name_for_checkpoint(
        checkpoint, candidates
    )
    candidate = candidates["models"].get(candidate_name) if candidate_name else None
    if model_name and candidate is None:
        raise ValueError(f"unsupported model candidate: {model_name!r}")
    if (
        candidate_name
        and Path(checkpoint).resolve()
        != candidate_checkpoint(candidate_name, model_config).resolve()
    ):
        raise ValueError(f"checkpoint does not match model candidate: {candidate_name}")
    if calibrated is None:
        calibrated = candidate is not None
    if calibrated and candidate is None:
        raise ValueError("calibrated inference requires a registered model candidate")

    scan = ionogram.read_validated(artifact)
    signal = np.where(scan.valid_mask, 1.0 - scan.intensity, 0.0).astype(np.float32)
    model, channels, architecture = load_model(checkpoint, torch)
    features = image_features(signal, channels)
    with torch.inference_mode():
        prediction = torch.sigmoid(model(torch.from_numpy(features[None])))
    prediction = prediction.squeeze().cpu().numpy().astype(np.float32)
    prediction = calibrate_prediction(
        prediction, candidate.get("calibration") if calibrated and candidate else None
    )
    np.savez_compressed(
        output,
        prediction=prediction,
        frequency_mhz=scan.frequency_mhz,
        virtual_height_km=scan.virtual_height_km,
        valid_mask=scan.valid_mask,
        meta_json=json.dumps(
            {
                "producer": "scripts/pipeline/infer_isis_model.py",
                "model": candidate_name or architecture,
                "architecture": architecture,
                "checkpoint": Path(checkpoint).name,
                "calibrated": bool(calibrated),
                "calibration": candidate.get("calibration")
                if calibrated and candidate
                else None,
                "orientation": "height,frequency",
                "scale": "unit",
                "source_artifact": Path(artifact).name,
            }
        ),
    )
    return prediction


def main(argv=None):
    """Parse CLI options and run model inference on one artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="validated isis.ionogram.v1 NPZ")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=candidate_checkpoint("norm_residual_unet"),
    )
    parser.add_argument(
        "--model",
        choices=tuple(load_model_candidates()["models"]),
        help="registered model",
    )
    parser.add_argument(
        "--uncalibrated", action="store_true", help="skip model contrast calibration"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = candidate_checkpoint(args.model) if args.model else args.checkpoint
    prediction = infer(
        args.artifact,
        checkpoint,
        args.output,
        model_name=args.model,
        calibrated=False if args.uncalibrated else None,
    )
    print(json.dumps({"output": str(args.output), "shape": list(prediction.shape)}))


if __name__ == "__main__":
    main()
