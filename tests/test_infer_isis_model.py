"""Tests for model checkpoint lookup and output calibration."""

from pathlib import Path

import numpy as np

from scripts.pipeline.infer_isis_model import (
    calibrate_prediction,
    candidate_checkpoint,
    load_model_candidates,
)


def test_registered_finalists_have_local_checkpoints():
    candidates = load_model_candidates()
    assert candidates["default_model"] == "norm_residual_unet"
    assert set(candidates["models"]) == {
        "contrast_aware_norm_residual",
        "norm_residual_unet",
        "hybrid_unet",
    }
    for name in candidates["models"]:
        assert Path(candidate_checkpoint(name)).is_file()


def test_calibration_clips_and_preserves_unit_output():
    output = calibrate_prediction(
        np.array([[-1.0, 0.5, 2.0]], dtype=np.float32),
        {"scale": 2.0, "bias": -0.5},
    )
    np.testing.assert_allclose(output, [[0.0, 0.5, 1.0]])
