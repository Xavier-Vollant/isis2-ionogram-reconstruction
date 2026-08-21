import numpy as np

from isis_research import ionogram
from scripts.pipeline.warp_calibrated_scan import _write_outputs, inverse_mapping, warp_array


def test_inverse_mapping_is_monotonic_and_handles_interpolation():
    mapping = {
        "breakpoints": [
            {"film_column": 10.0, "frequency_mhz": 1.0},
            {"film_column": 20.0, "frequency_mhz": 3.0},
            {"film_column": 40.0, "frequency_mhz": 7.0},
        ]
    }
    result = inverse_mapping(
        np.array([1.0, 2.0, 5.0, 7.0]), mapping, "film_column", "frequency_mhz"
    )
    assert np.allclose(result, [10.0, 15.0, 30.0, 40.0])


def test_warp_array_returns_regular_grid_and_valid_mask():
    image = np.arange(100, dtype=float).reshape(10, 10)
    frequency = {
        "breakpoints": [
            {"film_column": 0.0, "frequency_mhz": 1.0},
            {"film_column": 9.0, "frequency_mhz": 10.0},
        ]
    }
    height = {
        "breakpoints": [
            {"film_row": 0.0, "virtual_height_km": 0.0},
            {"film_row": 9.0, "virtual_height_km": 900.0},
        ]
    }
    warped, valid, frequencies, heights = warp_array(
        image, frequency, height, (1.0, 10.0, 0.0, 900.0), 8, 6
    )
    assert warped.shape == (6, 8)
    assert valid.shape == warped.shape
    assert valid.all()
    assert np.allclose(frequencies[[0, -1]], [1.0, 10.0])
    assert np.allclose(heights[[0, -1]], [0.0, 900.0])


def test_phase6_writer_emits_a_validated_canonical_artifact(tmp_path):
    arrays = {
        "warped": np.full((6, 8), 0.8, dtype=np.float32),
        "valid": np.ones((6, 8), dtype=bool),
        "frequency": np.linspace(1.0, 10.0, 8),
        "height": np.linspace(0.0, 900.0, 6),
    }
    result = {
        "schema": "isis.csa_warp_result.v1",
        "status": "review",
        "frequency_source": "film_only_profile",
        "height_source": "film_only_profile",
        "valid_coverage": 1.0,
        "confidence": 1.0,
        "confidence_metric": "valid_mask_coverage",
        "warnings": [],
    }

    _, path, _ = _write_outputs(
        tmp_path, "scan", result, arrays, "scan", write_plot=False, route="film_only"
    )

    artifact = ionogram.read_validated(path)
    assert artifact.meta["status"] == "review"
    assert artifact.meta["route"] == "film_only"
    assert artifact.meta["confidence"] == 1.0
    assert artifact.meta["provenance"]["legacy_phase6_schema"] == "isis.csa_warp_result.v1"
    assert artifact.meta["provenance"]["confidence_metric"] == "valid_mask_coverage"
