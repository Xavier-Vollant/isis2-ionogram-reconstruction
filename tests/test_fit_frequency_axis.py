"""Tests for frequency-axis fitting and coverage warnings."""

import numpy as np

from scripts.pipeline.fit_frequency_axis import fit_reference


def test_frequency_axis_is_monotonic_piecewise_mapping():
    result = fit_reference(
        [120.0, 140.0, 170.0, 210.0, 260.0, 320.0, 380.0, 450.0],
        {
            "marker_columns": [10.0, 20.0, 35.0, 55.0, 80.0, 110.0, 140.0, 175.0],
            "marker_frequencies": [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "source": "test",
        },
    )
    assert result["status"] == "usable"
    assert result["mapping"] == "monotonic_piecewise_linear"
    assert result["matched_marker_count"] == 8
    assert np.all(np.diff([item["film_column"] for item in result["breakpoints"]]) > 0)
    assert np.all(
        np.diff([item["frequency_mhz"] for item in result["breakpoints"]]) > 0
    )


def test_frequency_axis_keeps_partial_coverage_as_a_warning():
    result = fit_reference(
        [110.0, 130.0, 150.0, 180.0],
        {
            "marker_columns": [10.0, 20.0, 35.0, 55.0, 80.0, 110.0, 140.0, 175.0],
            "marker_frequencies": [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "source": "test",
        },
    )
    assert result["status"] in {"review", "usable"}
    assert result["marker_coverage"] < 1.0
    assert "partial_frequency_marker_coverage" in result["warnings"]
