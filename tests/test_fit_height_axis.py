"""Tests for virtual-height calibration paths."""

import numpy as np

from scripts.pipeline.fit_height_axis import (
    fit_from_landmark_reference,
    fit_from_profile,
)


def _structure():
    return {
        "image_shape": [120, 200],
        "film_region": {"top_row": 10.0, "bottom_row": 110.0},
        "horizontal_rulings": {
            "lattice": {
                "status": "regular_lattice",
                "spacing_px": 20.0,
                "rows": [30.0, 50.0, 70.0, 90.0],
            }
        },
    }


def test_profile_height_axis_labels_rulings_in_km():
    result = fit_from_profile(
        _structure(),
        {
            "profiles": {
                "narrow__sweep_10mhz": {
                    "sample_count": 25,
                    "height": {
                        "px_per_km": {"median": 0.1},
                        "top_offset_px": {"median": 2.0},
                        "ruling_spacing_px": {"median": 20.0},
                        "km_per_ruling": {"median": 200.0},
                    },
                }
            }
        },
        {"status": "usable", "profile": "narrow__sweep_10mhz"},
    )
    assert result["status"] == "usable"
    assert result["zero_row_px"] == 12.0
    assert np.allclose(
        [item["virtual_height_km"] for item in result["breakpoints"]],
        [180.0, 380.0, 580.0, 780.0],
    )


def test_cdf_height_axis_preserves_piecewise_monotonic_anchors():
    result = fit_from_landmark_reference(
        _structure(),
        {"status": "usable"},
        {
            "geometry": {
                "vertical_rows": [10.0, 110.0],
                "vertical_heights": [0.0, 2000.0],
            },
            "horizontal_matches": [
                {"status": "matched_csa_candidate", "match_score": 0.8}
            ],
        },
    )
    assert result["status"] == "usable"
    assert result["mapping"] == "affine_row_to_height"
    assert np.all(
        np.diff([item["virtual_height_km"] for item in result["breakpoints"]]) > 0
    )


def test_cdf_height_axis_uses_trusted_interior_match_as_local_anchor():
    result = fit_from_landmark_reference(
        _structure(),
        {"status": "usable"},
        {
            "geometry": {
                "vertical_rows": [10.0, 110.0],
                "vertical_heights": [0.0, 2000.0],
            },
            "horizontal_matches": [
                {
                    "status": "matched_csa_candidate",
                    "match_score": 0.8,
                    "csa_row": 58.0,
                    "virtual_height_km": 1000.0,
                }
            ],
        },
    )
    assert result["status"] == "usable"
    assert result["mapping"] == "piecewise_cdf_anchor"
    assert result["mapping_anchor_count"] == 3
    assert result["mapping_anchors"][1]["film_row"] == 58.0


def test_profile_height_axis_can_follow_observed_ruling_lattice():
    structure = _structure()
    structure["horizontal_rulings"]["lattice"]["rows"] = [30.0, 50.0, 70.0, 90.0, 105.0]
    result = fit_from_profile(
        structure,
        {
            "profiles": {
                "narrow__sweep_10mhz": {
                    "sample_count": 25,
                    "height": {
                        "px_per_km": {"median": 0.1},
                        "top_offset_px": {"median": 2.0},
                        "ruling_spacing_px": {"median": 20.0},
                        "km_per_ruling": {"median": 200.0},
                    },
                }
            }
        },
        {"status": "usable", "profile": "narrow__sweep_10mhz"},
    )
    assert result["mapping"] == "piecewise_ruling_lattice"
    assert result["mapping_anchor_count"] >= 3
