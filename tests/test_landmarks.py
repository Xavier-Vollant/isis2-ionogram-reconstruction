"""Tests for landmark matching and label classes."""

import numpy as np

from isis_research.registration import film
from isis_research.registration.landmarks import (
    build_ml_labels,
    label_consensus_rulings,
    label_csa_rulings,
    match_nasa_rows_to_csa,
    resolve_sweep_start,
    select_ruling_review_targets,
)


def test_fit_marker_axis_prefers_the_long_known_run():
    reference = np.array([10, 20, 31, 45, 62, 80, 101, 125], dtype=float)
    observed = np.array([2, 90, 100, 111, 125, 142, 160, 181, 205, 240], dtype=float)
    result = film.fit_marker_axis(observed, reference)
    assert result["count"] == 8
    assert result["rms_px"] < 1e-9


def test_horizontal_match_rejects_a_weak_or_distant_candidate():
    nasa = [
        {
            "cdf_row": 10,
            "virtual_height_km": 1000.0,
            "strength": 20.0,
            "strength_sigma": 5.0,
        }
    ]
    csa = [
        {"csa_row": 100.0, "strength_sigma": 2.1},
        {"csa_row": 120.0, "strength_sigma": 5.0},
    ]
    result = match_nasa_rows_to_csa(nasa, csa, np.array([0.02, 100.0]))
    assert result[0]["status"] == "matched_csa_candidate"
    assert result[0]["csa_row"] == 120.0

    rejected = match_nasa_rows_to_csa(
        nasa, [{"csa_row": 180.0, "strength_sigma": 5.0}], np.array([0.02, 100.0])
    )
    assert rejected[0]["status"] == "no_confident_csa_match"


def test_ruling_labels_separate_verified_relative_and_review_rows():
    candidates = [
        {"csa_row": 32.0, "support_fraction": 1.0, "segment_strength_sigma": 3.0},
        {"csa_row": 54.0, "support_fraction": 1.0, "segment_strength_sigma": 3.0},
        {"csa_row": 76.0, "support_fraction": 1.0, "segment_strength_sigma": 3.0},
    ]
    labels = label_csa_rulings(
        candidates,
        {"rows": [32.0, 54.0, 76.0]},
        [{"virtual_height_km": 440.0}],
        10.0,
        0.1,
        15.0,
    )
    assert labels[0]["status"] == "relative_ruling_candidate"
    assert labels[1]["status"] == "verified_height_ruling"
    assert len(select_ruling_review_targets(labels)) == 2


def test_consensus_rulings_become_explicit_ml_row_labels():
    labels = label_consensus_rulings(
        {"rows": [32.0, 54.0, 76.0]},
        54.0,
        [{"csa_row": 54.0, "status": "verified_height_ruling"}],
        {
            "rulings": [
                {
                    "relative_index": -1,
                    "scan_count": 5,
                    "mad_offset_rulings": 0.05,
                    "status": "consensus_relative_ruling",
                },
                {
                    "relative_index": 0,
                    "scan_count": 5,
                    "mad_offset_rulings": 0.0,
                    "status": "consensus_relative_ruling",
                },
            ]
        },
        10.0,
        0.1,
        100,
    )
    assert [item["class_name"] for item in labels] == [
        "csa_horizontal_ruling",
        "csa_horizontal_ruling",
    ]
    assert labels[0]["weak_label"] is True
    assert labels[1]["label_quality"] == "verified_height"
    assert labels[1]["row_start"] == 53
    assert labels[1]["row_end"] == 55


def test_ml_labels_carry_both_axes_and_ignore_the_unidentified():
    result = {
        "csa_consensus_ruling_labels": [
            {"class_id": 1, "class_name": "csa_horizontal_ruling", "csa_row": 54.0}
        ],
        "features": [
            {
                "name": "frequency_marker",
                "frequency_mhz": 0.5,
                "cdf_column": 393.0,
                "csa_x": 261.0,
                "residual_px": 0.4,
            },
            {
                "name": "frequency_marker_unmatched",
                "frequency_mhz": 10.0,
                "predicted_csa_x": 190.0,
                "status": "outside_detected_marker_run",
            },
            {
                "name": "start_of_frequency_sweep",
                "cdf_column": 291.0,
                "csa_x": 40.0,
                "residual_px": 1.2,
                "status": "matched_nearby_marker",
            },
            {"name": "top_of_ionogram", "cdf_row": 0.0, "csa_y": 9.0},
            {"name": "bottom_of_ionogram", "cdf_row": 222.0, "csa_y": 90.0},
        ],
        "csa_ruling_lattice": {"rows": [54.0]},
        "csa_horizontal_candidates": [
            {"csa_row": 54.4, "strength_sigma": 4.0},
            {"csa_row": 70.0, "strength_sigma": 2.6},
        ],
        "film": {"top_row": 9.0, "bottom_row": 90.0},
    }
    payload = build_ml_labels(result, (100, 300))

    # The frequency axis is what the rulings alone cannot supply.
    by_class = {item["class_name"] for item in payload["labels"]}
    assert by_class == {
        "csa_horizontal_ruling",
        "frequency_marker",
        "start_of_frequency_sweep",
        "top_of_ionogram",
        "bottom_of_ionogram",
    }
    marker = next(i for i in payload["labels"] if i["class_name"] == "frequency_marker")
    assert (marker["col_start"], marker["col_end"]) == (260, 262)
    assert marker["frequency_mhz"] == 0.5

    reasons = {item["reason"] for item in payload["ignore"]}
    assert reasons == {
        "marker_predicted_but_not_observed",
        "horizontal_line_outside_lattice",
        "outside_film_data_bounds",
    }
    # The line 0.4px off a lattice row is that row, not a second unknown line.
    off_lattice = [
        item
        for item in payload["ignore"]
        if item["reason"] == "horizontal_line_outside_lattice"
    ]
    assert [(item["row_start"], item["row_end"]) for item in off_lattice] == [(69, 71)]
    margins = sorted(
        (item["row_start"], item["row_end"])
        for item in payload["ignore"]
        if item["reason"] == "outside_film_data_bounds"
    )
    assert margins == [(0, 8), (91, 99)]


def test_ml_labels_drop_landmarks_that_fall_off_the_scan():
    result = {
        "features": [
            {"name": "frequency_marker", "csa_x": 400.0, "frequency_mhz": 9.0},
            {"name": "frequency_marker", "csa_x": None, "frequency_mhz": 8.0},
            {"name": "top_of_ionogram", "csa_y": -3.0},
        ],
        "film": {"top_row": 0.0, "bottom_row": 99.0},
    }
    payload = build_ml_labels(result, (100, 300))
    assert payload["labels"] == []
    assert payload["ignore"] == []


def test_sweep_start_uses_first_low_frequency_sample():
    frequency = np.r_[np.full(7, 4.0), np.linspace(0.1, 10.0, 40)]
    result = resolve_sweep_start(frequency, stored_start=23)
    assert result["inferred"] == 7
    assert result["resolved"] == 7
    assert result["delta_columns"] == 16
    assert result["frequency_mhz"] == 0.1


def test_sweep_start_does_not_choose_later_near_zero_sample():
    frequency = np.r_[np.full(7, 4.0), 0.1, np.linspace(0.09, -0.02, 20)]
    result = resolve_sweep_start(frequency, stored_start=23)
    assert result["resolved"] == 7
    assert result["frequency_mhz"] == 0.1
