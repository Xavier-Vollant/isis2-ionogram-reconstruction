"""Tests for scan-structure extraction outputs."""

import numpy as np

from scripts.pipeline.extract_scan_structure import extract_structure


def test_extract_structure_keeps_landmarks_unlabeled():
    image = np.full((120, 200), 180.0)
    image[:, [30, 60, 90, 120, 150, 180]] = 20.0
    image[[20, 40, 60, 80, 100], :] = 30.0
    result = extract_structure(image, marker_sigma=1.5, ruling_sigma=1.5)
    assert result["schema"] == "isis.csa_scan_structure.v1"
    assert result["vertical_markers"]["count"] >= 4
    assert result["horizontal_rulings"]["lattice"]["count"] >= 3
    assert "frequency_mhz" not in result
    assert "virtual_height_km" not in result
