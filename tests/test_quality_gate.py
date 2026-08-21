import numpy as np

from isis_research import ionogram
from scripts.dataset.quality_gate import evaluate_route, select_route


def _frequency(status="usable", coverage=1.0, rms=1.0, maximum=1.5, count=10):
    return {
        "status": status,
        "breakpoints": [
            {"film_column": 0, "frequency_mhz": 1},
            {"film_column": 9, "frequency_mhz": 2},
        ],
        "marker_coverage": coverage,
        "marker_rms_px": rms,
        "marker_max_error_px": maximum,
        "matched_marker_count": count,
        "warnings": [],
    }


def _height(status="usable", mapping="piecewise_cdf_anchor"):
    return {
        "status": status,
        "mapping": mapping,
        "mapping_anchors": [
            {"film_row": 0, "virtual_height_km": 0},
            {"film_row": 5, "virtual_height_km": 500},
        ],
        "height_min_km": 0,
        "height_max_km": 500,
        "warnings": [],
    }


def _warp(tmp_path, status="usable", coverage=1.0):
    graph = tmp_path / "warp.png"
    npz = tmp_path / "warp.npz"
    graph.write_bytes(b"graph")
    ionogram.write(
        npz,
        np.ones((6, 8), dtype=np.float32),
        np.ones((6, 8), dtype=bool),
        np.linspace(1, 2, 8),
        np.linspace(0, 500, 6),
        status=status,
        route="film_only",
        confidence=coverage,
        source={"test_warp": "warp"},
        provenance={"producer": "tests/test_quality_gate.py"},
    )
    return {
        "status": status,
        "valid_coverage": coverage,
        "frequency_min_mhz": 1,
        "frequency_max_mhz": 2,
        "height_min_km": 0,
        "height_max_km": 500,
        "graph": str(graph),
        "npz_sidecar": str(npz),
        "warnings": [],
    }


def _structure(status="structured"):
    return {
        "status": status,
        "image_shape": [10, 10],
        "film_region": {"top_row": 0, "bottom_row": 9},
        "vertical_markers": {
            "candidates": [{"x": 0, "strength_sigma": 4}, {"x": 9, "strength_sigma": 4}]
        },
        "horizontal_rulings": {
            "lattice": {"status": "regular_lattice", "rows": [1, 3, 5]}
        },
        "warnings": [],
    }


def test_quality_gate_reviews_large_frequency_residual(tmp_path):
    report = evaluate_route(
        "cdf_assisted",
        _structure(),
        _frequency(maximum=4.1),
        _height(),
        _warp(tmp_path),
    )
    assert report["status"] == "review"
    assert "frequency_max_residual_needs_review" in report["warnings"]


def test_quality_gate_rejects_low_warp_coverage(tmp_path):
    report = evaluate_route(
        "cdf_assisted",
        _structure(),
        _frequency(),
        _height(),
        _warp(tmp_path, coverage=0.5),
    )
    assert report["status"] == "not_usable"
    assert "warp_valid_coverage_below_80_percent" in report["errors"]


def test_quality_gate_rejects_warp_mask_that_disagrees_with_sidecar(tmp_path):
    warp = _warp(tmp_path)
    valid = np.ones((6, 8), dtype=bool)
    valid[0, 0] = False
    ionogram.write(
        warp["npz_sidecar"],
        np.ones((6, 8), dtype=np.float32),
        valid,
        np.linspace(1, 2, 8),
        np.linspace(0, 500, 6),
        status="usable",
        route="film_only",
        confidence=1.0,
        source={"test_warp": "warp"},
        provenance={"producer": "tests/test_quality_gate.py"},
    )
    report = evaluate_route(
        "film_only",
        _structure(),
        _frequency(),
        _height(mapping="piecewise_ruling_lattice"),
        warp,
    )
    assert report["status"] == "not_usable"
    assert "warp_coverage_does_not_match_valid_mask" in report["errors"]


def test_quality_gate_rejects_a_noncanonical_artifact(tmp_path):
    warp = _warp(tmp_path)
    np.savez_compressed(
        warp["npz_sidecar"],
        warped=np.ones((8, 6), dtype=np.float32),
        freq_axis=np.linspace(1, 2, 8),
        v_height=np.linspace(0, 500, 6),
    )
    report = evaluate_route(
        "film_only",
        _structure(),
        _frequency(),
        _height(mapping="piecewise_ruling_lattice"),
        warp,
    )
    assert report["status"] == "not_usable"
    assert "warp_npz_artifact_contract_invalid" in report["errors"]


def test_route_selection_prefers_usable_over_higher_scoring_review():
    reports = [
        {"route": "cdf_assisted", "status": "review", "quality_score": 99},
        {"route": "film_only", "status": "usable", "quality_score": 60},
    ]
    selected = select_route(reports)
    assert selected["status"] == "usable"
    assert selected["selected_route"] == "film_only"


def test_route_selection_uses_cdf_as_tiebreak():
    reports = [
        {"route": "cdf_assisted", "status": "usable", "quality_score": 80},
        {"route": "film_only", "status": "usable", "quality_score": 80},
    ]
    selected = select_route(reports)
    assert selected["selected_route"] == "cdf_assisted"
