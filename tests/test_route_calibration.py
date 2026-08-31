"""Tests for calibration route resolution and abstention."""

from pathlib import Path

import numpy as np

from scripts.pipeline.route_calibration import (
    cdf_validation_error,
    resolve_cdf,
    route_scan,
    select_film_profile,
)


def test_resolve_cdf_uses_metadata_nasa_id(tmp_path):
    cdf_dir = tmp_path / "cdf"
    cdf_dir.mkdir()
    expected = cdf_dir / "record.cdf"
    expected.write_bytes(b"cdf")
    path, source, warning = resolve_cdf(None, {"nasa_id": "record"}, cdf_dir)
    assert path == expected
    assert source == "metadata:nasa_id"
    assert warning is None


def test_film_profile_selection_can_abstain_when_candidates_are_tied():
    profile = {
        "source": {"min_profile_samples": 1, "min_fallback_samples": 1},
        "profiles": {
            "narrow__sweep_10mhz": {
                "sample_count": 10,
                "frequency": {
                    "frequencies_mhz": [1, 2, 3, 4],
                    "position_fraction": [0.2, 0.3, 0.4, 0.5],
                },
            },
            "narrow__sweep_20mhz": {
                "sample_count": 10,
                "frequency": {
                    "frequencies_mhz": [1, 2, 3, 4],
                    "position_fraction": [0.2, 0.3, 0.4, 0.5],
                },
            },
        },
        "format_fallbacks": {},
    }
    result = select_film_profile(
        [20, 30, 40, 50],
        {"width": 100, "format_class": "narrow", "sweep_class": "sweep_unknown"},
        profile,
        {},
    )
    assert result["status"] == "review"
    assert result["confidence"] == "medium"


def test_known_sweep_uses_the_exact_supported_profile():
    group = {
        "sample_count": 10,
        "frequency": {
            "frequencies_mhz": [1, 2, 3, 4],
            "position_fraction": [0.2, 0.3, 0.4, 0.5],
        },
    }
    profile = {
        "source": {"min_profile_samples": 1, "min_fallback_samples": 1},
        "profiles": {"narrow__sweep_10mhz": group},
        "format_fallbacks": {"narrow": group},
    }
    result = select_film_profile(
        [20, 30, 40, 50],
        {"width": 100, "format_class": "narrow", "sweep_class": "sweep_10mhz"},
        profile,
        {"sweep_class": "sweep_10mhz"},
    )
    assert result["status"] == "selected"
    assert result["selected"]["profile"] == "narrow__sweep_10mhz"


def test_cdf_validation_accepts_checked_in_sample():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/samples/i2_av_bur_1973077231124_v01.cdf"
    )
    assert cdf_validation_error(path) is None


def test_cdf_validation_rejects_marker_fill_values(monkeypatch, tmp_path):
    arrays = {
        "ampl": np.ones((4, 3)),
        "freq": np.array([1.0, 2.0, 3.0, 4.0]),
        "v_height": np.array([100.0, 200.0, 300.0]),
        "Epoch": np.array([1.0, 2.0, 3.0, 4.0]),
        "Time_mark": np.full(4, -1e31),
        "freq_mark": np.full(4, -1e31),
        "swept_start": np.array([0.0]),
    }

    class FakeCDF:
        def __init__(self, path):
            pass

        def varget(self, name):
            return arrays[name]

    monkeypatch.setattr("scripts.pipeline.route_calibration.cdflib.CDF", FakeCDF)
    assert "fewer than four" in cdf_validation_error(tmp_path / "fake.cdf")


def test_route_scan_falls_back_when_referenced_cdf_is_malformed(tmp_path):
    malformed = tmp_path / "broken.cdf"
    malformed.write_bytes(b"not a cdf")
    film = (
        Path(__file__).resolve().parents[1]
        / "data/raw/csa_verified_bur_1973077231124.png"
    )
    profile = (
        Path(__file__).resolve().parents[1] / "configs/film_calibration_profile.json"
    )

    result = route_scan(film, profile, cdf=malformed)

    assert result["route"] == "film_only"
    assert result["warnings"][0].startswith("referenced CDF is invalid:")
