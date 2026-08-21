import json

from scripts.pipeline.route_calibration import resolve_cdf, select_film_profile


def test_resolve_cdf_uses_metadata_nasa_id(tmp_path):
    cdf_dir = tmp_path / "cdf"
    cdf_dir.mkdir()
    expected = cdf_dir / "record.cdf"
    expected.write_bytes(b"cdf")
    path, source, warning = resolve_cdf(
        None, {"nasa_id": "record"}, cdf_dir
    )
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
