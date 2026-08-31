"""Tests for calibration-profile construction and reel-disjoint validation."""

from scripts.dataset.build_calibration_profile import (
    choose_group,
    display_path,
    split_reels,
)


def test_split_reels_never_leaks_a_reel():
    scans = [
        {
            "reel": "a",
            "width": 800,
            "width_class": "narrow",
            "format_class": "narrow",
            "sweep_class": "sweep_10mhz",
        },
        {
            "reel": "a",
            "width": 801,
            "width_class": "narrow",
            "format_class": "narrow",
            "sweep_class": "sweep_10mhz",
        },
        {
            "reel": "b",
            "width": 1200,
            "width_class": "wide",
            "format_class": "wide",
            "sweep_class": "sweep_20mhz",
        },
        {
            "reel": "c",
            "width": 1201,
            "width_class": "wide",
            "format_class": "wide",
            "sweep_class": "sweep_20mhz",
        },
    ]
    train, held_out, _ = split_reels(scans, fraction=0.25, seed=0)
    assert {scan["reel"] for scan in train}.isdisjoint(
        {scan["reel"] for scan in held_out}
    )


def test_choose_group_falls_back_when_exact_format_is_small():
    exact = {"sample_count": 1}
    fallback = {"sample_count": 40}
    profile = {
        "profiles": {"narrow__sweep_20mhz": exact},
        "format_fallbacks": {"narrow": fallback},
    }
    scan = {
        "width_class": "narrow",
        "format_class": "narrow",
        "sweep_class": "sweep_20mhz",
    }
    selected, name = choose_group(profile, scan)
    assert selected is fallback
    assert name == "narrow__fallback"


def test_display_path_supports_outputs_outside_repository(tmp_path):
    assert display_path(tmp_path / "profile.json") == str(tmp_path / "profile.json")
