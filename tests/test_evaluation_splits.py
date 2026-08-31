"""Checks for the reel/station-disjoint evaluation harness.

The failure these guard against is silent: a leaked reel raises no error and
simply returns a better score, which is indistinguishable from a better model.
"""

import numpy as np
import pytest

from isis_research.evaluation import splits


def _records(reels=("a", "b", "c"), per_reel=4, station_of=None):
    station_of = station_of or (lambda reel: f"ST-{reel}")
    return [
        {"reel": reel, "station": station_of(reel), "pair_name": f"{reel}-{index}"}
        for reel in reels
        for index in range(per_reel)
    ]


# --- leakage ------------------------------------------------------------------


def test_check_disjoint_accepts_a_clean_split():
    records = _records()
    fit = [r for r in records if r["reel"] != "c"]
    test = [r for r in records if r["reel"] == "c"]
    assert splits.check_disjoint(fit, test) is True


def test_check_disjoint_names_the_leaked_reel():
    records = _records()
    fit = records
    test = [r for r in records if r["reel"] == "c"]
    with pytest.raises(ValueError, match="leakage"):
        splits.check_disjoint(fit, test)


def test_grouped_folds_never_leak():
    records = _records(reels=tuple("abcdefgh"), per_reel=5)
    seen_test = []
    for fit, test in splits.grouped_folds(records, folds=4, seed=1):
        assert set(splits.groups_of(fit)) & set(splits.groups_of(test)) == set()
        seen_test.extend(splits.groups_of(test))
    assert sorted(seen_test) == list("abcdefgh"), (
        "every reel must be tested exactly once"
    )


def test_grouped_folds_refuses_more_folds_than_reels():
    with pytest.raises(ValueError, match="at least 5"):
        list(splits.grouped_folds(_records(), folds=5))


def test_grouped_folds_balances_scan_counts():
    records = _records(reels=("big",), per_reel=40) + _records(
        reels=tuple("abcdef"), per_reel=2
    )
    sizes = [len(test) for _, test in splits.grouped_folds(records, folds=3, seed=0)]
    assert max(sizes) - min(sizes) <= 40, "the one huge reel must not be split"
    assert sum(sizes) == len(records)


# --- leave one out ------------------------------------------------------------


def test_leave_one_reel_out_covers_every_reel_once():
    records = _records(reels=tuple("abcde"))
    groups = [group for _, _, group in splits.leave_one_out(records, key="reel")]
    assert groups == list("abcde")


def test_leave_one_station_out_groups_reels_by_station():
    records = _records(
        reels=("a", "b", "c"), station_of=lambda reel: "OTT" if reel != "c" else "RES"
    )
    folds = list(splits.leave_one_out(records, key="station"))
    assert [group for _, _, group in folds] == ["OTT", "RES"]
    ott_fit, ott_test, _ = folds[0]
    assert {r["reel"] for r in ott_test} == {"a", "b"}
    assert {r["reel"] for r in ott_fit} == {"c"}


def test_leave_one_out_skips_groups_below_min_test():
    records = _records(reels=("a", "b"), per_reel=4) + [
        {"reel": "tiny", "station": "X", "pair_name": "tiny-0"}
    ]
    groups = [g for _, _, g in splits.leave_one_out(records, key="reel", min_test=2)]
    assert "tiny" not in groups


# --- uncertainty --------------------------------------------------------------


def test_bootstrap_ci_brackets_the_median():
    values = np.random.default_rng(0).normal(10.0, 1.0, 200)
    point, low, high = splits.bootstrap_ci(values, seed=0)
    assert low < point < high
    assert 9.5 < point < 10.5


def test_bootstrap_ci_is_reproducible_from_the_seed():
    values = np.random.default_rng(1).normal(0.0, 1.0, 50)
    assert splits.bootstrap_ci(values, seed=7) == splits.bootstrap_ci(values, seed=7)
    assert splits.bootstrap_ci(values, seed=7) != splits.bootstrap_ci(values, seed=8)


def test_bootstrap_ci_withholds_an_interval_it_cannot_support():
    point, low, high = splits.bootstrap_ci([1.0, 2.0])
    assert point == 1.5
    assert low is None and high is None


def test_bootstrap_ci_ignores_non_finite_values():
    point, _, _ = splits.bootstrap_ci([1.0, 2.0, 3.0, np.nan, None])
    assert point == 2.0


def test_a_wider_sample_gives_a_wider_interval():
    rng = np.random.default_rng(0)
    _, tight_low, tight_high = splits.bootstrap_ci(rng.normal(0, 0.1, 100), seed=0)
    _, wide_low, wide_high = splits.bootstrap_ci(rng.normal(0, 5.0, 100), seed=0)
    assert (wide_high - wide_low) > (tight_high - tight_low)


# --- the report table ---------------------------------------------------------


def test_group_summary_reports_every_group_with_counts():
    records = _records(reels=("a", "b"), per_reel=5)
    values = [1.0] * 5 + [3.0] * 5
    report = splits.group_summary(records, values, key="reel")
    assert report["groups"] == 2
    rows = {row["reel"]: row for row in report["by_reel"]}
    assert rows["a"]["n"] == 5 and rows["a"]["value"] == 1.0
    assert rows["b"]["value"] == 3.0
    assert report["overall"]["value"] == 2.0


def test_overall_interval_is_over_groups_not_scans():
    """One huge reel must not set the width of the archive-level interval."""
    records = _records(reels=("huge",), per_reel=200) + _records(
        reels=("s1", "s2"), per_reel=2
    )
    values = [0.5] * 200 + [0.9] * 2 + [0.8] * 2
    report = splits.group_summary(records, values, key="reel")
    assert report["groups"] == 3
    assert report["overall"]["over"] == "reels"
    # Median of the three per-reel values (0.5, 0.9, 0.8) is 0.8. Median of the
    # 204 scans is 0.5, because the huge reel supplies 98% of them. Asserting
    # both is what makes this test fail if the overall is taken over scans.
    assert report["overall"]["value"] == 0.8
    assert np.median(values) == 0.5
    assert {row["n"] for row in report["by_reel"]} == {200, 2}
