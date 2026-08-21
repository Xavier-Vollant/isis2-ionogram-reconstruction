import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dataset"))

from build_candidate_matches import (  # noqa: E402
    SELECTORS,
    quota_caps,
    select_by_reel,
    select_by_station,
)


def candidate(dt, station, reel, year="1972"):
    return (
        dt,
        10.0,
        99.0,
        {
            "station_code_normalized": station,
            "film_subdir": reel,
            "year": year,
        },
        {},
    )


def archive(stations, years=None):
    return {
        "station_code_normalized": Counter(stations),
        "year": Counter(years or {"1972": 1}),
    }


def test_selection_spreads_over_reels_not_just_stations():
    # One station, ten reels: reel round-robin must touch every reel before
    # taking a second scan from any of them.
    eligible = [
        candidate(dt, "RES", f"reel{reel}")
        for reel in range(10)
        for dt in (0.1, 0.2, 0.3)
    ]
    chosen, _, _ = select_by_reel(eligible, 10, archive({"RES": 100}))
    assert len({item[3]["film_subdir"] for item in chosen}) == 10


def test_station_cap_stops_one_station_taking_the_batch():
    # RES holds far more reels, but the archive says OTT is the larger station.
    eligible = [candidate(0.1, "RES", f"res{i}") for i in range(50)]
    eligible += [candidate(0.1, "OTT", f"ott{i}") for i in range(50)]
    chosen, _, _ = select_by_reel(eligible, 20, archive({"RES": 20, "OTT": 80}))
    counts = Counter(item[3]["station_code_normalized"] for item in chosen)
    assert counts["OTT"] > counts["RES"]


def test_dense_reels_are_not_starved_by_thin_ones():
    # The regression this quota exists for: 1972 sits in two dense reels and
    # 1973 in twenty thin ones. Reel round-robin alone equalises reels, not
    # scans, and inverted a 61% majority into a 16% minority.
    eligible = [
        candidate(0.1, "RES", f"dense{reel}", "1972")
        for reel in range(2)
        for _ in range(500)
    ]
    eligible += [candidate(0.1, "RES", f"thin{reel}", "1973") for reel in range(20)]
    chosen, _, _ = select_by_reel(
        eligible, 100, archive({"RES": 1}, {"1972": 60, "1973": 20})
    )
    years = Counter(item[3]["year"] for item in chosen)
    assert years["1972"] > years["1973"]


def test_caps_renormalise_over_values_that_have_candidates():
    # ORR is a fifth of the archive and reachable nowhere: its quota must be
    # redistributed, not held open against scans that cannot exist.
    caps = quota_caps(
        [candidate(0.1, "RES", "reel")],
        Counter({"RES": 40, "ORR": 10}),
        10,
        "station_code_normalized",
    )
    assert "ORR" not in caps
    assert caps["RES"] >= 10


def test_batch_is_filled_even_when_caps_cannot_be_met():
    # Only RES has candidates, so a strict cap would under-deliver. The second
    # pass must fill the batch rather than silently return a short one.
    eligible = [candidate(0.1, "RES", f"reel{i}") for i in range(30)]
    chosen, _, _ = select_by_reel(eligible, 25, archive({"RES": 10, "OTT": 90}))
    assert len(chosen) == 25
    assert len({id(item) for item in chosen}) == 25


def test_quality_order_is_preserved_within_a_reel():
    eligible = [candidate(dt, "RES", "reel") for dt in (5.0, 0.5, 2.0)]
    chosen, _, _ = select_by_reel(eligible, 3, archive({"RES": 1}))
    assert [item[0] for item in chosen] == [0.5, 2.0, 5.0]


def test_station_quota_survives_an_unmeetable_year_quota():
    # Year and station cannot both be satisfied here: RES has only 1971 scans
    # and OTT only 1973, while the archive wants mostly 1972. Year must yield
    # first - relaxing both at once handed the batch to whoever had the most
    # candidates, taking RES from 23% to 64%.
    eligible = [candidate(0.1, "RES", f"res{i}", "1971") for i in range(400)]
    eligible += [candidate(0.1, "OTT", f"ott{i}", "1973") for i in range(400)]
    chosen, _, _ = select_by_reel(
        eligible,
        100,
        archive({"RES": 20, "OTT": 80}, {"1972": 90, "1971": 5, "1973": 5}),
    )
    counts = Counter(item[3]["station_code_normalized"] for item in chosen)
    assert len(chosen) == 100
    assert counts["OTT"] > 2 * counts["RES"]


def test_yield_compensation_favours_the_station_that_survives_less():
    # OTT survives alignment at half ULA's rate, so an equal archive share has
    # to become an unequal quota or the labelled batch under-represents it.
    eligible = [candidate(0.1, "OTT", f"ott{i}") for i in range(200)]
    eligible += [candidate(0.1, "ULA", f"ula{i}") for i in range(200)]
    counts = Counter({"OTT": 50, "ULA": 50})
    plain = quota_caps(eligible, counts, 100, "station_code_normalized")
    compensated = quota_caps(
        eligible, counts, 100, "station_code_normalized", {"OTT": 0.5, "ULA": 1.0}
    )
    assert plain["OTT"] == plain["ULA"]
    assert compensated["OTT"] > compensated["ULA"]


def test_a_station_that_never_survives_gets_no_quota():
    # Adelie failed all 40 attempts. Dividing by ~0 would hand it a vast quota
    # and spend the batch on scans that cannot pass.
    eligible = [candidate(0.1, "RES", "res")]
    caps = quota_caps(
        eligible,
        Counter({"ADL": 50, "RES": 50}),
        100,
        "station_code_normalized",
        {"ADL": 0.0, "RES": 0.7},
    )
    assert "ADL" not in caps
    assert caps["RES"] > 0


def test_a_dead_station_is_excluded_even_when_the_batch_falls_short():
    # The strong form: RES cannot fill the batch alone, so a merely-capped ADL
    # would be topped up by the uncapped pass. It must stay out entirely -
    # picks spent on a station that fails every attempt buy nothing.
    eligible = [candidate(0.1, "ADL", f"adl{i}") for i in range(50)]
    eligible += [candidate(0.1, "RES", f"res{i}") for i in range(10)]
    chosen, _, dropped = select_by_reel(
        eligible,
        30,
        archive({"ADL": 50, "RES": 50}),
        {"ADL": 0.0, "RES": 0.7},
    )
    stations = Counter(item[3]["station_code_normalized"] for item in chosen)
    assert stations["ADL"] == 0
    assert len(chosen) == 10
    assert dropped == ["ADL"]


def test_both_strategies_are_available_and_differ():
    # The station strategy gives every station the same count; the reel one
    # weights by archive share. Keeping both means batch1000 stays comparable.
    eligible = [candidate(0.1, "RES", f"res{i}") for i in range(100)]
    eligible += [candidate(0.1, "BUR", f"bur{i}") for i in range(100)]
    counts = archive({"RES": 90, "BUR": 10})

    by_station, _, _ = SELECTORS["station"](eligible, 40, counts)
    by_reel, _, _ = SELECTORS["reel"](eligible, 40, counts)

    station_split = Counter(i[3]["station_code_normalized"] for i in by_station)
    reel_split = Counter(i[3]["station_code_normalized"] for i in by_reel)
    assert station_split["RES"] == station_split["BUR"]
    assert reel_split["RES"] > reel_split["BUR"]


def test_station_strategy_ignores_quotas_and_yields():
    # It takes the same optional arguments so the two are interchangeable, but
    # must not act on them - that is the whole point of keeping it separate.
    eligible = [candidate(0.1, "RES", f"res{i}") for i in range(50)]
    eligible += [candidate(0.1, "ADL", f"adl{i}") for i in range(50)]
    plain, _, _ = select_by_station(eligible, 20, archive({"RES": 99, "ADL": 1}))
    with_yields, _, _ = select_by_station(
        eligible, 20, archive({"RES": 99, "ADL": 1}), {"ADL": 0.0, "RES": 0.7}
    )
    assert [id(i) for i in plain] == [id(i) for i in with_yields]
