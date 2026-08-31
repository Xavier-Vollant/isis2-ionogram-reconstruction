#!/usr/bin/env python
"""Rank possible CSA/NASA ionogram pairs from archive metadata.

The scores use time, position, and the gap to the next possible NASA record.
The results are candidates only; this command does not compare image pixels or
confirm a scientific match.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from isis_research.nasa.stations import separation_km

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# NASA passes run ~15 minutes with an ionogram every ~20-30 s, so a genuine
# pairing should land within seconds, not minutes.
WINDOW_SECONDS = 120.0

# 97% of candidates agree on position within 100 km, so requiring it discards
# almost nothing while removing pairs nothing corroborates.
MAX_POSITION_KM = 100.0

# The runner-up ionogram must be at least this many times further away in time,
# so a pair that fits its neighbour equally well never counts as "best".
AMBIGUITY_FACTOR = 2.0

COLUMNS = [
    "rank",
    "dt_seconds",
    "position_km",
    "margin_seconds",
    "ambiguous",
    "csa_id",
    "csa_timestamp_utc",
    "csa_station",
    "csa_station_code_raw",
    "csa_station_number",
    "csa_film_subdir",
    "csa_image_filename",
    "csa_image_url",
    "csa_max_depth",
    "csa_fmin",
    "csa_sat_lat",
    "csa_sat_lon",
    "csa_sat_height_km",
    "csa_station_elevation_deg",
    "csa_timestamp_trust",
    "nasa_id",
    "nasa_frame_sync_utc",
    "nasa_station",
    "nasa_pass_number",
    "nasa_header_file",
    "nasa_avg_bin_filename",
    "nasa_cdf_name_predicted",
    "nasa_gglat",
    "nasa_gglon",
    "nasa_hgt_km",
    "nasa_fixed_freq_mhz",
    "nasa_swept_freq_range",
    "nasa_sounding_status_determined",
    "nasa_freq_interp_available",
    "nasa_fixed_freq_suspect",
    "nasa_worldmap_zero",
    "nasa_parse_status",
]


def as_float(value):
    """Convert a CSV value to float, or return ``None`` when it is missing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_eligible(dt, position_km, margin):
    """Does this pair qualify as one of the best candidates?

    Requires an independent position check that agrees, and a runner-up
    ionogram clearly further away in time. A pair whose world-map fields were
    never computed has nothing corroborating it, however well the clocks agree.
    """
    if position_km is None or position_km > MAX_POSITION_KM:
        return False
    return margin >= AMBIGUITY_FACTOR * max(dt, 1.0)


def load_nasa():
    """-> {(station, year, doy): sorted [(sec_of_day, row)]}."""
    buckets = defaultdict(list)
    kept = skipped = 0
    with open(PROCESSED / "nasa_master.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sec = as_float(row["sec_of_day"])
            iso = row["frame_sync_utc"]
            if row["parse_status"] != "ok" or sec is None or not iso:
                skipped += 1
                continue
            year, doy = iso[:4], iso[5:8]
            buckets[(row["station_dir_code"], year, doy.lstrip("0") or "0")].append(
                (sec, row)
            )
            kept += 1
    for key in buckets:
        buckets[key].sort(key=lambda pair: pair[0])
    print(f"NASA ionograms indexed: {kept} (skipped {skipped})")
    return buckets


def nearest_two(entries, target):
    """-> (best, second) as (dt, row), searching a time-sorted pass bucket."""
    times = [sec for sec, _ in entries]
    position = bisect_left(times, target)
    neighbours = []
    for index in (position - 1, position, position + 1):
        if 0 <= index < len(entries):
            sec, row = entries[index]
            neighbours.append((abs(sec - target), row))
    neighbours.sort(key=lambda pair: pair[0])
    best = neighbours[0] if neighbours else None
    second = neighbours[1] if len(neighbours) > 1 else None
    return best, second


def build(limit, yields=None, strategy="reel"):
    """Build and rank CSA/NASA candidates under the requested quota strategy."""
    buckets = load_nasa()
    stats = Counter()
    candidates = []
    # Counted over every row, not the eligible ones, because the quota targets
    # the archive the model must finally cope with rather than the part of it
    # this crosswalk happens to reach.
    archive_counts = {field: Counter() for field in QUOTA_FIELDS}

    with open(PROCESSED / "csa_master.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stats["csa_rows"] += 1
            for field in QUOTA_FIELDS:
                if row[field]:
                    archive_counts[field][row[field]] += 1
            if row["timestamp_trust"] != "ok":
                stats["skip_untrusted_timestamp"] += 1
                continue
            station = row["station_code_normalized"]
            if not station:
                stats["skip_unmatched_station"] += 1
                continue
            sec = as_float(row["sec_of_day"])
            if sec is None or not row["year"] or not row["day_of_year"]:
                stats["skip_no_time"] += 1
                continue

            entries = buckets.get((station, row["year"], row["day_of_year"]))
            if not entries:
                stats["skip_no_nasa_pass_that_day"] += 1
                continue

            best, second = nearest_two(entries, sec)
            if best is None or best[0] > WINDOW_SECONDS:
                stats["skip_outside_window"] += 1
                continue

            dt, nasa_row = best
            margin = second[0] if second else float("inf")

            csa_lat, csa_lon = as_float(row["sat_lat"]), as_float(row["sat_lon"])
            nasa_lat = as_float(nasa_row["gglat"])
            nasa_lon = as_float(nasa_row["gglon"])
            if None in (csa_lat, csa_lon, nasa_lat, nasa_lon):
                position_km = None
                stats["no_position_check"] += 1
            elif nasa_lat == 0.0 and nasa_lon == 0.0:
                # NASA documents early records whose world-map fields were never
                # computed and left at zero. Absent, not a disagreement.
                position_km = None
                stats["nasa_worldmap_zero"] += 1
            else:
                position_km = separation_km(csa_lat, csa_lon, nasa_lat, nasa_lon)

            stats["candidates"] += 1
            candidates.append((dt, position_km, margin, row, nasa_row))

    # "Best" demands corroboration, not just a close clock. A pair qualifies
    # only if an independent position check agrees and no neighbouring ionogram
    # is a comparably good fit; otherwise a sub-second timestamp coincidence
    # with uncomputed world-map fields would outrank a genuine pairing.
    eligible = [c for c in candidates if is_eligible(c[0], c[1], c[2])]
    stats["eligible_for_best"] = len(eligible)
    chosen, relaxed, dropped = SELECTORS[strategy](
        eligible, limit, archive_counts, yields
    )
    if relaxed:
        stats["quota_surrendered"] = ",".join(relaxed)
    if dropped:
        stats["stations_dropped_zero_yield"] = ",".join(dropped)
    write(chosen, strategy)
    return stats, candidates


MIN_YIELD_SAMPLE = 20
DEAD_YIELD = 0.05


def measure_yield(manifest_path, minimum=MIN_YIELD_SAMPLE):
    """-> {station: share of attempts that reach a usable marker fit}.

    Selecting to archive shares balances what goes *in*; the alignment gate
    then re-weights what comes out, because it does not fail evenly. Ottawa
    survives at about half, Ulan at four fifths, Adelie at none. Read off a
    previous batch's manifest so the compensation is measured rather than
    assumed, and so it improves as batches accumulate.
    """
    records = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    attempted, passed = Counter(), Counter()
    for record in records:
        station = record["name"].split("_")[2].upper()
        attempted[station] += 1
        metrics = record.get("metrics") or {}
        count = metrics.get("marker_count")
        rms = metrics.get("marker_rms_px")
        if count is not None and rms is not None and count >= 8 and float(rms) <= 3.0:
            passed[station] += 1
    overall = sum(passed.values()) / max(sum(attempted.values()), 1)
    return {
        station: (passed[station] / total if total >= minimum else overall)
        for station, total in attempted.items()
    }


def quota_caps(eligible, archive_counts, limit, field, yields=None):
    """-> {value: maximum picks} in proportion to that value's share of the archive.

    Renormalised over the values that actually have eligible candidates.
    Orroral is 5.4% of the archive and shares no observing day with NASA at
    all, so holding a quota open for it would only shrink every other station's
    share in exchange for scans that cannot exist.

    With measured yields the quota is divided by the share that survives
    alignment, so it is the *labelled* batch that matches the archive rather
    than the candidate list feeding it. A value whose yield is effectively zero
    is dropped rather than given an enormous quota: Adelie failed all 40 of its
    attempts, so picks spent there buy nothing.
    """
    available = {candidate[3][field] for candidate in eligible}
    weights = {}
    for value, count in archive_counts.items():
        if value not in available or not count:
            continue
        weights[value] = count / (yields or {}).get(value, 1.0)
    total = sum(weights.values())
    if not total:
        return {}
    return {
        value: max(1, int(-(-limit * weight // total)))
        for value, weight in weights.items()
    }


def drop_dead_stations(eligible, yields):
    """-> candidates minus the stations that never survive alignment.

    Excluded outright rather than capped, because a cap is only honoured while
    quotas hold: the pass that fills the batch ignores them, and a station that
    fails every attempt would be topped up there precisely when it is least
    worth spending picks on. Adelie failed all 40 of its attempts.
    """
    if not yields:
        return eligible, []
    dead = {station for station, survival in yields.items() if survival <= DEAD_YIELD}
    if not dead:
        return eligible, []
    kept = [
        candidate
        for candidate in eligible
        if candidate[3]["station_code_normalized"] not in dead
    ]
    return kept, sorted(dead)


QUOTA_FIELDS = ("station_code_normalized", "year")


def select_by_station(eligible, limit, archive_counts=None, yields=None):
    """Pick `limit` candidates by round-robining across stations equally.

    Every station gets the same number of picks regardless of how large it is
    in the archive, which is the right shape when the question is "does this
    work everywhere" rather than "what will the model meet". Kept alongside the
    reel strategy because it is the selection batch1000 was built with, so
    comparisons against that batch stay like-for-like.

    Each station's own candidates stay ranked by quality - only which station
    gets the next pick is round-robined - so this never trades a better match
    for a worse one within a station.
    """
    by_station = defaultdict(list)
    for candidate in eligible:
        by_station[candidate[3]["station_code_normalized"]].append(candidate)
    for group in by_station.values():
        group.sort(key=lambda c: (round(c[0], 3), c[1]))
    stations = sorted(by_station)
    cursors = {station: 0 for station in stations}
    chosen = []
    while len(chosen) < limit:
        picked_this_round = False
        for station in stations:
            if cursors[station] < len(by_station[station]):
                chosen.append(by_station[station][cursors[station]])
                cursors[station] += 1
                picked_this_round = True
                if len(chosen) == limit:
                    break
        if not picked_this_round:
            break
    return chosen, [], []


def select_by_reel(eligible, limit, archive_counts=None, yields=None):
    """Pick `limit` candidates spread over film reels, quota'd to the archive.

    The reel is the primary axis because it is the one that governs how a scan
    *looks*: a reel is a physical strip with its own exposure, development,
    storage and scanner session, and that is the variation a pixel-level
    detector is most sensitive to.  Round-robining over stations instead left
    36 of 1021 reels represented.

    Round-robin alone equalises *reels*, not scans, so any property that varies
    with how many candidates a reel holds is silently inverted by it: a pool
    that was 61% 1972 came out 16% 1972, because the 1972 scans sit in a few
    dense reels while the 1973 ones are spread thin.  Station and year
    therefore carry their own caps, in proportion to the archive.

    Each reel's own candidates stay ranked by match quality - only the order
    reels are drawn from is round-robined - so this never trades a better match
    for a worse one within a reel.

    Caps bind only while they can still be met. Once every uncapped reel is
    exhausted the remainder is filled without them, because delivering a
    smaller batch than asked for silently is worse than delivering a slightly
    less even one visibly.
    """
    archive_counts = archive_counts or {}
    eligible, dropped = drop_dead_stations(eligible, yields)
    by_reel = defaultdict(list)
    for candidate in eligible:
        by_reel[candidate[3]["film_subdir"]].append(candidate)
    for group in by_reel.values():
        group.sort(key=lambda c: (round(c[0], 3), c[1]))

    caps = {
        field: quota_caps(
            eligible,
            archive_counts.get(field, {}),
            limit,
            field,
            # Only station has a measured survival rate; year rides along with
            # whichever stations the gate happens to keep.
            yields if field == "station_code_normalized" else None,
        )
        for field in QUOTA_FIELDS
    }
    reels = sorted(by_reel)
    cursors = {reel: 0 for reel in reels}
    taken = {field: Counter() for field in QUOTA_FIELDS}
    chosen = []

    def blocked(candidate, enforced):
        return any(
            taken[field][candidate[3][field]]
            >= caps[field].get(candidate[3][field], limit)
            for field in enforced
        )

    # Quotas are relaxed one at a time, least important first, rather than all
    # at once. Dropping both together hands the remainder of the batch to
    # whichever station simply has the most candidates: enforcing station and
    # year jointly stalls early, and the free pass that followed took Resolute
    # from 23% to 64%. Year yields first because station is the axis the model
    # is finally judged on.
    #
    # On the present crosswalk year yields immediately and entirely, because
    # the two archives' shared observing history confounds it with station:
    # the 1972 overlap is almost entirely a Resolute campaign, while Ottawa,
    # Ulan, Tromso, Quito and Adelie are each ~99% 1973. Matching both is not
    # possible, so the caller is told which quota was surrendered instead of
    # being handed a batch that quietly missed one.
    relaxed = []
    for enforced in (QUOTA_FIELDS, ("station_code_normalized",), ()):
        if chosen and len(enforced) < len(QUOTA_FIELDS):
            relaxed = [f for f in QUOTA_FIELDS if f not in enforced]
        while len(chosen) < limit:
            picked_this_round = False
            for reel in reels:
                if len(chosen) == limit:
                    break
                group = by_reel[reel]
                # Skip past candidates this reel can no longer place rather
                # than stalling the whole reel on its next one: a reel spans
                # years, so one exhausted year quota must not retire it.
                while cursors[reel] < len(group):
                    candidate = group[cursors[reel]]
                    if blocked(candidate, enforced):
                        cursors[reel] += 1
                        continue
                    cursors[reel] += 1
                    for field in QUOTA_FIELDS:
                        taken[field][candidate[3][field]] += 1
                    chosen.append(candidate)
                    picked_this_round = True
                    break
            if not picked_this_round:
                break
        if len(chosen) >= limit:
            break
        # The capped pass consumed its cursors; reopen them for the free pass.
        cursors = {reel: 0 for reel in reels}
        seen = {id(item) for item in chosen}
        for reel in reels:
            by_reel[reel] = [item for item in by_reel[reel] if id(item) not in seen]
    return chosen, relaxed, dropped


SELECTORS = {"station": select_by_station, "reel": select_by_reel}


def write(chosen, strategy):
    """Write selected candidate rows to a strategy-specific CSV file."""
    # The strategy is in the name so two selections of the same size do not
    # overwrite one another - the point of keeping both is comparing them.
    path = PROCESSED / f"candidate_matches_top{len(chosen)}_{strategy}.csv"
    with open(path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for rank, (dt, position_km, margin, csa, nasa) in enumerate(chosen, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "dt_seconds": f"{dt:.3f}",
                    "position_km": "" if position_km is None else f"{position_km:.1f}",
                    "margin_seconds": "" if margin == float("inf") else f"{margin:.3f}",
                    "ambiguous": str(margin < 2 * max(dt, 1.0)),
                    "csa_id": csa["csa_id"],
                    "csa_timestamp_utc": csa["timestamp_utc"],
                    "csa_station": csa["station_code_normalized"],
                    "csa_station_code_raw": csa["station_code_raw"],
                    # The film carries this number in its burned-in time code.
                    "csa_station_number": csa["station_number_raw"],
                    "csa_film_subdir": csa["film_subdir"],
                    "csa_image_filename": csa["image_filename"],
                    "csa_image_url": csa["image_url"],
                    "csa_max_depth": csa["max_depth"],
                    "csa_fmin": csa["fmin"],
                    "csa_sat_lat": csa["sat_lat"],
                    "csa_sat_lon": csa["sat_lon"],
                    "csa_sat_height_km": csa["sat_height_km"],
                    "csa_station_elevation_deg": csa["station_elevation_deg"],
                    "csa_timestamp_trust": csa["timestamp_trust"],
                    "nasa_id": nasa["nasa_id"],
                    "nasa_frame_sync_utc": nasa["frame_sync_utc"],
                    "nasa_station": nasa["station_dir_code"],
                    "nasa_pass_number": nasa["pass_number"],
                    "nasa_header_file": nasa["header_file"],
                    "nasa_avg_bin_filename": nasa["avg_bin_filename"],
                    "nasa_cdf_name_predicted": nasa["cdf_name_predicted"],
                    "nasa_gglat": nasa["gglat"],
                    "nasa_gglon": nasa["gglon"],
                    "nasa_hgt_km": nasa["hgt_km"],
                    "nasa_fixed_freq_mhz": nasa["fixed_freq_mhz"],
                    "nasa_swept_freq_range": nasa["swept_freq_range"],
                    "nasa_sounding_status_determined": nasa[
                        "sounding_status_determined"
                    ],
                    "nasa_freq_interp_available": nasa["freq_interp_available"],
                    "nasa_fixed_freq_suspect": nasa["fixed_freq_suspect"],
                    "nasa_worldmap_zero": nasa["worldmap_zero"],
                    "nasa_parse_status": nasa["parse_status"],
                }
            )
    print(f"\nwrote {len(chosen)} rows to {path.relative_to(ROOT)}")


def main():
    """Parse CLI options and write the selected CSA/NASA candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--strategy",
        choices=sorted(SELECTORS),
        default="reel",
        help="reel: spread over film reels with archive-proportional station "
        "quotas. station: equal picks per station, the batch1000 selection.",
    )
    parser.add_argument(
        "--yield-from",
        type=Path,
        default=None,
        help="quality_manifest.json of a previous batch; station quotas are "
        "divided by the share that survived alignment there, so the labelled "
        "batch matches the archive rather than the candidate list",
    )
    args = parser.parse_args()

    yields = measure_yield(args.yield_from) if args.yield_from else None
    stats, candidates = build(args.limit, yields, args.strategy)

    print("\n--- candidate generation ---")
    for name, value in sorted(stats.items()):
        print(f"{name:32} {value}")

    if not candidates:
        return
    print("\n--- distribution of all candidates ---")
    for threshold in (1, 5, 15, 30, 60, 120):
        count = sum(1 for c in candidates if c[0] <= threshold)
        print(f"within {threshold:>4}s : {count}")
    checked = [c[1] for c in candidates if c[1] is not None]
    if checked:
        for threshold in (50, 100, 250, 500):
            count = sum(1 for km in checked if km <= threshold)
            print(f"position agrees within {threshold:>4} km : {count}/{len(checked)}")

    # What the pool can offer, so a gap in the written batch can be read as a
    # selection failure or as an absent choice rather than guessed at.
    print("\n--- pool available to the selector ---")
    print(f"distinct film reels : {len({c[3]['film_subdir'] for c in candidates})}")
    stations = Counter(c[3]["station_code_normalized"] for c in candidates)
    for station, count in stations.most_common():
        print(f"  {station} : {count:6d} candidates")
    years = Counter(c[3]["year"] for c in candidates)
    total = sum(years.values())
    for year, count in sorted(years.items()):
        print(f"year {year} : {count:6d}  {100 * count / total:5.1f}%")


if __name__ == "__main__":
    main()
