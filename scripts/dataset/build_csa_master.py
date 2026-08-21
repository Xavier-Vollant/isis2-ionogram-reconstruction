#!/usr/bin/env python
"""Build the CSA master metadata table: one row per scanned ISIS-2 film image.

Joins the three CSA root inventories on (film subdirectory, image stem):

  result_master_ISIS2.csv  base inventory, one row per PNG
  microapp_ISIS.csv        subset carrying the max_depth and fmin measurements
  orbitcheck_isis_2.csv    TLE-propagated satellite geometry per image

Raw values are preserved verbatim alongside normalized ones; every
normalization that fails is recorded in a status column rather than dropped.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from urllib.parse import quote
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from isis_research.nasa.stations import STATIONS, reconcile  # noqa: E402

BASE_URL = (
    "https://donnees-data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes"
    "/pub/Alouette-ISIS/ISIS-2"
)
SOURCES = ("result_master_ISIS2.csv", "microapp_ISIS.csv", "orbitcheck_isis_2.csv")
IMAGE_URL_PREFIX = f"{BASE_URL}/"

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "csa"
OUT_DIR = ROOT / "data" / "processed"

# CSA labels ISIS-1 as satellite 3 / "ISIS A" and ISIS-2 as 4 / "ISIS B".
SATELLITE_IDS = {"3": "ISIS-1", "4": "ISIS-2"}
PATH_LABELS = {"ISIS A": "ISIS-1", "ISIS B": "ISIS-2"}

TIMESTAMP_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
COORD_RE = re.compile(r"^(-?[\d.]+)\s*([NSEW])?$", re.IGNORECASE)

COLUMNS = [
    "csa_id",
    "image_path_raw",
    "batch_dir",
    "film_subdir",
    "image_filename",
    "image_url",
    "in_result_master",
    "in_microapp",
    "in_orbitcheck",
    "duplicate_key",
    "satellite_number_raw",
    "satellite_label_from_path",
    "satellite_id",
    "satellite_conflict",
    "station_number_raw",
    "station_name_raw",
    "station_code_raw",
    "station_code_normalized",
    "station_code_conflict",
    "station_lat",
    "station_lon",
    "timestamp_raw",
    "timestamp_utc",
    "time_parse_status",
    "year",
    "day_of_year",
    "sec_of_day",
    "max_depth",
    "fmin",
    "tle_epoch",
    "orbit_flag_raw",
    "sat_lat",
    "sat_lon",
    "sat_height_km",
    "station_elevation_deg",
    "station_distance_km",
    "events_raw",
    "visibility_plausible",
    "timestamp_trust",
]


def download(refresh=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        target = RAW_DIR / name
        if target.exists() and not refresh:
            print(f"cached  {name} ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"fetching {name} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)
        print(f"  saved {target.stat().st_size / 1e6:.1f} MB")


def path_key(subdir, filename):
    """Join key tolerant of the archive's own path drift.

    The same image appears as `Image0103.png`, `image3793.png` and bare
    `Image0043` across the three files, and the batch directory prefix is
    sometimes `b14_R014207878` and sometimes `R014207878`. Subdirectory plus
    case-folded stem is the part that stays stable.
    """
    stem = re.sub(r"\.png$", "", filename.strip(), flags=re.IGNORECASE)
    return subdir.strip().lower(), stem.lower()


def split_image_path(raw):
    """`b11_R014207871/B1-35-20 ISIS B D-409/Image0001.png` -> its three parts."""
    parts = [p for p in raw.strip().split("/") if p]
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    return "", "", parts[0] if parts else ""


def parse_timestamp(raw):
    """-> (iso, status, year, day_of_year, sec_of_day).

    CSA timestamps are transcribed from film and include impossible values such
    as `1972-10-00` (day zero). Those keep their time-of-day, which is still
    usable for matching, but get no date.
    """
    raw = (raw or "").strip()
    if not raw:
        return "", "empty", "", "", ""
    match = TIMESTAMP_RE.match(raw)
    if not match:
        return "", "unparseable", "", "", ""

    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    time_ok = hour < 24 and minute < 60 and second < 60
    sec_of_day = str(hour * 3600 + minute * 60 + second) if time_ok else ""

    if not time_ok:
        return "", "invalid_time", str(year), "", ""
    if month == 0 or day == 0:
        return "", "invalid_day", str(year), "", sec_of_day
    try:
        day_of_year = date(year, month, day).timetuple().tm_yday
    except ValueError:
        return "", "invalid_day", str(year), "", sec_of_day

    iso = f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    return iso, "ok", str(year), str(day_of_year), sec_of_day


def parse_coord(raw):
    """`69.7N` and `-95` both appear; return signed degrees or None."""
    raw = (raw or "").strip()
    if not raw:
        return None
    match = COORD_RE.match(raw)
    if not match:
        return None
    value = float(match.group(1))
    if (match.group(2) or "").upper() in ("S", "W"):
        value = -abs(value)
    return value


def read_microapp(path):
    index = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            _, subdir, filename = split_image_path(row.get("file_name", ""))
            index[path_key(subdir, filename)] = (
                row.get("max_depth", ""),
                row.get("fmin", ""),
            )
    return index


def read_orbitcheck(path):
    index = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            key = path_key(row.get("Subdirectory", ""), row.get("Filename", ""))
            index[key] = (
                row.get("TLE_Epoch", ""),
                row.get("Flag", ""),
                row.get("Sat_Lat", ""),
                row.get("Sat_Lon", ""),
                row.get("Sat_Height", ""),
                row.get("Station_Alt", ""),
                row.get("Station_Distance", ""),
                row.get("Events", ""),
            )
    return index


def trust(time_status, visibility):
    if time_status != "ok":
        return f"suspect_{time_status}"
    if visibility == "False":
        return "suspect_below_horizon"
    if visibility == "":
        return "unverified"
    return "ok"


def build():
    micro = read_microapp(RAW_DIR / "microapp_ISIS.csv")
    print(f"microapp rows indexed:   {len(micro)}")
    orbit = read_orbitcheck(RAW_DIR / "orbitcheck_isis_2.csv")
    print(f"orbitcheck rows indexed: {len(orbit)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    station_audit = Counter()
    seen = set()

    with (
        open(
            RAW_DIR / "result_master_ISIS2.csv",
            newline="",
            encoding="utf-8",
            errors="replace",
        ) as source,
        open(OUT_DIR / "csa_master.csv", "w", newline="", encoding="utf-8") as out,
    ):
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()

        for row in csv.DictReader(source):
            raw_path = row.get("File name", "")
            batch, subdir, filename = split_image_path(raw_path)
            key = path_key(subdir, filename)

            duplicate = key in seen
            seen.add(key)
            stats["rows"] += 1
            stats["duplicate_key"] += duplicate

            satellite_raw = (row.get("Satellite number") or "").strip()
            label = next((v for k, v in PATH_LABELS.items() if k in subdir), "")
            satellite_id = SATELLITE_IDS.get(satellite_raw, "")
            conflict = "yes" if label and satellite_id and label != satellite_id else ""
            stats["satellite_conflict"] += bool(conflict)

            lat = parse_coord(row.get("Latitude", ""))
            lon = parse_coord(row.get("Longitude", ""))
            code_raw = (row.get("Ground station code") or "").strip()
            code_norm, code_conflict = reconcile(code_raw, lat, lon)
            if code_conflict:
                stats[f"station_{code_conflict}"] += 1
            station_audit[
                (
                    (row.get("Ground station number") or "").strip(),
                    (row.get("Ground station name") or "").strip(),
                    code_raw,
                    code_norm,
                    code_conflict,
                )
            ] += 1

            iso, time_status, year, doy, sec = parse_timestamp(row.get("Timestamp", ""))
            stats[f"time_{time_status}"] += 1

            max_depth, fmin = micro.get(key, ("", ""))
            stats["in_microapp"] += key in micro

            orbit_row = orbit.get(key)
            stats["in_orbitcheck"] += orbit_row is not None
            if orbit_row:
                tle, flag, slat, slon, sheight, elevation, distance, events = orbit_row
                try:
                    visibility = str(float(elevation) > 0)
                except (TypeError, ValueError):
                    visibility = ""
            else:
                tle = flag = slat = slon = sheight = elevation = distance = events = ""
                visibility = ""

            verdict = trust(time_status, visibility)
            stats[f"trust_{verdict}"] += 1

            writer.writerow(
                {
                    "csa_id": f"{key[0]}/{key[1]}",
                    "image_path_raw": raw_path,
                    "batch_dir": batch,
                    "film_subdir": subdir,
                    "image_filename": filename,
                    # Film subdirectories contain spaces, so the stored URL is
                    # percent-encoded and directly fetchable.
                    "image_url": IMAGE_URL_PREFIX + quote(raw_path.strip()),
                    "in_result_master": "True",
                    "in_microapp": str(key in micro),
                    "in_orbitcheck": str(orbit_row is not None),
                    "duplicate_key": str(duplicate),
                    "satellite_number_raw": satellite_raw,
                    "satellite_label_from_path": label,
                    "satellite_id": satellite_id,
                    "satellite_conflict": conflict,
                    "station_number_raw": (
                        row.get("Ground station number") or ""
                    ).strip(),
                    "station_name_raw": (row.get("Ground station name") or "").strip(),
                    "station_code_raw": code_raw,
                    "station_code_normalized": code_norm,
                    "station_code_conflict": code_conflict,
                    "station_lat": "" if lat is None else lat,
                    "station_lon": "" if lon is None else lon,
                    "timestamp_raw": (row.get("Timestamp") or "").strip(),
                    "timestamp_utc": iso,
                    "time_parse_status": time_status,
                    "year": year,
                    "day_of_year": doy,
                    "sec_of_day": sec,
                    "max_depth": max_depth,
                    "fmin": fmin,
                    "tle_epoch": tle,
                    "orbit_flag_raw": flag,
                    "sat_lat": slat,
                    "sat_lon": slon,
                    "sat_height_km": sheight,
                    "station_elevation_deg": elevation,
                    "station_distance_km": distance,
                    "events_raw": events,
                    "visibility_plausible": visibility,
                    "timestamp_trust": verdict,
                }
            )

    with open(
        OUT_DIR / "csa_station_audit.csv", "w", newline="", encoding="utf-8"
    ) as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "station_number",
                "station_name",
                "code_raw",
                "code_normalized",
                "conflict",
                "n_images",
            ]
        )
        for fields, count in sorted(station_audit.items(), key=lambda kv: -kv[1]):
            writer.writerow([*fields, count])

    return stats, station_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="re-download the CSA inventories"
    )
    parser.add_argument(
        "--skip-acquire",
        action="store_true",
        help="build from already-downloaded inventories",
    )
    args = parser.parse_args()

    if not args.skip_acquire:
        download(refresh=args.refresh)

    stats, station_audit = build()

    print("\n--- csa_master.csv ---")
    for name, value in sorted(stats.items()):
        print(f"{name:32} {value}")
    conflicts = [row for row in station_audit if row[4]]
    print(
        f"\nstations seen: {len({r[2] for r in station_audit})} codes, "
        f"{len(conflicts)} conflicting"
    )
    for number, name, raw, norm, conflict in conflicts:
        print(f"  station {number} {name!r}: {raw} -> {norm or '?'} ({conflict})")
    print(f"\nknown NASA stations: {len(STATIONS)}")


if __name__ == "__main__":
    main()
