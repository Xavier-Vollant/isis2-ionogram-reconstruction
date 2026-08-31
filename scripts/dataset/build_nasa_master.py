#!/usr/bin/env python
"""Build the NASA metadata table, with one row per digital ISIS-2 ionogram.

The builder crawls the SPDF pass-header tree, caches downloads, and reads fields
by label so missing telemetry values remain visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from isis_research.nasa.stations import STATION_DIRS

BASE_URL = (
    "https://spdf.gsfc.nasa.gov/pub/data/isis/topside_sounder/ionogram_header_ascii"
)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "nasa" / "headers"
INDEX_PATH = ROOT / "data" / "raw" / "nasa" / "pass_index.json"
OUT_DIR = ROOT / "data" / "processed"
USER_AGENT = "isis-2-research/0.1 (metadata inventory; contact repository owner)"
REQUEST_DELAY = 0.15  # per worker; SPDF bans bursts

HREF_RE = re.compile(r'href="([^"?/][^"]*)"', re.IGNORECASE)
LABEL_RE = re.compile(r"^\s*([A-Z][A-Z0-9 /\-]*?):\s*(.*)$")
PASS_TIME_RE = re.compile(
    r"(\d{1,2})/(\d{2})/(\d{2})\s+\(\s*(\d+)\)\s+(\d{2}):(\d{2}):(\d{2})"
)

# Label in the file -> column name. Anything else is kept as an unknown label.
FIELD_LABELS = {
    "SATELLITE": "satellite_label",
    "STATION": "station_code",
    "POWER": "transmitter_power",
    "SNDREC": "sounder_receiver",
    "SF": "swept_freq_range",
    "DMODE": "dmode",
    "GMODE": "gmode",
    "MIXED-MODE": "mixed_mode",
    "AITMODE": "ait_mode",
    "FIXED FREQ": "fixed_freq_mhz",
    "YR": "year_raw",
    "DAY": "day_of_year",
    "HR": "hour",
    "MIN": "minute",
    "SEC": "second",
    "LMT": "lmt",
    "GGLAT": "gglat",
    "GGLONG": "gglon",
    "HGT": "hgt_km",
    "GMLTM": "gmlt",
    "GMLAT": "gmlat",
    "GMLONG": "gmlon",
    "FH": "fh_mhz",
    "INVLAT": "invlat",
    "DIP": "dip",
    "CHI": "chi",
    "SUN": "sunlit",
    "L": "l_shell",
    "CEP": "cep",
    "VLF": "vlf",
    "RPA": "rpa",
    "IMS": "ims",
    "SPS": "sps",
    "EPD": "epd",
    "RLP": "rlp",
    "ASP": "asp",
}

PASS_LABELS = {
    "Satellite Number": "satellite_number",
    "Station Name": "station_code_pass",
    "Station Code": "station_code_num",
    "Tape Number": "tape_number",
    "Pass Number": "pass_number",
}

COLUMNS = [
    "nasa_id",
    "header_file",
    "station_dir",
    "station_dir_code",
    "year_dir",
    "ionogram_index",
    "avg_bin_filename",
    "full_bin_filename",
    "cdf_name_predicted",
    "quality_comments_raw",
    "satellite_number",
    "station_code_pass",
    "station_code_num",
    "tape_number",
    "pass_number",
    "pass_start_utc",
    "pass_end_utc",
    "ad_conversion_time",
    "station_log_comments",
    "operator_comments",
    "n_ionograms_declared",
    "satellite_label",
    "station_code",
    "transmitter_power",
    "sounder_receiver",
    "swept_freq_range",
    "dmode",
    "gmode",
    "mixed_mode",
    "ait_mode",
    "fixed_freq_mhz",
    "year_raw",
    "day_of_year",
    "hour",
    "minute",
    "second",
    "frame_sync_utc",
    "sec_of_day",
    "lmt",
    "gglat",
    "gglon",
    "hgt_km",
    "gmlt",
    "gmlat",
    "gmlon",
    "fh_mhz",
    "invlat",
    "dip",
    "chi",
    "sunlit",
    "l_shell",
    "cep",
    "vlf",
    "rpa",
    "ims",
    "sps",
    "epd",
    "rlp",
    "asp",
    "sounding_status_determined",
    "freq_interp_available",
    "worldmap_zero",
    "fixed_freq_suspect",
    "blank_field_count",
    "source_nul_bytes",
    "unknown_labels",
    "parse_status",
]


def fetch(url, attempts=5):
    """SPDF refuses connections under concurrent load, so back off and retry."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except OSError:
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    return None


def list_dir(url):
    """Return names from an SPDF directory listing, or an empty list on 404."""
    body = fetch(url)
    if body is None:
        return []
    names = HREF_RE.findall(body.decode("utf-8", errors="replace"))
    return [n for n in names if not n.startswith(("http", "mailto", "#"))]


def discover(limit_stations=None):
    """-> [(station_dir, year, filename)] for every ISIS-2 pass header."""
    targets = []
    stations = STATION_DIRS[:limit_stations] if limit_stations else STATION_DIRS
    for station in stations:
        years = [
            n.strip("/")
            for n in list_dir(f"{BASE_URL}/{station}/isis2/")
            if n.endswith("/") and n.strip("/").isdigit()
        ]
        if not years:
            print(f"  {station}: no isis2 data")
            continue
        count = 0
        for year in sorted(years):
            files = [
                n
                for n in list_dir(f"{BASE_URL}/{station}/isis2/{year}/")
                if n.endswith(".asc")
            ]
            targets.extend((station, year, name) for name in files)
            count += len(files)
        print(f"  {station}: {count} passes across {len(years)} years", flush=True)
    return targets


def cache_path(station, year, name):
    return RAW_DIR / station / year / name


def cache_ready(target):
    """Return whether a cached pass is present and non-empty."""
    path = cache_path(*target)
    return path.is_file() and path.stat().st_size > 0


def download_one(target):
    """Never raises: one refused connection must not abandon 11k passes."""
    station, year, name = target
    path = cache_path(station, year, name)
    if cache_ready(target):
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        body = fetch(f"{BASE_URL}/{station}/isis2/{year}/{name}")
    except (OSError, urllib.error.URLError, ValueError):
        return False
    if body is None:
        return False
    path.write_bytes(body)
    time.sleep(REQUEST_DELAY)
    return True


def acquire(targets, workers, passes=6, cooldown=120):
    """Fetch missing passes, sweeping repeatedly over whatever still failed.

    SPDF starts refusing connections when pushed (8 workers earned an instant
    ban that took minutes to lift), so this stays slow on purpose and waits out
    a refusal between sweeps rather than hammering through it.
    """
    for sweep in range(1, passes + 1):
        missing = [t for t in targets if not cache_ready(t)]
        if not missing:
            break
        print(f"sweep {sweep}: {len(missing)} passes missing", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(download_one, missing):
                done += 1
                if done % 250 == 0:
                    print(f"  {done}/{len(missing)}", flush=True)
        if sweep < passes and any(not cache_ready(t) for t in targets):
            print(f"  cooling down {cooldown}s", flush=True)
            time.sleep(cooldown)
    still_missing = [t for t in targets if not cache_ready(t)]
    if still_missing:
        print(f"WARNING: {len(still_missing)} passes could not be fetched")
    return still_missing


def pass_time(text):
    """`72/01/03  (72003)  02:44:04` -> `1972-01-03T02:44:04`."""
    match = PASS_TIME_RE.search(text or "")
    if not match:
        return ""
    yy, month, day, _, hour, minute, second = match.groups()
    year = int(yy)
    # Recording years are 1970s; A/D conversion years are 2000s.
    year += 2000 if year < 50 else 1900
    return f"{year:04d}-{int(month):02d}-{int(day):02d}T{hour}:{minute}:{second}"


def parse_pass_header(lines):
    header = {}
    for index, line in enumerate(lines):
        match = LABEL_RE.match(line) or re.match(r"^\s*([\w /]+):\s*(.*)$", line)
        if not match:
            if "IONOGRAM HEADERS FOLLOW" in line:
                header["n_ionograms_declared"] = line.strip().split()[0]
            continue
        label, value = match.group(1).strip(), match.group(2).strip()
        if label in PASS_LABELS:
            header[PASS_LABELS[label]] = value.split()[0] if value else ""
        elif label.startswith("Start Time"):
            header["pass_start_utc"] = pass_time(value)
        elif label.startswith("End Time"):
            header["pass_end_utc"] = pass_time(value)
        elif label.startswith("A/D Conversion"):
            header["ad_conversion_time"] = pass_time(value)
        elif label.startswith("Comments From Station Log"):
            header["station_log_comments"] = (
                lines[index + 1].strip() if value == "" else value
            )
        elif label.startswith("A/D Operator Comments"):
            header["operator_comments"] = (
                lines[index + 1].strip() if value == "" else value
            )
    return header


def parse_ionogram(chunk):
    fields, comments, unknown = {}, [], []
    full_bin = ""
    for index, line in enumerate(chunk):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("IONOGRAMS:"):
            fields["avg_bin_filename"] = stripped.split(":", 1)[1].strip()
            continue
        if index == 1 and stripped.upper().endswith(".BIN"):
            full_bin = stripped
            continue
        match = LABEL_RE.match(line)
        if match:
            label, value = match.group(1).strip(), match.group(2).strip()
            if label in FIELD_LABELS:
                fields[FIELD_LABELS[label]] = value
                continue
            unknown.append(label)
        comments.append(stripped)
    fields["full_bin_filename"] = full_bin
    fields["quality_comments_raw"] = " | ".join(comments)
    fields["unknown_labels"] = ",".join(sorted(set(unknown)))
    return fields


def frame_sync(fields):
    """-> (iso, sec_of_day, status)."""
    try:
        year = int(fields.get("year_raw", ""))
        doy = int(fields.get("day_of_year", ""))
        hour = int(fields.get("hour", ""))
        minute = int(fields.get("minute", ""))
        second = float(fields.get("second", ""))
    except (TypeError, ValueError):
        return "", "", "no_frame_sync_time"
    year += 2000 if year < 50 else 1900
    if not (0 < doy <= 366 and hour < 24 and minute < 60 and second < 61):
        return "", "", "invalid_frame_sync_time"
    return (
        f"{year:04d}-{doy:03d}T{hour:02d}:{minute:02d}:{second:09.6f}",
        f"{hour * 3600 + minute * 60 + second:.6f}",
        "ok",
    )


def parse_file(path, station, year_dir):
    """Parse one NASA pass-header file into normalized ionogram rows."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Three of NASA's 11000 pass files carry embedded NUL bytes; one has 512 of
    # them. Strip them so the row is still usable, but count them so a corrupt
    # source file is visible in the table rather than silently cleaned up.
    nul_bytes = text.count("\x00")
    if nul_bytes:
        text = text.replace("\x00", "")
    lines = text.splitlines()
    starts = [
        i for i, line in enumerate(lines) if line.strip().startswith("IONOGRAMS:")
    ]
    header = parse_pass_header(lines[: starts[0] if starts else len(lines)])

    rows = []
    for order, start in enumerate(starts):
        end = starts[order + 1] if order + 1 < len(starts) else len(lines)
        fields = parse_ionogram(lines[start:end])
        iso, sec_of_day, status = frame_sync(fields)
        comments = fields.get("quality_comments_raw", "")

        measured = [FIELD_LABELS[k] for k in FIELD_LABELS]
        blanks = sum(1 for name in measured if not fields.get(name, ""))
        fixed_freq = fields.get("fixed_freq_mhz", "")

        row = {name: "" for name in COLUMNS}
        row.update(header)
        row.update({k: v for k, v in fields.items() if k in row})
        row.update(
            {
                "nasa_id": f"{station}/{year_dir}/{path.stem}#{order:04d}",
                "header_file": path.name,
                "station_dir": station,
                "station_dir_code": station.split("_")[0],
                "year_dir": year_dir,
                "ionogram_index": str(order),
                "cdf_name_predicted": predicted_cdf(fields, station),
                "frame_sync_utc": iso,
                "sec_of_day": sec_of_day,
                "sounding_status_determined": str("NOT DETERMINED" not in comments),
                "freq_interp_available": str("NO FREQ INTERP" not in comments),
                "worldmap_zero": str(fields.get("gglat", "") in ("0.00", "0.", "0")),
                # NASA documents 0.25 MHz being recorded where 0.12 MHz was meant.
                "fixed_freq_suspect": str(fixed_freq.startswith("0.25")),
                "blank_field_count": str(blanks),
                "source_nul_bytes": str(nul_bytes),
                "parse_status": status,
            }
        )
        rows.append(row)
    return rows, header


def predicted_cdf(fields, station):
    """CDAWeb name for the average product, e.g. `I2_AV_RES_1972003024600`."""
    avg = fields.get("avg_bin_filename", "")
    # A4RES00306B01_03509_72003_024600.BIN -> yyddd 72003, hhmmss 024600
    match = re.search(r"_(\d{2})(\d{3})_(\d{6})\.BIN", avg, re.IGNORECASE)
    if not match:
        return ""
    yy, doy, hhmmss = match.groups()
    year = int(yy) + (2000 if int(yy) < 50 else 1900)
    return f"I2_AV_{station.split('_')[0]}_{year}{doy}{hhmmss}"


def main():
    """Parse CLI options and build the NASA master inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="number of download workers; keep this low for SPDF",
    )
    parser.add_argument(
        "--rediscover",
        action="store_true",
        help="re-walk the directory tree instead of using the cached index",
    )
    parser.add_argument(
        "--limit-stations",
        type=int,
        default=None,
        help="crawl only the first N stations (for testing)",
    )
    parser.add_argument(
        "--skip-acquire",
        action="store_true",
        help="parse the existing cache without crawling",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_acquire:
        targets = [
            (p.parent.parent.name, p.parent.name, p.name)
            for p in sorted(RAW_DIR.glob("*/*/*.asc"))
        ]
        print(f"parsing {len(targets)} cached passes")
    else:
        # Discovery costs ~220 listing requests; keep it so resumes skip it.
        if INDEX_PATH.exists() and not args.rediscover:
            targets = [tuple(t) for t in json.loads(INDEX_PATH.read_text())]
            print(f"using cached pass index: {len(targets)} passes")
        else:
            print("discovering pass headers ...", flush=True)
            targets = discover(args.limit_stations)
            INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            INDEX_PATH.write_text(json.dumps(targets))
            print(f"\n{len(targets)} passes discovered", flush=True)
        acquire(targets, args.workers)

    stats = Counter()
    with open(OUT_DIR / "nasa_master.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for station, year_dir, name in targets:
            path = cache_path(station, year_dir, name)
            if not path.exists():
                stats["missing_file"] += 1
                continue
            try:
                rows, _ = parse_file(path, station, year_dir)
            except (
                OSError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                csv.Error,
            ) as error:  # keep going; record the casualty
                stats["parse_error"] += 1
                print(f"  parse error {path.name}: {error}")
                continue
            stats["passes"] += 1
            stats["empty_passes"] += not rows
            for row in rows:
                stats["ionograms"] += 1
                stats[f"parse_{row['parse_status']}"] += 1
                stats["sounding_undetermined"] += (
                    row["sounding_status_determined"] == "False"
                )
                stats["no_freq_interp"] += row["freq_interp_available"] == "False"
                stats["fixed_freq_suspect"] += row["fixed_freq_suspect"] == "True"
                if row["unknown_labels"]:
                    stats["rows_with_unknown_labels"] += 1
                writer.writerow(row)

    print("\n--- nasa_master.csv ---")
    for name, value in sorted(stats.items()):
        print(f"{name:32} {value}")


if __name__ == "__main__":
    main()
