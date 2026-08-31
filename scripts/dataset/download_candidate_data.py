#!/usr/bin/env python3
"""Download, validate, and rank files behind metadata candidates.

The command fetches the CSA scan and NASA CDF, checks that both decode, and
writes the review table used by dataset preparation. Downloads are cached.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdflib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CSA_DIR = ROOT / "data" / "raw" / "matches" / "csa_png"
CDF_DIR = ROOT / "data" / "raw" / "matches" / "nasa_cdf"
REVIEW = ROOT / "outputs" / "review"
NASA_PNG = REVIEW / "nasa_png"

CDF_BASE = "https://spdf.gsfc.nasa.gov/pub/data/isis/topside_sounder/ionogram_cdf/isis2"
USER_AGENT = "final-isis/0.1 (candidate review; contact repository owner)"
HREF_RE = re.compile(r'href="([^"?/][^"]*)"', re.IGNORECASE)
FILL = -1e30
_listing_cache: dict[tuple[str, str, str], list[str]] = {}


def fetch(url: str, attempts: int = 4) -> bytes | None:
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


def list_day(station_dir: str, year: str, doy: str) -> list[str]:
    """Return CDF filenames in one SPDF station/year/day directory."""
    key = (station_dir, year, doy)
    if key not in _listing_cache:
        body = fetch(f"{CDF_BASE}/{station_dir}/{year}/{doy}/")
        names = HREF_RE.findall(body.decode("utf-8", "replace")) if body else []
        _listing_cache[key] = [name for name in names if name.lower().endswith(".cdf")]
    return _listing_cache[key]


def find_cdf(row: dict[str, str]) -> tuple[str | None, str]:
    """Find the versioned NASA CDF for a candidate row."""
    cached = row.get("nasa_cdf_file", "")
    if cached and (CDF_DIR / cached).is_file():
        return cached, "cached"
    stem = row.get("nasa_cdf_name_predicted", "").lower()
    if not stem:
        return None, "no_predicted_name"

    station_dir, _, _ = row["nasa_id"].split("/")[:3]
    iso = row.get("nasa_frame_sync_utc", "")
    if len(iso) < 8:
        return None, "no_frame_sync_time"
    year, doy = iso[:4], iso[5:8]
    names = list_day(station_dir, year, doy)
    if not names:
        return None, "no_day_directory"

    for name in names:
        if name.lower().startswith(stem):
            return name, "exact"

    # The header-derived name truncates frame-sync seconds. Allow a two-second
    # fallback when the archive's CDF filename differs by a recorded second.
    wanted_seconds = stem.rsplit("_", 1)[-1]
    best, best_gap = None, None
    for name in names:
        match = re.search(r"_(\d{13})_", name.lower())
        if not match:
            continue
        gap = abs(int(match.group(1)[-6:]) - int(wanted_seconds[-6:]))
        if best_gap is None or gap < best_gap:
            best, best_gap = name, gap
    if best is not None and best_gap is not None and best_gap <= 2:
        return best, f"nearest_{best_gap}s"
    return None, "not_found"


def download(url: str, target: Path, refresh: bool = False) -> bool:
    """Download one file atomically, reusing a non-empty cache entry."""
    if target.is_file() and target.stat().st_size > 0 and not refresh:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    try:
        body = fetch(url)
        if not body:
            return False
        temporary.write_bytes(body)
        temporary.replace(target)
        return True
    except (OSError, urllib.error.URLError, ValueError):
        return False
    finally:
        temporary.unlink(missing_ok=True)


def csa_target(row: dict[str, str]) -> Path:
    subdir = row["csa_film_subdir"].strip().replace(" ", "_").replace("/", "-")
    return CSA_DIR / f"{subdir}_{row['csa_image_filename']}"


def get_csa(row: dict[str, str], refresh: bool = False) -> Path | None:
    """Download one candidate's CSA image and return its cached path."""
    target = csa_target(row)
    return target if download(row["csa_image_url"], target, refresh) else None


def get_nasa(row: dict[str, str], refresh: bool = False) -> tuple[Path | None, str]:
    """Resolve, download, and return one candidate's NASA CDF."""
    name, kind = find_cdf(row)
    if not name:
        return None, kind
    station_dir = row["nasa_id"].split("/", 1)[0]
    iso = row["nasa_frame_sync_utc"]
    year, doy = iso[:4], iso[5:8]
    target = CDF_DIR / name
    url = f"{CDF_BASE}/{station_dir}/{year}/{doy}/{name}"
    return (target if download(url, target, refresh) else None), kind


def read_ionogram(path: Path) -> dict[str, object] | None:
    """Read the minimum CDF content needed for validation and review."""
    try:
        cdf = cdflib.CDF(str(path))
        amplitude = np.asarray(cdf.varget("ampl"))
        height = np.asarray(cdf.varget("v_height"), dtype=float).ravel()
        frequency = np.asarray(cdf.varget("freq"), dtype=float).ravel()
    except (OSError, KeyError, IndexError, TypeError, ValueError, RuntimeError):
        return None
    if amplitude.ndim != 2 or amplitude.size == 0 or height.size == 0:
        return None
    usable = np.isfinite(frequency) & (frequency > FILL)
    has_axis = (
        usable.size >= 2
        and int(usable.sum()) >= max(2, int(0.5 * usable.size))
        and float(frequency[usable].max() - frequency[usable].min()) > 0
    )
    return {
        "amplitude": amplitude,
        "height": height,
        "frequency": frequency,
        "has_freq_axis": has_axis,
        "signal_fraction": float(np.mean(amplitude > 0)),
        "shape": amplitude.shape,
    }


def validate_csa(path: Path) -> tuple[bool, str]:
    """Decode a CSA image and return ``(is_valid, width_by_height)``."""
    try:
        with Image.open(path) as image:
            size = f"{image.width}x{image.height}"
            image.load()
        return True, size
    except (OSError, SyntaxError, TypeError, ValueError):
        return False, ""


def render(ionogram: dict[str, object], path: Path, title: str) -> None:
    amplitude = ionogram["amplitude"]
    height = ionogram["height"]
    frequency = ionogram["frequency"]
    assert isinstance(amplitude, np.ndarray)
    assert isinstance(height, np.ndarray)
    assert isinstance(frequency, np.ndarray)
    fig, axis = plt.subplots(figsize=(5.2, 4.0), dpi=110)
    if bool(ionogram["has_freq_axis"]):
        usable = np.isfinite(frequency) & (frequency > FILL)
        extent = [
            float(frequency[usable].min()),
            float(frequency[usable].max()),
            float(height.max()),
            float(height.min()),
        ]
        axis.set_xlabel("frequency (MHz)")
    else:
        extent = [0, amplitude.shape[0], float(height.max()), float(height.min())]
        axis.set_xlabel("scan line (no frequency interpolation)")
    axis.imshow(
        amplitude.T,
        aspect="auto",
        cmap="gray_r",
        extent=extent,
        interpolation="nearest",
    )
    axis.set_ylabel("virtual height (km)")
    axis.set_title(title, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def process(row: dict[str, str], refresh: bool = False) -> dict[str, object]:
    """Fetch both sides of one candidate and add validation/review fields."""
    result: dict[str, object] = dict(row)
    csa_path = get_csa(row, refresh)
    result["csa_ok"], result["csa_size"] = (
        validate_csa(csa_path) if csa_path else (False, "")
    )
    if csa_path and result["csa_ok"]:
        result["csa_path"] = csa_path

    try:
        cdf_path, match_kind = get_nasa(row, refresh)
    except (
        OSError,
        urllib.error.URLError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        cdf_path, match_kind = None, f"download_error:{type(error).__name__}"
    result["cdf_match_kind"] = match_kind
    result["nasa_ok"] = False
    if cdf_path:
        ionogram = read_ionogram(cdf_path)
        if ionogram is not None:
            result["nasa_ok"] = True
            result.update(
                {
                    "nasa_cdf_path": cdf_path,
                    "nasa_cdf_file": cdf_path.name,
                    "has_freq_axis": ionogram["has_freq_axis"],
                    "signal_fraction": ionogram["signal_fraction"],
                    "nasa_shape": f"{ionogram['shape'][0]}x{ionogram['shape'][1]}",
                }
            )
            png = NASA_PNG / f"{int(row['rank']):03d}_{cdf_path.stem}.png"
            render(ionogram, png, f"{row['nasa_station']} {row['nasa_frame_sync_utc']}")
            result["nasa_png"] = png
    return result


def rank_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        0 if row.get("has_freq_axis") else 1,
        -round(float(row.get("signal_fraction", 0.0)), 2),
        float(row["dt_seconds"]),
        float(row.get("position_km") or 999),
    )


EXTRA_COLUMNS = [
    "review_rank",
    "has_freq_axis",
    "signal_fraction",
    "nasa_shape",
    "csa_size",
    "cdf_match_kind",
    "nasa_cdf_file",
    "csa_file",
    "nasa_png_file",
]


def write_ranked(
    rows: list[dict[str, object]], out_csv: Path, base_columns: list[str]
) -> None:
    """Write validated candidates with review rank and derived file fields."""
    base = [column for column in base_columns if column not in EXTRA_COLUMNS]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTRA_COLUMNS + base)
        writer.writeheader()
        for position, row in enumerate(rows, start=1):
            record = {column: row.get(column, "") for column in base}
            record.update(
                {
                    "review_rank": position,
                    "has_freq_axis": str(bool(row.get("has_freq_axis"))),
                    "signal_fraction": f"{float(row.get('signal_fraction', 0.0)):.4f}",
                    "nasa_shape": row.get("nasa_shape", ""),
                    "csa_size": row.get("csa_size", ""),
                    "cdf_match_kind": row.get("cdf_match_kind", ""),
                    "nasa_cdf_file": row.get("nasa_cdf_file", ""),
                    "csa_file": Path(row["csa_path"]).name
                    if row.get("csa_path")
                    else "",
                    "nasa_png_file": Path(row["nasa_png"]).name
                    if row.get("nasa_png")
                    else "",
                }
            )
            writer.writerow(record)


def write_page(rows: list[dict[str, object]]) -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    parts = [
        "<title>ISIS-2 candidate review</title>",
        (
            "<style>body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#111;color:#eee}"
            "h1{font-size:20px}.card{border:1px solid #333;border-radius:8px;margin-bottom:20px;padding:14px;background:#181818}"
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.pane{background:#000;padding:8px}"
            "img{width:100%;display:block}@media(max-width:800px){.pair{grid-template-columns:1fr}}"
        ),
        "</style>",
        "<h1>ISIS-2 candidate review</h1>",
        (
            f"<p>{len(rows)} metadata candidates, ranked after file validation. "
            "The comparison is not a pixel-level scientific match.</p>"
        ),
    ]
    for position, row in enumerate(rows, start=1):
        csa_image = (
            f"<img src='../../{Path(row['csa_path']).relative_to(ROOT)}'>"
            if row.get("csa_path")
            else "<p>CSA scan unavailable</p>"
        )
        nasa_image = (
            f"<img src='nasa_png/{Path(row['nasa_png']).name}'>"
            if row.get("nasa_png")
            else "<p>NASA CDF unavailable</p>"
        )
        parts.append(
            f"<div class='card'><h2>#{position} {html.escape(str(row['csa_station']))} "
            f"{html.escape(str(row['csa_timestamp_utc']))}</h2><p>"
            f"Δt {row['dt_seconds']} s · position {row.get('position_km', '')} km · "
            f"signal {float(row.get('signal_fraction', 0.0)):.1%} · "
            f"frequency axis {row.get('has_freq_axis', False)}</p>"
            f"<div class='pair'><div class='pane'>{csa_image}</div>"
            f"<div class='pane'>{nasa_image}</div></div></div>"
        )
    (REVIEW / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main(argv=None) -> None:
    """Parse CLI options and download/validate a candidate batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROCESSED / "candidate_matches_top100.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.start < 0 or args.workers < 1:
        raise SystemExit("--start must be non-negative and --workers must be positive")

    suffix = args.candidates.stem.replace("candidate_matches_", "")
    out_csv = args.output or PROCESSED / f"review_ranked_{suffix}.csv"
    global REVIEW, NASA_PNG
    REVIEW = ROOT / "outputs" / f"review_{suffix}"
    NASA_PNG = REVIEW / "nasa_png"

    with args.candidates.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        base_columns = list(reader.fieldnames or [])
        rows = list(reader)
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    for directory in (CSA_DIR, CDF_DIR, NASA_PNG):
        directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        processed = list(pool.map(lambda row: process(row, args.refresh), rows))

    usable = [row for row in processed if row.get("csa_ok") and row.get("nasa_ok")]
    usable.sort(key=rank_key)
    write_page(usable)
    write_ranked(usable, out_csv, base_columns)

    stats = Counter()
    for row in processed:
        stats["csa_valid"] += bool(row.get("csa_ok"))
        stats["nasa_valid"] += bool(row.get("nasa_ok"))
        stats[f"cdf_{row.get('cdf_match_kind', 'unknown')}"] += 1
        if row.get("csa_ok") and row.get("nasa_ok"):
            stats["usable_pairs"] += 1
            stats["with_freq_axis"] += bool(row.get("has_freq_axis"))
    print(f"processed {len(processed)} candidates")
    for name, value in sorted(stats.items()):
        print(f"{name:24} {value}")
    print(f"review CSV: {out_csv.relative_to(ROOT)}")
    print(f"review page: {(REVIEW / 'index.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
