#!/usr/bin/env python3
"""Build a larger, diverse Phase 1 pair set without copying scan files."""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT / "data/processed/review_ranked_top1500_reel.csv"
DEFAULT_EXISTING = ROOT / "outputs/calibration/phase1_pairs/manifest.csv"
DEFAULT_RECORDS = ROOT / "outputs/calibration/phase1_records.csv"
DEFAULT_LANDMARKS = ROOT / "outputs/landmarks/batch1500"
DEFAULT_OUT = ROOT / "outputs/calibration/phase1_pairs_target"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_name(row):
    return Path(row["nasa_png_file"]).stem.split("_", 1)[1]


def parse_size(value):
    width, height = (int(item) for item in value.lower().split("x", 1))
    return width, height


def choose_by_station(rows, target, quotas, randomize=False, seed=0):
    """Take rows while cycling across reels within each station."""
    rng = random.Random(seed)
    by_station_reel = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_station_reel[row["csa_station"]][row["csa_film_subdir"]].append(row)
    for station in by_station_reel:
        for reel in by_station_reel[station]:
            bucket = by_station_reel[station][reel]
            if randomize:
                rng.shuffle(bucket)
            else:
                bucket.sort(key=lambda item: int(item["rank"]))

    selected = []
    for station in sorted(by_station_reel, key=lambda item: (-quotas.get(item, 0), item)):
        reels = list(by_station_reel[station])
        if randomize:
            rng.shuffle(reels)
        else:
            reels.sort(key=lambda reel: int(by_station_reel[station][reel][0]["rank"]))
        count = 0
        while count < quotas.get(station, 0):
            added = False
            for reel in reels:
                bucket = by_station_reel[station][reel]
                if bucket and count < quotas[station]:
                    selected.append(bucket.pop(0))
                    count += 1
                    added = True
            if not added:
                break

    selected_names = {pair_name(row) for row in selected}
    remaining = [row for row in rows if pair_name(row) not in selected_names]
    if randomize:
        rng.shuffle(remaining)
    else:
        remaining.sort(key=lambda item: int(item["rank"]))
    for row in remaining:
        if len(selected) >= target:
            break
        selected.append(row)
    return selected[:target]


def assign_splits(rows, existing_reel_splits, seed):
    """Keep existing splits and assign unseen reels as held out at ~20%."""
    split_by_reel = dict(existing_reel_splits)
    new_reels = sorted({row["csa_film_subdir"] for row in rows} - split_by_reel.keys())
    random.Random(seed).shuffle(new_reels)
    held_target = max(1, round(len(rows) * 0.20)) if rows else 0
    held_count = 0
    for reel in new_reels:
        reel_count = sum(row["csa_film_subdir"] == reel for row in rows)
        if held_count < held_target:
            split_by_reel[reel] = "held_out"
            held_count += reel_count
        else:
            split_by_reel[reel] = "train"
    return [split_by_reel[row["csa_film_subdir"]] for row in rows], split_by_reel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        action="append",
        default=None,
        help="ranked candidate CSV; may be repeated for fetched segments",
    )
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument(
        "--used-manifest",
        type=Path,
        action="append",
        default=[],
        help="additional manifests whose pair names must not be selected",
    )
    parser.add_argument("--target", type=int, default=450, help="new candidate pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--fresh-only",
        action="store_true",
        help="write only the newly selected pairs instead of copying the base set",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"destination already exists: {args.out}")

    if args.fresh_only:
        # A fresh checkout has no historical Phase 1 set to merge. Reusing
        # those files remains supported for incremental batches.
        existing = read_csv(args.existing) if args.existing.is_file() else []
        old_record_rows = read_csv(args.records) if args.records.is_file() else []
    else:
        existing = read_csv(args.existing)
        old_record_rows = read_csv(args.records)
    old_records = {row["name"]: row for row in old_record_rows}
    existing_names = {row["pair_name"] for row in existing}
    existing_reel_splits = {row["reel"]: row["split"] for row in existing}
    used_names = set(existing_names)
    for used_manifest in args.used_manifest:
        used_rows = read_csv(used_manifest)
        used_names.update(row["pair_name"] for row in used_rows)
        existing_reel_splits.update(
            (row["reel"], row["split"]) for row in used_rows
        )

    csa_dir = ROOT / "data/raw/matches/csa_png"
    cdf_dir = ROOT / "data/raw/matches/nasa_cdf"
    landmark_names = {
        path.stem.removesuffix("_landmarks")
        for path in args.landmarks.glob("**/*_landmarks.json")
    }

    candidates = []
    seen = set()
    candidate_paths = args.candidates or [DEFAULT_CANDIDATES]
    for candidate_path in candidate_paths:
        for row in read_csv(candidate_path):
            name = pair_name(row)
            cdf_path = cdf_dir / f"{name}.cdf"
            csa_path = csa_dir / row["csa_file"]
            eligible = (
                name not in used_names
                and name not in seen
                and csa_path.is_file()
                and cdf_path.is_file()
                and row.get("cdf_match_kind") == "exact"
                and row.get("ambiguous") == "False"
                and row.get("has_freq_axis") == "True"
            )
            if eligible:
                candidates.append(row)
                seen.add(name)

    quotas = {
        "RES": round(args.target * 0.30),
        "OTT": round(args.target * 0.20),
        "TRO": round(args.target * 0.16),
        "QUI": round(args.target * 0.14),
        "ULA": round(args.target * 0.10),
        "BUR": round(args.target * 0.08),
        "KSH": round(args.target * 0.02),
    }
    # Correct rounding while keeping the intended station balance.
    quotas["RES"] += args.target - sum(quotas.values())
    selected = choose_by_station(
        candidates, args.target, quotas, args.randomize, args.seed
    )
    if len(selected) < args.target:
        raise SystemExit(
            f"only {len(selected)} eligible fresh pairs available; need {args.target}"
        )
    splits, reel_splits = assign_splits(selected, existing_reel_splits, args.seed)

    manifest = []
    records = []
    selection = []
    args.out.mkdir(parents=True)
    base_rows = [] if args.fresh_only else existing
    for index, row in enumerate(base_rows, start=1):
        name = row["pair_name"]
        old = old_records.get(name, {})
        csa_source = ROOT / row["csa_source"]
        cdf_source = ROOT / row["cdf_source"]
        pair_stem = f"{index:04d}__{name}"
        folder = args.out / row["split"]
        folder.mkdir(parents=True, exist_ok=True)
        csa_link = folder / f"{pair_stem}__CSA__{csa_source.name}"
        cdf_link = folder / f"{pair_stem}__NASA__{name}.cdf"
        csa_link.symlink_to(os.path.relpath(csa_source, csa_link.parent))
        cdf_link.symlink_to(os.path.relpath(cdf_source, cdf_link.parent))
        record = {
            "name": name,
            "split": row["split"],
            "reel": row["reel"],
            "station": old.get("station", ""),
            "width": old.get("width", ""),
            "height": old.get("height", ""),
            "width_class": old.get("width_class", ""),
            "format_class": row.get("format_class", ""),
            "sweep_class": row.get("sweep_class", "") or old.get("sweep_class", ""),
            "marker_count": old.get("marker_count", ""),
            "marker_rms_px": old.get("marker_rms_px", ""),
            "trace_within_1bin": old.get("trace_within_1bin", ""),
        }
        manifest.append(
            {
                "pair_number": index,
                "split": row["split"],
                "pair_name": name,
                "reel": row["reel"],
                "station": old.get("station", ""),
                "format_class": row.get("format_class", ""),
                "sweep_class": row.get("sweep_class", "") or old.get("sweep_class", ""),
                "csa_source": row["csa_source"],
                "cdf_source": row["cdf_source"],
                "csa_link": str(csa_link.relative_to(args.out)),
                "cdf_link": str(cdf_link.relative_to(args.out)),
                "selection_source": "existing_phase1",
                "candidate_rank": "",
            }
        )
        records.append(record)

    for row, split in zip(selected, splits):
        name = pair_name(row)
        width, height = parse_size(row["csa_size"])
        # Keep this import local so direct script execution also works.
        try:
            from scripts.dataset.build_calibration_profile import (
                format_class,
                sweep_class,
                width_class,
            )
        except ModuleNotFoundError:
            from scripts.dataset.build_calibration_profile import (
                format_class,
                sweep_class,
                width_class,
            )

        csa_source = csa_dir / row["csa_file"]
        cdf_source = cdf_dir / f"{name}.cdf"
        number = len(manifest) + 1
        pair_stem = f"{number:04d}__{name}"
        folder = args.out / split
        folder.mkdir(parents=True, exist_ok=True)
        csa_link = folder / f"{pair_stem}__CSA__{csa_source.name}"
        cdf_link = folder / f"{pair_stem}__NASA__{name}.cdf"
        csa_link.symlink_to(os.path.relpath(csa_source, csa_link.parent))
        cdf_link.symlink_to(os.path.relpath(cdf_source, cdf_link.parent))
        manifest.append(
            {
                "pair_number": number,
                "split": split,
                "pair_name": name,
                "reel": row["csa_film_subdir"],
                "station": row["csa_station"],
                "format_class": format_class(width, height),
                "sweep_class": sweep_class(row),
                "csa_source": str(csa_source.relative_to(ROOT)),
                "cdf_source": str(cdf_source.relative_to(ROOT)),
                "csa_link": str(csa_link.relative_to(args.out)),
                "cdf_link": str(cdf_link.relative_to(args.out)),
                "selection_source": "random_batch" if args.randomize else "top1500_diverse",
                "candidate_rank": row["rank"],
            }
        )
        records.append(
            {
                "name": name,
                "split": split,
                "reel": row["csa_film_subdir"],
                "station": row["csa_station"],
                "width": width,
                "height": height,
                "width_class": width_class(width),
                "format_class": format_class(width, height),
                "sweep_class": sweep_class(row),
                "marker_count": "",
                "marker_rms_px": "",
                "trace_within_1bin": "",
            }
        )
        selection.append(
            {
                "pair_name": name,
                "split": split,
                "reel": row["csa_film_subdir"],
                "station": row["csa_station"],
                "candidate_rank": row["rank"],
                "signal_fraction": row["signal_fraction"],
                "has_landmark_sidecar": name in landmark_names,
                "csa_file": row["csa_file"],
                "cdf_file": f"{name}.cdf",
            }
        )

    fields = [
        "pair_number",
        "split",
        "pair_name",
        "reel",
        "station",
        "format_class",
        "sweep_class",
        "csa_source",
        "cdf_source",
        "csa_link",
        "cdf_link",
        "selection_source",
        "candidate_rank",
    ]
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    record_fields = list(records[0])
    with (args.out / "phase1_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields)
        writer.writeheader()
        writer.writerows(records)
    with (args.out / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection[0]))
        writer.writeheader()
        writer.writerows(selection)

    counts = Counter(row["split"] for row in manifest)
    station_counts = Counter(row.get("station", "") for row in manifest)
    (args.out / "README.md").write_text(
        "# Target pair set\n\n"
        f"This set contains {len(manifest)} matched CSA/NASA pairs: {len(base_rows)} existing "
        f"pairs plus {len(selected)} fresh candidates. It intentionally includes a buffer "
        "because the Phase 7 quality gate, not file presence, defines usable warped data.\n\n"
        "- `manifest.csv` is the input to Phases 3–7.\n"
        "- `phase1_records.csv` supplies format and sweep metadata to frequency calibration.\n"
        "- `selection.csv` records the fresh-candidate selection and provenance.\n"
        "- `train/` and `held_out/` contain symlinks; raw scans and CDFs are not copied.\n\n"
        f"Split counts: train={counts['train']}, held_out={counts['held_out']}.\n"
        f"Station counts: {dict(sorted(station_counts.items()))}.\n",
        encoding="utf-8",
    )
    print(f"created {len(manifest)} pairs ({len(selected)} fresh) under {args.out}")
    print(f"split counts: {dict(counts)}")
    print(f"fresh station counts: {dict(Counter(row['station'] for row in selection))}")
    print(f"fresh landmark sidecars: {sum(row['has_landmark_sidecar'] for row in selection)}/{len(selection)}")


if __name__ == "__main__":
    main()
