#!/usr/bin/env python3
"""Render representative and worst held-out scans for a deep-evaluation model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram  # noqa: E402
from isis_research.models import image_features  # noqa: E402
from scripts.evaluation.evaluate_phase6_512_image_model import load_model  # noqa: E402
from scripts.evaluation.render_phase6_512_image_gallery import render_one, write_page  # noqa: E402
from scripts.training.train_phase6_512_image_model import load_sample, rows_for  # noqa: E402


DEFAULT_REPORT = ROOT / "outputs/evaluation/phase6_deep_report/report.json"
DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_deep_report/gallery"
CHECKPOINTS = {
    "hybrid_unet": ROOT / "outputs/evaluation/phase6_continual_models/hybrid_unet/best_model.pt",
    "norm_residual_unet": ROOT / "outputs/evaluation/phase6_continual_models/norm_residual_unet/best_model.pt",
}


def choose_rows(report, model, rows, count, seed):
    by_name = {row["pair_name"]: row for row in rows}
    ordered = []
    seen = set()
    for item in report["reports"][model]["image"]["worst_cases"][: min(6, count)]:
        row = by_name.get(item["pair_name"])
        if row is not None and row["pair_name"] not in seen:
            ordered.append(row)
            seen.add(row["pair_name"])
    ranked = sorted(
        report["records"].values(),
        key=lambda item: item["image"][model]["mae"],
    )
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
        item = ranked[int((len(ranked) - 1) * fraction)]
        row = by_name[item["pair_name"]]
        if row["pair_name"] not in seen:
            ordered.append(row)
            seen.add(row["pair_name"])
    station_rows = {}
    for item in ranked:
        station_rows.setdefault(item["station"], item)
    for station in sorted(station_rows):
        row = by_name[station_rows[station]["pair_name"]]
        if row["pair_name"] not in seen:
            ordered.append(row)
            seen.add(row["pair_name"])
    remainder = [row for row in rows if row["pair_name"] not in seen]
    random.Random(seed).shuffle(remainder)
    ordered.extend(remainder)
    return ordered[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), default="hybrid_unet")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--page", type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("count must be positive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or CHECKPOINTS[args.model]
    output_dir = args.output_dir or args.report.parent / f"gallery_{args.model}"
    page = args.page or args.report.parent / f"gallery_{args.model}.html"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output is not empty: {output_dir}")

    rows = [row for row in rows_for(args.corpus) if row["split"] == "held_out"]
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    selected = choose_rows(report, args.model, rows, args.count, args.seed)
    import torch

    model, checkpoint_data = load_model(checkpoint, torch)
    items = []
    for index, row in enumerate(selected, 1):
        scan = ionogram.read_validated(args.corpus / row["csa_artifact"])
        film, target, target_mask = load_sample(args.corpus, args.targets, row, target_rows)
        channels = checkpoint_data.get("input_channels", 1)
        with torch.no_grad():
            prediction = torch.sigmoid(
                model(torch.from_numpy(image_features(film, channels)[None, ...]))
            ).numpy()[0]
        raw_path = Path(row["raw_csa"])
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        output = output_dir / f"{index:02d}__{row['pair_name']}.png"
        item = render_one(
            index,
            row,
            scan,
            target,
            target_mask,
            prediction,
            raw_path,
            output,
        )
        items.append(item)
        print(f"rendered {index:02d}/{len(selected)}: {row['pair_name']}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "isis.phase6_deep_model_gallery.v1",
                "model": args.model,
                "checkpoint": str(checkpoint),
                "seed": args.seed,
                "items": items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_page(page, items, args.seed, checkpoint, output_dir)
    print(f"wrote {page}", flush=True)


if __name__ == "__main__":
    main()
