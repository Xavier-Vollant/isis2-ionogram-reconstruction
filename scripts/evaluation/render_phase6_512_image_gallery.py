#!/usr/bin/env python3
"""Render held-out examples for a native-resolution image model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram
from isis_research.models import model_constructor

DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_CHECKPOINT = ROOT / "outputs/evaluation/phase6_512_image_model_full_v2/model.pt"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/csa_cdf_side_by_side/image_model_gallery_10"
DEFAULT_PAGE = (
    ROOT / "outputs/evaluation/csa_cdf_side_by_side/gallery_image_model_10.html"
)
GRID_SHAPE = (512, 512)
PANEL_SIZE = (340, 340)
PANEL_GAP = 16
MARGIN = 20
COLS = 3
ROWS = 2
AXIS_LEFT = 46
AXIS_RIGHT = 8
AXIS_TOP = 5
AXIS_BOTTOM = 42
CANVAS_WIDTH = MARGIN * 2 + COLS * PANEL_SIZE[0] + (COLS - 1) * PANEL_GAP
CANVAS_HEIGHT = 900


def font(size, bold=False):
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


TITLE = font(15, True)
SMALL = font(11)
TINY = font(10)


def rows_for(path):
    with (Path(path) / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grey(values, valid=None, fixed=False):
    values = np.asarray(values, dtype=float)
    if fixed:
        low, high = 0.0, 1.0
    else:
        finite = np.isfinite(values)
        low, high = (
            np.percentile(values[finite], [2, 98]) if finite.any() else (0.0, 1.0)
        )
    pixels = np.clip(
        (np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-9), 0, 1
    )
    image = np.repeat(np.uint8(pixels * 255)[..., None], 3, axis=2)
    if valid is not None:
        image[~valid] = (16, 20, 27)
    return Image.fromarray(image, mode="RGB")


def raw_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return image.copy(), image.size


def axis_image(image, x_label, x_limits, y_label, y_limits):
    """Put readable data axes around one panel image."""
    image = ImageOps.contain(
        image,
        (
            PANEL_SIZE[0] - AXIS_LEFT - AXIS_RIGHT,
            PANEL_SIZE[1] - AXIS_TOP - AXIS_BOTTOM,
        ),
        method=Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", PANEL_SIZE, (8, 10, 14))
    left = AXIS_LEFT
    top = AXIS_TOP + (PANEL_SIZE[1] - AXIS_TOP - AXIS_BOTTOM - image.height) // 2
    right = left + image.width - 1
    bottom = top + image.height - 1
    canvas.paste(image, (left, top))
    draw = ImageDraw.Draw(canvas)
    axis_colour = (190, 202, 214)
    tick_colour = (126, 143, 160)
    draw.line((left, top, left, bottom), fill=axis_colour, width=1)
    draw.line((left, bottom, right, bottom), fill=axis_colour, width=1)

    def tick_text(value):
        value = float(value)
        if abs(value) >= 1000:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.2f}".rstrip("0").rstrip(".")

    for fraction in (0.0, 0.5, 1.0):
        x = round(left + fraction * (right - left))
        draw.line((x, bottom, x, bottom + 4), fill=tick_colour, width=1)
        draw.text(
            (x, bottom + 5),
            tick_text(np.linspace(*x_limits, 3)[round(fraction * 2)]),
            fill=axis_colour,
            font=TINY,
            anchor="mt",
        )

        y = round(top + fraction * (bottom - top))
        draw.line((left - 4, y, left, y), fill=tick_colour, width=1)
        draw.text(
            (left - 6, y),
            tick_text(np.linspace(*y_limits, 3)[round(fraction * 2)]),
            fill=axis_colour,
            font=TINY,
            anchor="rm",
        )

    draw.text(
        ((left + right) // 2, PANEL_SIZE[1] - 15),
        x_label,
        fill=axis_colour,
        font=TINY,
        anchor="mm",
    )
    y_title = Image.new("RGBA", (PANEL_SIZE[1], 20), (0, 0, 0, 0))
    ImageDraw.Draw(y_title).text(
        (PANEL_SIZE[1] // 2, 10), y_label, fill=axis_colour, font=TINY, anchor="mm"
    )
    y_title = y_title.rotate(90, expand=True)
    canvas.paste(y_title, (0, (PANEL_SIZE[1] - y_title.height) // 2), y_title)
    return canvas


def panel(canvas, position, title, image, detail, x_axis, y_axis):
    x, y = position
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (x, y, x + PANEL_SIZE[0], y + PANEL_SIZE[1] + 55),
        fill=(24, 30, 40),
        outline=(62, 74, 91),
    )
    draw.text((x + 8, y + 8), title, fill=(232, 238, 245), font=TITLE)
    image = axis_image(image, x_axis[0], x_axis[1], y_axis[0], y_axis[1])
    canvas.paste(image, (x, y + 36))
    draw.text((x + 8, y + PANEL_SIZE[1] + 41), detail, fill=(196, 207, 218), font=SMALL)


def load_model(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("model") != "unet":
        raise ValueError("the gallery expects the trained U-Net checkpoint")
    model = model_constructor("unet")(1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def render_one(index, row, scan, target, target_valid, prediction, raw_path, output):
    """Render one raw-film, target, prediction, and error gallery card."""
    film_signal = np.where(scan.valid_mask, 1.0 - scan.intensity, 0.0).astype(
        np.float32
    )
    model_display = grey(prediction, scan.valid_mask, fixed=True)
    target_display = grey(target, target_valid, fixed=True)
    error = np.abs(prediction - target)
    error_display = grey(error, target_valid)
    raw, raw_size = raw_image(raw_path)
    physical_x = (
        "frequency (MHz)",
        (float(scan.frequency_mhz[0]), float(scan.frequency_mhz[-1])),
    )
    physical_y = (
        "height (km)",
        (float(scan.virtual_height_km[0]), float(scan.virtual_height_km[-1])),
    )
    raw_x = ("pixel column", (0.0, float(raw_size[0] - 1)))
    raw_y = ("pixel row", (0.0, float(raw_size[1] - 1)))
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), (14, 19, 27))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (MARGIN, 12),
        f"{index:02d} · {row['pair_name']} · held-out",
        fill=(239, 244, 249),
        font=TITLE,
    )
    draw.text(
        (MARGIN, 32),
        "CSA → NASA-like image translation · native 512×512 comparison",
        fill=(168, 181, 196),
        font=SMALL,
    )
    positions = [
        (MARGIN, 58),
        (MARGIN + PANEL_SIZE[0] + PANEL_GAP, 58),
        (MARGIN + 2 * (PANEL_SIZE[0] + PANEL_GAP), 58),
        (MARGIN, 470),
        (MARGIN + PANEL_SIZE[0] + PANEL_GAP, 470),
        (MARGIN + 2 * (PANEL_SIZE[0] + PANEL_GAP), 470),
    ]
    panel(
        canvas,
        positions[0],
        "Raw CSA scan",
        raw,
        f"original PNG · {raw_size[0]}×{raw_size[1]} px",
        raw_x,
        raw_y,
    )
    panel(
        canvas,
        positions[1],
        "Standardized CSA input",
        grey(film_signal, scan.valid_mask),
        "usable film-only · native 512×512",
        physical_x,
        physical_y,
    )
    panel(
        canvas,
        positions[2],
        "NASA CDF target",
        grey(target, target_valid, fixed=True),
        "held-out reference · native 512×512",
        physical_x,
        physical_y,
    )
    panel(
        canvas,
        positions[3],
        "Model output",
        model_display,
        "continuous NASA-like output · native 512×512",
        physical_x,
        physical_y,
    )
    panel(
        canvas,
        positions[4],
        "NASA target",
        target_display,
        "matched reference · native 512×512",
        physical_x,
        physical_y,
    )
    panel(
        canvas,
        positions[5],
        "Absolute image error",
        error_display,
        f"mean masked error · {float(np.mean(error[target_valid])):.3f}",
        physical_x,
        physical_y,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {
        "pair_name": row["pair_name"],
        "station": row.get("station", ""),
        "raw_csa": str(raw_path),
        "standardized": row["csa_artifact"],
        "target": row["nasa_cdf"],
        "image": output.name,
        "shape": list(GRID_SHAPE),
        "masked_mae": float(np.mean(error[target_valid])),
    }


def write_page(page, items, seed, checkpoint, output_dir):
    """Write one HTML page containing the rendered gallery cards."""
    image_dir = output_dir.name
    figures = []
    for index, item in enumerate(items, 1):
        figures.append(
            f"<figure><figcaption>{index:02d}. {item['pair_name']} · {item['station']} · "
            f"masked MAE {item['masked_mae']:.3f}</figcaption>"
            f'<a href="{image_dir}/{item["image"]}" target="_blank">'
            f'<img loading="lazy" src="{image_dir}/{item["image"]}" '
            f'alt="Raw CSA, standardized CSA, NASA target, model output, and error for {item["pair_name"]}"></a></figure>'
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Direct CSA to NASA image model · 10 held-out scans</title>
<style>:root{{color-scheme:dark}}body{{margin:0;padding:24px;background:#10151c;color:#e8edf3;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1900px;margin:auto}}h1{{margin:0 0 6px;font-size:24px}}h2{{margin:24px 0 8px;font-size:19px}}p{{color:#aebac8;max-width:1500px}}figure{{margin:26px 0;border-top:1px solid #394554;padding-top:16px}}figcaption{{margin-bottom:10px;font-weight:600;color:#dce5ee}}img{{display:block;width:100%;height:auto;border:1px solid #394554;border-radius:8px}}a{{color:#8fc7ff}}code{{color:#b9e6d3;font-size:12px}}.callout{{padding:12px 14px;background:#18202b;border-left:4px solid #42dcd8}}</style></head>
<body><main><h1>Direct CSA → NASA image model · 10 held-out scans</h1>
<p>This is the new image-to-image model trained only on usable film-only standardized CSA scans. Every panel now includes visible axes: raw scans use source pixels, while calibrated panels use frequency (MHz) and virtual height (km).</p>
<p><b>Seed:</b> {seed} · <b>Checkpoint:</b> <code>{Path(checkpoint).name}</code> · <a href="gallery_native_models_10.html">previous seven-model gallery</a></p>
<div class="callout"><b>How to read it:</b> the model output is a continuous NASA-like image, not a thresholded line or occupancy mask. The absolute-error panel shows where the prediction differs from the matched CDF target.</div>
<h2>Random held-out examples</h2>{"".join(figures)}
</main></body></html>
"""
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(html, encoding="utf-8")


def main():
    """Parse CLI options and render the native-resolution comparison gallery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("count must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output is not empty: {args.output_dir}")

    corpus_rows = [row for row in rows_for(args.corpus) if row["split"] == "held_out"]
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    if len(corpus_rows) < args.count:
        raise SystemExit("not enough held-out rows")
    selected = random.Random(args.seed).sample(corpus_rows, args.count)
    model = load_model(args.checkpoint)
    import torch

    items = []
    for index, row in enumerate(selected, 1):
        artifact = args.corpus / row["csa_artifact"]
        scan = ionogram.read_validated(artifact)
        target_row = target_rows[row["pair_name"]]
        with np.load(
            args.targets / target_row["target_artifact"], allow_pickle=False
        ) as data:
            target = np.asarray(data["amplitude"], dtype=np.float32)
            target_valid = np.asarray(data["valid_mask"], dtype=bool) & scan.valid_mask
        film_signal = np.where(scan.valid_mask, 1.0 - scan.intensity, 0.0).astype(
            np.float32
        )
        with torch.no_grad():
            prediction = torch.sigmoid(
                model(torch.from_numpy(film_signal[None, None]))
            ).numpy()[0]
        raw_path = Path(row["raw_csa"])
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        item = render_one(
            index,
            row,
            scan,
            target,
            target_valid,
            prediction,
            raw_path,
            args.output_dir / f"{index:02d}__{row['pair_name']}.png",
        )
        items.append(item)
        print(f"rendered {index:02d}/{args.count}: {row['pair_name']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "isis.phase6_native_512_image_gallery.v1",
                "seed": args.seed,
                "checkpoint": str(args.checkpoint),
                "items": items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_page(args.page, items, args.seed, args.checkpoint, args.output_dir)
    print(f"wrote {args.page}", flush=True)


if __name__ == "__main__":
    main()
