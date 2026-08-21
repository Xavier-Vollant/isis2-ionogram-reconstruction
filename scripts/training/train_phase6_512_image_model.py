#!/usr/bin/env python3
"""Train a small masked CSA-to-NASA image translator.

This is the first image-to-image smoke path.  It intentionally trains only a
small subset when ``--train-limit`` is supplied; later experiments can reuse
the same loader with different models and losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram  # noqa: E402
from isis_research.models import model_constructor  # noqa: E402


DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_512_image_model_smoke"
GRID_SHAPE = (512, 512)


def rows_for(path):
    with (Path(path) / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_sample(corpus, targets, row, target_rows):
    scan = ionogram.read_validated(corpus / row["csa_artifact"])
    target_row = target_rows[row["pair_name"]]
    with np.load(targets / target_row["target_artifact"], allow_pickle=False) as data:
        target = np.asarray(data["amplitude"], dtype=np.float32)
        target_valid = np.asarray(data["valid_mask"], dtype=bool)
    if scan.intensity.shape != GRID_SHAPE or target.shape != GRID_SHAPE:
        raise ValueError(f"{row['pair_name']}: expected 512x512 input and target")
    film_valid = np.asarray(scan.valid_mask, dtype=bool)
    input_signal = np.where(film_valid, 1.0 - scan.intensity, 0.0).astype(np.float32)
    loss_mask = film_valid & target_valid & np.isfinite(target)
    return input_signal, target, loss_mask


def correlation(left, right, mask):
    x = left[mask].astype(float)
    y = right[mask].astype(float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-limit", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", choices=("unet", "cnn_2d"), default="unet")
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.train_limit < 1 or args.epochs < 1:
        raise SystemExit("--train-limit and --epochs must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")

    import torch
    import torch.nn.functional as functional

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)
    corpus_rows = [row for row in rows_for(args.corpus) if row["split"] == "train"]
    if args.train_limit < len(corpus_rows):
        corpus_rows = corpus_rows[: args.train_limit]
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    samples = [load_sample(args.corpus, args.targets, row, target_rows) for row in corpus_rows]

    model = model_constructor(args.model)(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(len(samples))
        epoch_losses = []
        for index in order:
            film, target, mask = samples[index]
            x = torch.from_numpy(film[None, None])
            y = torch.from_numpy(target[None])
            known = torch.from_numpy(mask[None])
            output = torch.sigmoid(model(x))
            loss = functional.smooth_l1_loss(output[known], y[known])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"epoch {epoch + 1}/{args.epochs}: loss={losses[-1]:.6f}", flush=True)

    model.eval()
    train_metrics = []
    with torch.no_grad():
        for row, (film, target, mask) in zip(corpus_rows, samples):
            prediction = torch.sigmoid(model(torch.from_numpy(film[None, None]))).numpy()[0]
            train_metrics.append(
                {
                    "pair_name": row["pair_name"],
                    "target_std": float(np.std(target[mask])),
                    "prediction_std": float(np.std(prediction[mask])),
                    "masked_mae": float(np.mean(np.abs(prediction[mask] - target[mask]))),
                    "masked_correlation": correlation(prediction, target, mask),
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": args.model,
            "input_channels": 1,
            "grid_shape": list(GRID_SHAPE),
            "target": "NASA_ampl_normalized",
        },
        checkpoint,
    )
    report = {
        "schema": "isis.phase6_native_512_image_model_smoke.v1",
        "model": args.model,
        "train_scans": len(samples),
        "epochs": args.epochs,
        "loss": "masked_smooth_l1",
        "losses": losses,
        "train_metrics": train_metrics,
        "output_checkpoint": str(checkpoint),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "prediction_nonconstant": bool(any(item["prediction_std"] > 1e-4 for item in train_metrics)),
        "mean_prediction_target_std_ratio": float(
            np.mean(
                [item["prediction_std"] / max(item["target_std"], 1e-9) for item in train_metrics]
            )
        ),
        "mean_train_correlation": float(
            np.mean([item["masked_correlation"] for item in train_metrics])
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
