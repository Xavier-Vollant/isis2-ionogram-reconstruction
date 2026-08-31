#!/usr/bin/env python3
"""Train a native-resolution image model on masked target patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research.models import image_features, model_constructor
from scripts.training.train_phase6_512_image_model import load_sample, rows_for

DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_512_image_model_experiment"


def make_patches(
    corpus, targets, rows, target_rows, patch_size, count, rng, input_channels=1
):
    """Sample masked training patches from the native-resolution corpus."""
    patches = []
    for row in rows:
        signal, target, mask = load_sample(corpus, targets, row, target_rows)
        height, width = signal.shape
        features = image_features(signal, input_channels)
        for _ in range(count):
            top = int(rng.integers(0, height - patch_size + 1))
            left = int(rng.integers(0, width - patch_size + 1))
            patches.append(
                (
                    features[:, top : top + patch_size, left : left + patch_size],
                    target[top : top + patch_size, left : left + patch_size],
                    mask[top : top + patch_size, left : left + patch_size],
                )
            )
    rng.shuffle(patches)
    return patches


def train(args):
    """Train one registered image model on the sampled patch set."""
    import torch
    from torch.nn import functional

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus_rows = [row for row in rows_for(args.corpus) if row["split"] == "train"]
    if args.train_limit is not None:
        if args.train_limit < 1:
            raise SystemExit("--train-limit must be positive")
        corpus_rows = corpus_rows[: args.train_limit]
    target_rows = {row["pair_name"]: row for row in rows_for(args.targets)}
    input_channels = 3 if args.model == "coord_unet" else 1
    model = model_constructor(args.model)(input_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []
    for epoch in range(args.epochs):
        patches = make_patches(
            args.corpus,
            args.targets,
            corpus_rows,
            target_rows,
            args.patch_size,
            args.patches_per_scan,
            np.random.default_rng(args.seed + epoch),
            input_channels,
        )
        model.train()
        epoch_losses = []
        for start in range(0, len(patches), args.batch_size):
            batch = patches[start : start + args.batch_size]
            films = torch.from_numpy(np.stack([item[0] for item in batch]))
            targets = torch.from_numpy(np.stack([item[1] for item in batch]))
            masks = torch.from_numpy(np.stack([item[2] for item in batch]))
            output = torch.sigmoid(model(films))
            loss = functional.smooth_l1_loss(output[masks], targets[masks])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        print(f"epoch {epoch + 1}/{args.epochs}: loss={losses[-1]:.6f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": args.model,
            "input_channels": input_channels,
            "grid_shape": [512, 512],
            "target": "NASA_ampl_normalized",
            "training": {
                "train_scans": len(corpus_rows),
                "epochs": args.epochs,
                "patch_size": args.patch_size,
                "patches_per_scan": args.patches_per_scan,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
        },
        checkpoint,
    )
    report = {
        "schema": "isis.phase6_native_512_image_model_patch_experiment.v1",
        "model": args.model,
        "loss": "masked_smooth_l1",
        "losses": losses,
        "loss_decreased": bool(losses[-1] < losses[0]),
        "output_checkpoint": str(checkpoint),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def main():
    """Parse CLI options and train native-resolution patch models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        choices=(
            "unet",
            "cnn_2d",
            "wide_unet",
            "residual_unet",
            "norm_residual_unet",
            "dilated_cnn",
            "coord_unet",
            "hybrid_unet",
        ),
        default="wide_unet",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-scan", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    if min(args.epochs, args.patch_size, args.patches_per_scan, args.batch_size) < 1:
        raise SystemExit("training parameters must be positive")
    train(args)


if __name__ == "__main__":
    main()
