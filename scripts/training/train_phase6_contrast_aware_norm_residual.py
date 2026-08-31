#!/usr/bin/env python3
"""Fine-tune the normalized residual model with a contrast-aware loss.

The initial checkpoint is read-only; a new candidate checkpoint is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research.models import image_features, model_constructor
from scripts.training.continual_train_phase6_models import (
    DEFAULT_CORPUS,
    DEFAULT_TARGETS,
    load_sample,
    make_patches,
    read_group_rows,
    split_pool,
    usable_rows,
)

DEFAULT_INITIAL = (
    ROOT / "outputs/evaluation/phase6_continual_models/norm_residual_unet/best_model.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_contrast_aware_norm_residual"
DEFAULT_GROUPS = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"


def contrast_loss(output, target, masks, torch):
    """Return the batch contrast penalty over pixels with known targets."""
    terms = []
    for prediction, expected, mask in zip(output, target, masks):
        if int(mask.sum()) > 1:
            known_prediction = prediction[mask]
            known_target = expected[mask]
            terms.append(
                torch.abs(
                    torch.std(known_prediction, unbiased=False)
                    - torch.std(known_target, unbiased=False)
                )
            )
    if not terms:
        return output.new_zeros(())
    return torch.stack(terms).mean()


def correlation(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def validate(model, refs, input_channels, batch_size, torch):
    """Evaluate a model on references and return macro image metrics."""
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(refs), batch_size):
            batch = refs[start : start + batch_size]
            samples = [load_sample(ref) for ref in batch]
            films, targets, masks = zip(*samples)
            output = torch.sigmoid(
                model(
                    torch.from_numpy(
                        np.stack(
                            [image_features(film, input_channels) for film in films]
                        )
                    )
                )
            ).numpy()
            for target, mask, prediction in zip(targets, masks, output):
                known = mask & np.isfinite(target) & np.isfinite(prediction)
                if int(known.sum()) < 2:
                    continue
                predicted = prediction[known].astype(float)
                expected = target[known].astype(float)
                records.append(
                    {
                        "mae": float(np.mean(np.abs(predicted - expected))),
                        "correlation": correlation(predicted, expected),
                        "prediction_std_ratio": float(
                            np.std(predicted) / max(np.std(expected), 1e-9)
                        ),
                    }
                )
    if not records:
        return {"scans": 0}
    return {
        "scans": len(records),
        "macro_mae": float(np.mean([item["mae"] for item in records])),
        "macro_correlation": float(np.mean([item["correlation"] for item in records])),
        "mean_prediction_std_ratio": float(
            np.mean([item["prediction_std_ratio"] for item in records])
        ),
    }


def main():
    """Parse CLI options and train the contrast-aware model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-scan", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--contrast-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    if (
        args.epochs < 1
        or args.patch_size < 1
        or args.patches_per_scan < 1
        or args.batch_size < 1
    ):
        raise SystemExit("training parameters must be positive")
    if args.contrast_weight < 0 or not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("contrast weight or validation fraction is invalid")

    import torch
    from torch.nn import functional

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    group_rows = read_group_rows(args.groups)
    pool = usable_rows(args.corpus, args.targets, group_rows=group_rows)
    train_refs, validation_refs = split_pool(pool, args.validation_fraction, args.seed)
    if not train_refs or not validation_refs:
        raise SystemExit("training/validation split is empty")

    checkpoint = torch.load(args.initial, map_location="cpu")
    if checkpoint.get("model") != "norm_residual_unet":
        raise SystemExit(
            f"expected a norm_residual_unet checkpoint, got {checkpoint.get('model')!r}"
        )
    model = model_constructor("norm_residual_unet")(1)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    history = []
    best_validation = None
    best_epoch = None
    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        patches, complete = make_patches(
            train_refs,
            args.patch_size,
            args.patches_per_scan,
            1,
            rng,
            balanced=True,
        )
        if not complete or not patches:
            raise SystemExit(f"could not create a complete patch epoch: {epoch}")
        model.train()
        pixel_losses = []
        contrast_losses = []
        for start in range(0, len(patches), args.batch_size):
            batch = patches[start : start + args.batch_size]
            films = torch.from_numpy(np.stack([item[0] for item in batch]))
            targets = torch.from_numpy(np.stack([item[1] for item in batch]))
            masks = torch.from_numpy(np.stack([item[2] for item in batch]))
            output = torch.sigmoid(model(films))
            pixel = functional.smooth_l1_loss(output[masks], targets[masks])
            contrast = contrast_loss(output, targets, masks, torch)
            loss = pixel + args.contrast_weight * contrast
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pixel_losses.append(float(pixel.detach()))
            contrast_losses.append(float(contrast.detach()))

        validation = validate(
            model,
            validation_refs,
            1,
            args.validation_batch_size,
            torch,
        )
        record = {
            "epoch": epoch,
            "patches": len(patches),
            "pixel_loss": float(np.mean(pixel_losses)),
            "contrast_loss": float(np.mean(contrast_losses)),
            "total_loss": float(
                np.mean(pixel_losses) + args.contrast_weight * np.mean(contrast_losses)
            ),
            "validation": validation,
        }
        history.append(record)
        print(
            f"epoch {epoch}/{args.epochs}: pixel={record['pixel_loss']:.6f} "
            f"contrast={record['contrast_loss']:.6f} "
            f"val_mae={validation['macro_mae']:.6f} "
            f"val_corr={validation['macro_correlation']:.4f}",
            flush=True,
        )
        better = best_validation is None or (
            validation["macro_mae"] < best_validation["macro_mae"] - 1e-6
            or (
                abs(validation["macro_mae"] - best_validation["macro_mae"]) <= 1e-6
                and validation["macro_correlation"]
                > best_validation["macro_correlation"]
            )
        )
        if better:
            best_validation = validation
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": "norm_residual_unet",
                    "input_channels": 1,
                    "grid_shape": [512, 512],
                    "target": "NASA_ampl_normalized",
                    "training": {
                        "mode": "contrast_aware_finetune",
                        "initial_checkpoint": str(args.initial),
                        "train_scans": len(train_refs),
                        "validation_scans": len(validation_refs),
                        "epochs": epoch,
                        "patch_size": args.patch_size,
                        "patches_per_scan": args.patches_per_scan,
                        "contrast_weight": args.contrast_weight,
                        "learning_rate": args.learning_rate,
                        "seed": args.seed,
                    },
                    "validation": validation,
                },
                best_path,
            )

    report = {
        "schema": "isis.phase6_contrast_aware_norm_residual_training.v1",
        "initial_checkpoint": str(args.initial),
        "output_checkpoint": str(best_path),
        "train_scans": len(train_refs),
        "validation_scans": len(validation_refs),
        "validation_reel_disjoint": True,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "contrast_weight": args.contrast_weight,
        "learning_rate": args.learning_rate,
        "history": history,
        "best_validation": best_validation,
        "production_checkpoint_changed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
