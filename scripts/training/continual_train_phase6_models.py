#!/usr/bin/env python3
"""Train models for a wall-clock budget with optional data replenishment.

New batches are validated before they join the training pool. NASA targets are
labels, never model inputs.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from isis_research import ionogram
from isis_research.evaluation import splits as evaluation_splits
from isis_research.models import image_features, model_constructor
from scripts.dataset.run_amplitude_batch import next_batch

DEFAULT_CORPUS = ROOT / "outputs/evaluation/phase6_usable_film_only_512"
DEFAULT_TARGETS = ROOT / "outputs/evaluation/phase6_usable_film_only_512_targets"
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/phase6_continual_models"
DEFAULT_INITIAL_ROOT = ROOT / "outputs/evaluation/phase6_512_image_model_experiments"
DEFAULT_BATCH_ROOT = ROOT / "outputs/amplitude_batches"
DEFAULT_BASE_MANIFEST = ROOT / "outputs/calibration/phase1_pairs_6400/manifest.csv"
DEFAULT_BASE_RECORDS = ROOT / "outputs/calibration/phase1_pairs_6400/phase1_records.csv"
DEFAULT_PROFILE = ROOT / "configs/film_calibration_profile.json"
# Entries are (model name, input channel count).
MODEL_SPECS = {
    "unet": ("unet", 1),
    "cnn_2d": ("cnn_2d", 1),
    "wide_unet": ("wide_unet", 1),
    "residual_unet": ("residual_unet", 1),
    "norm_residual_unet": ("norm_residual_unet", 1),
    "dilated_cnn": ("dilated_cnn", 1),
    "coord_unet": ("coord_unet", 3),
    "hybrid_unet": ("hybrid_unet", 1),
}
INITIAL_CHECKPOINT_DIRS = {
    "unet": "tiny_unet",
    "cnn_2d": "cnn_2d_full",
    "wide_unet": "wide_unet_retrained",
    "residual_unet": "residual_unet",
    "norm_residual_unet": "norm_residual_unet_full",
    "dilated_cnn": "dilated_cnn_full",
    "coord_unet": "coord_unet_full",
    "hybrid_unet": "hybrid_unet_full",
}


@dataclass(frozen=True)
class SampleRef:
    """Paths and metadata needed to load one training or validation sample."""

    corpus: Path
    targets: Path
    row: dict[str, str]
    target_artifact: str
    station: str
    axis_profile: str
    coverage_bin: str


def read_rows(path: Path) -> list[dict[str, str]]:
    with (path / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_group_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["pair_name"]: row for row in csv.DictReader(handle)}


def coverage_bin(value: float) -> str:
    if value < 0.9:
        return "<0.90"
    if value < 0.99:
        return "0.90-0.99"
    return ">=0.99"


def usable_rows(
    corpus: Path,
    targets: Path,
    limit: int | None = None,
    group_rows: dict[str, dict[str, str]] | None = None,
) -> list[SampleRef]:
    """Return only complete, aligned, finite, usable training pairs."""
    target_rows = {row["pair_name"]: row for row in read_rows(targets)}
    accepted = []
    rejected = 0
    for row in read_rows(corpus):
        if row.get("split") != "train" or row.get("status", "usable") != "usable":
            continue
        target_row = target_rows.get(row.get("pair_name", ""))
        group_row = (group_rows or {}).get(row.get("pair_name", ""), {})
        row = {
            **row,
            "reel": row.get("reel") or group_row.get("reel") or row["pair_name"],
        }
        csa_path = corpus / row["csa_artifact"]
        target_path = targets / target_row["target_artifact"] if target_row else None
        try:
            if (
                target_path is None
                or not csa_path.is_file()
                or not target_path.is_file()
            ):
                raise ValueError("missing artifact")
            scan = ionogram.read_validated(csa_path)
            with np.load(target_path, allow_pickle=False) as data:
                amplitude = np.asarray(data["amplitude"], dtype=np.float32)
                valid = np.asarray(data["valid_mask"], dtype=bool)
                frequency = np.asarray(data["frequency_mhz"], dtype=float)
                height = np.asarray(data["virtual_height_km"], dtype=float)
            if (
                scan.intensity.shape != (512, 512)
                or amplitude.shape != (512, 512)
                or valid.shape != (512, 512)
                or not valid.any()
                or not np.isfinite(amplitude[valid]).all()
                or not np.array_equal(frequency, scan.frequency_mhz)
                or not np.array_equal(height, scan.virtual_height_km)
                or not np.all(np.diff(frequency) > 0)
                or not np.all(np.diff(height) > 0)
            ):
                raise ValueError("shape, mask, finite-value, or axis check failed")
            accepted.append(
                SampleRef(
                    corpus,
                    targets,
                    row,
                    target_row["target_artifact"],
                    row.get("station") or "<blank>",
                    f"{scan.frequency_mhz[0]:.1f}-{scan.frequency_mhz[-1]:.1f}MHz",
                    coverage_bin(float(valid.mean())),
                )
            )
        except (KeyError, OSError, ValueError):
            rejected += 1
        if limit is not None and len(accepted) >= limit:
            break
    print(
        f"usable pool: {len(accepted)} accepted, {rejected} rejected from {corpus}",
        flush=True,
    )
    if not accepted:
        raise RuntimeError(f"no usable training pairs found in {corpus}")
    return accepted


def split_pool(pool: list[SampleRef], fraction: float, seed: int):
    """Make a deterministic reel-disjoint validation split."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    if fraction == 0.0 or len(pool) < 2:
        return pool, []
    groups = {ref.row.get("reel") or ref.row.get("pair_name") for ref in pool}
    if len(groups) < 2:
        return pool, []
    folds = min(len(groups), max(2, round(1.0 / fraction)))
    records = [
        {**ref.row, "_ref": ref, "reel": ref.row.get("reel") or ref.row["pair_name"]}
        for ref in pool
    ]
    _, validation_records = next(
        evaluation_splits.grouped_folds(records, folds, seed=seed, key="reel")
    )
    validation_keys = {
        (str(record["_ref"].corpus), record["_ref"].row["pair_name"])
        for record in validation_records
    }
    train = [
        ref
        for ref in pool
        if (str(ref.corpus), ref.row["pair_name"]) not in validation_keys
    ]
    validation = [
        ref
        for ref in pool
        if (str(ref.corpus), ref.row["pair_name"]) in validation_keys
    ]
    evaluation_splits.check_disjoint(
        [
            {**ref.row, "reel": ref.row.get("reel") or ref.row["pair_name"]}
            for ref in train
        ],
        [
            {**ref.row, "reel": ref.row.get("reel") or ref.row["pair_name"]}
            for ref in validation
        ],
        key="reel",
    )
    return train, validation


def balanced_refs(pool: list[SampleRef], rng):
    """Draw one epoch with equal probability for station/axis/coverage buckets."""
    buckets = {}
    for ref in pool:
        key = (ref.station, ref.axis_profile, ref.coverage_bin)
        buckets.setdefault(key, []).append(ref)
    if len(buckets) <= 1:
        order = list(pool)
        rng.shuffle(order)
        return order
    keys = list(buckets)
    order = []
    for _ in range(len(pool)):
        bucket = buckets[keys[int(rng.integers(len(keys)))]]
        order.append(bucket[int(rng.integers(len(bucket)))])
    return order


def load_sample(ref: SampleRef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a sample reference and return signal, target, and valid mask."""
    scan = ionogram.read_validated(ref.corpus / ref.row["csa_artifact"])
    with np.load(ref.targets / ref.target_artifact, allow_pickle=False) as data:
        target = np.asarray(data["amplitude"], dtype=np.float32)
        target_valid = np.asarray(data["valid_mask"], dtype=bool)
    film = np.where(scan.valid_mask, 1.0 - scan.intensity, 0.0).astype(np.float32)
    loss_mask = scan.valid_mask & target_valid & np.isfinite(target)
    return film, target, loss_mask


def make_patches(
    pool: list[SampleRef],
    patch_size: int,
    patches_per_scan: int,
    input_channels: int,
    rng,
    balanced: bool = True,
    deadline: float | None = None,
):
    """Create masked native-resolution patches from a set of references."""
    patches = []
    refs = balanced_refs(pool, rng) if balanced else list(pool)
    if not balanced:
        rng.shuffle(refs)
    for ref in refs:
        if deadline is not None and time.monotonic() >= deadline:
            return patches, False
        film, target, mask = load_sample(ref)
        height, width = film.shape
        features = image_features(film, input_channels)
        for _ in range(patches_per_scan):
            for _ in range(16):
                top = int(rng.integers(0, height - patch_size + 1))
                left = int(rng.integers(0, width - patch_size + 1))
                patch_mask = mask[top : top + patch_size, left : left + patch_size]
                if patch_mask.any():
                    patches.append(
                        (
                            features[
                                :, top : top + patch_size, left : left + patch_size
                            ],
                            target[top : top + patch_size, left : left + patch_size],
                            patch_mask,
                        )
                    )
                    break
    rng.shuffle(patches)
    return patches, True


def model_for(name, torch):
    model_name, input_channels = MODEL_SPECS[name]
    return model_constructor(model_name)(input_channels), input_channels


def load_or_create(
    name, output_root: Path, initial_root: Path, learning_rate: float, torch
):
    output = output_root / name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "model.pt"
    model, input_channels = model_for(name, torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    state = {
        "cycles": 0,
        "epochs": 0,
        "steps": 0,
        "elapsed_seconds": 0.0,
        "best_validation": None,
        "validation_no_improvement": 0,
    }
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("model") != name:
            raise ValueError(f"checkpoint model mismatch: {checkpoint_path}")
        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        state.update(checkpoint.get("continual", {}))
        print(f"resumed {name} from {checkpoint_path}", flush=True)
    else:
        initial_path = initial_root / INITIAL_CHECKPOINT_DIRS[name] / "model.pt"
        if initial_path.is_file():
            checkpoint = torch.load(initial_path, map_location="cpu")
            if checkpoint.get("model") != name and not (
                name == "unet" and checkpoint.get("model") == "unet"
            ):
                raise ValueError(f"initial checkpoint model mismatch: {initial_path}")
            model.load_state_dict(checkpoint["state_dict"])
            print(f"initialized {name} from {initial_path}", flush=True)
    return model, optimizer, input_channels, checkpoint_path, state


def save_checkpoint(
    name,
    model,
    optimizer,
    input_channels,
    checkpoint_path,
    state,
    train_scans,
    validation,
):
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model": name,
            "input_channels": input_channels,
            "grid_shape": [512, 512],
            "target": "NASA_ampl_normalized",
            "training": {
                "train_scans": train_scans,
                "validation_scans": state.get("validation_scans", 0),
                "mode": "continual_wall_clock",
            },
            "validation": validation,
            "continual": state,
        },
        checkpoint_path,
    )


def train_epoch(
    name, model, optimizer, input_channels, pool, args, rng, deadline, torch
):
    """Run one masked training cycle and return completion, losses, and steps."""
    from torch.nn import functional

    patches, patches_complete = make_patches(
        pool,
        args.patch_size,
        args.patches_per_scan,
        input_channels,
        rng,
        balanced=args.balanced_sampling,
        deadline=deadline,
    )
    if not patches:
        return False, [], 0
    model.train()
    losses = []
    steps = 0
    for start in range(0, len(patches), args.batch_size):
        if time.monotonic() >= deadline:
            return False, losses, steps
        batch = patches[start : start + args.batch_size]
        films = torch.from_numpy(np.stack([item[0] for item in batch]))
        targets = torch.from_numpy(np.stack([item[1] for item in batch]))
        masks = torch.from_numpy(np.stack([item[2] for item in batch]))
        output = torch.sigmoid(model(films))
        loss = functional.smooth_l1_loss(output[masks], targets[masks])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        steps += 1
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    print(
        f"{name}: completed pass over {len(pool)} scans, loss={mean_loss:.6f}",
        flush=True,
    )
    return patches_complete, losses, steps


def correlation(prediction, target):
    if prediction.size < 2 or np.std(prediction) == 0 or np.std(target) == 0:
        return 0.0
    return float(np.corrcoef(prediction, target)[0, 1])


def validate_model(model, input_channels, validation, args, deadline, torch):
    """Evaluate a model on validation references using macro image metrics."""
    if not validation:
        return None
    model.eval()
    per_scan = []
    with torch.no_grad():
        for start in range(0, len(validation), args.validation_batch_size):
            if time.monotonic() >= deadline:
                return None
            batch = validation[start : start + args.validation_batch_size]
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
                known = mask & np.isfinite(prediction) & np.isfinite(target)
                if known.sum() < 2:
                    continue
                predicted = prediction[known].astype(float)
                expected = target[known].astype(float)
                per_scan.append(
                    {
                        "mae": float(np.mean(np.abs(predicted - expected))),
                        "correlation": correlation(predicted, expected),
                    }
                )
    if not per_scan:
        return None
    return {
        "scans": len(per_scan),
        "macro_mae": float(np.mean([item["mae"] for item in per_scan])),
        "macro_correlation": float(np.mean([item["correlation"] for item in per_scan])),
    }


def validation_better(candidate, previous):
    if candidate is None:
        return False
    if previous is None:
        return True
    if candidate["macro_mae"] < previous["macro_mae"] - 1e-6:
        return True
    return (
        abs(candidate["macro_mae"] - previous["macro_mae"]) <= 1e-6
        and candidate["macro_correlation"] > previous["macro_correlation"]
    )


def replenish(args, batch_id: int, deadline: float):
    remaining = deadline - time.monotonic()
    if remaining < args.min_replenish_seconds:
        print(
            f"skipping replenishment: only {max(0.0, remaining):.1f}s remain "
            f"(reserve={args.min_replenish_seconds:.1f}s)",
            flush=True,
        )
        return None
    batch_root = args.batch_root / f"batch_{batch_id:04d}"
    env = None
    timeout = max(1.0, remaining - 5.0)
    candidates = args.candidate or sorted(
        (ROOT / "data/processed").glob("review_ranked*.csv")
    )
    if not [path for path in candidates if path.is_file()]:
        print(
            "no candidate CSVs available; continuing with the validated pool",
            flush=True,
        )
        return (None, None)
    command = [
        sys.executable,
        str(ROOT / "scripts/dataset/run_amplitude_batch.py"),
        "--size",
        str(args.replenish_size),
        "--seed",
        str(args.seed + batch_id),
        "--batch-id",
        str(batch_id),
        "--base-manifest",
        str(args.base_manifest),
        "--base-records",
        str(args.base_records),
        "--profile",
        str(args.profile),
        "--out-root",
        str(args.batch_root),
    ]
    for candidate in candidates:
        command.extend(("--candidates", str(candidate)))
    print("replenishing usable data: " + " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("replenishment timed out before the training deadline", flush=True)
        return None

    corpus = args.replenished_root / f"corpus_batch_{batch_id:04d}"
    targets = args.replenished_root / f"targets_batch_{batch_id:04d}"
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/training/prepare_phase6_usable_512.py"),
                "--dataset",
                str(batch_root / "dataset"),
                "--phase1",
                str(batch_root / "phase1_pairs" / "manifest.csv"),
                "--output",
                str(corpus),
            ],
            cwd=ROOT,
            check=True,
            timeout=max(1.0, deadline - time.monotonic() - 3.0),
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/training/prepare_phase6_512_image_targets.py"),
                "--corpus",
                str(corpus),
                "--output",
                str(targets),
            ],
            cwd=ROOT,
            check=True,
            timeout=max(1.0, deadline - time.monotonic() - 3.0),
        )
    except subprocess.TimeoutExpired:
        print(
            "replenishment preparation timed out before the training deadline",
            flush=True,
        )
        return None
    return corpus, targets


def existing_replenished(args):
    if args.corpus or not args.replenished_root.is_dir():
        return []
    pairs = []
    for corpus in sorted(args.replenished_root.glob("corpus_batch_*")):
        targets = args.replenished_root / corpus.name.replace("corpus_", "targets_", 1)
        if (corpus / "manifest.csv").is_file() and (targets / "manifest.csv").is_file():
            pairs.append((corpus, targets))
    return pairs


def parse_duration(args):
    values = [args.duration_hours, args.duration_minutes, args.duration_seconds]
    if sum(value is not None for value in values) != 1:
        raise SystemExit(
            "provide exactly one of --duration-hours, --duration-minutes, or --duration-seconds"
        )
    value = next(value for value in values if value is not None)
    multiplier = (
        3600
        if args.duration_hours is not None
        else 60
        if args.duration_minutes is not None
        else 1
    )
    seconds = float(value) * multiplier
    if seconds <= 0:
        raise SystemExit("duration must be positive")
    return seconds


def main():
    """Parse CLI options and continue training within the wall-clock budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument("--duration-hours", type=float)
    duration.add_argument("--duration-minutes", type=float)
    duration.add_argument("--duration-seconds", type=float)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(*MODEL_SPECS, "all"),
        default=["all"],
        help="model names to train, or all models by default",
    )
    parser.add_argument("--corpus", type=Path, action="append")
    parser.add_argument("--targets", type=Path, action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-root", type=Path, default=DEFAULT_INITIAL_ROOT)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument(
        "--replenished-root",
        type=Path,
        default=ROOT / "outputs/evaluation/phase6_continual_data",
    )
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--base-records", type=Path, default=DEFAULT_BASE_RECORDS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--replenish-size", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-scan", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-every", type=int, default=1)
    parser.add_argument(
        "--patience-cycles",
        type=int,
        default=5,
        help="stop after this many validation cycles without improvement; 0 disables",
    )
    parser.add_argument("--min-replenish-seconds", type=float, default=60.0)
    parser.add_argument(
        "--no-balanced-sampling",
        dest="balanced_sampling",
        action="store_false",
        help="visit each training scan once per epoch instead of balancing buckets",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--scan-limit", type=int)
    parser.add_argument("--no-replenish", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seconds = parse_duration(args)
    if "all" in args.models:
        args.models = list(MODEL_SPECS)
    args.models = list(dict.fromkeys(args.models))
    if (
        args.replenish_size < 1
        or args.patch_size < 1
        or args.patches_per_scan < 1
        or args.batch_size < 1
        or args.validation_batch_size < 1
        or args.validation_every < 1
        or args.patience_cycles < 0
        or args.min_replenish_seconds < 0
        or not 0.0 <= args.validation_fraction < 1.0
    ):
        raise SystemExit(
            "training, validation, and replenishment parameters are invalid"
        )
    if bool(args.corpus) != bool(args.targets):
        raise SystemExit("--corpus and --targets must be supplied together")
    if args.corpus:
        corpora = args.corpus
        targets = args.targets
    else:
        pairs = [(DEFAULT_CORPUS, DEFAULT_TARGETS), *existing_replenished(args)]
        corpora, targets = zip(*pairs)
        corpora, targets = list(corpora), list(targets)
    if len(corpora) != len(targets):
        raise SystemExit(
            "--corpus and --targets must be supplied the same number of times"
        )

    group_rows = read_group_rows(args.base_manifest)
    pool = []
    for corpus, target in zip(corpora, targets):
        pool.extend(usable_rows(corpus, target, args.scan_limit, group_rows))
    train_pool, validation = split_pool(pool, args.validation_fraction, args.seed)
    if not train_pool:
        raise SystemExit("validation split left no training pairs")
    print(f"models={','.join(args.models)} duration_seconds={seconds:.1f}", flush=True)
    print(
        f"training pool={len(train_pool)} validation pool={len(validation)} "
        f"balanced_sampling={args.balanced_sampling}",
        flush=True,
    )
    if args.dry_run:
        print("dry run: no model training or data generation performed", flush=True)
        return

    import torch

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    states = {}
    elapsed_bases = {}
    for name in args.models:
        states[name] = load_or_create(
            name, args.output_root, args.initial_root, args.learning_rate, torch
        )
        states[name][-1]["validation_scans"] = len(validation)
        elapsed_bases[name] = float(states[name][-1].get("elapsed_seconds", 0.0))
    deadline = time.monotonic() + seconds
    run_started = time.monotonic()
    cycle = max(int(state[-1].get("cycles", 0)) for state in states.values())
    batch_id = next_batch(args.batch_root)
    while time.monotonic() < deadline:
        cycle += 1
        cycle_complete = True
        offset = (cycle - 1) % len(args.models)
        ordered_models = args.models[offset:] + args.models[:offset]
        for name in ordered_models:
            if time.monotonic() >= deadline:
                cycle_complete = False
                break
            model, optimizer, input_channels, checkpoint_path, state = states[name]
            complete, losses, steps = train_epoch(
                name,
                model,
                optimizer,
                input_channels,
                train_pool,
                args,
                np.random.default_rng(
                    args.seed + cycle * 1000 + args.models.index(name)
                ),
                deadline,
                torch,
            )
            state.update(
                {
                    "cycles": cycle,
                    "epochs": int(state.get("epochs", 0)) + int(complete),
                    "steps": int(state.get("steps", 0)) + steps,
                    "elapsed_seconds": elapsed_bases[name]
                    + (time.monotonic() - run_started),
                    "last_loss": float(np.mean(losses)) if losses else None,
                }
            )
            save_checkpoint(
                name,
                model,
                optimizer,
                input_channels,
                checkpoint_path,
                state,
                len(train_pool),
                state.get("last_validation"),
            )
            if not complete:
                cycle_complete = False
                break
        if not cycle_complete or time.monotonic() >= deadline:
            break
        if args.validation_every and cycle % args.validation_every == 0 and validation:
            validation_complete = True
            for name in args.models:
                model, optimizer, input_channels, checkpoint_path, state = states[name]
                metrics = validate_model(
                    model, input_channels, validation, args, deadline, torch
                )
                if metrics is None:
                    validation_complete = False
                    break
                state["last_validation"] = metrics
                if validation_better(metrics, state.get("best_validation")):
                    state["best_validation"] = metrics
                    state["best_cycle"] = cycle
                    state["validation_no_improvement"] = 0
                    save_checkpoint(
                        name,
                        model,
                        optimizer,
                        input_channels,
                        checkpoint_path.parent / "best_model.pt",
                        state,
                        len(train_pool),
                        metrics,
                    )
                    print(
                        f"{name}: new best validation MAE={metrics['macro_mae']:.6f} "
                        f"corr={metrics['macro_correlation']:.4f}",
                        flush=True,
                    )
                else:
                    state["validation_no_improvement"] = (
                        int(state.get("validation_no_improvement", 0)) + 1
                    )
                save_checkpoint(
                    name,
                    model,
                    optimizer,
                    input_channels,
                    checkpoint_path,
                    state,
                    len(train_pool),
                    metrics,
                )
            if not validation_complete or time.monotonic() >= deadline:
                break
            if args.patience_cycles and all(
                int(states[name][-1].get("validation_no_improvement", 0))
                >= args.patience_cycles
                for name in args.models
            ):
                print(
                    "stopping: no model improved on validation within patience",
                    flush=True,
                )
                break
        if args.no_replenish:
            print("pool exhausted; --no-replenish requested, stopping", flush=True)
            break
        replenished = replenish(args, batch_id, deadline)
        if replenished is None:
            break
        if replenished == (None, None):
            continue
        corpus, target = replenished
        batch_id += 1
        pool.extend(usable_rows(corpus, target, args.scan_limit, group_rows))
        train_pool, validation = split_pool(pool, args.validation_fraction, args.seed)
        for name in args.models:
            states[name][-1]["validation_scans"] = len(validation)
        print(
            f"added replenished pool; training scans={len(train_pool)} "
            f"validation scans={len(validation)}",
            flush=True,
        )
    print(
        f"finished after {seconds - max(0.0, deadline - time.monotonic()):.1f}s and {cycle} cycles",
        flush=True,
    )


if __name__ == "__main__":
    main()
