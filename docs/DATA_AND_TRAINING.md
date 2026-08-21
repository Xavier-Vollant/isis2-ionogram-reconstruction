# Data, matching, and training

The one-scan workflow uses the checked-in calibration profile and model. The
training workflow is separate because it needs a much larger collection of
matched CSA scans and NASA CDFs.

## Build the catalog tables

Run these commands from the repository root. They download or parse metadata;
they do not download every image and CDF.

```sh
python scripts/dataset/build_csa_master.py
python scripts/dataset/build_nasa_master.py
python scripts/dataset/build_candidate_matches.py \
  --limit 1000 --strategy reel
```

The builders write `csa_master.csv`, `nasa_master.csv`, and a ranked candidate
crosswalk under `data/processed/`. The candidate rows are metadata matches,
not confirmed scientific matches.

For a small connectivity test, NASA discovery can be limited with
`--limit-stations 1`. Use `--skip-acquire` when the source inventories are
already cached.

## Download and review candidate pairs

Download the files behind a candidate table with:

```sh
python scripts/dataset/download_candidate_data.py \
  --candidates data/processed/candidate_matches_top1000_reel.csv \
  --limit 1000
```

The downloader caches CSA PNGs and versioned NASA CDFs under
`data/raw/matches/`, validates that both files decode, and writes a static
review page plus a reviewed CSV under `data/processed/`.

The downloader is resumable. Run it again to reuse cached files, or add
`--refresh` when the cached files should be replaced.

## Prepare a training batch

Use a reviewed candidate table to create a small amplitude batch:

```sh
python scripts/dataset/run_amplitude_batch.py \
  --size 50 \
  --candidates data/processed/review_ranked_candidate_matches_top1000_reel.csv
```

This performs landmark alignment, structure detection, calibration, warping,
quality gating, and dataset packaging. A batch can contain `usable`, `review`,
and rejected cases; only the appropriate usable records should continue into
training.

## Prepare targets and train

For a batch whose output is at `outputs/amplitude_batches/batch_0001`, run:

```sh
python scripts/training/prepare_phase6_usable_512.py \
  --dataset outputs/amplitude_batches/batch_0001/dataset \
  --phase1 outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --output outputs/amplitude_batches/batch_0001/phase6_corpus

python scripts/training/prepare_phase6_512_image_targets.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --output outputs/amplitude_batches/batch_0001/phase6_targets

python scripts/training/train_phase6_512_image_model.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/amplitude_batches/batch_0001/model
```

The training script defaults to a short smoke run. Increase `--epochs` for a
real experiment. It writes a new checkpoint under the requested output
directory and does not overwrite the checked-in models.

## Evaluate a checkpoint

```sh
python scripts/evaluation/evaluate_phase6_512_image_model.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --checkpoint outputs/amplitude_batches/batch_0001/model/model.pt \
  --output outputs/amplitude_batches/batch_0001/evaluation
```

Evaluation uses held-out usable scans. NASA CDFs are references for training
and scoring; they are not inputs to inference on a new CSA scan.

## What is not included

The repository contains examples and the builders, but not the complete
historical CSA/NASA archive or generated training corpus. Rebuilding a large
corpus requires network access, storage, and time. The checked-in model files
are ready for inference and are not replaced by a short training smoke run.
