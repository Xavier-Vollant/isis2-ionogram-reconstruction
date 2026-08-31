# Command reference

This page lists the runnable commands in the repository. Run them from the
repository root:

~~~sh
cd "/path/to/FINAL ISIS PUBLIC"
source .venv/bin/activate
~~~

The project requires Python 3.12. Install the complete development and
notebook environment once:

~~~sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
~~~

On Windows, activate the environment with .venv\Scripts\activate instead.

Every command supports `--help`:

~~~sh
python path/to/command.py --help
~~~

Use it to see the options for the installed version.

## Start here

### Process one CSA scan

This is the supported end-to-end workflow. It detects scan structure,
calibrates the physical axes, creates a validated artifact, runs the default
model, and exports a model-derived CDF-like file.

~~~sh
python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_bur_1973077111224.png \
  --output outputs/example \
  --station BUR \
  --diagnostics
~~~

Use another registered model with --model hybrid_unet,
--model norm_residual_unet, or --model contrast_aware_norm_residual. Use
--pair-name when the raw filename does not contain the observation identity
needed by the exported CDF header.

The output directory contains summary.json, a validated isis.ionogram.v1 NPZ
artifact, a prediction NPZ, a model-derived CDF-like file, and diagnostics
when --diagnostics is enabled.

### Run the notebooks

Start Jupyter from the repository root:

~~~sh
python -m jupyter lab
~~~

Open the notebooks in this order:

1. 01_end_to_end_pipeline.ipynb — build and inspect a calibrated artifact.
2. 02_structure_and_calibration.ipynb — inspect geometry and physical axes.
3. 03_reconstruction_and_cdf.ipynb — compare NASA and model CDF outputs.

To execute a notebook without opening Jupyter:

~~~sh
MPLCONFIGDIR=/tmp/final-isis-mpl \
python -m nbconvert --to notebook --execute \
  notebooks/03_reconstruction_and_cdf.ipynb \
  --output /tmp/final-isis-notebook-03.ipynb
~~~

On macOS, setting MPLCONFIGDIR to a writable directory avoids the
FigureCanvasAgg and Matplotlib-cache warnings.

### Run the checks

~~~sh
python -m pytest -p no:cacheprovider -q
python -m compileall -q src scripts tests
ruff format --check src scripts tests
ruff check src scripts tests
~~~

The test command disables pytest's cache plugin so the run stays local and
repeatable. Ruff checks style and may still report findings in existing test
files.

## Complete command index

The index includes all 35 runnable Python scripts.

| Area | Command |
| --- | --- |
| Dataset | scripts/dataset/align_landmarks.py |
| Dataset | scripts/dataset/align_phase1_pairs.py |
| Dataset | scripts/dataset/build_calibration_profile.py |
| Dataset | scripts/dataset/build_candidate_matches.py |
| Dataset | scripts/dataset/build_csa_master.py |
| Dataset | scripts/dataset/build_nasa_master.py |
| Dataset | scripts/dataset/download_candidate_data.py |
| Dataset | scripts/dataset/package_amplitude_dataset.py |
| Dataset | scripts/dataset/prepare_phase1_pairs_target.py |
| Dataset | scripts/dataset/quality_gate.py |
| Dataset | scripts/dataset/run_amplitude_batch.py |
| Pipeline | scripts/pipeline/export_model_as_cdf.py |
| Pipeline | scripts/pipeline/extract_scan_structure.py |
| Pipeline | scripts/pipeline/fit_frequency_axis.py |
| Pipeline | scripts/pipeline/fit_height_axis.py |
| Pipeline | scripts/pipeline/infer_isis_model.py |
| Pipeline | scripts/pipeline/route_calibration.py |
| Pipeline | scripts/pipeline/run_scan.py |
| Pipeline | scripts/pipeline/standardize_film_only_512.py |
| Pipeline | scripts/pipeline/warp_calibrated_scan.py |
| Evaluation | scripts/evaluation/benchmark_phase6_512_image_baselines.py |
| Evaluation | scripts/evaluation/evaluate_phase6_512_image_model.py |
| Evaluation | scripts/evaluation/evaluate_phase6_all_models.py |
| Evaluation | scripts/evaluation/evaluate_phase6_deep.py |
| Evaluation | scripts/evaluation/evaluate_phase6_final_candidates.py |
| Evaluation | scripts/evaluation/render_phase6_512_image_gallery.py |
| Evaluation | scripts/evaluation/render_phase6_deep_gallery.py |
| Experiments | scripts/experiments/run_phase6_all_models_contrast_test.py |
| Experiments | scripts/experiments/run_phase6_experiment_ab.py |
| Training | scripts/training/continual_train_phase6_models.py |
| Training | scripts/training/prepare_phase6_512_image_targets.py |
| Training | scripts/training/prepare_phase6_usable_512.py |
| Training | scripts/training/train_phase6_512_image_model.py |
| Training | scripts/training/train_phase6_512_image_model_patches.py |
| Training | scripts/training/train_phase6_contrast_aware_norm_residual.py |

## Dataset commands

These commands build metadata, match CSA and NASA records, download files,
align pairs, calibrate batches, and package training data. Most require
generated metadata or network access and should be run from the repository
root.

### Build the CSA master catalog

Purpose: build the CSA metadata table from the available source inventories.

~~~sh
python scripts/dataset/build_csa_master.py
~~~

Useful options:

- --refresh downloads the CSA inventories again.
- --skip-acquire uses inventories already present locally.

The result is written under data/processed. Use --skip-acquire when testing
parsing without network access.

### Build the NASA master catalog

Purpose: discover and parse NASA ISIS-2 pass-header metadata.

~~~sh
python scripts/dataset/build_nasa_master.py \
  --limit-stations 1 \
  --skip-acquire
~~~

Useful options:

- --workers N controls download/parse concurrency.
- --rediscover walks the NASA directory tree again.
- --limit-stations N limits a connectivity or parser test.
- --skip-acquire parses the existing cache without crawling.

### Build candidate CSA-NASA matches

Purpose: rank metadata-based CSA/NASA pairs. These are candidate matches, not
confirmed pixel-level scientific matches.

~~~sh
python scripts/dataset/build_candidate_matches.py \
  --limit 1000 \
  --strategy reel
~~~

Use --strategy station for equal station selection, or --yield-from PATH to
account for the survival rate of a previous aligned batch.

### Download candidate data

Purpose: download and validate CSA PNGs and NASA CDFs referenced by a
candidate CSV.

~~~sh
python scripts/dataset/download_candidate_data.py \
  --candidates data/processed/candidate_matches_top1000_reel.csv \
  --limit 1000
~~~

Useful options:

- --start N resumes from a later candidate.
- --workers N controls parallel downloads.
- --refresh replaces cached files.
- --output PATH changes the review output location.

Downloads are cached under data/raw/matches.

### Align one film and CDF pair

Purpose: detect and label the physical landmarks shared by one CSA film and
one NASA CDF.

~~~sh
python scripts/dataset/align_landmarks.py \
  --film data/raw/csa_verified_bur_1973077111224.png \
  --cdf data/samples/i2_av_bur_1973077111224_v01.cdf \
  --out /tmp/final-isis-landmarks.png \
  --fast
~~~

Use --consensus PATH for a cross-scan ruling consensus, --marker-sigma N to
change marker sensitivity, and omit --fast when expensive mutual-information
refinement is wanted. The command creates the requested PNG plus companion
JSON and NPZ files with the same base name.

### Align a pair manifest

Purpose: run the landmark aligner over many pairs.

~~~sh
python scripts/dataset/align_phase1_pairs.py \
  --pairs outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --out outputs/amplitude_batches/batch_0001/landmarks \
  --workers 2
~~~

Use --reuse PATH to reuse existing landmark directories.

### Build a calibration profile

Purpose: learn and validate the film-only calibration profile from an
existing CDF-assisted landmark batch.

~~~sh
python scripts/dataset/build_calibration_profile.py \
  --batch outputs/amplitude_batches/batch_0001/landmarks \
  --pairs outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --out configs/film_calibration_profile.json
~~~

Use --held-out-fraction and --seed for a reproducible validation split.

### Select a larger pair target

Purpose: choose a diverse set of candidate pairs without copying scan files.

~~~sh
python scripts/dataset/prepare_phase1_pairs_target.py \
  --candidates data/processed/review_ranked_candidate_matches_top1000_reel.csv \
  --target 100 \
  --seed 7 \
  --out outputs/phase1_target
~~~

Use --existing, --records, --landmarks, and --used-manifest to exclude
already-used pairs. Use --fresh-only to write only newly selected pairs.

### Run one amplitude batch

Purpose: run alignment, structure detection, calibration, warping, quality
gating, and packaging for a small CSA/NASA dataset batch.

~~~sh
python scripts/dataset/run_amplitude_batch.py \
  --size 50 \
  --seed 7 \
  --batch-id batch_0001 \
  --candidates data/processed/review_ranked_candidate_matches_top1000_reel.csv
~~~

Use --out-root to select the output root, --profile to select a calibration
profile, and --base-manifest or --base-records to extend an existing batch.

### Quality-gate a batch

Purpose: audit existing structure, frequency, height, warp, and comparison
sidecars and select the best route for each pair.

~~~sh
python scripts/dataset/quality_gate.py \
  --phase1-manifest outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --structure-dir outputs/amplitude_batches/batch_0001/structure \
  --frequency-dir outputs/amplitude_batches/batch_0001/frequency \
  --height-dir outputs/amplitude_batches/batch_0001/height \
  --warp-dir outputs/amplitude_batches/batch_0001/warp \
  --out-dir outputs/amplitude_batches/batch_0001/quality
~~~

Use --limit N for a small audit.

### Package the amplitude dataset

Purpose: package quality-gated CSA tensors with resampled NASA amplitude
targets.

~~~sh
python scripts/dataset/package_amplitude_dataset.py \
  --final outputs/amplitude_batches/batch_0001/quality/final.csv \
  --pairs outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --warp outputs/amplitude_batches/batch_0001/warp \
  --out outputs/amplitude_batches/batch_0001/dataset
~~~

Use --status review to package the review band separately. Adjust
--min-target-valid-fraction and --min-usable-pairs when changing quality
requirements.

## Pipeline commands

These commands expose the individual stages used by run_scan.py. They are
useful for debugging or for building a batch manually.

### Choose a calibration route

~~~sh
python scripts/pipeline/route_calibration.py \
  --film data/raw/csa_verified_bur_1973077111224.png \
  --profile configs/film_calibration_profile.json \
  --out /tmp/final-isis-route.json
~~~

Add --cdf PATH, --metadata PATH, or --cdf-dir PATH when matching reference
data is available. This command chooses a route; it does not warp the image.

### Extract scan structure

~~~sh
python scripts/pipeline/extract_scan_structure.py \
  --film data/raw/csa_verified_bur_1973077111224.png \
  --out-dir /tmp/final-isis-structure \
  --no-plots
~~~

Use --manifest PATH for a batch, --limit N for a small run, and --marker-sigma
or --ruling-sigma to tune detection sensitivity.

### Fit the frequency axis

Single-pair example:

~~~sh
python scripts/pipeline/fit_frequency_axis.py \
  --structure /tmp/final-isis-structure/structure.json \
  --cdf data/samples/i2_av_bur_1973077111224_v01.cdf \
  --out /tmp/final-isis-frequency.json
~~~

Use --profile PATH for film-only calibration. Batch mode uses --batch together
with --phase1-manifest, --phase1-records, --structure-dir, --landmark-dir,
and --out-dir.

### Fit the height axis

~~~sh
python scripts/pipeline/fit_height_axis.py \
  --phase1-manifest outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --structure-dir outputs/amplitude_batches/batch_0001/structure \
  --frequency-dir outputs/amplitude_batches/batch_0001/frequency \
  --out-dir outputs/amplitude_batches/batch_0001/height \
  --route both
~~~

Use --profile PATH for the film-only profile and --limit N for a small batch.

### Warp calibrated scans

~~~sh
python scripts/pipeline/warp_calibrated_scan.py \
  --manifest outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --structure-dir outputs/amplitude_batches/batch_0001/structure \
  --frequency-dir outputs/amplitude_batches/batch_0001/frequency \
  --height-dir outputs/amplitude_batches/batch_0001/height \
  --out-dir outputs/amplitude_batches/batch_0001/warp \
  --route both \
  --no-plots
~~~

Use --frequency-bins and --height-bins to change the regular grid.

### Standardize film-only 512x512 input

Purpose: create the native 512x512 film-only corpus used for model training.

~~~sh
python scripts/pipeline/standardize_film_only_512.py \
  --film data/raw/csa_verified_bur_1973077111224.png \
  --output outputs/film_only_512
~~~

For a batch, use one of --film-list or --film-dir. Only scans classified
usable are written for later model use.

### Run model inference

~~~sh
python scripts/pipeline/infer_isis_model.py \
  outputs/notebooks/01_scan.npz \
  --model norm_residual_unet \
  --output outputs/example/prediction.npz
~~~

Use --model to select a registered model, --checkpoint PATH for a custom
checkpoint, or --uncalibrated to skip contrast calibration.

### Export a model prediction as CDF-like output

~~~sh
python scripts/pipeline/export_model_as_cdf.py \
  outputs/example/prediction.npz \
  --csa outputs/notebooks/01_scan.npz \
  --pair-name i2_av_bur_1973077111224_v01 \
  --station BUR \
  --output outputs/example/model.cdf \
  --scale unit
~~~

Use --scale byte when a byte-scaled amplitude is required. The exported
amplitude is a model prediction, not a measured NASA observation.

### Run the complete single-scan pipeline

~~~sh
python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_bur_1973077111224.png \
  --output outputs/example \
  --model norm_residual_unet \
  --station BUR \
  --diagnostics
~~~

The positional argument is the raw CSA PNG. --output is required and should
be a new directory for each scan.

## Evaluation commands

Evaluation commands require a prepared 512x512 corpus and NASA image targets.
They score held-out scans; NASA targets are references for scoring, not model
inputs during inference.

### Benchmark non-learned baselines

~~~sh
python scripts/evaluation/benchmark_phase6_512_image_baselines.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/evaluation/baselines \
  --limit-held-out 16
~~~

### Evaluate one image model

~~~sh
python scripts/evaluation/evaluate_phase6_512_image_model.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --checkpoint outputs/amplitude_batches/batch_0001/model/model.pt \
  --output outputs/evaluation/model \
  --limit-held-out 16 \
  --batch-size 4
~~~

### Evaluate all stored models

~~~sh
python scripts/evaluation/evaluate_phase6_all_models.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/evaluation/all_models \
  --model-limit 3 \
  --seed 7
~~~

Use --groups and --checkpoints to select model groups or checkpoint
registries.

### Run deep evaluation

~~~sh
python scripts/evaluation/evaluate_phase6_deep.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/evaluation/deep \
  --limit 16 \
  --resume
~~~

Use repeated --checkpoint NAME=PATH options to evaluate explicit checkpoints.
--checkpoint-every N saves progress during a long run.

### Evaluate final candidates

~~~sh
python scripts/evaluation/evaluate_phase6_final_candidates.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/evaluation/final_candidates \
  --limit 16 \
  --seed 7 \
  --resume
~~~

Use --only NAME to evaluate only selected checkpoint names.

### Render a native-resolution gallery

~~~sh
python scripts/evaluation/render_phase6_512_image_gallery.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output-dir outputs/evaluation/gallery \
  --count 10 \
  --seed 7
~~~

Use --page N to render a later page and --checkpoint PATH for a specific
model.

### Render a deep-evaluation gallery

~~~sh
python scripts/evaluation/render_phase6_deep_gallery.py \
  --report outputs/evaluation/deep/report.json \
  --model norm_residual_unet \
  --output-dir outputs/evaluation/deep_gallery \
  --count 10 \
  --seed 7
~~~

Use --report PATH from a completed deep evaluation or provide the corpus,
targets, and checkpoint directly.

## Experiment commands

These commands are inference or comparison experiments. They keep production
checkpoints read-only and write separate reports.

### Compare contrast calibration on all models

~~~sh
python scripts/experiments/run_phase6_all_models_contrast_test.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/experiments/contrast_test
~~~

Use --benchmark PATH and --checkpoints PATH to select existing benchmark and
checkpoint registries.

### Run the marker and contrast A/B test

~~~sh
python scripts/experiments/run_phase6_experiment_ab.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/experiments/ab_test \
  --marker-strength 1.0
~~~

Use --benchmark PATH to reuse a fixed held-out benchmark and --checkpoint PATH
to select the Hybrid U-Net checkpoint.

## Training commands

Training can be expensive and needs a prepared corpus and target directory.
Each command writes to the output path; use a new directory for each run.

### Prepare the usable 512x512 corpus

~~~sh
python scripts/training/prepare_phase6_usable_512.py \
  --dataset outputs/amplitude_batches/batch_0001/dataset \
  --phase1 outputs/amplitude_batches/batch_0001/phase1_pairs/manifest.csv \
  --output outputs/amplitude_batches/batch_0001/phase6_corpus \
  --resume
~~~

Use --limit N for a small preparation test.

### Prepare direct NASA image targets

~~~sh
python scripts/training/prepare_phase6_512_image_targets.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --output outputs/amplitude_batches/batch_0001/phase6_targets \
  --resume
~~~

Use --limit N for a small preparation test.

### Train a basic image model

~~~sh
python scripts/training/train_phase6_512_image_model.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/training/basic \
  --model cnn_2d \
  --train-limit 8 \
  --epochs 1 \
  --seed 7
~~~

Use unet or cnn_2d with this command. This is a smoke-training path, not a
full research training run.

### Train a patch-based 512x512 model

~~~sh
python scripts/training/train_phase6_512_image_model_patches.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output outputs/training/patches \
  --model norm_residual_unet \
  --train-limit 8 \
  --epochs 1 \
  --patch-size 64 \
  --patches-per-scan 2 \
  --batch-size 2 \
  --seed 7
~~~

Adjust --patch-size, --patches-per-scan, --batch-size, and --learning-rate for
a real experiment.

### Fine-tune the contrast-aware model

~~~sh
python scripts/training/train_phase6_contrast_aware_norm_residual.py \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --initial models/phase6_norm_residual_unet.pt \
  --output outputs/training/contrast_aware \
  --epochs 1 \
  --patch-size 64 \
  --patches-per-scan 2 \
  --batch-size 2 \
  --seed 7
~~~

The checked-in initial checkpoint is read-only. This command writes an
isolated candidate checkpoint.

### Continue training within a time budget

~~~sh
python scripts/training/continual_train_phase6_models.py \
  --duration-minutes 5 \
  --models norm_residual_unet \
  --corpus outputs/amplitude_batches/batch_0001/phase6_corpus \
  --targets outputs/amplitude_batches/batch_0001/phase6_targets \
  --output-root outputs/training/continual \
  --dry-run
~~~

Remove --dry-run to train. Exactly one of --duration-hours,
--duration-minutes, or --duration-seconds is required. Use --no-replenish to
prevent creation of new data batches during the run.

## Development and inspection commands

### Show all options for any command

~~~sh
MPLCONFIGDIR=/tmp/final-isis-mpl \
python scripts/pipeline/run_scan.py --help
~~~

The same pattern works for every script in the command index.

### Run the test suite

~~~sh
pytest -q
~~~

Run one test file or one test:

~~~sh
pytest -q tests/test_run_scan.py
pytest -q tests/test_run_scan.py::test_run_scan_connects_the_three_product_stages
~~~

### Compile all Python files

~~~sh
python -m compileall -q src scripts tests
~~~

### Run the linter

~~~sh
ruff check src scripts tests
~~~

## Common problems

### Matplotlib says that FigureCanvasAgg is non-interactive

This is normally a display warning, not a data-processing failure. The
notebooks save static figures and can run without an interactive GUI. Set a
writable cache directory when launching commands:

~~~sh
MPLCONFIGDIR=/tmp/final-isis-mpl python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_bur_1973077111224.png \
  --output outputs/example
~~~

### A command cannot find a file

Run it from the repository root and check the paths in its --help output. The
dataset, training, and evaluation commands require generated files that are
not included in a fresh checkout.

### A scan is marked review or not_usable

Rerun the single-scan command with --diagnostics, then inspect the generated
structure and calibration products. Do not use a review or not_usable result
as training data unless the quality policy explicitly allows it.

### The output already exists

Choose a new output directory. Commands intentionally avoid silently
overwriting important artifacts.

## Verification status

This page is maintained against the runnable scripts in `scripts/`. When a
command changes, check its `--help` output and update this page if its purpose,
arguments, or outputs change.

For this checkout, the following checks were completed:

- all 35 runnable scripts passed --help;
- Python compilation passed for src, scripts, and tests;
- the complete test suite passed with `python -m pytest -p no:cacheprovider -q`:
  162 tests;
- the checked-in scan passed run_scan.py and produced a usable artifact,
  prediction, CDF, summary, and diagnostics;
- the standalone inference, CDF export, landmark alignment, frequency-axis
  fitting, and film-only standardization examples produced their expected
  temporary outputs.

The large catalog, batch, evaluation, experiment, and training examples also
passed their command-interface checks. Their full workflows require generated
corpora, network downloads, or long-running training data that are not part of
the small checked-in sample.
