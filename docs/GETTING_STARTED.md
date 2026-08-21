# Getting started

This is the shortest useful path through the project: install it, run the
included verified scan, and inspect the files it creates.

## 1. Install

Use Python 3.12 from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

## 2. Run the example

The repository includes a small CSA scan that is known to pass the current
quality gate:

```sh
python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_ksh_1972322002235.png \
  --output outputs/example \
  --station KSH \
  --diagnostics
```

The command runs structure detection, frequency calibration, height
calibration, warping, model inference, and CDF-like export.

It uses the default model registered in `configs/model_candidates.json`. To
try another included finalist, add for example `--model hybrid_unet`. A custom
checkpoint can still be supplied with `--checkpoint`; the named `--model`
takes precedence when both are provided.

## 3. Inspect the result

The output directory contains:

- `summary.json` — a compact record of the run and its physical axis ranges;
- an `isis.ionogram.v1` artifact — the calibrated image, axes, mask, and
  provenance;
- a model prediction NPZ file;
- a model-derived CDF-like file; and
- `diagnostics/` — static images and JSON records when `--diagnostics` is
  enabled.

The result is a prediction based on the CSA scan. It is not a measured NASA
amplitude observation. See [the project overview](OVERVIEW.md) for the terms
used here and [the pipeline guide](PIPELINE.md) for the stages in detail.

## 4. Open the notebooks

Start Jupyter from the repository root:

```sh
python -m jupyter lab
```

Open the notebooks in this order:

1. `01_end_to_end_pipeline.ipynb` — build and inspect one calibrated artifact;
2. `02_structure_and_calibration.ipynb` — investigate geometry and axes; and
3. `03_reconstruction_and_cdf.ipynb` — compare the three model predictions.

The notebook guide explains what each one expects and produces.

## If you have your own scan

Replace the example PNG in the command with your file. Keep the output in a
new directory so that results from different scans do not overwrite one
another. Use `--pair-name` when the filename does not contain the observation
identity expected by the CDF exporter.

If the scan is rejected, rerun with `--diagnostics` and inspect the reported
status and the generated structure and calibration files. The pipeline keeps
`review` and `not_usable` results visible; it does not silently turn them into
usable training data.
