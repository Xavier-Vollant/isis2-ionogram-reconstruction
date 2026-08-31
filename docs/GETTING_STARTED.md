# Getting started

Install the project, run the included scan, and inspect the result.

## 1. Install

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

## 2. Run the example

The repository includes a CSA scan that passes the current quality gate:

```sh
python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_ksh_1972322002235.png \
  --output outputs/example \
  --station KSH \
  --diagnostics
```

The command detects the scan structure, calibrates both axes, warps the image,
runs model inference, and writes a CDF-like export.

It uses the default model in `configs/model_candidates.json`. To try another
included model, add `--model hybrid_unet`. You can pass a custom checkpoint
with `--checkpoint`; `--model` takes precedence when both are provided.

## 3. Inspect the result

The output directory contains:

- `summary.json` — a compact run record and axis ranges;
- an `isis.ionogram.v1` artifact — image, axes, mask, and provenance;
- a model prediction NPZ file;
- a model-derived CDF-like file; and
- `diagnostics/` — static images and JSON records when enabled.

The result is a prediction from the CSA scan, not a measured NASA amplitude.
See [the project overview](OVERVIEW.md) for the terms and
[the pipeline guide](PIPELINE.md) for the processing stages.

## 4. Open the notebooks

Start Jupyter from the repository root:

```sh
python -m jupyter lab
```

Open the notebooks in this order:

1. `01_end_to_end_pipeline.ipynb` — build and inspect an artifact;
2. `02_structure_and_calibration.ipynb` — inspect geometry and axes; and
3. `03_reconstruction_and_cdf.ipynb` — compare the model predictions.

The notebook guide explains what each one expects and produces.

## If you have your own scan

Replace the example PNG in the command with your file. Keep the output in a
new directory so that results from different scans do not overwrite one
another. Use `--pair-name` when the filename does not contain the observation
identity expected by the CDF exporter.

If the scan is rejected, rerun with `--diagnostics` and inspect the status,
structure overlay, calibrated axes, warped image, and validity mask. The
pipeline keeps `review` and `not_usable` results visible; it does not silently
turn them into training data.
