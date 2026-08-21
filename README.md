# FINAL ISIS

This proof of concept turns a scanned ISIS-2 CSA ionogram into a calibrated
digital ionogram, predicts a NASA-like amplitude image, and writes a
NASA-CDF-like output. The predicted amplitude is not a measured NASA
observation.

## Start here

Install the project, test tools, and notebook tools:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
```

The dependency versions are pinned for reproducible development. Four small
raw CSA sample scans are included in `data/raw/`; the full archive is not.

Run the complete one-scan path on the included verified usable example:

```sh
python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_ksh_1972322002235.png \
  --output outputs/example --station KSH --diagnostics
```

The command uses `configs/film_calibration_profile.json` and the historical
best finalist, `norm_residual_unet`, by default. The three registered
finalists and their training-only contrast calibrations are listed in
`configs/model_candidates.json`; select another with `--checkpoint`. The
command writes a validated ionogram artifact, model prediction, model-derived
CDF, `summary.json`, and—when `--diagnostics` is enabled—static PNG/JSON
inspection products under `outputs/example/diagnostics/`.

The included verified pair is known to produce a `usable` 512×512 artifact,
model prediction, and model-derived CDF. Its matching NASA reference is in
`data/samples/` for structural/content comparison; the predicted amplitudes
are not expected to equal the measured NASA amplitudes.

To inspect one of the review-only frames immediately, run the structure stage
directly:

```sh
python scripts/pipeline/extract_scan_structure.py \
  --film data/raw/csa_sample_1972176033200.png \
  --out-dir outputs/sample_structure
```

See `data/raw/README.md` for the status of all included frames.

The same stages are explained in [docs/PIPELINE.md](docs/PIPELINE.md), and
the artifact contract is in
[`docs/product_contract.md`](docs/product_contract.md).
The current state and limitations are in [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).

## Pipeline at a glance

```text
raw CSA image
  → structure detection
  → frequency and virtual-height calibration
  → calibrated 512×512 ionogram artifact
  → three finalist image-model amplitude predictions
  → model-derived NASA-CDF-like file
```

The canonical intermediate artifact is `isis.ionogram.v1`, implemented in
`src/isis_research/ionogram.py`. Its array orientation is always
`(height, frequency)` and it carries its axes, validity mask, status, and
provenance.

## Repository map

- `scripts/pipeline/` — the one-scan path and its reusable stage scripts.
- `src/isis_research/` — reusable image, geometry, artifact, NASA-CDF, and
  model code.
- `scripts/dataset/` — CSA/NASA matching, calibration-profile construction,
  quality gates, and training-corpus packaging.
- `scripts/training/` — target preparation and model training.
- `scripts/evaluation/` — baselines, scoring, and static galleries.
- `scripts/experiments/` — retained alternative experiments, outside the main
  path.
- `notebooks/` — visual explanations of the main workflow and research work.
- `configs/`, `models/`, `data/samples/` — the small checked-in runtime and
  reference inputs.

NASA CDFs are used to build calibration profiles, prepare training targets,
and evaluate predictions. They are not required to create a new CSA-only
output. Only a few CDF samples are checked in; the full archive and generated
corpora remain external.

## Checks

```sh
pytest -q
python -m compileall -q src scripts tests
```

All visual comparison output is static. No local web server is required.
