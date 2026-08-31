# FINAL ISIS

Convert a scanned Canadian Space Agency (CSA) ISIS-2 ionogram into a
calibrated digital ionogram. The included image models can then estimate an
amplitude image and export it in a NASA-CDF-like format.

This is research software. The exported amplitude is a model prediction, not a
measured NASA observation. The project does not recover electron-density
profiles.

## Quick start

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'

python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_bur_1973077231124.png \
  --output outputs/example \
  --station BUR \
  --diagnostics
```

The command writes a calibrated `isis.ionogram.v1` artifact, a model
prediction, a model-derived CDF-like file, `summary.json`, and optional
diagnostic images.

## Read next

- [Understand the project](docs/OVERVIEW.md)
- [Run a scan and inspect its output](docs/GETTING_STARTED.md)
- [See every command](docs/COMMANDS.md)
- [Follow the pipeline](docs/PIPELINE.md)
- [Use the notebooks](notebooks/README.md)
- [Build data and train models](docs/DATA_AND_TRAINING.md)
- [Read the file contracts](docs/product_contract.md)
- [Troubleshoot a run](docs/TROUBLESHOOTING.md)
- [Read the research status](PROJECT_HANDOFF.md)
- [See what changed](CHANGELOG.md)
- [Acknowledgments](ACKNOWLEDGMENTS.md)

## What is included

This repository contains:

- the single-scan calibration, warping, inference, and export pipeline;
- three model checkpoints for inference;
- small CSA and NASA sample files;
- three notebooks that show the main workflow;
- catalog, matching, download, and review tools;
- training, evaluation, and experiment scripts; and
- reusable Python modules and tests.

The complete historical CSA/NASA archive and generated training corpus are not
included. The dataset tools can rebuild them when a larger training or
evaluation run is needed.

## Repository map

```text
src/isis_research/   image, geometry, artifact, NASA, and model code
scripts/pipeline/    one-scan processing commands
scripts/dataset/     catalog, matching, download, and packaging commands
scripts/training/    dataset preparation and model training
scripts/evaluation/  scoring, comparison, and gallery commands
scripts/experiments/ alternative model experiments
notebooks/           visual walkthroughs
configs/             calibration and model settings
models/              inference checkpoints
data/                small examples and external-data placeholders
tests/               automated checks
```

## Checks

From the repository root:

```sh
python -m pytest -p no:cacheprovider -q
python -m compileall -q src scripts tests
ruff format --check src scripts tests
ruff check src scripts tests
```

For a problem not covered here, see
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md). Each command also
provides its current options with `--help`.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

The Apache License covers the code, notebooks, configuration, and model
checkpoints. The redistributed CSA scans under [`data/raw/`](data/raw/) and the
NASA CDF references under [`data/samples/`](data/samples/) keep their
publishers' terms; see [`NOTICE`](NOTICE).
