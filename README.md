# FINAL ISIS

FINAL ISIS turns a scanned Canadian Space Agency (CSA) ISIS-2 ionogram into a
calibrated digital ionogram. Its image models can then estimate an amplitude
image and export that estimate in a NASA-CDF-like format.

This is a research proof of concept. The exported amplitude is a model
prediction, not a measured NASA observation. The project does not currently
recover physical electron-density profiles.

## Start here

Install the project and run the included verified example:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'

python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_ksh_1972322002235.png \
  --output outputs/example \
  --station KSH \
  --diagnostics
```

The command runs the complete single-scan path and writes a calibrated
`isis.ionogram.v1` artifact, a model prediction, a model-derived CDF-like file,
`summary.json`, and static diagnostics.

For the explanation of the terms and the meaning of each output, read
[`docs/OVERVIEW.md`](docs/OVERVIEW.md). For the steps, expected files, and
notebook instructions, read [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

## Choose a task

- [Understand the project](docs/OVERVIEW.md)
- [Run one scan](docs/GETTING_STARTED.md)
- [Follow the pipeline stage by stage](docs/PIPELINE.md)
- [Use the notebooks](notebooks/README.md)
- [Build catalogs, download pairs, train, or evaluate](docs/DATA_AND_TRAINING.md)
- [Understand the artifact and CDF contracts](docs/product_contract.md)
- [Fix common setup and output problems](docs/TROUBLESHOOTING.md)
- [Read the current research status and limitations](PROJECT_HANDOFF.md)

## What is included

The repository contains:

- the single-scan calibration, warping, inference, and export pipeline;
- three checked-in finalist model checkpoints;
- CSA and NASA sample files;
- three explanatory notebooks;
- CSA/NASA catalog and candidate-matching builders;
- candidate download and static review tools;
- training-target, training, evaluation, and experiment scripts; and
- reusable Python modules and tests.

The complete historical CSA/NASA archive and generated training corpus are not
checked in. The dataset tools can download or rebuild them when a larger
training or evaluation run is needed.

## Repository map

```text
src/isis_research/  reusable image, geometry, artifact, NASA, and model code
scripts/pipeline/   one-scan processing stages and runner
scripts/dataset/    catalogs, matching, downloading, calibration, and packaging
scripts/training/   corpus/target preparation and model training
scripts/evaluation/ scoring, comparisons, and static galleries
scripts/experiments/retained alternative experiments
notebooks/          visual walkthroughs of the supported workflow
configs/            calibration and model registrations
models/             checked-in inference checkpoints
data/               small examples and external-data placeholders
tests/              automated checks
```

## Checks

From the repository root:

```sh
pytest -q
python -m compileall -q src scripts tests
```

For a problem that is not covered by the quick-start guide, begin with
[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) and then use the relevant
script's `--help` output.
