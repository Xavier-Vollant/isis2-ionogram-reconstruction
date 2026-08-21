# ISIS-2 CSA film ionograms: project handoff

## Goal

The project tests whether scanned Canadian Space Agency ISIS-2 film
ionograms can be converted into useful digital ionograms. The current result
is a proof of concept: it calibrates the printed film geometry, predicts a
NASA-like amplitude image with an image model, and exports a CDF-like file.
The prediction is not a measured NASA observation.

## What currently works

The main path is `scripts/pipeline/run_scan.py`:

```text
raw CSA image
  → boundaries, marker candidates, and ruling lattice
  → frequency and virtual-height axes
  → calibrated 512×512 isis.ionogram.v1 artifact
  → image-model prediction
  → model-derived CDF-like output
```

The first three stages are deterministic and film-only. The amplitude stage
uses the checked-in image-model checkpoint. The exporter preserves the CSA
axes and validity mask, emits NASA-style variable names, and marks metadata
unavailable from a film scan as unknown.

Use `--diagnostics` with the runner to create static overlays and JSON records
for structure, frequency calibration, height calibration, and warping. This
makes the important transformations inspectable without a server.

The artifact contract lives in `src/isis_research/ionogram.py` and
`docs/product_contract.md`. Tests cover the contract, calibration helpers,
model naming, and the one-scan orchestration.

## NASA data and training

The checked-in calibration profile was learned from a CDF-assisted development
batch. NASA CDFs are used by the research side of the repository to:

- match CSA scans with NASA observations;
- align landmarks and build calibration profiles;
- create amplitude targets for training; and
- score predictions and render static comparisons.

The shipped image-training path predicts NASA's measured `ampl` image after it
has been put on the calibrated CSA grid. It does not use trace labels. The
full raw CSA archive, matched manifests, training corpus, and most generated
reports are intentionally external or reproducible outputs.

## Repository roles

- `scripts/pipeline/` — main processing stages and the single-scan runner.
- `src/isis_research/` — reusable geometry, image, artifact, NASA, and model
  modules.
- `scripts/dataset/` — matching and corpus preparation.
- `scripts/training/` — model-target preparation and training.
- `scripts/evaluation/` — scoring, baselines, and static galleries.
- `scripts/experiments/` — useful but non-canonical alternatives.
- `notebooks/` — researcher-facing visual walkthroughs.
- `configs/`, `models/`, `data/samples/` — small checked-in inputs.

## Status

Implemented:

- scan structure detection;
- film-only frequency and virtual-height calibration;
- regular-grid warping with an explicit support mask;
- validated `isis.ionogram.v1` artifacts;
- one-scan model inference;
- model-derived CDF-like export with explicit unknown metadata; and
- static evaluation and comparison tooling.

Partially validated:

- absolute height accuracy has no independent expert-reviewed gold set;
- marker fit residuals do not prove marker identity;
- the quality gate measures echo agreement, not landmark correctness;
- model evaluation is mainly internal and reel-disjoint; and
- wider sweep formats and station changes may reduce performance.

Future work:

- build a small gold set for marker identity, trace geometry, mode identity,
  and absolute height;
- measure calibration error before expanding the model;
- add calibrated uncertainty and abstention rules; and
- test an external reel/station set.

## Important limitations

A film scan cannot recover satellite orbit, magnetic coordinates, instrument
mode, or other pass metadata absent from the image. Matching NASA variable
names does not make the generated amplitude measured data or make it
byte-identical to a NASA CDF. The learned profile is a format-level summary,
not a guarantee that every scan is usable. `review` and `not_usable` results
must remain visible and must not silently enter training or published results.
