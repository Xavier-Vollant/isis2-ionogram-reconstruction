# Project handoff

This page records the current research status. Start with
[`README.md`](README.md) for commands and [`docs/PIPELINE.md`](docs/PIPELINE.md)
for the processing stages.

## Current result

The project can:

- standardized and checked for usable film structure;
- calibrated onto frequency and virtual-height axes;
- warped into a validated `isis.ionogram.v1` artifact;
- passed through the checked-in image models; and
- exported as a model-derived NASA-CDF-like amplitude file.

The verified KSH example completes this path as `usable`. Calibration is
deterministic for a given scan and profile. The amplitude stage uses a
checked-in image-model checkpoint.

## Implemented features

- structure detection and diagnostic overlays;
- film-only frequency and virtual-height calibration;
- CDF-assisted calibration when a matching reference is available;
- regular-grid warping with an explicit support mask;
- validated `isis.ionogram.v1` artifacts;
- three registered model checkpoints;
- one-scan model inference;
- model-derived CDF-like export with explicit unknown metadata;
- CSA/NASA catalog construction and candidate matching;
- candidate download, validation, and static review;
- training-target preparation and model training; and
- held-out evaluation and static comparison tools.

## What still needs scientific validation

- absolute height accuracy has no independent expert-reviewed gold set;
- marker fit residuals do not prove marker identity;
- the quality gate measures echo agreement, not landmark correctness;
- model evaluation is mainly internal and reel-disjoint; and
- wider sweep formats and station changes may reduce performance.

Useful next steps are an expert-reviewed reference set for marker identity,
trace geometry, mode identity, and absolute height; calibration-error
measurement before expanding the model; uncertainty and abstention rules; and
an external reel/station evaluation set.

## Important limits

A film scan cannot recover satellite orbit, magnetic coordinates, instrument
mode, or other pass metadata absent from the image. NASA-style names and CDF
structure do not make the generated amplitude measured data or byte-identical
to a NASA CDF. Electron-density profiles are outside the current scope.

The full CSA/NASA archive, matched manifests, training corpus, and most
generated reports are external or reproducible outputs rather than checked-in
files. See [`docs/DATA_AND_TRAINING.md`](docs/DATA_AND_TRAINING.md) for how to
rebuild the data side.
