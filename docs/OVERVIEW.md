# What this project does

FINAL ISIS reads scanned ISIS-2 ionograms from the Canadian Space Agency
(CSA). It calibrates a scan, saves a validated digital ionogram, and can use
an image model to estimate an amplitude image and export it in a
NASA-CDF-like format.

There are two workflows:

1. **Process one scan.** Start with a CSA PNG and run the supported pipeline.
2. **Build and study models.** Match CSA scans with NASA observations, prepare
   targets, train models, and evaluate them on held-out data.

The first workflow is the normal starting point. The second needs a larger
dataset than the examples included here.

## Words used in the project

| Term | Meaning here |
|---|---|
| **CSA scan** | A PNG of an ISIS-2 ionogram printed on film and scanned. |
| **NASA reference CDF** | A measured NASA ionogram in CDF format. It can support matching, calibration, training, or comparison. |
| **Calibration** | Estimating the frequency and virtual-height axes represented by the film. |
| **Ionogram artifact** | The validated intermediate file containing the image, axes, support mask, status, and processing history. |
| **Model prediction** | The amplitude image estimated by an included image model. |
| **Model-derived CDF-like file** | A NASA-like CDF file built from the CSA axes and model prediction. Its amplitude is not a measured NASA value. |
| **Training corpus** | Matched and processed CSA/NASA examples used for training or evaluation. |
| **Usable** | The artifact has enough valid support for the next stage. |
| **Review** | The artifact was produced, but its quality or calibration needs inspection. |
| **Not usable** | The pipeline could not produce a trustworthy artifact for the next stage. |

## What the output means

The stored image orientation is `(height, frequency)`. The artifact also stores
the frequency and virtual-height coordinates, so the image is not just an
unlabelled 512×512 array.

The model output is a prediction from the CSA scan. A matching CDF layout or
variable name does not make it measured data. Electron-density profiles are
outside the current scope.

## Future work

The model-derived file is only NASA-CDF-like today. A future goal is to make
its signal, axes, variables, metadata, and missing-value handling close enough
to measured NASA CDFs that TOPIST can process it in the same way as a NASA
observation. That will require validation with TOPIST, not just checking that
the file opens or has matching variable names.

## Other retained research utilities

The one-scan path does not call every module. The remaining research tools
cover:

- echo and trace extraction in `src/isis_research/extraction/`;
- signal-occupancy detection in `src/isis_research/signal_detection.py`;
- labelling and dataset helpers in `src/isis_research/labeling/`;
- NASA CDF reading, comparison, station metadata, and model export in
  `src/isis_research/nasa/`; and
- alternative training, evaluation, and contrast experiments under
  `scripts/training/`, `scripts/evaluation/`, and `scripts/experiments/`.

They are separate from the normal one-scan workflow.

## What is included

The repository includes the processing pipeline, three inference checkpoints,
small CSA and NASA examples, notebooks, catalog and matching tools, training
and evaluation scripts, and tests.

The complete historical CSA/NASA archive and generated training corpus are not
checked in. They can be downloaded or rebuilt with the dataset tools when a
larger training or evaluation run is needed.

For commands, start with [`README.md`](../README.md). For artifact fields and
validation rules, see [`product_contract.md`](product_contract.md).
