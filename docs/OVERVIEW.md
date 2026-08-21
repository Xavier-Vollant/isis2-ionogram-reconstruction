# What this project does

This project works with scanned ISIS-2 ionograms from the Canadian Space
Agency (CSA). It turns a film scan into a calibrated digital ionogram, uses an
image model to estimate an amplitude image, and can save that result in a
NASA-CDF-like file.

The project has two related uses:

1. **Process one scan.** Start with a CSA PNG and run the supported pipeline.
   The result includes the calibrated axes, the corrected image, diagnostics,
   and an optional model prediction.
2. **Build and study models.** Match CSA scans with NASA observations, prepare
   training targets, train a model, and evaluate it on held-out data.

The first use is the normal starting point. The second use needs a much larger
data collection than the small examples included in this repository.

## Words used in the project

| Term | Meaning here |
|---|---|
| **CSA scan** | A PNG image of an ISIS-2 ionogram printed on film and scanned. |
| **NASA reference CDF** | A measured NASA ionogram stored in CDF format. It can be used for matching, calibration, training, or comparison. |
| **Calibration** | Estimating the physical frequency and virtual-height axes represented by the film. |
| **Ionogram artifact** | The project's validated intermediate file. It stores the corrected image, axes, support mask, status, and processing history. |
| **Model prediction** | The amplitude image estimated from the calibrated CSA artifact by one of the included image models. |
| **Model-derived CDF-like file** | A CDF output built from the CSA axes and model prediction. It has a NASA-like structure, but its amplitude is a prediction rather than a measured NASA observation. |
| **Training corpus** | The matched and processed CSA/NASA examples used to train or evaluate a model. |
| **Usable** | The pipeline believes the artifact has enough valid support for the next stage. |
| **Review** | The artifact was produced, but its quality or calibration needs inspection. |
| **Not usable** | The pipeline could not produce a trustworthy artifact for the next stage. |

## What the output means

The pipeline preserves the image orientation as `(height, frequency)` and
records the real frequency and virtual-height coordinates instead of treating
the image as an unlabelled 512×512 array.

The model output is not a recovered NASA measurement. Matching the CDF file
layout or variable names does not make the prediction measured data, and this
project does not currently reconstruct electron-density profiles.

## Other retained research utilities

The quick-start path does not call every module in the repository. The other
capabilities are still present for research and follow-up work:

- echo and trace extraction under `src/isis_research/extraction/`;
- signal-occupancy detection in `src/isis_research/signal_detection.py`;
- labelling and dataset helpers under `src/isis_research/labeling/`;
- NASA CDF reading, comparison, station metadata, and model export under
  `src/isis_research/nasa/`; and
- alternative training, evaluation, and contrast experiments under
  `scripts/training/`, `scripts/evaluation/`, and `scripts/experiments/`.

These are documented as separate capabilities because they are useful for
research, but they are not extra steps in the normal one-scan workflow.

## What is included

The repository includes the supported processing pipeline, three inference
checkpoints, small CSA and NASA examples, notebooks, catalog/matching tools,
training scripts, evaluation scripts, and tests.

The complete historical CSA/NASA archive and generated training corpus are not
checked in. They can be downloaded or rebuilt with the dataset tools when a
larger training or evaluation run is needed.

For the recommended commands, start with the repository
[`README.md`](../README.md). For the exact artifact fields and validation rules,
see [`product_contract.md`](product_contract.md).
