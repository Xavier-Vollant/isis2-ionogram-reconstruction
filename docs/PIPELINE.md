# The pipeline, stage by stage

The canonical path is deliberately short:

```text
raw CSA image
  → structure detection
  → frequency calibration
  → height calibration
  → calibrated ionogram artifact
  → model prediction
  → CDF-like export
```

## One-scan path

Run it from the repository root:

```sh
python scripts/pipeline/run_scan.py \
  data/raw/csa_verified_ksh_1972322002235.png \
  --output outputs/example --station KSH --diagnostics
```

The runner uses the checked-in profile and the historical best finalist,
`norm_residual_unet`, by default and expects the scan to be accepted by that
profile. The three finalist checkpoints and their training-only contrast
calibrations are registered in `configs/model_candidates.json`. Use
`--checkpoint`, `--profile`, `--pair-name`, or `--station` when working with a
different calibration, model, or filename convention.

The verified sample above is a matched CSA/NASA development example and is
known to complete the current pipeline as `usable`. Its NASA reference is
`data/samples/i2_av_ksh_1972322002235_v01.cdf`. The generated CDF is checked
for compatible structure and variables; its model amplitudes remain a
prediction, not a measured reconstruction.

The review-only sample frames can be inspected directly with:

```sh
python scripts/pipeline/extract_scan_structure.py \
  --film data/raw/csa_sample_1972176033200.png \
  --out-dir outputs/sample_structure
```

## Stages

| Stage | Code | Input → output | Inspection |
|---|---|---|---|
| Load and detect | `extract_scan_structure.py`, `isis_research.registration` | grayscale image → film bounds, marker candidates, ruling lattice | `diagnostics/structure_overlay.png`, `structure.json` |
| Frequency axis | `fit_frequency_axis.py`, `route_calibration.py` | marker candidates + profile → MHz mapping | `diagnostics/frequency.json` |
| Height axis | `fit_height_axis.py` | ruling lattice + profile → km mapping | `diagnostics/height.json` |
| Warp and validate | `warp_calibrated_scan.py`, `ionogram.py` | image + two mappings → `isis.ionogram.v1` NPZ | `diagnostics/warped.png`, `valid_mask.png`, `warp.json` |
| Infer amplitude | `infer_isis_model.py`, `isis_research.models` | validated artifact + finalist checkpoint → calibrated prediction NPZ | prediction array and metadata |
| Export | `isis_research.nasa.model_cdf` | prediction + CSA axes → model-derived CDF-like file | `summary.json`, CDF variables and provenance |

The first four stages are deterministic and do not require a NASA CDF for the
new scan. The profile is a checked-in summary learned from a CDF-assisted
development batch. The final prediction is the only machine-learning stage.

The canonical artifact stores normalized film brightness in
`(height, frequency)` orientation, a validity mask, both physical axes, a
status, and provenance. A `review` or `not_usable` result is not silently
promoted to a usable artifact.

## Visual inspection

`--diagnostics` writes static products for a single scan. The structure image
shows detected boundaries, marker candidates, and ruling candidates. The
frequency and height JSON files expose the selected profile and mapping
breakpoints. The warped image and mask show what survives the geometric
correction and how much of the target grid is supported by the source film.

The evaluation scripts under `scripts/evaluation/` create static PNG, HTML,
JSON, and CSV galleries when the external corpus, targets, and checkpoints are
available. They do not start a web server.

## Research and training path

The training side is intentionally separate from the one-scan path:

```text
CSA/NASA matching
  → landmark alignment
  → structure/frequency/height calibration
  → quality gate
  → amplitude-target packaging
  → model training and held-out evaluation
```

The relevant scripts are grouped under `scripts/dataset/`,
`scripts/training/`, and `scripts/evaluation/`. They need the external raw
archive and generated manifests that are not part of this repository. The
scripts remain because they document how the profile and model were produced;
they are not required to understand or run one new scan.

## Current scientific boundary

The output is a calibrated digital ionogram and a model-derived amplitude
file. The project does not yet establish marker identity, absolute-height
accuracy, robust external generalization, or physical electron-density
reconstruction. Those claims require expert-reviewed references and should not
be inferred from file compatibility or fit residuals alone.
