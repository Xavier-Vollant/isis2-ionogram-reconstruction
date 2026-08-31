# The pipeline, stage by stage

[`GETTING_STARTED.md`](GETTING_STARTED.md) shows the complete command. This
page explains its stages and outputs.

## Main path

```text
raw CSA image
  → structure detection
  → frequency calibration
  → height calibration
  → calibrated ionogram artifact
  → model prediction
  → CDF-like export
```

## Stages

| Stage | Code | Input → output | Main result |
|---|---|---|---|
| Load and detect | `extract_scan_structure.py`, `isis_research.registration` | grayscale image → film bounds, marker candidates, ruling lattice | `structure_overlay.png`, `structure.json` |
| Frequency axis | `fit_frequency_axis.py`, `route_calibration.py` | marker candidates + profile → MHz mapping | `frequency.json` |
| Height axis | `fit_height_axis.py` | ruling lattice + profile → virtual-height mapping | `height.json` |
| Warp and validate | `warp_calibrated_scan.py`, `ionogram.py` | image + two mappings → `isis.ionogram.v1` NPZ | warped image, mask, and artifact metadata |
| Infer amplitude | `infer_isis_model.py`, `isis_research.models` | validated artifact + checkpoint → prediction NPZ | model prediction and provenance |
| Export | `isis_research.nasa.model_cdf` | prediction + CSA axes → model-derived CDF-like file | CDF variables and `summary.json` |

The first four stages are deterministic for a new scan and do not require a
matching NASA CDF. They use the checked-in calibration profile, learned from a
CDF-assisted development batch. Model inference is the machine-learning
stage.

## Calibration routes

The pipeline can use a matching CDF when one is available, or use the
film-only profile without a local NASA file. The route and warnings are stored
in the artifact metadata. `review` and `not_usable` results remain visible;
they are not promoted to `usable`.

## The canonical artifact

`isis.ionogram.v1` contains:

- normalized film intensity in `(height, frequency)` orientation;
- a validity mask for pixels supported by the source film;
- the frequency axis in MHz;
- the virtual-height axis in kilometres;
- a status and confidence record; and
- processing provenance.

The fields and validation rules are defined in
[`product_contract.md`](product_contract.md).

## Visual inspection

Use `--diagnostics` with the one-scan runner. The static products show:

- detected film boundaries, marker candidates, and ruling candidates;
- the selected frequency and height mappings;
- the warped image; and
- the valid-pixel mask.

The evaluation scripts create static PNG, HTML, JSON, and CSV galleries when
the external corpus, targets, and checkpoints are available. They do not start
a web server.

## Training path

Training is separate from processing one new scan:

```text
CSA/NASA matching
  → landmark alignment
  → structure/frequency/height calibration
  → quality gate
  → amplitude-target packaging
  → model training and held-out evaluation
```

The commands and data requirements are in
[`DATA_AND_TRAINING.md`](DATA_AND_TRAINING.md). The scripts are grouped under
`scripts/dataset/`, `scripts/training/`, and `scripts/evaluation/`.

## Scientific boundary

The output is a calibrated digital ionogram and a model-derived amplitude
file. File compatibility does not prove measured-data equivalence, marker
identity, absolute-height accuracy, external generalization, or physical
inversion. The current evidence and open questions are recorded in
[`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md).
