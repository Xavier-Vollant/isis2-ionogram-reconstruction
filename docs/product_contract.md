# Product contract

This page defines the files produced by the pipeline and the measurements used
to judge them. It describes the interface, not a claim of scientific
validation.

## Calibrated ionogram artifact

One CSA film scan produces a quality-gated `isis.ionogram.v1` artifact:

| Field | Meaning |
|---|---|
| `warped_intensity` | normalized film brightness in [0, 1], shaped `(height, frequency)` |
| `valid_mask` | false wherever the source film does not support a target pixel |
| `frequency_mhz` | strictly increasing frequency axis |
| `virtual_height_km` | strictly increasing virtual-height axis |
| `meta_json.orientation` | always `height,frequency` |
| `meta_json.status` | `usable`, `review`, or `not_usable` |
| `meta_json.route` | `film_only` or `cdf_assisted` |
| `meta_json.confidence` | route confidence |
| `meta_json.support` | axis bounds checked against the stored axes |
| `meta_json.source` | CSA scan identity and, when available, NASA identity |
| `meta_json.provenance` | producer and processing information |

The contract is implemented in `src/isis_research/ionogram.py`. `write` rejects
malformed artifacts. `read_validated` rejects non-canonical orientation,
missing metadata or masks, invalid scale, and inconsistent axes. The reader
also understands historical layouts for audit and migration.

Film traces are dark, so signal-positive consumers use
`1 - warped_intensity`. Storage remains `(height, frequency)`; consumers
transpose only at their own boundaries.

## Model-derived CDF-like output

The downstream file is a model-derived NASA-CDF-like amplitude file. Its axes,
validity mask, and signal come from the calibrated artifact and model output.
Fields unavailable from a film scan are explicit unknowns. It can be compared
with a measured NASA CDF, but it is not the same measurement. Electron-density
profiles are outside the current scope.

## Acceptance metrics

Keep these families separate. A good frequency fit must not hide a wrong
height zero.

| Family | Metric | Status |
|---|---|---|
| Marker identity | fraction of selected markers that are the correct frequency | needs expert-reviewed gold labels |
| Trace geometry | trace height error, continuity, and mode identity | needs expert-reviewed gold labels |
| Height | zero-row, scale, local variation, and fixed-height error before alignment | needs independent references |
| Coverage | `valid_mask.mean()` and support width | computable now |
| Abstention | error among claimed outputs versus fraction claimed, by status band | computable now |

Report by reel and station with uncertainty intervals. Do not subtract an
offset before scoring height; that measures shape agreement, not absolute
height accuracy.

The existing `usable`/`review`/`not_usable` gate measures echo agreement, not
landmark correctness. Marker identity, trace geometry, and absolute height
remain unmeasured until an expert-reviewed reference set exists.

## Current boundary

The current product is a calibrated digital ionogram plus a model-derived
amplitude file. It does not establish external generalization, absolute-height
accuracy, marker identity, or physical inversion. Those claims require
references beyond file compatibility and fit residuals.
