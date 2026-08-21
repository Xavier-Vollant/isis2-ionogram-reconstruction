# Notebooks

Run these from the repository root after installing the project and notebook
dependencies:

```sh
python -m pip install -e '.[dev,notebooks]'
```

1. `01_end_to_end_pipeline.ipynb` follows one raw CSA scan through the
   calibrated image artifact.
2. `02_structure_and_calibration.ipynb` focuses on boundaries, markers,
   rulings, physical axes, and diagnostic figures.
3. `03_reconstruction_and_cdf.ipynb` starts from a calibrated artifact and
   shows all three historical finalist predictions and CDF-like exports.

Four small raw CSA PNG examples are included in `data/raw/`. The notebooks
use the verified usable example `csa_verified_ksh_1972322002235.png` by
default. Notebook 03 compares its generated CDF with the matching reference
at `data/samples/i2_av_ksh_1972322002235_v01.cdf`.

Notebook 03 uses `configs/model_candidates.json`, including the saved
training-only contrast calibration for each finalist, so its outputs match
the calibrated comparison that was previously shown by the local viewer.

The notebooks call the functions used by the command-line scripts. They are
intended to explain and inspect the pipeline, not to contain a second copy of
the processing algorithms. For a non-interactive one-scan run, use
`scripts/pipeline/run_scan.py`.
