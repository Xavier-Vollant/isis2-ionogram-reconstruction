# Notebooks

The notebooks are visual companions to the command-line pipeline. Install the
project, then open Jupyter from the repository root:

```sh
python -m pip install -e '.[dev,notebooks]'
python -m jupyter lab
```

Run them in this order:

| Notebook | Use it for | Main result |
|---|---|---|
| `01_end_to_end_pipeline.ipynb` | Follow the verified CSA scan through loading, structure detection, calibration, and warping | calibrated `isis.ionogram.v1` artifact |
| `02_structure_and_calibration.ipynb` | Inspect boundaries, marker candidates, ruling lines, and physical axes | inspection figures and calibration records |
| `03_reconstruction_and_cdf.ipynb` | Run the three registered models and compare their CDF-like exports with the reference | prediction figures, amplitude metrics, and structural CDF comparison |

The notebooks use the checked-in matched pair by default:

- CSA: `data/raw/csa_verified_bur_1973077111224.png`
- NASA reference: `data/samples/i2_av_bur_1973077111224_v01.cdf`

Notebook 03 compares structure and shared variables, then resamples the NASA
amplitude grid onto each model grid. It shows NASA, model, and absolute-error
figures with overlap-aware MAE, RMSE, and correlation metrics. Predicted
amplitudes are not expected to equal measured NASA amplitudes.

The saved comparison figures are written to
`outputs/notebooks/03_cdf_compare_<model>.png`.

The notebooks call the same functions as the scripts. They do not contain a
second pipeline implementation. For a non-interactive run, use
[`scripts/pipeline/run_scan.py`](../scripts/pipeline/run_scan.py) as described
in [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md).

If a figure window cannot open, the notebooks and diagnostics still write
static figures. See [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md) for
the `FigureCanvasAgg` warning.
