# Notebooks

The notebooks are visual companions to the command-line pipeline. Install the
project and open Jupyter from the repository root:

```sh
python -m pip install -e '.[dev,notebooks]'
python -m jupyter lab
```

Run them in this order:

| Notebook | Use it for | Main result |
|---|---|---|
| `01_end_to_end_pipeline.ipynb` | Follow the verified CSA scan through loading, structure detection, calibration, and warping | calibrated `isis.ionogram.v1` artifact |
| `02_structure_and_calibration.ipynb` | Investigate boundaries, marker candidates, ruling lines, and physical axes when a scan looks wrong | inspection figures and calibration records |
| `03_reconstruction_and_cdf.ipynb` | Run the three registered finalist models and compare their CDF-like exports with the reference | prediction figures and structural CDF comparison |

The notebooks use the included verified pair by default:

- CSA: `data/raw/csa_verified_ksh_1972322002235.png`
- NASA reference: `data/samples/i2_av_ksh_1972322002235_v01.cdf`

Notebook 03 compares structure and shared variables. Its predicted amplitudes
are not expected to equal the measured NASA amplitudes.

The notebooks call the same functions as the scripts; they do not contain a
second implementation of the pipeline. For a non-interactive run, use
[`scripts/pipeline/run_scan.py`](../scripts/pipeline/run_scan.py) as described
in [`docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md).

If a figure window cannot open, static figures are still written by the
notebooks and diagnostics. See [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md)
for the `FigureCanvasAgg` warning.
