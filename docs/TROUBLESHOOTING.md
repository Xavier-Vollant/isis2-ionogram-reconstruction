# Troubleshooting

## Imports fail in a notebook

Start Jupyter from the repository root and install the project in the active
environment:

```sh
python -m pip install -e '.[dev,notebooks]'
python -m jupyter lab
```

The notebooks add the repository and `src/` to `sys.path`, but you still need
to install the dependencies.

## The notebook says that a file is missing

The example notebooks expect to be opened with the repository root as the
working directory. Confirm that these paths exist:

```text
data/raw/csa_verified_bur_1973077231124.png
data/samples/i2_av_bur_1973077231124_v01.cdf
configs/film_calibration_profile.json
configs/model_candidates.json
```

## Matplotlib says `FigureCanvasAgg is non-interactive`

This is a display warning, not a pipeline failure. The current Matplotlib
backend cannot open a separate window. The notebooks and diagnostics save
static images; inspect those images or use an inline Jupyter backend.

## The scan is marked `review` or `not_usable`

That status is part of the result. Run the command again with `--diagnostics`
and inspect the structure overlay, calibrated axes, warped image, and validity
mask. Use the verified BUR example as a fresh-installation check.

## The prediction does not look like the NASA amplitude

The model estimates amplitude from the CSA scan. The NASA CDF is a measured
reference for comparison, not an input to inference. Different amplitudes do
not by themselves mean that export failed.

## Training files are missing

Training needs a downloaded or rebuilt CSA/NASA corpus. Follow
[`DATA_AND_TRAINING.md`](DATA_AND_TRAINING.md) to build catalogs, download
candidate pairs, prepare a batch, and create training targets.

## The command cannot find a script or configuration file

Run commands from the repository root, or pass explicit paths in the command's
options. Most commands show their accepted paths with `--help`.
