# Troubleshooting

## Imports fail in a notebook

Start Jupyter from the repository root and install the project into the active
environment:

```sh
python -m pip install -e '.[dev,notebooks]'
python -m jupyter lab
```

The notebooks also add the repository and `src/` to `sys.path`, but that does
not replace installing the dependencies.

## The notebook says that a file is missing

The example notebooks expect to be opened with the repository root as the
working directory. Confirm that these paths exist:

```text
data/raw/csa_verified_ksh_1972322002235.png
data/samples/i2_av_ksh_1972322002235_v01.cdf
configs/film_calibration_profile.json
configs/model_candidates.json
```

## Matplotlib says `FigureCanvasAgg is non-interactive`

This is a display warning, not a pipeline failure. It means the current
matplotlib backend cannot open a separate interactive window. The notebooks
and diagnostics save static images; inspect those images or use a Jupyter
backend that supports inline display.

## The scan is marked `review` or `not_usable`

That status is part of the result, not an exception to hide. Run the command
again with `--diagnostics` and inspect the structure overlay, calibrated axes,
warped image, and validity mask. The verified KSH example is the recommended
sanity check for a fresh installation.

## The prediction does not look like the NASA amplitude

The model produces an estimate from the CSA scan. The NASA CDF is a measured
reference used for comparison, not a value supplied to the inference step.
Different amplitudes do not by themselves mean that the CDF export failed.

## Training files are missing

Training needs a downloaded or rebuilt CSA/NASA corpus. Follow
[`DATA_AND_TRAINING.md`](DATA_AND_TRAINING.md) to build catalogs, download
candidate pairs, prepare a batch, and create training targets.

## The command cannot find a script or configuration file

Run commands from the repository root, or pass explicit paths with the
script's options. Most commands show their accepted paths with `--help`.
