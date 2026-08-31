# Changelog

## 0.1.0 — Public release preparation

Since the previous GitHub version, the repository has been cleaned up so that
someone new can install it, run the example, and understand what the results
mean.

- Made the one-scan workflow the clear supported path, with updated examples,
  notebooks, output descriptions, and limitations.
- Replaced the old KSH quick-start example with a stronger BUR example, along
  with its matching sample CDF and an explanation of how it was selected.
- Renamed the README heading to match the repository and replaced the BUR
  example with a new scan/CDF pair selected from a 200-case check.
- Added guides for the pipeline, commands, data preparation, file contracts,
  notebooks, and common problems.
- Documented the three included inference checkpoints and the training,
  evaluation, catalog, matching, and review tools.
- Added validation and failure-case tests around artifacts, calibration,
  dataset preparation, quality checks, and CDF export.
- Added GitHub Actions checks for formatting, linting, compilation, notebooks,
  tests, and the end-to-end example.
- Added the Apache 2.0 license, third-party data notices, acknowledgments, and
  citation information.

The exported amplitude is still a model prediction, not a measured NASA
observation, and the project does not produce electron-density profiles. The
full CSA/NASA archive and generated training corpus remain outside the
repository and can be rebuilt with the dataset tools.
