"""Reading NASA CDF amplitude onto a target grid.

Lifted verbatim from ``scripts/benchmark_native_resolution_models.py``
(``cdf_amplitude``, native axes) and ``scripts/benchmark_amplitude_workflows.py``
(``_cdf_sweep`` / ``_resample_cdf``, the fixed 64x96 grid).  ``_cdf_sweep``
carries the sweep-prefix policy pinned by ``tests/test_cdf_sweep_policy.py``.
"""

from __future__ import annotations

from pathlib import Path

import cdflib
import numpy as np

from isis_research.grids import FREQUENCY, HEIGHT, _resample_grid


def cdf_amplitude(path: Path, frequency: np.ndarray, height: np.ndarray):
    cdf = cdflib.CDF(str(path))
    amplitude = np.asarray(cdf.varget("ampl"), dtype=float)
    source_frequency = np.asarray(cdf.varget("freq"), dtype=float).ravel()
    source_height = np.asarray(cdf.varget("v_height"), dtype=float).ravel()
    valid = (
        np.isfinite(source_frequency)
        & (source_frequency > 0.0)
        & np.isfinite(amplitude).all(axis=1)
    )
    source_frequency = source_frequency[valid]
    amplitude = amplitude[valid]
    # Some CDFs have a fixed-frequency prefix before the swept-frequency
    # portion.  Cut only when a genuinely negative step marks that boundary;
    # taking argmin unconditionally discards a valid prefix on already
    # monotonic CDF axes.
    steps = np.diff(source_frequency)
    if np.any(steps < 0.0):
        start = int(np.argmin(steps) + 1)
        source_frequency = source_frequency[start:]
        amplitude = amplitude[start:]
    order = np.argsort(source_frequency)
    source_frequency = source_frequency[order]
    amplitude = amplitude[order]
    source_frequency, unique = np.unique(source_frequency, return_index=True)
    amplitude = amplitude[unique]
    height_order = np.argsort(source_height)
    source_height = source_height[height_order]
    amplitude = amplitude[:, height_order]

    by_frequency = np.empty((len(source_height), len(frequency)), dtype=float)
    for index, row in enumerate(amplitude.T):
        by_frequency[index] = np.interp(frequency, source_frequency, row)
    output = np.empty((len(height), len(frequency)), dtype=float)
    for index in range(len(frequency)):
        output[:, index] = np.interp(height, source_height, by_frequency[:, index])
    inside = (
        (frequency[None, :] >= source_frequency[0])
        & (frequency[None, :] <= source_frequency[-1])
        & (height[:, None] >= source_height[0])
        & (height[:, None] <= source_height[-1])
    )
    return np.where(inside, output, np.nan), inside


def _cdf_sweep(data):
    amplitude = np.asarray(data["nasa_amplitude"], dtype=float)
    frequency = np.asarray(data["freq"], dtype=float).ravel()
    heights = np.asarray(data["v_height"], dtype=float).ravel()
    valid = (
        np.isfinite(frequency) & (frequency > 0.0) & np.isfinite(amplitude).all(axis=1)
    )
    frequency = frequency[valid]
    amplitude = amplitude[valid]
    if len(frequency) < 2:
        raise ValueError("CDF has too few finite frequency samples")
    # NASA files may contain a fixed-frequency prefix before the sweep, which
    # shows up as a negative step. Cut there - but only if such a step exists.
    # argmin returns an index unconditionally, so on an already-monotonic axis
    # the old form cut at the *smallest positive* step instead, discarding a
    # healthy prefix. Measured over 10,255 readable CDFs: 21 targets recover
    # coverage, 20 of them Kashima, three of which were producing an entirely
    # empty target. No target loses coverage.
    steps = np.diff(frequency)
    if np.any(steps < 0):
        start = int(np.argmin(steps) + 1)
        frequency = frequency[start:]
        amplitude = amplitude[start:]
    order = np.argsort(frequency)
    frequency = frequency[order]
    amplitude = amplitude[order]
    frequency, unique = np.unique(frequency, return_index=True)
    amplitude = amplitude[unique]
    if len(frequency) < 2 or frequency[-1] <= frequency[0]:
        raise ValueError("CDF sweep is not increasing")
    return amplitude, frequency, heights


def _resample_cdf(data):
    amplitude, frequency, heights = _cdf_sweep(data)
    values = _resample_grid(amplitude.T / 255.0, heights, frequency, np.nan)
    valid = np.isfinite(values)
    return np.nan_to_num(values, nan=0.0).astype(np.float32), valid
