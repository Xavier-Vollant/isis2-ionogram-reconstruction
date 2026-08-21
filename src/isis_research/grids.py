"""The common 64x96 evaluation grid, and the resamplers that land on it.

Lifted verbatim from ``scripts/benchmark_signal_detectors.py`` and
``scripts/benchmark_amplitude_workflows.py`` so the pipeline no longer imports
its constants out of a benchmark script.  The two resamplers are kept separate
on purpose: they differ in how they treat points outside the source support and
in their weight denominators, so they are not interchangeable.
"""

from __future__ import annotations

import numpy as np

from isis_research import ionogram

TARGET_HEIGHT = np.linspace(0.0, 3000.0, 64)
TARGET_FREQUENCY = np.linspace(0.1, 9.5, 96)

# The amplitude path spells the same two axes differently.
HEIGHT = TARGET_HEIGHT
FREQUENCY = TARGET_FREQUENCY


def resample_grid(values, source_y, source_x, fill_value=0.0):
    """Bilinearly resample a sorted grid onto the common 64x96 grid."""

    source_y = np.asarray(source_y, dtype=float).ravel()
    source_x = np.asarray(source_x, dtype=float).ravel()
    values = np.asarray(values, dtype=float)
    y_order = np.argsort(source_y)
    x_order = np.argsort(source_x)
    source_y = source_y[y_order]
    source_x = source_x[x_order]
    values = values[np.ix_(y_order, x_order)]
    y_index = np.clip(
        np.searchsorted(source_y, TARGET_HEIGHT, side="right") - 1,
        0,
        len(source_y) - 2,
    )
    x_index = np.clip(
        np.searchsorted(source_x, TARGET_FREQUENCY, side="right") - 1,
        0,
        len(source_x) - 2,
    )
    y_weight = (TARGET_HEIGHT - source_y[y_index]) / np.maximum(
        source_y[y_index + 1] - source_y[y_index], 1e-12
    )
    x_weight = (TARGET_FREQUENCY - source_x[x_index]) / np.maximum(
        source_x[x_index + 1] - source_x[x_index], 1e-12
    )
    top_left = values[np.ix_(y_index, x_index)]
    top_right = values[np.ix_(y_index, x_index + 1)]
    bottom_left = values[np.ix_(y_index + 1, x_index)]
    bottom_right = values[np.ix_(y_index + 1, x_index + 1)]
    wy = y_weight[:, None]
    wx = x_weight[None, :]
    output = (
        top_left * (1.0 - wy) * (1.0 - wx)
        + top_right * (1.0 - wy) * wx
        + bottom_left * wy * (1.0 - wx)
        + bottom_right * wy * wx
    )
    outside = (
        (TARGET_HEIGHT[:, None] < source_y[0])
        | (TARGET_HEIGHT[:, None] > source_y[-1])
        | (TARGET_FREQUENCY[None, :] < source_x[0])
        | (TARGET_FREQUENCY[None, :] > source_x[-1])
    )
    return np.where(outside, fill_value, output)


def _resample_grid(values, source_y, source_x, fill_value=0.0):
    """Bilinearly resample a sorted 2-D grid onto HEIGHT x FREQUENCY."""
    source_y = np.asarray(source_y, dtype=float)
    source_x = np.asarray(source_x, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(source_y) < 2 or len(source_x) < 2:
        raise ValueError("source grid has too few points")
    y_index = np.clip(
        np.searchsorted(source_y, HEIGHT, side="right") - 1, 0, len(source_y) - 2
    )
    x_index = np.clip(
        np.searchsorted(source_x, FREQUENCY, side="right") - 1, 0, len(source_x) - 2
    )
    y_weight = (HEIGHT - source_y[y_index]) / (
        source_y[y_index + 1] - source_y[y_index]
    )
    x_weight = (FREQUENCY - source_x[x_index]) / (
        source_x[x_index + 1] - source_x[x_index]
    )
    top_left = values[np.ix_(y_index, x_index)]
    top_right = values[np.ix_(y_index, x_index + 1)]
    bottom_left = values[np.ix_(y_index + 1, x_index)]
    bottom_right = values[np.ix_(y_index + 1, x_index + 1)]
    wy = y_weight[:, None]
    wx = x_weight[None, :]
    output = (
        top_left * (1.0 - wy) * (1.0 - wx)
        + top_right * (1.0 - wy) * wx
        + bottom_left * wy * (1.0 - wx)
        + bottom_right * wy * wx
    )
    outside = (
        (HEIGHT[:, None] < source_y[0])
        | (HEIGHT[:, None] > source_y[-1])
        | (FREQUENCY[None, :] < source_x[0])
        | (FREQUENCY[None, :] > source_x[-1])
    )
    return np.where(outside, fill_value, output)


def load_film(path):
    """Read a validated artifact and put its film signal on the common grid."""
    scan = ionogram.read_validated(path)
    film = 1.0 - scan.intensity
    valid = scan.valid_mask
    height = scan.virtual_height_km
    frequency = scan.frequency_mhz
    film = resample_grid(film, height, frequency)
    mask = resample_grid(valid.astype(float), height, frequency) > 0.5
    return film.astype(np.float32), mask
