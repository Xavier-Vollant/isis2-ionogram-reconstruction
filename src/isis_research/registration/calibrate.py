"""Fit a film scan's vertical calibration against a matched NASA ionogram.

The fit estimates the pixels-per-kilometre scale and the row representing zero
virtual height. It uses detrended mutual information so film density and NASA
amplitude can be compared without assuming the same intensity scale.
"""

from __future__ import annotations

import numpy as np

from ..extraction.echo import detrend, extract
from . import film as geometry


def mutual_information(film, nasa, valid, bins=64):
    """Measure shared information between two image arrays."""
    if valid.sum() < 1000:
        return -np.inf
    joint, _, _ = np.histogram2d(film[valid], nasa[valid], bins=bins)
    joint = joint / joint.sum()
    marginal = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / marginal[nonzero])))


def objective(image, base, zero_row, px_per_km, line_time, v_height, nasa, window):
    """Return the score and warped image for one vertical calibration."""
    warped = geometry.resample(
        image, base.with_vertical(zero_row, px_per_km), line_time, v_height
    )
    film = detrend(warped)
    valid = np.isfinite(film) & window
    return mutual_information(film, nasa, valid), warped


def evaluation_window(v_height, shape, low=200.0, high=2200.0):
    """Return the fixed height band used for calibration scoring."""
    return np.broadcast_to(((v_height >= low) & (v_height <= high))[None, :], shape)


def surface(image, base, line_time, v_height, nasa, window, scales, offsets):
    """Evaluate the objective over candidate scales and zero rows."""
    scores = np.empty((len(scales), len(offsets)))
    for i, px_per_km in enumerate(scales):
        for j, zero_row in enumerate(offsets):
            scores[i, j], _ = objective(
                image, base, zero_row, px_per_km, line_time, v_height, nasa, window
            )
    return scores


def search(image, base, line_time, v_height, nasa, window, scales, offsets):
    """Find the best vertical calibration and return its parameters."""
    scores = surface(image, base, line_time, v_height, nasa, window, scales, offsets)
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    return float(scores[i, j]), float(offsets[j]), float(scales[i])


def trace_offset(warped, ampl, freq, v_height, low, high, floor=3.0, ceiling=2500.0):
    """Return the median film-minus-NASA height offset between traces."""
    keep = v_height <= ceiling
    heights = v_height[keep]
    film_path, film_confidence = extract(warped[:, keep], traces=1)[0]
    nasa_path, nasa_confidence = extract(ampl.astype(float)[:, keep], traces=1)[0]
    both = (
        np.isfinite(film_confidence)
        & (film_confidence >= floor)
        & np.isfinite(nasa_confidence)
        & (nasa_confidence >= floor)
        & (freq > low)
        & (freq < high)
    )
    if both.sum() < 10:
        return None
    difference = heights[film_path[both]] - heights[nasa_path[both]]
    step = v_height[1] - v_height[0]
    return {
        "n": int(both.sum()),
        "median_km": float(np.median(difference)),
        "median_bins": float(np.median(difference) / step),
        "within_1bin": float(np.mean(np.abs(difference) <= step)),
        "within_60km": float(np.mean(np.abs(difference) <= 60)),
        "differences_km": difference,
    }
