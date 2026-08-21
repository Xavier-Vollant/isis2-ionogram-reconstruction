"""Fit a film scan's vertical calibration against a matched NASA ionogram.

Two numbers in the warp are invisible to the film itself: how many kilometres
lie between adjacent height rulings, and which row means zero kilometres. Both
have to come from somewhere outside the scan, and a matched NASA array is the
only thing available that knows.

The objective is mutual information, not correlation. Film density and receiver
amplitude are different physical quantities related by an unknown and possibly
non-monotone curve, and MI is the similarity measure that does not care which
curve it is. Both arrays are detrended first, so the film's rulings and both
archives' marker lines drop out and what remains to align is the echo.

`trace_offset` is deliberately not the objective. Keeping the acceptance test
separate from the thing being optimised is what stops a fit from grading its own
homework.
"""

from __future__ import annotations

import numpy as np

from ..extraction.echo import detrend, extract
from . import film as geometry


def mutual_information(film, nasa, valid, bins=64):
    """Shared information between two images of unrelated physical quantities."""
    if valid.sum() < 1000:
        return -np.inf
    joint, _, _ = np.histogram2d(film[valid], nasa[valid], bins=bins)
    joint = joint / joint.sum()
    marginal = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / marginal[nonzero])))


def objective(image, base, zero_row, px_per_km, line_time, v_height, nasa, window):
    """-> (score, warped) for one candidate vertical calibration."""
    warped = geometry.resample(
        image, base.with_vertical(zero_row, px_per_km), line_time, v_height
    )
    film = detrend(warped)
    valid = np.isfinite(film) & window
    return mutual_information(film, nasa, valid), warped


def evaluation_window(v_height, shape, low=200.0, high=2200.0):
    """A fixed band, so the score compares alignment and not how much each
    candidate happens to cover."""
    return np.broadcast_to(((v_height >= low) & (v_height <= high))[None, :], shape)


def surface(image, base, line_time, v_height, nasa, window, scales, offsets):
    """-> the objective over a grid of (px_per_km, zero_row).

    Returned whole rather than reduced to its argmax: the two parameters trade
    off against each other, and whether the peak is a point or a ridge is the
    difference between a calibration and a coincidence.
    """
    scores = np.empty((len(scales), len(offsets)))
    for i, px_per_km in enumerate(scales):
        for j, zero_row in enumerate(offsets):
            scores[i, j], _ = objective(
                image, base, zero_row, px_per_km, line_time, v_height, nasa, window
            )
    return scores


def search(image, base, line_time, v_height, nasa, window, scales, offsets):
    scores = surface(image, base, line_time, v_height, nasa, window, scales, offsets)
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    return float(scores[i, j]), float(offsets[j]), float(scales[i])


def trace_offset(warped, ampl, freq, v_height, low, high, floor=3.0, ceiling=2500.0):
    """Median film-minus-NASA height of the two independently extracted traces."""
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
