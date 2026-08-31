"""Find ionospheric echo traces in an ionogram array.

The extractor can process a warped CSA scan or a NASA array. It uses ridge
shape and continuity, not film density as amplitude. Reported confidence is
relative to the scan line's noise, not a calibrated signal strength.
"""

from __future__ import annotations

import numpy as np

# Zero-mean, so flat regions score zero however dense the film is there.
RIDGE_KERNEL = np.array([-1, -1, -1, 2, 3, 2, -1, -1, -1], dtype=float)
RIDGE_KERNEL -= RIDGE_KERNEL.mean()

# Every tuning constant below is expressed in BINS, which is only a physical
# quantity once the height axis is fixed. On the 64-bin, 0-3000 km benchmark
# grid one bin is 47.619 km, and the shipped defaults mean the kilometre values
# recorded here. `parameters_for` converts them back for any other axis.
#
# This matters more than it looks. `RIDGE_KERNEL` is a matched filter: 9 taps
# is a 429 km feature on the benchmark grid but a 53 km feature on a 223-bin
# native one, so an unconverted extractor searches for a different physical
# structure at every resolution. Measured on 199 clean-reference scans, running
# the native grid without converting the kernel costs more than converting the
# other three constants recovers - see docs/stage_i_trace_first.md.
REFERENCE_BIN_KM = 3000.0 / 63.0
KERNEL_WIDTH_KM = len(RIDGE_KERNEL) * REFERENCE_BIN_KM
CAP_KM = 20 * REFERENCE_BIN_KM
SMOOTHNESS_PER_KM = 0.5 / REFERENCE_BIN_KM
EXCLUDE_KM = 3 * REFERENCE_BIN_KM


def scaled_kernel(taps):
    """Return a zero-mean ridge kernel with `taps` samples."""
    taps = max(3, int(taps))
    if taps % 2 == 0:
        taps += 1
    if taps == len(RIDGE_KERNEL):
        return RIDGE_KERNEL.copy()
    resampled = np.interp(
        np.linspace(0.0, len(RIDGE_KERNEL) - 1.0, taps),
        np.arange(len(RIDGE_KERNEL), dtype=float),
        RIDGE_KERNEL,
    )
    return resampled - resampled.mean()


def parameters_for(height_km):
    """Return extractor settings scaled to a height axis in kilometres.

    The settings preserve the physical widths of the ridge kernel, transition
    cap, smoothness penalty, and exclusion band.
    """
    axis = np.asarray(height_km, dtype=float).ravel()
    if axis.size < 2:
        raise ValueError("a height axis needs at least two points")
    bin_km = abs(float(axis[-1] - axis[0])) / (axis.size - 1)
    if not np.isfinite(bin_km) or bin_km <= 0:
        raise ValueError("height axis has no positive spacing")
    return {
        "kernel": scaled_kernel(round(KERNEL_WIDTH_KM / bin_km)),
        "cap": max(1, round(CAP_KM / bin_km)),
        "smoothness": SMOOTHNESS_PER_KM * bin_km,
        "exclude": max(1, round(EXCLUDE_KM / bin_km)),
    }


def detrend(array):
    """Remove row and column background structure from an array."""
    values = np.where(np.isfinite(array), array, np.nan)
    values = values - np.nanmedian(values, axis=0, keepdims=True)
    return values - np.nanmedian(values, axis=1, keepdims=True)


def ridge_score(array, kernel=None):
    """Return per-line ridge scores and a mask of unsupported edge pixels."""
    kernel = RIDGE_KERNEL if kernel is None else np.asarray(kernel, dtype=float)
    values = detrend(array)
    missing = ~np.isfinite(values)
    response = np.apply_along_axis(
        lambda line: np.convolve(line, kernel, mode="same"),
        1,
        np.nan_to_num(values),
    )
    # Convolution zero-pads, so at the first and last bins the kernel sees a
    # step from nothing into the array - the strongest apparent ridge in the
    # image, and pure artifact. Those bins never carry a usable answer.
    edge = len(kernel) // 2
    missing[:, :edge] = True
    missing[:, -edge:] = True
    response[missing] = 0.0

    deviation = np.abs(response - np.median(response, axis=1, keepdims=True))
    noise = 1.4826 * np.median(deviation, axis=1, keepdims=True)
    floor = 0.5 * 1.4826 * np.median(np.abs(response - np.median(response)))
    return response / np.maximum(noise, max(floor, 1e-9)), missing


def _transition_cost(states, smoothness, cap):
    """Return the capped cost of moving between height bins."""
    steps = np.abs(np.subtract.outer(np.arange(states), np.arange(states)))
    return smoothness * np.minimum(steps, cap)


def best_path(score, smoothness, cap):
    """Find the highest-scoring continuous path through scan lines."""
    lines, states = score.shape
    transition = _transition_cost(states, smoothness, cap)
    cost = -score[0].copy()
    backtrack = np.zeros((lines, states), dtype=np.int16)
    for line in range(1, lines):
        total = cost[:, None] + transition
        backtrack[line] = np.argmin(total, axis=0)
        cost = total.min(axis=0) - score[line]

    path = np.zeros(lines, dtype=int)
    path[-1] = int(np.argmin(cost))
    for line in range(lines - 1, 0, -1):
        path[line - 1] = backtrack[line, path[line]]
    return path


def guided_score(array, probability, cnn_weight=1.0, kernel=None):
    """Add an occupancy probability to the ridge score."""
    score, missing = ridge_score(array, kernel)
    probability = np.asarray(probability, dtype=float)
    if probability.shape != score.shape:
        raise ValueError(
            f"probability shape {probability.shape} does not match array shape {score.shape}"
        )
    if not np.isfinite(cnn_weight) or cnn_weight < 0:
        raise ValueError("cnn_weight must be finite and non-negative")
    centred = 2.0 * np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0) - 1.0
    guided = score + float(cnn_weight) * centred
    missing = missing | ~np.isfinite(probability)
    guided[missing] = 0.0
    return guided, missing


def extract_guided(
    array,
    probability,
    traces=2,
    smoothness=0.5,
    cap=20,
    exclude=3,
    kernel=None,
    cnn_weight=1.0,
    forbid_missing=True,
):
    """Return traces selected from a ridge score plus occupancy guidance."""
    score, missing = guided_score(
        array, probability, cnn_weight=cnn_weight, kernel=kernel
    )
    # The ordinary extractor can return a path through an invalid edge state
    # and mark that point NaN afterwards. A strong CNN bonus can make that
    # zero-valued state attractive, so the candidate explicitly forbids it.
    # Keep weight zero bit-identical to `extract` for the regression control.
    if forbid_missing and cnn_weight > 0.0:
        score[missing] = -1.0e6
    found = []
    for _ in range(traces):
        path = best_path(score, smoothness, cap)
        lines = np.arange(len(path))
        confidence = score[lines, path].copy()
        confidence[missing[lines, path]] = np.nan
        found.append((path, confidence))
        for offset in range(-exclude, exclude + 1):
            score[lines, np.clip(path + offset, 0, score.shape[1] - 1)] = 0.0
    return found


def extract_cascade(
    array,
    probability,
    traces=3,
    smoothness=0.5,
    cap=20,
    exclude=3,
    kernel=None,
    cnn_weight=1.0,
    forbid_missing=True,
):
    """Return an ordered main path followed by residual paths."""
    score, missing = ridge_score(array, kernel)
    probability = np.asarray(probability, dtype=float)
    if probability.shape != score.shape:
        raise ValueError(
            f"probability shape {probability.shape} does not match array shape {score.shape}"
        )
    if not np.isfinite(cnn_weight) or cnn_weight < 0:
        raise ValueError("cnn_weight must be finite and non-negative")
    residual_probability = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0)
    found = []
    lines = np.arange(score.shape[0])
    for trace_index in range(traces):
        candidate = score.copy()
        if trace_index > 0 and cnn_weight > 0.0:
            candidate += float(cnn_weight) * (2.0 * residual_probability - 1.0)
            if forbid_missing:
                candidate[missing] = -1.0e6
        path = best_path(candidate, smoothness, cap)
        confidence = candidate[lines, path].copy()
        confidence[missing[lines, path]] = np.nan
        found.append((path, confidence))
        for offset in range(-exclude, exclude + 1):
            indices = np.clip(path + offset, 0, score.shape[1] - 1)
            score[lines, indices] = 0.0
            residual_probability[lines, indices] = 0.0
    return found


def extract(array, traces=2, smoothness=0.5, cap=20, exclude=3, kernel=None):
    """Return ordered `(path, confidence)` pairs, strongest trace first.

    Extractor settings are measured in bins; use `parameters_for` when the
    height axis is not the common evaluation grid.
    """
    score, missing = ridge_score(array, kernel)
    found = []
    for _ in range(traces):
        path = best_path(score, smoothness, cap)
        lines = np.arange(len(path))
        confidence = score[lines, path].copy()
        confidence[missing[lines, path]] = np.nan
        found.append((path, confidence))
        for offset in range(-exclude, exclude + 1):
            score[lines, np.clip(path + offset, 0, score.shape[1] - 1)] = 0.0
    return found
