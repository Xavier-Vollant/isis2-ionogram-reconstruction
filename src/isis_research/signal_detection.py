"""NASA-supervised signal occupancy for film ionograms.

This module deliberately stops before amplitude assignment.  It represents a
scan as three states: signal, no signal, and unknown.  NASA CDF amplitudes are
used only to construct training/evaluation labels; the feature functions read
the film warp only and are therefore usable when no NASA counterpart exists.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

from .extraction.echo import (
    extract as extract_traces,
    parameters_for,
    ridge_score,
    scaled_kernel,
)


def nasa_occupancy(
    amplitude: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 2.5,
    ambiguity: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(score, labels, usable)`` from a NASA amplitude grid.

    ``labels`` is 1 for locally elevated NASA signal, 0 for locally quiet
    pixels, and -1 for an ambiguous or invalid pixel.  This is a reproducible
    weak label, not a claim that every bright CDF pixel is a physical echo.
    The local median removes broad illumination/background structure and the
    MAD supplies a scan-specific noise scale.
    """

    values = np.asarray(amplitude, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("amplitude and valid_mask must be matching 2-D arrays")
    fill = float(np.nanmedian(values[valid])) if valid.any() else 0.0
    filled = np.where(valid, values, fill)
    baseline = ndimage.median_filter(filled, size=(7, 7), mode="nearest")
    residual = filled - baseline
    centre = float(np.median(residual[valid])) if valid.any() else 0.0
    deviation = np.abs(residual[valid] - centre) if valid.any() else np.array([0.0])
    noise = max(1.4826 * float(np.median(deviation)), 0.01)
    score = (residual - centre) / noise
    score[~valid] = np.nan

    labels = np.full(values.shape, -1, dtype=np.int8)
    positive = valid & (score >= threshold)
    negative = valid & (score <= threshold - ambiguity)
    labels[positive] = 1
    labels[negative] = 0
    usable = labels >= 0
    return score.astype(np.float32), labels, usable


def nasa_trace_occupancy(
    valid_mask: np.ndarray,
    paths: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 5.0,
    halfwidth: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(score, labels, usable)`` around confident NASA traces.

    ``paths`` and ``confidence`` are shaped ``(trace, frequency)`` and use
    height-bin indices.  Only frequency columns with a confident NASA trace
    become usable training examples; undetected columns remain unknown rather
    than being treated as negative echo evidence.
    """

    valid = np.asarray(valid_mask, dtype=bool)
    paths = np.asarray(paths)
    confidence = np.asarray(confidence, dtype=float)
    if valid.ndim != 2:
        raise ValueError("valid_mask must be 2-D")
    if paths.ndim == 1:
        paths = paths[None, :]
    if confidence.ndim == 1:
        confidence = confidence[None, :]
    if paths.shape != confidence.shape or paths.shape[1] != valid.shape[1]:
        raise ValueError("paths/confidence must match valid_mask frequency width")
    if halfwidth < 0:
        raise ValueError("halfwidth must be non-negative")

    labels = np.full(valid.shape, -1, dtype=np.int8)
    usable = np.zeros(valid.shape, dtype=bool)
    score = np.full(valid.shape, np.nan, dtype=np.float32)
    detected = np.isfinite(confidence) & (confidence >= threshold)
    for column in np.flatnonzero(detected.any(axis=0)):
        usable[:, column] = valid[:, column]
        labels[valid[:, column], column] = 0
    for trace, values in zip(paths, confidence):
        for column in np.flatnonzero(np.isfinite(values) & (values >= threshold)):
            centre = int(trace[column])
            low = max(0, centre - halfwidth)
            high = min(valid.shape[0], centre + halfwidth + 1)
            band = valid[low:high, column]
            labels[low:high, column][band] = 1
            score[low:high, column][band] = np.maximum(
                np.nan_to_num(score[low:high, column][band], nan=0.0),
                float(values[column]),
            )
    return score, labels, usable


def nasa_trace_soft_targets(
    labels: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 5.0,
    scale: float = 5.0,
) -> np.ndarray:
    """Return confidence-weighted positive targets with unknowns preserved."""

    labels = np.asarray(labels, dtype=np.int8)
    confidence = np.asarray(confidence, dtype=float)
    if labels.shape != confidence.shape:
        raise ValueError("labels and confidence must have matching shapes")
    if threshold <= 0 or scale <= 0:
        raise ValueError("threshold and scale must be positive")

    targets = np.full(labels.shape, -1.0, dtype=np.float32)
    targets[labels == 0] = 0.0
    positive = (labels == 1) & np.isfinite(confidence)
    strength = np.clip((confidence - threshold) / scale, 0.0, 1.0)
    targets[positive] = 0.5 + 0.5 * strength[positive]
    return targets


def nasa_trace_training_labels(
    labels: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 7.0,
) -> np.ndarray:
    """Hide low-confidence NASA positives from training without making negatives."""

    labels = np.asarray(labels, dtype=np.int8)
    confidence = np.asarray(confidence, dtype=float)
    if labels.shape != confidence.shape:
        raise ValueError("labels and confidence must have matching shapes")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    filtered = labels.copy()
    weak_positive = (filtered == 1) & (
        ~np.isfinite(confidence) | (confidence < threshold)
    )
    filtered[weak_positive] = -1
    return filtered


def nasa_trace_persistent_training_labels(
    labels: np.ndarray,
    confidence: np.ndarray,
    min_run: int = 3,
) -> np.ndarray:
    """Keep only NASA-labelled columns in runs that persist across frequency."""

    labels = np.asarray(labels, dtype=np.int8)
    confidence = np.asarray(confidence, dtype=float)
    if labels.shape != confidence.shape:
        raise ValueError("labels and confidence must have matching shapes")
    if labels.ndim < 2:
        raise ValueError("labels and confidence must have height and frequency axes")
    if min_run < 1:
        raise ValueError("min_run must be positive")

    detected = np.isfinite(confidence).any(axis=-2) & (labels >= 0).any(axis=-2)
    flat_detected = detected.reshape(-1, detected.shape[-1])
    flat_persistent = np.zeros_like(flat_detected)
    for row, columns in enumerate(flat_detected):
        run_ids, _ = ndimage.label(columns)
        run_lengths = np.bincount(run_ids)
        flat_persistent[row] = columns & (run_lengths[run_ids] >= min_run)
    persistent = flat_persistent.reshape(detected.shape)
    mask_shape = labels.shape[:-2] + (1, labels.shape[-1])
    return np.where(persistent.reshape(mask_shape), labels, -1).astype(np.int8)


FEATURE_NAMES = (
    "film_density",
    "local_contrast",
    "height_ridge",
    "frequency_ridge",
    "height_gradient",
    "frequency_gradient",
    "echo_ridge_score",
    "valid_mask",
)
MULTISCALE_WIDTHS_KM = (150.0, 250.0, 450.0, 750.0)


def film_feature_maps(film: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Build film-only features shaped ``(channels, height, frequency)``."""

    values = np.asarray(film, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("film and valid_mask must be matching 2-D arrays")
    values = np.nan_to_num(values, nan=0.0)

    height_background = ndimage.median_filter(values, size=(7, 1), mode="nearest")
    frequency_background = ndimage.median_filter(values, size=(1, 7), mode="nearest")
    local_background = ndimage.median_filter(values, size=(5, 5), mode="nearest")
    height_gradient, frequency_gradient = np.gradient(values)
    ridge, _ = ridge_score(values.T)

    layers = np.stack(
        [
            values,
            values - local_background,
            values - height_background,
            values - frequency_background,
            np.abs(height_gradient),
            np.abs(frequency_gradient),
            ridge.T,
            valid.astype(float),
        ],
        axis=0,
    ).astype(np.float32)
    layers[~np.broadcast_to(valid, layers.shape)] = 0.0
    return layers


def multiscale_ridge_score(
    film: np.ndarray,
    valid_mask: np.ndarray,
    height_km: np.ndarray,
    widths_km=MULTISCALE_WIDTHS_KM,
) -> np.ndarray:
    """Return the strongest normalized height-ridge response across widths."""

    return np.max(
        multiscale_ridge_features(film, valid_mask, height_km, widths_km),
        axis=0,
    )


def multiscale_ridge_features(
    film: np.ndarray,
    valid_mask: np.ndarray,
    height_km: np.ndarray,
    widths_km=MULTISCALE_WIDTHS_KM,
) -> np.ndarray:
    """Return one normalized height-ridge channel per physical width."""

    values = np.asarray(film, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    heights = np.asarray(height_km, dtype=float).ravel()
    widths = tuple(float(width) for width in widths_km)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("film and valid_mask must be matching 2-D arrays")
    if heights.ndim != 1 or len(heights) != values.shape[0] or len(heights) < 2:
        raise ValueError("height_km must match the film height axis")
    spacing = float(np.median(np.diff(heights)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("height_km must be strictly increasing")
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("widths_km must contain positive widths")

    # Keep the extractor finite on masked columns.  The response is masked
    # again below, while zero-filling avoids all-NaN median warnings inside
    # the shared ridge scorer.
    signal = np.where(valid, values, 0.0)
    responses = []
    for width in widths:
        kernel = scaled_kernel(round(width / spacing))
        response, _ = ridge_score(signal.T, kernel=kernel)
        responses.append(response.T)
    return np.where(valid, np.stack(responses), 0.0).astype(np.float32)


def continuity_ridge_score(
    film: np.ndarray,
    valid_mask: np.ndarray,
    height_km: np.ndarray,
    traces: int = 2,
    halfwidth: int = 1,
) -> np.ndarray:
    """Return score bands around film traces selected by frequency continuity."""

    values = np.asarray(film, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    heights = np.asarray(height_km, dtype=float).ravel()
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("film and valid_mask must be matching 2-D arrays")
    if heights.ndim != 1 or len(heights) != values.shape[0]:
        raise ValueError("height_km must match the film height axis")
    if traces < 1 or halfwidth < 0:
        raise ValueError("traces must be positive and halfwidth non-negative")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        found = extract_traces(
            np.where(valid, values, 0.0).T,
            traces=traces,
            **parameters_for(heights),
        )
    score = np.zeros(values.shape, dtype=np.float32)
    for path, confidence in found:
        for column in np.flatnonzero(np.isfinite(confidence)):
            centre = int(path[column])
            low = max(0, centre - halfwidth)
            high = min(values.shape[0], centre + halfwidth + 1)
            band = valid[low:high, column]
            band_score = score[low:high, column]
            band_score[band] = np.maximum(
                band_score[band], max(float(confidence[column]), 0.0)
            )
    return score


def film_model_features(
    film: np.ndarray,
    valid_mask: np.ndarray,
    height_km: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    """Build the persisted film feature channels required by a model."""

    channels = np.asarray(channels, dtype=int).ravel()
    features = film_feature_maps(film, valid_mask)
    if channels.size and channels.min() < 0:
        raise ValueError("model channels must be non-negative")
    if channels.size and channels.max() >= features.shape[0]:
        if channels.max() != features.shape[0]:
            raise ValueError("model requests unsupported film feature channels")
        features = np.concatenate(
            [features, continuity_ridge_score(film, valid_mask, height_km)[None]],
            axis=0,
        )
    if channels.size and channels.max() >= features.shape[0]:
        raise ValueError("model channel is not available")
    return features[channels]


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""

    values = np.asarray(values, dtype=float)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(np.float32)


def score_to_probability(
    score: np.ndarray, threshold: float, scale: float
) -> np.ndarray:
    """Map a fitted detector score to the Plan 1 probability contract."""

    return sigmoid((np.asarray(score, dtype=float) - threshold) / max(scale, 1e-3))


def robust_scale(values: np.ndarray) -> float:
    """Return a non-zero scale suitable for turning a score into a probability."""

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    return max(float((q75 - q25) / 1.349), 1e-3)
