"""Geometry that places a CSA film scan on NASA's frequency/height grid.

The horizontal mapping comes from physical features neither archive derives
from the other: the sounder's frequency markers, which NASA times outright and
the film records as bold full-height lines. Fitting one against the other needs
no assumption about sweep linearity - the sweep is markedly nonlinear, and the
fit absorbs that because the markers drive it.

The vertical mapping is the weak one. The film's height rulings give a spacing
in pixels, but their spacing in kilometres is supplied by neither archive, and
the row that means zero kilometres is likewise assumed. Both are therefore
parameters of `Geometry` rather than constants, so they can be fitted against a
matched NASA array instead of guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class Geometry:
    """Film pixel coordinates as a function of scan-line time and virtual height."""

    coefficients: np.ndarray  # film_x = polyval(coefficients, line_time)
    zero_row: float  # film row that means zero kilometres
    px_per_km: float  # film rows per kilometre of virtual height
    vertical_heights: np.ndarray | None = None
    vertical_rows: np.ndarray | None = None
    horizontal_reference: np.ndarray | None = None
    horizontal_columns: np.ndarray | None = None

    def columns(self, line_time):
        if (
            self.horizontal_reference is not None
            and self.horizontal_columns is not None
        ):
            # A piecewise fit has no basis for what happens past its own
            # knots. np.interp would clip to the boundary value there,
            # which reads as real (repeated) film content rather than the
            # absence of one - out of range is unknown, not the edge.
            line_time = np.asarray(line_time, dtype=float)
            columns = np.interp(
                line_time, self.horizontal_reference, self.horizontal_columns
            )
            outside = (line_time < self.horizontal_reference.min()) | (
                line_time > self.horizontal_reference.max()
            )
            return np.where(outside, np.nan, columns)
        return np.polyval(self.coefficients, line_time)

    def rows(self, v_height):
        if self.vertical_heights is not None and self.vertical_rows is not None:
            return np.interp(v_height, self.vertical_heights, self.vertical_rows)
        return self.zero_row + v_height * self.px_per_km

    def with_vertical(self, zero_row, px_per_km):
        return Geometry(self.coefficients, zero_row, px_per_km)


def rolling_median(values, window):
    padded = np.pad(values, window // 2, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def find_dark_lines(profile, window, sigma, merge):
    """Centroids of narrow dark lines, after removing broad density variation."""
    highpass = profile - rolling_median(profile, window)
    threshold = -sigma * highpass.std()
    hits = np.where(highpass < threshold)[0]
    groups = []
    for index in hits:
        if groups and index - groups[-1][-1] <= merge:
            groups[-1].append(index)
        else:
            groups.append([index])
    return np.array([np.average(g, weights=-highpass[g]) for g in groups])


def film_regions(image, merge=5):
    """-> (top, bottom) of the data area, excluding margin and time-code band.

    Bright rows are grouped, and groups separated by a short gap are merged:
    the height-ruling lines are themselves periodic dark rows, so without
    merging, a run-length definition fragments the real area on every ruling
    and can select a fragment far smaller than the true ionogram. The gap
    tolerance is kept short so a single bright row inside the time-code band
    (e.g. a wide digit stroke) stays its own isolated group instead of
    merging into and extending the real boundary.
    """
    row_mean = image.mean(axis=1)
    bright = np.flatnonzero(row_mean > row_mean.max() * 0.35)
    if not len(bright):
        return 0, len(row_mean)
    groups = [[int(bright[0])]]
    for row in bright[1:]:
        if row - groups[-1][-1] <= merge:
            groups[-1].append(int(row))
        else:
            groups.append([int(row)])
    group = max(groups, key=len)
    return group[0], group[-1] + 1


class FitFailed(Exception):
    """Raised when a scan cannot be placed on the grid, with the reason why.

    An exception rather than an exit, because this runs inside a batch driver
    and inside a calibration search, and neither can afford a library that
    terminates the process to report one unusable scan.
    """


def _windowed_fit(
    observed, reference, observed_start, reference_start, count, skip=None
):
    """-> (score-ingredients tuple) for one contiguous window, with an
    optional single interior observed index excluded."""
    x = reference[reference_start : reference_start + count]
    if skip is None:
        indices = np.arange(observed_start, observed_start + count)
    else:
        indices = np.concatenate(
            [
                np.arange(observed_start, observed_start + skip),
                np.arange(observed_start + skip + 1, observed_start + count + 1),
            ]
        )
    y = observed[indices]
    coefficients = np.polyfit(x, y, 1)
    residual = y - np.polyval(coefficients, x)
    rms = float(np.sqrt(np.mean(residual**2)))
    return rms, coefficients, residual, reference_start, indices


def fit_marker_axis(observed, reference, min_fraction=0.50, refine_window=5):
    """Fit reference marker positions (NASA's times, or a nominal index) to a
    long ordered run of detected film marker lines.

    Extra frame lines are common, so every sufficiently long contiguous pair of
    runs is tried on both sides - unlike a fit anchored at the start of each
    array, this finds a match even when the film's leading lines are spurious
    or NASA's own list does not begin at marker zero. The length floor
    prevents a coincidental four-line match from beating a marginally worse
    but much longer one.

    A single spurious extra dark line *inside* an otherwise clean run (a
    scratch, a crossing trace) is a different failure than an extra line at
    the frame edge: a purely contiguous match has no way to drop it without
    also discarding every genuine marker on one side, and keeping it shifts
    every following marker's assignment by one line - the residual then
    fails smoothly across many markers rather than as one clear outlier.

    Trying every interior skip position for every one of the O(n^3) plain
    windows is too slow to run per scan. Since a skip only ever helps a
    window that a plain fit already fits badly, it is tried only as a
    second pass, in a small neighbourhood around the best plain match - not
    across the whole search.
    """
    observed = np.asarray(observed, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    if len(observed) < 4 or len(reference) < 4:
        raise FitFailed("need at least four marker lines in both archives")

    minimum = max(4, int(math.ceil(min(len(observed), len(reference)) * min_fraction)))
    available = min(len(observed), len(reference))

    def score(count, rms, skipped):
        # One-pixel residual improvement is not worth throwing away a large
        # part of the known frequency scale, and an interior skip is a
        # stronger claim than a merely shorter run.
        return rms + 0.25 * (available - count) + (0.5 if skipped else 0.0)

    candidates = []
    for observed_start in range(len(observed)):
        for reference_start in range(len(reference)):
            maximum = min(
                len(observed) - observed_start, len(reference) - reference_start
            )
            for count in range(minimum, maximum + 1):
                rms, coefficients, residual, ref_start, indices = _windowed_fit(
                    observed, reference, observed_start, reference_start, count
                )
                candidates.append(
                    (
                        score(count, rms, False),
                        -count,
                        rms,
                        coefficients,
                        residual,
                        ref_start,
                        indices,
                    )
                )
    best = min(candidates, key=lambda item: item[:2])

    if best[2] > 3.0:
        base_reference_start = best[5]
        base_observed_start = int(best[6][0])
        for reference_start in range(
            max(0, base_reference_start - refine_window),
            min(len(reference), base_reference_start + refine_window + 1),
        ):
            for observed_start in range(
                max(0, base_observed_start - refine_window),
                min(len(observed), base_observed_start + refine_window + 1),
            ):
                maximum = min(
                    len(observed) - observed_start, len(reference) - reference_start
                )
                for count in range(minimum, maximum + 1):
                    skips = [None]
                    if observed_start + count < len(observed):
                        skips += list(range(1, count))
                    for skip in skips:
                        rms, coefficients, residual, ref_start, indices = _windowed_fit(
                            observed,
                            reference,
                            observed_start,
                            reference_start,
                            count,
                            skip,
                        )
                        candidates.append(
                            (
                                score(count, rms, skip is not None),
                                -count,
                                rms,
                                coefficients,
                                residual,
                                ref_start,
                                indices,
                            )
                        )
        best = min(candidates, key=lambda item: item[:2])

    _, _, rms, coefficients, residual, reference_start, indices = best
    count = len(residual)
    return {
        "coefficients": coefficients,
        "rms_px": rms,
        "max_error_px": float(np.max(np.abs(residual))),
        "observed_start": int(indices[0]),
        "reference_start": reference_start,
        "count": count,
        "residual": residual,
        "observed_indices": indices,
    }


def solve(image, mark_times, km_per_ruling, marker_sigma, ruling_sigma, tolerance):
    """-> (Geometry, diagnostics) from the film's own markers and rulings."""
    top, bottom = film_regions(image)
    band = image[top:bottom]
    markers = find_dark_lines(band.mean(axis=0), 61, marker_sigma, 4)
    fit = fit_marker_axis(markers, mark_times)
    if fit["rms_px"] > tolerance:
        raise FitFailed(
            f"frequency fit rms {fit['rms_px']:.2f}px exceeds {tolerance}px"
        )

    first_index = int(fit["observed_indices"][0])
    last_index = int(fit["observed_indices"][-1])
    signal = slice(int(markers[first_index]), int(markers[last_index]))
    rulings = (
        find_dark_lines(image[top:bottom, signal].mean(axis=1), 41, ruling_sigma, 2)
        + top
    )
    spacing = np.diff(rulings)
    keep = np.abs(spacing - np.median(spacing)) < 0.35 * np.median(spacing)
    regular = rulings[np.append(keep, True) & np.append(True, keep)]
    px_per_ruling = float(np.median(np.diff(regular)))

    geometry = Geometry(
        coefficients=fit["coefficients"],
        zero_row=float(regular[0]),
        px_per_km=px_per_ruling / km_per_ruling,
    )
    return geometry, {
        "top": top,
        "bottom": bottom,
        "markers": markers,
        **fit,
        "regular": regular,
        "px_per_ruling": px_per_ruling,
    }


def resample(image, geometry, line_time, v_height):
    """-> the film on NASA's (scan line x height bin) grid, signal-positive.

    Film traces are dark on a light base, so the image is inverted first: what
    comes back is large where the film is dark. That is a density, never an
    amplitude, and nothing here converts between the two.
    """
    grid_x, grid_y = np.meshgrid(
        geometry.columns(line_time), geometry.rows(v_height), indexing="ij"
    )
    warped = map_coordinates(
        255.0 - image,
        [grid_y.ravel(), grid_x.ravel()],
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(grid_x.shape)
    inside = (
        (grid_x >= 0)
        & (grid_x <= image.shape[1] - 1)
        & (grid_y >= 0)
        & (grid_y <= image.shape[0] - 1)
    )
    warped[~inside] = np.nan
    return warped
