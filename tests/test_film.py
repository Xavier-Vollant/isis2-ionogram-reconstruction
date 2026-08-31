"""Tests for film geometry and exposed-region detection."""

import numpy as np

from isis_research.registration.film import Geometry, film_regions


def test_geometry_columns_prefers_piecewise_over_polyval_when_set():
    # A straight line through these three points would badly misplace the
    # middle one - the same nonlinearity that made a global polyfit wrong for
    # column-vs-frequency (scripts/warp_film_only_sample.py).
    geometry = Geometry(
        coefficients=np.array([1000.0, 0.0]),  # would predict columns~1000x off
        zero_row=0.0,
        px_per_km=1.0,
        horizontal_reference=np.array([1.0, 2.0, 3.0]),
        horizontal_columns=np.array([10.0, 50.0, 55.0]),
    )
    assert geometry.columns(np.array([1.0, 2.0, 3.0])).tolist() == [10.0, 50.0, 55.0]
    assert geometry.columns(np.array([1.5])).item() == 30.0


def test_geometry_columns_falls_back_to_polyval_without_piecewise():
    geometry = Geometry(coefficients=np.array([2.0, 1.0]), zero_row=0.0, px_per_km=1.0)
    assert geometry.columns(np.array([1.0, 2.0])).tolist() == [3.0, 5.0]


def test_geometry_columns_piecewise_is_nan_outside_the_matched_knots():
    # np.interp alone would clip to the boundary value here (10.0 for 0.5,
    # 55.0 for 3.5) - real film content, at the wrong place. Past the fitted
    # knots the true column is unknown, not the edge one.
    geometry = Geometry(
        coefficients=np.array([0.0, 0.0]),
        zero_row=0.0,
        px_per_km=1.0,
        horizontal_reference=np.array([1.0, 2.0, 3.0]),
        horizontal_columns=np.array([10.0, 50.0, 55.0]),
    )
    result = geometry.columns(np.array([0.5, 2.0, 3.5]))
    assert np.isnan(result[0]) and np.isnan(result[2])
    assert result[1] == 50.0


def test_film_regions_bridges_a_single_row_dip():
    values = np.zeros(80)
    values[10:41] = 200.0
    values[25] = 0.0  # one ruling-line row inside the data area
    image = np.tile(values[:, None], (1, 5))
    assert film_regions(image) == (10, 41)


def test_film_regions_ignores_an_isolated_bright_row_past_the_true_edge():
    values = np.zeros(80)
    values[10:41] = 200.0
    values[65] = 200.0  # one stray bright row deep in the time-code band
    image = np.tile(values[:, None], (1, 5))
    assert film_regions(image) == (10, 41)
