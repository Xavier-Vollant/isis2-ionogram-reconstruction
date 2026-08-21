"""Checks for the CDF sweep-boundary policy.

NASA files may carry a fixed-frequency prefix before the sweep, which shows up
as a negative step in `freq`. The prefix has to be cut. The bug this pins is
cutting when there is no such step: `argmin` returns an index unconditionally,
so on an already-monotonic axis the cut landed at the smallest *positive* step
and discarded a healthy prefix.
"""

import numpy as np
import pytest

from isis_research.nasa.cdf import _cdf_sweep


def _data(frequency, heights=(0.0, 100.0)):
    frequency = np.asarray(frequency, dtype=float)
    amplitude = np.tile(
        np.arange(len(frequency), dtype=float)[:, None], (1, len(heights))
    )
    return {
        "nasa_amplitude": amplitude,
        "freq": frequency,
        "v_height": np.asarray(heights, dtype=float),
    }


def test_a_monotonic_axis_is_kept_whole():
    """The regression: nothing to cut, so nothing may be cut."""
    frequency = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    _, kept, _ = _cdf_sweep(_data(frequency))
    assert list(kept) == frequency


def test_a_monotonic_axis_with_uneven_steps_is_still_kept_whole():
    """The exact shape that used to trigger a spurious cut.

    The smallest positive step sits in the middle, which is where the old
    `argmin` landed - throwing away everything before it.
    """
    frequency = [0.1, 0.9, 1.7, 1.75, 2.6, 3.5]
    _, kept, _ = _cdf_sweep(_data(frequency))
    assert list(kept) == frequency
    assert kept[0] == 0.1, "the prefix before the smallest step must survive"


def test_a_fixed_frequency_prefix_is_still_cut():
    """The behaviour that must not regress."""
    frequency = [5.0, 5.0, 5.0, 0.5, 1.0, 1.5, 2.0]
    _, kept, _ = _cdf_sweep(_data(frequency))
    assert list(kept) == [0.5, 1.0, 1.5, 2.0]


def test_the_cut_takes_the_largest_negative_step():
    frequency = [2.0, 3.0, 1.0, 9.0, 0.5, 1.0, 1.5]
    _, kept, _ = _cdf_sweep(_data(frequency))
    assert list(kept) == [0.5, 1.0, 1.5]


def test_duplicate_frequencies_are_collapsed():
    _, kept, _ = _cdf_sweep(_data([0.5, 1.0, 1.0, 1.5]))
    assert list(kept) == [0.5, 1.0, 1.5]


def test_non_positive_and_non_finite_samples_are_dropped():
    _, kept, _ = _cdf_sweep(_data([0.0, -1.0, np.nan, 1.0, 2.0, 3.0]))
    assert list(kept) == [1.0, 2.0, 3.0]


def test_a_sweep_that_never_increases_is_rejected():
    with pytest.raises(ValueError):
        _cdf_sweep(_data([3.0, 2.0]))


def test_too_few_samples_is_rejected():
    with pytest.raises(ValueError):
        _cdf_sweep(_data([1.0]))


def test_amplitude_rows_follow_their_frequencies_through_the_cut():
    """The arrays must stay aligned, or the target is silently scrambled."""
    frequency = [5.0, 5.0, 0.5, 1.0, 1.5]
    amplitude, kept, _ = _cdf_sweep(_data(frequency))
    assert list(kept) == [0.5, 1.0, 1.5]
    # rows 2, 3, 4 of the original amplitude, whose values are 2, 3, 4
    assert [row[0] for row in amplitude] == [2.0, 3.0, 4.0]
