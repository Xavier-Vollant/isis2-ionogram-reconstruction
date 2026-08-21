"""`echo.parameters_for` must not change any number already published.

The whole point of expressing the extractor's constants in kilometres is that
the 64-bin benchmark grid keeps its shipped behaviour exactly. If that equality
ever breaks, every Stage F number silently moves.
"""

import numpy as np
import pytest

from isis_research.extraction import echo

BENCHMARK_HEIGHT = np.linspace(0.0, 3000.0, 64)
NATIVE_HEIGHT = np.linspace(0.0, 3330.000049620867, 223)


def test_benchmark_grid_reproduces_the_shipped_defaults():
    parameters = echo.parameters_for(BENCHMARK_HEIGHT)
    assert parameters["cap"] == 20
    assert parameters["exclude"] == 3
    assert parameters["smoothness"] == pytest.approx(0.5)
    assert np.array_equal(parameters["kernel"], echo.RIDGE_KERNEL)


def test_benchmark_grid_extraction_is_bit_identical():
    """Not just the constants - the trace itself must not move."""
    rng = np.random.default_rng(0)
    array = rng.normal(size=(40, 64))
    array[:, 20] += 6.0
    before = echo.extract(array, traces=2)
    after = echo.extract(array, traces=2, **echo.parameters_for(BENCHMARK_HEIGHT))
    for (path_a, conf_a), (path_b, conf_b) in zip(before, after):
        assert np.array_equal(path_a, path_b)
        np.testing.assert_array_equal(conf_a, conf_b)


def test_a_finer_axis_scales_every_constant():
    parameters = echo.parameters_for(NATIVE_HEIGHT)
    bin_km = 3330.000049620867 / 222
    assert parameters["cap"] == round(echo.CAP_KM / bin_km)
    assert parameters["exclude"] == round(echo.EXCLUDE_KM / bin_km)
    assert parameters["smoothness"] == pytest.approx(echo.SMOOTHNESS_PER_KM * bin_km)
    # a 429 km matched filter is many more taps once a bin is 15 km
    assert len(parameters["kernel"]) > len(echo.RIDGE_KERNEL)
    assert len(parameters["kernel"]) % 2 == 1


def test_physical_width_is_preserved_across_resolutions():
    coarse = echo.parameters_for(BENCHMARK_HEIGHT)
    fine = echo.parameters_for(NATIVE_HEIGHT)
    coarse_bin = 3000.0 / 63.0
    fine_bin = 3330.000049620867 / 222
    assert len(coarse["kernel"]) * coarse_bin == pytest.approx(
        len(fine["kernel"]) * fine_bin, rel=0.05
    )
    assert coarse["cap"] * coarse_bin == pytest.approx(fine["cap"] * fine_bin, rel=0.05)
    # cost per kilometre of height change, not per bin
    assert coarse["smoothness"] / coarse_bin == pytest.approx(
        fine["smoothness"] / fine_bin
    )


def test_scaled_kernel_stays_zero_mean_and_odd():
    for taps in (3, 4, 9, 28, 29, 73):
        kernel = echo.scaled_kernel(taps)
        assert len(kernel) % 2 == 1
        assert kernel.sum() == pytest.approx(0.0, abs=1e-9)


def test_a_degenerate_axis_is_refused():
    with pytest.raises(ValueError):
        echo.parameters_for([100.0])
    with pytest.raises(ValueError):
        echo.parameters_for([100.0, 100.0])


def test_guided_zero_weight_matches_ridge_extractor():
    rng = np.random.default_rng(12)
    array = rng.normal(size=(40, 64))
    probability = rng.uniform(size=array.shape)
    before = echo.extract(array, traces=2)
    after = echo.extract_guided(array, probability, traces=2, cnn_weight=0.0)
    for (path_a, conf_a), (path_b, conf_b) in zip(before, after):
        assert np.array_equal(path_a, path_b)
        np.testing.assert_array_equal(conf_a, conf_b)


def test_guided_probability_shape_is_checked():
    with pytest.raises(ValueError):
        echo.guided_score(np.zeros((40, 64)), np.zeros((40, 63)))


def test_cascade_zero_weight_matches_ridge_extractor():
    rng = np.random.default_rng(13)
    array = rng.normal(size=(40, 64))
    probability = rng.uniform(size=array.shape)
    before = echo.extract(array, traces=3)
    after = echo.extract_cascade(array, probability, traces=3, cnn_weight=0.0)
    for (path_a, conf_a), (path_b, conf_b) in zip(before, after):
        assert np.array_equal(path_a, path_b)
        np.testing.assert_array_equal(conf_a, conf_b)
