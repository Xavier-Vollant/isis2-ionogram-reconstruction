import numpy as np

from isis_research.signal_detection import (
    film_feature_maps,
    continuity_ridge_score,
    film_model_features,
    multiscale_ridge_features,
    multiscale_ridge_score,
    nasa_occupancy,
    nasa_trace_occupancy,
    nasa_trace_soft_targets,
    nasa_trace_training_labels,
    nasa_trace_persistent_training_labels,
)


def test_nasa_occupancy_marks_local_peak_and_invalid_as_unknown():
    amplitude = np.zeros((16, 20), dtype=float)
    amplitude[7, 10] = 10.0
    valid = np.ones_like(amplitude, dtype=bool)
    valid[0, 0] = False
    score, labels, usable = nasa_occupancy(amplitude, valid, threshold=2.0)
    assert score.shape == amplitude.shape
    assert labels[7, 10] == 1
    assert labels[0, 0] == -1
    assert not usable[0, 0]


def test_film_features_are_finite_and_keep_invalid_pixels_zero():
    film = np.zeros((16, 20), dtype=float)
    film[7, 10] = 1.0
    valid = np.ones_like(film, dtype=bool)
    valid[0, 0] = False
    features = film_feature_maps(film, valid)
    assert features.shape == (8, 16, 20)
    assert np.isfinite(features).all()
    assert np.all(features[:, 0, 0] == 0.0)


def test_nasa_trace_occupancy_marks_confident_band_and_unknown_gap():
    valid = np.ones((8, 5), dtype=bool)
    valid[0, 2] = False
    paths = np.array([[2, 3, 4, 3, 2]])
    confidence = np.array([[6.0, 6.0, 2.0, 6.0, 6.0]])

    score, labels, usable = nasa_trace_occupancy(
        valid, paths, confidence, threshold=5.0, halfwidth=1
    )

    assert labels[2, 0] == 1
    assert labels[1, 0] == 1
    assert labels[0, 2] == -1
    assert not usable[:, 2].any()
    assert labels[0, 0] == 0
    assert score[2, 0] == 6.0


def test_nasa_trace_soft_targets_preserve_unknowns_and_scale_confidence():
    labels = np.array([[0, 1, 1, -1]], dtype=np.int8)
    confidence = np.array([[np.nan, 5.0, 10.0, np.nan]])

    targets = nasa_trace_soft_targets(labels, confidence)

    assert np.array_equal(targets[0, [0, 3]], [0.0, -1.0])
    assert targets[0, 1] == 0.5
    assert targets[0, 2] == 1.0


def test_nasa_trace_training_labels_hide_weak_positives_as_unknown():
    labels = np.array([[0, 1, 1, -1]], dtype=np.int8)
    confidence = np.array([[np.nan, 6.0, 8.0, np.nan]])

    filtered = nasa_trace_training_labels(labels, confidence, threshold=7.0)

    assert np.array_equal(filtered, [[0, -1, 1, -1]])


def test_nasa_trace_persistent_training_labels_keep_only_long_runs():
    labels = np.zeros((3, 6), dtype=np.int8)
    labels[1, [0, 1, 3, 4, 5]] = 1
    labels[:, 2] = -1
    confidence = np.full(labels.shape, np.nan)
    confidence[1, [0, 1, 3, 4, 5]] = 6.0

    filtered = nasa_trace_persistent_training_labels(labels, confidence, min_run=3)

    assert np.all(filtered[:, 0:2] == -1)
    assert np.array_equal(filtered[:, 2], [-1, -1, -1])
    assert np.array_equal(filtered[:, 3:], labels[:, 3:])

    batched = nasa_trace_persistent_training_labels(
        np.stack([labels, labels]), np.stack([confidence, confidence]), min_run=3
    )
    assert batched.shape == (2, 3, 6)
    assert np.array_equal(batched[1], filtered)


def test_multiscale_ridge_score_is_finite_and_preserves_shape():
    height = np.linspace(0.0, 3000.0, 64)
    frequency = np.linspace(0.1, 9.5, 96)
    film = np.zeros((len(height), len(frequency)), dtype=float)
    centres = 18 + np.round(np.linspace(0, 12, len(frequency))).astype(int)
    for column, centre in enumerate(centres):
        film[max(0, centre - 1) : centre + 2, column] = 1.0

    valid = np.ones_like(film, dtype=bool)
    features = multiscale_ridge_features(film, valid, height, (250.0, 450.0))
    score = multiscale_ridge_score(film, valid, height, (250.0, 450.0))

    assert score.shape == film.shape
    assert features.shape == (2,) + film.shape
    assert np.allclose(score, features.max(axis=0))
    assert np.isfinite(features).all()
    assert np.isfinite(score).all()
    assert float(score.max()) > 0.0


def test_continuity_ridge_score_returns_a_finite_trace_band():
    rng = np.random.default_rng(4)
    height = np.linspace(0.0, 3000.0, 64)
    film = rng.normal(0.0, 0.02, (64, 96))
    centres = 20 + np.round(np.linspace(0, 12, 96)).astype(int)
    for column, centre in enumerate(centres):
        film[max(0, centre - 1) : centre + 2, column] += 1.0

    score = continuity_ridge_score(
        film, np.ones_like(film, dtype=bool), height, traces=1, halfwidth=1
    )

    assert score.shape == film.shape
    assert np.isfinite(score).all()
    assert float(score.max()) > 0.0
    assert np.count_nonzero(score) >= 96


def test_film_model_features_resolves_the_continuity_channel():
    rng = np.random.default_rng(5)
    height = np.linspace(0.0, 3000.0, 64)
    film = rng.normal(0.0, 0.02, (64, 96))
    valid = np.ones_like(film, dtype=bool)
    base = film_feature_maps(film, valid)

    features = film_model_features(film, valid, height, np.array([0, 8]))

    assert features.shape == (2,) + film.shape
    assert np.allclose(features[0], base[0])
    assert np.isfinite(features).all()
