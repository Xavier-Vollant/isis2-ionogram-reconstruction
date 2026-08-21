"""A small logistic model over film features, fitted by gradient descent.

Logistic regression rather than anything larger, for three reasons: the labels
are themselves an extraction rather than truth, so a model able to fit them
exactly would be fitting their errors; the fitted weights are readable, which
matters when the question is *what in the film marks an echo*; and it adds no
dependency to a project that has not yet chosen its scientific stack.

Written out in numpy rather than pulled from scikit-learn, which is not among
the installed candidate packages. If a bigger model is ever justified, that is
the moment to add the dependency, not before.
"""

from __future__ import annotations

import numpy as np


class Standardizer:
    """Centre and scale each feature, so one learning rate suits all of them."""

    def __init__(self, features):
        self.centre = features.mean(axis=0)
        self.scale = np.maximum(features.std(axis=0), 1e-9)

    def __call__(self, features):
        return (features - self.centre) / self.scale


def fit(features, labels, l2=1e-3, rate=0.5, steps=800, seed=0):
    """-> weights for the standardized features, bias last.

    Positives are reweighted to balance the classes. Echo pixels are well under
    2% of an ionogram, and an unweighted fit answers "no echo anywhere" with
    98% accuracy, which is both true and useless.
    """
    rng = np.random.default_rng(seed)
    design = np.hstack([features, np.ones((len(features), 1))])
    weights = rng.normal(0, 0.01, design.shape[1])

    positive = labels > 0
    balance = np.where(
        positive, 0.5 / max(positive.mean(), 1e-9), 0.5 / max(1 - positive.mean(), 1e-9)
    )
    balance = balance / balance.mean()

    velocity = np.zeros_like(weights)
    for _ in range(steps):
        predicted = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))
        gradient = design.T @ (balance * (predicted - labels)) / len(design)
        gradient[:-1] += l2 * weights[:-1]
        velocity = 0.9 * velocity + gradient
        weights -= rate * velocity
    return weights


def score(features, weights):
    """-> log-odds. Left unsquashed: the trace search wants an additive score."""
    return np.hstack([features, np.ones((len(features), 1))]) @ weights
