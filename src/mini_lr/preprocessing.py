"""Feature scaling, so every feature is on a similar scale before training."""

import numpy as np

from mini_lr.logistic_regression import NotFittedError


class StandardScaler:
    """Rescales each feature to mean 0 and standard deviation 1.

    It learns the mean and standard deviation from the training data in fit,
    then uses those same numbers in transform for any data you give it.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Learn each column's mean and standard deviation from X (training data only)."""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = (
            1.0  # a constant column would divide by zero; 1 leaves it as zeros
        )
        self.scale_ = scale
        return self

    def transform(self, X):
        """Scale X using the mean and standard deviation learned in fit."""
        if self.mean_ is None or self.scale_ is None:
            raise NotFittedError(
                "This StandardScaler is not fitted yet. Call fit first."
            )
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_
