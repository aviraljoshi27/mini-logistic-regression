"""Logistic regression built from scratch with NumPy."""

import numpy as np


def sigmoid(z):
    """Turn any number z into a probability between 0 and 1.

    I use exp(-|z|) so exp never gets a big positive number and overflows.
    For z >= 0 this is the usual 1 / (1 + e^-z). For z < 0 it's the same
    value written as e^z / (1 + e^z).
    """
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    """Average loss of the predicted probabilities against the true 0/1 labels.

    Confident wrong answers get punished hard. I clip the probabilities to
    [eps, 1 - eps] first, because log(0) is minus infinity and would turn
    the loss into inf or nan.
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)
    losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return np.mean(losses)


def predict_proba(X, w, b):
    """Probability that each row of X is class 1.

    First the straight-line score z = X @ w + b, one number per row,
    then sigmoid squashes each score into a probability.
    """
    z = X @ w + b
    return sigmoid(z)


def predict_class(X, w, b, threshold=0.5):
    """Turn probabilities into 0/1 predictions.

    A row becomes 1 if its probability is at least the threshold, else 0.
    0.5 is only the default. Moving it trades false alarms for missed cases.
    """
    proba = predict_proba(X, w, b)
    return (proba >= threshold).astype(int)


class NotFittedError(ValueError):
    """Raised when predict or predict_proba is called before fit."""


class LogisticRegression:
    """Logistic regression trained with plain gradient descent."""

    def __init__(self, learning_rate=0.1, n_iterations=1000, l2=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = None

    def __repr__(self):
        """Show the model's settings when it is printed."""
        return (
            f"LogisticRegression(learning_rate={self.learning_rate}, "
            f"n_iterations={self.n_iterations}, l2={self.l2})"
        )

    def fit(self, X, y):
        """Learn the weights and bias with gradient descent."""
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            p = predict_proba(X, self.weights_, self.bias_)
            penalty = (self.l2 / 2) * np.sum(self.weights_**2)
            self.loss_history_.append(binary_cross_entropy(y, p) + penalty)
            error = p - y
            grad_w = X.T @ error / n_samples + self.l2 * self.weights_
            grad_b = np.mean(error)
            self.weights_ = self.weights_ - self.learning_rate * grad_w
            self.bias_ = self.bias_ - self.learning_rate * grad_b

        return self

    def _check_is_fitted(self):
        """Stop with a clear message if fit hasn't been called yet."""
        if self.weights_ is None:
            raise NotFittedError(
                "This LogisticRegression is not fitted yet. Call fit(X, y) first."
            )

    def predict_proba(self, X):
        """Probability that each row of X is class 1, using the learned weights."""
        self._check_is_fitted()
        return predict_proba(X, self.weights_, self.bias_)

    def predict(self, X, threshold=0.5):
        """0 or 1 for each row of X, using the learned weights."""
        self._check_is_fitted()
        return predict_class(X, self.weights_, self.bias_, threshold)
