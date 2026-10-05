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
