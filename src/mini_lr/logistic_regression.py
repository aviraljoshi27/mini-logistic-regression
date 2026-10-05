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
