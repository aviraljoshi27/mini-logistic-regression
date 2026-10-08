import numpy as np
import pytest

from mini_lr.logistic_regression import NotFittedError
from mini_lr.preprocessing import StandardScaler

# Two features with the same shape but 100x different sizes (like area vs smoothness).
X_SCALE = np.array([[2.0, 200.0], [4.0, 400.0], [6.0, 600.0]])


def test_scaler_matches_hand_calculation():
    scaler = StandardScaler().fit(X_SCALE)
    assert scaler.mean_ is not None
    np.testing.assert_allclose(scaler.mean_, [4.0, 400.0])

    X_scaled = scaler.transform(X_SCALE)
    expected_column = [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]
    np.testing.assert_allclose(X_scaled[:, 0], expected_column)
    np.testing.assert_allclose(X_scaled[:, 1], expected_column)


def test_scaler_uses_training_statistics_only():
    scaler = StandardScaler().fit(X_SCALE)
    X_new = np.array([[20.0, 2000.0]])

    X_new_scaled = scaler.transform(X_new)
    expected = (20 - 4) / np.sqrt(8 / 3)
    np.testing.assert_allclose(X_new_scaled, [[expected, expected]])

    assert scaler.mean_ is not None
    np.testing.assert_allclose(scaler.mean_, [4.0, 400.0])


def test_transform_before_fit_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        StandardScaler().transform(X_SCALE)
