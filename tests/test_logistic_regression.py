"""Tests for the building blocks in mini_lr.logistic_regression."""

import numpy as np
import pytest

from mini_lr.logistic_regression import (
    LogisticRegression,
    NotFittedError,
    binary_cross_entropy,
    predict_class,
    predict_proba,
    sigmoid,
)


def test_sigmoid_of_zero_is_half():
    # 1 / (1 + e^0) = 1 / (1 + 1) = 0.5
    assert sigmoid(0.0) == 0.5


def test_sigmoid_known_values():
    # e^2 ≈ 7.389, so 1 / (1 + 7.389) ≈ 0.1192, and the other side is 1 - 0.1192
    z = np.array([-2.0, 0.0, 2.0])
    expected = np.array([0.11920292, 0.5, 0.88079708])
    assert sigmoid(z) == pytest.approx(expected)


def test_sigmoid_extreme_inputs_stay_between_0_and_1():
    # the simple formula overflows in exp for z like -1000 and warns
    z = np.array([-1000.0, -710.0, 0.0, 710.0, 1000.0])
    p = sigmoid(z)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_bce_known_value():
    # both examples are 80% sure of the right answer, so each loss is -ln(0.8)
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([0.8, 0.2])
    assert binary_cross_entropy(y_true, y_prob) == pytest.approx(0.22314355)


def test_bce_stays_finite_when_probabilities_hit_0_or_1():
    # without clipping, these give inf, inf and nan
    y_true = np.array([1.0, 0.0, 1.0])
    y_prob = np.array([0.0, 1.0, 1.0])
    assert np.isfinite(binary_cross_entropy(y_true, y_prob))


def test_bce_confident_wrong_costs_more_than_confident_right():
    y_true = np.array([1.0])
    assert binary_cross_entropy(y_true, np.array([0.1])) > binary_cross_entropy(
        y_true, np.array([0.9])
    )


def test_predict_proba_known_values():
    # z = X @ w + b: row 1 is 1 + 1 - 2 = 0, row 2 is 0 + 0 - 2 = -2
    X = np.array([[1.0, 1.0], [0.0, 0.0]])
    w = np.array([1.0, 1.0])
    b = -2.0
    p = predict_proba(X, w, b)
    assert p.shape == (2,)
    assert p == pytest.approx([0.5, 0.11920292])


def test_predict_class_gives_only_0_or_1():
    # z is 3, -3 and 0.5, so the probabilities are about 0.95, 0.05 and 0.62
    X = np.array([[3.0], [-3.0], [0.5]])
    w = np.array([1.0])
    pred = predict_class(X, w, 0.0)
    assert np.all(np.isin(pred, [0, 1]))
    assert list(pred) == [1, 0, 1]


def test_predict_class_threshold_changes_predictions():
    # same data, but now a row needs at least 0.7 to count as 1, so 0.62 drops to 0
    X = np.array([[3.0], [-3.0], [0.5]])
    w = np.array([1.0])
    pred = predict_class(X, w, 0.0, threshold=0.7)
    assert list(pred) == [1, 0, 0]


X_TINY = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
Y_TINY = np.array([0, 0, 0, 1, 1, 1])


def test_loss_goes_down_during_training():
    model = LogisticRegression(n_iterations=200).fit(X_TINY, Y_TINY)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_model_learns_simple_dataset():
    model = LogisticRegression(n_iterations=1000).fit(X_TINY, Y_TINY)
    assert np.array_equal(model.predict(X_TINY), Y_TINY)


def test_predict_before_fit_raises_not_fitted_error():
    model = LogisticRegression()
    with pytest.raises(NotFittedError):
        model.predict(X_TINY)


def test_fit_returns_the_model():
    model = LogisticRegression(n_iterations=10)
    assert model.fit(X_TINY, Y_TINY) is model


def test_l2_shrinks_the_weights():
    plain = LogisticRegression(n_iterations=1000).fit(X_TINY, Y_TINY)
    shrunk = LogisticRegression(n_iterations=1000, l2=0.1).fit(X_TINY, Y_TINY)
    assert np.sum(shrunk.weights_**2) < np.sum(plain.weights_**2)
