import numpy as np
from sklearn import metrics as sk_metrics

from mini_lr import metrics

# The 6-patient example worked out by hand: TP=2, FP=1, TN=2, FN=1.
Y_TRUE = [1, 1, 1, 0, 0, 0]
Y_PRED = [1, 1, 0, 0, 0, 1]


def test_metrics_match_hand_calculation():
    np.testing.assert_array_equal(
        metrics.confusion_matrix(Y_TRUE, Y_PRED), [[2, 1], [1, 2]]
    )
    for score in (
        metrics.accuracy_score,
        metrics.precision_score,
        metrics.recall_score,
        metrics.f1_score,
    ):
        assert np.isclose(score(Y_TRUE, Y_PRED), 2 / 3)


def test_metrics_match_sklearn():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)

    np.testing.assert_array_equal(
        metrics.confusion_matrix(y_true, y_pred),
        sk_metrics.confusion_matrix(y_true, y_pred),
    )
    assert np.isclose(
        metrics.accuracy_score(y_true, y_pred),
        sk_metrics.accuracy_score(y_true, y_pred),
    )
    assert np.isclose(
        metrics.precision_score(y_true, y_pred),
        sk_metrics.precision_score(y_true, y_pred),
    )
    assert np.isclose(
        metrics.recall_score(y_true, y_pred), sk_metrics.recall_score(y_true, y_pred)
    )
    assert np.isclose(
        metrics.f1_score(y_true, y_pred), sk_metrics.f1_score(y_true, y_pred)
    )


def test_precision_is_zero_when_nothing_predicted_positive():
    assert metrics.precision_score([1, 0, 1], [0, 0, 0]) == 0.0
