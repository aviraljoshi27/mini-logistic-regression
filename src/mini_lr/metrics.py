"""Metrics for checking how good the predictions are, written by hand."""

import numpy as np


def _confusion_counts(y_true, y_pred):
    """Count true positives, false positives, true negatives and false negatives."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def confusion_matrix(y_true, y_pred):
    """Return the 2x2 table [[TN, FP], [FN, TP]], laid out the same way as sklearn."""
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])


def _safe_divide(top, bottom):
    """Divide, but give 0.0 instead of crashing when the bottom is 0."""
    return top / bottom if bottom > 0 else 0.0


def accuracy_score(y_true, y_pred):
    """Share of all predictions that were right."""
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    return _safe_divide(tp + tn, tp + fp + tn + fn)


def precision_score(y_true, y_pred):
    """Of everything predicted as 1 (malignant), the share that really was 1."""
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fp)


def recall_score(y_true, y_pred):
    """Of everything that really was 1 (malignant), the share we caught."""
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fn)


def f1_score(y_true, y_pred):
    """One number that is only high when precision and recall are both high."""
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    return _safe_divide(2 * tp, 2 * tp + fp + fn)
