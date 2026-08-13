"""Tests for metrics computation module."""

import numpy as np
from src.metrics import compute_metrics


def test_compute_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    y_prob = np.eye(3)[y_pred]

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names=["C1", "C2", "C3"])

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "kappa" in metrics
    assert "macro_sensitivity" in metrics
    assert "macro_specificity" in metrics
    assert len(metrics["per_class"]) == 3
