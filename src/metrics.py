"""Evaluation metrics for imbalanced multi-class classification."""

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return headline metrics; add AUC from probabilities in evaluation code."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
