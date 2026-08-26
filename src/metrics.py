"""Comprehensive evaluation metrics for multi-class retinal disease classification."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    recall_score,
    precision_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive medical evaluation metrics.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted class labels (N,).
        y_prob: Predicted probability matrix of shape (N, num_classes).
        class_names: List of class labels.

    Returns:
        Dictionary containing aggregate metrics and per-class breakdown.
    """
    num_classes = len(np.unique(y_true)) if class_names is None else len(class_names)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # ROC-AUC calculation (One-vs-Rest)
    macro_auc = float("nan")
    if y_prob is not None and y_prob.ndim == 2:
        try:
            macro_auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            macro_auc = float("nan")

    # Per-class Sensitivity (Recall) and Specificity calculation
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    sensitivities = []
    specificities = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)

    macro_sensitivity = float(np.mean(sensitivities))
    macro_specificity = float(np.mean(specificities))

    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_auc": float(macro_auc),
        "kappa": float(kappa),
        "macro_sensitivity": float(macro_sensitivity),
        "macro_specificity": float(macro_specificity),
    }

    if class_names is not None:
        per_class_df = pd.DataFrame({
            "class_name": class_names,
            "sensitivity": sensitivities,
            "specificity": specificities,
            "f1_score": f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0),
        })
        metrics["per_class"] = per_class_df.to_dict(orient="records")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
