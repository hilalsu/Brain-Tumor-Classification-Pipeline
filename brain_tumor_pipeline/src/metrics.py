from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    *,
    average: str = "macro",
) -> Dict[str, float]:
    """
    y_score should be probabilities or decision scores shaped (N, num_classes).
    For ROC-AUC, we use One-vs-Rest.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    if y_score is not None:
        # ROC-AUC needs per-class scores.
        num_classes = len(np.unique(y_true))
        if y_score.ndim == 1:
            # Binary case: interpret as score for positive class.
            roc_auc = roc_auc_score(y_true, y_score)
        else:
            roc_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average=average)
        metrics["roc_auc"] = float(roc_auc)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def confusion_matrix_for_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

