from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: Sequence[str],
    *,
    save_path: str,
    cmap: str = "Blues",
    title: str = "Confusion matrix",
) -> None:
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap=cmap, cbar=False)
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curves_multiclass(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    *,
    save_path: str,
    title: str = "ROC curves (OvR)",
) -> None:
    """
    y_score: (N, C) probabilities or scores for each class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_classes = len(class_names)

    # Binarize labels for OvR ROC.
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = np.vstack([1 - y_bin, y_bin]).T

    plt.figure(figsize=(7.0, 5.5))
    colors = sns.color_palette("husl", n_classes)

    for i, cls_name in enumerate(class_names):
        fpr, tpr, _thr = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})", color=colors[i])

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_gwo_convergence(
    best_by_iter: List[Dict],
    *,
    save_path: str,
    title_prefix: str = "GWO convergence",
) -> None:
    iters = [d["iter"] for d in best_by_iter]
    fitness = [d["fitness"] for d in best_by_iter]
    feat_counts = [d["feature_count"] for d in best_by_iter]

    fig, ax1 = plt.subplots(figsize=(8.0, 5.5))
    ax1.plot(iters, fitness, color="tab:blue", lw=2, label="Best fitness")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best fitness", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(iters, feat_counts, color="tab:orange", lw=2, label="Feature count")
    ax2.set_ylabel("Selected feature count", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(title_prefix)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

