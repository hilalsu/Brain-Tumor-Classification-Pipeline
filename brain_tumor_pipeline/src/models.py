from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .metrics import compute_classification_metrics


@dataclass(frozen=True)
class ModelEvalResult:
    model_name: str
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_score: Optional[np.ndarray]
    confusion: np.ndarray


def _maybe_get_xgb():
    try:
        import xgboost as xgb

        return xgb
    except Exception:
        return None


def evaluate_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int,
    tune_hyperparameters: bool = False,
    xgb_max_estimators: int = 400,
) -> List[ModelEvalResult]:
    num_classes = int(len(np.unique(y_train)))
    results: List[ModelEvalResult] = []

    # k-NN baseline
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_score = knn.predict_proba(X_test) if hasattr(knn, "predict_proba") else None
    metrics = compute_classification_metrics(y_test, y_pred, y_score, average="macro")
    results.append(
        ModelEvalResult(
            model_name="knn",
            metrics=metrics,
            y_pred=y_pred,
            y_score=y_score,
            confusion=confusion_matrix(y_test, y_pred, labels=list(range(num_classes))),
        )
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_score, average="macro")
    results.append(
        ModelEvalResult(
            model_name="random_forest",
            metrics=metrics,
            y_pred=y_pred,
            y_score=y_score,
            confusion=confusion_matrix(y_test, y_pred, labels=list(range(num_classes))),
        )
    )

    # SVM
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_score = svm.predict_proba(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_score, average="macro")
    results.append(
        ModelEvalResult(
            model_name="svm_rbf",
            metrics=metrics,
            y_pred=y_pred,
            y_score=y_score,
            confusion=confusion_matrix(y_test, y_pred, labels=list(range(num_classes))),
        )
    )

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=random_state,
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_score = mlp.predict_proba(X_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_score, average="macro")
    results.append(
        ModelEvalResult(
            model_name="mlp",
            metrics=metrics,
            y_pred=y_pred,
            y_score=y_score,
            confusion=confusion_matrix(y_test, y_pred, labels=list(range(num_classes))),
        )
    )

    # XGBoost (optional)
    xgb = _maybe_get_xgb()
    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=xgb_max_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_clf.fit(X_train, y_train)
        y_pred = xgb_clf.predict(X_test)
        y_score = xgb_clf.predict_proba(X_test)
        metrics = compute_classification_metrics(y_test, y_pred, y_score, average="macro")
        results.append(
            ModelEvalResult(
                model_name="xgboost",
                metrics=metrics,
                y_pred=y_pred,
                y_score=y_score,
                confusion=confusion_matrix(y_test, y_pred, labels=list(range(num_classes))),
            )
        )

    return results

