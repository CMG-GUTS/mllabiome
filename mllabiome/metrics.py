from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from .utils import METRIC_COLUMNS


def _coerce_X_for_estimator(clf: BaseEstimator, X):
    """Use fitted feature names when an estimator was trained on a DataFrame."""
    names = getattr(clf, "feature_names_in_", None)
    if names is None:
        return X
    names = [str(c) for c in list(names)]
    if isinstance(X, pd.DataFrame):
        if list(map(str, X.columns)) == names:
            return X
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(names):
        return X
    return pd.DataFrame(arr, columns=names)


def _estimator_call(clf: BaseEstimator, method: str, X):
    return getattr(clf, method)(_coerce_X_for_estimator(clf, X))


def _predict_proba_aligned(
    clf: BaseEstimator, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        raw = np.asarray(_estimator_call(clf, "predict_proba", X), dtype=float)
        if raw.ndim == 1:
            raw = np.column_stack([1.0 - raw, raw])
        learned = np.asarray(getattr(clf, "classes_", classes), dtype=int)
        out = np.zeros((X.shape[0], len(classes)), dtype=float)
        for j, c in enumerate(learned):
            if c in set(classes.tolist()) and j < raw.shape[1]:
                out[:, int(np.where(classes == c)[0][0])] = raw[:, j]
        if out.sum(axis=1).min() <= 0:
            out = _renormalize_proba(raw, len(classes))
        return _renormalize_proba(out, len(classes)).astype(np.float32)
    if hasattr(clf, "decision_function"):
        score = np.asarray(_estimator_call(clf, "decision_function", X), dtype=float)
        if score.ndim == 1:
            p1 = expit(score)
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)
        return softmax(score, axis=1).astype(np.float32)
    pred = np.asarray(_estimator_call(clf, "predict", X), dtype=int)
    out = np.zeros((X.shape[0], len(classes)), dtype=float)
    for i, p in enumerate(pred):
        if p in set(classes.tolist()):
            out[i, int(np.where(classes == p)[0][0])] = 1.0
    return _renormalize_proba(out, len(classes)).astype(np.float32)


def _renormalize_proba(p: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.shape[1] != n_classes:
        q = np.zeros((p.shape[0], n_classes), dtype=float)
        width = min(n_classes, p.shape[1])
        q[:, :width] = p[:, :width]
        p = q
    p = np.nan_to_num(
        p, nan=1.0 / n_classes, posinf=1.0 / n_classes, neginf=1.0 / n_classes
    )
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    empty = s.squeeze() <= 1e-12
    s = np.where(s > 1e-12, s, 1.0)
    p = p / s
    if np.any(empty):
        p[empty, :] = 1.0 / n_classes
    return p


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = _renormalize_proba(y_proba, len(classes))
    out: dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}
    if len(y_true) == 0:
        return out
    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["BalAcc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["F1w"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["F1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["Precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    out["Recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    out["nMCC"] = float((matthews_corrcoef(y_true, y_pred) + 1.0) / 2.0)
    present = np.array([c for c in classes if c in set(y_true.tolist())], dtype=int)
    try:
        if len(classes) == 2:
            pos = classes[-1]
            pos_col = int(np.where(classes == pos)[0][0])
            yt = (y_true == pos).astype(int)
            if len(np.unique(yt)) == 2:
                out["AUC"] = float(roc_auc_score(yt, y_proba[:, pos_col]))
                out["PR_AUC"] = float(average_precision_score(yt, y_proba[:, pos_col]))
        elif len(present) >= 2:
            cols = [int(np.where(classes == c)[0][0]) for c in present]
            pp = _renormalize_proba(y_proba[:, cols], len(cols))
            out["AUC_macro"] = float(
                roc_auc_score(
                    y_true,
                    pp,
                    labels=present.tolist(),
                    multi_class="ovr",
                    average="macro",
                )
            )
            out["AUC_weighted"] = float(
                roc_auc_score(
                    y_true,
                    pp,
                    labels=present.tolist(),
                    multi_class="ovr",
                    average="weighted",
                )
            )
            out["AUC"] = out["AUC_macro"]
            out["PR_AUC_macro"] = float(
                average_precision_score(
                    pd.get_dummies(y_true).reindex(columns=present, fill_value=0),
                    pp,
                    average="macro",
                )
            )
    except Exception:
        pass
    return {
        k: (round(v, 6) if np.isfinite(v) else float("nan")) for k, v in out.items()
    }
