from __future__ import annotations

import numpy as np

import pandas as pd

from scipy.special import expit, softmax

from sklearn.base import BaseEstimator

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import METRIC_COLUMNS


def _coerce_X_for_estimator(clf: BaseEstimator, X):

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

        class_set = set(classes.tolist())

        for j, c in enumerate(learned):
            if c in class_set and j < raw.shape[1]:
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

    class_set = set(classes.tolist())

    for i, p in enumerate(pred):
        if p in class_set:
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


def _balanced_accuracy_known_classes(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> float:

    y_true = np.asarray(y_true, dtype=int)

    y_pred = np.asarray(y_pred, dtype=int)

    classes = np.asarray(classes, dtype=int)

    present = [int(c) for c in classes if np.any(y_true == int(c))]

    if not present:
        return float("nan")

    recalls: list[float] = []

    for c in present:
        mask = y_true == c

        support = int(mask.sum())

        if support > 0:
            recalls.append(float(np.mean(y_pred[mask] == c)))

    return float(np.mean(recalls)) if recalls else float("nan")


def _matthews_corrcoef_known_classes(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> float:

    y_true = np.asarray(y_true, dtype=int)

    y_pred = np.asarray(y_pred, dtype=int)

    base_classes = np.asarray(classes, dtype=int)

    labels = np.unique(np.concatenate([base_classes, y_true, y_pred])).astype(int)

    if labels.size == 0:
        return float("nan")

    index = {int(c): i for i, c in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)

    for yt, yp in zip(y_true, y_pred, strict=True):
        cm[index[int(yt)], index[int(yp)]] += 1

    true_marginals = cm.sum(axis=1, dtype=np.float64)

    pred_marginals = cm.sum(axis=0, dtype=np.float64)

    correct = float(np.trace(cm))

    total = float(cm.sum())

    numerator = correct * total - float(np.dot(pred_marginals, true_marginals))

    denominator_sq = (total**2 - float(np.dot(pred_marginals, pred_marginals))) * (
        total**2 - float(np.dot(true_marginals, true_marginals))
    )

    if denominator_sq <= 0.0:
        return 0.0

    return float(numerator / np.sqrt(denominator_sq))


def _multiclass_brier(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:

    index = {int(c): j for j, c in enumerate(classes)}

    one_hot = np.zeros_like(y_proba, dtype=float)

    for i, y in enumerate(np.asarray(y_true, dtype=int)):
        if int(y) in index:
            one_hot[i, index[int(y)]] = 1.0

    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> dict[str, float]:

    y_true = np.asarray(y_true, dtype=int)

    y_pred = np.asarray(y_pred, dtype=int)

    classes = np.asarray(classes, dtype=int)

    y_proba = _renormalize_proba(y_proba, len(classes))

    out: dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}

    out.update(
        {
            "Brier": float("nan"),
            "Brier_multiclass": float("nan"),
            "LogLoss": float("nan"),
        }
    )

    if len(y_true) == 0:
        return out

    out["Accuracy"] = float(accuracy_score(y_true, y_pred))

    out["BalAcc"] = _balanced_accuracy_known_classes(y_true, y_pred, classes)

    out["F1w"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    out["F1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    out["Precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )

    out["Recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )

    out["nMCC"] = float(
        (_matthews_corrcoef_known_classes(y_true, y_pred, classes) + 1.0) / 2.0
    )

    try:
        out["LogLoss"] = float(log_loss(y_true, y_proba, labels=classes.tolist()))

    except Exception:
        pass

    present = np.array([c for c in classes if c in set(y_true.tolist())], dtype=int)

    try:
        if len(classes) == 2:
            pos = classes[-1]

            pos_col = int(np.where(classes == pos)[0][0])

            yt = (y_true == pos).astype(int)

            p_pos = np.clip(y_proba[:, pos_col], 0.0, 1.0)

            out["Brier"] = float(np.mean((yt - p_pos) ** 2))

            if len(np.unique(yt)) == 2:
                out["AUC"] = float(roc_auc_score(yt, p_pos))

                out["PR_AUC"] = float(average_precision_score(yt, p_pos))

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

            out["Brier_multiclass"] = _multiclass_brier(y_true, y_proba, classes)

    except Exception:
        pass

    return {
        k: (round(v, 6) if np.isfinite(v) else float("nan")) for k, v in out.items()
    }
