from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .utils import METRIC_COLUMNS, REGRESSION_METRIC_COLUMNS


def canonical_metric_name(metric: str) -> str:
    text = str(metric).strip()
    key = text.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "auc": "AUROC",
        "roc_auc": "AUROC",
        "auroc": "AUROC",
        "auc_macro": "AUROC_macro",
        "roc_auc_macro": "AUROC_macro",
        "auroc_macro": "AUROC_macro",
        "auc_weighted": "AUROC_weighted",
        "roc_auc_weighted": "AUROC_weighted",
        "auroc_weighted": "AUROC_weighted",
        "pr_auc": "AUCPR",
        "prauc": "AUCPR",
        "aucpr": "AUCPR",
        "pr_auc_macro": "AUCPR_macro",
        "prauc_macro": "AUCPR_macro",
        "aucpr_macro": "AUCPR_macro",
        "pr_auc_weighted": "AUCPR_weighted",
        "prauc_weighted": "AUCPR_weighted",
        "aucpr_weighted": "AUCPR_weighted",
        "average_precision": "AP",
        "ap": "AP",
        "mcc": "MCC",
        "nmcc": "MCC",
        "f1w": "F1w",
        "precision": "Precision",
        "recall": "Recall",
        "logloss": "log_loss",
        "log_loss": "log_loss",
        "subject_macro_logloss": "subject_macro_log_loss",
        "subject_macro_log_loss": "subject_macro_log_loss",
        "cohort_macro_logloss": "cohort_macro_log_loss",
        "cohort_macro_log_loss": "cohort_macro_log_loss",
        "brier_loss": "brier",
        "brier": "brier",
    }
    return aliases.get(key, text)


def _metric_key(metric: str) -> str:
    return canonical_metric_name(metric).casefold().replace("-", "_").replace(" ", "_")


def metric_is_loss(metric: str) -> bool:
    return _metric_key(metric) in {
        "log_loss",
        "subject_macro_log_loss",
        "cohort_macro_log_loss",
        "brier",
        "brier_multiclass",
        "mae",
        "mse",
        "rmse",
        "medae",
    }


def metric_better(
    candidate: float, incumbent: float, metric: str, tol: float = 1e-12
) -> bool:
    if metric_is_loss(metric):
        return float(candidate) < float(incumbent) - float(tol)
    return float(candidate) > float(incumbent) + float(tol)


def metric_passes_threshold(score: float, threshold: float, metric: str) -> bool:
    if not np.isfinite(score) or not np.isfinite(threshold):
        return False
    raw = str(metric).strip().casefold().replace("-", "_").replace(" ", "_")
    boundary = float(threshold)
    if raw == "nmcc":
        boundary = 2.0 * boundary - 1.0
    if metric_is_loss(metric):
        return float(score) <= boundary
    return float(score) >= boundary


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


def _validate_probability_matrix(
    values: np.ndarray,
    n_rows: int,
    n_classes: int,
) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim == 1 and n_classes == 2:
        p = np.column_stack([1.0 - p, p])
    if p.ndim != 2:
        raise ValueError(
            f"predict_proba must return a 2D matrix; received shape {p.shape}."
        )
    if p.shape != (n_rows, n_classes):
        raise ValueError(
            f"predict_proba returned shape {p.shape}; expected {(n_rows, n_classes)}."
        )
    if not np.all(np.isfinite(p)):
        raise ValueError("predict_proba returned NaN or infinite values.")
    tolerance = 1e-7
    if np.any(p < -tolerance) or np.any(p > 1.0 + tolerance):
        raise ValueError("predict_proba returned values outside [0, 1].")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("predict_proba rows must sum to 1.")
    return np.clip(p, 0.0, 1.0)


def _predict_proba_aligned(
    clf: BaseEstimator, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    if not callable(getattr(clf, "predict_proba", None)):
        raise TypeError(
            f"{type(clf).__name__} does not provide predict_proba. Use a calibrated estimator before evaluation."
        )
    classes = np.asarray(classes, dtype=int)
    raw = np.asarray(_estimator_call(clf, "predict_proba", X), dtype=float)
    learned = np.asarray(getattr(clf, "classes_", classes), dtype=int)
    if learned.ndim != 1:
        raise ValueError("Estimator classes_ must be one-dimensional.")
    if len(learned) != len(classes) or set(learned.tolist()) != set(classes.tolist()):
        raise ValueError(
            f"Estimator classes {learned.tolist()!r} do not match expected classes {classes.tolist()!r}."
        )
    if raw.ndim == 1 and len(learned) == 2:
        raw = np.column_stack([1.0 - raw, raw])
    if raw.ndim != 2 or raw.shape[1] != len(learned):
        raise ValueError(
            f"predict_proba returned shape {raw.shape}; expected {len(learned)} class columns."
        )
    aligned = np.empty((raw.shape[0], len(classes)), dtype=float)
    for j, klass in enumerate(classes):
        source = np.flatnonzero(learned == klass)
        if len(source) != 1:
            raise ValueError(
                f"Could not align probability column for class {int(klass)}."
            )
        aligned[:, j] = raw[:, int(source[0])]
    return _validate_probability_matrix(aligned, len(X), len(classes)).astype(
        np.float32
    )


def _renormalize_proba(p: np.ndarray, n_classes: int) -> np.ndarray:
    values = np.asarray(p, dtype=float)
    if values.ndim == 1 and n_classes == 2:
        values = np.column_stack([1.0 - values, values])
    if values.ndim != 2 or values.shape[1] != n_classes:
        raise ValueError(
            f"Probability matrix must have {n_classes} columns; received shape {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Probability matrix contains NaN or infinite values.")
    if np.any(values < 0.0):
        raise ValueError("Probability matrix contains negative values.")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Probability matrix contains a row with zero total mass.")
    return values / row_sums


def log_loss_score(
    y_true: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> float:
    truth = np.asarray(y_true, dtype=int).reshape(-1)
    labels = np.asarray(classes, dtype=int).reshape(-1)
    probability = _renormalize_proba(np.asarray(y_proba, dtype=float), len(labels))
    if len(truth) != len(probability):
        raise ValueError("y_true and y_proba must contain the same number of rows.")
    mapping = {int(label): index for index, label in enumerate(labels)}
    try:
        columns = np.asarray([mapping[int(value)] for value in truth], dtype=int)
    except KeyError as exc:
        raise ValueError(
            "y_true contains classes outside the declared class set."
        ) from exc
    eps = np.finfo(float).eps
    picked = np.clip(probability[np.arange(len(truth)), columns], eps, 1.0)
    return float(-np.mean(np.log(picked))) if len(picked) else float("nan")


def grouped_log_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    groups: np.ndarray,
) -> float:
    truth = np.asarray(y_true, dtype=int).reshape(-1)
    labels = np.asarray(classes, dtype=int).reshape(-1)
    group_values = np.asarray(groups, dtype=object).reshape(-1)
    probability = _renormalize_proba(np.asarray(y_proba, dtype=float), len(labels))
    if len(truth) != len(group_values) or len(truth) != len(probability):
        raise ValueError(
            "y_true, y_proba, and groups must contain the same number of rows."
        )
    if not len(truth):
        return float("nan")
    mapping = {int(label): index for index, label in enumerate(labels)}
    try:
        columns = np.asarray([mapping[int(value)] for value in truth], dtype=int)
    except KeyError as exc:
        raise ValueError(
            "y_true contains classes outside the declared class set."
        ) from exc
    eps = np.finfo(float).eps
    losses = -np.log(np.clip(probability[np.arange(len(truth)), columns], eps, 1.0))
    codes, uniques = pd.factorize(
        pd.Series(group_values, dtype="object").astype(str), sort=False
    )
    if len(uniques) == 0:
        return float("nan")
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    totals = np.bincount(codes, weights=losses, minlength=len(uniques))
    valid = counts > 0.0
    means = totals[valid] / counts[valid]
    finite = means[np.isfinite(means)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> float:
    recalls = []
    for klass in classes:
        mask = y_true == klass
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == klass)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _matthews_corrcoef(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray
) -> float:
    mapping = {int(klass): i for i, klass in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=float)
    for truth, pred in zip(y_true, y_pred):
        if int(truth) not in mapping or int(pred) not in mapping:
            raise ValueError(
                "y_true or y_pred contains a class outside the declared class set."
            )
        matrix[mapping[int(truth)], mapping[int(pred)]] += 1.0
    total = float(matrix.sum())
    if total <= 0.0:
        return float("nan")
    trace = float(np.trace(matrix))
    true_totals = matrix.sum(axis=1)
    pred_totals = matrix.sum(axis=0)
    numerator = trace * total - float(np.dot(true_totals, pred_totals))
    denominator = float(
        np.sqrt(
            (total * total - float(np.dot(pred_totals, pred_totals)))
            * (total * total - float(np.dot(true_totals, true_totals)))
        )
    )
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _binary_diagnostic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    positive_class: int | None = None,
) -> dict[str, float]:
    classes = np.asarray(classes, dtype=int)
    if len(classes) != 2:
        return {
            "Sensitivity": float("nan"),
            "Specificity": float("nan"),
            "PPV": float("nan"),
            "NPV": float("nan"),
        }
    positive = int(classes[-1] if positive_class is None else positive_class)
    matches = np.flatnonzero(classes == positive)
    if len(matches) != 1:
        raise ValueError(
            f"positive_class={positive!r} is not in classes {classes.tolist()!r}."
        )
    negative = int(classes[0] if int(classes[1]) == positive else classes[1])
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((truth == positive) & (pred == positive)))
    fn = int(np.sum((truth == positive) & (pred == negative)))
    tn = int(np.sum((truth == negative) & (pred == negative)))
    fp = int(np.sum((truth == negative) & (pred == positive)))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator > 0 else float("nan")

    return {
        "Sensitivity": ratio(tp, tp + fn),
        "Specificity": ratio(tn, tn + fp),
        "PPV": ratio(tp, tp + fp),
        "NPV": ratio(tn, tn + fn),
    }


def _score_matrix(y_score: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    score = np.asarray(y_score, dtype=float)
    if score.ndim == 1:
        if n_classes != 2:
            raise ValueError(
                "A one-dimensional score is only valid for binary classification."
            )
        score = np.column_stack([-score, score])
    if score.shape != (n_rows, n_classes):
        raise ValueError(
            f"Score matrix has shape {score.shape}; expected {(n_rows, n_classes)}."
        )
    if not np.all(np.isfinite(score)):
        raise ValueError("Score matrix contains NaN or infinite values.")
    return score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    positive_class: int | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.asarray(classes, dtype=int)
    out: dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}
    if len(y_true) == 0:
        return out
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shape.")
    score = _score_matrix(y_proba, len(y_true), len(classes))
    declared = set(classes.tolist())
    if not set(np.unique(y_true).tolist()).issubset(declared):
        raise ValueError("y_true contains classes outside the declared class set.")
    if not set(np.unique(y_pred).tolist()).issubset(declared):
        raise ValueError("y_pred contains classes outside the declared class set.")

    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["BalAcc"] = _balanced_accuracy(y_true, y_pred, classes)
    out["F1w"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["F1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["Precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    out["Recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    out.update(_binary_diagnostic_metrics(y_true, y_pred, classes, positive_class))
    mcc = _matthews_corrcoef(y_true, y_pred, classes)
    out["MCC"] = float(mcc) if np.isfinite(mcc) else float("nan")

    class_to_col = {int(klass): j for j, klass in enumerate(classes)}
    true_cols = np.asarray([class_to_col[int(value)] for value in y_true], dtype=int)
    probability = _renormalize_proba(score, len(classes))
    out["log_loss"] = log_loss_score(y_true, probability, classes)
    target = np.zeros_like(probability)
    target[np.arange(len(y_true)), true_cols] = 1.0
    if len(classes) == 2:
        pos = int(classes[-1] if positive_class is None else positive_class)
        matches = np.flatnonzero(classes == pos)
        if len(matches) != 1:
            raise ValueError(
                f"positive_class={pos!r} is not in classes {classes.tolist()!r}."
            )
        pos_col = int(matches[0])
        binary = (y_true == pos).astype(float)
        out["brier"] = float(np.mean((probability[:, pos_col] - binary) ** 2))
    else:
        out["brier"] = float(np.mean(np.sum((probability - target) ** 2, axis=1)))

    if len(classes) == 2:
        pos = int(classes[-1] if positive_class is None else positive_class)
        matches = np.flatnonzero(classes == pos)
        if len(matches) != 1:
            raise ValueError(
                f"positive_class={pos!r} is not in classes {classes.tolist()!r}."
            )
        pos_col = int(matches[0])
        binary = (y_true == pos).astype(int)
        if len(np.unique(binary)) == 2:
            binary_score = score[:, pos_col]
            precision, recall, _ = precision_recall_curve(binary, binary_score)
            out["AUROC"] = float(roc_auc_score(binary, binary_score))
            out["AUCPR"] = float(auc(recall, precision))
            out["AP"] = float(average_precision_score(binary, binary_score))
    else:
        auc_values = []
        auc_weights = []
        pr_auc_values = []
        pr_auc_weights = []
        ap_values = []
        for j, klass in enumerate(classes):
            binary = (y_true == klass).astype(int)
            positives = int(binary.sum())
            negatives = int(len(binary) - positives)
            if positives == 0 or negatives == 0:
                continue
            class_score = score[:, j]
            precision, recall, _ = precision_recall_curve(binary, class_score)
            auc_values.append(float(roc_auc_score(binary, class_score)))
            auc_weights.append(float(positives))
            pr_auc_values.append(float(auc(recall, precision)))
            pr_auc_weights.append(float(positives))
            ap_values.append(float(average_precision_score(binary, class_score)))
        if auc_values:
            out["AUROC_macro"] = float(np.mean(auc_values))
            out["AUROC_weighted"] = float(np.average(auc_values, weights=auc_weights))
            out["AUROC"] = out["AUROC_macro"]
            out["AUCPR_macro"] = float(np.mean(pr_auc_values))
            out["AUCPR_weighted"] = float(
                np.average(pr_auc_values, weights=pr_auc_weights)
            )
            out["AUCPR"] = out["AUCPR_macro"]
            out["AP_macro"] = float(np.mean(ap_values))

    return {
        key: round(value, 6) if np.isfinite(value) else float("nan")
        for key, value in out.items()
    }


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    out = {key: float("nan") for key in REGRESSION_METRIC_COLUMNS}
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have identical shape.")
    if len(truth) == 0:
        return out
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Regression predictions and targets must be finite.")
    mse = float(mean_squared_error(truth, pred))
    out["MAE"] = float(mean_absolute_error(truth, pred))
    out["MSE"] = mse
    out["RMSE"] = float(np.sqrt(mse))
    out["MedAE"] = float(median_absolute_error(truth, pred))
    out["ExplainedVariance"] = float(explained_variance_score(truth, pred))
    out["R2"] = float(r2_score(truth, pred)) if len(truth) >= 2 else float("nan")
    if len(truth) >= 2 and np.std(truth) > 0 and np.std(pred) > 0:
        out["PearsonR"] = float(pearsonr(truth, pred).statistic)
        out["SpearmanR"] = float(spearmanr(truth, pred).statistic)
    return {
        key: round(value, 6) if np.isfinite(value) else float("nan")
        for key, value in out.items()
    }
