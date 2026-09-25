from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import _renormalize_proba
from .statistics_common import (
    OOF_METRIC_ORDER,
    _LODO_PROTOCOLS,
    _ensure_outer_split_key,
    _repeat_id,
    _stable_seed,
)


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if str(column).startswith("proba_")]


def _prepare_oof_frame(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    out = _ensure_outer_split_key(frame)
    required = {"outer_split_key", "sample_id", "y_true"}
    if out.empty or not required.issubset(out.columns):
        return pd.DataFrame()
    pcols = _probability_columns(out)
    if not pcols:
        return pd.DataFrame()
    out = out.copy()
    out["sample_id"] = out["sample_id"].astype(str)
    if "subject_id" in out.columns:
        if out["subject_id"].isna().any():
            raise ValueError(
                "Held-out strategy predictions contain missing subject_id values."
            )
        out["_subject_id"] = out["subject_id"].astype(str)
    else:
        out["_subject_id"] = out["sample_id"]
    out["outer_split_key"] = out["outer_split_key"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    for column in pcols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    valid = out["y_true"].notna()
    valid &= np.isfinite(out[pcols].to_numpy(dtype=float)).all(axis=1)
    out = out.loc[valid].copy()
    if out.empty:
        return out
    out["y_true"] = out["y_true"].astype(int)
    if len(pcols) == 2 and "y_proba_pos" in out.columns:
        positive = pd.to_numeric(out["y_proba_pos"], errors="coerce").to_numpy(
            dtype=float
        )
        second = out[pcols[1]].to_numpy(dtype=float)
        valid_positive = np.isfinite(positive)
        if bool(np.any(valid_positive)) and not np.allclose(
            positive[valid_positive],
            second[valid_positive],
            atol=1e-7,
            rtol=1e-7,
        ):
            raise ValueError(
                "Binary probability columns are not aligned with canonical class order."
            )
    if "y_pred" in out.columns:
        pred = pd.to_numeric(out["y_pred"], errors="coerce")
        fallback = np.argmax(out[pcols].to_numpy(dtype=float), axis=1)
        out["y_pred"] = np.where(
            pred.notna(), pred.fillna(0).astype(int), fallback
        ).astype(int)
    else:
        out["y_pred"] = np.argmax(out[pcols].to_numpy(dtype=float), axis=1).astype(int)
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        out["_cluster"] = out["outer_split_key"].astype(str)
        out["_repeat"] = "r0"
        duplicate_keys = ["_cluster", "sample_id"]
        subject_cohorts = out.groupby("_subject_id", sort=False)["_cluster"].nunique()
        if bool((subject_cohorts > 1).any()):
            raise ValueError(
                "A subject_id occurs in more than one held-out LODO cohort."
            )
    else:
        out["_repeat"] = out["outer_split_key"].map(_repeat_id)
        out["_cluster"] = out["_repeat"]
        duplicate_keys = ["_repeat", "sample_id"]
    if out.duplicated(duplicate_keys).any():
        duplicates = out.loc[
            out.duplicated(duplicate_keys, keep=False), duplicate_keys
        ].head(5)
        raise ValueError(
            "Held-out strategy predictions contain duplicate inference rows: "
            + duplicates.astype(str).agg("/".join, axis=1).str.cat(sep=", ")
        )
    return out.reset_index(drop=True)


def _probability_semantics(frame: pd.DataFrame) -> dict[str, Any]:
    pcols = _probability_columns(frame)
    if not pcols:
        return {"valid": False, "reason": "probability columns are unavailable"}
    if "probability_valid" in frame.columns:
        raw = frame["probability_valid"]
        normalized = raw.map(
            lambda value: (
                value
                if isinstance(value, (bool, np.bool_))
                else str(value).strip().lower() in {"1", "true", "yes"}
            )
        )
        if not bool(normalized.all()):
            aggregation = ""
            if "aggregation_strategy" in frame.columns:
                values = (
                    frame["aggregation_strategy"].dropna().astype(str).unique().tolist()
                )
                aggregation = ", ".join(values)
            reason = (
                f"aggregation {aggregation} is not probability-valued"
                if aggregation
                else "predictions are not probability-valued"
            )
            return {"valid": False, "reason": reason}
    values = frame[pcols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return {
            "valid": False,
            "reason": "probability matrix contains non-finite values",
        }
    if np.any(values < -1e-10) or np.any(values > 1.0 + 1e-10):
        return {"valid": False, "reason": "prediction scores fall outside [0, 1]"}
    sums = values.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=1e-6):
        return {"valid": False, "reason": "prediction rows do not sum to one"}
    return {"valid": True, "reason": "probability-valued predictions"}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def _calibration_binary(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    if len(y) < 4 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    logit = np.log(p / (1.0 - p))
    low = -50.0
    high = 50.0
    for _ in range(120):
        alpha = 0.5 * (low + high)
        score = float(np.sum(y - _sigmoid(logit + alpha)))
        if score > 0.0:
            low = alpha
        else:
            high = alpha
    alpha = 0.5 * (low + high)
    X = np.column_stack([np.ones(len(y), dtype=float), logit])
    beta = np.asarray([alpha, 1.0], dtype=float)
    converged = False
    for _ in range(100):
        eta = X @ beta
        mu = _sigmoid(eta)
        weight = np.clip(mu * (1.0 - mu), 1e-12, None)
        gradient = X.T @ (y - mu)
        information = X.T @ (weight[:, None] * X)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        if not np.isfinite(step).all():
            break
        beta = beta + step
        if not np.isfinite(beta).all() or float(np.max(np.abs(beta))) > 1e3:
            break
        if float(np.max(np.abs(step))) < 1e-8:
            converged = True
            break
    if not converged or not np.isfinite(beta).all():
        return float(alpha), float("nan"), float("nan")
    mu = _sigmoid(X @ beta)
    weight = np.clip(mu * (1.0 - mu), 1e-12, None)
    information = X.T @ (weight[:, None] * X)
    try:
        eigenvalues = np.linalg.eigvalsh(information)
    except np.linalg.LinAlgError:
        return float(alpha), float("nan"), float("nan")
    if (
        not np.isfinite(eigenvalues).all()
        or float(np.min(eigenvalues)) <= 1e-8
        or float(np.max(eigenvalues) / np.min(eigenvalues)) >= 1e10
        or float(np.max(np.abs(beta))) > 100.0
    ):
        return float(alpha), float("nan"), float("nan")
    return float(alpha), float(beta[0]), float(beta[1])


def _proper_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = _renormalize_proba(np.asarray(proba, dtype=float), proba.shape[1])
    n_classes = p.shape[1]
    clipped = np.clip(p, 1e-15, 1.0)
    out: dict[str, float] = {
        "brier": float("nan"),
        "brier_multiclass": float("nan"),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(y)), y]))),
        "CalibrationInTheLarge": float("nan"),
        "CalibrationIntercept": float("nan"),
        "CalibrationSlope": float("nan"),
    }
    if n_classes == 2:
        target = (y == 1).astype(float)
        out["brier"] = float(np.mean((p[:, 1] - target) ** 2))
        citl, intercept, slope = _calibration_binary(target, p[:, 1])
        out["CalibrationInTheLarge"] = citl
        out["CalibrationIntercept"] = intercept
        out["CalibrationSlope"] = slope
    else:
        one_hot = np.eye(n_classes, dtype=float)[y]
        out["brier_multiclass"] = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
        values = []
        for class_index in range(n_classes):
            target = (y == class_index).astype(float)
            values.append(_calibration_binary(target, p[:, class_index]))
        for output_index, name in enumerate(
            ("CalibrationInTheLarge", "CalibrationIntercept", "CalibrationSlope")
        ):
            finite = np.asarray(
                [
                    value[output_index]
                    for value in values
                    if np.isfinite(value[output_index])
                ],
                dtype=float,
            )
            if len(finite):
                out[name] = float(np.mean(finite))
    return out


def _binary_auc_pr_auc_ap(
    y_true: np.ndarray, score: np.ndarray
) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    ranks = np.empty(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        ranks[start:end] = rank
        start = end
    positive_rank_sum = float(np.sum(ranks[sorted_y == 1]))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / float(
        positives * negatives
    )
    desc = np.argsort(-s, kind="mergesort")
    y_desc = y[desc]
    s_desc = s[desc]
    distinct = np.where(np.diff(s_desc) != 0)[0]
    threshold_indices = np.r_[distinct, len(y_desc) - 1]
    tp = np.cumsum(y_desc)[threshold_indices].astype(float)
    fp = (threshold_indices + 1).astype(float) - tp
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(positives)
    recall_previous = np.r_[0.0, recall[:-1]]
    ap = float(np.sum((recall - recall_previous) * precision))
    curve_recall = np.r_[0.0, recall]
    curve_precision = np.r_[1.0, precision]
    pr_auc = float(
        np.sum(
            np.diff(curve_recall) * (curve_precision[:-1] + curve_precision[1:]) * 0.5
        )
    )
    return float(auc), pr_auc, ap


def _fast_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    values = np.asarray(score, dtype=float)
    n_classes = int(values.shape[1])
    out = {
        "AUROC": float("nan"),
        "AUROC_macro": float("nan"),
        "AUROC_weighted": float("nan"),
        "AUCPR": float("nan"),
        "AUCPR_macro": float("nan"),
        "AUCPR_weighted": float("nan"),
        "AP": float("nan"),
        "AP_macro": float("nan"),
        "MCC": float("nan"),
        "F1w": float("nan"),
        "F1_macro": float("nan"),
        "Precision": float("nan"),
        "Recall": float("nan"),
        "Sensitivity": float("nan"),
        "Specificity": float("nan"),
        "PPV": float("nan"),
        "NPV": float("nan"),
        "BalAcc": float("nan"),
        "Accuracy": float("nan"),
    }
    if len(y) == 0:
        return out
    matrix = (
        np.bincount(y * n_classes + pred, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(float)
    )
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    total = float(matrix.sum())
    out["Accuracy"] = float(diagonal.sum() / total) if total > 0 else float("nan")
    recall = np.divide(
        diagonal, support, out=np.zeros(n_classes, dtype=float), where=support > 0
    )
    precision = np.divide(
        diagonal, predicted, out=np.zeros(n_classes, dtype=float), where=predicted > 0
    )
    f1 = np.divide(
        2.0 * diagonal,
        support + predicted,
        out=np.zeros(n_classes, dtype=float),
        where=(support + predicted) > 0,
    )
    active = (support + predicted) > 0
    true_active = support > 0
    if np.any(active):
        out["Precision"] = float(np.mean(precision[active]))
        out["Recall"] = float(np.mean(recall[active]))
        out["F1_macro"] = float(np.mean(f1[active]))
    if total > 0:
        out["F1w"] = float(np.sum(f1 * support) / total)
    if np.any(true_active):
        out["BalAcc"] = float(np.mean(recall[true_active]))
    trace = float(diagonal.sum())
    numerator = trace * total - float(np.dot(support, predicted))
    denominator = float(
        np.sqrt(
            max(total * total - float(np.dot(predicted, predicted)), 0.0)
            * max(total * total - float(np.dot(support, support)), 0.0)
        )
    )
    mcc = numerator / denominator if denominator > 0 else 0.0
    out["MCC"] = float(mcc)
    if n_classes == 2:
        tn = float(matrix[0, 0])
        fp = float(matrix[0, 1])
        fn = float(matrix[1, 0])
        tp = float(matrix[1, 1])
        out["Sensitivity"] = tp / (tp + fn) if tp + fn > 0 else float("nan")
        out["Specificity"] = tn / (tn + fp) if tn + fp > 0 else float("nan")
        out["PPV"] = tp / (tp + fp) if tp + fp > 0 else float("nan")
        out["NPV"] = tn / (tn + fn) if tn + fn > 0 else float("nan")
        roc_auc, pr_auc, ap = _binary_auc_pr_auc_ap((y == 1).astype(int), values[:, 1])
        out["AUROC"] = roc_auc
        out["AUCPR"] = pr_auc
        out["AP"] = ap
    else:
        auc_values = []
        auc_weights = []
        pr_auc_values = []
        pr_auc_weights = []
        ap_values = []
        for class_index in range(n_classes):
            binary = (y == class_index).astype(int)
            roc_auc, pr_auc, ap = _binary_auc_pr_auc_ap(binary, values[:, class_index])
            if np.isfinite(roc_auc):
                auc_values.append(roc_auc)
                auc_weights.append(float(np.sum(binary)))
            if np.isfinite(pr_auc):
                pr_auc_values.append(pr_auc)
                pr_auc_weights.append(float(np.sum(binary)))
            if np.isfinite(ap):
                ap_values.append(ap)
        if auc_values:
            out["AUROC_macro"] = float(np.mean(auc_values))
            out["AUROC_weighted"] = float(np.average(auc_values, weights=auc_weights))
            out["AUROC"] = out["AUROC_macro"]
        if pr_auc_values:
            out["AUCPR_macro"] = float(np.mean(pr_auc_values))
            out["AUCPR_weighted"] = float(
                np.average(pr_auc_values, weights=pr_auc_weights)
            )
            out["AUCPR"] = out["AUCPR_macro"]
        if ap_values:
            out["AP_macro"] = float(np.mean(ap_values))
    return out


def _oof_metrics(frame: pd.DataFrame, probability_valid: bool) -> dict[str, float]:
    if frame.empty:
        return {metric: float("nan") for metric in OOF_METRIC_ORDER}
    pcols = _probability_columns(frame)
    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = frame["y_pred"].astype(int).to_numpy()
    score = frame[pcols].to_numpy(dtype=float)
    values = _fast_classification_metrics(y_true, y_pred, score)
    out = {metric: float(values.get(metric, np.nan)) for metric in OOF_METRIC_ORDER}
    if probability_valid:
        out.update(_proper_metrics(y_true, score))
    return out


def _mean_metric_dicts(values: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in OOF_METRIC_ORDER:
        array = np.asarray([value.get(metric, np.nan) for value in values], dtype=float)
        array = array[np.isfinite(array)]
        out[metric] = float(np.mean(array)) if len(array) else float("nan")
    return out


def _point_oof_estimands(
    frame: pd.DataFrame,
    protocol: str,
    probability_valid: bool,
) -> dict[str, dict[str, float]]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        pooled = _oof_metrics(frame, probability_valid)
        cohort_metrics = [
            _oof_metrics(group, probability_valid)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        return {
            "pooled_sample_weighted": pooled,
            "cohort_macro_equal_weight": _mean_metric_dicts(cohort_metrics),
        }
    repeat_metrics = [
        _oof_metrics(group, probability_valid)
        for _, group in frame.groupby("_repeat", sort=True)
    ]
    return {"mean_repeat_pooled_oof": _mean_metric_dicts(repeat_metrics)}


def _resample_subjects(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if group.empty:
        return group.copy()
    subjects = [
        subject_group.reset_index(drop=True)
        for _, subject_group in group.groupby("_subject_id", sort=True)
    ]
    chosen = rng.integers(0, len(subjects), size=len(subjects))
    return pd.concat([subjects[int(index)] for index in chosen], ignore_index=True)


def _bootstrap_oof_estimands(
    frame: pd.DataFrame,
    protocol: str,
    probability_valid: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    protocol_key = str(protocol).lower()
    names = (
        ("pooled_sample_weighted", "cohort_macro_equal_weight")
        if protocol_key in _LODO_PROTOCOLS
        else ("mean_repeat_pooled_oof",)
    )
    storage = {
        name: {
            metric: np.full(int(n_bootstrap), np.nan, dtype=float)
            for metric in OOF_METRIC_ORDER
        }
        for name in names
    }
    if frame.empty or int(n_bootstrap) <= 0:
        return storage
    if protocol_key in _LODO_PROTOCOLS:
        groups = [
            group.reset_index(drop=True)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        for bootstrap_index in range(int(n_bootstrap)):
            chosen = rng.integers(0, len(groups), size=len(groups))
            sampled_groups = [
                _resample_subjects(groups[int(index)], rng) for index in chosen
            ]
            pooled_metrics = _oof_metrics(
                pd.concat(sampled_groups, ignore_index=True), probability_valid
            )
            macro_metrics = _mean_metric_dicts(
                [_oof_metrics(group, probability_valid) for group in sampled_groups]
            )
            for metric in OOF_METRIC_ORDER:
                storage["pooled_sample_weighted"][metric][bootstrap_index] = (
                    pooled_metrics.get(metric, np.nan)
                )
                storage["cohort_macro_equal_weight"][metric][bootstrap_index] = (
                    macro_metrics.get(metric, np.nan)
                )
        return storage
    for bootstrap_index in range(int(n_bootstrap)):
        sampled = _resample_subjects(frame, rng)
        metrics = [
            _oof_metrics(group, probability_valid)
            for _, group in sampled.groupby("_repeat", sort=True)
        ]
        averaged = _mean_metric_dicts(metrics)
        for metric in OOF_METRIC_ORDER:
            storage["mean_repeat_pooled_oof"][metric][bootstrap_index] = averaged.get(
                metric, np.nan
            )
    return storage


def _performance_rows(
    strategy: str,
    frame: pd.DataFrame,
    protocol: str,
    probability_valid: bool,
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, Any]]:
    point = _point_oof_estimands(frame, protocol, probability_valid)
    boot = _bootstrap_oof_estimands(
        frame,
        protocol,
        probability_valid,
        n_bootstrap,
        np.random.default_rng(_stable_seed(random_state, "oof", strategy)),
    )
    rows: list[dict[str, Any]] = []
    for estimand, metrics in point.items():
        for metric in OOF_METRIC_ORDER:
            estimate = float(metrics.get(metric, np.nan))
            if not np.isfinite(estimate):
                continue
            samples = np.asarray(boot[estimand][metric], dtype=float)
            samples = samples[np.isfinite(samples)]
            low = high = np.nan
            if len(samples):
                low, high = np.quantile(samples, [0.025, 0.975])
            rows.append(
                {
                    "Strategy": strategy,
                    "estimand": estimand,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(low) if np.isfinite(low) else np.nan,
                    "ci_high": float(high) if np.isfinite(high) else np.nan,
                    "n_bootstrap_valid": int(len(samples)),
                    "n_rows": int(len(frame)),
                }
            )
    return rows


def _reliability_rows(
    strategy: str,
    frame: pd.DataFrame,
    probability_valid: bool,
    calibration_bins: int,
) -> list[dict[str, Any]]:
    if frame.empty or not probability_valid:
        return []
    pcols = _probability_columns(frame)
    proba = _renormalize_proba(frame[pcols].to_numpy(dtype=float), len(pcols))
    y_true = frame["y_true"].astype(int).to_numpy()
    classes = [len(pcols) - 1] if len(pcols) == 2 else list(range(len(pcols)))
    edges = np.linspace(0.0, 1.0, int(calibration_bins) + 1)
    rows: list[dict[str, Any]] = []
    for class_index in classes:
        target = (y_true == class_index).astype(float)
        score = proba[:, class_index]
        bins = np.digitize(score, edges[1:-1], right=True)
        for bin_index in range(int(calibration_bins)):
            mask = bins == bin_index
            if not np.any(mask):
                continue
            rows.append(
                {
                    "Strategy": strategy,
                    "class_index": int(class_index),
                    "bin": int(bin_index + 1),
                    "n": int(np.sum(mask)),
                    "mean_predicted_probability": float(np.mean(score[mask])),
                    "observed_frequency": float(np.mean(target[mask])),
                    "bin_lower": float(edges[bin_index]),
                    "bin_upper": float(edges[bin_index + 1]),
                }
            )
    return rows
