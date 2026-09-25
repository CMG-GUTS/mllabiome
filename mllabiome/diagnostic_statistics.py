from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve as sklearn_roc_curve

from .oof_statistics import _probability_columns, _resample_subjects
from .statistics_common import _LODO_PROTOCOLS, DIAGNOSTIC_METRICS, _stable_seed


def _validated_diagnostic_thresholds(values: Any) -> tuple[float, ...]:
    if values is None:
        return ()
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "diagnostic_thresholds must be an iterable of probabilities."
        ) from exc
    thresholds: list[float] = []
    for value in raw:
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "diagnostic_thresholds must contain numeric probabilities."
            ) from exc
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(
                "diagnostic_thresholds must contain finite probabilities strictly between 0 and 1."
            )
        if not any(abs(threshold - existing) <= 1e-12 for existing in thresholds):
            thresholds.append(threshold)
    return tuple(sorted(thresholds))


def _decision_curve_grid(
    minimum: float,
    maximum: float,
    points: int,
) -> np.ndarray:
    low = float(minimum)
    high = float(maximum)
    count = int(points)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Decision-curve threshold bounds must be finite.")
    if not 0.0 < low < high < 1.0:
        raise ValueError(
            "Decision-curve thresholds must satisfy 0 < minimum < maximum < 1."
        )
    if count < 2:
        raise ValueError("decision_curve_points must be at least 2.")
    return np.linspace(low, high, count, dtype=float)


def _binary_threshold_counts(
    frame: pd.DataFrame,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    pcols = _probability_columns(frame)
    if len(pcols) != 2:
        raise ValueError(
            "Binary diagnostic thresholds require exactly two probability columns."
        )
    y = frame["y_true"].astype(int).to_numpy()
    score = frame[pcols[1]].to_numpy(dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    positive_scores = np.sort(score[y == 1])
    negative_scores = np.sort(score[y == 0])
    tp = len(positive_scores) - np.searchsorted(
        positive_scores, thresholds, side="left"
    )
    fp = len(negative_scores) - np.searchsorted(
        negative_scores, thresholds, side="left"
    )
    fn = len(positive_scores) - tp
    tn = len(negative_scores) - fp
    return {
        "TN": tn.astype(float),
        "FP": fp.astype(float),
        "FN": fn.astype(float),
        "TP": tp.astype(float),
    }


def _safe_ratio_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _diagnostic_values_from_counts(
    counts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]
    total = tn + fp + fn + tp
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )
    mcc = np.where(total > 0.0, mcc, np.nan)
    return {
        "Sensitivity": _safe_ratio_array(tp, tp + fn),
        "Specificity": _safe_ratio_array(tn, tn + fp),
        "PPV": _safe_ratio_array(tp, tp + fp),
        "NPV": _safe_ratio_array(tn, tn + fn),
        "Accuracy": _safe_ratio_array(tp + tn, total),
        "MCC": mcc,
    }


def _threshold_diagnostic_values(
    frame: pd.DataFrame,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    return _diagnostic_values_from_counts(_binary_threshold_counts(frame, thresholds))


def _nanmean_vectors(values: list[np.ndarray], size: int) -> np.ndarray:
    if not values:
        return np.full(int(size), np.nan, dtype=float)
    stacked = np.vstack(values).astype(float)
    valid = np.isfinite(stacked)
    count = valid.sum(axis=0)
    total = np.where(valid, stacked, 0.0).sum(axis=0)
    return np.divide(
        total,
        count,
        out=np.full(int(size), np.nan, dtype=float),
        where=count > 0,
    )


def _mean_diagnostic_vectors(
    values: list[dict[str, np.ndarray]],
    size: int,
) -> dict[str, np.ndarray]:
    return {
        metric: _nanmean_vectors([value[metric] for value in values], size)
        for metric in DIAGNOSTIC_METRICS
    }


def _threshold_point_estimands(
    frame: pd.DataFrame,
    protocol: str,
    thresholds: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        cohort_values = [
            _threshold_diagnostic_values(group, thresholds)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        return {
            "pooled_sample_weighted": _threshold_diagnostic_values(frame, thresholds),
            "cohort_macro_equal_weight": _mean_diagnostic_vectors(
                cohort_values, len(thresholds)
            ),
        }
    repeat_values = [
        _threshold_diagnostic_values(group, thresholds)
        for _, group in frame.groupby("_repeat", sort=True)
    ]
    return {
        "mean_repeat_pooled_oof": _mean_diagnostic_vectors(
            repeat_values, len(thresholds)
        )
    }


def _bootstrap_threshold_estimands(
    frame: pd.DataFrame,
    protocol: str,
    thresholds: np.ndarray,
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
            metric: np.full((int(n_bootstrap), len(thresholds)), np.nan, dtype=float)
            for metric in DIAGNOSTIC_METRICS
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
            pooled = _threshold_diagnostic_values(
                pd.concat(sampled_groups, ignore_index=True), thresholds
            )
            macro = _mean_diagnostic_vectors(
                [
                    _threshold_diagnostic_values(group, thresholds)
                    for group in sampled_groups
                ],
                len(thresholds),
            )
            for metric in DIAGNOSTIC_METRICS:
                storage["pooled_sample_weighted"][metric][bootstrap_index] = pooled[
                    metric
                ]
                storage["cohort_macro_equal_weight"][metric][bootstrap_index] = macro[
                    metric
                ]
        return storage
    for bootstrap_index in range(int(n_bootstrap)):
        sampled = _resample_subjects(frame, rng)
        values = [
            _threshold_diagnostic_values(group, thresholds)
            for _, group in sampled.groupby("_repeat", sort=True)
        ]
        averaged = _mean_diagnostic_vectors(values, len(thresholds))
        for metric in DIAGNOSTIC_METRICS:
            storage["mean_repeat_pooled_oof"][metric][bootstrap_index] = averaged[
                metric
            ]
    return storage


def _threshold_metric_rows(
    strategy: str,
    frame: pd.DataFrame,
    protocol: str,
    thresholds: tuple[float, ...],
    n_bootstrap: int,
    random_state: int,
    positive_class_label: str,
) -> list[dict[str, Any]]:
    if not thresholds:
        return []
    threshold_array = np.asarray(thresholds, dtype=float)
    point = _threshold_point_estimands(frame, protocol, threshold_array)
    boot = _bootstrap_threshold_estimands(
        frame,
        protocol,
        threshold_array,
        n_bootstrap,
        np.random.default_rng(
            _stable_seed(random_state, "diagnostic_threshold", strategy)
        ),
    )
    rows: list[dict[str, Any]] = []
    for estimand, values in point.items():
        for threshold_index, threshold in enumerate(threshold_array):
            for metric in DIAGNOSTIC_METRICS:
                estimate = float(values[metric][threshold_index])
                if not np.isfinite(estimate):
                    continue
                samples = boot[estimand][metric][:, threshold_index]
                samples = samples[np.isfinite(samples)]
                low = high = np.nan
                if len(samples):
                    low, high = np.quantile(samples, [0.025, 0.975])
                rows.append(
                    {
                        "Strategy": strategy,
                        "estimand": estimand,
                        "threshold": float(threshold),
                        "positive_class_index": 1,
                        "positive_class_label": positive_class_label,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": float(low) if np.isfinite(low) else np.nan,
                        "ci_high": float(high) if np.isfinite(high) else np.nan,
                        "n_bootstrap_valid": len(samples),
                        "n_rows": len(frame),
                    }
                )
    return rows


def _binary_confusion(
    frame: pd.DataFrame,
    threshold: float | None,
) -> np.ndarray:
    y = frame["y_true"].astype(int).to_numpy()
    if threshold is None:
        pred = frame["y_pred"].astype(int).to_numpy()
    else:
        pcols = _probability_columns(frame)
        pred = (frame[pcols[1]].to_numpy(dtype=float) >= float(threshold)).astype(int)
    return np.bincount(y * 2 + pred, minlength=4).reshape(2, 2).astype(float)


def _confusion_components(matrices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not matrices:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
    mean_count = np.mean(np.stack(matrices, axis=0), axis=0)
    fractions = []
    for matrix in matrices:
        row_total = matrix.sum(axis=1, keepdims=True)
        fractions.append(
            np.divide(
                matrix,
                row_total,
                out=np.full_like(matrix, np.nan, dtype=float),
                where=row_total > 0.0,
            )
        )
    stacked = np.stack(fractions, axis=0)
    valid = np.isfinite(stacked)
    count = valid.sum(axis=0)
    total = np.where(valid, stacked, 0.0).sum(axis=0)
    mean_fraction = np.divide(
        total,
        count,
        out=np.full_like(mean_count, np.nan, dtype=float),
        where=count > 0,
    )
    return mean_count, mean_fraction


def _confusion_estimands(
    frame: pd.DataFrame,
    protocol: str,
    threshold: float | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        pooled = _binary_confusion(frame, threshold)
        pooled_total = pooled.sum(axis=1, keepdims=True)
        pooled_fraction = np.divide(
            pooled,
            pooled_total,
            out=np.full_like(pooled, np.nan, dtype=float),
            where=pooled_total > 0.0,
        )
        cohorts = [
            _binary_confusion(group, threshold)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        macro_count, macro_fraction = _confusion_components(cohorts)
        return {
            "pooled_sample_weighted": (pooled, pooled_fraction, 1),
            "cohort_macro_equal_weight": (
                macro_count,
                macro_fraction,
                len(cohorts),
            ),
        }
    repeats = [
        _binary_confusion(group, threshold)
        for _, group in frame.groupby("_repeat", sort=True)
    ]
    mean_count, mean_fraction = _confusion_components(repeats)
    return {"mean_repeat_pooled_oof": (mean_count, mean_fraction, len(repeats))}


def _confusion_rows(
    strategy: str,
    frame: pd.DataFrame,
    protocol: str,
    thresholds: tuple[float, ...],
    class_labels: tuple[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specifications: list[tuple[str, float | None]] = [
        ("probability_threshold", value) for value in thresholds
    ]
    for threshold_type, threshold in specifications:
        estimands = _confusion_estimands(frame, protocol, threshold)
        for estimand, (counts, fractions, n_units) in estimands.items():
            for actual in range(2):
                for predicted in range(2):
                    rows.append(
                        {
                            "Strategy": strategy,
                            "estimand": estimand,
                            "threshold_type": threshold_type,
                            "threshold": np.nan
                            if threshold is None
                            else float(threshold),
                            "positive_class_index": 1,
                            "positive_class_label": class_labels[1],
                            "actual_class_index": actual,
                            "actual_class_label": class_labels[actual],
                            "predicted_class_index": predicted,
                            "predicted_class_label": class_labels[predicted],
                            "count_estimate": float(counts[actual, predicted]),
                            "row_fraction": float(fractions[actual, predicted]),
                            "n_averaged_units": int(n_units),
                        }
                    )
    return rows


def _operating_confusion_matrix(
    frame: pd.DataFrame,
    n_classes: int,
) -> np.ndarray:
    y = frame["y_true"].astype(int).to_numpy()
    pred = frame["y_pred"].astype(int).to_numpy()
    return (
        np.bincount(
            y * int(n_classes) + pred,
            minlength=int(n_classes) * int(n_classes),
        )
        .reshape(int(n_classes), int(n_classes))
        .astype(float)
    )


def _operating_confusion_estimands(
    frame: pd.DataFrame,
    protocol: str,
    n_classes: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        pooled = _operating_confusion_matrix(frame, n_classes)
        pooled_total = pooled.sum(axis=1, keepdims=True)
        pooled_fraction = np.divide(
            pooled,
            pooled_total,
            out=np.full_like(pooled, np.nan, dtype=float),
            where=pooled_total > 0.0,
        )
        cohorts = [
            _operating_confusion_matrix(group, n_classes)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        macro_count, macro_fraction = _confusion_components(cohorts)
        return {
            "pooled_sample_weighted": (pooled, pooled_fraction, 1),
            "cohort_macro_equal_weight": (
                macro_count,
                macro_fraction,
                len(cohorts),
            ),
        }
    repeats = [
        _operating_confusion_matrix(group, n_classes)
        for _, group in frame.groupby("_repeat", sort=True)
    ]
    mean_count, mean_fraction = _confusion_components(repeats)
    return {"mean_repeat_pooled_oof": (mean_count, mean_fraction, len(repeats))}


def _operating_confusion_rows(
    strategy: str,
    frame: pd.DataFrame,
    protocol: str,
    class_labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_classes = len(class_labels)
    estimands = _operating_confusion_estimands(frame, protocol, n_classes)
    for estimand, (counts, fractions, n_units) in estimands.items():
        for actual in range(n_classes):
            for predicted in range(n_classes):
                rows.append(
                    {
                        "Strategy": strategy,
                        "estimand": estimand,
                        "threshold_type": "model_prediction",
                        "threshold": np.nan,
                        "positive_class_index": 1 if n_classes == 2 else np.nan,
                        "positive_class_label": class_labels[1]
                        if n_classes == 2
                        else "",
                        "actual_class_index": actual,
                        "actual_class_label": class_labels[actual],
                        "predicted_class_index": predicted,
                        "predicted_class_label": class_labels[predicted],
                        "count_estimate": float(counts[actual, predicted]),
                        "row_fraction": float(fractions[actual, predicted]),
                        "n_averaged_units": int(n_units),
                    }
                )
    return rows


def _decision_curve_values(
    frame: pd.DataFrame,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    counts = _binary_threshold_counts(frame, thresholds)
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    tp = counts["TP"]
    total = tn + fp + fn + tp
    prevalence = _safe_ratio_array(tp + fn, total)
    odds = thresholds / (1.0 - thresholds)
    net_benefit = _safe_ratio_array(tp, total) - _safe_ratio_array(fp, total) * odds
    treat_all = prevalence - (1.0 - prevalence) * odds
    standardized = np.divide(
        net_benefit,
        prevalence,
        out=np.full_like(net_benefit, np.nan, dtype=float),
        where=prevalence > 0.0,
    )
    return {
        "net_benefit": net_benefit,
        "treat_all_net_benefit": treat_all,
        "treat_none_net_benefit": np.zeros_like(net_benefit),
        "standardized_net_benefit": standardized,
        "prevalence": prevalence,
    }


def _mean_decision_vectors(
    values: list[dict[str, np.ndarray]],
    size: int,
) -> dict[str, np.ndarray]:
    names = (
        "net_benefit",
        "treat_all_net_benefit",
        "treat_none_net_benefit",
        "standardized_net_benefit",
        "prevalence",
    )
    return {
        name: _nanmean_vectors([value[name] for value in values], size)
        for name in names
    }


def _decision_curve_point_estimands(
    frame: pd.DataFrame,
    protocol: str,
    thresholds: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        cohort_values = [
            _decision_curve_values(group, thresholds)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
        return {
            "pooled_sample_weighted": _decision_curve_values(frame, thresholds),
            "cohort_macro_equal_weight": _mean_decision_vectors(
                cohort_values, len(thresholds)
            ),
        }
    repeat_values = [
        _decision_curve_values(group, thresholds)
        for _, group in frame.groupby("_repeat", sort=True)
    ]
    return {
        "mean_repeat_pooled_oof": _mean_decision_vectors(repeat_values, len(thresholds))
    }


def _bootstrap_decision_curve(
    frame: pd.DataFrame,
    protocol: str,
    thresholds: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    protocol_key = str(protocol).lower()
    names = (
        ("pooled_sample_weighted", "cohort_macro_equal_weight")
        if protocol_key in _LODO_PROTOCOLS
        else ("mean_repeat_pooled_oof",)
    )
    storage = {
        name: np.full((int(n_bootstrap), len(thresholds)), np.nan, dtype=float)
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
            pooled = _decision_curve_values(
                pd.concat(sampled_groups, ignore_index=True), thresholds
            )
            macro = _mean_decision_vectors(
                [_decision_curve_values(group, thresholds) for group in sampled_groups],
                len(thresholds),
            )
            storage["pooled_sample_weighted"][bootstrap_index] = pooled["net_benefit"]
            storage["cohort_macro_equal_weight"][bootstrap_index] = macro["net_benefit"]
        return storage
    for bootstrap_index in range(int(n_bootstrap)):
        sampled = _resample_subjects(frame, rng)
        values = [
            _decision_curve_values(group, thresholds)
            for _, group in sampled.groupby("_repeat", sort=True)
        ]
        averaged = _mean_decision_vectors(values, len(thresholds))
        storage["mean_repeat_pooled_oof"][bootstrap_index] = averaged["net_benefit"]
    return storage


def _decision_curve_rows(
    strategy: str,
    frame: pd.DataFrame,
    protocol: str,
    thresholds: np.ndarray,
    n_bootstrap: int,
    random_state: int,
    positive_class_label: str,
) -> list[dict[str, Any]]:
    point = _decision_curve_point_estimands(frame, protocol, thresholds)
    boot = _bootstrap_decision_curve(
        frame,
        protocol,
        thresholds,
        n_bootstrap,
        np.random.default_rng(_stable_seed(random_state, "decision_curve", strategy)),
    )
    rows: list[dict[str, Any]] = []
    for estimand, values in point.items():
        for threshold_index, threshold in enumerate(thresholds):
            samples = boot[estimand][:, threshold_index]
            samples = samples[np.isfinite(samples)]
            low = high = np.nan
            if len(samples):
                low, high = np.quantile(samples, [0.025, 0.975])
            rows.append(
                {
                    "Strategy": strategy,
                    "estimand": estimand,
                    "threshold": float(threshold),
                    "positive_class_index": 1,
                    "positive_class_label": positive_class_label,
                    "net_benefit": float(values["net_benefit"][threshold_index]),
                    "ci_low": float(low) if np.isfinite(low) else np.nan,
                    "ci_high": float(high) if np.isfinite(high) else np.nan,
                    "treat_all_net_benefit": float(
                        values["treat_all_net_benefit"][threshold_index]
                    ),
                    "treat_none_net_benefit": 0.0,
                    "standardized_net_benefit": float(
                        values["standardized_net_benefit"][threshold_index]
                    ),
                    "prevalence": float(values["prevalence"][threshold_index]),
                    "n_bootstrap_valid": len(samples),
                    "n_rows": len(frame),
                }
            )
    return rows


def _binary_roc(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pcols = _probability_columns(frame)
    if len(pcols) != 2 or frame.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    y = frame["y_true"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    score = frame[pcols[1]].to_numpy(dtype=float)
    fpr, tpr, _ = sklearn_roc_curve(y, score, pos_label=1, drop_intermediate=False)
    return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float)


def _interpolated_roc(frame: pd.DataFrame, fpr_grid: np.ndarray) -> np.ndarray:
    fpr, tpr = _binary_roc(frame)
    if len(fpr) == 0:
        return np.full(len(fpr_grid), np.nan, dtype=float)
    unique_fpr = np.unique(fpr)
    upper_tpr = np.asarray(
        [float(np.max(tpr[np.isclose(fpr, value)])) for value in unique_fpr],
        dtype=float,
    )
    values = np.interp(fpr_grid, unique_fpr, upper_tpr)
    values = np.maximum.accumulate(np.clip(values, 0.0, 1.0))
    values[0] = 0.0
    values[-1] = 1.0
    return values


def _mean_roc(
    groups: list[pd.DataFrame], fpr_grid: np.ndarray
) -> tuple[np.ndarray, int]:
    values = [_interpolated_roc(group, fpr_grid) for group in groups]
    values = [value for value in values if np.isfinite(value).any()]
    if not values:
        return np.full(len(fpr_grid), np.nan, dtype=float), 0
    return _nanmean_vectors(values, len(fpr_grid)), len(values)


def _roc_curve_rows(
    strategy: str,
    frame: pd.DataFrame,
    protocol: str,
    positive_class_label: str,
    operating_threshold: float = 0.5,
    grid_points: int = 401,
) -> list[dict[str, Any]]:
    fpr_grid = np.linspace(0.0, 1.0, max(2, int(grid_points)), dtype=float)
    protocol_key = str(protocol).lower()
    curves: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    if protocol_key in _LODO_PROTOCOLS:
        pooled_fpr, pooled_tpr = _binary_roc(frame)
        if len(pooled_fpr):
            curves["pooled_sample_weighted"] = (
                pooled_fpr,
                pooled_tpr,
                int(frame["_cluster"].nunique()),
            )
        groups = [group for _, group in frame.groupby("_cluster", sort=True)]
        macro_tpr, valid_groups = _mean_roc(groups, fpr_grid)
        if valid_groups:
            curves["cohort_macro_equal_weight"] = (
                fpr_grid,
                macro_tpr,
                valid_groups,
            )
    else:
        groups = [group for _, group in frame.groupby("_repeat", sort=True)]
        mean_tpr, valid_groups = _mean_roc(groups, fpr_grid)
        if valid_groups:
            curves["mean_repeat_pooled_oof"] = (fpr_grid, mean_tpr, valid_groups)
    operating = _threshold_point_estimands(
        frame, protocol, np.asarray([float(operating_threshold)], dtype=float)
    )
    rows: list[dict[str, Any]] = []
    for estimand, (fpr, tpr, n_units) in curves.items():
        diagnostics = operating.get(estimand, {})
        sensitivity = np.asarray(
            diagnostics.get("Sensitivity", np.asarray([np.nan], dtype=float)),
            dtype=float,
        )
        specificity = np.asarray(
            diagnostics.get("Specificity", np.asarray([np.nan], dtype=float)),
            dtype=float,
        )
        operating_tpr = float(sensitivity[0]) if len(sensitivity) else np.nan
        operating_fpr = (
            float(1.0 - specificity[0])
            if len(specificity) and np.isfinite(specificity[0])
            else np.nan
        )
        for index, (x, y) in enumerate(zip(fpr, tpr, strict=False)):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            rows.append(
                {
                    "Strategy": strategy,
                    "estimand": estimand,
                    "point_index": int(index),
                    "false_positive_rate": float(x),
                    "sensitivity": float(y),
                    "positive_class_index": 1,
                    "positive_class_label": positive_class_label,
                    "operating_threshold": float(operating_threshold),
                    "operating_false_positive_rate": operating_fpr,
                    "operating_sensitivity": operating_tpr,
                    "n_averaged_units": int(n_units),
                    "n_rows": len(frame),
                }
            )
    return rows
