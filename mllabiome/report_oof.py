from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import report as _report_module
from .console import console, path_table, phase_progress, stage, success
from .final_models import load_final_models
from .storage import read_table
from .utils import report_html_path

_PERCENT_METRICS = {
    "AUROC",
    "AUROC_macro",
    "AUROC_weighted",
    "AUCPR",
    "AUCPR_macro",
    "AUCPR_weighted",
    "AP",
    "AP_macro",
    "Accuracy",
    "BalAcc",
    "F1",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
}
_PRIMARY_METRIC_ORDER = (
    "AUROC",
    "AUCPR",
    "AP",
    "MCC",
    "F1w",
    "Precision",
    "Recall",
    "BalAcc",
    "Accuracy",
)
_MULTICLASS_METRIC_ORDER = (
    "AUROC_macro",
    "AUROC_weighted",
    "AUCPR_macro",
    "AUCPR_weighted",
    "AP_macro",
    "F1_macro",
)
_PROBABILITY_METRIC_ORDER = (
    "brier",
    "brier_multiclass",
    "log_loss",
)
_CONTRAST_METRIC_ORDER = (
    "AUROC",
    "AUROC_macro",
    "AUROC_weighted",
    "AUCPR",
    "AUCPR_macro",
    "AUCPR_weighted",
    "AP",
    "AP_macro",
    "MCC",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "BalAcc",
    "Accuracy",
    "brier",
    "brier_multiclass",
    "log_loss",
)
_CALIBRATION_METRIC_ORDER = (
    "CalibrationInTheLarge",
    "CalibrationIntercept",
    "CalibrationSlope",
    "CalibrationInTheLarge_macro_OvR",
    "CalibrationIntercept_macro_OvR",
    "CalibrationSlope_macro_OvR",
)
_OPERATING_DIAGNOSTIC_METRIC_ORDER = (
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "MCC",
)
_DIAGNOSTIC_METRIC_ORDER = (
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "Accuracy",
    "MCC",
)
_METRIC_LABELS = {
    "AUROC": "AUROC",
    "AUROC_macro": "AUROC macro (OvR)",
    "AUROC_weighted": "AUROC weighted (OvR)",
    "AUCPR": "AUCPR",
    "AUCPR_macro": "AUCPR macro (OvR)",
    "AUCPR_weighted": "AUCPR weighted (OvR)",
    "AP": "Average precision",
    "AP_macro": "Average precision macro",
    "MCC": "MCC",
    "Accuracy": "Accuracy",
    "BalAcc": "Balanced accuracy",
    "F1": "F1",
    "F1w": "F1w",
    "F1_macro": "F1 macro",
    "Precision": "Precision",
    "Recall": "Recall",
    "Sensitivity": "Sensitivity",
    "Specificity": "Specificity",
    "PPV": "PPV",
    "NPV": "NPV",
    "brier": "Brier score",
    "brier_multiclass": "Multiclass Brier score",
    "log_loss": "Log loss",
    "CalibrationInTheLarge": "Calibration-in-the-large",
    "CalibrationIntercept": "Calibration intercept",
    "CalibrationSlope": "Calibration slope",
    "CalibrationInTheLarge_macro_OvR": "Calibration-in-the-large macro (OvR)",
    "CalibrationIntercept_macro_OvR": "Calibration intercept macro (OvR)",
    "CalibrationSlope_macro_OvR": "Calibration slope macro (OvR)",
}
_ESTIMAND_LABELS = {
    "mean_repeat_pooled_oof": "Pooled out-of-fold across outer folds, mean across repeats",
    "pooled_sample_weighted": "Pooled out-of-fold, sample-weighted",
    "cohort_macro_equal_weight": "Pooled out-of-fold, equal-cohort weighting",
}
_SECTION_START = '<section id="oof-inference">'
_SECTION_END = "</section>"
_STATISTICS_ANCHOR = '<h3 id="statistics">Outer-unit strategy comparisons</h3>'
_COMPUTE_ANCHOR = '<h2 id="compute">Computational resources</h2>'
_FOOTER_ANCHOR = '<p class="report-footer">'
_PROCEDURE_ANCHOR = '<h2 id="performance-evaluation" class="first-section">Task definition and evaluation procedure</h2>'
_PERFORMANCE_ANCHOR = '<h2 id="performance-summary">Task performance summary</h2>'
_MPMA_B_SECTION_START = '<section id="mpma-b-composition">'
_MPMA_B_SECTION_END = "</section>"


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def _ensemble_summary_from_selected(root: Path) -> pd.DataFrame:
    selected = _read_json(root / "ensembling" / "selected_unit.json")
    ens = (
        selected.get("inner_val_best_mpmas_ensemble")
        or selected.get("inner_val_best_ensemble")
        or {}
    )
    if not isinstance(ens, dict) or not ens:
        return pd.DataFrame()
    members = _parse_list(ens.get("members", ens.get("member_config_ids")))
    weights = _parse_list(ens.get("weights"))
    effective = 0
    if weights and len(weights) == len(members):
        for weight in weights:
            value = _safe_float(weight)
            if np.isfinite(value) and value > 1e-12:
                effective += 1
    if effective == 0:
        value = _safe_float(ens.get("effective_member_count"))
        if np.isfinite(value) and value > 0:
            effective = int(value)
    if effective == 0:
        value = _safe_float(ens.get("member_count"))
        if np.isfinite(value) and value > 0:
            effective = int(value)
    if effective == 0:
        value = _safe_float(ens.get("ensemble_size"))
        if np.isfinite(value) and value > 0:
            effective = int(value)
    if effective == 0:
        effective = len(members)
    metric = str(ens.get("selection_metric", ens.get("optimize_metric", ""))).strip()
    row = {
        "Strategy": "MPMA-E",
        "Selection": str(ens.get("selection_strategy", ens.get("method", ""))).strip(),
        "Aggregation": str(ens.get("aggregation_strategy", "")).strip(),
        "Max size": int(_safe_float(ens.get("max_size")))
        if np.isfinite(_safe_float(ens.get("max_size")))
        else int(effective),
        "Members": int(effective),
        "Selection metric": metric,
    }
    return pd.DataFrame([row])


def _format_ci_cell(row: pd.Series, metric: str) -> str:
    estimate = _safe_float(row.get("estimate"))
    low = _safe_float(row.get("ci_low"))
    high = _safe_float(row.get("ci_high"))
    if not np.isfinite(estimate):
        return "NA"
    if metric in _PERCENT_METRICS:
        body = f"{100.0 * estimate:.3f}"
        if np.isfinite(low) and np.isfinite(high):
            body += f" [{100.0 * low:.3f}, {100.0 * high:.3f}]"
        return body
    body = f"{estimate:.3f}"
    if np.isfinite(low) and np.isfinite(high):
        body += f" [{low:.3f}, {high:.3f}]"
    return body


def _wide_metric_table(
    performance: pd.DataFrame, metric_order: tuple[str, ...], *, bold_best: bool = False
) -> pd.DataFrame:
    required = {"Strategy", "estimand", "metric", "estimate", "ci_low", "ci_high"}
    if performance.empty or not required.issubset(performance.columns):
        return pd.DataFrame()
    available = set(performance["metric"].astype(str))
    metrics = [metric for metric in metric_order if metric in available]
    if not metrics:
        return pd.DataFrame()
    estimands = [
        str(x)
        for x in performance["estimand"].dropna().astype(str).unique().tolist()
        if str(x)
    ]
    show_estimand = len(set(estimands)) > 1
    best: dict[tuple[str, str], str] = {}
    if bold_best:
        for estimand in estimands:
            group = performance[performance["estimand"].astype(str).eq(estimand)]
            for metric in metrics:
                sub = group[group["metric"].astype(str).eq(metric)].copy()
                sub["estimate"] = pd.to_numeric(sub.get("estimate"), errors="coerce")
                sub = sub[np.isfinite(sub["estimate"].to_numpy(dtype=float))]
                if sub.empty:
                    continue
                ascending = metric in {"brier", "brier_multiclass", "log_loss"}
                sub = sub.sort_values("estimate", ascending=ascending, kind="mergesort")
                best[(estimand, metric)] = str(sub.iloc[0]["Strategy"])
    rows: list[dict[str, Any]] = []
    for (strategy, estimand), group in performance.groupby(
        ["Strategy", "estimand"], sort=False
    ):
        row: dict[str, Any] = {"Strategy": str(strategy)}
        if show_estimand:
            row["Estimand"] = _ESTIMAND_LABELS.get(str(estimand), str(estimand))
        for metric in metrics:
            sub = group[group["metric"].astype(str).eq(metric)]
            label = _METRIC_LABELS.get(metric, metric)
            if sub.empty:
                row[label] = "NA"
            else:
                value = _format_ci_cell(sub.iloc[0], metric)
                if bold_best and best.get((str(estimand), metric)) == str(strategy):
                    value = f"<strong>{value}</strong>"
                row[label] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _performance_display(performance: pd.DataFrame) -> pd.DataFrame:
    return _wide_metric_table(performance, _PRIMARY_METRIC_ORDER, bold_best=True)


def _diagnostic_operating_display(
    performance: pd.DataFrame, n_classes: int | None
) -> pd.DataFrame:
    if n_classes is None or int(n_classes) != 2:
        return pd.DataFrame()
    return _wide_metric_table(performance, _OPERATING_DIAGNOSTIC_METRIC_ORDER)


def _multiclass_display(
    performance: pd.DataFrame, n_classes: int | None
) -> pd.DataFrame:
    if n_classes is None or int(n_classes) <= 2:
        return pd.DataFrame()
    return _wide_metric_table(performance, _MULTICLASS_METRIC_ORDER)


def _probability_display(
    performance: pd.DataFrame, n_classes: int | None
) -> pd.DataFrame:
    if n_classes is not None and int(n_classes) <= 2:
        order = ("brier", "log_loss")
    elif n_classes is not None and int(n_classes) > 2:
        order = ("brier_multiclass", "log_loss")
    else:
        order = _PROBABILITY_METRIC_ORDER
    return _wide_metric_table(performance, order)


def _calibration_display(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty or "metric" not in performance.columns:
        return pd.DataFrame()
    calibration = performance[
        performance["metric"].astype(str).isin(_CALIBRATION_METRIC_ORDER)
    ].copy()
    coefficient = (
        calibration["metric"]
        .astype(str)
        .isin(
            {
                "CalibrationIntercept",
                "CalibrationSlope",
                "CalibrationIntercept_macro_OvR",
                "CalibrationSlope_macro_OvR",
            }
        )
    )
    unstable = pd.Series(False, index=calibration.index)
    for column in ("estimate", "ci_low", "ci_high"):
        if column in calibration.columns:
            values = pd.to_numeric(calibration[column], errors="coerce")
            unstable = unstable | (coefficient & values.abs().gt(100.0))
    for column in ("estimate", "ci_low", "ci_high"):
        if column in calibration.columns:
            calibration.loc[unstable, column] = np.nan
    return _wide_metric_table(calibration, _CALIBRATION_METRIC_ORDER)


def _calibration_coefficients_display(
    coefficients: pd.DataFrame, n_classes: int | None
) -> pd.DataFrame:
    if n_classes is None or int(n_classes) <= 2 or coefficients.empty:
        return pd.DataFrame()
    required = {
        "Strategy",
        "estimand",
        "class_label",
        "CalibrationInTheLarge",
        "CalibrationIntercept",
        "CalibrationSlope",
    }
    if not required.issubset(coefficients.columns):
        return pd.DataFrame()
    out = coefficients[
        [
            "Strategy",
            "estimand",
            "class_label",
            "CalibrationInTheLarge",
            "CalibrationIntercept",
            "CalibrationSlope",
        ]
    ].copy()
    out = out.rename(
        columns={
            "estimand": "Estimand",
            "class_label": "Class",
            "CalibrationInTheLarge": "Calibration-in-the-large",
            "CalibrationIntercept": "Calibration intercept",
            "CalibrationSlope": "Calibration slope",
        }
    )
    return out


def _threshold_metrics_display(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Strategy",
        "estimand",
        "threshold",
        "positive_class_label",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
    }
    if threshold_metrics.empty or not required.issubset(threshold_metrics.columns):
        return pd.DataFrame()
    estimands = threshold_metrics["estimand"].dropna().astype(str).unique().tolist()
    show_estimand = len(estimands) > 1
    rows: list[dict[str, Any]] = []
    grouping = ["Strategy", "estimand", "threshold", "positive_class_label"]
    for keys, group in threshold_metrics.groupby(grouping, sort=False):
        strategy, estimand, threshold, positive_class = keys
        row: dict[str, Any] = {
            "Strategy": str(strategy),
            "Threshold": f"{float(threshold):.3f}",
            "Positive class": str(positive_class),
        }
        if show_estimand:
            row["Estimand"] = _ESTIMAND_LABELS.get(str(estimand), str(estimand))
        for metric in _DIAGNOSTIC_METRIC_ORDER:
            sub = group[group["metric"].astype(str).eq(metric)]
            if sub.empty:
                continue
            row[_METRIC_LABELS.get(metric, metric)] = _format_ci_cell(
                sub.iloc[0], metric
            )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    order = ["Strategy", "Estimand", "Threshold", "Positive class"]
    order.extend(
        _METRIC_LABELS.get(metric, metric) for metric in _DIAGNOSTIC_METRIC_ORDER
    )
    return out[[column for column in order if column in out.columns]]


def _confusion_display(confusion: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Strategy",
        "estimand",
        "threshold_type",
        "threshold",
        "actual_class_label",
        "predicted_class_label",
        "count_estimate",
        "row_fraction",
        "n_averaged_units",
    }
    if confusion.empty or not required.issubset(confusion.columns):
        return pd.DataFrame()
    estimands = confusion["estimand"].dropna().astype(str).unique().tolist()
    show_estimand = len(estimands) > 1
    predicted_labels = list(
        dict.fromkeys(confusion["predicted_class_label"].astype(str).tolist())
    )
    rows: list[dict[str, Any]] = []
    grouping = [
        "Strategy",
        "estimand",
        "threshold_type",
        "threshold",
        "actual_class_label",
    ]
    frame = confusion.copy()
    frame["threshold_key"] = pd.to_numeric(frame["threshold"], errors="coerce").fillna(
        -1.0
    )
    grouping = [
        "Strategy",
        "estimand",
        "threshold_type",
        "threshold_key",
        "actual_class_label",
    ]
    for keys, group in frame.groupby(grouping, sort=False):
        strategy, estimand, threshold_type, threshold_key, actual_class = keys
        operating_point = (
            "Model prediction"
            if str(threshold_type) == "model_prediction"
            else f"Probability ≥ {float(threshold_key):.3f}"
        )
        row: dict[str, Any] = {
            "Strategy": str(strategy),
            "Operating point": operating_point,
            "Actual class": str(actual_class),
        }
        if show_estimand:
            row["Estimand"] = _ESTIMAND_LABELS.get(str(estimand), str(estimand))
        for predicted_label in predicted_labels:
            sub = group[group["predicted_class_label"].astype(str).eq(predicted_label)]
            if sub.empty:
                row[f"Predicted {predicted_label}"] = "NA"
                continue
            item = sub.iloc[0]
            fraction = _safe_float(item.get("row_fraction"))
            count = _safe_float(item.get("count_estimate"))
            try:
                n_units = int(item.get("n_averaged_units", 1))
            except (TypeError, ValueError):
                n_units = 1
            if not np.isfinite(fraction):
                text = "NA"
            elif np.isfinite(count) and n_units <= 1:
                text = f"{100.0 * fraction:.1f}% ({count:.0f})"
            elif np.isfinite(count):
                text = f"{100.0 * fraction:.1f}% (mean {count:.1f})"
            else:
                text = f"{100.0 * fraction:.1f}%"
            row[f"Predicted {predicted_label}"] = text
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    order = ["Strategy", "Estimand", "Operating point", "Actual class"]
    order.extend(f"Predicted {label}" for label in predicted_labels)
    return out[[column for column in order if column in out.columns]]


def _decision_curve_display(
    decision_curve: pd.DataFrame, manifest: dict[str, Any]
) -> pd.DataFrame:
    required = {
        "Strategy",
        "estimand",
        "threshold",
        "positive_class_label",
        "net_benefit",
        "ci_low",
        "ci_high",
        "treat_all_net_benefit",
        "treat_none_net_benefit",
        "prevalence",
    }
    if decision_curve.empty or not required.issubset(decision_curve.columns):
        return pd.DataFrame()
    configured = manifest.get("diagnostic_thresholds", [])
    if not isinstance(configured, list) or not configured:
        return pd.DataFrame()
    thresholds = np.asarray(
        [float(value) for value in configured if np.isfinite(_safe_float(value))],
        dtype=float,
    )
    if len(thresholds) == 0:
        return pd.DataFrame()
    values = pd.to_numeric(decision_curve["threshold"], errors="coerce").to_numpy(
        dtype=float
    )
    mask = np.zeros(len(decision_curve), dtype=bool)
    for threshold in thresholds:
        mask |= np.isclose(values, threshold, atol=1e-12, rtol=1e-12)
    frame = decision_curve.loc[mask].copy()
    if frame.empty:
        return pd.DataFrame()
    estimands = frame["estimand"].dropna().astype(str).unique().tolist()
    show_estimand = len(estimands) > 1
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        estimate = _safe_float(item.get("net_benefit"))
        low = _safe_float(item.get("ci_low"))
        high = _safe_float(item.get("ci_high"))
        net = "NA"
        if np.isfinite(estimate):
            net = f"{estimate:.4f}"
            if np.isfinite(low) and np.isfinite(high):
                net += f" [{low:.4f}, {high:.4f}]"
        row: dict[str, Any] = {
            "Strategy": str(item.get("Strategy", "")),
            "Threshold": f"{_safe_float(item.get('threshold')):.3f}",
            "Positive class": str(item.get("positive_class_label", "")),
            "Net benefit (95% CI)": net,
            "Treat all": f"{_safe_float(item.get('treat_all_net_benefit')):.4f}",
            "Treat none": f"{_safe_float(item.get('treat_none_net_benefit')):.4f}",
            "Prevalence": f"{100.0 * _safe_float(item.get('prevalence')):.1f}%",
        }
        if show_estimand:
            estimand = str(item.get("estimand", ""))
            row["Estimand"] = _ESTIMAND_LABELS.get(estimand, estimand)
        rows.append(row)
    out = pd.DataFrame(rows)
    order = [
        "Strategy",
        "Estimand",
        "Threshold",
        "Positive class",
        "Net benefit (95% CI)",
        "Treat all",
        "Treat none",
        "Prevalence",
    ]
    return out[[column for column in order if column in out.columns]]


def _decision_curve_figure_html(
    decision_curve: pd.DataFrame,
    roc_curve: pd.DataFrame,
    performance: pd.DataFrame,
    manifest: dict[str, Any],
    report_dir: Path,
) -> str:
    dca_required = {
        "Strategy",
        "estimand",
        "threshold",
        "net_benefit",
        "ci_low",
        "ci_high",
        "treat_all_net_benefit",
        "treat_none_net_benefit",
    }
    roc_required = {
        "Strategy",
        "estimand",
        "point_index",
        "false_positive_rate",
        "sensitivity",
        "operating_threshold",
        "operating_false_positive_rate",
        "operating_sensitivity",
    }
    dca = (
        decision_curve.copy()
        if dca_required.issubset(decision_curve.columns)
        else pd.DataFrame()
    )
    roc = (
        roc_curve.copy() if roc_required.issubset(roc_curve.columns) else pd.DataFrame()
    )
    if dca.empty and roc.empty:
        return ""
    for column in (
        "threshold",
        "net_benefit",
        "ci_low",
        "ci_high",
        "treat_all_net_benefit",
        "treat_none_net_benefit",
    ):
        if column in dca.columns:
            dca[column] = pd.to_numeric(dca[column], errors="coerce")
    for column in (
        "point_index",
        "false_positive_rate",
        "sensitivity",
        "operating_threshold",
        "operating_false_positive_rate",
        "operating_sensitivity",
    ):
        if column in roc.columns:
            roc[column] = pd.to_numeric(roc[column], errors="coerce")
    if not dca.empty:
        dca = dca[np.isfinite(dca["threshold"])].copy()
    if not roc.empty:
        roc = roc[
            np.isfinite(roc["false_positive_rate"]) & np.isfinite(roc["sensitivity"])
        ].copy()
    settings = manifest.get("decision_curve", {})
    settings = settings if isinstance(settings, dict) else {}
    operating_threshold = _safe_float(settings.get("ordinary_operating_threshold", 0.5))
    if not np.isfinite(operating_threshold):
        operating_threshold = 0.5
    configured = manifest.get("diagnostic_thresholds", [])
    thresholds = []
    if isinstance(configured, list):
        thresholds = sorted(
            {
                float(value)
                for value in configured
                if np.isfinite(_safe_float(value)) and 0.0 < float(value) < 1.0
            }
        )
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from . import style as _style

    strategy_palette = {
        "MPMA-E": _style.C_NAVY,
        "MPMA-B": _style.C_MID,
        "Baseline RF": _style.C_DARK,
        "SIAMCAT": _style.C_TEAL,
        "AutoML": _style.C_BRIGHT,
    }
    fallback_palette = (
        _style.C_SKY,
        _style.C_HILITE,
        _style.MID,
        _style.DIM,
    )

    def auroc(strategy: str, estimand: str) -> float:
        if performance.empty:
            return float("nan")
        required_performance = {"Strategy", "estimand", "metric", "estimate"}
        if not required_performance.issubset(performance.columns):
            return float("nan")
        selected = performance[
            performance["Strategy"].astype(str).eq(strategy)
            & performance["estimand"].astype(str).eq(estimand)
            & performance["metric"].astype(str).eq("AUROC")
        ]
        if selected.empty:
            return float("nan")
        return _safe_float(selected.iloc[0].get("estimate"))

    estimands = list(
        dict.fromkeys(
            [
                *dca.get("estimand", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .tolist(),
                *roc.get("estimand", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .tolist(),
            ]
        )
    )
    if not estimands:
        return ""
    blocks: list[str] = []
    for index, estimand in enumerate(estimands, start=1):
        dca_sub = (
            dca[dca["estimand"].astype(str).eq(str(estimand))].copy()
            if not dca.empty
            else pd.DataFrame()
        )
        roc_sub = (
            roc[roc["estimand"].astype(str).eq(str(estimand))].copy()
            if not roc.empty
            else pd.DataFrame()
        )
        strategy_sequence = list(
            dict.fromkeys(
                [
                    *dca_sub.get("Strategy", pd.Series(dtype=str)).astype(str).tolist(),
                    *roc_sub.get("Strategy", pd.Series(dtype=str)).astype(str).tolist(),
                ]
            )
        )
        if not strategy_sequence:
            continue
        fallback_index = 0
        colors: dict[str, str] = {}
        for strategy in strategy_sequence:
            if strategy in strategy_palette:
                colors[strategy] = strategy_palette[strategy]
            else:
                colors[strategy] = fallback_palette[
                    fallback_index % len(fallback_palette)
                ]
                fallback_index += 1
        with mpl.rc_context(_style.RC):
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(_style.COL_W_2, 84 * _style.MM),
                gridspec_kw={"wspace": 0.28},
            )
            ax_dca, ax_roc = axes
            model_scale: list[np.ndarray] = []
            reference_groups: list[tuple[str, np.ndarray, np.ndarray]] = []
            reference_upper: list[float] = []
            dca_strategies = (
                dca_sub["Strategy"].astype(str).drop_duplicates().tolist()
                if not dca_sub.empty
                else []
            )
            for strategy in dca_strategies:
                group = dca_sub[
                    dca_sub["Strategy"].astype(str).eq(strategy)
                ].sort_values("threshold")
                x = group["threshold"].to_numpy(dtype=float)
                y = group["net_benefit"].to_numpy(dtype=float)
                low = group["ci_low"].to_numpy(dtype=float)
                high = group["ci_high"].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if not np.any(valid):
                    continue
                color = colors[strategy]
                ax_dca.plot(
                    x[valid],
                    y[valid],
                    linewidth=1.0,
                    color=color,
                    label=strategy,
                )
                band = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
                if np.any(band):
                    ax_dca.fill_between(
                        x[band],
                        low[band],
                        high[band],
                        color=color,
                        alpha=0.10,
                        linewidth=0.0,
                    )
                    model_scale.extend([low[band], high[band]])
                model_scale.append(y[valid])
                operating = group[
                    np.isclose(
                        group["threshold"].to_numpy(dtype=float),
                        operating_threshold,
                        atol=1e-12,
                        rtol=1e-12,
                    )
                ]
                if not operating.empty:
                    operating_net = _safe_float(operating.iloc[0].get("net_benefit"))
                    if np.isfinite(operating_net):
                        ax_dca.scatter(
                            [operating_threshold],
                            [operating_net],
                            s=15,
                            marker="o",
                            facecolor=color,
                            edgecolor=_style.BG,
                            linewidth=0.45,
                            zorder=8,
                        )
                reference_y = group["treat_all_net_benefit"].to_numpy(dtype=float)
                reference_valid = np.isfinite(x) & np.isfinite(reference_y)
                if np.any(reference_valid):
                    values = reference_y[reference_valid]
                    reference_groups.append((strategy, x[reference_valid], values))
                    reference_upper.append(float(np.max(values)))
            shared_reference = False
            if reference_groups:
                base_x = reference_groups[0][1]
                base_y = reference_groups[0][2]
                shared_reference = all(
                    len(x) == len(base_x)
                    and np.allclose(x, base_x, atol=1e-12, rtol=1e-12, equal_nan=True)
                    and np.allclose(y, base_y, atol=1e-10, rtol=1e-10, equal_nan=True)
                    for _, x, y in reference_groups[1:]
                )
                if shared_reference:
                    ax_dca.plot(
                        base_x,
                        base_y,
                        linestyle="--",
                        linewidth=0.9,
                        color=_style.MID,
                        label="Treat all",
                    )
                else:
                    for strategy, x, y in reference_groups:
                        ax_dca.plot(
                            x,
                            y,
                            linestyle="--",
                            linewidth=0.8,
                            color=colors.get(strategy, _style.MID),
                            alpha=0.75,
                            label=f"Treat all ({strategy})",
                        )
            ax_dca.axhline(
                0.0,
                color=_style.INK,
                linewidth=0.9,
                linestyle=":",
                label="Treat none",
            )
            ax_dca.axvline(
                operating_threshold,
                color=_style.DIM,
                linewidth=0.65,
                linestyle=(0, (3, 2)),
                alpha=0.9,
            )
            threshold_label_used = False
            for threshold in thresholds:
                if np.isclose(threshold, operating_threshold, atol=1e-12, rtol=1e-12):
                    continue
                ax_dca.axvline(
                    threshold,
                    color=_style.DIM,
                    linewidth=0.55,
                    linestyle=(0, (2, 2)),
                    alpha=0.65,
                    label=None if threshold_label_used else "Pre-specified threshold",
                )
                threshold_label_used = True
            finite_chunks = [
                values[np.isfinite(values)]
                for values in model_scale
                if np.any(np.isfinite(values))
            ]
            finite_scale = (
                np.concatenate(finite_chunks)
                if finite_chunks
                else np.asarray([], dtype=float)
            )
            if finite_scale.size:
                lower = min(0.0, float(np.min(finite_scale)))
                upper = max(
                    0.0,
                    float(np.max(finite_scale)),
                    max(reference_upper) if reference_upper else 0.0,
                )
                span = max(upper - lower, 0.05)
                ax_dca.set_ylim(lower - 0.08 * span, upper + 0.10 * span)
            if not dca_sub.empty:
                xmin = float(np.nanmin(dca_sub["threshold"].to_numpy(dtype=float)))
                xmax = float(np.nanmax(dca_sub["threshold"].to_numpy(dtype=float)))
                xspan = max(xmax - xmin, 1e-6)
                xleft = max(0.0, xmin - 0.02 * xspan)
                xright = min(1.0, xmax + 0.02 * xspan)
                if xmin <= 0.02:
                    xleft = 0.0
                if xmax >= 0.98:
                    xright = 1.0
                ax_dca.set_xlim(xleft, xright)
            else:
                ax_dca.set_xlim(0.0, 1.0)
                ax_dca.text(
                    0.5,
                    0.5,
                    "Decision-curve analysis unavailable",
                    transform=ax_dca.transAxes,
                    ha="center",
                    va="center",
                    color=_style.MID,
                )
            ymin, _ = ax_dca.get_ylim()
            clipped_reference = False
            for strategy, x, y in reference_groups:
                below = y < ymin
                if not np.any(below):
                    continue
                crossing = np.flatnonzero((y[:-1] >= ymin) & (y[1:] < ymin))
                if crossing.size:
                    i = int(crossing[0])
                    dy = y[i + 1] - y[i]
                    if np.isfinite(dy) and abs(dy) > 1e-15:
                        fraction = (ymin - y[i]) / dy
                        marker_x = x[i] + fraction * (x[i + 1] - x[i])
                    else:
                        marker_x = x[i + 1]
                else:
                    marker_x = x[int(np.flatnonzero(below)[0])]
                marker_color = (
                    _style.MID if shared_reference else colors.get(strategy, _style.MID)
                )
                ax_dca.scatter(
                    [marker_x],
                    [ymin],
                    marker="v",
                    s=14,
                    facecolor=marker_color,
                    edgecolor="none",
                    clip_on=False,
                    zorder=7,
                )
                clipped_reference = True
            ax_dca.set_xlabel("Threshold probability")
            ax_dca.set_ylabel("Net benefit")
            ax_dca.spines["top"].set_visible(False)
            ax_dca.spines["right"].set_visible(False)
            if dca_strategies:
                ax_dca.legend(loc="best", ncol=2 if len(dca_strategies) > 1 else 1)

            roc_strategies = (
                roc_sub["Strategy"].astype(str).drop_duplicates().tolist()
                if not roc_sub.empty
                else []
            )
            for strategy in roc_strategies:
                group = roc_sub[
                    roc_sub["Strategy"].astype(str).eq(strategy)
                ].sort_values("point_index")
                fpr = group["false_positive_rate"].to_numpy(dtype=float)
                tpr = group["sensitivity"].to_numpy(dtype=float)
                valid = np.isfinite(fpr) & np.isfinite(tpr)
                if not np.any(valid):
                    continue
                color = colors[strategy]
                auc = auroc(strategy, str(estimand))
                label = (
                    strategy
                    if not np.isfinite(auc)
                    else f"{strategy} (AUROC {auc:.3f})"
                )
                ax_roc.plot(
                    np.clip(fpr[valid], 0.0, 1.0),
                    np.clip(tpr[valid], 0.0, 1.0),
                    linewidth=1.0,
                    color=color,
                    label=label,
                )
                first = group.iloc[0]
                operating_fpr = _safe_float(first.get("operating_false_positive_rate"))
                operating_tpr = _safe_float(first.get("operating_sensitivity"))
                if np.isfinite(operating_fpr) and np.isfinite(operating_tpr):
                    ax_roc.scatter(
                        [operating_fpr],
                        [operating_tpr],
                        s=15,
                        marker="o",
                        facecolor=color,
                        edgecolor=_style.BG,
                        linewidth=0.45,
                        zorder=8,
                    )
            ax_roc.plot(
                [0.0, 1.0],
                [0.0, 1.0],
                linestyle="--",
                linewidth=0.8,
                color=_style.DIM,
                label="Chance",
            )
            ax_roc.set_xlim(0.0, 1.0)
            ax_roc.set_ylim(0.0, 1.0)
            ax_roc.set_xlabel("1 − specificity")
            ax_roc.set_ylabel("Sensitivity")
            ax_roc.spines["top"].set_visible(False)
            ax_roc.spines["right"].set_visible(False)
            if roc_strategies:
                ax_roc.legend(loc="lower right")
            fig.subplots_adjust(
                left=0.08, right=0.985, bottom=0.16, top=0.94, wspace=0.32
            )
            ax_dca.text(
                0.0,
                1.025,
                "a",
                transform=ax_dca.transAxes,
                fontweight="bold",
                va="bottom",
                ha="left",
                clip_on=False,
            )
            ax_roc.text(
                0.0,
                1.025,
                "b",
                transform=ax_roc.transAxes,
                fontweight="bold",
                va="bottom",
                ha="left",
                clip_on=False,
            )
            target = report_dir / "figures" / f"decision_roc__{index}.svg"
            _style.save_svg(fig, target)
            plt.close(fig)
        estimand_label = _ESTIMAND_LABELS.get(str(estimand), str(estimand))
        caption = "Decision-curve analysis with 95% bootstrap confidence intervals and receiver operating characteristic curves"
        if estimand_label:
            caption += f" ({estimand_label})"
        caption += f". Filled circles mark the ordinary binary operating threshold p={operating_threshold:.3f} in both panels"
        if clipped_reference:
            caption += "; a downward triangle marks where a treat-all reference continues below the displayed decision-curve y-range"
        caption += "."
        block = _report_module._fig(target, report_dir, caption)
        if block:
            blocks.append(block)
    return "\n".join(blocks)


def _html_table(df: pd.DataFrame, *, raw_html_cols: set[str] | None = None) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    raw_html_cols = raw_html_cols or set()
    cols = list(df.columns)
    parts = ['<div class="table-wrap"><table><thead><tr>']
    parts.extend(f"<th>{html.escape(str(c))}</th>" for c in cols)
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        parts.append("<tr>")
        for col in cols:
            value = row.get(col, "")
            if pd.isna(value):
                text = ""
            elif isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                text = str(int(value))
            elif isinstance(value, (float, np.floating)) and not isinstance(
                value, (bool, np.bool_)
            ):
                text = f"{float(value):.3f}" if np.isfinite(float(value)) else ""
            else:
                text = str(value) if col in raw_html_cols else html.escape(str(value))
            parts.append(f"<td>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _methodology_html(manifest: dict[str, Any]) -> str:
    protocol = str(manifest.get("protocol", "")).strip().lower()
    n_bootstrap = manifest.get("n_bootstrap", "")
    try:
        bootstrap = f"{int(n_bootstrap):,}"
    except (TypeError, ValueError):
        bootstrap = ""
    if protocol in {"lodo", "leave_one_dataset_out"}:
        method = "two-stage cohort-and-subject bootstrap"
        estimand = "sample-weighted and equal-cohort held-out performance"
    else:
        design = manifest.get("observed_oof_design", {})
        repeats = []
        if isinstance(design, dict):
            for value in design.values():
                if isinstance(value, dict):
                    try:
                        repeats.append(int(value.get("n_repeats")))
                    except (TypeError, ValueError):
                        pass
        method = (
            "subject-cluster bootstrap"
            if repeats and max(repeats) == 1
            else "subject-cluster and repeat bootstrap"
        )
        estimand = "pooled outer-fold out-of-fold performance"
    suffix = f" using {bootstrap} replicates" if bootstrap else ""
    return (
        f"<p>Estimates summarize {html.escape(estimand)}. "
        f"Uncertainty is reported as 95% percentile bootstrap confidence intervals "
        f"from a {html.escape(method)}{html.escape(suffix)}.</p>"
    )


def _probability_semantics_html(manifest: dict[str, Any]) -> str:
    semantics = manifest.get("probability_semantics", {})
    if not isinstance(semantics, dict):
        return ""
    excluded: list[str] = []
    for strategy, info in semantics.items():
        if not isinstance(info, dict) or bool(info.get("valid", False)):
            continue
        reason = str(info.get("reason", "")).lower()
        if "rank_mean" in reason:
            detail = "rank aggregation"
        elif "siamcat" in reason:
            detail = "score-valued output"
        elif "ridge" in reason:
            detail = "decision-score output"
        else:
            detail = "non-probability output"
        excluded.append(
            f"<strong>{html.escape(str(strategy))}</strong> ({html.escape(detail)})"
        )
    if not excluded:
        return ""
    return (
        "<p>Brier score, log loss, calibration, configured probability-threshold diagnostics, and decision-curve analysis are shown only for "
        "probability-valued predictions. Probability-based summaries are not shown for "
        + ", ".join(excluded)
        + ".</p>"
    )


def _format_contrast_value(value: Any, metric: str) -> str:
    v = _safe_float(value)
    if not np.isfinite(v):
        return "NA"
    if metric in _PERCENT_METRICS:
        return f"{100.0 * v:.3f}"
    return f"{v:.3f}"


def _contrast_display(
    contrasts: pd.DataFrame, n_classes: int | None = None
) -> pd.DataFrame:
    required = {
        "strategy_a",
        "strategy_b",
        "estimand",
        "metric",
        "estimate_a",
        "estimate_b",
        "advantage_a_over_b",
        "advantage_ci_low",
        "advantage_ci_high",
    }
    if contrasts.empty or not required.issubset(contrasts.columns):
        return pd.DataFrame()
    shown = contrasts.copy()
    metric_order = list(_CONTRAST_METRIC_ORDER)
    if n_classes is not None and int(n_classes) <= 2:
        excluded = {
            "AUROC_macro",
            "AUROC_weighted",
            "AUCPR_macro",
            "AUCPR_weighted",
            "AP_macro",
            "F1_macro",
            "brier_multiclass",
        }
        metric_order = [metric for metric in metric_order if metric not in excluded]
    metric_rank = {metric: index for index, metric in enumerate(metric_order)}
    shown = shown[shown["metric"].astype(str).isin(metric_order)].copy()
    shown["_metric_rank"] = shown["metric"].astype(str).map(metric_rank).fillna(999)
    shown = shown.sort_values(
        ["_metric_rank", "strategy_a", "strategy_b", "estimand"],
        kind="mergesort",
    )
    estimands = [
        str(x)
        for x in shown["estimand"].dropna().astype(str).unique().tolist()
        if str(x)
    ]
    show_estimand = len(set(estimands)) > 1
    rows: list[dict[str, Any]] = []
    for _, row in shown.iterrows():
        metric = str(row.get("metric", ""))
        metric_label = _METRIC_LABELS.get(metric, metric)
        advantage = _safe_float(row.get("advantage_a_over_b"))
        low = _safe_float(row.get("advantage_ci_low"))
        high = _safe_float(row.get("advantage_ci_high"))
        advantage_text = _format_contrast_value(advantage, metric)
        if np.isfinite(low) and np.isfinite(high):
            advantage_text += (
                f" [{_format_contrast_value(low, metric)}, "
                f"{_format_contrast_value(high, metric)}]"
            )
        item: dict[str, Any] = {
            "Strategy A": str(row.get("strategy_a", "")),
            "Strategy B": str(row.get("strategy_b", "")),
        }
        if show_estimand:
            item["Estimand"] = _ESTIMAND_LABELS.get(
                str(row.get("estimand", "")), str(row.get("estimand", ""))
            )
        if np.isfinite(low) and np.isfinite(high):
            if low > 0.0:
                interpretation = "95% CI excludes 0 and favors Strategy A"
            elif high < 0.0:
                interpretation = "95% CI excludes 0 and favors Strategy B"
            else:
                interpretation = "95% CI includes 0 with no clear difference"
        else:
            interpretation = "Confidence interval not estimable"
        item.update(
            {
                "Metric": metric_label,
                "A": _format_contrast_value(row.get("estimate_a"), metric),
                "B": _format_contrast_value(row.get("estimate_b"), metric),
                "Effect favoring A (95% CI)": advantage_text,
                "Interpretation": interpretation,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _design_display(manifest: dict[str, Any]) -> pd.DataFrame:
    design = manifest.get("observed_oof_design", {})
    if not isinstance(design, dict) or not design:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for strategy, values in design.items():
        if not isinstance(values, dict):
            continue
        row: dict[str, Any] = {"Strategy": str(strategy)}
        mapping = (
            ("n_unique_samples", "Subjects"),
            ("n_outer_units", "Outer units"),
            ("n_repeats", "Repeats"),
            ("n_cohorts", "Held-out cohorts"),
        )
        for source, label in mapping:
            value = values.get(source)
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                row[label] = number
        rows.append(row)
    return pd.DataFrame(rows)


def _classification_metric_note(manifest: dict[str, Any], n_classes: int | None) -> str:
    return (
        "<p>AUROC measures how well predicted scores rank classes across decision thresholds. "
        "AUCPR is trapezoidal area under the empirical precision-recall curve, while average precision (AP) is the recall-increment-weighted precision summary. "
        "MCC is reported on its conventional [-1, 1] scale. "
        "F1w is support-weighted F1. Precision and recall are macro-averaged across classes. "
        "Balanced accuracy is the mean class-specific recall, whereas accuracy is the overall fraction of correct predictions. Higher values are better for all metrics in this table.</p>"
    )


def _diagnostic_operating_note(manifest: dict[str, Any]) -> str:
    positive = (
        str(manifest.get("positive_class_label", "positive")).strip() or "positive"
    )
    return (
        f"<p>Sensitivity, specificity, PPV, and NPV use {html.escape(positive)} as the positive class at the model's ordinary held-out prediction operating point. "
        "PPV and NPV depend on outcome prevalence and therefore describe the observed evaluation population rather than transport unchanged to populations with different prevalence. "
        "MCC is shown on its conventional [-1, 1] scale. Confidence intervals use the same protocol-aware subject/cohort bootstrap as the other pooled out-of-fold estimates.</p>"
    )


def _diagnostic_threshold_note(manifest: dict[str, Any]) -> str:
    positive = (
        str(manifest.get("positive_class_label", "positive")).strip() or "positive"
    )
    return (
        f"<p>Configured probability thresholds are applied to held-out out-of-fold probabilities for the {html.escape(positive)} class only after model fitting and selection. "
        "They do not alter training, hyperparameter selection, or the ordinary model-prediction operating point. "
        "Sensitivity, specificity, PPV, NPV, accuracy and MCC are recomputed at each configured threshold with protocol-aware 95% bootstrap confidence intervals. "
        "PPV and NPV are prevalence-dependent. PPV or NPV is reported as NA when its denominator is zero in the corresponding estimate or bootstrap replicate.</p>"
    )


def _confusion_note() -> str:
    return (
        "<p>Confusion matrices are row-normalized by the observed class. Values in parentheses are raw counts for a single pooled estimand and mean counts when the estimand averages repeats or held-out cohorts. "
        "Probability-threshold matrices are shown only for explicitly configured binary diagnostic thresholds.</p>"
    )


def _decision_curve_note(manifest: dict[str, Any]) -> str:
    settings = manifest.get("decision_curve", {})
    if not isinstance(settings, dict):
        settings = {}
    low = _safe_float(settings.get("minimum_threshold"))
    high = _safe_float(settings.get("maximum_threshold"))
    points = settings.get("points", "")
    range_text = ""
    if np.isfinite(low) and np.isfinite(high):
        range_text = f" from {low:.3f} to {high:.3f}"
    try:
        point_text = f" at {int(points):,} threshold probabilities"
    except (TypeError, ValueError):
        point_text = ""
    operating = _safe_float(settings.get("ordinary_operating_threshold", 0.5))
    operating_text = f"{operating:.3f}" if np.isfinite(operating) else "0.500"
    return (
        "<p>Decision-curve analysis evaluates net benefit as TP/N − FP/N × p<sub>t</sub>/(1−p<sub>t</sub>), where p<sub>t</sub> is the threshold probability. "
        f"The curve is evaluated{html.escape(range_text)}{html.escape(point_text)} and compared with treat-all and treat-none reference strategies. "
        "Net-benefit confidence intervals use the same protocol-aware cluster bootstrap. Interpretation should be restricted to threshold probabilities that are clinically plausible because the threshold encodes the relative consequence of false-positive and false-negative decisions. "
        f"The paired ROC panel is constructed from held-out score rankings for the same protocol-defined estimand; filled circles mark sensitivity and 1−specificity at the ordinary binary operating threshold p={html.escape(operating_text)}. "
        "For repeated nested cross-validation, repeat-specific pooled out-of-fold ROC curves are averaged on a common false-positive-rate grid; for leave-dataset-out evaluation, both pooled sample-weighted and equal-weight cohort-macro ROC estimands are retained. "
        "AUROC values in the ROC legend are the corresponding held-out statistical estimates and are not recomputed from the rendered curve. Pre-specified diagnostic thresholds, when configured, remain marked in the decision-curve panel and summarized in the compact table.</p>"
    )


def _probability_quality_note(n_classes: int | None) -> str:
    brier = "Brier score is the mean squared error of predicted probabilities"
    if n_classes is not None and int(n_classes) > 2:
        brier = "Multiclass Brier score is the mean summed squared error across class probabilities"
    return (
        f"<p>{html.escape(brier)}. A value of 0 is ideal and lower values are better. "
        "Log loss is the mean negative log-probability assigned to the observed class. It penalizes confident wrong predictions more strongly. A value of 0 is ideal and lower values are better. "
        "Confidence intervals use the same protocol-aware bootstrap as the pooled performance estimates.</p>"
    )


def _paired_contrast_note(manifest: dict[str, Any]) -> str:
    protocol = str(manifest.get("protocol", "")).strip().lower()
    try:
        n_bootstrap = int(manifest.get("n_bootstrap", 0))
    except (TypeError, ValueError):
        n_bootstrap = 0
    count = f"{n_bootstrap:,} " if n_bootstrap > 0 else ""
    if protocol in {"lodo", "leave_one_dataset_out"}:
        sampling = "held-out cohorts and then subjects within sampled cohorts"
    else:
        sampling = "repeats and then subjects within sampled repeats"
    return (
        "<p>Contrasts are computed from predictions available for both strategies on the same held-out observations. "
        f"Each of the {count}paired bootstrap replicates resamples {html.escape(sampling)} and applies the identical draw to both strategies before recomputing each metric. "
        "For higher-is-better metrics the effect is A − B. For loss metrics it is B − A. Positive effects therefore favor Strategy A. "
        "The interval is the 2.5th to 97.5th percentile of the paired bootstrap effects.</p>"
    )


def _calibration_note() -> str:
    return (
        "<p>Calibration-in-the-large assesses systematic over- or under-prediction using an offset-only logistic recalibration and is ideally 0. "
        "Calibration intercept and slope come from logistic recalibration of the observed outcome on the logit of the predicted probability. Their ideal values are 0 and 1, respectively. "
        "For multiclass outcomes, calibration coefficients are computed one-vs-rest for each class and only explicitly labeled macro OvR summaries are reported as scalar performance metrics; reliability-curve data remain class-specific. "
        "Complete or quasi-complete separation, or nearly degenerate probabilities, can make the unpenalized calibration intercept and slope non-identifiable. Unstable estimates are reported as NA because extremely large coefficients are not interpretable.</p>"
    )


def _links_html(tables_dir: Path) -> str:
    names = [
        ("strategy_oof_performance.parquet", "Long-form out-of-fold estimates"),
        ("strategy_oof_calibration.parquet", "Reliability-curve data"),
        (
            "strategy_oof_calibration_coefficients.parquet",
            "Class-specific calibration coefficients",
        ),
        ("strategy_oof_pairwise_contrasts.parquet", "Paired out-of-fold contrasts"),
        (
            "strategy_oof_threshold_metrics.parquet",
            "Configured-threshold diagnostic metrics",
        ),
        ("strategy_oof_confusion_matrices.parquet", "Confusion matrices"),
        ("strategy_oof_decision_curve.parquet", "Decision-curve net benefit"),
        ("strategy_oof_roc_curve.parquet", "ROC coordinates"),
        ("strategy_oof_statistics_manifest.json", "Out-of-fold methods manifest"),
        ("strategy_oof_coverage.parquet", "Out-of-fold coverage"),
        ("strategy_oof_pairwise_coverage.parquet", "Paired out-of-fold coverage"),
    ]
    links = []
    for filename, label in names:
        path = tables_dir / filename
        if path.exists():
            links.append(
                f'<a href="tables/{html.escape(filename)}">{html.escape(label)}</a>'
            )
    if not links:
        return ""
    return '<p class="oof-downloads">' + " · ".join(links) + "</p>"


def _procedure_grid_html(procedure: pd.DataFrame) -> str:
    return _report_module._procedure_grid_html(procedure)


def _human_token(value: Any) -> str:
    label = _report_module._display_token(value)
    return label or "NA"


def _mpma_b_composition(
    root: Path, learner_labels: dict[str, str] | None = None
) -> pd.DataFrame:
    final = load_final_models(root)["MPMA-B"]
    representation = final.get("resolution", final.get("levels", ""))
    return pd.DataFrame(
        [
            {
                "Data representation": _human_token(representation),
                "Transformation": _human_token(final.get("count_transformation", "")),
                "Learner": _report_module._display_learner(
                    final.get("learner", ""), learner_labels
                )
                or "NA",
            }
        ]
    )


def _remove_mpma_b_section(text: str) -> str:
    start = text.find(_MPMA_B_SECTION_START)
    if start == -1:
        return text
    end = text.find(_MPMA_B_SECTION_END, start)
    if end == -1:
        raise RuntimeError(
            "Malformed MPMA-B composition section in the canonical HTML report"
        )
    end += len(_MPMA_B_SECTION_END)
    return text[:start] + text[end:]


def _replace_procedure_layout(text: str, report_dir: Path) -> str:
    procedure = _read_table(report_dir / "tables" / "evaluation_procedure.parquet")
    grid = _procedure_grid_html(procedure)
    if not grid:
        return text
    heading = text.find(_PROCEDURE_ANCHOR)
    if heading == -1:
        return text
    next_h2 = text.find("<h2 ", heading + len(_PROCEDURE_ANCHOR))
    grid_start = text.find(
        '<div class="procedure-grid">', heading + len(_PROCEDURE_ANCHOR)
    )
    if grid_start != -1 and (next_h2 == -1 or grid_start < next_h2):
        return text
    table_start = text.find(
        '<div class="table-wrap">', heading + len(_PROCEDURE_ANCHOR)
    )
    if table_start == -1:
        return text
    table_end = text.find("</div>", table_start)
    if table_end == -1:
        raise RuntimeError(
            "Malformed task-definition and evaluation-procedure table in the canonical HTML report"
        )
    table_end += len("</div>")
    return text[:table_start] + grid + text[table_end:]


def _insert_mpma_b_composition(
    text: str,
    report_dir: Path,
    learner_labels: dict[str, str] | None = None,
) -> str:
    text = _remove_mpma_b_section(text)
    table = _mpma_b_composition(report_dir.parent, learner_labels)
    if table.empty:
        return text
    heading = text.find(_PERFORMANCE_ANCHOR)
    if heading == -1:
        return text
    table_start = text.find(
        '<div class="table-wrap">', heading + len(_PERFORMANCE_ANCHOR)
    )
    if table_start == -1:
        return text
    table_end = text.find("</div>", table_start)
    if table_end == -1:
        raise RuntimeError(
            "Malformed Task performance table in the canonical HTML report"
        )
    table_end += len("</div>")
    section = "\n".join(
        [
            _MPMA_B_SECTION_START,
            "<h3>Final MPMA-B specification</h3>",
            _html_table(table),
            _MPMA_B_SECTION_END,
        ]
    )
    return text[:table_end] + "\n" + section + text[table_end:]


def _inject_compact_report_css(text: str) -> str:
    if ".procedure-grid {" in text:
        return text
    style_end = text.find("</style>")
    if style_end == -1:
        return text
    return (
        text[:style_end] + _report_module._publication_layout_css() + text[style_end:]
    )


def _enhance_compact_layout(
    text: str,
    report_dir: Path,
    learner_labels: dict[str, str] | None = None,
) -> str:
    text = _replace_procedure_layout(text, report_dir)
    text = _insert_mpma_b_composition(text, report_dir, learner_labels)
    return _inject_compact_report_css(text)


def _task_class_count(sweep: Any) -> int | None:
    data = getattr(sweep, "data", None)
    labels = getattr(data, "class_labels", None)
    if labels is not None:
        try:
            count = len(labels)
        except TypeError:
            count = 0
        if count > 0:
            return int(count)
    label_map = getattr(data, "label_map", None)
    if isinstance(label_map, dict) and label_map:
        values = {str(value) for value in label_map.values()}
        if values:
            return len(values)
    return None


def oof_section_html(
    report_dir: Path | str,
    n_classes: int | None = None,
    heading_level: int = 2,
    id_prefix: str = "",
    include_downloads: bool = False,
) -> str:
    report_dir = Path(report_dir)
    tables_dir = report_dir / "tables"
    performance = _read_table(tables_dir / "strategy_oof_performance.parquet")
    calibration_curve = _read_table(tables_dir / "strategy_oof_calibration.parquet")
    calibration_coefficients = _read_table(
        tables_dir / "strategy_oof_calibration_coefficients.parquet"
    )
    threshold_metrics = _read_table(
        tables_dir / "strategy_oof_threshold_metrics.parquet"
    )
    confusion = _read_table(tables_dir / "strategy_oof_confusion_matrices.parquet")
    decision_curve = _read_table(tables_dir / "strategy_oof_decision_curve.parquet")
    roc_curve = _read_table(tables_dir / "strategy_oof_roc_curve.parquet")
    contrasts = _read_table(tables_dir / "strategy_oof_pairwise_contrasts.parquet")
    manifest = _read_json(tables_dir / "strategy_oof_statistics_manifest.json")
    if n_classes is None:
        try:
            manifest_classes = int(manifest.get("n_classes", 0))
        except (TypeError, ValueError):
            manifest_classes = 0
        if manifest_classes > 0:
            n_classes = manifest_classes
    if performance.empty:
        return ""
    primary_table = _performance_display(performance)
    diagnostic_operating_table = _diagnostic_operating_display(performance, n_classes)
    multiclass_table = _multiclass_display(performance, n_classes)
    probability_table = _probability_display(performance, n_classes)
    calibration_table = _calibration_display(performance)
    calibration_coefficients_table = _calibration_coefficients_display(
        calibration_coefficients, n_classes
    )
    threshold_table = _threshold_metrics_display(threshold_metrics)
    confusion_table = _confusion_display(confusion)
    decision_curve_table = _decision_curve_display(decision_curve, manifest)
    decision_curve_figure = _decision_curve_figure_html(
        decision_curve, roc_curve, performance, manifest, report_dir
    )
    contrast_table = _contrast_display(contrasts, n_classes)
    design_table = _design_display(manifest)
    level = max(1, min(5, int(heading_level)))
    sublevel = min(6, level + 1)
    heading = f"h{level}"
    subheading = f"h{sublevel}"
    prefix = f"{id_prefix}-" if id_prefix else ""
    parts = [
        f'<section id="{prefix}oof-inference">',
        f'<{heading} id="{prefix}oof-performance">Pooled out-of-fold inference</{heading}>',
        "<p>Held-out predictions are pooled before the metrics are recomputed. Each estimate therefore reflects performance at the protocol-defined inference unit instead of an average of fold-level metric values. Point estimates and 95% confidence intervals are reported below.</p>",
        _methodology_html(manifest),
    ]
    if not design_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-design">Inference design</{subheading}>',
                _html_table(design_table),
            ]
        )
    if not primary_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-primary-performance">Performance</{subheading}>',
                _classification_metric_note(manifest, n_classes),
                _html_table(
                    primary_table,
                    raw_html_cols=set(primary_table.columns) - {"Strategy", "Estimand"},
                ),
            ]
        )
    if not diagnostic_operating_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-diagnostic-operating">Binary diagnostic operating point</{subheading}>',
                _diagnostic_operating_note(manifest),
                _html_table(diagnostic_operating_table),
            ]
        )
    if not multiclass_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-multiclass-performance">Multiclass summaries</{subheading}>',
                _html_table(multiclass_table),
            ]
        )
    if not probability_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-probability-quality">Probability quality</{subheading}>',
                _probability_quality_note(n_classes),
                _html_table(probability_table),
                _probability_semantics_html(manifest),
            ]
        )
    if not contrast_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-contrasts">Paired strategy contrasts</{subheading}>',
                _paired_contrast_note(manifest),
                _html_table(contrast_table),
            ]
        )
    if not calibration_table.empty:
        calibration_parts = [
            f'<{subheading} id="{prefix}oof-calibration-summary">Calibration</{subheading}>',
            _calibration_note(),
            _html_table(calibration_table),
        ]
        if not calibration_coefficients_table.empty:
            calibration_parts.extend(
                [
                    "<p>Class-specific one-vs-rest calibration coefficients:</p>",
                    _html_table(calibration_coefficients_table),
                ]
            )
        parts.extend(calibration_parts)
    elif not calibration_curve.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-calibration-summary">Calibration</{subheading}>',
                "<p>One-vs-rest reliability-curve data are available in "
                "<code>strategy_oof_calibration.parquet</code>.</p>",
            ]
        )
    if not confusion_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-confusion">Confusion matrices</{subheading}>',
                _confusion_note(),
                _html_table(confusion_table),
            ]
        )
    if not threshold_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-diagnostic-thresholds">Configured diagnostic thresholds</{subheading}>',
                _diagnostic_threshold_note(manifest),
                _html_table(threshold_table),
            ]
        )
    if not decision_curve.empty or not roc_curve.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-decision-curve">Decision-curve analysis and ROC</{subheading}>',
                _decision_curve_note(manifest),
            ]
        )
        if decision_curve_figure:
            parts.append(decision_curve_figure)
        if not decision_curve_table.empty:
            parts.append(_html_table(decision_curve_table))
        else:
            parts.append(
                "<p>No diagnostic probability thresholds were configured, so no compact threshold-specific net-benefit table is shown. The complete numerical curve remains available in <code>strategy_oof_decision_curve.parquet</code>.</p>"
            )
    if include_downloads:
        parts.append(_links_html(tables_dir))
    parts.append(_SECTION_END)
    return "\n".join(part for part in parts if part)


def _section_html(report_dir: Path, n_classes: int | None = None) -> str:
    return oof_section_html(report_dir, n_classes=n_classes, include_downloads=False)


def _remove_existing_section(text: str) -> str:
    start = text.find(_SECTION_START)
    if start == -1:
        return text
    end = text.find(_SECTION_END, start + len(_SECTION_START))
    if end == -1:
        raise RuntimeError(
            "Malformed existing pooled-OOF section in the canonical HTML report"
        )
    end += len(_SECTION_END)
    return text[:start] + text[end:]


def _statistics_section_end(text: str) -> int:
    start = text.find(_STATISTICS_ANCHOR)
    if start == -1:
        raise RuntimeError(
            "Cannot verify report layout: the statistical-comparisons HTML anchor was not found. The base report template has changed and the integration must be updated explicitly."
        )
    search_from = start + len(_STATISTICS_ANCHOR)
    candidates = [
        pos
        for pos in (text.find("<h2 ", search_from), text.find("<h3 ", search_from))
        if pos != -1
    ]
    if candidates:
        return min(candidates)
    footer = text.find(_FOOTER_ANCHOR, search_from)
    if footer != -1:
        return footer
    body_close = text.rfind("</body>")
    return body_close if body_close != -1 else len(text)


def _extract_compute_block(text: str) -> tuple[str, str]:
    start = text.find(_COMPUTE_ANCHOR)
    if start == -1:
        raise RuntimeError(
            "Cannot locate Computational resources. The base report template has changed and the integration must be updated explicitly."
        )
    next_h2 = text.find("<h2 ", start + len(_COMPUTE_ANCHOR))
    footer = text.find(_FOOTER_ANCHOR, start + len(_COMPUTE_ANCHOR))
    candidates = [x for x in (next_h2, footer) if x != -1]
    end = min(candidates) if candidates else len(text)
    block = text[start:end].strip()
    return text[:start] + text[end:], block


def enhance_html_report(
    report_dir: Path | str,
    n_classes: int | None = None,
    learner_labels: dict[str, str] | None = None,
) -> bool:
    report_dir = Path(report_dir)
    index_path = report_html_path(report_dir.parent)
    if not index_path.exists():
        raise FileNotFoundError(f"Report HTML does not exist: {index_path}")
    section = _section_html(report_dir, n_classes=n_classes)
    text = _enhance_compact_layout(
        index_path.read_text(encoding="utf-8"),
        report_dir,
        learner_labels,
    )
    text = _remove_existing_section(text)
    if section:
        insertion_pos = _statistics_section_end(text)
        text = text[:insertion_pos] + section + "\n" + text[insertion_pos:]
    text, compute_block = _extract_compute_block(text)
    footer_pos = text.find(_FOOTER_ANCHOR)
    if footer_pos == -1:
        body_close = text.rfind("</body>")
        footer_pos = body_close if body_close != -1 else len(text)
    abbreviations_pos = text.find('<h2 id="abbreviations">')
    compute_pos = abbreviations_pos if abbreviations_pos != -1 else footer_pos
    if compute_block:
        text = text[:compute_pos] + compute_block + "\n" + text[compute_pos:]
    text = text.replace("Learner family", "Learner type")
    index_path.write_text(text, encoding="utf-8")
    return bool(section)


def _terminal_oof_summary(report_dir: Path) -> None:
    performance = _read_table(
        report_dir / "tables" / "strategy_oof_performance.parquet"
    )
    if performance.empty:
        return
    display = _wide_metric_table(performance, _PRIMARY_METRIC_ORDER)
    if display.empty:
        return
    preferred = [
        "Strategy",
        "Estimand",
        "AUROC",
        "AUCPR",
        "Average precision",
        "MCC",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "Brier score",
        "Log loss",
    ]
    cols = [c for c in preferred if c in display.columns]
    if not cols:
        cols = list(display.columns[:8])
    from rich.table import Table

    table = Table(title="Pooled out-of-fold performance (95% CI)", show_lines=False)
    for col in cols:
        table.add_column(str(col), overflow="fold", no_wrap=False)
    for _, row in display[cols].iterrows():
        table.add_row(*[str(row.get(col, "")) for col in cols])
    console.print(table)


def _member_table_with_learner_type(
    table: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return table
    return table.rename(columns={"Learner family": "Learner type"})


def _procedure_table_publication(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or not {"Field", "Value"}.issubset(table.columns):
        return table
    out = table.copy()
    mask = out["Field"].astype(str).eq("Selection rule")
    out.loc[
        mask,
        "Value",
    ] = "Reported performance uses held-out outer evaluation predictions."
    return out


def write_report(sweep: Any, emit_html: bool = True) -> dict[str, Path]:
    root = Path(sweep.root())
    original_summary = _report_module._ensemble_summary_table
    original_procedure = _report_module._procedure_table
    original_terminal = _report_module._terminal_table

    def procedure_table(report_sweep: Any, report_root: Path) -> pd.DataFrame:
        return _procedure_table_publication(
            original_procedure(report_sweep, report_root)
        )

    def terminal_table(
        title: str,
        table: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return original_terminal(
            title,
            _member_table_with_learner_type(table),
            *args,
            **kwargs,
        )

    _report_module._ensemble_summary_table = _ensemble_summary_from_selected
    _report_module._procedure_table = procedure_table
    _report_module._terminal_table = terminal_table
    try:
        outputs = dict(_report_module.write_report(sweep, emit_html=emit_html))
    finally:
        _report_module._ensemble_summary_table = original_summary
        _report_module._procedure_table = original_procedure
        _report_module._terminal_table = original_terminal
    report_dir = root / "report"
    tables_dir = report_dir / "tables"
    performance_path = tables_dir / "strategy_oof_performance.parquet"
    if _read_table(performance_path).empty:
        raise RuntimeError(
            "Report statistics did not produce strategy_oof_performance.parquet."
        )
    inserted = False
    if emit_html:
        with phase_progress("Out-of-fold report integration", 2) as phase:
            phase.phase("HTML inference section")
            inserted = enhance_html_report(
                report_dir,
                n_classes=_task_class_count(sweep),
                learner_labels=_report_module._learner_display_map(sweep),
            )
            phase.phase("terminal inference summary")
            if inserted:
                _terminal_oof_summary(report_dir)
    new_outputs = {
        "strategy_oof_performance": tables_dir / "strategy_oof_performance.parquet",
        "strategy_oof_calibration": tables_dir / "strategy_oof_calibration.parquet",
        "strategy_oof_pairwise_contrasts": tables_dir
        / "strategy_oof_pairwise_contrasts.parquet",
        "strategy_oof_threshold_metrics": tables_dir
        / "strategy_oof_threshold_metrics.parquet",
        "strategy_oof_confusion_matrices": tables_dir
        / "strategy_oof_confusion_matrices.parquet",
        "strategy_oof_decision_curve": tables_dir
        / "strategy_oof_decision_curve.parquet",
        "strategy_oof_roc_curve": tables_dir / "strategy_oof_roc_curve.parquet",
        "strategy_oof_statistics_manifest": tables_dir
        / "strategy_oof_statistics_manifest.json",
        "strategy_oof_coverage": tables_dir / "strategy_oof_coverage.parquet",
        "strategy_oof_pairwise_coverage": tables_dir
        / "strategy_oof_pairwise_coverage.parquet",
    }
    existing = {name: path for name, path in new_outputs.items() if path.exists()}
    outputs.update(existing)
    if existing:
        stage("Out-of-fold inference", str(report_dir))
        path_table("Out-of-fold report outputs", existing)
    if inserted:
        success(f"Out-of-fold inference added to {report_html_path(root).name}")
    return outputs
