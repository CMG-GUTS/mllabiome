from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .evaluation_predictions import (
    evaluation_prediction_metadata,
    load_evaluation_predictions,
)
from .metrics import (
    _renormalize_proba,
    canonical_metric_name,
    compute_metrics,
    metric_is_loss,
)
from .storage import read_table, resolve_table_path, table_exists, write_table
from .utils import dump_json_standard

from .diagnostic_statistics import (
    _binary_confusion,
    _binary_threshold_counts,
    _bootstrap_decision_curve,
    _bootstrap_threshold_estimands,
    _confusion_components,
    _confusion_estimands,
    _decision_curve_grid,
    _confusion_rows,
    _decision_curve_rows,
    _decision_curve_point_estimands,
    _decision_curve_values,
    _diagnostic_values_from_counts,
    _mean_decision_vectors,
    _mean_diagnostic_vectors,
    _nanmean_vectors,
    _operating_confusion_rows,
    _roc_curve_rows,
    _operating_confusion_estimands,
    _operating_confusion_matrix,
    _safe_ratio_array,
    _threshold_metric_rows,
    _threshold_diagnostic_values,
    _threshold_point_estimands,
    _validated_diagnostic_thresholds,
)
from .oof_statistics import (
    _binary_auc_pr_auc_ap,
    _bootstrap_oof_estimands,
    _calibration_binary,
    _calibration_coefficient_rows,
    _fast_classification_metrics,
    _mean_metric_dicts,
    _oof_metrics,
    _point_oof_estimands,
    _proper_metrics,
    _performance_rows,
    _prepare_oof_frame,
    _probability_columns,
    _probability_semantics,
    _resample_subjects,
    _sigmoid,
    _reliability_rows,
)
from .paired_statistics import (
    _coverage_row,
    _match_key_columns,
    _matched_frames,
    _oof_design,
    _paired_bootstrap_advantages,
    _paired_contrast_rows,
    _paired_point_estimands,
    _paired_resample_indices,
    _subject_cluster_indices,
)
from .statistics_common import (
    DIAGNOSTIC_METRICS,
    OOF_METRIC_ORDER,
    _LODO_PROTOCOLS,
    _OOF_CONTRAST_METRICS,
    _ensure_outer_split_key,
    _repeat_id,
    _stable_seed,
)

DISPLAY_METRICS = ("AUROC", "AUCPR", "AP", "MCC", "F1w", "Precision", "Recall")
METRIC_LABELS = {
    "AUROC": "AUROC",
    "AUROC_macro": "AUROC macro (OvR)",
    "AUROC_weighted": "AUROC weighted (OvR)",
    "AUCPR": "AUCPR",
    "AUCPR_macro": "AUCPR macro (OvR)",
    "AUCPR_weighted": "AUCPR weighted (OvR)",
    "AP": "Average precision",
    "AP_macro": "Average precision macro",
    "MCC": "MCC",
    "F1w": "F1w",
    "Precision": "Precision",
    "Recall": "Recall",
    "Sensitivity": "Sensitivity",
    "Specificity": "Specificity",
    "PPV": "PPV",
    "NPV": "NPV",
    "log_loss": "Log loss",
    "brier": "Brier score",
}


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _strategy_prediction_frames(
    root: Path,
    strategy_rows: list[dict[str, Any]],
    selection_metric: str,
) -> dict[str, pd.DataFrame]:
    strategies = [
        str(row.get("Strategy", "")).strip()
        for row in strategy_rows
        if str(row.get("Strategy", "")).strip()
    ]
    frames: dict[str, pd.DataFrame] = {}
    for strategy in dict.fromkeys(strategies):
        frame = _ensure_outer_split_key(
            load_evaluation_predictions(root, strategy, selection_metric)
        )
        if not frame.empty:
            frames[strategy] = frame
    return frames


def _outer_split_sizes(root: Path) -> pd.DataFrame:
    frame = _read_table(root / "tables" / "cv_splits.parquet")
    required = {"stage", "split_key", "role", "sample_index"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[
        frame["stage"].astype(str).eq("outer")
        & frame["role"].astype(str).isin(["train", "test"])
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    counts = (
        frame.groupby(["split_key", "role"], sort=False)["sample_index"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
        .rename(
            columns={
                "split_key": "outer_split_key",
                "train": "n_train",
                "test": "n_test",
            }
        )
    )
    for column in ("n_train", "n_test"):
        if column not in counts.columns:
            counts[column] = 0
        counts[column] = (
            pd.to_numeric(counts[column], errors="coerce").fillna(0).astype(int)
        )
    counts["outer_split_key"] = counts["outer_split_key"].astype(str)
    return counts[["outer_split_key", "n_train", "n_test"]]


def _metric_frame(
    predictions: pd.DataFrame, split_sizes: pd.DataFrame | None = None
) -> pd.DataFrame:
    if predictions.empty or "outer_split_key" not in predictions.columns:
        return pd.DataFrame()
    pcols = [column for column in predictions.columns if column.startswith("proba_")]
    if not pcols or "y_true" not in predictions.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    classes = np.arange(len(pcols), dtype=int)
    for outer_split_key, group in predictions.groupby("outer_split_key", sort=True):
        y_true = pd.to_numeric(group["y_true"], errors="coerce")
        valid = y_true.notna()
        if not bool(valid.any()):
            continue
        y_true_array = y_true.loc[valid].astype(int).to_numpy()
        proba = (
            group.loc[valid, pcols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        if not np.isfinite(proba).all():
            continue
        proba = _renormalize_proba(proba, len(classes))
        if "y_pred" in group.columns:
            y_pred = pd.to_numeric(group.loc[valid, "y_pred"], errors="coerce")
            if y_pred.isna().any():
                y_pred_array = classes[np.argmax(proba, axis=1)]
            else:
                y_pred_array = y_pred.astype(int).to_numpy()
        else:
            y_pred_array = classes[np.argmax(proba, axis=1)]
        metrics = compute_metrics(y_true_array, y_pred_array, proba, classes)
        rows.append(
            {
                "outer_split_key": str(outer_split_key),
                "n_samples": int(len(y_true_array)),
                **metrics,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty or split_sizes is None or split_sizes.empty:
        return out
    sizes = split_sizes.copy()
    sizes["outer_split_key"] = sizes["outer_split_key"].astype(str)
    return out.merge(sizes, on="outer_split_key", how="left", validate="many_to_one")


def _bootstrap_unit_mean(
    values: pd.DataFrame,
    metric: str,
    protocol: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = values[["outer_split_key", metric]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[np.isfinite(frame[metric].to_numpy(dtype=float))]
    if frame.empty:
        return np.asarray([], dtype=float)
    out = np.empty(int(n_bootstrap), dtype=float)
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        values_array = frame[metric].to_numpy(dtype=float)
        n = len(values_array)
        indices = rng.integers(0, n, size=(int(n_bootstrap), n))
        return values_array[indices].mean(axis=1)
    frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
    groups = [
        group[metric].to_numpy(dtype=float)
        for _, group in frame.groupby("repeat", sort=True)
    ]
    if not groups:
        return np.asarray([], dtype=float)
    n_repeats = len(groups)
    for index in range(int(n_bootstrap)):
        sampled_repeat_indices = rng.integers(0, n_repeats, size=n_repeats)
        sampled: list[float] = []
        for repeat_index in sampled_repeat_indices:
            values_array = groups[int(repeat_index)]
            sampled.extend(
                values_array[
                    rng.integers(0, len(values_array), size=len(values_array))
                ].tolist()
            )
        out[index] = float(np.mean(sampled))
    return out


def _summary_rows(
    unit_metrics: pd.DataFrame,
    protocol: str,
    n_bootstrap: int,
    random_state: int,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, group in unit_metrics.groupby("Strategy", sort=False):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if values.empty:
                continue
            seed = _stable_seed(random_state, "summary", strategy, metric)
            boot = _bootstrap_unit_mean(
                group,
                metric,
                protocol,
                n_bootstrap,
                np.random.default_rng(seed),
            )
            ci_low = ci_high = np.nan
            if len(boot):
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "Strategy": str(strategy),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "estimate": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                    "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    "n_outer_units": int(len(values)),
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_method": (
                        "cluster_outer_cohort"
                        if str(protocol).lower() in _LODO_PROTOCOLS
                        else "hierarchical_repeat_outer_fold"
                    ),
                    "valid_bootstrap": int(np.isfinite(boot).sum()),
                }
            )
    return pd.DataFrame(rows)


from .statistical_tests import (
    _corrected_resampled_t_test,
    _exact_sign_flip_test,
    _paired_bootstrap_difference,
)


def _holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["p_holm"] = np.nan
    values = pd.to_numeric(out["p_value"], errors="coerce").to_numpy(dtype=float)
    indices = out.index[np.isfinite(values)]
    ordered = sorted(indices, key=lambda index: float(out.loc[index, "p_value"]))
    m = len(ordered)
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * float(out.loc[index, "p_value"]))
        running = max(running, adjusted)
        out.loc[index, "p_holm"] = running
    out["significant_holm_0_05"] = pd.to_numeric(out["p_holm"], errors="coerce").lt(
        0.05
    )
    return out


def _pairwise_rows(
    unit_metrics: pd.DataFrame,
    protocol: str,
    n_bootstrap: int,
    random_state: int,
    metrics: tuple[str, ...] = DISPLAY_METRICS,
) -> pd.DataFrame:
    if unit_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    strategies = list(dict.fromkeys(unit_metrics["Strategy"].astype(str).tolist()))
    for strategy_a, strategy_b in itertools.combinations(strategies, 2):
        left = unit_metrics[unit_metrics["Strategy"].eq(strategy_a)].copy()
        right = unit_metrics[unit_metrics["Strategy"].eq(strategy_b)].copy()
        merged = left.merge(
            right,
            on="outer_split_key",
            suffixes=("_a", "_b"),
            how="inner",
        )
        if merged.empty:
            continue
        for metric in metrics:
            metric_a = f"{metric}_a"
            metric_b = f"{metric}_b"
            if metric_a not in merged.columns or metric_b not in merged.columns:
                continue
            valid = merged[
                np.isfinite(
                    pd.to_numeric(merged[metric_a], errors="coerce").to_numpy(
                        dtype=float
                    )
                )
                & np.isfinite(
                    pd.to_numeric(merged[metric_b], errors="coerce").to_numpy(
                        dtype=float
                    )
                )
            ].copy()
            if valid.empty:
                continue
            estimate_a = float(pd.to_numeric(valid[metric_a], errors="coerce").mean())
            estimate_b = float(pd.to_numeric(valid[metric_b], errors="coerce").mean())
            seed = _stable_seed(
                random_state,
                "pairwise",
                strategy_a,
                strategy_b,
                metric,
            )
            ci_low, ci_high = _paired_bootstrap_difference(
                valid,
                metric_a,
                metric_b,
                protocol,
                n_bootstrap,
                np.random.default_rng(seed),
            )
            if str(protocol).lower() in _LODO_PROTOCOLS:
                statistic, p_value, test = _exact_sign_flip_test(
                    (
                        pd.to_numeric(valid[metric_a], errors="coerce")
                        - pd.to_numeric(valid[metric_b], errors="coerce")
                    ).to_numpy(dtype=float),
                    seed,
                )
            else:
                statistic, p_value, test = _corrected_resampled_t_test(
                    valid,
                    metric_a,
                    metric_b,
                )
            rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "estimate_a": estimate_a,
                    "estimate_b": estimate_b,
                    "difference_a_minus_b": estimate_a - estimate_b,
                    "difference_ci_low": ci_low,
                    "difference_ci_high": ci_high,
                    "test": test,
                    "test_statistic": statistic,
                    "p_value": p_value,
                    "n_paired_outer_units": int(len(valid)),
                }
            )
    return _holm_adjust(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _run_oof_statistics(
    frames: dict[str, pd.DataFrame],
    protocol: str,
    n_bootstrap: int,
    random_state: int,
    calibration_bins: int,
    diagnostic_thresholds: tuple[float, ...],
    decision_curve_thresholds: np.ndarray,
) -> dict[str, Any]:
    prepared: dict[str, pd.DataFrame] = {}
    semantics: dict[str, dict[str, Any]] = {}
    design: dict[str, dict[str, int]] = {}
    performance_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    calibration_coefficient_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    decision_curve_rows: list[dict[str, Any]] = []
    roc_curve_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    class_labels: tuple[str, ...] = ()
    n_classes = 0
    for strategy, frame in frames.items():
        current = _prepare_oof_frame(frame, protocol)
        if current.empty:
            continue
        current_pcols = _probability_columns(current)
        current_labels = tuple(column[len("proba_") :] for column in current_pcols)
        if class_labels and current_labels != class_labels:
            raise ValueError(
                "Displayed strategies do not share the same probability-column class order."
            )
        class_labels = current_labels
        current_n_classes = len(current_pcols)
        prepared[strategy] = current
        semantics[strategy] = _probability_semantics(current)
        design[strategy] = _oof_design(current, protocol)
        n_classes = max(n_classes, current_n_classes)
        probability_valid = bool(semantics[strategy].get("valid", False))
        performance_rows.extend(
            _performance_rows(
                strategy,
                current,
                protocol,
                probability_valid,
                n_bootstrap,
                random_state,
            )
        )
        calibration_rows.extend(
            _reliability_rows(
                strategy,
                current,
                probability_valid,
                calibration_bins,
            )
        )
        calibration_coefficient_rows.extend(
            _calibration_coefficient_rows(
                strategy,
                current,
                protocol,
                probability_valid,
            )
        )
        confusion_rows.extend(
            _operating_confusion_rows(strategy, current, protocol, current_labels)
        )
        if current_n_classes == 2:
            roc_curve_rows.extend(
                _roc_curve_rows(
                    strategy,
                    current,
                    protocol,
                    current_labels[1],
                    operating_threshold=0.5,
                )
            )
            if probability_valid:
                threshold_rows.extend(
                    _threshold_metric_rows(
                        strategy,
                        current,
                        protocol,
                        diagnostic_thresholds,
                        n_bootstrap,
                        random_state,
                        current_labels[1],
                    )
                )
                confusion_rows.extend(
                    _confusion_rows(
                        strategy,
                        current,
                        protocol,
                        diagnostic_thresholds,
                        (current_labels[0], current_labels[1]),
                    )
                )
                decision_curve_rows.extend(
                    _decision_curve_rows(
                        strategy,
                        current,
                        protocol,
                        decision_curve_thresholds,
                        n_bootstrap,
                        random_state,
                        current_labels[1],
                    )
                )
        coverage_rows.append(_coverage_row(strategy, current, protocol))
    contrasts, pairwise_coverage = _paired_contrast_rows(
        prepared,
        protocol,
        semantics,
        n_bootstrap,
        random_state,
    )
    return {
        "performance": pd.DataFrame(performance_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "calibration_coefficients": pd.DataFrame(calibration_coefficient_rows),
        "threshold_metrics": pd.DataFrame(threshold_rows),
        "confusion_matrices": pd.DataFrame(confusion_rows),
        "decision_curve": pd.DataFrame(decision_curve_rows),
        "roc_curve": pd.DataFrame(roc_curve_rows),
        "contrasts": contrasts,
        "coverage": pd.DataFrame(coverage_rows),
        "pairwise_coverage": pairwise_coverage,
        "probability_semantics": semantics,
        "observed_oof_design": design,
        "n_classes": int(n_classes),
        "class_labels": list(class_labels),
    }


def _selection_metric_from_run(
    root: Path,
    strategy_rows: list[dict[str, Any]],
    explicit: str | None,
) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    manifest = _read_json(root / "manifest.json")
    sweep = manifest.get("sweep", {})
    if isinstance(sweep, dict):
        evaluation = sweep.get("evaluation", {})
        if isinstance(evaluation, dict):
            metric = str(evaluation.get("optimize_metric", "")).strip()
            if metric:
                return metric
    for row in strategy_rows:
        if str(row.get("Strategy", "")).strip() == "MPMA-B":
            metric = str(
                row.get("selection_metric", row.get("optimize_metric", ""))
            ).strip()
            if metric:
                return metric
    return "log_loss"


def _file_signature(path: Path) -> dict[str, Any]:
    physical = resolve_table_path(path) if path.suffix == ".parquet" else path
    if not physical.exists():
        return {"path": str(path), "exists": False}
    stat = physical.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _statistics_fingerprint(
    root: Path,
    protocol: str,
    strategy_rows: list[dict[str, Any]],
    n_bootstrap: int,
    random_state: int,
    selection_metric: str,
    calibration_bins: int,
    diagnostic_thresholds: tuple[float, ...],
    decision_curve_thresholds: np.ndarray,
) -> str:
    relevant_rows = [
        {key: row.get(key) for key in ("Strategy", "source") if key in row}
        for row in strategy_rows
    ]
    paths = [
        root / "predictions" / "outer_predictions.parquet",
        root / "predictions" / "mpma_b_outer_predictions.parquet",
        root / "tables" / "mpma_b_outer_selection.parquet",
        root / "ensembling" / "ensemble_predictions.parquet",
        root / "ensembling" / "mpma_e_outer_selection.parquet",
        root / "configs.parquet",
        root / "inner_results" / "inner_results.parquet",
        root / "tables" / "qualification_gate.parquet",
        root / "tables" / "cv_splits.parquet",
    ]
    payload = {
        "protocol": str(protocol),
        "strategies": relevant_rows,
        "n_bootstrap": int(n_bootstrap),
        "random_state": int(random_state),
        "selection_metric": str(selection_metric),
        "display_metrics": list(DISPLAY_METRICS),
        "oof_metrics": list(OOF_METRIC_ORDER),
        "calibration_bins": int(calibration_bins),
        "diagnostic_thresholds": list(diagnostic_thresholds),
        "decision_curve_thresholds": [
            float(value) for value in decision_curve_thresholds
        ],
        "files": [_file_signature(path) for path in paths],
        "schema_version": 14,
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cached_result(
    root: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    tables = root / "report" / "tables"
    manifest_path = tables / "strategy_statistics_manifest.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("fingerprint", "")) != fingerprint:
        return None
    unit_path = tables / "strategy_outer_unit_metrics.parquet"
    summary_path = tables / "strategy_metrics_bootstrap.parquet"
    pairwise_path = tables / "strategy_pairwise_tests.parquet"
    oof_performance_path = tables / "strategy_oof_performance.parquet"
    oof_calibration_path = tables / "strategy_oof_calibration.parquet"
    oof_calibration_coefficients_path = (
        tables / "strategy_oof_calibration_coefficients.parquet"
    )
    oof_contrasts_path = tables / "strategy_oof_pairwise_contrasts.parquet"
    oof_manifest_path = tables / "strategy_oof_statistics_manifest.json"
    oof_coverage_path = tables / "strategy_oof_coverage.parquet"
    oof_pairwise_coverage_path = tables / "strategy_oof_pairwise_coverage.parquet"
    oof_threshold_metrics_path = tables / "strategy_oof_threshold_metrics.parquet"
    oof_confusion_matrices_path = tables / "strategy_oof_confusion_matrices.parquet"
    oof_decision_curve_path = tables / "strategy_oof_decision_curve.parquet"
    oof_roc_curve_path = tables / "strategy_oof_roc_curve.parquet"
    required = (
        unit_path,
        summary_path,
        pairwise_path,
        oof_performance_path,
        oof_calibration_path,
        oof_calibration_coefficients_path,
        oof_contrasts_path,
        oof_manifest_path,
        oof_coverage_path,
        oof_pairwise_coverage_path,
        oof_threshold_metrics_path,
        oof_confusion_matrices_path,
        oof_decision_curve_path,
        oof_roc_curve_path,
    )
    if any(not table_exists(path) for path in required):
        return None
    return {
        "unit_metrics": _read_table(unit_path),
        "summary": _read_table(summary_path),
        "pairwise": _read_table(pairwise_path),
        "oof_performance": _read_table(oof_performance_path),
        "oof_calibration": _read_table(oof_calibration_path),
        "oof_calibration_coefficients": _read_table(oof_calibration_coefficients_path),
        "oof_contrasts": _read_table(oof_contrasts_path),
        "oof_threshold_metrics": _read_table(oof_threshold_metrics_path),
        "oof_confusion_matrices": _read_table(oof_confusion_matrices_path),
        "oof_decision_curve": _read_table(oof_decision_curve_path),
        "oof_roc_curve": _read_table(oof_roc_curve_path),
        "unit_metrics_path": unit_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "oof_performance_path": oof_performance_path,
        "oof_calibration_path": oof_calibration_path,
        "oof_calibration_coefficients_path": oof_calibration_coefficients_path,
        "oof_contrasts_path": oof_contrasts_path,
        "oof_threshold_metrics_path": oof_threshold_metrics_path,
        "oof_confusion_matrices_path": oof_confusion_matrices_path,
        "oof_decision_curve_path": oof_decision_curve_path,
        "oof_roc_curve_path": oof_roc_curve_path,
        "oof_manifest_path": oof_manifest_path,
        "oof_coverage_path": oof_coverage_path,
        "oof_pairwise_coverage_path": oof_pairwise_coverage_path,
        "manifest_path": manifest_path,
        "selection_metric": str(manifest.get("selection_metric")),
        "cache_hit": True,
    }


def run_report_statistics(
    root: Path | str,
    protocol: str,
    strategy_rows: list[dict[str, Any]],
    n_bootstrap: int = 2000,
    random_state: int = 42,
    *,
    selection_metric: str | None = None,
    calibration_bins: int = 10,
    diagnostic_thresholds: Any = (),
    decision_curve_min_threshold: float = 0.01,
    decision_curve_max_threshold: float = 0.99,
    decision_curve_points: int = 99,
) -> dict[str, Any]:
    root = Path(root)
    diagnostic_thresholds = _validated_diagnostic_thresholds(diagnostic_thresholds)
    decision_curve_thresholds = _decision_curve_grid(
        decision_curve_min_threshold,
        decision_curve_max_threshold,
        decision_curve_points,
    )
    decision_curve_thresholds = np.unique(
        np.concatenate(
            [
                decision_curve_thresholds,
                np.asarray([0.5], dtype=float),
                np.asarray(diagnostic_thresholds, dtype=float),
            ]
        )
    )
    resolved_selection = _selection_metric_from_run(
        root, strategy_rows, selection_metric
    )
    fingerprint = _statistics_fingerprint(
        root,
        protocol,
        strategy_rows,
        n_bootstrap,
        random_state,
        resolved_selection,
        calibration_bins,
        diagnostic_thresholds,
        decision_curve_thresholds,
    )
    cached = _cached_result(root, fingerprint)
    if cached is not None:
        return cached
    frames = _strategy_prediction_frames(root, strategy_rows, resolved_selection)
    split_sizes = _outer_split_sizes(root)
    unit_frames: list[pd.DataFrame] = []
    for strategy, predictions in frames.items():
        metrics = _metric_frame(predictions, split_sizes)
        if metrics.empty:
            continue
        metrics.insert(0, "Strategy", strategy)
        unit_frames.append(metrics)
    unit_metrics = (
        pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    )
    metric_lookup = {
        str(column).casefold(): str(column) for column in unit_metrics.columns
    }
    requested_metric = canonical_metric_name(str(resolved_selection).strip())
    selected_metric = metric_lookup.get(requested_metric.casefold(), requested_metric)
    inference_metrics = tuple(
        dict.fromkeys(
            metric
            for metric in (*DISPLAY_METRICS, selected_metric)
            if unit_metrics.empty or metric in unit_metrics.columns
        )
    )
    summary_metrics = inference_metrics
    summary = (
        _summary_rows(
            unit_metrics,
            protocol,
            int(n_bootstrap),
            int(random_state),
            summary_metrics,
        )
        if not unit_metrics.empty
        else pd.DataFrame()
    )
    pairwise = (
        _pairwise_rows(
            unit_metrics,
            protocol,
            int(n_bootstrap),
            int(random_state),
            inference_metrics,
        )
        if not unit_metrics.empty
        else pd.DataFrame()
    )
    tables = root / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    unit_path = tables / "strategy_outer_unit_metrics.parquet"
    summary_path = tables / "strategy_metrics_bootstrap.parquet"
    pairwise_path = tables / "strategy_pairwise_tests.parquet"
    manifest_path = tables / "strategy_statistics_manifest.json"
    write_table(unit_path, unit_metrics)
    write_table(summary_path, summary)
    write_table(pairwise_path, pairwise)
    advanced = _run_oof_statistics(
        frames,
        protocol,
        int(n_bootstrap),
        int(random_state),
        int(calibration_bins),
        diagnostic_thresholds,
        decision_curve_thresholds,
    )
    oof_performance_path = tables / "strategy_oof_performance.parquet"
    oof_calibration_path = tables / "strategy_oof_calibration.parquet"
    oof_calibration_coefficients_path = (
        tables / "strategy_oof_calibration_coefficients.parquet"
    )
    oof_contrasts_path = tables / "strategy_oof_pairwise_contrasts.parquet"
    oof_manifest_path = tables / "strategy_oof_statistics_manifest.json"
    oof_coverage_path = tables / "strategy_oof_coverage.parquet"
    oof_pairwise_coverage_path = tables / "strategy_oof_pairwise_coverage.parquet"
    oof_threshold_metrics_path = tables / "strategy_oof_threshold_metrics.parquet"
    oof_confusion_matrices_path = tables / "strategy_oof_confusion_matrices.parquet"
    oof_decision_curve_path = tables / "strategy_oof_decision_curve.parquet"
    oof_roc_curve_path = tables / "strategy_oof_roc_curve.parquet"
    write_table(oof_performance_path, advanced["performance"])
    write_table(oof_calibration_path, advanced["calibration"])
    write_table(oof_calibration_coefficients_path, advanced["calibration_coefficients"])
    write_table(oof_contrasts_path, advanced["contrasts"])
    write_table(oof_coverage_path, advanced["coverage"])
    write_table(oof_pairwise_coverage_path, advanced["pairwise_coverage"])
    write_table(oof_threshold_metrics_path, advanced["threshold_metrics"])
    write_table(oof_confusion_matrices_path, advanced["confusion_matrices"])
    write_table(oof_decision_curve_path, advanced["decision_curve"])
    write_table(oof_roc_curve_path, advanced["roc_curve"])
    oof_manifest = {
        "schema_version": 11,
        "protocol": str(protocol),
        "strategies": list(frames),
        "n_classes": int(advanced["n_classes"]),
        "class_labels": advanced["class_labels"],
        "positive_class_index": 1 if int(advanced["n_classes"]) == 2 else None,
        "positive_class_label": (
            advanced["class_labels"][1]
            if int(advanced["n_classes"]) == 2 and len(advanced["class_labels"]) == 2
            else None
        ),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": 0.95,
        "calibration_bins": int(calibration_bins),
        "diagnostic_thresholds": list(diagnostic_thresholds),
        "decision_curve": {
            "minimum_threshold": float(decision_curve_thresholds[0]),
            "maximum_threshold": float(decision_curve_thresholds[-1]),
            "points": int(len(decision_curve_thresholds)),
            "net_benefit": "TP/N - FP/N * threshold/(1-threshold)",
            "reference_strategies": ["treat_none", "treat_all"],
            "ordinary_operating_threshold": 0.5,
        },
        "roc_curve": {
            "ordinary_operating_threshold": 0.5,
            "mean_curve_grid_points": 401,
            "coordinates": "false_positive_rate=1-specificity, true_positive_rate=sensitivity",
        },
        "metric_order": list(OOF_METRIC_ORDER),
        "estimands": (
            ["pooled_sample_weighted", "cohort_macro_equal_weight"]
            if str(protocol).lower() in _LODO_PROTOCOLS
            else ["mean_repeat_pooled_oof"]
        ),
        "observed_oof_design": advanced["observed_oof_design"],
        "probability_semantics": advanced["probability_semantics"],
        "bootstrap": (
            "two-stage cohort-and-subject cluster bootstrap"
            if str(protocol).lower() in _LODO_PROTOCOLS
            else "subject-cluster bootstrap preserving all repeat-specific predictions per sampled subject"
        ),
        "subject_identifier": "subject_id when available, otherwise sample_id",
        "paired_contrasts": "matched held-out observations with shared bootstrap draws",
        "metric_direction": {
            metric: ("lower" if metric_is_loss(metric) else "higher")
            for metric in _OOF_CONTRAST_METRICS
        },
        "scope": "displayed report strategies only",
        "evaluation_predictions": {
            strategy: evaluation_prediction_metadata(strategy) for strategy in frames
        },
        "final_refit_specifications_excluded_from_performance_estimation": True,
    }
    dump_json_standard(oof_manifest, oof_manifest_path)
    manifest = {
        "schema_version": 11,
        "fingerprint": fingerprint,
        "protocol": str(protocol),
        "strategies": list(frames),
        "display_metrics_bootstrapped": list(summary_metrics),
        "pairwise_metrics": list(inference_metrics),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": 0.95,
        "nested_cv_bootstrap": "hierarchical repeat/outer-fold bootstrap of held-out displayed-strategy metrics",
        "lodo_bootstrap": "cluster bootstrap of held-out cohorts for displayed-strategy metrics",
        "nested_cv_pairwise_test": "Nadeau-Bengio corrected resampled paired t-test using the mean observed outer-fold n_test/n_train ratio from the persisted split manifest",
        "lodo_pairwise_test": "paired sign-flip randomization test at held-out cohort level",
        "multiple_testing": "Holm adjustment across all displayed strategy-pair and metric hypotheses",
        "scope": "displayed report strategies only",
        "selection_metric": selected_metric,
        "diagnostic_thresholds": list(diagnostic_thresholds),
        "decision_curve_min_threshold": float(decision_curve_thresholds[0]),
        "decision_curve_max_threshold": float(decision_curve_thresholds[-1]),
        "decision_curve_points": int(len(decision_curve_thresholds)),
        "cache": "statistics are reused when source artefacts and statistical settings are unchanged",
        "evaluation_predictions": {
            strategy: evaluation_prediction_metadata(strategy) for strategy in frames
        },
        "final_refit_specifications_excluded_from_performance_estimation": True,
    }
    dump_json_standard(manifest, manifest_path)
    return {
        "unit_metrics": unit_metrics,
        "summary": summary,
        "pairwise": pairwise,
        "unit_metrics_path": unit_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "oof_performance": advanced["performance"],
        "oof_calibration": advanced["calibration"],
        "oof_calibration_coefficients": advanced["calibration_coefficients"],
        "oof_contrasts": advanced["contrasts"],
        "oof_threshold_metrics": advanced["threshold_metrics"],
        "oof_confusion_matrices": advanced["confusion_matrices"],
        "oof_decision_curve": advanced["decision_curve"],
        "oof_roc_curve": advanced["roc_curve"],
        "oof_performance_path": oof_performance_path,
        "oof_calibration_path": oof_calibration_path,
        "oof_calibration_coefficients_path": oof_calibration_coefficients_path,
        "oof_contrasts_path": oof_contrasts_path,
        "oof_threshold_metrics_path": oof_threshold_metrics_path,
        "oof_confusion_matrices_path": oof_confusion_matrices_path,
        "oof_decision_curve_path": oof_decision_curve_path,
        "oof_roc_curve_path": oof_roc_curve_path,
        "oof_manifest_path": oof_manifest_path,
        "oof_coverage_path": oof_coverage_path,
        "oof_pairwise_coverage_path": oof_pairwise_coverage_path,
        "manifest_path": manifest_path,
        "selection_metric": selected_metric,
        "cache_hit": False,
    }
