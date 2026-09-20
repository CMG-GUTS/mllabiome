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

from .storage import read_table, write_table, table_exists, resolve_table_path
from .final_models import build_final_models, fixed_strategy_predictions
from .metrics import _renormalize_proba, compute_metrics, metric_is_loss
from .utils import dump_json_standard

DISPLAY_METRICS = ("AUC", "PR_AUC", "nMCC", "F1w", "Precision", "Recall")
OOF_METRIC_ORDER = (
    "AUC",
    "AUC_macro",
    "AUC_weighted",
    "PR_AUC",
    "PR_AUC_macro",
    "nMCC",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
    "BalAcc",
    "Accuracy",
    "Brier",
    "Brier_multiclass",
    "LogLoss",
    "CalibrationInTheLarge",
    "CalibrationIntercept",
    "CalibrationSlope",
)
_OOF_CONTRAST_METRICS = (
    "AUC",
    "AUC_macro",
    "AUC_weighted",
    "PR_AUC",
    "PR_AUC_macro",
    "nMCC",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
    "BalAcc",
    "Accuracy",
    "Brier",
    "Brier_multiclass",
    "LogLoss",
)
METRIC_LABELS = {
    "AUC": "ROC-AUC",
    "PR_AUC": "PR-AUC (AP)",
    "nMCC": "nMCC",
    "F1w": "F1w",
    "Precision": "Precision",
    "Recall": "Recall",
}
_LODO_PROTOCOLS = {"lodo", "leave_one_dataset_out"}


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


def _ensure_outer_split_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outer_split_key" not in out.columns and "split_key" in out.columns:
        out["outer_split_key"] = out["split_key"]
    if "outer_split_key" in out.columns:
        out["outer_split_key"] = out["outer_split_key"].astype(str)
    return out


def _strategy_prediction_frames(
    root: Path, strategy_rows: list[dict[str, Any]]
) -> dict[str, pd.DataFrame]:
    rows = {
        str(row.get("Strategy", "")).strip(): row
        for row in strategy_rows
        if str(row.get("Strategy", "")).strip()
    }
    frames: dict[str, pd.DataFrame] = {}
    if "MPMA-B" in rows or "MPMA-E" in rows:
        build_final_models(root)
    outer = _ensure_outer_split_key(
        _read_table(root / "predictions" / "outer_predictions.parquet")
    )
    if "MPMA-B" in rows:
        frame = _ensure_outer_split_key(fixed_strategy_predictions(root, "MPMA-B"))
        if not frame.empty:
            frames["MPMA-B"] = frame
    if "MPMA-E" in rows:
        try:
            frame = _ensure_outer_split_key(fixed_strategy_predictions(root, "MPMA-E"))
            if not frame.empty:
                frames["MPMA-E"] = frame
        except ValueError as exc:
            if "No final MPMA-E specification" not in str(exc):
                raise
    if not outer.empty and "config_id" in outer.columns:
        outer["config_id"] = outer["config_id"].astype(str)
        for strategy in ("AutoML", "Baseline RF", "SIAMCAT"):
            row = rows.get(strategy)
            if row is None:
                continue
            config_id = str(row.get("config_id", "")).strip()
            if not config_id:
                continue
            frame = outer[outer["config_id"].eq(config_id)].copy()
            if not frame.empty:
                frames[strategy] = frame
    return frames


def _metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def _repeat_id(split_key: str) -> str:
    match = re.match(r"^r(\d+)_o\d+$", str(split_key))
    return f"r{match.group(1)}" if match else "r0"


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


def _paired_bootstrap_difference(
    merged: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    protocol: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    frame = merged[["outer_split_key", metric_a, metric_b]].copy()
    frame[metric_a] = pd.to_numeric(frame[metric_a], errors="coerce")
    frame[metric_b] = pd.to_numeric(frame[metric_b], errors="coerce")
    frame = frame[
        np.isfinite(frame[metric_a].to_numpy(dtype=float))
        & np.isfinite(frame[metric_b].to_numpy(dtype=float))
    ]
    if frame.empty:
        return float("nan"), float("nan")
    frame["diff"] = frame[metric_a] - frame[metric_b]
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        differences = frame["diff"].to_numpy(dtype=float)
        n = len(differences)
        indices = rng.integers(0, n, size=(int(n_bootstrap), n))
        boot = differences[indices].mean(axis=1)
    else:
        frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
        groups = [
            group["diff"].to_numpy(dtype=float)
            for _, group in frame.groupby("repeat", sort=True)
        ]
        boot = np.empty(int(n_bootstrap), dtype=float)
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
            boot[index] = float(np.mean(sampled))
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_test(
    differences: np.ndarray, random_state: int
) -> tuple[float, float, str]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), "paired_sign_flip"
    observed = float(np.mean(values))
    if n <= 16:
        total = 1 << n
        extreme = 0
        for bits in range(total):
            signs = np.fromiter(
                (1.0 if bits & (1 << index) else -1.0 for index in range(n)),
                dtype=float,
                count=n,
            )
            if abs(float(np.mean(signs * values))) >= abs(observed) - 1e-15:
                extreme += 1
        return observed, float(extreme / total), "exact_paired_sign_flip"
    rng = np.random.default_rng(int(random_state))
    n_permutations = 50000
    extreme = 0
    batch_size = 2000
    completed = 0
    while completed < n_permutations:
        current = min(batch_size, n_permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current, n))
        permuted = np.mean(signs * values[None, :], axis=1)
        extreme += int(np.sum(np.abs(permuted) >= abs(observed) - 1e-15))
        completed += current
    return (
        observed,
        float((extreme + 1) / (n_permutations + 1)),
        "monte_carlo_paired_sign_flip",
    )


def _corrected_resampled_t_test(
    merged: pd.DataFrame, metric_a: str, metric_b: str
) -> tuple[float, float, str]:
    frame = merged[["outer_split_key", metric_a, metric_b]].copy()
    frame[metric_a] = pd.to_numeric(frame[metric_a], errors="coerce")
    frame[metric_b] = pd.to_numeric(frame[metric_b], errors="coerce")
    frame = frame[
        np.isfinite(frame[metric_a].to_numpy(dtype=float))
        & np.isfinite(frame[metric_b].to_numpy(dtype=float))
    ]
    if len(frame) < 2:
        return float("nan"), float("nan"), "nadeau_bengio_corrected_t"
    differences = (frame[metric_a] - frame[metric_b]).to_numpy(dtype=float)
    mean_difference = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1))
    frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
    fold_counts = (
        frame.groupby("repeat")["outer_split_key"].nunique().to_numpy(dtype=float)
    )
    k = int(round(float(np.median(fold_counts)))) if len(fold_counts) else len(frame)
    if k < 2:
        return mean_difference, float("nan"), "nadeau_bengio_corrected_t"
    test_train_ratio = 1.0 / float(k - 1)
    corrected_variance = (1.0 / len(differences) + test_train_ratio) * variance
    if corrected_variance <= 0:
        p_value = 1.0 if abs(mean_difference) <= 1e-15 else 0.0
        return mean_difference, p_value, "nadeau_bengio_corrected_t"
    statistic = mean_difference / math.sqrt(corrected_variance)
    p_value = float(2.0 * student_t.sf(abs(statistic), df=len(differences) - 1))
    return float(statistic), p_value, "nadeau_bengio_corrected_t"


def _holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["p_holm"] = np.nan
    for metric, group in out.groupby("metric", sort=False):
        indices = group.index[
            np.isfinite(
                pd.to_numeric(group["p_value"], errors="coerce").to_numpy(dtype=float)
            )
        ]
        if len(indices) == 0:
            continue
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
        for metric in DISPLAY_METRICS:
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
    alpha = 0.0
    for _ in range(60):
        mu = _sigmoid(logit + alpha)
        weight = np.clip(mu * (1.0 - mu), 1e-10, None)
        step = float(np.sum(y - mu) / np.sum(weight))
        alpha += step
        if abs(step) < 1e-10:
            break
    X = np.column_stack([np.ones(len(y), dtype=float), logit])
    beta = np.asarray([alpha, 1.0], dtype=float)
    for _ in range(80):
        eta = X @ beta
        mu = _sigmoid(eta)
        weight = np.clip(mu * (1.0 - mu), 1e-10, None)
        gradient = X.T @ (y - mu)
        hessian = X.T @ (weight[:, None] * X)
        hessian += np.eye(2) * 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return float(alpha), float(beta[0]), float(beta[1])


def _proper_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = _renormalize_proba(np.asarray(proba, dtype=float), proba.shape[1])
    n_classes = p.shape[1]
    clipped = np.clip(p, 1e-15, 1.0)
    out: dict[str, float] = {
        "Brier": float("nan"),
        "Brier_multiclass": float("nan"),
        "LogLoss": float(-np.mean(np.log(clipped[np.arange(len(y)), y]))),
        "CalibrationInTheLarge": float("nan"),
        "CalibrationIntercept": float("nan"),
        "CalibrationSlope": float("nan"),
    }
    if n_classes == 2:
        target = (y == 1).astype(float)
        out["Brier"] = float(np.mean((p[:, 1] - target) ** 2))
        citl, intercept, slope = _calibration_binary(target, p[:, 1])
        out["CalibrationInTheLarge"] = citl
        out["CalibrationIntercept"] = intercept
        out["CalibrationSlope"] = slope
    else:
        one_hot = np.eye(n_classes, dtype=float)[y]
        out["Brier_multiclass"] = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
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


def _binary_auc_ap(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
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
    return float(auc), ap


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
        "AUC": float("nan"),
        "AUC_macro": float("nan"),
        "AUC_weighted": float("nan"),
        "PR_AUC": float("nan"),
        "PR_AUC_macro": float("nan"),
        "nMCC": float("nan"),
        "F1w": float("nan"),
        "F1_macro": float("nan"),
        "Precision": float("nan"),
        "Recall": float("nan"),
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
    out["nMCC"] = float((mcc + 1.0) / 2.0)
    if n_classes == 2:
        auc, ap = _binary_auc_ap((y == 1).astype(int), values[:, 1])
        out["AUC"] = auc
        out["PR_AUC"] = ap
    else:
        auc_values = []
        auc_weights = []
        ap_values = []
        for class_index in range(n_classes):
            binary = (y == class_index).astype(int)
            auc, ap = _binary_auc_ap(binary, values[:, class_index])
            if np.isfinite(auc):
                auc_values.append(auc)
                auc_weights.append(float(np.sum(binary)))
            if np.isfinite(ap):
                ap_values.append(ap)
        if auc_values:
            out["AUC_macro"] = float(np.mean(auc_values))
            out["AUC_weighted"] = float(np.average(auc_values, weights=auc_weights))
            out["AUC"] = out["AUC_macro"]
        if ap_values:
            out["PR_AUC_macro"] = float(np.mean(ap_values))
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


def _resample_rows(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if group.empty:
        return group.copy()
    indices = rng.integers(0, len(group), size=len(group))
    return group.iloc[indices].reset_index(drop=True)


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
                _resample_rows(groups[int(index)], rng) for index in chosen
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
    groups = [
        group.reset_index(drop=True) for _, group in frame.groupby("_repeat", sort=True)
    ]
    for bootstrap_index in range(int(n_bootstrap)):
        chosen = rng.integers(0, len(groups), size=len(groups))
        metrics = [
            _oof_metrics(_resample_rows(groups[int(index)], rng), probability_valid)
            for index in chosen
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


def _oof_design(frame: pd.DataFrame, protocol: str) -> dict[str, int]:
    protocol_key = str(protocol).lower()
    design = {
        "n_rows": int(len(frame)),
        "n_unique_samples": int(frame["sample_id"].nunique())
        if "sample_id" in frame.columns
        else 0,
        "n_outer_units": int(frame["outer_split_key"].nunique())
        if "outer_split_key" in frame.columns
        else 0,
        "n_repeats": int(frame["_repeat"].nunique())
        if "_repeat" in frame.columns
        else 0,
    }
    if protocol_key in _LODO_PROTOCOLS:
        design["n_cohorts"] = int(frame["_cluster"].nunique())
        design["n_unique_samples"] = int(
            frame[["_cluster", "sample_id"]].drop_duplicates().shape[0]
        )
    return design


def _coverage_row(strategy: str, frame: pd.DataFrame, protocol: str) -> dict[str, Any]:
    design = _oof_design(frame, protocol)
    return {"Strategy": strategy, **design}


def _match_key_columns(protocol: str) -> list[str]:
    return (
        ["_cluster", "sample_id"]
        if str(protocol).lower() in _LODO_PROTOCOLS
        else ["_repeat", "sample_id"]
    )


def _matched_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = _match_key_columns(protocol)
    left_keys = left[keys].astype(str).agg("\x1f".join, axis=1)
    right_keys = right[keys].astype(str).agg("\x1f".join, axis=1)
    common = set(left_keys).intersection(set(right_keys))
    left_match = left.loc[left_keys.isin(common)].copy()
    right_match = right.loc[right_keys.isin(common)].copy()
    left_match["_pair_key"] = left_match[keys].astype(str).agg("\x1f".join, axis=1)
    right_match["_pair_key"] = right_match[keys].astype(str).agg("\x1f".join, axis=1)
    left_match = left_match.sort_values("_pair_key", kind="mergesort").reset_index(
        drop=True
    )
    right_match = right_match.sort_values("_pair_key", kind="mergesort").reset_index(
        drop=True
    )
    if (
        len(left_match) != len(right_match)
        or left_match["_pair_key"].tolist() != right_match["_pair_key"].tolist()
    ):
        raise ValueError(
            "Could not align matched held-out predictions across strategies"
        )
    if not np.array_equal(
        left_match["y_true"].to_numpy(), right_match["y_true"].to_numpy()
    ):
        raise ValueError("Matched strategies disagree on held-out outcome labels")
    coverage = {
        "n_strategy_a": int(len(left)),
        "n_strategy_b": int(len(right)),
        "n_matched": int(len(left_match)),
        "coverage_fraction_a": float(len(left_match) / len(left))
        if len(left)
        else np.nan,
        "coverage_fraction_b": float(len(right_match) / len(right))
        if len(right)
        else np.nan,
    }
    return left_match, right_match, coverage


def _paired_resample_indices(
    frame: pd.DataFrame,
    protocol: str,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        groups = [
            group.index.to_numpy(dtype=int)
            for _, group in frame.groupby("_cluster", sort=True)
        ]
    else:
        groups = [
            group.index.to_numpy(dtype=int)
            for _, group in frame.groupby("_repeat", sort=True)
        ]
    chosen = rng.integers(0, len(groups), size=len(groups))
    return [
        indices[rng.integers(0, len(indices), size=len(indices))]
        for indices in (groups[int(index)] for index in chosen)
    ]


def _paired_point_estimands(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protocol: str,
    probability_valid_a: bool,
    probability_valid_b: bool,
) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        pooled = (
            _oof_metrics(left, probability_valid_a),
            _oof_metrics(right, probability_valid_b),
        )
        left_cohort = []
        right_cohort = []
        for cluster in left["_cluster"].drop_duplicates().tolist():
            left_cohort.append(
                _oof_metrics(left[left["_cluster"].eq(cluster)], probability_valid_a)
            )
            right_cohort.append(
                _oof_metrics(right[right["_cluster"].eq(cluster)], probability_valid_b)
            )
        return {
            "pooled_sample_weighted": pooled,
            "cohort_macro_equal_weight": (
                _mean_metric_dicts(left_cohort),
                _mean_metric_dicts(right_cohort),
            ),
        }
    left_repeat = []
    right_repeat = []
    for repeat in left["_repeat"].drop_duplicates().tolist():
        left_repeat.append(
            _oof_metrics(left[left["_repeat"].eq(repeat)], probability_valid_a)
        )
        right_repeat.append(
            _oof_metrics(right[right["_repeat"].eq(repeat)], probability_valid_b)
        )
    return {
        "mean_repeat_pooled_oof": (
            _mean_metric_dicts(left_repeat),
            _mean_metric_dicts(right_repeat),
        )
    }


def _paired_bootstrap_advantages(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protocol: str,
    probability_valid_a: bool,
    probability_valid_b: bool,
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
            for metric in _OOF_CONTRAST_METRICS
        }
        for name in names
    }
    for bootstrap_index in range(int(n_bootstrap)):
        sampled_indices = _paired_resample_indices(left, protocol, rng)
        if protocol_key in _LODO_PROTOCOLS:
            left_groups = [
                left.iloc[index].reset_index(drop=True) for index in sampled_indices
            ]
            right_groups = [
                right.iloc[index].reset_index(drop=True) for index in sampled_indices
            ]
            left_pooled = _oof_metrics(
                pd.concat(left_groups, ignore_index=True), probability_valid_a
            )
            right_pooled = _oof_metrics(
                pd.concat(right_groups, ignore_index=True), probability_valid_b
            )
            left_macro = _mean_metric_dicts(
                [_oof_metrics(group, probability_valid_a) for group in left_groups]
            )
            right_macro = _mean_metric_dicts(
                [_oof_metrics(group, probability_valid_b) for group in right_groups]
            )
            pairs = {
                "pooled_sample_weighted": (left_pooled, right_pooled),
                "cohort_macro_equal_weight": (left_macro, right_macro),
            }
        else:
            left_metrics = [
                _oof_metrics(
                    left.iloc[index].reset_index(drop=True), probability_valid_a
                )
                for index in sampled_indices
            ]
            right_metrics = [
                _oof_metrics(
                    right.iloc[index].reset_index(drop=True), probability_valid_b
                )
                for index in sampled_indices
            ]
            pairs = {
                "mean_repeat_pooled_oof": (
                    _mean_metric_dicts(left_metrics),
                    _mean_metric_dicts(right_metrics),
                )
            }
        for estimand, (metric_a, metric_b) in pairs.items():
            for metric in _OOF_CONTRAST_METRICS:
                a = float(metric_a.get(metric, np.nan))
                b = float(metric_b.get(metric, np.nan))
                if not np.isfinite(a) or not np.isfinite(b):
                    continue
                storage[estimand][metric][bootstrap_index] = (
                    b - a if metric_is_loss(metric) else a - b
                )
    return storage


def _paired_contrast_rows(
    frames: dict[str, pd.DataFrame],
    protocol: str,
    probability_semantics: dict[str, dict[str, Any]],
    n_bootstrap: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    prepared = {
        strategy: _prepare_oof_frame(frame, protocol)
        for strategy, frame in frames.items()
    }
    for strategy_a, strategy_b in itertools.combinations(prepared, 2):
        left, right, coverage = _matched_frames(
            prepared[strategy_a], prepared[strategy_b], protocol
        )
        coverage_rows.append(
            {"strategy_a": strategy_a, "strategy_b": strategy_b, **coverage}
        )
        if left.empty:
            continue
        valid_a = bool(probability_semantics[strategy_a].get("valid", False))
        valid_b = bool(probability_semantics[strategy_b].get("valid", False))
        point = _paired_point_estimands(left, right, protocol, valid_a, valid_b)
        boot = _paired_bootstrap_advantages(
            left,
            right,
            protocol,
            valid_a,
            valid_b,
            n_bootstrap,
            np.random.default_rng(
                _stable_seed(random_state, "paired_oof", strategy_a, strategy_b)
            ),
        )
        for estimand, (metric_a, metric_b) in point.items():
            for metric in _OOF_CONTRAST_METRICS:
                a = float(metric_a.get(metric, np.nan))
                b = float(metric_b.get(metric, np.nan))
                if not np.isfinite(a) or not np.isfinite(b):
                    continue
                difference = a - b
                advantage = b - a if metric_is_loss(metric) else difference
                samples = np.asarray(boot[estimand][metric], dtype=float)
                samples = samples[np.isfinite(samples)]
                low = high = np.nan
                if len(samples):
                    low, high = np.quantile(samples, [0.025, 0.975])
                rows.append(
                    {
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "estimand": estimand,
                        "metric": metric,
                        "estimate_a": a,
                        "estimate_b": b,
                        "difference_a_minus_b": difference,
                        "advantage_a_over_b": advantage,
                        "advantage_ci_low": float(low) if np.isfinite(low) else np.nan,
                        "advantage_ci_high": float(high)
                        if np.isfinite(high)
                        else np.nan,
                        "n_bootstrap_valid": int(len(samples)),
                        "n_matched": int(len(left)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def _run_oof_statistics(
    frames: dict[str, pd.DataFrame],
    protocol: str,
    n_bootstrap: int,
    random_state: int,
    calibration_bins: int,
) -> dict[str, Any]:
    prepared: dict[str, pd.DataFrame] = {}
    semantics: dict[str, dict[str, Any]] = {}
    design: dict[str, dict[str, int]] = {}
    performance_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    n_classes = 0
    for strategy, frame in frames.items():
        current = _prepare_oof_frame(frame, protocol)
        if current.empty:
            continue
        prepared[strategy] = current
        semantics[strategy] = _probability_semantics(current)
        design[strategy] = _oof_design(current, protocol)
        n_classes = max(n_classes, len(_probability_columns(current)))
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
        coverage_rows.append(_coverage_row(strategy, current, protocol))
    contrasts, pairwise_coverage = _paired_contrast_rows(
        prepared,
        protocol,
        semantics,
        n_bootstrap,
        random_state,
    )
    performance = pd.DataFrame(performance_rows)
    return {
        "performance": performance,
        "calibration": pd.DataFrame(calibration_rows),
        "contrasts": contrasts,
        "coverage": pd.DataFrame(coverage_rows),
        "pairwise_coverage": pairwise_coverage,
        "probability_semantics": semantics,
        "observed_oof_design": design,
        "n_classes": int(n_classes),
    }


def _stable_seed(random_state: int, *parts: Any) -> int:
    text = "|".join(str(value) for value in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little", signed=False)
    return int((int(random_state) + offset) % (2**32 - 1))


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
    return "nMCC"


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
) -> str:
    relevant_rows = []
    for row in strategy_rows:
        relevant_rows.append(
            {
                key: row.get(key)
                for key in (
                    "Strategy",
                    "source",
                    "config_id",
                    "ensemble_config_id",
                    "selection_strategy",
                    "aggregation_strategy",
                    "members",
                    "weights",
                )
                if key in row
            }
        )
    paths = [
        root / "predictions" / "outer_predictions.parquet",
        root / "configs.parquet",
        root / "inner_results" / "inner_results.parquet",
        root / "tables" / "mpma_b_final_candidate.json",
        root / "ensembling" / "selected_unit.json",
        root / "ensembling" / "mpma_e_final_candidate.json",
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
        "files": [_file_signature(path) for path in paths],
        "schema_version": 6,
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
    oof_contrasts_path = tables / "strategy_oof_pairwise_contrasts.parquet"
    oof_manifest_path = tables / "strategy_oof_statistics_manifest.json"
    oof_coverage_path = tables / "strategy_oof_coverage.parquet"
    oof_pairwise_coverage_path = tables / "strategy_oof_pairwise_coverage.parquet"
    required = (
        unit_path,
        summary_path,
        pairwise_path,
        oof_performance_path,
        oof_calibration_path,
        oof_contrasts_path,
        oof_manifest_path,
        oof_coverage_path,
        oof_pairwise_coverage_path,
    )
    if any(not table_exists(path) for path in required):
        return None
    return {
        "unit_metrics": _read_table(unit_path),
        "summary": _read_table(summary_path),
        "pairwise": _read_table(pairwise_path),
        "oof_performance": _read_table(oof_performance_path),
        "oof_calibration": _read_table(oof_calibration_path),
        "oof_contrasts": _read_table(oof_contrasts_path),
        "unit_metrics_path": unit_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "oof_performance_path": oof_performance_path,
        "oof_calibration_path": oof_calibration_path,
        "oof_contrasts_path": oof_contrasts_path,
        "oof_manifest_path": oof_manifest_path,
        "oof_coverage_path": oof_coverage_path,
        "oof_pairwise_coverage_path": oof_pairwise_coverage_path,
        "manifest_path": manifest_path,
        "selection_metric": str(manifest.get("selection_metric", "nMCC")),
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
) -> dict[str, Any]:
    root = Path(root)
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
    )
    cached = _cached_result(root, fingerprint)
    if cached is not None:
        return cached
    frames = _strategy_prediction_frames(root, strategy_rows)
    unit_frames: list[pd.DataFrame] = []
    for strategy, predictions in frames.items():
        metrics = _metric_frame(predictions)
        if metrics.empty:
            continue
        metrics.insert(0, "Strategy", strategy)
        unit_frames.append(metrics)
    unit_metrics = (
        pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    )
    summary_metrics = tuple(
        metric
        for metric in DISPLAY_METRICS
        if unit_metrics.empty or metric in unit_metrics.columns
    )
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
    )
    oof_performance_path = tables / "strategy_oof_performance.parquet"
    oof_calibration_path = tables / "strategy_oof_calibration.parquet"
    oof_contrasts_path = tables / "strategy_oof_pairwise_contrasts.parquet"
    oof_manifest_path = tables / "strategy_oof_statistics_manifest.json"
    oof_coverage_path = tables / "strategy_oof_coverage.parquet"
    oof_pairwise_coverage_path = tables / "strategy_oof_pairwise_coverage.parquet"
    write_table(oof_performance_path, advanced["performance"])
    write_table(oof_calibration_path, advanced["calibration"])
    write_table(oof_contrasts_path, advanced["contrasts"])
    write_table(oof_coverage_path, advanced["coverage"])
    write_table(oof_pairwise_coverage_path, advanced["pairwise_coverage"])
    oof_manifest = {
        "schema_version": 4,
        "protocol": str(protocol),
        "strategies": list(frames),
        "n_classes": int(advanced["n_classes"]),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": 0.95,
        "calibration_bins": int(calibration_bins),
        "metric_order": list(OOF_METRIC_ORDER),
        "estimands": (
            ["pooled_sample_weighted", "cohort_macro_equal_weight"]
            if str(protocol).lower() in _LODO_PROTOCOLS
            else ["mean_repeat_pooled_oof"]
        ),
        "observed_oof_design": advanced["observed_oof_design"],
        "probability_semantics": advanced["probability_semantics"],
        "bootstrap": (
            "two-stage cohort-and-subject bootstrap"
            if str(protocol).lower() in _LODO_PROTOCOLS
            else "subject-cluster and repeat bootstrap"
        ),
        "paired_contrasts": "matched held-out observations with shared bootstrap draws",
        "metric_direction": {
            metric: ("lower" if metric_is_loss(metric) else "higher")
            for metric in _OOF_CONTRAST_METRICS
        },
        "scope": "displayed report strategies only",
    }
    dump_json_standard(oof_manifest, oof_manifest_path)
    manifest = {
        "schema_version": 4,
        "fingerprint": fingerprint,
        "protocol": str(protocol),
        "strategies": list(frames),
        "display_metrics_bootstrapped": list(summary_metrics),
        "pairwise_metrics": list(DISPLAY_METRICS),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": 0.95,
        "nested_cv_bootstrap": "hierarchical repeat/outer-fold bootstrap of held-out displayed-strategy metrics",
        "lodo_bootstrap": "cluster bootstrap of held-out cohorts for displayed-strategy metrics",
        "nested_cv_pairwise_test": "Nadeau-Bengio corrected resampled paired t-test",
        "lodo_pairwise_test": "paired sign-flip randomization test at held-out cohort level",
        "multiple_testing": "Holm adjustment across displayed strategy pairs within each displayed metric",
        "scope": "displayed report strategies only",
        "selection_metric": resolved_selection,
        "cache": "statistics are reused when source artefacts and statistical settings are unchanged",
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
        "oof_contrasts": advanced["contrasts"],
        "oof_performance_path": oof_performance_path,
        "oof_calibration_path": oof_calibration_path,
        "oof_contrasts_path": oof_contrasts_path,
        "oof_manifest_path": oof_manifest_path,
        "oof_coverage_path": oof_coverage_path,
        "oof_pairwise_coverage_path": oof_pairwise_coverage_path,
        "manifest_path": manifest_path,
        "selection_metric": resolved_selection,
        "cache_hit": False,
    }
