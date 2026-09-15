from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .metrics import compute_metrics
from .utils import METRIC_COLUMNS, dump_json_standard

DISPLAY_METRICS = ("AUC", "PR_AUC", "nMCC", "F1w", "Precision", "Recall")
METRIC_LABELS = {
    "AUC": "ROC-AUC",
    "PR_AUC": "PR-AUC (AP)",
    "AUC_macro": "ROC-AUC macro",
    "AUC_weighted": "ROC-AUC weighted",
    "PR_AUC_macro": "PR-AUC macro",
    "nMCC": "nMCC",
    "F1w": "F1w",
    "F1_macro": "F1 macro",
    "Precision": "Precision",
    "Recall": "Recall",
    "BalAcc": "Balanced accuracy",
    "Accuracy": "Accuracy",
}


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


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
    rows = {str(r.get("Strategy", "")): r for r in strategy_rows}
    frames: dict[str, pd.DataFrame] = {}
    mpma_b = _ensure_outer_split_key(
        _read_tsv(root / "predictions" / "mpma_b_outer_predictions.tsv")
    )
    if not mpma_b.empty:
        frames["MPMA-B"] = mpma_b
    mpma_e = _ensure_outer_split_key(
        _read_tsv(root / "ensembling" / "ensemble_predictions.tsv")
    )
    if not mpma_e.empty:
        frames["MPMA-E"] = mpma_e
    outer = _ensure_outer_split_key(
        _read_tsv(root / "predictions" / "outer_predictions.tsv")
    )
    if not outer.empty and "config_id" in outer.columns:
        outer["config_id"] = outer["config_id"].astype(str)
        for strategy in ("AutoML", "Baseline RF", "SIAMCAT"):
            row = rows.get(strategy, {})
            config_id = str(row.get("config_id", "")).strip()
            if not config_id:
                continue
            sub = outer[outer["config_id"].eq(config_id)].copy()
            if not sub.empty:
                frames[strategy] = sub
    return frames


def _metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "outer_split_key" not in predictions.columns:
        return pd.DataFrame()
    pcols = [c for c in predictions.columns if c.startswith("proba_")]
    if not pcols or "y_true" not in predictions.columns:
        return pd.DataFrame()
    rows = []
    classes = np.arange(len(pcols), dtype=int)
    for outer_split_key, group in predictions.groupby("outer_split_key", sort=True):
        y_true = pd.to_numeric(group["y_true"], errors="coerce")
        valid = y_true.notna()
        if not valid.any():
            continue
        y_true_arr = y_true.loc[valid].astype(int).to_numpy()
        proba = (
            group.loc[valid, pcols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        if "y_pred" in group.columns:
            y_pred = pd.to_numeric(group.loc[valid, "y_pred"], errors="coerce")
            if y_pred.isna().any():
                y_pred_arr = classes[np.nanargmax(proba, axis=1)]
            else:
                y_pred_arr = y_pred.astype(int).to_numpy()
        else:
            y_pred_arr = classes[np.nanargmax(proba, axis=1)]
        metrics = compute_metrics(y_true_arr, y_pred_arr, proba, classes)
        row = {
            "outer_split_key": str(outer_split_key),
            "n_samples": int(len(y_true_arr)),
            **metrics,
        }
        rows.append(row)
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
    protocol = str(protocol).lower()
    out = np.empty(int(n_bootstrap), dtype=float)
    if protocol in {"lodo", "leave_one_dataset_out"}:
        arr = frame[metric].to_numpy(dtype=float)
        n = len(arr)
        for i in range(int(n_bootstrap)):
            out[i] = float(np.mean(arr[rng.integers(0, n, size=n)]))
        return out
    frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
    repeat_groups = {
        key: group[metric].to_numpy(dtype=float)
        for key, group in frame.groupby("repeat", sort=True)
    }
    repeat_keys = list(repeat_groups)
    n_repeats = len(repeat_keys)
    for i in range(int(n_bootstrap)):
        sampled_repeats = rng.integers(0, n_repeats, size=n_repeats)
        sampled_values = []
        for repeat_index in sampled_repeats:
            arr = repeat_groups[repeat_keys[int(repeat_index)]]
            sampled_values.extend(
                arr[rng.integers(0, len(arr), size=len(arr))].tolist()
            )
        out[i] = float(np.mean(sampled_values))
    return out


def _summary_rows(
    unit_metrics: pd.DataFrame,
    protocol: str,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    for strategy, group in unit_metrics.groupby("Strategy", sort=False):
        for metric in METRIC_COLUMNS:
            if metric not in group.columns:
                continue
            vals = (
                pd.to_numeric(group[metric], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            seed = int(random_state) + sum(ord(c) for c in f"{strategy}:{metric}")
            boot = (
                _bootstrap_unit_mean(
                    group, metric, protocol, n_bootstrap, np.random.default_rng(seed)
                )
                if not vals.empty
                else np.asarray([], dtype=float)
            )
            if len(boot):
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            else:
                ci_low = ci_high = np.nan
            rows.append(
                {
                    "Strategy": strategy,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "estimate": float(vals.mean()) if not vals.empty else np.nan,
                    "std": float(vals.std(ddof=1))
                    if len(vals) > 1
                    else (0.0 if len(vals) == 1 else np.nan),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_outer_units": int(len(vals)),
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_method": "cluster_outer_cohort"
                    if str(protocol).lower() in {"lodo", "leave_one_dataset_out"}
                    else "hierarchical_repeat_outer_fold",
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
    protocol = str(protocol).lower()
    out = np.empty(int(n_bootstrap), dtype=float)
    if protocol in {"lodo", "leave_one_dataset_out"}:
        arr = frame["diff"].to_numpy(dtype=float)
        n = len(arr)
        for i in range(int(n_bootstrap)):
            out[i] = float(np.mean(arr[rng.integers(0, n, size=n)]))
    else:
        frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
        repeat_groups = {
            key: group["diff"].to_numpy(dtype=float)
            for key, group in frame.groupby("repeat", sort=True)
        }
        repeat_keys = list(repeat_groups)
        n_repeats = len(repeat_keys)
        for i in range(int(n_bootstrap)):
            sampled_repeats = rng.integers(0, n_repeats, size=n_repeats)
            sampled_values = []
            for repeat_index in sampled_repeats:
                arr = repeat_groups[repeat_keys[int(repeat_index)]]
                sampled_values.extend(
                    arr[rng.integers(0, len(arr), size=len(arr))].tolist()
                )
            out[i] = float(np.mean(sampled_values))
    low, high = np.quantile(out, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_test(
    differences: np.ndarray, random_state: int
) -> tuple[float, float, str]:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), "paired_sign_flip"
    observed = float(np.mean(d))
    if n <= 16:
        total = 1 << n
        extreme = 0
        for bits in range(total):
            signs = np.fromiter(
                (1.0 if bits & (1 << j) else -1.0 for j in range(n)),
                dtype=float,
                count=n,
            )
            if abs(float(np.mean(signs * d))) >= abs(observed) - 1e-15:
                extreme += 1
        return observed, float(extreme / total), "exact_paired_sign_flip"
    rng = np.random.default_rng(int(random_state))
    n_perm = 50000
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        if abs(float(np.mean(signs * d))) >= abs(observed) - 1e-15:
            extreme += 1
    return observed, float((extreme + 1) / (n_perm + 1)), "monte_carlo_paired_sign_flip"


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
    mean_diff = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1))
    frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
    fold_counts = (
        frame.groupby("repeat")["outer_split_key"].nunique().to_numpy(dtype=float)
    )
    k = int(round(float(np.median(fold_counts)))) if len(fold_counts) else len(frame)
    if k < 2:
        return mean_diff, float("nan"), "nadeau_bengio_corrected_t"
    test_train_ratio = 1.0 / float(k - 1)
    corrected_variance = (1.0 / len(differences) + test_train_ratio) * variance
    if corrected_variance <= 0:
        p = 1.0 if abs(mean_diff) <= 1e-15 else 0.0
        return mean_diff, p, "nadeau_bengio_corrected_t"
    statistic = mean_diff / math.sqrt(corrected_variance)
    p = float(2.0 * student_t.sf(abs(statistic), df=len(differences) - 1))
    return float(statistic), p, "nadeau_bengio_corrected_t"


def _holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["p_holm"] = np.nan
    for metric, group in out.groupby("metric", sort=False):
        idx = group.index[
            np.isfinite(
                pd.to_numeric(group["p_value"], errors="coerce").to_numpy(dtype=float)
            )
        ]
        if len(idx) == 0:
            continue
        ordered = sorted(idx, key=lambda i: float(out.loc[i, "p_value"]))
        m = len(ordered)
        running = 0.0
        for rank, i in enumerate(ordered):
            adjusted = min(1.0, (m - rank) * float(out.loc[i, "p_value"]))
            running = max(running, adjusted)
            out.loc[i, "p_holm"] = running
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
    rows = []
    strategies = list(dict.fromkeys(unit_metrics["Strategy"].astype(str).tolist()))
    for strategy_a, strategy_b in itertools.combinations(strategies, 2):
        left = unit_metrics[unit_metrics["Strategy"].eq(strategy_a)].copy()
        right = unit_metrics[unit_metrics["Strategy"].eq(strategy_b)].copy()
        merged = left.merge(
            right, on="outer_split_key", suffixes=("_a", "_b"), how="inner"
        )
        if merged.empty:
            continue
        for metric in DISPLAY_METRICS:
            a = f"{metric}_a"
            b = f"{metric}_b"
            if a not in merged.columns or b not in merged.columns:
                continue
            valid = merged[
                np.isfinite(
                    pd.to_numeric(merged[a], errors="coerce").to_numpy(dtype=float)
                )
                & np.isfinite(
                    pd.to_numeric(merged[b], errors="coerce").to_numpy(dtype=float)
                )
            ].copy()
            if valid.empty:
                continue
            estimate_a = float(pd.to_numeric(valid[a], errors="coerce").mean())
            estimate_b = float(pd.to_numeric(valid[b], errors="coerce").mean())
            difference = estimate_a - estimate_b
            seed = int(random_state) + sum(
                ord(c) for c in f"{strategy_a}:{strategy_b}:{metric}"
            )
            ci_low, ci_high = _paired_bootstrap_difference(
                valid, a, b, protocol, n_bootstrap, np.random.default_rng(seed)
            )
            if str(protocol).lower() in {"lodo", "leave_one_dataset_out"}:
                statistic, p_value, test = _exact_sign_flip_test(
                    (
                        pd.to_numeric(valid[a], errors="coerce")
                        - pd.to_numeric(valid[b], errors="coerce")
                    ).to_numpy(dtype=float),
                    seed,
                )
            else:
                statistic, p_value, test = _corrected_resampled_t_test(valid, a, b)
            rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "estimate_a": estimate_a,
                    "estimate_b": estimate_b,
                    "difference_a_minus_b": difference,
                    "difference_ci_low": ci_low,
                    "difference_ci_high": ci_high,
                    "test": test,
                    "test_statistic": statistic,
                    "p_value": p_value,
                    "n_paired_outer_units": int(len(valid)),
                }
            )
    return _holm_adjust(pd.DataFrame(rows)) if rows else pd.DataFrame()


def run_report_statistics(
    root: Path | str,
    protocol: str,
    strategy_rows: list[dict[str, Any]],
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    root = Path(root)
    frames = _strategy_prediction_frames(root, strategy_rows)
    unit_frames = []
    for strategy, predictions in frames.items():
        metrics = _metric_frame(predictions)
        if metrics.empty:
            continue
        metrics.insert(0, "Strategy", strategy)
        unit_frames.append(metrics)
    unit_metrics = (
        pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    )
    summary = (
        _summary_rows(unit_metrics, protocol, n_bootstrap, random_state)
        if not unit_metrics.empty
        else pd.DataFrame()
    )
    pairwise = (
        _pairwise_rows(unit_metrics, protocol, n_bootstrap, random_state)
        if not unit_metrics.empty
        else pd.DataFrame()
    )
    tables = root / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    unit_path = tables / "strategy_outer_unit_metrics.tsv"
    summary_path = tables / "strategy_metrics_bootstrap.tsv"
    pairwise_path = tables / "strategy_pairwise_tests.tsv"
    unit_metrics.to_csv(unit_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    pairwise.to_csv(pairwise_path, sep="\t", index=False)
    manifest = {
        "protocol": str(protocol),
        "strategies": list(frames),
        "metrics_saved": list(METRIC_COLUMNS),
        "metrics_tested": list(DISPLAY_METRICS),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": 0.95,
        "nested_cv_bootstrap": "hierarchical repeat/outer-fold bootstrap of held-out strategy metrics",
        "lodo_bootstrap": "cluster bootstrap of held-out cohorts with equal cohort weighting",
        "nested_cv_pairwise_test": "Nadeau-Bengio corrected resampled paired t-test",
        "lodo_pairwise_test": "paired sign-flip randomization test at held-out cohort level",
        "multiple_testing": "Holm adjustment across strategy pairs within each metric",
        "scope": "report-stage held-out predictions for displayed strategies only; no model refitting or candidate re-evaluation",
    }
    manifest_path = tables / "strategy_statistics_manifest.json"
    dump_json_standard(manifest, manifest_path)
    return {
        "unit_metrics": unit_metrics,
        "summary": summary,
        "pairwise": pairwise,
        "unit_metrics_path": unit_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "manifest_path": manifest_path,
    }
