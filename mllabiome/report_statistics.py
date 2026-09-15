from __future__ import annotations
import hashlib
import itertools
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import t as student_t
from .metrics import _renormalize_proba, compute_metrics
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
    "Brier": "Brier score",
    "Brier_multiclass": "Multiclass Brier score",
    "LogLoss": "Log loss",
    "CalibrationInTheLarge": "Calibration-in-the-large",
    "CalibrationIntercept": "Calibration intercept",
    "CalibrationSlope": "Calibration slope",
}
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
_LOWER_IS_BETTER = {"Brier", "Brier_multiclass", "LogLoss"}
_TARGET_ZERO = {"CalibrationInTheLarge", "CalibrationIntercept"}
_TARGET_ONE = {"CalibrationSlope"}
_LODO_PROTOCOLS = {"lodo", "leave_one_dataset_out"}
_NESTED_PROTOCOLS = {"repeated_nested_cv", "nested_cv"}


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
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
        str(r.get("Strategy", "")): r
        for r in strategy_rows
        if str(r.get("Strategy", "")).strip()
    }
    frames: dict[str, pd.DataFrame] = {}
    if "MPMA-B" in rows:
        mpma_b = _ensure_outer_split_key(
            _read_tsv(root / "predictions" / "mpma_b_outer_predictions.tsv")
        )
        if not mpma_b.empty:
            frames["MPMA-B"] = mpma_b
    if "MPMA-E" in rows:
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
            if strategy not in rows:
                continue
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
        proba = _renormalize_proba(proba, len(classes))
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
    match = re.match("^r(\\d+)_o\\d+$", str(split_key))
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
    if protocol in _LODO_PROTOCOLS:
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
    unit_metrics: pd.DataFrame, protocol: str, n_bootstrap: int, random_state: int
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
            seed = int(random_state) + sum((ord(c) for c in f"{strategy}:{metric}"))
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
                    else 0.0
                    if len(vals) == 1
                    else np.nan,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_outer_units": int(len(vals)),
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_method": "cluster_outer_cohort"
                    if str(protocol).lower() in _LODO_PROTOCOLS
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
        return (float("nan"), float("nan"))
    frame["diff"] = frame[metric_a] - frame[metric_b]
    protocol = str(protocol).lower()
    out = np.empty(int(n_bootstrap), dtype=float)
    if protocol in _LODO_PROTOCOLS:
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
    return (float(low), float(high))


def _exact_sign_flip_test(
    differences: np.ndarray, random_state: int
) -> tuple[float, float, str]:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"), "paired_sign_flip")
    observed = float(np.mean(d))
    if n <= 16:
        total = 1 << n
        extreme = 0
        for bits in range(total):
            signs = np.fromiter(
                (1.0 if bits & 1 << j else -1.0 for j in range(n)), dtype=float, count=n
            )
            if abs(float(np.mean(signs * d))) >= abs(observed) - 1e-15:
                extreme += 1
        return (observed, float(extreme / total), "exact_paired_sign_flip")
    rng = np.random.default_rng(int(random_state))
    n_perm = 50000
    extreme = 0
    for _ in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        if abs(float(np.mean(signs * d))) >= abs(observed) - 1e-15:
            extreme += 1
    return (
        observed,
        float((extreme + 1) / (n_perm + 1)),
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
        return (float("nan"), float("nan"), "nadeau_bengio_corrected_t")
    differences = (frame[metric_a] - frame[metric_b]).to_numpy(dtype=float)
    mean_diff = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1))
    frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
    fold_counts = (
        frame.groupby("repeat")["outer_split_key"].nunique().to_numpy(dtype=float)
    )
    k = int(round(float(np.median(fold_counts)))) if len(fold_counts) else len(frame)
    if k < 2:
        return (mean_diff, float("nan"), "nadeau_bengio_corrected_t")
    test_train_ratio = 1.0 / float(k - 1)
    corrected_variance = (1.0 / len(differences) + test_train_ratio) * variance
    if corrected_variance <= 0:
        p = 1.0 if abs(mean_diff) <= 1e-15 else 0.0
        return (mean_diff, p, "nadeau_bengio_corrected_t")
    statistic = mean_diff / math.sqrt(corrected_variance)
    p = float(2.0 * student_t.sf(abs(statistic), df=len(differences) - 1))
    return (float(statistic), p, "nadeau_bengio_corrected_t")


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
    unit_metrics: pd.DataFrame, protocol: str, n_bootstrap: int, random_state: int
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
                (ord(c) for c in f"{strategy_a}:{strategy_b}:{metric}")
            )
            ci_low, ci_high = _paired_bootstrap_difference(
                valid, a, b, protocol, n_bootstrap, np.random.default_rng(seed)
            )
            if str(protocol).lower() in _LODO_PROTOCOLS:
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


def _stable_seed(random_state: int, *parts: Any) -> int:
    text = "|".join((str(x) for x in parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little", signed=False)
    return int((int(random_state) + offset) % (2**32 - 1))


def _sample_key_column(frame: pd.DataFrame) -> str | None:
    if "sample_id" in frame.columns:
        return "sample_id"
    if "sample_index" in frame.columns:
        return "sample_index"
    return None


def _prediction_probability_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("proba_")]


_PROPER_SCORING_METRICS = {"Brier", "Brier_multiclass", "LogLoss"}
_PROBABILITY_ONLY_METRICS = _PROPER_SCORING_METRICS | {
    "CalibrationInTheLarge",
    "CalibrationIntercept",
    "CalibrationSlope",
}
_PROBABILITY_PRESERVING_ENSEMBLES = {
    "mean_proba",
    "weighted_mean_proba",
    "median_proba",
}
_PROBABILITY_LEARNER_PREFIXES = (
    "lr",
    "logistic",
    "ridge",
    "rf",
    "et",
    "extratrees",
    "histgb",
    "histgradientboosting",
    "svc_rbf",
    "calib_lsvc",
    "caliblsvc",
    "knn",
    "nearestcentroid",
    "gnb",
    "gaussiannb",
    "bnb",
    "bernoullinb",
    "mnb",
    "multinomialnb",
    "lda",
    "qda",
    "sgd",
    "pa",
    "dt",
    "decisiontree",
    "flaml",
    "automl",
    "siamcat",
    "xgb",
    "lgb",
    "cat",
    "cb",
)


def _normalise_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int).ne(0)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "probability", "valid"})


def _learner_probability_semantics(learner: Any) -> tuple[bool, str]:
    key = str(learner or "").strip().lower()
    if not key:
        return (False, "learner probability semantics are unavailable")
    if key.startswith(_PROBABILITY_LEARNER_PREFIXES):
        return (
            True,
            "learner produces probabilities directly or through fold-local calibration",
        )
    return (False, f"probability semantics are unverified for learner {learner!r}")


def _config_learner_map(root: Path) -> dict[str, str]:
    configs = _read_tsv(root / "configs.tsv")
    if configs.empty or not {"config_id", "learner"}.issubset(configs.columns):
        return {}
    configs = configs.drop_duplicates("config_id", keep="last")
    return {str(row["config_id"]): str(row["learner"]) for _, row in configs.iterrows()}


def _parse_member_ids(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(x) for x in value]
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _strategy_probability_semantics(
    strategy: str,
    predictions: pd.DataFrame,
    strategy_row: dict[str, Any],
    config_learners: dict[str, str],
) -> dict[str, Any]:
    frame = predictions.copy()
    if str(strategy) == "MPMA-E":
        if "aggregation_strategy" not in frame.columns:
            return {
                "valid": False,
                "status": "unverified_probability",
                "reason": "ensemble aggregation strategy is unavailable in held-out predictions",
            }
        aggregations = sorted(
            {
                str(x).strip().lower()
                for x in frame["aggregation_strategy"].dropna().tolist()
                if str(x).strip()
            }
        )
        invalid_aggs = [
            a for a in aggregations if a not in _PROBABILITY_PRESERVING_ENSEMBLES
        ]
        if invalid_aggs:
            return {
                "valid": False,
                "status": "nonprobability_score",
                "reason": "ensemble aggregation is not probability-preserving: "
                + ", ".join(invalid_aggs),
                "aggregation_strategies": aggregations,
            }
        if "probability_valid" in frame.columns:
            flags = _normalise_bool_series(frame["probability_valid"])
            if len(flags) and bool(flags.all()):
                return {
                    "valid": True,
                    "status": "verified_probability",
                    "reason": "prediction file explicitly marks all rows as probability-valid",
                    "aggregation_strategies": aggregations,
                }
            return {
                "valid": False,
                "status": "nonprobability_score",
                "reason": "prediction file explicitly marks one or more rows as not probability-valid",
                "aggregation_strategies": aggregations,
            }
        if "members" not in frame.columns:
            return {
                "valid": False,
                "status": "unverified_probability",
                "reason": "ensemble member identities are unavailable, so member probability semantics cannot be verified",
                "aggregation_strategies": aggregations,
            }
        member_ids: set[str] = set()
        for value in frame["members"].dropna():
            member_ids.update(_parse_member_ids(value))
        if not member_ids:
            return {
                "valid": False,
                "status": "unverified_probability",
                "reason": "ensemble member identities could not be parsed",
                "aggregation_strategies": aggregations,
            }
        invalid_members: list[str] = []
        unknown_members: list[str] = []
        for cid in sorted(member_ids):
            learner = config_learners.get(str(cid), "")
            valid, reason = _learner_probability_semantics(learner)
            if not learner:
                unknown_members.append(str(cid))
            elif not valid:
                invalid_members.append(f"{cid}:{learner} ({reason})")
        if unknown_members or invalid_members:
            details = []
            if invalid_members:
                details.append(
                    "non-probabilistic member(s): " + "; ".join(invalid_members)
                )
            if unknown_members:
                details.append(
                    "member learner metadata missing for: " + ", ".join(unknown_members)
                )
            return {
                "valid": False,
                "status": "nonprobability_score",
                "reason": "; ".join(details),
                "aggregation_strategies": aggregations,
            }
        return {
            "valid": True,
            "status": "verified_probability",
            "reason": "probability-preserving ensemble of members with probability forecasts",
            "aggregation_strategies": aggregations,
        }
    if "probability_valid" in frame.columns:
        flags = _normalise_bool_series(frame["probability_valid"])
        if len(flags) and bool(flags.all()):
            return {
                "valid": True,
                "status": "verified_probability",
                "reason": "prediction file explicitly marks all rows as probability-valid",
            }
        return {
            "valid": False,
            "status": "nonprobability_score",
            "reason": "prediction file explicitly marks one or more rows as not probability-valid",
        }
    config_ids = (
        sorted(set(frame["config_id"].dropna().astype(str)))
        if "config_id" in frame.columns
        else []
    )
    learners: list[str] = []
    if config_ids:
        learners = [config_learners.get(cid, "") for cid in config_ids]
    if not learners or any((not x for x in learners)):
        fallback = str(strategy_row.get("learner", "")).strip()
        if fallback and len(config_ids) <= 1:
            learners = [fallback]
    if not learners or any((not str(x).strip() for x in learners)):
        return {
            "valid": False,
            "status": "unverified_probability",
            "reason": "learner probability semantics cannot be verified from the run metadata",
        }
    invalid = []
    for learner in sorted(set(map(str, learners))):
        valid, reason = _learner_probability_semantics(learner)
        if not valid:
            invalid.append(f"{learner} ({reason})")
    if invalid:
        return {
            "valid": False,
            "status": "nonprobability_score",
            "reason": "; ".join(invalid),
        }
    return {
        "valid": True,
        "status": "verified_probability",
        "reason": "all selected learners expose probability forecasts",
    }


def _expected_oof_design(root: Path, protocol: str) -> dict[str, int | None]:
    manifest = _read_json(root / "manifest.json")
    try:
        n_samples = int(manifest.get("n_samples"))
    except (TypeError, ValueError):
        n_samples = None
    sweep = manifest.get("sweep", {}) if isinstance(manifest.get("sweep"), dict) else {}
    evaluation = (
        sweep.get("evaluation", {}) if isinstance(sweep.get("evaluation"), dict) else {}
    )
    protocol_lower = str(protocol).lower()
    expected_repeats: int | None
    if protocol_lower == "repeated_nested_cv":
        try:
            expected_repeats = int(evaluation.get("repeats"))
        except (TypeError, ValueError):
            expected_repeats = None
    elif protocol_lower == "nested_cv":
        expected_repeats = 1
    else:
        expected_repeats = None
    return {"n_samples": n_samples, "n_repeats": expected_repeats}


def _prepare_oof_predictions(
    predictions: pd.DataFrame,
    protocol: str,
    *,
    strategy: str = "strategy",
    expected_n_samples: int | None = None,
    expected_repeats: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _ensure_outer_split_key(predictions)
    pcols = _prediction_probability_columns(frame)
    sample_key = _sample_key_column(frame)
    required = {"outer_split_key", "y_true"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{strategy}: OOF predictions are missing required columns: {missing}"
        )
    if not pcols:
        raise ValueError(f"{strategy}: OOF predictions contain no proba_* columns")
    if sample_key is None:
        raise ValueError(
            f"{strategy}: OOF predictions require sample_id or sample_index"
        )
    out = frame.copy()
    out["_sample_key"] = out[sample_key].astype(str)
    out["outer_split_key"] = out["outer_split_key"].astype(str)
    out["_repeat"] = out["outer_split_key"].map(_repeat_id)
    out["_cohort"] = out["outer_split_key"].astype(str)
    y = pd.to_numeric(out["y_true"], errors="coerce")
    pp = out[pcols].apply(pd.to_numeric, errors="coerce")
    valid = y.notna() & np.isfinite(pp.to_numpy(dtype=float)).all(axis=1)
    if not bool(valid.all()):
        bad = int((~valid).sum())
        raise ValueError(
            f"{strategy}: OOF predictions contain {bad} row(s) with missing/non-finite outcome or prediction values"
        )
    out["y_true"] = y.astype(int)
    proba = _renormalize_proba(pp.to_numpy(dtype=float), len(pcols))
    out.loc[:, pcols] = proba
    classes = np.arange(len(pcols), dtype=int)
    out["y_pred"] = classes[np.argmax(proba, axis=1)]
    protocol_lower = str(protocol).lower()
    duplicate_keys = (
        ["_repeat", "_sample_key"]
        if protocol_lower in _NESTED_PROTOCOLS
        else ["_cohort", "_sample_key"]
    )
    duplicate_mask = out.duplicated(duplicate_keys, keep=False)
    if bool(duplicate_mask.any()):
        raise ValueError(
            f"{strategy}: OOF predictions contain duplicate held-out observations for {duplicate_keys}"
        )
    n_unique_samples = int(out["_sample_key"].nunique())
    if expected_n_samples is not None and n_unique_samples != int(expected_n_samples):
        raise ValueError(
            f"{strategy}: incomplete OOF coverage: expected {int(expected_n_samples)} subjects, found {n_unique_samples}"
        )
    coverage: dict[str, Any] = {
        "status": "ok",
        "sample_key_column": sample_key,
        "probability_columns": pcols,
        "n_classes": int(len(pcols)),
        "n_input_rows": int(len(out)),
        "n_valid_rows": int(len(out)),
        "n_dropped_nonfinite_rows": 0,
        "n_duplicate_rows_removed": 0,
        "n_unique_samples": n_unique_samples,
        "n_outer_units": int(out["outer_split_key"].nunique()),
        "n_repeats": int(out["_repeat"].nunique()),
        "n_cohorts": int(out["_cohort"].nunique())
        if protocol_lower in _LODO_PROTOCOLS
        else 0,
    }
    if protocol_lower in _NESTED_PROTOCOLS:
        repeats = sorted(out["_repeat"].astype(str).unique())
        repeat_sets = {
            rep: set(out.loc[out["_repeat"].eq(rep), "_sample_key"].astype(str))
            for rep in repeats
        }
        union = set().union(*repeat_sets.values()) if repeat_sets else set()
        intersection = set.intersection(*repeat_sets.values()) if repeat_sets else set()
        if union != intersection:
            raise ValueError(
                f"{strategy}: incomplete repeated-CV OOF coverage: not every subject has one held-out prediction in every repeat"
            )
        if expected_repeats is not None and len(repeats) != int(expected_repeats):
            raise ValueError(
                f"{strategy}: incomplete repeated-CV OOF coverage: expected {int(expected_repeats)} repeat(s), found {len(repeats)}"
            )
        expected_repeat_count = (
            int(expected_n_samples) if expected_n_samples is not None else len(union)
        )
        for rep, subjects in repeat_sets.items():
            if len(subjects) != expected_repeat_count:
                raise ValueError(
                    f"{strategy}: repeat {rep} contains {len(subjects)} held-out subjects; expected {expected_repeat_count}"
                )
        expected_rows = len(union) * len(repeats)
        if len(out) != expected_rows:
            raise ValueError(
                f"{strategy}: repeated-CV OOF row count mismatch: expected {expected_rows}, found {len(out)}"
            )
        coverage.update(
            {
                "n_samples_any_repeat": int(len(union)),
                "n_samples_complete_repeats": int(len(intersection)),
                "complete_repeat_coverage_fraction": 1.0,
            }
        )
    elif protocol_lower in _LODO_PROTOCOLS:
        if bool(out["_sample_key"].duplicated(keep=False).any()):
            raise ValueError(
                f"{strategy}: LODO OOF predictions contain a subject in more than one held-out cohort"
            )
        coverage["complete_repeat_coverage_fraction"] = 1.0
    return (out.reset_index(drop=True), coverage)


def _logit(p: np.ndarray, eps: float = 1e-06) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _binary_calibration_parameters(
    y_true: np.ndarray, p_pos: np.ndarray
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pos, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    result = {
        "CalibrationInTheLarge": float("nan"),
        "CalibrationIntercept": float("nan"),
        "CalibrationSlope": float("nan"),
    }
    if len(y) < 4 or len(np.unique(y)) < 2:
        return result
    z = _logit(p)
    if not np.isfinite(z).all():
        return result

    def score_offset(a: float) -> float:
        eta = z + float(a)
        q = np.empty_like(eta)
        pos = eta >= 0
        q[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        ez = np.exp(eta[~pos])
        q[~pos] = ez / (1.0 + ez)
        return float(np.sum(y - q))

    try:
        lo, hi = (-30.0, 30.0)
        slo, shi = (score_offset(lo), score_offset(hi))
        if np.sign(slo) != np.sign(shi):
            result["CalibrationInTheLarge"] = float(brentq(score_offset, lo, hi))
    except Exception:
        pass

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = (float(beta[0]), float(beta[1]))
        eta = a + b * z
        nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        q = np.empty_like(eta)
        pos = eta >= 0
        q[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        ez = np.exp(eta[~pos])
        q[~pos] = ez / (1.0 + ez)
        residual = q - y
        grad = np.asarray([np.sum(residual), np.sum(residual * z)], dtype=float)
        return (nll, grad)

    try:
        fit = minimize(
            lambda beta: objective(beta)[0],
            x0=np.asarray([0.0, 1.0], dtype=float),
            jac=lambda beta: objective(beta)[1],
            method="BFGS",
            options={"gtol": 1e-08, "maxiter": 500},
        )
        beta = np.asarray(fit.x, dtype=float)
        if np.isfinite(beta).all() and np.max(np.abs(beta)) < 50:
            result["CalibrationIntercept"] = float(beta[0])
            result["CalibrationSlope"] = float(beta[1])
    except Exception:
        pass
    return result


def _extended_metrics(
    frame: pd.DataFrame, *, proper_probability: bool
) -> dict[str, float]:
    pcols = _prediction_probability_columns(frame)
    if frame.empty or not pcols:
        return {metric: float("nan") for metric in OOF_METRIC_ORDER}
    y_true = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    proba = frame[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(proba).all(axis=1)
    y_true = y_true[valid].astype(int)
    proba = _renormalize_proba(proba[valid], len(pcols))
    if len(y_true) == 0:
        return {metric: float("nan") for metric in OOF_METRIC_ORDER}
    classes = np.arange(len(pcols), dtype=int)
    y_pred = classes[np.argmax(proba, axis=1)]
    out = compute_metrics(y_true, y_pred, proba, classes)
    if not proper_probability:
        for metric in _PROBABILITY_ONLY_METRICS:
            out[metric] = float("nan")
        return out
    true_columns = np.searchsorted(classes, y_true)
    if np.any(true_columns < 0) or np.any(true_columns >= len(classes)):
        raise ValueError("y_true contains a class outside the probability columns.")
    chosen = np.clip(proba[np.arange(len(y_true)), true_columns], 1e-15, 1.0)
    out["LogLoss"] = float(-np.mean(np.log(chosen)))
    one_hot = np.eye(len(classes), dtype=float)[true_columns]
    if len(classes) == 2:
        yt = (y_true == classes[-1]).astype(float)
        out["Brier"] = float(np.mean((proba[:, -1] - yt) ** 2))
        out["Brier_multiclass"] = float("nan")
        out.update(_binary_calibration_parameters(yt.astype(int), proba[:, -1]))
    else:
        out["Brier"] = float("nan")
        out["Brier_multiclass"] = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
        out.update(
            {
                "CalibrationInTheLarge": float("nan"),
                "CalibrationIntercept": float("nan"),
                "CalibrationSlope": float("nan"),
            }
        )
    return out


def _mean_metric_dicts(dicts: Iterable[dict[str, float]]) -> dict[str, float]:
    items = list(dicts)
    if not items:
        return {metric: float("nan") for metric in OOF_METRIC_ORDER}
    out: dict[str, float] = {}
    keys = list(dict.fromkeys(OOF_METRIC_ORDER + tuple(METRIC_COLUMNS)))
    for key in keys:
        vals = np.asarray([d.get(key, np.nan) for d in items], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[key] = float(np.mean(vals)) if len(vals) else float("nan")
    return out


def _nested_point_metrics(
    frame: pd.DataFrame, *, proper_probability: bool
) -> dict[str, float]:
    return _mean_metric_dicts(
        (
            _extended_metrics(group, proper_probability=proper_probability)
            for _, group in frame.groupby("_repeat", sort=True)
        )
    )


def _resample_rows_by_sample_keys(
    frame: pd.DataFrame, sample_draw: np.ndarray
) -> pd.DataFrame:
    indexed = frame.set_index("_sample_key", drop=False)
    try:
        return indexed.loc[list(map(str, sample_draw))].reset_index(drop=True)
    except KeyError:
        available = [str(x) for x in sample_draw if str(x) in indexed.index]
        return (
            indexed.loc[available].reset_index(drop=True)
            if available
            else frame.iloc[0:0]
        )


def _nested_bootstrap_metric_dicts(
    frame: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    proper_probability: bool,
) -> list[dict[str, float]]:
    repeats = sorted(frame["_repeat"].astype(str).unique().tolist())
    samples = np.asarray(
        sorted(frame["_sample_key"].astype(str).unique()), dtype=object
    )
    if not repeats or len(samples) == 0:
        return []
    repeat_frames = {rep: frame[frame["_repeat"].eq(rep)].copy() for rep in repeats}
    out: list[dict[str, float]] = []
    for _ in range(int(n_bootstrap)):
        sample_draw = rng.choice(samples, size=len(samples), replace=True)
        repeat_draw = rng.choice(
            np.asarray(repeats, dtype=object), size=len(repeats), replace=True
        )
        repeat_metrics = []
        for rep in repeat_draw:
            boot_frame = _resample_rows_by_sample_keys(
                repeat_frames[str(rep)], sample_draw
            )
            if not boot_frame.empty:
                repeat_metrics.append(
                    _extended_metrics(boot_frame, proper_probability=proper_probability)
                )
        out.append(_mean_metric_dicts(repeat_metrics))
    return out


def _lodo_point_metrics(
    frame: pd.DataFrame, *, proper_probability: bool
) -> dict[str, dict[str, float]]:
    pooled = _extended_metrics(frame, proper_probability=proper_probability)
    cohort_macro = _mean_metric_dicts(
        (
            _extended_metrics(group, proper_probability=proper_probability)
            for _, group in frame.groupby("_cohort", sort=True)
        )
    )
    return {"pooled_sample_weighted": pooled, "cohort_macro_equal_weight": cohort_macro}


def _lodo_bootstrap_metric_dicts(
    frame: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    proper_probability: bool,
) -> dict[str, list[dict[str, float]]]:
    cohorts = sorted(frame["_cohort"].astype(str).unique().tolist())
    cohort_frames = {
        cohort: frame[frame["_cohort"].eq(cohort)].reset_index(drop=True)
        for cohort in cohorts
    }
    pooled_out: list[dict[str, float]] = []
    macro_out: list[dict[str, float]] = []
    if not cohorts:
        return {
            "pooled_sample_weighted": pooled_out,
            "cohort_macro_equal_weight": macro_out,
        }
    for _ in range(int(n_bootstrap)):
        cohort_draw = rng.choice(
            np.asarray(cohorts, dtype=object), size=len(cohorts), replace=True
        )
        sampled_cohorts: list[pd.DataFrame] = []
        cohort_metric_draw: list[dict[str, float]] = []
        for draw_no, cohort in enumerate(cohort_draw):
            source = cohort_frames[str(cohort)]
            if source.empty:
                continue
            row_idx = rng.integers(0, len(source), size=len(source))
            sampled = source.iloc[row_idx].copy().reset_index(drop=True)
            sampled["_bootstrap_cohort_draw"] = f"{cohort}__draw{draw_no}"
            sampled_cohorts.append(sampled)
            cohort_metric_draw.append(
                _extended_metrics(sampled, proper_probability=proper_probability)
            )
        if sampled_cohorts:
            pooled_out.append(
                _extended_metrics(
                    pd.concat(sampled_cohorts, ignore_index=True),
                    proper_probability=proper_probability,
                )
            )
        else:
            pooled_out.append({metric: np.nan for metric in OOF_METRIC_ORDER})
        macro_out.append(_mean_metric_dicts(cohort_metric_draw))
    return {
        "pooled_sample_weighted": pooled_out,
        "cohort_macro_equal_weight": macro_out,
    }


def _bootstrap_metric_array(
    bootstrap_dicts: list[dict[str, float]], metric: str
) -> np.ndarray:
    arr = np.asarray([row.get(metric, np.nan) for row in bootstrap_dicts], dtype=float)
    return arr[np.isfinite(arr)]


def _metric_direction(metric: str) -> str:
    if metric in _LOWER_IS_BETTER:
        return "lower_is_better"
    if metric in _TARGET_ZERO:
        return "target_0"
    if metric in _TARGET_ONE:
        return "target_1"
    return "higher_is_better"


def _strategy_selection_scope(strategy: str, row: dict[str, Any]) -> str:
    selection_basis = str(row.get("selection_basis", "")).strip()
    if strategy == "MPMA-B":
        return "nested_selected_mpma_strategy"
    if strategy == "MPMA-E":
        return "nested_selected_ensemble_strategy"
    if selection_basis:
        return selection_basis
    return "displayed_comparator_strategy"


def _selection_metric_from_run(
    root: Path, strategy_rows: list[dict[str, Any]], selection_metric: str | None
) -> str:
    if selection_metric is not None and str(selection_metric).strip():
        return str(selection_metric).strip()
    for row in strategy_rows:
        for key in ("selection_metric", "optimize_metric"):
            value = str(row.get(key, "")).strip()
            if value:
                return value
    selected = _read_json(root / "ensembling" / "selected_unit.json")
    for key in ("inner_val_best_mpma", "inner_val_best_mpmas_ensemble"):
        value = selected.get(key, {})
        if isinstance(value, dict):
            metric = str(value.get("selection_metric", "")).strip()
            if metric:
                return metric
    return "nMCC"


def _oof_summary_rows(
    strategy: str,
    strategy_row: dict[str, Any],
    protocol: str,
    point_by_estimand: dict[str, dict[str, float]],
    bootstrap_by_estimand: dict[str, list[dict[str, float]]],
    selection_metric: str,
    n_bootstrap: int,
    coverage: dict[str, Any],
    probability_semantics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for estimand, point in point_by_estimand.items():
        bootstrap_rows = bootstrap_by_estimand.get(estimand, [])
        for metric in OOF_METRIC_ORDER:
            estimate = float(point.get(metric, np.nan))
            if not np.isfinite(estimate):
                continue
            boot = _bootstrap_metric_array(bootstrap_rows, metric)
            ci_low = ci_high = se = np.nan
            if len(boot):
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                se = float(np.std(boot, ddof=1)) if len(boot) > 1 else 0.0
            protocol_lower = str(protocol).lower()
            if protocol_lower in _LODO_PROTOCOLS:
                bootstrap_method = "two_stage_cohort_then_sample_percentile_bootstrap"
                resampling_unit = "cohort_then_sample"
            else:
                bootstrap_method = "repeat_and_subject_cluster_percentile_bootstrap"
                resampling_unit = "repeat_and_subject"
            rows.append(
                {
                    "Strategy": strategy,
                    "selection_scope": _strategy_selection_scope(
                        strategy, strategy_row
                    ),
                    "estimand": estimand,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "is_selection_metric": bool(metric == selection_metric),
                    "direction": _metric_direction(metric),
                    "estimate": estimate,
                    "bootstrap_se": se,
                    "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                    "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    "confidence_level": 0.95,
                    "n_bootstrap": int(n_bootstrap),
                    "valid_bootstrap": int(len(boot)),
                    "bootstrap_method": bootstrap_method,
                    "resampling_unit": resampling_unit,
                    "n_samples": int(
                        coverage.get(
                            "n_samples_complete_repeats",
                            coverage.get("n_unique_samples", 0),
                        )
                        or 0
                    ),
                    "n_repeats": int(coverage.get("n_repeats", 0) or 0),
                    "n_cohorts": int(coverage.get("n_cohorts", 0) or 0),
                    "coverage_fraction": coverage.get(
                        "complete_repeat_coverage_fraction", 1.0
                    ),
                    "probability_metrics_valid": bool(
                        probability_semantics.get("valid", False)
                    ),
                    "probability_semantics_status": str(
                        probability_semantics.get("status", "")
                    ),
                    "probability_semantics_reason": str(
                        probability_semantics.get("reason", "")
                    ),
                }
            )
    return rows


def _calibration_curve_rows(
    strategy: str, frame: pd.DataFrame, n_bins: int = 10
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    pcols = _prediction_probability_columns(frame)
    if not pcols:
        return []
    y = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    proba = frame[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(proba).all(axis=1)
    y = y[valid].astype(int)
    proba = _renormalize_proba(proba[valid], len(pcols))
    if len(y) == 0:
        return []
    class_indices = [len(pcols) - 1] if len(pcols) == 2 else list(range(len(pcols)))
    rows: list[dict[str, Any]] = []
    for j in class_indices:
        p = proba[:, j]
        yt = (y == j).astype(int)
        if len(p) == 0:
            continue
        quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins) + 1))
        edges = np.unique(np.quantile(p, quantiles))
        if len(edges) < 2:
            edges = np.asarray(
                [max(0.0, float(p[0]) - 1e-12), min(1.0, float(p[0]) + 1e-12)]
            )
        bin_index = np.searchsorted(edges, p, side="right") - 1
        bin_index = np.clip(bin_index, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            mask = bin_index == b
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append(
                {
                    "Strategy": strategy,
                    "class_index": int(j),
                    "probability_column": pcols[j],
                    "bin": int(b + 1),
                    "n": n,
                    "mean_predicted": float(np.mean(p[mask])),
                    "observed_frequency": float(np.mean(yt[mask])),
                    "bin_probability_min": float(np.min(p[mask])),
                    "bin_probability_max": float(np.max(p[mask])),
                    "binning": "equal_frequency",
                    "requested_bins": int(n_bins),
                    "uncertainty": "descriptive_curve_only; use pooled calibration parameter bootstrap CIs for inference",
                }
            )
    return rows


def _canonical_pair_frame(
    strategy_a: str,
    frame_a: pd.DataFrame,
    strategy_b: str,
    frame_b: pd.DataFrame,
    protocol: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    a, _ = _prepare_oof_predictions(frame_a, protocol)
    b, _ = _prepare_oof_predictions(frame_b, protocol)
    if a.empty or b.empty:
        return (
            pd.DataFrame(),
            {
                "status": "unavailable",
                "reason": "one or both strategies lack valid OOF predictions",
            },
        )
    pcols_a = _prediction_probability_columns(a)
    pcols_b = _prediction_probability_columns(b)
    if len(pcols_a) != len(pcols_b):
        return (
            pd.DataFrame(),
            {
                "status": "unavailable",
                "reason": "strategies expose different class-probability dimensions",
            },
        )
    protocol_lower = str(protocol).lower()
    keys = (
        ["_cohort", "_sample_key"]
        if protocol_lower in _LODO_PROTOCOLS
        else ["_repeat", "_sample_key"]
    )
    left = a[keys + ["y_true"] + pcols_a].copy()
    right = b[keys + ["y_true"] + pcols_b].copy()
    left = left.rename(
        columns={
            "y_true": "y_true_a",
            **{c: f"a_proba_{i}" for i, c in enumerate(pcols_a)},
        }
    )
    right = right.rename(
        columns={
            "y_true": "y_true_b",
            **{c: f"b_proba_{i}" for i, c in enumerate(pcols_b)},
        }
    )
    left_keys = set(map(tuple, left[keys].astype(str).to_numpy()))
    right_keys = set(map(tuple, right[keys].astype(str).to_numpy()))
    if left_keys != right_keys:
        missing_from_b = len(left_keys - right_keys)
        missing_from_a = len(right_keys - left_keys)
        raise ValueError(
            f"{strategy_a} vs {strategy_b}: paired OOF coverage differs ({missing_from_b} observation(s) missing from {strategy_b}; {missing_from_a} observation(s) missing from {strategy_a})"
        )
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(
            f"{strategy_a} vs {strategy_b}: no paired OOF observations are available"
        )
    same_y = merged["y_true_a"].astype(int).eq(merged["y_true_b"].astype(int))
    n_label_mismatches = int((~same_y).sum())
    if n_label_mismatches:
        raise ValueError(
            f"{strategy_a} vs {strategy_b}: {n_label_mismatches} paired OOF observation(s) have inconsistent outcome labels"
        )
    info = {
        "status": "ok",
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "n_paired_rows": int(len(merged)),
        "n_paired_samples": int(merged["_sample_key"].nunique())
        if not merged.empty
        else 0,
        "n_repeats": int(merged["_repeat"].nunique())
        if not merged.empty and "_repeat" in merged.columns
        else 0,
        "n_cohorts": int(merged["_cohort"].nunique())
        if not merged.empty and "_cohort" in merged.columns
        else 0,
        "n_classes": int(len(pcols_a)),
        "n_label_mismatches_removed": n_label_mismatches,
    }
    return (merged.reset_index(drop=True), info)


def _pair_side_frame(frame: pd.DataFrame, side: str, n_classes: int) -> pd.DataFrame:
    out = pd.DataFrame({"y_true": frame["y_true_a"].astype(int).to_numpy()})
    for j in range(int(n_classes)):
        out[f"proba_{j}"] = pd.to_numeric(
            frame[f"{side}_proba_{j}"], errors="coerce"
        ).to_numpy(dtype=float)
    return out


def _metric_difference_dict(
    frame: pd.DataFrame, n_classes: int
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    a_metrics = _extended_metrics(
        _pair_side_frame(frame, "a", n_classes), proper_probability=True
    )
    b_metrics = _extended_metrics(
        _pair_side_frame(frame, "b", n_classes), proper_probability=True
    )
    raw_diff: dict[str, float] = {}
    for metric in OOF_METRIC_ORDER:
        av = float(a_metrics.get(metric, np.nan))
        bv = float(b_metrics.get(metric, np.nan))
        raw_diff[metric] = av - bv if np.isfinite(av) and np.isfinite(bv) else np.nan
    return (a_metrics, b_metrics, raw_diff)


def _nested_pair_point_metrics(
    pair: pd.DataFrame, n_classes: int
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    per_a: list[dict[str, float]] = []
    per_b: list[dict[str, float]] = []
    per_d: list[dict[str, float]] = []
    for _, group in pair.groupby("_repeat", sort=True):
        a, b, d = _metric_difference_dict(group, n_classes)
        per_a.append(a)
        per_b.append(b)
        per_d.append(d)
    return (
        _mean_metric_dicts(per_a),
        _mean_metric_dicts(per_b),
        _mean_metric_dicts(per_d),
    )


def _nested_pair_bootstrap_differences(
    pair: pd.DataFrame, n_classes: int, n_bootstrap: int, rng: np.random.Generator
) -> list[dict[str, float]]:
    repeats = sorted(pair["_repeat"].astype(str).unique().tolist())
    samples = np.asarray(sorted(pair["_sample_key"].astype(str).unique()), dtype=object)
    if not repeats or len(samples) == 0:
        return []
    repeat_frames = {
        rep: pair[pair["_repeat"].eq(rep)].set_index("_sample_key", drop=False)
        for rep in repeats
    }
    out: list[dict[str, float]] = []
    for _ in range(int(n_bootstrap)):
        sample_draw = rng.choice(samples, size=len(samples), replace=True)
        repeat_draw = rng.choice(
            np.asarray(repeats, dtype=object), size=len(repeats), replace=True
        )
        diffs: list[dict[str, float]] = []
        for rep in repeat_draw:
            source = repeat_frames[str(rep)]
            sampled = source.loc[list(map(str, sample_draw))].reset_index(drop=True)
            _, _, d = _metric_difference_dict(sampled, n_classes)
            diffs.append(d)
        out.append(_mean_metric_dicts(diffs))
    return out


def _lodo_pair_point_metrics(
    pair: pd.DataFrame, n_classes: int
) -> dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]]:
    pooled = _metric_difference_dict(pair, n_classes)
    cohort_a: list[dict[str, float]] = []
    cohort_b: list[dict[str, float]] = []
    cohort_d: list[dict[str, float]] = []
    for _, group in pair.groupby("_cohort", sort=True):
        a, b, d = _metric_difference_dict(group, n_classes)
        cohort_a.append(a)
        cohort_b.append(b)
        cohort_d.append(d)
    macro = (
        _mean_metric_dicts(cohort_a),
        _mean_metric_dicts(cohort_b),
        _mean_metric_dicts(cohort_d),
    )
    return {"pooled_sample_weighted": pooled, "cohort_macro_equal_weight": macro}


def _lodo_pair_bootstrap_differences(
    pair: pd.DataFrame, n_classes: int, n_bootstrap: int, rng: np.random.Generator
) -> dict[str, list[dict[str, float]]]:
    cohorts = sorted(pair["_cohort"].astype(str).unique().tolist())
    cohort_frames = {
        cohort: pair[pair["_cohort"].eq(cohort)].reset_index(drop=True)
        for cohort in cohorts
    }
    pooled_out: list[dict[str, float]] = []
    macro_out: list[dict[str, float]] = []
    if not cohorts:
        return {
            "pooled_sample_weighted": pooled_out,
            "cohort_macro_equal_weight": macro_out,
        }
    for _ in range(int(n_bootstrap)):
        cohort_draw = rng.choice(
            np.asarray(cohorts, dtype=object), size=len(cohorts), replace=True
        )
        sampled_frames: list[pd.DataFrame] = []
        cohort_diffs: list[dict[str, float]] = []
        for cohort in cohort_draw:
            source = cohort_frames[str(cohort)]
            idx = rng.integers(0, len(source), size=len(source))
            sampled = source.iloc[idx].reset_index(drop=True)
            sampled_frames.append(sampled)
            _, _, d = _metric_difference_dict(sampled, n_classes)
            cohort_diffs.append(d)
        if sampled_frames:
            pooled = pd.concat(sampled_frames, ignore_index=True)
            _, _, pooled_d = _metric_difference_dict(pooled, n_classes)
            pooled_out.append(pooled_d)
        else:
            pooled_out.append({metric: np.nan for metric in OOF_METRIC_ORDER})
        macro_out.append(_mean_metric_dicts(cohort_diffs))
    return {
        "pooled_sample_weighted": pooled_out,
        "cohort_macro_equal_weight": macro_out,
    }


def _oof_contrast_metrics() -> tuple[str, ...]:
    return tuple(
        (
            metric
            for metric in OOF_METRIC_ORDER
            if metric
            not in {"CalibrationInTheLarge", "CalibrationIntercept", "CalibrationSlope"}
        )
    )


def _run_oof_pairwise_contrasts(
    frames: dict[str, pd.DataFrame],
    protocol: str,
    selection_metric: str,
    n_bootstrap: int,
    random_state: int,
    probability_semantics: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = list(frames)
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    protocol_lower = str(protocol).lower()
    for strategy_a, strategy_b in itertools.combinations(strategies, 2):
        pair, info = _canonical_pair_frame(
            strategy_a, frames[strategy_a], strategy_b, frames[strategy_b], protocol
        )
        coverage_rows.append(info)
        if pair.empty:
            continue
        n_classes = int(info.get("n_classes", 0) or 0)
        seed = _stable_seed(
            random_state, "oof_pair", strategy_a, strategy_b, protocol_lower
        )
        rng = np.random.default_rng(seed)
        if protocol_lower in _LODO_PROTOCOLS:
            point_by_estimand = _lodo_pair_point_metrics(pair, n_classes)
            boot_by_estimand = _lodo_pair_bootstrap_differences(
                pair, n_classes, n_bootstrap, rng
            )
            bootstrap_method = (
                "paired_two_stage_cohort_then_sample_percentile_bootstrap"
            )
            resampling_unit = "paired_cohort_then_subject"
        else:
            a, b, d = _nested_pair_point_metrics(pair, n_classes)
            point_by_estimand = {"mean_repeat_pooled_oof": (a, b, d)}
            boot_by_estimand = {
                "mean_repeat_pooled_oof": _nested_pair_bootstrap_differences(
                    pair, n_classes, n_bootstrap, rng
                )
            }
            bootstrap_method = "paired_repeat_and_subject_cluster_percentile_bootstrap"
            resampling_unit = "paired_repeat_and_subject"
        for estimand, (a_metrics, b_metrics, d_metrics) in point_by_estimand.items():
            boot_rows = boot_by_estimand.get(estimand, [])
            for metric in _oof_contrast_metrics():
                if metric in _PROPER_SCORING_METRICS:
                    valid_a = bool(
                        probability_semantics.get(strategy_a, {}).get("valid", False)
                    )
                    valid_b = bool(
                        probability_semantics.get(strategy_b, {}).get("valid", False)
                    )
                    if not (valid_a and valid_b):
                        continue
                av = float(a_metrics.get(metric, np.nan))
                bv = float(b_metrics.get(metric, np.nan))
                diff = float(d_metrics.get(metric, np.nan))
                if not (np.isfinite(av) and np.isfinite(bv) and np.isfinite(diff)):
                    continue
                boot = _bootstrap_metric_array(boot_rows, metric)
                if len(boot):
                    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                    se = float(np.std(boot, ddof=1)) if len(boot) > 1 else np.nan
                else:
                    ci_low = ci_high = se = np.nan
                direction = _metric_direction(metric)
                if direction == "lower_is_better":
                    advantage = -diff
                    advantage_low = -float(ci_high) if np.isfinite(ci_high) else np.nan
                    advantage_high = -float(ci_low) if np.isfinite(ci_low) else np.nan
                else:
                    advantage = diff
                    advantage_low = float(ci_low) if np.isfinite(ci_low) else np.nan
                    advantage_high = float(ci_high) if np.isfinite(ci_high) else np.nan
                rows.append(
                    {
                        "estimand": estimand,
                        "metric": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "is_selection_metric": bool(metric == selection_metric),
                        "direction": direction,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "estimate_a": av,
                        "estimate_b": bv,
                        "difference_a_minus_b": diff,
                        "difference_ci_low": float(ci_low)
                        if np.isfinite(ci_low)
                        else np.nan,
                        "difference_ci_high": float(ci_high)
                        if np.isfinite(ci_high)
                        else np.nan,
                        "advantage_a_over_b": advantage,
                        "advantage_ci_low": advantage_low,
                        "advantage_ci_high": advantage_high,
                        "bootstrap_se_difference": se,
                        "confidence_level": 0.95,
                        "n_bootstrap": int(n_bootstrap),
                        "valid_bootstrap": int(len(boot)),
                        "bootstrap_method": bootstrap_method,
                        "resampling_unit": resampling_unit,
                        "n_paired_samples": int(info.get("n_paired_samples", 0) or 0),
                        "n_repeats": int(info.get("n_repeats", 0) or 0),
                        "n_cohorts": int(info.get("n_cohorts", 0) or 0),
                        "p_value": np.nan,
                        "p_value_policy": "not_reported_for_pooled_oof_contrasts",
                        "inference_scope": "paired_oof_prediction_level_uncertainty_conditional_on_the_cv_fits",
                    }
                )
    return (pd.DataFrame(rows), pd.DataFrame(coverage_rows))


def _oof_performance_table(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty:
        return pd.DataFrame()
    required = {"Strategy", "estimand", "metric", "estimate", "ci_low", "ci_high"}
    if not required.issubset(performance.columns):
        return pd.DataFrame()
    metric_order = [
        m for m in OOF_METRIC_ORDER if m in set(performance["metric"].astype(str))
    ]
    estimands = [
        str(x)
        for x in performance["estimand"].dropna().astype(str).unique().tolist()
        if str(x)
    ]
    show_estimand = len(set(estimands)) > 1
    rows: list[dict[str, Any]] = []
    for (strategy, estimand), group in performance.groupby(
        ["Strategy", "estimand"], sort=False
    ):
        row: dict[str, Any] = {"Strategy": strategy}
        if show_estimand:
            row["Estimand"] = estimand
        for metric in metric_order:
            sub = group[group["metric"].astype(str).eq(metric)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            estimate = pd.to_numeric(
                pd.Series([r.get("estimate")]), errors="coerce"
            ).iloc[0]
            low = pd.to_numeric(pd.Series([r.get("ci_low")]), errors="coerce").iloc[0]
            high = pd.to_numeric(pd.Series([r.get("ci_high")]), errors="coerce").iloc[0]
            if not np.isfinite(estimate):
                cell = ""
            elif np.isfinite(low) and np.isfinite(high):
                cell = f"{float(estimate):.4f} [{float(low):.4f}, {float(high):.4f}]"
            else:
                cell = f"{float(estimate):.4f}"
            label = METRIC_LABELS.get(metric, metric)
            row[label] = cell
        rows.append(row)
    return pd.DataFrame(rows)


def _run_oof_statistics(
    root: Path,
    frames: dict[str, pd.DataFrame],
    strategy_rows: list[dict[str, Any]],
    protocol: str,
    selection_metric: str,
    n_bootstrap: int,
    random_state: int,
    calibration_bins: int,
) -> dict[str, Any]:
    rows_by_strategy = {
        str(row.get("Strategy", "")): row
        for row in strategy_rows
        if str(row.get("Strategy", "")).strip()
    }
    config_learners = _config_learner_map(root)
    expected = _expected_oof_design(root, protocol)
    performance_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    probability_semantics_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy, raw_predictions in frames.items():
        strategy_row = rows_by_strategy.get(strategy, {})
        semantics = _strategy_probability_semantics(
            strategy, raw_predictions, strategy_row, config_learners
        )
        probability_semantics_by_strategy[strategy] = semantics
        if selection_metric in _PROBABILITY_ONLY_METRICS and (
            not bool(semantics.get("valid"))
        ):
            raise ValueError(
                f"{strategy}: selection metric {selection_metric!r} requires probabilistic outputs, but this strategy does not provide them ({semantics.get('reason', 'unverified probability semantics')})."
            )
        prepared, coverage = _prepare_oof_predictions(
            raw_predictions,
            protocol,
            strategy=strategy,
            expected_n_samples=expected.get("n_samples"),
            expected_repeats=expected.get("n_repeats"),
        )
        coverage_row = {
            "Strategy": strategy,
            **coverage,
            "probability_metrics_valid": bool(semantics.get("valid", False)),
            "probability_semantics_status": str(semantics.get("status", "")),
            "probability_semantics_reason": str(semantics.get("reason", "")),
        }
        protocol_lower = str(protocol).lower()
        strategy_seed = _stable_seed(random_state, "oof", strategy, protocol_lower)
        rng = np.random.default_rng(strategy_seed)
        proper_probability = bool(semantics.get("valid", False))
        if protocol_lower in _LODO_PROTOCOLS:
            point_by_estimand = _lodo_point_metrics(
                prepared, proper_probability=proper_probability
            )
            bootstrap_by_estimand = _lodo_bootstrap_metric_dicts(
                prepared, n_bootstrap, rng, proper_probability=proper_probability
            )
            coverage_row["n_unique_samples"] = int(prepared["_sample_key"].nunique())
            coverage_row["n_cohorts"] = int(prepared["_cohort"].nunique())
            coverage_row["inference_population"] = "all_held_out_cohorts_and_samples"
            curve_frame = prepared
        else:
            point_by_estimand = {
                "mean_repeat_pooled_oof": _nested_point_metrics(
                    prepared, proper_probability=proper_probability
                )
            }
            bootstrap_by_estimand = {
                "mean_repeat_pooled_oof": _nested_bootstrap_metric_dicts(
                    prepared, n_bootstrap, rng, proper_probability=proper_probability
                )
            }
            coverage_row["n_samples_any_repeat"] = int(
                prepared["_sample_key"].nunique()
            )
            coverage_row["n_samples_complete_repeats"] = int(
                prepared["_sample_key"].nunique()
            )
            coverage_row["complete_repeat_coverage_fraction"] = 1.0
            coverage_row["inference_population"] = (
                "all_subjects_with_complete_oof_coverage"
            )
            curve_frame = prepared
        performance_rows.extend(
            _oof_summary_rows(
                strategy,
                strategy_row,
                protocol,
                point_by_estimand,
                bootstrap_by_estimand,
                selection_metric,
                n_bootstrap,
                coverage,
                semantics,
            )
        )
        if proper_probability:
            calibration_rows.extend(
                _calibration_curve_rows(strategy, curve_frame, n_bins=calibration_bins)
            )
        coverage_rows.append(coverage_row)
    return {
        "performance": pd.DataFrame(performance_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "coverage": pd.DataFrame(coverage_rows),
        "probability_semantics": probability_semantics_by_strategy,
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
    legacy_manifest = {
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
        "note": "This manifest describes the original outer-unit reporting layer. See strategy_oof_statistics_manifest.json for the additive pooled-OOF layer.",
    }
    manifest_path = tables / "strategy_statistics_manifest.json"
    dump_json_standard(legacy_manifest, manifest_path)
    resolved_selection = _selection_metric_from_run(
        root, strategy_rows, selection_metric
    )
    advanced = _run_oof_statistics(
        root,
        frames,
        strategy_rows,
        protocol,
        resolved_selection,
        int(n_bootstrap),
        int(random_state),
        int(calibration_bins),
    )
    oof_performance = advanced["performance"]
    oof_calibration = advanced["calibration"]
    oof_coverage = advanced["coverage"]
    oof_performance_table = _oof_performance_table(oof_performance)
    oof_pairwise, oof_pairwise_coverage = _run_oof_pairwise_contrasts(
        frames,
        protocol,
        resolved_selection,
        int(n_bootstrap),
        int(random_state),
        advanced["probability_semantics"],
    )
    oof_performance_path = tables / "strategy_oof_performance.tsv"
    oof_performance_table_path = tables / "strategy_oof_performance_table.tsv"
    oof_calibration_path = tables / "strategy_oof_calibration.tsv"
    oof_pairwise_path = tables / "strategy_oof_pairwise_contrasts.tsv"
    oof_performance.to_csv(oof_performance_path, sep="\t", index=False)
    oof_performance_table.to_csv(oof_performance_table_path, sep="\t", index=False)
    oof_calibration.to_csv(oof_calibration_path, sep="\t", index=False)
    oof_pairwise.to_csv(oof_pairwise_path, sep="\t", index=False)
    for stale_name in (
        "strategy_oof_coverage.tsv",
        "strategy_oof_pairwise_coverage.tsv",
    ):
        stale_path = tables / stale_name
        if stale_path.exists():
            stale_path.unlink()
    available_additional = [
        metric
        for metric in OOF_METRIC_ORDER
        if metric != resolved_selection
        and (
            oof_performance.empty
            or "metric" not in oof_performance.columns
            or metric in set(oof_performance["metric"].astype(str))
        )
    ]
    protocol_lower = str(protocol).lower()
    observed_design: dict[str, Any] = {}
    if not oof_coverage.empty:
        for _, coverage_row in oof_coverage.iterrows():
            strategy_name = str(coverage_row.get("Strategy", ""))
            design_row: dict[str, int] = {}
            for key in ("n_unique_samples", "n_outer_units", "n_repeats", "n_cohorts"):
                value = pd.to_numeric(
                    pd.Series([coverage_row.get(key)]), errors="coerce"
                ).iloc[0]
                if pd.notna(value) and np.isfinite(float(value)):
                    design_row[key] = int(value)
            observed_design[strategy_name] = design_row
    advanced_manifest = {
        "protocol": str(protocol),
        "strategies": list(frames),
        "strategy_scope": "Strategies represented in the report table with available held-out predictions.",
        "selection_metric": resolved_selection,
        "selection_metric_definition": "Metric used for model and ensemble selection; defaults to nMCC when unavailable.",
        "additional_metrics": available_additional,
        "proper_scoring_rules": ["Brier", "Brier_multiclass", "LogLoss"],
        "calibration_summary": [
            "CalibrationInTheLarge",
            "CalibrationIntercept",
            "CalibrationSlope",
        ],
        "calibration_targets": {
            "CalibrationInTheLarge": 0.0,
            "CalibrationIntercept": 0.0,
            "CalibrationSlope": 1.0,
        },
        "calibration_curve": {
            "file": str(oof_calibration_path.name),
            "binning": "equal-frequency",
            "requested_bins": int(calibration_bins),
            "uncertainty": "Scalar calibration uncertainty is reported in strategy_oof_performance.tsv.",
            "multiclass": "one-vs-rest curves for every class",
            "binary": "curve for the package's positive probability column",
        },
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": 0.95,
        "ci_method": "percentile bootstrap",
        "nested_cv_estimand": "mean of repeat-specific pooled OOF metrics after concatenating all outer test folds within each repeat",
        "nested_cv_resampling": "resample subjects with replacement as clusters across repeats, resample repeat IDs with replacement, recompute pooled OOF metrics within each selected repeat, then average across repeats",
        "oof_coverage_policy": "Complete held-out coverage is required; missing, duplicate, or non-finite predictions raise an error.",
        "lodo_estimands": ["pooled_sample_weighted", "cohort_macro_equal_weight"],
        "lodo_resampling": "two-stage nonparametric bootstrap: resample held-out cohorts with replacement, then resample samples within each selected cohort; recompute both pooled and equal-cohort metrics",
        "inference_unit": "cohort_then_sample"
        if protocol_lower in _LODO_PROTOCOLS
        else "repeat_and_subject",
        "selection_interpretation": {
            "MPMA-B": "Fold-specific MPMA selected from inner-validation data.",
            "MPMA-E": "Fold-specific ensemble selected from inner-validation predictions.",
            "comparators": "Comparator config_id matches the report table.",
        },
        "oof_estimand": "Pooled held-out performance with protocol-aware bootstrap uncertainty.",
        "oof_pairwise_contrasts": {
            "file": str(oof_pairwise_path.name),
            "design": "paired resampling: the same subject/repeat or cohort/subject draws are applied to both strategies before recomputing the metric difference",
            "p_values": "not reported",
            "reason": "Pairwise OOF output is reported as an effect-size confidence interval.",
            "scope": "paired effect-size confidence intervals",
        },
        "calibration_testing_policy": "Calibration-in-the-large and intercept are referenced to 0; slope is referenced to 1.",
        "probability_metric_policy": "Proper scoring and calibration metrics require verified probabilistic outputs.",
        "probability_semantics": advanced["probability_semantics"],
        "observed_oof_design": observed_design,
        "files": {
            "performance_long": str(oof_performance_path.name),
            "performance_table": str(oof_performance_table_path.name),
            "calibration": str(oof_calibration_path.name),
            "pairwise_contrasts": str(oof_pairwise_path.name),
        },
    }
    oof_manifest_path = tables / "strategy_oof_statistics_manifest.json"
    dump_json_standard(advanced_manifest, oof_manifest_path)
    return {
        "unit_metrics": unit_metrics,
        "summary": summary,
        "pairwise": pairwise,
        "unit_metrics_path": unit_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "manifest_path": manifest_path,
        "oof_performance": oof_performance,
        "oof_performance_table": oof_performance_table,
        "oof_calibration": oof_calibration,
        "oof_coverage": oof_coverage,
        "oof_pairwise": oof_pairwise,
        "oof_pairwise_coverage": oof_pairwise_coverage,
        "oof_performance_path": oof_performance_path,
        "oof_performance_table_path": oof_performance_table_path,
        "oof_calibration_path": oof_calibration_path,
        "oof_pairwise_path": oof_pairwise_path,
        "oof_manifest_path": oof_manifest_path,
        "selection_metric": resolved_selection,
    }
