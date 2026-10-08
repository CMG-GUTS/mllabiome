from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .explainability_config import (
    ExplainabilityConfigurationError,
    ExplainabilityDependencyError,
    _auto_ale_bins,
)
from .explainability_methods import ALE, ALEInteractions, Permutation
from .explainability_model import _explain_predict_class_probability, _project_input
from .explainability_reporting import (
    _class_slug,
    _collapse_duplicate_feature_importance,
    _feature_distribution_stats,
    _method_display,
    _plain_taxon_label,
    _plot_feature_importance,
    _rank_support_from_importance,
)
from .explainability_runtime import _quiet_pyale_info
from .metrics import _predict_proba_aligned
from .storage import write_table
from .style import ACC_D, ACC_L, BG, INK, MID, TRACK
from .style import apply as apply_style
from .style import save_all
from .utils import _as_float_matrix


def _permutation_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    *,
    spec: Permutation,
    random_state: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    input_projector: Any | None = None,
) -> pd.DataFrame:
    classes = np.arange(len(class_labels), dtype=int)
    X = _as_float_matrix(X)
    y = np.asarray(y, dtype=int)
    scoring_name = str(spec.scoring).strip().lower()
    if scoring_name not in {"log_loss", "brier"}:
        raise ExplainabilityConfigurationError(
            "Permutation.scoring must be 'log_loss' or 'brier'."
        )
    n_samples, n_features = X.shape
    n_repeats = int(spec.n_repeats)
    if isinstance(spec.max_samples, float):
        n_draw = max(1, min(n_samples, int(round(float(spec.max_samples) * n_samples))))
    else:
        n_draw = max(1, min(n_samples, int(spec.max_samples)))
    seed_rng = np.random.RandomState(int(random_state))
    random_seed = int(seed_rng.randint(np.iinfo(np.int32).max + 1))
    values = np.full((len(class_indices), n_features, n_repeats), np.nan, dtype=float)
    total = max(1, n_features * n_repeats)
    done = 0
    if progress_callback is not None:
        progress_callback(0, total, f"0/{n_features} features")

    def scores(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
        out = np.empty(len(class_indices), dtype=float)
        eps = np.finfo(float).eps
        for pos, class_index in enumerate(class_indices):
            c = int(class_index)
            target = (np.asarray(y_true, dtype=int) == c).astype(float)
            p = np.clip(np.asarray(proba, dtype=float)[:, c], eps, 1.0 - eps)
            if scoring_name == "brier":
                out[pos] = float(np.mean((target - p) ** 2))
            else:
                out[pos] = float(
                    -np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
                )
        return out

    for feature_index, feature in enumerate(feature_names):
        feature_rng = np.random.RandomState(random_seed)
        if n_draw < n_samples:
            idx = np.sort(feature_rng.choice(n_samples, size=n_draw, replace=False))
            X_eval = X[idx].copy()
            y_eval = y[idx]
        else:
            X_eval = X.copy()
            y_eval = y
        baseline = scores(
            y_eval,
            _predict_proba_aligned(
                clf, _project_input(X_eval, input_projector), classes
            ),
        )
        shuffling_idx = np.arange(len(X_eval))
        for repeat_index in range(n_repeats):
            feature_rng.shuffle(shuffling_idx)
            X_permuted = X_eval.copy()
            X_permuted[:, feature_index] = X_eval[shuffling_idx, feature_index]
            permuted = _predict_proba_aligned(
                clf, _project_input(X_permuted, input_projector), classes
            )
            values[:, feature_index, repeat_index] = scores(y_eval, permuted) - baseline
            done += 1
            if progress_callback is not None:
                progress_callback(
                    done,
                    total,
                    f"feature {feature_index + 1}/{n_features} · repeat {repeat_index + 1}/{n_repeats} · {feature}",
                )

    rows: list[pd.DataFrame] = []
    for pos, class_index in enumerate(class_indices):
        c = int(class_index)
        arr = values[pos]
        rows.append(
            pd.DataFrame(
                {
                    "method": "permutation",
                    "class_index": c,
                    "class_label": str(class_labels[c]),
                    "feature": list(feature_names),
                    "importance_mean": np.nanmean(arr, axis=1),
                    "within_fold_importance_sd": np.nanstd(arr, axis=1, ddof=1)
                    if n_repeats > 1
                    else np.zeros(n_features, dtype=float),
                    "scoring": f"increase_in_one_vs_rest_{scoring_name}"
                    + ("_geometry_projected" if input_projector is not None else ""),
                    "perturbation_projection": "fitted_model_input_geometry"
                    if input_projector is not None
                    else "none",
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["class_index", "importance_mean", "feature"],
        ascending=[True, False, True],
    )


class _AleModelWrapper:
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self._fn = fn

    def predict(self, X: Any) -> np.ndarray:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.asarray(self._fn(arr), dtype=float)


def _require_pyale():
    try:
        from PyALE import ale as pyale
    except Exception as exc:
        raise ExplainabilityDependencyError(
            "Explainability.methods includes 'ale' or 'interactions', but the 'PyALE' package is not importable. "
            "Install PyALE or remove ALE-based methods from Explainability.methods. No fallback method will be used."
        ) from exc
    return pyale


def _ale_result_values(
    result: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    x1 = x2 = None
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        if "eff" in df.columns:
            vals = df["eff"].to_numpy(dtype=float)
        else:
            num = df.select_dtypes(include=[np.number])
            drop_cols = [
                c
                for c in num.columns
                if str(c).lower() in {"size", "count", "counts", "n"}
                or "ci" in str(c).lower()
            ]
            if drop_cols:
                num = num.drop(columns=drop_cols, errors="ignore")
            vals = num.to_numpy(dtype=float).ravel()
        if isinstance(df.index, pd.MultiIndex) and len(df.index) == len(vals):
            x1 = np.asarray([idx[0] for idx in df.index], dtype=float)
            x2 = np.asarray([idx[1] for idx in df.index], dtype=float)
        elif not isinstance(df.index, pd.MultiIndex) and len(df.index) == len(vals):
            try:
                x1 = df.index.to_numpy(dtype=float)
            except Exception:
                x1 = None
    elif isinstance(result, pd.Series):
        vals = result.to_numpy(dtype=float)
        if isinstance(result.index, pd.MultiIndex) and len(result.index) == len(vals):
            x1 = np.asarray([idx[0] for idx in result.index], dtype=float)
            x2 = np.asarray([idx[1] for idx in result.index], dtype=float)
        elif len(result.index) == len(vals):
            try:
                x1 = result.index.to_numpy(dtype=float)
            except Exception:
                x1 = None
    else:
        vals = np.asarray(result, dtype=float).ravel()
    vals = np.asarray(vals, dtype=float).ravel()
    finite = np.isfinite(vals)
    if x1 is not None and x2 is not None:
        finite = finite & np.isfinite(x1) & np.isfinite(x2)
        return vals[finite], x1[finite], x2[finite]
    if x1 is not None and x2 is None and len(x1) == len(vals):
        finite = finite & np.isfinite(x1)
        return vals[finite], x1[finite], None
    return vals[finite], None, None


def _ale_1d_effect_summary(result: Any, vals: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(vals, dtype=float).ravel()
    values = values[np.isfinite(values)]
    representative = values
    weights: np.ndarray | None = None
    if isinstance(result, pd.DataFrame) and {"eff", "size"}.issubset(result.columns):
        eff = pd.to_numeric(result["eff"], errors="coerce").to_numpy(dtype=float)
        size = pd.to_numeric(result["size"], errors="coerce").to_numpy(dtype=float)
        if len(eff) == len(size) and len(eff):
            if len(eff) > 1 and not np.isfinite(size[0]):
                representative = (eff[1:] + eff[:-1]) / 2.0
                weights = size[1:]
            else:
                representative = eff
                weights = size
    representative = np.asarray(representative, dtype=float).ravel()
    if weights is None:
        finite = np.isfinite(representative)
        representative = representative[finite]
        if not len(representative):
            return float("nan"), float("nan"), float("nan")
        mean = float(np.mean(representative))
        rms = float(np.sqrt(np.mean(representative**2)))
        sd = float(np.std(representative, ddof=1)) if len(representative) > 1 else 0.0
        return rms, mean, sd
    weights = np.asarray(weights, dtype=float).ravel()
    finite = np.isfinite(representative) & np.isfinite(weights) & (weights > 0)
    representative = representative[finite]
    weights = weights[finite]
    if not len(representative) or float(np.sum(weights)) <= 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.average(representative, weights=weights))
    rms = float(np.sqrt(np.average(representative**2, weights=weights)))
    sd = float(np.sqrt(np.average((representative - mean) ** 2, weights=weights)))
    return rms, mean, sd


def _ale_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    *,
    spec: ALE,
    top_features: Sequence[str] | None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    input_projector: Any | None = None,
) -> pd.DataFrame:
    pyale = _require_pyale()
    X = _as_float_matrix(X)
    feature_names = list(feature_names)
    name_to_index = {str(name): i for i, name in enumerate(feature_names)}
    features = list(top_features) if top_features else feature_names
    features = [str(f) for f in features if str(f) in name_to_index]
    if not features:
        raise ExplainabilityConfigurationError(
            "ALE was requested, but no candidate features were available."
        )
    df_X = pd.DataFrame(X, columns=feature_names)
    n_bins = _auto_ale_bins(X.shape[0], spec)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    classes = np.arange(len(class_labels), dtype=int)
    total = max(1, len(class_indices) * len(features))
    done = 0
    if progress_callback is not None:
        progress_callback(
            0, total, f"0/{len(features)} features · {len(class_indices)} classes"
        )
    for class_index in class_indices:
        c = int(class_index)
        predict_class = _explain_predict_class_probability(clf, classes, c)
        wrapper = _AleModelWrapper(
            lambda values, fn=predict_class: fn(_project_input(values, input_projector))
        )
        for fname in features:
            try:
                col = X[:, name_to_index[fname]]
                finite_col = col[np.isfinite(col)]
                n_distinct = int(np.unique(finite_col).size)
                if n_distinct < 2:
                    skipped.append(
                        {
                            "class_index": c,
                            "class_label": str(class_labels[c]),
                            "feature": fname,
                            "reason": "fewer_than_two_distinct_finite_values",
                            "n_finite": int(finite_col.size),
                            "n_distinct": n_distinct,
                        }
                    )
                    continue
                try:
                    with _quiet_pyale_info():
                        result = pyale(
                            df_X,
                            wrapper,
                            [fname],
                            grid_size=int(n_bins),
                            include_CI=False,
                            plot=False,
                        )
                    vals, grid, _ = _ale_result_values(result)
                except Exception as exc:
                    skipped.append(
                        {
                            "class_index": c,
                            "class_label": str(class_labels[c]),
                            "feature": fname,
                            "reason": "pyale_failed",
                            "error": str(exc)[:500],
                            "n_finite": int(finite_col.size),
                            "n_distinct": n_distinct,
                        }
                    )
                    continue
                if vals.size == 0:
                    skipped.append(
                        {
                            "class_index": c,
                            "class_label": str(class_labels[c]),
                            "feature": fname,
                            "reason": "no_finite_ale_effect_values",
                            "n_finite": int(finite_col.size),
                            "n_distinct": n_distinct,
                        }
                    )
                    continue
                strength, _, effect_sd = _ale_1d_effect_summary(result, vals)
                if not np.isfinite(strength):
                    continue
                rows.append(
                    {
                        "method": "ale",
                        "class_index": c,
                        "class_label": str(class_labels[c]),
                        "feature": fname,
                        "importance_mean": strength,
                        "within_fold_importance_sd": effect_sd,
                        "scoring": "rms_distribution_weighted_class_probability_ale"
                        + (
                            "_geometry_projected" if input_projector is not None else ""
                        ),
                        "perturbation_projection": "fitted_model_input_geometry"
                        if input_projector is not None
                        else "none",
                    }
                )
                if grid is not None and len(grid) == len(vals):
                    for x_value, effect in zip(grid, vals):
                        curves.append(
                            {
                                "class_index": c,
                                "class_label": str(class_labels[c]),
                                "feature": fname,
                                "grid_value": float(x_value),
                                "ale_effect": float(effect),
                            }
                        )
            finally:
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total, f"{class_labels[c]} · {fname}")
    frame = pd.DataFrame(rows)
    frame.attrs["curves"] = pd.DataFrame(curves)
    frame.attrs["skipped_features"] = pd.DataFrame(skipped)
    return (
        frame.sort_values(
            ["class_index", "importance_mean", "feature"],
            ascending=[True, False, True],
        )
        if not frame.empty
        else frame
    )


def _same_lineage_pair(f1: str, f2: str) -> bool:
    return str(f1).startswith(str(f2)) or str(f2).startswith(str(f1))


def _candidate_pairs_from_scores(
    X: np.ndarray, feature_names: Sequence[str], scores: pd.Series, max_pairs: int
) -> list[tuple[int, int]]:
    X = _as_float_matrix(X)
    if X.shape[1] < 2 or max_pairs <= 0:
        return []
    name_to_score = {str(k): float(v) for k, v in scores.items()}

    def usable(i: int) -> bool:
        x = X[:, i]
        return (
            np.isfinite(x).sum() >= 3
            and np.unique(x[np.isfinite(x)]).size >= 2
            and not str(feature_names[i]).startswith("ILR")
        )

    idx = [i for i in range(X.shape[1]) if usable(i)]
    if len(idx) < 2:
        return []
    order = sorted(
        idx,
        key=lambda i: name_to_score.get(
            str(feature_names[i]), float(np.nanvar(X[:, i]))
        ),
        reverse=True,
    )

    frontier = order[
        : min(len(order), max(4, int(math.ceil(math.sqrt(max_pairs * 2))) + 6))
    ]
    cand: list[tuple[float, int, int]] = []
    for a, i in enumerate(frontier):
        for j in frontier[a + 1 :]:
            if _same_lineage_pair(str(feature_names[i]), str(feature_names[j])):
                continue
            si = name_to_score.get(str(feature_names[i]), float(np.nanvar(X[:, i])))
            sj = name_to_score.get(str(feature_names[j]), float(np.nanvar(X[:, j])))
            cand.append((float(si + sj), i, j))
    cand.sort(key=lambda z: z[0], reverse=True)
    return [(i, j) for _, i, j in cand[:max_pairs]]


def _interp_unique(x: np.ndarray, y: np.ndarray, xp: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        raise ValueError("not enough finite support")
    ser = pd.Series(y, index=x).groupby(level=0).mean().sort_index()
    if len(ser) < 2:
        raise ValueError("not enough unique support")
    return np.interp(xp, ser.index.to_numpy(dtype=float), ser.to_numpy(dtype=float))


def _ale_interactions(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_index: int,
    *,
    spec: ALEInteractions,
    scores: pd.Series,
    progress_callback: Callable[[int, int, str], None] | None = None,
    input_projector: Any | None = None,
) -> dict[str, pd.DataFrame]:
    pyale = _require_pyale()
    X = _as_float_matrix(X)
    df_X = pd.DataFrame(X, columns=list(feature_names))
    c = int(class_index)
    predict_class = _explain_predict_class_probability(
        clf, np.arange(len(class_labels), dtype=int), c
    )
    wrapper = _AleModelWrapper(
        lambda values, fn=predict_class: fn(_project_input(values, input_projector))
    )
    n_bins = _auto_ale_bins(X.shape[0], spec)
    pairs = _candidate_pairs_from_scores(
        X, feature_names, scores, max_pairs=int(spec.top_k)
    )
    if not pairs:
        raise ExplainabilityConfigurationError(
            "Interactions were requested, but no usable feature pairs were available."
        )
    raw_rows = []
    total = len(pairs)
    if progress_callback is not None:
        progress_callback(0, total, f"0/{total} pairs")
    for pair_no, (i, j) in enumerate(pairs, start=1):
        f1, f2 = str(feature_names[i]), str(feature_names[j])
        try:
            try:
                with _quiet_pyale_info():
                    result2d = pyale(
                        df_X,
                        wrapper,
                        [f1, f2],
                        grid_size=int(n_bins),
                        include_CI=False,
                        plot=False,
                    )
                vals2d, x1, x2 = _ale_result_values(result2d)
            except Exception:
                continue
            if vals2d.size == 0:
                continue
            centred = vals2d - float(np.mean(vals2d))
            current = float(np.sqrt(np.mean(centred**2)))
            corrected = float("nan")
            correction_applied = False
            correction_error = ""
            if x1 is None or x2 is None:
                correction_error = "2D PyALE result does not expose coordinates for marginal correction"
            else:
                try:
                    with _quiet_pyale_info():
                        res1 = pyale(
                            df_X,
                            wrapper,
                            [f1],
                            grid_size=int(n_bins),
                            include_CI=False,
                            plot=False,
                        )
                        res2 = pyale(
                            df_X,
                            wrapper,
                            [f2],
                            grid_size=int(n_bins),
                            include_CI=False,
                            plot=False,
                        )
                    vals1, g1, _ = _ale_result_values(res1)
                    vals2, g2, _ = _ale_result_values(res2)
                    if g1 is None or g2 is None:
                        raise ValueError("1D PyALE result does not expose grids")
                    comp = (
                        vals2d
                        - _interp_unique(g1, vals1, x1)
                        - _interp_unique(g2, vals2, x2)
                    )
                    comp = comp - float(np.mean(comp))
                    corrected = float(np.sqrt(np.mean(comp**2)))
                    correction_applied = True
                except Exception as exc:
                    correction_error = str(exc)[:240]
            raw_rows.append(
                {
                    "class_index": c,
                    "class_label": str(class_labels[c]),
                    "feature_1": f1,
                    "feature_2": f2,
                    "current": current,
                    "corrected": corrected,
                    "fixed_pairs": current,
                    "corrected_fixed": corrected,
                    "n_ale_cells": int(vals2d.size),
                    "n_bins": int(n_bins),
                    "correction_applied": bool(correction_applied),
                    "correction_error": correction_error,
                    "perturbation_projection": "fitted_model_input_geometry"
                    if input_projector is not None
                    else "none",
                }
            )
        finally:
            if progress_callback is not None:
                progress_callback(
                    pair_no, total, f"pair {pair_no}/{total} · {f1} × {f2}"
                )
    if not raw_rows:
        raise ExplainabilityConfigurationError(
            "ALE interactions were requested, but no candidate pair produced a finite 2D ALE surface."
        )
    raw = pd.DataFrame(raw_rows)
    out: dict[str, pd.DataFrame] = {}
    for method in ("current", "corrected", "fixed_pairs", "corrected_fixed"):
        d = raw[
            [
                "class_index",
                "class_label",
                "feature_1",
                "feature_2",
                method,
                "n_ale_cells",
                "n_bins",
                "correction_applied",
                "correction_error",
                "perturbation_projection",
            ]
        ].copy()
        d = d.rename(columns={method: "interaction_strength"})
        d["method"] = method
        d = d.sort_values("interaction_strength", ascending=False, na_position="last")
        out[method] = d
    out["all_methods"] = raw
    return out


def _combine_feature_importance(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        return _collapse_duplicate_feature_importance(frames[0]).sort_values(
            ["class_index", "importance_mean", "feature"]
            if "class_index" in frames[0].columns
            else ["importance_mean", "feature"],
            ascending=[True, False, True]
            if "class_index" in frames[0].columns
            else [False, True],
            na_position="last",
        )
    long: list[pd.DataFrame] = []
    method_names: list[str] = []
    labels: dict[int, str] = {}
    for frame in frames:
        collapsed = _collapse_duplicate_feature_importance(frame)
        method = str(collapsed["method"].iloc[0])
        method_names.append(method)
        if {"class_index", "class_label"}.issubset(collapsed.columns):
            for _, row in (
                collapsed[["class_index", "class_label"]].drop_duplicates().iterrows()
            ):
                labels[int(row["class_index"])] = str(row["class_label"])
        support = _rank_support_from_importance(collapsed)
        if support.empty:
            continue
        d = support.rename("rank_support").reset_index()
        d["method"] = method
        d["normalized_rank"] = 1.0 - d["rank_support"]
        long.append(d)
    if not long:
        raise ExplainabilityConfigurationError(
            "No finite feature-importance values were available for consensus aggregation."
        )
    pooled = pd.concat(long, ignore_index=True)
    grouped = pooled.groupby(["class_index", "feature"], as_index=False).agg(
        importance_mean=("rank_support", "mean"),
        importance_sd=("rank_support", "std"),
        mean_rank=("normalized_rank", "mean"),
        n_methods=("method", "nunique"),
    )
    grouped["importance_sd"] = grouped["importance_sd"].fillna(0.0)
    grouped["consensus_score"] = grouped["importance_mean"]
    grouped["consensus_sd"] = grouped["importance_sd"]
    grouped["n_methods_total"] = len(dict.fromkeys(method_names))
    grouped["method_coverage"] = grouped["n_methods"] / grouped["n_methods_total"]
    grouped["method"] = "consensus"
    grouped["scoring"] = "mean_within_method_rank_support_available_methods"
    grouped["class_label"] = (
        grouped["class_index"]
        .map(labels)
        .fillna(grouped["class_index"].map(lambda x: f"class_{int(x)}"))
    )
    grouped = grouped.sort_values(
        ["class_index", "importance_mean", "n_methods", "feature"],
        ascending=[True, False, False, True],
    )
    return grouped[
        [
            "method",
            "class_index",
            "class_label",
            "feature",
            "importance_mean",
            "importance_sd",
            "consensus_score",
            "consensus_sd",
            "scoring",
            "mean_rank",
            "n_methods",
            "n_methods_total",
            "method_coverage",
        ]
    ]


def _single_method_support_table(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    frame = _collapse_duplicate_feature_importance(frame)
    if "class_index" not in frame.columns:
        frame["class_index"] = 0
        frame["class_label"] = "class_0"
    method = _method_display(str(frame["method"].iloc[0]))
    support = _rank_support_from_importance(frame, top_k=top_k)
    pieces: list[pd.DataFrame] = []
    for class_index, sub in frame.groupby("class_index", sort=True):
        top = (
            sub.sort_values("importance_mean", ascending=False, na_position="last")
            .head(int(top_k))
            .copy()
        )
        top["rank"] = np.arange(1, len(top) + 1, dtype=int)
        keys = pd.MultiIndex.from_arrays(
            [
                np.full(len(top), int(class_index), dtype=int),
                top["feature"].astype(str).to_numpy(),
            ]
        )
        top[method] = support.reindex(keys).to_numpy(dtype=float)
        top["consensus"] = top[method].astype(float)
        pieces.append(top)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    keep = [
        "class_index",
        "class_label",
        "rank",
        "feature",
        method,
        "consensus",
        "importance_mean",
        "importance_sd",
        "mean_rank",
        "median_rank",
        "rank_iqr",
        "top_k_frequency",
        "fold_coverage",
        "sign_consistency",
    ]
    return out[[c for c in keep if c in out.columns]]


def _write_fold_feature_importance(
    method: str, frames: Sequence[pd.DataFrame], target_dir: Path
) -> tuple[Path, pd.DataFrame]:
    prepared: list[pd.DataFrame] = []
    for fold_no, frame in enumerate(frames, start=1):
        d = frame.copy()
        d.attrs = {}
        d["method"] = str(method)
        if "fold_no" not in d.columns:
            d["fold_no"] = int(fold_no)
        if "fold_key" not in d.columns:
            d["fold_key"] = str(fold_no)
        prepared.append(d)
    out = pd.concat(prepared, ignore_index=True) if prepared else pd.DataFrame()
    path = target_dir / f"feature_importance_{method}_by_outer_fold.parquet"
    write_table(path, out)
    return path, out


def _write_method_outputs(
    frame: pd.DataFrame,
    *,
    target_dir: Path,
    figures_dir: Path,
    X_base: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_labels: list[str],
    top_k: int,
) -> dict[str, Path]:
    method = str(frame["method"].iloc[0]).strip().lower()
    table_path = target_dir / f"feature_importance_{method}.parquet"
    write_table(table_path, frame)
    stability_columns = [
        "method",
        "class_index",
        "class_label",
        "feature",
        "importance_mean",
        "importance_sd",
        "importance_median",
        "importance_q25",
        "importance_q75",
        "mean_rank",
        "median_rank",
        "rank_iqr",
        "top_k_frequency",
        "n_estimable_folds",
        "n_outer_folds_total",
        "fold_coverage",
        "signed_importance_mean",
        "sign_positive_fraction",
        "sign_negative_fraction",
        "sign_consistency",
    ]
    stability_path = target_dir / f"feature_stability_{method}.parquet"
    write_table(
        stability_path, frame[[c for c in stability_columns if c in frame.columns]]
    )
    top = _single_method_support_table(frame, top_k=top_k)
    top_path = target_dir / f"top_features_{method}.parquet"
    write_table(top_path, top)
    outputs: dict[str, Path] = {
        f"importance_{method}": table_path,
        f"stability_{method}": stability_path,
        f"top_features_{method}": top_path,
    }
    if top.empty:
        return outputs
    for class_index, class_top in top.groupby("class_index", sort=True):
        c = int(class_index)
        label = str(class_labels[c]) if 0 <= c < len(class_labels) else f"class_{c}"
        slug = _class_slug(label)
        dist = _feature_distribution_stats(
            class_top["feature"].astype(str).tolist(),
            feature_names,
            X_base,
            y,
            class_labels,
        )
        dist_path = target_dir / f"feature_distribution_stats_{method}__{slug}.parquet"
        write_table(dist_path, dist)
        importance_stem = figures_dir / f"feature_importance_{method}__{slug}"
        _plot_feature_importance(
            class_top, dist, importance_stem, top_k, class_labels, frame
        )
        outputs[f"feature_distribution_{method}_{slug}"] = dist_path
        outputs[f"figure_{method}_{slug}"] = importance_stem.with_suffix(".svg")
    return outputs


def _plot_ale_curves(
    curves: pd.DataFrame,
    top_features: Sequence[str],
    out_stem: Path,
    *,
    max_panels: int = 12,
    x_label: str = "Model-input feature value",
    y_label: str = "ALE effect",
) -> None:
    apply_style()
    import math as _math

    import matplotlib.patheffects as mpe
    import matplotlib.pyplot as plt

    from .style import MM

    if (
        curves is not None
        and "grid" not in curves.columns
        and "grid_value" in curves.columns
    ):
        curves = curves.rename(columns={"grid_value": "grid"}).copy()
    if (
        curves is None
        or curves.empty
        or not {"feature", "grid", "ale_effect"}.issubset(curves.columns)
    ):
        fig = plt.figure(figsize=(120 * MM, 40 * MM))
        ax = fig.add_axes([0.06, 0.15, 0.88, 0.70])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "ALE curves unavailable",
            ha="center",
            va="center",
            fontsize=7,
            color=MID,
        )
        save_all(fig, out_stem)
        plt.close(fig)
        return
    order = [
        str(f) for f in top_features if str(f) in set(curves["feature"].astype(str))
    ]
    if not order:
        order = list(dict.fromkeys(curves["feature"].astype(str).tolist()))
    order = order[: max(1, int(max_panels))]
    n = len(order)
    n_cols = 3 if n > 4 else min(2, n)
    n_rows = int(_math.ceil(n / max(n_cols, 1)))
    fig_w = 150 if n_cols == 3 else 112
    fig_h = max(44, 31 * n_rows + 10)
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))
    fig.patch.set_facecolor(BG)
    left, right, bottom, top = 0.075, 0.035, 0.095, 0.055
    gap_x, gap_y = 0.055, 0.115
    cell_w = (1 - left - right - gap_x * (n_cols - 1)) / n_cols
    cell_h = (1 - bottom - top - gap_y * (n_rows - 1)) / n_rows
    for idx, feat in enumerate(order):
        row, col = divmod(idx, n_cols)
        x0 = left + col * (cell_w + gap_x)
        y0 = 1 - top - (row + 1) * cell_h - row * gap_y
        ax = fig.add_axes([x0, y0, cell_w, cell_h])
        d = curves[curves["feature"].astype(str).eq(feat)].copy()
        d["grid"] = pd.to_numeric(d["grid"], errors="coerce")
        d["ale_effect"] = pd.to_numeric(d["ale_effect"], errors="coerce")
        d = (
            d.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["grid", "ale_effect"])
            .sort_values("grid")
        )
        ax.set_facecolor(BG)
        if d.empty:
            ax.text(
                0.5,
                0.5,
                "unavailable",
                ha="center",
                va="center",
                fontsize=5.5,
                color=MID,
                transform=ax.transAxes,
            )
        else:
            fold_col = next(
                (c for c in ("fold_key", "fold_no", "outer_fold") if c in d.columns),
                None,
            )
            grouped = (
                list(d.groupby(fold_col, sort=False, dropna=False))
                if fold_col is not None
                else [("all", d)]
            )
            fold_curves: list[tuple[np.ndarray, np.ndarray]] = []
            ax.axhline(0, color=MID, lw=0.45, ls=(0, (2.2, 2.2)), alpha=0.55, zorder=1)
            for _, fold_frame in grouped:
                fold_frame = (
                    fold_frame.groupby("grid", as_index=False)["ale_effect"]
                    .mean()
                    .sort_values("grid")
                )
                x_fold = fold_frame["grid"].to_numpy(float)
                y_fold = fold_frame["ale_effect"].to_numpy(float)
                finite = np.isfinite(x_fold) & np.isfinite(y_fold)
                x_fold = x_fold[finite]
                y_fold = y_fold[finite]
                if not len(x_fold):
                    continue
                fold_curves.append((x_fold, y_fold))
                if len(grouped) > 1:
                    ax.plot(
                        x_fold,
                        y_fold,
                        color=TRACK,
                        lw=0.55,
                        alpha=0.85,
                        zorder=2,
                        solid_capstyle="round",
                    )
            if len(fold_curves) == 1:
                x, y = fold_curves[0]
                ax.plot(
                    x,
                    y,
                    color=ACC_D,
                    lw=0.95,
                    zorder=4,
                    solid_capstyle="round",
                )
            elif len(fold_curves) > 1:
                lower = max(float(np.min(x)) for x, _ in fold_curves)
                upper = min(float(np.max(x)) for x, _ in fold_curves)
                if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                    grid_size = max(
                        32, min(128, max(len(x) for x, _ in fold_curves) * 4)
                    )
                    x_common = np.linspace(lower, upper, grid_size)
                    stack = np.empty((len(fold_curves), grid_size), dtype=float)
                    for fold_index, (x_fold, y_fold) in enumerate(fold_curves):
                        stack[fold_index] = np.interp(x_common, x_fold, y_fold)
                    median = np.median(stack, axis=0)
                    q25 = np.quantile(stack, 0.25, axis=0)
                    q75 = np.quantile(stack, 0.75, axis=0)
                    ax.fill_between(
                        x_common,
                        q25,
                        q75,
                        color=ACC_L,
                        alpha=0.52,
                        zorder=3,
                        linewidth=0,
                    )
                    ax.plot(
                        x_common,
                        median,
                        color=ACC_D,
                        lw=1.0,
                        zorder=4,
                        solid_capstyle="round",
                    )
        ax.text(
            0.0,
            1.055,
            _plain_taxon_label(feat, 46),
            ha="left",
            va="bottom",
            fontsize=5.2,
            color=INK,
            transform=ax.transAxes,
            path_effects=[mpe.withStroke(linewidth=1.0, foreground="white")],
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#000000")
            ax.spines[side].set_linewidth(0.45)
        ax.tick_params(
            axis="x",
            length=2.0,
            width=0.4,
            labelsize=5.0,
            pad=1,
            color="#000000",
            labelcolor="#000000",
        )
        ax.tick_params(
            axis="y",
            length=2.0,
            width=0.4,
            labelsize=5.0,
            pad=1,
            color="#000000",
            labelcolor="#000000",
        )
        ax.set_xlabel(str(x_label), fontsize=5.0, color="#000000", labelpad=2)
        ax.set_ylabel(str(y_label), fontsize=5.0, color="#000000", labelpad=2)
        try:
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
        except Exception:
            pass
    save_all(fig, out_stem)
    plt.close(fig)
