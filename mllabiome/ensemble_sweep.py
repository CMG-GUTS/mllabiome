from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .compute import ResourceTracker
from .ensemble_progress import EnsembleSearchProgress
from .configs_sweep import Ensemble, Sweep, sweep_task
from .console import path_table, stage, success, summary_table
from .data import load_dataset
from .ensemble_aggregation import (
    SUPPORTED_AGGREGATIONS,
    aggregate_member_predictions,
    effective_aggregation_weights,
)
from .metrics import (
    _renormalize_proba,
    compute_metrics,
    metric_better as _metric_better,
    metric_is_loss as _metric_is_loss,
)
from .mpma_e_figure import write_single_task_mpma_e_figure
from .selection import select_final_mpma_candidate
from .storage import read_table, write_table, table_exists
from .utils import TAXONOMIC_LEVELS, dump_json_standard


_ENSEMBLE_SEARCH_SCHEMA = "mpmae_search_v3"
_SUPER_LEARNER_WEIGHT_TOL = 1e-8
_SUPER_LEARNER_OPT_MAXITER = 1000
_SUPER_LEARNER_OPT_FTOL = 1e-12
_CARUANA_IMPROVEMENT_TOL = 1e-10


def _proba_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("proba_")]


def _excluded_config_ids(configs: pd.DataFrame, ensemble: Ensemble) -> set[str]:
    if configs is None or configs.empty or "config_id" not in configs.columns:
        return set()
    mask = pd.Series(False, index=configs.index, dtype=bool)
    ids = {str(x) for x in getattr(ensemble, "exclude_config_ids", ()) if str(x)}
    if ids:
        mask |= configs["config_id"].astype(str).isin(ids)

    def match(column: str, values: tuple[str, ...]) -> None:
        nonlocal mask
        vals = {str(x).casefold() for x in values if str(x)}
        if vals and column in configs.columns:
            mask |= configs[column].astype(str).str.casefold().isin(vals)

    match("learner", tuple(getattr(ensemble, "exclude_learners", ())))
    match("resolution", tuple(getattr(ensemble, "exclude_resolutions", ())))
    match(
        "count_transformation", tuple(getattr(ensemble, "exclude_transformations", ()))
    )
    return set(configs.loc[mask, "config_id"].astype(str))


def _eligible_config_ids(configs: pd.DataFrame, ensemble: Ensemble) -> set[str]:
    if configs is None or configs.empty or "config_id" not in configs.columns:
        return set()
    frame = configs.copy()
    if (
        not bool(getattr(ensemble, "include_inactive", False))
        and "active" in frame.columns
    ):
        active = pd.to_numeric(frame["active"], errors="coerce").fillna(0).astype(int)
        frame = frame[active.eq(1)]
    excluded = _excluded_config_ids(frame, ensemble)
    return set(frame["config_id"].astype(str)) - excluded


def _plan_methods(plan: Ensemble) -> tuple[str, ...]:
    value = getattr(plan, "methods", None)
    if value is None:
        value = getattr(plan, "selection_strategies", ())
    return tuple(str(x) for x in value)


def _plan_aggregations(plan: Ensemble) -> tuple[str, ...]:
    return tuple(str(x) for x in getattr(plan, "aggregation_strategies", ()))


def _plan_max_sizes(plan: Ensemble) -> tuple[int, ...]:
    legacy = getattr(plan, "sizes", None)
    canonical = tuple(int(x) for x in getattr(plan, "max_sizes", ()))
    if legacy is not None:
        legacy_sizes = tuple(int(x) for x in legacy)

        if canonical and canonical != (3,) and canonical != legacy_sizes:
            raise ValueError(
                "Ensemble.max_sizes and legacy Ensemble.sizes disagree. Use only max_sizes."
            )
        return legacy_sizes
    return canonical


def _resolved_super_learner_loss(plan: Ensemble) -> str:
    metric = str(plan.optimize_metric).strip().casefold()
    if metric in {"brier", "brier_loss"}:
        return "brier"
    return "log_loss"


def _caruana_controls(max_members: int) -> dict[str, int | float]:
    max_members = int(max_members)

    ceiling = min(1000, max(100, 25 * max_members))
    patience = max(20, 5 * max_members)
    minimum = min(ceiling, max(20, 5 * max_members))
    return {
        "max_iterations": int(ceiling),
        "patience": int(patience),
        "min_iterations": int(minimum),
        "improvement_tol": float(_CARUANA_IMPROVEMENT_TOL),
    }


def _validate_plan(plan: Ensemble) -> None:
    supported_selection = {
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    }
    methods = _plan_methods(plan)
    if not methods:
        raise ValueError(
            "Ensemble.selection_strategies must contain at least one supported method."
        )
    unknown = sorted(set(methods) - supported_selection)
    if unknown:
        raise ValueError(
            f"Unknown ensemble selection strategy(s): {unknown}. "
            f"Supported strategies: {sorted(supported_selection)}."
        )

    aggregations = _plan_aggregations(plan)
    if not aggregations:
        raise ValueError("Ensemble.aggregation_strategies must be non-empty.")
    unknown_aggregations = sorted(set(aggregations) - set(SUPPORTED_AGGREGATIONS))
    if unknown_aggregations:
        raise ValueError(
            f"Unknown ensemble aggregation strategy(s): {unknown_aggregations}. "
            f"Supported strategies: {list(SUPPORTED_AGGREGATIONS)}."
        )

    max_sizes = _plan_max_sizes(plan)
    if not max_sizes:
        raise ValueError("Ensemble.max_sizes must be non-empty.")
    if any(int(x) < 2 for x in max_sizes):
        raise ValueError("Every Ensemble.max_sizes entry must be at least 2.")

    learned = set(methods) & {"caruana", "super_learner"}
    simple = set(methods) - {"caruana", "super_learner"}
    if learned and "weighted_mean_proba" not in aggregations:
        raise ValueError(
            "Caruana and Super Learner are learned-weight ensemble selectors and require "
            "aggregation_strategies to include 'weighted_mean_proba'. Add that aggregation "
            "or remove those selection strategies."
        )
    nonweighted = [x for x in aggregations if x != "weighted_mean_proba"]
    if simple and not nonweighted:
        raise ValueError(
            "Simple ensemble selectors do not learn member weights, so they require at least "
            "one non-weighted aggregation strategy (for example 'mean_proba')."
        )


def _ensemble_configs(plan: Ensemble) -> list[dict[str, Any]]:
    _validate_plan(plan)
    rows: list[dict[str, Any]] = []
    learned_weight_selectors = {"caruana", "super_learner"}
    for method in _plan_methods(plan):
        for max_size in _plan_max_sizes(plan):
            for aggregation in _plan_aggregations(plan):
                if method in learned_weight_selectors:
                    if aggregation != "weighted_mean_proba":
                        continue
                elif aggregation == "weighted_mean_proba":
                    continue
                uid = "__".join(
                    [
                        _ENSEMBLE_SEARCH_SCHEMA,
                        str(plan.optimize_metric),
                        method,
                        aggregation,
                        str(max_size),
                        _resolved_super_learner_loss(plan)
                        if method == "super_learner"
                        else "na",
                    ]
                )
                rows.append(
                    {
                        "ensemble_config_id": hashlib.sha1(uid.encode()).hexdigest()[
                            :12
                        ],
                        "search_schema": _ENSEMBLE_SEARCH_SCHEMA,
                        "optimize_metric": str(plan.optimize_metric),
                        "selection_strategy": method,
                        "aggregation_strategy": aggregation,
                        "max_size": int(max_size),
                    }
                )
    if not rows:
        raise ValueError(
            "The requested selection × aggregation search space has no valid combinations."
        )
    return rows


def _prediction_frame_for_outer(
    predictions: pd.DataFrame, outer_split_key: str | None
) -> pd.DataFrame:
    frame = predictions.copy()
    if "outer_split_key" not in frame.columns:
        if "split_key" not in frame.columns:
            raise ValueError(
                "Prediction table must contain outer_split_key or split_key."
            )
        frame["outer_split_key"] = frame["split_key"]
    if outer_split_key is not None:
        frame = frame[frame["outer_split_key"].astype(str).eq(str(outer_split_key))]
    frame["outer_split_key"] = frame["outer_split_key"].astype(str)
    frame["config_id"] = frame["config_id"].astype(str)
    return frame


def _complete_inner_scores(
    inner: pd.DataFrame,
    predictions: pd.DataFrame,
    outer_split_key: str | None,
    metric: str,
    eligible_ids: set[str],
) -> pd.Series:
    required = {"inner_key", "config_id"}
    missing = required - set(inner.columns)
    if missing:
        raise ValueError(
            f"Inner-results table is missing required columns: {sorted(missing)}"
        )
    frame = inner.copy()
    if outer_split_key is not None:
        if "split_key" not in frame.columns:
            raise ValueError("Inner-results table must contain split_key.")
        frame = frame[frame["split_key"].astype(str).eq(str(outer_split_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    frame["inner_key"] = frame["inner_key"].astype(str)
    frame = frame[frame["config_id"].isin(eligible_ids)]
    if "ok" in frame.columns:
        ok = pd.to_numeric(frame["ok"], errors="coerce").fillna(0).astype(int)
        frame = frame[ok.eq(1)]
    expected = set(frame["inner_key"].dropna().astype(str))
    if not expected:
        return pd.Series(dtype=float)
    complete: list[str] = []
    for config_id, group in frame.groupby("config_id", sort=True):
        group = group.drop_duplicates("inner_key", keep="last")
        if set(group["inner_key"].astype(str)) == expected:
            complete.append(str(config_id))
    if not complete:
        return pd.Series(dtype=float)
    pred = _prediction_frame_for_outer(predictions, outer_split_key)
    pred = pred[pred["config_id"].isin(set(complete))].copy()
    pcols = _proba_cols(pred)
    if not pcols:
        raise ValueError("Inner predictions do not contain probability columns.")
    rows: list[tuple[str, float]] = []
    for config_id in complete:
        ordered, stack = _aligned_stack(pred, [config_id], pcols, inner=True)
        if ordered is None or stack is None:
            continue
        y_true = ordered["y_true"].to_numpy(dtype=int)
        value = _metric_value(y_true, stack[0], metric)
        if np.isfinite(value):
            rows.append((str(config_id), float(value)))
    if not rows:
        return pd.Series(dtype=float)
    scores = pd.Series(dict(rows), dtype=float)
    if _metric_is_loss(metric):
        order = sorted(scores.index, key=lambda cid: (float(scores[cid]), str(cid)))
    else:
        order = sorted(scores.index, key=lambda cid: (-float(scores[cid]), str(cid)))
    return scores.loc[order]


def _aligned_stack(
    predictions: pd.DataFrame, members: list[str], pcols: list[str], *, inner: bool
) -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    if not members:
        return (None, None)
    if inner:
        required = {"outer_split_key", "split_key", "sample_id", "y_true", "config_id"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(
                f"Inner prediction table is missing required columns: {sorted(missing)}"
            )
        key_cols = ["outer_split_key", "split_key", "sample_id"]
    else:
        required = {"sample_id", "y_true", "config_id"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(
                f"Outer prediction table is missing required columns: {sorted(missing)}"
            )
        key_cols = ["sample_id"]
    label_counts = predictions.groupby(key_cols, dropna=False)["y_true"].nunique()
    if (label_counts > 1).any():
        raise ValueError(
            "Prediction table contains conflicting y_true values for a sample."
        )
    ordered = (
        predictions[key_cols + ["y_true"]]
        .drop_duplicates(key_cols, keep="last")
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    stacks: list[np.ndarray] = []
    for cid in members:
        sub = predictions[predictions["config_id"].eq(str(cid))].copy()
        sub = sub.drop_duplicates(key_cols, keep="last")
        sub = ordered[key_cols].merge(
            sub[key_cols + pcols], on=key_cols, how="left", validate="one_to_one"
        )
        if sub[pcols].isna().any().any():
            return (None, None)
        stacks.append(sub[pcols].to_numpy(dtype=float))
    if len(stacks) != len(members):
        return (None, None)
    return (ordered, np.stack(stacks, axis=0))


def _missing_members(
    predictions: pd.DataFrame, members: list[str], pcols: list[str], *, inner: bool
) -> list[str]:
    missing: list[str] = []
    for cid in members:
        ordered, stack = _aligned_stack(predictions, [cid], pcols, inner=inner)
        if ordered is None or stack is None:
            missing.append(str(cid))
    return missing


def _metric_value(y_true: np.ndarray, proba: np.ndarray, metric: str) -> float:
    proba = _renormalize_proba(
        np.asarray(proba, dtype=float), np.asarray(proba).shape[1]
    )
    if metric == "log_loss":
        eps = np.finfo(float).eps
        picked = np.clip(
            proba[np.arange(len(y_true)), np.asarray(y_true, dtype=int)], eps, 1.0
        )
        value = float(-np.mean(np.log(picked)))
        return value if np.isfinite(value) else float("inf")
    if metric in {"brier", "brier_loss"}:
        target = np.zeros_like(proba)
        target[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0
        value = float(np.mean(np.sum((proba - target) ** 2, axis=1)))
        return value if np.isfinite(value) else float("inf")
    classes = np.arange(proba.shape[1], dtype=int)
    y_pred = classes[proba.argmax(axis=1)]
    values = compute_metrics(y_true, y_pred, proba, classes)
    if metric not in values:
        raise ValueError(f"Unsupported ensemble optimize_metric={metric!r}.")
    value = float(values[metric])
    return value if np.isfinite(value) else float("-inf")


def _weighted_probability_mean(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return aggregate_member_predictions(stack, "weighted_mean_proba", weights)


def _uniform_weights(n_members: int) -> np.ndarray:
    if n_members <= 0:
        raise ValueError("Cannot create weights for an empty ensemble.")
    return np.full(n_members, 1.0 / n_members, dtype=float)


def _pad_degenerate_ensemble(
    members: list[str], weights: np.ndarray, library: list[str]
) -> tuple[list[str], np.ndarray]:
    if len(members) >= 2:
        return members, weights
    if not members:
        raise ValueError("Cannot pad an empty learned ensemble.")
    for candidate in library:
        if candidate not in members:
            padded_members = [*members, candidate]
            padded_weights = np.asarray([float(weights[0]), 0.0], dtype=float)
            return padded_members, padded_weights
    return members, weights


def _select_top_k(scores: pd.Series, size: int) -> list[str]:
    return [str(x) for x in scores.index[: int(size)].tolist()]


def _evaluate_selected_members(
    spec: dict[str, Any],
    members: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    *,
    native_weights: np.ndarray | None = None,
    native_weight_source: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    aggregation = str(spec["aggregation_strategy"])
    active_weights: np.ndarray | None = None
    aggregation_weight_source = "not_applicable"
    if aggregation == "weighted_mean_proba":
        if native_weights is None:
            raise ValueError(
                f"{spec['selection_strategy']} does not define weights required by weighted_mean_proba."
            )
        active_weights = np.asarray(native_weights, dtype=float)
        aggregation_weight_source = native_weight_source or "learned"
    elif aggregation == "mean_proba":
        active_weights = effective_aggregation_weights(aggregation, len(members))
        aggregation_weight_source = "uniform"

    proba = aggregate_member_predictions(stack, aggregation, active_weights)
    classes = np.arange(proba.shape[1], dtype=int)
    metrics = compute_metrics(y_true, classes[proba.argmax(axis=1)], proba, classes)
    metrics[metric] = _metric_value(y_true, proba, metric)

    stored_weights = (
        [float(x) for x in active_weights] if active_weights is not None else []
    )
    selection_weights = (
        [float(x) for x in np.asarray(native_weights, dtype=float)]
        if native_weights is not None
        else []
    )
    result: dict[str, Any] = {
        **spec,
        "members": json.dumps(members),
        "weights": json.dumps(stored_weights),
        "selection_weights": json.dumps(selection_weights),
        "member_count": int(len(members)),
        "ensemble_size": int(len(members)),
        "effective_member_count": int(
            np.count_nonzero(np.asarray(stored_weights, dtype=float) > 1e-12)
        )
        if stored_weights
        else int(len(members)),
        "aggregation_weight_source": aggregation_weight_source,
        "weight_source": aggregation_weight_source,
        **{f"{key}_mean": float(value) for key, value in metrics.items()},
    }
    if extra:
        result.update(extra)
    return result


def _select_best_per_resolution(scores: pd.Series, configs: pd.DataFrame) -> list[str]:
    meta = configs.drop_duplicates("config_id").copy()
    meta["config_id"] = meta["config_id"].astype(str)
    meta = meta.set_index("config_id")
    out: list[str] = []
    seen: set[str] = set()
    for cid in [str(x) for x in scores.index.tolist()]:
        if cid not in meta.index:
            continue
        resolution = str(meta.loc[cid, "resolution"])
        if resolution in seen:
            continue
        out.append(cid)
        seen.add(resolution)
    return out


def _learner_type(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    compact = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    rules = (
        (("baseline_rf", "randomforest", "random_forest", "rf"), "random_forest"),
        (("extratrees", "extra_trees", "et"), "extra_trees"),
        (
            ("histgradientboosting", "hist_gradient_boosting", "histgb"),
            "hist_gradient_boosting",
        ),
        (("xgboost", "xgb"), "xgboost"),
        (("lightgbm", "lgbm", "lgb"), "lightgbm"),
        (("catboost", "cb", "cat"), "catboost"),
        (
            ("logisticregression", "logistic_regression", "logistic", "lr"),
            "logistic_regression",
        ),
        (("ridgeclassifier", "ridge_classifier", "ridge"), "ridge_classifier"),
        (("calib_lsvc", "linearsvc", "linear_svc", "lsvc"), "linear_svm"),
        (("svc_rbf", "svm_rbf", "svc"), "rbf_svm"),
        (("kneighbors", "k_neighbors", "knn"), "knn"),
        (("nearestcentroid", "nearest_centroid"), "nearest_centroid"),
        (("gaussiannb", "gaussian_nb", "gnb"), "gaussian_nb"),
        (("bernoullinb", "bernoulli_nb", "bnb"), "bernoulli_nb"),
        (("multinomialnb", "multinomial_nb", "mnb"), "multinomial_nb"),
        (("lda", "lineardiscriminant"), "lda"),
        (("qda", "quadraticdiscriminant"), "qda"),
        (("sgd_log", "sgdclassifier", "sgd_classifier", "sgd"), "sgd_classifier"),
        (("passiveaggressive", "passive_aggressive", "pa"), "passive_aggressive"),
        (("decisiontree", "decision_tree", "dt"), "decision_tree"),
        (("flaml", "automl"), "flaml"),
        (("siamcat",), "siamcat"),
    )
    for prefixes, learner_type in rules:
        if compact in prefixes or any(
            compact.startswith(prefix + "_") for prefix in prefixes
        ):
            return learner_type
    return compact.split("_", 1)[0] or compact


def _select_best_per_learner_type(
    scores: pd.Series, configs: pd.DataFrame
) -> list[str]:
    meta = configs.drop_duplicates("config_id").copy()
    meta["config_id"] = meta["config_id"].astype(str)
    meta = meta.set_index("config_id")
    out: list[str] = []
    seen: set[str] = set()
    for cid in [str(x) for x in scores.index.tolist()]:
        if cid not in meta.index:
            continue
        learner_type = _learner_type(str(meta.loc[cid, "learner"]))
        if learner_type in seen:
            continue
        out.append(cid)
        seen.add(learner_type)
    return out


def _caruana_select(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, float, list[float], dict[str, Any]]:
    if not member_ids or stack.shape[0] != len(member_ids):
        raise ValueError("Caruana library and prediction stack are inconsistent.")
    if max_members < 2:
        raise ValueError("Caruana max_members must be at least 2.")

    controls = _caruana_controls(max_members)
    max_iterations = int(controls["max_iterations"])
    patience = int(controls["patience"])
    min_iterations = int(controls["min_iterations"])
    improvement_tol = float(controls["improvement_tol"])

    running_sum = np.zeros_like(stack[0], dtype=float)
    chosen_indices: list[int] = []
    trajectory: list[float] = []
    best_score = float("inf") if _metric_is_loss(metric) else float("-inf")
    best_prefix = 0
    stale_steps = 0
    stop_reason = "iteration_ceiling"

    for step in range(max_iterations):
        distinct = set(chosen_indices)
        candidate_scores: list[tuple[float, float, int, str]] = []
        denom = float(step + 1)
        for j, cid in enumerate(member_ids):
            if j not in distinct and len(distinct) >= int(max_members):
                continue
            proba = _renormalize_proba((running_sum + stack[j]) / denom, stack.shape[2])
            score = _metric_value(y_true, proba, metric)
            eps = np.finfo(float).eps
            picked = np.clip(proba[np.arange(len(y_true)), y_true], eps, 1.0)
            tie_loss = float(-np.mean(np.log(picked)))
            candidate_scores.append((score, tie_loss, j, str(cid)))
        if not candidate_scores:
            stop_reason = "no_candidates"
            break
        if _metric_is_loss(metric):
            candidate_scores.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        else:
            candidate_scores.sort(
                key=lambda item: (-item[0], item[1], item[2], item[3])
            )
        score, _, selected_idx, _ = candidate_scores[0]
        chosen_indices.append(int(selected_idx))
        running_sum = running_sum + stack[selected_idx]
        trajectory.append(float(score))

        improved = best_prefix == 0 or _metric_better(
            float(score), float(best_score), metric, tol=improvement_tol
        )
        if improved:
            best_score = float(score)
            best_prefix = len(chosen_indices)
            stale_steps = 0
        else:
            stale_steps += 1

        if len(chosen_indices) >= min_iterations and stale_steps >= patience:
            stop_reason = "converged_patience"
            break

    executed_iterations = len(chosen_indices)
    chosen_indices = chosen_indices[:best_prefix]
    if not chosen_indices:
        raise RuntimeError("Caruana ensemble selection did not produce a candidate.")
    order: list[int] = []
    counts: dict[int, int] = {}
    for idx in chosen_indices:
        if idx not in counts:
            order.append(idx)
            counts[idx] = 0
        counts[idx] += 1
    members = [member_ids[idx] for idx in order]
    weights = np.asarray([counts[idx] for idx in order], dtype=float)
    weights /= weights.sum()
    diagnostics: dict[str, Any] = {
        "caruana_iterations_executed": int(executed_iterations),
        "caruana_iterations_selected": int(best_prefix),
        "caruana_stop_reason": stop_reason,
        "caruana_internal_iteration_ceiling": int(max_iterations),
        "caruana_internal_patience": int(patience),
        "caruana_internal_min_iterations": int(min_iterations),
        "caruana_internal_improvement_tol": float(improvement_tol),
    }
    return (
        members,
        weights,
        float(best_score),
        trajectory[:best_prefix],
        diagnostics,
    )


def _super_learner_loss(
    weights: np.ndarray, stack: np.ndarray, y_true: np.ndarray, loss: str
) -> float:
    proba = _weighted_probability_mean(stack, weights)
    if loss == "log_loss":
        eps = np.finfo(float).eps
        picked = np.clip(proba[np.arange(len(y_true)), y_true], eps, 1.0)
        return float(-np.mean(np.log(picked)))
    if loss == "brier":
        target = np.zeros_like(proba)
        target[np.arange(len(y_true)), y_true] = 1.0
        return float(np.mean(np.sum((proba - target) ** 2, axis=1)))
    raise ValueError(f"Unknown Super Learner loss: {loss!r}.")


def _fit_super_learner(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    loss: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, float]:
    if not member_ids or stack.shape[0] != len(member_ids):
        raise ValueError("Super Learner library and prediction stack are inconsistent.")
    if max_members < 2:
        raise ValueError("Super Learner max_members must be at least 2.")

    def optimise(local_stack: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        n_members = local_stack.shape[0]
        start = (
            _uniform_weights(n_members) if x0 is None else np.asarray(x0, dtype=float)
        )
        start = np.clip(start, 0.0, None)
        start /= start.sum()
        result = minimize(
            _super_learner_loss,
            start,
            args=(local_stack, y_true, str(loss)),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_members,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={
                "maxiter": _SUPER_LEARNER_OPT_MAXITER,
                "ftol": _SUPER_LEARNER_OPT_FTOL,
                "disp": False,
            },
        )
        if not bool(result.success):
            raise RuntimeError(f"Super Learner optimization failed: {result.message}")
        values = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
        if not np.all(np.isfinite(values)) or values.sum() <= 0.0:
            raise RuntimeError("Super Learner returned invalid ensemble weights.")
        return values / values.sum()

    full_weights = optimise(stack)
    keep = np.flatnonzero(full_weights > _SUPER_LEARNER_WEIGHT_TOL).tolist()

    if len(keep) < 2:
        order = sorted(
            range(len(member_ids)),
            key=lambda i: (-float(full_weights[i]), str(member_ids[i])),
        )
        keep = order[: min(2, len(order))]
    if len(keep) > int(max_members):
        keep = sorted(
            keep,
            key=lambda i: (-float(full_weights[i]), str(member_ids[i])),
        )[: int(max_members)]
    keep = sorted(keep)
    kept_stack = stack[keep]
    initial = full_weights[keep]
    kept_weights = optimise(kept_stack, initial)
    kept_ids = [member_ids[i] for i in keep]
    final_loss = _super_learner_loss(kept_weights, kept_stack, y_true, str(loss))
    return (kept_ids, kept_weights, float(final_loss))


def _candidate_from_simple_selector(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    configs: pd.DataFrame,
    pcols: list[str],
    metric: str,
) -> dict[str, Any] | None:
    method = str(spec["selection_strategy"])
    max_size = int(spec["max_size"])
    if method == "top_k":
        members = _select_top_k(scores, max_size)
        if len(members) != max_size:
            return None
    elif method == "best_per_resolution":
        members = _select_best_per_resolution(scores, configs)[:max_size]
    elif method == "best_per_learner_type":
        members = _select_best_per_learner_type(scores, configs)[:max_size]
    else:
        raise ValueError(f"Unsupported simple ensemble selector: {method!r}.")
    if len(members) < 2:
        return None
    ordered, stack = _aligned_stack(predictions, members, pcols, inner=True)
    if ordered is None or stack is None:
        return None
    return _evaluate_selected_members(
        spec,
        members,
        stack,
        ordered["y_true"].to_numpy(dtype=int),
        metric,
        extra={"inner_oof_rows": int(len(ordered)), "selection_weight_source": "none"},
    )


def _candidate_from_caruana(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    metric: str,
) -> dict[str, Any] | None:

    library = [str(x) for x in scores.index.tolist()]
    if len(library) < 2:
        return None
    ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    if ordered is None or stack is None:
        incomplete = set(_missing_members(predictions, library, pcols, inner=True))
        library = [cid for cid in library if cid not in incomplete]
        if len(library) < 2:
            return None
        ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    members, native_weights, _, trajectory, diagnostics = _caruana_select(
        library,
        stack,
        y_true,
        metric,
        int(spec["max_size"]),
    )
    members, native_weights = _pad_degenerate_ensemble(members, native_weights, library)
    if len(members) < 2:
        return None
    if (
        str(spec["aggregation_strategy"]) == "weighted_mean_proba"
        and int(np.count_nonzero(np.asarray(native_weights) > 1e-12)) < 2
    ):
        return None
    selected_indices = [library.index(cid) for cid in members]
    selected_stack = stack[selected_indices]
    return _evaluate_selected_members(
        spec,
        members,
        selected_stack,
        y_true,
        metric,
        native_weights=native_weights,
        native_weight_source="caruana_selection_frequency",
        extra={
            "inner_oof_rows": int(len(ordered)),
            "candidate_library_policy": "all_complete_inner_oof_mpmas",
            "candidate_library_size": int(len(library)),
            "selection_weight_source": "caruana_selection_frequency",
            "caruana_trajectory": json.dumps([float(x) for x in trajectory]),
            **diagnostics,
        },
    )


def _candidate_from_super_learner(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    plan: Ensemble,
    metric: str,
) -> dict[str, Any] | None:

    library = [str(x) for x in scores.index.tolist()]
    if len(library) < 2:
        return None
    ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    if ordered is None or stack is None:
        incomplete = set(_missing_members(predictions, library, pcols, inner=True))
        library = [cid for cid in library if cid not in incomplete]
        if len(library) < 2:
            return None
        ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    super_loss = _resolved_super_learner_loss(plan)
    members, native_weights, loss_value = _fit_super_learner(
        library,
        stack,
        y_true,
        super_loss,
        int(spec["max_size"]),
    )
    members, native_weights = _pad_degenerate_ensemble(members, native_weights, library)
    if len(members) < 2:
        return None
    if (
        str(spec["aggregation_strategy"]) == "weighted_mean_proba"
        and int(np.count_nonzero(np.asarray(native_weights) > 1e-12)) < 2
    ):
        return None
    selected_indices = [library.index(cid) for cid in members]
    selected_stack = stack[selected_indices]
    return _evaluate_selected_members(
        spec,
        members,
        selected_stack,
        y_true,
        metric,
        native_weights=native_weights,
        native_weight_source=f"convex_{super_loss}",
        extra={
            "inner_oof_rows": int(len(ordered)),
            "candidate_library_policy": "all_complete_inner_oof_mpmas",
            "candidate_library_size": int(len(library)),
            "selection_weight_source": f"convex_{super_loss}",
            "super_learner_loss": super_loss,
            "super_learner_loss_rule": "brier_if_optimize_metric_is_brier_else_log_loss",
            "super_learner_loss_value": float(loss_value),
            "super_learner_internal_weight_tol": float(_SUPER_LEARNER_WEIGHT_TOL),
            "super_learner_internal_optimizer": "SLSQP",
            "super_learner_internal_maxiter": int(_SUPER_LEARNER_OPT_MAXITER),
            "super_learner_internal_ftol": float(_SUPER_LEARNER_OPT_FTOL),
        },
    )


def _candidate_table_for_inner(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    outer_split_key: str | None,
    progress_callback=None,
    progress_scope: str | None = None,
) -> pd.DataFrame:
    _validate_plan(plan)
    eligible = _eligible_config_ids(configs, plan)
    scores = _complete_inner_scores(
        inner_results, inner_predictions, outer_split_key, metric, eligible
    )
    if scores.empty:
        return pd.DataFrame()
    pred = _prediction_frame_for_outer(inner_predictions, outer_split_key)
    pred = pred[pred["config_id"].isin(set(scores.index))].copy()
    pcols = _proba_cols(pred)
    if not pcols:
        raise ValueError("Inner predictions do not contain probability columns.")
    rows: list[dict[str, Any]] = []
    specs = _ensemble_configs(plan)
    scope = str(progress_scope or outer_split_key or "__final__")
    for index, spec in enumerate(specs, start=1):
        if progress_callback is not None:
            progress_callback(scope, "start", spec, index, len(specs), "evaluating")
        row = None
        failed = False
        try:
            method = str(spec["selection_strategy"])
            if method in {"top_k", "best_per_resolution", "best_per_learner_type"}:
                row = _candidate_from_simple_selector(
                    spec, scores, pred, configs, pcols, metric
                )
            elif method == "caruana":
                row = _candidate_from_caruana(spec, scores, pred, pcols, metric)
            elif method == "super_learner":
                row = _candidate_from_super_learner(
                    spec, scores, pred, pcols, plan, metric
                )
            else:
                raise ValueError(f"Unknown ensemble method: {method!r}.")
            if row is not None:
                rows.append(row)
        except Exception as exc:
            failed = True
            if progress_callback is not None:
                progress_callback(scope, "error", spec, index, len(specs), str(exc))
            raise
        finally:
            if not failed and progress_callback is not None:
                progress_callback(
                    scope,
                    "done",
                    spec,
                    index,
                    len(specs),
                    "valid" if row is not None else "skipped",
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    score_col = f"{metric}_mean"
    if score_col not in out.columns:
        raise ValueError(
            f"The requested ensemble metric {metric!r} was not produced for candidates."
        )
    out = out[np.isfinite(pd.to_numeric(out[score_col], errors="coerce"))].copy()
    method_priority = {name: i for i, name in enumerate(_plan_methods(plan))}
    aggregation_priority = {name: i for i, name in enumerate(_plan_aggregations(plan))}
    out["_method_priority"] = (
        out["selection_strategy"]
        .astype(str)
        .map(method_priority)
        .fillna(999)
        .astype(int)
    )
    out["_aggregation_priority"] = (
        out["aggregation_strategy"]
        .astype(str)
        .map(aggregation_priority)
        .fillna(999)
        .astype(int)
    )
    out = out.sort_values(
        [
            score_col,
            "_method_priority",
            "_aggregation_priority",
            "member_count",
            "max_size",
            "ensemble_config_id",
        ],
        ascending=[_metric_is_loss(metric), True, True, True, True, True],
        kind="mergesort",
    ).drop(columns=["_method_priority", "_aggregation_priority"])
    return out.reset_index(drop=True)


def _parse_json_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except Exception as exc:
        raise ValueError(f"Could not parse ensemble {field} JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Ensemble {field} must decode to a list.")
    return parsed


def select_mpma_e_by_outer_fold(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    outer_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_plan(plan)
    outer = _prediction_frame_for_outer(outer_predictions, None)
    outer_pcols = _proba_cols(outer)
    if not outer_pcols:
        raise ValueError("Outer predictions do not contain probability columns.")
    if "split_key" not in inner_results.columns:
        raise ValueError("Inner-results table must contain split_key.")
    selections: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    outer_keys = sorted(set(inner_results["split_key"].astype(str)))
    for outer_key in outer_keys:
        candidates = _candidate_table_for_inner(
            inner_results,
            inner_predictions,
            configs,
            plan,
            metric,
            outer_key,
            progress_callback=progress_callback,
            progress_scope=outer_key,
        )
        if candidates.empty:
            raise RuntimeError(
                f"No valid inner-only ensemble candidate was produced for outer fold {outer_key!r}."
            )
        score_col = f"{metric}_mean"
        winner = candidates.iloc[0].to_dict()
        if progress_callback is not None:
            progress_callback(
                outer_key,
                "selected",
                winner,
                len(_ensemble_configs(plan)),
                len(_ensemble_configs(plan)),
                f"inner {metric}={float(winner[score_col]):.4f}",
            )
        members = [str(x) for x in _parse_json_list(winner["members"], "members")]
        weights = [
            float(x) for x in _parse_json_list(winner.get("weights", "[]"), "weights")
        ]
        aggregation = str(winner["aggregation_strategy"])
        if aggregation == "weighted_mean_proba" and len(members) != len(weights):
            raise RuntimeError(
                f"Selected weighted ensemble for {outer_key!r} has inconsistent members/weights."
            )
        active_weights = weights if aggregation == "weighted_mean_proba" else None

        fold_outer = outer[outer["outer_split_key"].astype(str).eq(outer_key)].copy()
        if fold_outer.empty:
            raise RuntimeError(
                f"Outer fold {outer_key!r} has no outer-test predictions. Nested ensemble evaluation cannot silently drop the fold."
            )
        ordered, stack = _aligned_stack(fold_outer, members, outer_pcols, inner=False)
        if ordered is None or stack is None:
            missing = _missing_members(fold_outer, members, outer_pcols, inner=False)
            raise RuntimeError(
                f"Selected ensemble for outer fold {outer_key!r} cannot be evaluated because selected member predictions are missing: {missing}. The ensemble is not re-selected using outer-test availability."
            )
        proba = aggregate_member_predictions(stack, aggregation, active_weights)
        classes = np.arange(proba.shape[1], dtype=int)
        y_true = ordered["y_true"].to_numpy(dtype=int)
        y_pred = classes[proba.argmax(axis=1)]
        outer_metrics = compute_metrics(y_true, y_pred, proba, classes)
        weights_json = json.dumps(weights)
        members_json = json.dumps(members)
        selections.append(
            {
                "outer_split_key": outer_key,
                "ensemble_config_id": str(winner["ensemble_config_id"]),
                "selection_basis": "outer_fold_inner_oof_predictions_only",
                "selection_metric": str(metric),
                "inner_score": float(winner[score_col]),
                "selection_strategy": str(winner["selection_strategy"]),
                "aggregation_strategy": aggregation,
                "max_size": int(winner.get("max_size", len(members))),
                "ensemble_size": int(len(members)),
                "member_count": int(len(members)),
                "effective_member_count": int(
                    winner.get("effective_member_count", len(members))
                ),
                "members": members_json,
                "weights": weights_json,
                "selection_weights": str(winner.get("selection_weights", "[]")),
                "weight_source": str(winner.get("weight_source", "")),
                "aggregation_weight_source": str(
                    winner.get("aggregation_weight_source", "")
                ),
            }
        )
        metric_rows.append(
            {
                "outer_split_key": outer_key,
                "ensemble_config_id": str(winner["ensemble_config_id"]),
                **outer_metrics,
            }
        )
        ordered = ordered.copy()
        ordered["outer_split_key"] = outer_key
        ordered["ensemble_config_id"] = str(winner["ensemble_config_id"])
        ordered["selection_strategy"] = str(winner["selection_strategy"])
        ordered["aggregation_strategy"] = aggregation
        ordered["max_size"] = int(winner.get("max_size", len(members)))
        ordered["members"] = members_json
        ordered["weights"] = weights_json
        ordered["y_pred"] = y_pred.astype(int)
        for j, col in enumerate(outer_pcols):
            ordered[col] = proba[:, j]
        prediction_rows.extend(ordered.to_dict(orient="records"))
    if len(selections) != len(outer_keys):
        raise RuntimeError(
            "MPMA-E did not produce exactly one explicit selection per outer fold."
        )
    return (
        pd.DataFrame(selections),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
    )


def summarize_mpma_e_strategy(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    if fold_metrics is None or fold_metrics.empty:
        return {}
    out: dict[str, Any] = {
        "Strategy": "MPMA-E",
        "selection_basis": "outer_fold_inner_oof_predictions_only",
        "n_outer_folds": int(len(fold_metrics)),
    }
    for metric in fold_metrics.columns:
        if metric in {"outer_split_key", "ensemble_config_id"}:
            continue
        vals = (
            pd.to_numeric(fold_metrics[metric], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if vals.empty:
            continue
        out[f"outer_{metric}_mean"] = float(vals.mean())
        out[f"outer_{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"outer_{metric}_count"] = int(len(vals))
    return out


def select_final_mpma_e_candidate(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    member_score_metric: str | None = None,
    progress_callback=None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = _candidate_table_for_inner(
        inner_results,
        inner_predictions,
        configs,
        plan,
        metric,
        None,
        progress_callback=progress_callback,
        progress_scope="__final__",
    )
    if candidates.empty:
        return ({}, candidates)
    score_col = f"{metric}_mean"
    candidates = candidates.rename(columns={score_col: "inner_score"})
    first = candidates.iloc[0].to_dict()
    if progress_callback is not None:
        progress_callback(
            "__final__",
            "selected",
            first,
            len(_ensemble_configs(plan)),
            len(_ensemble_configs(plan)),
            f"inner {metric}={float(first['inner_score']):.4f}"
            if "inner_score" in first
            else f"inner {metric}={float(first[f'{metric}_mean']):.4f}",
        )
    compatibility_metric = str(member_score_metric or metric)
    best = {
        "ensemble_config_id": str(first["ensemble_config_id"]),
        "selection_basis": "all_inner_oof_predictions_for_final_refit",
        "selection_metric": str(metric),
        "optimize_metric": compatibility_metric,
        "member_score_metric": compatibility_metric,
        "inner_score": float(first["inner_score"]),
        "selection_strategy": str(first["selection_strategy"]),
        "aggregation_strategy": str(first["aggregation_strategy"]),
        "max_size": int(first.get("max_size", first.get("member_count", 0))),
        "ensemble_size": int(first["member_count"]),
        "member_count": int(first["member_count"]),
        "effective_member_count": int(
            first.get("effective_member_count", first["member_count"])
        ),
        "members": _parse_json_list(first["members"], "members"),
        "weights": [
            float(x) for x in _parse_json_list(first.get("weights", "[]"), "weights")
        ],
        "selection_weights": [
            float(x)
            for x in _parse_json_list(
                first.get("selection_weights", "[]"), "selection_weights"
            )
        ],
        "weight_source": str(first.get("weight_source", "")),
        "aggregation_weight_source": str(first.get("aggregation_weight_source", "")),
    }
    for key, value in first.items():
        if key in best or key == "optimize_metric":
            continue
        if key.startswith("outer_"):
            continue
        best[f"inner_{key}"] = value
    return (best, candidates)


def sweep_ensemble(sweep: Sweep) -> dict[str, Path]:
    if sweep_task(sweep) == "regression":
        from .regression_ensemble import sweep_regression_ensemble

        return sweep_regression_ensemble(sweep)
    root = sweep.root()
    ensemble_dir = root / "ensembling"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    outer_path = root / "predictions" / "outer_predictions.parquet"
    inner_result_path = root / "inner_results" / "inner_results.parquet"
    inner_prediction_path = root / "inner_predictions" / "inner_predictions.parquet"
    config_path = root / "configs.parquet"
    required = [outer_path, inner_result_path, inner_prediction_path, config_path]
    if any((not table_exists(path) for path in required)):
        raise FileNotFoundError("Run evaluate(sweep) before sweep_ensemble(sweep).")
    outer_predictions = read_table(outer_path)
    inner_results = read_table(inner_result_path)
    inner_predictions = read_table(inner_prediction_path)
    configs = read_table(config_path)
    metric = str(sweep.ensemble.optimize_metric)
    stage("Ensemble sweep", str(root))
    summary_table(
        "Ensemble search",
        {
            "candidate ensemble configurations": f"{len(_ensemble_configs(sweep.ensemble)):,}",
            "selection strategies": _plan_methods(sweep.ensemble),
            "aggregation strategies": _plan_aggregations(sweep.ensemble),
            "max sizes": _plan_max_sizes(sweep.ensemble),
            "learned-selector library": "all eligible complete inner-OOF MPMAs",
            "Caruana stopping": "automatic convergence (internally recorded)",
            "Super Learner loss": _resolved_super_learner_loss(sweep.ensemble),
            "excluded learners": sweep.ensemble.exclude_learners or "none",
            "excluded resolutions": sweep.ensemble.exclude_resolutions or "none",
            "excluded transformations": sweep.ensemble.exclude_transformations
            or "none",
            "optimize metric": metric,
        },
    )
    resource_tracker = ResourceTracker(
        sample_interval_s=float(
            getattr(sweep.evaluation, "resource_sample_interval_s", 0.1)
        )
    ).start()
    outer_keys = sorted(set(inner_results["split_key"].astype(str)))
    ensemble_specs = _ensemble_configs(sweep.ensemble)
    with EnsembleSearchProgress(
        outer_keys + ["__final__"], len(ensemble_specs)
    ) as live:
        selection, selected_outer_predictions, fold_metrics = (
            select_mpma_e_by_outer_fold(
                inner_results,
                inner_predictions,
                outer_predictions,
                configs,
                sweep.ensemble,
                metric,
                progress_callback=live.update,
            )
        )
        if selection.empty or selected_outer_predictions.empty or fold_metrics.empty:
            raise RuntimeError("No valid nested ensemble selections were produced.")
        member_score_metric = str(sweep.evaluation.optimize_metric)
        if member_score_metric not in inner_results.columns:
            raise ValueError(
                f"inner_results.parquet must contain the base evaluation metric {member_score_metric!r} required for final-model member metadata."
            )
        final_ensemble, candidates = select_final_mpma_e_candidate(
            inner_results,
            inner_predictions,
            configs,
            sweep.ensemble,
            metric,
            member_score_metric,
            progress_callback=live.update,
        )
    if not final_ensemble:
        raise RuntimeError(
            "No valid final ensemble candidate was produced from inner OOF predictions."
        )
    nested_summary = summarize_mpma_e_strategy(fold_metrics)
    final_mpma = select_final_mpma_candidate(
        inner_results,
        configs,
        str(sweep.evaluation.optimize_metric),
        plan=sweep.ensemble,
    )
    selection_path = ensemble_dir / "mpma_e_outer_selection.parquet"
    prediction_path = ensemble_dir / "ensemble_predictions.parquet"
    result_path = ensemble_dir / "mpma_e_outer_results.parquet"
    candidate_path = ensemble_dir / "ensemble_candidate_scores.parquet"
    summary_path = ensemble_dir / "mpma_e_strategy_summary.json"
    final_path = ensemble_dir / "mpma_e_final_candidate.json"
    write_table(selection_path, selection)
    write_table(prediction_path, selected_outer_predictions)
    write_table(result_path, fold_metrics)
    write_table(candidate_path, candidates)
    dump_json_standard(nested_summary, summary_path)
    report_final_ensemble = dict(final_ensemble)
    report_final_ensemble["optimize_metric"] = str(
        final_ensemble.get("selection_metric", metric)
    )
    report_final_ensemble["ensemble_size"] = int(
        final_ensemble.get("member_count", final_ensemble.get("ensemble_size", 0))
    )
    report_final_ensemble["max_size"] = int(
        final_ensemble.get("max_size", report_final_ensemble["ensemble_size"])
    )
    dump_json_standard(report_final_ensemble, final_path)
    selected = {
        "inner_val_best_mpma": final_mpma,
        "inner_val_best_mpmas_ensemble": final_ensemble,
        "nested_mpma_e": nested_summary,
        "terminology": {
            "MPMA-B": "fold-specific single MPMA selected by inner-validation scoring for nested performance; separate final candidate selected from all inner validation for refit",
            "MPMA-E": "fold-specific ensemble learned exclusively from inner out-of-fold predictions; member identities and weights are frozen before outer-test application",
        },
    }
    dump_json_standard(selected, ensemble_dir / "selected_unit.json")
    comparison = pd.DataFrame(
        [
            {
                "unit": "MPMA-B final candidate",
                "config_id": final_mpma.get("config_id", ""),
                "selection_basis": final_mpma.get("selection_basis", ""),
                "inner_score": final_mpma.get("inner_score", np.nan),
                "members": "",
                "weights": "",
            },
            {
                "unit": "MPMA-E final candidate",
                "config_id": final_ensemble.get("ensemble_config_id", ""),
                "selection_basis": final_ensemble.get("selection_basis", ""),
                "inner_score": final_ensemble.get("inner_score", np.nan),
                "members": final_ensemble.get("members", "[]"),
                "weights": final_ensemble.get("weights", "[]"),
            },
        ]
    )
    write_table(ensemble_dir / "final_model_comparison.parquet", comparison)
    resource_path = ensemble_dir / "mpma_e_selection_resources.json"
    dump_json_standard(resource_tracker.stop(), resource_path)
    if getattr(sweep, "uses_modalities", False):
        mpma_e_outputs = {}
    else:
        X_fig, taxa_fig, source_fig = _matrix_for_mpma_e_figure(sweep)
        mpma_e_outputs = write_single_task_mpma_e_figure(
            root,
            task_key=root.name,
            task_title=sweep.title,
            X=X_fig,
            taxa=taxa_fig,
            source=source_fig,
            out_dir=root / "figures",
            out_name="mpma_e",
            include_inactive_configs=True,
            max_members=20,
            seed=sweep.evaluation.random_state,
        )
    success(
        f"Ensemble sweep completed · final candidate={final_ensemble['ensemble_config_id']} · inner {metric}={final_ensemble['inner_score']:.4f}"
    )
    outputs = {
        "ensemble_dir": ensemble_dir,
        "selected_unit": ensemble_dir / "selected_unit.json",
        "comparison": ensemble_dir / "final_model_comparison.parquet",
        "mpma_e_selection": selection_path,
        "mpma_e_predictions": prediction_path,
        "mpma_e_outer_results": result_path,
        "mpma_e_candidates": candidate_path,
        "mpma_e_summary": summary_path,
        "mpma_e_final_candidate": final_path,
        "mpma_e_selection_resources": resource_path,
        **mpma_e_outputs,
    }
    path_table("Ensemble outputs", outputs)
    return outputs


def _matrix_for_mpma_e_figure(sweep: Sweep) -> tuple[np.ndarray, list[str], str]:
    X, feature_names = _raw_input_matrix_for_figure(sweep)
    if X.size == 0 or len(feature_names) == 0:
        raise RuntimeError(
            "No raw abundance matrix is available for MPMA-E visualisation."
        )
    return (X, feature_names, "raw input abundance matrix")


def _raw_input_matrix_for_figure(sweep: Sweep) -> tuple[np.ndarray, list[str]]:
    spec = sweep.data
    abundance_path = Path(spec.abundance_path)
    metadata_path = Path(spec.metadata_path) if spec.metadata_path is not None else None
    fmt = str(spec.format).strip().casefold().replace("-", "_")
    if fmt == "auto":
        fmt = (
            "mllab"
            if abundance_path.suffix.lower() in {".tsv", ".txt"} and metadata_path
            else "wide_csv"
        )
    if fmt in {"mllab", "matrix_tsv", "metaphlan", "metaphlan_tsv", "profile_tsv"}:
        if metadata_path is None:
            raise ValueError("Data.metadata_path is required for mllab TSV input.")
        meta = pd.read_csv(metadata_path, sep=None, engine="python", dtype=str)
        bio = pd.read_csv(abundance_path, sep="\t", index_col=0, low_memory=False)
        bio.index = bio.index.astype(str).str.strip()
        bio.columns = bio.columns.astype(str).str.strip()
        meta[spec.sample_id_col] = meta[spec.sample_id_col].astype(str).str.strip()
        common = [
            sid for sid in meta[spec.sample_id_col].tolist() if sid in set(bio.columns)
        ]
        if not common:
            raise ValueError(
                "No sample IDs overlap between metadata and abundance matrix."
            )
        X = bio[common].T.to_numpy(dtype=np.float32)
        return (X, bio.index.tolist())
    if fmt in {"csv", "wide_csv"}:
        df = pd.read_csv(abundance_path)
        if metadata_path is not None:
            meta = pd.read_csv(metadata_path, sep=None, engine="python")
            if (
                spec.sample_id_col not in df.columns
                or spec.sample_id_col not in meta.columns
            ):
                raise ValueError(
                    f"sample_id_col={spec.sample_id_col!r} must exist in both CSV files."
                )
            df = df.merge(
                meta, on=spec.sample_id_col, how="inner", suffixes=("", "__meta")
            )
        reserved = {spec.sample_id_col, spec.target_col, *(spec.metadata_cols or ())}
        if spec.group_col:
            reserved.add(spec.group_col)
        numeric_cols = [
            c
            for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            raise ValueError(
                "No numeric abundance columns found after excluding metadata columns."
            )
        return (
            df[numeric_cols].to_numpy(dtype=np.float32),
            [str(c) for c in numeric_cols],
        )
    dataset = load_dataset(sweep.data, TAXONOMIC_LEVELS)
    if "all" in dataset.X_by_level:
        return (
            dataset.X_by_level["all"],
            dataset.feature_names_by_level.get("all", []),
        )
    blocks = list(dataset.X_by_level.values())
    names = [name for lv in dataset.feature_names_by_level.values() for name in lv]
    return (np.concatenate(blocks, axis=1), names)
