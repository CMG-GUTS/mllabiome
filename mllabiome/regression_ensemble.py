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
from .configs_sweep import Ensemble, Sweep
from .console import path_table, stage, success, summary_table
from .metrics import compute_regression_metrics, metric_better, metric_is_loss
from .selection import (
    select_final_mpma_candidate,
    select_mpma_b_by_outer_fold,
    selected_mpma_b_outer_predictions,
)
from .storage import read_table, write_table, table_exists
from .utils import dump_json_standard


_REGRESSION_SEARCH_SCHEMA = "mpmae_regression_search_v1"
_WEIGHT_TOL = 1e-8
_OPT_MAXITER = 1000
_OPT_FTOL = 1e-12


def _plan_methods(plan: Ensemble) -> tuple[str, ...]:
    value = getattr(plan, "methods", None)
    if value is None:
        value = getattr(plan, "selection_strategies", ())
    return tuple(str(x) for x in value)


def _plan_max_sizes(plan: Ensemble) -> tuple[int, ...]:
    legacy = getattr(plan, "sizes", None)
    canonical = tuple(int(x) for x in getattr(plan, "max_sizes", ()))
    if legacy is None:
        return canonical
    values = tuple(int(x) for x in legacy)
    if canonical and canonical != (3,) and canonical != values:
        raise ValueError(
            "Ensemble.max_sizes and legacy Ensemble.sizes disagree. Use only max_sizes."
        )
    return values


def _canonical_aggregation(value: str) -> str | None:
    token = str(value).strip().casefold()
    mapping = {
        "mean": "mean_prediction",
        "mean_prediction": "mean_prediction",
        "mean_proba": "mean_prediction",
        "weighted_mean": "weighted_mean_prediction",
        "weighted_mean_prediction": "weighted_mean_prediction",
        "weighted_mean_proba": "weighted_mean_prediction",
        "median": "median_prediction",
        "median_prediction": "median_prediction",
        "median_proba": "median_prediction",
    }
    return mapping.get(token)


def _plan_aggregations(plan: Ensemble) -> tuple[str, ...]:
    out: list[str] = []
    for value in getattr(plan, "aggregation_strategies", ()):
        canonical = _canonical_aggregation(str(value))
        if canonical is not None and canonical not in out:
            out.append(canonical)
    return tuple(out)


def _validate_plan(plan: Ensemble) -> None:
    supported = {
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
    unknown = sorted(set(methods) - supported)
    if unknown:
        raise ValueError(f"Unknown ensemble selection strategy(s): {unknown}.")
    aggregations = _plan_aggregations(plan)
    if not aggregations:
        raise ValueError(
            "Regression ensembling requires mean, weighted mean, or median prediction aggregation."
        )
    sizes = _plan_max_sizes(plan)
    if not sizes or any(int(x) < 2 for x in sizes):
        raise ValueError("Every Ensemble.max_sizes entry must be at least 2.")
    learned = set(methods) & {"caruana", "super_learner"}
    simple = set(methods) - learned
    if learned and "weighted_mean_prediction" not in aggregations:
        raise ValueError(
            "Caruana and Super Learner require weighted mean prediction aggregation for regression."
        )
    if simple and not any(x != "weighted_mean_prediction" for x in aggregations):
        raise ValueError(
            "Simple regression ensemble selectors require mean or median prediction aggregation."
        )


def _ensemble_configs(plan: Ensemble) -> list[dict[str, Any]]:
    _validate_plan(plan)
    rows: list[dict[str, Any]] = []
    for method in _plan_methods(plan):
        for max_size in _plan_max_sizes(plan):
            for aggregation in _plan_aggregations(plan):
                learned = method in {"caruana", "super_learner"}
                if learned and aggregation != "weighted_mean_prediction":
                    continue
                if not learned and aggregation == "weighted_mean_prediction":
                    continue
                raw = "__".join(
                    [
                        _REGRESSION_SEARCH_SCHEMA,
                        str(plan.optimize_metric),
                        method,
                        aggregation,
                        str(max_size),
                    ]
                )
                rows.append(
                    {
                        "ensemble_config_id": hashlib.sha1(raw.encode()).hexdigest()[
                            :12
                        ],
                        "selection_strategy": method,
                        "aggregation_strategy": aggregation,
                        "max_size": int(max_size),
                        "optimize_metric": str(plan.optimize_metric),
                    }
                )
    if not rows:
        raise ValueError(
            "The requested regression ensemble search space has no valid combinations."
        )
    return rows


def _eligible_config_ids(configs: pd.DataFrame, plan: Ensemble) -> set[str]:
    frame = configs.copy()
    if "active" in frame.columns and not bool(getattr(plan, "include_inactive", False)):
        frame = frame[
            pd.to_numeric(frame["active"], errors="coerce").fillna(0).astype(int).eq(1)
        ]
    excluded = {str(x) for x in getattr(plan, "exclude_config_ids", ()) if str(x)}
    if excluded:
        frame = frame[~frame["config_id"].astype(str).isin(excluded)]
    for column, values in (
        ("learner", getattr(plan, "exclude_learners", ())),
        ("resolution", getattr(plan, "exclude_resolutions", ())),
        ("count_transformation", getattr(plan, "exclude_transformations", ())),
    ):
        blocked = {str(x).casefold() for x in values if str(x)}
        if blocked and column in frame.columns:
            frame = frame[~frame[column].astype(str).str.casefold().isin(blocked)]
    return set(frame["config_id"].astype(str))


def _complete_inner_scores(
    inner: pd.DataFrame, outer_key: str | None, metric: str, eligible: set[str]
) -> pd.Series:
    frame = inner.copy()
    if outer_key is not None:
        frame = frame[frame["split_key"].astype(str).eq(str(outer_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    frame["inner_key"] = frame["inner_key"].astype(str)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[frame["config_id"].isin(eligible)]
    if "ok" in frame.columns:
        frame = frame[
            pd.to_numeric(frame["ok"], errors="coerce").fillna(0).astype(int).eq(1)
        ]
    expected = set(frame["inner_key"].astype(str).unique())
    values: dict[str, float] = {}
    for cid, group in frame.groupby("config_id", sort=True):
        group = group.drop_duplicates("inner_key", keep="last")
        valid = group[np.isfinite(group[metric].to_numpy(dtype=float))]
        if set(valid["inner_key"].astype(str)) == expected and expected:
            values[str(cid)] = float(valid[metric].mean())
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, dtype=float).sort_values(
        ascending=metric_is_loss(metric), kind="mergesort"
    )


def _prediction_frame(predictions: pd.DataFrame, outer_key: str | None) -> pd.DataFrame:
    frame = predictions.copy()
    if "outer_split_key" not in frame.columns:
        frame["outer_split_key"] = frame["split_key"]
    if outer_key is not None:
        frame = frame[frame["outer_split_key"].astype(str).eq(str(outer_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    frame["outer_split_key"] = frame["outer_split_key"].astype(str)
    return frame


def _aligned_stack(
    predictions: pd.DataFrame, members: list[str], inner: bool
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    if len(members) < 1:
        return None, None
    keys = ["outer_split_key"]
    if inner:
        keys.append("split_key")
    keys.extend(["sample_id", "sample_index", "y_true"])
    base: pd.DataFrame | None = None
    columns: list[str] = []
    for index, cid in enumerate(members):
        part = predictions[predictions["config_id"].astype(str).eq(str(cid))].copy()
        if part.empty or "y_pred" not in part.columns:
            return None, None
        part = part[keys + ["y_pred"]].drop_duplicates(keys, keep="last")
        name = f"pred_{index}"
        part = part.rename(columns={"y_pred": name})
        base = (
            part
            if base is None
            else base.merge(part, on=keys, how="inner", validate="one_to_one")
        )
        columns.append(name)
    if base is None or base.empty:
        return None, None
    counts = [
        len(
            predictions[predictions["config_id"].astype(str).eq(str(cid))][
                keys
            ].drop_duplicates()
        )
        for cid in members
    ]
    if any(len(base) != count for count in counts):
        return None, None
    base = base.sort_values(keys, kind="mergesort").reset_index(drop=True)
    stack = np.vstack(
        [pd.to_numeric(base[c], errors="coerce").to_numpy(dtype=float) for c in columns]
    )
    if not np.isfinite(stack).all():
        return None, None
    return base[keys].copy(), stack


def aggregate_regression_predictions(
    stack: np.ndarray, aggregation: str, weights: np.ndarray | list[float] | None = None
) -> np.ndarray:
    values = np.asarray(stack, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or not np.isfinite(values).all():
        raise ValueError(
            "Regression ensemble predictions must have shape n_members x n_samples and be finite."
        )
    method = _canonical_aggregation(aggregation)
    if method == "mean_prediction":
        return np.mean(values, axis=0)
    if method == "median_prediction":
        return np.median(values, axis=0)
    if method == "weighted_mean_prediction":
        if weights is None:
            raise ValueError("weighted_mean_prediction requires ensemble weights.")
        w = np.asarray(weights, dtype=float).reshape(-1)
        if (
            len(w) != values.shape[0]
            or not np.isfinite(w).all()
            or np.any(w < 0)
            or float(w.sum()) <= 0
        ):
            raise ValueError("Regression ensemble weights are invalid.")
        w = w / float(w.sum())
        return np.tensordot(w, values, axes=(0, 0))
    raise ValueError(f"Unsupported regression aggregation {aggregation!r}.")


def _metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    metrics = compute_regression_metrics(
        np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    )
    if metric not in metrics:
        raise ValueError(f"Unknown regression ensemble metric {metric!r}.")
    return float(metrics[metric])


def _learner_type(value: str) -> str:
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    compact = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    for prefixes, name in (
        (("randomforest", "random_forest", "rf"), "random_forest"),
        (("extratrees", "extra_trees", "et"), "extra_trees"),
        (
            ("histgradientboosting", "hist_gradient_boosting", "hgb"),
            "hist_gradient_boosting",
        ),
        (("gradientboosting", "gradient_boosting", "gb"), "gradient_boosting"),
        (("ridge",), "ridge"),
        (("elasticnet", "elastic_net"), "elastic_net"),
        (("lasso",), "lasso"),
        (("linearregression", "linear_regression"), "linear_regression"),
        (("xgb", "xgboost"), "xgboost"),
        (("lgbm", "lightgbm"), "lightgbm"),
        (("catboost",), "catboost"),
    ):
        if compact in prefixes or any(
            compact.startswith(prefix + "_") for prefix in prefixes
        ):
            return name
    return compact.split("_", 1)[0] or compact


def _simple_members(
    method: str, scores: pd.Series, configs: pd.DataFrame, max_size: int
) -> list[str]:
    ordered = [str(x) for x in scores.index.tolist()]
    if method == "top_k":
        return ordered[:max_size] if len(ordered) >= max_size else []
    meta = configs.drop_duplicates("config_id").copy()
    meta["config_id"] = meta["config_id"].astype(str)
    meta = meta.set_index("config_id")
    out: list[str] = []
    seen: set[str] = set()
    for cid in ordered:
        if cid not in meta.index:
            continue
        if method == "best_per_resolution":
            key = str(meta.loc[cid, "resolution"])
        elif method == "best_per_learner_type":
            key = _learner_type(str(meta.loc[cid, "learner"]))
        else:
            raise ValueError(
                f"Unsupported simple regression ensemble selector {method!r}."
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(cid)
        if len(out) >= max_size:
            break
    return out


def _evaluate_candidate(
    spec: dict[str, Any],
    members: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    weights: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred = aggregate_regression_predictions(
        stack, str(spec["aggregation_strategy"]), weights
    )
    metrics = compute_regression_metrics(y_true, pred)
    row: dict[str, Any] = {
        **spec,
        "member_count": int(len(members)),
        "effective_member_count": int(np.count_nonzero(weights > _WEIGHT_TOL))
        if weights is not None
        else int(len(members)),
        "members": json.dumps(members),
        "weights": json.dumps([float(x) for x in weights])
        if weights is not None
        else json.dumps([]),
        f"{metric}_mean": float(metrics[metric]),
    }
    for name, value in metrics.items():
        row[f"{name}_mean"] = float(value)
    if extra:
        row.update(extra)
    return row


def _caruana(
    library: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, list[float]]:
    running = np.zeros(stack.shape[1], dtype=float)
    chosen: list[int] = []
    trajectory: list[float] = []
    best_score = float("inf") if metric_is_loss(metric) else float("-inf")
    best_prefix = 0
    stale = 0
    ceiling = min(1000, max(100, 25 * max_members))
    patience = max(20, 5 * max_members)
    minimum = min(ceiling, max(20, 5 * max_members))
    for step in range(ceiling):
        distinct = set(chosen)
        candidates: list[tuple[float, int, str]] = []
        for index, cid in enumerate(library):
            if index not in distinct and len(distinct) >= max_members:
                continue
            pred = (running + stack[index]) / float(step + 1)
            score = _metric_value(y_true, pred, metric)
            candidates.append((score, index, cid))
        if not candidates:
            break
        candidates.sort(
            key=lambda x: (
                (x[0], x[1], x[2]) if metric_is_loss(metric) else (-x[0], x[1], x[2])
            )
        )
        score, index, _ = candidates[0]
        chosen.append(index)
        running += stack[index]
        trajectory.append(float(score))
        if best_prefix == 0 or metric_better(score, best_score, metric, tol=1e-10):
            best_score = float(score)
            best_prefix = len(chosen)
            stale = 0
        else:
            stale += 1
        if len(chosen) >= minimum and stale >= patience:
            break
    chosen = chosen[:best_prefix]
    if not chosen:
        raise RuntimeError(
            "Caruana regression ensemble selection did not produce a candidate."
        )
    order: list[int] = []
    counts: dict[int, int] = {}
    for index in chosen:
        if index not in counts:
            order.append(index)
            counts[index] = 0
        counts[index] += 1
    members = [library[index] for index in order]
    weights = np.asarray([counts[index] for index in order], dtype=float)
    weights /= float(weights.sum())
    return members, weights, trajectory[:best_prefix]


def _fit_super_learner(
    library: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], np.ndarray]:
    def objective(weights: np.ndarray, local_stack: np.ndarray) -> float:
        pred = aggregate_regression_predictions(
            local_stack, "weighted_mean_prediction", weights
        )
        score = _metric_value(y_true, pred, metric)
        return float(score if metric_is_loss(metric) else -score)

    def optimise(
        local_stack: np.ndarray, start: np.ndarray | None = None
    ) -> np.ndarray:
        n = local_stack.shape[0]
        x0 = (
            np.full(n, 1.0 / n, dtype=float)
            if start is None
            else np.asarray(start, dtype=float)
        )
        x0 = np.clip(x0, 0.0, None)
        x0 /= float(x0.sum())
        result = minimize(
            objective,
            x0,
            args=(local_stack,),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": _OPT_MAXITER, "ftol": _OPT_FTOL, "disp": False},
        )
        if not bool(result.success):
            raise RuntimeError(
                f"Regression Super Learner optimization failed: {result.message}"
            )
        values = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
        if not np.isfinite(values).all() or float(values.sum()) <= 0:
            raise RuntimeError("Regression Super Learner returned invalid weights.")
        return values / float(values.sum())

    full = optimise(stack)
    keep = np.flatnonzero(full > _WEIGHT_TOL).tolist()
    if len(keep) < 2:
        keep = sorted(
            range(len(library)), key=lambda i: (-float(full[i]), str(library[i]))
        )[: min(2, len(library))]
    if len(keep) > max_members:
        keep = sorted(keep, key=lambda i: (-float(full[i]), str(library[i])))[
            :max_members
        ]
    keep = sorted(keep)
    weights = optimise(stack[keep], full[keep])
    return [library[i] for i in keep], weights


def _candidate_table(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    outer_key: str | None,
    progress_callback=None,
    progress_scope: str | None = None,
) -> pd.DataFrame:
    eligible = _eligible_config_ids(configs, plan)
    scores = _complete_inner_scores(inner_results, outer_key, metric, eligible)
    if scores.empty:
        return pd.DataFrame()
    pred = _prediction_frame(inner_predictions, outer_key)
    pred = pred[pred["config_id"].isin(set(scores.index))]
    rows: list[dict[str, Any]] = []
    specs = _ensemble_configs(plan)
    scope = str(progress_scope or outer_key or "__final__")
    for index, spec in enumerate(specs, start=1):
        if progress_callback is not None:
            progress_callback(scope, "start", spec, index, len(specs), "evaluating")
        method = str(spec["selection_strategy"])
        row = None
        failed = False
        try:
            if method in {"top_k", "best_per_resolution", "best_per_learner_type"}:
                members = _simple_members(
                    method, scores, configs, int(spec["max_size"])
                )
                if len(members) < 2:
                    continue
                ordered, stack = _aligned_stack(pred, members, True)
                if ordered is None or stack is None:
                    continue
                row = _evaluate_candidate(
                    spec,
                    members,
                    stack,
                    ordered["y_true"].to_numpy(dtype=float),
                    metric,
                    extra={"inner_oof_rows": int(len(ordered))},
                )
            else:
                library = [str(x) for x in scores.index.tolist()]
                ordered, stack = _aligned_stack(pred, library, True)
                if ordered is None or stack is None or len(library) < 2:
                    continue
                y_true = ordered["y_true"].to_numpy(dtype=float)
                if method == "caruana":
                    members, weights, trajectory = _caruana(
                        library, stack, y_true, metric, int(spec["max_size"])
                    )
                    if len(members) < 2:
                        continue
                    selected = stack[[library.index(cid) for cid in members]]
                    row = _evaluate_candidate(
                        spec,
                        members,
                        selected,
                        y_true,
                        metric,
                        weights,
                        {
                            "inner_oof_rows": int(len(ordered)),
                            "weight_source": "caruana_selection_frequency",
                            "caruana_trajectory": json.dumps(
                                [float(x) for x in trajectory]
                            ),
                        },
                    )
                elif method == "super_learner":
                    members, weights = _fit_super_learner(
                        library, stack, y_true, metric, int(spec["max_size"])
                    )
                    if len(members) < 2:
                        continue
                    selected = stack[[library.index(cid) for cid in members]]
                    row = _evaluate_candidate(
                        spec,
                        members,
                        selected,
                        y_true,
                        metric,
                        weights,
                        {
                            "inner_oof_rows": int(len(ordered)),
                            "weight_source": f"convex_{metric}",
                        },
                    )
                else:
                    raise ValueError(method)
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
    method_priority = {name: i for i, name in enumerate(_plan_methods(plan))}
    aggregation_priority = {name: i for i, name in enumerate(_plan_aggregations(plan))}
    out["_method"] = (
        out["selection_strategy"].map(method_priority).fillna(999).astype(int)
    )
    out["_aggregation"] = (
        out["aggregation_strategy"].map(aggregation_priority).fillna(999).astype(int)
    )
    out = out.sort_values(
        [
            score_col,
            "_method",
            "_aggregation",
            "member_count",
            "max_size",
            "ensemble_config_id",
        ],
        ascending=[metric_is_loss(metric), True, True, True, True, True],
        kind="mergesort",
    )
    return out.drop(columns=["_method", "_aggregation"]).reset_index(drop=True)


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return parsed


def _apply_candidate(
    candidate: dict[str, Any], predictions: pd.DataFrame, outer_key: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    members = [str(x) for x in _parse_json_list(candidate["members"])]
    weights = np.asarray(
        [float(x) for x in _parse_json_list(candidate.get("weights", "[]"))],
        dtype=float,
    )
    active_weights = (
        weights
        if str(candidate["aggregation_strategy"]) == "weighted_mean_prediction"
        else None
    )
    fold = _prediction_frame(predictions, outer_key)
    ordered, stack = _aligned_stack(fold, members, False)
    if ordered is None or stack is None:
        raise RuntimeError(
            f"Selected regression ensemble for outer fold {outer_key!r} has incomplete member predictions."
        )
    pred = aggregate_regression_predictions(
        stack, str(candidate["aggregation_strategy"]), active_weights
    )
    metrics = compute_regression_metrics(ordered["y_true"].to_numpy(dtype=float), pred)
    result = ordered.copy()
    result["ensemble_config_id"] = str(candidate["ensemble_config_id"])
    result["selection_strategy"] = str(candidate["selection_strategy"])
    result["aggregation_strategy"] = str(candidate["aggregation_strategy"])
    result["members"] = json.dumps(members)
    result["weights"] = json.dumps([float(x) for x in weights])
    result["y_pred"] = pred
    return result, metrics


def sweep_regression_ensemble(sweep: Sweep) -> dict[str, Path]:
    root = Path(sweep.root())
    ensemble_dir = root / "ensembling"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    outer_path = root / "predictions" / "outer_predictions.parquet"
    inner_result_path = root / "inner_results" / "inner_results.parquet"
    inner_prediction_path = root / "inner_predictions" / "inner_predictions.parquet"
    config_path = root / "configs.parquet"
    if any(
        not table_exists(path)
        for path in (outer_path, inner_result_path, inner_prediction_path, config_path)
    ):
        raise FileNotFoundError("Run evaluate(sweep) before sweep_ensemble(sweep).")
    outer = read_table(outer_path)
    inner_results = read_table(inner_result_path)
    inner_predictions = read_table(inner_prediction_path)
    configs = read_table(config_path)
    metric = str(sweep.ensemble.optimize_metric)
    if metric not in inner_results.columns:
        metric = str(sweep.evaluation.optimize_metric)
    if metric not in inner_results.columns:
        metric = "RMSE"
    stage("Regression ensemble sweep", str(root))
    summary_table(
        "Regression ensemble search",
        {
            "selection strategies": _plan_methods(sweep.ensemble),
            "aggregation strategies": _plan_aggregations(sweep.ensemble),
            "max sizes": _plan_max_sizes(sweep.ensemble),
            "optimize metric": metric,
        },
    )
    tracker = ResourceTracker(
        sample_interval_s=float(
            getattr(sweep.evaluation, "resource_sample_interval_s", 0.1)
        )
    ).start()
    mpma_b_selection = select_mpma_b_by_outer_fold(
        inner_results,
        configs,
        str(sweep.evaluation.optimize_metric),
        plan=sweep.ensemble,
    )
    mpma_b_predictions = selected_mpma_b_outer_predictions(mpma_b_selection, outer)
    mpma_b_metric_rows: list[dict[str, Any]] = []
    if not mpma_b_predictions.empty:
        for outer_key, group in mpma_b_predictions.groupby(
            "outer_split_key", sort=True
        ):
            config_ids = group["config_id"].astype(str).unique()
            if len(config_ids) != 1:
                raise ValueError(
                    "Each regression outer split must contain one selected MPMA-B config."
                )
            metrics = compute_regression_metrics(
                group["y_true"].to_numpy(dtype=float),
                group["y_pred"].to_numpy(dtype=float),
            )
            mpma_b_metric_rows.append(
                {
                    "outer_split_key": str(outer_key),
                    "config_id": str(config_ids[0]),
                    **metrics,
                }
            )
    mpma_b_metrics = pd.DataFrame(mpma_b_metric_rows)
    final_mpma = select_final_mpma_candidate(
        inner_results,
        configs,
        str(sweep.evaluation.optimize_metric),
        plan=sweep.ensemble,
    )
    if not final_mpma:
        raise RuntimeError(
            "No valid final regression MPMA-B candidate was produced from inner validation results."
        )
    tables_dir = root / "tables"
    results_dir = root / "results"
    predictions_dir = root / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    write_table(tables_dir / "mpma_b_outer_selection.parquet", mpma_b_selection)
    write_table(
        predictions_dir / "mpma_b_outer_predictions.parquet", mpma_b_predictions
    )
    write_table(results_dir / "mpma_b_outer_results.parquet", mpma_b_metrics)
    mpma_b_summary: dict[str, Any] = {
        "Strategy": "MPMA-B",
        "task": "regression",
        "selection_basis": "outer_fold_inner_validation",
        "n_outer_folds": int(len(mpma_b_metrics)),
    }
    for name in [
        "R2",
        "MAE",
        "MSE",
        "RMSE",
        "MedAE",
        "ExplainedVariance",
        "PearsonR",
        "SpearmanR",
    ]:
        if name in mpma_b_metrics.columns:
            values = (
                pd.to_numeric(mpma_b_metrics[name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(values):
                mpma_b_summary[f"outer_{name}_mean"] = float(values.mean())
                mpma_b_summary[f"outer_{name}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
    dump_json_standard(mpma_b_summary, tables_dir / "mpma_b_strategy_summary.json")
    dump_json_standard(final_mpma, tables_dir / "mpma_b_final_candidate.json")
    selections: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    outer_keys = sorted(set(inner_results["split_key"].astype(str)))
    ensemble_specs = _ensemble_configs(sweep.ensemble)
    with EnsembleSearchProgress(
        outer_keys + ["__final__"], len(ensemble_specs)
    ) as live:
        for outer_key in outer_keys:
            candidates = _candidate_table(
                inner_results,
                inner_predictions,
                configs,
                sweep.ensemble,
                metric,
                outer_key,
                progress_callback=live.update,
                progress_scope=outer_key,
            )
            if candidates.empty:
                raise RuntimeError(
                    f"No valid regression ensemble candidate was produced for outer fold {outer_key!r}."
                )
            winner = candidates.iloc[0].to_dict()
            pred_frame, metrics = _apply_candidate(winner, outer, outer_key)
            live.update(
                outer_key,
                "selected",
                winner,
                len(ensemble_specs),
                len(ensemble_specs),
                f"inner {metric}={float(winner[f'{metric}_mean']):.4f}",
            )
            selections.append(
                {
                    "outer_split_key": outer_key,
                    "ensemble_config_id": str(winner["ensemble_config_id"]),
                    "selection_basis": "outer_fold_inner_oof_predictions_only",
                    "selection_metric": metric,
                    "inner_score": float(winner[f"{metric}_mean"]),
                    "selection_strategy": str(winner["selection_strategy"]),
                    "aggregation_strategy": str(winner["aggregation_strategy"]),
                    "max_size": int(winner["max_size"]),
                    "member_count": int(winner["member_count"]),
                    "members": str(winner["members"]),
                    "weights": str(winner["weights"]),
                }
            )
            pred_frame["outer_split_key"] = outer_key
            prediction_frames.append(pred_frame)
            metric_rows.append(
                {
                    "outer_split_key": outer_key,
                    "ensemble_config_id": str(winner["ensemble_config_id"]),
                    **metrics,
                }
            )
        all_candidates = _candidate_table(
            inner_results,
            inner_predictions,
            configs,
            sweep.ensemble,
            metric,
            None,
            progress_callback=live.update,
            progress_scope="__final__",
        )
        if not all_candidates.empty:
            pooled = all_candidates.iloc[0].to_dict()
            live.update(
                "__final__",
                "selected",
                pooled,
                len(ensemble_specs),
                len(ensemble_specs),
                f"inner {metric}={float(pooled[f'{metric}_mean']):.4f}",
            )
    if all_candidates.empty:
        raise RuntimeError(
            "No valid final regression ensemble candidate was produced from inner OOF predictions."
        )
    best = all_candidates.iloc[0].to_dict()
    final_ensemble = {
        "ensemble_config_id": str(best["ensemble_config_id"]),
        "selection_basis": "all_inner_oof_predictions_for_final_refit",
        "selection_metric": metric,
        "optimize_metric": metric,
        "member_score_metric": str(sweep.evaluation.optimize_metric),
        "inner_score": float(best[f"{metric}_mean"]),
        "selection_strategy": str(best["selection_strategy"]),
        "aggregation_strategy": str(best["aggregation_strategy"]),
        "max_size": int(best["max_size"]),
        "ensemble_size": int(best["member_count"]),
        "member_count": int(best["member_count"]),
        "effective_member_count": int(
            best.get("effective_member_count", best["member_count"])
        ),
        "members": [str(x) for x in _parse_json_list(best["members"])],
        "weights": [float(x) for x in _parse_json_list(best.get("weights", "[]"))],
        "task": "regression",
    }
    selection_df = pd.DataFrame(selections)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(metric_rows)
    selection_path = ensemble_dir / "mpma_e_outer_selection.parquet"
    prediction_path = ensemble_dir / "ensemble_predictions.parquet"
    result_path = ensemble_dir / "mpma_e_outer_results.parquet"
    candidate_path = ensemble_dir / "ensemble_candidate_scores.parquet"
    final_path = ensemble_dir / "mpma_e_final_candidate.json"
    summary_path = ensemble_dir / "mpma_e_strategy_summary.json"
    write_table(selection_path, selection_df)
    write_table(prediction_path, predictions_df)
    write_table(result_path, metrics_df)
    write_table(candidate_path, all_candidates)
    dump_json_standard(final_ensemble, final_path)
    summary = {
        "Strategy": "MPMA-E",
        "task": "regression",
        "selection_basis": "outer_fold_inner_oof_predictions_only",
        "n_outer_folds": int(len(metrics_df)),
    }
    for name in [
        "R2",
        "MAE",
        "MSE",
        "RMSE",
        "MedAE",
        "ExplainedVariance",
        "PearsonR",
        "SpearmanR",
    ]:
        if name in metrics_df.columns:
            values = (
                pd.to_numeric(metrics_df[name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(values):
                summary[f"outer_{name}_mean"] = float(values.mean())
                summary[f"outer_{name}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
    dump_json_standard(summary, summary_path)
    selected = {
        "inner_val_best_mpma": final_mpma,
        "inner_val_best_mpmas_ensemble": final_ensemble,
        "nested_mpma_e": summary,
        "task": "regression",
    }
    selected_path = ensemble_dir / "selected_unit.json"
    dump_json_standard(selected, selected_path)
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
                "config_id": final_ensemble["ensemble_config_id"],
                "selection_basis": final_ensemble["selection_basis"],
                "inner_score": final_ensemble["inner_score"],
                "members": json.dumps(final_ensemble["members"]),
                "weights": json.dumps(final_ensemble["weights"]),
            },
        ]
    )
    comparison_path = ensemble_dir / "final_model_comparison.parquet"
    write_table(comparison_path, comparison)
    resource_path = ensemble_dir / "mpma_e_selection_resources.json"
    dump_json_standard(tracker.stop(), resource_path)
    if getattr(sweep, "uses_modalities", False):
        mpma_e_outputs = {}
    else:
        from .ensemble_sweep import _matrix_for_mpma_e_figure
        from .mpma_e_figure import write_single_task_mpma_e_figure

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
        f"Regression ensemble sweep completed · final candidate={final_ensemble['ensemble_config_id']} · inner {metric}={final_ensemble['inner_score']:.4f}"
    )
    outputs = {
        "ensemble_dir": ensemble_dir,
        "selected_unit": selected_path,
        "comparison": comparison_path,
        "mpma_e_selection": selection_path,
        "mpma_e_predictions": prediction_path,
        "mpma_e_outer_results": result_path,
        "mpma_e_candidates": candidate_path,
        "mpma_e_summary": summary_path,
        "mpma_e_final_candidate": final_path,
        "mpma_e_selection_resources": resource_path,
        **mpma_e_outputs,
    }
    path_table("Regression ensemble outputs", outputs)
    return outputs
