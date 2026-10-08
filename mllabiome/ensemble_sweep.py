from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .advanced_ensemble import ADVANCED_AGGREGATIONS, fit_crossfitted_advanced
from .compute import ResourceTracker
from .configs_sweep import Ensemble, Sweep, sweep_task
from .console import path_table, stage, success, summary_table
from .data import _wide_csv_feature_columns, load_dataset, metadata_path
from .ensemble_aggregation import (
    LEARNED_AGGREGATIONS,
    PROBABILITY_PRESERVING_AGGREGATIONS,
    SUPPORTED_AGGREGATIONS,
    aggregate_member_predictions,
    effective_aggregation_weights,
    stacking_features,
)
from .ensemble_progress import EnsembleSearchProgress
from .integrations import integration_modality_sets
from .metrics import (
    _renormalize_proba,
    aggregate_validation_metric,
    canonical_metric_name,
    compute_metrics,
)
from .metrics import metric_better as _metric_better
from .metrics import metric_is_loss as _metric_is_loss
from .metrics import metric_requires_probability_semantics
from .mpma_e_figure import write_single_task_mpma_e_figure
from .selection import _qualified_config_ids_for_splits, select_final_mpma_candidate
from .storage import read_table, table_exists, write_table
from .utils import TAXONOMIC_LEVELS, dump_json_standard

_ENSEMBLE_SEARCH_SCHEMA = "mpmae_search_v10"
_SUPER_LEARNER_WEIGHT_TOL = 1e-8
_SUPER_LEARNER_MIN_WEIGHT = 0.01
_SUPER_LEARNER_OPT_MAXITER = 1000
_SUPER_LEARNER_OPT_FTOL = 1e-12
_CARUANA_IMPROVEMENT_TOL = 1e-10
_BAGGED_CARUANA_MIN_BAGS = 8
_BAGGED_CARUANA_MAX_BAGS = 24


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
    metric = canonical_metric_name(str(plan.optimize_metric)).casefold()
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


def _late_integrations(plan: Ensemble) -> tuple[Any, ...]:
    return tuple(getattr(plan, "late_integrations", ()))


def _unimodal_config_modalities(configs: pd.DataFrame) -> dict[str, str]:
    required = {"config_id", "modalities"}
    missing = required - set(configs.columns)
    if missing:
        raise ValueError(
            f"Late fusion requires config metadata columns: {sorted(missing)}"
        )
    frame = configs.drop_duplicates("config_id", keep="last").copy()
    if "candidate_family" in frame.columns:
        frame = frame[frame["candidate_family"].astype(str).eq("unimodal")]
    elif "integration" in frame.columns:
        frame = frame[frame["integration"].astype(str).eq("unimodal")]
    else:
        raise ValueError(
            "Late fusion requires candidate_family or integration metadata to identify unimodal base models."
        )
    out: dict[str, str] = {}
    for row in frame[["config_id", "modalities"]].to_dict(orient="records"):
        parts = tuple(
            value.strip()
            for value in str(row["modalities"]).split(",")
            if value.strip()
        )
        if len(parts) == 1:
            out[str(row["config_id"])] = parts[0]
    return out


def _late_modality_names(configs: pd.DataFrame) -> tuple[str, ...]:
    names = tuple(sorted(set(_unimodal_config_modalities(configs).values())))
    if len(names) < 2:
        raise ValueError("Late fusion requires at least two unimodal modalities.")
    return names


def _validate_plan(plan: Ensemble) -> None:
    supported_selection = {
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "quality_diversity",
        "diversity",
        "performance_diversity",
        "caruana",
        "bagged_caruana",
        "caruana_multistart",
        "beam",
        "super_learner",
        "adaptive_super_learner",
        "safe_super_learner",
        "regularized_super_learner",
    }
    include_model_ensembles = bool(getattr(plan, "include_model_ensembles", True))
    late_integrations = _late_integrations(plan)
    if not include_model_ensembles and not late_integrations:
        raise ValueError(
            "Ensemble search has neither model ensembles nor late integrations enabled."
        )
    if include_model_ensembles:
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
        metric = canonical_metric_name(str(plan.optimize_metric))
        if metric_requires_probability_semantics(metric):
            invalid = sorted(
                set(aggregations) - set(PROBABILITY_PRESERVING_AGGREGATIONS)
            )
            if invalid:
                raise ValueError(
                    f"Ensemble.optimize_metric={metric!r} requires probability-valued aggregation. "
                    f"Remove incompatible aggregation strategy(s) {invalid}; allowed strategies are "
                    f"{sorted(PROBABILITY_PRESERVING_AGGREGATIONS)}."
                )
        max_sizes = _plan_max_sizes(plan)
        if not max_sizes:
            raise ValueError("Ensemble.max_sizes must be non-empty.")
        if any(int(x) < 2 for x in max_sizes):
            raise ValueError("Every Ensemble.max_sizes entry must be at least 2.")
        learned_methods = {
            "caruana",
            "bagged_caruana",
            "caruana_multistart",
            "beam",
            "super_learner",
            "adaptive_super_learner",
            "safe_super_learner",
            "regularized_super_learner",
        }
        learned = set(methods) & learned_methods
        simple = set(methods) - learned_methods
        if learned and "weighted_mean_proba" not in aggregations:
            raise ValueError(
                "Learned-weight ensemble selectors require aggregation_strategies to include "
                "'weighted_mean_proba'. Add that aggregation or remove those selection strategies."
            )
        nonweighted = [x for x in aggregations if x != "weighted_mean_proba"]
        if simple and not nonweighted:
            raise ValueError(
                "Simple ensemble selectors do not learn member weights, so they require at least "
                "one non-weighted aggregation strategy (for example 'mean_proba')."
            )
    allowed_late = {
        "late_mean_proba",
        "late_weighted_mean_proba",
        "late_super_learner",
    }
    invalid_late = sorted({item.key for item in late_integrations} - allowed_late)
    if invalid_late:
        raise ValueError(
            f"Unsupported classification late integration(s): {invalid_late}."
        )


def _ensemble_configs(
    plan: Ensemble, configs: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    _validate_plan(plan)
    rows: list[dict[str, Any]] = []
    if bool(getattr(plan, "include_model_ensembles", True)):
        learned_weight_selectors = {
            "caruana",
            "bagged_caruana",
            "caruana_multistart",
            "beam",
            "super_learner",
            "adaptive_super_learner",
            "safe_super_learner",
            "regularized_super_learner",
        }
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
                            "model_ensemble",
                            str(plan.optimize_metric),
                            method,
                            aggregation,
                            str(max_size),
                            _resolved_super_learner_loss(plan)
                            if method
                            in {
                                "super_learner",
                                "adaptive_super_learner",
                                "safe_super_learner",
                                "regularized_super_learner",
                            }
                            else "na",
                        ]
                    )
                    rows.append(
                        {
                            "ensemble_config_id": hashlib.sha1(
                                uid.encode()
                            ).hexdigest()[:12],
                            "search_schema": _ENSEMBLE_SEARCH_SCHEMA,
                            "ensemble_kind": "model_ensemble",
                            "optimize_metric": str(plan.optimize_metric),
                            "selection_strategy": method,
                            "aggregation_strategy": aggregation,
                            "max_size": int(max_size),
                        }
                    )
    late_integrations = _late_integrations(plan)
    if late_integrations:
        if configs is None:
            raise ValueError(
                "Late-fusion ensemble configuration requires config metadata."
            )
        modality_names = _late_modality_names(configs)
        aggregation_by_key = {
            "late_mean_proba": "mean_proba",
            "late_weighted_mean_proba": "weighted_mean_proba",
            "late_super_learner": "weighted_mean_proba",
        }
        for integration in late_integrations:
            for modality_set in integration_modality_sets(integration, modality_names):
                if len(modality_set) < 2:
                    raise ValueError(
                        "Late fusion modality sets must contain at least two modalities."
                    )
                if len(set(modality_set)) != len(modality_set):
                    raise ValueError(
                        "Late fusion modality sets cannot contain duplicate modalities."
                    )
                modalities = tuple(str(value) for value in modality_set)
                aggregation = aggregation_by_key[integration.key]
                uid = "__".join(
                    [
                        _ENSEMBLE_SEARCH_SCHEMA,
                        "late_fusion",
                        str(plan.optimize_metric),
                        integration.key,
                        aggregation,
                        ",".join(modalities),
                    ]
                )
                rows.append(
                    {
                        "ensemble_config_id": hashlib.sha1(uid.encode()).hexdigest()[
                            :12
                        ],
                        "search_schema": _ENSEMBLE_SEARCH_SCHEMA,
                        "ensemble_kind": "late_fusion",
                        "optimize_metric": str(plan.optimize_metric),
                        "selection_strategy": integration.key,
                        "aggregation_strategy": aggregation,
                        "max_size": len(modalities),
                        "integration": integration.key,
                        "modalities": json.dumps(list(modalities)),
                    }
                )
    unique = {str(row["ensemble_config_id"]): row for row in rows}
    rows = [unique[key] for key in sorted(unique)]
    if not rows:
        raise ValueError(
            "The requested ensemble search space has no valid combinations."
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
    qualification: pd.DataFrame | None = None,
) -> pd.Series:
    required = {"inner_key", "config_id"}
    missing = required - set(inner.columns)
    if missing:
        raise ValueError(
            f"Inner-results table is missing required columns: {sorted(missing)}"
        )
    frame = inner.copy()
    if qualification is not None and "split_key" not in frame.columns:
        raise ValueError(
            "Inner-results table must contain split_key when qualification filtering is enabled."
        )
    if outer_split_key is not None:
        if "split_key" not in frame.columns:
            raise ValueError("Inner-results table must contain split_key.")
        frame = frame[frame["split_key"].astype(str).eq(str(outer_split_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    frame["inner_key"] = frame["inner_key"].astype(str)
    expected = set(frame["inner_key"].dropna().astype(str))
    scope_split_keys = (
        set(frame["split_key"].astype(str).unique())
        if "split_key" in frame.columns
        else ({str(outer_split_key)} if outer_split_key is not None else set())
    )
    qualified_ids = _qualified_config_ids_for_splits(qualification, scope_split_keys)
    if qualified_ids is not None:
        eligible_ids = set(eligible_ids) & set(qualified_ids)
    frame = frame[frame["config_id"].isin(eligible_ids)]
    if "ok" in frame.columns:
        ok = pd.to_numeric(frame["ok"], errors="coerce").fillna(0).astype(int)
        frame = frame[ok.eq(1)]
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
        group = frame[frame["config_id"].eq(str(config_id))].copy()
        group = group.drop_duplicates("inner_key", keep="last")
        value, _ = aggregate_validation_metric(group, metric)
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
    metric = canonical_metric_name(metric)
    if metric in {"subject_macro_log_loss", "cohort_macro_log_loss"}:
        raise ValueError(
            f"Ensemble optimize_metric={metric!r} requires grouping information and is not supported for ensemble search; use log_loss for probability-ensemble optimization."
        )
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


def _select_top_k(scores: pd.Series, size: int) -> list[str]:
    return [str(x) for x in scores.index[: int(size)].tolist()]


def _prediction_correlation_matrix(stack: np.ndarray) -> np.ndarray:
    values = np.asarray(stack, dtype=float)
    eps = np.finfo(float).eps
    logged = np.log(np.clip(values, eps, 1.0))
    centered = logged - np.mean(logged, axis=2, keepdims=True)
    features = centered.reshape(centered.shape[0], -1)
    features = features - np.mean(features, axis=1, keepdims=True)
    norms = np.linalg.norm(features, axis=1)
    denom = np.outer(norms, norms)
    corr = np.divide(
        features @ features.T,
        denom,
        out=np.zeros((len(features), len(features)), dtype=float),
        where=denom > 0.0,
    )
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _quality_diversity_select(
    member_ids: list[str], stack: np.ndarray, size: int
) -> list[str]:
    count = min(int(size), len(member_ids))
    if count <= 0:
        return []
    if count == 1:
        return [member_ids[0]]
    corr = np.abs(_prediction_correlation_matrix(stack))
    n_members = len(member_ids)
    quality = (n_members - np.arange(n_members, dtype=float)) / float(n_members)
    selected = [0]
    remaining = set(range(1, n_members))
    while remaining and len(selected) < count:
        ranked: list[tuple[float, float, float, int, str]] = []
        for index in remaining:
            redundancy = max(float(corr[index, other]) for other in selected)
            novelty = 1.0 - redundancy
            utility = 0.5 * float(quality[index]) + 0.5 * novelty
            ranked.append(
                (utility, novelty, float(quality[index]), -index, member_ids[index])
            )
        ranked.sort(reverse=True)
        selected_index = int(-ranked[0][3])
        selected.append(selected_index)
        remaining.remove(selected_index)
    return [member_ids[index] for index in selected]


def _metric_gain(candidate: float, baseline: float, metric: str) -> float:
    if _metric_is_loss(metric):
        return float(baseline) - float(candidate)
    return float(candidate) - float(baseline)


def _stable_metric_improvement(
    ordered: pd.DataFrame,
    candidate_proba: np.ndarray,
    baseline_proba: np.ndarray,
    metric: str,
) -> tuple[bool, list[float]]:
    y_true = ordered["y_true"].to_numpy(dtype=int)
    candidate_score = _metric_value(y_true, candidate_proba, metric)
    baseline_score = _metric_value(y_true, baseline_proba, metric)
    if not _metric_better(candidate_score, baseline_score, metric):
        return False, []
    split_keys = ordered["split_key"].astype(str).to_numpy()
    gains: list[float] = []
    for split_key in sorted(set(split_keys.tolist())):
        mask = split_keys == split_key
        score_candidate = _metric_value(y_true[mask], candidate_proba[mask], metric)
        score_baseline = _metric_value(y_true[mask], baseline_proba[mask], metric)
        if np.isfinite(score_candidate) and np.isfinite(score_baseline):
            gains.append(_metric_gain(score_candidate, score_baseline, metric))
    if len(gains) < 2:
        return True, gains
    values = np.asarray(gains, dtype=float)
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    stable = float(np.mean(values)) > standard_error and float(np.median(values)) > 0.0
    return bool(stable), gains


def _evaluate_selected_members(
    spec: dict[str, Any],
    members: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    *,
    ordered: pd.DataFrame | None = None,
    native_weights: np.ndarray | None = None,
    native_weight_source: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("MPMA-E must contain at least two distinct members.")
    if stack.shape[0] != len(members):
        raise ValueError("MPMA-E member count and prediction stack disagree.")
    aggregation = str(spec["aggregation_strategy"])
    active_weights: np.ndarray | None = None
    aggregation_parameters: dict[str, Any] | None = None
    aggregation_weight_source = "not_applicable"
    aggregation_extra: dict[str, Any] = {}
    if aggregation == "weighted_mean_proba":
        if native_weights is None:
            raise ValueError(
                f"{spec['selection_strategy']} does not define weights required by weighted_mean_proba."
            )
        active_weights = np.asarray(native_weights, dtype=float)
        if (
            active_weights.shape != (len(members),)
            or not np.isfinite(active_weights).all()
            or np.any(active_weights < 0.0)
            or int(np.count_nonzero(active_weights > _SUPER_LEARNER_WEIGHT_TOL)) < 2
        ):
            raise ValueError(
                "Weighted MPMA-E must contain at least two positive-weight members."
            )
        aggregation_weight_source = native_weight_source or "learned"
    elif aggregation == "mean_proba":
        active_weights = effective_aggregation_weights(aggregation, len(members))
        aggregation_weight_source = "uniform"
    if aggregation in LEARNED_AGGREGATIONS:
        if ordered is None:
            raise ValueError(f"{aggregation} requires aligned inner-fold metadata.")
        if aggregation in ADVANCED_AGGREGATIONS:
            proba, aggregation_parameters, aggregation_extra = fit_crossfitted_advanced(
                stack,
                y_true,
                _stacking_fold_keys(ordered),
                aggregation,
                metric,
                _metric_value,
                _metric_is_loss,
            )
        else:
            proba, aggregation_parameters, aggregation_extra = (
                _fit_crossfitted_stacking(stack, y_true, ordered, aggregation, metric)
            )
        aggregation_weight_source = "meta_learner"
    else:
        proba = aggregate_member_predictions(
            stack, aggregation, active_weights, aggregation_parameters
        )
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
        "aggregation_parameters": json.dumps(aggregation_parameters or {}),
        "member_count": len(members),
        "ensemble_size": len(members),
        "effective_member_count": int(
            np.count_nonzero(np.asarray(stored_weights, dtype=float) > 1e-12)
        )
        if stored_weights
        else len(members),
        "aggregation_weight_source": aggregation_weight_source,
        "weight_source": aggregation_weight_source,
        **{f"{key}_mean": float(value) for key, value in metrics.items()},
        **aggregation_extra,
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


def _member_error_correlation(stack: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    target = np.zeros((len(y_true), stack.shape[2]), dtype=float)
    target[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0
    residual = (target[None, :, :] - np.asarray(stack, dtype=float)).reshape(
        stack.shape[0], -1
    )
    centered = residual - residual.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denom = norms[:, None] * norms[None, :]
    corr = np.divide(
        centered @ centered.T,
        denom,
        out=np.zeros((stack.shape[0], stack.shape[0]), dtype=float),
        where=denom > 0.0,
    )
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _rank_quality(n_members: int) -> np.ndarray:
    if n_members <= 1:
        return np.ones(n_members, dtype=float)
    return 1.0 - np.arange(n_members, dtype=float) / float(n_members - 1)


def _select_diversity(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    max_members: int,
) -> tuple[list[str], dict[str, Any]]:
    n_members = len(member_ids)
    limit = min(max(2, int(max_members)), n_members)
    corr = np.abs(_member_error_correlation(stack, y_true))
    quality = _rank_quality(n_members)
    selected = [0]
    best_selected: list[int] | None = None
    best_objective = float("-inf")
    trace: list[dict[str, Any]] = []
    while len(selected) < limit:
        remaining = [i for i in range(n_members) if i not in selected]
        candidates: list[tuple[float, float, int]] = []
        for index in remaining:
            complement = float(np.mean(1.0 - corr[index, selected]))
            score = complement + 0.10 * float(quality[index])
            candidates.append((score, complement, index))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected.append(int(candidates[0][2]))
        sub = corr[np.ix_(selected, selected)]
        upper = sub[np.triu_indices(len(selected), k=1)]
        complementarity = float(np.mean(1.0 - upper)) if len(upper) else 0.0
        quality_mean = float(np.mean(quality[selected]))
        objective = complementarity + 0.15 * quality_mean - 0.005 * (len(selected) - 2)
        trace.append(
            {
                "size": len(selected),
                "objective": objective,
                "complementarity": complementarity,
                "quality": quality_mean,
            }
        )
        if objective > best_objective + 1e-12:
            best_objective = objective
            best_selected = list(selected)
    if best_selected is None:
        best_selected = selected[:2]
    return [member_ids[i] for i in best_selected], {
        "diversity_objective": float(best_objective),
        "diversity_trace": json.dumps(trace),
    }


def _select_performance_diversity(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], dict[str, Any]]:
    n_members = len(member_ids)
    limit = min(max(2, int(max_members)), n_members)
    corr = np.abs(_member_error_correlation(stack, y_true))
    quality = _rank_quality(n_members)
    best_members: list[int] | None = None
    best_score = float("inf") if _metric_is_loss(metric) else float("-inf")
    best_alpha = 0.0
    best_size = 0
    evaluations: list[dict[str, Any]] = []
    for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
        selected = [0]
        for size in range(2, limit + 1):
            remaining = [i for i in range(n_members) if i not in selected]
            candidates: list[tuple[float, int]] = []
            for index in remaining:
                complement = float(np.mean(1.0 - corr[index, selected]))
                value = (1.0 - alpha) * float(quality[index]) + alpha * complement
                candidates.append((value, index))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected.append(int(candidates[0][1]))
            proba = np.mean(stack[selected], axis=0)
            score = _metric_value(y_true, proba, metric)
            evaluations.append(
                {"alpha": float(alpha), "size": int(size), "score": float(score)}
            )
            better = best_members is None or _metric_better(
                float(score), float(best_score), metric, tol=1e-12
            )
            tied = (
                best_members is not None
                and abs(float(score) - float(best_score)) <= 1e-12
            )
            if better or (tied and size < best_size):
                best_members = list(selected)
                best_score = float(score)
                best_alpha = float(alpha)
                best_size = int(size)
    if best_members is None:
        best_members = [0, 1]
        best_size = 2
    return [member_ids[i] for i in best_members], {
        "performance_diversity_alpha": float(best_alpha),
        "performance_diversity_selected_size": int(best_size),
        "performance_diversity_reference_score": float(best_score),
        "performance_diversity_trace": json.dumps(evaluations),
    }


def _caruana_select_from_start(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
    start_index: int,
) -> tuple[list[str], np.ndarray, float, list[float], dict[str, Any]]:
    if len(member_ids) < 2 or len(set(member_ids)) != len(member_ids):
        raise ValueError("Caruana requires at least two distinct candidate models.")
    if stack.ndim != 3 or stack.shape[0] != len(member_ids):
        raise ValueError("Caruana library and prediction stack are inconsistent.")
    if max_members < 2:
        raise ValueError("Caruana max_members must be at least 2.")
    controls = _caruana_controls(max_members)
    max_iterations = int(controls["max_iterations"])
    patience = int(controls["patience"])
    min_iterations = int(controls["min_iterations"])
    improvement_tol = float(controls["improvement_tol"])
    chosen_indices = [int(start_index)]
    running_sum = np.asarray(stack[int(start_index)], dtype=float).copy()
    trajectory = [_metric_value(y_true, running_sum, metric)]
    best_score = float("inf") if _metric_is_loss(metric) else float("-inf")
    best_prefix = 0
    stale_steps = 0
    stop_reason = "iteration_ceiling"
    for step in range(1, max_iterations):
        distinct = set(chosen_indices)
        candidate_scores: list[tuple[float, float, int, str]] = []
        denom = float(step + 1)
        for index, config_id in enumerate(member_ids):
            if len(distinct) == 1 and index in distinct:
                continue
            if index not in distinct and len(distinct) >= int(max_members):
                continue
            proba = _renormalize_proba(
                (running_sum + stack[index]) / denom, stack.shape[2]
            )
            score = _metric_value(y_true, proba, metric)
            eps = np.finfo(float).eps
            picked = np.clip(proba[np.arange(len(y_true)), y_true], eps, 1.0)
            tie_loss = float(-np.mean(np.log(picked)))
            candidate_scores.append((score, tie_loss, index, str(config_id)))
        if not candidate_scores:
            stop_reason = "no_candidates"
            break
        if _metric_is_loss(metric):
            candidate_scores.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        else:
            candidate_scores.sort(
                key=lambda item: (-item[0], item[1], item[2], item[3])
            )
        score, _, selected_index, _ = candidate_scores[0]
        chosen_indices.append(int(selected_index))
        running_sum = running_sum + stack[selected_index]
        trajectory.append(float(score))
        valid = len(set(chosen_indices)) >= 2
        improved = valid and (
            best_prefix == 0
            or _metric_better(
                float(score), float(best_score), metric, tol=improvement_tol
            )
        )
        if improved:
            best_score = float(score)
            best_prefix = len(chosen_indices)
            stale_steps = 0
        elif valid:
            stale_steps += 1
        if (
            best_prefix >= 2
            and len(chosen_indices) >= min_iterations
            and stale_steps >= patience
        ):
            stop_reason = "converged_patience"
            break
    if best_prefix < 2:
        raise RuntimeError("Multi-start Caruana did not produce a genuine ensemble.")
    chosen_indices = chosen_indices[:best_prefix]
    order: list[int] = []
    counts: dict[int, int] = {}
    for index in chosen_indices:
        if index not in counts:
            order.append(index)
            counts[index] = 0
        counts[index] += 1
    members = [member_ids[index] for index in order]
    weights = np.asarray([counts[index] for index in order], dtype=float)
    weights /= float(weights.sum())
    return (
        members,
        weights,
        float(best_score),
        trajectory[:best_prefix],
        {
            "caruana_multistart_start_member": str(member_ids[int(start_index)]),
            "caruana_multistart_iterations_selected": int(best_prefix),
            "caruana_multistart_stop_reason": stop_reason,
        },
    )


def _caruana_multistart_select(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, float, list[float], dict[str, Any]]:
    if len(member_ids) < 2:
        raise ValueError("Multi-start Caruana requires at least two candidates.")
    starts = min(len(member_ids), max(4, 2 * int(max_members)))
    best = None
    scores: list[dict[str, Any]] = []
    for start_index in range(starts):
        candidate = _caruana_select_from_start(
            member_ids, stack, y_true, metric, max_members, start_index
        )
        scores.append(
            {"member": str(member_ids[start_index]), "score": float(candidate[2])}
        )
        if best is None or _metric_better(candidate[2], best[2], metric, tol=1e-12):
            best = candidate
    if best is None:
        raise RuntimeError("Multi-start Caruana did not produce a valid ensemble.")
    members, weights, score, trajectory, diagnostics = best
    diagnostics = {
        **diagnostics,
        "caruana_multistart_starts_evaluated": int(starts),
        "caruana_multistart_scores": json.dumps(scores),
    }
    return members, weights, score, trajectory, diagnostics


def _beam_select(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, float, list[float], dict[str, Any]]:
    if len(member_ids) < 2:
        raise ValueError("Beam ensemble search requires at least two candidates.")
    width = min(64, max(16, 4 * int(max_members)))
    ceiling = min(100, max(20, 5 * int(max_members)))
    patience = max(5, int(max_members))
    states: list[np.ndarray] = []
    for index in range(len(member_ids)):
        counts = np.zeros(len(member_ids), dtype=np.int16)
        counts[index] = 1
        states.append(counts)
    best_score = float("inf") if _metric_is_loss(metric) else float("-inf")
    best_counts = None
    trajectory: list[float] = []
    stale = 0
    states_evaluated = 0
    stop_reason = "iteration_ceiling"
    for depth in range(2, ceiling + 1):
        expanded: dict[tuple[int, ...], np.ndarray] = {}
        for counts in states:
            active = set(np.flatnonzero(counts > 0).tolist())
            for index in range(len(member_ids)):
                if len(active) == 1 and depth == 2 and index in active:
                    continue
                if index not in active and len(active) >= int(max_members):
                    continue
                updated = counts.copy()
                updated[index] += 1
                expanded[tuple(int(x) for x in updated)] = updated
        if not expanded:
            stop_reason = "no_candidates"
            break
        scored: list[tuple[float, float, tuple[int, ...], np.ndarray]] = []
        for key, counts in expanded.items():
            weights = counts.astype(float) / float(np.sum(counts))
            proba = _weighted_probability_mean(stack, weights)
            score = _metric_value(y_true, proba, metric)
            eps = np.finfo(float).eps
            picked = np.clip(proba[np.arange(len(y_true)), y_true], eps, 1.0)
            tie_loss = float(-np.mean(np.log(picked)))
            scored.append((float(score), tie_loss, key, counts))
        states_evaluated += len(scored)
        if _metric_is_loss(metric):
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
        else:
            scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        states = [item[3] for item in scored[:width]]
        depth_score = float(scored[0][0])
        trajectory.append(depth_score)
        improved = best_counts is None or _metric_better(
            depth_score, best_score, metric, tol=1e-12
        )
        if improved:
            best_score = depth_score
            best_counts = scored[0][3].copy()
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            stop_reason = "converged_patience"
            break
    if best_counts is None or int(np.count_nonzero(best_counts)) < 2:
        raise RuntimeError("Beam search did not produce a genuine ensemble.")
    keep = np.flatnonzero(best_counts > 0).tolist()
    members = [member_ids[index] for index in keep]
    weights = best_counts[keep].astype(float)
    weights /= float(weights.sum())
    return (
        members,
        weights,
        float(best_score),
        trajectory,
        {
            "beam_width": int(width),
            "beam_iterations_executed": int(len(trajectory)),
            "beam_iteration_ceiling": int(ceiling),
            "beam_patience": int(patience),
            "beam_stop_reason": stop_reason,
            "beam_states_evaluated": int(states_evaluated),
        },
    )


def _caruana_select(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, float, list[float], dict[str, Any]]:
    if len(member_ids) < 2 or len(set(member_ids)) != len(member_ids):
        raise ValueError("Caruana requires at least two distinct candidate models.")
    if stack.ndim != 3 or stack.shape[0] != len(member_ids):
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
            if len(distinct) == 1 and j in distinct:
                continue
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

        valid = len(set(chosen_indices)) >= 2
        improved = valid and (
            best_prefix == 0
            or _metric_better(
                float(score), float(best_score), metric, tol=improvement_tol
            )
        )
        if improved:
            best_score = float(score)
            best_prefix = len(chosen_indices)
            stale_steps = 0
        elif valid:
            stale_steps += 1

        if (
            best_prefix >= 2
            and len(chosen_indices) >= min_iterations
            and stale_steps >= patience
        ):
            stop_reason = "converged_patience"
            break

    executed_iterations = len(chosen_indices)
    chosen_indices = chosen_indices[:best_prefix]
    if len(set(chosen_indices)) < 2:
        raise RuntimeError(
            "Caruana ensemble selection did not produce a genuine ensemble."
        )
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


def _bagged_caruana_select(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    if len(member_ids) < 2 or len(set(member_ids)) != len(member_ids):
        raise ValueError(
            "Bagged Caruana requires at least two distinct candidate models."
        )
    if stack.ndim != 3 or stack.shape[0] != len(member_ids):
        raise ValueError(
            "Bagged Caruana library and prediction stack are inconsistent."
        )
    if max_members < 2:
        raise ValueError("Bagged Caruana max_members must be at least 2.")
    n_members = len(member_ids)
    bag_count = int(
        min(
            _BAGGED_CARUANA_MAX_BAGS,
            max(_BAGGED_CARUANA_MIN_BAGS, np.ceil(2.0 * np.sqrt(n_members))),
        )
    )
    subset_size = int(
        min(
            n_members,
            max(int(max_members), int(np.ceil(2.0 * np.sqrt(n_members)))),
        )
    )
    digest = hashlib.sha1("|".join(member_ids).encode()).hexdigest()
    seed = int(digest[:8], 16)
    rng = np.random.default_rng(seed)
    accumulated = np.zeros(n_members, dtype=float)
    remaining = np.arange(1, n_members, dtype=int)
    for _ in range(bag_count):
        if subset_size >= n_members:
            indices = np.arange(n_members, dtype=int)
        else:
            sampled = rng.choice(remaining, size=max(0, subset_size - 1), replace=False)
            indices = np.asarray(
                [0, *sorted(int(value) for value in sampled)], dtype=int
            )
        subset_ids = [member_ids[int(index)] for index in indices]
        subset_stack = stack[indices]
        selected_ids, selected_weights, _, _, _ = _caruana_select(
            subset_ids,
            subset_stack,
            y_true,
            metric,
            min(int(max_members), len(subset_ids)),
        )
        for selected_id, weight in zip(selected_ids, selected_weights, strict=True):
            accumulated[member_ids.index(selected_id)] += float(weight)
    if float(accumulated.sum()) <= 0.0:
        raise RuntimeError("Bagged Caruana did not assign positive ensemble weight.")
    averaged = accumulated / float(accumulated.sum())
    keep = np.flatnonzero(averaged > 0.0).tolist()
    if len(keep) > int(max_members):
        keep = sorted(
            keep,
            key=lambda index: (-float(averaged[index]), str(member_ids[index])),
        )[: int(max_members)]
    keep = sorted(keep)
    weights = averaged[keep]
    weights = weights / float(weights.sum())
    members = [member_ids[index] for index in keep]
    return (
        members,
        weights,
        {
            "bagged_caruana_bags": int(bag_count),
            "bagged_caruana_subset_size": int(subset_size),
            "bagged_caruana_seed": int(seed),
        },
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


def _super_learner_gradient(
    weights: np.ndarray, stack: np.ndarray, y_true: np.ndarray, loss: str
) -> np.ndarray:
    proba = np.einsum("m,mnc->nc", weights, stack, optimize=True)
    if loss == "log_loss":
        picked = np.clip(
            proba[np.arange(len(y_true)), y_true], np.finfo(float).eps, 1.0
        )
        member_picked = stack[:, np.arange(len(y_true)), y_true]
        return -np.mean(member_picked / picked[None, :], axis=1)
    if loss == "brier":
        residual = proba.copy()
        residual[np.arange(len(y_true)), y_true] -= 1.0
        return (2.0 / len(y_true)) * np.einsum(
            "nc,mnc->m", residual, stack, optimize=True
        )
    raise ValueError(f"Unknown Super Learner loss: {loss!r}.")


def _fit_convex_probability_weights(
    stack: np.ndarray,
    y_true: np.ndarray,
    loss: str,
    start: np.ndarray | None = None,
    min_weight: float = 0.0,
) -> np.ndarray:
    n_members = int(stack.shape[0])
    if n_members < 2:
        raise ValueError("Convex probability weighting requires at least two members.")
    floor = float(min_weight)
    if not np.isfinite(floor) or floor < 0.0 or floor * n_members >= 1.0:
        raise ValueError("Infeasible convex probability weight floor.")
    x0 = (
        _uniform_weights(n_members) if start is None else np.asarray(start, dtype=float)
    )
    if x0.shape != (n_members,) or not np.isfinite(x0).all() or np.any(x0 < 0.0):
        raise ValueError("Initial convex ensemble weights are invalid.")
    if float(x0.sum()) <= 0.0:
        raise ValueError("Initial convex ensemble weights must have positive mass.")
    x0 = x0 / float(x0.sum())
    scale = 1.0 - floor * n_members
    if floor:
        x0 = np.maximum(x0 - floor, 0.0)
        x0 = (
            x0 / float(x0.sum())
            if float(x0.sum()) > 0.0
            else _uniform_weights(n_members)
        )

    def weights_from_simplex(values: np.ndarray) -> np.ndarray:
        return floor + scale * values

    def objective(values: np.ndarray) -> float:
        return _super_learner_loss(weights_from_simplex(values), stack, y_true, loss)

    def gradient(values: np.ndarray) -> np.ndarray:
        return scale * _super_learner_gradient(
            weights_from_simplex(values), stack, y_true, loss
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=[(0.0, 1.0)] * n_members,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        ],
        options={
            "maxiter": _SUPER_LEARNER_OPT_MAXITER,
            "ftol": _SUPER_LEARNER_OPT_FTOL,
            "disp": False,
        },
    )
    if not bool(result.success):
        raise RuntimeError(f"Super Learner optimization failed: {result.message}")
    values = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    if not np.all(np.isfinite(values)) or float(values.sum()) <= 0.0:
        raise RuntimeError("Super Learner returned invalid ensemble weights.")
    values /= float(values.sum())
    weights = weights_from_simplex(values)
    if not np.isfinite(weights).all() or np.any(weights < floor):
        raise RuntimeError("Super Learner returned infeasible ensemble weights.")
    return weights


def _regularized_super_learner_loss(
    weights: np.ndarray,
    stack: np.ndarray,
    y_true: np.ndarray,
    loss: str,
    anchor: np.ndarray,
    strength: float,
) -> float:
    base = _super_learner_loss(weights, stack, y_true, loss)
    penalty = float(strength) * float(np.sum((weights - anchor) ** 2))
    return float(base + penalty)


def _fit_regularized_convex_probability_weights(
    stack: np.ndarray,
    y_true: np.ndarray,
    loss: str,
    anchor: np.ndarray,
    strength: float,
    start: np.ndarray | None = None,
    min_weight: float = 0.0,
) -> np.ndarray:
    n_members = int(stack.shape[0])
    if n_members < 2:
        raise ValueError("Regularized Super Learner requires at least two members.")
    floor = float(min_weight)
    if not np.isfinite(floor) or floor < 0.0 or floor * n_members >= 1.0:
        raise ValueError("Infeasible regularized ensemble weight floor.")
    anchor_values = np.asarray(anchor, dtype=float)
    if anchor_values.shape != (n_members,):
        raise ValueError("Regularized Super Learner anchor has invalid shape.")
    if np.any(anchor_values < 0.0) or not np.isfinite(anchor_values).all():
        raise ValueError("Regularized Super Learner anchor is invalid.")
    anchor_total = float(anchor_values.sum())
    if anchor_total <= 0.0:
        raise ValueError("Regularized Super Learner anchor must have positive mass.")
    anchor_values = anchor_values / anchor_total
    x0 = anchor_values.copy() if start is None else np.asarray(start, dtype=float)
    if x0.shape != (n_members,) or not np.isfinite(x0).all() or np.any(x0 < 0.0):
        raise ValueError("Initial regularized ensemble weights are invalid.")
    if float(x0.sum()) <= 0.0:
        raise ValueError(
            "Initial regularized ensemble weights must have positive mass."
        )
    x0 = x0 / float(x0.sum())
    scale = 1.0 - floor * n_members
    if floor:
        x0 = np.maximum(x0 - floor, 0.0)
        x0 = (
            x0 / float(x0.sum())
            if float(x0.sum()) > 0.0
            else _uniform_weights(n_members)
        )

    def weights_from_simplex(values: np.ndarray) -> np.ndarray:
        return floor + scale * values

    def objective(values: np.ndarray) -> float:
        return _regularized_super_learner_loss(
            weights_from_simplex(values), stack, y_true, loss, anchor_values, strength
        )

    def gradient(values: np.ndarray) -> np.ndarray:
        weights = weights_from_simplex(values)
        return scale * (
            _super_learner_gradient(weights, stack, y_true, loss)
            + 2.0 * float(strength) * (weights - anchor_values)
        )

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_members,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        ],
        options={
            "maxiter": _SUPER_LEARNER_OPT_MAXITER,
            "ftol": _SUPER_LEARNER_OPT_FTOL,
            "disp": False,
        },
    )
    if not bool(result.success):
        raise RuntimeError(
            f"Regularized Super Learner optimization failed: {result.message}"
        )
    values = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    if not np.all(np.isfinite(values)) or float(values.sum()) <= 0.0:
        raise RuntimeError("Regularized Super Learner returned invalid weights.")
    values /= float(values.sum())
    weights = weights_from_simplex(values)
    if not np.isfinite(weights).all() or np.any(weights < floor):
        raise RuntimeError("Regularized Super Learner returned infeasible weights.")
    return weights


def _adaptive_regularization_strength(n_samples: int, n_members: int) -> float:
    numerator = float(np.log1p(max(int(n_members), 1)))
    denominator = float(np.log1p(max(int(n_samples), 2)))
    return numerator / denominator


def _select_super_learner_support(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    loss: str,
    full_weights: np.ndarray,
    max_members: int,
    anchor_index: int,
) -> list[int]:
    if len(member_ids) < 2 or len(set(member_ids)) != len(member_ids):
        raise ValueError("Super Learner needs at least two distinct candidate models.")
    if max_members < 2:
        raise ValueError("Super Learner max_members must be at least two.")
    if not 0 <= anchor_index < len(member_ids):
        raise ValueError("Super Learner anchor index is invalid.")
    ranked = sorted(
        range(len(member_ids)),
        key=lambda index: (-float(full_weights[index]), str(member_ids[index])),
    )
    selected = [anchor_index]
    selected.extend(
        index
        for index in ranked
        if index != anchor_index
        and float(full_weights[index]) >= _SUPER_LEARNER_MIN_WEIGHT
    )
    selected = selected[: min(max_members, len(member_ids))]
    if len(selected) < 2:
        partner_scores: list[tuple[float, str, int]] = []
        for index in range(len(member_ids)):
            if index == anchor_index:
                continue
            candidate = (1.0 - _SUPER_LEARNER_MIN_WEIGHT) * stack[
                anchor_index
            ] + _SUPER_LEARNER_MIN_WEIGHT * stack[index]
            score = _super_learner_loss(
                np.ones(1, dtype=float), candidate[None, :, :], y_true, loss
            )
            partner_scores.append((float(score), str(member_ids[index]), index))
        selected.append(min(partner_scores)[2])
    return sorted(selected)


def _fit_super_learner(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    loss: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, float]:
    if stack.shape[0] != len(member_ids):
        raise ValueError("Super Learner library and prediction stack are inconsistent.")
    if (
        len(member_ids) < 2
        or len(set(member_ids)) != len(member_ids)
        or max_members < 2
    ):
        raise ValueError("Super Learner requires at least two distinct members.")
    full_weights = _fit_convex_probability_weights(stack, y_true, loss)
    indices = _select_super_learner_support(
        member_ids,
        stack,
        y_true,
        loss,
        full_weights,
        max_members,
        int(np.argmax(full_weights)),
    )
    kept_weights = _fit_convex_probability_weights(
        stack[indices],
        y_true,
        loss,
        full_weights[indices],
        min_weight=_SUPER_LEARNER_MIN_WEIGHT,
    )
    final_loss = _super_learner_loss(kept_weights, stack[indices], y_true, loss)
    return [member_ids[index] for index in indices], kept_weights, float(final_loss)


def _fit_adaptive_super_learner(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    loss: str,
    max_members: int,
) -> tuple[list[str], np.ndarray, float, float]:
    if stack.shape[0] != len(member_ids):
        raise ValueError(
            "Adaptive Super Learner library and prediction stack are inconsistent."
        )
    if (
        len(member_ids) < 2
        or len(set(member_ids)) != len(member_ids)
        or max_members < 2
    ):
        raise ValueError(
            "Adaptive Super Learner requires at least two distinct members."
        )
    strength = _adaptive_regularization_strength(len(y_true), len(member_ids))
    anchor = np.zeros(len(member_ids), dtype=float)
    anchor[0] = 1.0
    full_weights = _fit_regularized_convex_probability_weights(
        stack, y_true, loss, anchor, strength
    )
    indices = _select_super_learner_support(
        member_ids, stack, y_true, loss, full_weights, max_members, 0
    )
    kept_anchor = np.zeros(len(indices), dtype=float)
    kept_anchor[indices.index(0)] = 1.0
    kept_weights = _fit_regularized_convex_probability_weights(
        stack[indices],
        y_true,
        loss,
        kept_anchor,
        strength,
        full_weights[indices],
        min_weight=_SUPER_LEARNER_MIN_WEIGHT,
    )
    value = _super_learner_loss(kept_weights, stack[indices], y_true, loss)
    return (
        [member_ids[index] for index in indices],
        kept_weights,
        float(value),
        float(strength),
    )


def _stacking_fold_keys(ordered: pd.DataFrame) -> np.ndarray:
    if "split_key" not in ordered.columns:
        raise ValueError("Stacking aggregation requires inner split identifiers.")
    if "outer_split_key" in ordered.columns:
        values = (
            ordered["outer_split_key"].astype(str)
            + "::"
            + ordered["split_key"].astype(str)
        )
    else:
        values = ordered["split_key"].astype(str)
    keys = values.to_numpy(dtype=str)
    if len(set(keys.tolist())) < 2:
        raise ValueError("Stacking aggregation requires at least two inner folds.")
    return keys


def _stacker_sample_weights(fold_keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(fold_keys).astype(str)
    unique, counts = np.unique(keys, return_counts=True)
    mapping = {key: 1.0 / float(count) for key, count in zip(unique, counts)}
    weights = np.asarray([mapping[key] for key in keys], dtype=float)
    return weights * (len(weights) / float(weights.sum()))


def _fit_logistic_stacker(
    features: np.ndarray,
    y_true: np.ndarray,
    *,
    c_value: float,
    l1_ratio: float | None,
    sample_weight: np.ndarray | None,
) -> LogisticRegression:
    penalty_default = (
        inspect.signature(LogisticRegression).parameters["penalty"].default
    )
    legacy_penalty_api = str(penalty_default) != "deprecated"
    if l1_ratio is None:
        values: dict[str, Any] = {
            "C": float(c_value),
            "solver": "lbfgs",
            "max_iter": 5000,
            "random_state": 0,
        }
        if legacy_penalty_api:
            values["penalty"] = "l2"
    else:
        values = {
            "C": float(c_value),
            "solver": "saga",
            "l1_ratio": float(l1_ratio),
            "max_iter": 20000,
            "tol": 1e-5,
            "random_state": 0,
        }
        if legacy_penalty_api:
            values["penalty"] = "elasticnet"
    model = LogisticRegression(**values)
    model.fit(features, y_true, sample_weight=sample_weight)
    return model


def _stacker_parameters(
    model: LogisticRegression,
    n_members: int,
    n_classes: int,
    *,
    clip: float,
    c_value: float,
    l1_ratio: float | None,
    fold_balanced: bool,
) -> dict[str, Any]:
    classes = np.asarray(model.classes_, dtype=int)
    if not np.array_equal(np.sort(classes), np.arange(n_classes, dtype=int)):
        raise ValueError("Stacking fit did not retain every probability class.")
    return {
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": np.asarray(model.intercept_, dtype=float).tolist(),
        "classes": classes.tolist(),
        "n_members": int(n_members),
        "n_classes": int(n_classes),
        "clip": float(clip),
        "C": float(c_value),
        "l1_ratio": None if l1_ratio is None else float(l1_ratio),
        "fold_balanced": bool(fold_balanced),
    }


def _stacker_grid(aggregation: str) -> list[tuple[float, float | None]]:
    if aggregation == "logistic_stacking":
        return [(value, None) for value in (0.05, 0.2, 1.0, 5.0, 20.0)]
    if aggregation == "elastic_net_stacking":
        return [
            (c_value, l1_ratio)
            for c_value in (0.05, 0.2, 1.0, 5.0)
            for l1_ratio in (0.1, 0.5, 0.9)
        ]
    if aggregation == "cohort_robust_stacking":
        return [(value, None) for value in (0.02, 0.05, 0.2, 1.0, 5.0)]
    raise ValueError(f"Unsupported learned aggregation {aggregation!r}.")


def _fit_crossfitted_stacking(
    stack: np.ndarray,
    y_true: np.ndarray,
    ordered: pd.DataFrame,
    aggregation: str,
    metric: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    clip = 1e-6
    features = stacking_features(stack, clip)
    fold_keys = _stacking_fold_keys(ordered)
    unique_folds = sorted(set(fold_keys.tolist()))
    n_classes = int(stack.shape[2])
    classes = np.arange(n_classes, dtype=int)
    if not np.array_equal(np.unique(y_true), classes):
        raise ValueError(
            "Stacking aggregation requires every class in the inner OOF data."
        )
    fold_balanced = aggregation == "cohort_robust_stacking"
    best = None
    trace: list[dict[str, Any]] = []
    for c_value, l1_ratio in _stacker_grid(aggregation):
        crossfit = np.zeros((len(y_true), n_classes), dtype=float)
        valid = True
        fold_scores: list[float] = []
        for fold in unique_folds:
            test_mask = fold_keys == fold
            train_mask = ~test_mask
            if not np.array_equal(np.unique(y_true[train_mask]), classes):
                valid = False
                break
            sample_weight = (
                _stacker_sample_weights(fold_keys[train_mask])
                if fold_balanced
                else None
            )
            try:
                model = _fit_logistic_stacker(
                    features[train_mask],
                    y_true[train_mask],
                    c_value=c_value,
                    l1_ratio=l1_ratio,
                    sample_weight=sample_weight,
                )
            except Exception:
                valid = False
                break
            predicted = np.asarray(
                model.predict_proba(features[test_mask]), dtype=float
            )
            local = np.zeros((int(np.count_nonzero(test_mask)), n_classes), dtype=float)
            for source, target in enumerate(np.asarray(model.classes_, dtype=int)):
                local[:, int(target)] = predicted[:, source]
            crossfit[test_mask] = local
            fold_scores.append(_metric_value(y_true[test_mask], local, metric))
        if (
            not valid
            or not np.isfinite(crossfit).all()
            or np.any(crossfit.sum(axis=1) <= 0)
        ):
            continue
        pooled_score = _metric_value(y_true, crossfit, metric)
        fold_array = np.asarray(fold_scores, dtype=float)
        fold_mean = float(np.mean(fold_array))
        fold_sd = float(np.std(fold_array, ddof=1)) if len(fold_array) > 1 else 0.0
        objective = (
            fold_mean + 0.25 * fold_sd
            if _metric_is_loss(metric) and fold_balanced
            else fold_mean - 0.25 * fold_sd
            if fold_balanced
            else pooled_score
        )
        trace.append(
            {
                "C": float(c_value),
                "l1_ratio": None if l1_ratio is None else float(l1_ratio),
                "pooled_score": float(pooled_score),
                "fold_mean": fold_mean,
                "fold_sd": fold_sd,
                "objective": float(objective),
            }
        )
        candidate = (float(objective), float(pooled_score), c_value, l1_ratio, crossfit)
        if best is None:
            best = candidate
        else:
            better = _metric_better(candidate[0], best[0], metric, tol=1e-12)
            tied = abs(candidate[0] - best[0]) <= 1e-12
            if better or (tied and float(c_value) < float(best[2])):
                best = candidate
    if best is None:
        raise RuntimeError(
            f"{aggregation} could not fit a valid cross-fitted meta-model."
        )
    _, pooled_score, c_value, l1_ratio, crossfit = best
    full_weight = _stacker_sample_weights(fold_keys) if fold_balanced else None
    final_model = _fit_logistic_stacker(
        features,
        y_true,
        c_value=float(c_value),
        l1_ratio=l1_ratio,
        sample_weight=full_weight,
    )
    parameters = _stacker_parameters(
        final_model,
        stack.shape[0],
        n_classes,
        clip=clip,
        c_value=float(c_value),
        l1_ratio=l1_ratio,
        fold_balanced=fold_balanced,
    )
    diagnostics = {
        "stacking_crossfit_score": float(pooled_score),
        "stacking_C": float(c_value),
        "stacking_l1_ratio": "" if l1_ratio is None else float(l1_ratio),
        "stacking_fold_balanced": int(fold_balanced),
        "stacking_internal_folds": len(unique_folds),
        "stacking_search_trace": json.dumps(trace),
    }
    return np.asarray(crossfit, dtype=float), parameters, diagnostics


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
    extra: dict[str, Any] = {"selection_weight_source": "none"}
    diagnostics: dict[str, Any] = {}
    if method in {"diversity", "performance_diversity"}:
        library = [str(value) for value in scores.index.tolist()]
        ordered, library_stack = _aligned_stack(predictions, library, pcols, inner=True)
        if ordered is None or library_stack is None:
            incomplete = set(_missing_members(predictions, library, pcols, inner=True))
            library = [value for value in library if value not in incomplete]
            if len(library) < 2:
                return None
            ordered, library_stack = _aligned_stack(
                predictions, library, pcols, inner=True
            )
        if ordered is None or library_stack is None or len(library) < 2:
            return None
        y_true = ordered["y_true"].to_numpy(dtype=int)
        if method == "diversity":
            members, diagnostics = _select_diversity(
                library, library_stack, y_true, max_size
            )
        else:
            members, diagnostics = _select_performance_diversity(
                library, library_stack, y_true, metric, max_size
            )
        indices = [library.index(config_id) for config_id in members]
        stack = library_stack[indices]
        extra["candidate_library_policy"] = "all_complete_inner_oof_mpmas"
        extra["candidate_library_size"] = len(library)
    else:
        if method == "top_k":
            members = _select_top_k(scores, max_size)
        elif method == "best_per_resolution":
            members = _select_best_per_resolution(scores, configs)[:max_size]
        elif method == "best_per_learner_type":
            members = _select_best_per_learner_type(scores, configs)[:max_size]
        elif method == "quality_diversity":
            library = [str(value) for value in scores.index.tolist()]
            ordered_library, library_stack = _aligned_stack(
                predictions, library, pcols, inner=True
            )
            if ordered_library is None or library_stack is None:
                incomplete = set(
                    _missing_members(predictions, library, pcols, inner=True)
                )
                library = [value for value in library if value not in incomplete]
                if len(library) < 2:
                    return None
                ordered_library, library_stack = _aligned_stack(
                    predictions, library, pcols, inner=True
                )
            if ordered_library is None or library_stack is None:
                return None
            members = _quality_diversity_select(library, library_stack, max_size)
            extra["candidate_library_policy"] = (
                "quality_diversity_log_probability_geometry"
            )
            extra["candidate_library_size"] = len(library)
        else:
            raise ValueError(f"Unsupported simple ensemble selector: {method!r}.")
        if len(members) < 2:
            return None
        ordered, stack = _aligned_stack(predictions, members, pcols, inner=True)
        if ordered is None or stack is None:
            return None
        y_true = ordered["y_true"].to_numpy(dtype=int)
    if len(members) < 2:
        return None
    extra["inner_oof_rows"] = len(ordered)
    extra.update(diagnostics)
    return _evaluate_selected_members(
        spec,
        members,
        stack,
        y_true,
        metric,
        ordered=ordered,
        extra=extra,
    )


def _late_fusion_members(
    spec: dict[str, Any], scores: pd.Series, configs: pd.DataFrame
) -> tuple[list[str], list[str]]:
    modalities = [str(value) for value in json.loads(str(spec["modalities"]))]
    config_modalities = _unimodal_config_modalities(configs)
    ordered = [str(value) for value in scores.index.tolist()]
    members: list[str] = []
    for modality in modalities:
        member = next(
            (
                config_id
                for config_id in ordered
                if config_modalities.get(config_id) == modality
            ),
            None,
        )
        if member is None:
            return [], modalities
        members.append(member)
    return members, modalities


def _fit_late_weighted_mean(
    stack: np.ndarray, y_true: np.ndarray, metric: str
) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    n_members = int(stack.shape[0])
    if n_members < 2:
        raise ValueError("Late weighted fusion requires at least two modalities.")
    counts = np.ones(n_members, dtype=float)
    running = np.sum(stack, axis=0)
    current_size = n_members
    best_score = _metric_value(y_true, running / float(current_size), metric)
    best_counts = counts.copy()
    trajectory = [float(best_score)]
    controls = _caruana_controls(n_members)
    stale = 0
    stop_reason = "iteration_ceiling"
    max_iterations = max(n_members, int(controls["max_iterations"]))
    min_iterations = max(n_members, int(controls["min_iterations"]))
    patience = int(controls["patience"])
    improvement_tol = float(controls["improvement_tol"])
    while current_size < max_iterations:
        candidates: list[tuple[float, int]] = []
        for index in range(n_members):
            proba = (running + stack[index]) / float(current_size + 1)
            candidates.append((_metric_value(y_true, proba, metric), index))
        candidates.sort(
            key=lambda item: (
                (item[0], item[1]) if _metric_is_loss(metric) else (-item[0], item[1])
            )
        )
        score, selected = candidates[0]
        counts[selected] += 1.0
        running += stack[selected]
        current_size += 1
        trajectory.append(float(score))
        if _metric_better(score, best_score, metric, tol=improvement_tol):
            best_score = float(score)
            best_counts = counts.copy()
            stale = 0
        else:
            stale += 1
        if current_size >= min_iterations and stale >= patience:
            stop_reason = "converged"
            break
    weights = best_counts / float(best_counts.sum())
    diagnostics = {
        "late_weight_iterations_executed": int(current_size),
        "late_weight_stop_reason": stop_reason,
        "late_weight_internal_iteration_ceiling": int(max_iterations),
        "late_weight_internal_patience": int(patience),
        "late_weight_internal_min_iterations": int(min_iterations),
        "late_weight_internal_improvement_tol": float(improvement_tol),
    }
    return weights, trajectory, diagnostics


def _candidate_from_late_fusion(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    configs: pd.DataFrame,
    pcols: list[str],
    plan: Ensemble,
    metric: str,
) -> dict[str, Any] | None:
    members, modalities = _late_fusion_members(spec, scores, configs)
    if len(members) != len(modalities) or len(members) < 2:
        return None
    ordered, stack = _aligned_stack(predictions, members, pcols, inner=True)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    integration = str(spec["integration"])
    extra: dict[str, Any] = {
        "inner_oof_rows": len(ordered),
        "candidate_library_policy": "best_complete_unimodal_mpma_per_modality",
        "candidate_library_size": len(members),
        "late_modalities": json.dumps(modalities),
    }
    if integration == "late_mean_proba":
        return _evaluate_selected_members(
            spec, members, stack, y_true, metric, extra=extra
        )
    if integration == "late_weighted_mean_proba":
        weights, trajectory, diagnostics = _fit_late_weighted_mean(
            stack, y_true, metric
        )
        extra.update(
            {
                "selection_weight_source": "late_caruana_selection_frequency",
                "late_weight_trajectory": json.dumps(
                    [float(value) for value in trajectory]
                ),
                **diagnostics,
            }
        )
        return _evaluate_selected_members(
            spec,
            members,
            stack,
            y_true,
            metric,
            native_weights=weights,
            native_weight_source="late_caruana_selection_frequency",
            extra=extra,
        )
    if integration == "late_super_learner":
        loss = _resolved_super_learner_loss(plan)
        weights = _fit_convex_probability_weights(
            stack, y_true, loss, min_weight=_SUPER_LEARNER_MIN_WEIGHT
        )
        loss_value = _super_learner_loss(weights, stack, y_true, loss)
        extra.update(
            {
                "selection_weight_source": f"late_convex_{loss}",
                "super_learner_loss": loss,
                "super_learner_loss_rule": "brier_if_optimize_metric_is_brier_else_log_loss",
                "super_learner_loss_value": float(loss_value),
                "super_learner_internal_weight_tol": float(_SUPER_LEARNER_WEIGHT_TOL),
                "super_learner_internal_min_weight": float(_SUPER_LEARNER_MIN_WEIGHT),
                "super_learner_internal_optimizer": "SLSQP",
                "super_learner_internal_maxiter": int(_SUPER_LEARNER_OPT_MAXITER),
                "super_learner_internal_ftol": float(_SUPER_LEARNER_OPT_FTOL),
            }
        )
        return _evaluate_selected_members(
            spec,
            members,
            stack,
            y_true,
            metric,
            native_weights=weights,
            native_weight_source=f"late_convex_{loss}",
            extra=extra,
        )
    raise ValueError(f"Unsupported late integration {integration!r}.")


def _complete_library_stack(
    scores: pd.Series, predictions: pd.DataFrame, pcols: list[str]
) -> tuple[list[str], pd.DataFrame, np.ndarray] | tuple[list[str], None, None]:
    library = [str(value) for value in scores.index.tolist()]
    if len(library) < 2:
        return library, None, None
    ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    if ordered is None or stack is None:
        incomplete = set(_missing_members(predictions, library, pcols, inner=True))
        library = [value for value in library if value not in incomplete]
        if len(library) < 2:
            return library, None, None
        ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    return library, ordered, stack


def _candidate_from_caruana(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    metric: str,
) -> dict[str, Any] | None:
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    members, native_weights, _, trajectory, diagnostics = _caruana_select(
        library, stack, y_true, metric, int(spec["max_size"])
    )
    selected_indices = [library.index(value) for value in members]
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
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "all_complete_inner_oof_mpmas",
            "candidate_library_size": len(library),
            "selection_weight_source": "caruana_selection_frequency",
            "caruana_trajectory": json.dumps([float(value) for value in trajectory]),
            **diagnostics,
        },
    )


def _candidate_from_bagged_caruana(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    metric: str,
) -> dict[str, Any] | None:
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    members, native_weights, diagnostics = _bagged_caruana_select(
        library, stack, y_true, metric, int(spec["max_size"])
    )
    selected_indices = [library.index(value) for value in members]
    selected_stack = stack[selected_indices]
    return _evaluate_selected_members(
        spec,
        members,
        selected_stack,
        y_true,
        metric,
        native_weights=native_weights,
        native_weight_source="bagged_caruana_mean_selection_weight",
        extra={
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "deterministic_model_subspace_bagging",
            "candidate_library_size": len(library),
            "selection_weight_source": "bagged_caruana_mean_selection_weight",
            **diagnostics,
        },
    )


def _candidate_from_caruana_multistart(
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
        library = [config_id for config_id in library if config_id not in incomplete]
        if len(library) < 2:
            return None
        ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    members, native_weights, _, trajectory, diagnostics = _caruana_multistart_select(
        library, stack, y_true, metric, int(spec["max_size"])
    )
    if len(members) < 2 or int(np.count_nonzero(native_weights > 1e-12)) < 2:
        return None
    indices = [library.index(config_id) for config_id in members]
    return _evaluate_selected_members(
        spec,
        members,
        stack[indices],
        y_true,
        metric,
        ordered=ordered,
        native_weights=native_weights,
        native_weight_source="caruana_multistart_selection_frequency",
        extra={
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "all_complete_inner_oof_mpmas",
            "candidate_library_size": len(library),
            "selection_weight_source": "caruana_multistart_selection_frequency",
            "caruana_multistart_trajectory": json.dumps(
                [float(value) for value in trajectory]
            ),
            **diagnostics,
        },
    )


def _candidate_from_beam(
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
        library = [config_id for config_id in library if config_id not in incomplete]
        if len(library) < 2:
            return None
        ordered, stack = _aligned_stack(predictions, library, pcols, inner=True)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    members, native_weights, _, trajectory, diagnostics = _beam_select(
        library, stack, y_true, metric, int(spec["max_size"])
    )
    if len(members) < 2 or int(np.count_nonzero(native_weights > 1e-12)) < 2:
        return None
    indices = [library.index(config_id) for config_id in members]
    return _evaluate_selected_members(
        spec,
        members,
        stack[indices],
        y_true,
        metric,
        ordered=ordered,
        native_weights=native_weights,
        native_weight_source="beam_selection_frequency",
        extra={
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "all_complete_inner_oof_mpmas",
            "candidate_library_size": len(library),
            "selection_weight_source": "beam_selection_frequency",
            "beam_trajectory": json.dumps([float(value) for value in trajectory]),
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
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    super_loss = _resolved_super_learner_loss(plan)
    members, native_weights, loss_value = _fit_super_learner(
        library, stack, y_true, super_loss, int(spec["max_size"])
    )
    selected_indices = [library.index(value) for value in members]
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
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "all_complete_inner_oof_mpmas",
            "candidate_library_size": len(library),
            "selection_weight_source": f"convex_{super_loss}",
            "super_learner_loss": super_loss,
            "super_learner_loss_rule": "brier_if_optimize_metric_is_brier_else_log_loss",
            "super_learner_loss_value": float(loss_value),
            "super_learner_internal_weight_tol": float(_SUPER_LEARNER_WEIGHT_TOL),
            "super_learner_internal_min_weight": float(_SUPER_LEARNER_MIN_WEIGHT),
            "super_learner_internal_optimizer": "SLSQP",
            "super_learner_internal_maxiter": int(_SUPER_LEARNER_OPT_MAXITER),
            "super_learner_internal_ftol": float(_SUPER_LEARNER_OPT_FTOL),
        },
    )


def _candidate_from_regularized_super_learner(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    plan: Ensemble,
    metric: str,
) -> dict[str, Any] | None:
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    members = library[: min(int(spec["max_size"]), len(library))]
    if len(members) < 2:
        return None
    selected_stack = stack[: len(members)]
    y_true = ordered["y_true"].to_numpy(dtype=int)
    folds = _stacking_fold_keys(ordered)
    unique = np.unique(folds)
    loss = _resolved_super_learner_loss(plan)
    anchor = np.full(len(members), 1.0 / float(len(members)))
    trace: list[dict[str, float]] = []
    best = None
    for strength in (0.03, 0.1, 0.3):
        heldout = np.zeros(selected_stack.shape[1:], dtype=float)
        valid = True
        for key in unique:
            training = folds != key
            testing = ~training
            try:
                weights = _fit_regularized_convex_probability_weights(
                    selected_stack[:, training],
                    y_true[training],
                    loss,
                    anchor,
                    float(strength),
                    min_weight=_SUPER_LEARNER_MIN_WEIGHT,
                )
                heldout[testing] = aggregate_member_predictions(
                    selected_stack[:, testing], "weighted_mean_proba", weights
                )
            except (RuntimeError, ValueError):
                valid = False
                break
        if not valid or np.any(heldout.sum(axis=1) <= 0):
            continue
        scored = float(_metric_value(y_true, heldout, metric))
        trace.append({"strength": float(strength), "score": scored})
        if best is None or _metric_better(scored, best[0], metric):
            best = (scored, float(strength), heldout)
    if best is None:
        raise RuntimeError(
            "Regularized Super Learner could not cross-fit a valid ensemble."
        )
    chosen_score, strength, heldout = best
    weights = _fit_regularized_convex_probability_weights(
        selected_stack,
        y_true,
        loss,
        anchor,
        strength,
        min_weight=_SUPER_LEARNER_MIN_WEIGHT,
    )
    result = _evaluate_selected_members(
        spec,
        members,
        selected_stack,
        y_true,
        metric,
        ordered=ordered,
        native_weights=weights,
        native_weight_source="regularized_super_learner",
        extra={
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "top_k_regularized_simplex",
            "candidate_library_size": len(library),
            "selection_weight_source": "regularized_super_learner",
            "super_learner_loss": loss,
            "regularized_super_learner_strength": strength,
            "stacking_crossfit_score": float(chosen_score),
            "stacking_internal_folds": len(unique),
            "stacking_search_trace": json.dumps(trace),
        },
    )
    classes = np.arange(heldout.shape[1], dtype=int)
    cv_metrics = compute_metrics(
        y_true, classes[np.argmax(heldout, axis=1)], heldout, classes
    )
    result.update({f"{key}_mean": float(value) for key, value in cv_metrics.items()})
    result[f"{metric}_mean"] = float(chosen_score)
    return result


def _candidate_from_adaptive_super_learner(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    plan: Ensemble,
    metric: str,
) -> dict[str, Any] | None:
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    super_loss = _resolved_super_learner_loss(plan)
    members, native_weights, loss_value, strength = _fit_adaptive_super_learner(
        library, stack, y_true, super_loss, int(spec["max_size"])
    )
    selected_indices = [library.index(value) for value in members]
    selected_stack = stack[selected_indices]
    return _evaluate_selected_members(
        spec,
        members,
        selected_stack,
        y_true,
        metric,
        native_weights=native_weights,
        native_weight_source=f"adaptive_convex_{super_loss}",
        extra={
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "best_member_anchored_complexity_adaptive_simplex",
            "candidate_library_size": len(library),
            "selection_weight_source": f"adaptive_convex_{super_loss}",
            "super_learner_loss": super_loss,
            "super_learner_loss_value": float(loss_value),
            "adaptive_regularization_strength": float(strength),
            "adaptive_regularization_anchor": str(library[0]),
            "super_learner_internal_min_weight": float(_SUPER_LEARNER_MIN_WEIGHT),
            "super_learner_internal_optimizer": "SLSQP",
        },
    )


def _candidate_from_safe_super_learner(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    plan: Ensemble,
    metric: str,
) -> dict[str, Any] | None:
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    super_loss = _resolved_super_learner_loss(plan)
    members, native_weights, loss_value = _fit_super_learner(
        library, stack, y_true, super_loss, int(spec["max_size"])
    )
    selected_indices = [library.index(value) for value in members]
    selected_stack = stack[selected_indices]
    candidate_proba = aggregate_member_predictions(
        selected_stack, "weighted_mean_proba", native_weights
    )
    baseline_proba = stack[0]
    accepted, gains = _stable_metric_improvement(
        ordered, candidate_proba, baseline_proba, metric
    )
    if not accepted:
        return None
    return _evaluate_selected_members(
        spec,
        members,
        selected_stack,
        y_true,
        metric,
        native_weights=native_weights,
        native_weight_source="safe_super_learner",
        extra={
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "stable_gain_over_best_single",
            "candidate_library_size": len(library),
            "selection_weight_source": "safe_super_learner",
            "super_learner_loss": super_loss,
            "super_learner_loss_value": float(loss_value),
            "super_learner_internal_min_weight": float(_SUPER_LEARNER_MIN_WEIGHT),
            "safe_super_learner_accepted": int(bool(accepted)),
            "safe_super_learner_split_gains": json.dumps(
                [float(value) for value in gains]
            ),
            "safe_super_learner_anchor": str(library[0]),
        },
    )


def _candidate_table_for_inner(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    outer_split_key: str | None,
    qualification: pd.DataFrame | None = None,
    progress_callback=None,
    progress_scope: str | None = None,
) -> pd.DataFrame:
    _validate_plan(plan)
    eligible = _eligible_config_ids(configs, plan)
    scores = _complete_inner_scores(
        inner_results,
        inner_predictions,
        outer_split_key,
        metric,
        eligible,
        qualification,
    )
    if scores.empty:
        return pd.DataFrame()
    pred = _prediction_frame_for_outer(inner_predictions, outer_split_key)
    pred = pred[pred["config_id"].isin(set(scores.index))].copy()
    pcols = _proba_cols(pred)
    if not pcols:
        raise ValueError("Inner predictions do not contain probability columns.")
    rows: list[dict[str, Any]] = []
    specs = _ensemble_configs(plan, configs)
    scope = str(progress_scope or outer_split_key or "__final__")
    for index, spec in enumerate(specs, start=1):
        if progress_callback is not None:
            progress_callback(scope, "start", spec, index, len(specs), "evaluating")
        row = None
        failed = False
        try:
            method = str(spec["selection_strategy"])
            if str(spec.get("ensemble_kind", "model_ensemble")) == "late_fusion":
                row = _candidate_from_late_fusion(
                    spec, scores, pred, configs, pcols, plan, metric
                )
            elif method in {
                "top_k",
                "best_per_resolution",
                "best_per_learner_type",
                "quality_diversity",
                "diversity",
                "performance_diversity",
            }:
                row = _candidate_from_simple_selector(
                    spec, scores, pred, configs, pcols, metric
                )
            elif method == "caruana":
                row = _candidate_from_caruana(spec, scores, pred, pcols, metric)
            elif method == "bagged_caruana":
                row = _candidate_from_bagged_caruana(spec, scores, pred, pcols, metric)
            elif method == "caruana_multistart":
                row = _candidate_from_caruana_multistart(
                    spec, scores, pred, pcols, metric
                )
            elif method == "beam":
                row = _candidate_from_beam(spec, scores, pred, pcols, metric)
            elif method == "super_learner":
                row = _candidate_from_super_learner(
                    spec, scores, pred, pcols, plan, metric
                )
            elif method == "regularized_super_learner":
                row = _candidate_from_regularized_super_learner(
                    spec, scores, pred, pcols, plan, metric
                )
            elif method == "adaptive_super_learner":
                row = _candidate_from_adaptive_super_learner(
                    spec, scores, pred, pcols, plan, metric
                )
            elif method == "safe_super_learner":
                row = _candidate_from_safe_super_learner(
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


def _parse_json_dict(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None or str(value).strip() in {"", "nan", "None"}:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception as exc:
        raise ValueError(f"Could not parse ensemble {field} JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Ensemble {field} must decode to an object.")
    return parsed


def select_mpma_e_by_outer_fold(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    outer_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    qualification: pd.DataFrame | None = None,
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
            qualification=qualification,
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
                len(_ensemble_configs(plan, configs)),
                len(_ensemble_configs(plan, configs)),
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
        aggregation_parameters = _parse_json_dict(
            winner.get("aggregation_parameters", "{}"), "aggregation_parameters"
        )

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
        proba = aggregate_member_predictions(
            stack, aggregation, active_weights, aggregation_parameters
        )
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
                "ensemble_size": len(members),
                "member_count": len(members),
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
                "aggregation_parameters": json.dumps(aggregation_parameters),
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
        ordered["aggregation_parameters"] = json.dumps(aggregation_parameters)
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
        "n_outer_folds": len(fold_metrics),
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
        out[f"outer_{metric}_count"] = len(vals)
    return out


def select_final_mpma_e_candidate(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    member_score_metric: str | None = None,
    qualification: pd.DataFrame | None = None,
    progress_callback=None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = _candidate_table_for_inner(
        inner_results,
        inner_predictions,
        configs,
        plan,
        metric,
        None,
        qualification=qualification,
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
            len(_ensemble_configs(plan, configs)),
            len(_ensemble_configs(plan, configs)),
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
        "aggregation_parameters": _parse_json_dict(
            first.get("aggregation_parameters", "{}"), "aggregation_parameters"
        ),
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
    if any(not table_exists(path) for path in required):
        raise FileNotFoundError("Run evaluate(sweep) before sweep_ensemble(sweep).")
    outer_predictions = read_table(outer_path)
    inner_results = read_table(inner_result_path)
    inner_predictions = read_table(inner_prediction_path)
    configs = read_table(config_path)
    qualification = None
    if sweep.gate.enabled:
        qualification_path = root / "tables" / "qualification_gate.parquet"
        if not table_exists(qualification_path):
            raise FileNotFoundError(
                "Qualification gate is enabled but qualification_gate.parquet is missing. Rerun evaluate(sweep)."
            )
        qualification = read_table(qualification_path)
    metric = canonical_metric_name(str(sweep.ensemble.optimize_metric))
    stage("Ensemble sweep", str(root))
    summary_table(
        "Ensemble search",
        {
            "candidate ensemble configurations": f"{len(_ensemble_configs(sweep.ensemble, configs)):,}",
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
    ensemble_specs = _ensemble_configs(sweep.ensemble, configs)
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
                qualification=qualification,
                progress_callback=live.update,
            )
        )
        if selection.empty or selected_outer_predictions.empty or fold_metrics.empty:
            raise RuntimeError("No valid nested ensemble selections were produced.")
        member_score_metric = canonical_metric_name(
            str(sweep.evaluation.optimize_metric)
        )
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
            qualification=qualification,
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
        canonical_metric_name(str(sweep.evaluation.optimize_metric)),
        plan=sweep.ensemble,
        qualification=qualification,
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
    resolved_metadata_path = metadata_path(spec)
    fmt = str(spec.format).strip().casefold().replace("-", "_")
    if fmt == "auto":
        fmt = (
            "mllab"
            if abundance_path.suffix.lower() in {".tsv", ".txt"}
            and resolved_metadata_path
            else "wide_csv"
        )
    if fmt in {"mllab", "matrix_tsv", "metaphlan", "metaphlan_tsv", "profile_tsv"}:
        if resolved_metadata_path is None:
            raise ValueError(
                "Data.metadata=Metadata(metadata_path=...) is required for mllab TSV input."
            )
        meta = pd.read_csv(resolved_metadata_path, sep=None, engine="python", dtype=str)
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
        if resolved_metadata_path is not None:
            meta = pd.read_csv(resolved_metadata_path, sep=None, engine="python")
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
        target_col = str(spec.target_col)
        feature_columns = _wide_csv_feature_columns(spec, df, target_col)
        return (
            df[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32),
            [str(c) for c in feature_columns],
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
