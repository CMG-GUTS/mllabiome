from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from .advanced_ensemble import ADVANCED_AGGREGATIONS, fit_crossfitted_advanced
from .modern_selection import drep_select, population_ensemble_select, pseo_bqp_select
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
    ecdf_stacking_features,
)
from .ensemble_progress import EnsembleSearchProgress
from .integrations import integration_modality_sets
from .learner_families import learner_family_from_spec
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
from .ensemble_quantile_learning import fit_grouped_quantile
from .selection_policy import rank_auroc_log_loss
from .pareto_selection import rank_pareto_auroc_logloss
from .selection import _qualified_config_ids_for_splits, select_final_mpma_candidate
from .mpma_vnext import rank_oof_loss, similarity_members
from .storage import read_table, table_exists, write_table
from .utils import TAXONOMIC_LEVELS, dump_json_standard

_ENSEMBLE_SEARCH_SCHEMA = "mpmae_search_v20"
_PARETO_SELECTORS = frozenset(
    {
        "pareto_optimal_auroc_logloss",
        "pareto_optimal_ml_dl_auroc_logloss",
        "pareto_optimal_ml_dl_auroc_logloss_capped",
        "pareto_optimal_taxonomic_rank_auroc_logloss",
        "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
    }
)
_PARETO_AGGREGATIONS = frozenset(
    {
        "mean_proba",
        "bayesian_model_averaging",
        "weighted_mean_proba",
        "cawpe",
    }
)
_SUPER_LEARNER_WEIGHT_TOL = 1e-8
_SUPER_LEARNER_MIN_WEIGHT = 0.01
_SUPER_LEARNER_OPT_MAXITER = 1000
_SUPER_LEARNER_OPT_FTOL = 1e-12
_CARUANA_IMPROVEMENT_TOL = 1e-10
_BAGGED_CARUANA_MIN_BAGS = 8
_BAGGED_CARUANA_MAX_BAGS = 24


def _proba_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("proba_")]


def _probability_diagnostics(
    predictions: pd.DataFrame,
    probability_columns: list[str],
    tolerance: float,
    source: str = "pooled_inner_oof",
    group_columns: tuple[str, ...] = ("config_id",),
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "config_id",
                "source",
                "probability_range",
                "probability_std",
                "near_constant",
            ]
        )
    if not probability_columns:
        raise ValueError("Probability diagnostics require probability columns.")
    columns = list(dict.fromkeys(str(value) for value in group_columns))
    if "config_id" not in columns or any(
        column not in predictions.columns for column in columns
    ):
        raise ValueError(
            "Probability diagnostics require config_id and valid group columns."
        )
    values = predictions[[*columns, *probability_columns]].copy()
    values["config_id"] = values["config_id"].astype(str)
    if not np.isfinite(values[probability_columns].to_numpy(dtype=float)).all():
        raise ValueError("Probability diagnostics found nonfinite predictions.")
    grouped = values.groupby(columns, sort=True)[probability_columns]
    minima = grouped.min()
    maxima = grouped.max()
    standard_deviations = grouped.std(ddof=0).fillna(0.0)
    spread = maxima - minima
    frame = spread.index.to_frame(index=False).reset_index(drop=True)
    frame["source"] = source
    frame["n_prediction_rows"] = grouped.size().to_numpy(dtype=int)
    frame["probability_range"] = spread.max(axis=1).to_numpy(dtype=float)
    frame["probability_std"] = standard_deviations.max(axis=1).to_numpy(dtype=float)
    frame["minimum_probability"] = minima.min(axis=1).to_numpy(dtype=float)
    frame["maximum_probability"] = maxima.max(axis=1).to_numpy(dtype=float)
    frame["near_constant"] = frame["probability_range"].le(float(tolerance))
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def _rank_pooled_grid_candidates(
    out: pd.DataFrame,
    metric: str,
    auroc_tolerance: float | None,
    priority: tuple[str, ...],
    final_selection_strategy: str = "single_metric",
) -> pd.DataFrame:
    score_col = f"{metric}_mean"
    if "member_count" not in out.columns:
        out["member_count"] = np.nan
    pareto = final_selection_strategy == "pareto_auroc_logloss_ideal"
    if final_selection_strategy not in {"single_metric", "pareto_auroc_logloss_ideal"}:
        raise ValueError(
            f"Unsupported final ensemble selection strategy: {final_selection_strategy!r}"
        )
    if pareto and auroc_tolerance is not None:
        raise ValueError(
            "Pareto final ensemble selection cannot be combined with auroc_tolerance."
        )
    if pareto:
        required = {"AUROC_mean", "log_loss_mean", "ensemble_config_id"}
        if not required.issubset(out.columns):
            raise ValueError(
                f"Pooled Pareto ranking requires columns: {sorted(required - set(out.columns))}"
            )
        for column, failure in (
            ("AUROC_mean", "invalid_auroc"),
            ("log_loss_mean", "invalid_log_loss"),
        ):
            values = pd.to_numeric(out[column], errors="coerce")
            invalid = ~np.isfinite(values)
            out.loc[invalid & out["search_status"].eq("valid"), "search_status"] = (
                failure
            )
    elif auroc_tolerance is not None:
        if "log_loss_mean" not in out.columns:
            raise ValueError(
                "Pooled AUROC/log-loss tolerance requires candidate log_loss_mean."
            )
        losses = pd.to_numeric(out["log_loss_mean"], errors="coerce")
        invalid_losses = ~np.isfinite(losses)
        out.loc[invalid_losses & out["search_status"].eq("valid"), "search_status"] = (
            "invalid_log_loss"
        )
    valid_mask = out["search_status"].eq("valid")
    valid_rows = out.loc[valid_mask].copy()
    invalid_rows = out.loc[~valid_mask].copy()
    ascending = [_metric_is_loss(metric), *([True] * len(priority))]
    if pareto:
        if not valid_rows.empty:
            valid_rows = rank_pareto_auroc_logloss(
                valid_rows, "AUROC_mean", "log_loss_mean", "ensemble_config_id"
            )
        else:
            valid_rows["pareto_optimal"] = pd.Series(dtype=bool)
            valid_rows["pareto_ideal_distance"] = pd.Series(dtype=float)
            valid_rows["pareto_rank"] = pd.Series(dtype=int)
    elif auroc_tolerance is not None and not valid_rows.empty:
        valid_rows = rank_auroc_log_loss(
            valid_rows, score_col, "log_loss_mean", float(auroc_tolerance), priority
        )
    else:
        valid_rows = valid_rows.sort_values(
            [score_col, *priority],
            ascending=ascending,
            kind="mergesort",
            na_position="last",
        )
    invalid_rows = invalid_rows.sort_values(
        [score_col, *priority],
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    )
    if pareto:
        invalid_rows["pareto_optimal"] = False
        invalid_rows["pareto_ideal_distance"] = np.nan
        invalid_rows["pareto_rank"] = np.nan
    out = pd.concat([valid_rows, invalid_rows], ignore_index=True)
    out["selection_rank"] = np.nan
    if len(valid_rows):
        out.loc[: len(valid_rows) - 1, "selection_rank"] = np.arange(
            1, len(valid_rows) + 1
        )
    out["selected"] = out["selection_rank"].eq(1).astype(int)
    out["selection_rule"] = (
        "pareto_auroc_logloss_ideal"
        if pareto
        else "auroc_constrained_log_loss"
        if auroc_tolerance is not None
        else "single_metric"
    )
    out["final_selection_strategy"] = final_selection_strategy
    out["auroc_tolerance"] = auroc_tolerance
    return out.drop(
        columns=["_method_priority", "_aggregation_priority"], errors="ignore"
    ).reset_index(drop=True)


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
        not bool(getattr(ensemble, "include_inactive", True))
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
    if metric == "auroc_mcc":
        return "AUROC_MCC"
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
    if str(getattr(plan, "selection_mode", "nested")) not in {"nested", "pooled_grid"}:
        raise ValueError("Ensemble.selection_mode must be nested or pooled_grid.")
    if str(
        getattr(plan, "selection_mode", "nested")
    ) == "pooled_grid" and _late_integrations(plan):
        raise ValueError(
            "Pooled ensemble grid search does not support late integrations."
        )
    if str(getattr(plan, "selection_mode", "nested")) == "pooled_grid" and set(
        _plan_aggregations(plan)
    ) & {"ecdf_grouped_stacking", "ecdf_learned_quantile"}:
        raise ValueError(
            "Pooled ensemble grid does not support aggregations that require separate development OOF fitting."
        )
    supported_selection = {
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "best_deep_ml_pair",
        "best_deep_ml_complementary_pair",
        "quality_diversity",
        "oof_top_k",
        "oof_similarity",
        "diversity",
        "performance_diversity",
        "pareto_optimal_auroc_logloss",
        "pareto_optimal_ml_dl_auroc_logloss",
        "pareto_optimal_ml_dl_auroc_logloss_capped",
        "pareto_optimal_taxonomic_rank_auroc_logloss",
        "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
        "caruana",
        "caruana_constrained",
        "bagged_caruana",
        "caruana_multistart",
        "beam",
        "super_learner",
        "adaptive_super_learner",
        "safe_super_learner",
        "regularized_super_learner",
        "qo_es",
        "qdo_es",
        "drep",
        "pseo_bqp",
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
        if set(methods) & _PARETO_SELECTORS:
            if not set(aggregations) & _PARETO_AGGREGATIONS:
                raise ValueError(
                    "Pareto ensembles need mean_proba, bayesian_model_averaging, weighted_mean_proba, or cawpe."
                )
            if (
                plan.auroc_tolerance is not None
                and str(plan.selection_mode) != "pooled_grid"
            ):
                raise ValueError(
                    "Pareto ensembles with an AUROC tolerance require selection_mode='pooled_grid'."
                )
        if not aggregations:
            raise ValueError("Ensemble.aggregation_strategies must be non-empty.")
        unknown_aggregations = sorted(set(aggregations) - set(SUPPORTED_AGGREGATIONS))
        if unknown_aggregations:
            raise ValueError(
                f"Unknown ensemble aggregation strategy(s): {unknown_aggregations}. "
                f"Supported strategies: {list(SUPPORTED_AGGREGATIONS)}."
            )
        metric = canonical_metric_name(str(plan.optimize_metric))
        if {"oof_top_k", "oof_similarity"} & set(methods) and metric != "log_loss":
            raise ValueError(
                "OOF top-k and similarity mechanisms currently require optimize_metric='log_loss'"
            )
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
            "caruana_constrained",
            "bagged_caruana",
            "caruana_multistart",
            "beam",
            "super_learner",
            "adaptive_super_learner",
            "safe_super_learner",
            "regularized_super_learner",
            "qo_es",
            "qdo_es",
        }
        simple = set(methods) - learned_methods - _PARETO_SELECTORS
        if (
            set(methods) & {"qo_es", "qdo_es"}
            and "weighted_mean_proba" not in aggregations
        ):
            raise ValueError(
                "QO-ES and QDO-ES require weighted_mean_proba to preserve learned selection frequencies."
            )
        if set(methods).issubset(learned_methods) and set(aggregations) == {"cawpe"}:
            raise ValueError(
                "CAWPE requires a non-learned member selector such as top_k or best_per_learner_type; learned selectors already determine their own weights."
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
            "caruana_constrained",
            "bagged_caruana",
            "caruana_multistart",
            "beam",
            "super_learner",
            "adaptive_super_learner",
            "safe_super_learner",
            "regularized_super_learner",
            "qo_es",
            "qdo_es",
        }
        for method in _plan_methods(plan):
            sizes = (
                (int(plan.caruana_iterations),)
                if method == "caruana"
                else (2,)
                if method in {"best_deep_ml_pair", "best_deep_ml_complementary_pair"}
                else _plan_max_sizes(plan)
            )
            for max_size in sizes:
                for aggregation in _plan_aggregations(plan):
                    if (
                        method in _PARETO_SELECTORS
                        and aggregation not in _PARETO_AGGREGATIONS
                    ):
                        continue
                    if method in learned_weight_selectors:
                        if aggregation == "cawpe":
                            continue
                        if aggregation != "weighted_mean_proba" and (
                            "weighted_mean_proba" in _plan_aggregations(plan)
                            or aggregation in LEARNED_AGGREGATIONS
                            or aggregation
                            in {"ecdf_weighted_quantile", "ecdf_learned_quantile"}
                        ):
                            continue
                    elif (
                        aggregation == "weighted_mean_proba"
                        and method not in _PARETO_SELECTORS
                    ):
                        continue
                    if (
                        method in {"qo_es", "qdo_es"}
                        and aggregation != "weighted_mean_proba"
                    ):
                        continue
                    uid = "__".join(
                        [
                            _ENSEMBLE_SEARCH_SCHEMA,
                            "model_ensemble",
                            str(plan.optimize_metric),
                            method,
                            aggregation,
                            str(max_size),
                            json.dumps(
                                [plan.ecdf_member_weights, plan.ecdf_reference_learner]
                            )
                            if aggregation == "ecdf_weighted_quantile"
                            else "na",
                            _resolved_super_learner_loss(plan)
                            if method
                            in {
                                "super_learner",
                                "adaptive_super_learner",
                                "safe_super_learner",
                                "regularized_super_learner",
                            }
                            else "na",
                            json.dumps(
                                [
                                    plan.caruana_iterations,
                                    plan.caruana_max_unique_members,
                                    plan.caruana_early_stopping,
                                    plan.caruana_patience,
                                    plan.caruana_min_iterations,
                                    plan.caruana_sorted_initial_models,
                                ]
                            )
                            if method == "caruana"
                            else "na",
                            *(
                                [
                                    json.dumps(
                                        [
                                            plan.advanced_selection_pool_size,
                                            plan.population_iterations,
                                            plan.population_archive_size,
                                            plan.population_random_state,
                                            plan.qdo_archive_bins,
                                            plan.drep_rho,
                                            plan.pseo_bqp_omega,
                                            plan.pseo_bqp_swap_passes,
                                        ]
                                    )
                                ]
                                if method in {"qo_es", "qdo_es", "drep", "pseo_bqp"}
                                else []
                            ),
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
    auroc_tolerance: float | None = None,
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
    if outer_split_key is None and "split_key" in frame.columns:
        frame["inner_key"] = frame["split_key"].astype(str) + "::" + frame["inner_key"]
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

    if metric == "AUROC_MCC" and metric not in frame.columns:
        if not {"AUROC", "MCC"}.issubset(frame.columns):
            raise ValueError(
                "AUROC_MCC selection requires inner AUROC and MCC columns."
            )
        auc = pd.to_numeric(frame["AUROC"], errors="coerce")
        mcc = pd.to_numeric(frame["MCC"], errors="coerce")
        frame[metric] = 0.5 * (2.0 * auc - 1.0) + 0.5 * mcc
    rows: list[tuple[str, float]] = []
    loss_rows: dict[str, float] = {}
    if auroc_tolerance is not None and "log_loss" not in frame.columns:
        raise ValueError(
            "AUROC/log-loss ensemble selection requires inner log_loss results."
        )
    for config_id in complete:
        ordered, stack = _aligned_stack(pred, [config_id], pcols, inner=True)
        if ordered is None or stack is None:
            continue
        group = frame[frame["config_id"].eq(str(config_id))].copy()
        group = group.drop_duplicates("inner_key", keep="last")
        if auroc_tolerance is not None:
            observed = group[["AUROC", "log_loss"]].apply(
                pd.to_numeric, errors="coerce"
            )
            if not np.isfinite(observed.to_numpy(dtype=float)).all():
                continue
        value, _ = aggregate_validation_metric(group, metric)
        if not np.isfinite(value):
            continue
        if auroc_tolerance is not None:
            loss, _ = aggregate_validation_metric(group, "log_loss")
            if not np.isfinite(loss):
                continue
            loss_rows[str(config_id)] = float(loss)
        rows.append((str(config_id), float(value)))
    if not rows:
        return pd.Series(dtype=float)
    scores = pd.Series(dict(rows), dtype=float)
    if auroc_tolerance is not None:
        table = pd.DataFrame(
            {"config_id": scores.index, "AUROC": scores.to_numpy(dtype=float)}
        )
        table["log_loss"] = table["config_id"].map(loss_rows)
        ranked = rank_auroc_log_loss(
            table, "AUROC", "log_loss", auroc_tolerance, ("config_id",)
        )
        return scores.loc[ranked["config_id"].tolist()]
    if _metric_is_loss(metric):
        order = sorted(scores.index, key=lambda cid: (float(scores[cid]), str(cid)))
    else:
        order = sorted(scores.index, key=lambda cid: (-float(scores[cid]), str(cid)))
    return scores.loc[order]


def _build_pooled_grid_matrix(
    predictions: pd.DataFrame,
    model_ids: list[str],
    progress_detail=None,
    max_bytes: int = 900_000_000,
) -> bool:
    pcols = _proba_cols(predictions)
    key_cols = ["outer_split_key", "split_key", "sample_id"]
    extra_cols = ["study_group"] if "study_group" in predictions.columns else []
    ordered = (
        predictions[key_cols + ["y_true"] + extra_cols]
        .drop_duplicates(key_cols, keep="last")
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    n_models = len(model_ids)
    n_samples = len(ordered)
    total = n_models * n_samples
    required_bytes = total * len(pcols) * 8
    if required_bytes > max_bytes:
        if progress_detail is not None:
            progress_detail(
                f"Aligned matrix needs {required_bytes / 1_000_000_000:.2f} GB; using on-demand alignment"
            )
        return False
    if progress_detail is not None:
        progress_detail(
            f"Indexing {n_models:,} models × {n_samples:,} samples into one reusable matrix"
        )
    model_index = pd.Index(model_ids)
    row_model = model_index.get_indexer(predictions["config_id"].astype(str))
    sample_index = pd.MultiIndex.from_frame(ordered[key_cols])
    row_sample = sample_index.get_indexer(
        pd.MultiIndex.from_frame(predictions[key_cols])
    )
    if np.any(row_model < 0) or np.any(row_sample < 0):
        raise ValueError("Pooled grid matrix contains unrecognized models or samples")
    offsets = row_model.astype(np.int64) * n_samples + row_sample
    counts = np.bincount(offsets, minlength=total)
    if len(counts) != total or np.any(counts != 1):
        raise ValueError(
            "Pooled grid matrix requires exactly one prediction per model and sample"
        )
    matrix = np.empty((n_models, n_samples, len(pcols)), dtype=float)
    matrix[row_model, row_sample] = predictions[pcols].to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Pooled grid matrix contains invalid probabilities")
    predictions.attrs["_pooled_grid_matrix"] = {
        "frame_id": id(predictions),
        "members": model_ids,
        "positions": {name: index for index, name in enumerate(model_ids)},
        "ordered": ordered,
        "matrix": matrix,
        "pcols": pcols,
    }
    return True


def _aligned_stack(
    predictions: pd.DataFrame, members: list[str], pcols: list[str], *, inner: bool
) -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    if not members:
        return (None, None)
    if inner:
        cache = predictions.attrs.get("_pooled_grid_matrix")
        if (
            cache is not None
            and cache["frame_id"] == id(predictions)
            and list(pcols) == cache["pcols"]
        ):
            positions = cache["positions"]
            if not all(str(member) in positions for member in members):
                return (None, None)
            names = cache["members"]
            if len(members) == len(names) and all(
                str(member) == name for member, name in zip(members, names)
            ):
                return cache["ordered"], cache["matrix"]
            indices = [positions[str(member)] for member in members]
            return cache["ordered"], cache["matrix"][indices]
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
    extra_cols = ["study_group"] if "study_group" in predictions.columns else []
    ordered = (
        predictions[key_cols + ["y_true"] + extra_cols]
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
    if metric == "AUROC_MCC":
        if proba.shape[1] != 2:
            raise ValueError("AUROC_MCC requires binary classification probabilities.")
        truth = np.asarray(y_true, dtype=int)
        positives = truth == 1
        n_pos = int(np.count_nonzero(positives))
        n_neg = len(truth) - n_pos
        if not n_pos or not n_neg:
            return float("-inf")
        ranks = rankdata(proba[:, 1], method="average")
        auc = (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        predicted = proba[:, 1] > proba[:, 0]
        tp = int(np.count_nonzero(positives & predicted))
        fp = int(np.count_nonzero(~positives & predicted))
        fn = n_pos - tp
        tn = n_neg - fp
        denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
        return round(float(0.5 * (2.0 * auc - 1.0) + 0.5 * mcc), 6)
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


def _cawpe_member_weights(
    stack: np.ndarray, y_true: np.ndarray, alpha: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(stack, dtype=float)
    truth = np.asarray(y_true)
    if predictions.ndim != 3 or predictions.shape[0] < 1 or predictions.shape[2] < 2:
        raise ValueError(
            "CAWPE requires member probabilities with n_members x n_samples x n_classes shape."
        )
    if truth.ndim != 1 or predictions.shape[1] != truth.size or truth.size < 1:
        raise ValueError("CAWPE requires one inner-OOF label per prediction row.")
    if (
        not np.issubdtype(truth.dtype, np.integer)
        or np.any(truth < 0)
        or np.any(truth >= predictions.shape[2])
    ):
        raise ValueError(
            "CAWPE requires class-index labels aligned with probability columns."
        )
    if not np.isfinite(predictions).all() or np.any(predictions < 0.0):
        raise ValueError("CAWPE requires finite, nonnegative inner-OOF probabilities.")
    if np.any(predictions.sum(axis=2) <= 0.0):
        raise ValueError("CAWPE requires strictly positive row probability mass.")
    exponent = float(alpha)
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("CAWPE exponent must be positive and finite.")
    accuracies = np.mean(predictions.argmax(axis=2) == truth[None, :], axis=1)
    with np.errstate(under="ignore"):
        raw = np.power(accuracies, exponent)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "CAWPE cannot compute weights: all candidate CV accuracies are zero."
        )
    return raw / total, accuracies


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
    aggregation_parameters_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    is_pareto = str(spec.get("selection_strategy")) in _PARETO_SELECTORS
    allow_singleton = (
        is_pareto and str(spec.get("aggregation_strategy")) in _PARETO_AGGREGATIONS
    ) or str(spec.get("selection_strategy")) == "caruana"
    if len(members) < (1 if allow_singleton else 2) or len(set(members)) != len(
        members
    ):
        raise ValueError(
            "MPMA-E requires distinct members (singleton allowed for Pareto or canonical Caruana)."
        )
    if stack.shape[0] != len(members):
        raise ValueError("MPMA-E member count and prediction stack disagree.")
    aggregation = str(spec["aggregation_strategy"])
    probabilistic_methods = {
        "bayesian_model_averaging",
        "mixture_of_experts",
        "cross_study_dro",
        "sparse_nonnegative_stacking",
    }
    samples_deduplicated = False
    if (
        (aggregation in probabilistic_methods or is_pareto)
        and ordered is not None
        and "sample_id" in ordered.columns
        and ordered["sample_id"].duplicated().any()
    ):
        identifiers = ordered["sample_id"].astype(str).to_numpy()
        unique = np.unique(identifiers)
        rows = []
        averages = []
        for identifier in unique:
            positions = np.flatnonzero(identifiers == identifier)
            frame = ordered.iloc[positions]
            if frame["y_true"].nunique() != 1:
                raise ValueError(
                    "Repeated OOF predictions conflict on the true class for a sample."
                )
            if "study_group" in frame and frame["study_group"].nunique() != 1:
                raise ValueError(
                    "Repeated OOF predictions conflict on the study assignment for a sample."
                )
            rows.append(frame.iloc[0].to_dict())
            averages.append(stack[:, positions, :].mean(axis=1))
        ordered = pd.DataFrame(rows).reset_index(drop=True)
        stack = np.stack(averages, axis=1)
        y_true = ordered["y_true"].to_numpy(dtype=int)
        samples_deduplicated = True
    active_weights: np.ndarray | None = None
    aggregation_parameters: dict[str, Any] | None = aggregation_parameters_override
    aggregation_weight_source = (
        "configuration"
        if aggregation_parameters_override is not None
        else "not_applicable"
    )
    aggregation_extra: dict[str, Any] = {}
    crossfit_proba: np.ndarray | None = None
    if aggregation == "cawpe":
        if native_weights is not None:
            raise ValueError(
                "CAWPE cannot be combined with weights from a learned selector."
            )
        active_weights, accuracies = _cawpe_member_weights(stack, y_true)
        aggregation_parameters = {
            "kind": "cawpe",
            "alpha": 4.0,
            "cv_accuracy": [float(value) for value in accuracies],
            "cv_prediction_rows": int(len(y_true)),
            "accuracy_source": "inner_oof_argmax_accuracy",
        }
        aggregation_weight_source = "inner_oof_accuracy_power_4"
        if int(np.count_nonzero(active_weights > 1e-12)) < (
            1 if allow_singleton else 2
        ):
            raise ValueError(
                "CAWPE requires at least two members with positive cross-validated accuracy."
            )
    elif aggregation == "weighted_mean_proba":
        if native_weights is None and is_pareto:
            if len(members) == 1:
                native_weights = np.ones(1, dtype=float)
                crossfit_proba = stack[0].copy()
                aggregation_weight_source = "singleton_identity"
            else:
                if ordered is None:
                    raise ValueError(
                        "Pareto weighted averaging needs aligned inner OOF metadata."
                    )
                crossfit_proba, native_weights, aggregation_extra = (
                    _fit_pareto_weighted_average(
                        stack, y_true, ordered, metric, samples_deduplicated
                    )
                )
                aggregation_weight_source = (
                    "crossfitted_convex_log_loss"
                    if metric not in {"brier", "brier_loss"}
                    else "crossfitted_convex_brier"
                )
        if native_weights is None:
            raise ValueError(
                f"{spec['selection_strategy']} does not define weights required by weighted_mean_proba."
            )
        active_weights = np.asarray(native_weights, dtype=float)
        if (
            active_weights.shape != (len(members),)
            or not np.isfinite(active_weights).all()
            or np.any(active_weights < 0.0)
            or int(np.count_nonzero(active_weights > _SUPER_LEARNER_WEIGHT_TOL))
            < (1 if allow_singleton else 2)
        ):
            raise ValueError(
                "Weighted MPMA-E does not contain the required positive-weight members."
            )
        if crossfit_proba is not None:
            aggregation_weight_source = (
                native_weight_source or aggregation_weight_source
            )
        else:
            aggregation_weight_source = native_weight_source or "learned"
    elif aggregation == "mean_proba":
        active_weights = effective_aggregation_weights(aggregation, len(members))
        aggregation_weight_source = "uniform"
    if (
        aggregation == "bayesian_model_averaging"
        and allow_singleton
        and len(members) == 1
    ):
        aggregation_parameters = {
            "kind": "bayesian_model_averaging",
            "n_members": 1,
            "n_classes": int(stack.shape[2]),
            "member_weights": [1.0],
            "weight_method": "singleton_identity",
        }
        proba = stack[0].copy()
        aggregation_weight_source = "singleton_identity"
    elif crossfit_proba is not None:
        proba = crossfit_proba
    elif aggregation == "ecdf_learned_quantile":
        if ordered is None or "split_key" not in ordered.columns:
            raise ValueError(
                "Learned quantile aggregation requires held-out group assignments."
            )
        proba, aggregation_parameters, aggregation_extra = fit_grouped_quantile(
            stack,
            y_true,
            ordered["split_key"].to_numpy(dtype=str),
        )
        aggregation_weight_source = "study_heldout_optimization"
    elif aggregation in LEARNED_AGGREGATIONS:
        if ordered is None:
            raise ValueError(f"{aggregation} requires aligned inner-fold metadata.")
        if aggregation in ADVANCED_AGGREGATIONS:
            proba, aggregation_parameters, aggregation_extra = fit_crossfitted_advanced(
                stack,
                y_true,
                ordered["study_group"].astype(str).to_numpy()
                if aggregation == "cross_study_dro" and "study_group" in ordered.columns
                else np.asarray(
                    [
                        f"sample_fold_{i % min(5, len(ordered))}"
                        for i in range(len(ordered))
                    ]
                )
                if aggregation in probabilistic_methods and samples_deduplicated
                else _probabilistic_meta_fold_keys(ordered, aggregation)
                if aggregation in probabilistic_methods
                else _stacking_fold_keys(ordered),
                aggregation,
                metric,
                _metric_value,
                _metric_is_loss,
                ordered["study_group"].to_numpy(dtype=str)
                if aggregation == "cross_study_dro" and "study_group" in ordered.columns
                else None,
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
        "inner_oof_unique_rows": int(len(y_true)),
        "inner_oof_repeated_samples_collapsed": int(samples_deduplicated),
        "effective_member_count": int(
            np.count_nonzero(np.asarray(stored_weights, dtype=float) > 1e-12)
        )
        if stored_weights
        else int(
            np.count_nonzero(
                np.asarray(aggregation_parameters["member_weights"], dtype=float)
                > 1e-12
            )
        )
        if aggregation_parameters is not None
        and "member_weights" in aggregation_parameters
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


def _deep_ml_candidates(
    scores: pd.Series, configs: pd.DataFrame
) -> tuple[list[str], list[str]]:
    required = {"config_id", "learner_class", "learner_params"}
    if not required.issubset(configs.columns):
        return [], []
    meta = configs.drop_duplicates("config_id", keep="last").set_index("config_id")
    families: dict[str, list[str]] = {"deep_learning": [], "machine_learning": []}
    for value in scores.index:
        config_id = str(value)
        if config_id not in meta.index:
            continue
        row = meta.loc[config_id]
        family = learner_family_from_spec(
            str(row["learner_class"]) if pd.notna(row["learner_class"]) else "",
            row["learner_params"] if pd.notna(row["learner_params"]) else None,
        )
        if family in families:
            families[family].append(config_id)
    return families["deep_learning"], families["machine_learning"]


def _select_deep_ml_pair(
    scores: pd.Series,
    predictions: pd.DataFrame,
    configs: pd.DataFrame,
    pcols: list[str],
    metric: str,
    complementary: bool,
) -> tuple[list[str], pd.DataFrame | None, np.ndarray | None, dict[str, Any]]:
    deep, machine = _deep_ml_candidates(scores, configs)
    extra: dict[str, Any] = {
        "learner_family_source": "persisted_estimator_class_and_nested_parameters",
        "deep_learning_candidates": len(deep),
        "machine_learning_candidates": len(machine),
    }
    if not deep or not machine:
        return [], None, None, extra
    if not complementary:
        members = [deep[0], machine[0]]
        ordered, stack = _aligned_stack(predictions, members, pcols, inner=True)
        return (
            (members, ordered, stack, extra)
            if ordered is not None and stack is not None
            else ([], None, None, extra)
        )
    search_limit = 20
    deep = deep[:search_limit]
    machine = machine[:search_limit]
    library = deep + machine
    common, aligned = _aligned_stack(predictions, library, pcols, inner=True)
    chosen: tuple[str, str] | None = None
    best_value: float | None = None
    best_ordered: pd.DataFrame | None = None
    best_stack: np.ndarray | None = None
    if common is not None and aligned is not None:
        labels = common["y_true"].to_numpy(dtype=int)
        positions = {config_id: index for index, config_id in enumerate(library)}
        for dl_id in deep:
            for ml_id in machine:
                stack = aligned[[positions[dl_id], positions[ml_id]]]
                probability = (stack[0] + stack[1]) / 2.0
                value = _metric_value(labels, probability, metric)
                if np.isfinite(value) and (
                    best_value is None or _metric_better(value, best_value, metric)
                ):
                    best_value = float(value)
                    chosen = (dl_id, ml_id)
                    best_ordered = common
                    best_stack = stack
    else:
        for dl_id in deep:
            for ml_id in machine:
                members = [dl_id, ml_id]
                ordered, stack = _aligned_stack(predictions, members, pcols, inner=True)
                if ordered is None or stack is None:
                    continue
                probability = (stack[0] + stack[1]) / 2.0
                value = _metric_value(
                    ordered["y_true"].to_numpy(dtype=int), probability, metric
                )
                if not np.isfinite(value):
                    continue
                if best_value is None or _metric_better(value, best_value, metric):
                    best_value = float(value)
                    chosen = (dl_id, ml_id)
                    best_ordered = ordered
                    best_stack = stack
    extra["candidate_library_policy"] = (
        "top_20_each_family_inner_oof_equal_probability_pair"
    )
    extra["pair_search_deep_count"] = len(deep)
    extra["pair_search_ml_count"] = len(machine)
    extra["pair_search_evaluated_on"] = "inner_validation_predictions_only"
    if chosen is None:
        return [], None, None, extra
    extra["pair_selection_score"] = best_value
    return list(chosen), best_ordered, best_stack, extra


def _member_error_correlation(stack: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    actual = np.asarray(y_true)
    if actual.ndim == 2:
        target = np.asarray(actual, dtype=float)
        if target.shape != stack.shape[1:]:
            raise ValueError(
                "Multi-target ensemble reference and predictions have incompatible dimensions."
            )
    else:
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
    score_function=None,
    minimize_score: bool | None = None,
) -> tuple[list[str], dict[str, Any]]:
    loss = _metric_is_loss(metric) if minimize_score is None else bool(minimize_score)
    scorer = (
        (lambda actual, proba: _metric_value(actual, proba, metric))
        if score_function is None
        else score_function
    )
    n_members = len(member_ids)
    limit = min(max(2, int(max_members)), n_members)
    corr = np.abs(_member_error_correlation(stack, y_true))
    quality = _rank_quality(n_members)
    best_members: list[int] | None = None
    best_score = float("inf") if loss else float("-inf")
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
            score = scorer(y_true, proba)
            evaluations.append(
                {"alpha": float(alpha), "size": int(size), "score": float(score)}
            )
            better = best_members is None or (
                float(score) < float(best_score) - 1e-12
                if loss
                else float(score) > float(best_score) + 1e-12
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


def _caruana_batch_scores(
    member_ids: list[str],
    stack: np.ndarray,
    running_sum: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    indices: list[int],
    denominator: float,
) -> list[tuple[float, float, int, str]]:
    metric = canonical_metric_name(metric)
    if metric not in {"AUROC", "AUROC_MCC", "log_loss", "brier"}:
        output = []
        for index in indices:
            proba = _renormalize_proba(
                (running_sum + stack[index]) / denominator, stack.shape[2]
            )
            score = _metric_value(y_true, proba, metric)
            picked = np.clip(
                proba[np.arange(len(y_true)), y_true], np.finfo(float).eps, 1.0
            )
            output.append(
                (score, float(-np.mean(np.log(picked))), index, member_ids[index])
            )
        return output
    truth = np.asarray(y_true, dtype=int)
    positions = np.arange(len(truth))
    positives = truth == 1
    n_pos = int(positives.sum())
    n_neg = len(truth) - n_pos
    output = []
    for start in range(0, len(indices), 192):
        batch = indices[start : start + 192]
        prob = (stack[batch] + running_sum[None, :, :]) / denominator
        row_sums = prob.sum(axis=2, keepdims=True)
        if np.any(prob < 0.0) or np.any(row_sums <= 0.0) or not np.isfinite(prob).all():
            raise ValueError("Caruana candidates contain invalid probabilities")
        prob /= row_sums
        picked = np.clip(prob[:, positions, truth], np.finfo(float).eps, 1.0)
        log_losses = -np.mean(np.log(picked), axis=1)
        if metric == "log_loss":
            values = log_losses
        elif metric == "brier":
            residual = prob.copy()
            residual[:, positions, truth] -= 1.0
            values = np.mean(np.sum(residual * residual, axis=2), axis=1)
        else:
            if prob.shape[2] != 2:
                values = np.array(
                    [_metric_value(truth, item, metric) for item in prob], dtype=float
                )
            elif n_pos == 0 or n_neg == 0:
                values = np.full(len(batch), float("-inf"))
            else:
                ranks = rankdata(prob[:, :, 1], axis=1, method="average")
                values = (
                    ranks[:, positives].sum(axis=1) - n_pos * (n_pos + 1) / 2.0
                ) / (n_pos * n_neg)
                if metric == "AUROC_MCC":
                    predicted = prob[:, :, 1] > prob[:, :, 0]
                    tp = np.sum(predicted & positives[None, :], axis=1)
                    fp = np.sum(predicted & ~positives[None, :], axis=1)
                    fn = n_pos - tp
                    tn = n_neg - fp
                    denominator_mcc = np.sqrt(
                        ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).astype(float)
                    )
                    mcc = np.divide(
                        tp * tn - fp * fn,
                        denominator_mcc,
                        out=np.zeros(len(batch), dtype=float),
                        where=denominator_mcc > 0,
                    )
                    values = 0.5 * (2.0 * values - 1.0) + 0.5 * mcc
        output.extend(
            (float(score), float(tie_loss), index, member_ids[index])
            for score, tie_loss, index in zip(values, log_losses, batch)
        )
    return output


def _caruana_canonical_select(
    member_ids: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    iterations: int,
    auroc_tolerance: float | None = None,
    max_unique_members: int | None = None,
    early_stopping: bool = False,
    patience: int = 25,
    min_iterations: int = 25,
    sorted_initial_models: int = 0,
) -> tuple[list[str], np.ndarray, float, list[float], dict[str, Any]]:
    if not member_ids or len(set(member_ids)) != len(member_ids):
        raise ValueError("Canonical Caruana requires distinct library identifiers.")
    if stack.ndim != 3 or stack.shape[0] != len(member_ids):
        raise ValueError("Caruana library and prediction stack are inconsistent.")
    if int(iterations) < 1 or (
        max_unique_members is not None and int(max_unique_members) < 1
    ):
        raise ValueError("Invalid canonical Caruana iteration or member limit.")
    if int(sorted_initial_models) < 0 or int(sorted_initial_models) > min(
        len(member_ids), int(iterations)
    ):
        raise ValueError(
            "Canonical Caruana sorted initialization exceeds the model library or iteration budget."
        )
    if max_unique_members is not None and int(sorted_initial_models) > int(
        max_unique_members
    ):
        raise ValueError(
            "Canonical Caruana sorted initialization exceeds the distinct-model cap."
        )
    running_sum = np.zeros_like(stack[0], dtype=float)
    chosen_indices: list[int] = []
    trajectory: list[float] = []
    trace: list[dict[str, Any]] = []
    prefixes: list[dict[str, float | int]] = []
    best_score = float("inf") if _metric_is_loss(metric) else float("-inf")
    best_prefix = 0
    stale_steps = 0
    stop_reason = "iteration_budget"
    previous_score: float | None = None
    previous_loss: float | None = None
    initial_order: list[int] = []
    if sorted_initial_models:
        initial_scores = _caruana_batch_scores(
            member_ids,
            stack,
            running_sum,
            y_true,
            metric,
            list(range(len(member_ids))),
            1.0,
        )
        initial_scores = [
            entry
            for entry in initial_scores
            if np.isfinite(entry[0]) and np.isfinite(entry[1])
        ]
        initial_scores.sort(
            key=lambda entry: (
                entry[0] if _metric_is_loss(metric) else -entry[0],
                entry[1],
                entry[2],
                entry[3],
            )
        )
        if len(initial_scores) < int(sorted_initial_models):
            raise RuntimeError(
                "Too few finite-scoring models for sorted Caruana initialization."
            )
        initial_order = [
            int(entry[2]) for entry in initial_scores[: int(sorted_initial_models)]
        ]
    for step in range(int(iterations)):
        distinct = set(chosen_indices)
        allowed = [
            index
            for index in range(len(member_ids))
            if max_unique_members is None
            or index in distinct
            or len(distinct) < int(max_unique_members)
        ]
        candidate_scores = _caruana_batch_scores(
            member_ids,
            stack,
            running_sum,
            y_true,
            metric,
            [initial_order[step]] if step < int(sorted_initial_models) else allowed,
            float(step + 1),
        )
        candidate_scores = [
            entry
            for entry in candidate_scores
            if np.isfinite(entry[0]) and np.isfinite(entry[1])
        ]
        if not candidate_scores:
            stop_reason = "no_finite_candidates"
            break
        if auroc_tolerance is not None:
            ranked = rank_auroc_log_loss(
                pd.DataFrame(
                    candidate_scores,
                    columns=["AUROC", "log_loss", "index", "config_id"],
                ),
                "AUROC",
                "log_loss",
                float(auroc_tolerance),
                ("index",),
            )
            winner = ranked.iloc[0]
            score, loss, selected_index = (
                float(winner["AUROC"]),
                float(winner["log_loss"]),
                int(winner["index"]),
            )
        else:
            candidate_scores.sort(
                key=lambda item: (
                    item[0] if _metric_is_loss(metric) else -item[0],
                    item[1],
                    item[2],
                    item[3],
                )
            )
            score, loss, selected_index, _ = candidate_scores[0]
        chosen_indices.append(int(selected_index))
        running_sum += stack[int(selected_index)]
        trajectory.append(float(score))
        eligible_prefix = len(chosen_indices) >= max(1, int(sorted_initial_models))
        if auroc_tolerance is not None:
            improved = False
            if eligible_prefix:
                prefixes.append(
                    {
                        "AUROC": float(score),
                        "log_loss": float(loss),
                        "length": len(chosen_indices),
                    }
                )
                ranked_prefixes = rank_auroc_log_loss(
                    pd.DataFrame(prefixes),
                    "AUROC",
                    "log_loss",
                    float(auroc_tolerance),
                    ("length",),
                )
                best = ranked_prefixes.iloc[0]
                next_best = int(best["length"])
                improved = next_best != best_prefix
                if improved:
                    best_prefix = next_best
                    best_score = float(best["AUROC"])
        else:
            improved = eligible_prefix and (
                best_prefix == 0
                or _metric_better(float(score), best_score, metric, tol=0.0)
            )
            if improved:
                best_prefix = len(chosen_indices)
                best_score = float(score)
        stale_steps = 0 if improved else stale_steps + 1
        trace.append(
            {
                "iteration": int(step + 1),
                "selected_config_id": str(member_ids[int(selected_index)]),
                "score": float(score),
                "log_loss": float(loss),
                "score_change": None
                if previous_score is None
                else float(score - previous_score),
                "log_loss_change": None
                if previous_loss is None
                else float(loss - previous_loss),
                "distinct_members": int(len(set(chosen_indices))),
                "best_prefix": int(best_prefix),
                "selected_as_best_prefix": bool(improved),
                "sorted_initialization": bool(step < int(sorted_initial_models)),
            }
        )
        previous_score = float(score)
        previous_loss = float(loss)
        if (
            early_stopping
            and step + 1 >= int(min_iterations)
            and stale_steps >= int(patience)
        ):
            stop_reason = "converged_patience"
            break
    if best_prefix < 1:
        raise RuntimeError("Canonical Caruana did not produce any finite candidate.")
    selected = chosen_indices[:best_prefix]
    counts: dict[int, int] = {}
    for index in selected:
        counts[index] = counts.get(index, 0) + 1
    ordered_indices = list(counts)
    members = [member_ids[index] for index in ordered_indices]
    weights = np.asarray(
        [counts[index] / float(best_prefix) for index in ordered_indices], dtype=float
    )
    diagnostics = {
        "caruana_variant": "canonical_replacement",
        "caruana_iterations_executed": int(len(chosen_indices)),
        "caruana_iterations_selected": int(best_prefix),
        "caruana_stop_reason": stop_reason,
        "caruana_iteration_budget": int(iterations),
        "caruana_sorted_initial_models": int(sorted_initial_models),
        "caruana_max_unique_members": max_unique_members,
        "caruana_early_stopping": bool(early_stopping),
        "caruana_patience": int(patience) if early_stopping else None,
        "caruana_min_iterations": int(min_iterations) if early_stopping else None,
        "caruana_selection_trace": json.dumps(trace, separators=(",", ":")),
        "caruana_selected_counts": json.dumps(
            {member_ids[index]: counts[index] for index in ordered_indices}
        ),
    }
    return members, weights, float(best_score), trajectory[:best_prefix], diagnostics


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
        denom = float(step + 1)
        allowed = [
            index
            for index in range(len(member_ids))
            if not (len(distinct) == 1 and index in distinct)
            and not (index not in distinct and len(distinct) >= int(max_members))
        ]
        candidate_scores = _caruana_batch_scores(
            member_ids, stack, running_sum, y_true, metric, allowed, denom
        )
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
    auroc_tolerance: float | None = None,
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
    prefixes: list[dict[str, float | int]] = []
    trace: list[dict[str, Any]] = []
    previous_score: float | None = None
    previous_loss: float | None = None

    for step in range(max_iterations):
        distinct = set(chosen_indices)
        denom = float(step + 1)
        allowed = [
            index
            for index in range(len(member_ids))
            if not (len(distinct) == 1 and index in distinct)
            and not (index not in distinct and len(distinct) >= int(max_members))
        ]
        candidate_scores = _caruana_batch_scores(
            member_ids, stack, running_sum, y_true, metric, allowed, denom
        )
        if not candidate_scores:
            stop_reason = "no_candidates"
            break
        if auroc_tolerance is not None:
            ranked = rank_auroc_log_loss(
                pd.DataFrame(
                    candidate_scores,
                    columns=["AUROC", "log_loss", "index", "config_id"],
                ),
                "AUROC",
                "log_loss",
                auroc_tolerance,
                ("index",),
            )
            selected = ranked.iloc[0]
            score = float(selected["AUROC"])
            candidate_loss = float(selected["log_loss"])
            selected_idx = int(selected["index"])
        else:
            if _metric_is_loss(metric):
                candidate_scores.sort(
                    key=lambda item: (item[0], item[1], item[2], item[3])
                )
            else:
                candidate_scores.sort(
                    key=lambda item: (-item[0], item[1], item[2], item[3])
                )
            score, candidate_loss, selected_idx, _ = candidate_scores[0]
        chosen_indices.append(int(selected_idx))
        running_sum = running_sum + stack[selected_idx]
        trajectory.append(float(score))
        trace.append(
            {
                "iteration": int(step + 1),
                "selected_config_id": str(member_ids[int(selected_idx)]),
                "score": float(score),
                "log_loss": float(candidate_loss),
                "score_change": None
                if previous_score is None
                else float(score - previous_score),
                "log_loss_change": None
                if previous_loss is None
                else float(candidate_loss - previous_loss),
                "distinct_members": int(len(set(chosen_indices))),
            }
        )
        previous_score, previous_loss = float(score), float(candidate_loss)

        valid = len(set(chosen_indices)) >= 2
        if auroc_tolerance is not None:
            if valid:
                prefixes.append(
                    {
                        "AUROC": float(score),
                        "log_loss": float(candidate_loss),
                        "length": len(chosen_indices),
                    }
                )
                ranked_prefixes = rank_auroc_log_loss(
                    pd.DataFrame(prefixes),
                    "AUROC",
                    "log_loss",
                    auroc_tolerance,
                    ("length",),
                )
                chosen_prefix = ranked_prefixes.iloc[0]
                next_best = int(chosen_prefix["length"])
                if next_best != best_prefix:
                    stale_steps = 0
                else:
                    stale_steps += 1
                best_prefix = next_best
                best_score = float(chosen_prefix["AUROC"])
        else:
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
        "caruana_variant": "legacy_constrained",
        "caruana_selection_trace": json.dumps(trace, separators=(",", ":")),
        "caruana_selected_counts": json.dumps(
            {member_ids[index]: counts[index] for index in order}
        ),
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
    if loss == "AUROC_MCC":
        return -_metric_value(y_true, proba, "AUROC_MCC")
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


def _fit_nonsmooth_composite_weights(
    stack: np.ndarray,
    y_true: np.ndarray,
    start: np.ndarray | None = None,
    min_weight: float = 0.0,
) -> np.ndarray:

    n_members = int(stack.shape[0])
    floor = float(min_weight)
    if n_members < 2 or floor < 0 or not np.isfinite(floor) or floor * n_members >= 1:
        raise ValueError("Infeasible composite ensemble simplex.")
    initial = (
        _uniform_weights(n_members) if start is None else np.asarray(start, dtype=float)
    )
    if (
        initial.shape != (n_members,)
        or not np.isfinite(initial).all()
        or np.any(initial < 0)
        or initial.sum() <= 0
    ):
        raise ValueError("Invalid initial composite ensemble weights.")
    initial = initial / initial.sum()

    initial = floor + (1 - floor * n_members) * initial

    def score(weights: np.ndarray) -> float:
        return _metric_value(
            y_true, _weighted_probability_mean(stack, weights), "AUROC_MCC"
        )

    seeds = [initial, _uniform_weights(n_members)]
    if n_members <= 10:
        seeds.extend(
            floor + (1 - floor * n_members) * np.eye(n_members)[index]
            for index in range(n_members)
        )
    best = initial.copy()
    best_value = score(best)
    fractions = (1.0, 0.5, 0.25, 0.1, 0.025)
    for seed in seeds:
        weights = np.asarray(seed, dtype=float).copy()
        value = score(weights)
        for _ in range(16):
            winning_weights = None
            winning_value = value
            for donor in range(n_members):
                available = max(0.0, float(weights[donor]) - floor)
                if available <= 1e-12:
                    continue
                for receiver in range(n_members):
                    if receiver == donor:
                        continue
                    for fraction in fractions:
                        proposed = weights.copy()
                        transfer = available * fraction
                        proposed[donor] -= transfer
                        proposed[receiver] += transfer
                        candidate_score = score(proposed)
                        if candidate_score > winning_value + 1e-9:
                            winning_value = candidate_score
                            winning_weights = proposed
            if winning_weights is None:
                break
            weights, value = winning_weights, winning_value
        if value > best_value + 1e-9:
            best, best_value = weights.copy(), value
    return best / best.sum()


def _fit_composite_super_learner(
    member_ids: list[str], stack: np.ndarray, y_true: np.ndarray, max_members: int
) -> tuple[list[str], np.ndarray, float]:

    n_members = len(member_ids)
    if n_members < 2 or len(set(member_ids)) != n_members or max_members < 2:
        raise ValueError("Composite Super Learner needs two distinct candidate models.")

    singles = [
        _metric_value(y_true, stack[index], "AUROC_MCC") for index in range(n_members)
    ]
    anchor = max(range(n_members), key=lambda index: (singles[index], -index))
    selected = [anchor]
    current = stack[anchor]
    current_score = float(singles[anchor])
    weights = np.array([1.0])
    for _ in range(min(max_members, n_members) - 1):
        candidate: tuple[float, int, float] | None = None
        for index in range(n_members):
            if index in selected:
                continue

            for alpha in (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75):
                mixed = (1.0 - alpha) * current + alpha * stack[index]
                value = _metric_value(y_true, mixed, "AUROC_MCC")
                if candidate is None or value > candidate[0] + 1e-9:
                    candidate = (value, index, alpha)
        if candidate is None:
            break
        if len(selected) >= 2 and candidate[0] <= current_score + 1e-9:
            break
        _, index, alpha = candidate
        selected.append(index)
        start = np.append((1 - alpha) * weights, alpha)
        weights = _fit_nonsmooth_composite_weights(
            stack[selected], y_true, start, min_weight=_SUPER_LEARNER_MIN_WEIGHT
        )
        current = _weighted_probability_mean(stack[selected], weights)
        current_score = _metric_value(y_true, current, "AUROC_MCC")
    if len(selected) < 2:
        raise RuntimeError("Composite Super Learner could not find two members.")

    return [member_ids[i] for i in selected], weights, float(-current_score)


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
    if loss == "AUROC_MCC":
        return _fit_nonsmooth_composite_weights(stack, y_true, x0, min_weight=floor)
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
    full_weights_override: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, float]:
    if stack.shape[0] != len(member_ids):
        raise ValueError("Super Learner library and prediction stack are inconsistent.")
    if (
        len(member_ids) < 2
        or len(set(member_ids)) != len(member_ids)
        or max_members < 2
    ):
        raise ValueError("Super Learner requires at least two distinct members.")
    if loss == "AUROC_MCC":
        return _fit_composite_super_learner(member_ids, stack, y_true, max_members)
    full_weights = (
        _fit_convex_probability_weights(stack, y_true, loss)
        if full_weights_override is None
        else np.asarray(full_weights_override, dtype=float)
    )
    if full_weights.shape != (len(member_ids),) or not np.isfinite(full_weights).all():
        raise ValueError(
            "Cached Super Learner weights do not match the candidate library"
        )
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


def _probabilistic_meta_fold_keys(ordered: pd.DataFrame, method: str) -> np.ndarray:
    if method == "cross_study_dro":
        if "study_group" not in ordered.columns:
            raise ValueError(
                "cross_study_dro requires explicit study_group labels aligned to held-out predictions."
            )
        groups = ordered["study_group"].astype(str).to_numpy()
        if len(np.unique(groups)) < 3:
            raise ValueError(
                "Cross-study DRO requires at least three studies for study-held-out meta-validation."
            )
        return groups
    sample_ids = ordered["sample_id"].astype(str).to_numpy()
    if len(np.unique(sample_ids)) != len(sample_ids):
        identifiers = np.unique(sample_ids)
        n_folds = min(5, len(identifiers))
        if n_folds < 2:
            raise ValueError(
                "Meta-model validation requires at least two unique samples."
            )
        keys = {
            sample_id: f"sample_group_{index % n_folds}"
            for index, sample_id in enumerate(identifiers)
        }
        return np.asarray([keys[ident] for ident in sample_ids], dtype=str)
    return _stacking_fold_keys(ordered)


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
    class_weight: str | None = None,
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
            "class_weight": class_weight,
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
            "class_weight": class_weight,
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
    if aggregation in {"logistic_stacking", "ecdf_logistic_stacking"}:
        return [(value, None) for value in (0.05, 0.2, 1.0, 5.0, 20.0)]
    if aggregation == "ecdf_grouped_stacking":
        return [(value, None) for value in (0.01, 0.05, 0.1, 0.2, 1.0)]
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
    ecdf = aggregation in {"ecdf_logistic_stacking", "ecdf_grouped_stacking"}
    if ecdf and int(stack.shape[2]) != 2:
        raise ValueError("ECDF stacking currently supports binary outcomes only.")
    references = (
        [np.sort(stack[index, :, 1]).tolist() for index in range(stack.shape[0])]
        if ecdf
        else []
    )
    features = (
        ecdf_stacking_features(stack, references)
        if ecdf
        else stacking_features(stack, clip)
    )
    fold_keys = _stacking_fold_keys(ordered)
    unique_folds = sorted(set(fold_keys.tolist()))
    n_classes = int(stack.shape[2])
    classes = np.arange(n_classes, dtype=int)
    if not np.array_equal(np.unique(y_true), classes):
        raise ValueError(
            "Stacking aggregation requires every class in the inner OOF data."
        )
    fold_balanced = aggregation in {"cohort_robust_stacking", "ecdf_grouped_stacking"}
    class_weight = "balanced" if aggregation == "ecdf_grouped_stacking" else None
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
            train_features = features[train_mask]
            validation_features = features[test_mask]
            if ecdf:
                fold_references = [
                    np.sort(stack[index, train_mask, 1]).tolist()
                    for index in range(stack.shape[0])
                ]
                train_features = ecdf_stacking_features(
                    stack[:, train_mask, :], fold_references
                )
                validation_features = ecdf_stacking_features(
                    stack[:, test_mask, :], fold_references
                )
            try:
                model = _fit_logistic_stacker(
                    train_features,
                    y_true[train_mask],
                    c_value=c_value,
                    l1_ratio=l1_ratio,
                    sample_weight=sample_weight,
                    class_weight=class_weight,
                )
            except Exception:
                valid = False
                break
            predicted = np.asarray(
                model.predict_proba(validation_features), dtype=float
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
        class_weight=class_weight,
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
    if ecdf:
        parameters["ecdf_references"] = references
    if class_weight is not None:
        parameters["class_weight"] = class_weight
    diagnostics = {
        "stacking_crossfit_score": float(pooled_score),
        "stacking_C": float(c_value),
        "stacking_l1_ratio": "" if l1_ratio is None else float(l1_ratio),
        "stacking_fold_balanced": int(fold_balanced),
        "stacking_internal_folds": len(unique_folds),
        "stacking_search_trace": json.dumps(trace),
    }
    return np.asarray(crossfit, dtype=float), parameters, diagnostics


def _ecdf_rank_parameters(
    plan: Ensemble,
    members: list[str],
    stack: np.ndarray,
    configs: pd.DataFrame,
) -> dict[str, Any]:
    metadata = configs.drop_duplicates("config_id").set_index("config_id")
    learner_weights = dict(plan.ecdf_member_weights)
    learners = [str(metadata.loc[member, "learner"]) for member in members]
    if len(learners) != len(learner_weights) or set(learners) != set(learner_weights):
        raise ValueError(
            "ECDF weighted quantile members do not match the configured learner weights."
        )
    reference_index = learners.index(plan.ecdf_reference_learner)
    if stack.ndim != 3 or stack.shape[2] != 2:
        raise ValueError(
            "ECDF weighted quantile aggregation requires binary member probabilities."
        )
    return {
        "ecdf_references": [
            np.sort(stack[i, :, 1].astype(float)).tolist()
            for i in range(stack.shape[0])
        ],
        "member_weights": [float(learner_weights[name]) for name in learners],
        "reference_member_index": int(reference_index),
        "member_learners": learners,
    }


def _pareto_inner_member_ids(
    inner_results: pd.DataFrame,
    eligible_ids: set[str],
    outer_split_key: str | None,
) -> list[str]:
    required = {"config_id", "inner_key", "AUROC", "log_loss"}
    if outer_split_key is not None:
        required.add("split_key")
    missing = required - set(inner_results.columns)
    if missing:
        raise ValueError(f"Pareto ensemble needs inner results: {sorted(missing)}")
    frame = inner_results.copy()
    if outer_split_key is not None:
        frame = frame.loc[frame["split_key"].astype(str).eq(str(outer_split_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    frame["inner_key"] = frame["inner_key"].astype(str)
    if "split_key" in frame:
        frame["split_key"] = frame["split_key"].astype(str)
    key_cols = ["split_key", "inner_key"] if "split_key" in frame else ["inner_key"]
    expected = set(
        map(tuple, frame[key_cols].drop_duplicates().itertuples(index=False, name=None))
    )
    frame = frame.loc[frame["config_id"].isin(eligible_ids)].copy()
    frame = frame.drop_duplicates(["config_id", *key_cols], keep="last")
    if "ok" in frame:
        frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").fillna(0).eq(1)]
    for col in ("AUROC", "log_loss"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame[["AUROC", "log_loss"]].to_numpy(dtype=float)).all(axis=1)
    ]
    if not expected:
        return []
    rows = []
    for config_id, group in frame.groupby("config_id", sort=True):
        observed = set(map(tuple, group[key_cols].itertuples(index=False, name=None)))
        if observed == expected:
            rows.append(
                {
                    "config_id": str(config_id),
                    "AUROC": float(group["AUROC"].mean()),
                    "log_loss": float(group["log_loss"].mean()),
                }
            )
    if not rows:
        return []
    ranked = rank_pareto_auroc_logloss(pd.DataFrame(rows))
    return ranked.loc[ranked["pareto_optimal"], "config_id"].astype(str).tolist()


def _fit_pareto_weighted_average(
    stack: np.ndarray,
    y_true: np.ndarray,
    ordered: pd.DataFrame,
    metric: str,
    samples_deduplicated: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    folds = (
        np.asarray(
            [
                f"sample_fold_{index % min(5, len(ordered))}"
                for index in range(len(ordered))
            ],
            dtype=str,
        )
        if samples_deduplicated
        else _probabilistic_meta_fold_keys(ordered, "bayesian_model_averaging")
    )
    unique = sorted(set(folds.tolist()))
    loss = "brier" if metric in {"brier", "brier_loss"} else "log_loss"
    predictions = np.zeros((len(y_true), stack.shape[2]), dtype=float)
    for fold in unique:
        heldout = folds == fold
        training = ~heldout
        if not np.array_equal(np.unique(y_true[training]), np.arange(stack.shape[2])):
            raise ValueError(
                "Pareto weight training requires all classes in every meta-training fold."
            )
        fold_weights = _fit_convex_probability_weights(
            stack[:, training, :], y_true[training], loss
        )
        predictions[heldout] = aggregate_member_predictions(
            stack[:, heldout, :], "weighted_mean_proba", fold_weights
        )
    weights = _fit_convex_probability_weights(stack, y_true, loss)
    return (
        predictions,
        weights,
        {
            "stacking_internal_folds": len(unique),
            "stacking_weight_objective": loss,
            "stacking_crossfit_score": _metric_value(y_true, predictions, metric),
        },
    )


def _pareto_ml_dl_member_ids(
    inner_results: pd.DataFrame,
    eligible_ids: set[str],
    outer_split_key: str | None,
    configs: pd.DataFrame,
) -> list[str]:
    ordered_ids = sorted(str(cid) for cid in eligible_ids)
    family_index = pd.Series(np.zeros(len(ordered_ids), dtype=float), index=ordered_ids)
    deep, machine = _deep_ml_candidates(family_index, configs)
    if not deep or not machine:
        raise ValueError(
            "Separate ML/DL Pareto selection needs at least one eligible classified "
            "conventional ML configuration and one deep-learning configuration."
        )
    deep_frontier = _pareto_inner_member_ids(inner_results, set(deep), outer_split_key)
    machine_frontier = _pareto_inner_member_ids(
        inner_results, set(machine), outer_split_key
    )
    if not deep_frontier or not machine_frontier:
        raise ValueError(
            "Separate ML/DL Pareto selection needs complete AUROC and log-loss "
            "inner-validation results from both learner families."
        )
    return deep_frontier + machine_frontier


def _capped_ml_dl_pareto_members(
    members: list[str],
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    max_size: int,
    outer_split_key: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    ids = list(dict.fromkeys(str(member) for member in members))
    cap = int(max_size)
    if cap < 2:
        raise ValueError("Capped ML/DL Pareto selection requires max_size >= 2.")
    if not ids:
        raise ValueError(
            "Capped ML/DL Pareto selection found no eligible frontier members."
        )
    deep, machine = _deep_ml_candidates(
        pd.Series(np.zeros(len(ids), dtype=float), index=ids), configs
    )
    if not deep or not machine:
        raise ValueError(
            "Capped ML/DL Pareto selection requires both complete ML and DL frontiers."
        )
    family = {member: "DL" for member in deep}
    family.update({member: "ML" for member in machine})
    counts_before = {"ML": len(machine), "DL": len(deep)}
    metadata: dict[str, Any] = {
        "pareto_full_frontier_count": len(ids),
        "pareto_family_counts_before_cap": json.dumps(counts_before, sort_keys=True),
        "pareto_cap_applied": int(len(ids) > cap),
        "pareto_cap_policy": "balanced_family_quality_prediction_diversity",
    }
    if len(ids) <= cap:
        metadata["pareto_family_counts_after_cap"] = json.dumps(
            counts_before, sort_keys=True
        )
        return ids, metadata
    metric_cols = ["config_id", "inner_key", "AUROC", "log_loss"]
    if "split_key" in inner_results.columns:
        metric_cols.append("split_key")
    if "ok" in inner_results.columns:
        metric_cols.append("ok")
    frame = inner_results.loc[
        inner_results["config_id"].astype(str).isin(ids), metric_cols
    ].copy()
    if outer_split_key is not None:
        frame = frame.loc[frame["split_key"].astype(str).eq(str(outer_split_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    key_cols = ["config_id", "inner_key"]
    if "split_key" in frame.columns:
        key_cols.append("split_key")
    frame = frame.drop_duplicates(key_cols, keep="last")
    if "ok" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
    frame["AUROC"] = pd.to_numeric(frame["AUROC"], errors="coerce")
    frame["log_loss"] = pd.to_numeric(frame["log_loss"], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame[["AUROC", "log_loss"]].to_numpy(dtype=float)).all(axis=1)
    ]
    scores = frame.groupby("config_id", sort=True)[["AUROC", "log_loss"]].mean()
    if set(ids) - set(scores.index):
        raise ValueError(
            "Capped ML/DL Pareto selection is missing candidate validation metrics."
        )
    quality: dict[str, float] = {}
    for group in (machine, deep):
        ranked = rank_pareto_auroc_logloss(
            scores.loc[group, ["AUROC", "log_loss"]].reset_index()
        )
        for record in ranked.to_dict(orient="records"):
            quality[str(record["config_id"])] = float(
                1.0 - float(record["pareto_ideal_distance"]) / np.sqrt(2.0)
            )
    pred = _prediction_frame_for_outer(inner_predictions, outer_split_key)
    pred = pred.loc[pred["config_id"].isin(ids)]
    pcols = _proba_cols(pred)
    if not pcols:
        raise ValueError(
            "Capped ML/DL Pareto selection requires inner validation probabilities."
        )
    _, stack = _aligned_stack(pred, ids, pcols, inner=True)
    if stack is None or not np.isfinite(stack).all():
        raise ValueError(
            "Capped ML/DL Pareto selection requires aligned complete inner probabilities."
        )
    correlations = np.abs(_prediction_correlation_matrix(stack))
    positions = {member: index for index, member in enumerate(ids)}
    target_size = min(cap, len(ids))
    quotas = {
        "ML": min(len(machine), target_size // 2),
        "DL": min(len(deep), target_size // 2),
    }
    pools = {"ML": machine, "DL": deep}
    while sum(quotas.values()) < target_size:
        options = [name for name in ("ML", "DL") if quotas[name] < len(pools[name])]
        next_family = min(
            options,
            key=lambda name: (
                quotas[name],
                -max(
                    quality[member]
                    for member in pools[name]
                    if member
                    not in set(
                        sorted(pools[name], key=lambda value: (-quality[value], value))[
                            : quotas[name]
                        ]
                    )
                ),
                name,
            ),
        )
        quotas[next_family] += 1
    selected = [
        min(machine, key=lambda member: (-quality[member], member)),
        min(deep, key=lambda member: (-quality[member], member)),
    ]
    selected_set = set(selected)
    selected_counts = {"ML": 1, "DL": 1}
    while len(selected) < target_size:
        available = [
            member
            for member in ids
            if member not in selected_set
            and selected_counts[family[member]] < quotas[family[member]]
        ]
        if not available:
            raise RuntimeError(
                "Capped ML/DL Pareto selection could not fill the requested family quotas."
            )

        def key(member: str) -> tuple[float, float, float, str]:
            index = positions[member]
            novelty = 1.0 - max(
                float(correlations[index, positions[prior]]) for prior in selected
            )
            value = 0.5 * quality[member] + 0.5 * novelty
            return (-value, -quality[member], -novelty, member)

        winner = min(available, key=key)
        selected.append(winner)
        selected_set.add(winner)
        selected_counts[family[winner]] += 1
    metadata["pareto_family_counts_after_cap"] = json.dumps(
        selected_counts, sort_keys=True
    )
    metadata["pareto_family_quota"] = json.dumps(quotas, sort_keys=True)
    return selected, metadata


def _rank_for_taxonomic_pareto(row: dict[str, Any]) -> str | None:
    available = TAXONOMIC_LEVELS[TAXONOMIC_LEVELS.index("phylum") :]
    resolution_value = row.get("resolution")
    resolution = (
        ""
        if resolution_value is None or pd.isna(resolution_value)
        else str(resolution_value).strip().casefold()
    )
    if resolution in {"raw", "all", "features", "asis"}:
        return None
    scope_value = row.get("representation_scope")
    scope = (
        ""
        if scope_value is None or pd.isna(scope_value)
        else str(scope_value).strip().casefold()
    )
    if scope == "multi-rank":
        return None
    for column in ("taxonomic_blocks", "levels", "resolution"):
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        if column == "resolution":
            levels = [str(value).strip().casefold()]
        else:
            levels = [
                part.strip().casefold()
                for part in str(value).split(",")
                if part.strip()
            ]
        if not levels or levels == ["all"]:
            continue
        if len(levels) != 1:
            return None
        if levels[0] in available:
            return levels[0]
        if levels[0] in TAXONOMIC_LEVELS:
            return None
    return None


def _pareto_taxonomic_rank_member_ids(
    inner_results: pd.DataFrame,
    eligible_ids: set[str],
    outer_split_key: str | None,
    configs: pd.DataFrame,
) -> list[str]:
    if not {"config_id", "resolution"}.issubset(configs.columns):
        raise ValueError(
            "Rank-wise Pareto selection requires config_id and resolution metadata"
        )
    frame = configs.drop_duplicates("config_id", keep="last")
    rank_members = {
        rank: set() for rank in TAXONOMIC_LEVELS[TAXONOMIC_LEVELS.index("phylum") :]
    }
    for row in frame.loc[frame["config_id"].astype(str).isin(eligible_ids)].to_dict(
        orient="records"
    ):
        rank = _rank_for_taxonomic_pareto(row)
        if rank is not None:
            rank_members[rank].add(str(row["config_id"]))
    members = []
    for rank, candidate_ids in rank_members.items():
        if candidate_ids:
            members.extend(
                _pareto_inner_member_ids(inner_results, candidate_ids, outer_split_key)
            )
    if not members:
        raise ValueError(
            "Rank-wise Pareto selection needs complete AUROC and log-loss "
            "inner-validation results for at least one single taxonomic rank "
            "from phylum through strain."
        )
    return members


def _capped_taxonomic_rank_pareto_members(
    members: list[str],
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    max_size: int,
    outer_split_key: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    ids = list(dict.fromkeys(str(member) for member in members))
    cap = int(max_size)
    if cap < 2:
        raise ValueError(
            "Capped taxonomic-rank Pareto selection requires max_size >= 2."
        )
    if not ids:
        raise ValueError(
            "Capped taxonomic-rank Pareto selection found no eligible frontier members."
        )
    config_rows = configs.drop_duplicates("config_id", keep="last").copy()
    config_rows["config_id"] = config_rows["config_id"].astype(str)
    config_index = config_rows.set_index("config_id")
    missing = sorted(set(ids) - set(config_index.index))
    if missing:
        raise ValueError(
            f"Capped taxonomic-rank Pareto selection is missing configuration metadata: {missing}"
        )
    ranks = TAXONOMIC_LEVELS[TAXONOMIC_LEVELS.index("phylum") :]
    pools: dict[str, list[str]] = {rank: [] for rank in ranks}
    for member in ids:
        rank = _rank_for_taxonomic_pareto(config_index.loc[member].to_dict())
        if rank is None or rank not in pools:
            raise ValueError(
                f"Capped taxonomic-rank Pareto selection cannot identify a single rank for {member}"
            )
        pools[rank].append(member)
    pools = {rank: values for rank, values in pools.items() if values}
    before = {rank: len(values) for rank, values in pools.items()}
    metadata: dict[str, Any] = {
        "pareto_full_frontier_count": len(ids),
        "pareto_taxonomic_rank_counts_before_cap": json.dumps(before, sort_keys=True),
        "pareto_cap_applied": int(len(ids) > cap),
        "pareto_cap_policy": "balanced_taxonomic_rank_quality_prediction_diversity",
    }
    if len(ids) <= cap:
        metadata["pareto_taxonomic_rank_counts_after_cap"] = json.dumps(
            before, sort_keys=True
        )
        metadata["pareto_taxonomic_rank_quota"] = json.dumps(before, sort_keys=True)
        metadata["pareto_taxonomic_ranks_not_represented"] = json.dumps([])
        return ids, metadata
    columns = ["config_id", "inner_key", "AUROC", "log_loss"]
    if "split_key" in inner_results.columns:
        columns.append("split_key")
    if "ok" in inner_results.columns:
        columns.append("ok")
    frame = inner_results.loc[
        inner_results["config_id"].astype(str).isin(ids), columns
    ].copy()
    if outer_split_key is not None:
        frame = frame.loc[frame["split_key"].astype(str).eq(str(outer_split_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    keys = ["config_id", "inner_key"]
    if "split_key" in frame.columns:
        keys.append("split_key")
    frame = frame.drop_duplicates(keys, keep="last")
    if "ok" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
    for column in ("AUROC", "log_loss"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame[["AUROC", "log_loss"]].to_numpy(dtype=float)).all(axis=1)
    ]
    scores = frame.groupby("config_id", sort=True)[["AUROC", "log_loss"]].mean()
    if set(ids) - set(scores.index):
        raise ValueError(
            "Capped taxonomic-rank Pareto selection is missing candidate validation metrics."
        )
    quality: dict[str, float] = {}
    for rank, group in pools.items():
        ranked = rank_pareto_auroc_logloss(
            scores.loc[group, ["AUROC", "log_loss"]].reset_index()
        )
        for record in ranked.to_dict(orient="records"):
            quality[str(record["config_id"])] = float(
                1.0 - float(record["pareto_ideal_distance"]) / np.sqrt(2.0)
            )
    auc = scores.loc[ids, "AUROC"]
    loss = scores.loc[ids, "log_loss"]
    auc_span = float(auc.max() - auc.min())
    loss_span = float(loss.max() - loss.min())
    global_quality = {
        member: float(
            1.0
            - np.hypot(
                (float(auc.max()) - float(auc.loc[member])) / auc_span
                if auc_span > 0
                else 0.0,
                (float(loss.loc[member]) - float(loss.min())) / loss_span
                if loss_span > 0
                else 0.0,
            )
            / np.sqrt(2.0)
        )
        for member in ids
    }
    pred = _prediction_frame_for_outer(inner_predictions, outer_split_key)
    pred = pred.loc[pred["config_id"].astype(str).isin(ids)].copy()
    pcols = _proba_cols(pred)
    if not pcols:
        raise ValueError(
            "Capped taxonomic-rank Pareto selection requires inner validation probabilities."
        )
    _, stack = _aligned_stack(pred, ids, pcols, inner=True)
    if stack is None or not np.isfinite(stack).all():
        raise ValueError(
            "Capped taxonomic-rank Pareto selection requires aligned complete inner probabilities."
        )
    correlations = np.abs(_prediction_correlation_matrix(stack))
    positions = {member: index for index, member in enumerate(ids)}
    rank_position = {rank: index for index, rank in enumerate(ranks)}
    ranked_pools = {
        rank: sorted(
            values,
            key=lambda member: (-quality[member], -global_quality[member], member),
        )
        for rank, values in pools.items()
    }
    ranked_ranks = sorted(
        pools,
        key=lambda rank: (
            -max(global_quality[member] for member in pools[rank]),
            -quality[ranked_pools[rank][0]],
            rank_position[rank],
        ),
    )
    target_size = min(cap, len(ids))
    covered_ranks = set(ranked_ranks[: min(target_size, len(ranked_ranks))])
    quotas = {rank: 1 for rank in covered_ranks}
    while sum(quotas.values()) < target_size:
        options = [rank for rank in covered_ranks if quotas[rank] < len(pools[rank])]
        if not options:
            raise RuntimeError(
                "Capped taxonomic-rank Pareto selection cannot fill the requested rank quotas."
            )
        rank = min(
            options,
            key=lambda name: (
                quotas[name],
                -quality[ranked_pools[name][quotas[name]]],
                -global_quality[ranked_pools[name][quotas[name]]],
                rank_position[name],
            ),
        )
        quotas[rank] += 1
    selected = [ranked_pools[rank][0] for rank in ranks if rank in covered_ranks]
    selected_set = set(selected)
    member_rank = {member: rank for rank, group in pools.items() for member in group}
    counts = {rank: 1 for rank in covered_ranks}
    while len(selected) < target_size:
        available = [
            member
            for member in ids
            if member not in selected_set
            and member_rank[member] in covered_ranks
            and counts[member_rank[member]] < quotas[member_rank[member]]
        ]
        if not available:
            raise RuntimeError(
                "Capped taxonomic-rank Pareto selection cannot fill the selected rank quotas."
            )

        def priority(member: str) -> tuple[float, float, float, float, str]:
            novelty = 1.0 - max(
                float(correlations[positions[member], positions[prior]])
                for prior in selected
            )
            value = 0.5 * quality[member] + 0.5 * novelty
            return (-value, -quality[member], -novelty, -global_quality[member], member)

        chosen = min(available, key=priority)
        selected.append(chosen)
        selected_set.add(chosen)
        counts[member_rank[chosen]] += 1
    after = {rank: counts[rank] for rank in ranks if rank in covered_ranks}
    metadata["pareto_taxonomic_rank_counts_after_cap"] = json.dumps(
        after, sort_keys=True
    )
    metadata["pareto_taxonomic_rank_quota"] = json.dumps(quotas, sort_keys=True)
    metadata["pareto_taxonomic_ranks_not_represented"] = json.dumps(
        [rank for rank in ranks if rank in pools and rank not in covered_ranks]
    )
    return selected, metadata


def _candidate_from_simple_selector(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    configs: pd.DataFrame,
    pcols: list[str],
    metric: str,
    plan: Ensemble,
    pareto_members: list[str] | None = None,
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
        if method in {"oof_top_k", "oof_similarity"}:
            library = [str(value) for value in scores.index.tolist()]
            ranking = rank_oof_loss(
                predictions, library, robust=method == "oof_similarity"
            )
            if method == "oof_top_k":
                members = ranking[:max_size]
            else:
                shortlisted = ranking[:30]
                aligned, available = _aligned_stack(
                    predictions, shortlisted, pcols, inner=True
                )
                if aligned is None or available is None:
                    raise ValueError("Incomplete OOF similarity library")
                members = similarity_members(available, shortlisted, max_size)
            extra["candidate_library_policy"] = (
                "pooled_oof_log_loss"
                if method == "oof_top_k"
                else "fold_robust_oof_similarity"
            )
            extra["candidate_library_size"] = len(ranking)
        elif method in _PARETO_SELECTORS:
            members = list(pareto_members or [])
            if not members:
                return None
            if len(members) > max_size:
                raise ValueError(
                    f"Inner Pareto frontier contains {len(members)} members, "
                    f"exceeding max_sizes={max_size}. Increase Ensemble.max_sizes "
                    "to avoid truncating the scientific frontier."
                )
            extra["candidate_library_policy"] = (
                "complete_inner_auroc_logloss_pareto_per_learner_family"
                if method
                in {
                    "pareto_optimal_ml_dl_auroc_logloss",
                    "pareto_optimal_ml_dl_auroc_logloss_capped",
                }
                else "complete_inner_auroc_logloss_pareto_per_taxonomic_rank"
                if method
                in {
                    "pareto_optimal_taxonomic_rank_auroc_logloss",
                    "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
                }
                else "complete_inner_auroc_logloss_pareto"
            )
            if method in {
                "pareto_optimal_ml_dl_auroc_logloss",
                "pareto_optimal_ml_dl_auroc_logloss_capped",
            }:
                deep, machine = _deep_ml_candidates(scores, configs)
                extra["pareto_deep_learning_count"] = len(set(members) & set(deep))
                extra["pareto_machine_learning_count"] = len(
                    set(members) & set(machine)
                )
                extra["learner_family_source"] = (
                    "persisted_estimator_class_and_nested_parameters"
                )
            if method in {
                "pareto_optimal_taxonomic_rank_auroc_logloss",
                "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
            }:
                rank_map = configs.drop_duplicates("config_id", keep="last").set_index(
                    "config_id"
                )
                selected_ranks = {
                    rank: 0
                    for rank in TAXONOMIC_LEVELS[TAXONOMIC_LEVELS.index("phylum") :]
                }
                for member in members:
                    rank = _rank_for_taxonomic_pareto(
                        rank_map.loc[str(member)].to_dict()
                    )
                    if rank is not None:
                        selected_ranks[rank] += 1
                extra["pareto_taxonomic_rank_counts"] = json.dumps(
                    {rank: count for rank, count in selected_ranks.items() if count},
                    separators=(",", ":"),
                )
            extra["candidate_library_size"] = len(scores)
            extra["degenerate_single_member"] = int(len(members) == 1)
        elif method == "top_k":
            members = _select_top_k(scores, max_size)
        elif method == "best_per_resolution":
            members = _select_best_per_resolution(scores, configs)[:max_size]
        elif method == "best_per_learner_type":
            members = _select_best_per_learner_type(scores, configs)[:max_size]
        elif method in {"best_deep_ml_pair", "best_deep_ml_complementary_pair"}:
            members, ordered_pair, stack_pair, pair_extra = _select_deep_ml_pair(
                scores,
                predictions,
                configs,
                pcols,
                metric,
                method == "best_deep_ml_complementary_pair",
            )
            if not members:
                return None
            extra.update(pair_extra)
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
        if len(members) < 2 and method not in _PARETO_SELECTORS:
            return None
        if method in {"best_deep_ml_pair", "best_deep_ml_complementary_pair"}:
            ordered, stack = ordered_pair, stack_pair
        else:
            ordered, stack = _aligned_stack(predictions, members, pcols, inner=True)
        if ordered is None or stack is None:
            return None
        y_true = ordered["y_true"].to_numpy(dtype=int)
    if len(members) < 2 and method not in _PARETO_SELECTORS:
        return None
    extra["inner_oof_rows"] = len(ordered)
    extra.update(diagnostics)
    rank_parameters = (
        _ecdf_rank_parameters(plan, members, stack, configs)
        if str(spec["aggregation_strategy"]) == "ecdf_weighted_quantile"
        else None
    )
    return _evaluate_selected_members(
        spec,
        members,
        stack,
        y_true,
        metric,
        ordered=ordered,
        extra=extra,
        aggregation_parameters_override=rank_parameters,
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
                "super_learner_loss": "negative_AUROC_MCC"
                if loss == "AUROC_MCC"
                else loss,
                "super_learner_loss_rule": "direct_composite_if_AUROC_MCC_else_brier_or_log_loss",
                "super_learner_loss_value": float(loss_value),
                "super_learner_objective_score": float(-loss_value)
                if loss == "AUROC_MCC"
                else "",
                "super_learner_internal_weight_tol": float(_SUPER_LEARNER_WEIGHT_TOL),
                "super_learner_internal_min_weight": float(_SUPER_LEARNER_MIN_WEIGHT),
                "super_learner_internal_optimizer": (
                    "pairwise_coordinate_search" if loss == "AUROC_MCC" else "SLSQP"
                ),
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


def _candidate_from_modern_selector(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    metric: str,
    plan: Ensemble,
    configs: pd.DataFrame,
) -> dict[str, Any] | None:
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    pool_limit = int(plan.advanced_selection_pool_size)
    if pool_limit > 0:
        limit = min(len(library), max(pool_limit, int(spec["max_size"])))
        library = library[:limit]
        stack = stack[:limit]
    y_true = ordered["y_true"].to_numpy(dtype=int)
    method = str(spec["selection_strategy"])
    max_members = int(spec["max_size"])
    native_weights = None
    if method in {"qo_es", "qdo_es"}:
        config_frame = configs.drop_duplicates("config_id", keep="last").set_index(
            "config_id"
        )
        signature_fields = ("learner", "transformation", "resolution", "levels")
        signatures = [
            tuple(
                str(config_frame.at[mid, col])
                if mid in config_frame.index and col in config_frame.columns
                else ""
                for col in signature_fields
            )
            for mid in library
        ]
        members, native_weights, details = population_ensemble_select(
            library,
            stack,
            y_true,
            lambda values: _metric_value(y_true, values, metric),
            higher_is_better=not _metric_is_loss(metric),
            max_unique=max_members,
            iterations=int(plan.population_iterations),
            archive_size=int(plan.population_archive_size),
            seed=int(plan.population_random_state),
            quality_diversity=method == "qdo_es",
            signatures=signatures,
            bins=int(plan.qdo_archive_bins),
        )
    elif method == "drep":
        members, details = drep_select(
            library, stack, y_true, max_members, plan.drep_rho
        )
    elif method == "pseo_bqp":
        members, details = pseo_bqp_select(
            library,
            stack,
            y_true,
            max_members,
            plan.pseo_bqp_omega,
            plan.pseo_bqp_swap_passes,
        )
    else:
        raise ValueError(f"Unknown modern selection strategy: {method}.")
    if len(members) < 2:
        return None
    positions = {mid: i for i, mid in enumerate(library)}
    selected_stack = stack[[positions[mid] for mid in members]]
    return _evaluate_selected_members(
        spec,
        members,
        selected_stack,
        y_true,
        metric,
        native_weights=native_weights,
        native_weight_source="population_selection_frequency"
        if native_weights is not None
        else None,
        ordered=ordered,
        extra={
            "inner_oof_rows": len(ordered),
            "candidate_library_policy": "top_ranked_complete_inner_oof_mpmas"
            if pool_limit > 0
            else "all_complete_inner_oof_mpmas",
            "candidate_library_size": len(library),
            "candidate_library_screening_limit": pool_limit,
            "selection_weight_source": "population_selection_frequency"
            if native_weights is not None
            else "none",
            **details,
        },
    )


def _candidate_from_caruana(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    pcols: list[str],
    metric: str,
    auroc_tolerance: float | None = None,
    plan: Ensemble | None = None,
) -> dict[str, Any] | None:
    library, ordered, stack = _complete_library_stack(scores, predictions, pcols)
    if ordered is None or stack is None:
        return None
    y_true = ordered["y_true"].to_numpy(dtype=int)
    if str(spec["selection_strategy"]) == "caruana":
        if plan is None:
            raise ValueError("Canonical Caruana requires an ensemble configuration.")
        members, native_weights, _, trajectory, diagnostics = _caruana_canonical_select(
            library,
            stack,
            y_true,
            metric,
            int(plan.caruana_iterations),
            auroc_tolerance,
            plan.caruana_max_unique_members,
            plan.caruana_early_stopping,
            plan.caruana_patience,
            plan.caruana_min_iterations,
            plan.caruana_sorted_initial_models,
        )
    else:
        members, native_weights, _, trajectory, diagnostics = _caruana_select(
            library, stack, y_true, metric, int(spec["max_size"]), auroc_tolerance
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
    full_library_size = len(library)
    pool_limit = int(getattr(plan, "pooled_super_learner_pool_size", 0))
    pooled = str(getattr(plan, "selection_mode", "nested")) == "pooled_grid"
    if (
        pooled
        and pool_limit
        and full_library_size > max(pool_limit, int(spec["max_size"]))
    ):
        limit = max(pool_limit, int(spec["max_size"]))
        library = library[:limit]
        stack = stack[:limit]
    y_true = ordered["y_true"].to_numpy(dtype=int)
    super_loss = _resolved_super_learner_loss(plan)
    fit_cache = predictions.attrs.get("_pooled_grid_matrix")
    if (
        fit_cache is not None
        and fit_cache["frame_id"] == id(predictions)
        and pooled
        and super_loss != "AUROC_MCC"
    ):
        weights_cache = fit_cache.setdefault("super_learner_full_weights", {})
        key = (tuple(library), super_loss)
        if key not in weights_cache:
            weights_cache[key] = _fit_convex_probability_weights(
                stack, y_true, super_loss
            )
        full_weights_override = weights_cache[key]
    else:
        full_weights_override = None
    members, native_weights, loss_value = _fit_super_learner(
        library,
        stack,
        y_true,
        super_loss,
        int(spec["max_size"]),
        full_weights_override=full_weights_override,
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
            "candidate_library_policy": (
                "top_ranked_inner_metric_screened_for_super_learner"
                if len(library) < full_library_size
                else "all_complete_inner_oof_mpmas"
            ),
            "candidate_library_size": full_library_size,
            "super_learner_optimizer_library_size": len(library),
            "super_learner_screening_applied": int(len(library) < full_library_size),
            "super_learner_screening_limit": pool_limit if pooled else 0,
            "selection_weight_source": f"convex_{super_loss}",
            "super_learner_loss": "negative_AUROC_MCC"
            if super_loss == "AUROC_MCC"
            else super_loss,
            "super_learner_loss_rule": "direct_composite_if_AUROC_MCC_else_brier_or_log_loss",
            "super_learner_loss_value": float(loss_value),
            "super_learner_objective_score": float(-loss_value)
            if super_loss == "AUROC_MCC"
            else "",
            "super_learner_internal_weight_tol": float(_SUPER_LEARNER_WEIGHT_TOL),
            "super_learner_internal_min_weight": float(_SUPER_LEARNER_MIN_WEIGHT),
            "super_learner_internal_optimizer": (
                "pairwise_coordinate_search" if super_loss == "AUROC_MCC" else "SLSQP"
            ),
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


def _aligned_outer_oof_stacking(
    predictions: pd.DataFrame,
    members: list[str],
    pcols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"sample_id", "outer_split_key", "config_id", "y_true"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Development OOF predictions are missing columns: {missing}.")
    frame = _prediction_frame_for_outer(predictions, None)
    frame = frame[frame["config_id"].isin(members)].copy()
    if frame.duplicated(["config_id", "sample_id"]).any():
        raise ValueError(
            "Development OOF stacking requires exactly one prediction per model and sample."
        )
    group_counts = frame.groupby("sample_id")["outer_split_key"].nunique()
    if (group_counts != 1).any():
        raise ValueError("Development OOF samples have inconsistent held-out studies.")
    ordered, stack = _aligned_stack(frame, members, pcols, inner=False)
    if ordered is None or stack is None:
        raise ValueError(
            "Development OOF predictions are incomplete for selected members."
        )
    groups = frame[["sample_id", "outer_split_key"]].drop_duplicates("sample_id")
    ordered = ordered.merge(groups, on="sample_id", how="left", validate="one_to_one")
    if ordered["outer_split_key"].isna().any():
        raise ValueError(
            "Development OOF samples are missing held-out study identifiers."
        )
    ordered["split_key"] = ordered["outer_split_key"]
    if ordered["sample_id"].duplicated().any():
        raise ValueError("Development OOF stacking contains repeated sample IDs.")
    return ordered, stack


def _candidate_table_for_pooled_pareto(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    qualification: pd.DataFrame | None,
    progress_callback,
    progress_scope: str,
) -> pd.DataFrame:
    specs = _ensemble_configs(plan, configs)
    spec = specs[0]

    def phase(step: int, detail: str) -> None:
        if progress_callback is not None:
            progress_callback(progress_scope, "phase", spec, step, 5, detail)

    phase(1, "Checking eligible configurations")
    required = {"config_id", "split_key", "inner_key", "AUROC", "log_loss"}
    if not required.issubset(inner_results.columns):
        raise ValueError(
            f"Pooled Pareto selection requires inner columns: {sorted(required - set(inner_results.columns))}"
        )
    eligible = _eligible_config_ids(configs, plan)
    expected_keys = inner_results[["split_key", "inner_key"]].drop_duplicates()
    expected_count = len(expected_keys)
    if not expected_count:
        raise ValueError(
            "No inner-validation splits are available for pooled Pareto selection"
        )
    qualified = _qualified_config_ids_for_splits(
        qualification, set(inner_results["split_key"].astype(str).unique())
    )
    if qualified is not None:
        eligible &= set(qualified)

    phase(2, "Aggregating complete inner AUROC and log loss")
    columns = ["config_id", "split_key", "inner_key", "AUROC", "log_loss"]
    if "ok" in inner_results.columns:
        columns.append("ok")
    frame = inner_results.loc[
        inner_results["config_id"].astype(str).isin(eligible), columns
    ].copy()
    frame["config_id"] = frame["config_id"].astype(str)
    frame = frame.drop_duplicates(["config_id", "split_key", "inner_key"], keep="last")
    if "ok" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
    for column in ("AUROC", "log_loss"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame[["AUROC", "log_loss"]].to_numpy(dtype=float)).all(axis=1)
    ]
    scores = frame.groupby("config_id", sort=False).agg(
        n_inner=("inner_key", "size"),
        AUROC=("AUROC", "mean"),
        log_loss=("log_loss", "mean"),
    )
    scores = scores.loc[scores["n_inner"].eq(expected_count)]
    if scores.empty:
        raise ValueError(
            "No complete inner-validation configurations are available for Pareto selection"
        )

    phase(3, "Checking inner prediction coverage")
    pred_keys = ["outer_split_key", "split_key", "sample_id"]
    if not {"config_id", "y_true", *pred_keys}.issubset(inner_predictions.columns):
        raise ValueError(
            "Inner predictions are missing required fields for pooled Pareto selection"
        )
    expected_predictions = len(inner_predictions[pred_keys].drop_duplicates())
    counts = inner_predictions.groupby("config_id", sort=False).size()

    phase(4, f"Computing Pareto frontier from {len(scores):,} complete candidates")
    if str(spec["selection_strategy"]) in {
        "pareto_optimal_ml_dl_auroc_logloss",
        "pareto_optimal_ml_dl_auroc_logloss_capped",
    }:
        members = _pareto_ml_dl_member_ids(
            inner_results, set(scores.index.astype(str)), None, configs
        )
    elif str(spec["selection_strategy"]) in {
        "pareto_optimal_taxonomic_rank_auroc_logloss",
        "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
    }:
        members = _pareto_taxonomic_rank_member_ids(
            inner_results, set(scores.index.astype(str)), None, configs
        )
    else:
        ranked = rank_pareto_auroc_logloss(
            scores.reset_index()[["config_id", "AUROC", "log_loss"]]
        )
        members = ranked.loc[ranked["pareto_optimal"], "config_id"].astype(str).tolist()
    if not members:
        raise ValueError("No valid AUROC-log-loss Pareto members were found")
    incomplete_predictions = [
        member
        for member in members
        if int(counts.get(member, 0)) < expected_predictions
    ]
    if incomplete_predictions:
        raise ValueError(
            f"Pooled Pareto members have incomplete inner OOF probabilities: {incomplete_predictions}"
        )
    selected_predictions = _prediction_frame_for_outer(
        inner_predictions.loc[inner_predictions["config_id"].isin(members)], None
    )
    pcols = _proba_cols(selected_predictions)
    if not pcols:
        raise ValueError(
            "Selected Pareto members have no inner probability predictions"
        )

    phase(5, f"Scoring fixed ensembles from {len(members)} pooled Pareto members")
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        chosen = members
        metadata: dict[str, Any] = {}
        if (
            str(spec["selection_strategy"])
            == "pareto_optimal_ml_dl_auroc_logloss_capped"
        ):
            chosen, metadata = _capped_ml_dl_pareto_members(
                members,
                inner_results,
                selected_predictions,
                configs,
                int(spec["max_size"]),
            )
        elif (
            str(spec["selection_strategy"])
            == "pareto_optimal_taxonomic_rank_auroc_logloss_capped"
        ):
            chosen, metadata = _capped_taxonomic_rank_pareto_members(
                members,
                inner_results,
                selected_predictions,
                configs,
                int(spec["max_size"]),
            )
        elif len(members) > int(spec["max_size"]):
            raise ValueError(
                f"Pooled inner Pareto frontier contains {len(members)} members, "
                f"exceeding max_sizes={spec['max_size']}. Raise the cap without changing fold membership."
            )
        selected_scores = scores.loc[chosen, "AUROC"].astype(float)
        row = _candidate_from_simple_selector(
            spec,
            selected_scores,
            selected_predictions,
            configs,
            pcols,
            metric,
            plan,
            pareto_members=chosen,
        )
        if row is None:
            raise ValueError(
                "The pooled Pareto members do not have matching complete inner OOF predictions"
            )
        row.update(metadata)
        row["candidate_library_size"] = len(scores)
        row["selection_basis"] = "pooled_inner_validation_fixed_members"
        rows.append(row)
        if progress_callback is not None:
            progress_callback(
                progress_scope,
                "start",
                spec,
                index,
                len(specs),
                f"{len(chosen)} fixed Pareto members",
            )
            progress_callback(progress_scope, "done", spec, index, len(specs), "valid")
    out = pd.DataFrame(rows)
    score_col = f"{metric}_mean"
    out = out.loc[np.isfinite(pd.to_numeric(out[score_col], errors="coerce"))].copy()
    if out.empty:
        raise ValueError(
            "No pooled Pareto candidate produced a finite selection metric"
        )
    return out.sort_values(
        [score_col, "member_count", "max_size", "ensemble_config_id"],
        ascending=[_metric_is_loss(metric), True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _pooled_grid_unique_predictions(
    predictions: pd.DataFrame,
    progress_detail=None,
) -> pd.DataFrame:
    pcols = _proba_cols(predictions)
    if progress_detail is not None:
        progress_detail("Checking repeated-sample label consistency")
    labels = predictions[["sample_id", "y_true"]].drop_duplicates()
    if labels["sample_id"].duplicated().any():
        raise ValueError("Pooled grid has contradictory labels for the same sample_id")
    keys = ["config_id", "sample_id"]
    agg = {col: "mean" for col in pcols}
    agg["y_true"] = "first"
    if "study_group" in predictions.columns:
        groups = predictions[["sample_id", "study_group"]].drop_duplicates()
        if groups["sample_id"].duplicated().any():
            raise ValueError(
                "Pooled grid has contradictory studies for the same sample_id"
            )
        agg["study_group"] = "first"
    if progress_detail is not None:
        progress_detail(f"Averaging {len(predictions):,} repeated OOF prediction rows")
    frame = predictions.groupby(keys, sort=False, as_index=False).agg(agg)
    if progress_detail is not None:
        progress_detail(
            f"Preparing meta-fold labels from {len(frame):,} unique predictions"
        )
    labels = frame[["sample_id", "y_true"]].drop_duplicates("sample_id").copy()
    classes = sorted(labels["y_true"].unique().tolist())
    n_folds = min(5, *(int((labels["y_true"] == cls).sum()) for cls in classes))
    n_folds = max(2, int(n_folds))
    assignments: dict[str, str] = {}
    for cls in classes:
        identifiers = sorted(
            labels.loc[labels["y_true"].eq(cls), "sample_id"].astype(str)
        )
        for index, identifier in enumerate(identifiers):
            assignments[identifier] = f"meta_{index % n_folds}"
    frame["outer_split_key"] = "pooled"
    frame["split_key"] = frame["sample_id"].astype(str).map(assignments)
    return frame


def _complete_pooled_grid_ids(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    eligible_ids: set[str],
    metric: str,
    progress_detail=None,
) -> set[str]:
    key_cols = ["split_key", "inner_key"]
    required = {"config_id", "AUROC", "log_loss", *key_cols}
    if not required.issubset(inner_results.columns):
        raise ValueError(
            f"Pooled grid metrics missing {sorted(required - set(inner_results.columns))}"
        )
    required_pred = {"config_id", "outer_split_key", "split_key", "sample_id", "y_true"}
    if not required_pred.issubset(inner_predictions.columns):
        raise ValueError(
            f"Pooled grid predictions missing {sorted(required_pred - set(inner_predictions.columns))}"
        )
    pcols = _proba_cols(inner_predictions)
    if not pcols:
        raise ValueError("Pooled grid needs class probability columns")
    expected_metrics = len(inner_results[key_cols].drop_duplicates())
    if not expected_metrics:
        raise ValueError("No inner-validation fold keys exist for pooled grid")
    if progress_detail is not None:
        progress_detail(f"Checking {len(inner_results):,} inner-fold metric records")
    frame = inner_results.loc[
        inner_results["config_id"].astype(str).isin(eligible_ids)
    ].copy()
    if "ok" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
    required_values = (
        ["AUROC", "log_loss", metric]
        if metric in frame.columns
        else ["AUROC", "log_loss"]
    )
    values = frame[required_values].apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)]
    frame = frame.drop_duplicates(["config_id", *key_cols], keep="last")
    counts = frame.groupby("config_id", sort=False).size()
    ids = set(counts.loc[counts.eq(expected_metrics)].index.astype(str))
    if not ids:
        raise ValueError("No complete inner-fold configurations for pooled grid")
    if progress_detail is not None:
        progress_detail(
            f"{len(ids):,} models have complete metrics · checking saved probabilities"
        )
    pred_keys = ["outer_split_key", "split_key", "sample_id"]
    expected_pred = len(inner_predictions[pred_keys].drop_duplicates())
    if not expected_pred:
        raise ValueError("No inner probability predictions for pooled grid")
    if progress_detail is not None:
        progress_detail(
            f"Validating {len(inner_predictions):,} probability records for {len(ids):,} models"
        )
    preds = inner_predictions.loc[
        inner_predictions["config_id"].astype(str).isin(ids),
        ["config_id", *pred_keys, *pcols],
    ].drop_duplicates(["config_id", *pred_keys], keep="last")
    finite = np.isfinite(preds[pcols].to_numpy(dtype=float)).all(axis=1)
    counts_pred = preds.loc[finite].groupby("config_id", sort=False).size()
    return ids & set(counts_pred.loc[counts_pred.eq(expected_pred)].index.astype(str))


def _pooled_grid_inner_scores(
    inner_results: pd.DataFrame,
    eligible: set[str],
    metric: str,
    auroc_tolerance: float | None,
) -> pd.Series:
    frame = inner_results.loc[
        inner_results["config_id"].astype(str).isin(eligible)
    ].copy()
    frame["config_id"] = frame["config_id"].astype(str)
    frame = frame.drop_duplicates(["config_id", "split_key", "inner_key"], keep="last")
    if metric == "AUROC_MCC" and metric not in frame.columns:
        if not {"AUROC", "MCC"}.issubset(frame.columns):
            raise ValueError(
                "AUROC_MCC selection requires inner AUROC and MCC columns."
            )
        auc = pd.to_numeric(frame["AUROC"], errors="coerce")
        mcc = pd.to_numeric(frame["MCC"], errors="coerce")
        frame[metric] = 0.5 * (2.0 * auc - 1.0) + 0.5 * mcc
    if metric not in frame.columns:
        raise ValueError(f"Missing inner evaluation metric {metric!r} for pooled grid")
    values = pd.to_numeric(frame[metric], errors="coerce")
    frame["_selection_value"] = values.where(np.isfinite(values), np.nan)
    groups = frame.groupby("config_id", sort=False)
    scores = groups["_selection_value"].mean().dropna()
    if metric in {"log_loss", "brier"} and "n_samples" in frame.columns:
        weights = pd.to_numeric(frame["n_samples"], errors="coerce")
        valid = np.isfinite(weights) & weights.gt(0) & frame["_selection_value"].notna()
        weighted = frame.loc[valid, ["config_id", "_selection_value"]].copy()
        weighted["_weighted_value"] = (
            frame.loc[valid, "_selection_value"] * weights.loc[valid]
        )
        weighted["_weight"] = weights.loc[valid]
        sums = weighted.groupby("config_id", sort=False)[
            ["_weighted_value", "_weight"]
        ].sum()
        if not sums.empty:
            scores.update(sums["_weighted_value"] / sums["_weight"])
    scores = scores.astype(float)
    if auroc_tolerance is not None:
        if "log_loss" not in frame.columns:
            raise ValueError(
                "AUROC/log-loss ensemble selection requires inner log_loss results."
            )
        losses_frame = frame[["config_id", "log_loss"]].copy()
        losses_frame["_loss"] = pd.to_numeric(losses_frame["log_loss"], errors="coerce")
        losses = losses_frame.groupby("config_id", sort=False)["_loss"].mean()
        if "n_samples" in frame.columns:
            weights = pd.to_numeric(frame["n_samples"], errors="coerce")
            valid = (
                np.isfinite(weights)
                & weights.gt(0)
                & np.isfinite(losses_frame["_loss"])
            )
            weighted = losses_frame.loc[valid, ["config_id", "_loss"]].copy()
            weighted["_weighted_loss"] = weighted["_loss"] * weights.loc[valid]
            weighted["_weight"] = weights.loc[valid]
            sums = weighted.groupby("config_id", sort=False)[
                ["_weighted_loss", "_weight"]
            ].sum()
            if not sums.empty:
                losses.update(sums["_weighted_loss"] / sums["_weight"])
        table = pd.DataFrame(
            {"config_id": scores.index, "AUROC": scores.to_numpy(dtype=float)}
        )
        table["log_loss"] = table["config_id"].map(losses)
        ranked = rank_auroc_log_loss(
            table, "AUROC", "log_loss", auroc_tolerance, ("config_id",)
        )
        return scores.loc[ranked["config_id"].tolist()]
    if _metric_is_loss(metric):
        ordered_ids = sorted(
            scores.index,
            key=lambda config_id: (float(scores[config_id]), str(config_id)),
        )
    else:
        ordered_ids = sorted(
            scores.index,
            key=lambda config_id: (-float(scores[config_id]), str(config_id)),
        )
    return scores.loc[ordered_ids]


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
    final_oof_predictions: pd.DataFrame | None = None,
    probability_diagnostics_path: Path | None = None,
) -> pd.DataFrame:
    _validate_plan(plan)
    pooled_grid = (
        outer_split_key is None
        and str(getattr(plan, "selection_mode", "nested")) == "pooled_grid"
    )
    if (
        not pooled_grid
        and outer_split_key is None
        and len(_plan_methods(plan)) == 1
        and _plan_methods(plan)[0] in _PARETO_SELECTORS
        and len(_plan_aggregations(plan)) == 1
        and _plan_aggregations(plan)[0] in _PARETO_AGGREGATIONS
        and not _late_integrations(plan)
    ):
        return _candidate_table_for_pooled_pareto(
            inner_results,
            inner_predictions,
            configs,
            plan,
            metric,
            qualification,
            progress_callback,
            str(progress_scope or "__final__"),
        )
    scope = str(progress_scope or outer_split_key or "__final__")

    def report_preparation(detail: str) -> None:
        if pooled_grid and progress_callback is not None:
            progress_callback(scope, "prepare", None, 0, 0, detail)

    report_preparation("Checking eligible configuration metadata")
    eligible = _eligible_config_ids(configs, plan)
    if pooled_grid:
        qualified = _qualified_config_ids_for_splits(
            qualification, set(inner_results["split_key"].astype(str).unique())
        )
        if qualified is not None:
            eligible &= set(qualified)
        report_preparation(
            f"Checking inner-fold and probability completeness for {len(eligible):,} models"
        )
        eligible &= _complete_pooled_grid_ids(
            inner_results,
            inner_predictions,
            eligible,
            metric,
            progress_detail=report_preparation,
        )
        if not eligible:
            raise ValueError(
                "No fully evaluated, probability-complete pooled-grid models"
            )
        report_preparation(
            f"Aggregating inner metrics for {len(eligible):,} complete models"
        )
        scores = _pooled_grid_inner_scores(
            inner_results, eligible, metric, plan.auroc_tolerance
        )
    else:
        scores = _complete_inner_scores(
            inner_results,
            inner_predictions,
            outer_split_key,
            metric,
            eligible,
            qualification,
            plan.auroc_tolerance,
        )
    if scores.empty:
        return pd.DataFrame()
    report_preparation(f"Filtering probabilities for {len(scores):,} ranked models")
    pred = _prediction_frame_for_outer(inner_predictions, outer_split_key)
    pred = pred[pred["config_id"].isin(set(scores.index))].copy()
    if pooled_grid:
        report_preparation(
            f"Collapsing repeated OOF observations across {len(scores):,} models"
        )
        pred = _pooled_grid_unique_predictions(pred, progress_detail=report_preparation)
    pcols = _proba_cols(pred)
    if not pcols:
        raise ValueError("Inner predictions do not contain probability columns.")
    probability_diagnostics = pd.DataFrame()
    if pooled_grid and plan.probability_degeneracy_policy != "off":
        report_preparation(
            f"Checking probability degeneracy across {len(scores):,} models"
        )
        probability_diagnostics = _probability_diagnostics(
            pred, pcols, float(plan.probability_degeneracy_tolerance)
        )
        if plan.probability_degeneracy_policy == "exclude":
            rejected = set(
                probability_diagnostics.loc[
                    probability_diagnostics["near_constant"], "config_id"
                ]
            )
            scores = scores.loc[~scores.index.astype(str).isin(rejected)]
            pred = pred.loc[
                pred["config_id"].astype(str).isin(set(scores.index))
            ].copy()
            if len(scores) < 2:
                raise ValueError(
                    "Probability degeneracy exclusion left fewer than two eligible models."
                )
        if probability_diagnostics_path is not None:
            probability_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            probability_diagnostics.to_csv(probability_diagnostics_path, index=False)
    if pooled_grid:
        _build_pooled_grid_matrix(
            pred,
            [str(value) for value in scores.index.tolist()],
            progress_detail=report_preparation,
        )
    rows: list[dict[str, Any]] = []
    specs = _ensemble_configs(plan, configs)
    if pooled_grid and progress_callback is not None:
        progress_callback(
            scope,
            "ready",
            None,
            0,
            0,
            f"{len(scores):,} complete models · {len(pred):,} pooled probability rows · {len(specs)} candidates",
        )
    pareto_cache: dict[str, list[str]] = {}
    capped_cache: dict[tuple[str, int], tuple[list[str], dict[str, Any]]] = {}
    for index, spec in enumerate(specs, start=1):
        if progress_callback is not None:
            progress_callback(scope, "start", spec, index, len(specs), "evaluating")
        row = None
        failed = False
        candidate_status = "skipped"
        try:
            method = str(spec["selection_strategy"])
            if (
                final_oof_predictions is not None
                and str(spec["aggregation_strategy"])
                in {
                    "ecdf_grouped_stacking",
                    "ecdf_weighted_quantile",
                    "ecdf_learned_quantile",
                }
                and method == "top_k"
                and str(spec.get("ensemble_kind", "model_ensemble")) == "model_ensemble"
            ):
                members = _select_top_k(scores, int(spec["max_size"]))
                if len(members) >= 2:
                    outer_columns = _proba_cols(final_oof_predictions)
                    if not outer_columns:
                        raise ValueError(
                            "Development OOF predictions have no probability columns."
                        )
                    ordered, outer_stack = _aligned_outer_oof_stacking(
                        final_oof_predictions, members, outer_columns
                    )
                    row = _evaluate_selected_members(
                        spec,
                        members,
                        outer_stack,
                        ordered["y_true"].to_numpy(dtype=int),
                        metric,
                        ordered=ordered,
                        aggregation_parameters_override=(
                            _ecdf_rank_parameters(plan, members, outer_stack, configs)
                            if str(spec["aggregation_strategy"])
                            == "ecdf_weighted_quantile"
                            else None
                        ),
                        extra={
                            "inner_oof_rows": len(ordered),
                            "candidate_library_policy": "unique_development_lodo_oof",
                            "candidate_library_size": len(scores),
                            "selection_weight_source": "none",
                            "stacking_oof_unique_samples": int(
                                ordered["sample_id"].nunique()
                            ),
                            "stacking_oof_studies": int(
                                ordered["outer_split_key"].nunique()
                            ),
                        },
                    )
            elif str(spec.get("ensemble_kind", "model_ensemble")) == "late_fusion":
                row = _candidate_from_late_fusion(
                    spec, scores, pred, configs, pcols, plan, metric
                )
            elif method in {
                "top_k",
                "oof_top_k",
                "oof_similarity",
                "best_per_resolution",
                "best_per_learner_type",
                "best_deep_ml_pair",
                "best_deep_ml_complementary_pair",
                "quality_diversity",
                "diversity",
                "performance_diversity",
                "pareto_optimal_auroc_logloss",
                "pareto_optimal_ml_dl_auroc_logloss",
                "pareto_optimal_ml_dl_auroc_logloss_capped",
                "pareto_optimal_taxonomic_rank_auroc_logloss",
                "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
            }:
                if pooled_grid and method in pareto_cache:
                    pareto_members = pareto_cache[method]
                else:
                    pareto_members = (
                        _pareto_ml_dl_member_ids(
                            inner_results,
                            set(scores.index.astype(str)),
                            outer_split_key,
                            configs,
                        )
                        if method
                        in {
                            "pareto_optimal_ml_dl_auroc_logloss",
                            "pareto_optimal_ml_dl_auroc_logloss_capped",
                        }
                        else _pareto_taxonomic_rank_member_ids(
                            inner_results,
                            set(scores.index.astype(str)),
                            outer_split_key,
                            configs,
                        )
                        if method
                        in {
                            "pareto_optimal_taxonomic_rank_auroc_logloss",
                            "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
                        }
                        else _pareto_inner_member_ids(
                            inner_results,
                            set(scores.index.astype(str)),
                            outer_split_key,
                        )
                        if method == "pareto_optimal_auroc_logloss"
                        else None
                    )
                    if pooled_grid:
                        pareto_cache[method] = list(pareto_members or [])
                capped_metadata: dict[str, Any] = {}
                cap_key = (method, int(spec["max_size"]))
                if pooled_grid and cap_key in capped_cache:
                    pareto_members, capped_metadata = capped_cache[cap_key]
                elif method == "pareto_optimal_ml_dl_auroc_logloss_capped":
                    pareto_members, capped_metadata = _capped_ml_dl_pareto_members(
                        pareto_members or [],
                        inner_results,
                        pred,
                        configs,
                        int(spec["max_size"]),
                        outer_split_key,
                    )
                    if pooled_grid:
                        capped_cache[cap_key] = (pareto_members, capped_metadata)
                elif method == "pareto_optimal_taxonomic_rank_auroc_logloss_capped":
                    pareto_members, capped_metadata = (
                        _capped_taxonomic_rank_pareto_members(
                            pareto_members or [],
                            inner_results,
                            pred,
                            configs,
                            int(spec["max_size"]),
                            outer_split_key,
                        )
                    )
                    if pooled_grid:
                        capped_cache[cap_key] = (pareto_members, capped_metadata)
                row = _candidate_from_simple_selector(
                    spec,
                    scores,
                    pred,
                    configs,
                    pcols,
                    metric,
                    plan,
                    pareto_members=pareto_members,
                )
                if row is not None:
                    row.update(capped_metadata)
            elif method in {"qo_es", "qdo_es", "drep", "pseo_bqp"}:
                row = _candidate_from_modern_selector(
                    spec, scores, pred, pcols, metric, plan, configs
                )
            elif method in {"caruana", "caruana_constrained"}:
                row = _candidate_from_caruana(
                    spec, scores, pred, pcols, metric, plan.auroc_tolerance, plan
                )
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
            if pooled_grid:
                if row is None:
                    rows.append(
                        {
                            **spec,
                            "search_status": "skipped",
                            "error_message": "Insufficient aligned members or predictions",
                        }
                    )
                else:
                    candidate_status = "valid"
                    row["search_status"] = "valid"
                    row["error_message"] = ""
                    row["selection_basis"] = "pooled_inner_oof_grid_search"
                    if not probability_diagnostics.empty:
                        index_diagnostics = probability_diagnostics.set_index(
                            "config_id"
                        )
                        selected = [
                            str(value)
                            for value in _parse_json_list(row["members"], "members")
                        ]
                        selected_diagnostics = index_diagnostics.reindex(selected)
                        row["near_constant_member_count"] = int(
                            selected_diagnostics["near_constant"].fillna(False).sum()
                        )
                        row["near_constant_member_ids"] = json.dumps(
                            [
                                value
                                for value in selected
                                if value in index_diagnostics.index
                                and bool(index_diagnostics.at[value, "near_constant"])
                            ]
                        )
                        row["selected_member_probability_ranges"] = json.dumps(
                            {
                                member: float(
                                    index_diagnostics.at[member, "probability_range"]
                                )
                                for member in selected
                                if member in index_diagnostics.index
                            }
                        )
                        row["probability_degeneracy_policy"] = str(
                            plan.probability_degeneracy_policy
                        )
                    rows.append(row)
            elif row is not None:
                rows.append(row)
                candidate_status = "valid"
        except Exception as exc:
            if pooled_grid:
                candidate_status = "failed"
                rows.append(
                    {
                        **spec,
                        "search_status": "failed",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )
                if progress_callback is not None:
                    progress_callback(scope, "error", spec, index, len(specs), str(exc))
            else:
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
                    candidate_status,
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    score_col = f"{metric}_mean"
    if score_col not in out.columns:
        if pooled_grid:
            out[score_col] = np.nan
        else:
            raise ValueError(
                f"The requested ensemble metric {metric!r} was not produced for candidates."
            )
    if pooled_grid:
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
        out.loc[~np.isfinite(out[score_col]), "search_status"] = out.loc[
            ~np.isfinite(out[score_col]), "search_status"
        ].replace("valid", "invalid_score")
        out["selection_score_basis"] = "pooled_inner_oof_apparent_score"
        out["selection_metric"] = str(metric)
    else:
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
    priority = (
        "_method_priority",
        "_aggregation_priority",
        "member_count",
        "max_size",
        "ensemble_config_id",
    )
    if pooled_grid:
        return _rank_pooled_grid_candidates(
            out, metric, plan.auroc_tolerance, priority, plan.final_selection_strategy
        )
    if plan.auroc_tolerance is not None:
        out = rank_auroc_log_loss(
            out, score_col, "log_loss_mean", plan.auroc_tolerance, priority
        )
        out["selection_rule"] = "auroc_constrained_log_loss"
        out["auroc_tolerance"] = float(plan.auroc_tolerance)
    else:
        out = out.sort_values(
            [score_col, *priority],
            ascending=[_metric_is_loss(metric), *([True] * len(priority))],
            kind="mergesort",
        )
    return out.drop(columns=["_method_priority", "_aggregation_priority"]).reset_index(
        drop=True
    )


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
        if aggregation in {"weighted_mean_proba", "cawpe"} and len(members) != len(
            weights
        ):
            raise RuntimeError(
                f"Selected weighted ensemble for {outer_key!r} has inconsistent members/weights."
            )
        active_weights = (
            weights if aggregation in {"weighted_mean_proba", "cawpe"} else None
        )
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
                "inner_log_loss": float(winner["log_loss_mean"]),
                "selection_rule": "auroc_constrained_log_loss"
                if plan.auroc_tolerance is not None
                else "single_metric",
                "auroc_tolerance": plan.auroc_tolerance,
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
    outer_predictions: pd.DataFrame | None = None,
    probability_diagnostics_path: Path | None = None,
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
        final_oof_predictions=outer_predictions,
        probability_diagnostics_path=probability_diagnostics_path,
    )
    if candidates.empty:
        return ({}, candidates)
    if str(getattr(plan, "selection_mode", "nested")) == "pooled_grid":
        valid = candidates["search_status"].astype(str).eq("valid")
        if not valid.any():
            return ({}, candidates)
    score_col = f"{metric}_mean"
    candidates["inner_score"] = pd.to_numeric(candidates[score_col], errors="coerce")
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
        "selection_basis": (
            "unique_development_lodo_oof_for_final_refit"
            if outer_predictions is not None
            and str(first["aggregation_strategy"])
            in {
                "ecdf_grouped_stacking",
                "ecdf_weighted_quantile",
                "ecdf_learned_quantile",
            }
            else "all_inner_oof_predictions_for_final_refit"
        ),
        "selection_metric": str(metric),
        "selection_rule": str(first.get("selection_rule", "single_metric")),
        "final_selection_strategy": str(
            getattr(plan, "final_selection_strategy", "single_metric")
        ),
        "auroc_tolerance": plan.auroc_tolerance,
        "pareto_optimal": bool(first.get("pareto_optimal", False)),
        "pareto_rank": int(first["pareto_rank"])
        if pd.notna(first.get("pareto_rank"))
        else None,
        "pareto_ideal_distance": float(first["pareto_ideal_distance"])
        if pd.notna(first.get("pareto_ideal_distance"))
        else None,
        "inner_log_loss": float(first["log_loss_mean"]),
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
    if bool(getattr(sweep.evaluation, "outer_only", False)):
        raise ValueError(
            "MPMA-E ensemble selection requires inner-fold validation results. "
            "Set outer_only=False and rerun evaluation in a new experiment "
            "directory, or skip the ensemble stage for outer-only runs."
        )
    if sweep_task(sweep) == "regression":
        from .regression_ensemble import sweep_regression_ensemble

        return sweep_regression_ensemble(sweep)
    from ._config_store import refresh_catalog

    refresh_catalog(Path(sweep.root()))
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
    from .console import info

    configs = read_table(config_path)
    qualification = None
    pooled_grid = (
        str(getattr(sweep.ensemble, "selection_mode", "nested")) == "pooled_grid"
    )
    fixed_pooled_only = pooled_grid or (
        bool(getattr(sweep.ensemble, "include_model_ensembles", True))
        and len(_plan_methods(sweep.ensemble)) == 1
        and _plan_methods(sweep.ensemble)[0] in _PARETO_SELECTORS
        and len(_plan_aggregations(sweep.ensemble)) == 1
        and _plan_aggregations(sweep.ensemble)[0] in _PARETO_AGGREGATIONS
        and not _late_integrations(sweep.ensemble)
    )
    if fixed_pooled_only and not pooled_grid:
        from ._config_store import (
            indexed_config_ids,
            read_configuration,
            read_configurations,
        )
        from .console import progress
        from concurrent.futures import ThreadPoolExecutor
        import sqlite3

        artifact_index = root / "configs.db"
        if artifact_index.is_file():
            with sqlite3.connect(artifact_index) as connection:
                has_index = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='config_artifacts'"
                ).fetchone()
                if has_index:
                    completed_ids = {
                        str(row[0])
                        for row in connection.execute(
                            "SELECT config_id FROM config_artifacts "
                            "WHERE expected_outer_count > 0 AND outer_count >= expected_outer_count "
                            "AND inner_count > 0"
                        )
                    }
                    configs = configs.loc[
                        configs["config_id"].astype(str).isin(completed_ids)
                    ].copy()
        stage("Ensemble sweep", str(root))
        info("Fixed pooled Pareto mode · no fold-specific selection")
        if inner_result_path.is_file() and inner_result_path.stat().st_size > 0:
            info("Reading cached inner-validation metrics")
            inner_results = read_table(inner_result_path)
        else:
            completed_set = set(configs["config_id"].astype(str))
            identifiers = [
                identifier
                for identifier in indexed_config_ids(root)
                if identifier in completed_set
            ]
            if not identifiers:
                raise ValueError(
                    "No configuration artifacts found for pooled Pareto selection"
                )
            frames = []
            with progress() as prog:
                task = prog.add_task(
                    "Loading inner-validation metrics", total=len(identifiers)
                )
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for index, frame in enumerate(
                        pool.map(
                            lambda identifier: read_configuration(
                                root, identifier, "inner_metrics"
                            ),
                            identifiers,
                        ),
                        start=1,
                    ):
                        if not frame.empty:
                            frames.append(frame)
                        prog.update(
                            task,
                            completed=index,
                            description=f"Loading inner-validation metrics · {index:,}/{len(identifiers):,}",
                        )
            inner_results = (
                pd.concat(frames, ignore_index=True, sort=False)
                if frames
                else pd.DataFrame()
            )
        info("Finding fully evaluated pooled AUROC–log-loss Pareto candidates")
        eligible = _eligible_config_ids(configs, sweep.ensemble)
        if sweep.gate.enabled:
            qualification_path = root / "tables" / "qualification_gate.parquet"
            if not table_exists(qualification_path):
                raise FileNotFoundError(
                    "Qualification gate is enabled but qualification data is missing"
                )
            qualification = read_table(qualification_path)
            qualified = _qualified_config_ids_for_splits(
                qualification, set(inner_results["split_key"].astype(str).unique())
            )
            if qualified is not None:
                eligible &= set(qualified)
        if _plan_methods(sweep.ensemble)[0] in {
            "pareto_optimal_ml_dl_auroc_logloss",
            "pareto_optimal_ml_dl_auroc_logloss_capped",
        }:
            members = _pareto_ml_dl_member_ids(inner_results, eligible, None, configs)
        elif _plan_methods(sweep.ensemble)[0] in {
            "pareto_optimal_taxonomic_rank_auroc_logloss",
            "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
        }:
            members = _pareto_taxonomic_rank_member_ids(
                inner_results, eligible, None, configs
            )
        else:
            members = _pareto_inner_member_ids(inner_results, eligible, None)
        if not members:
            raise ValueError("No complete pooled inner Pareto candidates found")
        info(
            f"Pooled Pareto selection identified {len(members)} candidate members · loading only their probabilities"
        )
        inner_predictions = read_configurations(root, "inner_predictions", members)
        if inner_predictions.empty:
            raise ValueError("Pooled Pareto members have no saved inner predictions")
        outer_predictions = pd.DataFrame()
    else:
        if not pooled_grid:
            info("Reading outer-test predictions")
            outer_predictions = read_table(outer_path)
        else:
            outer_predictions = pd.DataFrame()
        info("Reading inner-validation metrics")
        inner_results = read_table(inner_result_path)
        info("Reading inner-validation probabilities")
        inner_predictions = read_table(inner_prediction_path)
        if pooled_grid:
            stage("Ensemble sweep", str(root))
            info("Fixed pooled ensemble grid · no outer-fold selection")
    if "cross_study_dro" in sweep.ensemble.aggregation_strategies:
        if not sweep.data.group_col:
            raise ValueError(
                "cross_study_dro requires Data.group_col identifying true studies."
            )
        study_data = load_dataset(sweep.data, ("all",))
        if sweep.data.group_col not in study_data.metadata.columns:
            raise ValueError(
                "The declared study column is absent from training metadata."
            )
        mapping = pd.DataFrame(
            {
                "sample_id": [str(x) for x in study_data.sample_ids],
                "study_group": study_data.metadata[sweep.data.group_col]
                .astype(str)
                .str.strip()
                .to_numpy(),
            }
        )
        if (
            mapping["sample_id"].duplicated().any()
            or mapping["study_group"].eq("").any()
            or study_data.metadata[sweep.data.group_col].isna().any()
        ):
            raise ValueError(
                "Study identifiers must be nonmissing and uniquely linked to samples."
            )
        inner_predictions = inner_predictions.merge(
            mapping, on="sample_id", how="left", validate="many_to_one"
        )
        if inner_predictions["study_group"].isna().any():
            raise ValueError("Some inner prediction samples have no declared study.")
    if sweep.gate.enabled:
        qualification_path = root / "tables" / "qualification_gate.parquet"
        if not table_exists(qualification_path):
            raise FileNotFoundError(
                "Qualification gate is enabled but qualification_gate.parquet is missing. Rerun evaluate(sweep)."
            )
        qualification = read_table(qualification_path)
    metric = canonical_metric_name(str(sweep.ensemble.optimize_metric))
    if not fixed_pooled_only:
        stage("Ensemble sweep", str(root))
    summary_table(
        "Ensemble search",
        {
            "candidate ensemble configurations": f"{len(_ensemble_configs(sweep.ensemble, configs)):,}",
            "selection mode": "fixed pooled grid"
            if pooled_grid
            else "fixed pooled Pareto"
            if fixed_pooled_only
            else "nested CV",
            "selection strategies": _plan_methods(sweep.ensemble),
            "aggregation strategies": _plan_aggregations(sweep.ensemble),
            "max sizes": _plan_max_sizes(sweep.ensemble),
            "learned-selector library": "all eligible complete inner-OOF MPMAs",
            "pooled Super Learner screening": (
                f"top {sweep.ensemble.pooled_super_learner_pool_size} ranked models"
                if pooled_grid and sweep.ensemble.pooled_super_learner_pool_size
                else "disabled"
            ),
            "Caruana stopping": f"canonical fixed {sweep.ensemble.caruana_iterations} iterations"
            + (" with patience" if sweep.ensemble.caruana_early_stopping else ""),
            "Caruana unique-member cap": sweep.ensemble.caruana_max_unique_members
            or "none",
            "Caruana sorted initialization": sweep.ensemble.caruana_sorted_initial_models,
            "legacy constrained Caruana": "caruana_constrained"
            if "caruana_constrained" in _plan_methods(sweep.ensemble)
            else "not requested",
            "probability degeneracy": sweep.ensemble.probability_degeneracy_policy,
            "modern selection library screen": sweep.ensemble.advanced_selection_pool_size,
            "population search evaluations": sweep.ensemble.population_iterations,
            "population archive capacity": sweep.ensemble.population_archive_size,
            "Super Learner loss": _resolved_super_learner_loss(sweep.ensemble),
            "excluded learners": sweep.ensemble.exclude_learners or "none",
            "excluded resolutions": sweep.ensemble.exclude_resolutions or "none",
            "excluded transformations": sweep.ensemble.exclude_transformations
            or "none",
            "optimize metric": metric,
            "AUROC/log-loss tolerance": sweep.ensemble.auroc_tolerance,
            "final selection strategy": sweep.ensemble.final_selection_strategy,
        },
    )
    resource_tracker = ResourceTracker(
        sample_interval_s=float(
            getattr(sweep.evaluation, "resource_sample_interval_s", 0.1)
        )
    ).start()
    outer_keys = sorted(set(inner_results["split_key"].astype(str)))
    ensemble_specs = _ensemble_configs(sweep.ensemble, configs)
    scopes = ["__final__"] if fixed_pooled_only else outer_keys + ["__final__"]
    with EnsembleSearchProgress(
        scopes, len(ensemble_specs), pooled_grid=pooled_grid
    ) as live:
        if not fixed_pooled_only:
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
            if (
                selection.empty
                or selected_outer_predictions.empty
                or fold_metrics.empty
            ):
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
            outer_predictions=None if pooled_grid else outer_predictions,
            probability_diagnostics_path=(
                ensemble_dir / "pooled_model_probability_diagnostics.csv"
                if pooled_grid
                else None
            ),
        )
    if not final_ensemble:
        if pooled_grid and not candidates.empty:
            write_table(ensemble_dir / "ensemble_candidate_scores.parquet", candidates)
            candidates.to_csv(
                ensemble_dir / "ensemble_candidate_scorecard.csv", index=False
            )
        raise RuntimeError(
            "No valid final ensemble candidate was produced from inner OOF predictions."
        )
    nested_summary = (
        {} if fixed_pooled_only else summarize_mpma_e_strategy(fold_metrics)
    )
    final_mpma = select_final_mpma_candidate(
        inner_results,
        configs,
        canonical_metric_name(str(sweep.evaluation.optimize_metric)),
        plan=sweep.ensemble,
        qualification=qualification,
        inner_predictions=inner_predictions,
        base_selection=sweep.base_selection,
    )
    selection_path = ensemble_dir / "mpma_e_outer_selection.parquet"
    prediction_path = ensemble_dir / "ensemble_predictions.parquet"
    result_path = ensemble_dir / "mpma_e_outer_results.parquet"
    candidate_path = ensemble_dir / "ensemble_candidate_scores.parquet"
    summary_path = ensemble_dir / "mpma_e_strategy_summary.json"
    final_path = ensemble_dir / "mpma_e_final_candidate.json"
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
    from .fixed_final_performance import fixed_final_outer_performance

    if pooled_grid and sweep.ensemble.probability_degeneracy_policy != "off":
        heldout = read_table(outer_path)
        heldout_columns = _proba_cols(heldout)
        if heldout_columns:
            tolerance = float(sweep.ensemble.probability_degeneracy_tolerance)
            summary = [
                _probability_diagnostics(
                    heldout,
                    heldout_columns,
                    tolerance,
                    source="outer_fold_predictions_post_selection",
                )
            ]
            if "outer_split_key" in heldout.columns:
                summary.append(
                    _probability_diagnostics(
                        heldout,
                        heldout_columns,
                        tolerance,
                        source="outer_fold_by_split_post_selection",
                        group_columns=("config_id", "outer_split_key"),
                    )
                )
            pd.concat(summary, ignore_index=True, sort=False).to_csv(
                ensemble_dir / "heldout_model_probability_diagnostics.csv", index=False
            )
        del heldout
    if fixed_pooled_only:
        info(
            f"Pooled selection complete · {final_ensemble['member_count']} fixed members · evaluating held-out probabilities"
        )
    fixed_predictions, fixed_summary = fixed_final_outer_performance(
        root, "MPMA-E", persist=True
    )
    if fixed_pooled_only:
        info(
            f"Held-out evaluation complete · {len(fixed_summary) and fixed_summary.get('n_outer_folds', 0)} outer folds · saving results"
        )
    if fixed_pooled_only:
        selected_outer_predictions = fixed_predictions.copy()
        fold_metrics = read_table(ensemble_dir / "mpma_e_final_outer_results.parquet")
        members_json = json.dumps(final_ensemble["members"])
        weights_json = json.dumps(final_ensemble["weights"])
        selected_outer_predictions["members"] = members_json
        selected_outer_predictions["weights"] = weights_json
        selected_outer_predictions["selection_strategy"] = str(
            final_ensemble["selection_strategy"]
        )
        selected_outer_predictions["aggregation_strategy"] = str(
            final_ensemble["aggregation_strategy"]
        )
        selected_outer_predictions["max_size"] = int(final_ensemble["max_size"])
        selected_outer_predictions["selection_basis"] = (
            "pooled_inner_validation_fixed_members"
        )
        selection = pd.DataFrame(
            [
                {
                    "outer_split_key": str(outer_key),
                    "ensemble_config_id": str(final_ensemble["ensemble_config_id"]),
                    "selection_strategy": str(final_ensemble["selection_strategy"]),
                    "aggregation_strategy": str(final_ensemble["aggregation_strategy"]),
                    "max_size": int(final_ensemble["max_size"]),
                    "members": members_json,
                    "weights": weights_json,
                    "selection_basis": "pooled_inner_validation_fixed_members",
                }
                for outer_key in sorted(
                    selected_outer_predictions["outer_split_key"].astype(str).unique()
                )
            ]
        )
        nested_summary = {
            "Strategy": "MPMA-E",
            "selection_basis": "pooled_inner_validation_fixed_members",
            "performance_basis": "retrospective_fixed_final_specification_outer_cv",
            "n_outer_folds": len(fold_metrics),
            "member_count": len(final_ensemble["members"]),
        }
        nested_summary.update(
            {
                key: value
                for key, value in fixed_summary.items()
                if key.startswith("outer_")
            }
        )
    write_table(selection_path, selection)
    write_table(prediction_path, selected_outer_predictions)
    write_table(result_path, fold_metrics)
    write_table(candidate_path, candidates)
    if pooled_grid:
        csv_path = ensemble_dir / "ensemble_candidate_scorecard.csv"
        candidates.to_csv(csv_path, index=False)
        valid_candidates = candidates.loc[candidates["search_status"].eq("valid")]
        info(
            f"Pooled ensemble grid scored {len(valid_candidates)}/{len(candidates)} candidates · winner {final_ensemble['selection_strategy']} + {final_ensemble['aggregation_strategy']} · max {final_ensemble['max_size']} · inner {metric}={final_ensemble['inner_score']:.4f}"
        )
        summary_table(
            "Top ensemble candidates",
            {
                f"{index}. {row['selection_strategy']} + {row['aggregation_strategy']} · max {int(row['max_size'])}": (
                    f"AUROC={float(row['AUROC_mean']):.5f} · log_loss={float(row['log_loss_mean']):.5f} · Pareto rank={int(row['pareto_rank'])} · distance={float(row['pareto_ideal_distance']):.5f} · members={int(row['member_count'])}"
                    if sweep.ensemble.final_selection_strategy
                    == "pareto_auroc_logloss_ideal"
                    and bool(row["pareto_optimal"])
                    else f"{metric}={float(row['inner_score']):.5f} · members={int(row['member_count'])}"
                )
                for index, (_, row) in enumerate(valid_candidates.head(5).iterrows(), 1)
            },
        )
    dump_json_standard(nested_summary, summary_path)
    selected = {
        "inner_val_best_mpma": final_mpma,
        "inner_val_best_mpmas_ensemble": final_ensemble,
        "nested_mpma_e": nested_summary if not fixed_pooled_only else {},
        "fixed_mpma_e": nested_summary if fixed_pooled_only else {},
        "terminology": {
            "MPMA-B": "final candidate selected from pooled inner-validation results; retrospective fixed outer-CV performance stored separately",
            "MPMA-E": (
                "fixed pooled-inner ensemble grid winner evaluated retrospectively; validation score is apparent and retrospective outer CV is not independent"
                if pooled_grid
                else "fixed pooled-inner Pareto ensemble evaluated retrospectively with invariant members across all outer folds"
                if fixed_pooled_only
                else "fold-specific ensemble learned exclusively from inner out-of-fold predictions; member identities and weights are frozen before outer-test application"
            ),
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
        if fixed_pooled_only:
            info("Preparing MPMA-E figure")
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
        **(
            {
                "mpma_e_candidate_scorecard_csv": ensemble_dir
                / "ensemble_candidate_scorecard.csv"
            }
            if pooled_grid
            else {}
        ),
        "mpma_e_summary": summary_path,
        "mpma_e_final_candidate": final_path,
        "mpma_e_final_outer_predictions": ensemble_dir
        / "mpma_e_final_outer_predictions.parquet",
        "mpma_e_final_outer_results": ensemble_dir
        / "mpma_e_final_outer_results.parquet",
        "mpma_e_final_outer_summary": ensemble_dir / "mpma_e_final_outer_summary.json",
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
