from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import shutil
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_limits

from .configs_sweep import (
    Sweep,
    _effective_local_explanations_mode,
    _groups_from_metadata,
    _lodo_feature_pair,
    _resolved_evaluation_splits,
    _source_tree_sha256,
    _strata_from_metadata,
)
from .console import info, path_table, progress, stage, success, summary_table
from .data import load_dataset
from .explainability_context import (
    build_feature_relative_abundance_summary,
    build_local_relative_abundance_context,
)
from .explainability_methods import (
    ALE,
    LIME,
    SHAP,
    ALEInteractions,
    Permutation,
    coerce_method,
    method_has_global,
    method_has_local,
    method_name,
    method_to_dict,
)
from .explainability_support import top_k_rank_support
from .explainability_visuals import plot_feature_support as _plot_feature_support_visual
from .explainability_visuals import (
    plot_interaction_network as _plot_interaction_network_visual,
)
from .explainability_visuals import plot_local_attributions
from .learners import _learner_factory, fit_classifier
from .metrics import _predict_proba_aligned
from .metrics import metric_is_loss
from .resolutions import mask_feature_blocks, materialize_mpdr_with_blocks
from .runtime import (
    configure_estimator_threads,
    iter_parallel_tasks,
    resolve_execution_plan,
    thread_environment,
)
from .storage import glob_tables, read_table, table_exists, write_table
from .style import ACC_D, ACC_L, BG, COL_W_2, DIM, INK, MID, TRACK
from .style import apply as apply_style
from .style import save_all
from .transformations import CountTransformationAdapter, _count_transformation_factory
from .utils import _as_float_matrix, dump_json_standard, feature_tail_ellipsis

for _logger_name in ("PyALE", "PyALE._ALE_generic"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
    logging.getLogger(_logger_name).propagate = False


from .explainability_algorithms import (
    _AleModelWrapper,
    _ale_1d_effect_summary,
    _ale_feature_importance,
    _ale_interactions,
    _ale_result_values,
    _candidate_pairs_from_scores,
    _combine_feature_importance,
    _interp_unique,
    _permutation_feature_importance,
    _plot_ale_curves,
    _require_pyale,
    _same_lineage_pair,
    _single_method_support_table,
    _write_fold_feature_importance,
    _write_method_outputs,
)
from .explainability_model import (
    _FittedMpmaEnsemble,
    _aggregate_member_proba,
    _explain_predict_class_probability,
    _explain_predict_proba,
    _project_input,
    _projection_required,
    _sample_rows,
)
from .explainability_runtime import (
    _parallel_progress_results,
    _progress_callback,
    _queued_progress_callback,
    _quiet_pyale_info,
    _run_xai_task,
    _xai_execution_plan,
    _xai_task_iterator,
)
from .explainability_values import (
    _aggregate_fold_feature_importance,
    _lime_values_for_data,
    _shap_values_for_data,
    _value_frame_for_fold,
)


from .explainability_config import (
    ExplainabilityConfigurationError,
    ExplainabilityDependencyError,
    _EXPLAINABILITY_PIPELINE_SCHEMA,
    _auto_ale_bins,
    _configured_count_transformation_factory,
    _configured_learner_factory,
    _method_spec,
    _normalise_explainability_method_specs,
    _normalise_explainability_methods,
    _preflight_explainability_dependencies,
    _resolve_explainability_classes,
)


from .explainability_reporting import (
    _class_slug,
    _collapse_duplicate_feature_importance,
    _ensure_interaction_network_outputs,
    _feature_distribution_stats,
    _interaction_distribution_stats_for_class,
    _method_display,
    _method_support_table,
    _plain_taxon_label,
    _plot_feature_importance,
    _plot_interaction_network,
    _rank_support_from_importance,
    _standardized_group_shift,
    _terminal_taxon_label,
    _write_unavailable_interaction_network,
)


from .explainability_cache import (
    _cached_method_frame,
    _copy_explainability_cache,
    _existing_explainability_outputs,
    _existing_method_outputs,
    _explainability_cache_complete,
    _explainability_config_payload,
    _explainability_config_signature,
    _explainability_source_signature,
    _legacy_method_cache_valid,
    _load_method_cache_entry,
    _method_cache_entry,
    _method_cache_entry_status,
    _method_cache_files_complete,
    _method_cache_sidecar_path,
    _method_cache_signature,
    _persist_method_cache_entry,
    _refresh_target_visuals,
    _safe_cache_name,
    _signature_hash,
    _signature_value,
    _source_config_compatible,
    _visual_class_labels,
    refresh_explainability_visuals,
)


def _ensure_mpma_member_explanations(
    sweep: Sweep, rankings: pd.DataFrame, members: Sequence[str]
) -> None:
    if not members:
        return
    methods = _normalise_explainability_methods(sweep.explainability.methods)
    total = len(members)
    info(f"Preparing cached member explainability for MPMA-E · members={total}")
    for i, member in enumerate(members, start=1):
        matches = rankings[rankings["config_id"].astype(str).eq(str(member))]
        if matches.empty:
            matches = rankings[
                rankings["config_id"]
                .astype(str)
                .map(lambda x: x.startswith(str(member)) or str(member).startswith(x))
            ]
        cid = str(matches.iloc[0]["config_id"]) if not matches.empty else str(member)
        cache_dir = sweep.root() / "explainability" / "cache" / _safe_cache_name(cid)
        if _explainability_cache_complete(cache_dir, sweep.explainability, str(member)):
            info(f"MPMA-E member {i}/{total} · config={cid} · cached")
            continue
        if matches.empty:
            raise ExplainabilityConfigurationError(
                f"MPMA-E member {member!r} cannot be matched to mpma_inner_rankings.parquet for cached explanation."
            )
        row = matches.iloc[0]
        desc = " · ".join(
            [
                str(row.get("resolution", row.get("levels", "MPDR"))),
                str(row.get("count_transformation", "transformation")),
                str(row.get("learner", "learner")),
            ]
        )
        info(f"MPMA-E member {i}/{total} · config={cid} · {desc}")
        _explain_one(
            sweep,
            target_override=cid,
            output_slug_override=f"cache/{_safe_cache_name(cid)}",
            display_label_override=f"MPMA-E member {i}/{total}",
            allow_member_cache=False,
        )


def _configured_explainability_targets(explainability: Any) -> str | tuple[str, ...]:
    return getattr(explainability, "targets", "auto")


def _single_configured_explainability_target(explainability: Any) -> str:
    configured = _configured_explainability_targets(explainability)
    if isinstance(configured, str):
        return configured
    values = tuple(str(x) for x in configured)
    if len(values) != 1:
        raise ExplainabilityConfigurationError(
            "Internal error: _explain_one requires one explainability target. "
            "Use explain(sweep) to process multiple targets."
        )
    return values[0]


def _explainability_outer_splits(sweep: Sweep, dataset: Any) -> list[dict[str, Any]]:
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    y = np.asarray(dataset.y, dtype=int)
    strata = _strata_from_metadata(dataset.metadata, y, sweep.data.stratify_col)
    splits, _ = _resolved_evaluation_splits(
        sweep.root(),
        sweep.evaluation,
        dataset,
        groups,
        strata,
        sweep.data.stratify_col,
        sweep.data.group_col,
    )
    if not splits:
        raise ExplainabilityConfigurationError(
            "No outer splits are available for out-of-fold explainability."
        )
    return splits


def _ordered_member_rows(
    root: Path, rankings: pd.DataFrame, members: Sequence[str]
) -> pd.DataFrame:
    config_path = root / "configs.parquet"
    configs = read_table(config_path) if table_exists(config_path) else rankings
    configs = configs.copy()
    configs["config_id"] = configs["config_id"].astype(str)
    rows: list[pd.Series] = []
    for member in members:
        m = str(member)
        match = configs[configs["config_id"].eq(m)]
        if match.empty:
            match = configs[
                configs["config_id"].map(
                    lambda x: str(x).startswith(m) or m.startswith(str(x))
                )
            ]
        if match.empty:
            raise ExplainabilityConfigurationError(
                f"MPMA-E member {member!r} cannot be matched to configs.parquet."
            )
        rows.append(match.iloc[0])
    return pd.DataFrame(rows).drop_duplicates("config_id")


def _fit_oof_single_fold_task(
    split_no: int,
    split: dict[str, Any],
    X_base: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    class_count: int,
    protocol: str,
    feature_names: Sequence[str],
    expected_feature_mask: np.ndarray,
    ct_factory: Callable[[], Any],
    learner_factory: Callable[[], BaseEstimator],
    threads_per_worker: int,
) -> tuple[int, dict[str, Any] | None]:
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    if len(test_idx) == 0 or len(np.unique(y[train_idx])) < 2:
        return int(split_no), None
    X_train_raw, X_test_raw, feature_mask = _lodo_feature_pair(
        X_base, train_idx, test_idx, protocol
    )
    expected_feature_mask = np.asarray(expected_feature_mask, dtype=bool)
    if not np.array_equal(np.asarray(feature_mask, dtype=bool), expected_feature_mask):
        raise ExplainabilityConfigurationError(
            f"LODO feature mask changed while reconstructing outer split {split['split_key']!r}."
        )
    kept_feature_names = [
        str(name)
        for name, keep in zip(feature_names, expected_feature_mask)
        if bool(keep)
    ]
    ct = ct_factory()
    X_train, X_test = ct.apply_pair(X_train_raw, X_test_raw)
    coordinate_metadata = list(ct.coordinate_metadata(kept_feature_names))
    transformed_feature_names = [str(item.name) for item in coordinate_metadata]
    if X_train.shape[1] != len(transformed_feature_names) or X_test.shape[1] != len(
        transformed_feature_names
    ):
        raise ExplainabilityConfigurationError(
            f"Transformation metadata does not match the reconstructed outer split {split['split_key']!r}."
        )
    clf = configure_estimator_threads(learner_factory(), threads_per_worker)
    fit_classifier(
        clf, X_train, y[train_idx], None if groups is None else groups[train_idx]
    )
    proba = _predict_proba_aligned(clf, X_test, np.arange(int(class_count), dtype=int))
    geometry = (
        ct.perturbation_geometry()
        if hasattr(ct, "perturbation_geometry")
        else "unverified_custom"
    )
    projector = (
        ct.project_model_input
        if _projection_required(geometry) and hasattr(ct, "project_model_input")
        else None
    )
    return int(split_no), {
        "split_key": str(split["split_key"]),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "feature_mask": expected_feature_mask,
        "input_feature_names": kept_feature_names,
        "feature_names": transformed_feature_names,
        "coordinate_metadata": coordinate_metadata,
        "X_train": np.asarray(X_train, dtype=float),
        "X_test": np.asarray(X_test, dtype=float),
        "y_test": y[test_idx],
        "estimator": clf,
        "proba": np.asarray(proba, dtype=float),
        "input_projector": projector,
        "perturbation_geometry": geometry,
    }


def _coordinate_semantic_key(item: Any) -> tuple[Any, ...]:
    coefficients = np.asarray(item.coefficients, dtype=float)
    norm = float(np.linalg.norm(coefficients))
    if norm > 0.0:
        coefficients = coefficients / norm
    return (
        str(item.coordinate_type),
        "" if item.anchor_feature is None else str(item.anchor_feature),
        tuple(str(x) for x in item.components),
        tuple(float(x) for x in np.round(coefficients, 12)),
        bool(item.exact_feature_identity),
    )


def _coordinate_semantic_digest(key: tuple[Any, ...]) -> str:
    payload = json.dumps(key, sort_keys=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _align_oof_single_coordinates(
    folds: Sequence[dict[str, Any]], n_samples: int
) -> tuple[np.ndarray, list[str], list[Any], list[dict[str, Any]]]:
    name_keys: dict[str, set[tuple[Any, ...]]] = {}
    fold_pairs: list[list[tuple[Any, tuple[Any, ...]]]] = []
    for fold in folds:
        metadata = list(fold.get("coordinate_metadata", []))
        if len(metadata) != int(np.asarray(fold["X_train"]).shape[1]) or len(
            metadata
        ) != int(np.asarray(fold["X_test"]).shape[1]):
            raise ExplainabilityConfigurationError(
                f"Outer split {fold.get('split_key', '')!r} has coordinate metadata inconsistent with its fitted matrices."
            )
        pairs: list[tuple[Any, tuple[Any, ...]]] = []
        for item in metadata:
            key = _coordinate_semantic_key(item)
            base = str(item.name)
            name_keys.setdefault(base, set()).add(key)
            pairs.append((item, key))
        fold_pairs.append(pairs)
    global_names: list[str] = []
    global_metadata: list[Any] = []
    metadata_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold, pairs in zip(folds, fold_pairs):
        labels: list[str] = []
        renamed_metadata: list[Any] = []
        for item, key in pairs:
            base = str(item.name)
            label = (
                base
                if len(name_keys.get(base, {key})) == 1
                else f"{base}__{_coordinate_semantic_digest(key)}"
            )
            if label in labels:
                raise ExplainabilityConfigurationError(
                    f"Outer split {fold.get('split_key', '')!r} produced duplicate semantic coordinate {label!r}."
                )
            labels.append(label)
            renamed = replace(item, name=label)
            renamed_metadata.append(renamed)
            metadata_rows.append(
                {
                    "split_key": str(fold.get("split_key", "")),
                    "coordinate": label,
                    "fitted_coordinate": base,
                    "coordinate_type": str(item.coordinate_type),
                    "anchor_feature": ""
                    if item.anchor_feature is None
                    else str(item.anchor_feature),
                    "exact_feature_identity": bool(item.exact_feature_identity),
                    "components": json.dumps(
                        list(item.components), separators=(",", ":")
                    ),
                    "coefficients": json.dumps(
                        [float(x) for x in item.coefficients], separators=(",", ":")
                    ),
                    "semantic_digest": _coordinate_semantic_digest(key),
                }
            )
            if label not in seen:
                seen.add(label)
                global_names.append(label)
                global_metadata.append(renamed)
        fold["feature_names"] = labels
        fold["coordinate_metadata"] = renamed_metadata
    reference_sum = np.zeros((int(n_samples), len(global_names)), dtype=float)
    reference_count = np.zeros((int(n_samples), len(global_names)), dtype=np.int64)
    global_index = {name: i for i, name in enumerate(global_names)}
    for fold in folds:
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        X_test = np.asarray(fold["X_test"], dtype=float)
        labels = list(fold["feature_names"])
        for local_index, label in enumerate(labels):
            global_index_value = global_index[str(label)]
            values = X_test[:, local_index]
            finite = np.isfinite(values)
            if np.any(finite):
                rows = test_idx[finite]
                reference_sum[rows, global_index_value] += values[finite]
                reference_count[rows, global_index_value] += 1
    reference = np.full(reference_sum.shape, np.nan, dtype=float)
    valid = reference_count > 0
    reference[valid] = reference_sum[valid] / reference_count[valid]
    return reference, global_names, global_metadata, metadata_rows


def _assert_oof_prediction_reproduction(
    root: Path, row: pd.Series, dataset: Any, folds: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = root / "predictions" / "outer_predictions.parquet"
    if not table_exists(path):
        raise ExplainabilityConfigurationError(
            "Stored outer predictions are required to verify explainability refits."
        )
    stored = read_table(path).copy()
    if "outer_split_key" not in stored.columns and "split_key" in stored.columns:
        stored["outer_split_key"] = stored["split_key"]
    required = {"outer_split_key", "sample_id", "config_id"}
    if not required.issubset(stored.columns):
        raise ExplainabilityConfigurationError(
            "Stored outer predictions do not contain the identifiers required for refit verification."
        )
    config_id = str(row["config_id"])
    proba_columns = [f"proba_{label}" for label in dataset.class_labels]
    if any(column not in stored.columns for column in proba_columns):
        raise ExplainabilityConfigurationError(
            "Stored outer predictions do not contain all class-probability columns required for refit verification."
        )
    stored["outer_split_key"] = stored["outer_split_key"].astype(str)
    stored["sample_id"] = stored["sample_id"].astype(str)
    stored["config_id"] = stored["config_id"].astype(str)
    records: list[dict[str, Any]] = []
    for fold in folds:
        split_key = str(fold["split_key"])
        sample_ids = [
            str(dataset.sample_ids[int(i)])
            for i in np.asarray(fold["test_idx"], dtype=int)
        ]
        sub = stored[
            stored["outer_split_key"].eq(split_key) & stored["config_id"].eq(config_id)
        ].copy()
        if sub.empty or sub["sample_id"].duplicated().any():
            raise ExplainabilityConfigurationError(
                f"Stored outer predictions cannot uniquely verify config {config_id!r} on split {split_key!r}."
            )
        sub = sub.set_index("sample_id").reindex(sample_ids)
        if sub[proba_columns].isna().any().any():
            raise ExplainabilityConfigurationError(
                f"Stored outer predictions are incomplete for config {config_id!r} on split {split_key!r}."
            )
        expected = sub[proba_columns].to_numpy(dtype=float)
        observed = np.asarray(fold["proba"], dtype=float)
        if expected.shape != observed.shape:
            raise ExplainabilityConfigurationError(
                f"Explainability refit probability shape does not match stored evaluation output on split {split_key!r}."
            )
        error = float(np.max(np.abs(expected - observed))) if expected.size else 0.0
        if not np.allclose(expected, observed, rtol=1e-7, atol=1e-9):
            raise ExplainabilityConfigurationError(
                f"Explainability refit does not reproduce evaluated outer probabilities for config {config_id!r} on split {split_key!r}; max_abs_error={error:.3e}."
            )
        fold["prediction_reproduction_max_abs_error"] = error
        records.append(
            {
                "split_key": split_key,
                "config_id": config_id,
                "max_abs_probability_error": error,
                "rtol": 1e-7,
                "atol": 1e-9,
                "verified": 1,
            }
        )
    return records


def _fold_feature_names(fold: dict[str, Any], fallback: Sequence[str]) -> list[str]:
    names = [str(x) for x in fold.get("feature_names", fallback)]
    width = int(np.asarray(fold["X_test"]).shape[1])
    if len(names) != width:
        raise ExplainabilityConfigurationError(
            f"Outer split {fold.get('split_key', '')!r} has {width} model coordinates but {len(names)} coordinate labels."
        )
    return names


def _fit_oof_single_for_explainability(
    sweep: Sweep,
    row: pd.Series,
) -> dict[str, Any]:
    if getattr(sweep, "uses_modalities", False):
        from .multimodal_sweep import fit_modality_candidate_oof_for_explainability

        return fit_modality_candidate_oof_for_explainability(sweep, row)
    levels = tuple(str(row["levels"]).split(","))
    dataset = load_dataset(sweep.data, levels)
    X_base, feature_names, feature_blocks = materialize_mpdr_with_blocks(
        dataset, levels
    )
    X_base = np.asarray(X_base)
    feature_names = [str(x) for x in feature_names]
    splits = _explainability_outer_splits(sweep, dataset)
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    transformation_key = str(row["count_transformation"])
    learner_key = str(row["learner"])
    learner_factory = _configured_learner_factory(sweep, learner_key)
    execution = _xai_execution_plan(sweep, len(splits))
    tasks = []
    for split_no, split in enumerate(splits, start=1):
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
            feature_mask = np.ones(X_base.shape[1], dtype=bool)
        else:
            _, _, feature_mask = _lodo_feature_pair(
                X_base, train_idx, test_idx, str(sweep.evaluation.protocol)
            )
        fold_blocks = mask_feature_blocks(feature_blocks, feature_mask)
        ct_factory = _configured_count_transformation_factory(
            sweep,
            transformation_key,
            fold_blocks,
            resolution_feature_blocks=feature_blocks,
        )
        tasks.append(
            (
                _fit_oof_single_fold_task,
                (
                    split_no,
                    split,
                    X_base,
                    dataset.y,
                    groups,
                    len(dataset.class_labels),
                    str(sweep.evaluation.protocol),
                    feature_names,
                    np.asarray(feature_mask, dtype=bool),
                    ct_factory,
                    learner_factory,
                    int(execution.threads_per_worker),
                ),
                {},
            )
        )
    folds_by_no: dict[int, dict[str, Any]] = {}
    with progress() as prog:
        task = prog.add_task(
            f"Fitting OOF explanation models · {execution.workers} workers · {execution.threads_per_worker} threads/worker",
            total=len(tasks),
        )
        for split_no, fold in _xai_task_iterator(tasks, execution):
            if fold is not None:
                folds_by_no[int(split_no)] = fold
            prog.advance(task)
    folds = [folds_by_no[i] for i in sorted(folds_by_no)]
    if not folds:
        raise ExplainabilityConfigurationError(
            "No outer fold could be fitted for out-of-fold explainability."
        )
    reproduction = _assert_oof_prediction_reproduction(
        sweep.root(), row, dataset, folds
    )
    X_reference, transformed_feature_names, coordinate_metadata, coordinate_rows = (
        _align_oof_single_coordinates(folds, len(dataset.sample_ids))
    )
    geometries = {
        str(fold.get("perturbation_geometry", "unverified_custom")) for fold in folds
    }
    return {
        "dataset": dataset,
        "X_base": X_reference,
        "feature_names": transformed_feature_names,
        "coordinate_metadata": coordinate_metadata,
        "coordinate_metadata_by_fold": coordinate_rows,
        "prediction_reproduction": reproduction,
        "perturbation_geometry": sorted(geometries),
        "folds": folds,
        "execution": execution,
    }


def _mpma_e_reference_and_folds(
    sweep: Sweep,
    rankings: pd.DataFrame,
) -> tuple[Any, np.ndarray, list[str], list[dict[str, Any]], pd.Series, list[Any]]:
    root = sweep.root()
    members, aggregation = _selected_ensemble_members(root)
    member_rows = _ordered_member_rows(root, rankings, members)
    all_levels: list[str] = []
    for _, r in member_rows.iterrows():
        for lv in _row_levels(r):
            if lv not in all_levels:
                all_levels.append(lv)
    if not all_levels:
        all_levels = ["all"]
    dataset = load_dataset(sweep.data, tuple(all_levels))
    splits = _explainability_outer_splits(sweep, dataset)
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)

    feature_names: list[str] = []
    reference_blocks: list[np.ndarray] = []
    reference_coordinate_metadata: list[Any] = []
    member_materialized: list[dict[str, Any]] = []
    for member_i, (_, r) in enumerate(member_rows.iterrows(), start=1):
        levels = _row_levels(r) or ("all",)
        X_base_member, names_member, blocks_member = materialize_mpdr_with_blocks(
            dataset, levels
        )
        transformation_key = str(r["count_transformation"])
        learner_key = str(r["learner"])
        label_prefix = (
            f"{str(r.get('resolution', '+'.join(levels)))}"
            f"|{transformation_key}"
            f"|{learner_key}"
            f"|{str(r.get('config_id', member_i))}"
        )
        ct_ref = _configured_count_transformation_factory(
            sweep, transformation_key, blocks_member
        )()
        X_ref_member, _ = ct_ref.apply_pair(X_base_member, X_base_member)
        transformed_names = ct_ref.get_feature_names_out(list(names_member))
        transformed_metadata = ct_ref.coordinate_metadata(list(names_member))
        if X_ref_member.shape[1] != len(transformed_names) or len(
            transformed_metadata
        ) != len(transformed_names):
            raise ExplainabilityConfigurationError(
                f"Transformation {transformation_key!r} produced feature metadata inconsistent with its transformed matrix."
            )
        prefixed_names = [f"{label_prefix}|{name}" for name in transformed_names]
        feature_names.extend(prefixed_names)
        reference_coordinate_metadata.extend(
            replace(item, name=prefixed_name)
            for item, prefixed_name in zip(transformed_metadata, prefixed_names)
        )
        reference_blocks.append(X_ref_member)
        member_materialized.append(
            {
                "row": r,
                "levels": levels,
                "X_base": X_base_member,
                "feature_blocks": blocks_member,
                "n_features": X_base_member.shape[1],
                "transformation_key": transformation_key,
                "learner_key": learner_key,
            }
        )

    X_reference = (
        np.concatenate(reference_blocks, axis=1)
        if len(reference_blocks) > 1
        else reference_blocks[0]
    )
    folds: list[dict[str, Any]] = []
    total_members = len(member_materialized)
    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
            continue
        X_train_blocks: list[np.ndarray] = []
        X_test_blocks: list[np.ndarray] = []
        fitted_members: list[dict[str, Any]] = []
        start = 0
        for member_i, spec in enumerate(member_materialized, start=1):
            r = spec["row"]
            info(
                "Fitting MPMA-E OOF member "
                f"{member_i}/{total_members} · split={str(split['split_key'])} · "
                f"config={str(r.get('config_id', ''))} · "
                f"{str(r.get('resolution', r.get('levels', 'MPDR')))} · "
                f"{spec['transformation_key']} · {spec['learner_key']}"
            )
            ct = _configured_count_transformation_factory(
                sweep,
                spec["transformation_key"],
                spec["feature_blocks"],
                resolution_feature_blocks=spec["feature_blocks"],
            )()
            X_train_member, X_test_member = ct.apply_pair(
                spec["X_base"][train_idx], spec["X_base"][test_idx]
            )
            clf_member = _configured_learner_factory(sweep, spec["learner_key"])()
            fit_classifier(
                clf_member,
                X_train_member,
                dataset.y[train_idx],
                None if groups is None else groups[train_idx],
            )
            stop = start + X_train_member.shape[1]
            X_train_blocks.append(X_train_member)
            X_test_blocks.append(X_test_member)
            fitted_members.append(
                {
                    "config_id": str(r["config_id"]),
                    "slice": slice(start, stop),
                    "estimator": clf_member,
                    "classes": np.arange(len(dataset.class_labels), dtype=int),
                }
            )
            start = stop
        X_train = np.concatenate(X_train_blocks, axis=1)
        X_test = np.concatenate(X_test_blocks, axis=1)
        ensemble = _FittedMpmaEnsemble(fitted_members, aggregation=aggregation)
        proba = _predict_proba_aligned(
            ensemble, X_test, np.arange(len(dataset.class_labels), dtype=int)
        )
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": X_train,
                "X_test": X_test,
                "y_test": dataset.y[test_idx],
                "estimator": ensemble,
                "proba": proba,
            }
        )
    if not folds:
        raise ExplainabilityConfigurationError(
            "No outer fold could be fitted for MPMA-E out-of-fold explainability."
        )
    row = pd.Series(
        {
            "unit": "MPMA-E",
            "members": ",".join(member_rows["config_id"].astype(str).tolist()),
            "aggregation_strategy": aggregation,
            "levels": ",".join(all_levels),
            "count_transformation": "ensemble",
            "learner": "MPMA-E",
        }
    )
    return (
        dataset,
        X_reference,
        feature_names,
        folds,
        row,
        reference_coordinate_metadata,
    )


def _aggregate_oof_predictions(
    dataset: Any, folds: Sequence[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        proba = np.asarray(fold.get("proba"), dtype=float)
        pred = (
            np.argmax(proba, axis=1)
            if proba.ndim == 2 and proba.shape[0] == len(test_idx)
            else np.full(len(test_idx), -1)
        )
        for local_i, sample_index in enumerate(test_idx):
            rec = {
                "split_key": str(fold.get("split_key", "")),
                "sample_id": str(dataset.sample_ids[int(sample_index)]),
                "sample_index": int(sample_index),
                "y_true": int(dataset.y[int(sample_index)]),
                "y_pred": int(pred[local_i]),
            }
            if proba.ndim == 2 and local_i < proba.shape[0]:
                for j, label in enumerate(dataset.class_labels):
                    if j < proba.shape[1]:
                        rec[f"proba_{label}"] = float(proba[local_i, j])
                if proba.shape[1] > 1:
                    rec["y_proba_pos"] = float(proba[local_i, 1])
            rows.append(rec)
    return pd.DataFrame(rows)


def _write_oof_prediction_summary(
    dataset: Any, folds: Sequence[dict[str, Any]], target_dir: Path
) -> Path:
    pred_df = _aggregate_oof_predictions(dataset, folds)
    path = target_dir / "oof_predictions.parquet"
    write_table(path, pred_df)
    return path


def _select_local_sample_pairs(
    dataset: Any,
    folds: Sequence[dict[str, Any]],
    *,
    sample_ids: Sequence[str],
    representative: bool,
) -> list[tuple[int, str]]:
    sid_to_idx = {str(sid): i for i, sid in enumerate(dataset.sample_ids)}
    selected: list[tuple[int, str]] = []
    for sid in sample_ids:
        if str(sid) not in sid_to_idx:
            raise ExplainabilityConfigurationError(
                f"Requested instance sample_id {sid!r} was not found in the loaded dataset."
            )
        selected.append((int(sid_to_idx[str(sid)]), f"requested:{sid}"))
    if representative:
        records: list[dict[str, Any]] = []
        for fold in folds:
            test_idx = np.asarray(fold["test_idx"], dtype=int)
            proba = np.asarray(fold.get("proba"), dtype=float)
            if proba.ndim != 2 or proba.shape[0] != len(test_idx):
                continue
            for local_i, global_i in enumerate(test_idx):
                cls = int(dataset.y[int(global_i)])
                if 0 <= cls < proba.shape[1]:
                    records.append(
                        {
                            "sample_index": int(global_i),
                            "true_class": cls,
                            "p_own_class": float(proba[int(local_i), cls]),
                        }
                    )
        pred = pd.DataFrame(records)
        if not pred.empty:
            pred = pred.groupby(["sample_index", "true_class"], as_index=False).agg(
                p_own_class=("p_own_class", "mean")
            )
            for cls in sorted(set(int(x) for x in dataset.y.tolist())):
                sub = pred[pred["true_class"].eq(cls)].copy()
                if sub.empty:
                    continue
                centre = float(
                    pd.to_numeric(sub["p_own_class"], errors="coerce").median()
                )
                sub["_dist"] = (
                    pd.to_numeric(sub["p_own_class"], errors="coerce") - centre
                ).abs()
                row = sub.sort_values(["_dist", "sample_index"]).iloc[0]
                selected.append(
                    (int(row["sample_index"]), f"representative_class_{cls}")
                )
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for idx, role in selected:
        if idx not in seen:
            seen.add(idx)
            out.append((idx, role))
    return out


def _local_explanations_signature(
    source_signature: str,
    explainability: Any,
    specs_by_name: dict[str, Any],
    methods: Sequence[str],
    selected_pairs: Sequence[tuple[int, str]],
) -> str:
    local_methods = [
        name
        for name in ("shap", "lime")
        if name in methods and method_has_local(specs_by_name[name])
    ]
    payload = {
        "source_signature": str(source_signature),
        "methods": {
            name: method_to_dict(specs_by_name[name]) for name in local_methods
        },
        "random_state": int(explainability.random_state),
        "mode": _effective_local_explanations_mode(explainability),
        "selected_samples": [[int(i), str(role)] for i, role in selected_pairs],
        "local_top_k": int(explainability.local.stored_features),
        "top_instance_features": int(explainability.local.displayed_features),
    }
    return _signature_hash(payload)


def _compute_local_method_rows(
    method: str,
    spec: Any,
    folds: Sequence[dict[str, Any]],
    dataset: Any,
    feature_names: Sequence[str],
    random_state: int,
    threads_per_worker: int,
) -> list[dict[str, Any]]:
    class_indices = tuple(range(len(dataset.class_labels)))
    rows: list[dict[str, Any]] = []
    for fold_no, fold in enumerate(folds, start=1):
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        if len(test_idx) == 0:
            continue
        fold_feature_names = _fold_feature_names(fold, feature_names)
        estimator = configure_estimator_threads(
            fold["estimator"], int(threads_per_worker)
        )
        force_rows = tuple(range(len(test_idx)))
        if method == "shap":
            values, rows_ex, _ = _shap_values_for_data(
                estimator,
                fold["X_train"],
                fold["X_test"],
                fold_feature_names,
                dataset.class_labels,
                random_state=int(random_state) + fold_no * 997,
                spec=spec,
                force_explain_rows=force_rows,
                show_progress=False,
                input_projector=fold.get("input_projector"),
            )
        elif method == "lime":
            values, rows_ex = _lime_values_for_data(
                estimator,
                fold["X_train"],
                fold["X_test"],
                fold_feature_names,
                dataset.class_labels,
                class_indices,
                random_state=int(random_state) + fold_no * 997,
                spec=spec,
                force_explain_rows=force_rows,
                progress_callback=None,
                input_projector=fold.get("input_projector"),
            )
        else:
            continue
        rows.extend(
            _local_value_rows_for_fold(
                fold,
                values,
                rows_ex,
                class_indices,
                dataset.class_labels,
                dataset.sample_ids,
                dataset.y,
                positive_class=dataset.positive_class,
            )
        )
    return rows


def _compact_local_explanations(
    method: str,
    dataset: Any,
    feature_names: Sequence[str],
    rows: Sequence[dict[str, Any]],
    selected_pairs: Sequence[tuple[int, str]],
    local_top_k: int,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    selected = {int(i): str(role) for i, role in selected_pairs}
    records: list[dict[str, Any]] = []
    k = max(1, int(local_top_k))
    for row in rows:
        sample_index = int(row["sample_index"])
        true_class = int(row["true_class"])
        if int(row["class_index"]) != true_class:
            continue
        values = np.asarray(row["values"], dtype=float).reshape(-1)
        row_feature_names = [str(x) for x in row.get("feature_names", feature_names)]
        feature_values = np.asarray(
            row.get("feature_values", np.full(len(values), np.nan)), dtype=float
        ).reshape(-1)
        if len(values) != len(row_feature_names):
            raise ExplainabilityConfigurationError(
                f"Local {method.upper()} attribution width {len(values)} does not match {len(row_feature_names)} feature names."
            )
        is_selected = sample_index in selected
        if is_selected:
            chosen = np.argsort(-np.abs(values))
        else:
            chosen = np.argsort(-np.abs(values))[: min(k, len(values))]
        predicted_class = int(row.get("predicted_class", -1))
        predicted_label = (
            str(dataset.class_labels[predicted_class])
            if 0 <= predicted_class < len(dataset.class_labels)
            else ""
        )
        for rank, feature_index in enumerate(chosen, start=1):
            j = int(feature_index)
            value = float(values[j])
            records.append(
                {
                    "method": str(method),
                    "sample_id": str(row["sample_id"]),
                    "sample_index": sample_index,
                    "split_key": str(row.get("split_key", "")),
                    "selection_role": selected.get(sample_index, ""),
                    "selected_for_report": bool(is_selected),
                    "true_class": true_class,
                    "true_class_label": str(dataset.class_labels[true_class]),
                    "predicted_class": predicted_class,
                    "predicted_class_label": predicted_label,
                    "class_index": true_class,
                    "class_label": str(dataset.class_labels[true_class]),
                    "prediction": float(row.get("p_class", np.nan)),
                    "feature": str(row_feature_names[j]),
                    "feature_value": float(feature_values[j])
                    if j < len(feature_values)
                    else np.nan,
                    "attribution": value,
                    "abs_attribution": abs(value),
                    "local_rank": int(rank),
                    "retained_scope": "representative_full" if is_selected else "top_k",
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["method", "sample_index", "split_key", "local_rank"],
        kind="stable",
    )


def _ensure_local_explanation_outputs(
    target_dir: Path,
    source_signature: str,
    sweep: Any,
    specs_by_name: dict[str, Any],
    methods: Sequence[str],
    folds: Sequence[dict[str, Any]],
    dataset: Any,
    feature_names: Sequence[str],
    threads_per_worker: int,
    precomputed_rows: dict[str, Sequence[dict[str, Any]]] | None = None,
) -> dict[str, Path]:
    mode = _effective_local_explanations_mode(sweep.explainability)
    if mode == "none":
        return {}
    include_representative = mode in {"representative", "representative_and_requested"}
    include_requested = mode in {"requested", "representative_and_requested"}
    selected_pairs = _select_local_sample_pairs(
        dataset,
        folds,
        sample_ids=sweep.explainability.local.sample_ids if include_requested else (),
        representative=include_representative,
    )
    local_methods = [
        name
        for name in ("shap", "lime")
        if name in methods and method_has_local(specs_by_name[name])
    ]
    if not local_methods:
        return {}
    signature = _local_explanations_signature(
        source_signature, sweep.explainability, specs_by_name, methods, selected_pairs
    )
    local_path = target_dir / "local_explanations.parquet"
    cache_path = target_dir / "local_explanations_cache.json"
    try:
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    except Exception:
        cache = {}
    if str(cache.get("signature", "")) == signature and table_exists(local_path):
        local_table = read_table(local_path)
        info("Reusing cached sample-wise OOF explanations")
    else:
        frames: list[pd.DataFrame] = []
        supplied = precomputed_rows or {}
        for method in local_methods:
            rows = list(supplied.get(method, ()))
            if not rows:
                info(
                    f"Computing sample-wise OOF {method.upper()} explanations · all held-out samples"
                )
                rows = _compute_local_method_rows(
                    method,
                    specs_by_name[method],
                    folds,
                    dataset,
                    feature_names,
                    sweep.explainability.random_state,
                    threads_per_worker,
                )
            frame = _compact_local_explanations(
                method,
                dataset,
                feature_names,
                rows,
                selected_pairs,
                int(sweep.explainability.local.stored_features),
            )
            if not frame.empty:
                frames.append(frame)
        local_table = (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )
        write_table(local_path, local_table)
        dump_json_standard(
            {
                "signature": signature,
                "mode": mode,
                "selected_samples": [
                    {"sample_id": str(dataset.sample_ids[int(i)]), "role": str(role)}
                    for i, role in selected_pairs
                ],
            },
            cache_path,
        )
    outputs: dict[str, Path] = {"local_explanations": local_path}
    context_table = pd.DataFrame()
    if not local_table.empty:
        context_table = build_local_relative_abundance_context(
            dataset, local_table["feature"].astype(str).tolist()
        )
        if not context_table.empty:
            context_path = target_dir / "local_cohort_context.parquet"
            write_table(context_path, context_table)
            outputs["local_cohort_context"] = context_path
    if not local_table.empty and selected_pairs:
        figure_path = plot_local_attributions(
            local_table,
            target_dir / "figures" / "local_explanations",
            task="classification",
            top_n=int(sweep.explainability.local.displayed_features),
            cohort_context=context_table,
        )
        if figure_path is not None:
            outputs["local_explanations_figure"] = figure_path
    return outputs


def _aggregate_oof_interactions(
    tables_by_fold: Sequence[pd.DataFrame], method: str
) -> pd.DataFrame:
    if not tables_by_fold:
        return pd.DataFrame()
    raw = pd.concat(tables_by_fold, ignore_index=True)
    raw["interaction_strength"] = pd.to_numeric(
        raw["interaction_strength"], errors="coerce"
    )
    keys = ["class_index", "class_label", "feature_1", "feature_2"]
    out = raw.groupby(keys, as_index=False).agg(
        interaction_strength=("interaction_strength", "mean"),
        interaction_strength_sd=("interaction_strength", "std"),
        n_outer_folds=("fold_key", "nunique"),
        n_evaluations=("interaction_strength", "count"),
        n_ale_cells=("n_ale_cells", "mean"),
        correction_applied=("correction_applied", "max"),
        correction_error=(
            "correction_error",
            lambda x: "; ".join([str(v) for v in x if str(v) and str(v) != "nan"])[
                :240
            ],
        ),
        perturbation_projection=(
            "perturbation_projection",
            lambda x: (
                "fitted_model_input_geometry"
                if any(str(v) == "fitted_model_input_geometry" for v in x)
                else "none"
            ),
        ),
    )
    out["method"] = method
    return out.sort_values(
        ["class_index", "interaction_strength"],
        ascending=[True, False],
        na_position="last",
    )


def _shap_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    sample_ids: Sequence[Any],
    y: np.ndarray,
    positive_class: int | None,
    collect_local: bool,
    random_state: int,
    spec: SHAP,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame, list[dict[str, Any]], int, str]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    feature_names = _fold_feature_names(fold, feature_names)
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    force_rows = list(range(len(test_idx))) if collect_local else []
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    vals, rows_ex, backend = _shap_values_for_data(
        estimator,
        fold["X_train"],
        fold["X_test"],
        feature_names,
        class_labels,
        random_state=int(random_state),
        spec=spec,
        force_explain_rows=force_rows,
        show_progress=False,
        progress_callback=callback,
        input_projector=fold.get("input_projector"),
    )
    fold_frame = _value_frame_for_fold(
        "shap",
        vals,
        feature_names,
        class_indices,
        class_labels,
        "mean_abs_probability_shap_within_outer_fold",
        positive_class=positive_class,
    )
    fold_frame["fold_key"] = str(fold.get("split_key", ""))
    fold_frame["fold_no"] = int(fold_no)
    fold_frame["shap_backend"] = str(backend)
    fold_frame["perturbation_projection"] = (
        "fitted_model_input_geometry"
        if fold.get("input_projector") is not None
        else "none"
    )
    rows = (
        _local_value_rows_for_fold(
            fold,
            vals,
            rows_ex,
            tuple(range(len(class_labels))),
            class_labels,
            sample_ids,
            y,
            positive_class=positive_class,
        )
        if collect_local
        else []
    )
    return int(fold_no), fold_frame, rows, int(len(rows_ex)), str(backend)


def _local_value_rows_for_fold(
    fold: dict[str, Any],
    values: np.ndarray,
    rows_ex: Sequence[int],
    class_indices: Sequence[int],
    class_labels: Sequence[str],
    sample_ids: Sequence[Any],
    y: np.ndarray,
    *,
    positive_class: int | None = None,
) -> list[dict[str, Any]]:
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    X_test = np.asarray(fold["X_test"], dtype=float)
    proba = np.asarray(fold.get("proba"), dtype=float)
    pred = (
        np.argmax(proba, axis=1)
        if proba.ndim == 2 and proba.shape[0] == len(test_idx)
        else np.full(len(test_idx), -1)
    )
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    rows: list[dict[str, Any]] = []
    for local_value_row, local_test_row in enumerate(rows_ex):
        global_i = int(test_idx[int(local_test_row)])
        for class_pos, class_index in enumerate(class_indices):
            if arr.shape[2] == len(class_labels):
                local_values = arr[local_value_row, :, int(class_index)]
            elif arr.shape[2] == len(class_indices):
                local_values = arr[local_value_row, :, class_pos]
            elif arr.shape[2] == 1 and len(class_labels) == 2:
                positive = 1 if positive_class is None else int(positive_class)
                local_values = arr[local_value_row, :, 0]
                if int(class_index) != positive:
                    local_values = -local_values
            elif arr.shape[2] == 1 and len(class_indices) == 1:
                local_values = arr[local_value_row, :, 0]
            else:
                raise ExplainabilityConfigurationError(
                    f"Local attribution output shape {arr.shape} does not match configured classes."
                )
            p_class = (
                float(proba[int(local_test_row), int(class_index)])
                if proba.ndim == 2 and int(class_index) < proba.shape[1]
                else float("nan")
            )
            rows.append(
                {
                    "split_key": str(fold.get("split_key", "")),
                    "sample_id": str(sample_ids[global_i]),
                    "sample_index": global_i,
                    "true_class": int(y[global_i]),
                    "predicted_class": int(pred[int(local_test_row)])
                    if int(local_test_row) < len(pred)
                    else -1,
                    "class_index": int(class_index),
                    "class_label": str(class_labels[int(class_index)]),
                    "p_class": p_class,
                    "feature_names": _fold_feature_names(fold, ()),
                    "feature_values": X_test[int(local_test_row)].copy(),
                    "values": np.asarray(local_values, dtype=float).copy(),
                }
            )
    return rows


def _lime_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    sample_ids: Sequence[Any],
    y: np.ndarray,
    positive_class: int | None,
    collect_local: bool,
    random_state: int,
    spec: LIME,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame, list[dict[str, Any]]]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    feature_names = _fold_feature_names(fold, feature_names)
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    force_rows = list(range(len(test_idx))) if collect_local else []
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    coeffs, rows_ex = _lime_values_for_data(
        estimator,
        fold["X_train"],
        fold["X_test"],
        feature_names,
        class_labels,
        tuple(range(len(class_labels))) if collect_local else class_indices,
        random_state=int(random_state),
        spec=spec,
        force_explain_rows=force_rows,
        progress_callback=callback,
        input_projector=fold.get("input_projector"),
    )
    frame = _value_frame_for_fold(
        "lime",
        coeffs,
        feature_names,
        class_indices,
        class_labels,
        "mean_abs_lime_coefficient_within_outer_fold",
    )
    frame["fold_key"] = str(fold.get("split_key", ""))
    frame["fold_no"] = int(fold_no)
    frame["perturbation_projection"] = (
        "fitted_model_input_geometry"
        if fold.get("input_projector") is not None
        else "none"
    )
    rows = (
        _local_value_rows_for_fold(
            fold,
            coeffs,
            rows_ex,
            tuple(range(len(class_labels))) if collect_local else class_indices,
            class_labels,
            sample_ids,
            y,
            positive_class=positive_class,
        )
        if collect_local
        else []
    )
    return int(fold_no), frame, rows


def _permutation_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    random_state: int,
    spec: Permutation,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    feature_names = _fold_feature_names(fold, feature_names)
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    frame = _permutation_feature_importance(
        estimator,
        fold["X_test"],
        fold["y_test"],
        feature_names,
        class_labels,
        class_indices,
        spec=spec,
        random_state=int(random_state),
        progress_callback=callback,
        input_projector=fold.get("input_projector"),
    )
    frame["fold_key"] = str(fold.get("split_key", ""))
    frame["fold_no"] = int(fold_no)
    return int(fold_no), frame


def _ale_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    spec: ALE,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    feature_names = _fold_feature_names(fold, feature_names)
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    frame = _ale_feature_importance(
        estimator,
        fold["X_test"],
        feature_names,
        class_labels,
        class_indices,
        spec=spec,
        top_features=None,
        progress_callback=callback,
        input_projector=fold.get("input_projector"),
    )
    skipped = frame.attrs.get("skipped_features")
    skipped_out = None
    if isinstance(skipped, pd.DataFrame) and not skipped.empty:
        skipped_out = skipped.copy()
        skipped_out["fold_key"] = str(fold.get("split_key", ""))
        skipped_out["fold_no"] = int(fold_no)
    if frame.empty:
        return int(fold_no), frame, None, skipped_out
    curves = frame.attrs.get("curves")
    frame = frame.copy()
    frame["fold_key"] = str(fold.get("split_key", ""))
    frame["fold_no"] = int(fold_no)
    curves_out = None
    if isinstance(curves, pd.DataFrame) and not curves.empty:
        curves_out = curves.copy()
        curves_out["fold_key"] = str(fold.get("split_key", ""))
        curves_out["fold_no"] = int(fold_no)
    return int(fold_no), frame, curves_out, skipped_out


def _explain_one(
    sweep: Sweep,
    target_override: str | None = None,
    *,
    output_slug_override: str | None = None,
    display_label_override: str | None = None,
    allow_member_cache: bool = True,
) -> dict[str, Path]:
    method_specs = _normalise_explainability_method_specs(sweep.explainability.methods)
    methods = tuple(method_name(x) for x in method_specs)
    global_methods = tuple(method_name(x) for x in method_specs if method_has_global(x))
    local_methods = tuple(method_name(x) for x in method_specs if method_has_local(x))
    specs_by_name = {method_name(x): x for x in method_specs}
    _preflight_explainability_dependencies(methods)
    root = sweep.root()
    stage("Explainability", str(root))
    summary_table(
        "Explainability suite",
        {
            "target": target_override
            or _configured_explainability_targets(sweep.explainability),
            "methods": methods,
            "profile": str(getattr(sweep.explainability, "profile", "standard")),
            "top features": sweep.explainability.top_k,
            "local explanations": _effective_local_explanations_mode(
                sweep.explainability
            ),
            "interaction pairs": int(specs_by_name["interactions"].top_k)
            if "interactions" in methods
            else "not requested",
            "fallbacks": "off",
        },
    )
    rankings_path = root / "tables" / "mpma_inner_rankings.parquet"
    if not table_exists(rankings_path):
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = read_table(rankings_path)
    if rankings.empty:
        raise RuntimeError("No ranked MPMA is available for explainability.")
    target = target_override or _single_configured_explainability_target(
        sweep.explainability
    )
    ensemble_explain = False
    if target in {"best", "best_individual", "best_mpma", "mpma_b", "MPMA-B"}:
        row = rankings.iloc[0]
        target_label = "MPMA-B"
        target_slug = "mpma_b"
    elif target in {"ensemble", "mpma_e", "MPMA-E"}:
        ensemble_explain = True
        row = pd.Series({"unit": "MPMA-E"})
        target_label = "MPMA-E"
        target_slug = "mpma_e"
    elif target in {"baseline_rf", "Baseline RF", "baseline-rf"}:
        row = _resolve_baseline_rf_row(rankings)
        if row is None:
            raise ExplainabilityConfigurationError(
                "Baseline RF explainability was requested, but no completed MPMA matches "
                "RF_1000_msl5 + arcsin_sqrt + deepest available single rank."
            )
        target_label = "Baseline RF"
        target_slug = "baseline_rf"
    else:
        matches = rankings[rankings["config_id"].astype(str).eq(str(target))]
        if matches.empty:
            raise ValueError(f"No MPMA with config_id={target!r} in rankings.")
        row = matches.iloc[0]
        target_label = "MPMA"
        target_slug = str(target)

    if output_slug_override:
        target_slug = output_slug_override.strip("/")
    if display_label_override:
        target_label = display_label_override

    if ensemble_explain and allow_member_cache:
        members_for_cache, _ = _selected_ensemble_members(root)
        _ensure_mpma_member_explanations(sweep, rankings, members_for_cache)

    config_id_for_cache = None
    if (
        not ensemble_explain
        and "config_id" in row.index
        and not pd.isna(row.get("config_id"))
    ):
        config_id_for_cache = str(row.get("config_id"))
    target_dir = root / "explainability" / target_slug
    cache_dir = (
        root / "explainability" / "cache" / _safe_cache_name(config_id_for_cache)
        if config_id_for_cache
        else None
    )
    if (
        allow_member_cache
        and target_dir.exists()
        and _explainability_cache_complete(
            target_dir, sweep.explainability, config_id_for_cache
        )
    ):
        info(f"Reusing existing explainability for {target_label}")
        return _existing_explainability_outputs(target_dir)
    if (
        not ensemble_explain
        and allow_member_cache
        and cache_dir is not None
        and cache_dir.exists()
        and _explainability_cache_complete(
            cache_dir, sweep.explainability, config_id_for_cache
        )
    ):
        if target_dir != cache_dir:
            _copy_explainability_cache(cache_dir, target_dir)
        info(
            f"Reusing cached explainability for {target_label} · config={config_id_for_cache}"
        )
        return _existing_explainability_outputs(target_dir)

    coordinate_metadata: list[Any] = []
    coordinate_metadata_by_fold: list[dict[str, Any]] = []
    prediction_reproduction: list[dict[str, Any]] = []
    if ensemble_explain:
        info("Preparing selected MPMA-E outer-fold units for OOF explanation")
        (
            dataset,
            X_base,
            feature_names,
            oof_folds,
            row,
            coordinate_metadata,
        ) = _mpma_e_reference_and_folds(sweep, rankings)
    else:
        info("Preparing selected MPMA outer-fold units for OOF explanation")
        oof_bundle = _fit_oof_single_for_explainability(sweep, row)
        dataset = oof_bundle["dataset"]
        X_base = oof_bundle["X_base"]
        feature_names = oof_bundle["feature_names"]
        coordinate_metadata = list(oof_bundle.get("coordinate_metadata", []))
        coordinate_metadata_by_fold = list(
            oof_bundle.get("coordinate_metadata_by_fold", [])
        )
        prediction_reproduction = list(oof_bundle.get("prediction_reproduction", []))
        oof_folds = oof_bundle["folds"]
    for fold in oof_folds:
        if "feature_names" not in fold:
            fold["feature_names"] = list(feature_names)

    geometries = sorted(
        {
            str(value)
            for fold in oof_folds
            for value in (
                fold.get("perturbation_geometry", ())
                if isinstance(fold.get("perturbation_geometry", ()), (list, tuple, set))
                else (fold.get("perturbation_geometry", "unverified"),)
            )
        }
    )
    projection_applied = any(
        fold.get("input_projector") is not None for fold in oof_folds
    )
    perturbation_policy = {
        "geometry": geometries,
        "projection_applied": bool(projection_applied),
        "projection_scope": "generated perturbations are projected onto the fitted model-input geometry before prediction when a supported constraint is known",
        "methods": [
            name
            for name in global_methods
            if name in {"shap", "lime", "ale", "permutation", "interactions"}
        ],
        "interpretation": "predictive model-coordinate attribution under geometry-preserving perturbations; not a causal or isolated biological effect",
        "tree_shap_policy": "disabled for constrained projected inputs because TreeSHAP cannot apply the projection operator to masked samples",
    }
    perturbation_policy_path = target_dir / "perturbation_policy.json"
    dump_json_standard(perturbation_policy, perturbation_policy_path)

    class_indices = _resolve_explainability_classes(
        dataset, sweep.explainability.classes
    )
    execution = _xai_execution_plan(sweep, len(oof_folds))
    memory_gib = (
        "unknown"
        if execution.memory_bytes is None
        else f"{execution.memory_bytes / (1024**3):.1f} GiB"
    )
    summary_table(
        "Explainability workload",
        {
            "outer folds": len(oof_folds),
            "features": len(feature_names),
            "classes explained": len(class_indices),
            "class labels": [str(dataset.class_labels[int(i)]) for i in class_indices],
            "CPU logical/physical": f"{execution.logical_cpus}/{execution.physical_cpus}",
            "available memory": memory_gib,
            "workers": execution.workers,
            "threads per worker": execution.threads_per_worker,
            "parallel backend": execution.backend,
        },
    )
    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    coordinate_metadata_path: Path | None = None
    coordinate_metadata_by_fold_path: Path | None = None
    prediction_reproduction_path: Path | None = None
    if coordinate_metadata:
        coordinate_frame = pd.DataFrame(
            [
                {
                    "coordinate": str(item.name),
                    "coordinate_type": str(item.coordinate_type),
                    "anchor_feature": ""
                    if item.anchor_feature is None
                    else str(item.anchor_feature),
                    "exact_feature_identity": bool(item.exact_feature_identity),
                    "components": json.dumps(
                        list(item.components), separators=(",", ":")
                    ),
                    "coefficients": json.dumps(
                        [float(x) for x in item.coefficients], separators=(",", ":")
                    ),
                }
                for item in coordinate_metadata
            ]
        )
        coordinate_metadata_path = target_dir / "coordinate_metadata.parquet"
        write_table(coordinate_metadata_path, coordinate_frame)
    if coordinate_metadata_by_fold:
        coordinate_metadata_by_fold_path = (
            target_dir / "coordinate_metadata_by_outer_fold.parquet"
        )
        write_table(
            coordinate_metadata_by_fold_path,
            pd.DataFrame(coordinate_metadata_by_fold),
        )
    if prediction_reproduction:
        prediction_reproduction_path = target_dir / "prediction_reproduction.parquet"
        write_table(prediction_reproduction_path, pd.DataFrame(prediction_reproduction))

    source_signature = _explainability_source_signature(
        target_slug, row, oof_folds, feature_names, dataset.class_labels
    )
    previous_meta: dict[str, Any] = {}
    meta_path = target_dir / "explained_unit.json"
    if meta_path.exists():
        try:
            previous_meta = json.loads(meta_path.read_text())
        except Exception:
            previous_meta = {}
    previous_method_cache = dict(previous_meta.get("method_cache", {}))
    current_method_signatures = {
        name: _method_cache_signature(
            sweep.explainability, specs_by_name[name], source_signature, class_indices
        )
        for name in global_methods
    }
    current_method_entries = {
        name: _method_cache_entry(
            current_method_signatures[name],
            source_signature,
            specs_by_name[name],
            class_indices,
            sweep.explainability.top_k,
        )
        for name in current_method_signatures
    }
    reusable_methods: set[str] = set()
    for name, signature in current_method_signatures.items():
        if not _method_cache_files_complete(target_dir, name):
            info(
                f"Cache miss {name.upper()} · required method artifacts are incomplete"
            )
            continue
        entry, provenance_source = _load_method_cache_entry(
            target_dir, name, previous_method_cache
        )
        compatible, reason = _method_cache_entry_status(
            entry, signature, source_signature, specs_by_name[name], class_indices
        )
        legacy = not compatible and _legacy_method_cache_valid(
            previous_meta, row, sweep.explainability, specs_by_name[name], class_indices
        )
        if compatible or legacy:
            reusable_methods.add(name)
            _persist_method_cache_entry(
                meta_path, previous_meta, name, current_method_entries[name]
            )
            previous_method_cache[name] = current_method_entries[name]
            suffix = (
                "migrated from legacy metadata"
                if legacy and not compatible
                else provenance_source
            )
            info(
                f"Cache hit {name.upper()} · reusing completed global explainability · {suffix}"
            )
        else:
            info(f"Cache miss {name.upper()} · {reason}")

    method_frames: list[pd.DataFrame] = []
    fold_importance_frames: list[pd.DataFrame] = []
    method_outputs: dict[str, Path] = {}
    interaction_outputs: dict[str, Path] = {}
    resolved_shap_backends: set[str] = set()
    for name in global_methods:
        if name not in reusable_methods:
            continue
        if name == "interactions":
            interaction_outputs.update(_existing_method_outputs(target_dir, name))
            continue
        _, fold_long = _cached_method_frame(target_dir, name)
        fold_frames = (
            [frame.copy() for _, frame in fold_long.groupby("fold_no", sort=True)]
            if "fold_no" in fold_long.columns
            else [fold_long.copy()]
        )
        scoring_values = (
            fold_long.get("scoring", pd.Series(dtype=object)).dropna().astype(str)
        )
        scoring = scoring_values.iloc[0] if not scoring_values.empty else name
        frame = _aggregate_fold_feature_importance(
            name,
            fold_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            scoring,
            sweep.explainability.top_k,
        )
        method_frames.append(frame)
        fold_importance_frames.append(fold_long)
        if name == "shap" and "shap_backend" in fold_long.columns:
            resolved_shap_backends.update(
                str(x) for x in fold_long["shap_backend"].dropna().astype(str).unique()
            )
        method_outputs.update(
            _write_method_outputs(
                frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        method_outputs.update(_existing_method_outputs(target_dir, name))
    oof_pred_path = _write_oof_prediction_summary(dataset, oof_folds, target_dir)
    method_outputs["oof_predictions"] = oof_pred_path
    local_mode = _effective_local_explanations_mode(sweep.explainability)
    local_enabled = local_mode != "none" and bool(local_methods)

    shap_oof_rows: list[dict[str, Any]] = []
    lime_oof_rows: list[dict[str, Any]] = []
    shap_local_enabled = local_enabled and "shap" in local_methods
    if "shap" in global_methods and "shap" not in reusable_methods:
        spec = specs_by_name["shap"]
        info("Running global class-specific OOF SHAP feature attribution")
        shap_fold_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            with progress() as prog:
                for fold_no, fold in enumerate(oof_folds, start=1):
                    test_idx = np.asarray(fold["test_idx"], dtype=int)
                    fold_feature_names = _fold_feature_names(fold, feature_names)
                    expected_rows = max(
                        1,
                        len(test_idx)
                        if shap_local_enabled
                        else min(int(spec.max_explain), len(test_idx)),
                    )
                    prefix = (
                        f"SHAP fold {fold_no}/{len(oof_folds)} · "
                        f"{expected_rows} samples · {len(fold_feature_names)} features · {len(class_indices)} classes"
                    )
                    task = prog.add_task(f"{prefix} · preparing", total=expected_rows)
                    vals, rows_ex, backend = _shap_values_for_data(
                        configure_estimator_threads(
                            fold["estimator"], int(execution.threads_per_worker)
                        ),
                        fold["X_train"],
                        fold["X_test"],
                        fold_feature_names,
                        dataset.class_labels,
                        random_state=sweep.explainability.random_state + fold_no * 997,
                        spec=spec,
                        force_explain_rows=list(range(len(test_idx)))
                        if shap_local_enabled
                        else [],
                        show_progress=False,
                        progress_callback=_progress_callback(prog, task, prefix),
                        input_projector=fold.get("input_projector"),
                    )
                    fold_frame = _value_frame_for_fold(
                        "shap",
                        vals,
                        fold_feature_names,
                        class_indices,
                        dataset.class_labels,
                        "mean_abs_probability_shap_within_outer_fold",
                        positive_class=dataset.positive_class,
                    )
                    fold_frame["fold_key"] = str(fold.get("split_key", ""))
                    fold_frame["fold_no"] = int(fold_no)
                    fold_frame["shap_backend"] = str(backend)
                    fold_frame["perturbation_projection"] = (
                        "fitted_model_input_geometry"
                        if fold.get("input_projector") is not None
                        else "none"
                    )
                    shap_fold_frames.append(fold_frame)
                    if shap_local_enabled:
                        shap_oof_rows.extend(
                            _local_value_rows_for_fold(
                                fold,
                                vals,
                                rows_ex,
                                tuple(range(len(dataset.class_labels))),
                                dataset.class_labels,
                                dataset.sample_ids,
                                dataset.y,
                                positive_class=dataset.positive_class,
                            )
                        )
        else:
            fold_totals = {
                fold_no: max(
                    1,
                    len(np.asarray(fold["test_idx"], dtype=int))
                    if shap_local_enabled
                    else min(
                        int(spec.max_explain),
                        len(np.asarray(fold["test_idx"], dtype=int)),
                    ),
                )
                for fold_no, fold in enumerate(oof_folds, start=1)
            }
            info(
                f"SHAP workload · {sum(fold_totals.values()):,} explained samples · "
                f"{len(oof_folds)} folds × up to {int(spec.max_explain)} samples"
            )
            with Manager() as manager:
                progress_queue = manager.Queue()
                shap_tasks = [
                    (
                        _shap_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            dataset.sample_ids,
                            dataset.y,
                            dataset.positive_class,
                            shap_local_enabled,
                            sweep.explainability.random_state + fold_no * 997,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                parallel_results = _parallel_progress_results(
                    shap_tasks,
                    execution,
                    progress_queue,
                    label="SHAP",
                    unit_label="samples",
                    fold_totals=fold_totals,
                )
            fold_results: dict[
                int, tuple[pd.DataFrame, list[dict[str, Any]], int, str]
            ] = {}
            for fold_no, fold_frame, rows, explained_count, backend in parallel_results:
                fold_results[int(fold_no)] = (
                    fold_frame,
                    rows,
                    int(explained_count),
                    str(backend),
                )
            for fold_no in sorted(fold_results):
                fold_frame, rows, _, _ = fold_results[fold_no]
                shap_fold_frames.append(fold_frame)
                shap_oof_rows.extend(rows)
        backend_counts: dict[str, int] = {}
        for frame in shap_fold_frames:
            if "shap_backend" not in frame.columns or frame.empty:
                continue
            backend = str(frame["shap_backend"].iloc[0])
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        if backend_counts:
            resolved_shap_backends.update(backend_counts)
            info(
                "SHAP backend routing · "
                + ", ".join(f"{k}={v}" for k, v in sorted(backend_counts.items()))
            )
        info("Aggregating SHAP global importance and cross-fold stability")
        shap_frame = _aggregate_fold_feature_importance(
            "shap",
            shap_fold_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            "outer_fold_mean_abs_probability_shap",
            sweep.explainability.top_k,
        )
        if backend_counts:
            shap_frame["shap_backend"] = ",".join(sorted(backend_counts))
        method_frames.append(shap_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "shap", shap_fold_frames, target_dir
        )
        method_outputs["shap_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                shap_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path, previous_meta, "shap", current_method_entries["shap"]
        )
        previous_method_cache["shap"] = current_method_entries["shap"]
        success("Class-specific OOF SHAP completed")

    lime_local_enabled = local_enabled and "lime" in local_methods
    if "lime" in global_methods and "lime" not in reusable_methods:
        spec = specs_by_name["lime"]
        info("Running global class-specific OOF LIME feature attribution")
        lime_fold_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            prog = progress()
            prog.start()
            for fold_no, fold in enumerate(oof_folds, start=1):
                fold_feature_names = _fold_feature_names(fold, feature_names)
                prefix = f"LIME fold {fold_no}/{len(oof_folds)} · {len(fold_feature_names)} features · {len(class_indices)} classes"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                force_rows = (
                    list(range(len(np.asarray(fold["test_idx"], dtype=int))))
                    if lime_local_enabled
                    else []
                )
                local_class_indices = (
                    tuple(range(len(dataset.class_labels)))
                    if lime_local_enabled
                    else class_indices
                )
                coeffs, rows_ex = _lime_values_for_data(
                    configure_estimator_threads(
                        fold["estimator"], int(execution.threads_per_worker)
                    ),
                    fold["X_train"],
                    fold["X_test"],
                    fold_feature_names,
                    dataset.class_labels,
                    local_class_indices,
                    random_state=sweep.explainability.random_state + fold_no * 997,
                    spec=spec,
                    force_explain_rows=force_rows,
                    progress_callback=_progress_callback(prog, task, prefix),
                    input_projector=fold.get("input_projector"),
                )
                fold_frame = _value_frame_for_fold(
                    "lime",
                    coeffs,
                    fold_feature_names,
                    class_indices,
                    dataset.class_labels,
                    "mean_abs_lime_coefficient_within_outer_fold",
                )
                fold_frame["fold_key"] = str(fold.get("split_key", ""))
                fold_frame["fold_no"] = int(fold_no)
                fold_frame["perturbation_projection"] = (
                    "fitted_model_input_geometry"
                    if fold.get("input_projector") is not None
                    else "none"
                )
                lime_fold_frames.append(fold_frame)
                if lime_local_enabled:
                    lime_oof_rows.extend(
                        _local_value_rows_for_fold(
                            fold,
                            coeffs,
                            rows_ex,
                            local_class_indices,
                            dataset.class_labels,
                            dataset.sample_ids,
                            dataset.y,
                            positive_class=dataset.positive_class,
                        )
                    )
            prog.stop()
        else:
            fold_totals = {
                fold_no: max(
                    1,
                    len(np.asarray(fold["test_idx"], dtype=int))
                    if lime_local_enabled
                    else min(
                        int(spec.max_explain),
                        len(np.asarray(fold["test_idx"], dtype=int)),
                    ),
                )
                for fold_no, fold in enumerate(oof_folds, start=1)
            }
            info(
                f"LIME workload · {sum(fold_totals.values()):,} explained samples · "
                f"{len(oof_folds)} folds × up to {int(spec.max_explain)} samples"
            )
            with Manager() as manager:
                progress_queue = manager.Queue()
                tasks = [
                    (
                        _lime_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            dataset.sample_ids,
                            dataset.y,
                            dataset.positive_class,
                            lime_local_enabled,
                            sweep.explainability.random_state + fold_no * 997,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                parallel_results = _parallel_progress_results(
                    tasks,
                    execution,
                    progress_queue,
                    label="LIME",
                    unit_label="samples",
                    fold_totals=fold_totals,
                )
            results: dict[int, tuple[pd.DataFrame, list[dict[str, Any]]]] = {}
            for fold_no, frame, rows in parallel_results:
                results[int(fold_no)] = (frame, rows)
            lime_fold_frames = [results[i][0] for i in sorted(results)]
            for i in sorted(results):
                lime_oof_rows.extend(results[i][1])
        info("Aggregating LIME global importance and cross-fold stability")
        lime_frame = _aggregate_fold_feature_importance(
            "lime",
            lime_fold_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            "outer_fold_mean_abs_lime_coefficient",
            sweep.explainability.top_k,
        )
        method_frames.append(lime_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "lime", lime_fold_frames, target_dir
        )
        method_outputs["lime_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                lime_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path, previous_meta, "lime", current_method_entries["lime"]
        )
        previous_method_cache["lime"] = current_method_entries["lime"]
        success("Class-specific OOF LIME completed")

    if "permutation" in methods and "permutation" not in reusable_methods:
        spec = specs_by_name["permutation"]
        info("Running global class-specific OOF permutation importance")
        perm_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            prog = progress()
            prog.start()
            for fold_no, fold in enumerate(oof_folds, start=1):
                fold_feature_names = _fold_feature_names(fold, feature_names)
                prefix = f"Permutation fold {fold_no}/{len(oof_folds)} · {len(fold_feature_names)} features · {len(class_indices)} classes"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                frame = _permutation_feature_importance(
                    configure_estimator_threads(
                        fold["estimator"], int(execution.threads_per_worker)
                    ),
                    fold["X_test"],
                    fold["y_test"],
                    fold_feature_names,
                    dataset.class_labels,
                    class_indices,
                    spec=spec,
                    random_state=sweep.explainability.random_state + fold_no * 997,
                    progress_callback=_progress_callback(prog, task, prefix),
                    input_projector=fold.get("input_projector"),
                )
                frame["fold_key"] = str(fold.get("split_key", ""))
                frame["fold_no"] = int(fold_no)
                perm_frames.append(frame)
            prog.stop()
        else:
            total_per_fold = max(1, len(feature_names) * int(spec.n_repeats))
            fold_totals = {
                fold_no: total_per_fold for fold_no, _ in enumerate(oof_folds, start=1)
            }
            info(
                f"Permutation workload · {sum(fold_totals.values()):,} feature-repeat jobs · "
                f"{len(feature_names):,} features × {int(spec.n_repeats)} repeats × {len(oof_folds)} folds"
            )
            with Manager() as manager:
                progress_queue = manager.Queue()
                tasks = [
                    (
                        _permutation_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            sweep.explainability.random_state + fold_no * 997,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                parallel_results = _parallel_progress_results(
                    tasks,
                    execution,
                    progress_queue,
                    label="Permutation",
                    unit_label="feature-repeat jobs",
                    fold_totals=fold_totals,
                )
            results: dict[int, pd.DataFrame] = {}
            for fold_no, frame in parallel_results:
                results[int(fold_no)] = frame
            perm_frames = [results[i] for i in sorted(results)]
        info("Aggregating permutation global importance and cross-fold stability")
        permutation_frame = _aggregate_fold_feature_importance(
            "permutation",
            perm_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            f"outer_fold_increase_in_one_vs_rest_{spec.scoring}",
            sweep.explainability.top_k,
        )
        method_frames.append(permutation_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "permutation", perm_frames, target_dir
        )
        method_outputs["permutation_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                permutation_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path,
            previous_meta,
            "permutation",
            current_method_entries["permutation"],
        )
        previous_method_cache["permutation"] = current_method_entries["permutation"]
        success("Class-specific OOF permutation importance completed")

    if "ale" in methods and "ale" not in reusable_methods:
        spec = specs_by_name["ale"]
        info("Running global class-specific OOF ALE feature effects")
        ale_frames: list[pd.DataFrame] = []
        curve_frames: list[pd.DataFrame] = []
        skipped_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            prog = progress()
            prog.start()
            for fold_no, fold in enumerate(oof_folds, start=1):
                fold_feature_names = _fold_feature_names(fold, feature_names)
                prefix = f"ALE fold {fold_no}/{len(oof_folds)} · {len(fold_feature_names)} features · {len(class_indices)} classes"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                frame = _ale_feature_importance(
                    configure_estimator_threads(
                        fold["estimator"], int(execution.threads_per_worker)
                    ),
                    fold["X_test"],
                    fold_feature_names,
                    dataset.class_labels,
                    class_indices,
                    spec=spec,
                    top_features=None,
                    progress_callback=_progress_callback(prog, task, prefix),
                    input_projector=fold.get("input_projector"),
                )
                skipped = frame.attrs.get("skipped_features")
                if isinstance(skipped, pd.DataFrame) and not skipped.empty:
                    skipped = skipped.copy()
                    skipped["fold_key"] = str(fold.get("split_key", ""))
                    skipped["fold_no"] = int(fold_no)
                    skipped_frames.append(skipped)
                if frame.empty:
                    continue
                curves = frame.attrs.get("curves")
                frame = frame.copy()
                frame["fold_key"] = str(fold.get("split_key", ""))
                frame["fold_no"] = int(fold_no)
                ale_frames.append(frame)
                if isinstance(curves, pd.DataFrame) and not curves.empty:
                    curves = curves.copy()
                    curves["fold_key"] = str(fold.get("split_key", ""))
                    curves["fold_no"] = int(fold_no)
                    curve_frames.append(curves)
            prog.stop()
        else:
            total_per_fold = max(1, len(feature_names) * len(class_indices))
            total_work = max(1, len(oof_folds) * total_per_fold)
            info(
                f"ALE workload · {total_work:,} feature-class jobs · "
                f"{len(feature_names):,} features × {len(class_indices)} classes × {len(oof_folds)} folds"
            )
            results: dict[
                int, tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]
            ] = {}
            with Manager() as manager:
                progress_queue = manager.Queue()
                tasks = [
                    (
                        _ale_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                fold_progress = {fold_no: 0 for fold_no in range(1, len(oof_folds) + 1)}
                stop_monitor = Event()
                with progress() as prog:
                    task = prog.add_task(
                        f"ALE · 0/{total_work:,} feature-class jobs · {execution.workers} workers",
                        total=total_work,
                    )

                    def monitor() -> None:
                        while not stop_monitor.is_set():
                            try:
                                fold_no, completed, total, detail = progress_queue.get(
                                    timeout=0.25
                                )
                            except Empty:
                                continue
                            fold_no = int(fold_no)
                            fold_progress[fold_no] = max(
                                fold_progress.get(fold_no, 0),
                                min(max(0, int(completed)), max(1, int(total))),
                            )
                            overall = min(total_work, sum(fold_progress.values()))
                            prog.update(
                                task,
                                completed=overall,
                                description=(
                                    f"ALE · {overall:,}/{total_work:,} feature-class jobs · "
                                    f"fold {fold_no}/{len(oof_folds)} · {detail}"
                                ),
                            )

                    monitor_thread = Thread(target=monitor, daemon=True)
                    monitor_thread.start()
                    try:
                        for fold_no, frame, curves, skipped in _xai_task_iterator(
                            tasks, execution
                        ):
                            results[int(fold_no)] = (frame, curves, skipped)
                            fold_progress[int(fold_no)] = total_per_fold
                            overall = min(total_work, sum(fold_progress.values()))
                            prog.update(
                                task,
                                completed=overall,
                                description=(
                                    f"ALE · {overall:,}/{total_work:,} feature-class jobs · "
                                    f"completed fold {fold_no}/{len(oof_folds)}"
                                ),
                            )
                    finally:
                        stop_monitor.set()
                        monitor_thread.join(timeout=2.0)
                    prog.update(
                        task,
                        completed=total_work,
                        description=f"ALE · completed {total_work:,}/{total_work:,} feature-class jobs",
                    )
            for fold_no in sorted(results):
                frame, curves, skipped = results[fold_no]
                if skipped is not None and not skipped.empty:
                    skipped_frames.append(skipped)
                if not frame.empty:
                    ale_frames.append(frame)
                if curves is not None and not curves.empty:
                    curve_frames.append(curves)
        info("Aggregating ALE global effects and cross-fold stability")
        if skipped_frames:
            write_table(
                target_dir / "ale_skipped_features.parquet",
                pd.concat(skipped_frames, ignore_index=True),
            )
        if not ale_frames:
            raise ExplainabilityConfigurationError(
                "ALE was requested, but no candidate feature was estimable in any outer-test fold."
            )
        ale_frame = _aggregate_fold_feature_importance(
            "ale",
            ale_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            "outer_fold_rms_distribution_weighted_class_probability_ale",
            sweep.explainability.top_k,
        )
        if curve_frames:
            curves_all = pd.concat(curve_frames, ignore_index=True)
            write_table(target_dir / "ale_curves.parquet", curves_all)
            for class_index in class_indices:
                label = str(dataset.class_labels[int(class_index)])
                slug = _class_slug(label)
                class_imp = ale_frame[ale_frame["class_index"].eq(int(class_index))]
                top_features = (
                    class_imp.dropna(subset=["importance_mean"])
                    .sort_values("importance_mean", ascending=False)
                    .head(sweep.explainability.top_k)["feature"]
                    .astype(str)
                    .tolist()
                )
                class_curves = curves_all[
                    curves_all["class_index"].eq(int(class_index))
                ]
                curve_stem = figures_dir / f"ale_curves__{slug}"
                _plot_ale_curves(
                    class_curves,
                    top_features,
                    curve_stem,
                    max_panels=min(12, sweep.explainability.top_k),
                    x_label="Model-input feature value",
                    y_label=f"Centered ALE effect on P({label})",
                )
                method_outputs[f"ale_curves_{slug}"] = curve_stem.with_suffix(".svg")
        method_frames.append(ale_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "ale", ale_frames, target_dir
        )
        method_outputs["ale_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                ale_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path, previous_meta, "ale", current_method_entries["ale"]
        )
        previous_method_cache["ale"] = current_method_entries["ale"]
        success("Class-specific OOF ALE completed")

    if not method_frames and not local_enabled:
        raise ExplainabilityConfigurationError(
            "No global or local explainability method was executed."
        )

    imp = (
        _combine_feature_importance(method_frames) if method_frames else pd.DataFrame()
    )
    write_table(target_dir / "feature_importance.parquet", imp)
    stability_all = (
        pd.concat(
            [
                frame.assign(method=str(frame["method"].iloc[0]))
                for frame in method_frames
                if not frame.empty
            ],
            ignore_index=True,
        )
        if method_frames
        else pd.DataFrame()
    )
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
    write_table(
        target_dir / "feature_stability.parquet",
        stability_all[[c for c in stability_columns if c in stability_all.columns]],
    )
    method_outputs["feature_stability"] = target_dir / "feature_stability.parquet"
    fold_all = (
        pd.concat(fold_importance_frames, ignore_index=True)
        if fold_importance_frames
        else pd.DataFrame()
    )
    write_table(target_dir / "feature_importance_by_outer_fold.parquet", fold_all)
    method_outputs["feature_importance_by_outer_fold"] = (
        target_dir / "feature_importance_by_outer_fold.parquet"
    )

    if "interactions" in global_methods and "interactions" not in reusable_methods:
        spec = specs_by_name["interactions"]
        info("Running class-specific OOF ALE interaction analysis")
        by_method: dict[str, list[pd.DataFrame]] = {
            m: [] for m in ("current", "corrected", "fixed_pairs", "corrected_fixed")
        }
        all_raw: list[pd.DataFrame] = []
        prog = progress()
        prog.start()
        for class_index in class_indices:
            score_series = imp[imp["class_index"].eq(int(class_index))].set_index(
                "feature"
            )["importance_mean"]
            for fold_no, fold in enumerate(oof_folds, start=1):
                fold_feature_names = _fold_feature_names(fold, feature_names)
                prefix = f"Interactions · {dataset.class_labels[int(class_index)]} · fold {fold_no}/{len(oof_folds)}"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                try:
                    tables = _ale_interactions(
                        fold["estimator"],
                        fold["X_test"],
                        fold_feature_names,
                        dataset.class_labels,
                        int(class_index),
                        spec=spec,
                        scores=score_series,
                        progress_callback=_progress_callback(prog, task, prefix),
                        input_projector=fold.get("input_projector"),
                    )
                except ExplainabilityConfigurationError:
                    prog.update(
                        task, completed=1, total=1, description=f"{prefix} · skipped"
                    )
                    continue
                for name, tab in tables.items():
                    tab = tab.copy()
                    tab["fold_key"] = str(fold.get("split_key", ""))
                    if name == "all_methods":
                        all_raw.append(tab)
                    elif name in by_method:
                        by_method[name].append(tab)
        prog.stop()
        info("Aggregating interaction stability and writing outputs")
        interaction_tables = {
            name: _aggregate_oof_interactions(frames, name)
            for name, frames in by_method.items()
            if frames
        }
        if all_raw:
            all_path = target_dir / "feature_interactions_by_target_all_methods.parquet"
            write_table(all_path, pd.concat(all_raw, ignore_index=True))
            interaction_outputs["interactions_all_methods"] = all_path
        for name, tab in interaction_tables.items():
            path = target_dir / f"feature_interactions_{name}.parquet"
            write_table(path, tab)
            interaction_outputs[f"interactions_{name}"] = path
        _persist_method_cache_entry(
            meta_path,
            previous_meta,
            "interactions",
            current_method_entries["interactions"],
        )
        previous_method_cache["interactions"] = current_method_entries["interactions"]
        success("Class-specific OOF interaction analysis completed")

    if "interactions" in methods:
        interaction_outputs.update(
            _ensure_interaction_network_outputs(
                target_dir,
                figures_dir,
                X_base,
                dataset.y,
                feature_names,
                dataset.class_labels,
                class_indices,
                int(specs_by_name["interactions"].top_k),
                dataset,
                coordinate_metadata,
            )
        )

    method_outputs.update(
        _ensure_local_explanation_outputs(
            target_dir,
            source_signature,
            sweep,
            specs_by_name,
            methods,
            oof_folds,
            dataset,
            feature_names,
            int(execution.threads_per_worker),
            precomputed_rows={"shap": shap_oof_rows, "lime": lime_oof_rows},
        )
    )

    top_features = (
        _method_support_table(method_frames, imp, top_k=sweep.explainability.top_k)
        if method_frames and not imp.empty
        else pd.DataFrame()
    )
    write_table(target_dir / "top_features.parquet", top_features)
    dist_frames: list[pd.DataFrame] = []
    class_figure_paths: dict[str, Path] = {}
    grouped_top_features = (
        top_features.groupby("class_index", sort=True)
        if not top_features.empty and "class_index" in top_features.columns
        else ()
    )
    for class_index, class_top in grouped_top_features:
        c = int(class_index)
        label = str(dataset.class_labels[c])
        slug = _class_slug(label)
        dist = _feature_distribution_stats(
            class_top["feature"].astype(str).tolist(),
            feature_names,
            X_base,
            dataset.y,
            dataset.class_labels,
        )
        dist.insert(0, "class_label", label)
        dist.insert(0, "class_index", c)
        dist_frames.append(dist)
        support_stem = figures_dir / f"feature_support__{slug}"
        _plot_feature_importance(
            class_top,
            dist,
            support_stem,
            sweep.explainability.top_k,
            dataset.class_labels,
            stability_all,
        )
        class_figure_paths[f"feature_support_{slug}"] = support_stem.with_suffix(".svg")
    dist_all = (
        pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
    )
    write_table(target_dir / "feature_distribution_stats.parquet", dist_all)

    dump_json_standard(
        {
            "pipeline_schema": _EXPLAINABILITY_PIPELINE_SCHEMA,
            "target": target_slug,
            "target_label": target_label,
            "config": row.to_dict(),
            "methods": list(methods),
            "method_parameters": [method_to_dict(x) for x in method_specs],
            "method_cache": {
                **previous_method_cache,
                **current_method_entries,
            },
            "explanation_source_signature": source_signature,
            "explainability_config": _explainability_config_payload(
                sweep.explainability
            ),
            "explainability_config_signature": _explainability_config_signature(
                sweep.explainability
            ),
            "explained_class_indices": [int(x) for x in class_indices],
            "explained_class_labels": [
                str(dataset.class_labels[int(x)]) for x in class_indices
            ],
            "class_target": "class_probability",
            "explanation_layers": [
                *(["global", "cross_fold_stability"] if global_methods else []),
                *(["local"] if local_enabled else []),
            ],
            "local_explanations": local_mode,
            "local_methods": list(local_methods) if local_enabled else [],
            "local_target": "observed_class_probability",
            "local_storage": "top_k_per_oof_split_full_vectors_for_report_representatives",
            "local_top_k": int(sweep.explainability.local.stored_features),
            "global_methods": [x for x in global_methods if x != "interactions"],
            "interaction_scope": "global_exploratory"
            if "interactions" in global_methods
            else "not_requested",
            "shap_backends": sorted(resolved_shap_backends),
            "default_suite": ["shap", "lime", "ale", "permutation", "interactions"],
            "strict": True,
            "fallbacks": False,
            "cross_validation_explanations": "outer_test_folds",
            "out_of_fold": True,
            "perturbation_geometry": geometries,
            "geometry_projection_applied": bool(projection_applied),
            "perturbation_interpretation": perturbation_policy["interpretation"],
            "n_outer_folds_explained": len(oof_folds),
            "ale_feature_scope": "all_estimable_features",
            "consensus_strategy": "mean_within_method_rank_support_available_methods",
            "consensus_missing_values": "excluded_not_zero",
            "stability_unit": "outer_fold",
            "stability_interpretation": "descriptive_cross_fit_variability_not_independent_fold_confidence_intervals",
            "stability_statistics": [
                "importance_sd",
                "rank_iqr",
                "top_k_frequency",
                "fold_coverage",
                "sign_consistency",
            ],
            "fold_level_importance_file": "feature_importance_by_outer_fold.parquet",
            "execution": {
                "workers": int(execution.workers),
                "threads_per_worker": int(execution.threads_per_worker),
                "backend": str(execution.backend),
                "logical_cpus": int(execution.logical_cpus),
                "physical_cpus": int(execution.physical_cpus),
            },
        },
        target_dir / "explained_unit.json",
    )
    if not ensemble_explain and cache_dir is not None and target_dir != cache_dir:
        _copy_explainability_cache(target_dir, cache_dir)
    success(
        f"Explainability completed · target={target_label} · methods={','.join(methods)}"
    )
    outputs = {
        "explainability_dir": target_dir,
        "importance": target_dir / "feature_importance.parquet",
        "stability": target_dir / "feature_stability.parquet",
        "perturbation_policy": perturbation_policy_path,
    }
    if coordinate_metadata_path is not None:
        outputs["coordinate_metadata"] = coordinate_metadata_path
    if coordinate_metadata_by_fold_path is not None:
        outputs["coordinate_metadata_by_outer_fold"] = coordinate_metadata_by_fold_path
    if prediction_reproduction_path is not None:
        outputs["prediction_reproduction"] = prediction_reproduction_path
    outputs.update(class_figure_paths)
    outputs.update(method_outputs)
    outputs.update(interaction_outputs)
    path_table("Explainability outputs", outputs)
    return outputs


def _score_sort_column(df: pd.DataFrame) -> str | None:
    for col in (
        "rank",
        "inner_score",
        "inner_validation_score",
        "inner_log_loss_mean",
        "log_loss_mean",
        "score",
        "AUROC_mean",
        "MCC_mean",
    ):
        if col in df.columns:
            return col
    numeric = [
        c
        for c in df.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric[0] if numeric else None


def _row_levels(row: pd.Series) -> tuple[str, ...]:
    val = row.get("levels", row.get("resolution", ""))
    if pd.isna(val):
        return ()
    return tuple(x.strip() for x in str(val).replace("+", ",").split(",") if x.strip())


def _resolve_baseline_rf_row(rankings: pd.DataFrame) -> pd.Series | None:
    required = rankings.copy()
    if (
        "learner" not in required.columns
        or "count_transformation" not in required.columns
    ):
        return None
    required = required[
        required["learner"].astype(str).eq("RF_1000_msl5")
        & required["count_transformation"]
        .astype(str)
        .str.fullmatch(r"arcsine_sqrt(?:@(rank|global))?", case=False)
        .fillna(False)
    ].copy()
    if required.empty:
        return None
    required["_single_rank"] = required.apply(
        lambda r: _row_levels(r)[0] if len(_row_levels(r)) == 1 else "", axis=1
    )
    for rank in ("strain", "species", "genus"):
        sub = required[required["_single_rank"].eq(rank)].copy()
        if sub.empty:
            continue
        sort_col = _score_sort_column(sub)
        if sort_col and sort_col in sub.columns:
            if sort_col == "rank":
                ascending = True
            elif sort_col in {"inner_score", "inner_validation_score"}:
                metric = str(sub.get("selection_metric", pd.Series([""])).iloc[0])
                ascending = metric_is_loss(metric)
            else:
                ascending = "loss" in sort_col.casefold()
            sub = sub.sort_values(sort_col, ascending=ascending)
        return sub.iloc[0].drop(labels=["_single_rank"], errors="ignore")
    return None


def _parse_members(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [x.strip() for x in text.split(",") if x.strip()]
    else:
        parsed = value
    out: list[str] = []
    if isinstance(parsed, dict):
        parsed = parsed.get("members", [])
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, dict):
                cid = item.get("config_id") or item.get("id")
            else:
                cid = item
            if cid is not None:
                out.append(str(cid))
    return out


def _selected_ensemble_members(root: Path) -> tuple[list[str], str]:
    selected_path = root / "ensembling" / "selected_unit.json"
    if not selected_path.exists():
        raise ExplainabilityConfigurationError(
            "MPMA-E explainability requires ensembling/selected_unit.json."
        )
    with open(selected_path, "r", encoding="utf-8") as fh:
        selected = json.load(fh)
    unit = selected.get(
        "inner_val_best_mpmas_ensemble", selected.get("MPMA-E", selected)
    )
    members = _parse_members(unit.get("members") if isinstance(unit, dict) else None)
    aggregation = (
        str(unit.get("aggregation_strategy", "mean_proba"))
        if isinstance(unit, dict)
        else "mean_proba"
    )
    if not members:
        raise ExplainabilityConfigurationError(
            "MPMA-E selected unit does not contain members."
        )
    return members, aggregation


def _standard_explainability_targets(sweep: Sweep, rankings: pd.DataFrame) -> list[str]:
    targets: list[str] = []
    if (sweep.root() / "ensembling" / "selected_unit.json").exists() and not getattr(
        sweep, "uses_modalities", False
    ):
        targets.append("mpma_e")
    targets.append("mpma_b")
    if _resolve_baseline_rf_row(rankings) is not None:
        targets.append("baseline_rf")
    return targets


def _automatic_explainability_targets(
    sweep: Sweep, rankings: pd.DataFrame
) -> list[str]:
    configured = getattr(sweep.explainability, "targets", "auto")
    requested = [configured] if isinstance(configured, str) else list(configured)
    standard_aliases = {"auto", "all", "comparison", "standard", "standard_suite"}
    targets: list[str] = []
    for item in requested:
        target = str(item).strip()
        if not target:
            continue
        if target in standard_aliases:
            targets.extend(_standard_explainability_targets(sweep, rankings))
        else:
            targets.append(target)
    if not targets:
        targets.extend(_standard_explainability_targets(sweep, rankings))

    return list(dict.fromkeys(targets))


def explain(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    rankings_path = root / "tables" / "mpma_inner_rankings.parquet"
    if not table_exists(rankings_path):
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = read_table(rankings_path)
    targets = _automatic_explainability_targets(sweep, rankings)
    outputs: dict[str, Path] = {}
    for target in targets:
        out = _explain_one(sweep, target_override=target)
        key = {
            "mpma_e": "mpma_e",
            "ensemble": "mpma_e",
            "MPMA-E": "mpma_e",
            "baseline_rf": "baseline_rf",
            "Baseline RF": "baseline_rf",
            "baseline-rf": "baseline_rf",
            "mpma_b": "mpma_b",
            "best_individual": "mpma_b",
            "best": "mpma_b",
            "MPMA-B": "mpma_b",
        }.get(str(target), str(target))
        for name, path in out.items():
            outputs[f"{key}_{name}"] = path
    return outputs
