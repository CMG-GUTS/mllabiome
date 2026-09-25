from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from joblib import delayed
from threadpoolctl import threadpool_limits

from ._version import __version__
from .compute import ResourceTracker, machine_profile
from .configs_sweep import (
    MPDR,
    QualificationGate,
    _count_transformation_factory,
    _failed_metric_row,
    _lodo_feature_pair,
    _metric_row,
    _predict_proba_aligned,
    _prediction_rows_values,
    _regression_metric_row,
    _regression_prediction_rows,
    _backfill_metrics_from_predictions,
    _backfill_regression_metrics_from_predictions,
    _checkpoint_result,
    _clear_evaluation_checkpoints,
    _done_pairs,
    _existing_inner_scores,
    _existing_outputs,
    _experiment_fingerprint,
    _groups_from_metadata,
    _learner_factory,
    _learner_name,
    _load_existing_evaluation,
    _outer_pair_complete,
    _prepare_dirs,
    _qualification_map,
    _resolved_evaluation_splits,
    _scientific_digest,
    _scientific_value,
    _software_provenance,
    _source_tree_sha256,
    _strata_from_metadata,
    _subject_safe_groups,
    _validate_experiment_identity,
    _write_config_table,
    _write_rankings_and_figures,
    _write_tables,
    write_mpma_b_selection_outputs,
)
from .console import path_table, progress, stage, success, summary_table
from .estimator_protocol import EstimatorLike
from .figures import _write_representation_impact_figure
from .integrations import Integration
from .integrations import IntegrationModel as IntegrationModel
from .integrations import integration_modality_sets as integration_modality_sets
from .learners import fit_classifier
from .metrics import (
    _estimator_call,
    aggregate_validation_metric,
    canonical_metric_name,
    compute_metrics,
    compute_regression_metrics,
    grouped_log_loss,
)
from .modalities import (
    ModalityDataset,
    load_modalities,
    modality_dataset_fingerprint,
    modality_fingerprints,
    modality_source_fingerprints,
)
from .multimodal_candidates import CandidateSpec
from .multimodal_candidates import ModalityPath as ModalityPath
from .multimodal_candidates import _candidate_id as _candidate_id
from .multimodal_candidates import _canonical as _canonical
from .multimodal_candidates import (
    _IntegratedClassificationPredictor,
    _IntegratedRegressionPredictor,
)
from .multimodal_candidates import (
    _materialize_modality_representation as _materialize_modality_representation,
)
from .multimodal_candidates import _modality_paths as _modality_paths
from .multimodal_candidates import (
    _ModalityInputProjector,
    _normalise_modality_transformations,
    _prepare_candidate_pair,
    _prepare_candidate_pair_details,
    _representation_cache,
)
from .multimodal_candidates import _representation_specs as _representation_specs
from .multimodal_candidates import _transformation_specs as _transformation_specs
from .multimodal_candidates import build_modality_candidates
from .resolutions import FeatureBlocks
from .resolutions import mask_feature_blocks as mask_feature_blocks
from .resolutions import materialize_mpdr_with_blocks as materialize_mpdr_with_blocks
from .runtime import (
    ExecutionPlan,
    configure_estimator_threads,
    iter_parallel_tasks,
    resolve_execution_plan,
    thread_environment,
)
from .sweep_types import Sweep, validate_sweep_class_count
from .transformations import (
    _count_transformation_specs_for_blocks as _count_transformation_specs_for_blocks,
)
from .utils import dump_json_standard


_RepresentationKey: TypeAlias = tuple[str, str, tuple[str, ...]]
_RepresentationMatrices: TypeAlias = dict[_RepresentationKey, np.ndarray]
_RepresentationNames: TypeAlias = dict[
    _RepresentationKey, tuple[list[str], FeatureBlocks]
]
_LearnerFactory: TypeAlias = Callable[[], EstimatorLike]
_DelayedTask: TypeAlias = tuple[
    Callable[..., dict[str, Any]], tuple[Any, ...], dict[str, Any]
]
_ScheduledTask: TypeAlias = tuple[str, str, _DelayedTask]
_OuterSplits: TypeAlias = list[dict[str, Any]]
_InnerSplitsByOuter: TypeAlias = dict[str, list[tuple[np.ndarray, np.ndarray]]]


@dataclass(frozen=True)
class _PreparedModalityTask:
    split_key: str
    config_id: str
    fn: Callable[..., dict[str, Any]]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class _ModalityEvaluationRows:
    outer_metrics: list[dict[str, Any]]
    inner_metrics: list[dict[str, Any]]
    outer_predictions: list[dict[str, Any]]
    inner_predictions: list[dict[str, Any]]
    qualification: list[dict[str, Any]]
    job_resources: list[dict[str, Any]]

    @classmethod
    def empty(cls) -> _ModalityEvaluationRows:
        return cls([], [], [], [], [], [])

    def extend(self, result: Mapping[str, Any]) -> None:
        self.outer_metrics.extend(result.get("outer_metrics", []))
        self.inner_metrics.extend(result.get("inner_metrics", []))
        self.outer_predictions.extend(result.get("outer_predictions", []))
        self.inner_predictions.extend(result.get("inner_predictions", []))
        self.qualification.extend(result.get("qualification", []))
        self.job_resources.extend(result.get("job_resources", []))


@dataclass(frozen=True)
class _ModalitySweepContext:
    root: Path
    dataset: ModalityDataset
    learners: dict[str, _LearnerFactory]
    integrations: tuple[Integration, ...]
    matrices: _RepresentationMatrices
    names: _RepresentationNames
    specs: list[CandidateSpec]
    configs: pd.DataFrame
    groups: np.ndarray | None
    outer_splits: _OuterSplits
    inner_splits_by_outer: _InnerSplitsByOuter
    existing: dict[str, pd.DataFrame]
    inner_done: set[tuple[str, str]]
    outer_done: set[tuple[str, str]]
    qualification_map: dict[tuple[str, str], Any]


def _meta_row(base: dict[str, Any], spec: CandidateSpec) -> dict[str, Any]:
    base.update(
        {
            "candidate_family": spec.family,
            "modalities": ",".join(spec.modalities),
            "integration": spec.integration.key,
            "integration_n_components": ""
            if spec.n_components is None
            else str(int(spec.n_components)),
        }
    )
    return base


def _emit_evaluation_progress(
    progress_queue, split_key, config_id, phase, completed, total, detail
):
    if progress_queue is None:
        return
    progress_queue.put(
        (
            str(split_key),
            str(config_id),
            str(phase),
            int(completed),
            max(1, int(total)),
            str(detail),
        )
    )


def _result_failure_count(result):
    rows = list(result.get("inner_metrics", [])) + list(result.get("outer_metrics", []))
    return sum(1 for row in rows if int(row.get("ok", 1)) == 0)


def _new_modality_result(spec: CandidateSpec, split_key: str) -> dict[str, Any]:
    return {
        "split_key": split_key,
        "config_id": spec.config_id,
        "inner_metrics": [],
        "inner_predictions": [],
        "outer_metrics": [],
        "outer_predictions": [],
        "qualification": [],
        "job_resources": [],
        "fits": 0,
    }


def _fit_classification_candidate_fold(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    y: np.ndarray,
    groups: np.ndarray | None,
    classes: np.ndarray,
    subject_ids: Sequence[str],
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    protocol: str,
    learner_factory: _LearnerFactory,
    requested_metrics: set[str],
    random_state: int,
    threads_per_worker: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    X_train, X_eval, _ = _prepare_candidate_pair(
        spec,
        matrices,
        names,
        train_idx,
        eval_idx,
        protocol,
        random_state,
        _count_transformation_factory,
        _lodo_feature_pair,
    )
    estimator = configure_estimator_threads(learner_factory(), threads_per_worker)
    fit_classifier(
        estimator,
        X_train,
        y[train_idx],
        None if groups is None else groups[train_idx],
    )
    proba = _predict_proba_aligned(estimator, X_eval, classes)
    pred = classes[proba.argmax(axis=1)]
    metrics = compute_metrics(y[eval_idx], pred, proba, classes)
    if "subject_macro_log_loss" in requested_metrics:
        metrics["subject_macro_log_loss"] = grouped_log_loss(
            y[eval_idx],
            proba,
            classes,
            np.asarray(subject_ids, dtype=object)[eval_idx],
        )
    if "cohort_macro_log_loss" in requested_metrics and str(protocol).lower() in {
        "lodo",
        "leave_one_dataset_out",
    }:
        metrics["cohort_macro_log_loss"] = float(metrics["log_loss"])
    return np.asarray(pred), np.asarray(proba), metrics


def _classification_success_row(
    spec: CandidateSpec,
    mpdr: MPDR,
    metrics: Mapping[str, float],
    split_key: str,
    inner_key: str | None,
    stage_name: str,
    eval_idx: np.ndarray,
    subject_ids: Sequence[str],
) -> dict[str, Any]:
    row = _meta_row(
        _metric_row(
            dict(metrics),
            split_key,
            inner_key,
            spec.config_id,
            mpdr,
            spec.learner,
            stage_name,
        ),
        spec,
    )
    row["n_samples"] = len(eval_idx)
    row["n_subjects"] = len(pd.unique(np.asarray(subject_ids, dtype=object)[eval_idx]))
    return row


def _classification_failure_row(
    spec: CandidateSpec,
    mpdr: MPDR,
    split_key: str,
    inner_key: str | None,
    stage_name: str,
    exc: Exception,
) -> dict[str, Any]:
    return _meta_row(
        _failed_metric_row(
            split_key,
            inner_key,
            spec.config_id,
            mpdr,
            spec.learner,
            stage_name,
            exc,
        ),
        spec,
    )


def _classification_prediction_rows(
    spec: CandidateSpec,
    prediction_key: str,
    split_key: str,
    stage_name: str,
    eval_idx: np.ndarray,
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    y: np.ndarray,
    class_labels: Sequence[str],
    pred: np.ndarray,
    proba: np.ndarray,
) -> list[dict[str, Any]]:
    return _prediction_rows_values(
        prediction_key,
        spec.config_id,
        eval_idx,
        sample_ids,
        subject_ids,
        y,
        class_labels,
        pred,
        proba,
        stage_name,
        split_key,
    )


def _evaluate_classification_inner_folds(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    y: np.ndarray,
    groups: np.ndarray | None,
    classes: np.ndarray,
    class_labels: Sequence[str],
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    train_idx: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    split_key: str,
    protocol: str,
    learner_factory: _LearnerFactory,
    mpdr: MPDR,
    requested_metrics: set[str],
    gate_metric: str,
    existing_inner_keys: set[str],
    score_rows: list[dict[str, Any]],
    random_state: int,
    threads_per_worker: int,
    result: dict[str, Any],
    progress_queue: Any,
    completed_steps: int,
    total_steps: int,
) -> int:
    for inner_no, (train_local, validation_local) in enumerate(inner_splits):
        inner_key = f"{split_key}__i{inner_no}"
        if inner_key in existing_inner_keys:
            continue
        _emit_evaluation_progress(
            progress_queue,
            split_key,
            spec.config_id,
            "inner",
            completed_steps,
            total_steps,
            f"inner fold {inner_no + 1}/{len(inner_splits)}",
        )
        fold_train_idx = train_idx[np.asarray(train_local, dtype=int)]
        validation_idx = train_idx[np.asarray(validation_local, dtype=int)]
        if len(np.unique(y[fold_train_idx])) < 2 or len(validation_idx) == 0:
            completed_steps += 1
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "inner",
                completed_steps,
                total_steps,
                f"inner fold {inner_no + 1}/{len(inner_splits)} skipped",
            )
            continue
        try:
            pred, proba, metrics = _fit_classification_candidate_fold(
                spec,
                matrices,
                names,
                y,
                groups,
                classes,
                subject_ids,
                fold_train_idx,
                validation_idx,
                protocol,
                learner_factory,
                requested_metrics,
                random_state,
                threads_per_worker,
            )
            row = _classification_success_row(
                spec,
                mpdr,
                metrics,
                split_key,
                inner_key,
                "inner",
                validation_idx,
                subject_ids,
            )
            result["inner_metrics"].append(row)
            result["inner_predictions"].extend(
                _classification_prediction_rows(
                    spec,
                    inner_key,
                    split_key,
                    "inner",
                    validation_idx,
                    sample_ids,
                    subject_ids,
                    y,
                    class_labels,
                    pred,
                    proba,
                )
            )
            if np.isfinite(float(metrics.get(gate_metric, np.nan))):
                score_rows.append(dict(row))
            result["fits"] += 1
        except Exception as exc:
            result["inner_metrics"].append(
                _classification_failure_row(
                    spec, mpdr, split_key, inner_key, "inner", exc
                )
            )
        completed_steps += 1
        _emit_evaluation_progress(
            progress_queue,
            split_key,
            spec.config_id,
            "inner",
            completed_steps,
            total_steps,
            f"inner fold {inner_no + 1}/{len(inner_splits)} complete",
        )
    return completed_steps


def _classification_qualification(
    spec: CandidateSpec,
    split_key: str,
    gate: QualificationGate,
    gate_metric: str,
    score_rows: Sequence[Mapping[str, Any]],
    existing_qualification: Mapping[str, Any] | None,
    result: dict[str, Any],
) -> bool:
    if not gate.enabled:
        return True
    if existing_qualification is not None:
        return bool(int(existing_qualification.get("qualified", 0)))
    gate_score, _ = aggregate_validation_metric(pd.DataFrame(score_rows), gate_metric)
    qualified = QualificationGate(True, gate.metric, gate.threshold).qualifies(
        gate_score
    )
    result["qualification"].append(
        _meta_row(
            {
                "split_key": split_key,
                "config_id": spec.config_id,
                "mpdr_id": hashlib.sha1(
                    (spec.representation_label + spec.transformation_label).encode()
                ).hexdigest()[:12],
                "count_transformation": spec.transformation_label,
                "resolution": spec.representation_label,
                "levels": ",".join(spec.modalities),
                "learner": spec.learner,
                "gate_enabled": 1,
                "gate_metric": gate_metric,
                "gate_threshold": gate.threshold,
                "inner_score": gate_score,
                "qualified": int(qualified),
            },
            spec,
        )
    )
    return bool(qualified)


def _evaluate_classification_outer_fold(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    y: np.ndarray,
    groups: np.ndarray | None,
    classes: np.ndarray,
    class_labels: Sequence[str],
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_key: str,
    protocol: str,
    learner_factory: _LearnerFactory,
    mpdr: MPDR,
    requested_metrics: set[str],
    random_state: int,
    threads_per_worker: int,
    result: dict[str, Any],
) -> None:
    try:
        pred, proba, metrics = _fit_classification_candidate_fold(
            spec,
            matrices,
            names,
            y,
            groups,
            classes,
            subject_ids,
            train_idx,
            test_idx,
            protocol,
            learner_factory,
            requested_metrics,
            random_state,
            threads_per_worker,
        )
        result["outer_metrics"].append(
            _classification_success_row(
                spec,
                mpdr,
                metrics,
                split_key,
                None,
                "outer",
                test_idx,
                subject_ids,
            )
        )
        result["outer_predictions"].extend(
            _classification_prediction_rows(
                spec,
                split_key,
                split_key,
                "outer",
                test_idx,
                sample_ids,
                subject_ids,
                y,
                class_labels,
                pred,
                proba,
            )
        )
        result["fits"] += 1
    except Exception as exc:
        result["outer_metrics"].append(
            _classification_failure_row(spec, mpdr, split_key, None, "outer", exc)
        )


def _finalize_modality_candidate_result(
    spec: CandidateSpec,
    split_key: str,
    threads_per_worker: int,
    tracker: ResourceTracker,
    result: dict[str, Any],
    progress_queue: Any,
    total_steps: int,
) -> dict[str, Any]:
    measured = tracker.stop()
    result["job_resources"] = [
        _meta_row(
            {
                "split_key": split_key,
                "config_id": spec.config_id,
                "resolution": spec.representation_label,
                "count_transformation": spec.transformation_label,
                "learner": spec.learner,
                "fits": result["fits"],
                "threads_per_worker": threads_per_worker,
                **measured,
            },
            spec,
        )
    ]
    result["elapsed_s"] = float(measured.get("wall_time_s", 0.0))
    _emit_evaluation_progress(
        progress_queue,
        split_key,
        spec.config_id,
        "done",
        total_steps,
        total_steps,
        f"candidate complete · {result['fits']} fits · {_result_failure_count(result)} failures",
    )
    return result


def _classification_task(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    y: np.ndarray,
    groups: np.ndarray | None,
    classes: np.ndarray,
    class_labels: Sequence[str],
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    split_key: str,
    protocol: str,
    learner_factory: _LearnerFactory,
    gate: QualificationGate,
    selection_metric: str,
    existing_inner_keys: set[str],
    existing_inner_scores: Sequence[Mapping[str, Any]],
    existing_qualification: Mapping[str, Any] | None,
    needs_outer: bool,
    random_state: int,
    threads_per_worker: int,
    resource_sample_interval_s: float,
    progress_queue: Any = None,
) -> dict[str, Any]:
    tracker = ResourceTracker(sample_interval_s=resource_sample_interval_s).start()
    mpdr = MPDR(
        spec.representation_label, tuple(spec.modalities), spec.transformation_label
    )
    result = _new_modality_result(spec, split_key)
    gate_metric = canonical_metric_name(str(gate.metric))
    requested_metrics = {canonical_metric_name(str(selection_metric))}
    if gate.enabled:
        requested_metrics.add(gate_metric)
    score_rows = [
        dict(row) for row in existing_inner_scores if isinstance(row, Mapping)
    ]
    pending_inner = sum(
        1
        for inner_no in range(len(inner_splits))
        if f"{split_key}__i{inner_no}" not in existing_inner_keys
    )
    total_steps = max(1, pending_inner + int(bool(needs_outer)))
    completed_steps = 0
    _emit_evaluation_progress(
        progress_queue,
        split_key,
        spec.config_id,
        "start",
        completed_steps,
        total_steps,
        "starting candidate",
    )
    with (
        thread_environment(threads_per_worker),
        threadpool_limits(limits=max(1, int(threads_per_worker))),
    ):
        completed_steps = _evaluate_classification_inner_folds(
            spec,
            matrices,
            names,
            y,
            groups,
            classes,
            class_labels,
            sample_ids,
            subject_ids,
            train_idx,
            inner_splits,
            split_key,
            protocol,
            learner_factory,
            mpdr,
            requested_metrics,
            gate_metric,
            existing_inner_keys,
            score_rows,
            random_state,
            threads_per_worker,
            result,
            progress_queue,
            completed_steps,
            total_steps,
        )
        qualified = _classification_qualification(
            spec,
            split_key,
            gate,
            gate_metric,
            score_rows,
            existing_qualification,
            result,
        )
        if needs_outer and qualified:
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                "outer refit and evaluation",
            )
            _evaluate_classification_outer_fold(
                spec,
                matrices,
                names,
                y,
                groups,
                classes,
                class_labels,
                sample_ids,
                subject_ids,
                train_idx,
                test_idx,
                split_key,
                protocol,
                learner_factory,
                mpdr,
                requested_metrics,
                random_state,
                threads_per_worker,
                result,
            )
            completed_steps += 1
            detail = "outer evaluation complete"
        elif needs_outer:
            completed_steps += 1
            detail = "outer evaluation skipped by qualification gate"
        else:
            detail = ""
        if needs_outer:
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                detail,
            )
    return _finalize_modality_candidate_result(
        spec,
        split_key,
        threads_per_worker,
        tracker,
        result,
        progress_queue,
        total_steps,
    )


def _fit_regression_candidate_fold(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    protocol: str,
    learner_factory: _LearnerFactory,
    random_state: int,
    threads_per_worker: int,
) -> tuple[np.ndarray, dict[str, float]]:
    X_train, X_eval, _ = _prepare_candidate_pair(
        spec,
        matrices,
        names,
        train_idx,
        eval_idx,
        protocol,
        random_state,
        _count_transformation_factory,
        _lodo_feature_pair,
    )
    estimator = configure_estimator_threads(learner_factory(), threads_per_worker)
    estimator.fit(X_train, y[train_idx])
    pred = np.asarray(_estimator_call(estimator, "predict", X_eval), dtype=float)
    return pred, compute_regression_metrics(y[eval_idx], pred)


def _regression_success_row(
    spec: CandidateSpec,
    mpdr: MPDR,
    metrics: Mapping[str, float],
    split_key: str,
    inner_key: str | None,
    stage_name: str,
) -> dict[str, Any]:
    return _meta_row(
        _regression_metric_row(
            dict(metrics),
            split_key,
            inner_key,
            spec.config_id,
            mpdr,
            spec.learner,
            stage_name,
        ),
        spec,
    )


def _regression_failure_row(
    spec: CandidateSpec,
    mpdr: MPDR,
    split_key: str,
    inner_key: str | None,
    stage_name: str,
    exc: Exception,
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    if stage_name == "inner":
        metrics = {
            key: float("nan")
            for key in compute_regression_metrics(
                np.array([0.0, 1.0]), np.array([0.0, 1.0])
            )
        }
    row = _regression_metric_row(
        metrics,
        split_key,
        inner_key,
        spec.config_id,
        mpdr,
        spec.learner,
        stage_name,
    )
    row["ok"] = 0
    row["error"] = f"{type(exc).__name__}: {exc}"
    return _meta_row(row, spec)


def _evaluate_regression_inner_folds(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    dataset: ModalityDataset,
    train_idx: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    split_key: str,
    protocol: str,
    learner_factory: _LearnerFactory,
    mpdr: MPDR,
    gate_metric: str,
    existing_inner_keys: set[str],
    score_rows: list[dict[str, Any]],
    random_state: int,
    threads_per_worker: int,
    result: dict[str, Any],
    progress_queue: Any,
    completed_steps: int,
    total_steps: int,
) -> int:
    y = np.asarray(dataset.y, dtype=float)
    for inner_no, (train_local, validation_local) in enumerate(inner_splits):
        inner_key = f"{split_key}__i{inner_no}"
        if inner_key in existing_inner_keys:
            continue
        _emit_evaluation_progress(
            progress_queue,
            split_key,
            spec.config_id,
            "inner",
            completed_steps,
            total_steps,
            f"inner fold {inner_no + 1}/{len(inner_splits)}",
        )
        fold_train_idx = train_idx[np.asarray(train_local, dtype=int)]
        validation_idx = train_idx[np.asarray(validation_local, dtype=int)]
        try:
            pred, metrics = _fit_regression_candidate_fold(
                spec,
                matrices,
                names,
                y,
                fold_train_idx,
                validation_idx,
                protocol,
                learner_factory,
                random_state,
                threads_per_worker,
            )
            row = _regression_success_row(
                spec, mpdr, metrics, split_key, inner_key, "inner"
            )
            result["inner_metrics"].append(row)
            result["inner_predictions"].extend(
                _regression_prediction_rows(
                    inner_key,
                    spec.config_id,
                    validation_idx,
                    dataset.sample_ids,
                    dataset.subject_ids,
                    y,
                    pred,
                    "inner",
                    split_key,
                    dataset.target_name,
                )
            )
            if np.isfinite(float(metrics.get(gate_metric, np.nan))):
                score_rows.append(dict(row))
            result["fits"] += 1
        except Exception as exc:
            result["inner_metrics"].append(
                _regression_failure_row(spec, mpdr, split_key, inner_key, "inner", exc)
            )
        completed_steps += 1
        _emit_evaluation_progress(
            progress_queue,
            split_key,
            spec.config_id,
            "inner",
            completed_steps,
            total_steps,
            f"inner fold {inner_no + 1}/{len(inner_splits)} complete",
        )
    return completed_steps


def _regression_qualification(
    gate: QualificationGate,
    gate_metric: str,
    score_rows: Sequence[Mapping[str, Any]],
    existing_qualification: Mapping[str, Any] | None,
) -> bool:
    if not gate.enabled:
        return True
    if existing_qualification is not None:
        return bool(int(existing_qualification.get("qualified", 0)))
    gate_score, _ = aggregate_validation_metric(pd.DataFrame(score_rows), gate_metric)
    return bool(
        QualificationGate(True, gate.metric, gate.threshold).qualifies(gate_score)
    )


def _evaluate_regression_outer_fold(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    dataset: ModalityDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_key: str,
    protocol: str,
    learner_factory: _LearnerFactory,
    mpdr: MPDR,
    random_state: int,
    threads_per_worker: int,
    result: dict[str, Any],
) -> None:
    y = np.asarray(dataset.y, dtype=float)
    try:
        pred, metrics = _fit_regression_candidate_fold(
            spec,
            matrices,
            names,
            y,
            train_idx,
            test_idx,
            protocol,
            learner_factory,
            random_state,
            threads_per_worker,
        )
        result["outer_metrics"].append(
            _regression_success_row(spec, mpdr, metrics, split_key, None, "outer")
        )
        result["outer_predictions"].extend(
            _regression_prediction_rows(
                split_key,
                spec.config_id,
                test_idx,
                dataset.sample_ids,
                dataset.subject_ids,
                y,
                pred,
                "outer",
                split_key,
                dataset.target_name,
            )
        )
        result["fits"] += 1
    except Exception as exc:
        result["outer_metrics"].append(
            _regression_failure_row(spec, mpdr, split_key, None, "outer", exc)
        )


def _regression_task(
    spec: CandidateSpec,
    matrices: _RepresentationMatrices,
    names: _RepresentationNames,
    dataset: ModalityDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    split_key: str,
    protocol: str,
    learner_factory: _LearnerFactory,
    gate: QualificationGate,
    existing_inner_keys: set[str],
    existing_inner_scores: Sequence[Mapping[str, Any]],
    existing_qualification: Mapping[str, Any] | None,
    needs_outer: bool,
    random_state: int,
    threads_per_worker: int,
    resource_sample_interval_s: float,
    progress_queue: Any = None,
) -> dict[str, Any]:
    tracker = ResourceTracker(sample_interval_s=resource_sample_interval_s).start()
    mpdr = MPDR(
        spec.representation_label, tuple(spec.modalities), spec.transformation_label
    )
    result = _new_modality_result(spec, split_key)
    gate_metric = canonical_metric_name(str(gate.metric))
    score_rows = [
        dict(row) for row in existing_inner_scores if isinstance(row, Mapping)
    ]
    pending_inner = sum(
        1
        for inner_no in range(len(inner_splits))
        if f"{split_key}__i{inner_no}" not in existing_inner_keys
    )
    total_steps = max(1, pending_inner + int(bool(needs_outer)))
    completed_steps = 0
    _emit_evaluation_progress(
        progress_queue,
        split_key,
        spec.config_id,
        "start",
        completed_steps,
        total_steps,
        "starting candidate",
    )
    with (
        thread_environment(threads_per_worker),
        threadpool_limits(limits=max(1, int(threads_per_worker))),
    ):
        completed_steps = _evaluate_regression_inner_folds(
            spec,
            matrices,
            names,
            dataset,
            train_idx,
            inner_splits,
            split_key,
            protocol,
            learner_factory,
            mpdr,
            gate_metric,
            existing_inner_keys,
            score_rows,
            random_state,
            threads_per_worker,
            result,
            progress_queue,
            completed_steps,
            total_steps,
        )
        qualified = _regression_qualification(
            gate, gate_metric, score_rows, existing_qualification
        )
        if needs_outer and qualified:
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                "outer refit and evaluation",
            )
            _evaluate_regression_outer_fold(
                spec,
                matrices,
                names,
                dataset,
                train_idx,
                test_idx,
                split_key,
                protocol,
                learner_factory,
                mpdr,
                random_state,
                threads_per_worker,
                result,
            )
            completed_steps += 1
            detail = "outer evaluation complete"
        elif needs_outer:
            completed_steps += 1
            detail = "outer evaluation skipped by qualification gate"
        else:
            detail = ""
        if needs_outer:
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                detail,
            )
    return _finalize_modality_candidate_result(
        spec,
        split_key,
        threads_per_worker,
        tracker,
        result,
        progress_queue,
        total_steps,
    )


def _prepare_modality_sweep_context(sweep: Sweep) -> _ModalitySweepContext:
    samples = sweep.samples
    if samples is None:
        raise ValueError("Modality evaluation requires Sweep.samples.")
    dataset = load_modalities(samples, sweep.modalities)
    if dataset.task == "classification":
        validate_sweep_class_count(sweep, len(dataset.class_labels))
    root = sweep.root()
    _prepare_dirs(root)
    learner_specs = [
        _learner_factory(item, task=dataset.task) for item in sweep.learners
    ]
    learner_lookup = {name: factory for name, factory in learner_specs}
    integrations = tuple(sweep.integrations or (Integration("unimodal"),))
    matrices, names, feature_blocks_by_path = _representation_cache(
        dataset, sweep.representations
    )
    transformations = _normalise_modality_transformations(
        tuple(dataset.modalities), dataset.primary_modality, sweep.transformations
    )
    specs, configs = build_modality_candidates(
        tuple(dataset.modalities),
        sweep.representations,
        transformations,
        integrations,
        sweep.learners,
        _learner_name,
        feature_blocks_by_path=feature_blocks_by_path,
    )
    modality_hashes = modality_fingerprints(dataset)
    source_hashes = modality_source_fingerprints(samples, sweep.modalities)
    dataset_hash = modality_dataset_fingerprint(dataset)
    plan = sweep.evaluation
    configured_groups = _groups_from_metadata(dataset.metadata, samples.group_col)
    effective_groups, effective_group_col = _subject_safe_groups(
        plan, dataset, configured_groups, samples.group_col
    )
    fingerprint_strata = (
        _strata_from_metadata(dataset.metadata, dataset.y, samples.stratify_col)
        if dataset.task == "classification"
        else None
    )
    evaluation_fingerprint = _scientific_digest(
        {
            "schema": "multimodal-evaluation-fingerprint-v3",
            "source_tree_sha256": _source_tree_sha256(),
            "package_version": __version__,
            "model_runtime_dependencies": _software_provenance()["dependencies"],
            "dataset_fingerprint": dataset_hash,
            "modality_fingerprints": modality_hashes,
            "config_ids": sorted(configs["config_id"].astype(str).tolist()),
            "task": dataset.task,
            "target": dataset.target_name,
            "evaluation": {
                "protocol": str(plan.protocol),
                "outer_folds": int(plan.outer_folds),
                "inner_folds": int(plan.inner_folds),
                "repeats": int(plan.repeats),
                "random_state": int(plan.random_state),
                "optimize_metric": _scientific_value(plan.optimize_metric),
            },
            "group_col": samples.group_col,
            "effective_group_col": effective_group_col,
            "group_assignments": (
                None
                if effective_groups is None
                else _scientific_digest(
                    np.asarray(effective_groups, dtype=object).astype(str).tolist()
                )
            ),
            "subject_id_policy": "auto_group_repeated_subjects",
            "stratify_col": _scientific_value(samples.stratify_col),
            "strata_assignments": (
                None
                if fingerprint_strata is None
                else _scientific_digest(
                    np.asarray(fingerprint_strata, dtype=object).astype(str).tolist()
                )
            ),
            "gate": _scientific_value(sweep.gate),
            "multimodal_inclusion": _scientific_value(dataset.inclusion_report),
        }
    )
    _validate_experiment_identity(
        root, evaluation_fingerprint, redo=bool(sweep.evaluation.redo)
    )
    if sweep.evaluation.redo:
        _clear_evaluation_checkpoints(root)
    _write_config_table(root, configs)
    manifest = {
        "package": "mllabiome",
        "package_version": __version__,
        "software_provenance": _software_provenance(),
        "title": sweep.title,
        "version": "modality_sweep_v4",
        "primary_modality": dataset.primary_modality,
        "sample_alignment": "explicit_complete_case_intersection",
        "multimodal_inclusion": dataset.inclusion_report,
        "n_samples": len(dataset.sample_ids),
        "cv_splits": "tables/cv_splits.parquet",
        "dataset_fingerprint": dataset_hash,
        "dataset_fingerprint_algorithm": "sha256-multimodal-model-input-v1",
        "modality_fingerprints": modality_hashes,
        "source_file_sha256": source_hashes,
        "evaluation_fingerprint": evaluation_fingerprint,
        "evaluation_fingerprint_algorithm": "sha256-scientific-multimodal-evaluation-v3",
        "experiment_fingerprint": _experiment_fingerprint(
            sweep, evaluation_fingerprint
        ),
        "experiment_fingerprint_algorithm": "sha256-scientific-experiment-v1",
        "modalities": {
            name: {
                "n_features": int(modality.X.shape[1]),
                "format": modality.format,
                "fingerprint": modality_hashes[name],
            }
            for name, modality in dataset.modalities.items()
        },
        "integrations": [integration.key for integration in integrations],
    }
    dump_json_standard(manifest, root / "manifest_modalities.json")
    dump_json_standard(
        {
            **manifest,
            "class_labels": dataset.class_labels,
            "target": dataset.target_name,
            "task": dataset.task,
        },
        root / "manifest.json",
    )
    groups = _groups_from_metadata(dataset.metadata, samples.group_col)
    strata = (
        _strata_from_metadata(dataset.metadata, dataset.y, samples.stratify_col)
        if dataset.task == "classification"
        else None
    )
    outer_splits, inner_splits_by_outer = _resolved_evaluation_splits(
        root,
        sweep.evaluation,
        dataset,
        groups,
        strata,
        samples.stratify_col,
        samples.group_col,
    )
    current_ids = set(configs["config_id"].astype(str))
    outer_keys = {str(split["split_key"]) for split in outer_splits}
    existing = _load_existing_evaluation(
        root, current_ids, outer_keys, redo=sweep.evaluation.redo
    )
    if dataset.task == "classification":
        existing["inner_metrics"] = _backfill_metrics_from_predictions(
            existing["inner_metrics"],
            existing["inner_predictions"],
            dataset.classes,
            dataset.class_labels,
            "inner_key",
        )
        existing["outer_metrics"] = _backfill_metrics_from_predictions(
            existing["outer_metrics"],
            existing["outer_predictions"],
            dataset.classes,
            dataset.class_labels,
            "split_key",
        )
    else:
        existing["inner_metrics"] = _backfill_regression_metrics_from_predictions(
            existing["inner_metrics"], existing["inner_predictions"], "inner_key"
        )
        existing["outer_metrics"] = _backfill_regression_metrics_from_predictions(
            existing["outer_metrics"], existing["outer_predictions"], "split_key"
        )
    return _ModalitySweepContext(
        root=root,
        dataset=dataset,
        learners=learner_lookup,
        integrations=integrations,
        matrices=matrices,
        names=names,
        specs=specs,
        configs=configs,
        groups=groups,
        outer_splits=outer_splits,
        inner_splits_by_outer=inner_splits_by_outer,
        existing=existing,
        inner_done=_done_pairs(existing["inner_metrics"], "inner_key"),
        outer_done=_done_pairs(existing["outer_metrics"], "split_key"),
        qualification_map=(
            _qualification_map(existing["qualification"]) if sweep.gate.enabled else {}
        ),
    )


def _build_modality_tasks(
    sweep: Sweep, context: _ModalitySweepContext
) -> tuple[list[_ScheduledTask], dict[str, int]]:
    tasks: list[_ScheduledTask] = []
    split_counts: dict[str, int] = {}
    dataset = context.dataset
    for split in context.outer_splits:
        split_key = str(split["split_key"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if dataset.task == "classification" and (
            len(np.unique(dataset.y[train_idx])) < 2 or len(test_idx) == 0
        ):
            continue
        inner_splits = context.inner_splits_by_outer.get(split_key, [])
        inner_keys = [
            f"{split_key}__i{inner_no}" for inner_no in range(len(inner_splits))
        ]
        for spec in context.specs:
            missing_inner = {
                key
                for key in inner_keys
                if (key, spec.config_id) not in context.inner_done
            }
            needs_outer = not _outer_pair_complete(
                split_key,
                spec.config_id,
                context.outer_done,
                context.qualification_map,
            )
            if not missing_inner and not needs_outer:
                continue
            completed_inner = set(inner_keys) - missing_inner
            existing_scores = _existing_inner_scores(
                context.existing["inner_metrics"],
                split_key,
                spec.config_id,
                sweep.gate.metric,
            )
            common = (spec, context.matrices, context.names)
            if dataset.task == "classification":
                task = delayed(_classification_task)(
                    *common,
                    dataset.y,
                    context.groups,
                    dataset.classes,
                    tuple(dataset.class_labels),
                    tuple(dataset.sample_ids),
                    tuple(dataset.subject_ids),
                    train_idx,
                    test_idx,
                    tuple(inner_splits),
                    split_key,
                    sweep.evaluation.protocol,
                    context.learners[spec.learner],
                    sweep.gate,
                    sweep.evaluation.optimize_metric,
                    completed_inner,
                    existing_scores,
                    context.qualification_map.get((split_key, spec.config_id)),
                    needs_outer,
                    int(sweep.evaluation.random_state),
                    1,
                    float(sweep.evaluation.resource_sample_interval_s),
                )
            else:
                task = delayed(_regression_task)(
                    *common,
                    dataset,
                    train_idx,
                    test_idx,
                    tuple(inner_splits),
                    split_key,
                    sweep.evaluation.protocol,
                    context.learners[spec.learner],
                    sweep.gate,
                    completed_inner,
                    existing_scores,
                    context.qualification_map.get((split_key, spec.config_id)),
                    needs_outer,
                    int(sweep.evaluation.random_state),
                    1,
                    float(sweep.evaluation.resource_sample_interval_s),
                )
            tasks.append((split_key, spec.config_id, task))
            split_counts[split_key] = split_counts.get(split_key, 0) + 1
    return tasks, split_counts


def _prepare_modality_tasks(
    tasks: Sequence[_ScheduledTask], threads_per_worker: int
) -> list[_PreparedModalityTask]:
    prepared: list[_PreparedModalityTask] = []
    for split_key, config_id, task in tasks:
        fn, args, kwargs = task
        call_args = list(args)
        call_args[-2] = int(threads_per_worker)
        prepared.append(
            _PreparedModalityTask(
                split_key=str(split_key),
                config_id=str(config_id),
                fn=fn,
                args=tuple(call_args),
                kwargs=dict(kwargs),
            )
        )
    return prepared


def _show_modality_sweep_overview(
    sweep: Sweep,
    context: _ModalitySweepContext,
    execution: ExecutionPlan,
    pending_jobs: int,
) -> None:
    stage("Modality configuration sweep", sweep.title)
    summary_table(
        "Sweep overview",
        {
            "samples": len(context.dataset.sample_ids),
            "primary modality": context.dataset.primary_modality,
            "modalities": ", ".join(context.dataset.modalities),
            "candidates": len(context.specs),
            "protocol": sweep.evaluation.protocol,
            "outer splits": len(context.outer_splits),
            "pending jobs": pending_jobs,
            "workers": execution.workers,
            "threads per worker": execution.threads_per_worker,
            "experiment dir": context.root,
        },
    )


def _monitor_modality_progress(
    progress_queue: Any,
    stop_event: Event,
    prog: Any,
    split_tasks: Mapping[str, Any],
    split_counts: Mapping[str, int],
    completed_by_split: Mapping[str, int],
    spec_lookup: Mapping[str, CandidateSpec],
    candidate_order: Mapping[str, int],
    candidate_count: int,
) -> None:
    while not stop_event.is_set():
        try:
            (
                split_key,
                config_id,
                phase,
                step_done,
                step_total,
                detail,
            ) = progress_queue.get(timeout=0.25)
        except Empty:
            continue
        split_key = str(split_key)
        config_id = str(config_id)
        spec = spec_lookup.get(config_id)
        if spec is None or split_key not in split_tasks:
            continue
        ordinal = candidate_order.get(config_id, 0)
        done_jobs = completed_by_split.get(split_key, 0)
        description = (
            f"{split_key} · {done_jobs}/{split_counts.get(split_key, 0)} jobs · "
            f"candidate {ordinal}/{candidate_count} · {spec.integration.key} · "
            f"{spec.learner} · {phase} {int(step_done)}/{int(step_total)} · {detail}"
        )
        prog.update(split_tasks[split_key], description=description)


def _run_modality_tasks(
    context: _ModalitySweepContext,
    tasks: Sequence[_PreparedModalityTask],
    split_counts: Mapping[str, int],
    execution: ExecutionPlan,
) -> _ModalityEvaluationRows:
    rows = _ModalityEvaluationRows.empty()
    if not tasks:
        return rows
    spec_lookup = {spec.config_id: spec for spec in context.specs}
    candidate_order = {
        spec.config_id: index + 1 for index, spec in enumerate(context.specs)
    }
    completed_by_split = {str(key): 0 for key in split_counts}
    completed_jobs = 0
    completed_fits = 0
    failed_fits = 0
    stop_monitor = Event()
    with Manager() as manager:
        progress_queue = manager.Queue()
        with progress() as prog:
            overall_task = prog.add_task(
                f"Evaluation · {execution.workers} workers · 0 fits · 0 failures",
                total=len(tasks),
            )
            split_tasks = {
                str(split_key): prog.add_task(
                    f"{split_key} · queued · 0/{count} jobs",
                    total=max(1, int(count)),
                )
                for split_key, count in split_counts.items()
            }
            monitor_thread = Thread(
                target=_monitor_modality_progress,
                args=(
                    progress_queue,
                    stop_monitor,
                    prog,
                    split_tasks,
                    split_counts,
                    completed_by_split,
                    spec_lookup,
                    candidate_order,
                    len(context.specs),
                ),
                daemon=True,
            )
            monitor_thread.start()
            try:
                payloads = [
                    (
                        task.fn,
                        task.args,
                        {**task.kwargs, "progress_queue": progress_queue},
                    )
                    for task in tasks
                ]
                for result in iter_parallel_tasks(payloads, execution):
                    _checkpoint_result(context.root, result)
                    rows.extend(result)
                    completed_jobs += 1
                    completed_fits += int(result.get("fits", 0))
                    failed_fits += _result_failure_count(result)
                    split_key = str(result.get("split_key", ""))
                    config_id = str(result.get("config_id", ""))
                    completed_by_split[split_key] = (
                        completed_by_split.get(split_key, 0) + 1
                    )
                    spec = spec_lookup.get(config_id)
                    if split_key in split_tasks:
                        label = "completed"
                        if spec is not None:
                            label = (
                                f"completed · {spec.integration.key} · {spec.learner}"
                            )
                        prog.update(
                            split_tasks[split_key],
                            completed=completed_by_split[split_key],
                            description=(
                                f"{split_key} · {completed_by_split[split_key]}/"
                                f"{split_counts.get(split_key, 0)} jobs · {label} · "
                                f"last {float(result.get('elapsed_s', 0.0)):.1f}s"
                            ),
                        )
                    prog.update(
                        overall_task,
                        completed=completed_jobs,
                        description=(
                            f"Evaluation · {execution.workers} workers · "
                            f"{completed_fits:,} fits · {failed_fits:,} failures · "
                            "checkpoints current"
                        ),
                    )
            finally:
                stop_monitor.set()
                monitor_thread.join(timeout=2.0)
            prog.update(
                overall_task,
                completed=len(tasks),
                description=(
                    f"Evaluation · complete · {completed_fits:,} fits · "
                    f"{failed_fits:,} failures · checkpoints current"
                ),
            )
    return rows


def _finalize_modality_sweep(
    sweep: Sweep,
    context: _ModalitySweepContext,
    rows: _ModalityEvaluationRows,
    execution: ExecutionPlan,
    elapsed_s: float,
) -> dict[str, Path]:
    _write_tables(
        context.root,
        rows.outer_metrics,
        rows.inner_metrics,
        rows.outer_predictions,
        rows.inner_predictions,
        rows.qualification,
        rows.job_resources,
        existing=context.existing,
        gate_enabled=sweep.gate.enabled,
    )
    write_mpma_b_selection_outputs(
        context.root, sweep.evaluation.optimize_metric, plan=sweep.ensemble
    )
    _write_rankings_and_figures(
        context.root,
        context.dataset.class_labels,
        sweep.evaluation.optimize_metric,
    )
    _write_representation_impact_figure(
        context.root, metric_col=sweep.evaluation.optimize_metric
    )
    dump_json_standard(
        {
            "elapsed_s": float(elapsed_s),
            "workers": execution.workers,
            "threads_per_worker": execution.threads_per_worker,
            "machine": machine_profile(execution.logical_cpus, execution.physical_cpus),
        },
        context.root / "run_summary.json",
    )
    success("Modality configuration sweep completed")
    outputs = _existing_outputs(context.root)
    outputs["modality_manifest"] = context.root / "manifest_modalities.json"
    path_table("Configuration sweep outputs", outputs)
    return outputs


def evaluate_modality_sweep(sweep: Sweep) -> dict[str, Path]:
    context = _prepare_modality_sweep_context(sweep)
    tasks, split_counts = _build_modality_tasks(sweep, context)
    execution = resolve_execution_plan(
        sweep.evaluation.n_jobs,
        len(tasks) or 1,
        backend=sweep.evaluation.parallel_backend,
        memory_fraction=sweep.evaluation.memory_fraction,
        min_worker_memory_gib=sweep.evaluation.min_worker_memory_gib,
    )
    prepared = _prepare_modality_tasks(tasks, execution.threads_per_worker)
    _show_modality_sweep_overview(sweep, context, execution, len(prepared))
    started = time.perf_counter()
    rows = _run_modality_tasks(context, prepared, split_counts, execution)
    return _finalize_modality_sweep(
        sweep,
        context,
        rows,
        execution,
        time.perf_counter() - started,
    )


def candidate_from_row(
    sweep, row, dataset: ModalityDataset | None = None
) -> tuple[ModalityDataset, CandidateSpec, dict[Any, np.ndarray], dict[Any, Any]]:
    from .configs_sweep import _learner_name

    dataset = (
        load_modalities(sweep.samples, sweep.modalities) if dataset is None else dataset
    )
    matrices, names, feature_blocks_by_path = _representation_cache(
        dataset, sweep.representations
    )
    transformations = _normalise_modality_transformations(
        tuple(dataset.modalities), dataset.primary_modality, sweep.transformations
    )
    specs, _ = build_modality_candidates(
        tuple(dataset.modalities),
        sweep.representations,
        transformations,
        tuple(sweep.integrations or (Integration("unimodal"),)),
        sweep.learners,
        _learner_name,
        feature_blocks_by_path=feature_blocks_by_path,
    )
    config_id = str(row["config_id"])
    matches = [spec for spec in specs if spec.config_id == config_id]
    if len(matches) != 1:
        raise ValueError(
            f"Config {config_id!r} cannot be reconstructed uniquely from the configured modality search space."
        )
    spec = matches[0]
    return dataset, spec, matrices, names


def fit_modality_candidate_oof_for_explainability(sweep, row):
    from .configs_sweep import (
        _learner_factory,
        _predict_proba_aligned,
        _resolved_evaluation_splits,
        _strata_from_metadata,
    )
    from .explainability import (
        ExplainabilityConfigurationError,
        _xai_execution_plan,
        _xai_task_iterator,
    )

    dataset, spec, matrices, names = candidate_from_row(sweep, row)
    groups = (
        None
        if sweep.samples.group_col is None
        else dataset.metadata[sweep.samples.group_col].astype(str).to_numpy()
    )
    strata = _strata_from_metadata(
        dataset.metadata, dataset.y, sweep.samples.stratify_col
    )
    splits, _ = _resolved_evaluation_splits(
        sweep.root(),
        sweep.evaluation,
        dataset,
        groups,
        strata,
        sweep.samples.stratify_col,
        sweep.samples.group_col,
    )
    learner_lookup = dict(_learner_factory(item) for item in sweep.learners)
    learner_factory = learner_lookup[spec.learner]
    execution = _xai_execution_plan(sweep, len(splits))

    def fit_fold(split_no, split, threads_per_worker):
        from .configs_sweep import _count_transformation_factory, _lodo_feature_pair

        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
            return int(split_no), None
        details = _prepare_candidate_pair_details(
            spec,
            matrices,
            names,
            train_idx,
            test_idx,
            sweep.evaluation.protocol,
            int(sweep.explainability.random_state),
            _count_transformation_factory,
            _lodo_feature_pair,
        )
        clf = configure_estimator_threads(learner_factory(), threads_per_worker)
        fit_classifier(
            clf,
            details["X_train"],
            dataset.y[train_idx],
            None if groups is None else groups[train_idx],
        )
        direct_proba = _predict_proba_aligned(clf, details["X_test"], dataset.classes)
        if spec.integration.stage == "intermediate":
            estimator = _IntegratedClassificationPredictor(
                clf, details["integration_model"], details["modality_slices"]
            )
            X_train = details["X_train_source"]
            X_test = details["X_test_source"]
            coords = details["source_coordinates"]
            proba = _predict_proba_aligned(estimator, X_test, dataset.classes)
            if not np.allclose(proba, direct_proba, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    "Input-space classification explanation wrapper does not reproduce the fitted integrated model probabilities."
                )
        else:
            estimator = clf
            X_train = details["X_train"]
            X_test = details["X_test"]
            coords = details["coordinates"]
            proba = direct_proba
        projector = _ModalityInputProjector(
            details["transformation_models"], details["modality_slices"]
        )
        return int(split_no), {
            "split_key": str(split["split_key"]),
            "train_idx": train_idx,
            "test_idx": test_idx,
            "X_train": X_train,
            "X_test": X_test,
            "y_test": dataset.y[test_idx],
            "estimator": estimator,
            "proba": proba,
            "coordinate_metadata": coords,
            "explanation_space": "pre_integration"
            if spec.integration.stage == "intermediate"
            else "model_coordinates",
            "integration": spec.integration.key,
            "input_projector": projector if projector.requires_projection() else None,
            "perturbation_geometry": projector.geometry(),
        }

    tasks = [
        (fit_fold, (i, split, int(execution.threads_per_worker)), {})
        for i, split in enumerate(splits, start=1)
    ]
    folds_by_no = {}
    for split_no, fold in _xai_task_iterator(tasks, execution):
        if fold is not None:
            folds_by_no[int(split_no)] = fold
    folds = [folds_by_no[i] for i in sorted(folds_by_no)]
    if not folds:
        raise ExplainabilityConfigurationError(
            "No outer fold could be fitted for modality-based out-of-fold explainability."
        )
    first_meta = list(folds[0]["coordinate_metadata"])
    first_names = [str(item.name) for item in first_meta]
    for fold in folds[1:]:
        current = [str(item.name) for item in fold["coordinate_metadata"]]
        if current != first_names:
            raise ExplainabilityConfigurationError(
                "Modality-based OOF explainability produced different model coordinates across outer folds. This commonly occurs when LODO training-domain feature filtering changes the feature universe. The framework will not aggregate non-identical coordinates."
            )
    reference = np.zeros((len(dataset.sample_ids), len(first_names)), dtype=float)
    seen = np.zeros(len(dataset.sample_ids), dtype=bool)
    for fold in folds:
        idx = np.asarray(fold["test_idx"], dtype=int)
        if not np.any(seen[idx]):
            reference[idx] = np.asarray(fold["X_test"], dtype=float)
            seen[idx] = True
    return {
        "dataset": dataset,
        "X_base": reference,
        "feature_names": first_names,
        "coordinate_metadata": first_meta,
        "folds": folds,
        "execution": execution,
    }


def fit_modality_regression_candidate_folds(sweep, row, progress_callback=None):
    from .configs_sweep import (
        _count_transformation_factory,
        _learner_factory,
        _lodo_feature_pair,
        _resolved_evaluation_splits,
    )

    dataset, spec, matrices, names = candidate_from_row(sweep, row)
    groups = (
        None
        if sweep.samples.group_col is None
        else dataset.metadata[sweep.samples.group_col].astype(str).to_numpy()
    )
    splits, _ = _resolved_evaluation_splits(
        sweep.root(),
        sweep.evaluation,
        dataset,
        groups,
        group_col=sweep.samples.group_col,
    )
    learner_lookup = dict(
        _learner_factory(item, task="regression") for item in sweep.learners
    )
    folds = []
    for fold_no, split in enumerate(splits, start=1):
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        details = _prepare_candidate_pair_details(
            spec,
            matrices,
            names,
            train_idx,
            test_idx,
            sweep.evaluation.protocol,
            int(sweep.explainability.random_state),
            _count_transformation_factory,
            _lodo_feature_pair,
        )
        model = configure_estimator_threads(learner_lookup[spec.learner](), 1)
        model.fit(details["X_train"], dataset.y[train_idx])
        direct_pred = np.asarray(
            _estimator_call(model, "predict", details["X_test"]), dtype=float
        ).reshape(-1)
        if spec.integration.stage == "intermediate":
            estimator = _IntegratedRegressionPredictor(
                model, details["integration_model"], details["modality_slices"]
            )
            X_train = details["X_train_source"]
            X_test = details["X_test_source"]
            coords = details["source_coordinates"]
            pred = np.asarray(
                _estimator_call(estimator, "predict", X_test), dtype=float
            ).reshape(-1)
            if not np.allclose(pred, direct_pred, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    "Input-space regression explanation wrapper does not reproduce the fitted integrated model predictions."
                )
        else:
            estimator = model
            X_train = details["X_train"]
            X_test = details["X_test"]
            coords = details["coordinates"]
            pred = direct_pred
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": np.asarray(X_train, dtype=float),
                "X_test": np.asarray(X_test, dtype=float),
                "feature_names": [str(item.name) for item in coords],
                "coordinate_metadata_objects": coords,
                "y_test": np.asarray(dataset.y[test_idx], dtype=float),
                "y_pred": pred,
                "estimator": estimator,
                "explanation_space": "pre_integration"
                if spec.integration.stage == "intermediate"
                else "model_coordinates",
                "integration": spec.integration.key,
                "input_projector": _ModalityInputProjector(
                    details["transformation_models"], details["modality_slices"]
                ),
                "perturbation_geometry": _ModalityInputProjector(
                    details["transformation_models"], details["modality_slices"]
                ).geometry(),
            }
        )
        if progress_callback is not None:
            progress_callback(fold_no, len(splits), str(split["split_key"]))
    return dataset, folds
