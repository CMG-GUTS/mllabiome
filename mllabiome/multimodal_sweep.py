from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import delayed
from threadpoolctl import threadpool_limits

from ._version import __version__
from .compute import ResourceTracker, machine_profile
from .console import path_table, progress, stage, success, summary_table
from .integrations import Integration, IntegrationModel, integration_modality_sets
from .learners import fit_classifier
from .metrics import _estimator_call, compute_metrics, compute_regression_metrics
from .modalities import (
    ModalityDataset,
    load_modalities,
    modality_dataset_fingerprint,
    modality_fingerprints,
    modality_source_fingerprints,
)
from .resolutions import mask_feature_blocks, materialize_mpdr_with_blocks
from .runtime import (
    configure_estimator_threads,
    iter_parallel_tasks,
    resolve_execution_plan,
    thread_environment,
)
from .transformations import _count_transformation_specs_for_blocks
from .utils import dump_json_standard


from .multimodal_candidates import (
    CandidateSpec,
    ModalityPath,
    _IntegratedClassificationPredictor,
    _IntegratedRegressionPredictor,
    _ModalityInputProjector,
    _canonical,
    _candidate_id,
    _materialize_modality_representation,
    _modality_paths,
    _normalise_modality_transformations,
    _prepare_candidate_pair,
    _prepare_candidate_pair_details,
    _representation_cache,
    _representation_specs,
    _transformation_specs,
    build_modality_candidates,
)


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


def _classification_task(
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
    inner_splits,
    split_key,
    protocol,
    learner_factory,
    gate,
    existing_inner_keys,
    existing_inner_scores,
    existing_qualification,
    needs_outer,
    random_state,
    threads_per_worker,
    resource_sample_interval_s,
    progress_queue=None,
):
    from .configs_sweep import (
        MPDR,
        QualificationGate,
        _count_transformation_factory,
        _failed_metric_row,
        _lodo_feature_pair,
        _metric_row,
        _predict_proba_aligned,
        _prediction_rows_values,
    )

    tracker = ResourceTracker(sample_interval_s=resource_sample_interval_s).start()
    mpdr = MPDR(
        spec.representation_label, tuple(spec.modalities), spec.transformation_label
    )
    result = {
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
    scores = [float(x) for x in existing_inner_scores if np.isfinite(x)]
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
        for inner_no, (tr_local, va_local) in enumerate(inner_splits):
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
            tr_idx = train_idx[np.asarray(tr_local, dtype=int)]
            va_idx = train_idx[np.asarray(va_local, dtype=int)]
            if len(np.unique(y[tr_idx])) < 2 or len(va_idx) == 0:
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
                Xtr, Xva, _ = _prepare_candidate_pair(
                    spec,
                    matrices,
                    names,
                    tr_idx,
                    va_idx,
                    protocol,
                    random_state,
                    _count_transformation_factory,
                    _lodo_feature_pair,
                )
                clf = configure_estimator_threads(learner_factory(), threads_per_worker)
                fit_classifier(
                    clf, Xtr, y[tr_idx], None if groups is None else groups[tr_idx]
                )
                proba = _predict_proba_aligned(clf, Xva, classes)
                pred = classes[proba.argmax(axis=1)]
                metrics = compute_metrics(y[va_idx], pred, proba, classes)
                row = _meta_row(
                    _metric_row(
                        metrics,
                        split_key,
                        inner_key,
                        spec.config_id,
                        mpdr,
                        spec.learner,
                        "inner",
                    ),
                    spec,
                )
                prediction_rows = _prediction_rows_values(
                    inner_key,
                    spec.config_id,
                    va_idx,
                    sample_ids,
                    subject_ids,
                    y,
                    class_labels,
                    pred,
                    proba,
                    "inner",
                    split_key,
                )
                result["inner_metrics"].append(row)
                result["inner_predictions"].extend(prediction_rows)
                score = float(metrics.get(gate.metric, np.nan))
                if np.isfinite(score):
                    scores.append(score)
                result["fits"] += 1
            except Exception as exc:
                result["inner_metrics"].append(
                    _meta_row(
                        _failed_metric_row(
                            split_key,
                            inner_key,
                            spec.config_id,
                            mpdr,
                            spec.learner,
                            "inner",
                            exc,
                        ),
                        spec,
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
        qualified = True
        gate_score = float("nan")
        if gate.enabled:
            if existing_qualification is not None:
                qualified = bool(int(existing_qualification.get("qualified", 0)))
                gate_score = float(existing_qualification.get("inner_score", np.nan))
            else:
                gate_score = float(np.mean(scores)) if scores else float("nan")
                qualified = QualificationGate(
                    True, gate.metric, gate.threshold
                ).qualifies(gate_score)
                result["qualification"].append(
                    _meta_row(
                        {
                            "split_key": split_key,
                            "config_id": spec.config_id,
                            "mpdr_id": hashlib.sha1(
                                (
                                    spec.representation_label
                                    + spec.transformation_label
                                ).encode()
                            ).hexdigest()[:12],
                            "count_transformation": spec.transformation_label,
                            "resolution": spec.representation_label,
                            "levels": ",".join(spec.modalities),
                            "learner": spec.learner,
                            "gate_enabled": 1,
                            "gate_metric": gate.metric,
                            "gate_threshold": gate.threshold,
                            "inner_score": gate_score,
                            "qualified": int(qualified),
                        },
                        spec,
                    )
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
            try:
                Xtr, Xte, _ = _prepare_candidate_pair(
                    spec,
                    matrices,
                    names,
                    train_idx,
                    test_idx,
                    protocol,
                    random_state,
                    _count_transformation_factory,
                    _lodo_feature_pair,
                )
                clf = configure_estimator_threads(learner_factory(), threads_per_worker)
                fit_classifier(
                    clf,
                    Xtr,
                    y[train_idx],
                    None if groups is None else groups[train_idx],
                )
                proba = _predict_proba_aligned(clf, Xte, classes)
                pred = classes[proba.argmax(axis=1)]
                metrics = compute_metrics(y[test_idx], pred, proba, classes)
                row = _meta_row(
                    _metric_row(
                        metrics,
                        split_key,
                        None,
                        spec.config_id,
                        mpdr,
                        spec.learner,
                        "outer",
                    ),
                    spec,
                )
                prediction_rows = _prediction_rows_values(
                    split_key,
                    spec.config_id,
                    test_idx,
                    sample_ids,
                    subject_ids,
                    y,
                    class_labels,
                    pred,
                    proba,
                    "outer",
                    split_key,
                )
                result["outer_metrics"].append(row)
                result["outer_predictions"].extend(prediction_rows)
                result["fits"] += 1
            except Exception as exc:
                result["outer_metrics"].append(
                    _meta_row(
                        _failed_metric_row(
                            split_key,
                            None,
                            spec.config_id,
                            mpdr,
                            spec.learner,
                            "outer",
                            exc,
                        ),
                        spec,
                    )
                )
            completed_steps += 1
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                "outer evaluation complete",
            )
        elif needs_outer:
            completed_steps += 1
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                "outer evaluation skipped by qualification gate",
            )
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


def _regression_task(
    spec,
    matrices,
    names,
    dataset,
    train_idx,
    test_idx,
    inner_splits,
    split_key,
    protocol,
    learner_factory,
    gate,
    existing_inner_keys,
    existing_inner_scores,
    existing_qualification,
    needs_outer,
    random_state,
    threads_per_worker,
    resource_sample_interval_s,
    progress_queue=None,
):
    from .configs_sweep import (
        MPDR,
        QualificationGate,
        _count_transformation_factory,
        _lodo_feature_pair,
        _regression_metric_row,
        _regression_prediction_rows,
    )

    tracker = ResourceTracker(sample_interval_s=resource_sample_interval_s).start()
    mpdr = MPDR(
        spec.representation_label, tuple(spec.modalities), spec.transformation_label
    )
    result = {
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
    scores = [float(x) for x in existing_inner_scores if np.isfinite(x)]
    y = np.asarray(dataset.y, dtype=float)
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
        for inner_no, (tr_local, va_local) in enumerate(inner_splits):
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
            tr_idx = train_idx[np.asarray(tr_local, dtype=int)]
            va_idx = train_idx[np.asarray(va_local, dtype=int)]
            try:
                Xtr, Xva, _ = _prepare_candidate_pair(
                    spec,
                    matrices,
                    names,
                    tr_idx,
                    va_idx,
                    protocol,
                    random_state,
                    _count_transformation_factory,
                    _lodo_feature_pair,
                )
                reg = configure_estimator_threads(learner_factory(), threads_per_worker)
                reg.fit(Xtr, y[tr_idx])
                pred = np.asarray(_estimator_call(reg, "predict", Xva), dtype=float)
                metrics = compute_regression_metrics(y[va_idx], pred)
                row = _meta_row(
                    _regression_metric_row(
                        metrics,
                        split_key,
                        inner_key,
                        spec.config_id,
                        mpdr,
                        spec.learner,
                        "inner",
                    ),
                    spec,
                )
                prediction_rows = _regression_prediction_rows(
                    inner_key,
                    spec.config_id,
                    va_idx,
                    dataset.sample_ids,
                    dataset.subject_ids,
                    y,
                    pred,
                    "inner",
                    split_key,
                    dataset.target_name,
                )
                result["inner_metrics"].append(row)
                result["inner_predictions"].extend(prediction_rows)
                score = float(metrics.get(gate.metric, np.nan))
                if np.isfinite(score):
                    scores.append(score)
                result["fits"] += 1
            except Exception as exc:
                row = _regression_metric_row(
                    {
                        k: float("nan")
                        for k in compute_regression_metrics(
                            np.array([0.0, 1.0]), np.array([0.0, 1.0])
                        )
                    },
                    split_key,
                    inner_key,
                    spec.config_id,
                    mpdr,
                    spec.learner,
                    "inner",
                )
                row["ok"] = 0
                row["error"] = f"{type(exc).__name__}: {exc}"
                result["inner_metrics"].append(_meta_row(row, spec))
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
        qualified = True
        gate_score = float("nan")
        if gate.enabled:
            if existing_qualification is not None:
                qualified = bool(int(existing_qualification.get("qualified", 0)))
                gate_score = float(existing_qualification.get("inner_score", np.nan))
            else:
                gate_score = float(np.mean(scores)) if scores else float("nan")
                qualified = QualificationGate(
                    True, gate.metric, gate.threshold
                ).qualifies(gate_score)
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
            try:
                Xtr, Xte, _ = _prepare_candidate_pair(
                    spec,
                    matrices,
                    names,
                    train_idx,
                    test_idx,
                    protocol,
                    random_state,
                    _count_transformation_factory,
                    _lodo_feature_pair,
                )
                reg = configure_estimator_threads(learner_factory(), threads_per_worker)
                reg.fit(Xtr, y[train_idx])
                pred = np.asarray(_estimator_call(reg, "predict", Xte), dtype=float)
                metrics = compute_regression_metrics(y[test_idx], pred)
                row = _meta_row(
                    _regression_metric_row(
                        metrics,
                        split_key,
                        None,
                        spec.config_id,
                        mpdr,
                        spec.learner,
                        "outer",
                    ),
                    spec,
                )
                prediction_rows = _regression_prediction_rows(
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
                result["outer_metrics"].append(row)
                result["outer_predictions"].extend(prediction_rows)
                result["fits"] += 1
            except Exception as exc:
                row = _regression_metric_row(
                    {}, split_key, None, spec.config_id, mpdr, spec.learner, "outer"
                )
                row["ok"] = 0
                row["error"] = f"{type(exc).__name__}: {exc}"
                result["outer_metrics"].append(_meta_row(row, spec))
            completed_steps += 1
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                "outer evaluation complete",
            )
        elif needs_outer:
            completed_steps += 1
            _emit_evaluation_progress(
                progress_queue,
                split_key,
                spec.config_id,
                "outer",
                completed_steps,
                total_steps,
                "outer evaluation skipped by qualification gate",
            )
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


def evaluate_modality_sweep(sweep) -> dict[str, Path]:
    from .configs_sweep import (
        _backfill_metrics_from_predictions,
        _backfill_regression_metrics_from_predictions,
        _checkpoint_result,
        _clear_evaluation_checkpoints,
        _done_pairs,
        _existing_inner_scores,
        _existing_outputs,
        _groups_from_metadata,
        _learner_factory,
        _learner_name,
        _load_existing_evaluation,
        _outer_pair_complete,
        _prepare_dirs,
        _qualification_map,
        _resolved_evaluation_splits,
        _strata_from_metadata,
        _experiment_fingerprint,
        _scientific_digest,
        _scientific_value,
        _software_provenance,
        _source_tree_sha256,
        _subject_safe_groups,
        _validate_experiment_identity,
        _write_config_table,
        _write_rankings_and_figures,
        _write_tables,
        write_mpma_b_selection_outputs,
    )
    from .figures import _write_representation_impact_figure

    dataset = load_modalities(sweep.samples, sweep.modalities)
    root = sweep.root()
    _prepare_dirs(root)
    learners = [_learner_factory(item, task=dataset.task) for item in sweep.learners]
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
    source_hashes = modality_source_fingerprints(sweep.samples, sweep.modalities)
    dataset_hash = modality_dataset_fingerprint(dataset)
    plan = sweep.evaluation
    configured_groups = _groups_from_metadata(dataset.metadata, sweep.samples.group_col)
    effective_groups, effective_group_col = _subject_safe_groups(
        plan, dataset, configured_groups, sweep.samples.group_col
    )
    fingerprint_strata = (
        _strata_from_metadata(dataset.metadata, dataset.y, sweep.samples.stratify_col)
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
            "group_col": sweep.samples.group_col,
            "effective_group_col": effective_group_col,
            "group_assignments": None
            if effective_groups is None
            else _scientific_digest(
                np.asarray(effective_groups, dtype=object).astype(str).tolist()
            ),
            "subject_id_policy": "auto_group_repeated_subjects",
            "stratify_col": _scientific_value(sweep.samples.stratify_col),
            "strata_assignments": None
            if fingerprint_strata is None
            else _scientific_digest(
                np.asarray(fingerprint_strata, dtype=object).astype(str).tolist()
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
    groups = _groups_from_metadata(dataset.metadata, sweep.samples.group_col)
    strata = (
        _strata_from_metadata(dataset.metadata, dataset.y, sweep.samples.stratify_col)
        if dataset.task == "classification"
        else None
    )
    outer_splits, inner_splits_by_outer = _resolved_evaluation_splits(
        root,
        sweep.evaluation,
        dataset,
        groups,
        strata,
        sweep.samples.stratify_col,
        sweep.samples.group_col,
    )
    current_ids = set(configs["config_id"].astype(str))
    outer_keys = {str(x["split_key"]) for x in outer_splits}
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
    inner_done = _done_pairs(existing["inner_metrics"], "inner_key")
    outer_done = _done_pairs(existing["outer_metrics"], "split_key")
    qmap = _qualification_map(existing["qualification"]) if sweep.gate.enabled else {}
    learner_lookup = {name: factory for name, factory in learners}
    tasks = []
    split_counts = {}
    for split in outer_splits:
        split_key = str(split["split_key"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if dataset.task == "classification" and (
            len(np.unique(dataset.y[train_idx])) < 2 or len(test_idx) == 0
        ):
            continue
        inner = inner_splits_by_outer.get(split_key, [])
        inner_keys = [f"{split_key}__i{i}" for i in range(len(inner))]
        for spec in specs:
            missing_inner = {
                key for key in inner_keys if (key, spec.config_id) not in inner_done
            }
            needs_outer = not _outer_pair_complete(
                split_key, spec.config_id, outer_done, qmap
            )
            if not missing_inner and not needs_outer:
                continue
            args_common = (spec, matrices, names)
            if dataset.task == "classification":
                task = delayed(_classification_task)(
                    *args_common,
                    dataset.y,
                    groups,
                    dataset.classes,
                    tuple(dataset.class_labels),
                    tuple(dataset.sample_ids),
                    tuple(dataset.subject_ids),
                    train_idx,
                    test_idx,
                    tuple(inner),
                    split_key,
                    sweep.evaluation.protocol,
                    learner_lookup[spec.learner],
                    sweep.gate,
                    set(inner_keys) - missing_inner,
                    _existing_inner_scores(
                        existing["inner_metrics"],
                        split_key,
                        spec.config_id,
                        sweep.gate.metric,
                    ),
                    qmap.get((split_key, spec.config_id)),
                    needs_outer,
                    int(sweep.evaluation.random_state),
                    1,
                    float(sweep.evaluation.resource_sample_interval_s),
                )
            else:
                task = delayed(_regression_task)(
                    *args_common,
                    dataset,
                    train_idx,
                    test_idx,
                    tuple(inner),
                    split_key,
                    sweep.evaluation.protocol,
                    learner_lookup[spec.learner],
                    sweep.gate,
                    set(inner_keys) - missing_inner,
                    _existing_inner_scores(
                        existing["inner_metrics"],
                        split_key,
                        spec.config_id,
                        sweep.gate.metric,
                    ),
                    qmap.get((split_key, spec.config_id)),
                    needs_outer,
                    int(sweep.evaluation.random_state),
                    1,
                    float(sweep.evaluation.resource_sample_interval_s),
                )
            tasks.append((split_key, spec.config_id, task))
            split_counts[split_key] = split_counts.get(split_key, 0) + 1
    execution = resolve_execution_plan(
        sweep.evaluation.n_jobs,
        len(tasks) or 1,
        backend=sweep.evaluation.parallel_backend,
        memory_fraction=sweep.evaluation.memory_fraction,
        min_worker_memory_gib=sweep.evaluation.min_worker_memory_gib,
    )
    prepared = []
    for split_key, cid, task in tasks:
        fn, args, kwargs = task
        args = list(args)
        args[-2] = execution.threads_per_worker
        prepared.append((split_key, cid, fn, tuple(args), kwargs))
    stage("Modality configuration sweep", sweep.title)
    summary_table(
        "Sweep overview",
        {
            "samples": len(dataset.sample_ids),
            "primary modality": dataset.primary_modality,
            "modalities": ", ".join(dataset.modalities),
            "candidates": len(specs),
            "protocol": sweep.evaluation.protocol,
            "outer splits": len(outer_splits),
            "pending jobs": len(prepared),
            "workers": execution.workers,
            "threads per worker": execution.threads_per_worker,
            "experiment dir": root,
        },
    )
    rows = {
        "outer_metrics": [],
        "inner_metrics": [],
        "outer_predictions": [],
        "inner_predictions": [],
        "qualification": [],
        "job_resources": [],
    }
    t0 = time.perf_counter()
    if prepared:
        spec_lookup = {spec.config_id: spec for spec in specs}
        candidate_order = {spec.config_id: i + 1 for i, spec in enumerate(specs)}
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
                    total=len(prepared),
                )
                split_tasks = {
                    str(split_key): prog.add_task(
                        f"{split_key} · queued · 0/{count} jobs",
                        total=max(1, int(count)),
                    )
                    for split_key, count in split_counts.items()
                }

                def monitor():
                    while not stop_monitor.is_set():
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
                            f"candidate {ordinal}/{len(specs)} · {spec.integration.key} · {spec.learner} · "
                            f"{phase} {int(step_done)}/{int(step_total)} · {detail}"
                        )
                        prog.update(split_tasks[split_key], description=description)

                monitor_thread = Thread(target=monitor, daemon=True)
                monitor_thread.start()
                try:
                    payloads = [
                        (fn, args, {**kwargs, "progress_queue": progress_queue})
                        for _, _, fn, args, kwargs in prepared
                    ]
                    for result in iter_parallel_tasks(payloads, execution):
                        _checkpoint_result(root, result)
                        for key in rows:
                            rows[key].extend(result.get(key, []))
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
                                label = f"completed · {spec.integration.key} · {spec.learner}"
                            prog.update(
                                split_tasks[split_key],
                                completed=completed_by_split[split_key],
                                description=(
                                    f"{split_key} · {completed_by_split[split_key]}/{split_counts.get(split_key, 0)} jobs · "
                                    f"{label} · last {float(result.get('elapsed_s', 0.0)):.1f}s"
                                ),
                            )
                        prog.update(
                            overall_task,
                            completed=completed_jobs,
                            description=(
                                f"Evaluation · {execution.workers} workers · {completed_fits:,} fits · "
                                f"{failed_fits:,} failures · checkpoints current"
                            ),
                        )
                finally:
                    stop_monitor.set()
                    monitor_thread.join(timeout=2.0)
                prog.update(
                    overall_task,
                    completed=len(prepared),
                    description=(
                        f"Evaluation · complete · {completed_fits:,} fits · {failed_fits:,} failures · checkpoints current"
                    ),
                )
    _write_tables(
        root,
        rows["outer_metrics"],
        rows["inner_metrics"],
        rows["outer_predictions"],
        rows["inner_predictions"],
        rows["qualification"],
        rows["job_resources"],
        existing=existing,
        gate_enabled=sweep.gate.enabled,
    )
    write_mpma_b_selection_outputs(
        root, sweep.evaluation.optimize_metric, plan=sweep.ensemble
    )
    _write_rankings_and_figures(
        root, dataset.class_labels, sweep.evaluation.optimize_metric
    )
    _write_representation_impact_figure(
        root, metric_col=sweep.evaluation.optimize_metric
    )
    dump_json_standard(
        {
            "elapsed_s": time.perf_counter() - t0,
            "workers": execution.workers,
            "threads_per_worker": execution.threads_per_worker,
            "machine": machine_profile(execution.logical_cpus, execution.physical_cpus),
        },
        root / "run_summary.json",
    )
    success("Modality configuration sweep completed")
    outputs = _existing_outputs(root)
    outputs["modality_manifest"] = root / "manifest_modalities.json"
    path_table("Configuration sweep outputs", outputs)
    return outputs


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
        _groups_from_metadata,
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
        _groups_from_metadata,
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
