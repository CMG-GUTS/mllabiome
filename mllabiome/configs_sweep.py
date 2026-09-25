from __future__ import annotations

import hashlib
import importlib.metadata
import json
import marshal
import os
import platform
import sqlite3
import subprocess
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from threadpoolctl import threadpool_limits

from ._version import __version__
from .compute import ResourceTracker, machine_profile
from .console import info, path_table, progress, stage, success, summary_table
from .data import Data, Dataset, dataset_fingerprint, load_dataset
from .estimator_protocol import is_estimator_instance
from .explainability_methods import (
    ALE,
    SHAP,
    Permutation,
    apply_profile,
    method_has_local,
    normalise_profile,
)
from .figures import _write_representation_impact_figure
from .integrations import Integration
from .learners import (
    _learner_factory,
    _learner_name,
    fit_classifier,
    learner_display_label,
    validate_model_specs,
)
from .metrics import _estimator_call
from .metrics import _predict_proba_aligned as _metrics_predict_proba_aligned
from .metrics import (
    aggregate_validation_metric,
    canonical_metric_name,
    compute_metrics,
    compute_regression_metrics,
    grouped_log_loss,
    metric_is_loss,
    metric_passes_threshold,
)
from .modalities import Modality, Samples
from .resolutions import (
    _parse_resolution,
    mask_feature_blocks,
    materialize_mpdr_with_blocks,
)
from .runtime import (
    configure_estimator_threads,
    iter_parallel_tasks,
    resolve_execution_plan,
    thread_environment,
)
from .selection import write_mpma_b_selection_outputs
from .splits import resolve_cv_splits
from .storage import read_table, remove_table, table_exists, write_table
from .transformations import (
    TRANSFORMATION_LABELS,
    _count_transformation_factory,
    _count_transformation_name,
    _count_transformation_specs_for_blocks,
)
from .utils import METRIC_COLUMNS, TAXONOMIC_LEVELS, dump_json_standard


from .sweep_types import (
    _default_transformations,
    _legacy_local_explanations,
    _normalise_explainability_classes_config,
    _normalise_explainability_targets_config,
    Ensemble,
    Evaluation,
    Explainability,
    LocalExplanationMode,
    LocalExplanations,
    MPDR,
    MPMA,
    QualificationGate,
    Sweep,
    SweepTask,
    _effective_local_explanations_mode,
    _normalise_sweep_task,
    build_sweep_from_module,
    sweep_task,
    validate_sweep_class_count,
)


_MPDR_SEMANTICS = "select_then_transform_fold_local_rank_composition_v3"


def _mpdr_id(count_transformation: str, resolution: str) -> str:
    return hashlib.sha1(
        f"{_MPDR_SEMANTICS}__{count_transformation}__{resolution}".encode()
    ).hexdigest()[:12]


def _scientific_value(value: Any) -> Any:
    if is_estimator_instance(value):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "params": _scientific_value(value.get_params(deep=True)),
        }
    if isinstance(value, Mapping):
        return {
            str(k): _scientific_value(v)
            for k, v in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_scientific_value(v) for v in value]
    if isinstance(value, set):
        return sorted(
            (_scientific_value(v) for v in value),
            key=lambda x: json.dumps(x, sort_keys=True, default=str),
        )
    if isinstance(value, np.ndarray):
        return _scientific_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": _scientific_value(asdict(value)),
        }
    slots = getattr(type(value), "__slots__", None)
    if slots:
        slot_names = (slots,) if isinstance(slots, str) else tuple(slots)
        state = {
            str(name): _scientific_value(getattr(value, name))
            for name in slot_names
            if hasattr(value, name)
        }
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": state,
        }
    if callable(value):
        name = f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', getattr(value, '__name__', type(value).__qualname__))}"
        code = getattr(value, "__code__", None)
        if code is None:
            return {"callable": name}
        closure = getattr(value, "__closure__", None)
        closure_values = []
        if closure is not None:
            for cell in closure:
                try:
                    closure_values.append(_scientific_value(cell.cell_contents))
                except ValueError:
                    closure_values.append("<empty>")
        return {
            "callable": name,
            "code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest(),
            "defaults": _scientific_value(getattr(value, "__defaults__", None)),
            "kwdefaults": _scientific_value(getattr(value, "__kwdefaults__", None)),
            "closure": closure_values,
        }
    if value is None or isinstance(value, (str, int, bool)):
        return value
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": _scientific_value(state),
        }
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": str(value),
    }


def _scientific_digest(payload: Any) -> str:
    canonical = json.dumps(
        _scientific_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _transformation_fingerprint(identity: str, spec: Any) -> str:
    return _scientific_digest(
        {"identity": str(identity), "spec": _scientific_value(spec)}
    )


def _resolution_fingerprint(
    resolution: str, levels: Sequence[str], feature_blocks: Any
) -> str:
    blocks = []
    for name, indices in feature_blocks or ():
        blocks.append((str(name), tuple(int(index) for index in indices)))
    return _scientific_digest(
        {
            "resolution": str(resolution),
            "levels": tuple(str(level) for level in levels),
            "feature_blocks": blocks,
        }
    )


def _learner_payload(item: Any) -> dict[str, Any]:
    if (
        not isinstance(item, tuple)
        or len(item) != 2
        or not is_estimator_instance(item[1])
    ):
        raise TypeError("Learners must be explicit (name, estimator) pairs.")
    name, estimator = item
    return {
        "name": str(name),
        "class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "params": _scientific_value(estimator.get_params(deep=True)),
    }


def _learner_fingerprint(item: Any) -> str:
    return _scientific_digest(_learner_payload(item))


def _config_id(
    count_transformation: str,
    resolution: str,
    learner_name: str,
    learner_fingerprint: str | None = None,
    transformation_fingerprint: str | None = None,
    resolution_fingerprint: str | None = None,
) -> str:
    learner_identity = str(learner_fingerprint or learner_name)
    transformation_identity = str(transformation_fingerprint or count_transformation)
    resolution_identity = str(resolution_fingerprint or resolution)
    return hashlib.sha256(
        f"{_MPDR_SEMANTICS}__{count_transformation}__{resolution}__{resolution_identity}__{transformation_identity}__{learner_name}__{learner_identity}".encode()
    ).hexdigest()[:12]


def _evaluation_cache_payload(sweep: Sweep, dataset: Dataset) -> dict[str, Any]:
    plan = sweep.evaluation
    configured_groups = _groups_from_metadata(
        dataset.metadata, None if sweep.data is None else sweep.data.group_col
    )
    effective_groups, effective_group_col = _subject_safe_groups(
        plan,
        dataset,
        configured_groups,
        None if sweep.data is None else sweep.data.group_col,
    )
    strata = _strata_from_metadata(
        dataset.metadata,
        dataset.y,
        None if sweep.data is None else sweep.data.stratify_col,
    )
    return {
        "schema": "evaluation-cache-fingerprint-v1",
        "source_tree_sha256": _source_tree_sha256(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "model_runtime_dependencies": {
            name: _installed_distribution_version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "scikit-bio",
                "xgboost",
                "lightgbm",
                "catboost",
                "flaml",
                "joblib",
                "threadpoolctl",
            )
        },
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "task": str(dataset.task),
        "target": str(dataset.target_name),
        "evaluation": {
            "protocol": str(plan.protocol),
            "outer_folds": int(plan.outer_folds),
            "inner_folds": int(plan.inner_folds),
            "repeats": int(plan.repeats),
            "random_state": int(plan.random_state),
            "optimize_metric": _scientific_value(plan.optimize_metric),
        },
        "group_col": None if sweep.data is None else sweep.data.group_col,
        "effective_group_col": effective_group_col,
        "group_assignments": (
            None
            if effective_groups is None
            else _scientific_digest(
                np.asarray(effective_groups, dtype=object).astype(str).tolist()
            )
        ),
        "subject_id_policy": "auto_group_repeated_subjects",
        "stratify_col": (
            None if sweep.data is None else _scientific_value(sweep.data.stratify_col)
        ),
        "strata_assignments": (
            None
            if strata is None
            else _scientific_digest(
                np.asarray(strata, dtype=object).astype(str).tolist()
            )
        ),
        "gate": _scientific_value(sweep.gate),
    }


def _evaluation_cache_fingerprint(sweep: Sweep, dataset: Dataset) -> str:
    return _scientific_digest(_evaluation_cache_payload(sweep, dataset))


def _evaluation_fingerprint(
    sweep: Sweep, dataset: Dataset, configs: pd.DataFrame
) -> str:
    payload = {
        "schema": "evaluation-fingerprint-v4",
        "cache_fingerprint": _evaluation_cache_fingerprint(sweep, dataset),
        "config_ids": sorted(configs["config_id"].astype(str).tolist()),
    }
    return _scientific_digest(payload)


def _experiment_fingerprint(sweep: Sweep, evaluation_fingerprint: str) -> str:
    payload = {
        "schema": "experiment-fingerprint-v2",
        "evaluation_fingerprint": str(evaluation_fingerprint),
        "ensemble": _scientific_value(sweep.ensemble),
        "explainability": _scientific_value(sweep.explainability),
        "integrations": _scientific_value(sweep.integrations),
        "representations": _scientific_value(sweep.representations),
        "transformations": _scientific_value(sweep.transformations),
    }
    return _scientific_digest(payload)


def _has_evaluation_artifacts(root: Path) -> bool:
    paths = (
        root / "results" / "outer_results.parquet",
        root / "predictions" / "outer_predictions.parquet",
        root / "checkpoints",
    )
    return any(path.exists() for path in paths)


def _validate_experiment_identity(
    root: Path, evaluation_fingerprint: str, *, redo: bool
) -> None:
    path = root / "manifest.json"
    if not path.exists() or bool(redo):
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Existing experiment manifest cannot be read.") from exc
    stored = payload.get("evaluation_fingerprint")
    if stored is None:
        if _has_evaluation_artifacts(root):
            raise ValueError(
                "Existing evaluation artifacts predate scientific cache fingerprinting. Rerun with --redo or use a new experiment directory before reusing results."
            )
        return
    if str(stored) != str(evaluation_fingerprint):
        raise ValueError(
            "Existing experiment results do not match the current scientific evaluation fingerprint. Rerun with --redo or use a new experiment directory."
        )


def _incremental_context_payload_from_sweep(sweep: Sweep) -> dict[str, Any]:
    plan = sweep.evaluation
    data = sweep.data
    return {
        "evaluation": {
            "protocol": str(plan.protocol),
            "outer_folds": int(plan.outer_folds),
            "inner_folds": int(plan.inner_folds),
            "repeats": int(plan.repeats),
            "random_state": int(plan.random_state),
            "optimize_metric": _scientific_value(plan.optimize_metric),
        },
        "group_col": None if data is None else data.group_col,
        "stratify_col": (
            None if data is None else _scientific_value(data.stratify_col)
        ),
        "gate": _scientific_value(asdict(sweep.gate)),
    }


def _incremental_context_payload_from_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    sweep_payload = payload.get("sweep")
    if not isinstance(sweep_payload, Mapping):
        return None
    evaluation = sweep_payload.get("evaluation")
    data = sweep_payload.get("data")
    gate = sweep_payload.get("gate")
    if not isinstance(evaluation, Mapping) or not isinstance(data, Mapping):
        return None
    required = (
        "protocol",
        "outer_folds",
        "inner_folds",
        "repeats",
        "random_state",
        "optimize_metric",
    )
    if any(key not in evaluation for key in required):
        return None
    return {
        "evaluation": {key: evaluation.get(key) for key in required},
        "group_col": data.get("group_col"),
        "stratify_col": data.get("stratify_col"),
        "gate": gate,
    }


def _validate_incremental_experiment_identity(
    root: Path,
    sweep: Sweep,
    evaluation_cache_fingerprint: str,
    *,
    redo: bool,
) -> None:
    path = root / "manifest.json"
    if not path.exists() or bool(redo):
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Existing experiment manifest cannot be read.") from exc

    stored_cache = payload.get("evaluation_cache_fingerprint")
    if stored_cache is not None:
        if str(stored_cache) != str(evaluation_cache_fingerprint):
            raise ValueError(
                "Existing experiment results do not match the current evaluation cache context. "
                "The dataset, evaluation plan, grouping/stratification, gate, runtime, or package code changed. "
                "Rerun with --redo or use a new experiment directory."
            )
        return

    stored_legacy = payload.get("evaluation_fingerprint")
    if stored_legacy is None:
        if _has_evaluation_artifacts(root):
            raise ValueError(
                "Existing evaluation artifacts predate scientific cache fingerprinting. "
                "Rerun with --redo or use a new experiment directory before reusing results."
            )
        return

    stored_context = _incremental_context_payload_from_manifest(payload)
    current_context = _incremental_context_payload_from_sweep(sweep)
    if stored_context is None or _scientific_digest(
        stored_context
    ) != _scientific_digest(current_context):
        raise ValueError(
            "Existing experiment results use different evaluation, grouping/stratification, "
            "or qualification-gate settings. Rerun with --redo or use a new experiment directory."
        )
    if _has_evaluation_artifacts(root):
        info(
            "Migrating legacy evaluation cache metadata: completed MPMA/split pairs will be reused; "
            "new config IDs will be evaluated incrementally."
        )


def _scientifically_compatible_config(
    count_transformation: str, resolution: str, learner_payload: Mapping[str, Any]
) -> bool:
    learner_class = str(learner_payload.get("class", "")).rsplit(".", 1)[-1]
    if learner_class != "SIAMCATClassifier":
        return True
    transformation = str(count_transformation).strip().casefold()
    return transformation == "identity" and str(resolution).strip().casefold() == "raw"


def build_sweep_configs(
    resolutions: Sequence[Any],
    count_transformations: Sequence[Any],
    learners: Sequence[Any],
    resolution_feature_blocks: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    learner_specs = [
        (
            _learner_name(x),
            learner_display_label(x),
            _learner_fingerprint(x),
            _learner_payload(x),
        )
        for x in learners
    ]
    block_map = (
        {} if resolution_feature_blocks is None else dict(resolution_feature_blocks)
    )
    for res in resolutions:
        res_name, levels = _parse_resolution(res)
        feature_blocks = block_map.get(res_name)
        if feature_blocks is None and "all" not in levels:
            feature_blocks = tuple(
                (level, (index,)) for index, level in enumerate(levels)
            )
        nonempty_blocks = tuple(
            (str(name), tuple(indices))
            for name, indices in (feature_blocks or ())
            if tuple(indices)
        )
        taxonomic_blocks = tuple(
            name for name, _ in nonempty_blocks if name in TAXONOMIC_LEVELS
        )
        taxonomic_block_count = len(dict.fromkeys(taxonomic_blocks))
        if taxonomic_block_count == 0 and "all" not in levels:
            taxonomic_blocks = tuple(
                level for level in levels if level in TAXONOMIC_LEVELS
            )
            taxonomic_block_count = len(dict.fromkeys(taxonomic_blocks))
        representation_scope = (
            "single-rank"
            if taxonomic_block_count == 1
            else "multi-rank"
            if taxonomic_block_count > 1
            else "unresolved"
        )
        resolution_fingerprint = _resolution_fingerprint(
            res_name, levels, feature_blocks
        )
        transformation_specs = _count_transformation_specs_for_blocks(
            count_transformations, feature_blocks
        )
        for ct_name, ct_spec in transformation_specs:
            transformation_fingerprint = _transformation_fingerprint(ct_name, ct_spec)
            feature_filter = getattr(ct_spec, "feature_filter", None)
            feature_filter_identity = (
                "none" if feature_filter is None else str(feature_filter.identity)
            )
            prevalence_threshold = (
                float("nan")
                if feature_filter is None
                else float(feature_filter.threshold)
            )
            detection_threshold = (
                float("nan")
                if feature_filter is None
                else float(feature_filter.detection_threshold)
            )
            for (
                lname,
                learner_display,
                learner_fingerprint,
                learner_payload,
            ) in learner_specs:
                if not _scientifically_compatible_config(
                    ct_name, res_name, learner_payload
                ):
                    continue
                rows.append(
                    {
                        "config_id": _config_id(
                            ct_name,
                            res_name,
                            lname,
                            learner_fingerprint,
                            transformation_fingerprint,
                            resolution_fingerprint,
                        ),
                        "mpdr_id": _mpdr_id(ct_name, res_name),
                        "transformation_fingerprint": transformation_fingerprint,
                        "resolution_fingerprint": resolution_fingerprint,
                        "count_transformation": ct_name,
                        "feature_filter": feature_filter_identity,
                        "prevalence_threshold": prevalence_threshold,
                        "detection_threshold": detection_threshold,
                        "resolution": res_name,
                        "levels": ",".join(levels),
                        "taxonomic_blocks": ",".join(taxonomic_blocks),
                        "taxonomic_block_count": taxonomic_block_count,
                        "representation_scope": representation_scope,
                        "learner": lname,
                        "learner_display": learner_display,
                        "learner_fingerprint": learner_fingerprint,
                        "learner_class": learner_payload["class"],
                        "learner_params": json.dumps(
                            learner_payload["params"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "active": 1,
                    }
                )
    return pd.DataFrame(rows)


def _lodo_feature_pair(
    X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, protocol: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X[train_idx])
    X_test = np.asarray(X[test_idx])
    if str(protocol).lower() not in {"lodo", "leave_one_dataset_out"}:
        return X_train, X_test, np.ones(X.shape[1], dtype=bool)
    mask = np.any(np.isfinite(X_train) & (X_train != 0), axis=0)
    if not np.any(mask):
        raise ValueError(
            "LODO training partition contains no nonzero features at the selected resolution."
        )
    return X_train[:, mask], X_test[:, mask], mask


def _inner_validation_label(plan: Evaluation) -> str | int:
    if str(plan.protocol).lower() in {"lodo", "leave_one_dataset_out"}:
        return "leave-one-group-out across outer training groups"
    return plan.inner_folds


def _prediction_rows_values(
    key: str,
    cid: str,
    idx: np.ndarray,
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    y: np.ndarray,
    class_labels: Sequence[str],
    pred: np.ndarray,
    proba: np.ndarray,
    stage: str,
    outer_split_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_no, sample_idx in enumerate(idx):
        i = int(sample_idx)
        row = {
            "stage": stage,
            "split_key": key,
            "outer_split_key": outer_split_key,
            "sample_id": str(sample_ids[i]),
            "subject_id": str(subject_ids[i]),
            "sample_index": i,
            "config_id": cid,
            "y_true": int(y[i]),
            "y_pred": int(pred[row_no]),
        }
        for j, label in enumerate(class_labels):
            row[f"proba_{label}"] = float(proba[row_no, j])
        if len(class_labels) == 2:
            row["y_proba_pos"] = float(proba[row_no, 1])
        rows.append(row)
    return rows


def _evaluate_mpma_split_task(
    X_base: np.ndarray,
    feature_blocks: Any,
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
    res_name: str,
    levels: Sequence[str],
    ct_name: str,
    ct_item: Any,
    transformation_factory_builder: Any,
    learner_name: str,
    learner_factory: Any,
    cid: str,
    gate_enabled: bool,
    gate_metric: str,
    gate_threshold: float | None,
    selection_metric: str,
    existing_inner_keys: set[str],
    existing_inner_scores: Sequence[dict[str, Any]],
    existing_qualification: dict[str, Any] | None,
    needs_outer: bool,
    random_state: int,
    threads_per_worker: int,
    resource_sample_interval_s: float,
) -> dict[str, Any]:
    gate_metric_input = str(gate_metric)
    gate_metric = canonical_metric_name(gate_metric_input)
    requested_metrics = {canonical_metric_name(selection_metric)}
    if gate_enabled:
        requested_metrics.add(gate_metric)
    tracker = ResourceTracker(sample_interval_s=resource_sample_interval_s).start()
    mpdr = MPDR(
        resolution=res_name, levels=tuple(levels), count_transformation=str(ct_name)
    )
    result = {
        "split_key": split_key,
        "config_id": cid,
        "inner_metrics": [],
        "inner_predictions": [],
        "outer_metrics": [],
        "outer_predictions": [],
        "qualification": [],
        "job_resources": [],
        "fits": 0,
    }
    score_rows = [dict(x) for x in existing_inner_scores if isinstance(x, dict)]
    with (
        thread_environment(threads_per_worker),
        threadpool_limits(limits=max(1, int(threads_per_worker))),
    ):
        for inner_no, (inner_train_local, inner_val_local) in enumerate(inner_splits):
            inner_key = f"{split_key}__i{inner_no}"
            if inner_key in existing_inner_keys:
                continue
            tr_idx = train_idx[np.asarray(inner_train_local, dtype=int)]
            va_idx = train_idx[np.asarray(inner_val_local, dtype=int)]
            if len(np.unique(y[tr_idx])) < 2 or len(va_idx) == 0:
                continue
            X_inner_train, X_inner_val, feature_mask = _lodo_feature_pair(
                X_base, tr_idx, va_idx, protocol
            )
            inner_blocks = mask_feature_blocks(feature_blocks, feature_mask)
            _, inner_ct_factory = transformation_factory_builder(
                ct_item,
                random_state=random_state,
                feature_blocks=inner_blocks,
            )
            fitted = inner_ct_factory()
            try:
                X_tr, X_va = fitted.apply_pair(X_inner_train, X_inner_val)
                clf = configure_estimator_threads(learner_factory(), threads_per_worker)
                fit_classifier(
                    clf, X_tr, y[tr_idx], None if groups is None else groups[tr_idx]
                )
                proba = _predict_proba_aligned(clf, X_va, classes)
                pred = classes[proba.argmax(axis=1)]
                metrics = compute_metrics(y[va_idx], pred, proba, classes)
                if "subject_macro_log_loss" in requested_metrics:
                    metrics["subject_macro_log_loss"] = grouped_log_loss(
                        y[va_idx],
                        proba,
                        classes,
                        np.asarray(subject_ids, dtype=object)[va_idx],
                    )
                if "cohort_macro_log_loss" in requested_metrics and str(
                    protocol
                ).lower() in {"lodo", "leave_one_dataset_out"}:
                    metrics["cohort_macro_log_loss"] = float(metrics["log_loss"])
                row = _metric_row(
                    metrics, split_key, inner_key, cid, mpdr, learner_name, "inner"
                )
                row.update(fitted.feature_filter_metadata())
                row["n_samples"] = int(len(va_idx))
                row["n_subjects"] = int(
                    len(pd.unique(np.asarray(subject_ids, dtype=object)[va_idx]))
                )
                result["inner_metrics"].append(row)
                result["inner_predictions"].extend(
                    _prediction_rows_values(
                        inner_key,
                        cid,
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
                )
                score = float(metrics.get(gate_metric, np.nan))
                if np.isfinite(score):
                    score_rows.append(dict(row))
                result["fits"] += 1
            except Exception as exc:
                result["inner_metrics"].append(
                    _failed_metric_row(
                        split_key, inner_key, cid, mpdr, learner_name, "inner", exc
                    )
                )
        qualified = True
        gate_score = float("nan")
        if gate_enabled:
            if existing_qualification is not None:
                qualified = bool(int(existing_qualification.get("qualified", 0)))
                gate_score = float(existing_qualification.get("inner_score", np.nan))
            else:
                gate_score, _ = aggregate_validation_metric(
                    pd.DataFrame(score_rows), gate_metric
                )
                gate = QualificationGate(
                    enabled=True, metric=gate_metric_input, threshold=gate_threshold
                )
                qualified = gate.qualifies(gate_score)
                result["qualification"].append(
                    {
                        "split_key": split_key,
                        "config_id": cid,
                        "mpdr_id": _mpdr_id(ct_name, res_name),
                        "count_transformation": str(ct_name),
                        "resolution": res_name,
                        "levels": ",".join(levels),
                        "learner": learner_name,
                        "gate_enabled": 1,
                        "gate_metric": gate_metric,
                        "gate_threshold": gate_threshold,
                        "inner_score": gate_score,
                        "qualified": int(qualified),
                    }
                )
        if needs_outer and qualified:
            X_outer_train, X_outer_test, feature_mask = _lodo_feature_pair(
                X_base, train_idx, test_idx, protocol
            )
            outer_blocks = mask_feature_blocks(feature_blocks, feature_mask)
            _, outer_ct_factory = transformation_factory_builder(
                ct_item,
                random_state=random_state,
                feature_blocks=outer_blocks,
            )
            fitted_outer = outer_ct_factory()
            try:
                X_train, X_test = fitted_outer.apply_pair(X_outer_train, X_outer_test)
                clf = configure_estimator_threads(learner_factory(), threads_per_worker)
                fit_classifier(
                    clf,
                    X_train,
                    y[train_idx],
                    None if groups is None else groups[train_idx],
                )
                proba = _predict_proba_aligned(clf, X_test, classes)
                pred = classes[proba.argmax(axis=1)]
                metrics = compute_metrics(y[test_idx], pred, proba, classes)
                if "subject_macro_log_loss" in requested_metrics:
                    metrics["subject_macro_log_loss"] = grouped_log_loss(
                        y[test_idx],
                        proba,
                        classes,
                        np.asarray(subject_ids, dtype=object)[test_idx],
                    )
                if "cohort_macro_log_loss" in requested_metrics and str(
                    protocol
                ).lower() in {"lodo", "leave_one_dataset_out"}:
                    metrics["cohort_macro_log_loss"] = float(metrics["log_loss"])
                row = _metric_row(
                    metrics, split_key, None, cid, mpdr, learner_name, "outer"
                )
                row.update(fitted_outer.feature_filter_metadata())
                row["n_samples"] = int(len(test_idx))
                row["n_subjects"] = int(
                    len(pd.unique(np.asarray(subject_ids, dtype=object)[test_idx]))
                )
                result["outer_metrics"].append(row)
                result["outer_predictions"].extend(
                    _prediction_rows_values(
                        split_key,
                        cid,
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
                )
                result["fits"] += 1
            except Exception as exc:
                result["outer_metrics"].append(
                    _failed_metric_row(
                        split_key, None, cid, mpdr, learner_name, "outer", exc
                    )
                )
    measured = tracker.stop()
    resource_row = {
        "split_key": str(split_key),
        "config_id": str(cid),
        "resolution": str(res_name),
        "count_transformation": str(ct_name),
        "learner": str(learner_name),
        "fits": int(result.get("fits", 0)),
        "threads_per_worker": int(threads_per_worker),
        **measured,
    }
    result["job_resources"] = [resource_row]
    result["elapsed_s"] = float(measured.get("wall_time_s", 0.0))
    return result


def _checkpoint_result(root: Path, result: dict[str, Any]) -> None:
    payload = {
        "inner_metrics": result.get("inner_metrics", []),
        "inner_predictions": result.get("inner_predictions", []),
        "outer_metrics": result.get("outer_metrics", []),
        "outer_predictions": result.get("outer_predictions", []),
        "qualification": result.get("qualification", []),
        "job_resources": result.get("job_resources", []),
    }
    blob = zlib.compress(
        json.dumps(payload, allow_nan=True, separators=(",", ":")).encode("utf-8"),
        level=3,
    )
    conn = sqlite3.connect(root / "configs.db")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS evaluation_checkpoints (
                config_id TEXT NOT NULL,
                split_key TEXT NOT NULL,
                payload BLOB NOT NULL,
                elapsed_s REAL,
                PRIMARY KEY (config_id, split_key)
            )"""
        )
        conn.execute(
            """INSERT OR REPLACE INTO evaluation_checkpoints(config_id, split_key, payload, elapsed_s)
               VALUES (?, ?, ?, ?)""",
            (
                str(result.get("config_id", "")),
                str(result.get("split_key", "")),
                sqlite3.Binary(blob),
                float(result.get("elapsed_s", 0.0)),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _load_checkpoint_frames(
    root: Path, current_config_ids: set[str], outer_keys: set[str]
) -> dict[str, pd.DataFrame]:
    empty = {
        "outer_metrics": pd.DataFrame(),
        "inner_metrics": pd.DataFrame(),
        "outer_predictions": pd.DataFrame(),
        "inner_predictions": pd.DataFrame(),
        "qualification": pd.DataFrame(),
        "job_resources": pd.DataFrame(),
    }
    db_path = root / "configs.db"
    if not db_path.exists():
        return empty
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evaluation_checkpoints'"
        ).fetchone()
        if not exists:
            return empty
        rows = conn.execute(
            "SELECT config_id, split_key, payload FROM evaluation_checkpoints"
        ).fetchall()
    finally:
        conn.close()
    buckets = {k: [] for k in empty}
    for config_id, split_key, blob in rows:
        if str(split_key) not in outer_keys:
            continue
        try:
            payload = json.loads(zlib.decompress(blob).decode("utf-8"))
        except Exception:
            continue
        for key in buckets:
            buckets[key].extend(payload.get(key, []))
    return {
        key: pd.DataFrame(rows) if rows else pd.DataFrame()
        for key, rows in buckets.items()
    }


def _clear_evaluation_checkpoints(root: Path, compact: bool = False) -> None:
    db_path = root / "configs.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS evaluation_checkpoints")
        conn.commit()
        if compact:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
    finally:
        conn.close()


def _backfill_metrics_from_predictions(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    classes: np.ndarray,
    class_labels: Sequence[str],
    metric_key: str,
    requested_metrics: Sequence[str] = (),
) -> pd.DataFrame:
    if metrics.empty or predictions.empty:
        return metrics
    if metric_key not in metrics.columns or "config_id" not in metrics.columns:
        return metrics
    required = {"split_key", "config_id", "y_true", "y_pred"}
    proba_cols = [f"proba_{label}" for label in class_labels]
    if not required.issubset(predictions.columns) or not set(proba_cols).issubset(
        predictions.columns
    ):
        return metrics
    requested = {canonical_metric_name(metric) for metric in requested_metrics}
    out = metrics.copy()
    for metric in METRIC_COLUMNS:
        if metric not in out.columns:
            out[metric] = np.nan
    for column in ("n_samples", "n_subjects"):
        if column not in out.columns:
            out[column] = np.nan
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for (split_key, config_id), group in predictions.groupby(
        ["split_key", "config_id"], sort=False
    ):
        y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(group["y_pred"], errors="coerce").to_numpy(dtype=float)
        proba = (
            group[proba_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        if (
            len(y_true) == 0
            or not np.all(np.isfinite(y_true))
            or not np.all(np.isfinite(y_pred))
            or not np.all(np.isfinite(proba))
        ):
            continue
        values = compute_metrics(y_true.astype(int), y_pred.astype(int), proba, classes)
        if "subject_macro_log_loss" in requested and "subject_id" in group.columns:
            values["subject_macro_log_loss"] = grouped_log_loss(
                y_true.astype(int),
                proba,
                classes,
                group["subject_id"].astype(str).to_numpy(dtype=object),
            )
        if "cohort_macro_log_loss" in requested and str(split_key).startswith("lodo_"):
            values["cohort_macro_log_loss"] = float(values["log_loss"])
        values["n_samples"] = int(len(group))
        values["n_subjects"] = int(
            group["subject_id"].astype(str).nunique()
            if "subject_id" in group.columns
            else len(group)
        )
        lookup[(str(split_key), str(config_id))] = values
    ok = (
        pd.to_numeric(out["ok"], errors="coerce").fillna(0).astype(int).eq(1)
        if "ok" in out.columns
        else pd.Series(True, index=out.index)
    )
    for index in out.index[ok]:
        key = (str(out.at[index, metric_key]), str(out.at[index, "config_id"]))
        values = lookup.get(key)
        if values is None:
            continue
        for metric in METRIC_COLUMNS:
            current = pd.to_numeric(
                pd.Series([out.at[index, metric]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(current):
                value = float(values.get(metric, np.nan))
                if np.isfinite(value):
                    out.at[index, metric] = value
        for column in ("n_samples", "n_subjects"):
            current = pd.to_numeric(
                pd.Series([out.at[index, column]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(current):
                out.at[index, column] = float(values[column])
    return out


from .evaluation_splits import (
    _groups_from_metadata,
    _inner_splits,
    _normalise_column_names,
    _outer_splits,
    _regression_inner_splits,
    _regression_outer_splits,
    _resolved_evaluation_splits,
    _safe_group_n_splits,
    _safe_n_splits,
    _strata_from_metadata,
    _stratification_error_context,
    _subject_safe_groups,
)
from .target_sweeps import (
    _explainability_for_target,
    _learners_for_target,
    _metric_for_target,
    _target_columns,
    _target_task,
    target_sweeps,
)


def _regression_metric_row(
    metrics: dict[str, float],
    split_key: str,
    inner_key: str | None,
    cid: str,
    mpdr: MPDR,
    learner: str,
    stage_name: str,
) -> dict[str, Any]:
    row = _metric_row(metrics, split_key, inner_key, cid, mpdr, learner, stage_name)
    row["task"] = "regression"
    return row


def _regression_prediction_rows(
    key: str,
    cid: str,
    idx: np.ndarray,
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    y: np.ndarray,
    pred: np.ndarray,
    stage_name: str,
    outer_split_key: str,
    target_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_no, sample_idx in enumerate(idx):
        i = int(sample_idx)
        rows.append(
            {
                "stage": stage_name,
                "split_key": key,
                "outer_split_key": outer_split_key,
                "sample_id": str(sample_ids[i]),
                "subject_id": str(subject_ids[i]),
                "sample_index": i,
                "config_id": cid,
                "y_true": float(y[i]),
                "y_pred": float(pred[row_no]),
                "task": "regression",
                "target": str(target_name),
            }
        )
    return rows


def _evaluate_regression_split_task(
    X_base: np.ndarray,
    feature_blocks: Any,
    y: np.ndarray,
    sample_ids: Sequence[str],
    subject_ids: Sequence[str],
    target_name: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    split_key: str,
    protocol: str,
    res_name: str,
    levels: Sequence[str],
    ct_name: str,
    ct_item: Any,
    transformation_factory_builder: Any,
    learner_name: str,
    learner_factory: Any,
    cid: str,
    gate_enabled: bool,
    gate_metric: str,
    gate_threshold: float | None,
    existing_inner_keys: set[str],
    existing_inner_scores: Sequence[dict[str, Any]],
    existing_qualification: dict[str, Any] | None,
    needs_outer: bool,
    random_state: int,
    threads_per_worker: int,
    resource_sample_interval_s: float,
) -> dict[str, Any]:
    gate_metric_input = str(gate_metric)
    gate_metric = canonical_metric_name(gate_metric_input)
    tracker = ResourceTracker(sample_interval_s=resource_sample_interval_s).start()
    mpdr = MPDR(
        resolution=res_name, levels=tuple(levels), count_transformation=str(ct_name)
    )
    result = {
        "split_key": split_key,
        "config_id": cid,
        "inner_metrics": [],
        "inner_predictions": [],
        "outer_metrics": [],
        "outer_predictions": [],
        "qualification": [],
        "job_resources": [],
        "fits": 0,
    }
    score_rows = [dict(x) for x in existing_inner_scores if isinstance(x, dict)]
    with (
        thread_environment(threads_per_worker),
        threadpool_limits(limits=max(1, int(threads_per_worker))),
    ):
        for inner_no, (inner_train_local, inner_val_local) in enumerate(inner_splits):
            inner_key = f"{split_key}__i{inner_no}"
            if inner_key in existing_inner_keys:
                continue
            tr_idx = train_idx[np.asarray(inner_train_local, dtype=int)]
            va_idx = train_idx[np.asarray(inner_val_local, dtype=int)]
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            X_inner_train, X_inner_val, feature_mask = _lodo_feature_pair(
                X_base, tr_idx, va_idx, protocol
            )
            inner_blocks = mask_feature_blocks(feature_blocks, feature_mask)
            _, inner_ct_factory = transformation_factory_builder(
                ct_item,
                random_state=random_state,
                feature_blocks=inner_blocks,
            )
            fitted = inner_ct_factory()
            try:
                X_tr, X_va = fitted.apply_pair(X_inner_train, X_inner_val)
                reg = configure_estimator_threads(learner_factory(), threads_per_worker)
                reg.fit(X_tr, y[tr_idx])
                pred = np.asarray(
                    _estimator_call(reg, "predict", X_va), dtype=float
                ).reshape(-1)
                metrics = compute_regression_metrics(y[va_idx], pred)
                row = _regression_metric_row(
                    metrics, split_key, inner_key, cid, mpdr, learner_name, "inner"
                )
                row.update(fitted.feature_filter_metadata())
                result["inner_metrics"].append(row)
                result["inner_predictions"].extend(
                    _regression_prediction_rows(
                        inner_key,
                        cid,
                        va_idx,
                        sample_ids,
                        subject_ids,
                        y,
                        pred,
                        "inner",
                        split_key,
                        target_name,
                    )
                )
                score = float(metrics.get(gate_metric, np.nan))
                if np.isfinite(score):
                    score_rows.append(dict(row))
                result["fits"] += 1
            except Exception as exc:
                row = _failed_metric_row(
                    split_key, inner_key, cid, mpdr, learner_name, "inner", exc
                )
                row["task"] = "regression"
                result["inner_metrics"].append(row)
        qualified = True
        gate_score = float("nan")
        if gate_enabled:
            if existing_qualification is not None:
                qualified = bool(int(existing_qualification.get("qualified", 0)))
                gate_score = float(existing_qualification.get("inner_score", np.nan))
            else:
                gate_score, _ = aggregate_validation_metric(
                    pd.DataFrame(score_rows), gate_metric
                )
                gate = QualificationGate(
                    enabled=True, metric=gate_metric_input, threshold=gate_threshold
                )
                qualified = gate.qualifies(gate_score)
                result["qualification"].append(
                    {
                        "split_key": split_key,
                        "config_id": cid,
                        "mpdr_id": _mpdr_id(ct_name, res_name),
                        "count_transformation": str(ct_name),
                        "resolution": res_name,
                        "levels": ",".join(levels),
                        "learner": learner_name,
                        "gate_enabled": 1,
                        "gate_metric": gate_metric,
                        "gate_threshold": gate_threshold,
                        "inner_score": gate_score,
                        "qualified": int(qualified),
                        "task": "regression",
                    }
                )
        if needs_outer and qualified:
            X_outer_train, X_outer_test, feature_mask = _lodo_feature_pair(
                X_base, train_idx, test_idx, protocol
            )
            outer_blocks = mask_feature_blocks(feature_blocks, feature_mask)
            _, outer_ct_factory = transformation_factory_builder(
                ct_item,
                random_state=random_state,
                feature_blocks=outer_blocks,
            )
            fitted_outer = outer_ct_factory()
            try:
                X_train, X_test = fitted_outer.apply_pair(X_outer_train, X_outer_test)
                reg = configure_estimator_threads(learner_factory(), threads_per_worker)
                reg.fit(X_train, y[train_idx])
                pred = np.asarray(
                    _estimator_call(reg, "predict", X_test), dtype=float
                ).reshape(-1)
                metrics = compute_regression_metrics(y[test_idx], pred)
                row = _regression_metric_row(
                    metrics, split_key, None, cid, mpdr, learner_name, "outer"
                )
                row.update(fitted_outer.feature_filter_metadata())
                result["outer_metrics"].append(row)
                result["outer_predictions"].extend(
                    _regression_prediction_rows(
                        split_key,
                        cid,
                        test_idx,
                        sample_ids,
                        subject_ids,
                        y,
                        pred,
                        "outer",
                        split_key,
                        target_name,
                    )
                )
                result["fits"] += 1
            except Exception as exc:
                row = _failed_metric_row(
                    split_key, None, cid, mpdr, learner_name, "outer", exc
                )
                row["task"] = "regression"
                result["outer_metrics"].append(row)
    measured = tracker.stop()
    result["job_resources"] = [
        {
            "split_key": str(split_key),
            "config_id": str(cid),
            "resolution": str(res_name),
            "count_transformation": str(ct_name),
            "learner": str(learner_name),
            "fits": int(result.get("fits", 0)),
            "threads_per_worker": int(threads_per_worker),
            "task": "regression",
            **measured,
        }
    ]
    result["elapsed_s"] = float(measured.get("wall_time_s", 0.0))
    return result


def _backfill_regression_metrics_from_predictions(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    metric_key: str,
) -> pd.DataFrame:
    if metrics.empty or predictions.empty:
        return metrics
    if metric_key not in metrics.columns or "config_id" not in metrics.columns:
        return metrics
    required = {"split_key", "config_id", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        return metrics
    out = metrics.copy()
    for metric in METRIC_COLUMNS:
        if metric not in out.columns:
            out[metric] = np.nan
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for (split_key, config_id), group in predictions.groupby(
        ["split_key", "config_id"], sort=False
    ):
        y_true = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(group["y_pred"], errors="coerce").to_numpy(dtype=float)
        if (
            len(y_true) == 0
            or not np.all(np.isfinite(y_true))
            or not np.all(np.isfinite(y_pred))
        ):
            continue
        lookup[(str(split_key), str(config_id))] = compute_regression_metrics(
            y_true, y_pred
        )
    ok = (
        pd.to_numeric(out["ok"], errors="coerce").fillna(0).astype(int).eq(1)
        if "ok" in out.columns
        else pd.Series(True, index=out.index)
    )
    for index in out.index[ok]:
        key = (str(out.at[index, metric_key]), str(out.at[index, "config_id"]))
        values = lookup.get(key)
        if values is None:
            continue
        for metric, value in values.items():
            current = pd.to_numeric(
                pd.Series([out.at[index, metric]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(current) and np.isfinite(value):
                out.at[index, metric] = float(value)
    return out


def _evaluate_regression(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    _prepare_dirs(root)
    resolutions = [_parse_resolution(r) for r in sweep.resolutions]
    levels = tuple(
        dict.fromkeys(
            lv
            for _, values in resolutions
            for lv in values
            if lv not in {"all", "features", "asis", "raw"}
        )
    )
    dataset = load_dataset(sweep.data, levels or ("all",))
    _validate_dataset_identity(root, dataset)
    learners = [_learner_factory(item, task="regression") for item in sweep.learners]
    learner_fingerprints = {
        _learner_name(item): _learner_fingerprint(item) for item in sweep.learners
    }
    mpdr_cache = {}
    for name, lvls in resolutions:
        matrix, _, blocks = materialize_mpdr_with_blocks(dataset, lvls)
        mpdr_cache[name] = (np.asarray(matrix, dtype=np.float32), blocks)
    feature_blocks_by_resolution = {
        name: blocks for name, (_, blocks) in mpdr_cache.items()
    }
    transformation_specs_by_resolution = {
        name: _count_transformation_specs_for_blocks(
            sweep.count_transformations, feature_blocks_by_resolution[name]
        )
        for name, _ in resolutions
    }
    configs = build_sweep_configs(
        sweep.resolutions,
        sweep.count_transformations,
        sweep.learners,
        resolution_feature_blocks=feature_blocks_by_resolution,
    )
    evaluation_cache_fingerprint = _evaluation_cache_fingerprint(sweep, dataset)
    evaluation_fingerprint = _evaluation_fingerprint(sweep, dataset, configs)
    _validate_incremental_experiment_identity(
        root,
        sweep,
        evaluation_cache_fingerprint,
        redo=bool(sweep.evaluation.redo),
    )
    if sweep.evaluation.redo:
        _clear_evaluation_checkpoints(root)
    _write_config_table(root, configs)
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    outer_splits, inner_splits_by_outer = _resolved_evaluation_splits(
        root,
        sweep.evaluation,
        dataset,
        groups,
        group_col=sweep.data.group_col,
    )
    _write_manifest(
        root,
        sweep,
        dataset,
        evaluation_fingerprint=evaluation_fingerprint,
        evaluation_cache_fingerprint=evaluation_cache_fingerprint,
    )
    current_config_ids = set(configs["config_id"].astype(str))
    current_outer_keys = {str(split["split_key"]) for split in outer_splits}
    existing = _load_existing_evaluation(
        root,
        current_config_ids,
        current_outer_keys,
        redo=sweep.evaluation.redo,
    )
    existing["inner_metrics"] = _backfill_regression_metrics_from_predictions(
        existing["inner_metrics"], existing["inner_predictions"], "inner_key"
    )
    existing["outer_metrics"] = _backfill_regression_metrics_from_predictions(
        existing["outer_metrics"], existing["outer_predictions"], "split_key"
    )
    if not sweep.gate.enabled:
        existing["qualification"] = pd.DataFrame()
        qpath = root / "tables" / "qualification_gate.parquet"
        if table_exists(qpath):
            remove_table(qpath)
    inner_done = _done_pairs(existing["inner_metrics"], "inner_key")
    outer_done = _done_pairs(existing["outer_metrics"], "split_key")
    qualification_map = (
        _qualification_map(existing["qualification"]) if sweep.gate.enabled else {}
    )
    tasks = []
    split_task_counts: dict[str, int] = {}
    for split in outer_splits:
        split_key = str(split["split_key"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        inner_splits = inner_splits_by_outer.get(split_key, [])
        inner_keys = [
            f"{split_key}__i{inner_no}" for inner_no in range(len(inner_splits))
        ]
        for res_name, lvls in resolutions:
            X_base, feature_blocks = mpdr_cache[res_name]
            resolution_fingerprint = _resolution_fingerprint(
                res_name, lvls, feature_blocks
            )
            for ct_name, ct_spec in transformation_specs_by_resolution[res_name]:
                ct_item = (ct_name, ct_spec) if ct_spec is not None else ct_name
                transformation_fingerprint = _transformation_fingerprint(
                    str(ct_name), ct_spec
                )
                for learner_name, learner_factory in learners:
                    cid = _config_id(
                        str(ct_name),
                        res_name,
                        learner_name,
                        learner_fingerprints[learner_name],
                        transformation_fingerprint,
                        resolution_fingerprint,
                    )
                    missing_inner = {
                        key for key in inner_keys if (key, cid) not in inner_done
                    }
                    needs_outer = not _outer_pair_complete(
                        split_key, cid, outer_done, qualification_map
                    )
                    if not missing_inner and not needs_outer:
                        continue
                    task = delayed(_evaluate_regression_split_task)(
                        X_base,
                        feature_blocks,
                        dataset.y,
                        tuple(dataset.sample_ids),
                        tuple(dataset.subject_ids),
                        dataset.target_name,
                        train_idx,
                        test_idx,
                        tuple(inner_splits),
                        split_key,
                        sweep.evaluation.protocol,
                        res_name,
                        tuple(lvls),
                        str(ct_name),
                        ct_item,
                        _count_transformation_factory,
                        learner_name,
                        learner_factory,
                        cid,
                        bool(sweep.gate.enabled),
                        str(sweep.gate.metric),
                        sweep.gate.threshold,
                        set(inner_keys) - missing_inner,
                        _existing_inner_scores(
                            existing["inner_metrics"],
                            split_key,
                            cid,
                            sweep.gate.metric,
                        ),
                        qualification_map.get((split_key, cid)),
                        bool(needs_outer),
                        int(sweep.evaluation.random_state),
                        1,
                        float(sweep.evaluation.resource_sample_interval_s),
                    )
                    tasks.append((split_key, cid, task))
                    split_task_counts[split_key] = (
                        split_task_counts.get(split_key, 0) + 1
                    )
    expected_pairs = len(outer_splits) * len(configs)
    completed_pairs = sum(
        1
        for split in outer_splits
        for cid in current_config_ids
        if _outer_pair_complete(
            str(split["split_key"]), cid, outer_done, qualification_map
        )
    )
    execution = resolve_execution_plan(
        sweep.evaluation.n_jobs,
        len(tasks) or 1,
        backend=sweep.evaluation.parallel_backend,
        memory_fraction=sweep.evaluation.memory_fraction,
        min_worker_memory_gib=sweep.evaluation.min_worker_memory_gib,
    )
    prepared_tasks = []
    for split_key, cid, task in tasks:
        fn, args, kwargs = task
        args = list(args)
        args[-2] = execution.threads_per_worker
        prepared_tasks.append((split_key, cid, fn, tuple(args), kwargs))
    stage("Regression configuration sweep", sweep.title)
    memory_gib = (
        "unknown"
        if execution.memory_bytes is None
        else f"{execution.memory_bytes / (1024**3):.1f} GiB"
    )
    y = np.asarray(dataset.y, dtype=float)
    summary_table(
        "Sweep overview",
        {
            "task": "regression",
            "target": dataset.target_name,
            "samples": f"{len(y):,}",
            "target range": f"{float(np.min(y)):.4g} to {float(np.max(y)):.4g}",
            "MPDRs": f"{configs['mpdr_id'].nunique():,}",
            "MPMAs": f"{len(configs):,}",
            "protocol": sweep.evaluation.protocol,
            "outer splits": f"{len(outer_splits):,}",
            "inner folds": _inner_validation_label(sweep.evaluation),
            "selection metric": sweep.evaluation.optimize_metric,
            "gate": "on" if sweep.gate.enabled else "off",
            "completed MPMA/split pairs": f"{completed_pairs:,}/{expected_pairs:,}",
            "pending jobs": f"{len(prepared_tasks):,}",
            "CPU logical/physical": f"{execution.logical_cpus}/{execution.physical_cpus}",
            "available memory": memory_gib,
            "workers": execution.workers,
            "threads per worker": execution.threads_per_worker,
            "parallel backend": execution.backend,
            "experiment dir": root,
        },
    )
    if completed_pairs and completed_pairs < expected_pairs:
        info(
            "Resuming regression sweep: completed MPMA/split pairs are kept; only new or missing pairs will be evaluated."
        )
    if not prepared_tasks:
        success(
            "Current regression sweep is already complete; no evaluation jobs to run."
        )
        _write_tables(
            root,
            [],
            [],
            [],
            [],
            [],
            [],
            existing=existing,
            gate_enabled=sweep.gate.enabled,
        )
        selection_tracker = ResourceTracker(
            sample_interval_s=sweep.evaluation.resource_sample_interval_s
        ).start()
        write_mpma_b_selection_outputs(
            root, str(sweep.evaluation.optimize_metric), plan=sweep.ensemble
        )
        dump_json_standard(
            selection_tracker.stop(),
            root / "tables" / "mpma_b_selection_resources.json",
        )
        _write_rankings_and_figures(root, [], str(sweep.evaluation.optimize_metric))
        _write_representation_impact_figure(
            root, metric_col=str(sweep.evaluation.optimize_metric)
        )
        outputs = _existing_outputs(root)
        path_table("Regression sweep outputs", outputs)
        return outputs
    t0 = time.perf_counter()
    outer_metric_rows: list[dict[str, Any]] = []
    inner_metric_rows: list[dict[str, Any]] = []
    outer_pred_rows: list[dict[str, Any]] = []
    inner_pred_rows: list[dict[str, Any]] = []
    qualification_rows: list[dict[str, Any]] = []
    job_resource_rows: list[dict[str, Any]] = []
    completed_by_split: dict[str, int] = {}
    with progress() as prog:
        job_task = prog.add_task(
            "Regression MPMA/split jobs", total=len(prepared_tasks)
        )
        split_task = prog.add_task(
            "Outer splits completed", total=len(split_task_counts)
        )
        task_payloads = [
            (fn, args, kwargs) for _, _, fn, args, kwargs in prepared_tasks
        ]
        for result in iter_parallel_tasks(task_payloads, execution):
            _checkpoint_result(root, result)
            inner_metric_rows.extend(result.get("inner_metrics", []))
            inner_pred_rows.extend(result.get("inner_predictions", []))
            outer_metric_rows.extend(result.get("outer_metrics", []))
            outer_pred_rows.extend(result.get("outer_predictions", []))
            qualification_rows.extend(result.get("qualification", []))
            job_resource_rows.extend(result.get("job_resources", []))
            split_key = str(result.get("split_key", ""))
            config_id = str(result.get("config_id", ""))
            completed_by_split[split_key] = completed_by_split.get(split_key, 0) + 1
            if completed_by_split[split_key] == split_task_counts.get(split_key, 0):
                prog.advance(split_task)
            prog.update(
                job_task,
                advance=1,
                description=f"Regression MPMA/split jobs · {split_key} · {config_id}",
            )
    _write_tables(
        root,
        outer_metric_rows,
        inner_metric_rows,
        outer_pred_rows,
        inner_pred_rows,
        qualification_rows,
        job_resource_rows,
        existing=existing,
        gate_enabled=sweep.gate.enabled,
    )
    selection_tracker = ResourceTracker(
        sample_interval_s=sweep.evaluation.resource_sample_interval_s
    ).start()
    write_mpma_b_selection_outputs(
        root, str(sweep.evaluation.optimize_metric), plan=sweep.ensemble
    )
    dump_json_standard(
        selection_tracker.stop(), root / "tables" / "mpma_b_selection_resources.json"
    )
    _write_rankings_and_figures(root, [], str(sweep.evaluation.optimize_metric))
    _write_representation_impact_figure(
        root, metric_col=str(sweep.evaluation.optimize_metric)
    )
    elapsed = time.perf_counter() - t0
    dump_json_standard(
        {
            "task": "regression",
            "target": dataset.target_name,
            "elapsed_s": elapsed,
            "workers": execution.workers,
            "threads_per_worker": execution.threads_per_worker,
            "logical_cpus": execution.logical_cpus,
            "physical_cpus": execution.physical_cpus,
            "n_jobs_completed": len(prepared_tasks),
            "n_outer_rows_added": len(outer_metric_rows),
            "n_inner_rows_added": len(inner_metric_rows),
            "n_qualification_rows_added": len(qualification_rows),
            "cpu_core_hours_added": float(
                sum(float(r.get("cpu_core_hours", 0.0)) for r in job_resource_rows)
            ),
            "model_fits_added": int(
                sum(int(r.get("fits", 0)) for r in job_resource_rows)
            ),
            "peak_job_rss_gib": float(
                max(
                    [float(r.get("peak_rss_gib", 0.0)) for r in job_resource_rows]
                    or [0.0]
                )
            ),
            "machine": machine_profile(execution.logical_cpus, execution.physical_cpus),
        },
        root / "run_summary.json",
    )
    success(
        f"Regression sweep completed in {elapsed:.1f}s · workers={execution.workers} · added {len(outer_metric_rows):,} outer rows and {len(inner_metric_rows):,} inner rows"
    )
    outputs = _existing_outputs(root)
    path_table("Regression sweep outputs", outputs)
    return outputs


def _write_multi_target_summary(
    sweep: Sweep, children: Sequence[Sweep]
) -> dict[str, Path]:
    root = sweep.root()
    root.mkdir(parents=True, exist_ok=True)
    targets = []
    for child in children:
        target = str(child.data.target_col)
        task = str(child.data.task)
        targets.append(
            {
                "target": target,
                "task": task,
                "experiment_dir": str(child.root()),
                "outer_metrics": str(
                    child.root() / "results" / "outer_results.parquet"
                ),
                "outer_predictions": str(
                    child.root() / "predictions" / "outer_predictions.parquet"
                ),
            }
        )
    dump_json_standard(
        {"task": _normalise_sweep_task(sweep.data.task), "targets": targets},
        root / "multi_target_manifest.json",
    )
    return {"manifest": root / "multi_target_manifest.json"}


def evaluate(sweep: Sweep) -> dict[str, Path]:
    if sweep.uses_modalities:
        from .multimodal_sweep import evaluate_modality_sweep

        return evaluate_modality_sweep(sweep)
    children = target_sweeps(sweep)
    if len(children) > 1 or children[0] is not sweep:
        for index, child in enumerate(children, start=1):
            info(
                f"Multi-target evaluation · target {index}/{len(children)} · {child.data.target_col}"
            )
            evaluate(child)
        info("Multi-target evaluation · writing target manifest")
        return _write_multi_target_summary(sweep, children)
    task = _target_task(sweep.data, _target_columns(sweep.data)[0])
    if task == "regression":
        return _evaluate_regression(sweep)
    return _evaluate_classification(sweep)


def _evaluate_classification(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    _prepare_dirs(root)
    resolutions = [_parse_resolution(r) for r in sweep.resolutions]
    all_levels = tuple(
        dict.fromkeys(
            lv
            for _, levels in resolutions
            for lv in levels
            if lv not in {"all", "features", "asis", "raw"}
        )
    )
    dataset = load_dataset(sweep.data, all_levels or ("all",))
    validate_sweep_class_count(sweep, len(dataset.class_labels))
    _validate_dataset_identity(root, dataset)
    learner_factories = [_learner_factory(x) for x in sweep.learners]
    learner_fingerprints = {
        _learner_name(item): _learner_fingerprint(item) for item in sweep.learners
    }
    mpdr_cache = {}
    for res_name, levels in resolutions:
        matrix, _, blocks = materialize_mpdr_with_blocks(dataset, levels)
        mpdr_cache[res_name] = (np.asarray(matrix, dtype=np.float32), blocks)
    feature_blocks_by_resolution = {
        name: blocks for name, (_, blocks) in mpdr_cache.items()
    }
    transformation_specs_by_resolution = {
        name: _count_transformation_specs_for_blocks(
            sweep.count_transformations, feature_blocks_by_resolution[name]
        )
        for name, _ in resolutions
    }
    configs = build_sweep_configs(
        sweep.resolutions,
        sweep.count_transformations,
        sweep.learners,
        resolution_feature_blocks=feature_blocks_by_resolution,
    )
    evaluation_cache_fingerprint = _evaluation_cache_fingerprint(sweep, dataset)
    evaluation_fingerprint = _evaluation_fingerprint(sweep, dataset, configs)
    _validate_incremental_experiment_identity(
        root,
        sweep,
        evaluation_cache_fingerprint,
        redo=bool(sweep.evaluation.redo),
    )
    if sweep.evaluation.redo:
        _clear_evaluation_checkpoints(root)
    _write_config_table(root, configs)
    y = dataset.y
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    strata = _strata_from_metadata(dataset.metadata, y, sweep.data.stratify_col)
    outer_splits, inner_splits_by_outer = _resolved_evaluation_splits(
        root,
        sweep.evaluation,
        dataset,
        groups,
        strata,
        sweep.data.stratify_col,
        sweep.data.group_col,
    )
    _write_manifest(
        root,
        sweep,
        dataset,
        evaluation_fingerprint=evaluation_fingerprint,
        evaluation_cache_fingerprint=evaluation_cache_fingerprint,
    )
    current_config_ids = set(configs["config_id"].astype(str))
    current_outer_keys = {str(s["split_key"]) for s in outer_splits}
    existing = _load_existing_evaluation(
        root,
        current_config_ids,
        current_outer_keys,
        redo=sweep.evaluation.redo,
    )
    requested_metrics = [sweep.evaluation.optimize_metric]
    if sweep.gate.enabled:
        requested_metrics.append(sweep.gate.metric)
    existing["inner_metrics"] = _backfill_metrics_from_predictions(
        existing["inner_metrics"],
        existing["inner_predictions"],
        dataset.classes,
        dataset.class_labels,
        "inner_key",
        requested_metrics,
    )
    existing["outer_metrics"] = _backfill_metrics_from_predictions(
        existing["outer_metrics"],
        existing["outer_predictions"],
        dataset.classes,
        dataset.class_labels,
        "split_key",
        requested_metrics,
    )
    if not sweep.gate.enabled:
        existing["qualification"] = pd.DataFrame()
        qpath = root / "tables" / "qualification_gate.parquet"
        if table_exists(qpath):
            remove_table(qpath)
    inner_done = _done_pairs(existing["inner_metrics"], "inner_key")
    outer_done = _done_pairs(existing["outer_metrics"], "split_key")
    qualification_map = (
        _qualification_map(existing["qualification"]) if sweep.gate.enabled else {}
    )
    tasks = []
    split_task_counts: dict[str, int] = {}
    for split in outer_splits:
        split_key = str(split["split_key"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
            continue
        inner_splits = inner_splits_by_outer.get(split_key, [])
        inner_keys = [
            f"{split_key}__i{inner_no}" for inner_no in range(len(inner_splits))
        ]
        for res_name, levels in resolutions:
            X_base, feature_blocks = mpdr_cache[res_name]
            resolution_fingerprint = _resolution_fingerprint(
                res_name, levels, feature_blocks
            )
            for ct_name, ct_spec in transformation_specs_by_resolution[res_name]:
                ct_item = (ct_name, ct_spec) if ct_spec is not None else ct_name
                transformation_fingerprint = _transformation_fingerprint(
                    str(ct_name), ct_spec
                )
                for learner_name, learner_factory in learner_factories:
                    cid = _config_id(
                        str(ct_name),
                        res_name,
                        learner_name,
                        learner_fingerprints[learner_name],
                        transformation_fingerprint,
                        resolution_fingerprint,
                    )
                    missing_inner = {
                        key for key in inner_keys if (key, cid) not in inner_done
                    }
                    needs_outer = not _outer_pair_complete(
                        split_key, cid, outer_done, qualification_map
                    )
                    if not missing_inner and not needs_outer:
                        continue
                    task = delayed(_evaluate_mpma_split_task)(
                        X_base,
                        feature_blocks,
                        y,
                        groups,
                        dataset.classes,
                        tuple(dataset.class_labels),
                        tuple(dataset.sample_ids),
                        tuple(dataset.subject_ids),
                        train_idx,
                        test_idx,
                        tuple(inner_splits),
                        split_key,
                        sweep.evaluation.protocol,
                        res_name,
                        tuple(levels),
                        str(ct_name),
                        ct_item,
                        _count_transformation_factory,
                        learner_name,
                        learner_factory,
                        cid,
                        bool(sweep.gate.enabled),
                        str(sweep.gate.metric),
                        sweep.gate.threshold,
                        str(sweep.evaluation.optimize_metric),
                        set(inner_keys) - missing_inner,
                        _existing_inner_scores(
                            existing["inner_metrics"], split_key, cid, sweep.gate.metric
                        ),
                        qualification_map.get((split_key, cid)),
                        bool(needs_outer),
                        int(sweep.evaluation.random_state),
                        1,
                        float(sweep.evaluation.resource_sample_interval_s),
                    )
                    tasks.append((split_key, cid, task))
                    split_task_counts[split_key] = (
                        split_task_counts.get(split_key, 0) + 1
                    )
    expected_pairs = len(outer_splits) * len(configs)
    completed_pairs = sum(
        1
        for split in outer_splits
        for cid in current_config_ids
        if _outer_pair_complete(
            str(split["split_key"]), cid, outer_done, qualification_map
        )
    )
    execution = resolve_execution_plan(
        sweep.evaluation.n_jobs,
        len(tasks) or 1,
        backend=sweep.evaluation.parallel_backend,
        memory_fraction=sweep.evaluation.memory_fraction,
        min_worker_memory_gib=sweep.evaluation.min_worker_memory_gib,
    )
    prepared_tasks = []
    for split_key, cid, task in tasks:
        fn, args, kwargs = task
        args = list(args)
        args[-2] = execution.threads_per_worker
        prepared_tasks.append((split_key, cid, fn, tuple(args), kwargs))
    stage("Configuration sweep", sweep.title)
    memory_gib = (
        "unknown"
        if execution.memory_bytes is None
        else f"{execution.memory_bytes / (1024**3):.1f} GiB"
    )
    summary_table(
        "Sweep overview",
        {
            "samples": f"{len(y):,}",
            "classes": dataset.class_labels,
            "MPDRs": f"{configs['mpdr_id'].nunique():,}",
            "MPMAs": f"{len(configs):,}",
            "protocol": sweep.evaluation.protocol,
            "stratification": "target"
            if not sweep.data.stratify_col
            else f"target + {sweep.data.stratify_col}",
            "outer splits": f"{len(outer_splits):,}",
            "inner folds": _inner_validation_label(sweep.evaluation),
            "gate": "on" if sweep.gate.enabled else "off",
            "completed MPMA/split pairs": f"{completed_pairs:,}/{expected_pairs:,}",
            "pending jobs": f"{len(prepared_tasks):,}",
            "CPU logical/physical": f"{execution.logical_cpus}/{execution.physical_cpus}",
            "available memory": memory_gib,
            "workers": execution.workers,
            "threads per worker": execution.threads_per_worker,
            "parallel backend": execution.backend,
            "experiment dir": root,
        },
    )
    if completed_pairs and completed_pairs < expected_pairs:
        info(
            "Resuming sweep: completed MPMA/split pairs are kept; only new or missing pairs will be evaluated."
        )
    if not prepared_tasks:
        success(
            "Current sweep configuration is already complete; no evaluation jobs to run."
        )
        _write_tables(
            root,
            [],
            [],
            [],
            [],
            [],
            [],
            existing=existing,
            gate_enabled=sweep.gate.enabled,
        )
        selection_tracker = ResourceTracker(
            sample_interval_s=sweep.evaluation.resource_sample_interval_s
        ).start()
        write_mpma_b_selection_outputs(
            root, sweep.evaluation.optimize_metric, plan=sweep.ensemble
        )
        dump_json_standard(
            selection_tracker.stop(),
            root / "tables" / "mpma_b_selection_resources.json",
        )
        _write_rankings_and_figures(
            root, dataset.class_labels, sweep.evaluation.optimize_metric
        )
        _write_representation_impact_figure(
            root, metric_col=sweep.evaluation.optimize_metric
        )
        path_table("Configuration sweep outputs", _existing_outputs(root))
        return _existing_outputs(root)
    t0 = time.perf_counter()
    outer_metric_rows: list[dict[str, Any]] = []
    inner_metric_rows: list[dict[str, Any]] = []
    outer_pred_rows: list[dict[str, Any]] = []
    inner_pred_rows: list[dict[str, Any]] = []
    qualification_rows: list[dict[str, Any]] = []
    job_resource_rows: list[dict[str, Any]] = []
    completed_by_split: dict[str, int] = {}
    with progress() as prog:
        job_task = prog.add_task("MPMA/split jobs", total=len(prepared_tasks))
        split_task = prog.add_task(
            "Outer splits completed", total=len(split_task_counts)
        )
        task_payloads = [
            (fn, args, kwargs) for _, _, fn, args, kwargs in prepared_tasks
        ]
        for result in iter_parallel_tasks(task_payloads, execution):
            _checkpoint_result(root, result)
            inner_metric_rows.extend(result.get("inner_metrics", []))
            inner_pred_rows.extend(result.get("inner_predictions", []))
            outer_metric_rows.extend(result.get("outer_metrics", []))
            outer_pred_rows.extend(result.get("outer_predictions", []))
            qualification_rows.extend(result.get("qualification", []))
            job_resource_rows.extend(result.get("job_resources", []))
            split_key = str(result.get("split_key", ""))
            config_id = str(result.get("config_id", ""))
            completed_by_split[split_key] = completed_by_split.get(split_key, 0) + 1
            if completed_by_split[split_key] == split_task_counts.get(split_key, 0):
                prog.advance(split_task)
            prog.update(
                job_task,
                advance=1,
                description=f"MPMA/split jobs · {split_key} · {config_id}",
            )
    _write_tables(
        root,
        outer_metric_rows,
        inner_metric_rows,
        outer_pred_rows,
        inner_pred_rows,
        qualification_rows,
        job_resource_rows,
        existing=existing,
        gate_enabled=sweep.gate.enabled,
    )
    selection_tracker = ResourceTracker(
        sample_interval_s=sweep.evaluation.resource_sample_interval_s
    ).start()
    write_mpma_b_selection_outputs(
        root, sweep.evaluation.optimize_metric, plan=sweep.ensemble
    )
    dump_json_standard(
        selection_tracker.stop(), root / "tables" / "mpma_b_selection_resources.json"
    )
    _write_rankings_and_figures(
        root, dataset.class_labels, sweep.evaluation.optimize_metric
    )
    _write_representation_impact_figure(
        root, metric_col=sweep.evaluation.optimize_metric
    )
    elapsed = time.perf_counter() - t0
    dump_json_standard(
        {
            "elapsed_s": elapsed,
            "workers": execution.workers,
            "threads_per_worker": execution.threads_per_worker,
            "logical_cpus": execution.logical_cpus,
            "physical_cpus": execution.physical_cpus,
            "n_jobs_completed": len(prepared_tasks),
            "n_outer_rows_added": len(outer_metric_rows),
            "n_inner_rows_added": len(inner_metric_rows),
            "n_qualification_rows_added": len(qualification_rows),
            "cpu_core_hours_added": float(
                sum(float(r.get("cpu_core_hours", 0.0)) for r in job_resource_rows)
            ),
            "model_fits_added": int(
                sum(int(r.get("fits", 0)) for r in job_resource_rows)
            ),
            "peak_job_rss_gib": float(
                max(
                    [float(r.get("peak_rss_gib", 0.0)) for r in job_resource_rows]
                    or [0.0]
                )
            ),
            "machine": machine_profile(execution.logical_cpus, execution.physical_cpus),
        },
        root / "run_summary.json",
    )
    success(
        f"Configuration sweep completed in {elapsed:.1f}s · workers={execution.workers} · added {len(outer_metric_rows):,} outer rows and {len(inner_metric_rows):,} inner rows"
    )
    outputs = _existing_outputs(root)
    path_table("Configuration sweep outputs", outputs)
    return outputs


def _prepare_dirs(root: Path) -> None:
    for d in [
        "results",
        "predictions",
        "inner_results",
        "inner_predictions",
        "tables",
        "figures",
        "ensembling",
        "explainability",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)


def _existing_outputs(root: Path) -> dict[str, Path]:
    return {
        "experiment_dir": root,
        "configs": root / "configs.db",
        "rankings": root / "tables" / "mpma_inner_rankings.parquet",
        "outer_performance_summary": root
        / "tables"
        / "mpma_outer_performance_summary.parquet",
        "outer_results": root / "results" / "outer_results.parquet",
        "inner_results": root / "inner_results" / "inner_results.parquet",
        "outer_predictions": root / "predictions" / "outer_predictions.parquet",
        "mpma_b_selection": root / "tables" / "mpma_b_outer_selection.parquet",
        "mpma_b_predictions": root / "predictions" / "mpma_b_outer_predictions.parquet",
        "mpma_b_outer_results": root / "results" / "mpma_b_outer_results.parquet",
        "mpma_b_summary": root / "tables" / "mpma_b_strategy_summary.json",
        "mpma_b_final_candidate": root / "tables" / "mpma_b_final_candidate.json",
        "job_resources": root / "tables" / "job_resources.parquet",
        "cv_splits": root / "tables" / "cv_splits.parquet",
        "mpma_b_selection_resources": root
        / "tables"
        / "mpma_b_selection_resources.json",
    }


def _read_table(path: Path) -> pd.DataFrame:
    return read_table(path)


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    write_table(path, frame)


def _filter_existing(df: pd.DataFrame, outer_keys: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "outer_split_key" in out.columns:
        out = out[out["outer_split_key"].astype(str).isin(outer_keys)]
    elif "split_key" in out.columns:
        split_s = out["split_key"].astype(str)
        out = out[
            split_s.isin(outer_keys)
            | split_s.str.rsplit("__i", n=1).str[0].isin(outer_keys)
        ]
    return out.reset_index(drop=True)


def _load_existing_evaluation(
    root: Path, current_config_ids: set[str], outer_keys: set[str], redo: bool = False
) -> dict[str, pd.DataFrame]:
    tables = {
        "outer_metrics": root / "results" / "outer_results.parquet",
        "inner_metrics": root / "inner_results" / "inner_results.parquet",
        "outer_predictions": root / "predictions" / "outer_predictions.parquet",
        "inner_predictions": root / "inner_predictions" / "inner_predictions.parquet",
        "qualification": root / "tables" / "qualification_gate.parquet",
        "job_resources": root / "tables" / "job_resources.parquet",
    }
    base = {
        name: _filter_existing(_read_table(path), outer_keys)
        for name, path in tables.items()
    }
    checkpoints = _load_checkpoint_frames(root, current_config_ids, outer_keys)
    subsets = {
        "outer_metrics": ["split_key", "config_id"],
        "inner_metrics": ["inner_key", "config_id"],
        "outer_predictions": ["split_key", "config_id", "sample_id"],
        "inner_predictions": ["split_key", "config_id", "sample_id"],
        "qualification": ["split_key", "config_id"],
        "job_resources": ["split_key", "config_id"],
    }
    for key in base:
        cp = checkpoints.get(key, pd.DataFrame())
        base[key] = _concat_existing_new(
            base[key],
            cp.to_dict(orient="records") if cp is not None and not cp.empty else [],
            subsets[key],
        )
        if redo and not base[key].empty and "config_id" in base[key].columns:
            base[key] = base[key][
                ~base[key]["config_id"].astype(str).isin(current_config_ids)
            ].reset_index(drop=True)
    return base


def _done_pairs(df: pd.DataFrame, split_col: str) -> set[tuple[str, str]]:
    if df.empty or split_col not in df.columns or "config_id" not in df.columns:
        return set()
    complete = df
    if "ok" in complete.columns:
        ok = pd.to_numeric(complete["ok"], errors="coerce").fillna(0).astype(int)
        complete = complete.loc[ok.eq(1)]
    return set(zip(complete[split_col].astype(str), complete["config_id"].astype(str)))


def _qualification_map(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if df.empty or "split_key" not in df.columns or "config_id" not in df.columns:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        out[(str(row.get("split_key")), str(row.get("config_id")))] = row
    return out


def _outer_pair_complete(
    split_key: str,
    config_id: str,
    outer_done: set[tuple[str, str]],
    qualification_map: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    pair = (str(split_key), str(config_id))
    qrow = qualification_map.get(pair)
    if qrow is not None:
        try:
            if int(qrow.get("qualified", 0)) == 0:
                return True
        except Exception:
            pass
        return pair in outer_done
    return pair in outer_done


def _existing_inner_scores(
    df: pd.DataFrame, outer_split_key: str, config_id: str, metric: str
) -> list[dict[str, Any]]:
    metric = canonical_metric_name(metric)
    if (
        df.empty
        or metric not in df.columns
        or "config_id" not in df.columns
        or "split_key" not in df.columns
    ):
        return []
    sub = df[
        df["config_id"].astype(str).eq(str(config_id))
        & df["split_key"].astype(str).eq(str(outer_split_key))
    ].copy()
    if "ok" in sub.columns:
        ok = pd.to_numeric(sub["ok"], errors="coerce").fillna(0).astype(int)
        sub = sub[ok.eq(1)]
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub[np.isfinite(sub[metric].to_numpy(dtype=float))]
    keep = [
        column
        for column in (metric, "n_samples", "n_subjects")
        if column in sub.columns
    ]
    return sub[keep].to_dict(orient="records")


def _concat_existing_new(
    existing_df: pd.DataFrame, new_rows: list[dict[str, Any]], subset: list[str]
) -> pd.DataFrame:
    frames = []
    if existing_df is not None and not existing_df.empty:
        frames.append(existing_df)
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    keep_subset = [c for c in subset if c in out.columns]
    if keep_subset:
        out = out.drop_duplicates(subset=keep_subset, keep="last")
    return out


def _validate_dataset_identity(root: Path, dataset: Dataset) -> None:
    path = root / "manifest.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Existing experiment manifest cannot be read.") from exc
    stored = payload.get("dataset_fingerprint")
    if not stored:
        return
    current = dataset_fingerprint(dataset)
    if str(stored) != current:
        raise ValueError(
            "Existing experiment results were produced from different dataset content. Use a new experiment directory or restore the original scientific inputs."
        )


_PROVENANCE_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "polars",
    "pyarrow",
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "lime",
    "PyALE",
    "networkx",
    "scikit-bio",
    "rich",
    "flaml",
    "tqdm",
    "psutil",
    "joblib",
    "threadpoolctl",
    "ruff",
)


def _installed_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_tree_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and (path.suffix in {".py", ".R"} or path.name == "py.typed")
    )
    hasher = hashlib.sha256()
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return hasher.hexdigest()


def _git_source_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = None
    dirty = None
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip() or None
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit = os.environ.get("MLLABIOME_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    return {"git_commit": commit, "git_dirty": dirty}


def _software_provenance() -> dict[str, Any]:
    distribution_version = _installed_distribution_version("mllabiome")
    return {
        "mllabiome_source_version": __version__,
        "mllabiome_distribution_version": distribution_version,
        "version_consistent": distribution_version in {None, __version__},
        "source_tree_sha256": _source_tree_sha256(),
        "source_tree_fingerprint_algorithm": "sha256-package-source-v1",
        **_git_source_state(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependencies": {
            name: _installed_distribution_version(name)
            for name in _PROVENANCE_DISTRIBUTIONS
        },
    }


def _write_manifest(
    root: Path,
    sweep: Sweep,
    dataset: Dataset,
    *,
    evaluation_fingerprint: str | None = None,
    evaluation_cache_fingerprint: str | None = None,
) -> None:
    safe_sweep = asdict(sweep)
    for section in ["data"]:
        for k, v in list(safe_sweep[section].items()):
            if isinstance(v, Path):
                safe_sweep[section][k] = str(v)
    safe_sweep["experiment_dir"] = str(safe_sweep["experiment_dir"])
    manifest = {
        "package": "mllabiome",
        "package_version": __version__,
        "software_provenance": _software_provenance(),
        "terminology": {
            "MPDR": "Microbiome Profile Data Representation = taxonomic resolution + count transformation",
            "MPMA": "Microbiome Profile Modelling Algorithm = MPDR + learner",
        },
        "sweep": safe_sweep,
        "task": dataset.task,
        "target": dataset.target_name,
        "class_labels": dataset.class_labels,
        "n_samples": len(dataset.y),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "dataset_fingerprint_algorithm": "sha256-model-input-v2",
        "evaluation_fingerprint": evaluation_fingerprint,
        "evaluation_fingerprint_algorithm": "sha256-scientific-evaluation-v4"
        if evaluation_fingerprint
        else None,
        "evaluation_cache_fingerprint": evaluation_cache_fingerprint,
        "evaluation_cache_fingerprint_algorithm": (
            "sha256-evaluation-cache-context-v1"
            if evaluation_cache_fingerprint
            else None
        ),
        "experiment_fingerprint": _experiment_fingerprint(sweep, evaluation_fingerprint)
        if evaluation_fingerprint
        else None,
        "experiment_fingerprint_algorithm": "sha256-scientific-experiment-v1"
        if evaluation_fingerprint
        else None,
        "cv_splits": "tables/cv_splits.parquet",
        "transformations": [label.key for label in TRANSFORMATION_LABELS],
        "mpdr_semantics": _MPDR_SEMANTICS,
    }
    dump_json_standard(manifest, root / "manifest.json")


def _write_config_table(root: Path, configs: pd.DataFrame) -> None:
    db_path = root / "configs.db"
    conn = sqlite3.connect(db_path)
    extra_columns = {
        "candidate_family": "TEXT",
        "modalities": "TEXT",
        "integration": "TEXT",
        "integration_n_components": "TEXT",
        "taxonomic_blocks": "TEXT",
        "taxonomic_block_count": "INTEGER",
        "representation_scope": "TEXT",
        "feature_filter": "TEXT",
        "prevalence_threshold": "REAL",
        "detection_threshold": "REAL",
        "learner_display": "TEXT",
        "transformation_fingerprint": "TEXT",
        "resolution_fingerprint": "TEXT",
        "learner_fingerprint": "TEXT",
        "learner_class": "TEXT",
        "learner_params": "TEXT",
    }
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS configs (
                config_id TEXT PRIMARY KEY,
                mpdr_id TEXT,
                count_transformation TEXT NOT NULL,
                resolution TEXT NOT NULL,
                levels TEXT NOT NULL,
                learner TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                candidate_family TEXT,
                modalities TEXT,
                integration TEXT,
                integration_n_components TEXT
            )"""
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(configs)")}
        if "transformation_abbreviation" in columns:
            conn.execute("DROP TABLE configs")
            conn.execute(
                """CREATE TABLE configs (
                    config_id TEXT PRIMARY KEY, mpdr_id TEXT, count_transformation TEXT NOT NULL,
                    resolution TEXT NOT NULL, levels TEXT NOT NULL, learner TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1, candidate_family TEXT, modalities TEXT,
                    integration TEXT, integration_n_components TEXT
                )"""
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(configs)")}
        for name, sql_type in extra_columns.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE configs ADD COLUMN {name} {sql_type}")
        conn.execute("UPDATE configs SET active=0")
        for r in configs.to_dict(orient="records"):
            conn.execute(
                """INSERT INTO configs
                   (config_id, mpdr_id, count_transformation, resolution, levels, learner, active, candidate_family, modalities, integration, integration_n_components, taxonomic_blocks, taxonomic_block_count, representation_scope, feature_filter, prevalence_threshold, detection_threshold, learner_display, transformation_fingerprint, resolution_fingerprint, learner_fingerprint, learner_class, learner_params)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(config_id) DO UPDATE SET
                     mpdr_id=excluded.mpdr_id, count_transformation=excluded.count_transformation,
                     resolution=excluded.resolution, levels=excluded.levels, learner=excluded.learner,
                     active=1, candidate_family=excluded.candidate_family, modalities=excluded.modalities,
                     integration=excluded.integration, integration_n_components=excluded.integration_n_components,
                     taxonomic_blocks=excluded.taxonomic_blocks, taxonomic_block_count=excluded.taxonomic_block_count,
                     representation_scope=excluded.representation_scope, feature_filter=excluded.feature_filter, prevalence_threshold=excluded.prevalence_threshold, detection_threshold=excluded.detection_threshold, learner_display=excluded.learner_display,
                     transformation_fingerprint=excluded.transformation_fingerprint, resolution_fingerprint=excluded.resolution_fingerprint, learner_fingerprint=excluded.learner_fingerprint, learner_class=excluded.learner_class,
                     learner_params=excluded.learner_params""",
                (
                    str(r.get("config_id")),
                    str(r.get("mpdr_id", "")),
                    str(r.get("count_transformation")),
                    str(r.get("resolution")),
                    str(r.get("levels")),
                    str(r.get("learner")),
                    None
                    if pd.isna(r.get("candidate_family"))
                    else str(r.get("candidate_family")),
                    None if pd.isna(r.get("modalities")) else str(r.get("modalities")),
                    None
                    if pd.isna(r.get("integration"))
                    else str(r.get("integration")),
                    None
                    if pd.isna(r.get("integration_n_components"))
                    else str(r.get("integration_n_components")),
                    None
                    if pd.isna(r.get("taxonomic_blocks"))
                    else str(r.get("taxonomic_blocks")),
                    None
                    if pd.isna(r.get("taxonomic_block_count"))
                    else int(r.get("taxonomic_block_count")),
                    None
                    if pd.isna(r.get("representation_scope"))
                    else str(r.get("representation_scope")),
                    None
                    if pd.isna(r.get("feature_filter"))
                    else str(r.get("feature_filter")),
                    None
                    if pd.isna(r.get("prevalence_threshold"))
                    else float(r.get("prevalence_threshold")),
                    None
                    if pd.isna(r.get("detection_threshold"))
                    else float(r.get("detection_threshold")),
                    None
                    if pd.isna(r.get("learner_display"))
                    else str(r.get("learner_display")),
                    None
                    if pd.isna(r.get("transformation_fingerprint"))
                    else str(r.get("transformation_fingerprint")),
                    None
                    if pd.isna(r.get("resolution_fingerprint"))
                    else str(r.get("resolution_fingerprint")),
                    None
                    if pd.isna(r.get("learner_fingerprint"))
                    else str(r.get("learner_fingerprint")),
                    None
                    if pd.isna(r.get("learner_class"))
                    else str(r.get("learner_class")),
                    None
                    if pd.isna(r.get("learner_params"))
                    else str(r.get("learner_params")),
                ),
            )
        conn.commit()
        export = pd.read_sql_query(
            "SELECT * FROM configs ORDER BY active DESC, count_transformation, resolution, learner",
            conn,
        )
    finally:
        conn.close()
    write_table(root / "configs.parquet", export)


def _predict_proba_aligned(
    clf: BaseEstimator, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    return _metrics_predict_proba_aligned(clf, X, classes)


def _metric_row(
    metrics: dict[str, float],
    split_key: str,
    inner_key: str | None,
    cid: str,
    mpdr: MPDR,
    learner: str,
    stage: str,
) -> dict[str, Any]:
    row = {
        "stage": stage,
        "split_key": split_key,
        "inner_key": inner_key or "",
        "config_id": cid,
        "mpdr_id": _mpdr_id(mpdr.count_transformation, mpdr.resolution),
        "count_transformation": mpdr.count_transformation,
        "resolution": mpdr.resolution,
        "levels": ",".join(mpdr.levels),
        "learner": learner,
        "ok": 1,
        "error": "",
    }
    row.update(metrics)
    return row


def _failed_metric_row(
    split_key: str,
    inner_key: str | None,
    cid: str,
    mpdr: MPDR,
    learner: str,
    stage: str,
    exc: Exception,
) -> dict[str, Any]:
    row = _metric_row(
        {k: float("nan") for k in METRIC_COLUMNS},
        split_key,
        inner_key,
        cid,
        mpdr,
        learner,
        stage,
    )
    row["ok"] = 0
    row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _write_tables(
    root: Path,
    outer_metrics: list[dict[str, Any]],
    inner_metrics: list[dict[str, Any]],
    outer_preds: list[dict[str, Any]],
    inner_preds: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    job_resources: list[dict[str, Any]],
    existing: dict[str, pd.DataFrame] | None = None,
    gate_enabled: bool = False,
) -> None:
    existing = existing or {}
    outer_df = _concat_existing_new(
        existing.get("outer_metrics", pd.DataFrame()),
        outer_metrics,
        ["split_key", "config_id"],
    )
    inner_df = _concat_existing_new(
        existing.get("inner_metrics", pd.DataFrame()),
        inner_metrics,
        ["inner_key", "config_id"],
    )
    outer_pred_df = _concat_existing_new(
        existing.get("outer_predictions", pd.DataFrame()),
        outer_preds,
        ["split_key", "config_id", "sample_id"],
    )
    inner_pred_df = _concat_existing_new(
        existing.get("inner_predictions", pd.DataFrame()),
        inner_preds,
        ["split_key", "config_id", "sample_id"],
    )
    qual_df = (
        _concat_existing_new(
            existing.get("qualification", pd.DataFrame()),
            qualification,
            ["split_key", "config_id"],
        )
        if gate_enabled
        else pd.DataFrame()
    )
    resource_df = _concat_existing_new(
        existing.get("job_resources", pd.DataFrame()),
        job_resources,
        ["split_key", "config_id"],
    )

    for frame in (outer_df, inner_df):
        if "transformation_abbreviation" in frame.columns:
            frame.drop(columns=["transformation_abbreviation"], inplace=True)
        if "count_transformation" in frame.columns:
            frame["count_transformation"] = frame["count_transformation"].map(
                _count_transformation_name
            )
    _write_dataframe(root / "results" / "outer_results.parquet", outer_df)
    _write_dataframe(root / "inner_results" / "inner_results.parquet", inner_df)
    _write_dataframe(root / "predictions" / "outer_predictions.parquet", outer_pred_df)
    _write_dataframe(
        root / "inner_predictions" / "inner_predictions.parquet", inner_pred_df
    )
    _write_dataframe(root / "tables" / "job_resources.parquet", resource_df)
    qpath = root / "tables" / "qualification_gate.parquet"
    if gate_enabled:
        _write_dataframe(qpath, qual_df)
    elif table_exists(qpath):
        remove_table(qpath)
    _clear_evaluation_checkpoints(root, compact=True)


def _write_rankings_and_figures(
    root: Path, class_labels: list[str], optimize_metric: str
) -> None:
    metric = canonical_metric_name(optimize_metric)
    inner_path = root / "inner_results" / "inner_results.parquet"
    configs_path = root / "configs.parquet"
    ranking_path = root / "tables" / "mpma_inner_rankings.parquet"
    legacy_path = root / "tables" / "mpma_rankings.parquet"
    if table_exists(inner_path) and table_exists(configs_path):
        inner = read_table(inner_path)
        configs = read_table(configs_path)
        if not inner.empty and metric in inner.columns and "config_id" in inner.columns:
            frame = inner.copy()
            frame["config_id"] = frame["config_id"].astype(str)
            expected = (
                set(frame["inner_key"].dropna().astype(str))
                if "inner_key" in frame.columns
                else set()
            )
            if "ok" in frame.columns:
                ok = pd.to_numeric(frame["ok"], errors="coerce").fillna(0).astype(int)
                frame = frame[ok.eq(1)]
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
            frame = frame[np.isfinite(frame[metric].to_numpy(dtype=float))]
            rows: list[dict[str, Any]] = []
            for config_id, group in frame.groupby("config_id", sort=True):
                if expected and "inner_key" in group.columns:
                    group = group.drop_duplicates("inner_key", keep="last")
                    if set(group["inner_key"].astype(str)) != expected:
                        continue
                score, spread = aggregate_validation_metric(group, metric)
                if not np.isfinite(score):
                    continue
                rows.append(
                    {
                        "config_id": str(config_id),
                        "selection_metric": metric,
                        "inner_score": float(score),
                        "inner_score_std": float(spread),
                        "n_inner_folds": int(len(group)),
                    }
                )
            ranking = pd.DataFrame(rows)
            if not ranking.empty:
                metadata = configs.drop_duplicates("config_id", keep="last").copy()
                metadata["config_id"] = metadata["config_id"].astype(str)
                ranking = ranking.merge(
                    metadata, on="config_id", how="left", validate="one_to_one"
                )
                ranking = ranking.sort_values(
                    ["inner_score", "config_id"],
                    ascending=[metric_is_loss(metric), True],
                    kind="mergesort",
                ).reset_index(drop=True)
                ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
                write_table(ranking_path, ranking)
            elif table_exists(ranking_path):
                remove_table(ranking_path)
        elif table_exists(ranking_path):
            remove_table(ranking_path)
    if table_exists(legacy_path):
        remove_table(legacy_path)
    outer_path = root / "results" / "outer_results.parquet"
    summary_path = root / "tables" / "mpma_outer_performance_summary.parquet"
    if not table_exists(outer_path):
        if table_exists(summary_path):
            remove_table(summary_path)
        return
    outer = read_table(outer_path)
    if outer.empty or "config_id" not in outer.columns:
        if table_exists(summary_path):
            remove_table(summary_path)
        return
    metrics = [column for column in METRIC_COLUMNS if column in outer.columns]
    group_cols = [
        column
        for column in (
            "config_id",
            "mpdr_id",
            "count_transformation",
            "resolution",
            "levels",
            "learner",
        )
        if column in outer.columns
    ]
    valid = outer.copy()
    if "ok" in valid.columns:
        ok = pd.to_numeric(valid["ok"], errors="coerce").fillna(0).astype(int)
        valid = valid[ok.eq(1)]
    if valid.empty or not metrics or not group_cols:
        if table_exists(summary_path):
            remove_table(summary_path)
        return
    summary = valid.groupby(group_cols, dropna=False)[metrics].agg(
        ["mean", "std", "count"]
    )
    summary.columns = [f"{metric_name}_{stat}" for metric_name, stat in summary.columns]
    summary = summary.reset_index()
    write_table(summary_path, summary)
