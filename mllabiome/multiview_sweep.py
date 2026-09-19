from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import delayed
from threadpoolctl import threadpool_limits

from .compute import ResourceTracker, machine_profile
from .console import info, path_table, progress, stage, success, summary_table
from .integrations import Integration, IntegrationModel, integration_view_sets
from .metrics import compute_metrics, compute_regression_metrics
from .resolutions import materialize_mpdr
from .runtime import (
    configure_estimator_threads,
    iter_parallel_tasks,
    resolve_execution_plan,
    thread_environment,
)
from .transformations import _count_transformation_spec
from .utils import dump_json_standard
from .views import ViewDataset, load_views


@dataclass(frozen=True)
class ViewPath:
    view: str
    representation: str
    selector: tuple[str, ...]
    transformation: str
    transformation_item: Any


@dataclass(frozen=True)
class CandidateSpec:
    config_id: str
    family: str
    view_paths: tuple[ViewPath, ...]
    integration: Integration
    n_components: int | None
    learner: str

    @property
    def views(self) -> tuple[str, ...]:
        return tuple(path.view for path in self.view_paths)

    @property
    def representation_label(self) -> str:
        return "+".join(f"{p.view}:{p.representation}" for p in self.view_paths)

    @property
    def transformation_label(self) -> str:
        return "+".join(f"{p.view}:{p.transformation}" for p in self.view_paths)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _candidate_id(
    paths: Sequence[ViewPath],
    integration: Integration,
    n_components: int | None,
    learner: str,
) -> str:
    payload = {
        "paths": [
            (p.view, p.representation, p.selector, p.transformation) for p in paths
        ],
        "integration": integration.key,
        "n_components": n_components,
        "learner": str(learner),
        "semantics": "view_candidate_nested_v1",
    }
    return hashlib.sha1(_canonical(payload).encode()).hexdigest()[:12]


def _representation_specs(
    view: str, representations: Mapping[str, Sequence[Any]] | None
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not representations or view not in representations:
        return (("all", ("all",)),)
    out = []
    for item in representations[view]:
        if isinstance(item, str):
            out.append((str(item), (str(item),)))
        else:
            name, selector = item
            values = (
                (str(selector),)
                if isinstance(selector, str)
                else tuple(str(x) for x in selector)
            )
            out.append((str(name), values))
    return tuple(out)


def _transformation_specs(
    view: str, transformations: Mapping[str, Sequence[Any]] | None
) -> tuple[tuple[str, Any], ...]:
    items = (
        ("identity",)
        if not transformations or view not in transformations
        else tuple(transformations[view])
    )
    return tuple((str(_count_transformation_spec(item)[0]), item) for item in items)


def _view_paths(
    view_names: Sequence[str], representations, transformations
) -> dict[str, tuple[ViewPath, ...]]:
    result = {}
    for view in view_names:
        paths = []
        for rep_name, selector in _representation_specs(view, representations):
            for tr_name, tr_item in _transformation_specs(view, transformations):
                paths.append(ViewPath(view, rep_name, selector, tr_name, tr_item))
        result[view] = tuple(paths)
    return result


def build_view_candidates(
    view_names: Sequence[str],
    representations,
    transformations,
    integrations: Sequence[Integration],
    learners: Sequence[Any],
    learner_name_fn,
) -> tuple[list[CandidateSpec], pd.DataFrame]:
    paths = _view_paths(view_names, representations, transformations)
    learner_names = [learner_name_fn(item) for item in learners]
    specs: list[CandidateSpec] = []
    for integration in integrations:
        if integration.stage == "late":
            continue
        for view_set in integration_view_sets(integration, view_names):
            path_lists = [paths[name] for name in view_set]
            for selected in itertools.product(*path_lists):
                for n_components in integration.component_values():
                    for learner in learner_names:
                        cid = _candidate_id(
                            selected, integration, n_components, learner
                        )
                        family = integration.stage
                        specs.append(
                            CandidateSpec(
                                cid,
                                family,
                                tuple(selected),
                                integration,
                                n_components,
                                learner,
                            )
                        )
    unique = {spec.config_id: spec for spec in specs}
    specs = [unique[key] for key in sorted(unique)]
    rows = []
    for spec in specs:
        rows.append(
            {
                "config_id": spec.config_id,
                "mpdr_id": hashlib.sha1(
                    (
                        spec.representation_label + "__" + spec.transformation_label
                    ).encode()
                ).hexdigest()[:12],
                "count_transformation": spec.transformation_label,
                "resolution": spec.representation_label,
                "levels": ";".join(
                    f"{p.view}:{','.join(p.selector)}" for p in spec.view_paths
                ),
                "learner": spec.learner,
                "active": 1,
                "candidate_family": spec.family,
                "views": ",".join(spec.views),
                "integration": spec.integration.key,
                "integration_n_components": ""
                if spec.n_components is None
                else str(int(spec.n_components)),
            }
        )
    return specs, pd.DataFrame(rows)


def _materialize_view_representation(
    dataset: ViewDataset, path: ViewPath
) -> tuple[np.ndarray, list[str]]:
    ds = dataset.as_dataset(path.view)
    selector = tuple(path.selector)
    taxonomy_tokens = {
        "all",
        "features",
        "asis",
        "raw",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "strain",
    }
    if all(token in taxonomy_tokens for token in selector):
        X, names = materialize_mpdr(ds, selector)
        return np.asarray(X, dtype=np.float32), list(names)
    all_names = list(ds.feature_names_by_level["all"])
    index = {name: i for i, name in enumerate(all_names)}
    missing = [name for name in selector if name not in index]
    if missing:
        raise ValueError(
            f"Representation {path.representation!r} for View {path.view!r} references unknown features: {missing[:8]!r}."
        )
    columns = np.asarray([index[name] for name in selector], dtype=int)
    return np.asarray(ds.X_by_level["all"][:, columns], dtype=np.float32), [
        all_names[i] for i in columns
    ]


def _prepare_candidate_pair_details(
    spec: CandidateSpec,
    matrices,
    names,
    train_idx,
    test_idx,
    protocol,
    random_state,
    transformation_factory_builder,
    lodo_feature_pair,
):
    tr_blocks = {}
    te_blocks = {}
    coordinate_names = {}
    coordinate_metadata = {}
    for path in spec.view_paths:
        X = matrices[(path.view, path.representation, path.selector)]
        feature_names = names[(path.view, path.representation, path.selector)]
        Xtr_raw, Xte_raw, mask = lodo_feature_pair(X, train_idx, test_idx, protocol)
        kept_names = [
            name
            for name, keep in zip(feature_names, np.asarray(mask, dtype=bool))
            if bool(keep)
        ]
        _, factory = transformation_factory_builder(
            path.transformation_item, random_state=random_state
        )
        fitted = factory()
        Xtr, Xte = fitted.apply_pair(Xtr_raw, Xte_raw)
        tr_blocks[path.view] = np.asarray(Xtr, dtype=float)
        te_blocks[path.view] = np.asarray(Xte, dtype=float)
        if hasattr(fitted, "get_feature_names_out"):
            coordinate_names[path.view] = list(fitted.get_feature_names_out(kept_names))
        else:
            coordinate_names[path.view] = kept_names
        if hasattr(fitted, "coordinate_metadata"):
            coordinate_metadata[path.view] = list(
                fitted.coordinate_metadata(kept_names)
            )
    integration = Integration(
        spec.integration.name,
        n_components=spec.n_components,
        view_sets=spec.integration.view_sets,
        min_views=spec.integration.min_views,
        max_views=spec.integration.max_views,
    )
    model = IntegrationModel(integration, random_state=random_state)
    Xtr, Xte, coords = model.fit_transform_pair(
        tr_blocks, te_blocks, coordinate_names, coordinate_metadata
    )
    source_model = IntegrationModel(
        Integration("early_concat"), random_state=random_state
    )
    Xtr_source, Xte_source, source_coords = source_model.fit_transform_pair(
        tr_blocks, te_blocks, coordinate_names, coordinate_metadata
    )
    slices = {}
    start = 0
    for view in tr_blocks:
        stop = start + int(tr_blocks[view].shape[1])
        slices[view] = slice(start, stop)
        start = stop
    return {
        "X_train": np.asarray(Xtr, dtype=float),
        "X_test": np.asarray(Xte, dtype=float),
        "coordinates": coords,
        "X_train_source": np.asarray(Xtr_source, dtype=float),
        "X_test_source": np.asarray(Xte_source, dtype=float),
        "source_coordinates": source_coords,
        "integration_model": model,
        "view_slices": slices,
    }


def _prepare_candidate_pair(
    spec: CandidateSpec,
    matrices,
    names,
    train_idx,
    test_idx,
    protocol,
    random_state,
    transformation_factory_builder,
    lodo_feature_pair,
):
    details = _prepare_candidate_pair_details(
        spec,
        matrices,
        names,
        train_idx,
        test_idx,
        protocol,
        random_state,
        transformation_factory_builder,
        lodo_feature_pair,
    )
    return details["X_train"], details["X_test"], details["coordinates"]


class _IntegratedRegressionPredictor:
    def __init__(self, estimator, integration_model, view_slices):
        self.estimator = estimator
        self.integration_model = integration_model
        self.view_slices = dict(view_slices)

    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        blocks = {view: arr[:, slc] for view, slc in self.view_slices.items()}
        return np.asarray(
            self.estimator.predict(self.integration_model.transform(blocks)),
            dtype=float,
        )


class _IntegratedClassificationPredictor:
    def __init__(self, estimator, integration_model, view_slices):
        self.estimator = estimator
        self.integration_model = integration_model
        self.view_slices = dict(view_slices)
        self.classes_ = getattr(estimator, "classes_", None)

    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        blocks = {view: arr[:, slc] for view, slc in self.view_slices.items()}
        return np.asarray(
            self.estimator.predict(self.integration_model.transform(blocks))
        )

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        blocks = {view: arr[:, slc] for view, slc in self.view_slices.items()}
        return np.asarray(
            self.estimator.predict_proba(self.integration_model.transform(blocks)),
            dtype=float,
        )


def _meta_row(base: dict[str, Any], spec: CandidateSpec) -> dict[str, Any]:
    base.update(
        {
            "candidate_family": spec.family,
            "views": ",".join(spec.views),
            "integration": spec.integration.key,
            "integration_n_components": ""
            if spec.n_components is None
            else str(int(spec.n_components)),
        }
    )
    return base


def _classification_task(
    spec,
    matrices,
    names,
    y,
    classes,
    class_labels,
    sample_ids,
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
):
    from .configs_sweep import (
        MPDR,
        QualificationGate,
        _count_transformation_factory,
        _failed_metric_row,
        _lodo_feature_pair,
        _metric_row,
        _prediction_rows_values,
        _predict_proba_aligned,
    )

    tracker = ResourceTracker(sample_interval_s=resource_sample_interval_s).start()
    mpdr = MPDR(spec.representation_label, tuple(spec.views), spec.transformation_label)
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
    with (
        thread_environment(threads_per_worker),
        threadpool_limits(limits=max(1, int(threads_per_worker))),
    ):
        for inner_no, (tr_local, va_local) in enumerate(inner_splits):
            inner_key = f"{split_key}__i{inner_no}"
            if inner_key in existing_inner_keys:
                continue
            tr_idx = train_idx[np.asarray(tr_local, dtype=int)]
            va_idx = train_idx[np.asarray(va_local, dtype=int)]
            if len(np.unique(y[tr_idx])) < 2 or len(va_idx) == 0:
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
                clf.fit(Xtr, y[tr_idx])
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
                result["inner_metrics"].append(row)
                result["inner_predictions"].extend(
                    _prediction_rows_values(
                        inner_key,
                        spec.config_id,
                        va_idx,
                        sample_ids,
                        y,
                        class_labels,
                        pred,
                        proba,
                        "inner",
                        split_key,
                    )
                )
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
                            "levels": ",".join(spec.views),
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
                clf.fit(Xtr, y[train_idx])
                proba = _predict_proba_aligned(clf, Xte, classes)
                pred = classes[proba.argmax(axis=1)]
                metrics = compute_metrics(y[test_idx], pred, proba, classes)
                result["outer_metrics"].append(
                    _meta_row(
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
                )
                result["outer_predictions"].extend(
                    _prediction_rows_values(
                        split_key,
                        spec.config_id,
                        test_idx,
                        sample_ids,
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
    mpdr = MPDR(spec.representation_label, tuple(spec.views), spec.transformation_label)
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
    with (
        thread_environment(threads_per_worker),
        threadpool_limits(limits=max(1, int(threads_per_worker))),
    ):
        for inner_no, (tr_local, va_local) in enumerate(inner_splits):
            inner_key = f"{split_key}__i{inner_no}"
            if inner_key in existing_inner_keys:
                continue
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
                pred = np.asarray(reg.predict(Xva), dtype=float)
                metrics = compute_regression_metrics(y[va_idx], pred)
                result["inner_metrics"].append(
                    _meta_row(
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
                )
                result["inner_predictions"].extend(
                    _regression_prediction_rows(
                        inner_key,
                        spec.config_id,
                        va_idx,
                        dataset.sample_ids,
                        y,
                        pred,
                        "inner",
                        split_key,
                        dataset.target_name,
                    )
                )
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
                pred = np.asarray(reg.predict(Xte), dtype=float)
                metrics = compute_regression_metrics(y[test_idx], pred)
                result["outer_metrics"].append(
                    _meta_row(
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
                )
                result["outer_predictions"].extend(
                    _regression_prediction_rows(
                        split_key,
                        spec.config_id,
                        test_idx,
                        dataset.sample_ids,
                        y,
                        pred,
                        "outer",
                        split_key,
                        dataset.target_name,
                    )
                )
                result["fits"] += 1
            except Exception as exc:
                row = _regression_metric_row(
                    {}, split_key, None, spec.config_id, mpdr, spec.learner, "outer"
                )
                row["ok"] = 0
                row["error"] = f"{type(exc).__name__}: {exc}"
                result["outer_metrics"].append(_meta_row(row, spec))
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
    return result


def evaluate_view_sweep(sweep) -> dict[str, Path]:
    from .configs_sweep import (
        _backfill_metrics_from_predictions,
        _backfill_regression_metrics_from_predictions,
        _checkpoint_result,
        _clear_evaluation_checkpoints,
        _done_pairs,
        _existing_inner_scores,
        _existing_outputs,
        _groups_from_metadata,
        _inner_splits,
        _load_existing_evaluation,
        _outer_pair_complete,
        _outer_splits,
        _prepare_dirs,
        _qualification_map,
        _regression_inner_splits,
        _regression_outer_splits,
        _strata_from_metadata,
        _write_config_table,
        _write_manifest,
        _write_rankings_and_figures,
        _write_tables,
        _learner_factory,
        _learner_name,
        write_mpma_b_selection_outputs,
    )
    from .figures import _write_representation_impact_figure

    dataset = load_views(sweep.samples, sweep.views)
    root = sweep.root()
    _prepare_dirs(root)
    learners = [_learner_factory(item, task=dataset.task) for item in sweep.learners]
    integrations = tuple(sweep.integrations or (Integration("unimodal"),))
    specs, configs = build_view_candidates(
        tuple(dataset.views),
        sweep.representations,
        sweep.transformations,
        integrations,
        sweep.learners,
        _learner_name,
    )
    _write_config_table(root, configs)
    if sweep.evaluation.redo:
        _clear_evaluation_checkpoints(root)
    manifest = {
        "title": sweep.title,
        "version": "view_sweep_v1",
        "primary_view": dataset.primary_view,
        "sample_alignment": "primary_view_required_in_all_views",
        "n_samples": len(dataset.sample_ids),
        "views": {
            name: {"n_features": int(view.X.shape[1]), "format": view.format}
            for name, view in dataset.views.items()
        },
        "integrations": [integration.key for integration in integrations],
    }
    dump_json_standard(manifest, root / "manifest_views.json")
    dump_json_standard(
        {
            **manifest,
            "class_labels": dataset.class_labels,
            "target": dataset.target_name,
            "task": dataset.task,
        },
        root / "manifest.json",
    )
    matrices = {}
    names = {}
    for spec in specs:
        for path in spec.view_paths:
            key = (path.view, path.representation, path.selector)
            if key not in matrices:
                X, feature_names = _materialize_view_representation(dataset, path)
                matrices[key] = X
                names[key] = feature_names
    groups = (
        None
        if sweep.samples.group_col is None
        else dataset.metadata[sweep.samples.group_col].to_numpy()
    )
    if dataset.task == "classification":
        strata = _strata_from_metadata(
            dataset.metadata, dataset.y, sweep.samples.stratify_col
        )
        outer_splits = _outer_splits(
            sweep.evaluation, dataset.y, groups, strata, sweep.samples.stratify_col
        )
    else:
        outer_splits = _regression_outer_splits(
            sweep.evaluation, len(dataset.y), groups
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
        if dataset.task == "classification":
            if len(np.unique(dataset.y[train_idx])) < 2 or len(test_idx) == 0:
                continue
            inner = _inner_splits(
                sweep.evaluation,
                dataset.y,
                train_idx,
                groups,
                split,
                strata,
                sweep.samples.stratify_col,
            )
        else:
            inner = _regression_inner_splits(sweep.evaluation, train_idx, groups, split)
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
                    dataset.classes,
                    tuple(dataset.class_labels),
                    tuple(dataset.sample_ids),
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
    stage("View configuration sweep", sweep.title)
    summary_table(
        "Sweep overview",
        {
            "samples": len(dataset.sample_ids),
            "primary view": dataset.primary_view,
            "views": ", ".join(dataset.views),
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
        payloads = [(fn, args, kwargs) for _, _, fn, args, kwargs in prepared]
        completed_by_split: dict[str, int] = {}
        with progress() as prog:
            job_task = prog.add_task("View candidate/split jobs", total=len(prepared))
            split_task = prog.add_task(
                "Outer splits completed", total=max(1, len(split_counts))
            )
            for result in iter_parallel_tasks(payloads, execution):
                _checkpoint_result(root, result)
                for key in rows:
                    rows[key].extend(result.get(key, []))
                split_key = str(result.get("split_key", ""))
                config_id = str(result.get("config_id", ""))
                completed_by_split[split_key] = completed_by_split.get(split_key, 0) + 1
                if completed_by_split[split_key] == split_counts.get(split_key, 0):
                    prog.advance(split_task)
                prog.update(
                    job_task,
                    advance=1,
                    description=f"View candidate/split jobs · {split_key} · {config_id}",
                )
                prog.update(
                    split_task,
                    description=(
                        f"Outer splits completed · {sum(1 for key, value in completed_by_split.items() if value == split_counts.get(key, 0))}"
                        f"/{len(split_counts)}"
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
    success("View configuration sweep completed")
    outputs = _existing_outputs(root)
    outputs["view_manifest"] = root / "manifest_views.json"
    path_table("Configuration sweep outputs", outputs)
    return outputs


def candidate_from_row(
    sweep, row, dataset: ViewDataset | None = None
) -> tuple[ViewDataset, CandidateSpec, dict[Any, np.ndarray], dict[Any, list[str]]]:
    from .configs_sweep import _learner_name

    dataset = load_views(sweep.samples, sweep.views) if dataset is None else dataset
    specs, _ = build_view_candidates(
        tuple(dataset.views),
        sweep.representations,
        sweep.transformations,
        tuple(sweep.integrations or (Integration("unimodal"),)),
        sweep.learners,
        _learner_name,
    )
    config_id = str(row["config_id"])
    matches = [spec for spec in specs if spec.config_id == config_id]
    if len(matches) != 1:
        raise ValueError(
            f"Config {config_id!r} cannot be reconstructed uniquely from the configured view search space."
        )
    spec = matches[0]
    matrices = {}
    names = {}
    for path in spec.view_paths:
        key = (path.view, path.representation, path.selector)
        if key not in matrices:
            X, feature_names = _materialize_view_representation(dataset, path)
            matrices[key] = X
            names[key] = feature_names
    return dataset, spec, matrices, names


def fit_view_candidate_oof_for_explainability(sweep, row):
    from .configs_sweep import (
        _groups_from_metadata,
        _outer_splits,
        _strata_from_metadata,
        _learner_factory,
        _predict_proba_aligned,
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
    splits = _outer_splits(
        sweep.evaluation, dataset.y, groups, strata, sweep.samples.stratify_col
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
        clf.fit(details["X_train"], dataset.y[train_idx])
        direct_proba = _predict_proba_aligned(clf, details["X_test"], dataset.classes)
        if spec.integration.stage == "intermediate":
            estimator = _IntegratedClassificationPredictor(
                clf, details["integration_model"], details["view_slices"]
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
            "No outer fold could be fitted for view-based out-of-fold explainability."
        )
    first_meta = list(folds[0]["coordinate_metadata"])
    first_names = [str(item.name) for item in first_meta]
    for fold in folds[1:]:
        current = [str(item.name) for item in fold["coordinate_metadata"]]
        if current != first_names:
            raise ExplainabilityConfigurationError(
                "View-based OOF explainability produced different model coordinates across outer folds. This commonly occurs when LODO training-domain feature filtering changes the feature universe. The framework will not aggregate non-identical coordinates."
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


def fit_view_regression_candidate_folds(sweep, row):
    from .configs_sweep import (
        _groups_from_metadata,
        _regression_outer_splits,
        _learner_factory,
        _count_transformation_factory,
        _lodo_feature_pair,
    )

    dataset, spec, matrices, names = candidate_from_row(sweep, row)
    groups = (
        None
        if sweep.samples.group_col is None
        else dataset.metadata[sweep.samples.group_col].astype(str).to_numpy()
    )
    splits = _regression_outer_splits(sweep.evaluation, len(dataset.y), groups)
    learner_lookup = dict(
        _learner_factory(item, task="regression") for item in sweep.learners
    )
    folds = []
    for split in splits:
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
        direct_pred = np.asarray(model.predict(details["X_test"]), dtype=float).reshape(
            -1
        )
        if spec.integration.stage == "intermediate":
            estimator = _IntegratedRegressionPredictor(
                model, details["integration_model"], details["view_slices"]
            )
            X_train = details["X_train_source"]
            X_test = details["X_test_source"]
            coords = details["source_coordinates"]
            pred = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
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
            }
        )
    return dataset, folds
