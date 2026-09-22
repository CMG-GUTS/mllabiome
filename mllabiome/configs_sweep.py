from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold
from threadpoolctl import threadpool_limits

from .data import Data, Dataset, load_dataset
from .integrations import Integration
from .modalities import Samples, Modality
from .console import info, path_table, progress, stage, success, summary_table
from .figures import _write_representation_impact_figure
from .explainability_methods import (
    ALE,
    Permutation,
    SHAP,
    apply_profile,
    method_has_local,
    normalise_profile,
)
from .learners import _learner_factory, _learner_name, validate_model_specs
from .metrics import (
    _estimator_call,
    _predict_proba_aligned as _metrics_predict_proba_aligned,
    compute_metrics,
    compute_regression_metrics,
    metric_is_loss,
    metric_passes_threshold,
)
from .resolutions import _parse_resolution, materialize_mpdr
from .utils import METRIC_COLUMNS, dump_json_standard
from .transformations import (
    TRANSFORMATION_LABELS,
    _count_transformation_factory,
    _count_transformation_name,
    _count_transformation_spec,
)
from .selection import write_mpma_b_selection_outputs
from .storage import read_table, write_table, table_exists, remove_table
from .splits import resolve_cv_splits
from .runtime import (
    configure_estimator_threads,
    iter_parallel_tasks,
    resolve_execution_plan,
    thread_environment,
)
from .compute import ResourceTracker, machine_profile


def _default_transformations():
    from .transformations import Transformation

    return (Transformation("arcsine_sqrt"),)


@dataclass(frozen=True)
class MPDR:
    resolution: str
    levels: tuple[str, ...]
    count_transformation: str


@dataclass(frozen=True)
class MPMA:
    config_id: str
    mpdr: MPDR
    learner: str


@dataclass
class QualificationGate:
    enabled: bool = False
    metric: str = "nMCC"
    threshold: float | None = None

    def qualifies(self, score: float) -> bool:
        if not self.enabled:
            return True
        if self.threshold is None:
            raise ValueError(
                "QualificationGate.threshold is required when the gate is enabled."
            )
        return metric_passes_threshold(score, float(self.threshold), self.metric)


@dataclass
class Evaluation:
    protocol: Literal[
        "repeated_nested_cv", "nested_cv", "lodo", "leave_one_dataset_out"
    ] = "repeated_nested_cv"
    outer_folds: int = 5
    inner_folds: int = 3
    repeats: int = 2
    random_state: int = 42
    optimize_metric: Any = "nMCC"
    n_jobs: int | str = 1
    parallel_backend: str = "loky"
    memory_fraction: float = 0.80
    min_worker_memory_gib: float = 1.0
    resource_sample_interval_s: float = 0.10
    redo: bool = False


@dataclass
class Ensemble:
    max_sizes: tuple[int, ...] = (3,)
    sizes: tuple[int, ...] | None = None
    selection_strategies: tuple[str, ...] = (
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    )
    aggregation_strategies: tuple[str, ...] = (
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "rank_mean",
        "majority_vote",
    )
    optimize_metric: Any = "nMCC"

    include_inactive: bool = True

    threshold_score: float = 0.30
    threshold_max_members: int = 50

    exclude_config_ids: tuple[str, ...] = ()
    exclude_learners: tuple[str, ...] = ()
    exclude_resolutions: tuple[str, ...] = ()
    exclude_transformations: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalExplanations:
    representatives: bool = True
    sample_ids: tuple[str, ...] = ()
    stored_features: int = 15
    displayed_features: int = 8
    regression_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", tuple(str(x) for x in self.sample_ids))
        object.__setattr__(
            self,
            "regression_quantiles",
            tuple(float(x) for x in self.regression_quantiles),
        )
        if int(self.stored_features) < 1:
            raise ValueError("LocalExplanations.stored_features must be at least 1.")
        if int(self.displayed_features) < 1:
            raise ValueError("LocalExplanations.displayed_features must be at least 1.")
        if not self.regression_quantiles or any(
            not 0.0 < x < 1.0 for x in self.regression_quantiles
        ):
            raise ValueError(
                "LocalExplanations.regression_quantiles must contain values strictly between 0 and 1."
            )


@dataclass(init=False)
class Explainability:
    targets: str | tuple[str, ...] = "auto"
    top_k: int = 30
    methods: tuple[Any, ...] = (SHAP(), Permutation(), ALE())
    classes: str | tuple[int | str, ...] = "auto"
    profile: str = "standard"
    random_state: int = 42
    local: LocalExplanations = LocalExplanations()
    keep_cache: bool = False
    n_jobs: int | str | None = None
    parallel_backend: str | None = None

    def __init__(
        self,
        targets: str | Sequence[str] = "auto",
        top_k: int = 30,
        methods: Sequence[Any] = (SHAP(), Permutation(), ALE()),
        classes: str | Sequence[int | str] = "auto",
        profile: str = "standard",
        random_state: int = 42,
        local: LocalExplanations | None = None,
        keep_cache: bool = False,
        n_jobs: int | str | None = None,
        parallel_backend: str | None = None,
        **unknown_options: Any,
    ) -> None:
        legacy_keys = {
            "local_explanations",
            "representative_instances",
            "instance_sample_ids",
            "local_top_k",
            "top_instance_features",
            "representative_quantiles",
        }
        legacy = {
            key: unknown_options.pop(key)
            for key in tuple(unknown_options)
            if key in legacy_keys
        }
        if unknown_options:
            unknown = ", ".join(sorted(unknown_options))
            raise TypeError(f"Unknown Explainability option(s): {unknown}.")
        if local is not None and legacy:
            raise TypeError(
                "Use either Explainability.local or legacy local-explanation options, not both."
            )
        self.targets = _normalise_explainability_targets_config(targets)
        self.top_k = int(top_k)
        self.profile = normalise_profile(profile)
        self.methods = tuple(apply_profile(m, self.profile) for m in methods)
        self.classes = _normalise_explainability_classes_config(classes)
        self.random_state = int(random_state)
        self.local = local if local is not None else _legacy_local_explanations(legacy)
        self.keep_cache = bool(keep_cache)
        self.n_jobs = n_jobs
        self.parallel_backend = (
            None if parallel_backend is None else str(parallel_backend)
        )

    @property
    def local_explanations(self) -> str:
        return _effective_local_explanations_mode(self)

    @property
    def representative_instances(self) -> bool:
        return bool(self.local.representatives)

    @property
    def instance_sample_ids(self) -> tuple[str, ...]:
        return tuple(self.local.sample_ids)

    @property
    def local_top_k(self) -> int:
        return int(self.local.stored_features)

    @property
    def top_instance_features(self) -> int:
        return int(self.local.displayed_features)

    @property
    def representative_quantiles(self) -> tuple[float, ...]:
        return tuple(self.local.regression_quantiles)


def _legacy_local_explanations(options: Mapping[str, Any]) -> LocalExplanations:
    if not options:
        return LocalExplanations()
    representatives = bool(options.get("representative_instances", True))
    sample_ids = tuple(str(x) for x in options.get("instance_sample_ids", ()))
    stored_features = int(options.get("local_top_k", 15))
    displayed_features = int(options.get("top_instance_features", 8))
    quantiles = tuple(
        float(x) for x in options.get("representative_quantiles", (0.25, 0.50, 0.75))
    )
    mode = (
        str(options.get("local_explanations", "auto")).strip().lower().replace("-", "_")
    )
    aliases = {
        "": "auto",
        "off": "none",
        "false": "none",
        "representatives": "representative",
        "samples": "requested",
        "both": "representative_and_requested",
        "representative_requested": "representative_and_requested",
    }
    mode = aliases.get(mode, mode)
    if mode == "none":
        representatives = False
        sample_ids = ()
    elif mode == "representative":
        representatives = True
        sample_ids = ()
    elif mode == "requested":
        representatives = False
    elif mode == "representative_and_requested":
        representatives = True
    elif mode != "auto":
        raise ValueError(
            "Explainability.local_explanations must be 'auto', 'none', 'representative', 'requested', or 'representative_and_requested'."
        )
    return LocalExplanations(
        representatives=representatives,
        sample_ids=sample_ids,
        stored_features=stored_features,
        displayed_features=displayed_features,
        regression_quantiles=quantiles,
    )


def _effective_local_explanations_mode(explainability: Any) -> str:
    methods = tuple(getattr(explainability, "methods", ()))
    if not any(method_has_local(method) for method in methods):
        return "none"
    local = getattr(explainability, "local", None)
    if local is None:
        local = _legacy_local_explanations(
            {
                "representative_instances": getattr(
                    explainability, "representative_instances", True
                ),
                "instance_sample_ids": getattr(
                    explainability, "instance_sample_ids", ()
                ),
            }
        )
    representative = bool(local.representatives)
    requested = bool(local.sample_ids)
    if representative and requested:
        return "representative_and_requested"
    if representative:
        return "representative"
    if requested:
        return "requested"
    return "none"


def _normalise_explainability_targets_config(
    targets: str | Sequence[str],
) -> str | tuple[str, ...]:
    if isinstance(targets, str):
        text = targets.strip()
        return text or "auto"
    out = tuple(str(x).strip() for x in targets if str(x).strip())
    return out or "auto"


def _normalise_explainability_classes_config(
    classes: str | Sequence[int | str],
) -> str | tuple[int | str, ...]:
    if isinstance(classes, str):
        text = classes.strip()
        return text or "auto"
    out = tuple(x for x in classes if str(x).strip())
    return out or "auto"


@dataclass
class Sweep:
    data: Data | None
    experiment_dir: Path | str
    resolutions: Sequence[tuple[str, Sequence[str]] | str] = field(
        default_factory=lambda: (("species", ("species",)),)
    )
    count_transformations: Sequence[Any] = field(
        default_factory=_default_transformations
    )
    learners: Any = field(default_factory=tuple)
    evaluation: Evaluation = field(default_factory=Evaluation)
    gate: QualificationGate = field(default_factory=QualificationGate)
    ensemble: Ensemble = field(default_factory=Ensemble)
    explainability: Explainability = field(default_factory=Explainability)
    title: str = "mllabiome sweep"
    samples: Samples | None = None
    modalities: Sequence[Modality] = field(default_factory=tuple)
    representations: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    transformations: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    integrations: Sequence[Integration] = field(
        default_factory=lambda: (Integration("unimodal"),)
    )

    def __post_init__(self) -> None:
        validate_model_specs(self.learners, context="Sweep.learners / MODELS")

    def root(self) -> Path:
        return Path(self.experiment_dir)

    @property
    def uses_modalities(self) -> bool:
        return self.samples is not None or bool(self.modalities)


def build_sweep_from_module(mod: Any) -> Sweep:

    if hasattr(mod, "VIEWS"):
        raise TypeError(
            "VIEWS is not supported. Use MODALITIES with mllabiome.Modality(...)."
        )

    if hasattr(mod, "build_sweep"):
        obj = mod.build_sweep()
        if not isinstance(obj, Sweep):
            raise TypeError("build_sweep() must return mllabiome.Sweep.")
        return obj
    if hasattr(mod, "SWEEP"):
        obj = mod.SWEEP
        if not isinstance(obj, Sweep):
            raise TypeError("SWEEP must be an instance of mllabiome.Sweep.")
        return obj

    if hasattr(mod, "SAMPLES") or hasattr(mod, "MODALITIES"):
        missing = [
            name
            for name in ("EXPERIMENT_DIR", "SAMPLES", "MODALITIES")
            if not hasattr(mod, name)
        ]
        if missing:
            raise TypeError(f"Modality-based config is missing: {', '.join(missing)}.")
        samples = getattr(mod, "SAMPLES")
        modalities = tuple(getattr(mod, "MODALITIES"))
        if not isinstance(samples, Samples):
            raise TypeError("SAMPLES must be an instance of mllabiome.Samples(...).")
        if not modalities or not all(
            isinstance(modality, Modality) for modality in modalities
        ):
            raise TypeError(
                "MODALITIES must contain one or more mllabiome.Modality(...) instances."
            )
        if hasattr(mod, "TRANSFORMATIONS"):
            transformations = getattr(mod, "TRANSFORMATIONS")
        elif hasattr(mod, "_build_transformations"):
            transformations = mod._build_transformations()
        else:
            transformations = {}
        if hasattr(mod, "MODELS"):
            learners = getattr(mod, "MODELS")
        elif hasattr(mod, "_build_models"):
            learners = mod._build_models()
        else:
            raise TypeError(
                "Modality-based config must define MODELS or _build_models()."
            )
        integrations = tuple(getattr(mod, "INTEGRATIONS", (Integration("unimodal"),)))
        if not all(isinstance(item, Integration) for item in integrations):
            raise TypeError(
                "INTEGRATIONS must contain mllabiome.Integration(...) instances."
            )
        ensemble_explicit = hasattr(mod, "ENSEMBLE")
        ensemble = getattr(mod, "ENSEMBLE", Ensemble())
        late = {item.key for item in integrations if item.stage == "late"}
        if late:
            task = _normalise_sweep_task(samples.task)
            if task == "regression":
                mapping = {
                    "late_mean_prediction": "mean_prediction",
                    "late_weighted_mean_prediction": "weighted_mean_prediction",
                    "late_median_prediction": "median_prediction",
                    "late_mean_proba": "mean_prediction",
                    "late_weighted_mean_proba": "weighted_mean_prediction",
                }
                learned_aggregation = "weighted_mean_prediction"
            else:
                mapping = {
                    "late_mean_proba": "mean_proba",
                    "late_weighted_mean_proba": "weighted_mean_proba",
                }
                learned_aggregation = "weighted_mean_proba"
            if ensemble_explicit:
                aggregations = list(ensemble.aggregation_strategies)
                selections = list(ensemble.selection_strategies)
            else:
                aggregations = []
                selections = ["top_k"]
            for key in sorted(late):
                if key in mapping and mapping[key] not in aggregations:
                    aggregations.append(mapping[key])
                if key == "late_super_learner":
                    if "super_learner" not in selections:
                        selections.append("super_learner")
                    if learned_aggregation not in aggregations:
                        aggregations.append(learned_aggregation)
            ensemble = replace(
                ensemble,
                aggregation_strategies=tuple(aggregations),
                selection_strategies=tuple(selections),
            )
        return Sweep(
            data=None,
            title=getattr(mod, "TITLE", Path(getattr(mod, "EXPERIMENT_DIR")).name),
            experiment_dir=getattr(mod, "EXPERIMENT_DIR"),
            learners=learners,
            evaluation=getattr(mod, "EVALUATION", Evaluation()),
            gate=getattr(mod, "GATE", QualificationGate()),
            ensemble=ensemble,
            explainability=getattr(mod, "EXPLAINABILITY", Explainability()),
            samples=samples,
            modalities=modalities,
            representations=getattr(mod, "REPRESENTATIONS", {}),
            transformations=transformations,
            integrations=integrations,
        )

    required = ["EXPERIMENT_DIR", "DATA"]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise TypeError(
            "Config must define "
            + ", ".join(required)
            + f". Missing: {', '.join(missing)}."
        )
    data = getattr(mod, "DATA")
    if not isinstance(data, Data):
        raise TypeError("DATA must be an instance of mllabiome.Data(...).")

    if hasattr(mod, "RESOLUTIONS"):
        resolutions = getattr(mod, "RESOLUTIONS")
    elif hasattr(mod, "_RESOLUTION_SETS"):
        resolutions = getattr(mod, "_RESOLUTION_SETS")
    else:
        raise TypeError(
            "Config must define RESOLUTIONS. Legacy _RESOLUTION_SETS is still accepted for compatibility."
        )

    if hasattr(mod, "COUNT_TRANSFORMATIONS"):
        count_transformations = getattr(mod, "COUNT_TRANSFORMATIONS")
    elif hasattr(mod, "_build_count_transformations"):
        count_transformations = mod._build_count_transformations()
    else:
        raise TypeError(
            "Config must define COUNT_TRANSFORMATIONS. Legacy _build_count_transformations() is still accepted for compatibility."
        )

    if hasattr(mod, "MODELS"):
        learners = getattr(mod, "MODELS")
    elif hasattr(mod, "_build_models"):
        learners = mod._build_models()
    else:
        raise TypeError(
            "Config must define MODELS as explicit (name, estimator) pairs. Legacy _build_models() is still accepted for compatibility."
        )
    validate_model_specs(learners, context="MODELS")

    return Sweep(
        title=getattr(mod, "TITLE", Path(getattr(mod, "EXPERIMENT_DIR")).name),
        experiment_dir=getattr(mod, "EXPERIMENT_DIR"),
        data=data,
        resolutions=resolutions,
        count_transformations=count_transformations,
        learners=learners,
        evaluation=getattr(mod, "EVALUATION", Evaluation()),
        gate=getattr(mod, "GATE", QualificationGate()),
        ensemble=getattr(mod, "ENSEMBLE", Ensemble()),
        explainability=getattr(mod, "EXPLAINABILITY", Explainability()),
    )


_MPDR_SEMANTICS = "select_then_transform_fold_local_lodo_v2"


def _mpdr_id(count_transformation: str, resolution: str) -> str:
    return hashlib.sha1(
        f"{_MPDR_SEMANTICS}__{count_transformation}__{resolution}".encode()
    ).hexdigest()[:12]


def _config_id(count_transformation: str, resolution: str, learner_name: str) -> str:
    return hashlib.sha1(
        f"{_MPDR_SEMANTICS}__{count_transformation}__{resolution}__{learner_name}".encode()
    ).hexdigest()[:12]


def build_sweep_configs(
    resolutions: Sequence[Any],
    count_transformations: Sequence[Any],
    learners: Sequence[Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    learner_names = [_learner_name(x) for x in learners]
    for res in resolutions:
        res_name, levels = _parse_resolution(res)
        for ct in count_transformations:
            ct_name = _count_transformation_name(ct)
            for lname in learner_names:
                rows.append(
                    {
                        "config_id": _config_id(ct_name, res_name, lname),
                        "mpdr_id": _mpdr_id(ct_name, res_name),
                        "count_transformation": ct_name,
                        "resolution": res_name,
                        "levels": ",".join(levels),
                        "learner": lname,
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
    y: np.ndarray,
    classes: np.ndarray,
    class_labels: Sequence[str],
    sample_ids: Sequence[str],
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
    existing_inner_scores: Sequence[float],
    existing_qualification: dict[str, Any] | None,
    needs_outer: bool,
    random_state: int,
    threads_per_worker: int,
    resource_sample_interval_s: float,
) -> dict[str, Any]:
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
    scores = [float(x) for x in existing_inner_scores if np.isfinite(x)]
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
            _, inner_ct_factory = transformation_factory_builder(
                ct_item, random_state=random_state
            )
            fitted = inner_ct_factory()
            X_inner_train, X_inner_val, _ = _lodo_feature_pair(
                X_base, tr_idx, va_idx, protocol
            )
            try:
                X_tr, X_va = fitted.apply_pair(X_inner_train, X_inner_val)
                clf = configure_estimator_threads(learner_factory(), threads_per_worker)
                clf.fit(X_tr, y[tr_idx])
                proba = _predict_proba_aligned(clf, X_va, classes)
                pred = classes[proba.argmax(axis=1)]
                metrics = compute_metrics(y[va_idx], pred, proba, classes)
                row = _metric_row(
                    metrics, split_key, inner_key, cid, mpdr, learner_name, "inner"
                )
                result["inner_metrics"].append(row)
                result["inner_predictions"].extend(
                    _prediction_rows_values(
                        inner_key,
                        cid,
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
                score = float(metrics.get(gate_metric, np.nan))
                if np.isfinite(score):
                    scores.append(score)
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
                gate_score = float(np.mean(scores)) if scores else float("nan")
                gate = QualificationGate(
                    enabled=True, metric=gate_metric, threshold=gate_threshold
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
            _, outer_ct_factory = transformation_factory_builder(
                ct_item, random_state=random_state
            )
            fitted_outer = outer_ct_factory()
            X_outer_train, X_outer_test, _ = _lodo_feature_pair(
                X_base, train_idx, test_idx, protocol
            )
            try:
                X_train, X_test = fitted_outer.apply_pair(X_outer_train, X_outer_test)
                clf = configure_estimator_threads(learner_factory(), threads_per_worker)
                clf.fit(X_train, y[train_idx])
                proba = _predict_proba_aligned(clf, X_test, classes)
                pred = classes[proba.argmax(axis=1)]
                metrics = compute_metrics(y[test_idx], pred, proba, classes)
                row = _metric_row(
                    metrics, split_key, None, cid, mpdr, learner_name, "outer"
                )
                result["outer_metrics"].append(row)
                result["outer_predictions"].extend(
                    _prediction_rows_values(
                        split_key,
                        cid,
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
        lookup[(str(split_key), str(config_id))] = compute_metrics(
            y_true.astype(int), y_pred.astype(int), proba, classes
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
        for metric in METRIC_COLUMNS:
            current = pd.to_numeric(
                pd.Series([out.at[index, metric]]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(current):
                value = float(values.get(metric, np.nan))
                if np.isfinite(value):
                    out.at[index, metric] = value
    return out


def _normalise_sweep_task(value: str) -> str:
    task = str(value).strip().casefold().replace("-", "_")
    aliases = {
        "binary": "classification",
        "multiclass": "classification",
        "continuous": "regression",
        "multilabel": "multilabel",
        "multi_output": "multioutput",
        "multioutput": "multioutput",
    }
    task = aliases.get(task, task)
    if task not in {"classification", "regression", "multilabel", "multioutput"}:
        raise ValueError(f"Unsupported task {value!r}.")
    return task


def sweep_task(sweep: Sweep) -> str:
    if sweep.uses_modalities:
        if sweep.samples is None:
            raise ValueError("Modality-based sweeps require Samples.")
        return _normalise_sweep_task(sweep.samples.task)
    if sweep.data is None:
        raise ValueError("Data-based sweeps require Data.")
    return _normalise_sweep_task(sweep.data.task)


def _target_columns(data: Data) -> tuple[str, ...]:
    if isinstance(data.target_col, str):
        return (data.target_col,)
    return tuple(str(x) for x in data.target_col)


def _target_task(data: Data, target: str) -> str:
    root_task = _normalise_sweep_task(data.task)
    if root_task == "multilabel":
        return "classification"
    if root_task == "multioutput":
        if not data.target_tasks or target not in data.target_tasks:
            raise ValueError(
                f"Data.target_tasks must define task for multioutput target {target!r}."
            )
        value = _normalise_sweep_task(data.target_tasks[target])
        if value not in {"classification", "regression"}:
            raise ValueError(
                f"Multioutput target {target!r} must be classification or regression."
            )
        return value
    if data.target_tasks and target in data.target_tasks:
        value = _normalise_sweep_task(data.target_tasks[target])
        if value in {"classification", "regression"}:
            return value
    return root_task


def _learners_for_target(sweep: Sweep, target: str, task: str):
    learners = sweep.learners
    if not isinstance(learners, dict):
        return learners
    for key in (target, task, "default"):
        if key in learners:
            return learners[key]
    raise ValueError(f"No learners configured for target {target!r} ({task}).")


def _metric_for_target(value: Any, target: str, task: str, fallback: str) -> str:
    if isinstance(value, dict):
        for key in (target, task, "default"):
            if key in value:
                return str(value[key])
        return fallback
    return str(value)


def _explainability_for_target(
    plan: Explainability, task: str, optimize_metric: str
) -> Explainability:
    methods = []
    regression_scores = {
        "r2",
        "mae",
        "mse",
        "rmse",
        "medae",
        "explainedvariance",
        "explained_variance",
        "pearsonr",
        "spearmanr",
    }
    for method in plan.methods:
        if isinstance(method, Permutation):
            scoring = str(method.scoring).strip().casefold()
            if task == "regression":
                if scoring not in regression_scores:
                    fallback = str(optimize_metric).strip().casefold()
                    scoring = fallback if fallback in regression_scores else "rmse"
            elif scoring not in {"log_loss", "brier"}:
                scoring = "log_loss"
            method = replace(method, scoring=scoring)
        methods.append(method)
    return replace(plan, methods=tuple(methods))


def target_sweeps(sweep: Sweep) -> list[Sweep]:
    targets = _target_columns(sweep.data)
    if len(targets) == 1 and _normalise_sweep_task(sweep.data.task) not in {
        "multilabel",
        "multioutput",
    }:
        return [sweep]
    out = []
    for target in targets:
        task = _target_task(sweep.data, target)
        labels = sweep.data.class_labels
        if sweep.data.target_class_labels and target in sweep.data.target_class_labels:
            labels = tuple(sweep.data.target_class_labels[target])
        positive = sweep.data.positive_class
        if (
            sweep.data.target_positive_classes
            and target in sweep.data.target_positive_classes
        ):
            positive = sweep.data.target_positive_classes[target]
        extra_metadata = tuple(
            dict.fromkeys(
                (*sweep.data.metadata_cols, *[x for x in targets if x != target])
            )
        )
        child_data = replace(
            sweep.data,
            target_col=target,
            task=task,
            class_labels=labels,
            positive_class=positive,
            metadata_cols=extra_metadata,
        )
        default_metric = "RMSE" if task == "regression" else "log_loss"
        child_evaluation = replace(
            sweep.evaluation,
            optimize_metric=_metric_for_target(
                sweep.evaluation.optimize_metric, target, task, default_metric
            ),
        )
        child_ensemble = replace(
            sweep.ensemble,
            optimize_metric=_metric_for_target(
                sweep.ensemble.optimize_metric, target, task, default_metric
            ),
        )
        child_explainability = _explainability_for_target(
            sweep.explainability, task, str(child_evaluation.optimize_metric)
        )
        child = replace(
            sweep,
            data=child_data,
            experiment_dir=sweep.root() / "targets" / target,
            learners=_learners_for_target(sweep, target, task),
            evaluation=child_evaluation,
            ensemble=child_ensemble,
            explainability=child_explainability,
            title=f"{sweep.title} · {target}",
        )
        out.append(child)
    return out


def _regression_outer_splits(
    plan: Evaluation, n_samples: int, groups: np.ndarray | None
) -> list[dict[str, Any]]:
    protocol = plan.protocol.lower()
    if protocol in {"lodo", "leave_one_dataset_out"}:
        if groups is None:
            raise ValueError("LODO requires DATA.group_col.")
        out = []
        for i, group in enumerate(pd.unique(groups)):
            out.append(
                {
                    "split_key": f"lodo_{i}__{group}",
                    "repeat": 0,
                    "outer_fold": i,
                    "outer_group": str(group),
                    "train_idx": np.where(groups != group)[0],
                    "test_idx": np.where(groups == group)[0],
                }
            )
        return out
    repeats = plan.repeats if protocol == "repeated_nested_cv" else 1
    out = []
    n_splits = min(int(plan.outer_folds), int(n_samples))
    if n_splits < 2:
        raise ValueError(
            "Regression cross-validation requires at least two outer folds."
        )
    for repeat_no in range(repeats):
        splitter = KFold(
            n_splits=n_splits, shuffle=True, random_state=plan.random_state + repeat_no
        )
        for fold_no, (train_idx, test_idx) in enumerate(
            splitter.split(np.arange(n_samples))
        ):
            out.append(
                {
                    "split_key": f"r{repeat_no}_o{fold_no}",
                    "repeat": repeat_no,
                    "outer_fold": fold_no,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    return out


def _regression_inner_splits(
    plan: Evaluation,
    outer_train_idx: np.ndarray,
    groups: np.ndarray | None,
    split: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if (
        plan.protocol.lower() in {"lodo", "leave_one_dataset_out"}
        and groups is not None
    ):
        local_groups = groups[outer_train_idx]
        result = []
        for group in pd.unique(local_groups):
            val = np.where(local_groups == group)[0]
            train = np.where(local_groups != group)[0]
            if len(train) and len(val):
                result.append((train, val))
        if result:
            return result
    n_splits = min(int(plan.inner_folds), len(outer_train_idx))
    if n_splits < 2:
        return []
    seed = (
        plan.random_state
        + int(split.get("repeat", 0)) * 1009
        + int(split.get("outer_fold", 0))
    )
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (train, val) for train, val in splitter.split(np.arange(len(outer_train_idx)))
    ]


def _resolved_evaluation_splits(
    root: Path,
    plan: Evaluation,
    dataset: Any,
    groups: np.ndarray | None,
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
    group_col: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    task = str(dataset.task).lower()

    def create():
        if task == "regression":
            outer = _regression_outer_splits(plan, len(dataset.y), groups)
            inner = {
                str(split["split_key"]): _regression_inner_splits(
                    plan, np.asarray(split["train_idx"], dtype=int), groups, split
                )
                for split in outer
            }
            return outer, inner
        outer = _outer_splits(plan, dataset.y, groups, strata, stratify_col)
        inner = {
            str(split["split_key"]): _inner_splits(
                plan,
                dataset.y,
                np.asarray(split["train_idx"], dtype=int),
                groups,
                split,
                strata,
                stratify_col,
            )
            for split in outer
        }
        return outer, inner

    outer, inner, _ = resolve_cv_splits(
        root=Path(root),
        sample_ids=tuple(dataset.sample_ids),
        y=dataset.y,
        groups=groups,
        strata=strata,
        task=task,
        target_name=str(dataset.target_name),
        protocol=str(plan.protocol),
        outer_folds=int(plan.outer_folds),
        inner_folds=int(plan.inner_folds),
        repeats=int(plan.repeats),
        random_state=int(plan.random_state),
        group_col=group_col,
        stratify_col=stratify_col,
        create=create,
    )
    return outer, inner


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
    y: np.ndarray,
    sample_ids: Sequence[str],
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
    existing_inner_scores: Sequence[float],
    existing_qualification: dict[str, Any] | None,
    needs_outer: bool,
    random_state: int,
    threads_per_worker: int,
    resource_sample_interval_s: float,
) -> dict[str, Any]:
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
    scores = [float(x) for x in existing_inner_scores if np.isfinite(x)]
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
            _, inner_ct_factory = transformation_factory_builder(
                ct_item, random_state=random_state
            )
            fitted = inner_ct_factory()
            X_inner_train, X_inner_val, _ = _lodo_feature_pair(
                X_base, tr_idx, va_idx, protocol
            )
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
                result["inner_metrics"].append(row)
                result["inner_predictions"].extend(
                    _regression_prediction_rows(
                        inner_key,
                        cid,
                        va_idx,
                        sample_ids,
                        y,
                        pred,
                        "inner",
                        split_key,
                        target_name,
                    )
                )
                score = float(metrics.get(gate_metric, np.nan))
                if np.isfinite(score):
                    scores.append(score)
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
                gate_score = float(np.mean(scores)) if scores else float("nan")
                gate = QualificationGate(
                    enabled=True, metric=gate_metric, threshold=gate_threshold
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
            _, outer_ct_factory = transformation_factory_builder(
                ct_item, random_state=random_state
            )
            fitted_outer = outer_ct_factory()
            X_outer_train, X_outer_test, _ = _lodo_feature_pair(
                X_base, train_idx, test_idx, protocol
            )
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
                result["outer_metrics"].append(row)
                result["outer_predictions"].extend(
                    _regression_prediction_rows(
                        split_key,
                        cid,
                        test_idx,
                        sample_ids,
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
    learners = [_learner_factory(item, task="regression") for item in sweep.learners]
    count_transformation_specs = [
        _count_transformation_spec(x) for x in sweep.count_transformations
    ]
    configs = build_sweep_configs(
        sweep.resolutions, sweep.count_transformations, sweep.learners
    )
    _write_config_table(root, configs)
    if sweep.evaluation.redo:
        _clear_evaluation_checkpoints(root)
    _write_manifest(root, sweep, dataset)
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    outer_splits, inner_splits_by_outer = _resolved_evaluation_splits(
        root,
        sweep.evaluation,
        dataset,
        groups,
        group_col=sweep.data.group_col,
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
    mpdr_cache = {
        name: np.asarray(materialize_mpdr(dataset, lvls)[0], dtype=np.float32)
        for name, lvls in resolutions
    }
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
            X_base = mpdr_cache[res_name]
            for ct_name, ct_spec in count_transformation_specs:
                ct_item = (ct_name, ct_spec) if ct_spec is not None else ct_name
                for learner_name, learner_factory in learners:
                    cid = _config_id(str(ct_name), res_name, learner_name)
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
                        dataset.y,
                        tuple(dataset.sample_ids),
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
            "MPDRs": f"{len(sweep.resolutions) * len(count_transformation_specs):,}",
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
    learner_factories = [_learner_factory(x) for x in sweep.learners]
    count_transformation_specs = [
        _count_transformation_spec(x) for x in sweep.count_transformations
    ]
    configs = build_sweep_configs(
        sweep.resolutions, sweep.count_transformations, sweep.learners
    )
    _write_config_table(root, configs)
    if sweep.evaluation.redo:
        _clear_evaluation_checkpoints(root)
    _write_manifest(root, sweep, dataset)
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
    current_config_ids = set(configs["config_id"].astype(str))
    current_outer_keys = {str(s["split_key"]) for s in outer_splits}
    existing = _load_existing_evaluation(
        root,
        current_config_ids,
        current_outer_keys,
        redo=sweep.evaluation.redo,
    )
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
    mpdr_cache = {
        res_name: np.asarray(materialize_mpdr(dataset, levels)[0], dtype=np.float32)
        for res_name, levels in resolutions
    }
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
            X_base = mpdr_cache[res_name]
            for ct_name, ct_spec in count_transformation_specs:
                ct_item = (ct_name, ct_spec) if ct_spec is not None else ct_name
                for learner_name, learner_factory in learner_factories:
                    cid = _config_id(str(ct_name), res_name, learner_name)
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
                        y,
                        dataset.classes,
                        tuple(dataset.class_labels),
                        tuple(dataset.sample_ids),
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
            "MPDRs": f"{len(sweep.resolutions) * len(count_transformation_specs):,}",
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
        "rankings": root / "tables" / "mpma_rankings.parquet",
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
    return set(zip(df[split_col].astype(str), df["config_id"].astype(str)))


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
) -> list[float]:
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
    ]
    if "ok" in sub.columns:
        sub = sub[sub["ok"].eq(1)]
    vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
    return [float(x) for x in vals if np.isfinite(x)]


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


def _write_manifest(root: Path, sweep: Sweep, dataset: Dataset) -> None:
    safe_sweep = asdict(sweep)
    for section in ["data"]:
        for k, v in list(safe_sweep[section].items()):
            if isinstance(v, Path):
                safe_sweep[section][k] = str(v)
    safe_sweep["experiment_dir"] = str(safe_sweep["experiment_dir"])
    manifest = {
        "package": "mllabiome",
        "terminology": {
            "MPDR": "Microbiome Profile Data Representation = taxonomic resolution + count transformation",
            "MPMA": "Microbiome Profile Modelling Algorithm = MPDR + learner",
        },
        "sweep": safe_sweep,
        "task": dataset.task,
        "target": dataset.target_name,
        "class_labels": dataset.class_labels,
        "n_samples": len(dataset.y),
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
        conn.execute("UPDATE configs SET active=1")
        for r in configs.to_dict(orient="records"):
            conn.execute(
                """INSERT INTO configs
                   (config_id, mpdr_id, count_transformation, resolution, levels, learner, active, candidate_family, modalities, integration, integration_n_components)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                   ON CONFLICT(config_id) DO UPDATE SET
                     mpdr_id=excluded.mpdr_id, count_transformation=excluded.count_transformation,
                     resolution=excluded.resolution, levels=excluded.levels, learner=excluded.learner,
                     active=1, candidate_family=excluded.candidate_family, modalities=excluded.modalities,
                     integration=excluded.integration, integration_n_components=excluded.integration_n_components""",
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
                ),
            )
        conn.commit()
        export = pd.read_sql_query(
            "SELECT * FROM configs ORDER BY active DESC, count_transformation, resolution, learner",
            conn,
        )
    finally:
        conn.close()


def _groups_from_metadata(
    meta: pd.DataFrame, group_col: str | None
) -> np.ndarray | None:
    if not group_col:
        return None
    if group_col not in meta.columns:
        raise ValueError(f"Group column {group_col!r} not found in metadata.")
    return meta[group_col].astype(str).to_numpy()


def _normalise_column_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None or value == "":
        return tuple()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value if str(v))


def _strata_from_metadata(
    meta: pd.DataFrame, y: np.ndarray, stratify_col: str | Sequence[str] | None
) -> np.ndarray:

    cols = _normalise_column_names(stratify_col)
    if not cols:
        return np.asarray(y, dtype=str)
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise ValueError(f"Stratification column(s) not found in metadata: {missing}")
    base = pd.Series(np.asarray(y, dtype=str), index=meta.index).astype(str)
    parts = [base]
    for c in cols:
        vals = meta[c].astype(str).fillna("NA").reset_index(drop=True)
        parts.append(vals)
    joined = parts[0].reset_index(drop=True)
    for vals in parts[1:]:
        joined = joined.str.cat(vals.astype(str), sep="__strata__")
    return joined.to_numpy(dtype=str)


def _safe_n_splits(strata: np.ndarray, requested: int) -> int:
    counts = pd.Series(strata).value_counts()
    if counts.empty:
        return 0
    return int(max(0, min(requested, counts.min(), len(strata))))


def _stratification_error_context(
    plan: Evaluation, stratify_col: str | Sequence[str] | None
) -> str:
    cols = _normalise_column_names(stratify_col)
    if not cols:
        return "target labels"
    return "target labels plus " + ", ".join(cols)


def _outer_splits(
    plan: Evaluation,
    y: np.ndarray,
    groups: np.ndarray | None,
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    protocol = plan.protocol.lower()
    out: list[dict[str, Any]] = []
    if protocol in {"lodo", "leave_one_dataset_out"}:
        if groups is None:
            raise ValueError("LODO requires DATA.group_col.")
        for i, g in enumerate(pd.unique(groups)):
            test_idx = np.where(groups == g)[0]
            train_idx = np.where(groups != g)[0]
            out.append(
                {
                    "split_key": f"lodo_{i}__{g}",
                    "repeat": 0,
                    "outer_fold": i,
                    "outer_group": str(g),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
        return out
    n_repeats = plan.repeats if protocol == "repeated_nested_cv" else 1
    split_strata = np.asarray(strata if strata is not None else y, dtype=str)
    for r in range(n_repeats):
        n_splits = _safe_n_splits(split_strata, plan.outer_folds)
        if n_splits < 2:
            context = _stratification_error_context(plan, stratify_col)
            raise ValueError(
                f"Not enough samples in each stratum for outer cross-validation using {context}. "
                "Reduce outer_folds, remove/merge sparse strata, or omit DATA.stratify_col."
            )
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=plan.random_state + r
        )
        for o, (tr, te) in enumerate(skf.split(np.zeros(len(y)), split_strata)):
            out.append(
                {
                    "split_key": f"r{r}_o{o}",
                    "repeat": r,
                    "outer_fold": o,
                    "train_idx": tr,
                    "test_idx": te,
                }
            )
    return out


def _inner_splits(
    plan: Evaluation,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    groups: np.ndarray | None,
    outer_split: dict[str, Any],
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if (
        plan.protocol.lower() in {"lodo", "leave_one_dataset_out"}
        and groups is not None
    ):
        g_train = groups[outer_train_idx]
        unique = list(pd.unique(g_train))
        if len(unique) >= 2:
            out = []
            for g in unique:
                va = np.where(g_train == g)[0]
                tr = np.where(g_train != g)[0]
                if len(np.unique(y[outer_train_idx[tr]])) >= 2:
                    out.append((tr, va))
            if out:
                return out
    y_train = y[outer_train_idx]
    split_strata = np.asarray(
        strata[outer_train_idx] if strata is not None else y_train, dtype=str
    )
    n_splits = _safe_n_splits(split_strata, plan.inner_folds)
    if n_splits < 2:
        return []
    seed = (
        plan.random_state
        + int(outer_split.get("repeat", 0)) * 1009
        + int(outer_split.get("outer_fold", 0))
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y_train)), split_strata)]


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
    path = root / "results" / "outer_results.parquet"
    if not table_exists(path):
        return
    df = read_table(path)
    if df.empty:
        return
    metrics = [c for c in METRIC_COLUMNS if c in df.columns]
    group_cols = [
        "config_id",
        "mpdr_id",
        "count_transformation",
        "resolution",
        "levels",
        "learner",
    ]
    agg = (
        df[df["ok"].eq(1)]
        .groupby(group_cols, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    rank = agg.reset_index()
    sort_metric = (
        optimize_metric if f"{optimize_metric}_mean" in rank.columns else "nMCC"
    )
    sort_col = f"{sort_metric}_mean"
    rank = rank.sort_values(sort_col, ascending=metric_is_loss(sort_metric))
    rank.insert(0, "rank", np.arange(1, len(rank) + 1))
    write_table(root / "tables" / "mpma_rankings.parquet", rank)
