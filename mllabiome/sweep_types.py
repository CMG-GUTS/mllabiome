from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .data import Data
from .explainability_methods import (
    ALE,
    SHAP,
    Permutation,
    apply_profile,
    method_has_local,
    normalise_profile,
)
from .integrations import Integration
from .learners import validate_model_specs
from .ensemble_aggregation import (
    PROBABILITY_PRESERVING_AGGREGATIONS,
    SUPPORTED_AGGREGATIONS,
)
from .metrics import (
    canonical_metric_name,
    metric_passes_threshold,
    metric_requires_probability_semantics,
)
from .utils import CLASSIFICATION_METRIC_COLUMNS, REGRESSION_METRIC_COLUMNS
from .modalities import Modality, Samples


class SweepTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTILABEL = "multilabel"
    MULTIOUTPUT = "multioutput"

    @classmethod
    def parse(cls, value: Any) -> "SweepTask":
        token = str(value).strip().casefold().replace("-", "_")
        aliases = {
            "binary": cls.CLASSIFICATION.value,
            "multiclass": cls.CLASSIFICATION.value,
            "continuous": cls.REGRESSION.value,
            "multi_output": cls.MULTIOUTPUT.value,
        }
        try:
            return cls(aliases.get(token, token))
        except ValueError as exc:
            raise ValueError(f"Unsupported task {value!r}.") from exc


class LocalExplanationMode(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REPRESENTATIVE = "representative"
    REQUESTED = "requested"
    REPRESENTATIVE_AND_REQUESTED = "representative_and_requested"

    @classmethod
    def parse(cls, value: Any) -> "LocalExplanationMode":
        token = str(value).strip().casefold().replace("-", "_")
        aliases = {
            "": cls.AUTO.value,
            "off": cls.NONE.value,
            "false": cls.NONE.value,
            "representatives": cls.REPRESENTATIVE.value,
            "samples": cls.REQUESTED.value,
            "both": cls.REPRESENTATIVE_AND_REQUESTED.value,
            "representative_requested": cls.REPRESENTATIVE_AND_REQUESTED.value,
        }
        try:
            return cls(aliases.get(token, token))
        except ValueError as exc:
            allowed = ", ".join(repr(mode.value) for mode in cls)
            raise ValueError(
                f"Explainability.local_explanations must be one of {allowed}."
            ) from exc


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
    metric: str = "MCC"
    threshold: float | None = None

    def __post_init__(self) -> None:
        raw = str(self.metric).strip().casefold().replace("-", "_").replace(" ", "_")
        if raw == "nmcc" and self.threshold is not None:
            self.threshold = 2.0 * float(self.threshold) - 1.0
        self.metric = canonical_metric_name(str(self.metric))
        if self.enabled and self.threshold is None:
            raise ValueError(
                "QualificationGate.threshold is required when the gate is enabled."
            )

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
    optimize_metric: Any = "log_loss"
    n_jobs: int | str = 1
    parallel_backend: str = "loky"
    memory_fraction: float = 0.80
    min_worker_memory_gib: float = 1.0
    resource_sample_interval_s: float = 0.10
    diagnostic_thresholds: tuple[float, ...] = ()
    decision_curve_min_threshold: float = 0.01
    decision_curve_max_threshold: float = 0.99
    decision_curve_points: int = 99
    redo: bool = False

    def __post_init__(self) -> None:
        protocol = str(self.protocol).strip().casefold().replace("-", "_")
        if protocol not in {
            "repeated_nested_cv",
            "nested_cv",
            "lodo",
            "leave_one_dataset_out",
        }:
            raise ValueError(f"Unsupported evaluation protocol {self.protocol!r}.")
        self.protocol = protocol
        self.optimize_metric = canonical_metric_name(str(self.optimize_metric))
        if int(self.inner_folds) < 2:
            raise ValueError("Evaluation.inner_folds must be at least 2.")
        if (
            protocol in {"repeated_nested_cv", "nested_cv"}
            and int(self.outer_folds) < 2
        ):
            raise ValueError("Evaluation.outer_folds must be at least 2 for nested CV.")
        if int(self.repeats) < 1:
            raise ValueError("Evaluation.repeats must be at least 1.")


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
    )
    optimize_metric: Any = "auto"

    include_inactive: bool = True

    threshold_score: float = 0.30
    threshold_max_members: int = 50

    exclude_config_ids: tuple[str, ...] = ()
    exclude_learners: tuple[str, ...] = ()
    exclude_resolutions: tuple[str, ...] = ()
    exclude_transformations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.optimize_metric = canonical_metric_name(str(self.optimize_metric))
        self.max_sizes = tuple(int(value) for value in self.max_sizes)
        if self.sizes is not None:
            self.sizes = tuple(int(value) for value in self.sizes)
        self.selection_strategies = tuple(
            str(value) for value in self.selection_strategies
        )
        self.aggregation_strategies = tuple(
            str(value) for value in self.aggregation_strategies
        )
        if not self.max_sizes or any(value < 2 for value in self.max_sizes):
            raise ValueError("Every Ensemble.max_sizes entry must be at least 2.")
        if not self.selection_strategies:
            raise ValueError("Ensemble.selection_strategies must be non-empty.")
        if not self.aggregation_strategies:
            raise ValueError("Ensemble.aggregation_strategies must be non-empty.")


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
    mode = LocalExplanationMode.parse(options.get("local_explanations", "auto"))
    if mode is LocalExplanationMode.NONE:
        representatives = False
        sample_ids = ()
    elif mode is LocalExplanationMode.REPRESENTATIVE:
        representatives = True
        sample_ids = ()
    elif mode is LocalExplanationMode.REQUESTED:
        representatives = False
    elif mode is LocalExplanationMode.REPRESENTATIVE_AND_REQUESTED:
        representatives = True
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


def _configured_class_count(source: Any) -> int | None:
    labels = getattr(source, "class_labels", None)
    if labels is not None:
        return len(tuple(labels))
    label_map = getattr(source, "label_map", None)
    if label_map:
        return len({int(value) for value in label_map.values()})
    return None


def _validate_metric(
    metric: str,
    task: str,
    protocol: str,
    n_classes: int | None,
    role: str,
) -> str:
    canonical = canonical_metric_name(metric)
    if task == "classification":
        allowed = set(CLASSIFICATION_METRIC_COLUMNS)
        if canonical not in allowed:
            raise ValueError(
                f"{role}={metric!r} is not a supported classification metric. "
                f"Supported metrics: {sorted(allowed)}."
            )
        if canonical == "cohort_macro_log_loss" and protocol not in {
            "lodo",
            "leave_one_dataset_out",
        }:
            raise ValueError(
                f"{role}='cohort_macro_log_loss' requires protocol='lodo' or 'leave_one_dataset_out'."
            )
        if n_classes is not None and n_classes > 2:
            binary_only = {
                "AUROC",
                "AUCPR",
                "AP",
                "Sensitivity",
                "Specificity",
                "PPV",
                "NPV",
            }
            if canonical in binary_only:
                alternatives = {
                    "AUROC": "AUROC_macro",
                    "AUCPR": "AUCPR_macro",
                    "AP": "AP_macro",
                }
                suggestion = alternatives.get(canonical)
                suffix = f" Use {suggestion!r}." if suggestion else ""
                raise ValueError(
                    f"{role}={canonical!r} is binary-only when the configured outcome has {n_classes} classes.{suffix}"
                )
    elif task == "regression":
        allowed = set(REGRESSION_METRIC_COLUMNS)
        if canonical not in allowed:
            raise ValueError(
                f"{role}={metric!r} is not a supported regression metric. "
                f"Supported metrics: {sorted(allowed)}."
            )
    else:
        raise ValueError(f"Unsupported task {task!r}.")
    return canonical


def _validate_sweep_configuration(sweep: Any) -> None:
    source = sweep.samples if sweep.uses_modalities else sweep.data
    if source is None:
        raise ValueError("Sweep requires a data or samples specification.")
    task = _normalise_sweep_task(source.task)
    protocol = str(sweep.evaluation.protocol)
    n_classes = _configured_class_count(source) if task == "classification" else None
    sweep.evaluation.optimize_metric = _validate_metric(
        str(sweep.evaluation.optimize_metric),
        task,
        protocol,
        n_classes,
        "Evaluation.optimize_metric",
    )
    if sweep.gate.enabled:
        sweep.gate.metric = _validate_metric(
            str(sweep.gate.metric),
            task,
            protocol,
            n_classes,
            "QualificationGate.metric",
        )
    ensemble_metric = str(sweep.ensemble.optimize_metric).strip()
    if ensemble_metric.casefold() == "auto":
        ensemble_metric = (
            "log_loss"
            if task == "classification"
            else str(sweep.evaluation.optimize_metric)
        )
    sweep.ensemble.optimize_metric = _validate_metric(
        ensemble_metric,
        task,
        protocol,
        n_classes,
        "Ensemble.optimize_metric",
    )
    supported_selection = {
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    }
    unknown_selection = sorted(
        set(sweep.ensemble.selection_strategies) - supported_selection
    )
    if unknown_selection:
        raise ValueError(
            f"Unsupported ensemble selection strategy(s): {unknown_selection}."
        )
    learned = set(sweep.ensemble.selection_strategies) & {"caruana", "super_learner"}
    simple = set(sweep.ensemble.selection_strategies) - learned
    if task == "classification":
        grouped_metrics = {"subject_macro_log_loss", "cohort_macro_log_loss"}
        if sweep.ensemble.optimize_metric in grouped_metrics:
            raise ValueError(
                "Ensemble.optimize_metric does not support grouped macro log-loss objectives; use 'log_loss' for probability-ensemble optimization."
            )
        if (
            sweep.evaluation.optimize_metric == "subject_macro_log_loss"
            or (sweep.gate.enabled and sweep.gate.metric == "subject_macro_log_loss")
        ) and not getattr(source, "subject_id_col", None):
            raise ValueError(
                "subject_macro_log_loss requires an explicit subject_id_col so repeated observations are aggregated by biological subject."
            )
        unknown = sorted(
            set(sweep.ensemble.aggregation_strategies) - set(SUPPORTED_AGGREGATIONS)
        )
        if unknown:
            raise ValueError(
                f"Unsupported classification ensemble aggregation strategy(s): {unknown}."
            )
        if metric_requires_probability_semantics(sweep.ensemble.optimize_metric):
            invalid = sorted(
                set(sweep.ensemble.aggregation_strategies)
                - set(PROBABILITY_PRESERVING_AGGREGATIONS)
            )
            if invalid:
                raise ValueError(
                    f"Ensemble.optimize_metric={sweep.ensemble.optimize_metric!r} requires probability-valued aggregation. "
                    f"Remove incompatible aggregation strategy(s) {invalid}; allowed strategies are "
                    f"{sorted(PROBABILITY_PRESERVING_AGGREGATIONS)}."
                )
        if (
            learned
            and "weighted_mean_proba" not in sweep.ensemble.aggregation_strategies
        ):
            raise ValueError(
                "Caruana and Super Learner require 'weighted_mean_proba' in Ensemble.aggregation_strategies."
            )
        if simple and not any(
            aggregation != "weighted_mean_proba"
            for aggregation in sweep.ensemble.aggregation_strategies
        ):
            raise ValueError(
                "Simple ensemble selectors require at least one non-weighted aggregation strategy."
            )


def validate_sweep_class_count(sweep: Any, n_classes: int) -> None:
    if int(n_classes) < 2:
        raise ValueError("Classification requires at least two outcome classes.")
    protocol = str(sweep.evaluation.protocol)
    sweep.evaluation.optimize_metric = _validate_metric(
        str(sweep.evaluation.optimize_metric),
        "classification",
        protocol,
        int(n_classes),
        "Evaluation.optimize_metric",
    )
    if sweep.gate.enabled:
        sweep.gate.metric = _validate_metric(
            str(sweep.gate.metric),
            "classification",
            protocol,
            int(n_classes),
            "QualificationGate.metric",
        )
    sweep.ensemble.optimize_metric = _validate_metric(
        str(sweep.ensemble.optimize_metric),
        "classification",
        protocol,
        int(n_classes),
        "Ensemble.optimize_metric",
    )


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
        _validate_sweep_configuration(self)

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


def _normalise_sweep_task(value: str) -> str:
    return SweepTask.parse(value).value


def sweep_task(sweep: Sweep) -> str:
    if sweep.uses_modalities:
        if sweep.samples is None:
            raise ValueError("Modality-based sweeps require Samples.")
        return _normalise_sweep_task(sweep.samples.task)
    if sweep.data is None:
        raise ValueError("Data-based sweeps require Data.")
    return _normalise_sweep_task(sweep.data.task)
