from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
from .explainability_methods import ALE, ALEInteractions, coerce_method, method_name
from .learners import _learner_factory
from ._historical_configs import (
    HistoricalConfigurationError,
    recorded_configuration,
    recorded_learner_factory,
    recorded_transformation_spec,
)
from .sweep_types import Sweep
from .transformations import (
    CountTransformationAdapter,
    _count_transformation_factory,
    _count_transformation_specs_for_blocks,
)

_EXPLAINABILITY_METHODS = {"shap", "lime", "ale", "permutation", "interactions"}
_EXPLAINABILITY_PIPELINE_SCHEMA = "oof-coordinate-reconstruction-v3-taxon-proxy"


class ExplainabilityConfigurationError(RuntimeError):
    pass


class ExplainabilityDependencyError(RuntimeError):
    pass


def _normalise_explainability_method_specs(methods: Sequence[Any]) -> tuple[Any, ...]:
    if not methods:
        raise ExplainabilityConfigurationError(
            "Explainability.methods must contain at least one explicit method."
        )
    specs: list[Any] = []
    unsupported: list[str] = []
    for method in methods:
        try:
            spec = coerce_method(method)
        except Exception:
            unsupported.append(str(method))
            continue
        if method_name(spec) not in _EXPLAINABILITY_METHODS:
            unsupported.append(str(method))
            continue
        specs.append(spec)
    if unsupported:
        raise ExplainabilityConfigurationError(
            "Unsupported explainability method(s): "
            + ", ".join(unsupported)
            + ". Supported methods are: "
            + ", ".join(sorted(_EXPLAINABILITY_METHODS))
            + "."
        )
    names = [method_name(x) for x in specs]
    if len(set(names)) != len(names):
        raise ExplainabilityConfigurationError(
            "Explainability.methods must not contain duplicate method types."
        )
    return tuple(specs)


def _normalise_explainability_methods(methods: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        method_name(x) for x in _normalise_explainability_method_specs(methods)
    )


def _method_spec(methods: Sequence[Any], name: str) -> Any:
    target = str(name)
    for spec in _normalise_explainability_method_specs(methods):
        if method_name(spec) == target:
            return spec
    raise ExplainabilityConfigurationError(
        f"Explainability method {target!r} is not configured."
    )


def _resolve_explainability_classes(dataset: Any, configured: Any) -> tuple[int, ...]:
    n_classes = len(dataset.class_labels)
    if isinstance(configured, str):
        token = configured.strip().lower()
        if token in {"", "auto"}:
            if n_classes == 2:
                value = getattr(dataset, "positive_class", None)
                return (int(1 if value is None else value),)
            return tuple(range(n_classes))
        if token == "all":
            return tuple(range(n_classes))
        configured = (configured,)
    indices: list[int] = []
    labels = [str(x) for x in dataset.class_labels]
    for value in configured:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
        else:
            text = str(value).strip()
            matches = [
                i
                for i, label in enumerate(labels)
                if label.casefold() == text.casefold()
            ]
            if len(matches) == 1:
                idx = int(matches[0])
            else:
                try:
                    idx = int(text)
                except ValueError as exc:
                    raise ExplainabilityConfigurationError(
                        f"Unknown explainability class {value!r}. Available classes: {labels!r}."
                    ) from exc
        if idx < 0 or idx >= n_classes:
            raise ExplainabilityConfigurationError(
                f"Explainability class index {idx} is outside 0..{n_classes - 1}."
            )
        if idx not in indices:
            indices.append(idx)
    if not indices:
        raise ExplainabilityConfigurationError(
            "No explainability classes were selected."
        )
    return tuple(indices)


def _auto_ale_bins(n_samples: int, spec: ALE | ALEInteractions) -> int:
    if isinstance(spec.bins, (int, np.integer)):
        return max(2, int(spec.bins))
    value = int(np.floor(np.sqrt(max(1, int(n_samples)))))
    return int(max(int(spec.min_bins), min(int(spec.max_bins), value)))


def _preflight_explainability_dependencies(methods: Sequence[str]) -> None:
    missing: list[str] = []
    if "shap" in methods:
        try:
            pass
        except Exception as exc:
            missing.append(f"shap ({exc})")
    if "lime" in methods:
        try:
            pass
        except Exception as exc:
            missing.append(f"lime ({exc})")
    if "ale" in methods or "interactions" in methods:
        try:
            pass
        except Exception as exc:
            missing.append(f"PyALE ({exc})")
    if "interactions" in methods:
        try:
            pass
        except Exception as exc:
            missing.append(f"networkx ({exc})")
    if missing:
        raise ExplainabilityDependencyError(
            "Strict explainability cannot start because requested method dependencies are missing: "
            + "; ".join(missing)
            + ". Install the required package dependencies or explicitly remove the corresponding method(s). No fallback method will be used."
        )


def _configured_count_transformation_factory(
    sweep: Sweep,
    key: str,
    feature_blocks: Any = None,
    *,
    resolution_feature_blocks: Any = None,
    feature_names: Any = None,
    config_id: str | None = None,
) -> Callable[[], CountTransformationAdapter]:
    reference_blocks = (
        feature_blocks
        if resolution_feature_blocks is None
        else resolution_feature_blocks
    )
    specifications = _count_transformation_specs_for_blocks(
        sweep.count_transformations, reference_blocks
    )
    matches = [(name, spec) for name, spec in specifications if str(name) == str(key)]
    if config_id is not None:
        from .configs_sweep import _transformation_fingerprint

        try:
            saved = recorded_configuration(sweep.root(), config_id)
            expected = str(saved.get("transformation_fingerprint"))
            matches = [
                (name, spec)
                for name, spec in matches
                if _transformation_fingerprint(name, spec) == expected
            ]
            if not matches:
                spec = recorded_transformation_spec(sweep.root(), config_id, key)
                matches = [(key, spec)]
        except HistoricalConfigurationError as exc:
            raise ExplainabilityConfigurationError(str(exc)) from exc
    if len(matches) != 1:
        available = [str(name) for name, _ in specifications]
        raise ExplainabilityConfigurationError(
            f"Cannot recover exactly the stored count transformation {key!r}; "
            f"available effective transformations: {available!r}."
        )
    name, spec = matches[0]
    item = (name, spec) if spec is not None else name
    resolved_name, factory = _count_transformation_factory(
        item,
        random_state=sweep.evaluation.random_state,
        feature_blocks=feature_blocks,
        feature_names=feature_names,
    )
    if str(resolved_name) != str(key):
        raise ExplainabilityConfigurationError(
            f"Explainability reconstructed count transformation {resolved_name!r} for "
            f"evaluated transformation {key!r}."
        )
    return factory


def _configured_learner_factory(
    sweep: Sweep,
    key: str,
    *,
    config_id: str | None = None,
    task: str = "classification",
) -> Callable[[], BaseEstimator]:
    from .configs_sweep import _learner_fingerprint

    if config_id is not None:
        try:
            row = recorded_configuration(sweep.root(), config_id)
        except HistoricalConfigurationError as exc:
            raise ExplainabilityConfigurationError(str(exc)) from exc
        if str(row.get("learner")) != str(key):
            raise ExplainabilityConfigurationError(
                f"Configuration {config_id!r} does not record learner {key!r}."
            )
        expected = str(row.get("learner_fingerprint"))
        for item in sweep.learners:
            if str(item[0]) == key and _learner_fingerprint(item) == expected:
                return _learner_factory(item, task=task)[1]
        try:
            return recorded_learner_factory(sweep.root(), config_id, key, task=task)
        except HistoricalConfigurationError as exc:
            raise ExplainabilityConfigurationError(str(exc)) from exc
    factories = dict(_learner_factory(x, task=task) for x in sweep.learners)
    if key not in factories:
        raise ExplainabilityConfigurationError(
            f"Learner {key!r} is not in sweep.learners and no historical config_id was provided."
        )
    return factories[key]
