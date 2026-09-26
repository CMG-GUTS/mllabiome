from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .integrations import Integration, IntegrationModel, integration_modality_sets
from .metrics import _estimator_call
from .modalities import ModalityDataset
from .resolutions import mask_feature_blocks, materialize_mpdr_with_blocks
from .transformations import _count_transformation_specs_for_blocks


@dataclass(frozen=True)
class ModalityPath:
    modality: str
    representation: str
    selector: tuple[str, ...]
    transformation: str
    transformation_item: Any


@dataclass(frozen=True)
class CandidateSpec:
    config_id: str
    family: str
    modality_paths: tuple[ModalityPath, ...]
    integration: Integration
    n_components: int | None
    learner: str
    learner_fingerprint: str

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(path.modality for path in self.modality_paths)

    @property
    def representation_label(self) -> str:
        return "+".join(f"{p.modality}:{p.representation}" for p in self.modality_paths)

    @property
    def transformation_label(self) -> str:
        return "+".join(f"{p.modality}:{p.transformation}" for p in self.modality_paths)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _candidate_id(
    paths: Sequence[ModalityPath],
    integration: Integration,
    n_components: int | None,
    learner: str,
    learner_fingerprint: str,
) -> str:
    from .configs_sweep import _scientific_digest, _scientific_value

    payload = {
        "paths": [
            {
                "modality": p.modality,
                "representation": p.representation,
                "selector": p.selector,
                "transformation": p.transformation,
                "transformation_spec": _scientific_value(p.transformation_item),
            }
            for p in paths
        ],
        "integration": _scientific_value(integration),
        "n_components": n_components,
        "learner": str(learner),
        "learner_fingerprint": str(learner_fingerprint),
        "semantics": "modality_candidate_nested_v3",
    }
    return _scientific_digest(payload)[:12]


def _representation_specs(
    modality: str, representations: Mapping[str, Sequence[Any]] | None
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not representations or modality not in representations:
        return (("all", ("all",)),)
    out = []
    for item in representations[modality]:
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


def _normalise_modality_transformations(
    modality_names: Sequence[str],
    primary_modality: str,
    transformations: Any,
) -> dict[str, tuple[Any, ...]]:
    names = tuple(str(name) for name in modality_names)
    primary = str(primary_modality)
    if primary not in names:
        raise ValueError(
            f"Primary modality {primary!r} is not present in the loaded modalities."
        )
    if transformations is None:
        return {}
    if isinstance(transformations, Mapping):
        unknown = sorted(
            str(name) for name in transformations if str(name) not in names
        )
        if unknown:
            raise ValueError(
                f"TRANSFORMATIONS references unknown modalities: {unknown!r}."
            )
        return {str(name): tuple(values) for name, values in transformations.items()}
    if isinstance(transformations, (str, bytes)):
        raise TypeError(
            "TRANSFORMATIONS must be a per-modality mapping or a sequence of transformation specifications."
        )
    try:
        values = tuple(transformations)
    except TypeError as exc:
        raise TypeError(
            "TRANSFORMATIONS must be a per-modality mapping or a sequence of transformation specifications."
        ) from exc
    if not values:
        return {}
    return {primary: values}


def _transformation_specs(
    modality: str,
    transformations: Mapping[str, Sequence[Any]] | None,
    feature_blocks: Any = None,
) -> tuple[tuple[str, Any], ...]:
    items = (
        ("identity",)
        if not transformations or modality not in transformations
        else tuple(transformations[modality])
    )
    out: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        specs = _count_transformation_specs_for_blocks((item,), feature_blocks)
        if not specs:
            continue
        name, spec = specs[0]
        if name in seen:
            continue
        seen.add(name)
        effective_item = (
            spec if spec is not None and not isinstance(item, tuple) else item
        )
        out.append((str(name), effective_item))
    return tuple(out)


def _modality_paths(
    modality_names: Sequence[str],
    representations,
    transformations,
    feature_blocks_by_path: Mapping[Any, Any] | None = None,
) -> dict[str, tuple[ModalityPath, ...]]:
    result = {}
    block_map = {} if feature_blocks_by_path is None else dict(feature_blocks_by_path)
    for modality in modality_names:
        paths = []
        for rep_name, selector in _representation_specs(modality, representations):
            key = (modality, rep_name, selector)
            for tr_name, tr_item in _transformation_specs(
                modality, transformations, block_map.get(key)
            ):
                paths.append(
                    ModalityPath(modality, rep_name, selector, tr_name, tr_item)
                )
        result[modality] = tuple(paths)
    return result


def build_modality_candidates(
    modality_names: Sequence[str],
    representations,
    transformations,
    integrations: Sequence[Integration],
    learners: Sequence[Any],
    learner_name_fn,
    feature_blocks_by_path: Mapping[Any, Any] | None = None,
) -> tuple[list[CandidateSpec], pd.DataFrame]:
    paths = _modality_paths(
        modality_names, representations, transformations, feature_blocks_by_path
    )
    from .configs_sweep import (
        _learner_fingerprint,
        _learner_payload,
        _scientific_digest,
        _scientific_value,
    )

    learner_specs = [
        (learner_name_fn(item), _learner_fingerprint(item), _learner_payload(item))
        for item in learners
    ]
    specs: list[CandidateSpec] = []
    candidate_integrations = [item for item in integrations if item.stage != "late"]
    late_integrations = [item for item in integrations if item.stage == "late"]
    required_late_modalities = {
        modality
        for integration in late_integrations
        for modality_set in integration_modality_sets(integration, modality_names)
        for modality in modality_set
    }
    covered_unimodal_modalities = {
        modality_set[0]
        for integration in candidate_integrations
        if integration.key == "unimodal"
        for modality_set in integration_modality_sets(integration, modality_names)
        if len(modality_set) == 1
    }
    missing_late_modalities = tuple(
        name
        for name in modality_names
        if name in required_late_modalities and name not in covered_unimodal_modalities
    )
    if missing_late_modalities:
        candidate_integrations.insert(
            0,
            Integration(
                "unimodal",
                modality_sets=tuple((name,) for name in missing_late_modalities),
            ),
        )
    for integration in candidate_integrations:
        for modality_set in integration_modality_sets(integration, modality_names):
            path_lists = [paths[name] for name in modality_set]
            for selected in itertools.product(*path_lists):
                for n_components in integration.component_values():
                    for learner, learner_fingerprint, learner_payload in learner_specs:
                        cid = _candidate_id(
                            selected,
                            integration,
                            n_components,
                            learner,
                            learner_fingerprint,
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
                                learner_fingerprint,
                            )
                        )
    unique = {spec.config_id: spec for spec in specs}
    specs = [unique[key] for key in sorted(unique)]
    learner_payloads = {name: payload for name, _, payload in learner_specs}
    rows = []
    for spec in specs:
        learner_payload = learner_payloads[spec.learner]
        transformation_fingerprint = _scientific_digest(
            [
                {
                    "modality": path.modality,
                    "transformation": path.transformation,
                    "spec": _scientific_value(path.transformation_item),
                }
                for path in spec.modality_paths
            ]
        )
        resolution_fingerprint = _scientific_digest(
            [
                {
                    "modality": path.modality,
                    "representation": path.representation,
                    "selector": path.selector,
                }
                for path in spec.modality_paths
            ]
        )
        rows.append(
            {
                "config_id": spec.config_id,
                "mpdr_id": _scientific_digest(
                    {
                        "representation": spec.representation_label,
                        "transformation_fingerprint": transformation_fingerprint,
                    }
                )[:12],
                "count_transformation": spec.transformation_label,
                "transformation_fingerprint": transformation_fingerprint,
                "resolution_fingerprint": resolution_fingerprint,
                "resolution": spec.representation_label,
                "levels": ";".join(
                    f"{p.modality}:{','.join(p.selector)}" for p in spec.modality_paths
                ),
                "learner": spec.learner,
                "learner_display": spec.learner,
                "learner_fingerprint": spec.learner_fingerprint,
                "learner_class": learner_payload["class"],
                "learner_params": json.dumps(
                    learner_payload["params"], sort_keys=True, separators=(",", ":")
                ),
                "active": 1,
                "candidate_family": spec.family,
                "modalities": ",".join(spec.modalities),
                "integration": spec.integration.key,
                "integration_n_components": ""
                if spec.n_components is None
                else str(int(spec.n_components)),
            }
        )
    return specs, pd.DataFrame(rows)


def _materialize_modality_representation(
    dataset: ModalityDataset, path: ModalityPath
) -> tuple[np.ndarray, list[str], Any]:
    ds = dataset.as_dataset(path.modality)
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
        X, names, feature_blocks = materialize_mpdr_with_blocks(ds, selector)
        return np.asarray(X, dtype=np.float32), list(names), feature_blocks
    all_names = list(ds.feature_names_by_level["all"])
    index = {name: i for i, name in enumerate(all_names)}
    missing = [name for name in selector if name not in index]
    if missing:
        raise ValueError(
            f"Representation {path.representation!r} for Modality {path.modality!r} references unknown features: {missing[:8]!r}."
        )
    columns = np.asarray([index[name] for name in selector], dtype=int)
    selected_names = [all_names[i] for i in columns]
    feature_blocks = (("all", tuple(range(len(selected_names)))),)
    return (
        np.asarray(ds.X_by_level["all"][:, columns], dtype=np.float32),
        selected_names,
        feature_blocks,
    )


def _representation_cache(dataset: ModalityDataset, representations):
    matrices = {}
    names = {}
    feature_blocks_by_path = {}
    for modality in dataset.modalities:
        for rep_name, selector in _representation_specs(modality, representations):
            path = ModalityPath(modality, rep_name, selector, "identity", "identity")
            key = (modality, rep_name, selector)
            X, feature_names, feature_blocks = _materialize_modality_representation(
                dataset, path
            )
            matrices[key] = X
            names[key] = (feature_names, feature_blocks)
            feature_blocks_by_path[key] = feature_blocks
    return matrices, names, feature_blocks_by_path


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
    transformation_models = {}
    for path in spec.modality_paths:
        X = matrices[(path.modality, path.representation, path.selector)]
        feature_names, feature_blocks = names[
            (path.modality, path.representation, path.selector)
        ]
        Xtr_raw, Xte_raw, mask = lodo_feature_pair(X, train_idx, test_idx, protocol)
        kept_names = [
            name
            for name, keep in zip(feature_names, np.asarray(mask, dtype=bool))
            if bool(keep)
        ]
        local_blocks = mask_feature_blocks(feature_blocks, mask)
        _, factory = transformation_factory_builder(
            path.transformation_item,
            random_state=random_state,
            feature_blocks=local_blocks,
        )
        fitted = factory()
        Xtr, Xte = fitted.apply_pair(Xtr_raw, Xte_raw)
        transformation_models[path.modality] = fitted
        tr_blocks[path.modality] = np.asarray(Xtr, dtype=float)
        te_blocks[path.modality] = np.asarray(Xte, dtype=float)
        if hasattr(fitted, "get_feature_names_out"):
            coordinate_names[path.modality] = list(
                fitted.get_feature_names_out(kept_names)
            )
        else:
            coordinate_names[path.modality] = kept_names
        if hasattr(fitted, "coordinate_metadata"):
            coordinate_metadata[path.modality] = list(
                fitted.coordinate_metadata(kept_names)
            )
    integration = Integration(
        spec.integration.name,
        n_components=spec.n_components,
        modality_sets=spec.integration.modality_sets,
        min_modalities=spec.integration.min_modalities,
        max_modalities=spec.integration.max_modalities,
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
    for modality in tr_blocks:
        stop = start + int(tr_blocks[modality].shape[1])
        slices[modality] = slice(start, stop)
        start = stop
    return {
        "X_train": np.asarray(Xtr, dtype=float),
        "X_test": np.asarray(Xte, dtype=float),
        "coordinates": coords,
        "X_train_source": np.asarray(Xtr_source, dtype=float),
        "X_test_source": np.asarray(Xte_source, dtype=float),
        "source_coordinates": source_coords,
        "integration_model": model,
        "modality_slices": slices,
        "transformation_models": transformation_models,
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


class _ModalityInputProjector:
    def __init__(self, transformation_models, modality_slices):
        self.transformation_models = dict(transformation_models)
        self.modality_slices = dict(modality_slices)

    def geometry(self) -> str:
        parts = []
        for modality, model in self.transformation_models.items():
            value = (
                model.perturbation_geometry()
                if hasattr(model, "perturbation_geometry")
                else "unverified_custom"
            )
            parts.append(f"{modality}:{value}")
        return "multimodal[" + ",".join(parts) + "]"

    def requires_projection(self) -> bool:
        geometry = self.geometry()
        constrained = (
            "simplex",
            "sphere",
            "clr",
            "rank",
            "binary",
        )
        return any(token in geometry for token in constrained)

    def __call__(self, X):
        arr = np.asarray(X, dtype=float).copy()
        for modality, slc in self.modality_slices.items():
            model = self.transformation_models.get(modality)
            if model is None or not hasattr(model, "project_model_input"):
                continue
            arr[:, slc] = np.asarray(
                model.project_model_input(arr[:, slc]), dtype=float
            )
        return arr


class _IntegratedRegressionPredictor:
    def __init__(self, estimator, integration_model, modality_slices):
        self.estimator = estimator
        self.integration_model = integration_model
        self.modality_slices = dict(modality_slices)

    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        blocks = {
            modality: arr[:, slc] for modality, slc in self.modality_slices.items()
        }
        return np.asarray(
            _estimator_call(
                self.estimator, "predict", self.integration_model.transform(blocks)
            ),
            dtype=float,
        )


class _IntegratedClassificationPredictor:
    def __init__(self, estimator, integration_model, modality_slices):
        self.estimator = estimator
        self.integration_model = integration_model
        self.modality_slices = dict(modality_slices)
        self.classes_ = getattr(estimator, "classes_", None)

    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        blocks = {
            modality: arr[:, slc] for modality, slc in self.modality_slices.items()
        }
        return np.asarray(
            _estimator_call(
                self.estimator, "predict", self.integration_model.transform(blocks)
            )
        )

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        blocks = {
            modality: arr[:, slc] for modality, slc in self.modality_slices.items()
        }
        return np.asarray(
            _estimator_call(
                self.estimator,
                "predict_proba",
                self.integration_model.transform(blocks),
            ),
            dtype=float,
        )
