from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from .multimodal_fusion import (
    AlignmentKernelFusion,
    BlockPLSFusion,
    InductiveSNFFusion,
    SparseCovarianceFusion,
)
from .mofa_integration import GaussianMOFAFusion


@dataclass(frozen=True)
class Integration:
    name: str
    n_components: tuple[int, ...] | int | None = None
    modality_sets: tuple[tuple[str, ...], ...] | None = None
    min_modalities: int = 2
    max_modalities: int | None = None
    sparsity: float = 0.25
    n_neighbors: int = 10
    n_iterations: int = 15
    mofa_max_iter: int = 500

    @property
    def key(self) -> str:
        value = str(self.name).strip().casefold().replace("-", "_")
        if "view" in value:
            raise ValueError(
                "Integration terminology uses modality, not view. Use intermediate_modality_pca and modality_sets/min_modalities/max_modalities."
            )
        aliases = {
            "single": "unimodal",
            "concat": "early_concat",
            "early": "early_concat",
            "modality_pca": "intermediate_modality_pca",
            "joint_pca": "intermediate_joint_pca",
            "mean_proba": "late_mean_proba",
            "weighted_mean_proba": "late_weighted_mean_proba",
            "mean_prediction": "late_mean_prediction",
            "weighted_mean_prediction": "late_weighted_mean_prediction",
            "median_prediction": "late_median_prediction",
            "super_learner": "late_super_learner",
            "mkl": "intermediate_mkl",
            "block_pls": "intermediate_block_pls",
            "sparse_cross_covariance": "intermediate_sparse_cross_covariance",
            "snf_inductive": "intermediate_snf_inductive",
            "mofa": "intermediate_mofa",
            "mofa_plus": "intermediate_mofa",
        }
        return aliases.get(value, value)

    @property
    def stage(self) -> str:
        if self.key == "unimodal":
            return "unimodal"
        if self.key.startswith("early_"):
            return "early"
        if self.key.startswith("intermediate_"):
            return "intermediate"
        if self.key.startswith("late_"):
            return "late"
        raise ValueError(f"Unknown integration {self.name!r}.")

    def component_values(self) -> tuple[int | None, ...]:
        if self.n_components is None:
            return (None,)
        if isinstance(self.n_components, int):
            return (int(self.n_components),)
        values = tuple(int(x) for x in self.n_components)
        if not values or any(x < 1 for x in values):
            raise ValueError("Integration n_components must contain positive integers.")
        return values


@dataclass(frozen=True)
class IntegratedCoordinate:
    name: str
    coordinate_type: str
    modality: str | None
    source_coordinates: tuple[str, ...]
    coefficients: tuple[float, ...]
    exact_feature_identity: bool

    @property
    def anchor_feature(self) -> str | None:
        if self.exact_feature_identity and len(self.source_coordinates) == 1:
            return self.source_coordinates[0]
        return None

    @property
    def components(self) -> tuple[str, ...]:
        return self.source_coordinates


class IntegrationModel:
    def __init__(self, spec: Integration, random_state: int = 42) -> None:
        self.spec = spec
        self.random_state = int(random_state)
        self.models_: dict[str, PCA] = {}
        self.joint_model_: PCA | None = None
        self.coordinates_: list[IntegratedCoordinate] = []
        self.fusion_model_: Any = None

    def fit_transform_pair(
        self,
        train: Mapping[str, np.ndarray],
        test: Mapping[str, np.ndarray],
        coordinate_names: Mapping[str, Sequence[str]],
        coordinate_metadata: Mapping[str, Sequence[Any]] | None = None,
        y_train: np.ndarray | None = None,
        task: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[IntegratedCoordinate]]:
        key = self.spec.key
        names = tuple(train)
        if key == "unimodal":
            if len(names) != 1:
                raise ValueError("Unimodal integration requires exactly one modality.")
            modality = names[0]
            Xtr = np.asarray(train[modality], dtype=float)
            Xte = np.asarray(test[modality], dtype=float)
            source_meta = (
                None
                if coordinate_metadata is None
                else coordinate_metadata.get(modality)
            )
            if source_meta:
                coords = [
                    IntegratedCoordinate(
                        str(item.name),
                        str(item.coordinate_type),
                        modality,
                        tuple(str(x) for x in item.components),
                        tuple(float(x) for x in item.coefficients),
                        bool(item.exact_feature_identity),
                    )
                    for item in source_meta
                ]
            else:
                coords = [
                    IntegratedCoordinate(
                        str(name),
                        "feature_coordinate",
                        modality,
                        (str(name),),
                        (1.0,),
                        True,
                    )
                    for name in coordinate_names[modality]
                ]
            self.coordinates_ = coords
            return Xtr, Xte, coords
        if key == "early_concat":
            return self._concat(train, test, coordinate_names, coordinate_metadata)
        if key == "intermediate_modality_pca":
            return self._modality_pca(train, test, coordinate_names)
        if key == "intermediate_joint_pca":
            return self._joint_pca(train, test, coordinate_names)
        if key in {
            "intermediate_mkl",
            "intermediate_block_pls",
            "intermediate_sparse_cross_covariance",
            "intermediate_snf_inductive",
            "intermediate_mofa",
        }:
            return self._fusion_pair(train, test, coordinate_names, y_train, task)
        raise ValueError(
            f"Integration {self.spec.name!r} is not a feature-space integration."
        )

    def transform(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        key = self.spec.key
        names = tuple(data)
        if key == "unimodal":
            if len(names) != 1:
                raise ValueError("Unimodal integration requires exactly one modality.")
            return np.asarray(data[names[0]], dtype=float)
        if key == "early_concat":
            return np.concatenate(
                [np.asarray(data[modality], dtype=float) for modality in names], axis=1
            )
        if key == "intermediate_modality_pca":
            missing = [modality for modality in names if modality not in self.models_]
            if missing:
                raise ValueError(
                    f"Integration model is missing fitted PCA model(s) for {missing!r}."
                )
            return np.concatenate(
                [
                    self.models_[modality].transform(
                        np.asarray(data[modality], dtype=float)
                    )
                    for modality in names
                ],
                axis=1,
            )
        if key in {
            "intermediate_mkl",
            "intermediate_block_pls",
            "intermediate_sparse_cross_covariance",
            "intermediate_snf_inductive",
            "intermediate_mofa",
        }:
            if self.fusion_model_ is None:
                raise ValueError("Fusion integration model is not fitted.")
            return np.asarray(self.fusion_model_.transform(data), dtype=float)
        if key == "intermediate_joint_pca":
            if self.joint_model_ is None:
                raise ValueError("Joint PCA integration model is not fitted.")
            X = np.concatenate(
                [np.asarray(data[modality], dtype=float) for modality in names], axis=1
            )
            return self.joint_model_.transform(X)
        raise ValueError(
            f"Integration {self.spec.name!r} is not a feature-space integration."
        )

    def _fusion_pair(self, train, test, coordinate_names, y_train, task):
        key = self.spec.key
        requested = self.spec.component_values()[0]
        if requested is not None and requested < 1:
            raise ValueError("Fusion n_components must be a positive integer.")
        if key == "intermediate_mkl":
            if y_train is None:
                raise ValueError(
                    "Supervised kernel fusion requires fold-training outcomes."
                )
            model = AlignmentKernelFusion(requested).fit(train, y_train, task)
        elif key == "intermediate_block_pls":
            if y_train is None:
                raise ValueError(
                    "Supervised block PLS requires fold-training outcomes."
                )
            model = BlockPLSFusion(requested).fit(train, y_train, task)
        elif key == "intermediate_mofa":
            model = GaussianMOFAFusion(
                requested if requested is not None else 8,
                max_iter=self.spec.mofa_max_iter,
                random_state=self.random_state,
            ).fit(train)
        elif key == "intermediate_sparse_cross_covariance":
            model = SparseCovarianceFusion(requested, self.spec.sparsity).fit(train)
        else:
            model = InductiveSNFFusion(
                requested, self.spec.n_neighbors, self.spec.n_iterations
            ).fit(train)
        self.fusion_model_ = model
        Xtr = np.asarray(model.transform(train), dtype=float)
        Xte = np.asarray(model.transform(test), dtype=float)
        coords = []
        if key == "intermediate_sparse_cross_covariance":
            for name in model.modalities_:
                source = tuple(str(value) for value in coordinate_names[name])
                loadings = model.loadings_[name] / model.scales_[name][:, None]
                for i, weights in enumerate(loadings.T):
                    coords.append(
                        IntegratedCoordinate(
                            f"{name}::SPARSE_COV_{i + 1:04d}",
                            "sparse_cross_covariance_component",
                            name,
                            source,
                            tuple(float(value) for value in weights),
                            False,
                        )
                    )
        elif key == "intermediate_mofa":
            source = tuple(
                str(value)
                for name in model.modalities_
                for value in coordinate_names[name]
            )
            for i, weights in enumerate(model.projection_.T):
                coords.append(
                    IntegratedCoordinate(
                        f"MOFA_{i + 1:04d}",
                        "gaussian_mofa_posterior_factor",
                        None,
                        source,
                        tuple(float(value) for value in weights),
                        False,
                    )
                )
        elif key == "intermediate_block_pls":
            source = tuple(
                str(value)
                for name in model.modalities_
                for value in coordinate_names[name]
            )
            scale = np.concatenate([model.scales_[name] for name in model.modalities_])
            for i, weights in enumerate(model.model_.x_rotations_.T):
                coords.append(
                    IntegratedCoordinate(
                        f"BLOCK_PLS_{i + 1:04d}",
                        "supervised_block_pls_component",
                        None,
                        source,
                        tuple(float(value) for value in weights / scale),
                        False,
                    )
                )
        else:
            source = tuple(
                str(value) for name in train for value in coordinate_names[name]
            )
            kind = (
                "supervised_kernel_component"
                if key == "intermediate_mkl"
                else "inductive_snf_component"
            )
            for i in range(Xtr.shape[1]):
                coords.append(
                    IntegratedCoordinate(
                        f"{key.upper()}_{i + 1:04d}",
                        kind,
                        None,
                        source,
                        (),
                        False,
                    )
                )
        if Xtr.shape[1] != len(coords):
            raise RuntimeError(
                "Fusion feature coordinates do not match the integrated matrix."
            )
        self.coordinates_ = coords
        return Xtr, Xte, coords

    def _concat(self, train, test, coordinate_names, coordinate_metadata=None):
        tr_parts = []
        te_parts = []
        coords = []
        for modality in train:
            tr_parts.append(np.asarray(train[modality], dtype=float))
            te_parts.append(np.asarray(test[modality], dtype=float))
            source_meta = (
                None
                if coordinate_metadata is None
                else coordinate_metadata.get(modality)
            )
            if source_meta:
                for item in source_meta:
                    label = f"{modality}::{item.name}"
                    coords.append(
                        IntegratedCoordinate(
                            label,
                            str(item.coordinate_type),
                            modality,
                            tuple(str(x) for x in item.components),
                            tuple(float(x) for x in item.coefficients),
                            bool(item.exact_feature_identity),
                        )
                    )
            else:
                for name in coordinate_names[modality]:
                    label = f"{modality}::{name}"
                    coords.append(
                        IntegratedCoordinate(
                            label,
                            "feature_coordinate",
                            modality,
                            (str(name),),
                            (1.0,),
                            True,
                        )
                    )
        self.coordinates_ = coords
        return (
            np.concatenate(tr_parts, axis=1),
            np.concatenate(te_parts, axis=1),
            coords,
        )

    def _n_components(self, shape: tuple[int, int]) -> int:
        requested = self.spec.component_values()[0]
        limit = max(1, min(shape[0], shape[1]))
        return limit if requested is None else min(int(requested), limit)

    def _modality_pca(self, train, test, coordinate_names):
        tr_parts = []
        te_parts = []
        coords = []
        for modality in train:
            Xtr = np.asarray(train[modality], dtype=float)
            Xte = np.asarray(test[modality], dtype=float)
            model = PCA(
                n_components=self._n_components(Xtr.shape),
                svd_solver="full",
                random_state=self.random_state,
            )
            tr_parts.append(model.fit_transform(Xtr))
            te_parts.append(model.transform(Xte))
            self.models_[modality] = model
            source = tuple(str(x) for x in coordinate_names[modality])
            for i, row in enumerate(np.asarray(model.components_, dtype=float)):
                coords.append(
                    IntegratedCoordinate(
                        f"{modality}::PCA_{i + 1:04d}",
                        "pca_component",
                        modality,
                        source,
                        tuple(float(x) for x in row),
                        False,
                    )
                )
        self.coordinates_ = coords
        return (
            np.concatenate(tr_parts, axis=1),
            np.concatenate(te_parts, axis=1),
            coords,
        )

    def _joint_pca(self, train, test, coordinate_names):
        Xtr, Xte, source_coords = self._concat(train, test, coordinate_names, None)
        model = PCA(
            n_components=self._n_components(Xtr.shape),
            svd_solver="full",
            random_state=self.random_state,
        )
        Ztr = model.fit_transform(Xtr)
        Zte = model.transform(Xte)
        self.joint_model_ = model
        source = tuple(c.name for c in source_coords)
        coords = [
            IntegratedCoordinate(
                f"JOINT_PCA_{i + 1:04d}",
                "joint_pca_component",
                None,
                source,
                tuple(float(x) for x in row),
                False,
            )
            for i, row in enumerate(np.asarray(model.components_, dtype=float))
        ]
        self.coordinates_ = coords
        return Ztr, Zte, coords


def integration_modality_sets(
    spec: Integration, modality_names: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    names = tuple(str(x) for x in modality_names)
    if spec.modality_sets is not None:
        out = tuple(tuple(str(x) for x in values) for values in spec.modality_sets)
        unknown = sorted({x for values in out for x in values if x not in names})
        if unknown:
            raise ValueError(
                f"Integration {spec.name!r} references unknown modalities: {unknown!r}."
            )
        expected = 1 if spec.stage == "unimodal" else 2
        if any(
            len(set(values)) != len(values) or len(values) < expected for values in out
        ):
            raise ValueError(
                "Integration modality sets must contain distinct modalities and satisfy the required minimum size."
            )
        if spec.key == "intermediate_sparse_cross_covariance" and any(
            len(values) != 2 for values in out
        ):
            raise ValueError(
                "Sparse cross-covariance integration requires exactly two modalities."
            )
        return out
    if spec.stage == "unimodal":
        return tuple((name,) for name in names)
    minimum = max(2, int(spec.min_modalities))
    maximum = (
        len(names)
        if spec.max_modalities is None
        else min(len(names), int(spec.max_modalities))
    )
    if spec.key == "intermediate_sparse_cross_covariance":
        minimum = max(minimum, 2)
        maximum = min(maximum, 2)
    return tuple(
        values
        for size in range(minimum, maximum + 1)
        for values in combinations(names, size)
    )
