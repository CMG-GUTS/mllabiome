from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    Dataset,
    _dataset_from_feature_matrix,
    _encode_regression,
    _encode_y,
    _normalise_task,
    _read_feature_by_sample_tsv,
)


@dataclass(frozen=True)
class Samples:
    path: Path | str
    sample_id_col: str = "sample_id"
    target_col: str | tuple[str, ...] = "label"
    task: str = "classification"
    target_tasks: Mapping[str, str] | None = None
    group_col: str | None = None
    stratify_col: str | tuple[str, ...] | None = None
    metadata_cols: tuple[str, ...] = ()
    label_map: Mapping[Any, int] | None = None
    class_labels: tuple[str, ...] | None = None
    positive_class: int | str = 1
    target_class_labels: Mapping[str, tuple[str, ...]] | None = None
    target_positive_classes: Mapping[str, int | str] | None = None


@dataclass(frozen=True)
class Modality:
    name: str
    path: Path | str
    format: str = "table"
    primary: bool = False
    sample_id_col: str | None = None


@dataclass
class ModalityMatrix:
    name: str
    X: np.ndarray
    feature_names: list[str]
    format: str
    dataset: Dataset


@dataclass
class ModalityDataset:
    samples: Samples
    sample_ids: list[str]
    y: np.ndarray
    metadata: pd.DataFrame
    class_labels: list[str]
    positive_class: int | None
    task: str
    target_name: str
    primary_modality: str
    modalities: dict[str, ModalityMatrix]

    @property
    def classes(self) -> np.ndarray:
        return np.arange(len(self.class_labels), dtype=int)

    def as_dataset(self, modality_name: str) -> Dataset:
        return self.modalities[str(modality_name)].dataset


def _read_samples(
    samples: Samples,
) -> tuple[pd.DataFrame, str, np.ndarray, list[str], int | None, str]:
    meta = pd.read_csv(Path(samples.path), sep=None, engine="python", dtype=str)
    if samples.sample_id_col not in meta.columns:
        raise ValueError(
            f"Samples table missing sample ID column {samples.sample_id_col!r}."
        )
    if not isinstance(samples.target_col, str):
        raise ValueError(
            "Multi-target modality sweeps must be expanded before loading modalities."
        )
    target = str(samples.target_col)
    if target not in meta.columns:
        raise ValueError(f"Samples table missing target column {target!r}.")
    meta[samples.sample_id_col] = meta[samples.sample_id_col].astype(str).str.strip()
    if meta[samples.sample_id_col].duplicated().any():
        duplicates = meta.loc[
            meta[samples.sample_id_col].duplicated(), samples.sample_id_col
        ].tolist()
        raise ValueError(
            f"Samples table contains duplicate sample IDs: {duplicates[:5]!r}."
        )
    task = _normalise_task(samples.task)
    if task == "regression":
        y = _encode_regression(meta[target].tolist())
        labels: list[str] = []
        positive = None
    else:
        y, labels, positive = _encode_y(
            meta[target].tolist(),
            samples.label_map,
            samples.class_labels,
            samples.positive_class,
        )
    return meta, target, y, labels, positive, task


def _read_table_modality(
    modality: Modality, samples: Samples
) -> tuple[list[str], np.ndarray, list[str]]:
    frame = pd.read_csv(Path(modality.path), sep=None, engine="python")
    id_col = str(modality.sample_id_col or samples.sample_id_col)
    if id_col not in frame.columns:
        raise ValueError(
            f"Modality {modality.name!r} is missing sample ID column {id_col!r}."
        )
    frame[id_col] = frame[id_col].astype(str).str.strip()
    if frame[id_col].duplicated().any():
        duplicates = frame.loc[frame[id_col].duplicated(), id_col].tolist()
        raise ValueError(
            f"Modality {modality.name!r} contains duplicate sample IDs: {duplicates[:5]!r}."
        )
    feature_cols = [str(c) for c in frame.columns if str(c) != id_col]
    if not feature_cols:
        raise ValueError(f"Modality {modality.name!r} contains no feature columns.")
    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = [str(c) for c in numeric.columns[numeric.isna().any()].tolist()]
        raise ValueError(
            f"Modality {modality.name!r} contains missing or non-numeric feature values in columns: {bad[:8]!r}."
        )
    X = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(
            f"Modality {modality.name!r} contains NaN or infinite feature values."
        )
    return frame[id_col].tolist(), X, feature_cols


def _read_feature_matrix_modality(
    modality: Modality, samples: Samples, sample_ids: list[str]
) -> tuple[list[str], np.ndarray, list[str]]:
    bio, available = _read_feature_by_sample_tsv(Path(modality.path), sample_ids)
    X = bio[available].T.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(
            f"Modality {modality.name!r} contains NaN or infinite feature values."
        )
    return available, X, bio.index.astype(str).tolist()


def load_modalities(
    samples: Samples, modalities: Sequence[Modality]
) -> ModalityDataset:
    items = tuple(modalities)
    if not items:
        raise ValueError("At least one Modality is required.")
    names = [str(v.name).strip() for v in items]
    if any(not name for name in names):
        raise ValueError("Every Modality requires a non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError("Modality names must be unique.")
    primary_items = [v for v in items if bool(v.primary)]
    if len(primary_items) != 1:
        raise ValueError(
            f"Exactly one primary Modality is required; found {len(primary_items)}."
        )
    meta, target, y_all, labels, positive, task = _read_samples(samples)
    meta_index = meta.set_index(samples.sample_id_col, drop=False)
    metadata_ids = meta[samples.sample_id_col].astype(str).tolist()
    loaded: dict[str, tuple[list[str], np.ndarray, list[str], str]] = {}
    for modality in items:
        fmt = str(modality.format).strip().casefold().replace("-", "_")
        if fmt in {"mllab", "metaphlan", "metaphlan_tsv", "profile_tsv", "matrix_tsv"}:
            ids, X, features = _read_feature_matrix_modality(
                modality, samples, metadata_ids
            )
            canonical_fmt = "matrix_tsv" if fmt == "matrix_tsv" else "mllab"
        elif fmt in {"table", "tabular", "csv", "tsv", "matrix"}:
            ids, X, features = _read_table_modality(modality, samples)
            canonical_fmt = "table"
        else:
            raise ValueError(
                f"Unsupported format {modality.format!r} for Modality {modality.name!r}."
            )
        unknown = [sid for sid in ids if sid not in meta_index.index]
        if unknown:
            raise ValueError(
                f"Modality {modality.name!r} contains sample IDs not present in Samples: {unknown[:5]!r}."
            )
        loaded[str(modality.name)] = (ids, X, features, canonical_fmt)
    primary = primary_items[0]
    primary_ids = loaded[str(primary.name)][0]
    if not primary_ids:
        raise ValueError(f"Primary Modality {primary.name!r} contains no samples.")
    missing_meta = [sid for sid in primary_ids if sid not in meta_index.index]
    if missing_meta:
        raise ValueError(
            f"Primary Modality {primary.name!r} contains sample IDs absent from Samples: {missing_meta[:5]!r}."
        )
    modality_matrices: dict[str, ModalityMatrix] = {}
    for modality in items:
        ids, X, features, fmt = loaded[str(modality.name)]
        index = {sid: i for i, sid in enumerate(ids)}
        missing = [sid for sid in primary_ids if sid not in index]
        if missing:
            preview = missing[:8]
            raise ValueError(
                f"Modality {modality.name!r} is missing {len(missing)} sample(s) required by primary Modality {primary.name!r}: {preview!r}. Prepare matched input tables before running integration."
            )
        order = np.asarray([index[sid] for sid in primary_ids], dtype=int)
        X_aligned = np.asarray(X[order], dtype=np.float32)
        primary_meta = meta_index.loc[primary_ids].reset_index(drop=True)
        y_lookup = dict(zip(meta[samples.sample_id_col].astype(str), np.asarray(y_all)))
        y = np.asarray(
            [y_lookup[sid] for sid in primary_ids],
            dtype=float if task == "regression" else int,
        )
        ds = _dataset_from_feature_matrix(
            X_aligned,
            list(features),
            y,
            list(primary_ids),
            primary_meta,
            list(labels),
            positive,
            ("all",),
            task,
            target,
        )
        modality_matrices[str(modality.name)] = ModalityMatrix(
            str(modality.name), X_aligned, list(features), fmt, ds
        )
    primary_meta = meta_index.loc[primary_ids].reset_index(drop=True)
    y_lookup = dict(zip(meta[samples.sample_id_col].astype(str), np.asarray(y_all)))
    y = np.asarray(
        [y_lookup[sid] for sid in primary_ids],
        dtype=float if task == "regression" else int,
    )
    return ModalityDataset(
        samples,
        list(primary_ids),
        y,
        primary_meta,
        list(labels),
        positive,
        task,
        target,
        str(primary.name),
        modality_matrices,
    )
