from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    Dataset,
    _dataset_from_feature_matrix,
    _encode_regression,
    dataset_fingerprint,
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
    subject_id_col: str | None = None


@dataclass(frozen=True)
class Modality:
    name: str
    path: Path | str
    format: str = "table"
    primary: bool = False
    sample_id_col: str | None = None
    metadata_cols: tuple[str, ...] = ()
    feature_cols: str | tuple[str, ...] | None = None
    feature_prefixes: str | tuple[str, ...] | None = None
    allow_implicit_numeric_features: bool = False


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
    subject_ids: list[str]
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


def _subject_ids(
    samples: Samples, frame: pd.DataFrame, sample_ids: Sequence[str]
) -> list[str]:
    if not samples.subject_id_col:
        return [str(value) for value in sample_ids]
    if samples.subject_id_col not in frame.columns:
        raise ValueError(
            f"subject_id_col={samples.subject_id_col!r} was not found in the Samples table."
        )
    values = frame[samples.subject_id_col]
    if values.isna().any():
        raise ValueError(
            f"subject_id_col={samples.subject_id_col!r} contains missing values."
        )
    return values.astype(str).tolist()


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
    reserved = {id_col, *(str(c) for c in modality.metadata_cols)}
    missing_metadata = sorted(c for c in reserved if c != id_col and c not in frame.columns)
    if missing_metadata:
        raise ValueError(
            f"Modality {modality.name!r} is missing declared metadata columns: {missing_metadata!r}."
        )
    explicit = modality.feature_cols
    prefixes = modality.feature_prefixes
    if explicit is not None and prefixes is not None:
        raise ValueError(
            f"Modality {modality.name!r} must use feature_cols or feature_prefixes, not both."
        )
    if explicit is not None:
        requested = (explicit,) if isinstance(explicit, str) else tuple(explicit)
        feature_cols = [str(c) for c in requested]
        missing = [c for c in feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(
                f"Modality {modality.name!r} is missing declared feature columns: {missing[:8]!r}."
            )
    elif prefixes is not None:
        requested_prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
        normalized = tuple(str(prefix) for prefix in requested_prefixes if str(prefix))
        if not normalized:
            raise ValueError(
                f"Modality {modality.name!r} feature_prefixes must not be empty."
            )
        feature_cols = [
            str(c)
            for c in frame.columns
            if str(c) not in reserved and any(str(c).startswith(prefix) for prefix in normalized)
        ]
    elif modality.allow_implicit_numeric_features:
        feature_cols = [str(c) for c in frame.columns if str(c) not in reserved]
    else:
        raise ValueError(
            f"Tabular Modality {modality.name!r} requires explicit feature_cols or feature_prefixes. Set allow_implicit_numeric_features=True only for a verified feature-only table."
        )
    overlap = sorted(set(feature_cols) & reserved)
    if overlap:
        raise ValueError(
            f"Modality {modality.name!r} feature selection overlaps reserved metadata columns: {overlap!r}."
        )
    if not feature_cols:
        raise ValueError(f"Modality {modality.name!r} contains no selected feature columns.")
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
        subject_ids = _subject_ids(samples, primary_meta, primary_ids)
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
            subject_ids,
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
    subject_ids = _subject_ids(samples, primary_meta, primary_ids)
    y_lookup = dict(zip(meta[samples.sample_id_col].astype(str), np.asarray(y_all)))
    y = np.asarray(
        [y_lookup[sid] for sid in primary_ids],
        dtype=float if task == "regression" else int,
    )
    return ModalityDataset(
        samples,
        list(primary_ids),
        subject_ids,
        y,
        primary_meta,
        list(labels),
        positive,
        task,
        target,
        str(primary.name),
        modality_matrices,
    )

def modality_fingerprints(dataset: ModalityDataset) -> dict[str, str]:
    return {
        str(name): dataset_fingerprint(matrix.dataset)
        for name, matrix in sorted(dataset.modalities.items(), key=lambda item: str(item[0]))
    }


def modality_dataset_fingerprint(dataset: ModalityDataset) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"modality-dataset-v1")
    for value in (dataset.task, dataset.target_name, dataset.primary_modality):
        payload = str(value).encode("utf-8")
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    for sample_id, subject_id in zip(dataset.sample_ids, dataset.subject_ids):
        for value in (sample_id, subject_id):
            payload = str(value).encode("utf-8")
            hasher.update(len(payload).to_bytes(8, "big"))
            hasher.update(payload)
    for name, fingerprint in modality_fingerprints(dataset).items():
        for value in (name, fingerprint):
            payload = str(value).encode("utf-8")
            hasher.update(len(payload).to_bytes(8, "big"))
            hasher.update(payload)
    return hasher.hexdigest()

def source_file_sha256(path: Path | str) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def modality_source_fingerprints(
    samples: Samples, modalities: Sequence[Modality]
) -> dict[str, Any]:
    return {
        "samples": source_file_sha256(samples.path),
        "modalities": {
            str(modality.name): source_file_sha256(modality.path)
            for modality in modalities
        },
    }

