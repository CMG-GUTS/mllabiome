from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    Dataset,
    _dataset_from_feature_matrix,
    _encode_regression,
    _encode_y,
    _normalise_task,
    _read_feature_by_sample_tsv,
    dataset_fingerprint,
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
    inclusion_report: dict[str, Any]

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
    missing_metadata = sorted(
        c for c in reserved if c != id_col and c not in frame.columns
    )
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
        requested_prefixes = (
            (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
        )
        normalized = tuple(str(prefix) for prefix in requested_prefixes if str(prefix))
        if not normalized:
            raise ValueError(
                f"Modality {modality.name!r} feature_prefixes must not be empty."
            )
        feature_cols = [
            str(c)
            for c in frame.columns
            if str(c) not in reserved
            and any(str(c).startswith(prefix) for prefix in normalized)
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
        raise ValueError(
            f"Modality {modality.name!r} contains no selected feature columns."
        )
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


def _count_distribution(values: Sequence[Any]) -> dict[str, int]:
    series = pd.Series(list(values), dtype="object").dropna().astype(str)
    if series.empty:
        return {}
    counts = series.value_counts(sort=False)
    return {
        str(key): int(value)
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def _target_distribution(
    y: np.ndarray, labels: Sequence[str], task: str
) -> dict[str, Any]:
    arr = np.asarray(y)
    if task == "classification":
        counts: dict[str, int] = {}
        for idx in range(len(labels)):
            counts[str(labels[idx])] = int(np.sum(arr.astype(int) == idx))
        return {"kind": "classification", "counts": counts}
    values = (
        pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy(dtype=float)
    )
    if values.size == 0:
        return {"kind": "regression", "n": 0}
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "kind": "regression",
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _categorical_summary(values: pd.Series, limit: int = 8) -> str:
    series = values.dropna().astype(str)
    if series.empty:
        return "n=0"
    counts = series.value_counts()
    total = int(counts.sum())
    parts = [
        f"{key!s}: {int(value)} ({100.0 * float(value) / total:.1f}%)"
        for key, value in counts.iloc[:limit].items()
    ]
    remaining = int(counts.iloc[limit:].sum())
    if remaining:
        parts.append(f"other: {remaining} ({100.0 * remaining / total:.1f}%)")
    return "; ".join(parts)


def _numeric_summary(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return "n=0"
    q1, median, q3 = np.quantile(numeric, [0.25, 0.5, 0.75])
    return f"{float(median):.4g} [{float(q1):.4g}, {float(q3):.4g}] (n={int(numeric.size)})"


def _characteristic_balance(
    name: str, primary: pd.Series, complete: pd.Series, excluded: pd.Series
) -> dict[str, Any]:
    primary_nonmissing = primary.dropna()
    numeric_primary = pd.to_numeric(primary_nonmissing, errors="coerce")
    is_numeric = bool(len(primary_nonmissing)) and bool(numeric_primary.notna().all())
    if is_numeric:
        c = pd.to_numeric(complete, errors="coerce").dropna().to_numpy(dtype=float)
        e = pd.to_numeric(excluded, errors="coerce").dropna().to_numpy(dtype=float)
        balance = None
        if c.size and e.size:
            c_var = float(np.var(c, ddof=1)) if c.size > 1 else 0.0
            e_var = float(np.var(e, ddof=1)) if e.size > 1 else 0.0
            pooled = float(np.sqrt((c_var + e_var) / 2.0))
            if pooled > 0:
                balance = float((np.mean(c) - np.mean(e)) / pooled)
            elif float(np.mean(c)) == float(np.mean(e)):
                balance = 0.0
        return {
            "variable": str(name),
            "kind": "continuous",
            "primary": _numeric_summary(primary),
            "complete": _numeric_summary(complete),
            "excluded": _numeric_summary(excluded),
            "balance_metric": "standardized_mean_difference_complete_minus_excluded",
            "balance_value": balance,
        }
    categories = sorted(
        set(primary.dropna().astype(str).tolist())
        | set(complete.dropna().astype(str).tolist())
        | set(excluded.dropna().astype(str).tolist())
    )
    complete_counts = complete.dropna().astype(str).value_counts()
    excluded_counts = excluded.dropna().astype(str).value_counts()
    complete_n = int(complete_counts.sum())
    excluded_n = int(excluded_counts.sum())
    max_diff = None
    if complete_n and excluded_n:
        diffs = [
            abs(
                float(complete_counts.get(category, 0)) / complete_n
                - float(excluded_counts.get(category, 0)) / excluded_n
            )
            for category in categories
        ]
        max_diff = float(max(diffs)) if diffs else 0.0
    return {
        "variable": str(name),
        "kind": "categorical",
        "primary": _categorical_summary(primary),
        "complete": _categorical_summary(complete),
        "excluded": _categorical_summary(excluded),
        "balance_metric": "max_absolute_proportion_difference_complete_vs_excluded",
        "balance_value": max_diff,
    }


def _multimodal_inclusion_report(
    samples: Samples,
    items: Sequence[Modality],
    loaded: Mapping[str, tuple[list[str], np.ndarray, list[str], str]],
    primary_ids: Sequence[str],
    complete_ids: Sequence[str],
    meta_index: pd.DataFrame,
    y_lookup: Mapping[str, Any],
    labels: Sequence[str],
    task: str,
) -> dict[str, Any]:
    primary_ids = [str(value) for value in primary_ids]
    complete_ids = [str(value) for value in complete_ids]
    complete_set = set(complete_ids)
    excluded_ids = [sid for sid in primary_ids if sid not in complete_set]
    primary_meta = meta_index.loc[primary_ids].reset_index(drop=True)
    complete_meta = meta_index.loc[complete_ids].reset_index(drop=True)
    excluded_meta = (
        meta_index.loc[excluded_ids].reset_index(drop=True)
        if excluded_ids
        else primary_meta.iloc[0:0].copy()
    )
    primary_subjects = _subject_ids(samples, primary_meta, primary_ids)
    complete_subjects = _subject_ids(samples, complete_meta, complete_ids)
    excluded_subject_samples = (
        _subject_ids(samples, excluded_meta, excluded_ids) if excluded_ids else []
    )
    primary_subject_set = set(primary_subjects)
    complete_subject_set = set(complete_subjects)
    excluded_subject_sample_set = set(excluded_subject_samples)
    fully_excluded_subjects = primary_subject_set - complete_subject_set
    partially_retained_subjects = complete_subject_set & excluded_subject_sample_set
    availability: dict[str, Any] = {}
    availability_sets: dict[str, set[str]] = {}
    for modality in items:
        ids = set(str(value) for value in loaded[str(modality.name)][0])
        availability_sets[str(modality.name)] = ids
        available = sum(sid in ids for sid in primary_ids)
        missing = len(primary_ids) - available
        availability[str(modality.name)] = {
            "available_primary_samples": int(available),
            "missing_primary_samples": int(missing),
            "availability_fraction": float(available / len(primary_ids))
            if primary_ids
            else 0.0,
        }
    patterns: dict[tuple[str, ...], int] = {}
    for sid in primary_ids:
        missing = tuple(
            str(modality.name)
            for modality in items
            if sid not in availability_sets[str(modality.name)]
        )
        patterns[missing] = patterns.get(missing, 0) + 1
    pattern_rows = [
        {
            "missing_modalities": list(pattern),
            "n_samples": int(count),
            "fraction_primary": float(count / len(primary_ids)) if primary_ids else 0.0,
        }
        for pattern, count in sorted(
            patterns.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    primary_y = np.asarray([y_lookup[sid] for sid in primary_ids])
    complete_y = np.asarray([y_lookup[sid] for sid in complete_ids])
    excluded_y = np.asarray([y_lookup[sid] for sid in excluded_ids])
    characteristics: list[dict[str, Any]] = []
    requested_characteristics: list[str] = []
    for value in samples.metadata_cols:
        key = str(value)
        if key not in requested_characteristics:
            requested_characteristics.append(key)
    if samples.group_col and str(samples.group_col) not in requested_characteristics:
        requested_characteristics.append(str(samples.group_col))
    stratify = samples.stratify_col
    stratify_cols = (stratify,) if isinstance(stratify, str) else tuple(stratify or ())
    for value in stratify_cols:
        key = str(value)
        if key not in requested_characteristics:
            requested_characteristics.append(key)
    for column in requested_characteristics:
        if column in primary_meta.columns:
            characteristics.append(
                _characteristic_balance(
                    column,
                    primary_meta[column],
                    complete_meta[column],
                    excluded_meta[column],
                )
            )
    group_distribution = None
    if samples.group_col and str(samples.group_col) in primary_meta.columns:
        column = str(samples.group_col)
        group_distribution = {
            "column": column,
            "primary": _count_distribution(primary_meta[column]),
            "complete": _count_distribution(complete_meta[column]),
            "excluded": _count_distribution(excluded_meta[column]),
        }
    return {
        "policy": "complete_case",
        "estimand": (
            "samples with complete observations across all requested modalities "
            "within the primary-modality cohort"
        ),
        "reference_population": "samples present in the primary modality and Samples metadata",
        "n_primary_samples": len(primary_ids),
        "n_complete_samples": len(complete_ids),
        "n_excluded_samples": len(excluded_ids),
        "sample_retention_fraction": (
            float(len(complete_ids) / len(primary_ids)) if primary_ids else 0.0
        ),
        "n_primary_subjects": len(primary_subject_set),
        "n_complete_subjects": len(complete_subject_set),
        "n_fully_excluded_subjects": len(fully_excluded_subjects),
        "n_subjects_with_incomplete_samples": len(excluded_subject_sample_set),
        "n_partially_retained_subjects": len(partially_retained_subjects),
        "subject_retention_fraction": (
            float(len(complete_subject_set) / len(primary_subject_set))
            if primary_subject_set
            else 0.0
        ),
        "modality_availability": availability,
        "missingness_patterns": pattern_rows,
        "target_distribution": {
            "primary": _target_distribution(primary_y, labels, task),
            "complete": _target_distribution(complete_y, labels, task),
            "excluded": _target_distribution(excluded_y, labels, task),
        },
        "group_distribution": group_distribution,
        "characteristic_balance": characteristics,
    }


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
    availability_sets = {
        str(modality.name): set(loaded[str(modality.name)][0]) for modality in items
    }
    complete_ids = [
        sid
        for sid in primary_ids
        if all(sid in availability_sets[str(modality.name)] for modality in items)
    ]
    if not complete_ids:
        raise ValueError(
            "No complete-case samples remain after intersecting the requested modalities."
        )
    y_lookup = dict(zip(meta[samples.sample_id_col].astype(str), np.asarray(y_all)))
    inclusion_report = _multimodal_inclusion_report(
        samples,
        items,
        loaded,
        primary_ids,
        complete_ids,
        meta_index,
        y_lookup,
        labels,
        task,
    )
    complete_meta = meta_index.loc[complete_ids].reset_index(drop=True)
    subject_ids = _subject_ids(samples, complete_meta, complete_ids)
    y = np.asarray(
        [y_lookup[sid] for sid in complete_ids],
        dtype=float if task == "regression" else int,
    )
    modality_matrices: dict[str, ModalityMatrix] = {}
    for modality in items:
        ids, X, features, fmt = loaded[str(modality.name)]
        index = {sid: i for i, sid in enumerate(ids)}
        order = np.asarray([index[sid] for sid in complete_ids], dtype=int)
        X_aligned = np.asarray(X[order], dtype=np.float32)
        ds = _dataset_from_feature_matrix(
            X_aligned,
            list(features),
            y,
            list(complete_ids),
            subject_ids,
            complete_meta,
            list(labels),
            positive,
            ("all",),
            task,
            target,
        )
        modality_matrices[str(modality.name)] = ModalityMatrix(
            str(modality.name), X_aligned, list(features), fmt, ds
        )
    return ModalityDataset(
        samples,
        list(complete_ids),
        subject_ids,
        y,
        complete_meta,
        list(labels),
        positive,
        task,
        target,
        str(primary.name),
        modality_matrices,
        inclusion_report,
    )


def modality_fingerprints(dataset: ModalityDataset) -> dict[str, str]:
    return {
        str(name): dataset_fingerprint(matrix.dataset)
        for name, matrix in sorted(
            dataset.modalities.items(), key=lambda item: str(item[0])
        )
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
