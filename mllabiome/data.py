from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import TAXONOMIC_LEVELS


@dataclass
class Data:
    abundance_path: Path | str
    metadata_path: Path | str | None = None
    format: str = "auto"
    sample_id_col: str = "sample_id"
    target_col: str = "label"
    group_col: str | None = None
    stratify_col: str | tuple[str, ...] | None = None
    metadata_cols: tuple[str, ...] = ()
    label_map: Mapping[Any, int] | None = None
    class_labels: tuple[str, ...] | None = None
    positive_class: int | str = 1


@dataclass
class Dataset:
    X_by_level: dict[str, np.ndarray]
    feature_names_by_level: dict[str, list[str]]
    y: np.ndarray
    sample_ids: list[str]
    metadata: pd.DataFrame
    class_labels: list[str]

    @property
    def classes(self) -> np.ndarray:
        return np.arange(len(self.class_labels), dtype=int)


def load_dataset(spec: Data, levels_needed: Iterable[str] | None = None) -> Dataset:
    levels_needed = tuple(dict.fromkeys(levels_needed or TAXONOMIC_LEVELS))
    fmt = spec.format
    abundance_path = Path(spec.abundance_path)
    metadata_path = Path(spec.metadata_path) if spec.metadata_path is not None else None
    if fmt == "auto":
        fmt = (
            "metaphlan_tsv"
            if abundance_path.suffix.lower() in {".tsv", ".txt"} and metadata_path
            else "wide_csv"
        )
    if fmt in {"csv", "wide_csv"}:
        return _load_csv_dataset(spec, levels_needed)
    if fmt in {"matrix_tsv", "metaphlan_tsv", "profile_tsv"}:
        if metadata_path is None:
            raise ValueError(
                "Data.metadata_path is required for matrix-style TSV input."
            )
        return _load_matrix_tsv_dataset(spec, levels_needed)
    raise ValueError(
        f"Unsupported data format {fmt!r}. Supported formats are 'auto', 'wide_csv', and 'metaphlan_tsv'."
    )


def _encode_y(
    values: Sequence[Any],
    label_map: Mapping[Any, int] | None = None,
    labels: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    s = pd.Series(values)
    if label_map is not None:
        mapping = {str(k).strip().lower(): int(v) for k, v in label_map.items()}
        y = np.array([mapping.get(str(v).strip().lower(), -1) for v in s], dtype=int)
        valid_codes = sorted(set(mapping.values()))
        if labels is not None:
            labels_out = [str(x) for x in labels]
        else:
            inv = {int(v): str(k) for k, v in label_map.items()}
            labels_out = [inv.get(code, str(code)) for code in valid_codes]
        return y, labels_out
    if labels is None:
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().all():
            vals = numeric.astype(int).to_numpy()
            uniq = sorted(pd.unique(vals).tolist())
            mapping = {v: i for i, v in enumerate(uniq)}
            y = np.array([mapping[v] for v in vals], dtype=int)
            return y, [str(v) for v in uniq]
        labels_out = sorted(str(x) for x in s.dropna().unique())
    else:
        labels_out = [str(x) for x in labels]
    mapping = {v: i for i, v in enumerate(labels_out)}
    y = np.array([mapping.get(str(v), -1) for v in s], dtype=int)
    return y, labels_out


def _load_csv_dataset(spec: Data, levels_needed: tuple[str, ...]) -> Dataset:
    abundance_path = Path(spec.abundance_path)
    df = pd.read_csv(abundance_path)
    if spec.metadata_path is not None:
        meta = pd.read_csv(Path(spec.metadata_path), sep=None, engine="python")
        if (
            spec.sample_id_col not in df.columns
            or spec.sample_id_col not in meta.columns
        ):
            raise ValueError(
                f"sample_id_col={spec.sample_id_col!r} must exist in both CSV files."
            )
        df = df.merge(meta, on=spec.sample_id_col, how="inner", suffixes=("", "__meta"))
    if spec.target_col not in df.columns:
        raise ValueError(f"Target column {spec.target_col!r} not found.")
    if spec.sample_id_col in df.columns:
        sample_ids = df[spec.sample_id_col].astype(str).tolist()
    else:
        sample_ids = [str(i) for i in range(len(df))]
    y, class_labels = _encode_y(
        df[spec.target_col].tolist(), spec.label_map, spec.class_labels
    )
    keep = y >= 0
    df = df.loc[keep].reset_index(drop=True)
    y = y[keep]
    sample_ids = [sid for sid, ok in zip(sample_ids, keep) if ok]
    reserved = {spec.sample_id_col, spec.target_col, *(spec.metadata_cols or ())}
    if spec.group_col:
        reserved.add(spec.group_col)
    if spec.stratify_col:
        if isinstance(spec.stratify_col, str):
            reserved.add(spec.stratify_col)
        else:
            reserved.update(str(c) for c in spec.stratify_col)
    numeric_cols = [
        c
        for c in df.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        raise ValueError(
            "No numeric abundance columns found after excluding metadata columns."
        )
    feature_names = [str(c) for c in numeric_cols]
    X_all = df[numeric_cols].to_numpy(dtype=np.float32)
    return _dataset_from_feature_matrix(
        X_all, feature_names, y, sample_ids, df, class_labels, levels_needed
    )


_MATRIX_HEADER_TOKENS = {
    "clade_name",
    "feature",
    "feature_id",
    "features",
    "taxon",
    "taxa",
    "taxonomy",
    "otu",
    "otu_id",
    "asv",
    "asv_id",
}


def _first_nonempty_tsv_row(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                return [field.strip() for field in line.rstrip("\r\n").split("\t")]
    raise ValueError(f"Abundance matrix {path} is empty.")


def _read_feature_by_sample_tsv(
    abundance_path: Path,
    metadata_sample_ids: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    first = _first_nonempty_tsv_row(abundance_path)
    if len(first) < 2:
        raise ValueError(
            "Matrix-style abundance TSV must contain one feature column and at least one sample column."
        )
    first_field = first[0].lstrip("#").strip()
    first_values = [str(x).strip() for x in first[1:]]
    sample_ids = [str(x).strip() for x in metadata_sample_ids]
    sample_set = set(sample_ids)
    overlap = sum(value in sample_set for value in first_values)
    first_rank = _taxonomic_rank(first_field)
    explicit_header = first_field.lower() in _MATRIX_HEADER_TOKENS
    strong_sample_header = overlap >= min(3, max(1, len(sample_ids)))
    exact_positional_width = len(first_values) == len(sample_ids)
    if explicit_header or strong_sample_header:
        bio = pd.read_csv(abundance_path, sep="\t", index_col=0, low_memory=False)
        bio.index = bio.index.astype(str).str.strip()
        bio.columns = bio.columns.astype(str).str.strip()
        common = [sid for sid in sample_ids if sid in set(bio.columns)]
        if not common:
            raise ValueError(
                "No sample IDs overlap between metadata and the headered abundance matrix."
            )
        return bio, common
    if exact_positional_width and first_rank is not None:
        bio = pd.read_csv(
            abundance_path, sep="\t", header=None, index_col=0, low_memory=False
        )
        bio.index = bio.index.astype(str).str.strip()
        bio.columns = sample_ids
        return bio, sample_ids
    if overlap > 0:
        bio = pd.read_csv(abundance_path, sep="\t", index_col=0, low_memory=False)
        bio.index = bio.index.astype(str).str.strip()
        bio.columns = bio.columns.astype(str).str.strip()
        common = [sid for sid in sample_ids if sid in set(bio.columns)]
        if not common:
            raise ValueError(
                "No sample IDs overlap between metadata and the headered abundance matrix."
            )
        return bio, common
    if exact_positional_width:
        bio = pd.read_csv(
            abundance_path, sep="\t", header=None, index_col=0, low_memory=False
        )
        bio.index = bio.index.astype(str).str.strip()
        bio.columns = sample_ids
        return bio, sample_ids
    raise ValueError(
        "Cannot determine whether the abundance TSV is headered or headerless. "
        "Headerless matrices require exactly one abundance column per metadata row."
    )


def _load_matrix_tsv_dataset(spec: Data, levels_needed: tuple[str, ...]) -> Dataset:
    abundance_path = Path(spec.abundance_path)
    metadata_path = Path(spec.metadata_path)
    meta = pd.read_csv(metadata_path, sep=None, engine="python", dtype=str)
    if spec.sample_id_col not in meta.columns:
        raise ValueError(f"Metadata missing sample ID column {spec.sample_id_col!r}.")
    if spec.target_col not in meta.columns:
        raise ValueError(f"Metadata missing target column {spec.target_col!r}.")
    meta[spec.sample_id_col] = meta[spec.sample_id_col].astype(str).str.strip()
    if meta[spec.sample_id_col].duplicated().any():
        duplicates = meta.loc[
            meta[spec.sample_id_col].duplicated(), spec.sample_id_col
        ].tolist()
        raise ValueError(f"Metadata contains duplicate sample IDs: {duplicates[:5]!r}.")
    metadata_sample_ids = meta[spec.sample_id_col].tolist()
    bio, selected_sample_ids = _read_feature_by_sample_tsv(
        abundance_path, metadata_sample_ids
    )
    meta = meta.set_index(spec.sample_id_col).loc[selected_sample_ids].reset_index()
    y, class_labels = _encode_y(
        meta[spec.target_col].tolist(), spec.label_map, spec.class_labels
    )
    keep = y >= 0
    selected_sample_ids = [sid for sid, ok in zip(selected_sample_ids, keep) if ok]
    meta = meta.loc[keep].reset_index(drop=True)
    y = y[keep]
    X = bio[selected_sample_ids].T.to_numpy(dtype=np.float32)
    feature_names = bio.index.tolist()
    return _dataset_from_feature_matrix(
        X,
        feature_names,
        y,
        selected_sample_ids,
        meta,
        class_labels,
        levels_needed,
    )


def _dataset_from_feature_matrix(
    X_all: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    sample_ids: list[str],
    meta: pd.DataFrame,
    class_labels: list[str],
    levels_needed: tuple[str, ...],
) -> Dataset:
    X_all = np.asarray(X_all, dtype=np.float32)
    if X_all.ndim != 2:
        raise ValueError("Abundance matrix must be two-dimensional.")
    if X_all.shape[0] != len(sample_ids):
        raise ValueError(
            "Abundance matrix row count does not match the number of sample IDs."
        )
    if X_all.shape[1] != len(feature_names):
        raise ValueError(
            "Abundance matrix column count does not match the number of feature names."
        )
    level_to_idx: dict[str, list[int]] = {lv: [] for lv in TAXONOMIC_LEVELS}
    for j, name in enumerate(feature_names):
        level = _taxonomic_rank(name)
        if level is not None:
            level_to_idx[level].append(j)
    X_by_level: dict[str, np.ndarray] = {"all": X_all.astype(np.float32, copy=False)}
    names_by_level: dict[str, list[str]] = {"all": list(feature_names)}
    for level in TAXONOMIC_LEVELS:
        idx = level_to_idx[level]
        if idx:
            X_by_level[level] = X_all[:, idx].astype(np.float32, copy=False)
            names_by_level[level] = [feature_names[i] for i in idx]
    return Dataset(
        X_by_level=X_by_level,
        feature_names_by_level=names_by_level,
        y=y,
        sample_ids=sample_ids,
        metadata=meta,
        class_labels=class_labels,
    )


def _taxonomic_rank(name: str) -> str | None:
    text = str(name)
    order = [
        ("strain", ["___t__", "|t__", ";t__", " t__"]),
        ("species", ["___s__", "|s__", ";s__", " s__"]),
        ("genus", ["___g__", "|g__", ";g__", " g__"]),
        ("family", ["___f__", "|f__", ";f__", " f__"]),
        ("order", ["___o__", "|o__", ";o__", " o__"]),
        ("class", ["___c__", "|c__", ";c__", " c__"]),
        ("phylum", ["___p__", "|p__", ";p__", " p__"]),
        ("domain", ["d__", "k__", "|k__", ";k__"]),
    ]
    for level, markers in order:
        if any(
            marker in text or text.startswith(marker.strip("|; ")) for marker in markers
        ):
            return level
    return None
