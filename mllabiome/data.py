from __future__ import annotations

import hashlib
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
    target_col: str | tuple[str, ...] = "label"
    task: str = "classification"
    target_tasks: Mapping[str, str] | None = None
    group_col: str | None = None
    stratify_col: str | tuple[str, ...] | None = None
    subject_id_col: str | None = None
    metadata_cols: tuple[str, ...] = ()
    feature_cols: str | tuple[str, ...] | None = None
    feature_prefixes: str | tuple[str, ...] | None = None
    allow_implicit_numeric_features: bool = False
    label_map: Mapping[Any, int] | None = None
    class_labels: tuple[str, ...] | None = None
    positive_class: int | str = 1
    target_class_labels: Mapping[str, tuple[str, ...]] | None = None
    target_positive_classes: Mapping[str, int | str] | None = None


@dataclass
class Dataset:
    X_by_level: dict[str, np.ndarray]
    feature_names_by_level: dict[str, list[str]]
    y: np.ndarray
    sample_ids: list[str]
    subject_ids: list[str]
    metadata: pd.DataFrame
    class_labels: list[str]
    positive_class: int | None
    task: str = "classification"
    target_name: str = "label"

    @property
    def classes(self) -> np.ndarray:
        return np.arange(len(self.class_labels), dtype=int)

    @property
    def positive_class_label(self) -> str | None:
        if self.positive_class is None:
            return None
        return self.class_labels[int(self.positive_class)]


def _hash_text(hasher: Any, value: Any) -> None:
    payload = str(value).encode("utf-8")
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _hash_text_sequence(hasher: Any, values: Sequence[Any]) -> None:
    hasher.update(len(values).to_bytes(8, "big"))
    for value in values:
        _hash_text(hasher, value)


def _hash_array(hasher: Any, values: np.ndarray, dtype: str) -> None:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    hasher.update(len(array.shape).to_bytes(4, "big"))
    for size in array.shape:
        hasher.update(int(size).to_bytes(8, "big"))
    hasher.update(array.tobytes(order="C"))


def dataset_fingerprint(dataset: Dataset) -> str:
    hasher = hashlib.sha256()
    _hash_text(hasher, dataset.task)
    _hash_text(hasher, dataset.target_name)
    _hash_text_sequence(hasher, dataset.sample_ids)
    _hash_text_sequence(hasher, dataset.subject_ids)
    _hash_text_sequence(hasher, dataset.feature_names_by_level.get("all", ()))
    _hash_text_sequence(hasher, dataset.class_labels)
    _hash_text(hasher, dataset.positive_class)
    if dataset.task == "classification":
        _hash_array(hasher, dataset.y, "<i8")
    else:
        _hash_array(hasher, dataset.y, "<f8")
    _hash_array(hasher, dataset.X_by_level["all"], "<f4")
    return hasher.hexdigest()


def _single_target_name(spec: Data) -> str:
    if not isinstance(spec.target_col, str):
        raise ValueError(
            "load_dataset requires a single target column; multi-target sweeps are expanded by evaluate()."
        )
    return spec.target_col


def _normalise_task(task: str) -> str:
    value = str(task).strip().casefold().replace("-", "_")
    aliases = {
        "binary": "classification",
        "multiclass": "classification",
        "continuous": "regression",
    }
    value = aliases.get(value, value)
    if value not in {"classification", "regression"}:
        raise ValueError(
            f"Unsupported single-target task {task!r}. Use 'classification' or 'regression'."
        )
    return value


def _encode_regression(values: Sequence[Any]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if series.isna().any():
        bad = np.flatnonzero(series.isna().to_numpy()).tolist()
        raise ValueError(
            f"Regression target contains missing or non-numeric values at {len(bad)} row(s); first positions: {bad[:5]!r}."
        )
    y = series.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Regression target contains NaN or infinite values.")
    if len(y) < 2:
        raise ValueError("Regression requires at least two samples.")
    return y


def load_dataset(spec: Data, levels_needed: Iterable[str] | None = None) -> Dataset:
    levels_needed = tuple(dict.fromkeys(levels_needed or TAXONOMIC_LEVELS))
    _single_target_name(spec)
    fmt = str(spec.format).strip().casefold().replace("-", "_")
    abundance_path = Path(spec.abundance_path)
    metadata_path = Path(spec.metadata_path) if spec.metadata_path is not None else None
    if fmt == "auto":
        fmt = (
            "mllab"
            if abundance_path.suffix.lower() in {".tsv", ".txt"} and metadata_path
            else "wide_csv"
        )
    if fmt in {"csv", "wide_csv"}:
        return _load_csv_dataset(spec, levels_needed)
    if fmt in {"mllab", "matrix_tsv", "metaphlan", "metaphlan_tsv", "profile_tsv"}:
        if metadata_path is None:
            raise ValueError(
                "Data.metadata_path is required for matrix-style TSV input."
            )
        return _load_matrix_tsv_dataset(spec, levels_needed)
    raise ValueError(
        f"Unsupported data format {fmt!r}. Supported formats are 'auto', 'wide_csv', and 'mllab'."
    )


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalise_text(value: Any) -> str:
    return str(value).strip()


def _normalise_lookup(value: Any) -> str:
    return _normalise_text(value).casefold()


def _resolve_positive_from_codes(
    positive_class: int | str,
    unique_codes: list[int],
    code_labels: dict[int, str],
    raw_key_to_code: dict[str, int],
) -> int:
    if isinstance(positive_class, (int, np.integer)):
        value = int(positive_class)
        if value in unique_codes:
            return value
        if 0 <= value < len(unique_codes):
            return unique_codes[value]
    token = _normalise_lookup(positive_class)
    if token in raw_key_to_code:
        return int(raw_key_to_code[token])
    for code, label in code_labels.items():
        if _normalise_lookup(label) == token:
            return int(code)
    try:
        numeric = int(_normalise_text(positive_class))
    except ValueError:
        numeric = None
    if numeric is not None and numeric in unique_codes:
        return numeric
    allowed = [code_labels[code] for code in unique_codes]
    raise ValueError(
        f"positive_class={positive_class!r} does not identify a binary class. Available classes: {allowed!r}."
    )


def _resolve_positive_from_labels(
    positive_class: int | str,
    labels: list[str],
    raw_values: list[Any] | None = None,
) -> int:
    if isinstance(positive_class, (int, np.integer)):
        value = int(positive_class)
        if raw_values is not None:
            for index, raw in enumerate(raw_values):
                if isinstance(raw, (int, np.integer, float, np.floating)):
                    if float(raw) == float(value):
                        return index
        if 0 <= value < len(labels):
            return value
    token = _normalise_lookup(positive_class)
    matches = [i for i, label in enumerate(labels) if _normalise_lookup(label) == token]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"positive_class={positive_class!r} does not identify a binary class. Available classes: {labels!r}."
    )


def _encode_y(
    values: Sequence[Any],
    label_map: Mapping[Any, int] | None = None,
    labels: tuple[str, ...] | None = None,
    positive_class: int | str = 1,
) -> tuple[np.ndarray, list[str], int | None]:
    raw = list(values)
    if not raw:
        raise ValueError("Target column is empty.")
    missing = [i for i, value in enumerate(raw) if _is_missing(value)]
    if missing:
        raise ValueError(
            f"Target column contains missing values at {len(missing)} row(s); first positions: {missing[:5]!r}."
        )

    if label_map is not None:
        normalised_map: dict[str, tuple[Any, int]] = {}
        for key, code in label_map.items():
            token = _normalise_lookup(key)
            if not token:
                raise ValueError("label_map contains an empty label key.")
            if token in normalised_map:
                raise ValueError(
                    f"label_map contains duplicate labels after case-insensitive normalization: {key!r}."
                )
            try:
                mapped_code = int(code)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"label_map value for {key!r} must be an integer class code."
                ) from exc
            normalised_map[token] = (key, mapped_code)

        unknown = sorted(
            {
                _normalise_text(value)
                for value in raw
                if _normalise_lookup(value) not in normalised_map
            }
        )
        if unknown:
            raise ValueError(
                f"Target column contains labels not present in label_map: {unknown!r}."
            )

        external_codes = np.asarray(
            [normalised_map[_normalise_lookup(value)][1] for value in raw], dtype=int
        )
        unique_codes = sorted(set(external_codes.tolist()))
        if len(unique_codes) < 2:
            raise ValueError("Classification requires at least two observed classes.")

        keys_by_code: dict[int, list[str]] = {code: [] for code in unique_codes}
        for original_key, code in normalised_map.values():
            if code in keys_by_code:
                keys_by_code[code].append(_normalise_text(original_key))

        if labels is not None:
            labels_out = [_normalise_text(value) for value in labels]
            if len(labels_out) != len(unique_codes):
                raise ValueError(
                    f"class_labels has {len(labels_out)} entries but {len(unique_codes)} classes are observed."
                )
            if len(set(labels_out)) != len(labels_out):
                raise ValueError("class_labels must be unique.")
            code_labels = dict(zip(unique_codes, labels_out))
        else:
            merged = [code for code, keys in keys_by_code.items() if len(keys) != 1]
            if merged:
                raise ValueError(
                    "class_labels is required when label_map maps multiple raw labels to one class code."
                )
            code_labels = {code: keys_by_code[code][0] for code in unique_codes}

        if len(unique_codes) == 2:
            raw_key_to_code = {
                token: code
                for token, (_, code) in normalised_map.items()
                if code in unique_codes
            }
            positive_code = _resolve_positive_from_codes(
                positive_class, unique_codes, code_labels, raw_key_to_code
            )
            ordered_codes = [code for code in unique_codes if code != positive_code] + [
                positive_code
            ]
            positive_index = 1
        else:
            ordered_codes = unique_codes
            positive_index = None

        internal = {code: index for index, code in enumerate(ordered_codes)}
        y = np.asarray([internal[int(code)] for code in external_codes], dtype=int)
        labels_final = [code_labels[code] for code in ordered_codes]
        return y, labels_final, positive_index

    if labels is not None:
        labels_out = [_normalise_text(value) for value in labels]
        if len(labels_out) < 2:
            raise ValueError("class_labels must contain at least two classes.")
        if len(set(labels_out)) != len(labels_out):
            raise ValueError("class_labels must be unique.")
        raw_text = [_normalise_text(value) for value in raw]
        unknown = sorted(set(raw_text) - set(labels_out))
        if unknown:
            raise ValueError(
                f"Target column contains labels not present in class_labels: {unknown!r}."
            )
        ordered_labels = list(labels_out)
        if len(ordered_labels) == 2:
            positive_index_original = _resolve_positive_from_labels(
                positive_class, ordered_labels
            )
            positive_label = ordered_labels[positive_index_original]
            ordered_labels = [
                label for label in ordered_labels if label != positive_label
            ] + [positive_label]
            positive_index = 1
        else:
            positive_index = None
        mapping = {label: index for index, label in enumerate(ordered_labels)}
        y = np.asarray([mapping[value] for value in raw_text], dtype=int)
        return y, ordered_labels, positive_index

    numeric = pd.to_numeric(pd.Series(raw), errors="coerce")
    if numeric.notna().all():
        raw_values = numeric.tolist()
        unique_values = sorted(pd.unique(numeric).tolist())
        labels_out = [_normalise_text(value) for value in unique_values]
        if len(unique_values) < 2:
            raise ValueError("Classification requires at least two observed classes.")
        if len(unique_values) == 2:
            positive_index_original = _resolve_positive_from_labels(
                positive_class, labels_out, unique_values
            )
            positive_value = unique_values[positive_index_original]
            ordered_values = [
                value for value in unique_values if value != positive_value
            ] + [positive_value]
            positive_index = 1
        else:
            ordered_values = unique_values
            positive_index = None
        value_to_index = {value: index for index, value in enumerate(ordered_values)}
        y = np.asarray([value_to_index[value] for value in raw_values], dtype=int)
        return y, [_normalise_text(value) for value in ordered_values], positive_index

    raw_text = [_normalise_text(value) for value in raw]
    unique_labels = sorted(set(raw_text))
    if len(unique_labels) < 2:
        raise ValueError("Classification requires at least two observed classes.")
    if len(unique_labels) == 2:
        positive_index_original = _resolve_positive_from_labels(
            positive_class, unique_labels
        )
        positive_label = unique_labels[positive_index_original]
        ordered_labels = [
            label for label in unique_labels if label != positive_label
        ] + [positive_label]
        positive_index = 1
    else:
        ordered_labels = unique_labels
        positive_index = None
    mapping = {label: index for index, label in enumerate(ordered_labels)}
    y = np.asarray([mapping[value] for value in raw_text], dtype=int)
    return y, ordered_labels, positive_index


def _reserved_wide_csv_columns(spec: Data, target_col: str) -> set[str]:
    reserved = {spec.sample_id_col, target_col, *(spec.metadata_cols or ())}
    if spec.subject_id_col:
        reserved.add(spec.subject_id_col)
    if spec.group_col:
        reserved.add(spec.group_col)
    if spec.stratify_col:
        if isinstance(spec.stratify_col, str):
            reserved.add(spec.stratify_col)
        else:
            reserved.update(str(c) for c in spec.stratify_col)
    return {str(value) for value in reserved}


def _wide_csv_feature_columns(
    spec: Data, df: pd.DataFrame, target_col: str
) -> list[str]:
    reserved = _reserved_wide_csv_columns(spec, target_col)
    if isinstance(spec.feature_cols, str):
        explicit = [spec.feature_cols]
    else:
        explicit = (
            None if spec.feature_cols is None else [str(c) for c in spec.feature_cols]
        )
    prefixes_raw = spec.feature_prefixes
    if isinstance(prefixes_raw, str):
        prefixes = (prefixes_raw,)
    else:
        prefixes = tuple(str(value) for value in (prefixes_raw or ()))
    if explicit is not None and prefixes:
        raise ValueError(
            "Use either Data.feature_cols or Data.feature_prefixes, not both."
        )
    if explicit is not None:
        if not explicit:
            raise ValueError("Data.feature_cols cannot be empty.")
        duplicates = sorted({c for c in explicit if explicit.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"Data.feature_cols contains duplicate columns: {duplicates[:5]!r}."
            )
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(
                f"Configured abundance columns are missing: {missing[:10]!r}."
            )
        forbidden = [c for c in explicit if c in reserved]
        if forbidden:
            raise ValueError(
                "Configured abundance columns overlap reserved metadata/target columns: "
                f"{forbidden[:10]!r}."
            )
        selected = explicit
    elif prefixes:
        if any(not prefix for prefix in prefixes):
            raise ValueError("Data.feature_prefixes cannot contain empty prefixes.")
        selected = [
            str(c)
            for c in df.columns
            if str(c) not in reserved
            and any(str(c).startswith(prefix) for prefix in prefixes)
        ]
        if not selected:
            raise ValueError(
                f"No abundance columns match Data.feature_prefixes={prefixes!r}."
            )
    elif bool(spec.allow_implicit_numeric_features):
        selected = [
            str(c)
            for c in df.columns
            if str(c) not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        candidates = [
            str(c)
            for c in df.columns
            if str(c) not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        preview = candidates[:10]
        raise ValueError(
            "Wide CSV abundance columns must be selected explicitly with Data.feature_cols or Data.feature_prefixes. "
            f"Implicit numeric-column inference is disabled to prevent covariate leakage. Numeric candidates: {preview!r}. "
            "Set allow_implicit_numeric_features=True only for a verified feature-only table."
        )
    if not selected:
        raise ValueError("No abundance columns were selected.")
    converted = df[selected].apply(pd.to_numeric, errors="coerce")
    invalid = converted.isna() & df[selected].notna()
    if bool(invalid.any().any()):
        bad_columns = invalid.columns[invalid.any(axis=0)].astype(str).tolist()
        raise ValueError(
            f"Selected abundance columns contain non-numeric values: {bad_columns[:10]!r}."
        )
    return selected


def _subject_ids(
    spec: Data, frame: pd.DataFrame, sample_ids: Sequence[str]
) -> list[str]:
    if not spec.subject_id_col:
        return [str(value) for value in sample_ids]
    if spec.subject_id_col not in frame.columns:
        raise ValueError(
            f"subject_id_col={spec.subject_id_col!r} was not found in metadata."
        )
    values = frame[spec.subject_id_col]
    if values.isna().any():
        raise ValueError(
            f"subject_id_col={spec.subject_id_col!r} contains missing values."
        )
    return values.astype(str).tolist()


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
    target_col = _single_target_name(spec)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found.")
    if spec.sample_id_col in df.columns:
        sample_series = df[spec.sample_id_col].astype(str)
        if sample_series.duplicated().any():
            duplicates = sample_series[sample_series.duplicated()].tolist()
            raise ValueError(
                f"Wide CSV contains duplicate sample IDs: {duplicates[:5]!r}."
            )
        sample_ids = sample_series.tolist()
    else:
        sample_ids = [str(i) for i in range(len(df))]
    subject_ids = _subject_ids(spec, df, sample_ids)
    task = _normalise_task(spec.task)
    if task == "regression":
        y = _encode_regression(df[target_col].tolist())
        class_labels = []
        positive_class = None
    else:
        y, class_labels, positive_class = _encode_y(
            df[target_col].tolist(),
            spec.label_map,
            spec.class_labels,
            spec.positive_class,
        )
    feature_columns = _wide_csv_feature_columns(spec, df, target_col)
    feature_names = [str(c) for c in feature_columns]
    X_all = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    return _dataset_from_feature_matrix(
        X_all,
        feature_names,
        y,
        sample_ids,
        subject_ids,
        df,
        class_labels,
        positive_class,
        levels_needed,
        task,
        target_col,
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
        "Cannot determine whether the abundance TSV is headered or headerless. Headerless matrices require exactly one abundance column per metadata row."
    )


def _load_matrix_tsv_dataset(spec: Data, levels_needed: tuple[str, ...]) -> Dataset:
    abundance_path = Path(spec.abundance_path)
    metadata_path = Path(spec.metadata_path)
    meta = pd.read_csv(metadata_path, sep=None, engine="python", dtype=str)
    if spec.sample_id_col not in meta.columns:
        raise ValueError(f"Metadata missing sample ID column {spec.sample_id_col!r}.")
    target_col = _single_target_name(spec)
    if target_col not in meta.columns:
        raise ValueError(f"Metadata missing target column {target_col!r}.")
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
    task = _normalise_task(spec.task)
    if task == "regression":
        y = _encode_regression(meta[target_col].tolist())
        class_labels = []
        positive_class = None
    else:
        y, class_labels, positive_class = _encode_y(
            meta[target_col].tolist(),
            spec.label_map,
            spec.class_labels,
            spec.positive_class,
        )
    X = bio[selected_sample_ids].T.to_numpy(dtype=np.float32)
    feature_names = bio.index.tolist()
    subject_ids = _subject_ids(spec, meta, selected_sample_ids)
    return _dataset_from_feature_matrix(
        X,
        feature_names,
        y,
        selected_sample_ids,
        subject_ids,
        meta,
        class_labels,
        positive_class,
        levels_needed,
        task,
        target_col,
    )


def _dataset_from_feature_matrix(
    X_all: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    sample_ids: list[str],
    subject_ids: list[str],
    meta: pd.DataFrame,
    class_labels: list[str],
    positive_class: int | None,
    levels_needed: tuple[str, ...],
    task: str = "classification",
    target_name: str = "label",
) -> Dataset:
    X_all = np.asarray(X_all, dtype=np.float32)
    if X_all.ndim != 2:
        raise ValueError("Abundance matrix must be two-dimensional.")
    if X_all.shape[0] != len(sample_ids):
        raise ValueError(
            "Abundance matrix row count does not match the number of sample IDs."
        )
    if len(subject_ids) != len(sample_ids):
        raise ValueError("Subject ID length does not match the number of sample IDs.")
    if X_all.shape[1] != len(feature_names):
        raise ValueError(
            "Abundance matrix column count does not match the number of feature names."
        )
    if len(y) != len(sample_ids):
        raise ValueError("Target length does not match the number of sample IDs.")
    if task == "classification":
        if not np.all(np.isin(y, np.arange(len(class_labels), dtype=int))):
            raise ValueError(
                "Encoded target contains values outside the canonical class range."
            )
        if len(class_labels) == 2 and positive_class != 1:
            raise ValueError(
                "Binary positive class must be canonical internal class 1."
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
        y=np.asarray(y, dtype=int if task == "classification" else float),
        sample_ids=sample_ids,
        subject_ids=[str(value) for value in subject_ids],
        metadata=meta,
        class_labels=class_labels,
        positive_class=positive_class,
        task=task,
        target_name=target_name,
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
