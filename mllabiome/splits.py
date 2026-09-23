from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .storage import read_table, table_exists, write_table


_SCHEMA_VERSION = 2
_REQUIRED_COLUMNS = {
    "schema_version",
    "data_signature",
    "plan_signature",
    "task",
    "protocol",
    "stage",
    "split_key",
    "inner_key",
    "repeat",
    "outer_fold",
    "inner_fold",
    "outer_group",
    "role",
    "sample_id",
    "sample_index",
}


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _digest(payload: Any) -> str:
    data = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _normalise_optional(values: np.ndarray | Sequence[Any] | None, n: int) -> list[Any]:
    if values is None:
        return [None] * n
    array = np.asarray(values, dtype=object).reshape(-1)
    if len(array) != n:
        raise ValueError("Split metadata length does not match the number of samples.")
    return [_json_value(value) for value in array.tolist()]


def _plan_signature(
    protocol: str,
    outer_folds: int,
    inner_folds: int,
    repeats: int,
    random_state: int,
    group_col: str | None,
    stratify_col: str | Sequence[str] | None,
    schema_version: int,
) -> str:
    return _digest(
        {
            "schema_version": int(schema_version),
            "protocol": str(protocol).lower(),
            "outer_folds": int(outer_folds),
            "inner_folds": int(inner_folds),
            "repeats": int(repeats),
            "random_state": int(random_state),
            "group_col": group_col,
            "stratify_col": _json_value(stratify_col),
        }
    )


def split_signatures(
    sample_ids: Sequence[str],
    y: np.ndarray | Sequence[Any],
    groups: np.ndarray | Sequence[Any] | None,
    strata: np.ndarray | Sequence[Any] | None,
    task: str,
    target_name: str,
    protocol: str,
    outer_folds: int,
    inner_folds: int,
    repeats: int,
    random_state: int,
    group_col: str | None,
    stratify_col: str | Sequence[str] | None,
) -> tuple[str, str]:
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Cross-validation split persistence requires unique sample IDs."
        )
    target = np.asarray(y, dtype=object).reshape(-1)
    if len(target) != len(ids):
        raise ValueError("Target length does not match the number of sample IDs.")
    group_values = _normalise_optional(groups, len(ids))
    stratum_values = _normalise_optional(strata, len(ids))
    records = sorted(
        (
            ids[index],
            _json_value(target[index]),
            group_values[index],
            stratum_values[index],
        )
        for index in range(len(ids))
    )
    data_signature = _digest(
        {
            "task": str(task),
            "target_name": str(target_name),
            "samples": records,
        }
    )
    plan_signature = _plan_signature(
        protocol,
        outer_folds,
        inner_folds,
        repeats,
        random_state,
        group_col,
        stratify_col,
        _SCHEMA_VERSION,
    )
    return data_signature, plan_signature


def _manifest_frame(
    sample_ids: Sequence[str],
    task: str,
    protocol: str,
    data_signature: str,
    plan_signature: str,
    outer_splits: Sequence[dict[str, Any]],
    inner_splits: dict[str, Sequence[tuple[np.ndarray, np.ndarray]]],
) -> pd.DataFrame:
    ids = [str(value) for value in sample_ids]
    rows: list[dict[str, Any]] = []
    for split in outer_splits:
        split_key = str(split["split_key"])
        repeat = int(split.get("repeat", 0))
        outer_fold = int(split.get("outer_fold", 0))
        outer_group = str(split.get("outer_group", ""))
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        for role, indices in (("train", train_idx), ("test", test_idx)):
            for sample_index in indices:
                index = int(sample_index)
                rows.append(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "data_signature": data_signature,
                        "plan_signature": plan_signature,
                        "task": str(task),
                        "protocol": str(protocol).lower(),
                        "stage": "outer",
                        "split_key": split_key,
                        "inner_key": "",
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "inner_fold": -1,
                        "outer_group": outer_group,
                        "role": role,
                        "sample_id": ids[index],
                        "sample_index": index,
                    }
                )
        for inner_fold, pair in enumerate(inner_splits.get(split_key, ())):
            inner_train, inner_validation = pair
            inner_key = f"{split_key}__i{inner_fold}"
            for role, local_indices in (
                ("train", np.asarray(inner_train, dtype=int)),
                ("validation", np.asarray(inner_validation, dtype=int)),
            ):
                global_indices = train_idx[local_indices]
                for sample_index in global_indices:
                    index = int(sample_index)
                    rows.append(
                        {
                            "schema_version": _SCHEMA_VERSION,
                            "data_signature": data_signature,
                            "plan_signature": plan_signature,
                            "task": str(task),
                            "protocol": str(protocol).lower(),
                            "stage": "inner",
                            "split_key": split_key,
                            "inner_key": inner_key,
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "inner_fold": int(inner_fold),
                            "outer_group": outer_group,
                            "role": role,
                            "sample_id": ids[index],
                            "sample_index": index,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Cross-validation split generation produced no assignments.")
    role_order = pd.Categorical(
        frame["role"], categories=["train", "validation", "test"], ordered=True
    )
    frame = frame.assign(_role_order=role_order)
    frame = frame.sort_values(
        ["repeat", "outer_fold", "stage", "inner_fold", "_role_order", "sample_index"],
        kind="stable",
    ).drop(columns="_role_order")
    return frame.reset_index(drop=True)


def _read_manifest(
    path: Path,
    sample_ids: Sequence[str],
    data_signature: str,
    plan_signature: str,
    legacy_plan_signature: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    frame = read_table(path)
    missing = _REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Stored cross-validation split manifest is missing columns: {sorted(missing)!r}."
        )
    if frame.empty:
        raise ValueError("Stored cross-validation split manifest is empty.")
    schema_values = set(
        pd.to_numeric(frame["schema_version"], errors="coerce").dropna()
    )
    if schema_values == {_SCHEMA_VERSION}:
        expected_plan_signature = plan_signature
    elif schema_values == {1} and legacy_plan_signature is not None:
        expected_plan_signature = legacy_plan_signature
    else:
        raise ValueError(
            "Stored cross-validation split manifest has an unsupported schema."
        )
    data_values = set(frame["data_signature"].astype(str))
    plan_values = set(frame["plan_signature"].astype(str))
    if data_values != {data_signature} or plan_values != {expected_plan_signature}:
        raise ValueError(
            "Stored cross-validation splits do not match the current dataset or evaluation plan. Use a new experiment directory for a different split definition."
        )
    ids = [str(value) for value in sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Cross-validation split persistence requires unique sample IDs."
        )
    current_index = {sample_id: index for index, sample_id in enumerate(ids)}
    stored_ids = set(frame["sample_id"].astype(str))
    if stored_ids != set(ids):
        raise ValueError(
            "Stored cross-validation splits contain a different set of sample IDs from the current dataset."
        )
    outer_frame = frame[frame["stage"].astype(str).eq("outer")].copy()
    if outer_frame.empty:
        raise ValueError(
            "Stored cross-validation split manifest contains no outer splits."
        )
    split_meta = (
        outer_frame[["split_key", "repeat", "outer_fold", "outer_group"]]
        .drop_duplicates()
        .sort_values(["repeat", "outer_fold", "split_key"], kind="stable")
    )
    outer_splits: list[dict[str, Any]] = []
    inner_map: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for meta in split_meta.to_dict(orient="records"):
        split_key = str(meta["split_key"])
        split_rows = outer_frame[outer_frame["split_key"].astype(str).eq(split_key)]
        train_ids = split_rows.loc[
            split_rows["role"].astype(str).eq("train"), "sample_id"
        ].astype(str)
        test_ids = split_rows.loc[
            split_rows["role"].astype(str).eq("test"), "sample_id"
        ].astype(str)
        train_idx = np.asarray([current_index[value] for value in train_ids], dtype=int)
        test_idx = np.asarray([current_index[value] for value in test_ids], dtype=int)
        if len(set(train_idx)) != len(train_idx) or len(set(test_idx)) != len(test_idx):
            raise ValueError(
                f"Stored outer split {split_key!r} contains duplicate samples."
            )
        if set(train_idx).intersection(set(test_idx)):
            raise ValueError(
                f"Stored outer split {split_key!r} has overlapping train and test samples."
            )
        if set(train_idx).union(test_idx) != set(range(len(ids))):
            raise ValueError(
                f"Stored outer split {split_key!r} does not partition all samples."
            )
        split: dict[str, Any] = {
            "split_key": split_key,
            "repeat": int(meta["repeat"]),
            "outer_fold": int(meta["outer_fold"]),
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
        outer_group = str(meta.get("outer_group", ""))
        if outer_group:
            split["outer_group"] = outer_group
        outer_splits.append(split)
        global_to_local = {int(value): index for index, value in enumerate(train_idx)}
        inner_rows = frame[
            frame["stage"].astype(str).eq("inner")
            & frame["split_key"].astype(str).eq(split_key)
        ].copy()
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        if not inner_rows.empty:
            inner_meta = (
                inner_rows[["inner_key", "inner_fold"]]
                .drop_duplicates()
                .sort_values(["inner_fold", "inner_key"], kind="stable")
            )
            for inner in inner_meta.to_dict(orient="records"):
                inner_key = str(inner["inner_key"])
                fold_rows = inner_rows[
                    inner_rows["inner_key"].astype(str).eq(inner_key)
                ]
                train_global = [
                    current_index[value]
                    for value in fold_rows.loc[
                        fold_rows["role"].astype(str).eq("train"), "sample_id"
                    ].astype(str)
                ]
                validation_global = [
                    current_index[value]
                    for value in fold_rows.loc[
                        fold_rows["role"].astype(str).eq("validation"), "sample_id"
                    ].astype(str)
                ]
                if (
                    not set(train_global)
                    .union(validation_global)
                    .issubset(global_to_local)
                ):
                    raise ValueError(
                        f"Stored inner split {inner_key!r} contains samples outside its outer training partition."
                    )
                inner_train = np.asarray(
                    [global_to_local[value] for value in train_global], dtype=int
                )
                inner_validation = np.asarray(
                    [global_to_local[value] for value in validation_global], dtype=int
                )
                if len(set(inner_train)) != len(inner_train) or len(
                    set(inner_validation)
                ) != len(inner_validation):
                    raise ValueError(
                        f"Stored inner split {inner_key!r} contains duplicate samples."
                    )
                if set(inner_train).intersection(set(inner_validation)):
                    raise ValueError(
                        f"Stored inner split {inner_key!r} has overlapping partitions."
                    )
                if set(inner_train).union(inner_validation) != set(
                    range(len(train_idx))
                ):
                    raise ValueError(
                        f"Stored inner split {inner_key!r} does not partition the outer training samples."
                    )
                pairs.append((inner_train, inner_validation))
        inner_map[split_key] = pairs
    return outer_splits, inner_map


def _prediction_sample_sets(path: Path, key_column: str) -> dict[str, set[str]]:
    if not table_exists(path):
        return {}
    frame = read_table(path)
    if (
        frame.empty
        or key_column not in frame.columns
        or "sample_id" not in frame.columns
    ):
        return {}
    return {
        str(key): set(group["sample_id"].astype(str))
        for key, group in frame.groupby(key_column, sort=False)
    }


def _checkpoint_sample_sets(
    root: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    db_path = root / "configs.db"
    if not db_path.exists():
        return {}, {}
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evaluation_checkpoints'"
        ).fetchone()
        if not exists:
            return {}, {}
        rows = conn.execute("SELECT payload FROM evaluation_checkpoints").fetchall()
    finally:
        conn.close()
    outer: dict[str, set[str]] = {}
    inner: dict[str, set[str]] = {}
    for (blob,) in rows:
        try:
            payload = json.loads(zlib.decompress(blob).decode("utf-8"))
        except Exception:
            continue
        for row in payload.get("outer_predictions", []):
            key = str(row.get("split_key", ""))
            sample_id = str(row.get("sample_id", ""))
            if key and sample_id:
                outer.setdefault(key, set()).add(sample_id)
        for row in payload.get("inner_predictions", []):
            key = str(row.get("split_key", ""))
            sample_id = str(row.get("sample_id", ""))
            if key and sample_id:
                inner.setdefault(key, set()).add(sample_id)
    return outer, inner


def _merge_sample_sets(
    first: dict[str, set[str]], second: dict[str, set[str]]
) -> dict[str, set[str]]:
    out = {key: set(values) for key, values in first.items()}
    for key, values in second.items():
        out.setdefault(key, set()).update(values)
    return out


def _validate_legacy_assignments(
    root: Path,
    sample_ids: Sequence[str],
    outer_splits: Sequence[dict[str, Any]],
    inner_splits: dict[str, Sequence[tuple[np.ndarray, np.ndarray]]],
) -> None:
    outer_saved = _prediction_sample_sets(
        root / "predictions" / "outer_predictions.parquet", "split_key"
    )
    inner_saved = _prediction_sample_sets(
        root / "inner_predictions" / "inner_predictions.parquet", "split_key"
    )
    checkpoint_outer, checkpoint_inner = _checkpoint_sample_sets(root)
    outer_saved = _merge_sample_sets(outer_saved, checkpoint_outer)
    inner_saved = _merge_sample_sets(inner_saved, checkpoint_inner)
    if not outer_saved and not inner_saved:
        return
    ids = [str(value) for value in sample_ids]
    expected_outer: dict[str, set[str]] = {}
    expected_inner: dict[str, set[str]] = {}
    for split in outer_splits:
        split_key = str(split["split_key"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        expected_outer[split_key] = {ids[int(index)] for index in test_idx}
        for inner_fold, pair in enumerate(inner_splits.get(split_key, ())):
            validation = np.asarray(pair[1], dtype=int)
            expected_inner[f"{split_key}__i{inner_fold}"] = {
                ids[int(train_idx[int(index)])] for index in validation
            }
    for key, observed in outer_saved.items():
        expected = expected_outer.get(key)
        if expected is not None and observed != expected:
            raise ValueError(
                f"Existing outer predictions for {key!r} do not match the regenerated split assignment."
            )
    for key, observed in inner_saved.items():
        expected = expected_inner.get(key)
        if expected is not None and observed != expected:
            raise ValueError(
                f"Existing inner predictions for {key!r} do not match the regenerated split assignment."
            )


def resolve_cv_splits(
    root: Path,
    sample_ids: Sequence[str],
    y: np.ndarray | Sequence[Any],
    groups: np.ndarray | Sequence[Any] | None,
    strata: np.ndarray | Sequence[Any] | None,
    task: str,
    target_name: str,
    protocol: str,
    outer_folds: int,
    inner_folds: int,
    repeats: int,
    random_state: int,
    group_col: str | None,
    stratify_col: str | Sequence[str] | None,
    create: Callable[
        [],
        tuple[
            list[dict[str, Any]],
            dict[str, list[tuple[np.ndarray, np.ndarray]]],
        ],
    ],
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[np.ndarray, np.ndarray]]], Path]:
    root = Path(root)
    path = root / "tables" / "cv_splits.parquet"
    data_signature, plan_signature = split_signatures(
        sample_ids,
        y,
        groups,
        strata,
        task,
        target_name,
        protocol,
        outer_folds,
        inner_folds,
        repeats,
        random_state,
        group_col,
        stratify_col,
    )
    if table_exists(path):
        frame = read_table(path)
        schema_values = (
            set(pd.to_numeric(frame["schema_version"], errors="coerce").dropna())
            if "schema_version" in frame.columns
            else set()
        )
        legacy_allowed = group_col is None or str(protocol).lower() in {
            "lodo",
            "leave_one_dataset_out",
        }
        if schema_values == {1} and legacy_allowed:
            legacy_plan_signature = _plan_signature(
                protocol,
                outer_folds,
                inner_folds,
                repeats,
                random_state,
                group_col,
                stratify_col,
                1,
            )
            outer_splits, inner_splits = _read_manifest(
                path,
                sample_ids,
                data_signature,
                plan_signature,
                legacy_plan_signature,
            )
            upgraded = _manifest_frame(
                sample_ids,
                task,
                protocol,
                data_signature,
                plan_signature,
                outer_splits,
                inner_splits,
            )
            write_table(path, upgraded)
            outer_splits, inner_splits = _read_manifest(
                path, sample_ids, data_signature, plan_signature
            )
            return outer_splits, inner_splits, path
        outer_splits, inner_splits = _read_manifest(
            path, sample_ids, data_signature, plan_signature
        )
        return outer_splits, inner_splits, path
    outer_splits, inner_splits = create()
    _validate_legacy_assignments(root, sample_ids, outer_splits, inner_splits)
    frame = _manifest_frame(
        sample_ids,
        task,
        protocol,
        data_signature,
        plan_signature,
        outer_splits,
        inner_splits,
    )
    write_table(path, frame)
    outer_splits, inner_splits = _read_manifest(
        path, sample_ids, data_signature, plan_signature
    )
    return outer_splits, inner_splits, path
