from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from .splits import resolve_cv_splits
from .sweep_types import Evaluation


def _regression_outer_splits(
    plan: Evaluation, n_samples: int, groups: np.ndarray | None
) -> list[dict[str, Any]]:
    protocol = plan.protocol.lower()
    if protocol in {"lodo", "leave_one_dataset_out"}:
        if groups is None:
            raise ValueError("LODO requires DATA.group_col.")
        out = []
        for i, group in enumerate(pd.unique(groups)):
            out.append(
                {
                    "split_key": f"lodo_{i}__{group}",
                    "repeat": 0,
                    "outer_fold": i,
                    "outer_group": str(group),
                    "train_idx": np.where(groups != group)[0],
                    "test_idx": np.where(groups == group)[0],
                }
            )
        return out
    repeats = plan.repeats if protocol == "repeated_nested_cv" else 1
    out = []
    if groups is None:
        n_splits = min(int(plan.outer_folds), int(n_samples))
    else:
        n_splits = min(int(plan.outer_folds), int(pd.Series(groups).nunique()))
    if n_splits < 2:
        raise ValueError(
            "Regression cross-validation requires at least two samples or groups."
        )
    for repeat_no in range(repeats):
        seed = plan.random_state + repeat_no
        if groups is None:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(np.arange(n_samples))
        else:
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(np.arange(n_samples), groups=groups)
        for fold_no, (train_idx, test_idx) in enumerate(iterator):
            out.append(
                {
                    "split_key": f"r{repeat_no}_o{fold_no}",
                    "repeat": repeat_no,
                    "outer_fold": fold_no,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
    return out


def _regression_inner_splits(
    plan: Evaluation,
    outer_train_idx: np.ndarray,
    groups: np.ndarray | None,
    split: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if (
        plan.protocol.lower() in {"lodo", "leave_one_dataset_out"}
        and groups is not None
    ):
        local_groups = groups[outer_train_idx]
        result = []
        for group in pd.unique(local_groups):
            val = np.where(local_groups == group)[0]
            train = np.where(local_groups != group)[0]
            if len(train) and len(val):
                result.append((train, val))
        if result:
            return result
    local_groups = None if groups is None else groups[outer_train_idx]
    if local_groups is None:
        n_splits = min(int(plan.inner_folds), len(outer_train_idx))
    else:
        n_splits = min(int(plan.inner_folds), int(pd.Series(local_groups).nunique()))
    if n_splits < 2:
        return []
    seed = (
        plan.random_state
        + int(split.get("repeat", 0)) * 1009
        + int(split.get("outer_fold", 0))
    )
    if local_groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.arange(len(outer_train_idx)))
    else:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.arange(len(outer_train_idx)), groups=local_groups)
    return [(train, val) for train, val in iterator]


def _subject_safe_groups(
    plan: Evaluation,
    dataset: Any,
    groups: np.ndarray | None,
    group_col: str | None,
) -> tuple[np.ndarray | None, str | None]:
    subject_ids = np.asarray([str(x) for x in dataset.subject_ids], dtype=object)
    if len(subject_ids) != len(dataset.y):
        raise ValueError("subject_ids must align one-to-one with model rows.")
    repeated = len(set(subject_ids.tolist())) < len(subject_ids)
    protocol = str(plan.protocol).strip().lower()
    if groups is not None:
        group_values = np.asarray(groups, dtype=object)
        if len(group_values) != len(subject_ids):
            raise ValueError(
                "Configured CV groups must align one-to-one with model rows."
            )
        mapping = (
            pd.DataFrame({"subject": subject_ids, "group": group_values})
            .groupby("subject", sort=False)["group"]
            .nunique(dropna=False)
        )
        if bool((mapping > 1).any()):
            bad = mapping[mapping > 1].index.astype(str).tolist()[:5]
            raise ValueError(
                f"Repeated subjects map to multiple CV groups, which can leak a subject across train/test partitions: {bad!r}. Use a grouping column that is constant within subject."
            )
        return group_values, group_col
    if protocol in {"lodo", "leave_one_dataset_out"}:
        return None, group_col
    if repeated:
        return subject_ids, "__subject_id__"
    return None, group_col


def _resolved_evaluation_splits(
    root: Path,
    plan: Evaluation,
    dataset: Any,
    groups: np.ndarray | None,
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
    group_col: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    task = str(dataset.task).lower()
    groups, group_col = _subject_safe_groups(plan, dataset, groups, group_col)

    def create():
        if task == "regression":
            outer = _regression_outer_splits(plan, len(dataset.y), groups)
            inner = {
                str(split["split_key"]): _regression_inner_splits(
                    plan, np.asarray(split["train_idx"], dtype=int), groups, split
                )
                for split in outer
            }
            return outer, inner
        outer = _outer_splits(plan, dataset.y, groups, strata, stratify_col)
        inner = {
            str(split["split_key"]): _inner_splits(
                plan,
                dataset.y,
                np.asarray(split["train_idx"], dtype=int),
                groups,
                split,
                strata,
                stratify_col,
            )
            for split in outer
        }
        return outer, inner

    outer, inner, _ = resolve_cv_splits(
        root=Path(root),
        sample_ids=tuple(dataset.sample_ids),
        y=dataset.y,
        groups=groups,
        strata=strata,
        task=task,
        target_name=str(dataset.target_name),
        protocol=str(plan.protocol),
        outer_folds=int(plan.outer_folds),
        inner_folds=int(plan.inner_folds),
        repeats=int(plan.repeats),
        random_state=int(plan.random_state),
        group_col=group_col,
        stratify_col=stratify_col,
        create=create,
    )
    subjects = np.asarray([str(x) for x in dataset.subject_ids], dtype=object)
    for split in outer:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        overlap = set(subjects[train_idx].tolist()) & set(subjects[test_idx].tolist())
        if overlap:
            raise ValueError(
                f"CV split {split['split_key']!r} leaks subject(s) across outer train/test partitions: {sorted(overlap)[:5]!r}."
            )
        for inner_no, (inner_train, inner_val) in enumerate(
            inner.get(str(split["split_key"]), [])
        ):
            inner_train_idx = train_idx[np.asarray(inner_train, dtype=int)]
            inner_val_idx = train_idx[np.asarray(inner_val, dtype=int)]
            inner_overlap = set(subjects[inner_train_idx].tolist()) & set(
                subjects[inner_val_idx].tolist()
            )
            if inner_overlap:
                raise ValueError(
                    f"CV split {split['split_key']!r} inner fold {inner_no} leaks subject(s) across train/validation partitions: {sorted(inner_overlap)[:5]!r}."
                )
    return outer, inner


def _groups_from_metadata(
    meta: pd.DataFrame, group_col: str | None
) -> np.ndarray | None:
    if not group_col:
        return None
    if group_col not in meta.columns:
        raise ValueError(f"Group column {group_col!r} not found in metadata.")
    values = meta[group_col]
    if values.isna().any():
        raise ValueError(f"Group column {group_col!r} contains missing values.")
    groups = values.astype(str).str.strip()
    if groups.eq("").any():
        raise ValueError(f"Group column {group_col!r} contains empty values.")
    return groups.to_numpy()


def _normalise_column_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None or value == "":
        return tuple()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value if str(v))


def _strata_from_metadata(
    meta: pd.DataFrame, y: np.ndarray, stratify_col: str | Sequence[str] | None
) -> np.ndarray:

    cols = _normalise_column_names(stratify_col)
    if not cols:
        return np.asarray(y, dtype=str)
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise ValueError(f"Stratification column(s) not found in metadata: {missing}")
    base = pd.Series(np.asarray(y, dtype=str), index=meta.index).astype(str)
    parts = [base]
    for c in cols:
        vals = meta[c].astype(str).fillna("NA").reset_index(drop=True)
        parts.append(vals)
    joined = parts[0].reset_index(drop=True)
    for vals in parts[1:]:
        joined = joined.str.cat(vals.astype(str), sep="__strata__")
    return joined.to_numpy(dtype=str)


def _safe_n_splits(strata: np.ndarray, requested: int) -> int:
    counts = pd.Series(strata).value_counts()
    if counts.empty:
        return 0
    return int(max(0, min(requested, counts.min(), len(strata))))


def _safe_group_n_splits(strata: np.ndarray, groups: np.ndarray, requested: int) -> int:
    frame = pd.DataFrame({"stratum": strata, "group": groups}).drop_duplicates()
    if frame.empty:
        return 0
    per_stratum = frame.groupby("stratum", dropna=False)["group"].nunique()
    return int(
        max(
            0,
            min(
                requested,
                int(frame["group"].nunique()),
                int(per_stratum.min()),
            ),
        )
    )


def _stratification_error_context(
    plan: Evaluation, stratify_col: str | Sequence[str] | None
) -> str:
    cols = _normalise_column_names(stratify_col)
    if not cols:
        return "target labels"
    return "target labels plus " + ", ".join(cols)


def _outer_splits(
    plan: Evaluation,
    y: np.ndarray,
    groups: np.ndarray | None,
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    protocol = plan.protocol.lower()
    out: list[dict[str, Any]] = []
    if protocol in {"lodo", "leave_one_dataset_out"}:
        if groups is None:
            raise ValueError("LODO requires DATA.group_col.")
        for i, g in enumerate(pd.unique(groups)):
            test_idx = np.where(groups == g)[0]
            train_idx = np.where(groups != g)[0]
            out.append(
                {
                    "split_key": f"lodo_{i}__{g}",
                    "repeat": 0,
                    "outer_fold": i,
                    "outer_group": str(g),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
        return out
    n_repeats = plan.repeats if protocol == "repeated_nested_cv" else 1
    split_strata = np.asarray(strata if strata is not None else y, dtype=str)
    for r in range(n_repeats):
        if groups is None:
            n_splits = _safe_n_splits(split_strata, plan.outer_folds)
        else:
            n_splits = _safe_group_n_splits(split_strata, groups, plan.outer_folds)
        if n_splits < 2:
            context = _stratification_error_context(plan, stratify_col)
            raise ValueError(
                f"Not enough samples or groups for outer cross-validation using {context}. "
                "Reduce outer_folds, remove/merge sparse strata, or omit DATA.stratify_col."
            )
        seed = plan.random_state + r
        if groups is None:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            iterator = splitter.split(np.zeros(len(y)), split_strata)
        else:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            iterator = splitter.split(np.zeros(len(y)), split_strata, groups)
        for o, (tr, te) in enumerate(iterator):
            out.append(
                {
                    "split_key": f"r{r}_o{o}",
                    "repeat": r,
                    "outer_fold": o,
                    "train_idx": tr,
                    "test_idx": te,
                }
            )
    return out


def _inner_splits(
    plan: Evaluation,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    groups: np.ndarray | None,
    outer_split: dict[str, Any],
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if (
        plan.protocol.lower() in {"lodo", "leave_one_dataset_out"}
        and groups is not None
    ):
        g_train = groups[outer_train_idx]
        unique = list(pd.unique(g_train))
        if len(unique) >= 2:
            out = []
            for g in unique:
                va = np.where(g_train == g)[0]
                tr = np.where(g_train != g)[0]
                if len(np.unique(y[outer_train_idx[tr]])) >= 2:
                    out.append((tr, va))
            if out:
                return out
    y_train = y[outer_train_idx]
    split_strata = np.asarray(
        strata[outer_train_idx] if strata is not None else y_train, dtype=str
    )
    local_groups = None if groups is None else groups[outer_train_idx]
    if local_groups is None:
        n_splits = _safe_n_splits(split_strata, plan.inner_folds)
    else:
        n_splits = _safe_group_n_splits(split_strata, local_groups, plan.inner_folds)
    if n_splits < 2:
        return []
    seed = (
        plan.random_state
        + int(outer_split.get("repeat", 0)) * 1009
        + int(outer_split.get("outer_fold", 0))
    )
    if local_groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(len(y_train)), split_strata)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(np.zeros(len(y_train)), split_strata, local_groups)
    return [(tr, va) for tr, va in iterator]
