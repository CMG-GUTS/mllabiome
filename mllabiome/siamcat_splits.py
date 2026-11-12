from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _valid(y: np.ndarray, allocation: np.ndarray, n_folds: int, min_train: int) -> bool:
    for fold in range(n_folds):
        test = allocation == fold
        if not np.any(test):
            return False
        for klass in (0, 1):
            if np.count_nonzero((y == klass) & test) < 1:
                return False
            if np.count_nonzero((y == klass) & ~test) < min_train:
                return False
    return True


def _score(y: np.ndarray, allocation: np.ndarray, n_folds: int) -> float:
    targets = np.array([np.count_nonzero(y == klass) / n_folds for klass in (0, 1)])
    total = 0.0
    for fold in range(n_folds):
        count = np.array(
            [np.count_nonzero((allocation == fold) & (y == klass)) for klass in (0, 1)]
        )
        total += float(np.sum(((count - targets) / np.maximum(targets, 1.0)) ** 2))
    return total


def _group_assignment(
    y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int, min_train: int
) -> np.ndarray | None:
    codes, uniques = _factor_groups(groups)
    n_groups = len(uniques)
    if n_groups < n_folds:
        return None
    best = None
    best_score = float("inf")
    if n_folds == 2 and n_groups <= 14:
        for count in range(1, n_groups):
            for left in combinations(range(1, n_groups), count - 1):
                chosen = np.zeros(n_groups, dtype=int)
                chosen[[0, *left]] = 1
                allocation = chosen[codes]
                if not _valid(y, allocation, n_folds, min_train):
                    continue
                value = _score(y, allocation, n_folds)
                if value < best_score:
                    best, best_score = allocation.copy(), value
        return best
    for i in range(96):
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=int((seed + 7919 * i) % (2**32 - 1)),
            )
            allocation = np.full(len(y), -1, dtype=int)
            for fold, (_, test_idx) in enumerate(
                splitter.split(np.zeros((len(y), 1)), y, groups)
            ):
                allocation[test_idx] = fold
        except ValueError:
            break
        if not _valid(y, allocation, n_folds, min_train):
            continue
        value = _score(y, allocation, n_folds)
        if value < best_score:
            best, best_score = allocation.copy(), value
            if value < 0.0001:
                break
    return best


def _factor_groups(groups: np.ndarray) -> tuple[np.ndarray, list[Any]]:
    values = np.asarray(groups, dtype=object)
    mapping: dict[Any, int] = {}
    codes = np.empty(len(values), dtype=int)
    unique = []
    for idx, value in enumerate(values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(unique)
            unique.append(key)
        codes[idx] = mapping[key]
    return codes, unique


def make_siamcat_splits(
    y: Any,
    groups: Any = None,
    *,
    n_folds: int = 2,
    repeats: int = 1,
    random_state: int = 42,
    min_train_class: int = 3,
) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    if labels.ndim != 1 or len(labels) == 0 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("SIAMCAT internal CV requires a nonempty binary label vector.")
    if int(n_folds) < 2 or int(repeats) < 1:
        raise ValueError(
            "SIAMCAT internal CV requires at least two folds and one repetition."
        )
    if min(np.bincount(labels, minlength=2)) < 2 * min_train_class:
        counts = np.bincount(labels, minlength=2)
        raise ValueError(
            f"SIAMCAT cannot construct safe {n_folds}-fold internal CV: training class counts are {counts.tolist()}, requiring at least {2 * min_train_class} of each class for two-fold resampling."
        )
    group_values = None
    if groups is not None:
        group_values = np.asarray(groups, dtype=object)
        if group_values.ndim != 1 or len(group_values) != len(labels):
            raise ValueError("SIAMCAT grouping vector must match the training labels.")
        if bool(pd.isna(group_values).any()):
            raise ValueError("SIAMCAT internal grouping values cannot be missing.")
        if len(_factor_groups(group_values)[1]) < int(n_folds):
            raise ValueError(
                f"SIAMCAT needs at least {n_folds} independent groups for internal CV."
            )
    result = np.zeros((int(repeats), len(labels)), dtype=int)
    for repeat in range(int(repeats)):
        seed = int((int(random_state) + 1299721 * repeat) % (2**32 - 1))
        if group_values is not None:
            allocation = _group_assignment(
                labels, group_values, int(n_folds), seed, min_train_class
            )
            if allocation is None:
                counts = np.bincount(labels, minlength=2)
                raise ValueError(
                    f"SIAMCAT cannot construct class-balanced, group-disjoint internal CV with {n_folds} folds and training class counts {counts.tolist()}; choose a valid subject-level grouping or supply more minority-class groups."
                )
        else:
            splitter = StratifiedKFold(
                n_splits=int(n_folds), shuffle=True, random_state=seed
            )
            allocation = np.full(len(labels), -1, dtype=int)
            for fold, (_, test_idx) in enumerate(
                splitter.split(np.zeros((len(labels), 1)), labels)
            ):
                allocation[test_idx] = fold
            if not _valid(labels, allocation, int(n_folds), min_train_class):
                counts = np.bincount(labels, minlength=2)
                raise ValueError(
                    f"SIAMCAT cannot construct class-balanced sample CV with {n_folds} folds and training class counts {counts.tolist()}."
                )
        result[repeat] = allocation + 1
    return result
