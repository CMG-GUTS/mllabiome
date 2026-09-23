from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mllabiome.configs_sweep import (
    Evaluation,
    _groups_from_metadata,
    _inner_splits,
    _outer_splits,
    _regression_inner_splits,
    _regression_outer_splits,
)


def test_grouped_classification_outer_and_inner_splits_are_disjoint():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"])
    plan = Evaluation(
        protocol="repeated_nested_cv",
        outer_folds=3,
        inner_folds=2,
        repeats=2,
        random_state=42,
    )
    outer = _outer_splits(plan, y, groups)
    assert len(outer) == 6
    for split in outer:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        inner = _inner_splits(plan, y, train_idx, groups, split)
        assert inner
        for inner_train, inner_validation in inner:
            outer_groups = groups[train_idx]
            assert set(outer_groups[inner_train]).isdisjoint(
                set(outer_groups[inner_validation])
            )


def test_grouped_regression_outer_and_inner_splits_are_disjoint():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"])
    plan = Evaluation(
        protocol="repeated_nested_cv",
        outer_folds=3,
        inner_folds=2,
        repeats=2,
        random_state=42,
    )
    outer = _regression_outer_splits(plan, len(groups), groups)
    assert len(outer) == 6
    for split in outer:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        inner = _regression_inner_splits(plan, train_idx, groups, split)
        assert inner
        for inner_train, inner_validation in inner:
            outer_groups = groups[train_idx]
            assert set(outer_groups[inner_train]).isdisjoint(
                set(outer_groups[inner_validation])
            )


def test_group_column_rejects_missing_or_empty_values():
    with pytest.raises(ValueError):
        _groups_from_metadata(pd.DataFrame({"subject": ["a", None]}), "subject")
    with pytest.raises(ValueError):
        _groups_from_metadata(pd.DataFrame({"subject": ["a", " "]}), "subject")
