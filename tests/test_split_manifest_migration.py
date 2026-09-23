from __future__ import annotations

import numpy as np
import pytest

import mllabiome.splits as splits


def _legacy_frame(sample_ids, y, groups, group_col):
    outer = [
        {
            "split_key": "r0_o0",
            "repeat": 0,
            "outer_fold": 0,
            "train_idx": np.array([0, 1]),
            "test_idx": np.array([2, 3]),
        },
        {
            "split_key": "r0_o1",
            "repeat": 0,
            "outer_fold": 1,
            "train_idx": np.array([2, 3]),
            "test_idx": np.array([0, 1]),
        },
    ]
    inner = {
        "r0_o0": [(np.array([0]), np.array([1]))],
        "r0_o1": [(np.array([0]), np.array([1]))],
    }
    data_signature, _ = splits.split_signatures(
        sample_ids,
        y,
        groups,
        None,
        "classification",
        "group",
        "nested_cv",
        2,
        2,
        1,
        42,
        group_col,
        None,
    )
    legacy_plan = splits._plan_signature("nested_cv", 2, 2, 1, 42, group_col, None, 1)
    frame = splits._manifest_frame(
        sample_ids,
        "classification",
        "nested_cv",
        data_signature,
        legacy_plan,
        outer,
        inner,
    )
    frame["schema_version"] = 1
    frame["plan_signature"] = legacy_plan
    return frame


def test_legacy_ungrouped_manifest_is_upgraded_without_regenerating(
    monkeypatch, tmp_path
):
    sample_ids = ["s1", "s2", "s3", "s4"]
    y = np.array([0, 1, 0, 1])
    state = {"frame": _legacy_frame(sample_ids, y, None, None)}
    monkeypatch.setattr(splits, "table_exists", lambda path: True)
    monkeypatch.setattr(splits, "read_table", lambda path: state["frame"].copy())
    monkeypatch.setattr(
        splits,
        "write_table",
        lambda path, frame: state.update(frame=frame.copy()) or path,
    )

    def create():
        raise AssertionError("legacy ungrouped splits must not be regenerated")

    outer, inner, _ = splits.resolve_cv_splits(
        root=tmp_path,
        sample_ids=sample_ids,
        y=y,
        groups=None,
        strata=None,
        task="classification",
        target_name="group",
        protocol="nested_cv",
        outer_folds=2,
        inner_folds=2,
        repeats=1,
        random_state=42,
        group_col=None,
        stratify_col=None,
        create=create,
    )
    assert len(outer) == 2
    assert set(inner) == {"r0_o0", "r0_o1"}
    assert set(state["frame"]["schema_version"]) == {2}


def test_legacy_grouped_nested_manifest_is_rejected(monkeypatch, tmp_path):
    sample_ids = ["s1", "s2", "s3", "s4"]
    y = np.array([0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b"])
    state = {"frame": _legacy_frame(sample_ids, y, groups, "subject")}
    monkeypatch.setattr(splits, "table_exists", lambda path: True)
    monkeypatch.setattr(splits, "read_table", lambda path: state["frame"].copy())
    with pytest.raises(ValueError):
        splits.resolve_cv_splits(
            root=tmp_path,
            sample_ids=sample_ids,
            y=y,
            groups=groups,
            strata=None,
            task="classification",
            target_name="group",
            protocol="nested_cv",
            outer_folds=2,
            inner_folds=2,
            repeats=1,
            random_state=42,
            group_col="subject",
            stratify_col=None,
            create=lambda: ([], {}),
        )
