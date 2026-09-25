import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

import mllabiome.configs_sweep as cs
from mllabiome.configs_sweep import (
    Ensemble,
    Evaluation,
    QualificationGate,
    Sweep,
    evaluate,
)
from mllabiome.data import Data, Dataset
from mllabiome.ensemble_sweep import select_mpma_e_by_outer_fold
from mllabiome.selection import select_mpma_b_by_outer_fold


def _groups():
    return np.array(
        ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4 + ["E"] * 4 + ["F"] * 4
    )


def _labels():
    return np.tile(np.array([0, 1, 0, 1], dtype=int), 6)


def _configs():
    return pd.DataFrame(
        [
            {
                "config_id": "A",
                "learner": "RF",
                "resolution": "genus",
                "count_transformation": "identity",
                "active": 1,
            },
            {
                "config_id": "B",
                "learner": "LR",
                "resolution": "species",
                "count_transformation": "relative_abundance",
                "active": 1,
            },
        ]
    )


def _inner_results(outer_splits, score_a=0.8, score_b=0.7):
    rows = []
    for split in outer_splits:
        outer_key = str(split["split_key"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        inner = cs._inner_splits(
            Evaluation(protocol="lodo", inner_folds=3, random_state=42),
            _labels(),
            train_idx,
            _groups(),
            split,
        )
        for inner_no, _ in enumerate(inner):
            for cid, score in (("A", score_a), ("B", score_b)):
                rows.append(
                    {
                        "stage": "inner",
                        "split_key": outer_key,
                        "inner_key": f"{outer_key}__i{inner_no}",
                        "config_id": cid,
                        "ok": 1,
                        "MCC": float(score),
                    }
                )
    return pd.DataFrame(rows)


def _predictions(outer_splits, inner):
    rows = []
    y = _labels()
    groups = _groups()
    for split in outer_splits:
        outer_key = str(split["split_key"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        if inner:
            inner_splits = cs._inner_splits(
                Evaluation(protocol="lodo", inner_folds=3, random_state=42),
                y,
                train_idx,
                groups,
                split,
            )
            for inner_no, (_, val_local) in enumerate(inner_splits):
                val_idx = train_idx[np.asarray(val_local, dtype=int)]
                for cid, offset in (("A", 0.0), ("B", 0.1)):
                    for sample_idx in val_idx:
                        truth = int(y[sample_idx])
                        p1 = 0.8 if truth == 1 else 0.2
                        if cid == "B":
                            p1 = min(
                                max(p1 + (offset if truth == 1 else -offset), 0.01),
                                0.99,
                            )
                        rows.append(
                            {
                                "stage": "inner",
                                "split_key": f"{outer_key}__i{inner_no}",
                                "outer_split_key": outer_key,
                                "sample_id": f"s{sample_idx}",
                                "sample_index": int(sample_idx),
                                "config_id": cid,
                                "y_true": truth,
                                "y_pred": int(p1 >= 0.5),
                                "proba_control": 1.0 - p1,
                                "proba_case": p1,
                            }
                        )
        else:
            for cid in ("A", "B"):
                for sample_idx in np.asarray(split["test_idx"], dtype=int):
                    truth = int(y[sample_idx])
                    p1 = 0.8 if truth == 1 else 0.2
                    rows.append(
                        {
                            "stage": "outer",
                            "split_key": outer_key,
                            "outer_split_key": outer_key,
                            "sample_id": f"s{sample_idx}",
                            "sample_index": int(sample_idx),
                            "config_id": cid,
                            "y_true": truth,
                            "y_pred": int(p1 >= 0.5),
                            "proba_control": 1.0 - p1,
                            "proba_case": p1,
                        }
                    )
    return pd.DataFrame(rows)


def test_lodo_with_six_groups_creates_six_outer_splits():
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    splits = cs._outer_splits(plan, _labels(), _groups())
    assert len(splits) == 6
    assert {str(split["outer_group"]) for split in splits} == set("ABCDEF")


def test_lodo_outer_test_is_exactly_one_complete_group():
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    groups = _groups()
    splits = cs._outer_splits(plan, _labels(), groups)
    for split in splits:
        held = str(split["outer_group"])
        test_idx = np.asarray(split["test_idx"], dtype=int)
        train_idx = np.asarray(split["train_idx"], dtype=int)
        assert set(groups[test_idx]) == {held}
        assert held not in set(groups[train_idx])
        assert len(test_idx) == int(np.sum(groups == held))


def test_lodo_inner_validation_leaves_out_each_remaining_group_once():
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    groups = _groups()
    y = _labels()
    splits = cs._outer_splits(plan, y, groups)
    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        held_outer = str(split["outer_group"])
        inner = cs._inner_splits(plan, y, train_idx, groups, split)
        remaining = set(groups[train_idx])
        assert len(inner) == 5
        seen = []
        for train_local, val_local in inner:
            train_global = train_idx[np.asarray(train_local, dtype=int)]
            val_global = train_idx[np.asarray(val_local, dtype=int)]
            val_groups = set(groups[val_global])
            assert len(val_groups) == 1
            val_group = next(iter(val_groups))
            seen.append(val_group)
            assert val_group not in set(groups[train_global])
            assert held_outer not in set(groups[train_global])
            assert held_outer not in val_groups
        assert set(seen) == remaining


def test_lodo_inner_folds_parameter_does_not_truncate_leave_one_group_out():
    y = _labels()
    groups = _groups()
    for configured_inner_folds in (2, 3, 4, 10):
        plan = Evaluation(
            protocol="lodo",
            inner_folds=configured_inner_folds,
            random_state=42,
        )
        split = cs._outer_splits(plan, y, groups)[0]
        inner = cs._inner_splits(
            plan,
            y,
            np.asarray(split["train_idx"], dtype=int),
            groups,
            split,
        )
        assert len(inner) == 5


def test_lodo_mpma_b_selection_is_outer_cohort_specific_and_inner_only():
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    outer = cs._outer_splits(plan, _labels(), _groups())
    inner = _inner_results(outer)
    first_key = str(outer[0]["split_key"])
    second_key = str(outer[1]["split_key"])
    inner.loc[
        inner["split_key"].eq(first_key) & inner["config_id"].eq("B"),
        "MCC",
    ] = 0.95
    inner.loc[
        inner["split_key"].eq(second_key) & inner["config_id"].eq("A"),
        "MCC",
    ] = 0.95
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "MCC")
    selected = dict(zip(selection["outer_split_key"], selection["config_id"]))
    assert selected[first_key] == "B"
    assert selected[second_key] == "A"


def test_lodo_mpma_b_averages_inner_cohorts_equally_not_by_sample_count():
    outer_key = "lodo_0__A"
    inner = pd.DataFrame(
        [
            {
                "split_key": outer_key,
                "inner_key": f"{outer_key}__i0",
                "config_id": "A",
                "ok": 1,
                "MCC": 0.9,
            },
            {
                "split_key": outer_key,
                "inner_key": f"{outer_key}__i1",
                "config_id": "A",
                "ok": 1,
                "MCC": 0.1,
            },
            {
                "split_key": outer_key,
                "inner_key": f"{outer_key}__i0",
                "config_id": "B",
                "ok": 1,
                "MCC": 0.6,
            },
            {
                "split_key": outer_key,
                "inner_key": f"{outer_key}__i1",
                "config_id": "B",
                "ok": 1,
                "MCC": 0.6,
            },
        ]
    )
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "MCC")
    assert selection.iloc[0]["config_id"] == "B"
    assert float(selection.iloc[0]["inner_score"]) == pytest.approx(0.6)


def test_lodo_mpma_e_selection_uses_only_inner_held_out_cohort_predictions():
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    outer_splits = cs._outer_splits(plan, _labels(), _groups())
    inner_results = _inner_results(outer_splits)
    inner_predictions = _predictions(outer_splits, True)
    outer_predictions = _predictions(outer_splits, False)
    ensemble = Ensemble(
        sizes=(2,),
        selection_strategies=("top_k",),
        aggregation_strategies=("mean_proba", "weighted_mean_proba"),
        optimize_metric="MCC",
    )
    first, _, _ = select_mpma_e_by_outer_fold(
        inner_results,
        inner_predictions,
        outer_predictions,
        _configs(),
        ensemble,
        "MCC",
    )
    changed = outer_predictions.copy()
    changed["proba_control"] = outer_predictions["proba_case"].to_numpy()
    changed["proba_case"] = outer_predictions["proba_control"].to_numpy()
    changed["y_pred"] = 1 - outer_predictions["y_pred"].to_numpy()
    second, _, _ = select_mpma_e_by_outer_fold(
        inner_results,
        inner_predictions,
        changed,
        _configs(),
        ensemble,
        "MCC",
    )
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    assert len(first) == 6


def test_lodo_outer_split_keys_preserve_held_out_group_identity():
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    splits = cs._outer_splits(plan, _labels(), _groups())
    for split in splits:
        assert str(split["outer_group"]) in str(split["split_key"])


class _ConstantEstimator:
    def __init__(self):
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.tile(np.array([[0.5, 0.5]], dtype=float), (len(X), 1))


class _AuditTransformer:
    def __init__(self, calls):
        self.calls = calls

    def apply_pair(self, X_train, X_test):
        train = np.asarray(X_train, dtype=float).copy()
        test = np.asarray(X_test, dtype=float).copy()
        self.calls.append((train, test))
        return train, test


def _feature_dataset():
    groups = _groups()
    y = _labels()
    X = np.zeros((len(groups), 7), dtype=np.float32)
    X[:, 0] = 1.0
    for group_no, group in enumerate("ABCDEF", start=1):
        X[groups == group, group_no] = 1.0
    names = ["g__shared"] + [f"g__only_{group}" for group in "ABCDEF"]
    metadata = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(groups))],
            "study_id": groups,
        }
    )
    return Dataset(
        X_by_level={"genus": X, "all": X},
        feature_names_by_level={"genus": names, "all": names},
        y=y,
        sample_ids=metadata["sample_id"].tolist(),
        subject_ids=metadata["sample_id"].tolist(),
        metadata=metadata,
        class_labels=["control", "case"],
        positive_class=1,
    )


def test_lodo_feature_mask_excludes_outer_test_only_features():
    dataset = _feature_dataset()
    groups = dataset.metadata["study_id"].to_numpy()
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    split = cs._outer_splits(plan, dataset.y, groups)[0]
    X = dataset.X_by_level["genus"]
    train, test, mask = cs._lodo_feature_pair(
        X,
        np.asarray(split["train_idx"], dtype=int),
        np.asarray(split["test_idx"], dtype=int),
        plan.protocol,
    )
    assert str(split["outer_group"]) == "A"
    assert mask.tolist() == [True, False, True, True, True, True, True]
    assert train.shape[1] == 6
    assert test.shape[1] == 6


def test_lodo_feature_mask_excludes_inner_validation_only_features():
    dataset = _feature_dataset()
    groups = dataset.metadata["study_id"].to_numpy()
    plan = Evaluation(protocol="lodo", inner_folds=3, random_state=42)
    outer = cs._outer_splits(plan, dataset.y, groups)[0]
    outer_train = np.asarray(outer["train_idx"], dtype=int)
    inner = cs._inner_splits(plan, dataset.y, outer_train, groups, outer)
    inner_train_local, inner_val_local = inner[0]
    inner_train = outer_train[np.asarray(inner_train_local, dtype=int)]
    inner_val = outer_train[np.asarray(inner_val_local, dtype=int)]
    held_inner = str(groups[inner_val[0]])
    X = dataset.X_by_level["genus"]
    _, _, mask = cs._lodo_feature_pair(X, inner_train, inner_val, plan.protocol)
    expected = np.ones(7, dtype=bool)
    expected[1] = False
    expected[1 + "ABCDEF".index(held_inner)] = False
    np.testing.assert_array_equal(mask, expected)


def test_non_lodo_feature_view_preserves_complete_selected_feature_space():
    dataset = _feature_dataset()
    X = dataset.X_by_level["genus"]
    train_idx = np.arange(0, 16, dtype=int)
    test_idx = np.arange(16, 24, dtype=int)
    train, test, mask = cs._lodo_feature_pair(X, train_idx, test_idx, "nested_cv")
    assert mask.tolist() == [True] * 7
    np.testing.assert_array_equal(train, X[train_idx])
    np.testing.assert_array_equal(test, X[test_idx])


def test_evaluate_lodo_applies_fold_local_feature_vocabulary_before_transformation(
    tmp_path, monkeypatch
):
    dataset = _feature_dataset()
    calls = []
    monkeypatch.setattr(cs, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        cs,
        "_learner_factory",
        lambda item: (str(item[0]), lambda: _ConstantEstimator()),
    )
    monkeypatch.setattr(
        cs,
        "_count_transformation_factory",
        lambda item, random_state, feature_blocks=None: (
            "identity",
            lambda: _AuditTransformer(calls),
        ),
    )
    monkeypatch.setattr(cs, "_write_rankings_and_figures", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cs, "_write_representation_impact_figure", lambda *args, **kwargs: None
    )
    sweep = Sweep(
        data=Data(
            abundance_path="unused",
            group_col="study_id",
            class_labels=("control", "case"),
        ),
        experiment_dir=tmp_path / "lodo",
        resolutions=(("genus", ("genus",)),),
        count_transformations=("identity",),
        learners=(("constant", DummyClassifier(strategy="prior")),),
        evaluation=Evaluation(
            protocol="lodo",
            inner_folds=3,
            repeats=1,
            optimize_metric="MCC",
            random_state=42,
        ),
        gate=QualificationGate(enabled=False, metric="MCC", threshold=0.51),
        ensemble=Ensemble(sizes=(2,), optimize_metric="MCC"),
    )
    evaluate(sweep)
    assert len(calls) == 36
    for outer_no, outer_group in enumerate("ABCDEF"):
        fold_calls = calls[outer_no * 6 : outer_no * 6 + 6]
        for inner_train, inner_val in fold_calls[:5]:
            assert inner_train.shape[1] == 5
            assert inner_val.shape[1] == 5
        outer_train, outer_test = fold_calls[5]
        assert outer_train.shape[1] == 6
        assert outer_test.shape[1] == 6
