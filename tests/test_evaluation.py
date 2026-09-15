import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import mllabiome.configs_sweep as cs
from mllabiome.configs_sweep import Evaluation, QualificationGate, Sweep, evaluate
from mllabiome.data import Data, Dataset


class ConstantEstimator:
    def __init__(self, value=0, fit_log=None):
        self.value = int(value)
        self.fit_log = fit_log
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y):
        if self.fit_log is not None:
            self.fit_log.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        out = np.zeros((len(X), 2), dtype=float)
        out[:, self.value] = 1.0
        return out


class SignalEstimator:
    def __init__(self, fit_log=None):
        self.fit_log = fit_log
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y):
        if self.fit_log is not None:
            self.fit_log.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = (X[:, 0] > 0.5).astype(float)
        return np.column_stack([1.0 - p1, p1])


class AuditTransformer:
    def __init__(self, calls):
        self.calls = calls

    def apply_pair(self, X_tr, X_te):
        a = np.asarray(X_tr, dtype=float).copy()
        b = np.asarray(X_te, dtype=float).copy()
        self.calls.append((a, b))
        return a, b


def _dataset(n=12, y=None, outer_feature_shift=None):
    if y is None:
        y = np.array([i % 2 for i in range(n)], dtype=int)
    else:
        y = np.asarray(y, dtype=int)
    ids = np.arange(n, dtype=float)
    signal = np.array([i % 2 for i in range(n)], dtype=float)
    nuisance = ids + 10.0
    X = np.column_stack([signal, ids, nuisance]).astype(np.float32)
    if outer_feature_shift is not None:
        idx, shift = outer_feature_shift
        X[np.asarray(idx, dtype=int), 2] += float(shift)
    names = ["g__signal", "g__sample_index", "g__nuisance"]
    return Dataset(
        X_by_level={"genus": X, "all": X},
        feature_names_by_level={"genus": names, "all": names},
        y=y,
        sample_ids=[f"s{i}" for i in range(n)],
        metadata=pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)]}),
        class_labels=["control", "case"],
        positive_class=1,
    )


def _sweep(
    root,
    evaluation=None,
    gate=None,
    learners=("constant",),
    transformations=("identity",),
):
    return Sweep(
        data=Data(abundance_path="unused"),
        experiment_dir=root,
        resolutions=(("genus", ("genus",)),),
        count_transformations=transformations,
        learners=learners,
        evaluation=evaluation
        or Evaluation(
            protocol="nested_cv",
            outer_folds=3,
            inner_folds=2,
            repeats=1,
            optimize_metric="nMCC",
            random_state=17,
            n_jobs=1,
        ),
        gate=gate or QualificationGate(enabled=False, metric="nMCC", threshold=0.75),
        title="evaluation test",
    )


def _patch_runtime(monkeypatch, dataset, learner_factories, fit_log=None):
    monkeypatch.setattr(cs, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(cs, "_learner_name", lambda item: str(item))

    def learner_factory(item):
        name = str(item)
        factory = learner_factories[name]
        return name, factory

    monkeypatch.setattr(cs, "_learner_factory", learner_factory)
    monkeypatch.setattr(cs, "_write_rankings_and_figures", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cs, "_write_representation_impact_figure", lambda *args, **kwargs: None
    )


def _read_eval_tables(root):
    return {
        "outer": pd.read_csv(root / "results" / "outer_results.tsv", sep="\t"),
        "inner": pd.read_csv(root / "inner_results" / "inner_results.tsv", sep="\t"),
        "outer_pred": pd.read_csv(
            root / "predictions" / "outer_predictions.tsv", sep="\t"
        ),
        "inner_pred": pd.read_csv(
            root / "inner_predictions" / "inner_predictions.tsv", sep="\t"
        ),
    }


def test_repeated_nested_outer_splits_are_disjoint_complete_stratified_and_reproducible():
    y = np.array([i % 2 for i in range(24)], dtype=int)
    plan = Evaluation(
        protocol="repeated_nested_cv",
        outer_folds=4,
        inner_folds=3,
        repeats=2,
        random_state=23,
    )
    first = cs._outer_splits(plan, y, None)
    second = cs._outer_splits(plan, y, None)
    assert len(first) == 8
    assert [x["split_key"] for x in first] == [x["split_key"] for x in second]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a["train_idx"], b["train_idx"])
        np.testing.assert_array_equal(a["test_idx"], b["test_idx"])
        train = set(map(int, a["train_idx"]))
        test = set(map(int, a["test_idx"]))
        assert train.isdisjoint(test)
        assert train | test == set(range(len(y)))
        counts = Counter(y[a["test_idx"]].tolist())
        assert counts[0] == counts[1]
    for repeat in range(2):
        test_indices = np.concatenate(
            [x["test_idx"] for x in first if int(x["repeat"]) == repeat]
        )
        assert Counter(map(int, test_indices)) == Counter(range(len(y)))


def test_nested_cv_uses_one_outer_repeat_even_if_repeats_is_larger():
    y = np.array([i % 2 for i in range(20)], dtype=int)
    plan = Evaluation(
        protocol="nested_cv",
        outer_folds=5,
        inner_folds=2,
        repeats=9,
        random_state=7,
    )
    splits = cs._outer_splits(plan, y, None)
    assert len(splits) == 5
    assert {int(x["repeat"]) for x in splits} == {0}


def test_inner_splits_are_strictly_inside_outer_training_and_cover_it_once_as_validation():
    y = np.array([i % 2 for i in range(24)], dtype=int)
    plan = Evaluation(
        protocol="nested_cv",
        outer_folds=4,
        inner_folds=3,
        repeats=1,
        random_state=13,
    )
    outer = cs._outer_splits(plan, y, None)
    for outer_split in outer:
        outer_train = np.asarray(outer_split["train_idx"], dtype=int)
        outer_test = set(map(int, outer_split["test_idx"]))
        inner = cs._inner_splits(plan, y, outer_train, None, outer_split)
        assert len(inner) == 3
        validation_global = []
        for train_local, val_local in inner:
            train_global = outer_train[np.asarray(train_local, dtype=int)]
            val_global = outer_train[np.asarray(val_local, dtype=int)]
            assert set(map(int, train_global)).isdisjoint(set(map(int, val_global)))
            assert set(map(int, train_global)).isdisjoint(outer_test)
            assert set(map(int, val_global)).isdisjoint(outer_test)
            assert set(map(int, train_global)) | set(map(int, val_global)) == set(
                map(int, outer_train)
            )
            assert len(np.unique(y[train_global])) == 2
            validation_global.extend(map(int, val_global))
        assert Counter(validation_global) == Counter(map(int, outer_train))


def test_joint_metadata_strata_include_target_and_all_requested_columns():
    y = np.array([0, 1, 0, 1], dtype=int)
    meta = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B"],
            "sex": ["F", "M", "F", "M"],
        }
    )
    strata = cs._strata_from_metadata(meta, y, ("site", "sex"))
    assert strata.tolist() == [
        "0__strata__A__strata__F",
        "1__strata__A__strata__M",
        "0__strata__B__strata__F",
        "1__strata__B__strata__M",
    ]
    with pytest.raises(ValueError, match="missing"):
        cs._strata_from_metadata(meta, y, ("site", "missing"))


def test_outer_split_count_is_reduced_to_smallest_stratum_when_possible():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    plan = Evaluation(
        protocol="nested_cv", outer_folds=5, inner_folds=2, random_state=3
    )
    splits = cs._outer_splits(plan, y, None)
    assert len(splits) == 3


def test_outer_split_fails_when_any_stratum_has_fewer_than_two_samples():
    y = np.array([0, 0, 0, 1], dtype=int)
    plan = Evaluation(
        protocol="nested_cv", outer_folds=3, inner_folds=2, random_state=3
    )
    with pytest.raises(ValueError, match="Not enough samples"):
        cs._outer_splits(plan, y, None)


def test_lodo_holds_out_exactly_one_group_and_inner_splits_exclude_outer_group():
    y = np.array([0, 1, 0, 1] * 3, dtype=int)
    groups = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    plan = Evaluation(
        protocol="lodo", outer_folds=5, inner_folds=3, repeats=2, random_state=5
    )
    outer = cs._outer_splits(plan, y, groups)
    assert len(outer) == 3
    for split in outer:
        held = str(split["outer_group"])
        test_idx = np.asarray(split["test_idx"], dtype=int)
        train_idx = np.asarray(split["train_idx"], dtype=int)
        assert set(groups[test_idx]) == {held}
        assert held not in set(groups[train_idx])
        inner = cs._inner_splits(plan, y, train_idx, groups, split)
        assert inner
        for train_local, val_local in inner:
            train_global = train_idx[np.asarray(train_local, dtype=int)]
            val_global = train_idx[np.asarray(val_local, dtype=int)]
            assert held not in set(groups[train_global])
            assert held not in set(groups[val_global])
            assert len(set(groups[val_global])) == 1


def test_qualification_gate_contract():
    gate = QualificationGate(enabled=False, metric="nMCC", threshold=None)
    assert gate.qualifies(float("nan"))
    enabled = QualificationGate(enabled=True, metric="nMCC", threshold=0.75)
    assert enabled.qualifies(0.75)
    assert enabled.qualifies(0.80)
    assert not enabled.qualifies(0.74)
    assert not enabled.qualifies(float("nan"))
    with pytest.raises(ValueError, match="threshold"):
        QualificationGate(enabled=True, metric="nMCC", threshold=None).qualifies(0.9)


def test_config_ids_include_mpdr_semantics_version(monkeypatch):
    original = cs._MPDR_SEMANTICS
    first = cs._config_id("identity", "genus", "model")
    assert first == cs._config_id("identity", "genus", "model")
    monkeypatch.setattr(cs, "_MPDR_SEMANTICS", original + "_different")
    second = cs._config_id("identity", "genus", "model")
    assert first != second


def test_evaluate_output_accounting_is_exact_and_predictions_align_to_samples(
    tmp_path, monkeypatch
):
    dataset = _dataset(12)
    factories = {
        "a": lambda: ConstantEstimator(0),
        "b": lambda: ConstantEstimator(1),
    }
    _patch_runtime(monkeypatch, dataset, factories)
    sweep = _sweep(
        tmp_path / "run",
        learners=("a", "b"),
        transformations=("identity", "relative_abundance"),
    )
    evaluate(sweep)
    tables = _read_eval_tables(sweep.root())
    configs = pd.read_csv(sweep.root() / "configs.tsv", sep="\t")
    assert len(configs[configs["active"].eq(1)]) == 4
    assert len(tables["outer"]) == 12
    assert len(tables["inner"]) == 24
    assert len(tables["outer_pred"]) == 48
    assert len(tables["inner_pred"]) == 96
    assert not tables["outer"].duplicated(["split_key", "config_id"]).any()
    assert not tables["inner"].duplicated(["inner_key", "config_id"]).any()
    assert (
        not tables["outer_pred"]
        .duplicated(["split_key", "config_id", "sample_id"])
        .any()
    )
    assert (
        not tables["inner_pred"]
        .duplicated(["split_key", "config_id", "sample_id"])
        .any()
    )
    lookup = dict(zip(dataset.sample_ids, dataset.y))
    assert all(
        int(row.y_true) == int(lookup[row.sample_id])
        for row in tables["outer_pred"].itertuples()
    )
    assert all(
        int(row.y_true) == int(lookup[row.sample_id])
        for row in tables["inner_pred"].itertuples()
    )
    assert set(tables["outer"]["count_transformation"]) == {
        "identity",
        "relative_abundance",
    }
    assert "transformation_abbreviation" not in tables["outer"].columns
    assert "transformation_abbreviation" not in tables["inner"].columns


def test_gate_is_computed_from_inner_validation_and_unqualified_configs_skip_outer_evaluation(
    tmp_path, monkeypatch
):
    dataset = _dataset(12)
    factories = {
        "signal": lambda: SignalEstimator(),
        "constant": lambda: ConstantEstimator(0),
    }
    _patch_runtime(monkeypatch, dataset, factories)
    sweep = _sweep(
        tmp_path / "gate",
        learners=("signal", "constant"),
        transformations=("identity",),
        gate=QualificationGate(enabled=True, metric="nMCC", threshold=0.75),
    )
    evaluate(sweep)
    tables = _read_eval_tables(sweep.root())
    qual = pd.read_csv(sweep.root() / "tables" / "qualification_gate.tsv", sep="\t")
    configs = pd.read_csv(sweep.root() / "configs.tsv", sep="\t")
    signal_id = configs.loc[configs["learner"].eq("signal"), "config_id"].iloc[0]
    constant_id = configs.loc[configs["learner"].eq("constant"), "config_id"].iloc[0]
    assert len(qual) == 6
    assert set(qual.loc[qual["config_id"].eq(signal_id), "qualified"]) == {1}
    assert set(qual.loc[qual["config_id"].eq(constant_id), "qualified"]) == {0}
    assert set(tables["outer"]["config_id"]) == {signal_id}
    assert set(tables["outer_pred"]["config_id"]) == {signal_id}
    assert set(tables["inner"]["config_id"]) == {signal_id, constant_id}
    for row in qual.itertuples():
        sub = tables["inner"][
            tables["inner"]["split_key"].astype(str).eq(str(row.split_key))
            & tables["inner"]["config_id"].astype(str).eq(str(row.config_id))
            & tables["inner"]["ok"].eq(1)
        ]
        assert len(sub) == 2
        assert float(row.inner_score) == pytest.approx(float(sub["nMCC"].mean()))


def test_evaluate_passes_only_inner_training_to_inner_transform_and_only_outer_training_to_outer_transform(
    tmp_path, monkeypatch
):
    dataset = _dataset(12)
    factories = {"constant": lambda: ConstantEstimator(0)}
    _patch_runtime(monkeypatch, dataset, factories)
    calls = []
    seeds = []

    def transformation_factory(item, *, random_state):
        seeds.append(int(random_state))
        name = str(item[0]) if isinstance(item, tuple) else str(item)
        return name, lambda: AuditTransformer(calls)

    monkeypatch.setattr(cs, "_count_transformation_factory", transformation_factory)
    plan = Evaluation(
        protocol="nested_cv",
        outer_folds=3,
        inner_folds=2,
        repeats=1,
        optimize_metric="nMCC",
        random_state=31,
    )
    sweep = _sweep(tmp_path / "audit", evaluation=plan)
    expected_outer = cs._outer_splits(plan, dataset.y, None)
    evaluate(sweep)
    assert len(seeds) == 9
    assert set(seeds) == {31}
    assert len(calls) == 9
    for outer_no, outer_split in enumerate(expected_outer):
        group = calls[outer_no * 3 : outer_no * 3 + 3]
        outer_train = set(map(int, outer_split["train_idx"]))
        outer_test = set(map(int, outer_split["test_idx"]))
        inner = cs._inner_splits(
            plan,
            dataset.y,
            np.asarray(outer_split["train_idx"], dtype=int),
            None,
            outer_split,
        )
        for inner_no in range(2):
            train_matrix, val_matrix = group[inner_no]
            train_ids = set(map(int, train_matrix[:, 1]))
            val_ids = set(map(int, val_matrix[:, 1]))
            assert train_ids.isdisjoint(val_ids)
            assert train_ids | val_ids == outer_train
            assert train_ids.isdisjoint(outer_test)
            assert val_ids.isdisjoint(outer_test)
            local_train, local_val = inner[inner_no]
            global_train = set(
                map(int, np.asarray(outer_split["train_idx"])[local_train])
            )
            global_val = set(map(int, np.asarray(outer_split["train_idx"])[local_val]))
            assert train_ids == global_train
            assert val_ids == global_val
        train_matrix, test_matrix = group[2]
        assert set(map(int, train_matrix[:, 1])) == outer_train
        assert set(map(int, test_matrix[:, 1])) == outer_test


def test_outer_test_labels_cannot_change_inner_results_or_gate_decisions(
    tmp_path, monkeypatch
):
    train_idx = np.arange(8, dtype=int)
    test_idx = np.arange(8, 12, dtype=int)
    fixed_outer = {
        "split_key": "fixed_outer",
        "repeat": 0,
        "outer_fold": 0,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    fixed_inner = [
        (np.array([0, 1, 4, 5]), np.array([2, 3, 6, 7])),
        (np.array([2, 3, 6, 7]), np.array([0, 1, 4, 5])),
    ]
    base_y = np.array([i % 2 for i in range(12)], dtype=int)
    flipped_y = base_y.copy()
    flipped_y[test_idx] = 1 - flipped_y[test_idx]
    datasets = [_dataset(12, y=base_y), _dataset(12, y=flipped_y)]
    outputs = []

    for run_no, dataset in enumerate(datasets):
        factories = {
            "signal": lambda: SignalEstimator(),
            "constant": lambda: ConstantEstimator(0),
        }
        _patch_runtime(monkeypatch, dataset, factories)
        monkeypatch.setattr(cs, "_outer_splits", lambda *args, **kwargs: [fixed_outer])
        monkeypatch.setattr(cs, "_inner_splits", lambda *args, **kwargs: fixed_inner)
        sweep = _sweep(
            tmp_path / f"leakage_{run_no}",
            learners=("signal", "constant"),
            transformations=("identity",),
            gate=QualificationGate(enabled=True, metric="nMCC", threshold=0.75),
        )
        evaluate(sweep)
        tables = _read_eval_tables(sweep.root())
        qual = pd.read_csv(sweep.root() / "tables" / "qualification_gate.tsv", sep="\t")
        outputs.append((tables, qual))

    inner_a = (
        outputs[0][0]["inner"]
        .sort_values(["config_id", "inner_key"])
        .reset_index(drop=True)
    )
    inner_b = (
        outputs[1][0]["inner"]
        .sort_values(["config_id", "inner_key"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(inner_a, inner_b, check_dtype=False)
    qual_a = (
        outputs[0][1].sort_values(["config_id", "split_key"]).reset_index(drop=True)
    )
    qual_b = (
        outputs[1][1].sort_values(["config_id", "split_key"]).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(qual_a, qual_b, check_dtype=False)
    outer_a = outputs[0][0]["outer"].sort_values("config_id").reset_index(drop=True)
    outer_b = outputs[1][0]["outer"].sort_values("config_id").reset_index(drop=True)
    assert not np.allclose(
        outer_a["nMCC"].to_numpy(dtype=float),
        outer_b["nMCC"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_completed_evaluation_resumes_without_refitting_or_duplicating_rows(
    tmp_path, monkeypatch
):
    dataset = _dataset(12)
    fit_log = []
    factories = {
        "a": lambda: ConstantEstimator(0, fit_log),
        "b": lambda: ConstantEstimator(1, fit_log),
    }
    _patch_runtime(monkeypatch, dataset, factories)
    sweep = _sweep(
        tmp_path / "resume",
        learners=("a", "b"),
        transformations=("identity",),
    )
    evaluate(sweep)
    assert fit_log
    first = _read_eval_tables(sweep.root())
    fit_log.clear()
    evaluate(sweep)
    second = _read_eval_tables(sweep.root())
    assert fit_log == []
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key], check_dtype=False)


def test_redo_recomputes_but_does_not_duplicate_rows(tmp_path, monkeypatch):
    dataset = _dataset(12)
    fit_log = []
    factories = {"a": lambda: ConstantEstimator(0, fit_log)}
    _patch_runtime(monkeypatch, dataset, factories)
    plan = Evaluation(
        protocol="nested_cv",
        outer_folds=3,
        inner_folds=2,
        repeats=1,
        random_state=17,
        redo=False,
    )
    sweep = _sweep(tmp_path / "redo", evaluation=plan, learners=("a",))
    evaluate(sweep)
    first = _read_eval_tables(sweep.root())
    fit_log.clear()
    sweep.evaluation.redo = True
    evaluate(sweep)
    assert fit_log
    second = _read_eval_tables(sweep.root())
    assert len(first["outer"]) == len(second["outer"]) == 3
    assert len(first["inner"]) == len(second["inner"]) == 6
    assert not second["outer"].duplicated(["split_key", "config_id"]).any()
    assert not second["inner"].duplicated(["inner_key", "config_id"]).any()


def test_manifest_records_evaluation_protocol_and_mpdr_semantics(tmp_path, monkeypatch):
    dataset = _dataset(12)
    factories = {"constant": lambda: ConstantEstimator(0)}
    _patch_runtime(monkeypatch, dataset, factories)
    sweep = _sweep(tmp_path / "manifest")
    evaluate(sweep)
    manifest = json.loads((sweep.root() / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mpdr_semantics"] == cs._MPDR_SEMANTICS
    assert manifest["sweep"]["evaluation"]["protocol"] == "nested_cv"
    assert manifest["sweep"]["evaluation"]["outer_folds"] == 3
    assert manifest["sweep"]["evaluation"]["inner_folds"] == 2
    assert manifest["sweep"]["evaluation"]["random_state"] == 17
