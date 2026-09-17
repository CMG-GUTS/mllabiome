from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mllabiome import explainability as core
from mllabiome import final_explainability as final_xai
from mllabiome import mpma_e_explainability as mpmae_xai
from mllabiome.configs_sweep import Explainability
from mllabiome.explainability_methods import Permutation


class IdentityTransformation:
    def apply_pair(self, train, test):
        return np.asarray(train, dtype=float), np.asarray(test, dtype=float)


class RecordingEstimator:
    def __init__(self):
        self.train_sample_ids = set()
        self.classes_ = np.asarray([0, 1], dtype=int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.train_sample_ids = set(X[:, 0].astype(int).tolist())
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p1 = np.full(X.shape[0], 0.5, dtype=float)
        return np.column_stack([1.0 - p1, p1])


def _dataset(n=6):
    return SimpleNamespace(
        y=np.asarray([0, 1, 0, 1, 0, 1][:n], dtype=int),
        class_labels=("A", "B"),
        sample_ids=np.asarray([f"s{i}" for i in range(n)], dtype=object),
        metadata=pd.DataFrame(index=np.arange(n)),
    )


def _splits():
    return [
        {
            "split_key": "outer0",
            "train_idx": np.asarray([2, 3, 4, 5], dtype=int),
            "test_idx": np.asarray([0, 1], dtype=int),
        },
        {
            "split_key": "outer1",
            "train_idx": np.asarray([0, 1, 4, 5], dtype=int),
            "test_idx": np.asarray([2, 3], dtype=int),
        },
        {
            "split_key": "outer2",
            "train_idx": np.asarray([0, 1, 2, 3], dtype=int),
            "test_idx": np.asarray([4, 5], dtype=int),
        },
    ]


def _sweep(root: Path, targets=("mpma_b", "mpma_e")):
    return SimpleNamespace(
        root=lambda: root,
        data=SimpleNamespace(group_col=None, stratify_col=None),
        evaluation=SimpleNamespace(protocol="repeated_nested_cv"),
        explainability=Explainability(
            targets=targets,
            methods=(Permutation(),),
            classes="auto",
            representative_instances=False,
        ),
    )


def test_final_mpma_b_specification_is_the_explanation_target(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame(
        [
            {"config_id": "not_final", "score": 0.99},
            {"config_id": "final_b", "score": 0.50},
        ]
    ).to_csv(tables / "mpma_rankings.tsv", sep="\t", index=False)
    models = {"MPMA-B": {"config_id": "final_b"}}
    seen = {}

    monkeypatch.setattr(final_xai, "build_final_models", lambda root: models)
    monkeypatch.setattr(
        final_xai, "_invalidate_stale", lambda root, models, explainability: None
    )
    monkeypatch.setattr(
        final_xai._core,
        "_automatic_explainability_targets",
        lambda sweep, rankings: ["mpma_b"],
    )

    def fake_explain_one(sweep, target_override=None, **kwargs):
        seen["target_override"] = target_override
        return {}

    monkeypatch.setattr(final_xai._core, "_explain_one", fake_explain_one)
    final_xai.explain(_sweep(tmp_path, targets=("mpma_b",)))
    assert seen["target_override"] == "final_b"


def test_mpma_b_oof_refits_exclude_every_test_sample(monkeypatch):
    dataset = _dataset()
    X = np.column_stack(
        [np.arange(len(dataset.y), dtype=float), np.linspace(0.0, 1.0, len(dataset.y))]
    )

    monkeypatch.setattr(core, "load_dataset", lambda data, levels: dataset)
    monkeypatch.setattr(
        core, "materialize_mpdr", lambda dataset, levels: (X, ["id", "x"])
    )
    monkeypatch.setattr(
        core, "_explainability_outer_splits", lambda sweep, dataset: _splits()
    )
    monkeypatch.setattr(
        core,
        "_configured_count_transformation_factory",
        lambda sweep, key: IdentityTransformation,
    )
    monkeypatch.setattr(
        core,
        "_configured_learner_factory",
        lambda sweep, key: RecordingEstimator,
    )

    row = pd.Series(
        {
            "levels": "all",
            "count_transformation": "identity",
            "learner": "dummy",
        }
    )
    bundle = core._fit_oof_single_for_explainability(_sweep(Path(".")), row)
    held_out = []
    for fold in bundle["folds"]:
        train = set(np.asarray(fold["train_idx"], dtype=int).tolist())
        test = set(np.asarray(fold["test_idx"], dtype=int).tolist())
        assert train.isdisjoint(test)
        assert fold["estimator"].train_sample_ids == train
        assert fold["estimator"].train_sample_ids.isdisjoint(test)
        held_out.extend(sorted(test))
    assert sorted(held_out) == list(range(len(dataset.y)))


def test_final_mpma_e_specification_is_passed_to_oof_fitter(tmp_path, monkeypatch):
    mpma_e = {
        "ensemble_config_id": "final_e",
        "selection_strategy": "super_learner",
        "aggregation_strategy": "weighted_mean_proba",
        "max_size": 3,
        "member_count": 2,
        "members": [
            {"config_id": "mA", "aggregation_weight": 0.7},
            {"config_id": "mB", "aggregation_weight": 0.3},
        ],
    }
    models = {"MPMA-B": {"config_id": "final_b"}, "MPMA-E": mpma_e}
    seen = {}

    monkeypatch.setattr(mpmae_xai, "build_final_models", lambda root: models)
    monkeypatch.setattr(
        mpmae_xai._core, "_normalise_explainability_methods", lambda methods: ()
    )

    def fake_fit(sweep, rankings, received):
        seen["ensemble"] = received
        return {
            "dataset": _dataset(),
            "reproduction": pd.DataFrame(),
            "linear_weights": np.asarray([0.7, 0.3], dtype=float),
        }

    monkeypatch.setattr(mpmae_xai, "_fit_oof_members", fake_fit)
    monkeypatch.setattr(
        mpmae_xai, "_write_prediction_tables", lambda bundle, model, out_dir: {}
    )
    mpmae_xai.explain_mpma_e(_sweep(tmp_path), rankings=pd.DataFrame())

    assert seen["ensemble"] == mpma_e
    metadata = json.loads(
        (tmp_path / "explainability" / "mpma_e" / "explained_unit.json").read_text()
    )
    assert metadata["ensemble_config_id"] == "final_e"
    assert [x["config_id"] for x in metadata["members"]] == ["mA", "mB"]
    assert metadata["aggregation_strategy"] == "weighted_mean_proba"
    assert (
        metadata["cross_validation_explanations"]
        == "outer_test_folds_of_final_selected_specification"
    )


def test_mpma_e_oof_refits_use_final_members_and_exclude_test_samples(
    tmp_path, monkeypatch
):
    dataset = _dataset()
    X = np.column_stack(
        [np.arange(len(dataset.y), dtype=float), np.linspace(0.0, 1.0, len(dataset.y))]
    )
    rankings = pd.DataFrame(
        [
            {
                "config_id": "mA",
                "levels": "all",
                "resolution": "raw",
                "count_transformation": "identity",
                "learner": "dummy",
            },
            {
                "config_id": "mB",
                "levels": "all",
                "resolution": "raw",
                "count_transformation": "identity",
                "learner": "dummy",
            },
        ]
    )
    mpma_e = {
        "ensemble_config_id": "final_e",
        "selection_strategy": "super_learner",
        "aggregation_strategy": "weighted_mean_proba",
        "max_size": 3,
        "member_count": 2,
        "members": [
            {"config_id": "mA", "aggregation_weight": 0.7},
            {"config_id": "mB", "aggregation_weight": 0.3},
        ],
    }

    monkeypatch.setattr(mpmae_xai, "load_dataset", lambda data, levels: dataset)
    monkeypatch.setattr(
        mpmae_xai, "materialize_mpdr", lambda dataset, levels: (X, ["id", "x"])
    )
    monkeypatch.setattr(
        mpmae_xai._core,
        "_explainability_outer_splits",
        lambda sweep, dataset: _splits(),
    )
    monkeypatch.setattr(
        mpmae_xai,
        "_lodo_feature_pair",
        lambda X, train_idx, test_idx, protocol: (
            np.asarray(X)[train_idx],
            np.asarray(X)[test_idx],
            np.ones(np.asarray(X).shape[1], dtype=bool),
        ),
    )
    monkeypatch.setattr(
        mpmae_xai._core,
        "_configured_count_transformation_factory",
        lambda sweep, key: IdentityTransformation,
    )
    monkeypatch.setattr(
        mpmae_xai._core,
        "_configured_learner_factory",
        lambda sweep, key: RecordingEstimator,
    )
    monkeypatch.setattr(
        mpmae_xai, "_stored_member_outer_predictions", lambda root: None
    )

    bundle = mpmae_xai._fit_oof_members(_sweep(tmp_path), rankings, mpma_e)
    held_out = []
    for fold in bundle["folds"]:
        train = set(np.asarray(fold["train_idx"], dtype=int).tolist())
        test = set(np.asarray(fold["test_idx"], dtype=int).tolist())
        assert train.isdisjoint(test)
        assert [member["config_id"] for member in fold["members"]] == ["mA", "mB"]
        for member in fold["members"]:
            assert member["estimator"].train_sample_ids == train
            assert member["estimator"].train_sample_ids.isdisjoint(test)
        held_out.extend(sorted(test))
    assert sorted(held_out) == list(range(len(dataset.y)))
