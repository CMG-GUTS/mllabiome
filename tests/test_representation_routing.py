from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from mllabiome import mll
from mllabiome.configs_sweep import (
    _build_classification_tasks,
    _learner_fingerprint,
    build_sweep_configs,
)
from mllabiome.learners import _learner_factory
from mllabiome.transformations import _count_transformation_specs_for_blocks


def _candidates():
    return (
        ("SIAMCAT_Default", mll.SIAMCATClassifier()),
        ("BNB_a0p1", BernoulliNB(alpha=0.1)),
        ("MNB_a0p5", MultinomialNB(alpha=0.5)),
        ("CNB_a0p5", ComplementNB(alpha=0.5)),
        ("RF_named", RandomForestClassifier(n_estimators=5, random_state=42)),
        ("RF_alias", RandomForestClassifier(n_estimators=5, random_state=42)),
    )


def test_siamcat_is_raw_identity_once_with_aliased_resolutions():
    configurations = build_sweep_configs(
        (("raw", ("raw",)), ("raw", ("all",)), ("genus", ("genus",))),
        (mll.Transformation("identity"), mll.Transformation("arcsine_sqrt")),
        _candidates(),
    )
    siamcat = configurations[configurations.learner.eq("SIAMCAT_Default")]
    assert siamcat[["resolution", "count_transformation"]].values.tolist() == [
        ["raw", "identity"]
    ]
    assert configurations.config_id.is_unique
    assert "RF_alias" not in configurations.learner.tolist()


def test_naive_bayes_requires_compatible_representation():
    configurations = build_sweep_configs(
        (("genus", ("genus",)),),
        (
            mll.Transformation("presence_absence"),
            mll.Transformation("relative_abundance"),
            mll.Transformation("clr"),
            mll.Transformation("arcsine_sqrt", composition_scope="joint"),
        ),
        _candidates(),
    )
    bnb = configurations[configurations.learner.eq("BNB_a0p1")]
    assert bnb.count_transformation.tolist() == ["presence_absence"]
    for name in ("MNB_a0p5", "CNB_a0p5"):
        transforms = set(
            configurations[configurations.learner.eq(name)].count_transformation
        )
        assert all("centered_log_ratio" not in transform for transform in transforms)
        assert "presence_absence" in transforms


def test_resolution_restrictions_and_unknown_names():
    configurations = build_sweep_configs(
        (("raw", ("all",)), ("genus", ("genus",))),
        (mll.Transformation("identity"),),
        _candidates(),
        learner_resolutions={
            name: ("genus",) for name, _ in _candidates() if name != "SIAMCAT_Default"
        },
    )
    assert set(configurations[configurations.resolution.eq("raw")].learner) == {
        "SIAMCAT_Default"
    }
    with pytest.raises(ValueError, match="Unknown learner_resolutions"):
        build_sweep_configs(
            (("genus", ("genus",)),),
            (mll.Transformation("identity"),),
            _candidates(),
            learner_resolutions={"missing": ("genus",)},
        )


def test_conflicting_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="conflicting estimator"):
        build_sweep_configs(
            (("genus", ("genus",)),),
            (mll.Transformation("identity"),),
            (
                ("RF", RandomForestClassifier(n_estimators=3)),
                ("RF", RandomForestClassifier(n_estimators=9)),
            ),
        )


def test_scheduler_only_queues_eligible_unique_pairs():
    models = (
        ("SIAMCAT_Default", mll.SIAMCATClassifier()),
        ("RF", RandomForestClassifier(n_estimators=5, random_state=42)),
    )
    resolutions = (("raw", ("all",)), ("raw", ("all",)), ("genus", ("genus",)))
    transforms = (
        mll.Transformation("identity"),
        mll.Transformation("presence_absence"),
    )
    blocks = {"raw": (("all", (0, 1)),), "genus": (("genus", (0, 1)),)}
    configurations = build_sweep_configs(
        resolutions,
        transforms,
        models,
        resolution_feature_blocks=blocks,
        learner_resolutions={"RF": ("genus",)},
    )
    matrix = np.array([[0, 1], [2, 0], [3, 3], [0, 0], [1, 1], [1, 2]], dtype=float)
    dataset = SimpleNamespace(
        y=np.array([0, 1, 0, 1, 0, 1]),
        classes=np.array([0, 1]),
        class_labels=("a", "b"),
        sample_ids=[str(index) for index in range(6)],
        subject_ids=[str(index) for index in range(6)],
    )
    context = SimpleNamespace(
        dataset=dataset,
        outer_splits=[
            {"split_key": "o0", "train_idx": [0, 1, 2, 3], "test_idx": [4, 5]}
        ],
        inner_splits_by_outer={"o0": [([0, 1], [2, 3])]},
        resolutions=resolutions,
        mpdr_cache={
            name: (matrix, ("a", "b"), blocks[name]) for name, _ in resolutions
        },
        transformation_specs_by_resolution={
            name: _count_transformation_specs_for_blocks(
                transforms, blocks[name], resolution=name
            )
            for name, _ in resolutions
        },
        learners=[_learner_factory(model) for model in models],
        learner_fingerprints={
            name: _learner_fingerprint(item)
            for name, *other in models
            for item in [(name, other[0])]
        },
        current_config_ids=set(configurations.config_id),
        groups=np.arange(6),
        inner_done=set(),
        outer_done=set(),
        qualification_map={},
        existing={"inner_metrics": pd.DataFrame()},
    )
    sweep = SimpleNamespace(
        evaluation=SimpleNamespace(
            protocol="nested_cv",
            optimize_metric="log_loss",
            random_state=42,
            resource_sample_interval_s=0.1,
        ),
        gate=SimpleNamespace(enabled=False, metric="MCC", threshold=None),
    )
    tasks, counts = _build_classification_tasks(sweep, context)
    assert len(configurations) == 3
    assert len(tasks) == 3
    assert counts == {"o0": 3}
    assert len({(key, config_id) for key, config_id, _ in tasks}) == 3
    assert all(config_id in context.current_config_ids for _, config_id, _ in tasks)
