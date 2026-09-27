from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from sklearn import get_config, set_config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _load_learners():
    root = Path(__file__).resolve().parents[1] / "mllabiome"
    package = sys.modules.get("mllabiome")
    if package is None:
        package = types.ModuleType("mllabiome")
        package.__path__ = [str(root)]
        sys.modules["mllabiome"] = package
    for name in ("estimator_protocol", "learners"):
        key = f"mllabiome.{name}"
        if key in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(key, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules["mllabiome.learners"]


learners = _load_learners()
FLAMLClassifier = learners.FLAMLClassifier
GroupAwareCalibratedClassifier = learners.GroupAwareCalibratedClassifier
fit_classifier = learners.fit_classifier


def _dataset():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(12), 2)
    y = np.repeat(np.arange(12) % 2, 2)
    X = rng.normal(size=(24, 5))
    X[:, 0] += y
    return X, y, groups


def _assert_group_disjoint(cv, groups):
    assert isinstance(cv, list)
    assert len(cv) >= 2
    for train, test in cv:
        assert set(groups[train]).isdisjoint(set(groups[test]))


@pytest.mark.parametrize(
    "estimator",
    [
        GridSearchCV(
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1.0]},
            cv=3,
        ),
        RandomizedSearchCV(
            LogisticRegression(max_iter=1000),
            {"C": [0.1, 1.0, 10.0]},
            n_iter=2,
            cv=3,
            random_state=7,
        ),
        CalibratedClassifierCV(LogisticRegression(max_iter=1000), cv=3),
        LogisticRegressionCV(
            Cs=[0.1, 1.0],
            cv=3,
            max_iter=1000,
            l1_ratios=(0.0,),
            scoring="accuracy",
            use_legacy_attributes=False,
        ),
    ],
)
def test_internal_cv_is_materialized_as_group_disjoint_splits(estimator):
    X, y, groups = _dataset()
    fit_classifier(estimator, X, y, groups)
    _assert_group_disjoint(estimator.cv, groups)


def test_grid_search_inside_pipeline_is_group_disjoint():
    X, y, groups = _dataset()
    search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1.0]},
        cv=3,
    )
    estimator = Pipeline([("scale", StandardScaler()), ("search", search)])
    fit_classifier(estimator, X, y, groups)
    _assert_group_disjoint(estimator.named_steps["search"].cv, groups)


def test_unsafe_user_cv_is_rejected():
    X, y, groups = _dataset()
    estimator = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [1.0]},
        cv=KFold(n_splits=3, shuffle=True, random_state=7),
    )
    with pytest.raises(ValueError, match="not group-disjoint"):
        fit_classifier(estimator, X, y, groups)


def test_nested_internal_cv_is_rejected():
    X, y, groups = _dataset()
    estimator = GridSearchCV(
        CalibratedClassifierCV(LogisticRegression(max_iter=1000), cv=3),
        {"method": ["sigmoid"]},
        cv=3,
    )
    with pytest.raises(ValueError, match="requires its own grouped fitting"):
        fit_classifier(estimator, X, y, groups)


def test_grid_search_rejects_group_sensitive_estimator_inside_pipeline():
    X, y, groups = _dataset()
    inner = Pipeline(
        [("scale", StandardScaler()), ("automl", FLAMLClassifier(time_budget=1))]
    )
    estimator = GridSearchCV(inner, {"scale__with_mean": [True, False]}, cv=3)
    with pytest.raises(ValueError, match="requires its own grouped fitting"):
        fit_classifier(estimator, X, y, groups)


def test_internal_cv_inside_opaque_meta_estimator_is_rejected():
    X, y, groups = _dataset()
    search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.1, 1.0]},
        cv=3,
    )
    estimator = VotingClassifier(
        [("search", search), ("lr", LogisticRegression(max_iter=1000))],
        voting="soft",
    )
    with pytest.raises(ValueError, match="Only transparent Pipeline nesting"):
        fit_classifier(estimator, X, y, groups)


def test_svc_probability_internal_cv_is_rejected():
    X, y, groups = _dataset()
    with pytest.raises(ValueError, match="internal sample-level cross-validation"):
        fit_classifier(SVC(probability=True), X, y, groups)


def test_group_aware_calibration_receives_groups():
    X, y, groups = _dataset()
    estimator = GroupAwareCalibratedClassifier(
        LogisticRegression(max_iter=1000),
        n_splits=3,
        random_state=7,
    )
    fit_classifier(estimator, X, y, groups)
    assert estimator.model_.classes_.shape[0] == 2


class GroupProbeClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, groups=None):
        if groups is None:
            raise AssertionError("groups were not routed")
        self.groups_ = np.asarray(groups).copy()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))

    def predict_proba(self, X):
        result = np.zeros((len(X), len(self.classes_)), dtype=float)
        result[:, 0] = 1.0
        return result


def test_pipeline_routes_groups_to_group_sensitive_final_estimator():
    X, y, groups = _dataset()
    estimator = Pipeline(
        [("scale", StandardScaler()), ("model", GroupProbeClassifier())]
    )
    previous = bool(get_config().get("enable_metadata_routing", False))
    set_config(enable_metadata_routing=True)
    try:
        fit_classifier(estimator, X, y, groups)
        np.testing.assert_array_equal(estimator.named_steps["model"].groups_, groups)
        assert bool(get_config().get("enable_metadata_routing", False)) is True
    finally:
        set_config(enable_metadata_routing=previous)


def test_flaml_backend_receives_group_split_configuration(monkeypatch):
    X, y, groups = _dataset()

    class AutoML:
        def fit(self, **kwargs):
            self.fit_kwargs_ = kwargs
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            result = np.zeros((len(X), 2), dtype=float)
            result[:, 0] = 1.0
            return result

    module = types.ModuleType("flaml")
    module.AutoML = AutoML
    monkeypatch.setitem(sys.modules, "flaml", module)
    estimator = Pipeline(
        [("scale", StandardScaler()), ("automl", FLAMLClassifier(time_budget=1))]
    )
    fit_classifier(estimator, X, y, groups)
    kwargs = estimator.named_steps["automl"].model_.fit_kwargs_
    np.testing.assert_array_equal(kwargs["groups"], groups)
    assert kwargs["split_type"] == "group"
