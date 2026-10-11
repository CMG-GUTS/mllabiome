from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC
from xgboost import XGBClassifier

from mllabiome.learners import (
    GroupAwareCalibratedClassifier,
    _learner_factory,
    fit_classifier,
)
from mllabiome.ridge_softmax import RidgeSoftmaxClassifier


@pytest.fixture(scope="module")
def grouped_data():
    X, _ = make_classification(
        n_samples=90,
        n_features=20,
        n_informative=9,
        n_redundant=3,
        random_state=42,
    )
    groups = np.repeat(np.arange(30), 3)
    y = np.repeat(np.arange(30) % 2, 3)
    return X, y, groups


@pytest.mark.parametrize(
    "name,estimator",
    [
        (
            "LS_RBF_g0p25_a0p2",
            LabelSpreading(kernel="rbf", gamma=0.25, alpha=0.2, max_iter=1000),
        ),
        (
            "XGB_500_d6_lr0p05",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=1,
                random_state=42,
            ),
        ),
        (
            "ET_500_leaf3_Bal",
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            ),
        ),
        (
            "SVM_RBF_C10_Bal_GroupCal",
            GroupAwareCalibratedClassifier(
                estimator=SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=False,
                    class_weight="balanced",
                ),
                method="sigmoid",
                n_splits=3,
                random_state=42,
            ),
        ),
        ("RC_a0p1_Softmax", RidgeSoftmaxClassifier(alpha=0.1)),
        ("BNB_a0p1", BernoulliNB(alpha=0.1)),
        (
            "RF_750_leaf3",
            RandomForestClassifier(
                n_estimators=750, min_samples_leaf=3, n_jobs=1, random_state=42
            ),
        ),
    ],
)
def test_added_manuscript_learner(grouped_data, name, estimator):
    X, y, groups = grouped_data
    if name.startswith("BNB_"):
        X = (X > 0.0).astype(float)
    returned_name, factory = _learner_factory((name, estimator))
    assert returned_name == name
    model = factory()
    fit_classifier(model, X, y, groups=groups)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    assert proba.shape == (X.shape[0], 2)
    assert np.isfinite(proba).all()
    assert np.all(proba >= 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    cloned_params = clone(estimator).get_params(deep=True)
    original_params = estimator.get_params(deep=True)
    assert cloned_params.keys() == original_params.keys()
    for key in original_params:
        if key != "estimator":
            lhs, rhs = cloned_params[key], original_params[key]
            if isinstance(lhs, (float, np.floating)) and np.isnan(lhs):
                assert isinstance(rhs, (float, np.floating)) and np.isnan(rhs)
            else:
                assert lhs == rhs


def test_ridge_softmax_binary_mapping_and_clone(grouped_data):
    X, y, _ = grouped_data
    model = RidgeSoftmaxClassifier(alpha=0.1, temperature=0.8).fit(X, y)
    scores = model.decision_function(X)
    proba = model.predict_proba(X)
    expected = 1.0 / (1.0 + np.exp(-2.0 * scores / 0.8))
    assert np.allclose(proba[:, 1], expected)
    assert np.array_equal(model.predict(X), model.classes_[np.argmax(proba, axis=1)])
    assert isinstance(clone(model), RidgeSoftmaxClassifier)


def test_ridge_softmax_multiclass(grouped_data):
    X, _, _ = grouped_data
    y = np.arange(len(X)) % 3
    model = RidgeSoftmaxClassifier(alpha=0.1).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(y), 3)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0)


@pytest.mark.parametrize("temperature", [0.0, -0.1, np.inf, np.nan])
def test_ridge_softmax_rejects_invalid_temperature(grouped_data, temperature):
    X, y, _ = grouped_data
    with pytest.raises(ValueError, match="temperature"):
        RidgeSoftmaxClassifier(temperature=temperature).fit(X, y)


@pytest.mark.parametrize(
    "name,estimator",
    [
        (
            "MLP_32_LBFGS",
            __import__("sklearn").pipeline.Pipeline(
                [
                    ("scaler", __import__("sklearn").preprocessing.StandardScaler()),
                    (
                        "mlp",
                        __import__("sklearn").neural_network.MLPClassifier(
                            hidden_layer_sizes=(32,),
                            activation="relu",
                            solver="lbfgs",
                            alpha=1e-3,
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "RF_1000_leaf5",
            RandomForestClassifier(
                n_estimators=1000, min_samples_leaf=5, n_jobs=1, random_state=42
            ),
        ),
        (
            "LDA_Eigen_Shrink",
            __import__("sklearn").discriminant_analysis.LinearDiscriminantAnalysis(
                solver="eigen", shrinkage="auto"
            ),
        ),
    ],
)
def test_existing_table_learner(grouped_data, name, estimator):
    X, y, groups = grouped_data
    _, factory = _learner_factory((name, estimator))
    model = factory()
    fit_classifier(model, X, y, groups=groups)
    proba = np.asarray(model.predict_proba(X))
    assert proba.shape == (len(y), 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_group_calibration_splits_are_disjoint(grouped_data):
    X, y, groups = grouped_data
    model = GroupAwareCalibratedClassifier(
        SVC(kernel="rbf", C=10, class_weight="balanced", probability=False),
        method="sigmoid",
        n_splits=3,
        random_state=42,
    )
    for train_idx, validation_idx in model._splits(X, y, groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[validation_idx]))
