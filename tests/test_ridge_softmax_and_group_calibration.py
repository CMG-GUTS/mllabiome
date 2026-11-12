"""General learner regression tests for calibration and ridge-softmax."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from mllabiome.learners import GroupAwareCalibratedClassifier
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
