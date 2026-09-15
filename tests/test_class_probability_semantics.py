import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier

from mllabiome.data import _encode_y
from mllabiome.learners import _learner_factory, build_learner
from mllabiome.metrics import _predict_proba_aligned, compute_metrics


class HardOnlyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_positive_class_is_canonical_internal_class_one():
    y, labels, positive = _encode_y(
        ["case", "control", "case", "control"],
        label_map={"control": 8, "case": 3},
        labels=("case label", "control label"),
        positive_class=3,
    )
    assert positive == 1
    assert labels == ["control label", "case label"]
    assert y.tolist() == [1, 0, 1, 0]


def test_positive_class_can_be_named():
    y, labels, positive = _encode_y(
        ["Placebo", "Active", "Active", "Placebo"],
        labels=("Placebo", "Active"),
        positive_class="Active",
    )
    assert positive == 1
    assert labels == ["Placebo", "Active"]
    assert y.tolist() == [0, 1, 1, 0]


def test_unknown_target_label_raises():
    with pytest.raises(ValueError, match="not present in label_map"):
        _encode_y(
            ["control", "unexpected"],
            label_map={"control": 0, "case": 1},
            labels=("control", "case"),
            positive_class=1,
        )


def test_noncontiguous_label_map_is_canonicalized():
    y, labels, positive = _encode_y(
        ["negative", "positive", "negative", "positive"],
        label_map={"negative": 20, "positive": 90},
        positive_class=90,
    )
    assert set(y.tolist()) == {0, 1}
    assert labels == ["negative", "positive"]
    assert positive == 1


def test_invalid_positive_class_raises():
    with pytest.raises(ValueError, match="does not identify a binary class"):
        _encode_y(
            ["control", "case"],
            label_map={"control": 0, "case": 1},
            positive_class="disease",
        )


def test_predict_proba_alignment_rejects_hard_only_estimators():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = HardOnlyClassifier().fit(X, y)
    with pytest.raises(TypeError, match="does not provide predict_proba"):
        _predict_proba_aligned(clf, X, np.array([0, 1]))


def test_custom_decision_estimator_is_calibrated():
    name, factory = _learner_factory(("ridge_custom", RidgeClassifier(alpha=1.0)))
    assert name == "ridge_custom"
    clf = factory()
    assert callable(getattr(clf, "predict_proba", None))


def test_builtin_ridge_is_calibrated():
    X = np.arange(48, dtype=float).reshape(24, 2)
    y = np.array([0] * 12 + [1] * 12)
    clf = build_learner("ridge_a1")
    clf.fit(X, y)
    proba = _predict_proba_aligned(clf, X, np.array([0, 1]))
    assert proba.shape == (24, 2)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_hard_only_custom_estimator_is_rejected():
    _, factory = _learner_factory(("hard_only", HardOnlyClassifier()))
    with pytest.raises(TypeError, match="neither predict_proba nor decision_function"):
        factory()


def test_binary_metrics_follow_canonical_positive_class():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    metrics = compute_metrics(y_true, y_pred, proba, np.array([0, 1]), positive_class=1)
    assert metrics["AUC"] == 1.0
    assert metrics["PR_AUC"] == 1.0


def test_probability_matrix_validation_is_strict():
    class BadProbabilityClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            return np.tile(np.array([[0.9, 0.9]]), (len(X), 1))

    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    clf = BadProbabilityClassifier().fit(X, y)
    with pytest.raises(ValueError, match="rows must sum to 1"):
        _predict_proba_aligned(clf, X, np.array([0, 1]))
