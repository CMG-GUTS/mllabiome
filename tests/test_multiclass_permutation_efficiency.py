import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from mllabiome.explainability import _permutation_feature_importance
from mllabiome.explainability_methods import Permutation


class CountingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.predict_proba_calls = 0

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        self.predict_proba_calls += 1
        X = np.asarray(X, dtype=float)
        logits = np.column_stack(
            [
                X[:, 0],
                X[:, 1],
                X[:, 2],
                -X[:, 0] - X[:, 1] - X[:, 2],
            ]
        )
        logits = logits - np.max(logits, axis=1, keepdims=True)
        values = np.exp(logits)
        return values / np.sum(values, axis=1, keepdims=True)


def test_multiclass_permutation_reuses_predictions_across_classes():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 3))
    y = np.tile(np.arange(4), 6)
    clf = CountingClassifier().fit(X, y)
    frame = _permutation_feature_importance(
        clf,
        X,
        y,
        ("a", "b", "c"),
        ("c0", "c1", "c2", "c3"),
        (0, 1, 2, 3),
        spec=Permutation(n_repeats=2),
        random_state=42,
    )
    assert clf.predict_proba_calls == X.shape[1] * (1 + 2)
    assert len(frame) == X.shape[1] * 4
    assert set(frame["class_index"]) == {0, 1, 2, 3}


def test_binary_permutation_remains_class_specific():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 3))
    y = np.tile(np.arange(2), 10)

    class BinaryClassifier(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=float)
            p = 1.0 / (1.0 + np.exp(-X[:, 0]))
            return np.column_stack([1.0 - p, p])

    frame = _permutation_feature_importance(
        BinaryClassifier().fit(X, y),
        X,
        y,
        ("a", "b", "c"),
        ("negative", "positive"),
        (1,),
        spec=Permutation(n_repeats=1),
        random_state=7,
    )
    assert set(frame["class_index"]) == {1}
    assert set(frame["class_label"]) == {"positive"}
