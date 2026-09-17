import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from mllabiome.explainability import _permutation_feature_importance
from mllabiome.explainability_methods import Permutation


class _Classifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - p, p])


def test_permutation_progress_reaches_total():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = np.tile(np.arange(2), 10)
    events = []
    _permutation_feature_importance(
        _Classifier().fit(X, y),
        X,
        y,
        ("a", "b", "c"),
        ("negative", "positive"),
        (1,),
        spec=Permutation(n_repeats=2),
        random_state=42,
        progress_callback=lambda done, total, detail: events.append(
            (done, total, detail)
        ),
    )
    assert events[0][0] == 0
    assert events[-1][0] == events[-1][1] == 6
    assert "feature 3/3" in events[-1][2]
