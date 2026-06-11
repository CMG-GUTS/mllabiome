from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pandas as pd

from mllabiome.learners import FLAMLClassifier, build_learner
from mllabiome.metrics import _predict_proba_aligned


class _RequiresNamedFrame:
    classes_ = np.array([0, 1], dtype=int)
    feature_names_in_ = np.array(["a", "b"], dtype=object)

    def predict_proba(self, X):
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == ["a", "b"]
        return np.tile(np.array([[0.25, 0.75]], dtype=float), (len(X), 1))


def test_predict_proba_aligned_preserves_fitted_feature_names() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    proba = _predict_proba_aligned(_RequiresNamedFrame(), X, np.array([0, 1], dtype=int))
    assert proba.shape == (2, 2)
    assert np.allclose(proba[:, 1], 0.75)


def test_flaml_wrapper_uses_named_frames_for_fit_and_predict(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    class FakeAutoML:
        def fit(self, **kwargs):
            X_train = kwargs["X_train"]
            assert isinstance(X_train, pd.DataFrame)
            calls.append(("fit", list(X_train.columns)))
            self.columns_ = list(X_train.columns)
            self.classes_ = np.array([0, 1], dtype=int)
            return self

        def predict_proba(self, X):
            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == self.columns_
            calls.append(("predict_proba", list(X.columns)))
            return np.tile(np.array([[0.4, 0.6]], dtype=float), (len(X), 1))

        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == self.columns_
            calls.append(("predict", list(X.columns)))
            return np.ones(len(X), dtype=int)

    fake_flaml = types.ModuleType("flaml")
    fake_flaml.AutoML = FakeAutoML
    monkeypatch.setitem(sys.modules, "flaml", fake_flaml)

    clf = FLAMLClassifier(time_budget=1, estimator_list=["lgbm"], verbose=0)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    clf.fit(X, y)
    clf.predict_proba(X)
    clf.predict(X)

    assert list(clf.feature_names_in_) == ["feature_000000", "feature_000001"]
    assert calls == [
        ("fit", ["feature_000000", "feature_000001"]),
        ("predict_proba", ["feature_000000", "feature_000001"]),
        ("predict", ["feature_000000", "feature_000001"]),
    ]


def test_builtin_logistic_learners_do_not_emit_penalty_deprecation_warnings() -> None:
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0], [0.0, 0.1], [1.0, 1.1]],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    for key in ("lr_l2", "lr_l1"):
        clf = build_learner(key)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf.fit(X, y)
        messages = [str(w.message) for w in caught]
        assert not any("'penalty' was deprecated" in msg for msg in messages)
        assert not any("Inconsistent values: penalty=" in msg for msg in messages)
