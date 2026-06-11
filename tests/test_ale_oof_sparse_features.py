from __future__ import annotations

import numpy as np
import pandas as pd

from mllabiome import explainability as ex


class _DummyClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X[:, 1] if X.shape[1] > 1 else X[:, 0]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def test_oof_ale_skips_fold_constant_features_without_method_fallback(monkeypatch):
    def fake_ale(X, model, features, grid_size, include_CI, plot):
        fname = features[0]
        values = np.asarray(X[fname], dtype=float)
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
        return pd.DataFrame({"eff": [-0.25, 0.25]}, index=[lo, hi])

    monkeypatch.setattr(ex, "_require_pyale", lambda: fake_ale)

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=float,
    )
    frame = ex._ale_feature_importance(
        _DummyClassifier(),
        X,
        ["constant_in_fold", "variable_in_fold"],
        ["control", "case"],
        n_bins=4,
        top_features=["constant_in_fold", "variable_in_fold"],
    )

    assert frame["feature"].tolist() == ["variable_in_fold"]
    skipped = frame.attrs["skipped_features"]
    assert skipped["feature"].tolist() == ["constant_in_fold"]
    assert skipped["reason"].tolist() == ["fewer_than_two_distinct_finite_values"]


def test_oof_ale_mean_uses_estimable_folds_only():
    fold1 = pd.DataFrame(
        {
            "method": ["ale"],
            "feature": ["A"],
            "importance_mean": [2.0],
            "importance_sd": [0.0],
        }
    )
    fold2 = pd.DataFrame(
        {
            "method": ["ale"],
            "feature": ["B"],
            "importance_mean": [4.0],
            "importance_sd": [0.0],
        }
    )

    out = ex._mean_feature_importance_frames(
        "ale",
        [fold1, fold2],
        ["A", "B"],
        "mean_estimable_outer_fold_rms_centered_ale",
    ).set_index("feature")

    assert out.loc["A", "importance_mean"] == 2.0
    assert out.loc["B", "importance_mean"] == 4.0
    assert out.loc["A", "n_estimable_folds"] == 1
    assert out.loc["B", "n_estimable_folds"] == 1
