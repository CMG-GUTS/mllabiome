from types import SimpleNamespace

import numpy as np
import pandas as pd

from mllabiome import Permutation
from mllabiome import explainability as xai


def _fold(class_index, label, values, signed=None):
    frame = pd.DataFrame(
        {
            "method": "shap",
            "class_index": class_index,
            "class_label": label,
            "feature": list(values),
            "importance_mean": list(values.values()),
        }
    )
    if signed is not None:
        frame["signed_importance_mean"] = [signed[k] for k in values]
    return frame


def test_multiclass_consensus_does_not_pool_classes():
    shap = pd.DataFrame(
        {
            "method": ["shap"] * 4,
            "class_index": [0, 0, 1, 1],
            "class_label": ["A", "A", "B", "B"],
            "feature": ["x", "y", "x", "y"],
            "importance_mean": [10.0, 1.0, 1.0, 10.0],
        }
    )
    ale = pd.DataFrame(
        {
            "method": ["ale"] * 4,
            "class_index": [0, 0, 1, 1],
            "class_label": ["A", "A", "B", "B"],
            "feature": ["x", "y", "x", "y"],
            "importance_mean": [5.0, 2.0, 2.0, 5.0],
        }
    )
    out = xai._combine_feature_importance([shap, ale])
    c0 = out[out["class_index"].eq(0)].set_index("feature")
    c1 = out[out["class_index"].eq(1)].set_index("feature")
    assert c0.loc["x", "consensus_score"] > c0.loc["y", "consensus_score"]
    assert c1.loc["y", "consensus_score"] > c1.loc["x", "consensus_score"]


def test_outer_fold_stability_reports_rank_frequency_coverage_and_sign():
    frames = [
        _fold(0, "A", {"x": 4.0, "y": 2.0}, {"x": 0.5, "y": -0.2}),
        _fold(0, "A", {"x": 3.0, "y": 1.0}, {"x": 0.3, "y": -0.1}),
        _fold(0, "A", {"x": 1.0, "y": 5.0}, {"x": -0.2, "y": -0.4}),
    ]
    out = xai._aggregate_fold_feature_importance(
        "shap", frames, ["x", "y"], [0], ["A"], "test", top_k=1
    ).set_index("feature")
    assert np.isclose(out.loc["x", "importance_mean"], 8.0 / 3.0)
    assert np.isclose(out.loc["x", "top_k_frequency"], 2.0 / 3.0)
    assert np.isclose(out.loc["x", "fold_coverage"], 1.0)
    assert int(out.loc["x", "n_estimable_folds"]) == 3
    assert np.isclose(out.loc["x", "sign_positive_fraction"], 2.0 / 3.0)
    assert np.isclose(out.loc["x", "sign_negative_fraction"], 1.0 / 3.0)
    assert np.isclose(out.loc["x", "sign_consistency"], 2.0 / 3.0)
    assert out.loc["x", "rank_iqr"] >= 0.0


def test_missing_feature_in_some_folds_is_missing_not_zero():
    first = _fold(0, "A", {"x": 2.0, "y": 1.0})
    second = _fold(0, "A", {"x": 4.0})
    out = xai._aggregate_fold_feature_importance(
        "ale", [first, second], ["x", "y"], [0], ["A"], "ale", top_k=1
    ).set_index("feature")
    assert int(out.loc["y", "n_estimable_folds"]) == 1
    assert np.isclose(out.loc["y", "fold_coverage"], 0.5)
    assert np.isclose(out.loc["y", "importance_mean"], 1.0)


def test_class_specific_permutation_emits_one_table_per_requested_class():
    class Estimator:
        classes_ = np.asarray([0, 1, 2])

        def predict_proba(self, X):
            X = np.asarray(X)
            return np.tile(np.asarray([0.2, 0.3, 0.5]), (len(X), 1))

    out = xai._permutation_feature_importance(
        Estimator(),
        np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
        np.asarray([0, 1, 2]),
        ["x", "y"],
        ["A", "B", "C"],
        [0, 1, 2],
        spec=Permutation(n_repeats=3),
        random_state=42,
    )
    assert set(out["class_index"]) == {0, 1, 2}
    assert set(out["class_label"]) == {"A", "B", "C"}
    assert len(out) == 6
