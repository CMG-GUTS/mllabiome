import numpy as np
import pandas as pd

from mllabiome.explainability import (
    _combine_feature_importance,
    _mean_feature_importance_frames,
    _method_support_table,
    _rank_support_from_importance,
)
from mllabiome.explainability_visuals import _mean_support_column


def _frame(method, values):
    return pd.DataFrame(
        {
            "method": method,
            "feature": list(values),
            "importance_mean": list(values.values()),
            "importance_sd": 0.0,
            "scoring": method,
            "class_index": 0,
            "class_label": "class_0",
        }
    )


def test_ale_never_estimable_feature_remains_missing():
    fold1 = _frame("ale", {"A": 2.0})
    fold2 = _frame("ale", {"A": 4.0})
    out = _mean_feature_importance_frames(
        "ale", [fold1, fold2], ["A", "B"], "ale"
    ).set_index("feature")
    assert out.loc["A", "importance_mean"] == 3.0
    assert int(out.loc["A", "n_estimable_folds"]) == 2
    assert np.isnan(out.loc["B", "importance_mean"])
    assert np.isnan(out.loc["B", "importance_sd"])
    assert int(out.loc["B", "n_estimable_folds"]) == 0


def test_consensus_uses_within_method_ranks_not_raw_scales():
    shap = _frame("shap", {"A": 3000.0, "B": 2000.0, "C": 1000.0})
    permutation = _frame("permutation", {"A": 0.01, "B": 0.03, "C": 0.02})
    first = _combine_feature_importance([shap, permutation]).set_index("feature")
    shap_scaled = _frame("shap", {"A": 3.0, "B": 2.0, "C": 1.0})
    permutation_scaled = _frame("permutation", {"A": 1.0, "B": 3.0, "C": 2.0})
    second = _combine_feature_importance([shap_scaled, permutation_scaled]).set_index(
        "feature"
    )
    np.testing.assert_allclose(
        first.loc[["A", "B", "C"], "consensus_score"],
        second.loc[["A", "B", "C"], "consensus_score"],
    )


def test_missing_ale_is_excluded_not_converted_to_zero():
    shap = _frame("shap", {"A": 3.0, "B": 2.0, "C": 1.0})
    permutation = _frame("permutation", {"A": 1.0, "B": 3.0, "C": 2.0})
    ale = _frame("ale", {"A": 2.0, "C": 1.0})
    consensus = _combine_feature_importance([shap, permutation, ale]).set_index(
        "feature"
    )
    assert np.isclose(consensus.loc["B", "consensus_score"], 0.75)
    assert int(consensus.loc["B", "n_methods"]) == 2
    assert int(consensus.loc["B", "n_methods_total"]) == 3
    assert np.isclose(consensus.loc["B", "method_coverage"], 2.0 / 3.0)


def test_method_support_table_preserves_missing_ale():
    shap = _frame("shap", {"A": 3.0, "B": 2.0, "C": 1.0})
    permutation = _frame("permutation", {"A": 1.0, "B": 3.0, "C": 2.0})
    ale = _frame("ale", {"A": 2.0, "C": 1.0})
    consensus = _combine_feature_importance([shap, permutation, ale])
    table = _method_support_table([shap, permutation, ale], consensus, 3).set_index(
        "feature"
    )
    assert np.isnan(table.loc["B", "ALE"])
    assert np.isclose(table.loc["B", "consensus"], 0.75)
    assert int(table.loc["B", "n_methods"]) == 2


def test_visual_mean_support_ignores_missing_methods():
    frame = pd.DataFrame(
        {
            "feature": ["A", "B"],
            "SHAP": [1.0, 0.5],
            "ALE": [np.nan, 0.5],
        }
    )
    values = _mean_support_column(frame)
    np.testing.assert_allclose(values, [1.0, 0.5])


def test_rank_support_maps_best_to_one_and_worst_to_zero():
    support = _rank_support_from_importance(
        _frame("shap", {"A": 9.0, "B": 5.0, "C": 1.0})
    )
    assert support.loc[(0, "A")] == 1.0
    assert support.loc[(0, "B")] == 0.5
    assert support.loc[(0, "C")] == 0.0
