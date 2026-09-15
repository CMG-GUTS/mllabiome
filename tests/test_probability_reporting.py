import json

import numpy as np
import pandas as pd
import pytest

from mllabiome.report_statistics import (
    _extended_metrics,
    _learner_probability_semantics,
    _strategy_probability_semantics,
)


def test_probability_metrics_are_computed_for_verified_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1, 0, 1, 0, 1],
            "proba_0": [0.9, 0.7, 0.2, 0.1, 0.6, 0.3, 0.8, 0.25],
            "proba_1": [0.1, 0.3, 0.8, 0.9, 0.4, 0.7, 0.2, 0.75],
        }
    )
    result = _extended_metrics(frame, proper_probability=True)
    expected_brier = np.mean(
        (frame["proba_1"].to_numpy() - frame["y_true"].to_numpy()) ** 2
    )
    assert result["Brier"] == pytest.approx(expected_brier)
    assert np.isfinite(result["LogLoss"])
    assert np.isfinite(result["CalibrationInTheLarge"])
    assert np.isfinite(result["CalibrationIntercept"])
    assert np.isfinite(result["CalibrationSlope"])


def test_calibrated_builtin_learners_are_probability_valid() -> None:
    for learner in (
        "RF_1000_msl5",
        "Ridge_a1",
        "SIAMCAT",
        "nearestcentroid_raw",
    ):
        valid, _ = _learner_probability_semantics(learner)
        assert valid


def test_mixed_selected_outer_fold_aggregations_disable_probability_metrics() -> None:
    frame = pd.DataFrame(
        {
            "aggregation_strategy": ["weighted_mean_proba", "rank_mean"],
            "members": [json.dumps(["c1"]), json.dumps(["c1"])],
        }
    )
    result = _strategy_probability_semantics(
        "MPMA-E", frame, {}, {"c1": "RF_1000_msl5"}
    )
    assert not result["valid"]
    assert set(result["aggregation_strategies"]) == {
        "weighted_mean_proba",
        "rank_mean",
    }


def test_probability_preserving_selected_ensemble_is_valid() -> None:
    frame = pd.DataFrame(
        {
            "aggregation_strategy": ["weighted_mean_proba", "mean_proba"],
            "members": [json.dumps(["c1"]), json.dumps(["c1"])],
        }
    )
    result = _strategy_probability_semantics("MPMA-E", frame, {}, {"c1": "Ridge_a1"})
    assert result["valid"]
