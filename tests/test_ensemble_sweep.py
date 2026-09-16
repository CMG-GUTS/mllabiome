from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mllabiome.configs_sweep import Ensemble
from mllabiome.ensemble_sweep import (
    _candidate_table_for_inner,
    select_final_mpma_e_candidate,
    select_mpma_e_by_outer_fold,
)


def _configs():
    return pd.DataFrame(
        [
            {"config_id": "A", "resolution": "genus", "learner": "RF"},
            {"config_id": "B", "resolution": "raw", "learner": "BNB"},
            {"config_id": "C", "resolution": "genus", "learner": "LR"},
            {"config_id": "D", "resolution": "raw", "learner": "XGB"},
        ]
    )


def _probabilities():
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    wrong = 0.6
    a = []
    b = []
    for i, target in enumerate(y):
        if i < 4:
            a.append(0.95 if target else 0.05)
            b.append(1.0 - wrong if target else wrong)
        else:
            b.append(0.95 if target else 0.05)
            a.append(1.0 - wrong if target else wrong)
    c = np.full(len(y), 0.5, dtype=float)
    d = np.asarray([0.65, 0.35, 0.62, 0.38, 0.62, 0.38, 0.65, 0.35], dtype=float)
    return y, {
        "A": np.asarray(a, dtype=float),
        "B": np.asarray(b, dtype=float),
        "C": c,
        "D": d,
    }


def _inner_results():
    rows = []
    for inner_key in ("inner0", "inner1"):
        for config_id in ("A", "B", "C", "D"):
            rows.append(
                {
                    "split_key": "outer0",
                    "inner_key": inner_key,
                    "config_id": config_id,
                    "ok": 1,
                    "nMCC": 0.5,
                }
            )
    return pd.DataFrame(rows)


def _inner_predictions():
    y, values = _probabilities()
    rows = []
    for inner_key, indices in (("inner0", range(0, 4)), ("inner1", range(4, 8))):
        for config_id, p1 in values.items():
            for index in indices:
                rows.append(
                    {
                        "outer_split_key": "outer0",
                        "split_key": inner_key,
                        "sample_id": f"i{index}",
                        "y_true": int(y[index]),
                        "config_id": config_id,
                        "proba_0": float(1.0 - p1[index]),
                        "proba_1": float(p1[index]),
                    }
                )
    return pd.DataFrame(rows)


def _outer_predictions(labels=None):
    _, values = _probabilities()
    y = (
        np.asarray([0, 1, 0, 1], dtype=int)
        if labels is None
        else np.asarray(labels, dtype=int)
    )
    rows = []
    for config_id, p1 in values.items():
        for index in range(4):
            rows.append(
                {
                    "outer_split_key": "outer0",
                    "sample_id": f"o{index}",
                    "y_true": int(y[index]),
                    "config_id": config_id,
                    "proba_0": float(1.0 - p1[index]),
                    "proba_1": float(p1[index]),
                }
            )
    return pd.DataFrame(rows)


def _plan(method):
    aggregation = (
        "weighted_mean_proba"
        if method in {"caruana", "super_learner"}
        else "mean_proba"
    )
    return Ensemble(
        max_sizes=(2,),
        selection_strategies=(method,),
        aggregation_strategies=(aggregation,),
        optimize_metric="log_loss",
    )


@pytest.mark.parametrize(
    "method",
    [
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    ],
)
def test_outer_labels_cannot_change_selected_ensemble(method):
    selected_a, _, _ = select_mpma_e_by_outer_fold(
        _inner_results(),
        _inner_predictions(),
        _outer_predictions([0, 1, 0, 1]),
        _configs(),
        _plan(method),
        "log_loss",
    )
    selected_b, _, _ = select_mpma_e_by_outer_fold(
        _inner_results(),
        _inner_predictions(),
        _outer_predictions([1, 0, 1, 0]),
        _configs(),
        _plan(method),
        "log_loss",
    )
    columns = [
        "ensemble_config_id",
        "selection_strategy",
        "aggregation_strategy",
        "members",
        "weights",
    ]
    pd.testing.assert_frame_equal(
        selected_a[columns].reset_index(drop=True),
        selected_b[columns].reset_index(drop=True),
    )


def test_log_loss_can_select_ensemble_without_log_loss_column_in_inner_results():
    plan = Ensemble(
        max_sizes=(2,),
        selection_strategies=(
            "top_k",
            "best_per_resolution",
            "best_per_learner_type",
            "caruana",
            "super_learner",
        ),
        aggregation_strategies=("mean_proba", "weighted_mean_proba"),
        optimize_metric="log_loss",
    )
    selection, predictions, metrics = select_mpma_e_by_outer_fold(
        _inner_results(),
        _inner_predictions(),
        _outer_predictions(),
        _configs(),
        plan,
        "log_loss",
    )
    assert len(selection) == 1
    assert not predictions.empty
    assert not metrics.empty
    assert np.isfinite(float(selection.iloc[0]["inner_score"]))


def test_final_log_loss_candidate_is_the_minimum_inner_loss():
    plan = Ensemble(
        max_sizes=(2, 3),
        selection_strategies=("top_k", "best_per_resolution"),
        aggregation_strategies=("mean_proba", "median_proba"),
        optimize_metric="log_loss",
    )
    best, candidates = select_final_mpma_e_candidate(
        _inner_results(),
        _inner_predictions(),
        _configs(),
        plan,
        "log_loss",
    )
    assert not candidates.empty
    assert float(best["inner_score"]) == pytest.approx(
        float(candidates["inner_score"].min())
    )


def test_candidate_table_contains_only_requested_compatible_pairs():
    plan = Ensemble(
        max_sizes=(2,),
        selection_strategies=("top_k", "caruana", "super_learner"),
        aggregation_strategies=(
            "mean_proba",
            "weighted_mean_proba",
            "median_proba",
        ),
        optimize_metric="log_loss",
    )
    table = _candidate_table_for_inner(
        _inner_results(),
        _inner_predictions(),
        _configs(),
        plan,
        "log_loss",
        "outer0",
    )
    pairs = set(zip(table["selection_strategy"], table["aggregation_strategy"]))
    assert pairs <= {
        ("top_k", "mean_proba"),
        ("top_k", "median_proba"),
        ("caruana", "weighted_mean_proba"),
        ("super_learner", "weighted_mean_proba"),
    }
    assert ("caruana", "weighted_mean_proba") in pairs
    assert ("super_learner", "weighted_mean_proba") in pairs


def test_learned_selectors_respect_max_size_and_store_weights():
    for method in ("caruana", "super_learner"):
        selection, _, _ = select_mpma_e_by_outer_fold(
            _inner_results(),
            _inner_predictions(),
            _outer_predictions(),
            _configs(),
            _plan(method),
            "log_loss",
        )
        row = selection.iloc[0]
        members = json.loads(row["members"])
        weights = np.asarray(json.loads(row["weights"]), dtype=float)
        assert 2 <= len(members) <= 2
        assert len(weights) == len(members)
        assert np.all(weights >= 0.0)
        np.testing.assert_allclose(weights.sum(), 1.0)


def test_caruana_and_super_learner_use_complete_inner_oof_library():
    plan = Ensemble(
        max_sizes=(2,),
        selection_strategies=("caruana", "super_learner"),
        aggregation_strategies=("weighted_mean_proba",),
        optimize_metric="log_loss",
    )
    table = _candidate_table_for_inner(
        _inner_results(),
        _inner_predictions(),
        _configs(),
        plan,
        "log_loss",
        "outer0",
    )
    assert set(table["candidate_library_policy"]) == {"all_complete_inner_oof_mpmas"}
    assert set(table["candidate_library_size"].astype(int)) == {4}
