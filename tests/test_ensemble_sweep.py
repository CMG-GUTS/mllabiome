import json
import numpy as np
import pandas as pd
import pytest
from mllabiome import ensemble_sweep
from mllabiome.configs_sweep import Ensemble
from mllabiome.ensemble_sweep import (
    _caruana_select,
    _ensemble_configs,
    _fit_super_learner,
    _metric_value,
    _weighted_probability_mean,
    select_mpma_e_by_outer_fold,
)


def _configs():
    return pd.DataFrame(
        [
            {
                "config_id": "A",
                "resolution": "species",
                "learner": "RF",
                "count_transformation": "relative_abundance",
                "active": 1,
            },
            {
                "config_id": "B",
                "resolution": "genus",
                "learner": "LR",
                "count_transformation": "centered_log_ratio_multiplicative_replacement",
                "active": 1,
            },
            {
                "config_id": "C",
                "resolution": "family",
                "learner": "BNB",
                "count_transformation": "presence_absence",
                "active": 1,
            },
        ]
    )


def _inner_results():
    rows = []
    scores = {"A": (0.9, 0.88), "B": (0.82, 0.8), "C": (0.6, 0.58)}
    for outer in ("o0", "o1"):
        for j in range(2):
            for cid, values in scores.items():
                rows.append(
                    {
                        "split_key": outer,
                        "inner_key": f"{outer}__i{j}",
                        "config_id": cid,
                        "nMCC": values[j],
                        "ok": 1,
                    }
                )
    return pd.DataFrame(rows)


def _append_prediction_rows(rows, outer, inner_key, sample_prefix, cid, y, p1):
    for i, (truth, positive) in enumerate(zip(y, p1)):
        rows.append(
            {
                "outer_split_key": outer,
                "split_key": inner_key,
                "sample_id": f"{sample_prefix}{i}",
                "config_id": cid,
                "y_true": int(truth),
                "y_pred": int(positive >= 0.5),
                "proba_0": float(1.0 - positive),
                "proba_1": float(positive),
            }
        )


def _inner_predictions():
    rows = []
    y0 = np.array([0, 1, 0, 1], dtype=int)
    y1 = np.array([1, 0, 1, 0], dtype=int)
    predictions = {
        "A": ([0.1, 0.8, 0.2, 0.78], [0.82, 0.25, 0.76, 0.2]),
        "B": ([0.2, 0.72, 0.35, 0.88], [0.75, 0.3, 0.68, 0.22]),
        "C": ([0.55, 0.45, 0.6, 0.4], [0.45, 0.55, 0.4, 0.6]),
    }
    for outer in ("o0", "o1"):
        for cid, (p0, p1) in predictions.items():
            _append_prediction_rows(
                rows, outer, f"{outer}__i0", f"{outer}_a_", cid, y0, p0
            )
            _append_prediction_rows(
                rows, outer, f"{outer}__i1", f"{outer}_b_", cid, y1, p1
            )
    return pd.DataFrame(rows)


def _outer_predictions():
    rows = []
    for outer, y in (("o0", [0, 1, 0, 1]), ("o1", [1, 0, 1, 0])):
        probs = {
            "A": [0.15, 0.82, 0.22, 0.77]
            if outer == "o0"
            else [0.81, 0.22, 0.75, 0.18],
            "B": [0.25, 0.74, 0.3, 0.85] if outer == "o0" else [0.73, 0.28, 0.7, 0.24],
            "C": [0.55, 0.45, 0.6, 0.4] if outer == "o0" else [0.45, 0.55, 0.4, 0.6],
        }
        for cid, p1 in probs.items():
            for i, (truth, positive) in enumerate(zip(y, p1)):
                rows.append(
                    {
                        "outer_split_key": outer,
                        "split_key": outer,
                        "sample_id": f"{outer}_test_{i}",
                        "config_id": cid,
                        "y_true": int(truth),
                        "y_pred": int(positive >= 0.5),
                        "proba_0": float(1.0 - positive),
                        "proba_1": float(positive),
                    }
                )
    return pd.DataFrame(rows)


def test_default_ensemble_space_contains_only_publication_methods():
    plan = Ensemble(
        selection_strategies=(
            "top_k",
            "best_per_resolution",
            "best_per_learner_type",
            "caruana",
            "super_learner",
        ),
        aggregation_strategies=("mean_proba",),
    )
    expected = {
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    }
    configs = _ensemble_configs(plan)
    assert {x["selection_strategy"] for x in configs} == expected


def test_unknown_ensemble_method_fails_loudly():
    with pytest.raises(ValueError, match="Unknown ensemble method"):
        _ensemble_configs(
            Ensemble(
                selection_strategies=("topk_typo",),
                aggregation_strategies=("mean_proba",),
            )
        )


def test_weighted_probability_mean_matches_hand_calculation_and_is_batch_invariant():
    stack = np.array([[[0.8, 0.2], [0.1, 0.9]], [[0.6, 0.4], [0.3, 0.7]]], dtype=float)
    weights = np.array([0.75, 0.25])
    result = _weighted_probability_mean(stack, weights)
    expected = np.array([[0.75, 0.25], [0.15, 0.85]])
    np.testing.assert_allclose(result, expected)
    alone = _weighted_probability_mean(stack[:, :1, :], weights)[0]
    np.testing.assert_allclose(alone, result[0])


def test_caruana_selection_learns_nonnegative_normalized_frequency_weights():
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    perfect = np.column_stack([1 - y * 0.9 - (1 - y) * 0.1, y * 0.9 + (1 - y) * 0.1])
    complementary = np.array(
        [
            [0.8, 0.2],
            [0.25, 0.75],
            [0.65, 0.35],
            [0.35, 0.65],
            [0.75, 0.25],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    poor = np.full((len(y), 2), 0.5, dtype=float)
    stack = np.stack([perfect, complementary, poor], axis=0)
    members, weights, score, trajectory = _caruana_select(
        ["A", "B", "C"], stack, y, "nMCC", max_iterations=8
    )
    assert members
    assert len(members) == len(weights)
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.isfinite(score)
    assert trajectory


def test_super_learner_weights_are_simplex_constrained_and_prefer_strong_model():
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    strong = np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.92, 0.08],
            [0.08, 0.92],
        ],
        dtype=float,
    )
    weak = np.full((len(y), 2), 0.5, dtype=float)
    misleading = strong[:, ::-1]
    stack = np.stack([strong, weak, misleading], axis=0)
    members, weights, loss = _fit_super_learner(
        ["strong", "weak", "bad"], stack, y, "log_loss", 1e-08
    )
    assert len(members) == len(weights)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)
    assert "strong" in members
    strong_weight = weights[members.index("strong")]
    assert strong_weight > 0.9
    assert np.isfinite(loss)


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
    plan = Ensemble(
        selection_strategies=(method,),
        aggregation_strategies=("mean_proba",),
        sizes=(2,),
        optimize_metric="nMCC",
    )
    inner_results = _inner_results()
    inner_predictions = _inner_predictions()
    outer = _outer_predictions()
    selected_a, _, _ = select_mpma_e_by_outer_fold(
        inner_results, inner_predictions, outer, _configs(), plan, "nMCC"
    )
    corrupted_outer = outer.copy()
    corrupted_outer["y_true"] = 1 - corrupted_outer["y_true"].astype(int)
    corrupted_outer[["proba_0", "proba_1"]] = corrupted_outer[
        ["proba_1", "proba_0"]
    ].to_numpy()
    selected_b, _, _ = select_mpma_e_by_outer_fold(
        inner_results, inner_predictions, corrupted_outer, _configs(), plan, "nMCC"
    )
    cols = [
        "outer_split_key",
        "ensemble_config_id",
        "selection_strategy",
        "members",
        "weights",
        "inner_score",
    ]
    pd.testing.assert_frame_equal(
        selected_a[cols].reset_index(drop=True), selected_b[cols].reset_index(drop=True)
    )


def test_missing_selected_outer_member_fails_instead_of_reselecting():
    plan = Ensemble(
        selection_strategies=("top_k",),
        aggregation_strategies=("mean_proba",),
        sizes=(2,),
        optimize_metric="nMCC",
    )
    outer = _outer_predictions()
    broken = outer[
        ~(outer["outer_split_key"].eq("o0") & outer["config_id"].eq("B"))
    ].copy()
    with pytest.raises(RuntimeError, match="not re-selected|missing"):
        select_mpma_e_by_outer_fold(
            _inner_results(), _inner_predictions(), broken, _configs(), plan, "nMCC"
        )


def test_every_inner_outer_key_must_produce_exactly_one_selection():
    plan = Ensemble(
        selection_strategies=("top_k",),
        aggregation_strategies=("mean_proba",),
        sizes=(2,),
        optimize_metric="nMCC",
    )
    selection, predictions, metrics = select_mpma_e_by_outer_fold(
        _inner_results(),
        _inner_predictions(),
        _outer_predictions(),
        _configs(),
        plan,
        "nMCC",
    )
    assert selection["outer_split_key"].tolist() == ["o0", "o1"]
    assert metrics["outer_split_key"].tolist() == ["o0", "o1"]
    assert set(predictions["outer_split_key"]) == {"o0", "o1"}
    for value in selection["weights"]:
        weights = np.asarray(json.loads(value), dtype=float)
        assert weights.sum() == pytest.approx(1.0)


def test_log_loss_metric_matches_manual_binary_cross_entropy():
    y = np.array([0, 1, 1, 0], dtype=int)
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.75, 0.25]], dtype=float)
    expected = -np.mean(np.log([0.9, 0.8, 0.7, 0.75]))
    assert _metric_value(y, proba, "log_loss") == pytest.approx(expected)


def test_caruana_log_loss_uses_lower_is_better():
    y = np.array([0, 1, 0, 1], dtype=int)
    strong = np.array([[0.95, 0.05], [0.05, 0.95], [0.9, 0.1], [0.1, 0.9]], dtype=float)
    weak = np.full((4, 2), 0.5, dtype=float)
    stack = np.stack([strong, weak], axis=0)
    members, weights, score, trajectory = _caruana_select(
        ["strong", "weak"], stack, y, "log_loss", max_iterations=4
    )
    assert members[0] == "strong"
    assert weights[0] > 0.5
    assert score == pytest.approx(min(trajectory))


def test_log_loss_can_select_ensemble_without_log_loss_column_in_inner_results():
    plan = Ensemble(
        selection_strategies=(
            "top_k",
            "best_per_resolution",
            "best_per_learner_type",
            "caruana",
            "super_learner",
        ),
        aggregation_strategies=("mean_proba",),
        sizes=(2,),
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
    assert not selection.empty
    assert not predictions.empty
    assert not metrics.empty
    assert set(selection["selection_metric"]) == {"log_loss"}
    assert np.isfinite(selection["inner_score"].to_numpy(dtype=float)).all()
    assert (selection["member_count"].to_numpy(dtype=int) > 0).all()


def test_best_per_learner_type_keeps_one_configuration_per_algorithm():
    scores = pd.Series([0.9, 0.8, 0.7, 0.6], index=["A", "B", "C", "D"])
    configs = pd.DataFrame(
        [
            {
                "config_id": "A",
                "resolution": "species",
                "learner": "RF_1000_msl5",
                "count_transformation": "relative_abundance",
                "active": 1,
            },
            {
                "config_id": "B",
                "resolution": "genus",
                "learner": "RF_fast",
                "count_transformation": "relative_abundance",
                "active": 1,
            },
            {
                "config_id": "C",
                "resolution": "family",
                "learner": "LR_L2",
                "count_transformation": "relative_abundance",
                "active": 1,
            },
            {
                "config_id": "D",
                "resolution": "order",
                "learner": "XGB_depth3",
                "count_transformation": "relative_abundance",
                "active": 1,
            },
        ]
    )
    members = ensemble_sweep._select_best_per_learner_type(scores, configs)
    assert members == ["A", "C", "D"]
    assert ensemble_sweep._learner_type("RF_1000_msl5") == "random_forest"
    assert ensemble_sweep._learner_type("RF_fast") == "random_forest"
    assert ensemble_sweep._learner_type("LR_L2") == "logistic_regression"
