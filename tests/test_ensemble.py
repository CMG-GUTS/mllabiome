from __future__ import annotations

import numpy as np
import pytest

from mllabiome.configs_sweep import Ensemble
from mllabiome.ensemble_aggregation import aggregate_member_predictions
from mllabiome.ensemble_sweep import (
    _caruana_select,
    _ensemble_configs,
    _fit_super_learner,
    _resolved_super_learner_loss,
)


def _binary_library(seed: int = 7, n_samples: int = 80, n_members: int = 9):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)
    stack = []
    for member in range(n_members):
        score = (0.18 + 0.035 * member) * (2 * y - 1) + rng.normal(
            scale=1.0, size=n_samples
        )
        p1 = 1.0 / (1.0 + np.exp(-score))
        stack.append(np.column_stack([1.0 - p1, p1]))
    return [f"m{index}" for index in range(n_members)], np.stack(stack), y


def test_default_ensemble_space_contains_only_valid_selection_aggregation_pairs():
    plan = Ensemble(
        max_sizes=(3,),
        selection_strategies=(
            "top_k",
            "best_per_resolution",
            "best_per_learner_type",
            "caruana",
            "super_learner",
        ),
        aggregation_strategies=(
            "mean_proba",
            "weighted_mean_proba",
            "median_proba",
            "rank_mean",
            "majority_vote",
        ),
    )
    rows = _ensemble_configs(plan)
    pairs = {(row["selection_strategy"], row["aggregation_strategy"]) for row in rows}
    expected = {
        ("top_k", "mean_proba"),
        ("top_k", "median_proba"),
        ("top_k", "rank_mean"),
        ("top_k", "majority_vote"),
        ("best_per_resolution", "mean_proba"),
        ("best_per_resolution", "median_proba"),
        ("best_per_resolution", "rank_mean"),
        ("best_per_resolution", "majority_vote"),
        ("best_per_learner_type", "mean_proba"),
        ("best_per_learner_type", "median_proba"),
        ("best_per_learner_type", "rank_mean"),
        ("best_per_learner_type", "majority_vote"),
        ("caruana", "weighted_mean_proba"),
        ("super_learner", "weighted_mean_proba"),
    }
    assert pairs == expected


def test_learned_selectors_require_weighted_mean_proba():
    plan = Ensemble(
        max_sizes=(3,),
        selection_strategies=("caruana", "super_learner"),
        aggregation_strategies=("mean_proba",),
    )
    with pytest.raises(ValueError, match="weighted_mean_proba"):
        _ensemble_configs(plan)


def test_simple_selectors_require_nonweighted_aggregation():
    plan = Ensemble(
        max_sizes=(3,),
        selection_strategies=("top_k",),
        aggregation_strategies=("weighted_mean_proba",),
    )
    with pytest.raises(ValueError, match="non-weighted aggregation"):
        _ensemble_configs(plan)


def test_unknown_ensemble_method_fails_loudly():
    with pytest.raises(ValueError, match="Unknown ensemble selection strategy"):
        _ensemble_configs(
            Ensemble(
                selection_strategies=("topk_typo",),
                aggregation_strategies=("mean_proba",),
            )
        )


def test_legacy_sizes_alias_matches_max_sizes():
    canonical = Ensemble(
        max_sizes=(6,),
        selection_strategies=("top_k",),
        aggregation_strategies=("mean_proba",),
    )
    legacy = Ensemble(
        sizes=(6,),
        selection_strategies=("top_k",),
        aggregation_strategies=("mean_proba",),
    )
    assert _ensemble_configs(canonical) == _ensemble_configs(legacy)


def test_numerical_controls_are_not_public_ensemble_options():
    for key, value in (
        ("learned_library_size", 50),
        ("caruana_max_iterations", 25),
        ("super_learner_weight_tol", 1e-6),
        ("super_learner_loss", "log_loss"),
    ):
        with pytest.raises(TypeError):
            Ensemble(**{key: value})


def test_super_learner_loss_is_automatic():
    common = dict(
        max_sizes=(3,),
        selection_strategies=("super_learner",),
        aggregation_strategies=("weighted_mean_proba",),
    )
    assert (
        _resolved_super_learner_loss(Ensemble(optimize_metric="brier", **common))
        == "brier"
    )
    assert (
        _resolved_super_learner_loss(Ensemble(optimize_metric="log_loss", **common))
        == "log_loss"
    )
    assert (
        _resolved_super_learner_loss(Ensemble(optimize_metric="nMCC", **common))
        == "log_loss"
    )


@pytest.mark.parametrize(
    "aggregation,weights",
    [
        ("mean_proba", None),
        ("weighted_mean_proba", [0.25, 0.75]),
        ("median_proba", None),
        ("rank_mean", None),
        ("majority_vote", None),
    ],
)
def test_public_aggregations_return_valid_probability_matrices(aggregation, weights):
    stack = np.asarray(
        [
            [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]],
            [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]],
        ],
        dtype=float,
    )
    result = aggregate_member_predictions(stack, aggregation, weights)
    assert result.shape == (3, 2)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)


def test_rank_mean_is_batch_invariant():
    single = np.asarray(
        [
            [[0.7, 0.2, 0.1]],
            [[0.2, 0.6, 0.2]],
        ],
        dtype=float,
    )
    batch = np.asarray(
        [
            [[0.7, 0.2, 0.1], [0.05, 0.15, 0.80]],
            [[0.2, 0.6, 0.2], [0.70, 0.20, 0.10]],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        aggregate_member_predictions(single, "rank_mean")[0],
        aggregate_member_predictions(batch, "rank_mean")[0],
    )


def test_caruana_selection_learns_nonnegative_normalized_frequency_weights():
    member_ids, stack, y = _binary_library(seed=17)
    members, weights, score, trajectory, diagnostics = _caruana_select(
        member_ids, stack, y, "log_loss", 4
    )
    assert 1 <= len(members) <= 4
    assert len(weights) == len(members)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.isfinite(score)
    assert len(trajectory) == diagnostics["caruana_iterations_selected"]
    assert diagnostics["caruana_iterations_executed"] >= len(trajectory)
    assert diagnostics["caruana_stop_reason"] in {
        "converged_patience",
        "iteration_ceiling",
        "no_candidates",
    }


def test_caruana_log_loss_uses_lower_is_better():
    y = np.asarray([0, 1, 0, 1], dtype=int)
    strong = np.asarray(
        [[0.95, 0.05], [0.05, 0.95], [0.9, 0.1], [0.1, 0.9]], dtype=float
    )
    weak = np.full((4, 2), 0.5, dtype=float)
    members, weights, score, trajectory, _ = _caruana_select(
        ["strong", "weak"], np.stack([strong, weak]), y, "log_loss", 2
    )
    assert members[0] == "strong"
    assert weights[0] > 0.5
    assert score <= trajectory[0] + 1e-12


def test_super_learner_weights_are_simplex_constrained_and_prefer_strong_model():
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
    strong = np.asarray(
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
    members, weights, loss = _fit_super_learner(
        ["strong", "weak", "bad"],
        np.stack([strong, weak, misleading]),
        y,
        "log_loss",
        3,
    )
    assert 2 <= len(members) <= 3
    assert len(weights) == len(members)
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.isfinite(loss)
    assert weights[members.index("strong")] == pytest.approx(max(weights), abs=1e-6)


def test_super_learner_respects_max_members():
    member_ids, stack, y = _binary_library(seed=23)
    members, weights, _ = _fit_super_learner(member_ids, stack, y, "log_loss", 4)
    assert 2 <= len(members) <= 4
    assert len(weights) == len(members)
    np.testing.assert_allclose(weights.sum(), 1.0)
