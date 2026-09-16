from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mllabiome.configs_sweep import Ensemble
from mllabiome.ensemble_aggregation import aggregate_member_predictions
from mllabiome.ensemble_sweep import (
    _candidate_from_caruana,
    _caruana_select,
    _ensemble_configs,
    _fit_super_learner,
    _resolved_super_learner_loss,
)


def test_learned_selectors_require_weighted_mean() -> None:
    plan = Ensemble(
        max_sizes=(6,),
        selection_strategies=("top_k", "caruana", "super_learner"),
        aggregation_strategies=("mean_proba",),
    )
    with pytest.raises(ValueError, match="weighted_mean_proba"):
        _ensemble_configs(plan)


def test_compatibility_matrix_is_explicit() -> None:
    plan = Ensemble(
        max_sizes=(6,),
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
        ),
    )
    pairs = {
        (row["selection_strategy"], row["aggregation_strategy"])
        for row in _ensemble_configs(plan)
    }
    assert ("top_k", "mean_proba") in pairs
    assert ("top_k", "median_proba") in pairs
    assert ("top_k", "weighted_mean_proba") not in pairs
    assert ("caruana", "weighted_mean_proba") in pairs
    assert ("caruana", "mean_proba") not in pairs
    assert ("super_learner", "weighted_mean_proba") in pairs
    assert ("super_learner", "median_proba") not in pairs


def test_legacy_sizes_alias_matches_max_sizes() -> None:
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


def test_numerical_controls_are_not_public_ensemble_options() -> None:
    for key, value in (
        ("learned_library_size", 50),
        ("caruana_max_iterations", 25),
        ("super_learner_weight_tol", 1e-6),
        ("super_learner_loss", "log_loss"),
    ):
        with pytest.raises(TypeError):
            Ensemble(**{key: value})


def test_super_learner_loss_is_automatic() -> None:
    common = dict(
        max_sizes=(6,),
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


def test_rank_mean_is_batch_invariant() -> None:

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
    a = aggregate_member_predictions(single, "rank_mean")[0]
    b = aggregate_member_predictions(batch, "rank_mean")[0]
    np.testing.assert_allclose(a, b)


def _synthetic_library(seed: int = 7, n_samples: int = 80, n_members: int = 9):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_samples)
    stack = []
    for member in range(n_members):
        score = (0.2 + 0.04 * member) * (2 * y - 1) + rng.normal(
            scale=1.0, size=n_samples
        )
        p1 = 1.0 / (1.0 + np.exp(-score))
        stack.append(np.column_stack([1.0 - p1, p1]))
    member_ids = [f"m{index}" for index in range(n_members)]
    return member_ids, np.stack(stack, axis=0), y


def test_super_learner_respects_max_members() -> None:
    member_ids, stack, y = _synthetic_library()
    selected, weights, _ = _fit_super_learner(
        member_ids,
        stack,
        y,
        "log_loss",
        4,
    )
    assert 2 <= len(selected) <= 4
    assert len(weights) == len(selected)
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_caruana_respects_max_members_and_records_automatic_stopping() -> None:
    member_ids, stack, y = _synthetic_library(seed=17)
    selected, weights, _, trajectory, diagnostics = _caruana_select(
        member_ids,
        stack,
        y,
        "log_loss",
        4,
    )
    assert 1 <= len(selected) <= 4
    assert len(weights) == len(selected)
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert diagnostics["caruana_iterations_executed"] >= len(trajectory)
    assert diagnostics["caruana_iterations_selected"] == len(trajectory)
    assert diagnostics["caruana_stop_reason"] in {
        "converged_patience",
        "iteration_ceiling",
        "no_candidates",
    }


def test_caruana_uses_full_eligible_library_without_top50_prescreen() -> None:
    rng = np.random.default_rng(23)
    n_members = 55
    n_samples = 36
    y = rng.integers(0, 2, size=n_samples)
    rows = []
    member_ids = [f"m{i:02d}" for i in range(n_members)]
    for j, member_id in enumerate(member_ids):
        score = (0.1 + 0.002 * j) * (2 * y - 1) + rng.normal(size=n_samples)
        p1 = 1.0 / (1.0 + np.exp(-score))
        for i in range(n_samples):
            rows.append(
                {
                    "outer_split_key": "outer0",
                    "split_key": "inner0",
                    "sample_id": f"s{i:03d}",
                    "y_true": int(y[i]),
                    "config_id": member_id,
                    "proba_0": float(1.0 - p1[i]),
                    "proba_1": float(p1[i]),
                }
            )
    scores = pd.Series(
        np.linspace(0.1, 0.9, n_members),
        index=member_ids,
        dtype=float,
    )
    spec = {
        "ensemble_config_id": "test",
        "search_schema": "mpmae_search_v3",
        "optimize_metric": "log_loss",
        "selection_strategy": "caruana",
        "aggregation_strategy": "weighted_mean_proba",
        "max_size": 4,
    }
    candidate = _candidate_from_caruana(
        spec, scores, pd.DataFrame(rows), ["proba_0", "proba_1"], "log_loss"
    )
    assert candidate is not None
    assert candidate["candidate_library_size"] == n_members
    assert candidate["candidate_library_policy"] == "all_complete_inner_oof_mpmas"
