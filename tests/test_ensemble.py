import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mllabiome.configs_sweep import Ensemble
from mllabiome.ensemble_sweep import (
    _aggregate_proba,
    _candidate_table_for_inner,
    _eligible_config_ids,
    _ensemble_configs,
    _select_members,
    select_final_mpma_e_candidate,
    select_mpma_e_by_outer_fold,
    summarize_mpma_e_strategy,
    sweep_ensemble,
)


def _configs():
    return pd.DataFrame(
        [
            {
                "config_id": "A",
                "learner": "RF_A",
                "resolution": "genus",
                "count_transformation": "identity",
                "active": 1,
            },
            {
                "config_id": "B",
                "learner": "LR_B",
                "resolution": "species",
                "count_transformation": "relative_abundance",
                "active": 1,
            },
            {
                "config_id": "C",
                "learner": "RF_C",
                "resolution": "family",
                "count_transformation": "hellinger",
                "active": 1,
            },
            {
                "config_id": "S",
                "learner": "SIAMCAT",
                "resolution": "raw",
                "count_transformation": "identity",
                "active": 1,
            },
        ]
    )


def _plan(
    sizes=(2,),
    selection_strategies=("top_k",),
    aggregation_strategies=("mean_proba", "weighted_mean_proba"),
    **kwargs,
):
    return Ensemble(
        sizes=tuple(sizes),
        selection_strategies=tuple(selection_strategies),
        aggregation_strategies=tuple(aggregation_strategies),
        optimize_metric="nMCC",
        **kwargs,
    )


def _inner_results(scores):
    rows = []
    for outer_key, config_scores in scores.items():
        for cid, values in config_scores.items():
            for inner_no, value in enumerate(values):
                rows.append(
                    {
                        "stage": "inner",
                        "split_key": outer_key,
                        "inner_key": f"{outer_key}__i{inner_no}",
                        "config_id": cid,
                        "ok": 1,
                        "nMCC": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _prediction_rows(outer_key, cid, p1_values, y_values, inner=True):
    rows = []
    for i, (p1, y) in enumerate(zip(p1_values, y_values)):
        inner_no = i // 2
        row = {
            "stage": "inner" if inner else "outer",
            "split_key": f"{outer_key}__i{inner_no}" if inner else outer_key,
            "outer_split_key": outer_key,
            "sample_id": f"{outer_key}_{'in' if inner else 'out'}_{i}",
            "sample_index": i,
            "config_id": cid,
            "y_true": int(y),
            "y_pred": int(float(p1) >= 0.5),
            "proba_control": 1.0 - float(p1),
            "proba_case": float(p1),
        }
        rows.append(row)
    return rows


def _inner_predictions_by_fold(specs):
    rows = []
    for outer_key, config_data in specs.items():
        y = config_data["y"]
        for cid, p1 in config_data["p"].items():
            rows.extend(_prediction_rows(outer_key, cid, p1, y, inner=True))
    return pd.DataFrame(rows)


def _outer_predictions_by_fold(specs):
    rows = []
    for outer_key, config_data in specs.items():
        y = config_data["y"]
        for cid, p1 in config_data["p"].items():
            rows.extend(_prediction_rows(outer_key, cid, p1, y, inner=False))
    return pd.DataFrame(rows)


def _weighted_inner_fixture():
    inner_results = _inner_results(
        {
            "o0": {
                "A": [0.90, 0.90],
                "B": [0.60, 0.60],
            }
        }
    )
    y = [0, 1, 0, 1]
    inner_predictions = _inner_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.30, 0.70, 0.30, 0.70],
                    "B": [0.90, 0.10, 0.90, 0.10],
                },
            }
        }
    )
    return inner_results, inner_predictions


def test_ensemble_config_grid_is_exact_and_deterministic():
    plan = _plan(
        sizes=(2, 3),
        selection_strategies=("top_k", "best_per_family"),
        aggregation_strategies=("mean_proba", "weighted_mean_proba"),
    )
    first = _ensemble_configs(plan)
    second = _ensemble_configs(plan)
    assert first == second
    assert len(first) == 8
    assert len({row["ensemble_config_id"] for row in first}) == 8


def test_excluded_comparator_and_inactive_configs_are_not_eligible():
    configs = _configs()
    configs.loc[configs["config_id"].eq("C"), "active"] = 0
    plan = _plan(exclude_learners=("SIAMCAT",))
    assert _eligible_config_ids(configs, plan) == {"A", "B"}


def test_top_k_member_selection_uses_inner_score_order():
    scores = pd.Series({"A": 0.9, "B": 0.8, "C": 0.7})
    spec = {
        "ensemble_size": 2,
        "selection_strategy": "top_k",
    }
    assert _select_members(scores, _configs(), spec, _plan()) == ["A", "B"]


def test_best_per_family_and_best_per_resolution_enforce_diversity():
    scores = pd.Series({"A": 0.9, "C": 0.85, "B": 0.8})
    family = _select_members(
        scores,
        _configs(),
        {"ensemble_size": 2, "selection_strategy": "best_per_family"},
        _plan(),
    )
    resolution = _select_members(
        scores,
        _configs(),
        {"ensemble_size": 2, "selection_strategy": "best_per_resolution"},
        _plan(),
    )
    assert family == ["A", "B"]
    assert resolution == ["A", "C"]


def test_weighted_mean_uses_inner_member_scores_not_outer_performance():
    stack = np.array(
        [
            [[0.7, 0.3], [0.3, 0.7]],
            [[0.1, 0.9], [0.9, 0.1]],
        ],
        dtype=float,
    )
    weighted = _aggregate_proba(stack, np.array([0.9, 0.6]), "weighted_mean_proba")
    np.testing.assert_allclose(weighted, stack[0], atol=1e-6)


def test_candidate_ranking_is_computed_from_inner_oof_predictions():
    inner_results, inner_predictions = _weighted_inner_fixture()
    candidates = _candidate_table_for_inner(
        inner_results,
        inner_predictions,
        _configs(),
        _plan(),
        "nMCC",
        "o0",
        {"A", "B"},
    )
    assert not candidates.empty
    assert candidates.iloc[0]["aggregation_strategy"] == "weighted_mean_proba"
    assert float(candidates.iloc[0]["nMCC_mean"]) == pytest.approx(1.0)


def test_outer_predictions_cannot_change_selected_ensemble_specification():
    inner_results, inner_predictions = _weighted_inner_fixture()
    y = [0, 1, 0, 1]
    outer_a = _outer_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.30, 0.70, 0.30, 0.70],
                    "B": [0.90, 0.10, 0.90, 0.10],
                },
            }
        }
    )
    outer_b = _outer_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.90, 0.10, 0.90, 0.10],
                    "B": [0.10, 0.90, 0.10, 0.90],
                },
            }
        }
    )
    sel_a, pred_a, met_a = select_mpma_e_by_outer_fold(
        inner_results,
        inner_predictions,
        outer_a,
        _configs(),
        _plan(),
        "nMCC",
    )
    sel_b, pred_b, met_b = select_mpma_e_by_outer_fold(
        inner_results,
        inner_predictions,
        outer_b,
        _configs(),
        _plan(),
        "nMCC",
    )
    pd.testing.assert_frame_equal(sel_a, sel_b, check_dtype=False)
    assert sel_a.iloc[0]["aggregation_strategy"] == "weighted_mean_proba"
    assert not np.allclose(
        met_a["nMCC"].to_numpy(dtype=float),
        met_b["nMCC"].to_numpy(dtype=float),
    )


def test_member_set_cannot_change_when_only_outer_member_performance_changes():
    inner_results, inner_predictions = _weighted_inner_fixture()
    y = [0, 1, 0, 1]
    outer = _outer_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.90, 0.10, 0.90, 0.10],
                    "B": [0.10, 0.90, 0.10, 0.90],
                    "C": [0.10, 0.90, 0.10, 0.90],
                },
            }
        }
    )
    configs = _configs()
    inner_results = pd.concat(
        [
            inner_results,
            _inner_results({"o0": {"C": [0.50, 0.50]}}),
        ],
        ignore_index=True,
    )
    inner_predictions = pd.concat(
        [
            inner_predictions,
            _inner_predictions_by_fold(
                {
                    "o0": {
                        "y": y,
                        "p": {"C": [0.10, 0.90, 0.10, 0.90]},
                    }
                }
            ),
        ],
        ignore_index=True,
    )
    selection, _, _ = select_mpma_e_by_outer_fold(
        inner_results,
        inner_predictions,
        outer,
        configs,
        _plan(aggregation_strategies=("mean_proba",)),
        "nMCC",
    )
    members = json.loads(selection.iloc[0]["members"])
    assert members == ["A", "B"]
    assert "C" not in members


def test_different_outer_folds_may_select_different_ensemble_specifications():
    inner_results = _inner_results(
        {
            "o0": {"A": [0.9, 0.9], "B": [0.6, 0.6]},
            "o1": {"A": [0.9, 0.9], "B": [0.6, 0.6]},
        }
    )
    y = [0, 1, 0, 1]
    inner_predictions = _inner_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.30, 0.70, 0.30, 0.70],
                    "B": [0.90, 0.10, 0.90, 0.10],
                },
            },
            "o1": {
                "y": y,
                "p": {
                    "A": [0.70, 0.30, 0.70, 0.30],
                    "B": [0.01, 0.99, 0.01, 0.99],
                },
            },
        }
    )
    outer = _outer_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.30, 0.70, 0.30, 0.70],
                    "B": [0.90, 0.10, 0.90, 0.10],
                },
            },
            "o1": {
                "y": y,
                "p": {
                    "A": [0.70, 0.30, 0.70, 0.30],
                    "B": [0.01, 0.99, 0.01, 0.99],
                },
            },
        }
    )
    selection, _, _ = select_mpma_e_by_outer_fold(
        inner_results,
        inner_predictions,
        outer,
        _configs(),
        _plan(),
        "nMCC",
    )
    observed = dict(
        zip(selection["outer_split_key"], selection["aggregation_strategy"])
    )
    assert observed["o0"] == "weighted_mean_proba"
    assert observed["o1"] == "mean_proba"


def test_incomplete_inner_member_predictions_make_candidate_invalid():
    inner_results, inner_predictions = _weighted_inner_fixture()
    inner_predictions = inner_predictions[
        ~(
            inner_predictions["config_id"].eq("B")
            & inner_predictions["sample_id"].eq("o0_in_3")
        )
    ].reset_index(drop=True)
    candidates = _candidate_table_for_inner(
        inner_results,
        inner_predictions,
        _configs(),
        _plan(),
        "nMCC",
        "o0",
        {"A", "B"},
    )
    assert candidates.empty


def test_failed_or_incomplete_inner_metric_history_excludes_member_from_ranking():
    inner_results, inner_predictions = _weighted_inner_fixture()
    mask = inner_results["config_id"].eq("B") & inner_results["inner_key"].eq("o0__i1")
    inner_results.loc[mask, "ok"] = 0
    inner_results.loc[mask, "nMCC"] = np.nan
    candidates = _candidate_table_for_inner(
        inner_results,
        inner_predictions,
        _configs(),
        _plan(),
        "nMCC",
        "o0",
        {"A", "B"},
    )
    assert candidates.empty


def test_siamcat_is_excluded_from_mpma_e_even_when_its_inner_score_is_highest():
    inner_results = _inner_results(
        {
            "o0": {
                "A": [0.8, 0.8],
                "B": [0.7, 0.7],
                "S": [0.99, 0.99],
            }
        }
    )
    y = [0, 1, 0, 1]
    inner_predictions = _inner_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.2, 0.8, 0.2, 0.8],
                    "B": [0.3, 0.7, 0.3, 0.7],
                    "S": [0.1, 0.9, 0.1, 0.9],
                },
            }
        }
    )
    plan = _plan(
        aggregation_strategies=("mean_proba",),
        exclude_learners=("SIAMCAT",),
    )
    candidates = _candidate_table_for_inner(
        inner_results,
        inner_predictions,
        _configs(),
        plan,
        "nMCC",
        "o0",
        {"A", "B", "S"},
    )
    assert not candidates.empty
    assert all("S" not in json.loads(x) for x in candidates["members"])


def test_final_ensemble_candidate_is_selected_from_inner_predictions_only():
    inner_results, inner_predictions = _weighted_inner_fixture()
    final, candidates = select_final_mpma_e_candidate(
        inner_results,
        inner_predictions,
        _configs(),
        _plan(),
        "nMCC",
    )
    assert not candidates.empty
    assert final["aggregation_strategy"] == "weighted_mean_proba"
    assert (
        final["selection_basis"] == "all_inner_validation_predictions_for_final_refit"
    )
    assert final["inner_score"] == pytest.approx(1.0)
    assert not any(str(key).startswith("outer_") for key in final)


def test_nested_strategy_summary_aggregates_only_selected_outer_fold_metrics():
    fold_metrics = pd.DataFrame(
        [
            {
                "outer_split_key": "o0",
                "ensemble_config_id": "e1",
                "nMCC": 0.8,
                "AUC": 0.9,
            },
            {
                "outer_split_key": "o1",
                "ensemble_config_id": "e2",
                "nMCC": 0.6,
                "AUC": 0.7,
            },
        ]
    )
    summary = summarize_mpma_e_strategy(fold_metrics)
    assert summary["Strategy"] == "MPMA-E"
    assert summary["selection_basis"] == "outer_fold_inner_validation_predictions"
    assert summary["n_outer_folds"] == 2
    assert summary["outer_nMCC_mean"] == pytest.approx(0.7)
    assert summary["outer_nMCC_count"] == 2


def test_sweep_outputs_nested_and_final_ensemble_separately(tmp_path, monkeypatch):
    root = tmp_path / "run"
    for rel in (
        "predictions",
        "inner_results",
        "inner_predictions",
        "ensembling",
        "tables",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    inner_results, inner_predictions = _weighted_inner_fixture()
    y = [0, 1, 0, 1]
    outer = _outer_predictions_by_fold(
        {
            "o0": {
                "y": y,
                "p": {
                    "A": [0.90, 0.10, 0.90, 0.10],
                    "B": [0.10, 0.90, 0.10, 0.90],
                },
            }
        }
    )
    inner_results.to_csv(
        root / "inner_results" / "inner_results.tsv", sep="\t", index=False
    )
    inner_predictions.to_csv(
        root / "inner_predictions" / "inner_predictions.tsv", sep="\t", index=False
    )
    outer.to_csv(root / "predictions" / "outer_predictions.tsv", sep="\t", index=False)
    _configs().query("config_id in ['A', 'B']").to_csv(
        root / "configs.tsv", sep="\t", index=False
    )

    class DummySweep:
        def __init__(self):
            self.ensemble = _plan()
            self.evaluation = SimpleNamespace(optimize_metric="nMCC", random_state=42)
            self.title = "ensemble test"
            self.data = SimpleNamespace()

        def root(self):
            return root

    import mllabiome.ensemble_sweep as es

    monkeypatch.setattr(
        es,
        "_matrix_for_mpma_e_figure",
        lambda sweep: (np.ones((2, 2), dtype=float), ["a", "b"], "test"),
    )
    monkeypatch.setattr(
        es, "write_single_task_mpma_e_figure", lambda *args, **kwargs: {}
    )
    outputs = sweep_ensemble(DummySweep())
    assert outputs["mpma_e_selection"].exists()
    assert outputs["mpma_e_predictions"].exists()
    assert outputs["mpma_e_outer_results"].exists()
    assert outputs["mpma_e_candidates"].exists()
    assert outputs["mpma_e_summary"].exists()
    assert outputs["mpma_e_final_candidate"].exists()
    selection = pd.read_csv(outputs["mpma_e_selection"], sep="\t")
    assert selection.iloc[0]["aggregation_strategy"] == "weighted_mean_proba"
    nested = json.loads(outputs["mpma_e_summary"].read_text(encoding="utf-8"))
    final = json.loads(outputs["mpma_e_final_candidate"].read_text(encoding="utf-8"))
    assert nested["selection_basis"] == "outer_fold_inner_validation_predictions"
    assert (
        final["selection_basis"] == "all_inner_validation_predictions_for_final_refit"
    )
    selected_unit = json.loads(outputs["selected_unit"].read_text(encoding="utf-8"))
    assert "nested_mpma_e" in selected_unit
