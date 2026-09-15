import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mllabiome.selection import (_eligible_config_ids, mpma_b_fold_metrics,
                                 select_final_mpma_candidate,
                                 select_mpma_b_by_outer_fold,
                                 selected_mpma_b_outer_predictions,
                                 summarize_mpma_b_strategy,
                                 write_mpma_b_selection_outputs)


def _configs():
    return pd.DataFrame(
        [
            {
                "config_id": "A",
                "mpdr_id": "mpdr_a",
                "count_transformation": "identity",
                "resolution": "genus",
                "levels": "genus",
                "learner": "ModelA",
                "active": 1,
            },
            {
                "config_id": "B",
                "mpdr_id": "mpdr_b",
                "count_transformation": "relative_abundance",
                "resolution": "species",
                "levels": "species",
                "learner": "ModelB",
                "active": 1,
            },
            {
                "config_id": "S",
                "mpdr_id": "mpdr_s",
                "count_transformation": "identity",
                "resolution": "raw",
                "levels": "all",
                "learner": "SIAMCAT",
                "active": 1,
            },
        ]
    )


def _inner_rows(scores_by_split):
    rows = []
    for split_key, config_scores in scores_by_split.items():
        for config_id, scores in config_scores.items():
            for inner_no, score in enumerate(scores):
                rows.append(
                    {
                        "stage": "inner",
                        "split_key": split_key,
                        "inner_key": f"{split_key}__i{inner_no}",
                        "config_id": config_id,
                        "ok": 1,
                        "nMCC": float(score),
                    }
                )
    return pd.DataFrame(rows)


def _outer_predictions(predictions_by_split):
    rows = []
    sample_index = 0
    for split_key, config_predictions in predictions_by_split.items():
        for config_id, predictions in config_predictions.items():
            for y_true, p1 in predictions:
                p1 = float(p1)
                rows.append(
                    {
                        "stage": "outer",
                        "split_key": split_key,
                        "outer_split_key": split_key,
                        "sample_id": f"{split_key}_s{sample_index}",
                        "sample_index": sample_index,
                        "config_id": config_id,
                        "y_true": int(y_true),
                        "y_pred": int(p1 >= 0.5),
                        "proba_control": 1.0 - p1,
                        "proba_case": p1,
                    }
                )
                sample_index += 1
    return pd.DataFrame(rows)


def _plan(**kwargs):
    base = {
        "include_inactive": False,
        "exclude_config_ids": (),
        "exclude_learners": (),
        "exclude_resolutions": (),
        "exclude_transformations": (),
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_mpma_b_selection_uses_inner_validation_even_when_outer_performance_prefers_another_config():
    inner = _inner_rows(
        {
            "o0": {
                "A": [0.90, 0.90],
                "B": [0.70, 0.70],
            }
        }
    )
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    assert len(selection) == 1
    assert selection.iloc[0]["config_id"] == "A"
    outer = _outer_predictions(
        {
            "o0": {
                "A": [(0, 0.9), (1, 0.1), (0, 0.9), (1, 0.1)],
                "B": [(0, 0.1), (1, 0.9), (0, 0.1), (1, 0.9)],
            }
        }
    )
    selected = selected_mpma_b_outer_predictions(selection, outer)
    assert set(selected["config_id"]) == {"A"}
    metrics = mpma_b_fold_metrics(selected)
    assert float(metrics.iloc[0]["nMCC"]) < 0.5


def test_mpma_b_can_select_different_configs_on_different_outer_folds():
    inner = _inner_rows(
        {
            "o0": {"A": [0.90, 0.88], "B": [0.70, 0.72]},
            "o1": {"A": [0.60, 0.62], "B": [0.91, 0.89]},
            "o2": {"A": [0.85, 0.87], "B": [0.80, 0.79]},
        }
    )
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    observed = dict(zip(selection["outer_split_key"], selection["config_id"]))
    assert observed == {"o0": "A", "o1": "B", "o2": "A"}


def test_selected_outer_predictions_use_exactly_the_fold_specific_selected_config():
    inner = _inner_rows(
        {
            "o0": {"A": [0.9, 0.8], "B": [0.7, 0.6]},
            "o1": {"A": [0.6, 0.5], "B": [0.9, 0.8]},
        }
    )
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    outer = _outer_predictions(
        {
            "o0": {
                "A": [(0, 0.1), (1, 0.9)],
                "B": [(0, 0.9), (1, 0.1)],
            },
            "o1": {
                "A": [(0, 0.9), (1, 0.1)],
                "B": [(0, 0.1), (1, 0.9)],
            },
        }
    )
    selected = selected_mpma_b_outer_predictions(selection, outer)
    by_fold = {
        split: set(group["config_id"])
        for split, group in selected.groupby("outer_split_key")
    }
    assert by_fold == {"o0": {"A"}, "o1": {"B"}}
    assert len(selected) == 4


def test_changing_outer_predictions_cannot_change_mpma_b_selection():
    inner = _inner_rows(
        {
            "o0": {"A": [0.9, 0.8], "B": [0.7, 0.6]},
            "o1": {"A": [0.6, 0.5], "B": [0.9, 0.8]},
        }
    )
    first = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    outer_a = _outer_predictions(
        {
            "o0": {"A": [(0, 0.1), (1, 0.9)], "B": [(0, 0.9), (1, 0.1)]},
            "o1": {"A": [(0, 0.1), (1, 0.9)], "B": [(0, 0.9), (1, 0.1)]},
        }
    )
    outer_b = outer_a.copy()
    outer_b["proba_control"] = outer_a["proba_case"].to_numpy()
    outer_b["proba_case"] = outer_a["proba_control"].to_numpy()
    outer_b["y_pred"] = 1 - outer_a["y_pred"].to_numpy()
    second = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    metrics_a = mpma_b_fold_metrics(selected_mpma_b_outer_predictions(first, outer_a))
    metrics_b = mpma_b_fold_metrics(selected_mpma_b_outer_predictions(second, outer_b))
    assert not np.allclose(
        metrics_a["nMCC"].to_numpy(dtype=float),
        metrics_b["nMCC"].to_numpy(dtype=float),
    )


def test_incomplete_inner_cv_config_is_not_eligible_for_selection():
    inner = _inner_rows(
        {
            "o0": {
                "A": [0.80, 0.80, 0.80],
                "B": [0.99, 0.99, 0.99],
            }
        }
    )
    inner = inner[
        ~(inner["config_id"].eq("B") & inner["inner_key"].eq("o0__i2"))
    ].reset_index(drop=True)
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    assert selection.iloc[0]["config_id"] == "A"
    assert int(selection.iloc[0]["n_inner_folds"]) == 3


def test_failed_inner_fold_config_is_not_eligible_for_selection():
    inner = _inner_rows(
        {
            "o0": {
                "A": [0.80, 0.80, 0.80],
                "B": [0.99, 0.99, 0.99],
            }
        }
    )
    mask = inner["config_id"].eq("B") & inner["inner_key"].eq("o0__i1")
    inner.loc[mask, "ok"] = 0
    inner.loc[mask, "nMCC"] = np.nan
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    assert selection.iloc[0]["config_id"] == "A"


def test_ties_are_broken_deterministically_by_config_id():
    inner = _inner_rows(
        {
            "o0": {
                "B": [0.80, 0.80],
                "A": [0.80, 0.80],
            }
        }
    )
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    assert selection.iloc[0]["config_id"] == "A"


def test_comparator_exclusions_apply_before_mpma_b_selection():
    inner = _inner_rows(
        {
            "o0": {
                "A": [0.80, 0.80],
                "B": [0.70, 0.70],
                "S": [0.99, 0.99],
            }
        }
    )
    plan = _plan(exclude_learners=("SIAMCAT",))
    eligible = _eligible_config_ids(_configs(), plan)
    assert eligible == {"A", "B"}
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC", plan=plan)
    assert selection.iloc[0]["config_id"] == "A"


def test_inactive_configs_are_excluded_unless_explicitly_included():
    configs = _configs()
    configs.loc[configs["config_id"].eq("B"), "active"] = 0
    default_ids = _eligible_config_ids(configs, _plan())
    all_ids = _eligible_config_ids(configs, _plan(include_inactive=True))
    assert default_ids == {"A", "S"}
    assert all_ids == {"A", "B", "S"}


def test_qualification_gate_filters_selection_per_outer_fold():
    inner = _inner_rows(
        {
            "o0": {
                "A": [0.85, 0.85],
                "B": [0.90, 0.90],
            },
            "o1": {
                "A": [0.90, 0.90],
                "B": [0.85, 0.85],
            },
        }
    )
    qualification = pd.DataFrame(
        [
            {"split_key": "o0", "config_id": "A", "qualified": 1},
            {"split_key": "o0", "config_id": "B", "qualified": 0},
            {"split_key": "o1", "config_id": "A", "qualified": 0},
            {"split_key": "o1", "config_id": "B", "qualified": 1},
        ]
    )
    selection = select_mpma_b_by_outer_fold(
        inner,
        _configs(),
        "nMCC",
        qualification=qualification,
    )
    observed = dict(zip(selection["outer_split_key"], selection["config_id"]))
    assert observed == {"o0": "A", "o1": "B"}


def test_strategy_summary_is_based_on_fold_specific_selected_predictions():
    inner = _inner_rows(
        {
            "o0": {"A": [0.9, 0.9], "B": [0.7, 0.7]},
            "o1": {"A": [0.7, 0.7], "B": [0.9, 0.9]},
        }
    )
    selection = select_mpma_b_by_outer_fold(inner, _configs(), "nMCC")
    outer = _outer_predictions(
        {
            "o0": {
                "A": [(0, 0.1), (1, 0.9), (0, 0.1), (1, 0.9)],
                "B": [(0, 0.9), (1, 0.1), (0, 0.9), (1, 0.1)],
            },
            "o1": {
                "A": [(0, 0.9), (1, 0.1), (0, 0.9), (1, 0.1)],
                "B": [(0, 0.1), (1, 0.9), (0, 0.1), (1, 0.9)],
            },
        }
    )
    selected = selected_mpma_b_outer_predictions(selection, outer)
    fold_metrics = mpma_b_fold_metrics(selected)
    summary = summarize_mpma_b_strategy(fold_metrics)
    assert summary["Strategy"] == "MPMA-B"
    assert summary["selection_basis"] == "outer_fold_inner_validation"
    assert summary["n_outer_folds"] == 2
    assert summary["outer_nMCC_mean"] == pytest.approx(1.0)
    assert summary["outer_nMCC_count"] == 2


def test_global_final_candidate_is_inner_only_and_explicitly_not_nested_performance():
    inner = _inner_rows(
        {
            "o0": {"A": [0.95, 0.95], "B": [0.60, 0.60]},
            "o1": {"A": [0.55, 0.55], "B": [0.90, 0.90]},
            "o2": {"A": [0.95, 0.95], "B": [0.60, 0.60]},
        }
    )
    final_candidate = select_final_mpma_candidate(inner, _configs(), "nMCC")
    assert final_candidate["config_id"] == "A"
    assert final_candidate["selection_basis"] == "all_inner_validation_for_final_refit"
    assert final_candidate["selection_metric"] == "nMCC"
    assert not any(str(key).startswith("outer_") for key in final_candidate)


def test_global_final_candidate_excludes_incomplete_configs():
    inner = _inner_rows(
        {
            "o0": {"A": [0.70, 0.70], "B": [0.99, 0.99]},
            "o1": {"A": [0.70, 0.70], "B": [0.99, 0.99]},
        }
    )
    inner = inner[
        ~(inner["config_id"].eq("B") & inner["inner_key"].eq("o1__i1"))
    ].reset_index(drop=True)
    final_candidate = select_final_mpma_candidate(inner, _configs(), "nMCC")
    assert final_candidate["config_id"] == "A"


def test_write_outputs_keeps_nested_strategy_and_final_candidate_separate(tmp_path):
    root = tmp_path / "run"
    (root / "inner_results").mkdir(parents=True)
    (root / "predictions").mkdir(parents=True)
    (root / "tables").mkdir(parents=True)
    inner = _inner_rows(
        {
            "o0": {"A": [0.95, 0.95], "B": [0.60, 0.60]},
            "o1": {"A": [0.55, 0.55], "B": [0.90, 0.90]},
            "o2": {"A": [0.95, 0.95], "B": [0.60, 0.60]},
        }
    )
    outer = _outer_predictions(
        {
            "o0": {
                "A": [(0, 0.1), (1, 0.9)],
                "B": [(0, 0.9), (1, 0.1)],
            },
            "o1": {
                "A": [(0, 0.9), (1, 0.1)],
                "B": [(0, 0.1), (1, 0.9)],
            },
            "o2": {
                "A": [(0, 0.1), (1, 0.9)],
                "B": [(0, 0.9), (1, 0.1)],
            },
        }
    )
    inner.to_csv(root / "inner_results" / "inner_results.tsv", sep="\t", index=False)
    outer.to_csv(root / "predictions" / "outer_predictions.tsv", sep="\t", index=False)
    _configs().to_csv(root / "configs.tsv", sep="\t", index=False)
    outputs = write_mpma_b_selection_outputs(root, "nMCC")
    for path in outputs.values():
        assert path.exists()
    selection = pd.read_csv(outputs["mpma_b_selection"], sep="\t")
    assert dict(zip(selection["outer_split_key"], selection["config_id"])) == {
        "o0": "A",
        "o1": "B",
        "o2": "A",
    }
    selected_predictions = pd.read_csv(outputs["mpma_b_predictions"], sep="\t")
    assert {
        split: set(group["config_id"])
        for split, group in selected_predictions.groupby("outer_split_key")
    } == {
        "o0": {"A"},
        "o1": {"B"},
        "o2": {"A"},
    }
    summary = json.loads(outputs["mpma_b_summary"].read_text(encoding="utf-8"))
    assert summary["selection_basis"] == "outer_fold_inner_validation"
    final_candidate = json.loads(
        outputs["mpma_b_final_candidate"].read_text(encoding="utf-8")
    )
    assert final_candidate["config_id"] == "A"
    assert final_candidate["selection_basis"] == "all_inner_validation_for_final_refit"


def test_selected_prediction_builder_rejects_multiple_selected_configs_for_one_outer_fold():
    selection = pd.DataFrame(
        [
            {"outer_split_key": "o0", "config_id": "A"},
            {"outer_split_key": "o0", "config_id": "B"},
        ]
    )
    outer = _outer_predictions(
        {
            "o0": {
                "A": [(0, 0.1), (1, 0.9)],
                "B": [(0, 0.1), (1, 0.9)],
            }
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        selected_mpma_b_outer_predictions(selection, outer)
