from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mllabiome.metrics import compute_metrics
from mllabiome.report_statistics import _extended_metrics, run_report_statistics


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_configs(root: Path) -> None:
    _write_tsv(
        pd.DataFrame(
            [
                {
                    "config_id": "b",
                    "resolution": "genus",
                    "levels": "genus",
                    "count_transformation": "identity",
                    "learner": "RF_1000_msl5",
                },
                {
                    "config_id": "e1",
                    "resolution": "genus",
                    "levels": "genus",
                    "count_transformation": "identity",
                    "learner": "RF_1000_msl5",
                },
                {
                    "config_id": "e2",
                    "resolution": "species",
                    "levels": "species",
                    "count_transformation": "relative_abundance",
                    "learner": "XGB",
                },
            ]
        ),
        root / "configs.tsv",
    )


def _write_final_selection(root: Path, aggregation: str | None = None) -> None:
    _write_json(
        {"config_id": "b", "selection_metric": "nMCC"},
        root / "tables" / "mpma_b_final_candidate.json",
    )
    if aggregation is None:
        return
    unit = {
        "ensemble_config_id": "ens",
        "selection_strategy": "top_k",
        "aggregation_strategy": aggregation,
        "optimize_metric": "nMCC",
        "members": json.dumps(["e1", "e2"]),
    }
    if aggregation == "weighted_mean_proba":
        unit["weights"] = [0.25, 0.75]
    _write_json(
        {"inner_val_best_mpmas_ensemble": unit},
        root / "ensembling" / "selected_unit.json",
    )
    _write_tsv(
        pd.DataFrame(
            [
                {"config_id": "e1", "ok": 1, "nMCC": 0.60},
                {"config_id": "e1", "ok": 1, "nMCC": 0.70},
                {"config_id": "e2", "ok": 1, "nMCC": 0.80},
                {"config_id": "e2", "ok": 1, "nMCC": 0.90},
            ]
        ),
        root / "inner_results" / "inner_results.tsv",
    )


def _append_prediction(
    rows: list[dict],
    config_id: str,
    split_key: str,
    sample_id: str,
    y_true: int,
    p1: float,
) -> None:
    rows.append(
        {
            "config_id": config_id,
            "outer_split_key": split_key,
            "sample_id": sample_id,
            "y_true": int(y_true),
            "y_pred": int(p1 >= 0.5),
            "proba_0": 1.0 - float(p1),
            "proba_1": float(p1),
        }
    )


def _write_nested_run(
    root: Path,
    repeats: int = 2,
    n_samples: int = 8,
    aggregation: str | None = None,
    missing_last: bool = False,
) -> None:
    _write_configs(root)
    _write_final_selection(root, aggregation)
    _write_json(
        {"n_samples": n_samples, "sweep": {"evaluation": {"repeats": repeats}}},
        root / "manifest.json",
    )
    y = np.asarray(
        [0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2), dtype=int
    )
    rows: list[dict] = []
    for repeat in range(repeats):
        for i, yy in enumerate(y):
            if missing_last and repeat == repeats - 1 and i == n_samples - 1:
                continue
            split_key = f"r{repeat}_o{i % 2}"
            base = float(np.clip(0.15 + 0.70 * yy + 0.01 * repeat, 0.01, 0.99))
            _append_prediction(rows, "b", split_key, f"s{i}", int(yy), base)
            if aggregation is not None:
                _append_prediction(
                    rows,
                    "e1",
                    split_key,
                    f"s{i}",
                    int(yy),
                    float(np.clip(base + (0.04 if yy else -0.04), 0.01, 0.99)),
                )
                _append_prediction(
                    rows,
                    "e2",
                    split_key,
                    f"s{i}",
                    int(yy),
                    float(np.clip(base + (0.08 if yy else -0.08), 0.01, 0.99)),
                )
    _write_tsv(pd.DataFrame(rows), root / "predictions" / "outer_predictions.tsv")


def _write_lodo_run(root: Path, aggregation: str = "mean_proba") -> None:
    _write_configs(root)
    _write_final_selection(root, aggregation)
    rows: list[dict] = []
    sample_no = 0
    for cohort in ("study_a", "study_b", "study_c"):
        for yy, p1 in ((0, 0.10), (0, 0.25), (1, 0.70), (1, 0.90)):
            sample_id = f"s{sample_no}"
            _append_prediction(rows, "b", cohort, sample_id, yy, p1)
            _append_prediction(
                rows,
                "e1",
                cohort,
                sample_id,
                yy,
                float(np.clip(p1 + (0.03 if yy else -0.03), 0.01, 0.99)),
            )
            _append_prediction(
                rows,
                "e2",
                cohort,
                sample_id,
                yy,
                float(np.clip(p1 + (0.06 if yy else -0.06), 0.01, 0.99)),
            )
            sample_no += 1
    _write_tsv(pd.DataFrame(rows), root / "predictions" / "outer_predictions.tsv")
    _write_json(
        {"n_samples": sample_no, "sweep": {"evaluation": {}}}, root / "manifest.json"
    )


def test_compute_metrics_single_class_resample_is_warning_free() -> None:
    y_true = np.asarray([0, 0, 0], dtype=int)
    y_pred = np.asarray([0, 0, 0], dtype=int)
    proba = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=float)
    out = compute_metrics(y_true, y_pred, proba, np.asarray([0, 1], dtype=int))
    assert np.isclose(out["BalAcc"], 1.0)
    assert np.isclose(out["nMCC"], 0.5)
    assert np.isnan(out["AUC"])
    assert np.isnan(out["PR_AUC"])


def test_extended_metrics_adds_proper_scoring_rules_binary() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "proba_0": [0.9, 0.8, 0.3, 0.1],
            "proba_1": [0.1, 0.2, 0.7, 0.9],
        }
    )
    out = _extended_metrics(frame, proper_probability=True)
    expected_brier = np.mean(
        (frame["y_true"].to_numpy(dtype=float) - frame["proba_1"].to_numpy(dtype=float))
        ** 2
    )
    assert np.isclose(out["Brier"], expected_brier, atol=1e-6)
    assert np.isfinite(out["LogLoss"])
    assert np.isnan(out["Brier_multiclass"])


def test_repeated_nested_cv_uses_final_mpma_b_and_reports_oof_statistics(
    tmp_path: Path,
) -> None:
    _write_nested_run(tmp_path)
    result = run_report_statistics(
        tmp_path,
        "repeated_nested_cv",
        [{"Strategy": "MPMA-B"}],
        n_bootstrap=40,
        random_state=7,
        selection_metric="AUC",
        calibration_bins=4,
    )
    tables = tmp_path / "report" / "tables"
    assert (tmp_path / "final_models.json").exists()
    assert (tables / "strategy_outer_unit_metrics.tsv").exists()
    assert (tables / "strategy_metrics_bootstrap.tsv").exists()
    assert (tables / "strategy_pairwise_tests.tsv").exists()
    assert (tables / "strategy_oof_performance.tsv").exists()
    assert (tables / "strategy_oof_calibration.tsv").exists()
    assert not (tables / "strategy_oof_coverage.tsv").exists()
    assert not (tables / "strategy_oof_pairwise_coverage.tsv").exists()
    perf = result["oof_performance"]
    assert set(perf["Strategy"]) == {"MPMA-B"}
    assert set(perf["selection_scope"]) == {"final_selected_mpma_configuration"}
    assert set(perf["estimand"]) == {"mean_repeat_pooled_oof"}
    assert {"Brier", "LogLoss", "CalibrationSlope"}.issubset(set(perf["metric"]))
    selected = perf[perf["metric"].eq("AUC")]
    assert not selected.empty
    assert selected["is_selection_metric"].all()
    assert set(selected["resampling_unit"]) == {"repeat_and_subject"}
    assert set(selected["n_samples"]) == {8}
    assert set(selected["n_repeats"]) == {2}


def test_lodo_uses_one_final_mpma_e_and_reports_both_estimands(tmp_path: Path) -> None:
    _write_lodo_run(tmp_path, "mean_proba")
    result = run_report_statistics(
        tmp_path,
        "lodo",
        [{"Strategy": "MPMA-E"}],
        n_bootstrap=40,
        random_state=11,
        selection_metric="nMCC",
        calibration_bins=3,
    )
    perf = result["oof_performance"]
    assert set(perf["Strategy"]) == {"MPMA-E"}
    assert set(perf["selection_scope"]) == {"final_selected_ensemble_specification"}
    assert set(perf["estimand"]) == {
        "pooled_sample_weighted",
        "cohort_macro_equal_weight",
    }
    assert set(perf["resampling_unit"]) == {"cohort_then_sample"}
    assert set(perf["n_cohorts"]) == {3}
    assert {"Brier", "LogLoss", "CalibrationInTheLarge"}.issubset(set(perf["metric"]))


def test_comparator_only_statistics_do_not_require_final_model_artifacts(
    tmp_path: Path,
) -> None:
    outer = pd.DataFrame(
        [
            {
                "outer_split_key": "r0_o0",
                "sample_id": "s0",
                "config_id": "reported",
                "y_true": 0,
                "y_pred": 0,
                "proba_0": 0.8,
                "proba_1": 0.2,
            },
            {
                "outer_split_key": "r0_o1",
                "sample_id": "s1",
                "config_id": "reported",
                "y_true": 1,
                "y_pred": 1,
                "proba_0": 0.2,
                "proba_1": 0.8,
            },
            {
                "outer_split_key": "r0_o0",
                "sample_id": "s0",
                "config_id": "not_reported",
                "y_true": 0,
                "y_pred": 1,
                "proba_0": 0.1,
                "proba_1": 0.9,
            },
            {
                "outer_split_key": "r0_o1",
                "sample_id": "s1",
                "config_id": "not_reported",
                "y_true": 1,
                "y_pred": 0,
                "proba_0": 0.9,
                "proba_1": 0.1,
            },
        ]
    )
    _write_tsv(outer, tmp_path / "predictions" / "outer_predictions.tsv")
    result = run_report_statistics(
        tmp_path,
        "nested_cv",
        [{"Strategy": "Baseline RF", "config_id": "reported"}],
        n_bootstrap=20,
        random_state=3,
    )
    perf = result["oof_performance"]
    assert set(perf["Strategy"]) == {"Baseline RF"}
    auc = perf[perf["metric"].eq("AUC")]["estimate"].iloc[0]
    assert np.isclose(float(auc), 1.0)
    assert not (tmp_path / "final_models.json").exists()


def test_oof_pairwise_contrasts_compare_fixed_final_mpma_b_and_mpma_e(
    tmp_path: Path,
) -> None:
    _write_nested_run(tmp_path, aggregation="mean_proba")
    result = run_report_statistics(
        tmp_path,
        "repeated_nested_cv",
        [{"Strategy": "MPMA-B"}, {"Strategy": "MPMA-E"}],
        n_bootstrap=50,
        random_state=17,
        selection_metric="nMCC",
    )
    pairwise = result["oof_pairwise"]
    assert not pairwise.empty
    assert set(pairwise["strategy_a"]) == {"MPMA-B"}
    assert set(pairwise["strategy_b"]) == {"MPMA-E"}
    assert set(pairwise["estimand"]) == {"mean_repeat_pooled_oof"}
    assert set(pairwise["n_paired_samples"]) == {8}
    assert set(pairwise["n_repeats"]) == {2}
    assert set(pairwise["p_value_policy"]) == {"not_reported_for_pooled_oof_contrasts"}
    assert pairwise["p_value"].isna().all()
    assert set(pairwise["resampling_unit"]) == {"paired_repeat_and_subject"}
    selected = pairwise[pairwise["metric"].eq("nMCC")]
    assert not selected.empty
    assert selected["is_selection_metric"].all()


def test_incomplete_final_mpma_b_oof_coverage_is_a_hard_error(tmp_path: Path) -> None:
    _write_nested_run(tmp_path, repeats=1, n_samples=4, missing_last=True)
    with pytest.raises(ValueError, match="incomplete OOF coverage"):
        run_report_statistics(
            tmp_path,
            "repeated_nested_cv",
            [{"Strategy": "MPMA-B"}],
            n_bootstrap=10,
            random_state=1,
        )


def test_rank_mean_final_mpma_e_omits_probability_metrics_and_calibration(
    tmp_path: Path,
) -> None:
    _write_nested_run(tmp_path, repeats=1, n_samples=6, aggregation="rank_mean")
    result = run_report_statistics(
        tmp_path,
        "nested_cv",
        [{"Strategy": "MPMA-E"}],
        n_bootstrap=20,
        random_state=2,
        selection_metric="nMCC",
    )
    perf = result["oof_performance"]
    metrics = set(perf["metric"].astype(str))
    assert {"AUC", "PR_AUC", "nMCC"}.issubset(metrics)
    assert "Brier" not in metrics
    assert "Brier_multiclass" not in metrics
    assert "LogLoss" not in metrics
    assert "CalibrationInTheLarge" not in metrics
    assert "CalibrationIntercept" not in metrics
    assert "CalibrationSlope" not in metrics
    assert result["oof_calibration"].empty
    semantics = perf.iloc[0]
    assert bool(semantics["probability_metrics_valid"]) is False
    assert "rank_mean" in str(semantics["probability_semantics_reason"])
