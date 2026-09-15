from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mllabiome.metrics import compute_metrics
from mllabiome.report_statistics import run_report_statistics


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def test_compute_metrics_single_class_resample_is_warning_free() -> None:
    # Bootstrap and LODO resamples can legitimately contain one observed class.
    # This must be handled explicitly rather than delegated to sklearn metrics
    # that emit single-label confusion-matrix warnings.
    y_true = np.asarray([0, 0, 0], dtype=int)
    y_pred = np.asarray([0, 0, 0], dtype=int)
    proba = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ],
        dtype=float,
    )
    out = compute_metrics(y_true, y_pred, proba, np.asarray([0, 1], dtype=int))
    assert np.isclose(out["BalAcc"], 1.0)
    # Degenerate MCC is 0.0; nMCC therefore equals 0.5.
    assert np.isclose(out["nMCC"], 0.5)
    assert np.isnan(out["AUC"])
    assert np.isnan(out["PR_AUC"])
    assert np.isfinite(out["Brier"])
    assert np.isfinite(out["LogLoss"])


def test_compute_metrics_adds_proper_scoring_rules_binary() -> None:
    y_true = np.asarray([0, 0, 1, 1], dtype=int)
    proba = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    y_pred = np.argmax(proba, axis=1)
    out = compute_metrics(y_true, y_pred, proba, np.asarray([0, 1], dtype=int))
    expected_brier = np.mean((y_true - proba[:, 1]) ** 2)
    assert np.isclose(out["Brier"], expected_brier, atol=1e-6)
    assert np.isfinite(out["LogLoss"])
    assert np.isnan(out["Brier_multiclass"])


def test_repeated_nested_cv_adds_sample_aware_oof_statistics_without_replacing_legacy(
    tmp_path: Path,
) -> None:
    rows = []
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    for repeat in range(2):
        for i, yy in enumerate(y):
            fold = i % 2
            p1 = 0.15 + 0.7 * yy + 0.02 * repeat
            p1 = float(np.clip(p1, 0.01, 0.99))
            rows.append(
                {
                    "outer_split_key": f"r{repeat}_o{fold}",
                    "sample_id": f"s{i}",
                    "y_true": int(yy),
                    "y_pred": int(p1 >= 0.5),
                    "proba_control": 1.0 - p1,
                    "proba_case": p1,
                }
            )
    _write_tsv(
        pd.DataFrame(rows),
        tmp_path / "predictions" / "mpma_b_outer_predictions.tsv",
    )

    result = run_report_statistics(
        tmp_path,
        "repeated_nested_cv",
        [{"Strategy": "MPMA-B", "selection_basis": "outer_fold_inner_validation"}],
        n_bootstrap=40,
        random_state=7,
        primary_metric="AUC",
        calibration_bins=4,
    )

    tables = tmp_path / "report" / "tables"
    # Existing outputs remain.
    assert (tables / "strategy_outer_unit_metrics.tsv").exists()
    assert (tables / "strategy_metrics_bootstrap.tsv").exists()
    assert (tables / "strategy_pairwise_tests.tsv").exists()
    legacy = pd.read_csv(tables / "strategy_metrics_bootstrap.tsv", sep="\t")
    assert "Brier" not in set(legacy["metric"].astype(str))
    assert "LogLoss" not in set(legacy["metric"].astype(str))
    # New additive outputs exist.
    assert (tables / "strategy_oof_performance.tsv").exists()
    assert (tables / "strategy_oof_calibration.tsv").exists()
    assert (tables / "strategy_oof_coverage.tsv").exists()
    assert (tables / "strategy_oof_statistics_manifest.json").exists()

    perf = result["oof_performance"]
    assert set(perf["Strategy"]) == {"MPMA-B"}
    assert set(perf["estimand"]) == {"mean_repeat_pooled_oof"}
    assert {"Brier", "LogLoss", "CalibrationSlope"}.issubset(set(perf["metric"]))
    primary = perf[perf["metric"].eq("AUC")]
    assert not primary.empty
    assert set(primary["metric_role"]) == {"primary"}
    assert set(primary["resampling_unit"]) == {"repeat_and_subject"}
    assert set(primary["n_samples"]) == {8}
    assert set(primary["n_repeats"]) == {2}

    coverage = result["oof_coverage"].iloc[0]
    assert int(coverage["n_samples_complete_repeats"]) == 8
    assert np.isclose(float(coverage["complete_repeat_coverage_fraction"]), 1.0)


def test_lodo_reports_pooled_and_equal_cohort_estimands(tmp_path: Path) -> None:
    rows = []
    sample_no = 0
    for cohort in ("study_a", "study_b", "study_c"):
        for yy, p1 in ((0, 0.10), (0, 0.25), (1, 0.70), (1, 0.90)):
            rows.append(
                {
                    "outer_split_key": cohort,
                    "sample_id": f"s{sample_no}",
                    "y_true": yy,
                    "y_pred": int(p1 >= 0.5),
                    "proba_control": 1.0 - p1,
                    "proba_case": p1,
                }
            )
            sample_no += 1
    _write_tsv(
        pd.DataFrame(rows),
        tmp_path / "ensembling" / "ensemble_predictions.tsv",
    )

    result = run_report_statistics(
        tmp_path,
        "lodo",
        [
            {
                "Strategy": "MPMA-E",
                "selection_basis": "outer_fold_inner_validation_predictions",
            }
        ],
        n_bootstrap=40,
        random_state=11,
        primary_metric="nMCC",
        calibration_bins=3,
    )
    perf = result["oof_performance"]
    assert set(perf["Strategy"]) == {"MPMA-E"}
    assert set(perf["estimand"]) == {
        "pooled_sample_weighted",
        "cohort_macro_equal_weight",
    }
    assert set(perf["resampling_unit"]) == {"cohort_then_sample"}
    assert set(perf["n_cohorts"]) == {3}
    assert {"Brier", "LogLoss", "CalibrationInTheLarge"}.issubset(set(perf["metric"]))


def test_statistics_only_use_reported_strategy_config_ids(tmp_path: Path) -> None:
    outer = pd.DataFrame(
        [
            {
                "outer_split_key": "r0_o0",
                "sample_id": "s0",
                "config_id": "reported",
                "y_true": 0,
                "y_pred": 0,
                "proba_a": 0.8,
                "proba_b": 0.2,
            },
            {
                "outer_split_key": "r0_o1",
                "sample_id": "s1",
                "config_id": "reported",
                "y_true": 1,
                "y_pred": 1,
                "proba_a": 0.2,
                "proba_b": 0.8,
            },
            {
                "outer_split_key": "r0_o0",
                "sample_id": "s0",
                "config_id": "not_reported",
                "y_true": 0,
                "y_pred": 1,
                "proba_a": 0.1,
                "proba_b": 0.9,
            },
            {
                "outer_split_key": "r0_o1",
                "sample_id": "s1",
                "config_id": "not_reported",
                "y_true": 1,
                "y_pred": 0,
                "proba_a": 0.9,
                "proba_b": 0.1,
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
