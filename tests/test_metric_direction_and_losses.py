from __future__ import annotations

import numpy as np
import pandas as pd

from mllabiome.configs_sweep import _backfill_metrics_from_predictions
from mllabiome.metrics import (
    compute_metrics,
    metric_better,
    metric_is_loss,
    metric_passes_threshold,
)


def test_loss_metric_direction_is_lower_is_better():
    for metric in ("log_loss", "logloss", "brier", "brier_loss", "brier_multiclass"):
        assert metric_is_loss(metric)
        assert metric_better(0.2, 0.4, metric)
        assert not metric_better(0.4, 0.2, metric)
        assert metric_passes_threshold(0.2, 0.3, metric)
        assert not metric_passes_threshold(0.4, 0.3, metric)


def test_score_metric_direction_is_higher_is_better():
    for metric in ("nMCC", "AUC", "Accuracy"):
        assert not metric_is_loss(metric)
        assert metric_better(0.8, 0.6, metric)
        assert not metric_better(0.6, 0.8, metric)
        assert metric_passes_threshold(0.8, 0.7, metric)
        assert not metric_passes_threshold(0.6, 0.7, metric)


def test_compute_metrics_includes_binary_log_loss_and_brier():
    y_true = np.asarray([0, 1, 0, 1], dtype=int)
    proba = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.75, 0.25], [0.1, 0.9]], dtype=float)
    y_pred = proba.argmax(axis=1)
    values = compute_metrics(y_true, y_pred, proba, np.asarray([0, 1], dtype=int))
    expected_log_loss = -np.mean(np.log(proba[np.arange(len(y_true)), y_true]))
    expected_brier = np.mean((proba[:, 1] - y_true.astype(float)) ** 2)
    assert values["log_loss"] == round(float(expected_log_loss), 6)
    assert values["brier"] == round(float(expected_brier), 6)


def test_compute_metrics_includes_multiclass_log_loss_and_brier():
    y_true = np.asarray([0, 1, 2, 1], dtype=int)
    proba = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.75, 0.15],
            [0.15, 0.15, 0.7],
            [0.1, 0.8, 0.1],
        ],
        dtype=float,
    )
    y_pred = proba.argmax(axis=1)
    values = compute_metrics(y_true, y_pred, proba, np.asarray([0, 1, 2], dtype=int))
    assert np.isfinite(values["log_loss"])
    assert np.isfinite(values["brier"])
    assert values["log_loss"] > 0.0
    assert values["brier"] >= 0.0


def test_existing_metric_rows_are_backfilled_from_saved_probabilities():
    metrics = pd.DataFrame(
        [
            {
                "inner_key": "inner0",
                "config_id": "A",
                "ok": 1,
                "nMCC": 1.0,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "split_key": "inner0",
                "config_id": "A",
                "sample_id": "s0",
                "y_true": 0,
                "y_pred": 0,
                "proba_A": 0.9,
                "proba_B": 0.1,
            },
            {
                "split_key": "inner0",
                "config_id": "A",
                "sample_id": "s1",
                "y_true": 1,
                "y_pred": 1,
                "proba_A": 0.2,
                "proba_B": 0.8,
            },
        ]
    )
    result = _backfill_metrics_from_predictions(
        metrics,
        predictions,
        np.asarray([0, 1], dtype=int),
        ("A", "B"),
        "inner_key",
    )
    assert np.isfinite(float(result.loc[0, "log_loss"]))
    assert np.isfinite(float(result.loc[0, "brier"]))
