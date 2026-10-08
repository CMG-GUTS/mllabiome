from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "mllabiome" / "advanced_ensemble.py"
SPEC = importlib.util.spec_from_file_location(
    "mllabiome_advanced_ensemble_test", SOURCE
)
ADVANCED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADVANCED)


def _log_loss(labels, probabilities, metric):
    assert metric == "log_loss"
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.mean(np.log(probabilities[np.arange(len(labels)), labels])))


@pytest.mark.parametrize("classes", (2, 3))
@pytest.mark.parametrize("method", sorted(ADVANCED.ADVANCED_AGGREGATIONS))
def test_advanced_aggregation_crossfit_serialization(classes, method):
    rng = np.random.default_rng(734 + classes)
    n = 90
    labels = np.arange(n) % classes
    rng.shuffle(labels)
    stack = rng.gamma(2.0, 1.0, (4, n, classes))
    stack /= stack.sum(axis=2, keepdims=True)
    for member in range(4):
        stack[member, np.arange(n), labels] += 0.15 + 0.03 * member
        stack[member] /= stack[member].sum(axis=1, keepdims=True)
    folds = np.asarray([f"fold{index % 3}" for index in range(n)])
    crossfit, parameters, diagnostics = ADVANCED.fit_crossfitted_advanced(
        stack,
        labels,
        folds,
        method,
        "log_loss",
        _log_loss,
        lambda metric: metric == "log_loss",
    )
    reconstructed = ADVANCED.apply_advanced_aggregation(
        stack, json.loads(json.dumps(parameters))
    )
    for probabilities in (crossfit, reconstructed):
        assert probabilities.shape == (n, classes)
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
    assert diagnostics["stacking_internal_folds"] == 3
    assert np.isfinite(diagnostics["stacking_crossfit_score"])
    if method in {"rf_stacking", "extra_trees_stacking", "boosted_stacking"}:
        direct = ADVANCED._fit_meta_tree(method, stack, labels)
        np.testing.assert_allclose(
            reconstructed,
            ADVANCED.apply_advanced_aggregation(stack, direct),
            atol=1e-12,
        )


def test_advanced_aggregation_rejects_singleton():
    singleton = np.asarray([[[0.8, 0.2], [0.3, 0.7]]], dtype=float)
    with pytest.raises(ValueError, match="at least two members"):
        ADVANCED._probability_matrix(singleton)
