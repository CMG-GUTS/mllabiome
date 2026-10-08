from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1] / "mllabiome"


def load_functions(file: str, names: list[str], namespace: dict) -> dict:
    tree = ast.parse((ROOT / file).read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), file, "exec"), namespace)
    return namespace


def classification_stack(n=60):
    y = np.arange(n) % 2
    strong = np.column_stack(
        [np.where(y == 0, 0.99, 0.01), np.where(y == 1, 0.99, 0.01)]
    )
    medium = np.column_stack([np.where(y == 0, 0.8, 0.2), np.where(y == 1, 0.8, 0.2)])
    weak = np.column_stack([np.where(y == 0, 0.25, 0.75), np.where(y == 1, 0.25, 0.75)])
    return np.stack([strong, medium, weak]), y


def test_classification():
    names = [
        "_uniform_weights",
        "_super_learner_loss",
        "_super_learner_gradient",
        "_fit_convex_probability_weights",
        "_regularized_super_learner_loss",
        "_fit_regularized_convex_probability_weights",
        "_adaptive_regularization_strength",
        "_select_super_learner_support",
        "_fit_super_learner",
        "_fit_adaptive_super_learner",
        "_evaluate_selected_members",
        "_candidate_from_safe_super_learner",
        "_candidate_from_super_learner",
        "_candidate_from_adaptive_super_learner",
        "_candidate_from_late_fusion",
    ]

    def agg(stack, strategy, weights, params=None):
        return np.einsum(
            "m,mnc->nc", np.asarray(weights, dtype=float), stack, optimize=True
        )

    ns = dict(
        np=np,
        minimize=minimize,
        _SUPER_LEARNER_WEIGHT_TOL=1e-8,
        _SUPER_LEARNER_MIN_WEIGHT=0.01,
        _SUPER_LEARNER_OPT_MAXITER=1000,
        _SUPER_LEARNER_OPT_FTOL=1e-12,
        aggregate_member_predictions=agg,
        _weighted_probability_mean=lambda stack, weights: agg(
            stack, "weighted_mean_proba", weights
        ),
        _resolved_super_learner_loss=lambda plan: "log_loss",
    )
    load_functions("ensemble_sweep.py", names, ns)
    ids = ["a", "b", "c"]
    stack, y = classification_stack()
    assert np.argmax(ns["_fit_convex_probability_weights"](stack, y, "log_loss")) == 0
    for loss in ("log_loss", "brier"):
        for max_members in (2, 3):
            selected, weights, value = ns["_fit_super_learner"](
                ids, stack, y, loss, max_members
            )
            assert 2 <= len(selected) <= max_members
            assert len(set(selected)) == len(selected)
            assert len(weights) == len(selected)
            assert np.all(weights >= 0.01 - 1e-14)
            assert abs(weights.sum() - 1) < 1e-10
            assert np.isfinite(value)
            print(
                "classification", loss, max_members, selected, weights.tolist(), value
            )
        selected, weights, value, strength = ns["_fit_adaptive_super_learner"](
            ids, stack, y, loss, 3
        )
        assert len(selected) >= 2 and "a" in selected
        assert min(weights) >= 0.01 - 1e-14
        assert abs(weights.sum() - 1) < 1e-10
        print("adaptive", loss, selected, weights.tolist(), value, strength)
        rng = np.random.default_rng(19)
        probe = np.array([0.2, 0.3, 0.5])
        random_stack = rng.dirichlet([1, 1], size=(3, 24))
        labels = rng.integers(0, 2, size=24)
        analytic = ns["_super_learner_gradient"](probe, random_stack, labels, loss)
        numerical = []
        for k in range(3):
            d = np.zeros(3)
            d[k] = 1e-6
            numerical.append(
                (
                    ns["_super_learner_loss"](probe + d, random_stack, labels, loss)
                    - ns["_super_learner_loss"](probe - d, random_stack, labels, loss)
                )
                / 2e-6
            )
        assert np.allclose(analytic, numerical, atol=1e-6), (analytic, numerical)
    for f in (ns["_fit_super_learner"], ns["_fit_adaptive_super_learner"]):
        for case in (["a"], ["a", "a"], ["a", "b"]):
            try:
                f(case, stack[: len(case)], y, "log_loss", 1 if len(case) == 2 else 3)
            except ValueError:
                pass
            else:
                raise AssertionError("Expected invalid library/max_members error")
    for n in (2, 3, 4, 9):
        rg = np.random.default_rng(99 + n)
        s = rg.dirichlet(np.ones(2), size=(n, 25))
        yy = rg.integers(0, 2, 25)
        for loss in ("brier", "log_loss"):
            w = ns["_fit_convex_probability_weights"](s, yy, loss, min_weight=0.01)
            assert np.all(w >= 0.01 - 1e-14) and abs(w.sum() - 1) < 1e-10
            wa = ns["_fit_regularized_convex_probability_weights"](
                s, yy, loss, np.eye(n)[0], 0.1, min_weight=0.01
            )
            assert np.all(wa >= 0.01 - 1e-14) and abs(wa.sum() - 1) < 1e-10
    ns["json"] = json
    ns["_metric_value"] = lambda labels, preds, metric: float(
        -np.mean(np.log(np.clip(preds[np.arange(len(labels)), labels], 1e-15, 1.0)))
    )
    ns["compute_metrics"] = lambda labels, classes, proba, possible: {
        "log_loss": ns["_metric_value"](labels, proba, "log_loss")
    }
    ns["effective_aggregation_weights"] = lambda kind, n: np.full(n, 1.0 / n)
    ns["LEARNED_AGGREGATIONS"] = set()
    ns["_stable_metric_improvement"] = lambda *args: (False, [])
    ns["_complete_library_stack"] = lambda *args: (
        ids,
        pd.DataFrame({"y_true": y}),
        stack,
    )
    spec = {
        "max_size": 3,
        "aggregation_strategy": "weighted_mean_proba",
        "selection_strategy": "super_learner",
    }
    candidate = ns["_candidate_from_super_learner"](
        spec, pd.Series(), pd.DataFrame(), [], SimpleNamespace(), "log_loss"
    )
    assert candidate["member_count"] >= 2 and candidate["effective_member_count"] >= 2
    assert len(json.loads(candidate["weights"])) >= 2
    print(
        "classification candidate selection", candidate["members"], candidate["weights"]
    )
    spec["selection_strategy"] = "adaptive_super_learner"
    candidate = ns["_candidate_from_adaptive_super_learner"](
        spec, pd.Series(), pd.DataFrame(), [], SimpleNamespace(), "log_loss"
    )
    assert candidate["member_count"] >= 2 and candidate["effective_member_count"] >= 2
    print("adaptive candidate selection", candidate["members"], candidate["weights"])
    spec["selection_strategy"] = "safe_super_learner"
    result = ns["_candidate_from_safe_super_learner"](
        spec, pd.Series(), pd.DataFrame(), [], SimpleNamespace(), "log_loss"
    )
    assert result is None
    print("safe_super_learner rejected unstable candidate, no singleton fallback")
    ns["_late_fusion_members"] = lambda *args: (["a", "b"], ["m1", "m2"])
    ns["_aligned_stack"] = lambda *args, **kwargs: (
        pd.DataFrame({"y_true": y}),
        stack[:2],
    )
    late_spec = {
        "max_size": 2,
        "integration": "late_super_learner",
        "aggregation_strategy": "weighted_mean_proba",
        "selection_strategy": "late_super_learner",
    }
    candidate = ns["_candidate_from_late_fusion"](
        late_spec,
        pd.Series(),
        pd.DataFrame(),
        pd.DataFrame(),
        [],
        SimpleNamespace(),
        "log_loss",
    )
    assert candidate["effective_member_count"] == 2
    print(
        "late-super-learner candidate selection",
        candidate["members"],
        candidate["weights"],
    )
    try:
        ns["_evaluate_selected_members"](
            spec,
            ["a", "b"],
            stack[:2],
            y,
            "log_loss",
            native_weights=np.array([1.0, 0.0]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected candidate guard to reject effective singleton")


def test_final():
    ns = dict(
        np=np,
        pd=pd,
        json=json,
        Any=object,
        Path=Path,
        _SUPPORTED_ENSEMBLES={"weighted_mean_proba", "mean_proba"},
        effective_aggregation_weights=lambda kind, n, weights: (
            np.asarray(weights, dtype=float) if weights is not None else np.ones(n) / n
        ),
    )
    load_functions(
        "final_models.py",
        [
            "_parse_members",
            "_resolve_config_id",
            "_stored_weights",
            "_build_mpma_e",
            "_core_config",
        ],
        ns,
    )
    conf = pd.DataFrame({"config_id": ["a", "b", "c"], "learner": ["RF", "CB", "LR"]})
    ns["_member_scores"] = lambda root, ids, metric: {k: 0.33 for k in ids}
    unit = {
        "members": ["a", "b"],
        "aggregation_strategy": "weighted_mean_proba",
        "selection_strategy": "super_learner",
        "weights": [0.99, 0.01],
        "effective_member_count": 2,
        "selection_metric": "log_loss",
    }
    ns["_mpma_e_unit"] = lambda root: dict(unit)
    model = ns["_build_mpma_e"](Path("."), conf)
    assert model["member_count"] == 2 and model["effective_member_count"] == 2
    assert len(model["members"]) == 2
    print(
        "final-model reconstruction",
        [(member["config_id"], member["weight"]) for member in model["members"]],
    )
    unit["weights"] = [1.0, 0.0]
    try:
        ns["_build_mpma_e"](Path("."), conf)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected singleton weights to be rejected")
    unit["weights"] = [0.5, 0.5]
    unit["effective_member_count"] = 1
    try:
        ns["_build_mpma_e"](Path("."), conf)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected stored effective-member mismatch to be rejected")
    print("final-model singleton and metadata mismatch rejected")


def test_regression():
    def predict(stack, method, weights):
        return np.average(stack, axis=0, weights=np.asarray(weights, dtype=float))

    def metric(y, p, metric):
        return np.mean((y - p) ** 2)

    ns = dict(
        np=np,
        minimize=minimize,
        _WEIGHT_TOL=1e-8,
        _MIN_EFFECTIVE_WEIGHT=0.01,
        _OPT_MAXITER=1000,
        _OPT_FTOL=1e-12,
        aggregate_regression_predictions=predict,
        _metric_value=metric,
        metric_is_loss=lambda metric: True,
    )
    load_functions(
        "regression_ensemble.py",
        ["_fit_convex_regression_weights", "_fit_super_learner"],
        ns,
    )
    rng = np.random.default_rng(26)
    y = rng.normal(size=64)
    stack = np.stack([y.copy(), y + 1, y + 3])
    for max_members in (2, 3):
        members, weights = ns["_fit_super_learner"](
            ["perfect", "bias1", "bias3"], stack, y, "MSE", max_members
        )
        assert 2 <= len(members) <= max_members
        assert np.all(weights >= 0.01 - 1e-14)
        assert abs(weights.sum() - 1) < 1e-10
        print("regression", max_members, members, weights.tolist())


if __name__ == "__main__":
    test_classification()
    test_final()
    test_regression()
    print("ALL INVARIANT TESTS PASSED")
