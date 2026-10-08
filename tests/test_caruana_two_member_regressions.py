from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "mllabiome"


def load_functions(names: list[str], namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse((ROOT / "ensemble_sweep.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(nodes) != len(names):
        raise RuntimeError("The requested core functions were not found.")
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "ensemble_sweep.py", "exec"),
        namespace,
    )
    return namespace


def loss(y: np.ndarray, p: np.ndarray, metric: str) -> float:
    p = np.clip(np.asarray(p), np.finfo(float).eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    if metric == "log_loss":
        return float(-np.mean(np.log(p[np.arange(len(y)), y])))
    if metric == "brier":
        target = np.eye(p.shape[1])[y]
        return float(np.mean(np.sum((p - target) ** 2, axis=1)))
    raise ValueError(metric)


def assert_invariant(
    result: tuple,
    ids: list[str],
    stack: np.ndarray,
    y: np.ndarray,
    metric: str,
    max_members: int,
) -> None:
    members, weights, score, trajectory, diagnostics = result
    assert 2 <= len(members) <= max_members
    assert len(members) == len(set(members))
    assert set(members) <= set(ids)
    assert weights.shape == (len(members),)
    assert np.isfinite(weights).all() and np.all(weights > 0.0)
    assert np.isclose(weights.sum(), 1.0, atol=1e-12)
    selected = stack[[ids.index(member) for member in members]]
    probabilities = np.einsum("m,mnc->nc", weights, selected)
    assert np.isclose(loss(y, probabilities, metric), score, atol=1e-10)
    assert np.isclose(trajectory[-1], score, atol=1e-10)
    assert len(trajectory) >= 2
    assert diagnostics


def run() -> None:
    names = [
        "_caruana_controls",
        "_caruana_select",
        "_caruana_select_from_start",
        "_caruana_multistart_select",
        "_bagged_caruana_select",
        "_complete_library_stack",
        "_candidate_from_caruana",
        "_candidate_from_bagged_caruana",
        "_candidate_from_caruana_multistart",
        "_evaluate_selected_members",
    ]

    def better(new: float, old: float, metric: str, tol: float = 0.0) -> bool:
        return new < old - tol

    def weighted(
        stack: np.ndarray,
        method: str,
        weights: np.ndarray | None = None,
        params: dict | None = None,
    ) -> np.ndarray:
        if weights is None:
            return np.mean(stack, axis=0)
        return np.einsum("m,mnc->nc", weights, stack, optimize=True)

    ns = dict(
        np=np,
        pd=pd,
        json=json,
        hashlib=hashlib,
        Any=Any,
        _CARUANA_IMPROVEMENT_TOL=1e-10,
        _BAGGED_CARUANA_MIN_BAGS=8,
        _BAGGED_CARUANA_MAX_BAGS=24,
        _SUPER_LEARNER_WEIGHT_TOL=1e-8,
        _metric_is_loss=lambda metric: True,
        _metric_better=better,
        _metric_value=loss,
        _renormalize_proba=lambda proba, n: (
            np.maximum(proba, 0) / np.maximum(proba, 0).sum(axis=1, keepdims=True)
        ),
        _weighted_probability_mean=lambda stack, weights: weighted(
            stack, "weighted_mean_proba", weights
        ),
        aggregate_member_predictions=weighted,
        compute_metrics=lambda y, pred, p, classes: {
            "log_loss": loss(y, p, "log_loss"),
            "brier": loss(y, p, "brier"),
        },
        effective_aggregation_weights=lambda method, n: np.full(n, 1 / n),
        LEARNED_AGGREGATIONS=set(),
        _aligned_stack=lambda predictions, library, pcols, inner=True: (
            pd.DataFrame({"y_true": predictions.attrs["y"]}),
            predictions.attrs["stack"][
                [predictions.attrs["ids"].index(member) for member in library]
            ],
        ),
        _missing_members=lambda *args, **kwargs: [],
    )
    load_functions(names, ns)
    y = np.arange(80) % 2
    best = np.eye(2)[y] * 0.98 + 0.01
    medium = np.eye(2)[y] * 0.58 + 0.21
    poor = np.eye(2)[1 - y] * 0.58 + 0.21
    ids = ["best", "medium", "poor"]
    experiments = [
        ("dominant", np.stack([best, medium, poor])),
        ("identical", np.stack([best, best, best])),
        ("two", np.stack([best, poor])),
        ("near_identical", np.stack([best, best * 0.999 + 0.001 * medium, medium])),
    ]
    counts = {"caruana": 0, "multistart": 0, "bagged": 0, "candidate": 0, "negative": 0}
    for scenario, stack in experiments:
        subset_ids = ids[: stack.shape[0]]
        for metric in ("log_loss", "brier"):
            for max_members in (2, 3):
                members_limit = min(max_members, len(subset_ids))
                result = ns["_caruana_select"](
                    subset_ids, stack, y, metric, members_limit
                )
                assert_invariant(result, subset_ids, stack, y, metric, members_limit)
                counts["caruana"] += 1
                for start in range(len(subset_ids)):
                    result = ns["_caruana_select_from_start"](
                        subset_ids, stack, y, metric, members_limit, start
                    )
                    assert_invariant(
                        result, subset_ids, stack, y, metric, members_limit
                    )
                    counts["multistart"] += 1
                result = ns["_caruana_multistart_select"](
                    subset_ids, stack, y, metric, members_limit
                )
                assert_invariant(result, subset_ids, stack, y, metric, members_limit)
                counts["multistart"] += 1
                members, weights, diagnostics = ns["_bagged_caruana_select"](
                    subset_ids, stack, y, metric, members_limit
                )
                assert 2 <= len(members) <= members_limit
                assert len(set(members)) == len(members)
                assert np.all(weights > 0) and np.isclose(np.sum(weights), 1.0)
                assert diagnostics["bagged_caruana_bags"] >= 1
                counts["bagged"] += 1
    prediction_frame = pd.DataFrame()
    prediction_frame.attrs["ids"] = ids
    prediction_frame.attrs["stack"] = experiments[0][1]
    prediction_frame.attrs["y"] = y
    scores = pd.Series([0.01, 0.6, 1.0], index=ids)
    for method in (
        "_candidate_from_caruana",
        "_candidate_from_bagged_caruana",
        "_candidate_from_caruana_multistart",
    ):
        row = ns[method](
            {
                "max_size": 3,
                "selection_strategy": method,
                "aggregation_strategy": "weighted_mean_proba",
            },
            scores,
            prediction_frame,
            ["proba_0", "proba_1"],
            "log_loss",
        )
        assert row is not None
        members = json.loads(row["members"])
        weights = json.loads(row["weights"])
        assert len(members) >= 2 and row["effective_member_count"] >= 2
        assert all(weight > 0 for weight in weights)
        counts["candidate"] += 1
    bad = [
        (["x"], np.stack([best]), 2),
        (["x", "x"], np.stack([best, medium]), 2),
        (["x", "y"], np.stack([best, medium]), 1),
    ]
    for ids_bad, stack_bad, max_members in bad:
        for method in (
            "_caruana_select",
            "_bagged_caruana_select",
            "_caruana_multistart_select",
        ):
            try:
                ns[method](ids_bad, stack_bad, y, "log_loss", max_members)
            except ValueError:
                counts["negative"] += 1
            else:
                raise AssertionError(
                    f"{method} accepted an invalid ensemble library or size"
                )
    print("CARUANA INVARIANT TESTS PASSED", counts)


if __name__ == "__main__":
    run()


def test_caruana_selection_keeps_two_effective_members():
    run()
