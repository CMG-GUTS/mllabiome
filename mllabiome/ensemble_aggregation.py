from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import rankdata

SUPPORTED_AGGREGATIONS: tuple[str, ...] = (
    "mean_proba",
    "weighted_mean_proba",
    "median_proba",
    "rank_mean",
    "majority_vote",
    "max_proba",
    "min_proba",
)

LINEAR_PROBABILITY_AGGREGATIONS = frozenset({"mean_proba", "weighted_mean_proba"})
PROBABILITY_PRESERVING_AGGREGATIONS = frozenset(
    {"mean_proba", "weighted_mean_proba", "median_proba"}
)


def normalise_weights(weights: Iterable[float], n_members: int) -> np.ndarray:
    arr = np.asarray(list(weights), dtype=float)
    if arr.ndim != 1 or len(arr) != int(n_members):
        raise ValueError("Expected exactly one ensemble weight per member.")
    if not np.isfinite(arr).all() or np.any(arr < 0.0):
        raise ValueError("Ensemble weights must be finite and non-negative.")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("Ensemble weights must have positive total mass.")
    return arr / total


def effective_aggregation_weights(
    aggregation: str,
    n_members: int,
    weights: Iterable[float] | None = None,
) -> np.ndarray | None:
    method = str(aggregation).strip()
    if method == "mean_proba":
        if int(n_members) < 1:
            raise ValueError("Cannot aggregate an empty ensemble.")
        return np.full(int(n_members), 1.0 / float(n_members), dtype=float)
    if method == "weighted_mean_proba":
        if weights is None:
            raise ValueError(
                "weighted_mean_proba requires learned/stored member weights."
            )
        return normalise_weights(weights, int(n_members))
    return None


def aggregate_member_predictions(
    stack: np.ndarray,
    aggregation: str,
    weights: Iterable[float] | None = None,
) -> np.ndarray:
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError(
            "Expected member predictions with shape n_members x n_samples x n_classes"
        )
    if not np.isfinite(stack).all() or np.any(stack < 0.0):
        raise ValueError("Member predictions must be finite and non-negative.")

    method = str(aggregation).strip()
    if method not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"Unsupported MPMA-E aggregation_strategy {aggregation!r}")

    if method == "mean_proba":
        proba = np.mean(stack, axis=0)
    elif method == "weighted_mean_proba":
        w = effective_aggregation_weights(method, stack.shape[0], weights)
        assert w is not None
        proba = np.tensordot(w, stack, axes=(0, 0))
    elif method == "median_proba":
        proba = np.median(stack, axis=0)
    elif method == "rank_mean":
        ranked = np.empty_like(stack, dtype=float)
        for member_index in range(stack.shape[0]):
            for sample_index in range(stack.shape[1]):
                ranked[member_index, sample_index, :] = (
                    rankdata(stack[member_index, sample_index, :], method="average")
                    - 1.0
                )
        proba = np.mean(ranked, axis=0)
    elif method == "majority_vote":
        votes = np.argmax(stack, axis=2)
        proba = np.zeros(stack.shape[1:], dtype=float)
        for sample_index in range(stack.shape[1]):
            counts = np.bincount(
                votes[:, sample_index], minlength=stack.shape[2]
            ).astype(float)
            proba[sample_index] = counts
    elif method == "max_proba":
        proba = np.max(stack, axis=0)
    else:
        proba = np.min(stack, axis=0)

    proba = np.asarray(proba, dtype=float)
    if not np.isfinite(proba).all() or np.any(proba < 0.0):
        raise ValueError("Aggregated MPMA-E predictions contain invalid values")
    sums = proba.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        zero_rows = np.flatnonzero((sums[:, 0] <= 0.0))
        if method == "rank_mean" and len(zero_rows):
            proba[zero_rows, :] = 1.0
            sums = proba.sum(axis=1, keepdims=True)
        else:
            raise ValueError(
                "Aggregated MPMA-E predictions contain rows with zero total score"
            )
    return proba / sums
