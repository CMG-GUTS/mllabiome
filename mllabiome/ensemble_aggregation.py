from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from scipy.special import expit, softmax
from scipy.stats import rankdata

from .advanced_ensemble import ADVANCED_AGGREGATIONS, apply_advanced_aggregation

SUPPORTED_AGGREGATIONS: tuple[str, ...] = (
    "mean_proba",
    "weighted_mean_proba",
    "median_proba",
    "trimmed_mean_proba",
    "geometric_mean_proba",
    "rank_mean",
    "majority_vote",
    "max_proba",
    "min_proba",
    "trimmed_mean",
    "logit_mean",
    "logistic_stacking",
    "elastic_net_stacking",
    "cohort_robust_stacking",
    *sorted(ADVANCED_AGGREGATIONS),
)

LEARNED_AGGREGATIONS = frozenset(
    {
        "logistic_stacking",
        "elastic_net_stacking",
        "cohort_robust_stacking",
        *ADVANCED_AGGREGATIONS,
    }
)

LINEAR_PROBABILITY_AGGREGATIONS = frozenset({"mean_proba", "weighted_mean_proba"})
PROBABILITY_PRESERVING_AGGREGATIONS = frozenset(
    {
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "trimmed_mean_proba",
        "geometric_mean_proba",
        "trimmed_mean",
        "logit_mean",
        "logistic_stacking",
        "elastic_net_stacking",
        "cohort_robust_stacking",
        *ADVANCED_AGGREGATIONS,
    }
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


def stacking_features(stack: np.ndarray, clip: float = 1e-6) -> np.ndarray:
    values = np.asarray(stack, dtype=float)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[2] < 2:
        raise ValueError(
            "Expected member probabilities with shape n_members x n_samples x n_classes."
        )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Member probabilities must be finite and non-negative.")
    sums = values.sum(axis=2, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("Member probabilities contain rows with zero total mass.")
    values = values / sums
    eps = float(clip)
    if not np.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise ValueError("Stacking probability clip must lie in (0, 0.5).")
    values = np.clip(values, eps, 1.0)
    reference = values[:, :, -1:]
    transformed = np.log(values[:, :, :-1]) - np.log(reference)
    return np.transpose(transformed, (1, 0, 2)).reshape(values.shape[1], -1)


def _stacking_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    if parameters is None:
        raise ValueError("Learned stacking aggregation requires stored parameters.")
    values = dict(parameters)
    required = {"coef", "intercept", "classes", "n_members", "n_classes"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"Stored stacking parameters are missing fields: {missing}.")
    return values


def _apply_stacking(
    stack: np.ndarray, parameters: Mapping[str, Any] | None
) -> np.ndarray:
    values = _stacking_parameters(parameters)
    n_members = int(values["n_members"])
    n_classes = int(values["n_classes"])
    if int(stack.shape[0]) != n_members or int(stack.shape[2]) != n_classes:
        raise ValueError(
            "Stored stacking parameters do not match the member prediction shape."
        )
    features = stacking_features(stack, float(values.get("clip", 1e-6)))
    coef = np.asarray(values["coef"], dtype=float)
    intercept = np.asarray(values["intercept"], dtype=float).reshape(-1)
    classes = np.asarray(values["classes"], dtype=int).reshape(-1)
    if not np.array_equal(np.sort(classes), np.arange(n_classes, dtype=int)):
        raise ValueError(
            "Stored stacking classes are incompatible with probability columns."
        )
    if coef.ndim != 2 or coef.shape[1] != features.shape[1]:
        raise ValueError("Stored stacking coefficients have incompatible shape.")
    if coef.shape[0] == 1 and len(classes) == 2:
        if len(intercept) != 1:
            raise ValueError("Stored binary stacking intercept has incompatible shape.")
        positive = expit(features @ coef[0] + intercept[0])
        raw = np.column_stack([1.0 - positive, positive])
    else:
        if coef.shape[0] != len(classes) or len(intercept) != len(classes):
            raise ValueError(
                "Stored multiclass stacking parameters have incompatible shape."
            )
        raw = softmax(features @ coef.T + intercept.reshape(1, -1), axis=1)
    proba = np.zeros((features.shape[0], n_classes), dtype=float)
    for source, target in enumerate(classes):
        proba[:, int(target)] = raw[:, source]
    return proba


def aggregate_member_predictions(
    stack: np.ndarray,
    aggregation: str,
    weights: Iterable[float] | None = None,
    parameters: Mapping[str, Any] | None = None,
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
    elif method == "trimmed_mean_proba":
        n_members = int(stack.shape[0])
        trim = int(np.floor(0.1 * n_members))
        if trim < 1 or 2 * trim >= n_members:
            proba = np.mean(stack, axis=0)
        else:
            ordered = np.sort(stack, axis=0)
            proba = np.mean(ordered[trim : n_members - trim], axis=0)
    elif method == "geometric_mean_proba":
        eps = np.finfo(float).eps
        proba = np.exp(np.mean(np.log(np.clip(stack, eps, 1.0)), axis=0))
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
    elif method == "min_proba":
        proba = np.min(stack, axis=0)
    elif method == "trimmed_mean":
        ordered = np.sort(stack, axis=0)
        trim = int(np.floor(0.2 * stack.shape[0]))
        trim = min(trim, max(0, (stack.shape[0] - 2) // 2))
        proba = np.mean(ordered[trim : stack.shape[0] - trim], axis=0)
    elif method == "logit_mean":
        clipped = np.clip(stack, 1e-8, 1.0)
        log_pool = np.mean(np.log(clipped), axis=0)
        proba = softmax(log_pool, axis=1)
    elif method in ADVANCED_AGGREGATIONS:
        proba = apply_advanced_aggregation(stack, parameters)
    else:
        proba = _apply_stacking(stack, parameters)

    proba = np.asarray(proba, dtype=float)
    if not np.isfinite(proba).all() or np.any(proba < 0.0):
        raise ValueError("Aggregated MPMA-E predictions contain invalid values")
    sums = proba.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.0):
        zero_rows = np.flatnonzero(sums[:, 0] <= 0.0)
        if method == "rank_mean" and len(zero_rows):
            proba[zero_rows, :] = 1.0
            sums = proba.sum(axis=1, keepdims=True)
        else:
            raise ValueError(
                "Aggregated MPMA-E predictions contain rows with zero total score"
            )
    return proba / sums
