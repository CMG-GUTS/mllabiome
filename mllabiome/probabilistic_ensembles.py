from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


METHODS = frozenset(
    {
        "bayesian_model_averaging",
        "mixture_of_experts",
        "cross_study_dro",
        "sparse_nonnegative_stacking",
    }
)


def _validated(stack: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(stack, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] < 2:
        raise ValueError("At least two ensemble members and two classes are required.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Member probabilities must be finite and nonnegative.")
    totals = values.sum(axis=2, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Every member probability row must have positive mass.")
    values = values / totals
    if labels is not None:
        y = np.asarray(labels, dtype=int)
        if (
            y.shape != (values.shape[1],)
            or (y < 0).any()
            or (y >= values.shape[2]).any()
        ):
            raise ValueError("Meta-learner labels do not match member probabilities.")
    return values


def _fit_bma(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    truth = np.clip(values[:, np.arange(len(labels)), labels].T, 1e-12, 1)
    log_predictive = np.log(truth)
    draws = 128
    rng = np.random.default_rng(42)
    concentration = max(1.0, min(40.0, float(np.sqrt(len(labels)))))
    posterior = np.empty((draws, values.shape[0]), dtype=float)
    for index in range(draws):
        sampling = rng.dirichlet(np.ones(len(labels)))
        score = sampling @ log_predictive
        posterior[index] = softmax(concentration * score)
    weights = posterior.mean(axis=0)
    return {
        "kind": "bayesian_model_averaging",
        "weight_method": "bayesian_bootstrap_pseudo_bma_plus",
        "member_weights": weights.tolist(),
        "member_weight_std": posterior.std(axis=0, ddof=1).tolist(),
        "member_weight_interval_95": np.quantile(
            posterior, [0.025, 0.975], axis=0
        ).T.tolist(),
        "bootstrap_replicates": draws,
        "effective_evidence_size": concentration,
        "n_members": int(values.shape[0]),
        "n_classes": int(values.shape[2]),
    }


def _simplex_weights(
    scores: np.ndarray, strength: float, study_keys: np.ndarray | None = None
) -> np.ndarray:
    n_samples, n_members = scores.shape
    if study_keys is not None:
        groups, inverse = np.unique(study_keys, return_inverse=True)
        if len(groups) < 2:
            raise ValueError("Cross-study DRO requires two or more observed studies.")
        counts = np.bincount(inverse)
        if np.min(counts) < 3:
            raise ValueError(
                "Cross-study DRO requires at least three held-out predictions per study."
            )
    else:
        inverse = None
        groups = None

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        mixture = np.clip(scores @ weights, 1e-12, 1.0)
        losses = -np.log(mixture)
        derivatives = -scores / mixture[:, None]
        if groups is None:
            value = float(losses.mean())
            gradient = derivatives.mean(axis=0)
        else:
            values = np.bincount(inverse, weights=losses) / np.bincount(inverse)
            temperature = float(strength)
            active = softmax(values / temperature)
            value = float(
                temperature
                * np.log(np.exp((values - values.max()) / temperature).sum())
                + values.max()
            )
            gradient = np.zeros(n_members, dtype=float)
            for index in range(len(groups)):
                gradient += active[index] * derivatives[inverse == index].mean(axis=0)
        penalty = 0.002 if groups is not None else float(strength)
        baseline = np.full(n_members, 1.0 / n_members)
        value += float(penalty * np.sum((weights - baseline) ** 2))
        gradient += 2.0 * penalty * (weights - baseline)
        return value, gradient

    fitted = minimize(
        objective,
        np.full(n_members, 1.0 / n_members),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_members,
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - 1.0,
                "jac": lambda x: np.ones_like(x),
            }
        ],
        options={"maxiter": 600, "ftol": 1e-11},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Ensemble convex optimisation failed: {fitted.message}")
    weights = np.maximum(fitted.x, 0.0)
    return weights / weights.sum()


def _fit_sparse(
    values: np.ndarray, labels: np.ndarray, strength: float
) -> dict[str, Any]:
    truth = np.clip(values[:, np.arange(len(labels)), labels].T, 1e-12, 1)
    selected: list[int] = []
    remaining = set(range(len(values)))
    best_loss = float("inf")
    best_weights = np.zeros(len(values), dtype=float)
    max_members = min(len(values), max(2, int(np.ceil(np.sqrt(len(values))))))
    for _ in range(max_members):
        winner = None
        for candidate in sorted(remaining):
            indices = selected + [candidate]
            fitted = _simplex_weights(truth[:, indices], 0.0001)
            mix = np.maximum(truth[:, indices] @ fitted, 1e-12)
            loss = float(-np.log(mix).mean())
            if winner is None or (loss, candidate) < (winner[0], winner[1]):
                winner = (loss, candidate, fitted)
        if winner is None:
            break
        loss, member, weights = winner
        if selected and best_loss - loss <= float(strength):
            break
        selected.append(member)
        remaining.remove(member)
        best_loss = loss
        best_weights[:] = 0.0
        best_weights[selected] = weights
    return {
        "kind": "sparse_nonnegative_stacking",
        "member_weights": best_weights.tolist(),
        "active_members": [int(i) for i in np.flatnonzero(best_weights > 1e-7)],
        "sparsity_improvement_threshold": float(strength),
        "n_members": int(values.shape[0]),
        "n_classes": int(values.shape[2]),
    }


def fit_probabilistic(
    method: str,
    stack: np.ndarray,
    labels: np.ndarray,
    strength: float | None = None,
    study_keys: np.ndarray | None = None,
) -> dict[str, Any]:
    values = _validated(stack, labels)
    y = np.asarray(labels, dtype=int)
    if not np.array_equal(np.unique(y), np.arange(values.shape[2])):
        raise ValueError("Every class is required for meta-learner training.")
    if method == "bayesian_model_averaging":
        return _fit_bma(values, y)
    if method == "sparse_nonnegative_stacking":
        return _fit_sparse(
            values, y, float(strength if strength is not None else 0.002)
        )
    if method == "cross_study_dro":
        if study_keys is None:
            raise ValueError(
                "Cross-study DRO needs explicit biological study labels from Data.group_col."
            )
        groups = np.asarray(study_keys, dtype=str)
        if len(groups) != len(y) or any(not x.strip() for x in groups):
            raise ValueError("Cross-study DRO study identifiers are invalid.")
        scores = np.clip(values[:, np.arange(len(y)), y].T, 1e-12, 1)
        weights = _simplex_weights(
            scores, float(strength if strength is not None else 0.1), groups
        )
        return {
            "kind": method,
            "member_weights": weights.tolist(),
            "study_count": int(len(np.unique(groups))),
            "temperature": float(strength if strength is not None else 0.1),
            "n_members": int(values.shape[0]),
            "n_classes": int(values.shape[2]),
        }
    raise ValueError(f"Unknown probabilistic aggregation {method!r}.")


def predict_probabilistic(
    stack: np.ndarray, parameters: Mapping[str, Any]
) -> np.ndarray:
    values = _validated(stack)
    if (
        str(parameters.get("kind")) not in METHODS - {"mixture_of_experts"}
        or int(parameters.get("n_members", -1)) != values.shape[0]
        or int(parameters.get("n_classes", -1)) != values.shape[2]
    ):
        raise ValueError(
            "Stored probabilistic ensemble parameters do not match predictions."
        )
    weights = np.asarray(parameters.get("member_weights"), dtype=float)
    if (
        weights.shape != (values.shape[0],)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid stored probabilistic ensemble weights.")
    weights /= weights.sum()
    return np.tensordot(weights, values, axes=(0, 0))
