from __future__ import annotations

from typing import Any

import json

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .ensemble_aggregation import aggregate_member_predictions


def _validated_inputs(
    stack: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(stack, dtype=float)
    target = np.asarray(labels, dtype=int).reshape(-1)
    studies = np.asarray(groups).astype(str).reshape(-1)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] != 2:
        raise ValueError(
            "Learned ECDF quantile aggregation requires at least two binary members."
        )
    if values.shape[1] != len(target) or len(target) != len(studies):
        raise ValueError(
            "Member predictions, targets, and study assignments are misaligned."
        )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Member probabilities must be finite and nonnegative.")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-5):
        raise ValueError("Member probability rows must sum to one.")
    if len(np.unique(target)) != 2 or not set(target.tolist()).issubset({0, 1}):
        raise ValueError(
            "Learned ECDF quantile aggregation requires both target classes."
        )
    if len(np.unique(studies)) < 2:
        raise ValueError(
            "Study-held-out learning requires at least two distinct study groups."
        )
    return values, target, studies


def _sample_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    _, indices, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[indices].astype(float)
    unique_classes, labels, class_counts = np.unique(
        y, return_inverse=True, return_counts=True
    )
    if len(unique_classes) != 2:
        raise ValueError("Both classes are required to estimate ensemble weights.")
    weights /= class_counts[labels].astype(float) / float(len(y))
    return weights / weights.sum()


def _log_loss(proba: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    selected = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.dot(weights, np.log(selected)))


def _ecdf_columns(stack: np.ndarray, references: list[list[float]]) -> np.ndarray:
    columns = []
    for index, reference in enumerate(references):
        values = np.asarray(reference, dtype=float)
        scores = stack[index, :, 1]
        left = np.searchsorted(values, scores, side="left")
        right = np.searchsorted(values, scores, side="right")
        columns.append((left + right) / (2.0 * len(values)))
    return np.column_stack(columns)


def _fit_on_studies(
    stack: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    n_members = int(stack.shape[0])
    weights = _sample_weights(y, groups)
    reference_losses = [
        _log_loss(stack[index], y, weights) for index in range(n_members)
    ]
    anchor = int(np.argmin(reference_losses))
    references = [np.sort(stack[index, :, 1]).tolist() for index in range(n_members)]
    features = _ecdf_columns(stack, references)
    points = (np.arange(len(y), dtype=float) + 0.5) / len(y)
    distribution = np.asarray(references[anchor], dtype=float)

    def objective(candidate: np.ndarray) -> float:
        score = features @ candidate
        positives = np.interp(
            score,
            points,
            distribution,
            left=distribution[0],
            right=distribution[-1],
        )
        proba = np.column_stack((1.0 - positives, positives))
        return _log_loss(proba, y, weights)

    if n_members == 2:
        optimize = lambda fraction: objective(
            np.array([float(fraction), 1.0 - float(fraction)], dtype=float)
        )
        fitted = minimize_scalar(
            optimize,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-5},
        )
        options = [(optimize(0.0), 0.0), (optimize(1.0), 1.0)]
        if bool(fitted.success):
            options.append((float(fitted.fun), float(fitted.x)))
        _, fraction = min(options, key=lambda item: (item[0], item[1]))
        member_weights = np.array([fraction, 1.0 - fraction], dtype=float)
    else:
        constraints = [{"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}]
        starts = [np.full(n_members, 1.0 / n_members)]
        starts.extend(np.eye(n_members, dtype=float))
        solutions = []
        for start in starts:
            result = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_members,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if bool(result.success):
                solution = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
                solution /= solution.sum()
                solutions.append((objective(solution), solution.tolist(), solution))
        if not solutions:
            raise RuntimeError("Convex quantile ensemble weight optimization failed.")
        member_weights = min(solutions, key=lambda entry: (entry[0], entry[1]))[2]
    return {
        "ecdf_references": references,
        "member_weights": member_weights.tolist(),
        "reference_member_index": anchor,
        "weight_objective": "study_and_class_balanced_log_loss",
        "weight_objective_value": objective(member_weights),
        "reference_selection_losses": reference_losses,
        "reference_selection_rule": "minimum_study_and_class_balanced_log_loss",
        "training_samples": int(len(y)),
        "training_studies": int(len(np.unique(groups))),
    }


def fit_grouped_quantile(
    stack: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    values, targets, studies = _validated_inputs(stack, y, groups)
    predictions = np.empty((len(targets), 2), dtype=float)
    group_results = []
    for study in sorted(np.unique(studies).tolist()):
        valid = studies == study
        train = ~valid
        if len(np.unique(targets[train])) != 2:
            raise ValueError(
                "Study-held-out reference fitting requires both classes in each training partition."
            )
        parameters = _fit_on_studies(
            values[:, train, :], targets[train], studies[train]
        )
        predictions[valid] = aggregate_member_predictions(
            values[:, valid, :],
            "ecdf_learned_quantile",
            parameters=parameters,
        )
        group_results.append(
            {
                "study": str(study),
                "n_validation": int(valid.sum()),
                "training_samples": int(train.sum()),
                "member_weights": parameters["member_weights"],
                "reference_member_index": int(parameters["reference_member_index"]),
            }
        )
    fitted = _fit_on_studies(values, targets, studies)
    diagnostics = {
        "weight_learning_protocol": "leave_one_study_out",
        "weight_learning_groups": int(len(group_results)),
        "weight_learning_samples": int(len(targets)),
        "weight_learning_crossfit_log_loss": _log_loss(
            predictions,
            targets,
            _sample_weights(targets, studies),
        ),
        "weight_learning_folds": json.dumps(group_results, separators=(",", ":")),
    }
    return predictions, fitted, diagnostics
