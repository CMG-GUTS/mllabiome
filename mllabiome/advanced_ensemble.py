from __future__ import annotations

import json
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, softmax
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

ADVANCED_AGGREGATIONS = frozenset(
    {
        "rf_stacking",
        "extra_trees_stacking",
        "boosted_stacking",
        "gated_stacking",
        "hierarchical_dirichlet_stacking",
        "temperature_scaled_mean_proba",
        "sigmoid_calibrated_mean_proba",
    }
)


def _probability_matrix(stack: np.ndarray) -> np.ndarray:
    values = np.asarray(stack, dtype=float)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] < 2:
        raise ValueError(
            "Advanced ensembles require at least two members and two classes."
        )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(
            "Advanced ensemble predictions must be finite and nonnegative."
        )
    sums = values.sum(axis=2, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Advanced ensemble probability rows must have positive mass.")
    return values / sums


def _features(stack: np.ndarray) -> np.ndarray:
    values = np.clip(_probability_matrix(stack), 1e-6, 1.0)
    logarithms = np.log(values[:, :, :-1]) - np.log(values[:, :, -1:])
    return logarithms.transpose(1, 0, 2).reshape(values.shape[1], -1)


def _context_features(stack: np.ndarray) -> np.ndarray:
    values = _probability_matrix(stack)
    flattened = _features(stack)
    consensus = np.mean(values, axis=0)
    dispersion = np.mean(np.std(values, axis=0), axis=1)
    confidence = np.max(consensus, axis=1)
    return np.column_stack((flattened, confidence - 0.5, dispersion))


def _contexts(stack: np.ndarray) -> np.ndarray:
    values = _probability_matrix(stack)
    consensus = np.mean(values, axis=0)
    confidence = np.max(consensus, axis=1)
    disagreement = np.mean(np.std(values, axis=0), axis=1)
    return (confidence >= 0.7).astype(int) + 2 * (disagreement >= 0.1).astype(int)


def _tree_record(estimator: Any) -> dict[str, Any]:
    tree = estimator.tree_
    return {
        "left": tree.children_left.astype(int).tolist(),
        "right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": np.asarray(tree.value, dtype=float)[:, 0, :].tolist(),
    }


def _tree_values(features: np.ndarray, record: Mapping[str, Any]) -> np.ndarray:
    left = np.asarray(record["left"], dtype=int)
    right = np.asarray(record["right"], dtype=int)
    indices = np.asarray(record["feature"], dtype=int)
    thresholds = np.asarray(record["threshold"], dtype=float)
    values = np.asarray(record["value"], dtype=float)
    if not (len(left) == len(right) == len(indices) == len(thresholds) == len(values)):
        raise ValueError("Malformed serialized decision tree.")
    if np.any((left >= len(left)) | (right >= len(left))):
        raise ValueError("Serialized decision tree points outside its node array.")
    nodes = np.zeros(len(features), dtype=int)
    for _ in range(len(left)):
        active = indices[nodes] >= 0
        if not np.any(active):
            return values[nodes]
        rows = np.flatnonzero(active)
        current = nodes[rows]
        nodes[rows] = np.where(
            features[rows, indices[current]] <= thresholds[current],
            left[current],
            right[current],
        )
    raise ValueError("Serialized decision tree contains a cycle.")


def _fit_meta_tree(
    method: str, stack: np.ndarray, y_true: np.ndarray
) -> dict[str, Any]:
    features = _features(stack)
    n = len(y_true)
    leaf = max(4, min(20, n // 9))
    if method == "rf_stacking":
        estimator = RandomForestClassifier(
            n_estimators=160,
            max_depth=2,
            min_samples_leaf=leaf,
            max_features=0.8,
            bootstrap=True,
            random_state=42,
            n_jobs=1,
        )
    elif method == "extra_trees_stacking":
        estimator = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=2,
            min_samples_leaf=leaf,
            max_features=0.8,
            bootstrap=False,
            random_state=42,
            n_jobs=1,
        )
    else:
        estimator = GradientBoostingClassifier(
            n_estimators=70,
            learning_rate=0.04,
            max_depth=1,
            min_samples_leaf=leaf,
            subsample=0.85,
            random_state=42,
        )
    estimator.fit(features, y_true)
    classes = np.asarray(estimator.classes_, dtype=int)
    if not np.array_equal(classes, np.arange(stack.shape[2], dtype=int)):
        raise ValueError("Meta-learner did not retain all outcome classes.")
    if method == "boosted_stacking":
        prior = np.asarray(estimator.init_.class_prior_, dtype=float)
        return {
            "kind": method,
            "n_members": int(stack.shape[0]),
            "n_classes": int(stack.shape[2]),
            "learning_rate": float(estimator.learning_rate),
            "prior": prior.tolist(),
            "trees": [
                [_tree_record(tree) for tree in stage]
                for stage in estimator.estimators_
            ],
            "blend": 0.15,
        }
    return {
        "kind": method,
        "n_members": int(stack.shape[0]),
        "n_classes": int(stack.shape[2]),
        "trees": [_tree_record(tree) for tree in estimator.estimators_],
        "blend": 0.15,
    }


def _predict_meta_tree(stack: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    features = _features(stack)
    method = str(params["kind"])
    classes = int(params["n_classes"])
    if method == "boosted_stacking":
        prior = np.maximum(np.asarray(params["prior"], dtype=float), 1e-12)
        if classes == 2:
            raw = np.full(len(features), np.log(prior[1] / prior[0]))
            for stage in params["trees"]:
                raw += (
                    float(params["learning_rate"])
                    * _tree_values(features, stage[0])[:, 0]
                )
            positive = expit(raw)
            predicted = np.column_stack((1.0 - positive, positive))
        else:
            raw = np.tile(np.log(prior), (len(features), 1))
            for stage in params["trees"]:
                for class_index, record in enumerate(stage):
                    raw[:, class_index] += (
                        float(params["learning_rate"])
                        * _tree_values(features, record)[:, 0]
                    )
            predicted = softmax(raw, axis=1)
    else:
        predicted = np.zeros((len(features), classes), dtype=float)
        trees = params["trees"]
        if not trees:
            raise ValueError("Serialized forest cannot be empty.")
        for record in trees:
            leaves = _tree_values(features, record)
            totals = leaves.sum(axis=1, keepdims=True)
            if np.any(totals <= 0) or leaves.shape[1] != classes:
                raise ValueError("Invalid serialized forest leaf values.")
            predicted += leaves / totals
        predicted /= float(len(trees))
    blend = float(params["blend"])
    return (1.0 - blend) * predicted + blend * np.mean(stack, axis=0)


def _fit_temperature(
    stack: np.ndarray, labels: np.ndarray, method: str
) -> dict[str, Any]:
    pooled = np.clip(np.mean(_probability_matrix(stack), axis=0), 1e-8, 1.0)
    logits = np.log(pooled)
    rows = np.arange(len(labels), dtype=int)
    if method == "temperature_scaled_mean_proba":

        def loss(value: float) -> float:
            prob = softmax(logits / float(value), axis=1)
            return float(-np.mean(np.log(np.maximum(prob[rows, labels], 1e-12))))

        fitted = minimize_scalar(loss, bounds=(0.5, 4.0), method="bounded")
        if not fitted.success:
            raise RuntimeError("Temperature optimization did not converge.")
        return {
            "kind": method,
            "temperature": float(fitted.x),
            "n_members": int(stack.shape[0]),
            "n_classes": int(stack.shape[2]),
        }
    classes = int(stack.shape[2])

    def loss(theta: np.ndarray) -> float:
        slope = float(theta[0])
        offsets = np.concatenate((theta[1:], [0.0]))
        predicted = softmax(slope * logits + offsets, axis=1)
        nll = -np.mean(np.log(np.maximum(predicted[rows, labels], 1e-12)))
        return float(nll + 0.02 * (slope - 1.0) ** 2 + 0.02 * np.sum(offsets**2))

    initial = np.concatenate(([1.0], np.zeros(classes - 1)))
    bounds = [(0.1, 4.0)] + [(-2.0, 2.0)] * (classes - 1)
    fitted = minimize(loss, initial, method="L-BFGS-B", bounds=bounds)
    if not fitted.success:
        raise RuntimeError("Sigmoid calibration optimization did not converge.")
    return {
        "kind": method,
        "slope": float(fitted.x[0]),
        "offsets": np.concatenate((fitted.x[1:], [0.0])).tolist(),
        "n_members": int(stack.shape[0]),
        "n_classes": classes,
    }


def _predict_temperature(stack: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    pooled = np.clip(np.mean(stack, axis=0), 1e-8, 1.0)
    logits = np.log(pooled)
    if params["kind"] == "temperature_scaled_mean_proba":
        return softmax(logits / float(params["temperature"]), axis=1)
    return softmax(
        float(params["slope"]) * logits + np.asarray(params["offsets"]), axis=1
    )


def _fit_gating(
    stack: np.ndarray, labels: np.ndarray, strength: float
) -> dict[str, Any]:
    values = _probability_matrix(stack)
    features = _context_features(values)
    features = np.column_stack((np.ones(len(features)), np.clip(features, -8, 8)))
    n_members = int(values.shape[0])
    n_features = features.shape[1]
    floor = 0.01
    targets = values[:, np.arange(len(labels)), labels].T

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coef = theta.reshape(n_features, n_members)
        raw = softmax(features @ coef, axis=1)
        weights = floor + (1.0 - floor * n_members) * raw
        proba = np.maximum(np.sum(weights * targets, axis=1), 1e-12)
        value = -np.mean(np.log(proba)) + 0.5 * strength * np.sum(coef**2)
        gradient_weights = -targets / proba[:, None] / len(labels)
        gradient_logits = (
            (1.0 - floor * n_members)
            * raw
            * (gradient_weights - np.sum(gradient_weights * raw, axis=1, keepdims=True))
        )
        gradient = features.T @ gradient_logits + strength * coef
        return float(value), gradient.ravel()

    fitted = minimize(
        loss,
        np.zeros(n_members * n_features),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError("Gated stacking optimization did not converge.")
    return {
        "kind": "gated_stacking",
        "coef": fitted.x.reshape(n_features, n_members).tolist(),
        "floor": floor,
        "n_members": n_members,
        "n_classes": int(values.shape[2]),
        "strength": float(strength),
    }


def _predict_gating(stack: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    values = _probability_matrix(stack)
    features = np.column_stack(
        (np.ones(values.shape[1]), np.clip(_context_features(values), -8, 8))
    )
    coef = np.asarray(params["coef"], dtype=float)
    if coef.shape != (features.shape[1], values.shape[0]):
        raise ValueError(
            "Gating parameter dimensions disagree with member predictions."
        )
    floor = float(params["floor"])
    weights = floor + (1.0 - floor * values.shape[0]) * softmax(features @ coef, axis=1)
    return np.einsum("nm,mnc->nc", weights, values, optimize=True)


def _fit_simplex(
    stack: np.ndarray,
    labels: np.ndarray,
    prior: np.ndarray,
    strength: float,
    floor: float = 0.01,
) -> np.ndarray:
    values = _probability_matrix(stack)
    n_members = int(values.shape[0])
    if n_members * floor >= 1:
        raise ValueError("Positive ensemble floor is infeasible.")
    prior = np.asarray(prior, dtype=float)
    prior = np.maximum(prior, 1e-8)
    prior /= prior.sum()
    scores = values[:, np.arange(len(labels)), labels].T
    scale = 1.0 - n_members * floor

    def loss(v: np.ndarray) -> tuple[float, np.ndarray]:
        weights = floor + scale * v
        pooled = np.maximum(scores @ weights, 1e-12)
        objective = -np.mean(np.log(pooled)) - float(strength) * np.sum(
            prior * np.log(weights)
        )
        gradient = scale * (
            -np.mean(scores / pooled[:, None], axis=0) - strength * prior / weights
        )
        return float(objective), gradient

    initial = np.maximum(prior - floor, 0)
    initial = (
        initial / initial.sum()
        if initial.sum() > 0
        else np.full(n_members, 1.0 / n_members)
    )
    result = minimize(
        lambda x: loss(x)[0],
        initial,
        jac=lambda x: loss(x)[1],
        method="SLSQP",
        bounds=[(0, 1)] * n_members,
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - 1,
                "jac": lambda x: np.ones_like(x),
            }
        ],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(
            f"Hierarchical stacking optimization failed: {result.message}"
        )
    solution = np.maximum(result.x, 0)
    solution /= solution.sum()
    return floor + scale * solution


def _fit_hierarchical(
    stack: np.ndarray, labels: np.ndarray, strength: float
) -> dict[str, Any]:
    n_members = int(stack.shape[0])
    uniform = np.full(n_members, 1.0 / n_members)
    global_weights = _fit_simplex(stack, labels, uniform, 0.05)
    contexts = _contexts(stack)
    locals_: dict[str, list[float]] = {}
    for key in np.unique(contexts):
        mask = contexts == key
        if np.count_nonzero(mask) < 6:
            continue
        local = _fit_simplex(stack[:, mask], labels[mask], global_weights, strength)
        locals_[str(int(key))] = local.tolist()
    return {
        "kind": "hierarchical_dirichlet_stacking",
        "global_weights": global_weights.tolist(),
        "local_weights": locals_,
        "strength": float(strength),
        "n_members": n_members,
        "n_classes": int(stack.shape[2]),
        "floor": 0.01,
    }


def _predict_hierarchical(stack: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    values = _probability_matrix(stack)
    contexts = _contexts(values)
    global_weights = np.asarray(params["global_weights"], dtype=float)
    output = np.zeros(values.shape[1:], dtype=float)
    for context in np.unique(contexts):
        rows = contexts == context
        local = params["local_weights"].get(str(int(context)), global_weights)
        weights = np.asarray(local, dtype=float)
        if weights.shape != (values.shape[0],) or np.any(weights < 0.01 - 1e-8):
            raise ValueError(
                "Hierarchical ensemble weights violate the effective-member floor."
            )
        output[rows] = np.tensordot(weights, values[:, rows, :], axes=(0, 0))
    return output


def fit_advanced_model(
    method: str, stack: np.ndarray, y_true: np.ndarray, strength: float | None = None
) -> dict[str, Any]:
    values = _probability_matrix(stack)
    labels = np.asarray(y_true, dtype=int)
    if values.shape[1] != len(labels) or not np.array_equal(
        np.unique(labels), np.arange(values.shape[2])
    ):
        raise ValueError(
            "Advanced stacking requires matching labels and all outcome classes."
        )
    if method in {"rf_stacking", "extra_trees_stacking", "boosted_stacking"}:
        return _fit_meta_tree(method, values, labels)
    if method in {"temperature_scaled_mean_proba", "sigmoid_calibrated_mean_proba"}:
        return _fit_temperature(values, labels, method)
    if method == "gated_stacking":
        return _fit_gating(
            values, labels, float(strength if strength is not None else 0.1)
        )
    if method == "hierarchical_dirichlet_stacking":
        return _fit_hierarchical(
            values, labels, float(strength if strength is not None else 0.5)
        )
    raise ValueError(f"Unsupported advanced aggregation: {method!r}")


def apply_advanced_aggregation(
    stack: np.ndarray, parameters: Mapping[str, Any]
) -> np.ndarray:
    values = _probability_matrix(stack)
    if (
        not isinstance(parameters, Mapping)
        or str(parameters.get("kind")) not in ADVANCED_AGGREGATIONS
    ):
        raise ValueError(
            "Advanced aggregation requires valid stored meta-model parameters."
        )
    if (
        int(parameters["n_members"]) != values.shape[0]
        or int(parameters["n_classes"]) != values.shape[2]
    ):
        raise ValueError(
            "Advanced meta-model dimensions do not match selected members."
        )
    kind = str(parameters["kind"])
    if kind in {"rf_stacking", "extra_trees_stacking", "boosted_stacking"}:
        output = _predict_meta_tree(values, parameters)
    elif kind in {"temperature_scaled_mean_proba", "sigmoid_calibrated_mean_proba"}:
        output = _predict_temperature(values, parameters)
    elif kind == "gated_stacking":
        output = _predict_gating(values, parameters)
    else:
        output = _predict_hierarchical(values, parameters)
    if (
        not np.isfinite(output).all()
        or np.any(output < 0.0)
        or np.any(output.sum(axis=1) <= 0)
    ):
        raise ValueError("Advanced aggregation returned invalid probabilities.")
    return output / output.sum(axis=1, keepdims=True)


def fit_crossfitted_advanced(
    stack: np.ndarray,
    y_true: np.ndarray,
    fold_keys: np.ndarray,
    method: str,
    metric: str,
    score: Callable[[np.ndarray, np.ndarray, str], float],
    loss_metric: Callable[[str], bool],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    values = _probability_matrix(stack)
    labels = np.asarray(y_true, dtype=int)
    keys = np.asarray(fold_keys).astype(str)
    if len(keys) != len(labels):
        raise ValueError("Meta-model folds and labels have different lengths.")
    unique = np.unique(keys)
    if len(unique) < 2:
        raise ValueError("Advanced stacking requires at least two validation folds.")
    choices: tuple[float | None, ...] = (
        (0.05, 0.5)
        if method == "gated_stacking"
        else (0.25, 1.0)
        if method == "hierarchical_dirichlet_stacking"
        else (None,)
    )
    best: tuple[float, np.ndarray, float | None] | None = None
    trace: list[dict[str, Any]] = []
    for strength in choices:
        heldout = np.zeros((len(labels), values.shape[2]), dtype=float)
        valid = True
        for key in unique:
            train = keys != key
            test = ~train
            if not np.array_equal(np.unique(labels[train]), np.arange(values.shape[2])):
                valid = False
                break
            try:
                fitted = fit_advanced_model(
                    method, values[:, train], labels[train], strength
                )
                heldout[test] = apply_advanced_aggregation(values[:, test], fitted)
            except (ValueError, RuntimeError, FloatingPointError):
                valid = False
                break
        if (
            not valid
            or not np.isfinite(heldout).all()
            or np.any(heldout.sum(axis=1) <= 0)
        ):
            continue
        scored = float(score(labels, heldout, metric))
        if not np.isfinite(scored):
            continue
        trace.append({"strength": strength, "score": scored})
        if best is None or (
            scored < best[0] if loss_metric(metric) else scored > best[0]
        ):
            best = (scored, heldout, strength)
    if best is None:
        raise RuntimeError(
            f"{method} could not produce complete cross-fitted predictions."
        )
    chosen_score, crossfit, strength = best
    parameters = fit_advanced_model(method, values, labels, strength)
    diagnostics = {
        "stacking_crossfit_score": float(chosen_score),
        "stacking_internal_folds": int(len(unique)),
        "stacking_strength": "" if strength is None else float(strength),
        "stacking_search_trace": json.dumps(trace),
    }
    return crossfit, parameters, diagnostics
