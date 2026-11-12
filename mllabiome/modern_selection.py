from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


def _checked_inputs(
    stack: np.ndarray, y_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(stack, dtype=float)
    target = np.asarray(y_true, dtype=int)
    if (
        values.ndim != 3
        or values.shape[0] < 2
        or values.shape[1] != len(target)
        or values.shape[2] < 2
    ):
        raise ValueError(
            "Modern selectors require at least two models and aligned class probabilities."
        )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0 + 1e-8)
    ):
        raise ValueError("Modern selectors require finite, nonnegative probabilities.")
    if not np.allclose(values.sum(axis=2), 1.0, atol=1e-5):
        raise ValueError("Modern selectors require normalized class probabilities.")
    if len(target) < 2 or target.min() < 0 or target.max() >= values.shape[2]:
        raise ValueError(
            "Labels must be class indices aligned with probability columns."
        )
    return values, target


def _ensemble_probability(counts: np.ndarray, stack: np.ndarray) -> np.ndarray:
    positions = np.flatnonzero(counts)
    return np.tensordot(
        counts[positions] / counts[positions].sum(), stack[positions], axes=(0, 0)
    )


def _configuration_distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    return float(np.mean([a != b for a, b in zip(left, right)]))


def _behaviour(
    counts: np.ndarray,
    stack: np.ndarray,
    labels: np.ndarray,
    signatures: list[tuple[str, ...]],
) -> tuple[float, float]:
    positions = np.flatnonzero(counts)
    if len(positions) < 2:
        return 1.0, 0.0
    losses = (
        1.0
        - stack[positions[:, None], np.arange(len(labels))[None, :], labels[None, :]]
    )
    centered = losses - losses.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=1))
    correlation = np.divide(
        centered @ centered.T,
        norms[:, None] * norms[None, :],
        out=np.zeros((len(positions), len(positions))),
        where=(norms[:, None] * norms[None, :]) > 1e-12,
    )
    upper = np.triu_indices(len(positions), k=1)
    alc = float(np.clip(np.mean(correlation[upper]), -1.0, 1.0))
    css = float(
        np.mean(
            [
                _configuration_distance(signatures[int(i)], signatures[int(j)])
                for a, i in enumerate(positions)
                for j in positions[a + 1 :]
            ]
        )
    )
    return alc, css


def population_ensemble_select(
    library: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    *,
    higher_is_better: bool,
    max_unique: int,
    iterations: int,
    archive_size: int,
    seed: int,
    quality_diversity: bool = False,
    signatures: list[tuple[str, ...]] | None = None,
    bins: int = 6,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    values, labels = _checked_inputs(stack, y_true)
    n_models = len(library)
    if len(library) != len(values) or len(set(library)) != len(library):
        raise ValueError("Population selectors require distinct model IDs.")
    max_unique = min(n_models, int(max_unique))
    if max_unique < 2 or iterations < 1 or archive_size < 2 or bins < 2:
        raise ValueError("Invalid population-selection search controls.")
    if quality_diversity and (signatures is None or len(signatures) != n_models):
        raise ValueError("QDO-ES requires one configuration signature per model.")
    rng = np.random.default_rng(int(seed))
    seen: set[tuple[int, ...]] = set()
    archive: list[tuple[float, tuple[int, ...], tuple[float, float]]] = []
    best: tuple[float, tuple[int, ...], tuple[float, float]] | None = None
    evaluations = 0

    def evaluate(source: np.ndarray) -> None:
        nonlocal best, evaluations, archive
        key = tuple(int(x) for x in source)
        if key in seen or not sum(key):
            return
        seen.add(key)
        probability = _ensemble_probability(source, values)
        measured = float(score_fn(probability))
        if not math.isfinite(measured):
            return
        loss = -measured if higher_is_better else measured
        behaviour = (
            _behaviour(source, values, labels, signatures)
            if quality_diversity
            else (0.0, 0.0)
        )
        item = (loss, key, behaviour)
        evaluations += 1
        if sum(x > 0 for x in key) >= 2 and (
            best is None or (loss, key) < (best[0], best[1])
        ):
            best = item
        if quality_diversity:
            archive.append(item)
            buckets: dict[
                tuple[int, int], tuple[float, tuple[int, ...], tuple[float, float]]
            ] = {}
            for candidate in archive:
                a, b = candidate[2]
                cell = (
                    min(bins - 1, int((a + 1.0) * 0.5 * bins)),
                    min(bins - 1, int(b * bins)),
                )
                if cell not in buckets or candidate[:2] < buckets[cell][:2]:
                    buckets[cell] = candidate
            archive = sorted(buckets.values(), key=lambda entry: (entry[0], entry[1]))[
                :archive_size
            ]
        else:
            archive = sorted([*archive, item], key=lambda entry: (entry[0], entry[1]))[
                :archive_size
            ]

    starter = min(n_models, max(8, min(archive_size, 24)))
    for index in range(starter):
        seed_counts = np.zeros(n_models, dtype=np.int32)
        seed_counts[index] = 1
        evaluate(seed_counts)
    for index in range(1, min(starter, 12)):
        seed_counts = np.zeros(n_models, dtype=np.int32)
        seed_counts[0] = 1
        seed_counts[index] = 1
        evaluate(seed_counts)
    for _ in range(iterations):
        if not archive:
            break
        parent_ids = rng.choice(len(archive), size=min(3, len(archive)), replace=False)
        parent = min(
            (archive[int(i)] for i in parent_ids), key=lambda item: (item[0], item[1])
        )
        counts = np.array(parent[1], dtype=np.int32)
        if len(archive) >= 2 and rng.random() < 0.35:
            other = archive[int(rng.integers(len(archive)))][1]
            mask = rng.random(n_models) < 0.5
            counts[mask] = np.asarray(other, dtype=np.int32)[mask]
        if rng.random() < 0.40 and counts.sum() > 2:
            selected = np.flatnonzero(counts)
            counts[int(rng.choice(selected))] -= 1
        else:
            active = int(np.count_nonzero(counts))
            choices = (
                np.flatnonzero(counts) if active >= max_unique else np.arange(n_models)
            )
            if rng.random() < 0.7:
                head = choices[choices < max(2, n_models // 5)]
                if len(head):
                    choices = head
            counts[int(rng.choice(choices))] += 1
        if np.count_nonzero(counts) > max_unique:
            active = np.flatnonzero(counts)
            drop = int(rng.choice(active))
            counts[drop] = 0
        if counts.sum() == 0:
            counts[int(rng.integers(n_models))] = 1
        evaluate(counts)
    if best is None:
        raise ValueError("Population search produced no valid multi-member ensemble.")
    support = np.flatnonzero(best[1])
    weights = np.asarray(best[1], dtype=float)[support]
    weights /= weights.sum()
    return (
        [library[int(i)] for i in support],
        weights,
        {
            "population_method": "qdo_es" if quality_diversity else "qo_es",
            "population_evaluations": int(evaluations),
            "population_iterations": int(iterations),
            "population_archive_size": int(len(archive)),
            "population_search_seed": int(seed),
            "population_unique_member_cap": int(max_unique),
            "population_selected_count_total": int(sum(best[1])),
            "population_selected_loss": float(best[0]),
            "population_selected_alc": float(best[2][0]) if quality_diversity else None,
            "population_selected_configuration_distance": float(best[2][1])
            if quality_diversity
            else None,
            "population_search_variant": "bounded_archive_stochastic_crossover_mutation",
        },
    )


def drep_select(
    library: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    max_members: int,
    rho: float = 0.5,
) -> tuple[list[str], dict[str, Any]]:
    values, labels = _checked_inputs(stack, y_true)
    if not 0.0 < rho < 1.0 or max_members < 2:
        raise ValueError("DREP requires 0 < rho < 1 and max_members >= 2.")
    classes = np.argmax(values, axis=2)
    n = len(library)
    accuracy = np.mean(classes == labels[None, :], axis=1)
    first = int(np.argmax(accuracy))
    chosen = [first]
    remaining = set(range(n)) - {first}
    votes = np.eye(values.shape[2])[classes[first]].astype(float)
    current_errors = int(np.count_nonzero(np.argmax(votes, axis=1) != labels))
    trace = [current_errors]
    while remaining and len(chosen) < max_members:
        signs = 2.0 * np.eye(values.shape[2])[classes] - 1.0
        current_signs = 2.0 * votes / len(chosen) - 1.0
        distance = {
            i: float(np.mean(np.sum(signs[i] * current_signs, axis=1)))
            for i in remaining
        }
        eligible = sorted(remaining, key=lambda i: (distance[i], i))[
            : max(1, int(math.ceil(rho * len(remaining))))
        ]
        trial = [
            (
                int(
                    np.count_nonzero(
                        np.argmax(votes + np.eye(values.shape[2])[classes[i]], axis=1)
                        != labels
                    )
                ),
                i,
            )
            for i in eligible
        ]
        errors, picked = min(trial)
        if errors >= current_errors:
            break
        votes += np.eye(values.shape[2])[classes[picked]]
        chosen.append(picked)
        remaining.remove(picked)
        current_errors = errors
        trace.append(errors)
    return [library[i] for i in chosen], {
        "drep_rho": float(rho),
        "drep_validation_errors": trace,
        "drep_stop_reason": "member_limit"
        if len(chosen) >= max_members
        else "no_error_improvement",
        "drep_vote_mode": "hard_class_majority",
    }


def pseo_bqp_select(
    library: list[str],
    stack: np.ndarray,
    y_true: np.ndarray,
    max_members: int,
    omega: float = 0.5,
    swap_passes: int = 4,
) -> tuple[list[str], dict[str, Any]]:
    values, labels = _checked_inputs(stack, y_true)
    if not 0.0 <= omega <= 1.0 or max_members < 2:
        raise ValueError("PSEO BQP requires omega in [0, 1] and max_members >= 2.")
    n, n_samples, n_classes = values.shape
    errors = np.eye(n_classes)[labels][None, :, :] - values
    flat = errors.reshape(n, -1)
    gram = flat @ flat.T / n_samples
    costs = float(omega) * gram
    np.fill_diagonal(costs, (1.0 - float(omega)) * np.diag(gram))
    selected: list[int] = []
    available = set(range(n))
    for _ in range(min(max_members, n)):
        chosen = min(
            available,
            key=lambda index: (
                float(costs[index, index] + 2.0 * costs[index, selected].sum()),
                index,
            ),
        )
        selected.append(chosen)
        available.remove(chosen)
    swaps = 0
    for _ in range(max(0, int(swap_passes))):
        current = float(costs[np.ix_(selected, selected)].sum())
        replacement: tuple[float, int, int] | None = None
        for old in selected:
            rest = [i for i in selected if i != old]
            for candidate in available:
                delta = float(
                    costs[candidate, candidate]
                    - costs[old, old]
                    + 2.0 * (costs[candidate, rest].sum() - costs[old, rest].sum())
                )
                proposed = (delta, old, candidate)
                if delta < -1e-12 and (replacement is None or proposed < replacement):
                    replacement = proposed
        if replacement is None:
            break
        _, old, new = replacement
        selected[selected.index(old)] = new
        available.remove(new)
        available.add(old)
        swaps += 1
    selected = sorted(selected)
    return [library[i] for i in selected], {
        "pseo_bqp_omega": float(omega),
        "pseo_bqp_approximator": "greedy_forward_with_one_exchange_local_search",
        "pseo_bqp_swap_count": int(swaps),
        "pseo_bqp_objective": float(costs[np.ix_(selected, selected)].sum()),
        "pseo_bqp_library_size": int(n),
        "pseo_bqp_full_multilayer_stacking": False,
    }
