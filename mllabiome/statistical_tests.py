from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

_LODO_PROTOCOLS = frozenset({"lodo", "leave_one_dataset_out"})


def _repeat_id(split_key: str) -> str:
    match = re.match(r"^r(\d+)_o\d+$", str(split_key))
    return f"r{match.group(1)}" if match else "r0"


def _paired_bootstrap_difference(
    merged: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    protocol: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    frame = merged[["outer_split_key", metric_a, metric_b]].copy()
    frame[metric_a] = pd.to_numeric(frame[metric_a], errors="coerce")
    frame[metric_b] = pd.to_numeric(frame[metric_b], errors="coerce")
    frame = frame[
        np.isfinite(frame[metric_a].to_numpy(dtype=float))
        & np.isfinite(frame[metric_b].to_numpy(dtype=float))
    ]
    if frame.empty:
        return float("nan"), float("nan")
    frame["diff"] = frame[metric_a] - frame[metric_b]
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        differences = frame["diff"].to_numpy(dtype=float)
        n = len(differences)
        indices = rng.integers(0, n, size=(int(n_bootstrap), n))
        boot = differences[indices].mean(axis=1)
    else:
        frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
        groups = [
            group["diff"].to_numpy(dtype=float)
            for _, group in frame.groupby("repeat", sort=True)
        ]
        boot = np.empty(int(n_bootstrap), dtype=float)
        n_repeats = len(groups)
        for index in range(int(n_bootstrap)):
            sampled_repeat_indices = rng.integers(0, n_repeats, size=n_repeats)
            sampled: list[float] = []
            for repeat_index in sampled_repeat_indices:
                values_array = groups[int(repeat_index)]
                sampled.extend(
                    values_array[
                        rng.integers(0, len(values_array), size=len(values_array))
                    ].tolist()
                )
            boot[index] = float(np.mean(sampled))
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_test(
    differences: np.ndarray, random_state: int
) -> tuple[float, float, str]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), "paired_sign_flip"
    observed = float(np.mean(values))
    if n <= 16:
        total = 1 << n
        extreme = 0
        for bits in range(total):
            signs = np.fromiter(
                (1.0 if bits & (1 << index) else -1.0 for index in range(n)),
                dtype=float,
                count=n,
            )
            if abs(float(np.mean(signs * values))) >= abs(observed) - 1e-15:
                extreme += 1
        return observed, float(extreme / total), "exact_paired_sign_flip"
    rng = np.random.default_rng(int(random_state))
    n_permutations = 50000
    extreme = 0
    batch_size = 2000
    completed = 0
    while completed < n_permutations:
        current = min(batch_size, n_permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current, n))
        permuted = np.mean(signs * values[None, :], axis=1)
        extreme += int(np.sum(np.abs(permuted) >= abs(observed) - 1e-15))
        completed += current
    return (
        observed,
        float((extreme + 1) / (n_permutations + 1)),
        "monte_carlo_paired_sign_flip",
    )


def _corrected_resampled_t_test(
    merged: pd.DataFrame, metric_a: str, metric_b: str
) -> tuple[float, float, str]:
    columns = ["outer_split_key", metric_a, metric_b]
    for column in ("n_train_a", "n_test_a", "n_train_b", "n_test_b"):
        if column in merged.columns:
            columns.append(column)
    frame = merged[columns].copy()
    frame[metric_a] = pd.to_numeric(frame[metric_a], errors="coerce")
    frame[metric_b] = pd.to_numeric(frame[metric_b], errors="coerce")
    frame = frame[
        np.isfinite(frame[metric_a].to_numpy(dtype=float))
        & np.isfinite(frame[metric_b].to_numpy(dtype=float))
    ]
    if len(frame) < 2:
        return float("nan"), float("nan"), "nadeau_bengio_corrected_t"
    differences = (frame[metric_a] - frame[metric_b]).to_numpy(dtype=float)
    mean_difference = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1))
    ratios: list[np.ndarray] = []
    for suffix in ("a", "b"):
        train_column = f"n_train_{suffix}"
        test_column = f"n_test_{suffix}"
        if train_column not in frame.columns or test_column not in frame.columns:
            continue
        train = pd.to_numeric(frame[train_column], errors="coerce").to_numpy(
            dtype=float
        )
        test = pd.to_numeric(frame[test_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(train) & np.isfinite(test) & (train > 0) & (test > 0)
        if bool(valid.any()):
            ratios.append(test[valid] / train[valid])
    if ratios:
        test_train_ratio = float(np.mean(np.concatenate(ratios)))
    else:
        frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
        fold_counts = (
            frame.groupby("repeat")["outer_split_key"].nunique().to_numpy(dtype=float)
        )
        k = (
            int(round(float(np.median(fold_counts))))
            if len(fold_counts)
            else len(frame)
        )
        if k < 2:
            return mean_difference, float("nan"), "nadeau_bengio_corrected_t"
        test_train_ratio = 1.0 / float(k - 1)
    corrected_variance = (1.0 / len(differences) + test_train_ratio) * variance
    if corrected_variance <= 0:
        p_value = 1.0 if abs(mean_difference) <= 1e-15 else 0.0
        return mean_difference, p_value, "nadeau_bengio_corrected_t"
    statistic = mean_difference / math.sqrt(corrected_variance)
    p_value = float(2.0 * student_t.sf(abs(statistic), df=len(differences) - 1))
    return float(statistic), p_value, "nadeau_bengio_corrected_t"
