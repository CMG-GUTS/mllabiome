from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .metrics import metric_is_loss
from .oof_statistics import (_lodo_cluster_subject_indices, _mean_metric_dicts,
                             _oof_metrics, _oof_metrics_arrays,
                             _point_oof_estimands, _prepare_oof_frame,
                             _probability_columns,
                             _sample_lodo_cluster_indices)
from .statistics_common import (_LODO_PROTOCOLS, _OOF_CONTRAST_METRICS,
                                _stable_seed)


def _oof_design(frame: pd.DataFrame, protocol: str) -> dict[str, int]:
    protocol_key = str(protocol).lower()
    design = {
        "n_rows": len(frame),
        "n_unique_samples": int(frame["sample_id"].nunique())
        if "sample_id" in frame.columns
        else 0,
        "n_unique_subjects": int(frame["_subject_id"].nunique())
        if "_subject_id" in frame.columns
        else 0,
        "n_outer_units": int(frame["outer_split_key"].nunique())
        if "outer_split_key" in frame.columns
        else 0,
        "n_repeats": int(frame["_repeat"].nunique())
        if "_repeat" in frame.columns
        else 0,
    }
    if protocol_key in _LODO_PROTOCOLS:
        design["n_cohorts"] = int(frame["_cluster"].nunique())
        design["n_unique_samples"] = int(
            frame[["_cluster", "sample_id"]].drop_duplicates().shape[0]
        )
    return design


def _coverage_row(strategy: str, frame: pd.DataFrame, protocol: str) -> dict[str, Any]:
    design = _oof_design(frame, protocol)
    return {"Strategy": strategy, **design}


def _match_key_columns(protocol: str) -> list[str]:
    return (
        ["_cluster", "sample_id"]
        if str(protocol).lower() in _LODO_PROTOCOLS
        else ["_repeat", "sample_id"]
    )


def _matched_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = _match_key_columns(protocol)
    left_keys = left[keys].astype(str).agg("\x1f".join, axis=1)
    right_keys = right[keys].astype(str).agg("\x1f".join, axis=1)
    common = set(left_keys).intersection(set(right_keys))
    left_match = left.loc[left_keys.isin(common)].copy()
    right_match = right.loc[right_keys.isin(common)].copy()
    left_match["_pair_key"] = left_match[keys].astype(str).agg("\x1f".join, axis=1)
    right_match["_pair_key"] = right_match[keys].astype(str).agg("\x1f".join, axis=1)
    left_match = left_match.sort_values("_pair_key", kind="mergesort").reset_index(
        drop=True
    )
    right_match = right_match.sort_values("_pair_key", kind="mergesort").reset_index(
        drop=True
    )
    if (
        len(left_match) != len(right_match)
        or left_match["_pair_key"].tolist() != right_match["_pair_key"].tolist()
    ):
        raise ValueError(
            "Could not align matched held-out predictions across strategies"
        )
    if not np.array_equal(
        left_match["y_true"].to_numpy(), right_match["y_true"].to_numpy()
    ):
        raise ValueError("Matched strategies disagree on held-out outcome labels")
    coverage = {
        "n_strategy_a": len(left),
        "n_strategy_b": len(right),
        "n_matched": len(left_match),
        "coverage_fraction_a": float(len(left_match) / len(left))
        if len(left)
        else np.nan,
        "coverage_fraction_b": float(len(right_match) / len(right))
        if len(right)
        else np.nan,
    }
    return left_match, right_match, coverage


def _subject_cluster_indices(
    frame: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    groups = [
        group.index.to_numpy(dtype=int)
        for _, group in frame.groupby("_subject_id", sort=True)
    ]
    chosen = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[int(index)] for index in chosen])


def _paired_resample_indices(
    frame: pd.DataFrame,
    protocol: str,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        groups = [group for _, group in frame.groupby("_cluster", sort=True)]
        chosen = rng.integers(0, len(groups), size=len(groups))
        return [_subject_cluster_indices(groups[int(index)], rng) for index in chosen]
    return [_subject_cluster_indices(frame, rng)]


def _paired_point_estimands(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protocol: str,
    probability_valid_a: bool,
    probability_valid_b: bool,
) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        pooled = (
            _oof_metrics(left, probability_valid_a, include_calibration=False),
            _oof_metrics(right, probability_valid_b, include_calibration=False),
        )
        left_cohort = []
        right_cohort = []
        for cluster in left["_cluster"].drop_duplicates().tolist():
            left_cohort.append(
                _oof_metrics(
                    left[left["_cluster"].eq(cluster)],
                    probability_valid_a,
                    include_calibration=False,
                )
            )
            right_cohort.append(
                _oof_metrics(
                    right[right["_cluster"].eq(cluster)],
                    probability_valid_b,
                    include_calibration=False,
                )
            )
        return {
            "pooled_sample_weighted": pooled,
            "cohort_macro_equal_weight": (
                _mean_metric_dicts(left_cohort),
                _mean_metric_dicts(right_cohort),
            ),
        }
    left_repeat = []
    right_repeat = []
    for repeat in left["_repeat"].drop_duplicates().tolist():
        left_repeat.append(
            _oof_metrics(
                left[left["_repeat"].eq(repeat)],
                probability_valid_a,
                include_calibration=False,
            )
        )
        right_repeat.append(
            _oof_metrics(
                right[right["_repeat"].eq(repeat)],
                probability_valid_b,
                include_calibration=False,
            )
        )
    return {
        "mean_repeat_pooled_oof": (
            _mean_metric_dicts(left_repeat),
            _mean_metric_dicts(right_repeat),
        )
    }


def _paired_bootstrap_advantages(
    left: pd.DataFrame,
    right: pd.DataFrame,
    protocol: str,
    probability_valid_a: bool,
    probability_valid_b: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
    progress_label: str = "paired OOF bootstrap",
) -> dict[str, dict[str, np.ndarray]]:
    protocol_key = str(protocol).lower()
    names = (
        ("pooled_sample_weighted", "cohort_macro_equal_weight")
        if protocol_key in _LODO_PROTOCOLS
        else ("mean_repeat_pooled_oof",)
    )
    storage = {
        name: {
            metric: np.full(int(n_bootstrap), np.nan, dtype=float)
            for metric in _OOF_CONTRAST_METRICS
        }
        for name in names
    }
    for bootstrap_index in range(int(n_bootstrap)):
        sampled_indices = _paired_resample_indices(left, protocol, rng)
        if protocol_key in _LODO_PROTOCOLS:
            left_groups = [
                left.iloc[index].reset_index(drop=True) for index in sampled_indices
            ]
            right_groups = [
                right.iloc[index].reset_index(drop=True) for index in sampled_indices
            ]
            left_pooled = _oof_metrics(
                pd.concat(left_groups, ignore_index=True),
                probability_valid_a,
                include_calibration=False,
            )
            right_pooled = _oof_metrics(
                pd.concat(right_groups, ignore_index=True),
                probability_valid_b,
                include_calibration=False,
            )
            left_macro = _mean_metric_dicts(
                [
                    _oof_metrics(group, probability_valid_a, include_calibration=False)
                    for group in left_groups
                ]
            )
            right_macro = _mean_metric_dicts(
                [
                    _oof_metrics(group, probability_valid_b, include_calibration=False)
                    for group in right_groups
                ]
            )
            pairs = {
                "pooled_sample_weighted": (left_pooled, right_pooled),
                "cohort_macro_equal_weight": (left_macro, right_macro),
            }
        else:
            index = sampled_indices[0]
            left_sample = left.iloc[index].reset_index(drop=True)
            right_sample = right.iloc[index].reset_index(drop=True)
            left_metrics = [
                _oof_metrics(group, probability_valid_a, include_calibration=False)
                for _, group in left_sample.groupby("_repeat", sort=True)
            ]
            right_metrics = [
                _oof_metrics(group, probability_valid_b, include_calibration=False)
                for _, group in right_sample.groupby("_repeat", sort=True)
            ]
            pairs = {
                "mean_repeat_pooled_oof": (
                    _mean_metric_dicts(left_metrics),
                    _mean_metric_dicts(right_metrics),
                )
            }
        for estimand, (metric_a, metric_b) in pairs.items():
            for metric in _OOF_CONTRAST_METRICS:
                a = float(metric_a.get(metric, np.nan))
                b = float(metric_b.get(metric, np.nan))
                if not np.isfinite(a) or not np.isfinite(b):
                    continue
                storage[estimand][metric][bootstrap_index] = (
                    b - a if metric_is_loss(metric) else a - b
                )
        completed = bootstrap_index + 1
        update_every = max(1, int(n_bootstrap) // 100)
        if progress_callback is not None and (
            completed == 1
            or completed == int(n_bootstrap)
            or completed % update_every == 0
        ):
            progress_callback(progress_label, completed, int(n_bootstrap))
    return storage


def _shared_lodo_pairwise_bootstrap(
    prepared: dict[str, pd.DataFrame],
    probability_semantics: dict[str, dict[str, Any]],
    n_bootstrap: int,
    rng: np.random.Generator,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    strategies = list(prepared)
    storage = {
        strategy: {
            estimand: {
                metric: np.full(int(n_bootstrap), np.nan, dtype=float)
                for metric in _OOF_CONTRAST_METRICS
            }
            for estimand in (
                "pooled_sample_weighted",
                "cohort_macro_equal_weight",
            )
        }
        for strategy in strategies
    }
    if not strategies or int(n_bootstrap) <= 0:
        return storage
    base = prepared[strategies[0]]
    clusters = _lodo_cluster_subject_indices(base)
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = {}
    for strategy in strategies:
        frame = prepared[strategy]
        pcols = _probability_columns(frame)
        arrays[strategy] = (
            frame["y_true"].astype(int).to_numpy(),
            frame["y_pred"].astype(int).to_numpy(),
            frame[pcols].to_numpy(dtype=float),
            bool(probability_semantics[strategy].get("valid", False)),
        )
    n_bootstrap = int(n_bootstrap)
    update_every = max(1, n_bootstrap // 100)
    label = f"LODO paired bootstrap · {len(strategies)} strategies"
    for bootstrap_index in range(n_bootstrap):
        sampled_indices = _sample_lodo_cluster_indices(clusters, rng)
        pooled_index = np.concatenate(sampled_indices)
        for strategy in strategies:
            y_true, y_pred, score, probability_valid = arrays[strategy]
            pooled = _oof_metrics_arrays(
                y_true[pooled_index],
                y_pred[pooled_index],
                score[pooled_index],
                probability_valid,
                include_calibration=False,
            )
            macro = _mean_metric_dicts(
                [
                    _oof_metrics_arrays(
                        y_true[index],
                        y_pred[index],
                        score[index],
                        probability_valid,
                        include_calibration=False,
                    )
                    for index in sampled_indices
                ]
            )
            for metric in _OOF_CONTRAST_METRICS:
                storage[strategy]["pooled_sample_weighted"][metric][bootstrap_index] = (
                    pooled.get(metric, np.nan)
                )
                storage[strategy]["cohort_macro_equal_weight"][metric][
                    bootstrap_index
                ] = macro.get(metric, np.nan)
        completed = bootstrap_index + 1
        if progress_callback is not None and (
            completed == 1 or completed == n_bootstrap or completed % update_every == 0
        ):
            progress_callback(label, completed, n_bootstrap)
    return storage


def _aligned_lodo_frames(
    prepared: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame] | None:
    strategies = list(prepared)
    if len(strategies) < 2:
        return None
    keys = ["_cluster", "sample_id"]
    aligned: dict[str, pd.DataFrame] = {}
    reference_keys: np.ndarray | None = None
    reference_y: np.ndarray | None = None
    for strategy in strategies:
        frame = prepared[strategy].copy()
        frame["_shared_pair_key"] = frame[keys].astype(str).agg("\x1f".join, axis=1)
        if frame["_shared_pair_key"].duplicated().any():
            return None
        frame = frame.sort_values("_shared_pair_key", kind="mergesort").reset_index(
            drop=True
        )
        current_keys = frame["_shared_pair_key"].to_numpy()
        current_y = frame["y_true"].to_numpy()
        if reference_keys is None:
            reference_keys = current_keys
            reference_y = current_y
        elif (
            len(current_keys) != len(reference_keys)
            or not np.array_equal(current_keys, reference_keys)
            or not np.array_equal(current_y, reference_y)
        ):
            return None
        aligned[strategy] = frame.drop(columns=["_shared_pair_key"])
    return aligned


def _paired_contrast_rows(
    frames: dict[str, pd.DataFrame],
    protocol: str,
    probability_semantics: dict[str, dict[str, Any]],
    n_bootstrap: int,
    random_state: int,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    prepared = {
        strategy: _prepare_oof_frame(frame, protocol)
        for strategy, frame in frames.items()
    }
    protocol_key = str(protocol).lower()
    shared_bootstrap = None
    shared_points: dict[str, dict[str, dict[str, float]]] = {}
    aligned_lodo = (
        _aligned_lodo_frames(prepared) if protocol_key in _LODO_PROTOCOLS else None
    )
    if aligned_lodo is not None:
        for strategy, frame in aligned_lodo.items():
            shared_points[strategy] = _point_oof_estimands(
                frame,
                protocol,
                bool(probability_semantics[strategy].get("valid", False)),
                include_calibration=False,
            )
        shared_bootstrap = _shared_lodo_pairwise_bootstrap(
            aligned_lodo,
            probability_semantics,
            n_bootstrap,
            np.random.default_rng(
                _stable_seed(random_state, "paired_oof_shared", *prepared.keys())
            ),
            progress_callback=progress_callback,
        )
    for strategy_a, strategy_b in itertools.combinations(prepared, 2):
        left, right, coverage = _matched_frames(
            prepared[strategy_a], prepared[strategy_b], protocol
        )
        coverage_rows.append(
            {"strategy_a": strategy_a, "strategy_b": strategy_b, **coverage}
        )
        if left.empty:
            continue
        valid_a = bool(probability_semantics[strategy_a].get("valid", False))
        valid_b = bool(probability_semantics[strategy_b].get("valid", False))
        if shared_bootstrap is not None:
            point = {
                estimand: (
                    shared_points[strategy_a][estimand],
                    shared_points[strategy_b][estimand],
                )
                for estimand in shared_points[strategy_a]
            }
            boot = {
                estimand: {
                    metric: (
                        shared_bootstrap[strategy_b][estimand][metric]
                        - shared_bootstrap[strategy_a][estimand][metric]
                        if metric_is_loss(metric)
                        else shared_bootstrap[strategy_a][estimand][metric]
                        - shared_bootstrap[strategy_b][estimand][metric]
                    )
                    for metric in _OOF_CONTRAST_METRICS
                }
                for estimand in shared_points[strategy_a]
            }
        else:
            point = _paired_point_estimands(left, right, protocol, valid_a, valid_b)
            boot = _paired_bootstrap_advantages(
                left,
                right,
                protocol,
                valid_a,
                valid_b,
                n_bootstrap,
                np.random.default_rng(
                    _stable_seed(random_state, "paired_oof", strategy_a, strategy_b)
                ),
                progress_callback=progress_callback,
                progress_label=f"paired OOF bootstrap · {strategy_a} vs {strategy_b}",
            )
        for estimand, (metric_a, metric_b) in point.items():
            for metric in _OOF_CONTRAST_METRICS:
                a = float(metric_a.get(metric, np.nan))
                b = float(metric_b.get(metric, np.nan))
                if not np.isfinite(a) or not np.isfinite(b):
                    continue
                difference = a - b
                advantage = b - a if metric_is_loss(metric) else difference
                samples = np.asarray(boot[estimand][metric], dtype=float)
                samples = samples[np.isfinite(samples)]
                low = high = np.nan
                if len(samples):
                    low, high = np.quantile(samples, [0.025, 0.975])
                rows.append(
                    {
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "estimand": estimand,
                        "metric": metric,
                        "estimate_a": a,
                        "estimate_b": b,
                        "difference_a_minus_b": difference,
                        "advantage_a_over_b": advantage,
                        "advantage_ci_low": float(low) if np.isfinite(low) else np.nan,
                        "advantage_ci_high": float(high)
                        if np.isfinite(high)
                        else np.nan,
                        "n_bootstrap_valid": len(samples),
                        "n_matched": len(left),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)
