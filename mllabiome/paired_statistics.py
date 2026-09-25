from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from .metrics import metric_is_loss
from .oof_statistics import _mean_metric_dicts, _oof_metrics, _prepare_oof_frame
from .statistics_common import _LODO_PROTOCOLS, _OOF_CONTRAST_METRICS, _stable_seed


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
            _oof_metrics(left, probability_valid_a),
            _oof_metrics(right, probability_valid_b),
        )
        left_cohort = []
        right_cohort = []
        for cluster in left["_cluster"].drop_duplicates().tolist():
            left_cohort.append(
                _oof_metrics(left[left["_cluster"].eq(cluster)], probability_valid_a)
            )
            right_cohort.append(
                _oof_metrics(right[right["_cluster"].eq(cluster)], probability_valid_b)
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
            _oof_metrics(left[left["_repeat"].eq(repeat)], probability_valid_a)
        )
        right_repeat.append(
            _oof_metrics(right[right["_repeat"].eq(repeat)], probability_valid_b)
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
                pd.concat(left_groups, ignore_index=True), probability_valid_a
            )
            right_pooled = _oof_metrics(
                pd.concat(right_groups, ignore_index=True), probability_valid_b
            )
            left_macro = _mean_metric_dicts(
                [_oof_metrics(group, probability_valid_a) for group in left_groups]
            )
            right_macro = _mean_metric_dicts(
                [_oof_metrics(group, probability_valid_b) for group in right_groups]
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
                _oof_metrics(group, probability_valid_a)
                for _, group in left_sample.groupby("_repeat", sort=True)
            ]
            right_metrics = [
                _oof_metrics(group, probability_valid_b)
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
    return storage


def _paired_contrast_rows(
    frames: dict[str, pd.DataFrame],
    protocol: str,
    probability_semantics: dict[str, dict[str, Any]],
    n_bootstrap: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    prepared = {
        strategy: _prepare_oof_frame(frame, protocol)
        for strategy, frame in frames.items()
    }
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
