from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._config_store import read_configuration
from .ensemble_aggregation import aggregate_member_predictions
from .metrics import compute_metrics
from .storage import read_table, table_exists, write_table
from .utils import dump_json_standard


def _specification(root: Path, strategy: str) -> dict[str, Any]:
    relative = (
        "tables/mpma_b_final_candidate.json"
        if strategy == "MPMA-B"
        else "ensembling/mpma_e_final_candidate.json"
    )
    path = root / relative
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid final model specification in {path}")
    return value


def _prediction_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        str(column) for column in frame.columns if str(column).startswith("proba_")
    ]
    columns = sorted(columns, key=lambda column: int(column.removeprefix("proba_")))
    if not columns or columns != [f"proba_{index}" for index in range(len(columns))]:
        raise ValueError(
            "Fixed-final evaluation requires consecutive probability columns"
        )
    return columns


def _member_outer_predictions(root: Path, config_id: str) -> pd.DataFrame:
    frame = read_configuration(root, config_id, "outer_predictions")
    if frame.empty:
        path = root / "predictions" / "outer_predictions.parquet"
        if table_exists(path):
            frame = read_table(path)
            if "config_id" not in frame.columns:
                raise ValueError("Outer predictions have no configuration identifier")
            frame = frame[frame["config_id"].astype(str).eq(str(config_id))].copy()
    if frame.empty:
        raise ValueError(f"Missing outer predictions for final member {config_id}")
    if "outer_split_key" not in frame.columns:
        if "split_key" not in frame.columns:
            raise ValueError("Outer predictions have no split identifier")
        frame = frame.assign(outer_split_key=frame["split_key"].astype(str))
    else:
        frame["outer_split_key"] = frame["outer_split_key"].astype(str)
    required = {"outer_split_key", "sample_id", "y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Incomplete outer prediction schema for {config_id}")
    _prediction_columns(frame)
    return frame


def _fixed_mpma_b(root: Path, specification: dict[str, Any]) -> pd.DataFrame:
    config_id = str(specification.get("config_id", "")).strip()
    if not config_id:
        raise ValueError("Final MPMA-B specification has no config_id")
    result = _member_outer_predictions(root, config_id)
    from .evaluation_predictions import _align_softened_mpma_b_positive_probability

    return _align_softened_mpma_b_positive_probability(result)


def _fixed_mpma_e(root: Path, specification: dict[str, Any]) -> pd.DataFrame:
    members = specification.get("members", [])
    if isinstance(members, str):
        members = json.loads(members)
    if (
        not isinstance(members, list)
        or not members
        or len(set(map(str, members))) != len(members)
    ):
        raise ValueError("Final MPMA-E must specify unique ensemble members")
    members = [str(member) for member in members]
    prepared = [_member_outer_predictions(root, member) for member in members]
    probability_columns = _prediction_columns(prepared[0])
    if any(_prediction_columns(frame) != probability_columns for frame in prepared[1:]):
        raise ValueError("Final MPMA-E members use incompatible class probabilities")
    identity = (
        "sample_index"
        if all("sample_index" in frame.columns for frame in prepared)
        else "sample_id"
    )
    key = ["outer_split_key", identity]
    reference = prepared[0]
    if any(frame.duplicated(key).any() for frame in prepared):
        raise ValueError("Final MPMA-E members contain duplicate outer observations")
    expected = set(map(tuple, reference[key].astype(str).to_numpy()))
    if any(
        set(map(tuple, frame[key].astype(str).to_numpy())) != expected
        for frame in prepared[1:]
    ):
        raise ValueError(
            "Final MPMA-E members do not cover identical outer observations"
        )
    ordered = reference.copy()
    for column in key:
        ordered[column] = ordered[column].astype(str)
    ordered = ordered.sort_values(key).reset_index(drop=True)
    aligned = []
    for frame in prepared:
        frame = frame.copy()
        for column in key:
            frame[column] = frame[column].astype(str)
        indexed = frame.set_index(key)
        observation_keys = pd.MultiIndex.from_frame(ordered[key], names=key)
        aligned_frame = indexed.reindex(observation_keys)
        if aligned_frame[probability_columns].isna().any().any():
            raise ValueError("Final MPMA-E member probabilities contain missing values")
        if not np.array_equal(
            pd.to_numeric(aligned_frame["y_true"], errors="raise").to_numpy(),
            pd.to_numeric(ordered["y_true"], errors="raise").to_numpy(),
        ):
            raise ValueError("Final MPMA-E member ground-truth labels disagree")
        if "sample_id" in aligned_frame and not np.array_equal(
            aligned_frame["sample_id"].astype(str).to_numpy(),
            ordered["sample_id"].astype(str).to_numpy(),
        ):
            raise ValueError("Final MPMA-E member sample identifiers disagree")
        aligned.append(aligned_frame[probability_columns].to_numpy(dtype=float))
    weights = specification.get("weights", [])
    if isinstance(weights, str):
        weights = json.loads(weights)
    parameters = specification.get("aggregation_parameters", {})
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    probabilities = aggregate_member_predictions(
        np.stack(aligned, axis=0),
        str(specification.get("aggregation_strategy", "")),
        weights=weights,
        parameters=parameters,
    )
    result = ordered.drop(
        columns=[
            "config_id",
            "learner",
            "resolution",
            "levels",
            "count_transformation",
            "mpdr_id",
        ],
        errors="ignore",
    ).copy()
    for index, column in enumerate(probability_columns):
        result[column] = probabilities[:, index]
    result["y_pred"] = np.argmax(probabilities, axis=1).astype(int)
    if "y_proba_pos" in result.columns and len(probability_columns) == 2:
        result["y_proba_pos"] = probabilities[:, 1]
    result["ensemble_config_id"] = str(specification.get("ensemble_config_id", ""))
    return result


def _summarize_outer(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    probability_columns = _prediction_columns(predictions)
    classes = np.arange(len(probability_columns), dtype=int)
    fold_rows = []
    for split_key, fold in predictions.groupby("outer_split_key", sort=True):
        truth = pd.to_numeric(fold["y_true"], errors="raise").to_numpy(dtype=int)
        predicted = pd.to_numeric(fold["y_pred"], errors="raise").to_numpy(dtype=int)
        probabilities = fold[probability_columns].to_numpy(dtype=float)
        metrics = compute_metrics(truth, predicted, probabilities, classes)
        fold_rows.append(
            {"outer_split_key": str(split_key), "n_samples": len(fold), **metrics}
        )
    frame = pd.DataFrame(fold_rows)
    summary: dict[str, Any] = {
        "selection_basis": "all_inner_validation_for_final_refit",
        "performance_basis": "retrospective_fixed_final_specification_outer_cv",
        "n_outer_folds": len(frame),
    }
    for metric in frame.columns:
        if metric in {"outer_split_key", "n_samples"}:
            continue
        values = (
            pd.to_numeric(frame[metric], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if values.empty:
            continue
        summary[f"outer_{metric}_mean"] = float(values.mean())
        summary[f"outer_{metric}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )
        summary[f"outer_{metric}_count"] = int(len(values))
    return frame, summary


def fixed_final_outer_performance(
    root: Path | str,
    strategy: str,
    *,
    persist: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(root)
    if strategy not in {"MPMA-B", "MPMA-E"}:
        raise ValueError(f"Unsupported fixed-final strategy: {strategy!r}")
    specification = _specification(root, strategy)
    if not specification:
        return pd.DataFrame(), {}
    predictions = (
        _fixed_mpma_b(root, specification)
        if strategy == "MPMA-B"
        else _fixed_mpma_e(root, specification)
    )
    from .evaluation_predictions import (
        _validate_outer_coverage,
        _validate_prediction_values,
    )

    _validate_prediction_values(predictions, strategy)
    _validate_outer_coverage(root, predictions, strategy)
    fold_metrics, summary = _summarize_outer(predictions)
    selected_id = (
        str(specification.get("config_id", ""))
        if strategy == "MPMA-B"
        else str(specification.get("ensemble_config_id", ""))
    )
    summary.update(specification)
    summary["performance_basis"] = "retrospective_fixed_final_specification_outer_cv"
    summary["final_specification_id"] = selected_id
    summary["n_outer_folds"] = len(fold_metrics)
    if persist:
        directory = root / ("tables" if strategy == "MPMA-B" else "ensembling")
        prefix = "mpma_b_final_outer" if strategy == "MPMA-B" else "mpma_e_final_outer"
        write_table(directory / f"{prefix}_predictions.parquet", predictions)
        write_table(directory / f"{prefix}_results.parquet", fold_metrics)
        dump_json_standard(summary, directory / f"{prefix}_summary.json")
    return predictions, summary
