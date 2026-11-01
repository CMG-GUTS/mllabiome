from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    for delimiter in ("\t", ",", ";"):
        frame = pd.read_csv(path, sep=delimiter, dtype=str, encoding="utf-8-sig")
        frame.columns = frame.columns.astype(str).str.strip()
        if "sample_id" in frame.columns:
            frame["sample_id"] = frame["sample_id"].astype(str).str.strip()
            if (
                frame["sample_id"].isna().any()
                or frame["sample_id"].eq("").any()
                or frame["sample_id"].duplicated().any()
            ):
                raise ValueError(f"Invalid sample IDs in conformal input {path}.")
            return frame
    raise ValueError(f"Conformal input {path} requires a sample_id column.")


def _probability_columns(class_labels: list[str], prefix: str) -> list[str]:
    import re

    used: set[str] = set()
    columns: list[str] = []
    for label in class_labels:
        base = re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_") or "class"
        token = base
        suffix = 2
        while token in used:
            token = f"{base}_{suffix}"
            suffix += 1
        used.add(token)
        columns.append(f"{prefix}_probability_{token}")
    return columns


def split_conformal_sets(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    test_probabilities: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    cal = np.asarray(calibration_probabilities, dtype=float)
    test = np.asarray(test_probabilities, dtype=float)
    y = np.asarray(calibration_labels, dtype=int)
    if (
        cal.ndim != 2
        or test.ndim != 2
        or cal.shape[1] < 2
        or cal.shape[1] != test.shape[1]
    ):
        raise ValueError(
            "Conformal calibration and inference class dimensions disagree."
        )
    if (
        len(cal) < 2
        or y.shape != (len(cal),)
        or (y < 0).any()
        or (y >= cal.shape[1]).any()
    ):
        raise ValueError(
            "Conformal calibration requires at least two labeled samples and valid class indices."
        )
    if not 0 < float(alpha) < 1:
        raise ValueError("Conformal alpha must lie strictly between zero and one.")
    for matrix in (cal, test):
        if (
            not np.isfinite(matrix).all()
            or (matrix < 0).any()
            or (matrix > 1).any()
            or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError(
                "Conformal input must contain normalized probability vectors."
            )
    scores = 1.0 - cal[np.arange(len(cal)), y]
    order_statistic = int(np.ceil((len(cal) + 1) * (1.0 - float(alpha))))
    threshold = (
        float(np.sort(scores)[order_statistic - 1])
        if order_statistic <= len(cal)
        else float("inf")
    )
    include = 1.0 - test <= threshold
    metadata = {
        "method": "split_conformal_least_ambiguous_class",
        "alpha": float(alpha),
        "n_calibration": int(len(cal)),
        "order_statistic": order_statistic,
        "nonconformity_threshold": threshold if np.isfinite(threshold) else None,
        "full_set_due_to_small_calibration_sample": bool(not np.isfinite(threshold)),
        "mean_set_size": float(include.sum(axis=1).mean()) if len(test) else 0.0,
        "assumptions": "Calibration and test examples are exchangeable, calibration labels are independent of model fitting and selection, and the scoring model is fixed.",
    }
    return include, metadata


def external_conformal_sets(
    predictions: pd.DataFrame,
    calibration_predictions_path: Path,
    calibration_labels_path: Path,
    *,
    class_labels: list[str],
    strategies: tuple[str, ...],
    alpha: float,
    training_ids: list[str],
    test_ids: list[str],
    model_signature: str,
    model_artifact_hashes: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    calibration_predictions_path = Path(calibration_predictions_path)
    calibration_labels_path = Path(calibration_labels_path)
    origin = calibration_predictions_path.parent / "manifest.json"
    if not origin.is_file():
        raise ValueError(
            f"Conformal calibration predictions require the accompanying inference manifest: {origin}."
        )
    manifest = json.loads(origin.read_text(encoding="utf-8"))
    if str(manifest.get("model_signature", "")) != str(model_signature):
        raise ValueError(
            "Conformal calibration predictions were not generated with the same training data and frozen final model specification."
        )
    if (
        not model_artifact_hashes
        or manifest.get("model_artifact_hashes") != model_artifact_hashes
    ):
        raise ValueError(
            "Conformal calibration predictions require exactly matching fitted and saved model artifacts."
        )
    calibration = _table(calibration_predictions_path)
    labels = _table(calibration_labels_path)
    if "y_true" not in labels.columns:
        raise ValueError(
            "Conformal calibration labels require a y_true column containing the original class labels."
        )
    if labels["y_true"].isna().any() or labels["y_true"].eq("").any():
        raise ValueError("Conformal calibration outcomes contain missing labels.")
    if set(calibration["sample_id"]) != set(labels["sample_id"]):
        raise ValueError(
            "Conformal prediction and label sample IDs must agree exactly."
        )
    forbidden = set(str(x) for x in training_ids) | set(str(x) for x in test_ids)
    overlap = forbidden & set(calibration["sample_id"])
    if overlap:
        raise ValueError(
            f"Conformal calibration samples overlap training or external test samples: {sorted(overlap)[:5]!r}."
        )
    matched = calibration.merge(
        labels, on="sample_id", how="inner", validate="one_to_one"
    )
    codes = {str(label): index for index, label in enumerate(class_labels)}
    unknown = sorted(set(matched["y_true"].astype(str)) - set(codes))
    if unknown:
        raise ValueError(
            f"Conformal calibration contains unknown outcome labels: {unknown!r}."
        )
    y = np.asarray([codes[str(label)] for label in matched["y_true"]], dtype=int)
    result = predictions.copy()
    report: dict[str, Any] = {}
    for strategy in strategies:
        prefix = strategy.casefold().replace("-", "_")
        columns = _probability_columns(class_labels, prefix)
        missing = set(columns) - set(matched.columns)
        if missing:
            raise ValueError(
                f"Conformal calibration predictions are missing columns: {sorted(missing)!r}."
            )
        if not set(columns).issubset(predictions.columns):
            raise ValueError(
                "External predictions are missing required probability columns."
            )
        cal_probabilities = (
            matched[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        )
        test_probabilities = predictions[columns].to_numpy(dtype=float)
        included, summary = split_conformal_sets(
            cal_probabilities, y, test_probabilities, alpha
        )
        result[f"{prefix}_conformal_prediction_set"] = [
            json.dumps([str(class_labels[j]) for j in np.flatnonzero(row)])
            for row in included
        ]
        result[f"{prefix}_conformal_set_size"] = included.sum(axis=1).astype(int)
        for j, label in enumerate(class_labels):
            result[
                f"{prefix}_conformal_contains_{columns[j].split('_probability_', 1)[1]}"
            ] = included[:, j]
        report[strategy] = summary
    return result, report
