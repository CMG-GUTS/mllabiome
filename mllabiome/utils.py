from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

TAXONOMIC_LEVELS: tuple[str, ...] = (
    "domain",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "strain",
)

CLASSIFICATION_METRIC_COLUMNS = [
    "AUC",
    "AUC_macro",
    "AUC_weighted",
    "nMCC",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
    "BalAcc",
    "Accuracy",
    "PR_AUC",
    "PR_AUC_macro",
    "log_loss",
    "brier",
]

REGRESSION_METRIC_COLUMNS = [
    "R2",
    "MAE",
    "MSE",
    "RMSE",
    "MedAE",
    "ExplainedVariance",
    "PearsonR",
    "SpearmanR",
]

METRIC_COLUMNS = CLASSIFICATION_METRIC_COLUMNS + REGRESSION_METRIC_COLUMNS


def _as_float_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {X.shape}.")
    return X


def _json_clean(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_clean(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "name") and isinstance(getattr(obj, "name"), str):
        return getattr(obj, "name")
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    try:
        json.dumps(obj, allow_nan=False)
        return obj
    except TypeError:
        return str(obj)


def dump_json_standard(obj: Any, path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_clean(obj), fh, indent=indent, allow_nan=False)
