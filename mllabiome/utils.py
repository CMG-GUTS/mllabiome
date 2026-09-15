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

METRIC_COLUMNS = [
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
]


def _as_float_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {X.shape}.")
    return X


def _finite(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def _relative(X: np.ndarray) -> np.ndarray:
    X = np.clip(_finite(X), 0.0, None)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return X / row_sum


def _clr(X: np.ndarray, pseudo_count: float) -> np.ndarray:
    R = _relative(X)
    R = np.clip(R, pseudo_count, None)
    R = R / R.sum(axis=1, keepdims=True)
    L = np.log(R)
    return L - L.mean(axis=1, keepdims=True)


def _json_clean(obj: Any) -> Any:
    """Return JSON-standard data: NaN/Inf become None recursively."""
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
    """Write strict standards-compliant JSON. Python NaN/Inf are not emitted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_clean(obj), fh, indent=indent, allow_nan=False)
