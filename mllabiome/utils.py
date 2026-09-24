from __future__ import annotations

import json
import re
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
    "PR_AUC",
    "PR_AUC_macro",
    "PR_AUC_weighted",
    "AP",
    "AP_macro",
    "MCC",
    "nMCC",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "BalAcc",
    "Accuracy",
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


def tail_ellipsis(value: Any, max_len: int, *, ellipsis: str = "...") -> str:
    text = str(value)
    limit = max(1, int(max_len))
    if len(text) <= limit:
        return text
    marker = str(ellipsis)
    if limit <= len(marker):
        return marker[:limit]
    return marker + text[-(limit - len(marker)) :].lstrip()


def rank_tail_ellipsis(value: Any, max_len: int, *, ellipsis: str = "...") -> str:
    text = str(value).strip()
    limit = max(1, int(max_len))
    if len(text) <= limit:
        return text
    marker = str(ellipsis)
    match = re.match(r"^([A-Za-z]\.\s+)(.*)$", text)
    if match is None:
        return tail_ellipsis(text, limit, ellipsis=marker)
    prefix = match.group(1)
    body = match.group(2).strip()
    if limit <= len(prefix):
        return prefix[:limit]
    if limit <= len(prefix) + len(marker):
        return (prefix + marker)[:limit]
    budget = limit - len(prefix) - len(marker)
    return prefix + marker + body[-budget:].lstrip()


def feature_tail_ellipsis(value: Any, max_len: int, *, ellipsis: str = "...") -> str:
    text = str(value).strip()
    limit = max(1, int(max_len))
    if len(text) <= limit:
        return text
    if text.startswith("ALR[") and text.endswith("]") and " / " in text:
        left, right = text[4:-1].split(" / ", 1)
        available = limit - len("ALR[") - len(" / ") - len("]")
        minimum_left = 8
        if available >= minimum_left + 4:
            if len(right) <= available - minimum_left:
                right_budget = len(right)
                left_budget = available - right_budget
            else:
                right_budget = max(4, int(round(available * 0.58)))
                left_budget = available - right_budget
            left_text = rank_tail_ellipsis(left, left_budget, ellipsis=ellipsis)
            right_text = rank_tail_ellipsis(right, right_budget, ellipsis=ellipsis)
            return f"ALR[{left_text} / {right_text}]"
    return rank_tail_ellipsis(text, limit, ellipsis=ellipsis)


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
