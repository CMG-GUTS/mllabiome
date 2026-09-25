from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

DIAGNOSTIC_METRICS = (
    "Sensitivity",
    "Specificity",
    "PPV",
    "NPV",
    "Accuracy",
    "MCC",
    "nMCC",
)
OOF_METRIC_ORDER = (
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
    "Brier",
    "Brier_multiclass",
    "LogLoss",
    "CalibrationInTheLarge",
    "CalibrationIntercept",
    "CalibrationSlope",
)
_OOF_CONTRAST_METRICS = (
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
    "Brier",
    "Brier_multiclass",
    "LogLoss",
)
_LODO_PROTOCOLS = {"lodo", "leave_one_dataset_out"}


def _ensure_outer_split_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outer_split_key" not in out.columns and "split_key" in out.columns:
        out["outer_split_key"] = out["split_key"]
    if "outer_split_key" in out.columns:
        out["outer_split_key"] = out["outer_split_key"].astype(str)
    return out


def _repeat_id(split_key: str) -> str:
    match = re.match(r"^r(\d+)_o\d+$", str(split_key))
    return f"r{match.group(1)}" if match else "r0"


def _stable_seed(random_state: int, *parts: Any) -> int:
    text = "|".join(str(value) for value in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little", signed=False)
    return int((int(random_state) + offset) % (2**32 - 1))
