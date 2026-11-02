from __future__ import annotations

import math

import numpy as np
import pandas as pd


def validate_auroc_tolerance(tolerance: float | None, role: str) -> float | None:
    if tolerance is None:
        return None
    value = float(tolerance)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{role} must be finite and between 0 and 1.")
    return value


def rank_auroc_log_loss(
    frame: pd.DataFrame,
    auc_col: str,
    loss_col: str,
    tolerance: float,
    tie_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    missing = {auc_col, loss_col, *tie_columns} - set(frame.columns)
    if missing:
        raise ValueError(f"AUROC/log-loss selection requires columns: {sorted(missing)}.")
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    auc = pd.to_numeric(ranked[auc_col], errors="coerce")
    loss = pd.to_numeric(ranked[loss_col], errors="coerce")
    ranked = ranked.loc[np.isfinite(auc) & np.isfinite(loss)].copy()
    ranked[auc_col] = auc.loc[ranked.index].astype(float)
    ranked[loss_col] = loss.loc[ranked.index].astype(float)
    if ranked.empty:
        return ranked
    maximum = float(ranked[auc_col].max())
    ranked["_auroc_eligible"] = ranked[auc_col].lt(maximum - tolerance - 1e-12)
    eligible = ranked.loc[~ranked["_auroc_eligible"]].sort_values(
        [loss_col, auc_col, *tie_columns],
        ascending=[True, False, *([True] * len(tie_columns))],
        kind="mergesort",
    )
    ineligible = ranked.loc[ranked["_auroc_eligible"]].sort_values(
        [auc_col, loss_col, *tie_columns],
        ascending=[False, True, *([True] * len(tie_columns))],
        kind="mergesort",
    )
    result = pd.concat([eligible, ineligible], axis=0).drop(columns="_auroc_eligible")
    result.attrs["best_auroc"] = maximum
    result.attrs["auroc_tolerance"] = float(tolerance)
    return result
