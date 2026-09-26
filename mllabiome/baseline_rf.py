from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASELINE_RF_RANK_PRIORITY = ("strain", "species", "genus")


def _params(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _rf_protocol_params_match(value: Any) -> bool:
    params = _params(value)
    if params is None:
        return False
    expected = RandomForestClassifier(
        n_estimators=1000,
        min_samples_leaf=5,
    ).get_params(deep=False)
    ignored = {"n_jobs", "random_state", "verbose"}
    for key, expected_value in expected.items():
        if key in ignored:
            continue
        if key not in params or params[key] != expected_value:
            return False
    return True


def _rank_value(value: Any) -> str:
    text = str(value).strip().casefold()
    return text if text in BASELINE_RF_RANK_PRIORITY else ""


def baseline_rf_single_rank(frame: pd.DataFrame) -> pd.Series:
    resolution = frame.get("resolution", pd.Series("", index=frame.index)).map(
        _rank_value
    )
    levels = frame.get("levels", pd.Series("", index=frame.index)).map(_rank_value)
    result = resolution.copy()
    result.loc[result.eq("")] = levels.loc[result.eq("")]
    return result.astype("object")


def baseline_rf_mask(frame: pd.DataFrame, *, active_only: bool = True) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if "count_transformation" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    transform = frame["count_transformation"].astype(str).str.strip().str.casefold()
    mask &= transform.eq("arcsine_sqrt@rank-wise")
    if "feature_filter" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    feature_filter = frame["feature_filter"].astype(str).str.strip().str.casefold()
    mask &= feature_filter.isin({"none", "", "nan"})
    if "learner_class" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    learner_class = frame["learner_class"].astype(str)
    mask &= learner_class.str.endswith("RandomForestClassifier", na=False)
    if "learner_params" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    mask &= frame["learner_params"].map(_rf_protocol_params_match)
    mask &= baseline_rf_single_rank(frame).isin(BASELINE_RF_RANK_PRIORITY)
    if active_only and "active" in frame.columns:
        active = pd.to_numeric(frame["active"], errors="coerce").fillna(0).astype(int)
        mask &= active.eq(1)
    return mask.fillna(False)


def resolve_baseline_rf_configs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    candidates = frame[baseline_rf_mask(frame, active_only=True)].copy()
    if candidates.empty:
        return candidates
    ranks = baseline_rf_single_rank(candidates)
    for rank in BASELINE_RF_RANK_PRIORITY:
        selected = candidates[ranks.eq(rank)].copy()
        if selected.empty:
            continue
        if "config_id" in selected.columns:
            selected = selected.sort_values("config_id", kind="stable")
        return selected.iloc[[0]].copy()
    return candidates.iloc[0:0].copy()


def resolve_baseline_rf_config_id(frame: pd.DataFrame) -> str | None:
    selected = resolve_baseline_rf_configs(frame)
    if selected.empty or "config_id" not in selected.columns:
        return None
    value = selected.iloc[0].get("config_id")
    if value is None or pd.isna(value):
        return None
    return str(value)
