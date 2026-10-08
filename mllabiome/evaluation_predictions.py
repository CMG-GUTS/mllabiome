from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_rf import resolve_baseline_rf_configs
from .metrics import canonical_metric_name
from .selection import select_mpma_b_by_outer_fold, selected_mpma_b_outer_predictions
from .storage import read_table, table_exists

_RANK_PRIORITY = (
    "strain",
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "domain",
)
_COMPARATORS = ("AutoML", "Baseline RF", "SIAMCAT")


def _read_required_table(path: Path) -> pd.DataFrame:
    if not table_exists(path):
        raise FileNotFoundError(path)
    frame = read_table(path)
    if frame.empty:
        raise ValueError(f"Expected non-empty evaluation artifact: {path}")
    return frame


def _outer_split_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "outer_split_key" not in out.columns:
        if "split_key" not in out.columns:
            raise ValueError(
                "Evaluation predictions must contain outer_split_key or split_key."
            )
        out["outer_split_key"] = out["split_key"]
    if out["outer_split_key"].isna().any():
        raise ValueError(
            "Evaluation predictions contain missing outer_split_key values."
        )
    out["outer_split_key"] = out["outer_split_key"].astype(str)
    return out


def _selection_outer_split_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "outer_split_key" not in out.columns:
        if "split_key" not in out.columns:
            raise ValueError(
                "Selection artifact must contain outer_split_key or split_key."
            )
        out["outer_split_key"] = out["split_key"]
    if out["outer_split_key"].isna().any():
        raise ValueError("Selection artifact contains missing outer_split_key values.")
    out["outer_split_key"] = out["outer_split_key"].astype(str)
    return out


def _expected_outer_splits(root: Path) -> set[str]:
    path = root / "tables" / "cv_splits.parquet"
    if not table_exists(path):
        return set()
    frame = read_table(path)
    required = {"stage", "split_key", "role"}
    if frame.empty or not required.issubset(frame.columns):
        return set()
    frame = frame[
        frame["stage"].astype(str).eq("outer") & frame["role"].astype(str).eq("test")
    ]
    return set(frame["split_key"].dropna().astype(str))


def _identity_token(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Held-out sample identity cannot be missing.")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return str(value)


def _expected_outer_samples(root: Path) -> dict[str, set[str]]:
    path = root / "tables" / "cv_splits.parquet"
    if not table_exists(path):
        return {}
    frame = read_table(path)
    required = {"stage", "split_key", "role", "sample_index"}
    if frame.empty or not required.issubset(frame.columns):
        return {}
    frame = frame[
        frame["stage"].astype(str).eq("outer") & frame["role"].astype(str).eq("test")
    ].copy()
    if frame.empty:
        return {}
    frame["split_key"] = frame["split_key"].astype(str)
    frame["sample_index"] = frame["sample_index"].map(_identity_token)
    return {
        str(split_key): set(group["sample_index"].astype(str))
        for split_key, group in frame.groupby("split_key", sort=False)
    }


def _validate_selection_basis(
    selection: pd.DataFrame, allowed: set[str], strategy: str
) -> None:
    if "selection_basis" not in selection.columns:
        return
    observed = set(selection["selection_basis"].dropna().astype(str))
    if observed and not observed.issubset(allowed):
        raise ValueError(
            f"{strategy} evaluation selection has incompatible selection_basis values: "
            f"{sorted(observed)}"
        )


def _validate_prediction_values(frame: pd.DataFrame, strategy: str) -> None:
    required = {"outer_split_key", "sample_id", "y_true", "y_pred"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{strategy} evaluation predictions are missing columns: {missing}"
        )
    if frame["sample_id"].isna().any():
        raise ValueError(
            f"{strategy} evaluation predictions contain missing sample_id values."
        )
    identity = "sample_index" if "sample_index" in frame.columns else "sample_id"
    if frame.duplicated(["outer_split_key", identity]).any():
        raise ValueError(
            f"{strategy} evaluation predictions contain duplicate held-out observations "
            "within an outer split."
        )
    y_true = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(
            f"{strategy} evaluation predictions contain non-finite labels."
        )
    pcols = [column for column in frame.columns if str(column).startswith("proba_")]
    if not pcols:
        raise ValueError(
            f"{strategy} evaluation predictions contain no probability columns."
        )
    proba = frame[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    tolerance = 1e-6
    if (
        not np.isfinite(proba).all()
        or np.any(proba < -tolerance)
        or np.any(proba > 1.0 + tolerance)
    ):
        raise ValueError(
            f"{strategy} evaluation predictions contain invalid probabilities."
        )
    if not np.allclose(proba.sum(axis=1), 1.0, atol=tolerance, rtol=tolerance):
        raise ValueError(f"{strategy} evaluation probability rows must sum to one.")
    n_classes = len(pcols)
    for values, label in ((y_true, "y_true"), (y_pred, "y_pred")):
        if not np.allclose(values, np.rint(values), atol=tolerance, rtol=0.0):
            raise ValueError(
                f"{strategy} evaluation {label} values must be integer encoded."
            )
        encoded = np.rint(values).astype(int)
        if np.any(encoded < 0) or np.any(encoded >= n_classes):
            raise ValueError(
                f"{strategy} evaluation {label} values are incompatible with the "
                "probability columns."
            )


def _validate_outer_coverage(root: Path, frame: pd.DataFrame, strategy: str) -> None:
    expected_splits = _expected_outer_splits(root)
    observed_splits = set(frame["outer_split_key"].astype(str))
    if expected_splits and observed_splits != expected_splits:
        missing = sorted(expected_splits - observed_splits)
        extra = sorted(observed_splits - expected_splits)
        raise ValueError(
            f"{strategy} evaluation predictions do not match the persisted outer-test "
            f"split set; missing={missing}, extra={extra}."
        )
    if "sample_index" not in frame.columns:
        return
    expected_samples = _expected_outer_samples(root)
    if not expected_samples:
        return
    observed = frame.copy()
    observed["sample_index"] = observed["sample_index"].map(_identity_token)
    for split_key, expected in expected_samples.items():
        actual = set(
            observed.loc[
                observed["outer_split_key"].astype(str).eq(str(split_key)),
                "sample_index",
            ].astype(str)
        )
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{strategy} held-out sample coverage does not match cv_splits for "
                f"{split_key!r}; missing={missing}, extra={extra}."
            )


def _validate_selected_predictions(
    root: Path,
    strategy: str,
    predictions: pd.DataFrame,
    selection: pd.DataFrame,
    prediction_id: str,
    selection_id: str,
    allowed_basis: set[str],
) -> pd.DataFrame:
    pred = _outer_split_key(predictions)
    sel = _selection_outer_split_key(selection)
    required = {"outer_split_key", selection_id}
    missing = sorted(required - set(sel.columns))
    if missing:
        raise ValueError(f"{strategy} selection artifact is missing columns: {missing}")
    if prediction_id not in pred.columns:
        raise ValueError(
            f"{strategy} evaluation predictions are missing identifier column "
            f"{prediction_id!r}."
        )
    if sel.duplicated("outer_split_key").any():
        raise ValueError(f"{strategy} must have exactly one selection per outer split.")
    sel[selection_id] = sel[selection_id].astype(str)
    pred[prediction_id] = pred[prediction_id].astype(str)
    _validate_selection_basis(sel, allowed_basis, strategy)
    selection_splits = set(sel["outer_split_key"].astype(str))
    prediction_splits = set(pred["outer_split_key"].astype(str))
    if selection_splits != prediction_splits:
        missing = sorted(selection_splits - prediction_splits)
        extra = sorted(prediction_splits - selection_splits)
        raise ValueError(
            f"{strategy} prediction and selection outer splits differ; "
            f"missing={missing}, extra={extra}."
        )
    selected_ids = sel.set_index("outer_split_key")[selection_id].astype(str).to_dict()
    for split_key, group in pred.groupby("outer_split_key", sort=False):
        observed_ids = set(group[prediction_id].astype(str))
        expected_id = str(selected_ids[str(split_key)])
        if observed_ids != {expected_id}:
            raise ValueError(
                f"{strategy} predictions for {split_key!r} do not match its "
                f"inner-selected specification; expected={expected_id!r}, "
                f"observed={sorted(observed_ids)}."
            )
    _validate_prediction_values(pred, strategy)
    _validate_outer_coverage(root, pred, strategy)
    return pred.reset_index(drop=True)


def _single_rank(frame: pd.DataFrame) -> pd.Series:
    resolution = frame.get("resolution", pd.Series("", index=frame.index)).astype(str)
    levels = frame.get("levels", pd.Series("", index=frame.index)).astype(str)
    resolution = resolution.str.strip().str.casefold()
    levels = levels.str.strip().str.casefold().str.replace(" ", "", regex=False)
    result = pd.Series("", index=frame.index, dtype="object")
    for rank in _RANK_PRIORITY:
        mask = result.eq("") & (resolution.eq(rank) | levels.eq(rank))
        result.loc[mask] = rank
    return result


def _comparator_configs(configs: pd.DataFrame, strategy: str) -> pd.DataFrame:
    frame = configs.copy()
    if "config_id" not in frame.columns:
        raise ValueError("configs.parquet must contain config_id.")
    if "active" in frame.columns:
        active = pd.to_numeric(frame["active"], errors="coerce").fillna(0).astype(int)
        frame = frame[active.eq(1)].copy()
    learner = frame.get("learner", pd.Series("", index=frame.index)).astype(str)
    learner_class = frame.get("learner_class", pd.Series("", index=frame.index)).astype(
        str
    )
    transform = frame.get(
        "count_transformation", pd.Series("", index=frame.index)
    ).astype(str)
    resolution = frame.get("resolution", pd.Series("", index=frame.index)).astype(str)
    if strategy == "AutoML":
        frame = frame[
            learner.str.contains("FLAML|AutoML", case=False, regex=True).fillna(False)
            & transform.str.fullmatch(
                r"relative_abundance(?:@(rank-wise|joint))?", case=False
            ).fillna(False)
        ].copy()
        if frame.empty:
            return frame
        frame["_single_rank"] = _single_rank(frame)
        for rank in _RANK_PRIORITY:
            ranked = frame[frame["_single_rank"].eq(rank)].copy()
            if not ranked.empty:
                return ranked.drop(columns=["_single_rank"])
        return frame.iloc[0:0].drop(columns=["_single_rank"])
    if strategy == "Baseline RF":
        return resolve_baseline_rf_configs(frame)
    if strategy == "SIAMCAT":
        siamcat = learner.str.fullmatch("SIAMCAT", case=False).fillna(False)
        siamcat |= learner_class.str.endswith("SIAMCATClassifier", na=False)
        return frame[
            siamcat
            & transform.str.fullmatch("identity", case=False).fillna(False)
            & resolution.str.fullmatch("raw", case=False).fillna(False)
        ].copy()
    raise ValueError(f"Unsupported comparator strategy {strategy!r}.")


def _metric_column(frame: pd.DataFrame, metric: str) -> str:
    target = canonical_metric_name(metric)
    if target in frame.columns:
        return target
    normalized = {
        canonical_metric_name(str(column)).casefold(): str(column)
        for column in frame.columns
    }
    resolved = normalized.get(target.casefold())
    if resolved is None:
        raise ValueError(
            f"Inner-results table does not contain selection metric {metric!r}."
        )
    return resolved


def _load_mpma_b(root: Path) -> pd.DataFrame:
    predictions = _read_required_table(
        root / "predictions" / "mpma_b_outer_predictions.parquet"
    )
    selection = _read_required_table(root / "tables" / "mpma_b_outer_selection.parquet")
    return _validate_selected_predictions(
        root,
        "MPMA-B",
        predictions,
        selection,
        "config_id",
        "config_id",
        {"outer_fold_inner_validation"},
    )


def _load_mpma_e(root: Path) -> pd.DataFrame:
    predictions = _read_required_table(
        root / "ensembling" / "ensemble_predictions.parquet"
    )
    selection = _read_required_table(
        root / "ensembling" / "mpma_e_outer_selection.parquet"
    )
    return _validate_selected_predictions(
        root,
        "MPMA-E",
        predictions,
        selection,
        "ensemble_config_id",
        "ensemble_config_id",
        {"outer_fold_inner_oof_predictions_only"},
    )


def _load_comparator(root: Path, strategy: str, metric: str) -> pd.DataFrame:
    inner = _read_required_table(root / "inner_results" / "inner_results.parquet")
    outer = _read_required_table(root / "predictions" / "outer_predictions.parquet")
    configs = _read_required_table(root / "configs.parquet")
    candidates = _comparator_configs(configs, strategy)
    if candidates.empty:
        raise ValueError(f"No eligible configurations are available for {strategy}.")
    qualification_path = root / "tables" / "qualification_gate.parquet"
    qualification = (
        read_table(qualification_path) if table_exists(qualification_path) else None
    )
    selection_metric = _metric_column(inner, metric)
    selection = select_mpma_b_by_outer_fold(
        inner,
        candidates,
        selection_metric,
        qualification=qualification,
    )
    if selection.empty:
        raise ValueError(
            "No complete outer-fold inner-validation selections are available for "
            f"{strategy}."
        )
    selection = selection.copy()
    selection["selection_basis"] = "outer_fold_inner_validation_comparator"
    predictions = selected_mpma_b_outer_predictions(selection, outer)
    return _validate_selected_predictions(
        root,
        strategy,
        predictions,
        selection,
        "config_id",
        "config_id",
        {"outer_fold_inner_validation_comparator"},
    )


def load_evaluation_predictions(
    root: Path | str, strategy: str, selection_metric: str
) -> pd.DataFrame:
    root = Path(root)
    strategy = str(strategy).strip()
    if strategy == "MPMA-B":
        return _load_mpma_b(root)
    if strategy == "MPMA-E":
        return _load_mpma_e(root)
    if strategy in _COMPARATORS:
        return _load_comparator(root, strategy, selection_metric)
    raise ValueError(f"Unsupported evaluation strategy {strategy!r}.")


def evaluation_prediction_metadata(strategy: str) -> dict[str, Any]:
    strategy = str(strategy).strip()
    if strategy == "MPMA-B":
        return {
            "estimand": "nested_model_selection_procedure_performance",
            "prediction_artifact": "predictions/mpma_b_outer_predictions.parquet",
            "selection_artifact": "tables/mpma_b_outer_selection.parquet",
            "selection_scope": "outer_fold_inner_validation_only",
            "final_refit_specification_used_for_performance": False,
        }
    if strategy == "MPMA-E":
        return {
            "estimand": "nested_ensemble_selection_procedure_performance",
            "prediction_artifact": "ensembling/ensemble_predictions.parquet",
            "selection_artifact": "ensembling/mpma_e_outer_selection.parquet",
            "selection_scope": "outer_fold_inner_oof_predictions_only",
            "final_refit_specification_used_for_performance": False,
        }
    if strategy in _COMPARATORS:
        return {
            "estimand": "nested_comparator_procedure_performance",
            "prediction_artifact": "predictions/outer_predictions.parquet",
            "selection_artifact": None,
            "selection_scope": (
                "outer_fold_inner_validation_within_prespecified_comparator_family"
            ),
            "final_refit_specification_used_for_performance": False,
        }
    raise ValueError(f"Unsupported evaluation strategy {strategy!r}.")


def comparator_config_ids(root: Path | str, strategy: str) -> set[str]:
    root = Path(root)
    strategy = str(strategy).strip()
    if strategy not in _COMPARATORS:
        raise ValueError(f"Unsupported comparator strategy {strategy!r}.")
    configs = _read_required_table(root / "configs.parquet")
    candidates = _comparator_configs(configs, strategy)
    if candidates.empty:
        return set()
    return set(candidates["config_id"].astype(str))
