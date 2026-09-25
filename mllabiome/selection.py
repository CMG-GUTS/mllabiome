from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    canonical_metric_name,
    compute_metrics,
    compute_regression_metrics,
    metric_is_loss,
)
from .storage import read_table, table_exists, write_table
from .transformations import _count_transformation_name
from .utils import dump_json_standard


def _eligible_config_ids(configs: pd.DataFrame, plan: Any | None = None) -> set[str]:
    if configs is None or configs.empty or "config_id" not in configs.columns:
        return set()
    frame = configs.copy()
    include_inactive = (
        bool(getattr(plan, "include_inactive", False)) if plan is not None else False
    )
    if not include_inactive and "active" in frame.columns:
        active = pd.to_numeric(frame["active"], errors="coerce").fillna(0).astype(int)
        frame = frame[active.eq(1)]
    if plan is not None:
        exclude_ids = {str(x) for x in getattr(plan, "exclude_config_ids", ())}
        exclude_learners = {str(x) for x in getattr(plan, "exclude_learners", ())}
        exclude_resolutions = {str(x) for x in getattr(plan, "exclude_resolutions", ())}
        exclude_transformations = {
            _count_transformation_name(x)
            for x in getattr(plan, "exclude_transformations", ())
        }
        if exclude_ids:
            frame = frame[~frame["config_id"].astype(str).isin(exclude_ids)]
        if exclude_learners and "learner" in frame.columns:
            frame = frame[~frame["learner"].astype(str).isin(exclude_learners)]
        if exclude_resolutions and "resolution" in frame.columns:
            frame = frame[~frame["resolution"].astype(str).isin(exclude_resolutions)]
        if exclude_transformations and "count_transformation" in frame.columns:
            frame = frame[
                ~frame["count_transformation"].astype(str).isin(exclude_transformations)
            ]
    return set(frame["config_id"].astype(str))


def _qualified_pairs(qualification: pd.DataFrame | None) -> set[tuple[str, str]] | None:
    if qualification is None or qualification.empty:
        return None
    required = {"split_key", "config_id", "qualified"}
    if not required.issubset(qualification.columns):
        return None
    q = qualification.copy()
    q["qualified"] = (
        pd.to_numeric(q["qualified"], errors="coerce").fillna(0).astype(int)
    )
    q = q[q["qualified"].eq(1)]
    return set(zip(q["split_key"].astype(str), q["config_id"].astype(str)))


def _config_metadata(configs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if configs is None or configs.empty or "config_id" not in configs.columns:
        return {}
    frame = configs.drop_duplicates("config_id", keep="last")
    return {str(row["config_id"]): row for row in frame.to_dict(orient="records")}


def _inner_score(group: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = group[metric].to_numpy(dtype=float)
    weight_column = {
        "log_loss": "n_samples",
        "subject_macro_log_loss": "n_subjects",
    }.get(metric)
    if weight_column is not None and weight_column in group.columns:
        weights = pd.to_numeric(group[weight_column], errors="coerce").to_numpy(
            dtype=float
        )
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if np.any(valid):
            score = float(np.average(values[valid], weights=weights[valid]))
        else:
            score = float(np.mean(values))
    else:
        score = float(np.mean(values))
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return score, spread


def _complete_config_scores(
    inner: pd.DataFrame,
    metric: str,
    eligible_ids: set[str],
    qualification: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metric = canonical_metric_name(metric)
    required = {"split_key", "inner_key", "config_id", metric}
    missing = required - set(inner.columns)
    if missing:
        raise ValueError(
            f"Inner-results table is missing required columns: {sorted(missing)}"
        )
    frame = inner.copy()
    frame["split_key"] = frame["split_key"].astype(str)
    frame["inner_key"] = frame["inner_key"].astype(str)
    frame["config_id"] = frame["config_id"].astype(str)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    if eligible_ids:
        frame = frame[frame["config_id"].isin(eligible_ids)]
    if "ok" in frame.columns:
        frame["_ok"] = pd.to_numeric(frame["ok"], errors="coerce").fillna(0).astype(int)
    else:
        frame["_ok"] = 1
    qualified = _qualified_pairs(qualification)
    rows: list[dict[str, Any]] = []
    for split_key, split_frame in frame.groupby("split_key", sort=True):
        expected_keys = tuple(sorted(x for x in split_frame["inner_key"].unique() if x))
        expected_set = set(expected_keys)
        if not expected_set:
            continue
        for config_id, group in split_frame.groupby("config_id", sort=True):
            if (
                qualified is not None
                and (str(split_key), str(config_id)) not in qualified
            ):
                continue
            group = group.drop_duplicates("inner_key", keep="last")
            valid = group[
                group["_ok"].eq(1) & np.isfinite(group[metric].to_numpy(dtype=float))
            ]
            valid_keys = set(valid["inner_key"].astype(str))
            if valid_keys != expected_set:
                continue
            score, spread = _inner_score(valid, metric)
            rows.append(
                {
                    "outer_split_key": str(split_key),
                    "config_id": str(config_id),
                    "selection_metric": str(metric),
                    "inner_score": score,
                    "inner_score_std": spread,
                    "n_inner_folds": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def select_mpma_b_by_outer_fold(
    inner: pd.DataFrame,
    configs: pd.DataFrame,
    metric: str,
    plan: Any | None = None,
    qualification: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metric = canonical_metric_name(metric)
    eligible_ids = _eligible_config_ids(configs, plan)
    if not eligible_ids:
        return pd.DataFrame()
    scores = _complete_config_scores(inner, metric, eligible_ids, qualification)
    if scores.empty:
        return pd.DataFrame()
    metadata = _config_metadata(configs)
    selected: list[dict[str, Any]] = []
    for outer_split_key, group in scores.groupby("outer_split_key", sort=True):
        winner = (
            group.sort_values(
                ["inner_score", "config_id"],
                ascending=[metric_is_loss(metric), True],
                kind="mergesort",
            )
            .iloc[0]
            .to_dict()
        )
        meta = metadata.get(str(winner["config_id"]), {})
        row = {
            "outer_split_key": str(outer_split_key),
            "config_id": str(winner["config_id"]),
            "selection_basis": "outer_fold_inner_validation",
            "selection_metric": str(metric),
            "inner_score": float(winner["inner_score"]),
            "inner_score_std": float(winner["inner_score_std"]),
            "n_inner_folds": int(winner["n_inner_folds"]),
        }
        for key in (
            "mpdr_id",
            "count_transformation",
            "resolution",
            "levels",
            "learner",
            "candidate_family",
            "modalities",
            "integration",
            "integration_n_components",
        ):
            if key in meta:
                row[key] = meta[key]
        selected.append(row)
    return pd.DataFrame(selected)


def select_final_mpma_candidate(
    inner: pd.DataFrame,
    configs: pd.DataFrame,
    metric: str,
    plan: Any | None = None,
) -> dict[str, Any]:
    metric = canonical_metric_name(metric)
    eligible_ids = _eligible_config_ids(configs, plan)
    if not eligible_ids:
        return {}
    required = {"inner_key", "config_id", metric}
    missing = required - set(inner.columns)
    if missing:
        raise ValueError(
            f"Inner-results table is missing required columns: {sorted(missing)}"
        )
    frame = inner.copy()
    frame["inner_key"] = frame["inner_key"].astype(str)
    frame["config_id"] = frame["config_id"].astype(str)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    if eligible_ids:
        frame = frame[frame["config_id"].isin(eligible_ids)]
    if "ok" in frame.columns:
        ok = pd.to_numeric(frame["ok"], errors="coerce").fillna(0).astype(int)
        frame = frame[ok.eq(1)]
    frame = frame[np.isfinite(frame[metric].to_numpy(dtype=float))]
    expected_keys = set(inner["inner_key"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for config_id, group in frame.groupby("config_id", sort=True):
        group = group.drop_duplicates("inner_key", keep="last")
        if set(group["inner_key"].astype(str)) != expected_keys:
            continue
        score, spread = _inner_score(group, metric)
        rows.append(
            {
                "config_id": str(config_id),
                "inner_score": score,
                "inner_score_std": spread,
                "n_inner_folds": int(len(group)),
            }
        )
    if not rows:
        return {}
    ranked = pd.DataFrame(rows).sort_values(
        ["inner_score", "config_id"],
        ascending=[metric_is_loss(metric), True],
        kind="mergesort",
    )
    best = ranked.iloc[0].to_dict()
    meta = _config_metadata(configs).get(str(best["config_id"]), {})
    out = {
        "config_id": str(best["config_id"]),
        "selection_basis": "all_inner_validation_for_final_refit",
        "selection_metric": str(metric),
        "inner_score": float(best["inner_score"]),
        "inner_score_std": float(best["inner_score_std"]),
        "n_inner_folds": int(best["n_inner_folds"]),
    }
    for key in (
        "mpdr_id",
        "count_transformation",
        "resolution",
        "levels",
        "learner",
    ):
        if key in meta:
            out[key] = meta[key]
    return out


def selected_mpma_b_outer_predictions(
    selection: pd.DataFrame,
    outer_predictions: pd.DataFrame,
) -> pd.DataFrame:
    if selection is None or selection.empty:
        return pd.DataFrame()
    required_selection = {"outer_split_key", "config_id"}
    required_predictions = {"config_id", "sample_id", "y_true", "y_pred"}
    if not required_selection.issubset(selection.columns):
        raise ValueError("Selection table must contain outer_split_key and config_id.")
    if not required_predictions.issubset(outer_predictions.columns):
        raise ValueError(
            "Outer-predictions table is missing required prediction columns."
        )
    pred = outer_predictions.copy()
    if "outer_split_key" not in pred.columns:
        if "split_key" not in pred.columns:
            raise ValueError(
                "Outer-predictions table must contain outer_split_key or split_key."
            )
        pred["outer_split_key"] = pred["split_key"]
    pred["outer_split_key"] = pred["outer_split_key"].astype(str)
    pred["config_id"] = pred["config_id"].astype(str)
    sel = selection.copy()
    sel["outer_split_key"] = sel["outer_split_key"].astype(str)
    sel["config_id"] = sel["config_id"].astype(str)
    if sel.duplicated("outer_split_key").any():
        raise ValueError(
            "MPMA-B selection must contain exactly one selected config per outer split."
        )
    key_cols = ["outer_split_key", "config_id"]
    keep = [
        c
        for c in (
            "outer_split_key",
            "config_id",
            "selection_metric",
            "inner_score",
            "inner_score_std",
            "n_inner_folds",
        )
        if c in sel.columns
    ]
    out = pred.merge(sel[keep], on=key_cols, how="inner", validate="many_to_one")
    expected_splits = set(sel["outer_split_key"].astype(str))
    observed_splits = set(out["outer_split_key"].astype(str))
    missing_splits = sorted(expected_splits - observed_splits)
    if missing_splits:
        raise ValueError(
            f"Selected MPMA-B outer predictions are missing for outer split(s): {missing_splits}"
        )
    if "sample_index" in out.columns:
        out = out.sort_values(["outer_split_key", "sample_index"], kind="mergesort")
    else:
        out = out.sort_values(["outer_split_key", "sample_id"], kind="mergesort")
    return out.reset_index(drop=True)


def mpma_b_fold_metrics(selected_predictions: pd.DataFrame) -> pd.DataFrame:
    if selected_predictions is None or selected_predictions.empty:
        return pd.DataFrame()
    pcols = [c for c in selected_predictions.columns if c.startswith("proba_")]
    rows: list[dict[str, Any]] = []
    for outer_split_key, group in selected_predictions.groupby(
        "outer_split_key", sort=True
    ):
        config_ids = group["config_id"].astype(str).unique()
        if len(config_ids) != 1:
            raise ValueError(
                "Each outer split must contain predictions from exactly one selected MPMA-B config."
            )
        if pcols:
            y_true = (
                pd.to_numeric(group["y_true"], errors="raise").astype(int).to_numpy()
            )
            y_pred = (
                pd.to_numeric(group["y_pred"], errors="raise").astype(int).to_numpy()
            )
            proba = (
                group[pcols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            )
            classes = np.arange(len(pcols), dtype=int)
            metrics = compute_metrics(y_true, y_pred, proba, classes)
            task = "classification"
        else:
            y_true = pd.to_numeric(group["y_true"], errors="raise").to_numpy(
                dtype=float
            )
            y_pred = pd.to_numeric(group["y_pred"], errors="raise").to_numpy(
                dtype=float
            )
            metrics = compute_regression_metrics(y_true, y_pred)
            task = "regression"
        rows.append(
            {
                "outer_split_key": str(outer_split_key),
                "config_id": str(config_ids[0]),
                "task": task,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_mpma_b_strategy(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    if fold_metrics is None or fold_metrics.empty:
        return {}
    out: dict[str, Any] = {
        "Strategy": "MPMA-B",
        "selection_basis": "outer_fold_inner_validation",
        "n_outer_folds": int(len(fold_metrics)),
    }
    metric_cols = [
        c
        for c in fold_metrics.columns
        if c not in {"outer_split_key", "config_id"}
        and pd.api.types.is_numeric_dtype(fold_metrics[c])
    ]
    for metric in metric_cols:
        vals = (
            pd.to_numeric(fold_metrics[metric], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if vals.empty:
            continue
        out[f"outer_{metric}_mean"] = float(vals.mean())
        out[f"outer_{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"outer_{metric}_count"] = int(len(vals))
    return out


def write_mpma_b_selection_outputs(
    root: Path | str,
    metric: str,
    plan: Any | None = None,
) -> dict[str, Path]:
    root = Path(root)
    metric = canonical_metric_name(metric)
    inner_path = root / "inner_results" / "inner_results.parquet"
    outer_pred_path = root / "predictions" / "outer_predictions.parquet"
    configs_path = root / "configs.parquet"
    if (
        not table_exists(inner_path)
        or not table_exists(outer_pred_path)
        or not table_exists(configs_path)
    ):
        return {}
    inner = read_table(inner_path)
    outer_predictions = read_table(outer_pred_path)
    configs = read_table(configs_path)
    qualification_path = root / "tables" / "qualification_gate.parquet"
    qualification = (
        read_table(qualification_path) if table_exists(qualification_path) else None
    )
    selection = select_mpma_b_by_outer_fold(
        inner,
        configs,
        metric,
        plan=plan,
        qualification=qualification,
    )
    selected_predictions = selected_mpma_b_outer_predictions(
        selection, outer_predictions
    )
    fold_metrics = mpma_b_fold_metrics(selected_predictions)
    summary = summarize_mpma_b_strategy(fold_metrics)
    final_candidate = select_final_mpma_candidate(inner, configs, metric, plan=plan)
    tables_dir = root / "tables"
    predictions_dir = root / "predictions"
    results_dir = root / "results"
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    selection_path = tables_dir / "mpma_b_outer_selection.parquet"
    prediction_path = predictions_dir / "mpma_b_outer_predictions.parquet"
    fold_metrics_path = results_dir / "mpma_b_outer_results.parquet"
    summary_path = tables_dir / "mpma_b_strategy_summary.json"
    final_path = tables_dir / "mpma_b_final_candidate.json"
    write_table(selection_path, selection)
    write_table(prediction_path, selected_predictions)
    write_table(fold_metrics_path, fold_metrics)
    dump_json_standard(summary, summary_path)
    dump_json_standard(final_candidate, final_path)
    return {
        "mpma_b_selection": selection_path,
        "mpma_b_predictions": prediction_path,
        "mpma_b_outer_results": fold_metrics_path,
        "mpma_b_summary": summary_path,
        "mpma_b_final_candidate": final_path,
    }
