from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .configs_sweep import Ensemble, Sweep
from .console import path_table, stage, success, summary_table
from .data import load_dataset
from .metrics import _renormalize_proba, compute_metrics
from .mpma_e_figure import write_single_task_mpma_e_figure
from .selection import select_final_mpma_candidate
from .utils import TAXONOMIC_LEVELS, dump_json_standard
from .compute import ResourceTracker


def _load_manifest(root: Path) -> dict[str, Any]:
    with open(root / "manifest.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _proba_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("proba_")]


def _excluded_config_ids(configs: pd.DataFrame, ensemble: Ensemble) -> set[str]:
    if configs is None or configs.empty or "config_id" not in configs.columns:
        return set()
    mask = pd.Series(False, index=configs.index, dtype=bool)
    ids = {str(x) for x in getattr(ensemble, "exclude_config_ids", ()) if str(x)}
    if ids:
        mask |= configs["config_id"].astype(str).isin(ids)

    def match(column: str, values: tuple[str, ...]) -> None:
        nonlocal mask
        vals = {str(x).casefold() for x in values if str(x)}
        if vals and column in configs.columns:
            mask |= configs[column].astype(str).str.casefold().isin(vals)

    match("learner", tuple(getattr(ensemble, "exclude_learners", ())))
    match("resolution", tuple(getattr(ensemble, "exclude_resolutions", ())))
    match(
        "count_transformation", tuple(getattr(ensemble, "exclude_transformations", ()))
    )
    return set(configs.loc[mask, "config_id"].astype(str))


def _eligible_config_ids(configs: pd.DataFrame, ensemble: Ensemble) -> set[str]:
    if configs is None or configs.empty or "config_id" not in configs.columns:
        return set()
    frame = configs.copy()
    if (
        not bool(getattr(ensemble, "include_inactive", False))
        and "active" in frame.columns
    ):
        active = pd.to_numeric(frame["active"], errors="coerce").fillna(0).astype(int)
        frame = frame[active.eq(1)]
    excluded = _excluded_config_ids(frame, ensemble)
    return set(frame["config_id"].astype(str)) - excluded


def _ensemble_configs(plan: Ensemble) -> list[dict[str, Any]]:
    rows = []
    for sel in plan.selection_strategies:
        for agg in plan.aggregation_strategies:
            sizes = (plan.threshold_max_members,) if sel == "threshold" else plan.sizes
            for size in sizes:
                uid = f"{plan.optimize_metric}__{sel}_{size}__{agg}"
                rows.append(
                    {
                        "ensemble_config_id": hashlib.sha1(uid.encode()).hexdigest()[
                            :12
                        ],
                        "optimize_metric": plan.optimize_metric,
                        "selection_strategy": str(sel),
                        "ensemble_size": int(size),
                        "threshold_score": plan.threshold_score
                        if sel == "threshold"
                        else np.nan,
                        "aggregation_strategy": str(agg),
                    }
                )
    return rows


def _model_family(name: str) -> str:
    mn = str(name).lower()
    for fam, keys in {
        "RF": ["rf", "randomforest"],
        "ET": ["et", "extra"],
        "GB": ["gb", "xgb", "lgb", "cat", "hist"],
        "LR": ["lr", "logistic"],
        "Ridge": ["ridge"],
        "SVM": ["svc", "svm", "lsvc"],
        "NB": ["nb", "gnb", "bnb", "mnb"],
        "kNN": ["knn", "nearest"],
        "DA": ["lda", "qda"],
    }.items():
        if any(k in mn for k in keys):
            return fam
    return mn.split("_")[0]


def _select_members(
    scores: pd.Series,
    configs: pd.DataFrame,
    spec: dict[str, Any],
    plan: Ensemble,
) -> list[str]:
    size = int(spec["ensemble_size"])
    sel = str(spec["selection_strategy"])
    ordered = [str(x) for x in scores.index.tolist()]
    meta = configs.drop_duplicates("config_id").copy()
    meta["config_id"] = meta["config_id"].astype(str)
    meta = meta.set_index("config_id")
    if sel == "top_k":
        return ordered[:size]
    if sel == "threshold":
        return [cid for cid in ordered if float(scores[cid]) >= plan.threshold_score][
            :size
        ]
    if sel == "best_per_family":
        out = []
        seen = set()
        for cid in ordered:
            fam = _model_family(meta.loc[cid, "learner"] if cid in meta.index else cid)
            if fam not in seen:
                out.append(cid)
                seen.add(fam)
            if len(out) >= size:
                break
        return out
    if sel == "best_per_resolution":
        out = []
        seen = set()
        for cid in ordered:
            res = str(meta.loc[cid, "resolution"] if cid in meta.index else "")
            if res not in seen:
                out.append(cid)
                seen.add(res)
            if len(out) >= size:
                break
        return out
    if sel == "diverse_top_k":
        out = _select_members(
            scores,
            configs,
            {**spec, "selection_strategy": "best_per_family"},
            plan,
        )
        for cid in ordered:
            if len(out) >= size:
                break
            if cid not in out:
                out.append(cid)
        return out
    return ordered[:size]


def _aggregate_proba(stack: np.ndarray, weights: np.ndarray, method: str) -> np.ndarray:
    stack = np.asarray(stack, dtype=float)
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)
    if stack.ndim != 3:
        raise ValueError("Expected stack shape: n_members × n_samples × n_classes")
    method = str(method)
    if method == "median_proba":
        proba = np.median(stack, axis=0)
    elif method == "weighted_mean_proba":
        w = np.clip(weights - np.nanmin(weights), 0, None) + 1e-8
        w = w / w.sum()
        proba = np.tensordot(w, stack, axes=(0, 0))
    elif method == "rank_mean":
        ranked = np.empty_like(stack)
        for m in range(stack.shape[0]):
            for c in range(stack.shape[2]):
                ranked[m, :, c] = rankdata(stack[m, :, c])
        proba = ranked.mean(axis=0)
    elif method == "majority_vote":
        preds = stack.argmax(axis=2)
        proba = np.zeros(stack.shape[1:], dtype=float)
        for i in range(stack.shape[1]):
            counts = np.bincount(preds[:, i], minlength=stack.shape[2]).astype(float)
            proba[i, :] = counts / max(1, counts.sum())
    elif method == "max_proba":
        proba = stack.max(axis=0)
    elif method == "min_proba":
        proba = stack.min(axis=0)
    else:
        proba = stack.mean(axis=0)
    return _renormalize_proba(proba, stack.shape[2])


def _complete_inner_scores(
    inner: pd.DataFrame,
    outer_split_key: str | None,
    metric: str,
    eligible_ids: set[str],
) -> pd.Series:
    required = {"inner_key", "config_id", metric}
    missing = required - set(inner.columns)
    if missing:
        raise ValueError(
            f"Inner-results table is missing required columns: {sorted(missing)}"
        )
    frame = inner.copy()
    if outer_split_key is not None:
        if "split_key" not in frame.columns:
            raise ValueError("Inner-results table must contain split_key.")
        frame = frame[frame["split_key"].astype(str).eq(str(outer_split_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    frame["inner_key"] = frame["inner_key"].astype(str)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[frame["config_id"].isin(eligible_ids)]
    if "ok" in frame.columns:
        ok = pd.to_numeric(frame["ok"], errors="coerce").fillna(0).astype(int)
        frame = frame[ok.eq(1)]
    expected = set(frame["inner_key"].dropna().astype(str))
    if not expected:
        return pd.Series(dtype=float)
    rows = []
    for config_id, group in frame.groupby("config_id", sort=True):
        group = group.drop_duplicates("inner_key", keep="last")
        valid = group[np.isfinite(group[metric].to_numpy(dtype=float))]
        if set(valid["inner_key"].astype(str)) != expected:
            continue
        rows.append((str(config_id), float(valid[metric].mean())))
    if not rows:
        return pd.Series(dtype=float)
    scores = pd.Series(dict(rows), dtype=float)
    order = sorted(scores.index, key=lambda cid: (-float(scores[cid]), str(cid)))
    return scores.loc[order]


def _prediction_frame_for_outer(
    predictions: pd.DataFrame,
    outer_split_key: str | None,
) -> pd.DataFrame:
    frame = predictions.copy()
    if "outer_split_key" not in frame.columns:
        if "split_key" not in frame.columns:
            raise ValueError(
                "Prediction table must contain outer_split_key or split_key."
            )
        frame["outer_split_key"] = frame["split_key"]
    if outer_split_key is not None:
        frame = frame[frame["outer_split_key"].astype(str).eq(str(outer_split_key))]
    frame["config_id"] = frame["config_id"].astype(str)
    return frame


def _aligned_stack(
    predictions: pd.DataFrame,
    members: list[str],
    pcols: list[str],
    inner: bool,
) -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    if not members:
        return None, None
    if inner:
        if "split_key" not in predictions.columns:
            raise ValueError("Inner predictions must contain split_key.")
        key_cols = ["split_key", "sample_id"]
    else:
        key_cols = ["sample_id"]
    base_cols = key_cols + ["y_true"]
    ordered = (
        predictions[base_cols]
        .drop_duplicates(key_cols)
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    stacks = []
    for cid in members:
        sub = predictions[predictions["config_id"].eq(str(cid))].copy()
        sub = sub.drop_duplicates(key_cols, keep="last")
        sub = ordered[key_cols].merge(
            sub[key_cols + pcols], on=key_cols, how="left", validate="one_to_one"
        )
        if sub[pcols].isna().any().any():
            return None, None
        stacks.append(sub[pcols].to_numpy(dtype=float))
    if len(stacks) != len(members):
        return None, None
    return ordered, np.stack(stacks, axis=0)


def _score_inner_candidate(
    spec: dict[str, Any],
    scores: pd.Series,
    predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    pcols: list[str],
) -> dict[str, Any] | None:
    members = _select_members(scores, configs, spec, plan)
    if len(members) < 2:
        return None
    weights = np.asarray(
        [float(scores.get(cid, np.nan)) for cid in members], dtype=float
    )
    fold_metrics = []
    for inner_key, fold in predictions.groupby("split_key", sort=True):
        ordered, stack = _aligned_stack(fold, members, pcols, False)
        if ordered is None or stack is None:
            return None
        proba = _aggregate_proba(stack, weights, str(spec["aggregation_strategy"]))
        classes = np.arange(proba.shape[1], dtype=int)
        y_true = ordered["y_true"].to_numpy(dtype=int)
        y_pred = classes[proba.argmax(axis=1)]
        fold_metrics.append(compute_metrics(y_true, y_pred, proba, classes))
    if not fold_metrics:
        return None
    tab = pd.DataFrame(fold_metrics)
    means = tab.mean(numeric_only=True).to_dict()
    stds = tab.std(numeric_only=True, ddof=1).fillna(0.0).to_dict()
    counts = tab.count(numeric_only=True).to_dict()
    row = {
        **spec,
        "members": json.dumps(members),
        "member_count": int(len(members)),
        "n_inner_folds": int(len(fold_metrics)),
    }
    row.update({f"{key}_mean": float(value) for key, value in means.items()})
    row.update({f"{key}_std": float(value) for key, value in stds.items()})
    row.update({f"{key}_count": int(value) for key, value in counts.items()})
    return row


def _candidate_table_for_inner(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
    outer_split_key: str | None,
    available_ids: set[str] | None = None,
) -> pd.DataFrame:
    eligible = _eligible_config_ids(configs, plan)
    if available_ids is not None:
        eligible &= {str(x) for x in available_ids}
    scores = _complete_inner_scores(inner_results, outer_split_key, metric, eligible)
    if scores.empty:
        return pd.DataFrame()
    pred = _prediction_frame_for_outer(inner_predictions, outer_split_key)
    pred = pred[pred["config_id"].isin(set(scores.index))].copy()
    pcols = _proba_cols(pred)
    if not pcols:
        raise ValueError("Inner predictions do not contain probability columns.")
    rows = []
    for spec in _ensemble_configs(plan):
        row = _score_inner_candidate(spec, scores, pred, configs, plan, pcols)
        if row is None:
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    score_col = f"{metric}_mean" if f"{metric}_mean" in out.columns else "nMCC_mean"
    out = out.sort_values(
        [score_col, "ensemble_config_id"], ascending=[False, True], kind="mergesort"
    )
    return out.reset_index(drop=True)


def select_mpma_e_by_outer_fold(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    outer_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = _prediction_frame_for_outer(outer_predictions, None)
    inner_pred = _prediction_frame_for_outer(inner_predictions, None)
    outer_pcols = _proba_cols(outer)
    if not outer_pcols:
        raise ValueError("Outer predictions do not contain probability columns.")
    selections = []
    prediction_rows = []
    metric_rows = []
    outer_keys = sorted(set(outer["outer_split_key"].astype(str)))
    for outer_key in outer_keys:
        fold_outer = outer[outer["outer_split_key"].astype(str).eq(outer_key)].copy()
        fold_inner_pred = inner_pred[
            inner_pred["outer_split_key"].astype(str).eq(outer_key)
        ].copy()
        if fold_outer.empty or fold_inner_pred.empty:
            continue
        available_ids = set(fold_outer["config_id"].astype(str))
        candidates = _candidate_table_for_inner(
            inner_results,
            inner_predictions,
            configs,
            plan,
            metric,
            outer_key,
            available_ids,
        )
        if candidates.empty:
            continue
        score_col = (
            f"{metric}_mean" if f"{metric}_mean" in candidates.columns else "nMCC_mean"
        )
        winner = candidates.iloc[0].to_dict()
        members = json.loads(str(winner["members"]))
        eligible = _eligible_config_ids(configs, plan) & available_ids
        scores = _complete_inner_scores(inner_results, outer_key, metric, eligible)
        ordered, stack = _aligned_stack(fold_outer, members, outer_pcols, False)
        if ordered is None or stack is None:
            continue
        weights = np.asarray(
            [float(scores.get(cid, np.nan)) for cid in members], dtype=float
        )
        proba = _aggregate_proba(stack, weights, str(winner["aggregation_strategy"]))
        classes = np.arange(proba.shape[1], dtype=int)
        y_true = ordered["y_true"].to_numpy(dtype=int)
        y_pred = classes[proba.argmax(axis=1)]
        outer_metrics = compute_metrics(y_true, y_pred, proba, classes)
        selections.append(
            {
                "outer_split_key": outer_key,
                "ensemble_config_id": str(winner["ensemble_config_id"]),
                "selection_basis": "outer_fold_inner_validation_predictions",
                "selection_metric": str(metric),
                "inner_score": float(winner[score_col]),
                "selection_strategy": str(winner["selection_strategy"]),
                "aggregation_strategy": str(winner["aggregation_strategy"]),
                "ensemble_size": int(winner["ensemble_size"]),
                "member_count": int(winner["member_count"]),
                "members": json.dumps(members),
            }
        )
        metric_rows.append(
            {
                "outer_split_key": outer_key,
                "ensemble_config_id": str(winner["ensemble_config_id"]),
                **outer_metrics,
            }
        )
        ordered = ordered.copy()
        ordered["outer_split_key"] = outer_key
        ordered["ensemble_config_id"] = str(winner["ensemble_config_id"])
        ordered["selection_strategy"] = str(winner["selection_strategy"])
        ordered["aggregation_strategy"] = str(winner["aggregation_strategy"])
        ordered["members"] = json.dumps(members)
        ordered["y_pred"] = y_pred.astype(int)
        for j, col in enumerate(outer_pcols):
            ordered[col] = proba[:, j]
        prediction_rows.extend(ordered.to_dict(orient="records"))
    return (
        pd.DataFrame(selections),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
    )


def summarize_mpma_e_strategy(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    if fold_metrics is None or fold_metrics.empty:
        return {}
    out: dict[str, Any] = {
        "Strategy": "MPMA-E",
        "selection_basis": "outer_fold_inner_validation_predictions",
        "n_outer_folds": int(len(fold_metrics)),
    }
    for metric in fold_metrics.columns:
        if metric in {"outer_split_key", "ensemble_config_id"}:
            continue
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


def select_final_mpma_e_candidate(
    inner_results: pd.DataFrame,
    inner_predictions: pd.DataFrame,
    configs: pd.DataFrame,
    plan: Ensemble,
    metric: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = _candidate_table_for_inner(
        inner_results,
        inner_predictions,
        configs,
        plan,
        metric,
        None,
        None,
    )
    if candidates.empty:
        return {}, candidates
    score_col = (
        f"{metric}_mean" if f"{metric}_mean" in candidates.columns else "nMCC_mean"
    )
    candidates = candidates.rename(columns={score_col: "inner_score"})
    first = candidates.iloc[0].to_dict()
    best = {
        "ensemble_config_id": str(first["ensemble_config_id"]),
        "selection_basis": "all_inner_validation_predictions_for_final_refit",
        "selection_metric": str(metric),
        "inner_score": float(first["inner_score"]),
        "selection_strategy": str(first["selection_strategy"]),
        "aggregation_strategy": str(first["aggregation_strategy"]),
        "ensemble_size": int(first["ensemble_size"]),
        "member_count": int(first["member_count"]),
        "members": str(first["members"]),
    }
    for key, value in first.items():
        if key in best or key in {"optimize_metric", "threshold_score"}:
            continue
        if key.startswith("outer_"):
            continue
        best[f"inner_{key}"] = value
    return best, candidates


def sweep_ensemble(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    ensemble_dir = root / "ensembling"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    outer_path = root / "predictions" / "outer_predictions.tsv"
    inner_result_path = root / "inner_results" / "inner_results.tsv"
    inner_prediction_path = root / "inner_predictions" / "inner_predictions.tsv"
    config_path = root / "configs.tsv"
    required = [outer_path, inner_result_path, inner_prediction_path, config_path]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("Run evaluate(sweep) before sweep_ensemble(sweep).")
    outer_predictions = pd.read_csv(outer_path, sep="\t")
    inner_results = pd.read_csv(inner_result_path, sep="\t")
    inner_predictions = pd.read_csv(inner_prediction_path, sep="\t")
    configs = pd.read_csv(config_path, sep="\t")
    metric = sweep.ensemble.optimize_metric
    if metric not in inner_results.columns:
        metric = (
            sweep.evaluation.optimize_metric
            if sweep.evaluation.optimize_metric in inner_results.columns
            else "nMCC"
        )
    stage("Ensemble sweep", str(root))
    summary_table(
        "Ensemble search",
        {
            "candidate ensembles": f"{len(_ensemble_configs(sweep.ensemble)):,}",
            "selection strategies": sweep.ensemble.selection_strategies,
            "aggregation strategies": sweep.ensemble.aggregation_strategies,
            "ensemble sizes": sweep.ensemble.sizes,
            "excluded learners": sweep.ensemble.exclude_learners or "none",
            "excluded resolutions": sweep.ensemble.exclude_resolutions or "none",
            "excluded transformations": sweep.ensemble.exclude_transformations
            or "none",
            "optimize metric": metric,
        },
    )
    resource_tracker = ResourceTracker(
        sample_interval_s=float(
            getattr(sweep.evaluation, "resource_sample_interval_s", 0.10)
        )
    ).start()
    selection, selected_outer_predictions, fold_metrics = select_mpma_e_by_outer_fold(
        inner_results,
        inner_predictions,
        outer_predictions,
        configs,
        sweep.ensemble,
        metric,
    )
    if selection.empty or selected_outer_predictions.empty or fold_metrics.empty:
        raise RuntimeError("No valid nested ensemble selections were produced.")
    final_ensemble, candidates = select_final_mpma_e_candidate(
        inner_results,
        inner_predictions,
        configs,
        sweep.ensemble,
        metric,
    )
    if not final_ensemble:
        raise RuntimeError(
            "No valid final ensemble candidate was produced from inner validation predictions."
        )
    nested_summary = summarize_mpma_e_strategy(fold_metrics)
    final_mpma = select_final_mpma_candidate(
        inner_results,
        configs,
        metric,
        plan=sweep.ensemble,
    )
    selection_path = ensemble_dir / "mpma_e_outer_selection.tsv"
    prediction_path = ensemble_dir / "ensemble_predictions.tsv"
    result_path = ensemble_dir / "mpma_e_outer_results.tsv"
    candidate_path = ensemble_dir / "ensemble_candidate_scores.tsv"
    summary_path = ensemble_dir / "mpma_e_strategy_summary.json"
    final_path = ensemble_dir / "mpma_e_final_candidate.json"
    selection.to_csv(selection_path, sep="\t", index=False)
    selected_outer_predictions.to_csv(prediction_path, sep="\t", index=False)
    fold_metrics.to_csv(result_path, sep="\t", index=False)
    candidates.to_csv(candidate_path, sep="\t", index=False)
    dump_json_standard(nested_summary, summary_path)
    dump_json_standard(final_ensemble, final_path)
    selected = {
        "inner_val_best_mpma": final_mpma,
        "inner_val_best_mpmas_ensemble": final_ensemble,
        "nested_mpma_e": nested_summary,
        "terminology": {
            "MPMA-B": "fold-specific single MPMA selected by inner-validation scoring for nested performance; separate final candidate selected from all inner validation for refit",
            "MPMA-E": "fold-specific ensemble specification selected by inner-validation predictions for nested performance; separate final candidate selected from all inner validation predictions for refit",
        },
    }
    dump_json_standard(selected, ensemble_dir / "selected_unit.json")
    comparison = pd.DataFrame(
        [
            {
                "unit": "MPMA-B final candidate",
                "config_id": final_mpma.get("config_id", ""),
                "selection_basis": final_mpma.get("selection_basis", ""),
                "inner_score": final_mpma.get("inner_score", np.nan),
                "members": "",
            },
            {
                "unit": "MPMA-E final candidate",
                "config_id": final_ensemble.get("ensemble_config_id", ""),
                "selection_basis": final_ensemble.get("selection_basis", ""),
                "inner_score": final_ensemble.get("inner_score", np.nan),
                "members": final_ensemble.get("members", "[]"),
            },
        ]
    )
    comparison.to_csv(
        ensemble_dir / "final_model_comparison.tsv", sep="\t", index=False
    )
    resource_path = ensemble_dir / "mpma_e_selection_resources.json"
    dump_json_standard(resource_tracker.stop(), resource_path)
    stale_candidate_plot = ensemble_dir / "ensemble_candidates.png"
    if stale_candidate_plot.exists():
        stale_candidate_plot.unlink()
    X_fig, taxa_fig, source_fig = _matrix_for_mpma_e_figure(sweep)
    mpma_e_outputs = write_single_task_mpma_e_figure(
        root,
        task_key=root.name,
        task_title=sweep.title,
        X=X_fig,
        taxa=taxa_fig,
        source=source_fig,
        out_dir=root / "figures",
        out_name="mpma_e",
        include_inactive_configs=True,
        max_members=20,
        seed=sweep.evaluation.random_state,
    )
    success(
        f"Ensemble sweep completed · final candidate={final_ensemble['ensemble_config_id']} · inner {metric}={final_ensemble['inner_score']:.4f}"
    )
    outputs = {
        "ensemble_dir": ensemble_dir,
        "selected_unit": ensemble_dir / "selected_unit.json",
        "comparison": ensemble_dir / "final_model_comparison.tsv",
        "mpma_e_selection": selection_path,
        "mpma_e_predictions": prediction_path,
        "mpma_e_outer_results": result_path,
        "mpma_e_candidates": candidate_path,
        "mpma_e_summary": summary_path,
        "mpma_e_final_candidate": final_path,
        "mpma_e_selection_resources": resource_path,
        **mpma_e_outputs,
    }
    path_table("Ensemble outputs", outputs)
    return outputs


def _matrix_for_mpma_e_figure(sweep: Sweep) -> tuple[np.ndarray, list[str], str]:
    X, feature_names = _raw_input_matrix_for_figure(sweep)
    if X.size == 0 or len(feature_names) == 0:
        raise RuntimeError(
            "No raw abundance matrix is available for MPMA-E visualisation."
        )
    return X, feature_names, "raw input abundance matrix"


def _raw_input_matrix_for_figure(sweep: Sweep) -> tuple[np.ndarray, list[str]]:
    spec = sweep.data
    abundance_path = Path(spec.abundance_path)
    metadata_path = Path(spec.metadata_path) if spec.metadata_path is not None else None
    fmt = spec.format
    if fmt == "auto":
        fmt = (
            "metaphlan_tsv"
            if abundance_path.suffix.lower() in {".tsv", ".txt"} and metadata_path
            else "wide_csv"
        )
    if fmt in {"matrix_tsv", "metaphlan_tsv", "profile_tsv"}:
        if metadata_path is None:
            raise ValueError(
                "Data.metadata_path is required for MetaPhlAn-style TSV input."
            )
        meta = pd.read_csv(metadata_path, sep=None, engine="python", dtype=str)
        bio = pd.read_csv(abundance_path, sep="\t", index_col=0, low_memory=False)
        bio.index = bio.index.astype(str).str.strip()
        bio.columns = bio.columns.astype(str).str.strip()
        meta[spec.sample_id_col] = meta[spec.sample_id_col].astype(str).str.strip()
        common = [
            sid for sid in meta[spec.sample_id_col].tolist() if sid in set(bio.columns)
        ]
        if not common:
            raise ValueError(
                "No sample IDs overlap between metadata and abundance matrix."
            )
        X = bio[common].T.to_numpy(dtype=np.float32)
        return X, bio.index.tolist()
    if fmt in {"csv", "wide_csv"}:
        df = pd.read_csv(abundance_path)
        if metadata_path is not None:
            meta = pd.read_csv(metadata_path, sep=None, engine="python")
            if (
                spec.sample_id_col not in df.columns
                or spec.sample_id_col not in meta.columns
            ):
                raise ValueError(
                    f"sample_id_col={spec.sample_id_col!r} must exist in both CSV files."
                )
            df = df.merge(
                meta, on=spec.sample_id_col, how="inner", suffixes=("", "__meta")
            )
        reserved = {spec.sample_id_col, spec.target_col, *(spec.metadata_cols or ())}
        if spec.group_col:
            reserved.add(spec.group_col)
        numeric_cols = [
            c
            for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_cols:
            raise ValueError(
                "No numeric abundance columns found after excluding metadata columns."
            )
        return df[numeric_cols].to_numpy(dtype=np.float32), [
            str(c) for c in numeric_cols
        ]
    dataset = load_dataset(sweep.data, TAXONOMIC_LEVELS)
    if "all" in dataset.X_by_level:
        return dataset.X_by_level["all"], dataset.feature_names_by_level.get("all", [])
    blocks = list(dataset.X_by_level.values())
    names = [name for lv in dataset.feature_names_by_level.values() for name in lv]
    return np.concatenate(blocks, axis=1), names
