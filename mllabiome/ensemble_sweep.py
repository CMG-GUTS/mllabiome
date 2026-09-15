from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .configs_sweep import Ensemble, Sweep
from .console import path_table, progress, stage, success, summary_table
from .data import load_dataset
from .metrics import _renormalize_proba, compute_metrics
from .mpma_e_figure import write_single_task_mpma_e_figure
from .utils import TAXONOMIC_LEVELS, dump_json_standard


def _load_manifest(root: Path) -> dict[str, Any]:
    with open(root / "manifest.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _proba_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("proba_")]


def _excluded_config_ids(configs: pd.DataFrame, ensemble: Ensemble) -> set[str]:
    """Return evaluated configuration IDs excluded from framework selection.

    Excluded configurations remain in the evaluation outputs and can therefore
    be surfaced as independent report comparators.  They are removed only from
    MPMA-B selection and MPMA-E member/candidate selection.
    """
    if configs is None or configs.empty or "config_id" not in configs.columns:
        return set()

    mask = pd.Series(False, index=configs.index, dtype=bool)

    ids = {str(x) for x in getattr(ensemble, "exclude_config_ids", ()) if str(x)}
    if ids:
        mask |= configs["config_id"].astype(str).isin(ids)

    def _match(column: str, values: tuple[str, ...]) -> None:
        nonlocal mask
        vals = {str(x).casefold() for x in values if str(x)}
        if vals and column in configs.columns:
            mask |= configs[column].astype(str).str.casefold().isin(vals)

    _match("learner", tuple(getattr(ensemble, "exclude_learners", ())))
    _match("resolution", tuple(getattr(ensemble, "exclude_resolutions", ())))
    _match(
        "count_transformation", tuple(getattr(ensemble, "exclude_transformations", ()))
    )

    return set(configs.loc[mask, "config_id"].astype(str))


def sweep_ensemble(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    ensemble_dir = root / "ensembling"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    pred_path = root / "predictions" / "outer_predictions.tsv"
    inner_path = root / "inner_results" / "inner_results.tsv"
    config_path = root / "configs.tsv"
    if not pred_path.exists() or not inner_path.exists():
        raise FileNotFoundError("Run evaluate(sweep) before sweep_ensemble(sweep).")

    preds = pd.read_csv(pred_path, sep="\t")
    inner = pd.read_csv(inner_path, sep="\t")
    configs = pd.read_csv(config_path, sep="\t")
    pcols = _proba_cols(preds)
    if not pcols:
        raise ValueError("Outer predictions do not contain probability columns.")

    metric = sweep.ensemble.optimize_metric
    if metric not in inner.columns:
        metric = (
            sweep.evaluation.optimize_metric
            if sweep.evaluation.optimize_metric in inner.columns
            else "nMCC"
        )

    all_candidate_rows: list[dict[str, Any]] = []
    all_fold_predictions: list[dict[str, Any]] = []
    candidate_specs = _ensemble_configs(sweep.ensemble)

    stage("Ensemble sweep", str(root))
    summary_table(
        "Ensemble search",
        {
            "candidate ensembles": f"{len(candidate_specs):,}",
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

    excluded_ids = _excluded_config_ids(configs, sweep.ensemble)
    complete_ids = (
        set(configs["config_id"].astype(str)) & set(preds["config_id"].astype(str))
    ) - excluded_ids
    preds = preds[preds["config_id"].astype(str).isin(complete_ids)].copy()
    inner = inner[
        inner["config_id"].astype(str).isin(complete_ids) & inner["ok"].eq(1)
    ].copy()
    if preds.empty or inner.empty:
        raise RuntimeError(
            "No complete MPMA predictions are available for ensemble search."
        )

    with progress() as prog:
        candidate_task = prog.add_task(
            "Ensemble candidates", total=len(candidate_specs)
        )
        for spec in candidate_specs:
            prog.update(
                candidate_task,
                description=f"Ensemble {spec['selection_strategy']} + {spec['aggregation_strategy']}",
            )
            fold_metrics: list[dict[str, float]] = []
            fold_member_sets: list[list[str]] = []
            for outer_split, fold_pred in preds.groupby("outer_split_key", sort=False):
                inner_subset = inner[
                    inner["split_key"].astype(str).eq(str(outer_split))
                ]
                if inner_subset.empty:
                    inner_subset = inner
                scores = (
                    inner_subset.groupby("config_id")[metric]
                    .mean()
                    .dropna()
                    .sort_values(ascending=False)
                )
                scores = scores[scores.index.astype(str).isin(complete_ids)]
                members = _select_members(scores, configs, spec, sweep.ensemble)
                if len(members) < 2:
                    continue
                ordered_samples = (
                    fold_pred[["sample_id", "y_true"]]
                    .drop_duplicates("sample_id")
                    .sort_values("sample_id")
                )
                stack, weights = [], []
                valid = True
                for cid in members:
                    sub = (
                        fold_pred[fold_pred["config_id"].astype(str).eq(str(cid))]
                        .set_index("sample_id")
                        .reindex(ordered_samples["sample_id"])
                    )
                    if sub[pcols].isna().any().any():
                        valid = False
                        break
                    stack.append(sub[pcols].to_numpy(dtype=float))
                    weights.append(float(scores.get(cid, np.nan)))
                if not valid or len(stack) < 2:
                    continue
                proba = _aggregate_proba(
                    np.stack(stack, axis=0),
                    np.asarray(weights, dtype=float),
                    spec["aggregation_strategy"],
                )
                classes = np.arange(proba.shape[1], dtype=int)
                y_true = ordered_samples["y_true"].to_numpy(dtype=int)
                y_pred = classes[proba.argmax(axis=1)]
                mets = compute_metrics(y_true, y_pred, proba, classes)
                fold_metrics.append(mets)
                fold_member_sets.append(members)
                for i, sid in enumerate(ordered_samples["sample_id"].tolist()):
                    row = {
                        "ensemble_config_id": spec["ensemble_config_id"],
                        "outer_split_key": outer_split,
                        "sample_id": sid,
                        "y_true": int(y_true[i]),
                        "y_pred": int(y_pred[i]),
                    }
                    for j, col in enumerate(pcols):
                        row[col] = float(proba[i, j])
                    all_fold_predictions.append(row)
            prog.advance(candidate_task)
            if not fold_metrics:
                continue
            fold_tab = pd.DataFrame(fold_metrics)
            means = fold_tab.mean(numeric_only=True).to_dict()
            stds = fold_tab.std(numeric_only=True, ddof=1).fillna(0.0).to_dict()
            counts = fold_tab.count(numeric_only=True).to_dict()
            row = {
                **spec,
                **{f"{k}_mean": float(v) for k, v in means.items()},
                "n_folds": len(fold_metrics),
            }
            row.update({f"{k}_std": float(v) for k, v in stds.items()})
            row.update({f"{k}_count": int(v) for k, v in counts.items()})
            row["members"] = json.dumps(_mode_member_set(fold_member_sets))
            all_candidate_rows.append(row)

    cand = pd.DataFrame(all_candidate_rows)
    if cand.empty:
        raise RuntimeError("No valid ensemble candidates were produced.")
    score_col = f"{sweep.ensemble.optimize_metric}_mean"
    if score_col not in cand.columns:
        score_col = "nMCC_mean"
    cand = cand.sort_values(score_col, ascending=False)
    cand.to_csv(ensemble_dir / "ensemble_candidate_scores.tsv", sep="\t", index=False)
    pd.DataFrame(all_fold_predictions).to_csv(
        ensemble_dir / "ensemble_predictions.tsv", sep="\t", index=False
    )

    rankings = pd.read_csv(root / "tables" / "mpma_rankings.tsv", sep="\t")
    best_mpma = _inner_val_best_mpma(inner, rankings, metric)
    best_ensemble = cand.iloc[0].to_dict()
    selected = {
        "inner_val_best_mpma": best_mpma,
        "inner_val_best_mpmas_ensemble": best_ensemble,
        "terminology": {
            "MPMA-B": "best single MPMA selected by inner-validation scoring",
            "MPMA-E": "ensemble of MPMAs selected and aggregated by inner-validation scoring",
        },
    }
    dump_json_standard(selected, ensemble_dir / "selected_unit.json")

    comp = _final_model_comparison(best_mpma, best_ensemble, score_col)
    comp.to_csv(ensemble_dir / "final_model_comparison.tsv", sep="\t", index=False)
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
        f"Ensemble sweep completed · best={best_ensemble['ensemble_config_id']} · {score_col}={best_ensemble[score_col]:.4f}"
    )
    outputs = {
        "ensemble_dir": ensemble_dir,
        "selected_unit": ensemble_dir / "selected_unit.json",
        "comparison": ensemble_dir / "final_model_comparison.tsv",
        **mpma_e_outputs,
    }
    path_table("Ensemble outputs", outputs)
    return outputs


def _matrix_for_mpma_e_figure(sweep: Sweep) -> tuple[np.ndarray, list[str], str]:
    """Return the raw input abundance matrix for the MPMA-E schematic.

    Member strips are derived from this matrix by aggregation to their selected
    MPDR ranks. The input strip reflects the complete profile table, not only
    the subset of ranks enabled in the sweep configuration.
    """
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
                        "selection_strategy": sel,
                        "ensemble_size": int(size),
                        "threshold_score": plan.threshold_score
                        if sel == "threshold"
                        else np.nan,
                        "aggregation_strategy": agg,
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
    scores: pd.Series, configs: pd.DataFrame, spec: dict[str, Any], plan: Ensemble
) -> list[str]:
    size = int(spec["ensemble_size"])
    sel = str(spec["selection_strategy"])
    ordered = [str(x) for x in scores.index.tolist()]
    meta = configs.drop_duplicates("config_id").set_index("config_id")
    if sel == "top_k":
        return ordered[:size]
    if sel == "threshold":
        return [cid for cid in ordered if float(scores[cid]) >= plan.threshold_score][
            :size
        ]
    if sel == "best_per_family":
        out, seen = [], set()
        for cid in ordered:
            fam = _model_family(meta.loc[cid, "learner"] if cid in meta.index else cid)
            if fam not in seen:
                out.append(cid)
                seen.add(fam)
            if len(out) >= size:
                break
        return out
    if sel == "best_per_resolution":
        out, seen = [], set()
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
            scores, configs, {**spec, "selection_strategy": "best_per_family"}, plan
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
    weights = np.nan_to_num(weights, nan=0.0)
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


def _mode_member_set(member_sets: list[list[str]]) -> list[str]:
    if not member_sets:
        return []
    counts: dict[str, int] = {}
    for members in member_sets:
        for cid in members:
            counts[cid] = counts.get(cid, 0) + 1
    first = member_sets[0]
    return sorted(
        counts, key=lambda c: (-counts[c], first.index(c) if c in first else 9999)
    )[: len(first)]


def _inner_val_best_mpma(
    inner: pd.DataFrame, outer_rankings: pd.DataFrame, metric: str
) -> dict[str, Any]:
    """Return the MPMA-B row selected by inner validation, augmented with outer estimates."""
    if inner is None or inner.empty or "config_id" not in inner.columns:
        return (
            outer_rankings.iloc[0].to_dict()
            if outer_rankings is not None and not outer_rankings.empty
            else {}
        )
    score_metric = (
        metric
        if metric in inner.columns
        else ("nMCC" if "nMCC" in inner.columns else None)
    )
    if score_metric is None:
        return (
            outer_rankings.iloc[0].to_dict()
            if outer_rankings is not None and not outer_rankings.empty
            else {}
        )
    ok = inner.copy()
    if "ok" in ok.columns:
        ok = ok[pd.to_numeric(ok["ok"], errors="coerce").fillna(0).eq(1)]
    if ok.empty:
        return (
            outer_rankings.iloc[0].to_dict()
            if outer_rankings is not None and not outer_rankings.empty
            else {}
        )
    id_cols = [
        c
        for c in [
            "config_id",
            "mpdr_id",
            "count_transformation",
            "transformation_abbreviation",
            "resolution",
            "levels",
            "learner",
        ]
        if c in ok.columns
    ]
    metrics = [
        c for c in ["nMCC", "AUC", "BalAcc", "Accuracy", "F1w"] if c in ok.columns
    ]
    agg = ok.groupby(id_cols, dropna=False)[metrics].agg(["mean", "std", "count"])
    agg.columns = [f"inner_{m}_{stat}" for m, stat in agg.columns]
    rank = agg.reset_index().sort_values(f"inner_{score_metric}_mean", ascending=False)
    best = rank.iloc[0].to_dict()
    if (
        outer_rankings is not None
        and not outer_rankings.empty
        and "config_id" in outer_rankings.columns
    ):
        cid = str(best.get("config_id", ""))
        match = outer_rankings[outer_rankings["config_id"].astype(str).eq(cid)]
        if not match.empty:
            outer = match.iloc[0].to_dict()
            for k, v in outer.items():
                if k.endswith("_mean") or k.endswith("_std") or k.endswith("_count"):
                    best[f"outer_{k}"] = v
                elif k not in best:
                    best[k] = v
    best["selection_basis"] = "inner_validation"
    best["score"] = best.get(f"inner_{score_metric}_mean", best.get("inner_nMCC_mean"))
    return best


def _final_model_comparison(
    best_mpma: dict[str, Any], best_ensemble: dict[str, Any], score_col: str
) -> pd.DataFrame:
    rows = []
    if best_mpma:
        rows.append(
            {
                "unit": "MPMA-B",
                "config_id": best_mpma.get("config_id", ""),
                "score": best_mpma.get(
                    "score",
                    best_mpma.get(score_col, best_mpma.get("nMCC_mean", np.nan)),
                ),
                "count_transformation": best_mpma.get("count_transformation", ""),
                "resolution": best_mpma.get("resolution", ""),
                "learner": best_mpma.get("learner", ""),
                "members": "",
            }
        )
    rows.append(
        {
            "unit": "MPMA-E",
            "config_id": best_ensemble.get("ensemble_config_id", ""),
            "score": best_ensemble.get(score_col, np.nan),
            "count_transformation": "",
            "resolution": "",
            "learner": f"{best_ensemble.get('selection_strategy')} + {best_ensemble.get('aggregation_strategy')}",
            "members": best_ensemble.get("members", "[]"),
        }
    )
    return pd.DataFrame(rows)


def _plot_ensemble_candidates(
    cand: pd.DataFrame, score_col: str, out: Path, top_n: int = 20
) -> None:
    if cand.empty:
        return
    import matplotlib.pyplot as plt

    top = cand.head(top_n).copy()
    labels = top.apply(
        lambda r: f"{r['selection_strategy']}\n{r['aggregation_strategy']}", axis=1
    )
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(top))))
    ax.barh(np.arange(len(top)), top[score_col].astype(float))
    ax.set_yticks(np.arange(len(top)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(score_col.replace("_", " "))
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
