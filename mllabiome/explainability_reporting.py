from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .explainability_context import build_feature_relative_abundance_summary
from .explainability_support import top_k_rank_support
from .explainability_visuals import plot_feature_support as _plot_feature_support_visual
from .explainability_visuals import (
    plot_interaction_network as _plot_interaction_network_visual,
)
from .storage import read_table, table_exists
from .style import COL_W_2, DIM, MID
from .style import apply as apply_style
from .style import save_all
from .utils import feature_tail_ellipsis


def _class_slug(label: Any) -> str:
    text = str(label).strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "class"


def _collapse_duplicate_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "feature" not in frame.columns:
        return frame.copy()
    keys = ["feature"]
    if "class_index" in frame.columns:
        keys = ["class_index", "feature"]
    if not frame.duplicated(keys).any():
        return frame.copy()
    d = frame.copy()
    d["feature"] = d["feature"].astype(str)
    d["importance_mean"] = pd.to_numeric(
        d.get("importance_mean", np.nan), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    if "importance_sd" not in d.columns:
        d["importance_sd"] = np.nan
    agg: dict[str, Any] = {
        "importance_mean": lambda x: pd.to_numeric(x, errors="coerce").sum(min_count=1),
        "importance_sd": lambda x: float(
            np.sqrt(
                np.nansum(
                    np.square(pd.to_numeric(x, errors="coerce").to_numpy(dtype=float))
                )
            )
        ),
    }
    for column in (
        "method",
        "class_label",
        "scoring",
        "n_methods",
        "n_methods_total",
        "method_coverage",
        "n_estimable_folds",
        "n_outer_folds_total",
        "fold_coverage",
        "importance_median",
        "importance_q25",
        "importance_q75",
        "mean_rank",
        "median_rank",
        "rank_iqr",
        "top_k_frequency",
        "signed_importance_mean",
        "sign_positive_fraction",
        "sign_negative_fraction",
        "sign_consistency",
    ):
        if column in d.columns:
            agg[column] = "first"
    return d.groupby(keys, as_index=False, sort=False).agg(agg)


def _rank_support_from_importance(
    frame: pd.DataFrame, top_k: int | None = None
) -> pd.Series:
    d = _collapse_duplicate_feature_importance(frame).copy()
    if "class_index" not in d.columns:
        d["class_index"] = 0
    pieces: list[pd.Series] = []
    for class_index, sub in d.groupby("class_index", sort=True):
        values = pd.to_numeric(sub["importance_mean"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = values.notna()
        if not valid.any():
            continue
        ranked = values.loc[valid].rank(ascending=False, method="average")
        n = int(valid.sum())
        if top_k is None:
            support = (
                pd.Series(1.0, index=ranked.index, dtype=float)
                if n == 1
                else 1.0 - (ranked - 1.0) / float(n - 1)
            )
        else:
            support = top_k_rank_support(values.loc[valid], int(top_k))
        index = pd.MultiIndex.from_arrays(
            [
                np.full(len(support), int(class_index), dtype=int),
                sub.loc[valid, "feature"].astype(str).to_numpy(),
            ],
            names=["class_index", "feature"],
        )
        pieces.append(
            pd.Series(support.to_numpy(dtype=float), index=index, dtype=float)
        )
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def _method_display(method: str) -> str:
    m = str(method).strip().lower()
    return {
        "shap": "SHAP",
        "lime": "LIME",
        "ale": "ALE",
        "permutation": "Permutation",
        "consensus": "consensus",
    }.get(m, str(method))


def _method_support_table(
    frames: Sequence[pd.DataFrame], imp: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    if "class_index" not in imp.columns:
        imp = imp.copy()
        imp["class_index"] = 0
        imp["class_label"] = "class_0"
    pieces: list[pd.DataFrame] = []
    supports: dict[str, pd.Series] = {}
    for frame in frames:
        method = _method_display(frame["method"].iloc[0])
        supports[method] = _rank_support_from_importance(frame, top_k=top_k)
    for class_index, class_imp in imp.groupby("class_index", sort=True):
        top = class_imp.copy()
        keys = pd.MultiIndex.from_arrays(
            [
                np.full(len(top), int(class_index), dtype=int),
                top["feature"].astype(str).to_numpy(),
            ]
        )
        for method, support in supports.items():
            top[method] = support.reindex(keys).to_numpy(dtype=float)
        present_columns = [
            col for col in ("SHAP", "LIME", "ALE", "Permutation") if col in top.columns
        ]
        for col in ("SHAP", "LIME", "ALE", "Permutation"):
            if col not in top.columns:
                top[col] = np.nan
        top["consensus"] = (
            top[present_columns].mean(axis=1, skipna=True)
            if present_columns
            else np.nan
        )
        top["n_methods"] = (
            top[present_columns].notna().sum(axis=1) if present_columns else 0
        )
        top["n_methods_total"] = len(present_columns)
        top["method_coverage"] = (
            top["n_methods"] / float(len(present_columns))
            if present_columns
            else np.nan
        )
        top = top.sort_values(
            ["consensus", "n_methods", "importance_mean", "feature"],
            ascending=[False, False, False, True],
            na_position="last",
        ).head(int(top_k))
        top["rank"] = np.arange(1, len(top) + 1, dtype=int)
        pieces.append(top)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    cols = [
        "class_index",
        "class_label",
        "rank",
        "feature",
        "SHAP",
        "LIME",
        "ALE",
        "Permutation",
        "consensus",
        "importance_mean",
        "mean_rank",
        "n_methods",
        "n_methods_total",
        "method_coverage",
    ]
    return out[[c for c in cols if c in out.columns]]


def _terminal_taxon_label(value: str) -> str:
    text = str(value).split("___")[-1].split("|")[-1]
    rank = ""
    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if text.startswith(pfx):
            rank = pfx[0] + ". "
            text = text[len(pfx) :]
            break
    text = text.replace("_", " ").strip()
    return f"{rank}{text}" if text else str(value).replace("_", " ")


def _plain_taxon_label(feature_name: str, max_len: int = 34) -> str:
    text = str(feature_name)
    if text.startswith("ALR[") and text.endswith("]"):
        body = text[4:-1]
        if "/" in body:
            numerator, reference = (part.strip() for part in body.split("/", 1))
            out = f"ALR[{_terminal_taxon_label(numerator)} / {_terminal_taxon_label(reference)}]"
        else:
            out = text
    elif text.startswith("ILR_"):
        parts = text.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            out = f"ILR balance {int(parts[1])} · {parts[2][:6]}"
        else:
            out = text.replace("_", " ")
    else:
        out = _terminal_taxon_label(text)
    return feature_tail_ellipsis(out, max_len)


def _plot_feature_importance(
    top_features: pd.DataFrame,
    stats: pd.DataFrame,
    out_stem: Path,
    top_k: int,
    class_labels: Sequence[str],
    stability: pd.DataFrame | None = None,
) -> None:
    _plot_feature_support_visual(
        top_features, stats, out_stem, top_k, class_labels, stability
    )


def _plot_interaction_network(
    tab: pd.DataFrame,
    stats: pd.DataFrame,
    out_stem: Path,
    top_k: int,
    class_labels: Sequence[str],
    layout: str = "default",
) -> bool:
    try:
        return bool(
            _plot_interaction_network_visual(
                tab, stats, out_stem, top_k, class_labels, layout=layout
            )
        )
    except Exception as exc:
        _write_unavailable_interaction_network(out_stem, str(exc))
        return False


def _write_unavailable_interaction_network(out_stem: Path, message: str) -> None:
    apply_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(COL_W_2, 82 * (1.0 / 25.4)))
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "Interaction network unavailable\nno finite 2D-ALE interaction strengths",
        ha="center",
        va="center",
        fontsize=7,
        color=DIM,
        linespacing=1.15,
    )
    ax.text(
        0.5,
        0.38,
        str(message)[:180],
        ha="center",
        va="center",
        fontsize=5.2,
        color=MID,
        linespacing=1.12,
    )
    save_all(fig, out_stem)
    plt.close(fig)


def _standardized_group_shift(
    control: np.ndarray,
    case: np.ndarray,
) -> tuple[float, float, float, float, float]:
    control = np.asarray(control, dtype=float)
    case = np.asarray(case, dtype=float)
    control = control[np.isfinite(control)]
    case = case[np.isfinite(case)]
    control_mean = float(np.mean(control)) if control.size else np.nan
    case_mean = float(np.mean(case)) if case.size else np.nan
    difference = (
        float(case_mean - control_mean)
        if np.isfinite(control_mean) and np.isfinite(case_mean)
        else np.nan
    )
    control_sd = float(np.std(control, ddof=1)) if control.size > 1 else np.nan
    case_sd = float(np.std(case, ddof=1)) if case.size > 1 else np.nan
    pooled_sd = np.nan
    if control.size > 1 and case.size > 1:
        denominator = control.size + case.size - 2
        if denominator > 0:
            pooled_variance = (
                (control.size - 1) * control_sd**2 + (case.size - 1) * case_sd**2
            ) / float(denominator)
            if np.isfinite(pooled_variance) and pooled_variance >= 0:
                pooled_sd = float(np.sqrt(pooled_variance))
    if np.isfinite(difference) and np.isfinite(pooled_sd) and pooled_sd > 0:
        standardized = float(difference / pooled_sd)
    elif np.isfinite(difference) and difference == 0:
        standardized = 0.0
    else:
        standardized = np.nan
    return control_mean, case_mean, difference, pooled_sd, standardized


def _interaction_distribution_stats_for_class(
    features: Sequence[str],
    feature_names: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    class_index: int,
) -> pd.DataFrame:
    index = {str(feature): i for i, feature in enumerate(feature_names)}
    target_mask = np.asarray(y, dtype=int) == int(class_index)
    other_mask = ~target_mask
    rows: list[dict[str, Any]] = []
    for feature in features:
        key = str(feature)
        if key not in index:
            continue
        values = np.asarray(X[:, index[key]], dtype=float)
        control_mean, case_mean, difference, pooled_sd, standardized = (
            _standardized_group_shift(values[other_mask], values[target_mask])
        )
        rows.append(
            {
                "feature": key,
                "control_mean_coordinate": control_mean,
                "case_mean_coordinate": case_mean,
                "case_minus_control_coordinate": difference,
                "pooled_sd_coordinate": pooled_sd,
                "standardized_mean_difference": standardized,
            }
        )
    return pd.DataFrame(rows)


def _ensure_interaction_network_outputs(
    target_dir: Path,
    figures_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    top_k: int,
    dataset: Any,
    coordinate_metadata: Sequence[Any] | None = None,
) -> dict[str, Path]:
    path = target_dir / "feature_interactions_current.parquet"
    if not table_exists(path):
        return {}
    table = read_table(path)
    if table.empty or "class_index" not in table.columns:
        return {}
    outputs: dict[str, Path] = {}
    for class_index in class_indices:
        c = int(class_index)
        class_table = table[table["class_index"].eq(c)].copy()
        class_table["interaction_strength"] = pd.to_numeric(
            class_table.get("interaction_strength", np.nan), errors="coerce"
        )
        class_table = class_table.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["interaction_strength"]
        )
        if class_table.empty:
            continue
        class_table = class_table.sort_values(
            "interaction_strength", ascending=False
        ).head(int(top_k))
        features = list(
            dict.fromkeys(
                class_table.get("feature_1", pd.Series(dtype=object))
                .astype(str)
                .tolist()
                + class_table.get("feature_2", pd.Series(dtype=object))
                .astype(str)
                .tolist()
            )
        )
        stats = _interaction_distribution_stats_for_class(
            features, feature_names, X, y, c
        )
        abundance = build_feature_relative_abundance_summary(
            dataset, features, coordinate_metadata
        )
        if not abundance.empty:
            stats = stats.merge(abundance, on="feature", how="outer")
        label = str(class_labels[c])
        other_labels = [str(v) for i, v in enumerate(class_labels) if i != c]
        reference_label = other_labels[0] if len(other_labels) == 1 else "Other classes"
        slug = _class_slug(label)
        stem = figures_dir / f"interaction_network_current__{slug}"
        _plot_interaction_network(
            class_table,
            stats,
            stem,
            int(top_k),
            (reference_label, label),
            layout="default",
        )
        outputs[stem.name] = stem.with_suffix(".svg")
    return outputs


def _feature_distribution_stats(
    features: list[str],
    feature_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    idx = {f: i for i, f in enumerate(feature_names)}
    rows = []
    for feat in features:
        if feat not in idx:
            continue
        vals = np.asarray(X[:, idx[feat]], dtype=float)
        finite = vals[np.isfinite(vals)]
        row = {
            "feature": feat,
            "overall_mean_coordinate": float(np.mean(finite))
            if finite.size
            else np.nan,
            "overall_median_coordinate": float(np.median(finite))
            if finite.size
            else np.nan,
            "overall_sd_coordinate": float(np.std(finite, ddof=1))
            if finite.size > 1
            else np.nan,
        }
        for c, label in enumerate(labels):
            sub = vals[np.asarray(y) == c]
            sub = sub[np.isfinite(sub)]
            row[f"mean_coordinate_{label}"] = (
                float(np.mean(sub)) if sub.size else np.nan
            )
            row[f"median_coordinate_{label}"] = (
                float(np.median(sub)) if sub.size else np.nan
            )
        if len(labels) >= 2:
            control_mean, case_mean, difference, pooled_sd, standardized = (
                _standardized_group_shift(
                    vals[np.asarray(y) == 0], vals[np.asarray(y) == 1]
                )
            )
            row["control_mean_coordinate"] = control_mean
            row["case_mean_coordinate"] = case_mean
            row["case_minus_control_coordinate"] = difference
            row["pooled_sd_coordinate"] = pooled_sd
            row["standardized_mean_difference"] = standardized
        rows.append(row)
    return pd.DataFrame(rows)
