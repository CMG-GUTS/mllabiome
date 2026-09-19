from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .storage import read_table, write_table
from .metrics import metric_is_loss
from .transformations import TRANSFORMATION_SPACE, transformation_label
from .utils import TAXONOMIC_LEVELS


def _plot_metric_bars(
    rank: pd.DataFrame, metric_col: str, out: Path, top_n: int = 20
) -> None:
    if rank.empty or metric_col not in rank.columns:
        return
    import matplotlib.pyplot as plt

    top = rank.head(top_n).copy()
    labels = top.apply(
        lambda r: (
            f"{r['learner']}\n{r['resolution']} | {r.get('transformation_abbreviation', r['count_transformation'])}"
        ),
        axis=1,
    )
    fig_h = max(3.5, 0.34 * len(top))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(np.arange(len(top)), top[metric_col].astype(float))
    ax.set_yticks(np.arange(len(top)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric_col.replace("_", " "))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _representation_metric_name(metric_col: str, df: pd.DataFrame) -> str:
    if metric_col in df.columns:
        return metric_col
    if f"{metric_col}_mean" in df.columns:
        return f"{metric_col}_mean"
    for candidate in ("nMCC", "AUC", "AUC_macro", "BalAcc", "Accuracy"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No supported metric column is available for the representation-impact figure."
    )


def _resolution_order_key(x: Any) -> tuple[int, int, int, str]:
    lab = str(x).strip().lower()
    pos = {r: i for i, r in enumerate(TAXONOMIC_LEVELS)}
    if lab in pos:
        return (0, pos[lab], -1, lab)
    sep = "-" if "-" in lab else ("+" if "+" in lab else None)
    if sep:
        parts = [p.strip() for p in lab.split(sep)]
        if len(parts) == 2 and parts[0] in pos and parts[1] in pos:
            return (1 if sep == "-" else 2, pos[parts[0]], pos[parts[1]], lab)
    return (9, 9, 9, lab)


def _best_cell_indices(values: np.ndarray, k: int, lower_is_better: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(arr))
    if finite.size == 0:
        return np.asarray([], dtype=int)
    k = max(1, min(int(k), int(finite.size)))
    vals = arr[finite]
    if lower_is_better:
        threshold = np.partition(vals, k - 1)[k - 1]
        return finite[vals <= threshold]
    threshold = np.partition(vals, finite.size - k)[finite.size - k]
    return finite[vals >= threshold]


def _eta2_one_way(df: pd.DataFrame, factor: str, metric: str) -> float:
    if factor not in df.columns or metric not in df.columns:
        return 0.0
    d = df[[factor, metric]].dropna()
    if d.empty:
        return 0.0
    y = d[metric].astype(float).to_numpy()
    ss_total = float(np.sum((y - y.mean()) ** 2))
    if ss_total <= 1e-12:
        return 0.0
    ss_between = 0.0
    for _, g in d.groupby(factor, dropna=False):
        vals = g[metric].astype(float).to_numpy()
        if len(vals):
            ss_between += len(vals) * float((vals.mean() - y.mean()) ** 2)
    return float(max(0.0, min(1.0, ss_between / ss_total)))


def _write_representation_impact_figure(root: Path, metric_col: str = "nMCC") -> None:
    pass
    result_path = root / "results" / "outer_results.parquet"
    if not result_path.exists() or result_path.stat().st_size == 0:
        return
    df = read_table(result_path)
    if df.empty or "ok" not in df.columns:
        return
    df = df[df["ok"].eq(1)].copy()
    if df.empty:
        return
    metric = _representation_metric_name(metric_col, df)
    if metric not in df.columns:
        return
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df[np.isfinite(df[metric])].copy()
    if df.empty:
        return

    if "transformation_abbreviation" not in df.columns:
        df["transformation_abbreviation"] = df["count_transformation"].map(
            lambda x: transformation_label(x).abbreviation
        )

    cells = (
        df.groupby(
            ["resolution", "count_transformation", "transformation_abbreviation"],
            dropna=False,
        )[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"{metric}_mean"})
    )
    (root / "tables").mkdir(parents=True, exist_ok=True)
    write_table(root / "tables" / "representation_impact_cells.parquet", cells)

    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    from .style import (
        C_DARK,
        C_HILITE,
        C_MID,
        C_SKY,
        COL_W_2,
        HMAP_CMAP,
        REPRESENTATION_RC,
    )

    mpl.rcParams.update(REPRESENTATION_RC)

    heat = (
        df.groupby(["resolution", "count_transformation"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    res_order = sorted(
        heat["resolution"].astype(str).unique(), key=_resolution_order_key
    )
    tr_ref = [info.key for info in TRANSFORMATION_SPACE]
    present = set(heat["count_transformation"].astype(str))
    tr_order = [x for x in tr_ref if x in present] + sorted(present - set(tr_ref))
    pivot = heat.pivot_table(
        index="resolution",
        columns="count_transformation",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=res_order, columns=tr_order)
    label_map = (
        df.drop_duplicates("count_transformation")
        .set_index("count_transformation")["transformation_abbreviation"]
        .to_dict()
    )
    tr_labels = [
        str(label_map.get(c, transformation_label(c).abbreviation))
        for c in pivot.columns
    ]

    mat = pivot.to_numpy(dtype=float)
    vals = mat[np.isfinite(mat)]
    if not len(vals):
        return
    vmin, vmax = float(vals.min()), float(vals.max())
    if abs(vmax - vmin) < 1e-12:
        pad = max(abs(vmin) * 0.01, 0.01)
        vmin -= pad
        vmax += pad
    vmid = 0.5 * (vmin + vmax)

    fig = plt.figure(figsize=(COL_W_2, 122 / 25.4), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[0.78, 1.68],
        width_ratios=[1.12, 0.95],
        hspace=0.52,
        wspace=0.45,
        figure=fig,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    def tag(ax, label, x=-0.13, y=1.08):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )

    def trim(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    is_modality_sweep = "candidate_family" in df.columns or "modalities" in df.columns
    representation_label = (
        "Modality representation" if is_modality_sweep else "Taxonomic resolution"
    )
    learner_label = "Learner" if is_modality_sweep else "Classifier"
    transformation_factor_label = (
        "Transformation" if is_modality_sweep else "Feature representation"
    )
    res_summary = (
        df.groupby("resolution", dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reindex(res_order)
    )
    xs = np.arange(len(res_summary))
    yv = res_summary["mean"].to_numpy(dtype=float)
    se = res_summary["std"].fillna(0).to_numpy(dtype=float) / np.sqrt(
        np.maximum(res_summary["count"].to_numpy(dtype=float), 1.0)
    )
    ax_a.fill_between(xs, yv - se, yv + se, color=C_MID, alpha=0.16, linewidth=0)
    ax_a.plot(xs, yv, color=C_DARK, marker="o", markersize=3.5, linewidth=1.0)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([str(x) for x in res_summary.index], rotation=35, ha="right")
    ax_a.set_xlabel(representation_label, labelpad=3)
    ax_a.set_ylabel(metric.replace("_", "-"), labelpad=3)
    tag(ax_a, "a")
    trim(ax_a)

    factors = [
        (transformation_factor_label, "count_transformation"),
        (representation_label, "resolution"),
    ]
    if "learner" in df.columns and df["learner"].astype(str).nunique(dropna=True) > 1:
        factors.append((learner_label, "learner"))
    eta_vals = [_eta2_one_way(df, col, metric) for _, col in factors]
    ypos = np.arange(len(factors))
    colors = [C_DARK, C_MID, C_SKY][: len(factors)]
    ax_b.barh(ypos, eta_vals, color=colors, alpha=0.88, height=0.62, zorder=3)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([label for label, _ in factors])
    ax_b.set_xlim(0, max(1.0, max(eta_vals) * 1.05 if eta_vals else 1.0))
    ax_b.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax_b.set_xlabel(
        f"Proportion of {metric.replace('_', '-')} variance (η²)", labelpad=3
    )
    ax_b.invert_yaxis()
    tag(ax_b, "b")
    trim(ax_b)

    im = ax_c.imshow(
        mat,
        cmap=HMAP_CMAP,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    lower_is_better = metric_is_loss(metric)
    for ri in range(mat.shape[0]):
        finite_count = int(np.isfinite(mat[ri]).sum())
        if finite_count:
            k = max(1, int(np.ceil(0.10 * finite_count)))
            selected = _best_cell_indices(mat[ri], k, lower_is_better)
            for ci in selected:
                ax_c.add_patch(
                    plt.Rectangle(
                        (ci - 0.5, ri - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor=C_HILITE,
                        linewidth=0.42,
                    )
                )
    ax_c.set_yticks(np.arange(len(res_order)))
    ax_c.set_yticklabels(res_order)
    ax_c.set_xticks(np.arange(len(tr_labels)))
    ax_c.set_xticklabels(tr_labels, rotation=50, ha="right")
    ax_c.set_xlabel("Feature representation", labelpad=3)
    ax_c.set_ylabel(representation_label, labelpad=3)
    tag(ax_c, "c")
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    ax_c.tick_params(axis="both", which="both", length=0)
    cb = fig.colorbar(im, ax=ax_c, fraction=0.018, pad=0.012)
    cb.set_label(metric.replace("_", "-"), fontsize=5.5, labelpad=2)
    cb.set_ticks([vmin, vmid, vmax])
    ndp = 3 if (vmax - vmin) < 0.08 else 2
    cb.set_ticklabels([f"{v:.{ndp}f}" for v in (vmin, vmid, vmax)])
    cb.ax.tick_params(labelsize=5, length=2, width=0.4)
    cb.outline.set_visible(False)

    out_base = root / "figures" / "representation_impact"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def _plot_feature_importance(imp: pd.DataFrame, out: Path, top_k: int) -> None:
    import matplotlib.pyplot as plt

    top = imp.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.25 * len(top))))
    ax.barh(
        np.arange(len(top)),
        top["importance_mean"].astype(float),
        xerr=top["importance_sd"].astype(float),
    )
    ax.set_yticks(np.arange(len(top)), [_short_taxon(x) for x in top["feature"]])
    ax.set_xlabel("Permutation importance")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_interaction_network(tab: pd.DataFrame, out: Path, top_k: int) -> None:
    try:
        import networkx as nx
    except Exception as exc:
        raise ExplainabilityDependencyError(
            "Interaction-network plotting requires networkx. No fallback figure will be used."
        ) from exc
    import matplotlib.pyplot as plt

    d = tab.dropna(subset=["interaction_strength"]).head(int(top_k)).copy()
    if d.empty:
        raise ExplainabilityConfigurationError(
            "Interaction network was requested, but no finite interaction strengths were available. No fallback figure will be used."
        )
    G = nx.Graph()
    for _, r in d.iterrows():
        f1, f2 = str(r["feature_1"]), str(r["feature_2"])
        w = float(r["interaction_strength"])
        G.add_edge(f1, f2, weight=w)
    pos = nx.spring_layout(G, seed=42, weight="weight")
    weights = np.asarray([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    if len(weights) == 0:
        raise ExplainabilityConfigurationError(
            "Interaction network had no edges after filtering. No fallback figure will be used."
        )
    scale = weights / max(float(weights.max()), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.5 + 3.0 * scale, alpha=0.65)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=130)
    nx.draw_networkx_labels(
        G, pos, ax=ax, labels={n: _short_taxon(n, 28) for n in G.nodes()}, font_size=6
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _short_taxon(name: str, max_len: int = 70) -> str:
    s = str(name).split("|")[-1].split("___")[-1]
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


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
        vals = X[:, idx[feat]].astype(float)
        row = {
            "feature": feat,
            "overall_mean": float(np.mean(vals)),
            "overall_median": float(np.median(vals)),
            "prevalence": float(np.mean(vals > 0)),
        }
        for c, label in enumerate(labels):
            sub = vals[y == c]
            row[f"mean_{label}"] = float(np.mean(sub)) if len(sub) else np.nan
            row[f"prevalence_{label}"] = float(np.mean(sub > 0)) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
