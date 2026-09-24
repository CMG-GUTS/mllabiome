from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .explainability_support import top_k_rank_support
from .style import COL_W_2, compact_svg, enforce_nature_figure
from .utils import feature_tail_ellipsis

try:
    import networkx as nx
except Exception:
    nx = None


MM = 1.0 / 25.4
INK = "#0f172a"
MID = "#64748b"
DIM = "#94a3b8"
TRACK = "#e2e8f0"
BG = "#ffffff"
PANEL_BG = "#f8fafc"
ACC = "#2563eb"
ACC_D = "#1d4ed8"
ACC_L = "#dbeafe"
CLASS_CTRL = "#64748b"
CLASS_CASE = "#2563eb"
NET_NEUT = "#F7F7F7"
NET_EDGE_CMAP = LinearSegmentedColormap.from_list(
    "net_edge", ["#f7f9fc", "#DBE2E9"], N=256
)
SUPPORT_CMAP = LinearSegmentedColormap.from_list(
    "support", ["#ffffff", "#eff6ff", "#bfdbfe", "#60a5fa", "#1d4ed8"], N=256
)


def _support_text_color(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return INK
    rgba = SUPPORT_CMAP(float(np.clip(v, 0.0, 1.0)))

    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "#ffffff" if lum < 0.48 else INK


def _annotate_support_matrix(axis: plt.Axes, matrix: np.ndarray) -> None:
    if matrix.ndim != 2:
        return
    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            value = float(matrix[yi, xi])
            if not np.isfinite(value):
                continue
            axis.text(
                xi,
                yi,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=3.8,
                color=_support_text_color(value),
                zorder=5,
            )


def _draw_mean_support(axis: plt.Axes, values: np.ndarray) -> None:
    vals = np.asarray(values, dtype=float)
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([0.0, 1.0])
    axis.set_xticklabels(["0", "1"], fontsize=4.6, color="#000000")
    axis.tick_params(axis="x", length=1.6, width=0.35, pad=1, colors="#000000")
    axis.spines["bottom"].set_visible(True)
    axis.spines["bottom"].set_color("#000000")
    axis.spines["bottom"].set_linewidth(0.4)
    for yi, value in enumerate(vals):
        axis.barh(
            yi, 1.0, height=0.48, left=0.0, color=TRACK, edgecolor="none", zorder=1
        )
        if np.isfinite(value):
            v = float(np.clip(value, 0.0, 1.0))
            axis.plot([v, v], [yi - 0.20, yi + 0.20], color=ACC, lw=0.9, zorder=3)


RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "font.size": 7.0,
    "axes.linewidth": 0.45,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 6.0,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.02,
}


def apply_style() -> None:
    mpl.rcParams.update(RC)


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_suffix(".svg")
    enforce_nature_figure(fig)
    fig.savefig(path, dpi=300, metadata={"Date": None}, bbox_inches=None, pad_inches=0)
    compact_svg(path)


def _bbox(
    panel: Sequence[float], x: float, y: float, w: float, h: float
) -> list[float]:
    px, py, pw, ph = panel
    return [px + x * pw, py + y * ph, w * pw, h * ph]


def _xy(panel: Sequence[float], x: float, y: float) -> tuple[float, float]:
    px, py, pw, ph = panel
    return px + x * pw, py + y * ph


def _txt(
    fig: plt.Figure,
    panel: Sequence[float],
    x: float,
    y: float,
    text: str,
    *,
    size=7,
    color=INK,
    weight="normal",
    ha="left",
    va="center",
    rotation=0,
    linespacing=1.05,
    **kwargs,
):
    fx, fy = _xy(panel, x, y)
    return fig.text(
        fx,
        fy,
        text,
        fontsize=size,
        color=color,
        weight=weight,
        ha=ha,
        va=va,
        rotation=rotation,
        linespacing=linespacing,
        **kwargs,
    )


def _line(
    fig: plt.Figure,
    panel: Sequence[float],
    xs,
    ys,
    *,
    color=DIM,
    lw=0.5,
    zorder=3,
    **kwargs,
) -> None:
    fig.add_artist(
        mlines.Line2D(
            [_xy(panel, x, 0)[0] for x in xs],
            [_xy(panel, 0, y)[1] for y in ys],
            transform=fig.transFigure,
            color=color,
            lw=lw,
            clip_on=False,
            zorder=zorder,
            **kwargs,
        )
    )


def _bracket(
    fig: plt.Figure,
    panel: Sequence[float],
    x0: float,
    x1: float,
    y: float,
    label: str,
    *,
    tick=0.010,
    label_pad=0.006,
    color=MID,
    lw=0.6,
) -> None:
    _line(fig, panel, [x0, x0, x1, x1], [y - tick, y, y, y - tick], color=color, lw=lw)
    _txt(
        fig,
        panel,
        (x0 + x1) / 2,
        y + label_pad,
        label,
        size=5.6,
        color=color,
        ha="center",
        va="bottom",
    )


def _soft_missing(ax: plt.Axes, message: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=6.2,
        color=DIM,
        linespacing=1.15,
    )


def _style_black_bottom_axis(ax: plt.Axes, *, show_left: bool = False) -> None:
    axis_black = "#000000"
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(axis_black)
    ax.spines["bottom"].set_linewidth(0.45)
    ax.spines["left"].set_visible(show_left)
    if show_left:
        ax.spines["left"].set_color(axis_black)
        ax.spines["left"].set_linewidth(0.45)
    ax.tick_params(
        axis="x", length=2.2, width=0.45, color=axis_black, labelcolor=axis_black, pad=1
    )
    ax.tick_params(axis="y", length=0, color=axis_black, labelcolor=axis_black)


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


def _plain_taxon_label(feature_name: str, max_len: int | None = 32) -> str:
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
    return out if max_len is None else feature_tail_ellipsis(out, max_len)


def _net_short_label(feature_name: str) -> str:
    return _plain_taxon_label(str(feature_name), 42)


def _net_italic(label: str) -> str:
    text = str(label)
    if text.startswith("ALR[") or text.startswith("ILR balance "):
        return text
    parts = text.split(". ", 1)
    if len(parts) < 2:
        return text
    prefix, name = parts
    return rf"$\mathit{{{prefix}.}}$ {name.replace('$', '')}"


def _feature_label(feature_name: str) -> str:
    try:
        return _net_italic(_net_short_label(feature_name))
    except Exception:
        return _plain_taxon_label(str(feature_name), 42)


def _full_feature_label(feature_name: str) -> str:
    try:
        return _net_italic(_plain_taxon_label(str(feature_name), None))
    except Exception:
        return _plain_taxon_label(str(feature_name), None)


def _deduplicate_top_features(top: pd.DataFrame, max_features: int) -> pd.DataFrame:
    if "rank" in top.columns:
        top = top.sort_values("rank", ascending=True)
    top = top.drop_duplicates("feature", keep="first").head(max_features).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    return top


def _method_key(value: str) -> str:
    text = str(value).strip().casefold().replace(".", "")
    if text in {"perm", "permutation"}:
        return "permutation"
    return text


def _fold_stability_matrix(
    stability: pd.DataFrame | None,
    features: Sequence[str],
    methods: Sequence[str],
    class_index: int | None = None,
) -> np.ndarray:
    matrix = np.full((len(features), len(methods)), np.nan, dtype=float)
    if stability is None or stability.empty:
        return matrix
    if not {"method", "feature", "top_k_frequency"}.issubset(stability.columns):
        return matrix
    d = stability.copy()
    if class_index is not None and "class_index" in d.columns:
        values = pd.to_numeric(d["class_index"], errors="coerce")
        d = d[values.eq(int(class_index))].copy()
    if d.empty:
        return matrix
    d["_method_key"] = d["method"].map(_method_key)
    d["_feature_key"] = d["feature"].astype(str)
    d["_stability"] = pd.to_numeric(d["top_k_frequency"], errors="coerce")
    lookup = (
        d.dropna(subset=["_stability"])
        .drop_duplicates(["_method_key", "_feature_key"], keep="first")
        .set_index(["_method_key", "_feature_key"])["_stability"]
    )
    for i, feature in enumerate(features):
        for j, method in enumerate(methods):
            key = (_method_key(method), str(feature))
            if key in lookup.index:
                matrix[i, j] = float(np.clip(lookup.loc[key], 0.0, 1.0))
    return matrix


def plot_feature_support(
    top_features: pd.DataFrame,
    stats: pd.DataFrame,
    out_stem: Path,
    top_k: int,
    class_labels: Sequence[str] | None = None,
    stability: pd.DataFrame | None = None,
) -> bool:
    apply_style()
    raw_top = top_features.copy()
    if raw_top.empty or "feature" not in raw_top.columns:
        return False
    top = _deduplicate_top_features(raw_top, int(top_k))
    n = len(top)
    if n == 0:
        return False
    stat_map = (
        stats.drop_duplicates("feature").set_index("feature").to_dict("index")
        if stats is not None and "feature" in stats.columns
        else {}
    )
    features = top["feature"].astype(str).tolist()
    method_cols = [
        m for m in ("SHAP", "LIME", "Permutation", "ALE") if m in top.columns
    ]
    solo_method = len(method_cols) == 1
    class_index = None
    if "class_index" in top.columns:
        values = pd.to_numeric(top["class_index"], errors="coerce").dropna().unique()
        if len(values) == 1:
            class_index = int(values[0])
    fold_stability = _fold_stability_matrix(
        stability, features, method_cols, class_index=class_index
    )
    if solo_method and "top_k_frequency" in top.columns:
        fallback = pd.to_numeric(top["top_k_frequency"], errors="coerce").to_numpy(
            dtype=float
        )
        valid = np.isfinite(fallback)
        if fold_stability.shape == (len(top), 1):
            fold_stability[valid, 0] = np.clip(fallback[valid], 0.0, 1.0)

    fig_h_mm = max(52.0, 3.0 * n + 18.0)
    fig_w = (136.0 * MM) if solo_method else COL_W_2
    fig = plt.figure(figsize=(fig_w, fig_h_mm * MM))
    fig.patch.set_facecolor(BG)
    panel = [0.035, 0.035, 0.930, 0.930]

    if solo_method:
        ax_lab = fig.add_axes(_bbox(panel, 0.004, 0.035, 0.455, 0.820), zorder=5)
        ax_dir = fig.add_axes(_bbox(panel, 0.485, 0.035, 0.145, 0.820), zorder=5)
        ax_hm = fig.add_axes(_bbox(panel, 0.715, 0.035, 0.085, 0.820), zorder=5)
        ax_stab = fig.add_axes(_bbox(panel, 0.885, 0.035, 0.085, 0.820), zorder=5)
        ax_mean = None
        bracket_lab = (0.020, 0.455)
        bracket_dir = (0.485, 0.630)
        bracket_hm = (0.715, 0.800)
        bracket_stab = (0.885, 0.970)
    else:
        ax_lab = fig.add_axes(_bbox(panel, 0.004, 0.035, 0.350, 0.820), zorder=5)
        ax_dir = fig.add_axes(_bbox(panel, 0.380, 0.035, 0.095, 0.820), zorder=5)
        ax_hm = fig.add_axes(_bbox(panel, 0.505, 0.035, 0.195, 0.820), zorder=5)
        ax_mean = fig.add_axes(_bbox(panel, 0.725, 0.035, 0.060, 0.820), zorder=5)
        ax_stab = fig.add_axes(_bbox(panel, 0.815, 0.035, 0.155, 0.820), zorder=5)
        bracket_lab = (0.020, 0.350)
        bracket_dir = (0.380, 0.475)
        bracket_hm = (0.505, 0.700)
        bracket_mean = (0.725, 0.785)
        bracket_stab = (0.815, 0.970)

    axes_to_style = [ax_lab, ax_dir, ax_hm, ax_stab]
    if ax_mean is not None:
        axes_to_style.append(ax_mean)
    for ax in axes_to_style:
        ax.set_facecolor(BG)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for yi in np.arange(n + 1) - 0.5:
            ax.axhline(yi, color=TRACK, lw=0.25, zorder=0)

    ax_lab.set_xlim(0, 1)
    ax_lab.set_xticks([])
    for i, feat in enumerate(features):
        rank_val = int(top.iloc[i]["rank"])
        ax_lab.text(
            0.010,
            i,
            str(rank_val),
            ha="left",
            va="center",
            fontsize=5.0,
            color=DIM,
            clip_on=False,
        )
        ax_lab.text(
            0.120,
            i,
            _full_feature_label(feat),
            ha="left",
            va="center",
            fontsize=5.0,
            color=INK,
            clip_on=False,
        )

    shifts = []
    for feat in features:
        row = stat_map.get(feat, {})
        try:
            v = float(row.get("standardized_mean_difference", np.nan))
        except Exception:
            v = np.nan
        shifts.append(v)
    shift_clip = np.clip(
        np.nan_to_num(np.asarray(shifts, dtype=float), nan=0.0), -2.0, 2.0
    )
    ax_dir.set_xlim(-2.75, 2.75)
    ax_dir.axvline(0, color="#000000", lw=0.38, zorder=1, alpha=0.75)
    for i, v in enumerate(shift_clip):
        col = CLASS_CASE if v >= 0 else CLASS_CTRL
        ax_dir.plot(
            [0, v],
            [i, i],
            color=col,
            lw=1.0,
            solid_capstyle="round",
            alpha=0.86,
            clip_on=False,
        )
        ax_dir.scatter(
            [v], [i], s=9, color=col, edgecolors="none", zorder=3, clip_on=False
        )
    ax_dir.set_xticks([-2.0, 0.0, 2.0])
    ax_dir.set_xticklabels(["Control", "0", "Case"], fontsize=5.0, color="#000000")
    for lab in ax_dir.get_xticklabels():
        lab.set_clip_on(False)
    _style_black_bottom_axis(ax_dir, show_left=False)

    if method_cols:
        support_matrix = (
            top[method_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
        support_matrix = np.clip(support_matrix, 0, 1)
        cmap = SUPPORT_CMAP.copy()
        cmap.set_bad(TRACK)
        for axis, matrix in ((ax_hm, support_matrix), (ax_stab, fold_stability)):
            masked = np.ma.masked_invalid(matrix)
            axis.imshow(
                masked,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            axis.set_xticks(np.arange(len(method_cols)))
            labels = [m.replace("Permutation", "Perm.") for m in method_cols]
            axis.set_xticklabels(labels, fontsize=5.0, color="#000000", rotation=0)
            axis.tick_params(axis="x", length=0, pad=2, colors="#000000")
            for lab in axis.get_xticklabels():
                lab.set_clip_on(False)
            for x in np.arange(-0.5, len(method_cols) + 0.5, 1):
                axis.axvline(x, color="white", lw=0.45)
            for yline in np.arange(-0.5, n + 0.5, 1):
                axis.axhline(yline, color="white", lw=0.35)
            _annotate_support_matrix(axis, matrix)
        if ax_mean is not None:
            with np.errstate(invalid="ignore"):
                mean_support = np.nanmean(support_matrix, axis=1)
            _draw_mean_support(ax_mean, mean_support)
    else:
        _soft_missing(ax_hm, "no method\nscores")
        _soft_missing(ax_stab, "no fold\nfrequency")
        if ax_mean is not None:
            _soft_missing(ax_mean, "no mean")

    _bracket(fig, panel, bracket_lab[0], bracket_lab[1], 0.900, "Ranked feature")
    _bracket(fig, panel, bracket_dir[0], bracket_dir[1], 0.900, "Class shift")
    _bracket(fig, panel, bracket_hm[0], bracket_hm[1], 0.900, "Top-k support")
    if ax_mean is not None:
        _bracket(fig, panel, bracket_mean[0], bracket_mean[1], 0.900, "Mean")
    _bracket(
        fig, panel, bracket_stab[0], bracket_stab[1], 0.900, "Fold top-k frequency"
    )
    save_all(fig, out_stem)
    plt.close(fig)
    return True


def plot_regression_feature_support(
    importance: pd.DataFrame,
    out_stem: Path,
    top_k: int,
) -> bool:
    apply_style()
    required = {"method", "feature", "importance_mean"}
    if (
        importance is None
        or importance.empty
        or not required.issubset(importance.columns)
    ):
        return False
    d = importance.copy()
    d["importance_mean"] = pd.to_numeric(d["importance_mean"], errors="coerce")
    d = d[np.isfinite(d["importance_mean"].to_numpy(dtype=float))]
    if d.empty:
        return False
    d["method"] = d["method"].astype(str)
    d["feature"] = d["feature"].astype(str)
    methods = [str(x) for x in d["method"].drop_duplicates().tolist()]
    if not methods:
        return False
    rows = []
    for method in methods:
        sub = d[d["method"].eq(method)].copy()
        sub = sub.sort_values(["importance_mean", "feature"], ascending=[False, True])
        sub["support"] = top_k_rank_support(
            sub["importance_mean"], int(top_k)
        ).to_numpy(dtype=float)
        rows.append(sub[["method", "feature", "support"]])
    support = pd.concat(rows, ignore_index=True)
    pivot = support.pivot_table(
        index="feature", columns="method", values="support", aggfunc="first"
    )
    pivot = pivot.reindex(columns=methods)
    pivot["mean_support"] = pivot.mean(axis=1, skipna=True)
    pivot["methods_available"] = pivot[methods].notna().sum(axis=1)
    top = (
        pivot.reset_index()
        .sort_values(
            ["mean_support", "methods_available", "feature"],
            ascending=[False, False, True],
        )
        .head(int(top_k))
    )
    if top.empty:
        return False
    top = top.reset_index(drop=True)
    features = top["feature"].astype(str).tolist()
    fold_stability = _fold_stability_matrix(d, features, methods)
    n = len(top)
    solo_method = len(methods) == 1
    fig_h_mm = max(52.0, 3.0 * n + 18.0)
    fig_w = (136.0 * MM) if solo_method else COL_W_2
    fig = plt.figure(figsize=(fig_w, fig_h_mm * MM))
    fig.patch.set_facecolor(BG)
    panel = [0.035, 0.035, 0.930, 0.930]
    if solo_method:
        ax_lab = fig.add_axes(_bbox(panel, 0.004, 0.035, 0.570, 0.820), zorder=5)
        ax_hm = fig.add_axes(_bbox(panel, 0.665, 0.035, 0.125, 0.820), zorder=5)
        ax_stab = fig.add_axes(_bbox(panel, 0.845, 0.035, 0.125, 0.820), zorder=5)
        ax_mean = None
        bracket_lab = (0.020, 0.570)
        bracket_hm = (0.665, 0.790)
        bracket_stab = (0.845, 0.970)
    else:
        ax_lab = fig.add_axes(_bbox(panel, 0.004, 0.035, 0.455, 0.820), zorder=5)
        ax_hm = fig.add_axes(_bbox(panel, 0.505, 0.035, 0.215, 0.820), zorder=5)
        ax_mean = fig.add_axes(_bbox(panel, 0.745, 0.035, 0.065, 0.820), zorder=5)
        ax_stab = fig.add_axes(_bbox(panel, 0.835, 0.035, 0.145, 0.820), zorder=5)
        bracket_lab = (0.020, 0.455)
        bracket_hm = (0.505, 0.720)
        bracket_mean = (0.745, 0.810)
        bracket_stab = (0.835, 0.980)
    axes_to_style = [ax_lab, ax_hm, ax_stab]
    if ax_mean is not None:
        axes_to_style.append(ax_mean)
    for ax in axes_to_style:
        ax.set_facecolor(BG)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for yi in np.arange(n + 1) - 0.5:
            ax.axhline(yi, color=TRACK, lw=0.25, zorder=0)
    ax_lab.set_xlim(0, 1)
    ax_lab.set_xticks([])
    for i, feature in enumerate(features):
        ax_lab.text(
            0.010,
            i,
            str(i + 1),
            ha="left",
            va="center",
            fontsize=5.0,
            color=DIM,
            clip_on=False,
        )
        ax_lab.text(
            0.095,
            i,
            _full_feature_label(feature),
            ha="left",
            va="center",
            fontsize=5.0,
            color=INK,
            clip_on=False,
        )
    support_matrix = (
        top[methods].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    support_matrix = np.clip(support_matrix, 0, 1)
    cmap = SUPPORT_CMAP.copy()
    cmap.set_bad(TRACK)
    display_names = {
        "shap": "SHAP",
        "lime": "LIME",
        "permutation": "Perm.",
        "ale": "ALE",
    }
    labels = [display_names.get(m.casefold(), m) for m in methods]
    for axis, matrix in ((ax_hm, support_matrix), (ax_stab, fold_stability)):
        masked = np.ma.masked_invalid(matrix)
        axis.imshow(
            masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1
        )
        axis.set_xticks(np.arange(len(methods)))
        axis.set_xticklabels(labels, fontsize=5.0, color="#000000", rotation=0)
        axis.tick_params(axis="x", length=0, pad=2, colors="#000000")
        for lab in axis.get_xticklabels():
            lab.set_clip_on(False)
        for x in np.arange(-0.5, len(methods) + 0.5, 1):
            axis.axvline(x, color="white", lw=0.45)
        for yline in np.arange(-0.5, n + 0.5, 1):
            axis.axhline(yline, color="white", lw=0.35)
        _annotate_support_matrix(axis, matrix)
    if ax_mean is not None:
        mean_support = pd.to_numeric(top["mean_support"], errors="coerce").to_numpy(
            dtype=float
        )
        _draw_mean_support(ax_mean, mean_support)
    _bracket(fig, panel, bracket_lab[0], bracket_lab[1], 0.900, "Ranked feature")
    _bracket(fig, panel, bracket_hm[0], bracket_hm[1], 0.900, "Top-k support")
    if ax_mean is not None:
        _bracket(fig, panel, bracket_mean[0], bracket_mean[1], 0.900, "Mean")
    _bracket(
        fig, panel, bracket_stab[0], bracket_stab[1], 0.900, "Fold top-k frequency"
    )
    save_all(fig, out_stem)
    plt.close(fig)
    return True


def _interp_col_net(t: float, col0: str, col_mid: str, col1: str):
    t = float(np.clip(t, 0, 1))
    a = np.array(mcolors.to_rgba(col0))
    b = np.array(mcolors.to_rgba(col_mid))
    c = np.array(mcolors.to_rgba(col1))
    if t < 0.5:
        u = t / 0.5
        out = (1 - u) * a + u * b
    else:
        u = (t - 0.5) / 0.5
        out = (1 - u) * b + u * c
    return tuple(out)


def _net_node_color(standardized_shift: float):
    shift = float(standardized_shift) if np.isfinite(standardized_shift) else 0.0
    neut = np.array(mcolors.to_rgba(NET_NEUT))
    if shift >= 0:
        t = min(abs(shift) / 2.0, 1.0)
        target = np.array(mcolors.to_rgba(CLASS_CASE))
    else:
        t = min(abs(shift) / 2.0, 1.0)
        target = np.array(mcolors.to_rgba(CLASS_CTRL))
    return tuple((1.0 - t) * neut + t * target)


def _net_node_radius(strength_norm: float, is_hub: bool = False) -> float:
    norm = float(np.clip(strength_norm, 0.0, 1.0))
    r = 0.08 + norm * 0.12
    if is_hub:
        r = max(r * 1.6, 0.20)
    return float(r)


def _net_bezier(p1, p2, curv: float = 0.10):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    cpx, cpy = mx - curv * dy, my + curv * dx
    t = np.linspace(0, 1, 80)
    x = (1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * cpx + t**2 * p2[0]
    y = (1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * cpy + t**2 * p2[1]
    m = 0.5
    mx2 = (1 - m) ** 2 * p1[0] + 2 * (1 - m) * m * cpx + m**2 * p2[0]
    my2 = (1 - m) ** 2 * p1[1] + 2 * (1 - m) * m * cpy + m**2 * p2[1]
    return x, y, float(mx2), float(my2)


def _get_abundance(
    stat_map: dict[str, dict], feature: str, key: str, default: float = 0.5
) -> float:
    row = stat_map.get(feature, {})
    try:
        v = float(row.get(key, default))
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _build_network_graph(interactions: pd.DataFrame, stats: pd.DataFrame | None):
    if nx is None:
        raise RuntimeError("networkx is not installed")
    cols = set(interactions.columns)
    f1_col = (
        "feature1"
        if "feature1" in cols
        else ("feature_1" if "feature_1" in cols else None)
    )
    f2_col = (
        "feature2"
        if "feature2" in cols
        else ("feature_2" if "feature_2" in cols else None)
    )
    if f1_col is None or f2_col is None or "interaction_strength" not in cols:
        raise RuntimeError(
            "interaction table is missing feature columns or interaction_strength"
        )
    d = interactions.copy()
    d["interaction_strength"] = pd.to_numeric(
        d["interaction_strength"], errors="coerce"
    )
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[f1_col, f2_col, "interaction_strength"]
    )
    d = d[d["interaction_strength"].notna()]
    if d.empty:
        raise RuntimeError("no finite interaction strengths")
    stat_map = (
        stats.drop_duplicates("feature").set_index("feature").to_dict("index")
        if stats is not None and "feature" in stats.columns
        else {}
    )
    G = nx.Graph()
    for row_i, row in d.reset_index(drop=True).iterrows():
        f1, f2 = str(row[f1_col]), str(row[f2_col])
        try:
            w = float(row["interaction_strength"])
        except Exception:
            continue
        if not np.isfinite(w) or f1 == f2:
            continue
        for feat in (f1, f2):
            if feat not in G:
                G.add_node(
                    feat,
                    label=_net_short_label(feat),
                    standardized_shift=_get_abundance(
                        stat_map, feat, "standardized_mean_difference", 0.0
                    ),
                )
        if G.has_edge(f1, f2):
            if w > G[f1][f2]["weight"]:
                G[f1][f2].update(weight=w, rank=row_i + 1)
        else:
            G.add_edge(f1, f2, weight=w, rank=row_i + 1)
    if len(G.edges()) == 0:
        raise RuntimeError("no plottable interaction edges")
    weighted_degree = {node: float(value) for node, value in G.degree(weight="weight")}
    max_degree = max(weighted_degree.values()) if weighted_degree else 0.0
    for node in G.nodes():
        value = weighted_degree.get(node, 0.0)
        G.nodes[node]["strength_norm"] = (
            float(value / max_degree) if max_degree > 0 else 0.0
        )
    return G


def _net_layout_component(subG, seed=42):
    n = len(subG)
    if n == 1:
        return {list(subG.nodes())[0]: np.array([0.0, 0.0])}
    if n == 2:
        nodes = list(subG.nodes())
        return {nodes[0]: np.array([-0.4, 0.0]), nodes[1]: np.array([0.4, 0.0])}
    max_deg = max(d for _, d in subG.degree())
    if max_deg <= 2:
        endpoints = [v for v, d in subG.degree() if d == 1]
        source = endpoints[0] if endpoints else list(subG.nodes())[0]
        ordered = list(nx.dfs_preorder_nodes(subG, source=source))
        angles = np.linspace(0, 2 * np.pi, len(ordered), endpoint=False)
        return {
            node: np.array([np.cos(a), np.sin(a)]) for node, a in zip(ordered, angles)
        }
    try:
        return nx.kamada_kawai_layout(subG, weight="weight")
    except Exception:
        return nx.spring_layout(subG, weight="weight", k=1.5, seed=seed)


def _net_grid_layout(G, seed=42):
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_cols = 2
    n_rows = int(np.ceil(len(components) / n_cols))
    cell_w, cell_h = 4.5, 4.5
    pos = {}
    for i, comp in enumerate(components):
        row, col = i // n_cols, i % n_cols
        cx = (col - (n_cols - 1) / 2) * cell_w
        cy = ((n_rows - 1) / 2 - row) * cell_h
        subG = G.subgraph(comp)
        cpos = _net_layout_component(subG, seed=seed)
        if len(comp) > 1:
            keys = list(cpos.keys())
            arr = np.array([cpos[k] for k in keys], dtype=float)
            arr -= arr.mean(axis=0)
            ext = np.abs(arr).max()
            if ext > 0:
                arr = arr / ext * 1.0
            cpos = {k: arr[j] for j, k in enumerate(keys)}
        for node, pnt in cpos.items():
            pos[node] = np.array([pnt[0] + cx, pnt[1] + cy])
    return pos


def _normalise_layout(pos, target: float) -> dict:
    keys = list(pos.keys())
    arr = np.array([pos[k] for k in keys], dtype=float)
    arr -= arr.mean(axis=0)
    ext = np.abs(arr).max()
    if ext > 0:
        arr = arr / ext * target
    return {k: arr[i] for i, k in enumerate(keys)}


def _net_compute_layout(G, seed=42):
    if nx is None:
        raise RuntimeError("networkx not installed")
    degrees = dict(G.degree())
    hub = max(degrees, key=degrees.get)
    hub_deg = int(degrees[hub])
    components = list(nx.connected_components(G))
    if hub_deg >= 4:
        seed_pos = {hub: np.array([0.0, 0.0])}
        pos = nx.spring_layout(
            G,
            pos=seed_pos,
            fixed=[hub],
            weight="weight",
            k=4.5,
            seed=seed,
            iterations=400,
        )
        target = 4.0
    elif len(components) > 1:
        pos = _net_grid_layout(G, seed=seed)
        target = 4.2
    else:
        try:
            pos = nx.kamada_kawai_layout(G, weight="weight")
        except Exception:
            pos = nx.spring_layout(G, weight="weight", k=4.5, seed=seed)
        target = 4.0
    return _normalise_layout(pos, target), hub, hub_deg


def _net_compute_kamada_kawai_layout(G):
    if nx is None:
        raise RuntimeError("networkx not installed")
    degrees = dict(G.degree())
    hub = max(degrees, key=degrees.get)
    hub_deg = int(degrees[hub])
    pos = nx.kamada_kawai_layout(G, weight="weight")
    return _normalise_layout(pos, 4.0), hub, hub_deg


def _net_edge_linewidth(norm_strength: float) -> float:
    return 0.9 + float(norm_strength) * 4.2


def _draw_network(
    ax: plt.Axes,
    interactions: pd.DataFrame,
    stats: pd.DataFrame | None,
    top_k: int,
    layout: str = "default",
) -> tuple[float, float]:
    G = _build_network_graph(interactions.head(int(top_k)), stats)
    if len(G.nodes()) == 0:
        raise RuntimeError("No nodes in graph")
    if layout == "kamada_kawai":
        pos, hub, hub_deg = _net_compute_kamada_kawai_layout(G)
    else:
        pos, hub, hub_deg = _net_compute_layout(G, seed=42)
    pos = {node: np.asarray(value, dtype=float) * 0.78 for node, value in pos.items()}
    strengths = [G[u][v]["weight"] for u, v in G.edges()]
    s_min, s_max = float(min(strengths)), float(max(strengths))

    def _ns(value):
        return (float(value) - s_min) / (s_max - s_min) if s_max > s_min else 0.6

    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    edge_list = sorted(G.edges(data=True), key=lambda edge: edge[2]["weight"])
    for idx, (u, v, edge) in enumerate(edge_list):
        p1, p2 = pos[u], pos[v]
        norm = _ns(edge["weight"])
        curvature = 0.10 * (1 if idx % 2 == 0 else -1)
        xe, ye, _, _ = _net_bezier(p1, p2, curv=curvature)
        ax.plot(
            xe,
            ye,
            color=NET_EDGE_CMAP(0.12 + norm * 0.88),
            linewidth=_net_edge_linewidth(norm),
            alpha=0.80,
            zorder=1,
            solid_capstyle="round",
        )

    radii: dict[Any, float] = {}
    for node in G.nodes():
        data = G.nodes[node]
        is_hub = bool(node == hub and hub_deg >= 4)
        radius = _net_node_radius(data.get("strength_norm", 0.0), is_hub=is_hub)
        radii[node] = radius
        x, y = pos[node]
        ax.add_patch(
            plt.Circle(
                (x, y),
                radius,
                facecolor=_net_node_color(data.get("standardized_shift", 0.0)),
                edgecolor=INK,
                linewidth=0.8,
                zorder=3,
                alpha=0.96,
            )
        )

    nodes = list(G.nodes())
    left = [node for node in nodes if float(pos[node][0]) < 0.0]
    right = [node for node in nodes if float(pos[node][0]) >= 0.0]
    while abs(len(left) - len(right)) > 2:
        source = left if len(left) > len(right) else right
        target = right if source is left else left
        move = min(source, key=lambda node: abs(float(pos[node][0])))
        source.remove(move)
        target.append(move)

    def _draw_side(side_nodes: list[Any], side: str) -> None:
        if not side_nodes:
            return
        ordered = sorted(side_nodes, key=lambda node: float(pos[node][1]), reverse=True)
        count = len(ordered)
        center = float(np.median([float(pos[node][1]) for node in ordered]))
        half_span = min(3.15, 0.39 * max(count - 1, 0))
        center_limit = max(0.0, 3.40 - half_span)
        center = float(np.clip(center, -center_limit, center_limit))
        slots = np.linspace(center + half_span, center - half_span, count)
        sign = -1.0 if side == "left" else 1.0
        text_x = sign * 4.10
        line_end = sign * 4.06
        elbow_x = sign * 3.52
        horizontal_alignment = "right" if side == "left" else "left"
        for node, slot_y in zip(ordered, slots):
            x, y = map(float, pos[node])
            vector = np.array([elbow_x - x, float(slot_y) - y], dtype=float)
            length = float(np.linalg.norm(vector))
            if length <= 1e-12:
                vector = np.array([sign, 0.0], dtype=float)
                length = 1.0
            anchor = np.array([x, y], dtype=float) + vector / length * (
                radii[node] + 0.05
            )
            ax.plot(
                [anchor[0], elbow_x, line_end],
                [anchor[1], float(slot_y), float(slot_y)],
                color=INK,
                linewidth=0.42,
                alpha=0.58,
                linestyle=(0, (1.5, 2.2)),
                solid_capstyle="round",
                zorder=4,
            )
            label = _net_italic(str(G.nodes[node]["label"]))
            is_hub = bool(node == hub and hub_deg >= 4)
            ax.text(
                text_x,
                float(slot_y),
                label,
                ha=horizontal_alignment,
                va="center",
                fontsize=6.2 if not is_hub else 6.5,
                color=INK,
                fontweight="bold" if is_hub else "normal",
                zorder=6,
            )

    _draw_side(left, "left")
    _draw_side(right, "right")
    ax.set_xlim(-7.15, 7.15)
    ax.set_ylim(-5.0, 5.0)
    return float(s_min), float(s_max)


def _legend_size_colour_net(
    ax: plt.Axes, ctrl_text="Control", case_text="Case"
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lo_r, hi_r = np.log10(0.05), np.log10(60.0)
    gxs = [0.32, 0.50, 0.68, 0.86]
    gys = [0.75, 0.61, 0.47, 0.33]
    abs_vals = [0.1, 1.0, 5.0, 15.0]
    abs_lbl = ["0.1%", "1%", "5%", "15%"]
    ax_width_pt = max(
        ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0, 1e-6
    )
    pts_per_data = ax_width_pt / 10.0

    def _grid_s(ab):
        mean = max(float(ab), 0.05)
        norm = np.clip((np.log10(mean) - lo_r) / (hi_r - lo_r), 0, 1)
        r_data = 0.08 + norm * 0.12
        r_pt = r_data * pts_per_data
        return np.pi * (r_pt**2)

    for cx, al in zip(gxs, abs_lbl):
        ax.text(cx, 0.82, al, ha="center", va="center", fontsize=6.4, color="black")
    for ry, t in zip(gys, [0.0, 1 / 3, 2 / 3, 1.0]):
        col = _interp_col_net(t, CLASS_CTRL, NET_NEUT, CLASS_CASE)
        for cx, ab in zip(gxs, abs_vals):
            ax.scatter(
                [cx],
                [ry],
                s=_grid_s(ab),
                facecolor=col,
                edgecolors=INK,
                linewidths=0.28,
                zorder=3,
                clip_on=False,
            )

    sx0, sx1 = 0.08, 0.13
    sxc = (sx0 + sx1) / 2
    sy0, sy1 = gys[-1], gys[0]
    n_s = 80
    sy = np.linspace(sy0, sy1, n_s + 1)
    st = np.linspace(1.0, 0.0, n_s + 1)
    for i in range(n_s):
        t = (st[i] + st[i + 1]) / 2
        c = _interp_col_net(t, CLASS_CTRL, NET_NEUT, CLASS_CASE)
        ax.fill(
            [sx0, sx1, sx1, sx0],
            [sy[i], sy[i], sy[i + 1], sy[i + 1]],
            color=c,
            linewidth=0,
            zorder=2,
        )
    ext = 0.05
    ax.annotate(
        "",
        xy=(sxc, sy1 + ext),
        xytext=(sxc, sy1),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5, mutation_scale=4.5),
    )
    ax.annotate(
        "",
        xy=(sxc, sy0 - ext),
        xytext=(sxc, sy0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5, mutation_scale=4.5),
    )
    ax.text(
        sxc,
        sy1 + ext + 0.01,
        f"Relatively more abundant in {ctrl_text}",
        ha="center",
        va="bottom",
        fontsize=5.9,
        color="black",
        rotation=90,
        clip_on=False,
    )
    ax.text(
        sxc,
        sy0 - ext - 0.01,
        f"Relatively more abundant in {case_text}",
        ha="center",
        va="top",
        fontsize=5.9,
        color="black",
        rotation=90,
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(gxs[-1] + 0.06, 0.20),
        xytext=(gxs[0] - 0.06, 0.20),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5, mutation_scale=4.5),
    )
    ax.text(
        (gxs[0] + gxs[-1]) / 2,
        0.11,
        "Mean relative abundance",
        ha="center",
        va="center",
        fontsize=6.4,
        color="black",
    )


def _legend_edge_net(ax: plt.Axes, s_min: float, s_max: float) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fy0, fy1 = 0.18, 0.82
    n_f = 180
    fy = np.linspace(fy0, fy1, n_f + 1)
    ts = np.linspace(0, 1, n_f + 1)
    ax_width_pt = max(
        ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0, 1e-6
    )

    def _half_width_axes(t):
        return 0.5 * _net_edge_linewidth(float(t)) / ax_width_pt

    xc = 0.42
    hw = np.array([_half_width_axes(t) for t in ts])
    for i in range(n_f):
        t = (ts[i] + ts[i + 1]) / 2
        col = NET_EDGE_CMAP(0.12 + t * 0.88)
        xs = [xc - hw[i], xc + hw[i], xc + hw[i + 1], xc - hw[i + 1]]
        ys = [fy[i], fy[i], fy[i + 1], fy[i + 1]]
        ax.fill(xs, ys, color=col, linewidth=0, zorder=2)
    ax.text(
        xc,
        fy1 + 0.040,
        f"{s_max:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="black",
    )
    ax.text(
        xc,
        fy0 - 0.040,
        f"{s_min:.3f}",
        ha="center",
        va="top",
        fontsize=6.5,
        color="black",
    )
    ax.text(
        0.66,
        (fy0 + fy1) / 2,
        "Interaction strength",
        ha="center",
        va="center",
        fontsize=6.6,
        color="black",
        rotation=90,
        clip_on=False,
    )


def plot_interaction_network(
    tab: pd.DataFrame,
    stats: pd.DataFrame,
    out_stem: Path,
    top_k: int,
    class_labels: Sequence[str] | None = None,
    layout: str = "default",
) -> bool:
    apply_style()
    labels = [str(x) for x in (class_labels or ())]
    regression_mode = len(labels) < 2
    fig = plt.figure(figsize=(COL_W_2, 125 * MM))
    fig.patch.set_facecolor(BG)
    if regression_mode:
        ax_net = fig.add_axes([0.02, 0.06, 0.79, 0.88], zorder=4)
        ax_leg1 = fig.add_axes([0.82, 0.53, 0.15, 0.37], zorder=12)
        ax_leg2 = fig.add_axes([0.82, 0.24, 0.14, 0.34], zorder=12)
    else:
        ax_net = fig.add_axes([0.02, 0.06, 0.69, 0.88], zorder=4)
        ax_leg1 = fig.add_axes([0.73, 0.53, 0.25, 0.37], zorder=12)
        ax_leg2 = fig.add_axes([0.75, 0.17, 0.20, 0.28], zorder=12)
    try:
        s_min, s_max = _draw_network(ax_net, tab, stats, top_k, layout=layout)
        if regression_mode:
            ax_leg1.axis("off")
        else:
            _legend_size_colour_net(ax_leg1, ctrl_text=labels[0], case_text=labels[1])
        _legend_edge_net(ax_leg2, s_min, s_max)
    except Exception as exc:
        ax_net.clear()
        _soft_missing(ax_net, f"Interaction network unavailable\n{exc}")
        ax_leg1.axis("off")
        ax_leg2.axis("off")
    save_all(fig, out_stem)
    plt.close(fig)
    return True


def _context_palette(
    context: pd.DataFrame, task: str
) -> tuple[list[tuple[Any, str, str]], dict[Any, str]]:
    if str(task).lower() != "classification" or context.empty:
        return [("cohort", "cohort", CLASS_CTRL)], {"cohort": CLASS_CTRL}
    data = context.copy()
    data["class_index"] = pd.to_numeric(data.get("class_index"), errors="coerce")
    data = data[np.isfinite(data["class_index"].to_numpy(dtype=float))]
    if data.empty:
        return [("cohort", "cohort", CLASS_CTRL)], {"cohort": CLASS_CTRL}
    colors = [
        DIM,
        ACC,
        "#8b5cf6",
        "#0f766e",
        "#c2410c",
        "#a16207",
        "#be185d",
        "#334155",
    ]
    groups: list[tuple[Any, str, str]] = []
    mapping: dict[Any, str] = {}
    for pos, class_index in enumerate(
        sorted(data["class_index"].astype(int).unique().tolist())
    ):
        rows = data[data["class_index"].astype(int).eq(int(class_index))]
        labels = [
            str(x)
            for x in rows.get("class_label", pd.Series(dtype=str)).dropna().tolist()
            if str(x)
        ]
        label = labels[0] if labels else str(class_index)
        color = colors[pos % len(colors)]
        groups.append((int(class_index), label, color))
        mapping[int(class_index)] = color
    return groups, mapping


def _class_display_label(value: Any) -> str:
    text = str(value).strip().replace("_", " ")
    return text[:1].upper() + text[1:] if text else text


def _local_direction_colors(
    context: pd.DataFrame,
    task: str,
    class_label: str,
    class_index: int | None = None,
) -> tuple[str, str]:
    if str(task).lower() != "classification":
        return ACC, CLASS_CTRL
    groups, _ = _context_palette(context, task)
    if not groups:
        if class_index == 0:
            return CLASS_CTRL, CLASS_CASE
        if class_index == 1:
            return CLASS_CASE, CLASS_CTRL
        return ACC, DIM
    matched_key: Any = None
    positive = ACC
    if class_index is not None:
        for key, _, color in groups:
            if key == int(class_index):
                matched_key = key
                positive = color
                break
    if matched_key is None:
        target = str(class_label).strip().casefold()
        for key, label, color in groups:
            if str(label).strip().casefold() == target:
                matched_key = key
                positive = color
                break
    if len(groups) == 2 and matched_key is not None:
        for key, _, color in groups:
            if key != matched_key:
                return positive, color
    return positive, DIM


def _draw_cohort_context(
    ax: plt.Axes,
    context: pd.DataFrame,
    features: Sequence[str],
    sample_id: str,
    task: str,
) -> tuple[list[tuple[Any, str, str]], dict[Any, str]]:
    groups, colors = _context_palette(context, task)
    if context.empty:
        ax.set_xticks([])
        return groups, colors
    data = context.copy()
    data["feature"] = data.get("feature", "").astype(str)
    data["sample_id"] = data.get("sample_id", "").astype(str)
    data["relative_abundance"] = pd.to_numeric(
        data.get("relative_abundance"), errors="coerce"
    )
    data = data[np.isfinite(data["relative_abundance"].to_numpy(dtype=float))]
    data["relative_abundance"] = data["relative_abundance"].clip(lower=0.0)
    n_groups = max(1, len(groups))
    if n_groups == 1:
        offsets = {groups[0][0]: 0.0}
    else:
        values = np.linspace(0.17, -0.17, n_groups)
        offsets = {key: float(values[i]) for i, (key, _, _) in enumerate(groups)}
    for row_no, feature in enumerate(features):
        rows = data[data["feature"].eq(str(feature))].copy()
        if rows.empty:
            continue
        if str(task).lower() == "classification" and "class_index" in rows.columns:
            rows["class_index"] = pd.to_numeric(rows["class_index"], errors="coerce")
            for class_key, _, color in groups:
                group = rows[rows["class_index"].eq(float(class_key))].sort_values(
                    "sample_id", kind="stable"
                )
                if group.empty:
                    continue
                values = group["relative_abundance"].to_numpy(dtype=float)
                jitter = 0.015 * np.sin(
                    np.arange(len(group), dtype=float) * 2.399963229728653
                )
                y = (
                    np.full(len(group), row_no + offsets[class_key], dtype=float)
                    + jitter
                )
                ax.scatter(
                    values,
                    y,
                    s=7.5,
                    color=color,
                    alpha=0.58 if class_key != groups[0][0] else 0.78,
                    edgecolors="none",
                    zorder=2,
                )
        else:
            group = rows.sort_values("sample_id", kind="stable")
            values = group["relative_abundance"].to_numpy(dtype=float)
            jitter = 0.025 * np.sin(
                np.arange(len(group), dtype=float) * 2.399963229728653
            )
            ax.scatter(
                values,
                row_no + jitter,
                s=7.5,
                color=CLASS_CTRL,
                alpha=0.68,
                edgecolors="none",
                zorder=2,
            )
        focal = rows[rows["sample_id"].eq(str(sample_id))]
        if not focal.empty:
            point = focal.iloc[0]
            x = float(point["relative_abundance"])
            if str(task).lower() == "classification":
                key_value = pd.to_numeric(
                    pd.Series([point.get("class_index", np.nan)]), errors="coerce"
                ).iloc[0]
                if np.isfinite(key_value):
                    key = int(key_value)
                    y = row_no + offsets.get(key, 0.0)
                    face = colors.get(key, ACC)
                else:
                    y = float(row_no)
                    face = ACC
            else:
                y = float(row_no)
                face = CLASS_CTRL
            ax.scatter(
                [x],
                [y],
                s=15.0,
                facecolor=face,
                edgecolor=INK,
                linewidth=0.55,
                zorder=5,
            )
    maximum = float(data["relative_abundance"].max()) if not data.empty else 0.1
    maximum = max(maximum, 1e-4)
    ax.set_xscale("symlog", linthresh=1e-5, linscale=0.70, base=10)
    if maximum > 0.20:
        ax.set_xlim(-5e-5, 2.50)
        ax.set_xticks([0.0, 1e-4, 1e-2, 1.0])
        ax.set_xticklabels(["0%", "0.01%", "1%", "100%"])
    else:
        ax.set_xlim(-5e-5, max(0.25, maximum * 1.50))
        ax.set_xticks([0.0, 1e-4, 1e-2, 1e-1])
        ax.set_xticklabels(["0%", "0.01%", "1%", "10%"])
    ax.set_xlabel("Relative abundance", fontsize=5.0, color=INK, labelpad=2)
    ax.tick_params(axis="x", labelsize=4.4, length=1.8, width=0.35, pad=1)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#000000")
    ax.spines["bottom"].set_linewidth(0.4)
    return groups, colors


def _context_legend(
    fig: plt.Figure,
    context: pd.DataFrame,
    sample_id: str,
    task: str,
    anchor: tuple[float, float],
) -> None:
    groups, colors = _context_palette(context, task)
    handles: list[mlines.Line2D] = []
    if str(task).lower() == "classification":
        for key, label, color in groups:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markersize=3.0,
                    markerfacecolor=color,
                    markeredgecolor="none",
                    label=label,
                )
            )
        focal_color = ACC
        focal = (
            context[
                context.get("sample_id", pd.Series(dtype=str))
                .astype(str)
                .eq(str(sample_id))
            ]
            if not context.empty
            else pd.DataFrame()
        )
        if not focal.empty:
            value = pd.to_numeric(focal.get("class_index"), errors="coerce").dropna()
            if not value.empty:
                focal_color = colors.get(int(value.iloc[0]), ACC)
    else:
        handles.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=3.0,
                markerfacecolor=CLASS_CTRL,
                markeredgecolor="none",
                label="cohort",
            )
        )
        focal_color = CLASS_CTRL
    handles.append(
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor=focal_color,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            label="explained sample",
        )
    )
    legend = fig.legend(
        handles=handles,
        ncol=len(handles),
        loc="center",
        bbox_to_anchor=anchor,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.38,
        columnspacing=0.8,
        handlelength=0.8,
        handletextpad=0.3,
        fontsize=4.4,
    )
    legend.get_frame().set_edgecolor("#B9B9B9")
    legend.get_frame().set_linewidth(0.65)
    legend.get_frame().set_facecolor("white")


def plot_local_attributions(
    table: pd.DataFrame,
    out_stem: Path,
    *,
    task: str,
    top_n: int,
    cohort_context: pd.DataFrame | None = None,
) -> Path | None:
    if table is None or table.empty:
        return None
    context = cohort_context.copy() if cohort_context is not None else pd.DataFrame()
    selected = table.copy()
    if "selected_for_report" in selected.columns:
        marker = selected["selected_for_report"]
        if marker.dtype == bool:
            selected = selected[marker]
        else:
            selected = selected[
                marker.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
            ]
    if selected.empty or "method" not in selected.columns:
        return None
    present = {str(x).strip().lower() for x in selected["method"].dropna().tolist()}
    methods = [name for name in ("shap", "lime") if name in present]
    if not methods:
        return None
    apply_style()
    sample_ids = list(dict.fromkeys(selected["sample_id"].astype(str).tolist()))
    if not sample_ids:
        return None
    k = max(1, int(top_n))
    n_panels = len(sample_ids)
    fig_h = max(64.0, 50.0 * n_panels + 10.0)
    fig = plt.figure(figsize=(COL_W_2, fig_h * MM))
    fig.patch.set_facecolor(BG)
    panel_h = 0.92 / n_panels
    for panel_no, sample_id in enumerate(sample_ids):
        sub = selected[selected["sample_id"].astype(str).eq(sample_id)].copy()
        aggregates: dict[str, pd.DataFrame] = {}
        for method in methods:
            local = sub[sub["method"].astype(str).str.lower().eq(method)].copy()
            if local.empty:
                continue
            agg = local.groupby("feature", as_index=False, dropna=False).agg(
                attribution=("attribution", "mean"),
                attribution_sd=("attribution", "std"),
                feature_value=("feature_value", "mean"),
            )
            agg["attribution"] = pd.to_numeric(
                agg["attribution"], errors="coerce"
            ).fillna(0.0)
            agg["abs_attribution"] = agg["attribution"].abs()
            support_rows: list[dict[str, float | str]] = []
            split_values = (
                local["split_key"].astype(str)
                if "split_key" in local.columns
                else pd.Series(["oof"] * len(local), index=local.index)
            )
            for split_key in dict.fromkeys(split_values.tolist()):
                split = local[split_values.eq(split_key)].copy()
                split = split.groupby("feature", as_index=False, dropna=False).agg(
                    attribution=("attribution", "mean")
                )
                split["attribution"] = pd.to_numeric(
                    split["attribution"], errors="coerce"
                ).fillna(0.0)
                split["abs_attribution"] = split["attribution"].abs()
                split = split.sort_values(
                    ["abs_attribution", "feature"],
                    ascending=[False, True],
                    kind="stable",
                ).reset_index(drop=True)
                split["rank"] = np.arange(1, len(split) + 1, dtype=int)
                for _, item in split.iterrows():
                    rank = int(item["rank"])
                    value = float(item["attribution"])
                    score = (
                        float(np.sign(value) / rank)
                        if rank <= k and value != 0.0
                        else 0.0
                    )
                    support_rows.append(
                        {
                            "feature": str(item["feature"]),
                            "signed_rank_support": score,
                            "rank_support": abs(score),
                        }
                    )
            support = pd.DataFrame(support_rows)
            if not support.empty:
                support = support.groupby("feature", as_index=False).agg(
                    signed_rank_support=("signed_rank_support", "mean"),
                    rank_support=("rank_support", "mean"),
                )
                agg = agg.merge(support, on="feature", how="left")
            else:
                agg["signed_rank_support"] = 0.0
                agg["rank_support"] = 0.0
            agg["signed_rank_support"] = pd.to_numeric(
                agg["signed_rank_support"], errors="coerce"
            ).fillna(0.0)
            agg["rank_support"] = pd.to_numeric(
                agg["rank_support"], errors="coerce"
            ).fillna(0.0)
            agg = agg.sort_values(
                ["rank_support", "abs_attribution", "feature"],
                ascending=[False, False, True],
                kind="stable",
            ).reset_index(drop=True)
            aggregates[method] = agg
        if not aggregates:
            continue
        features: set[str] = set()
        for agg in aggregates.values():
            features.update(agg.head(k)["feature"].astype(str).tolist())
        rows: list[dict[str, float | str | int]] = []
        for feature in features:
            row: dict[str, float | str | int] = {"feature": feature}
            signed_supports: list[float] = []
            abs_supports: list[float] = []
            support_count = 0
            signs: list[int] = []
            for method in methods:
                agg = aggregates.get(method)
                value = 0.0
                score = 0.0
                abs_score = 0.0
                if agg is not None:
                    match = agg[agg["feature"].astype(str).eq(feature)]
                    if not match.empty:
                        value = float(match.iloc[0]["attribution"])
                        score = float(match.iloc[0]["signed_rank_support"])
                        abs_score = float(match.iloc[0]["rank_support"])
                row[f"{method}_value"] = value
                if abs_score > 0.0:
                    support_count += 1
                if score != 0.0:
                    signs.append(1 if score > 0 else -1)
                signed_supports.append(score)
                abs_supports.append(abs_score)
            row["consensus"] = (
                float(np.mean(signed_supports)) if signed_supports else 0.0
            )
            row["selection_score"] = (
                float(np.mean(abs_supports)) if abs_supports else 0.0
            )
            row["support_count"] = int(support_count)
            row["direction_agreement"] = int(len(set(signs)) <= 1 and len(signs) > 1)
            rows.append(row)
        merged = pd.DataFrame(rows)
        if merged.empty:
            continue
        merged = (
            merged.sort_values(
                ["selection_score", "feature"], ascending=[False, True], kind="stable"
            )
            .head(k)
            .reset_index(drop=True)
        )
        y0 = 0.04 + (n_panels - 1 - panel_no) * panel_h
        role = str(sub.get("selection_role", pd.Series([""])).iloc[0]).strip()
        split_key = (
            sub["split_key"].astype(str)
            if "split_key" in sub.columns
            else pd.Series(["out_of_fold"] * len(sub), index=sub.index)
        )
        n_explanations = max(1, int(split_key.nunique()))
        if str(task).lower() == "classification":
            class_label = str(sub.get("class_label", pd.Series([""])).iloc[0])
            true_label = str(
                sub.get("true_class_label", pd.Series([class_label])).iloc[0]
            )
            predicted_values = sub.get(
                "predicted_class_label", pd.Series([""] * len(sub), index=sub.index)
            ).astype(str)
            predicted_nonempty = predicted_values[predicted_values.str.len().gt(0)]
            predicted_label = (
                str(predicted_nonempty.mode().iloc[0])
                if not predicted_nonempty.empty
                else ""
            )
            prediction_values = pd.to_numeric(
                sub.get("prediction", np.nan), errors="coerce"
            )
            pred_frame = pd.DataFrame(
                {"split_key": split_key, "prediction": prediction_values}
            ).dropna(subset=["prediction"])
            pred_by_split = (
                pred_frame.groupby("split_key", sort=False)["prediction"].mean()
                if not pred_frame.empty
                else pd.Series(dtype=float)
            )
            prediction_mean = (
                float(pred_by_split.mean()) if not pred_by_split.empty else float("nan")
            )
            prediction_sd = (
                float(pred_by_split.std(ddof=1)) if len(pred_by_split) > 1 else 0.0
            )
            correct_frame = pd.DataFrame(
                {
                    "split_key": split_key,
                    "predicted": predicted_values,
                }
            ).drop_duplicates("split_key", keep="first")
            correct_rate = (
                float(correct_frame["predicted"].eq(true_label).mean()) * 100.0
                if not correct_frame.empty
                else float("nan")
            )
            class_display = _class_display_label(class_label)
            true_display = _class_display_label(true_label)
            predicted_display = _class_display_label(predicted_label)
            heading = (
                f"Observed: {true_display}   Predicted: {predicted_display}   "
                f"Mean P({class_display}) = {prediction_mean:.3f} ± {prediction_sd:.3f}"
            )
            detail = (
                f"Out-of-fold explanations: n = {n_explanations}   "
                f"Correct out-of-fold predictions: {correct_rate:.0f}%"
            )
            shap_label = f"SHAP contribution to P({class_display})"
        else:
            prediction_values = pd.to_numeric(
                sub.get("prediction", np.nan), errors="coerce"
            )
            pred_frame = pd.DataFrame(
                {"split_key": split_key, "prediction": prediction_values}
            ).dropna(subset=["prediction"])
            pred_by_split = (
                pred_frame.groupby("split_key", sort=False)["prediction"].mean()
                if not pred_frame.empty
                else pd.Series(dtype=float)
            )
            prediction_mean = (
                float(pred_by_split.mean()) if not pred_by_split.empty else float("nan")
            )
            prediction_sd = (
                float(pred_by_split.std(ddof=1)) if len(pred_by_split) > 1 else 0.0
            )
            observed = pd.to_numeric(
                sub.get("observed_response", np.nan), errors="coerce"
            ).mean()
            heading = (
                f"Observed: {observed:.3f}   Mean prediction = "
                f"{prediction_mean:.3f} ± {prediction_sd:.3f}"
            )
            detail = f"Out-of-fold explanations: n = {n_explanations}"
            shap_label = "SHAP contribution to predicted response"
        fig.text(
            0.035,
            y0 + panel_h * 0.91,
            heading,
            ha="left",
            va="center",
            fontsize=6.8,
            color=INK,
            weight="bold",
        )
        fig.text(
            0.035,
            y0 + panel_h * 0.82,
            detail,
            ha="left",
            va="center",
            fontsize=5.4,
            color=MID,
            weight="normal",
        )
        has_context = not context.empty
        if len(methods) == 2:
            if has_context:
                axes = {
                    "labels": fig.add_axes(
                        [0.035, y0 + 0.10 * panel_h, 0.270, panel_h * 0.61]
                    ),
                    "context": fig.add_axes(
                        [0.320, y0 + 0.10 * panel_h, 0.115, panel_h * 0.61]
                    ),
                    "shap": fig.add_axes(
                        [0.455, y0 + 0.10 * panel_h, 0.150, panel_h * 0.61]
                    ),
                    "lime": fig.add_axes(
                        [0.625, y0 + 0.10 * panel_h, 0.150, panel_h * 0.61]
                    ),
                    "consensus": fig.add_axes(
                        [0.795, y0 + 0.10 * panel_h, 0.170, panel_h * 0.61]
                    ),
                }
            else:
                axes = {
                    "labels": fig.add_axes(
                        [0.035, y0 + 0.10 * panel_h, 0.305, panel_h * 0.61]
                    ),
                    "shap": fig.add_axes(
                        [0.355, y0 + 0.10 * panel_h, 0.180, panel_h * 0.61]
                    ),
                    "lime": fig.add_axes(
                        [0.555, y0 + 0.10 * panel_h, 0.180, panel_h * 0.61]
                    ),
                    "consensus": fig.add_axes(
                        [0.755, y0 + 0.10 * panel_h, 0.210, panel_h * 0.61]
                    ),
                }
        else:
            method = methods[0]
            if has_context:
                axes = {
                    "labels": fig.add_axes(
                        [0.055, y0 + 0.10 * panel_h, 0.300, panel_h * 0.61]
                    ),
                    "context": fig.add_axes(
                        [0.375, y0 + 0.10 * panel_h, 0.145, panel_h * 0.61]
                    ),
                    method: fig.add_axes(
                        [0.55, y0 + 0.10 * panel_h, 0.39, panel_h * 0.61]
                    ),
                }
            else:
                axes = {
                    "labels": fig.add_axes(
                        [0.055, y0 + 0.10 * panel_h, 0.39, panel_h * 0.61]
                    ),
                    method: fig.add_axes(
                        [0.47, y0 + 0.10 * panel_h, 0.47, panel_h * 0.61]
                    ),
                }
        n_rows = len(merged)
        y = np.arange(n_rows)
        for ax in axes.values():
            ax.set_ylim(n_rows - 0.5, -0.5)
            ax.set_yticks([])
            ax.set_facecolor(BG)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for yi in np.arange(n_rows + 1) - 0.5:
                ax.axhline(yi, color=TRACK, lw=0.25, zorder=0)
        axes["labels"].set_xlim(0, 1)
        axes["labels"].set_xticks([])
        display_features = merged["feature"].astype(str).tolist()
        for i, feature in enumerate(display_features):
            axes["labels"].text(
                0.01,
                i,
                _net_italic(_plain_taxon_label(feature, 46)),
                ha="left",
                va="center",
                fontsize=5.0,
                color=INK,
                path_effects=[mpe.withStroke(linewidth=1.0, foreground="white")],
            )
        if "context" in axes:
            panel_context = context[
                context["feature"].astype(str).isin(display_features)
            ].copy()
            _draw_cohort_context(
                axes["context"],
                panel_context,
                display_features,
                sample_id,
                task,
            )
            if panel_no == 0 and not panel_context.empty:
                position = axes["context"].get_position()
                _context_legend(
                    fig,
                    panel_context,
                    sample_id,
                    task,
                    (0.805, y0 + panel_h * 0.91),
                )
        explained_class_index = None
        if str(task).lower() == "classification" and "class_index" in sub.columns:
            class_values = pd.to_numeric(sub["class_index"], errors="coerce").dropna()
            if not class_values.empty:
                explained_class_index = int(class_values.iloc[0])
        positive_color, negative_color = _local_direction_colors(
            context,
            task,
            class_label if str(task).lower() == "classification" else "",
            explained_class_index,
        )
        for method in methods:
            ax = axes[method]
            values = (
                pd.to_numeric(
                    merged.get(f"{method}_value", pd.Series(np.zeros(n_rows))),
                    errors="coerce",
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            vmax = max(float(np.nanmax(np.abs(values))) if len(values) else 1.0, 1e-12)
            colors = [
                positive_color if value >= 0 else negative_color for value in values
            ]
            ax.barh(y, values, height=0.54, color=colors, edgecolor="none", zorder=2)
            ax.axvline(0.0, color="#000000", lw=0.5, zorder=3)
            ax.set_xlim(-2.05 * vmax, 2.05 * vmax)
            if method == "shap":
                xlabel = shap_label
                title = "SHAP"
            else:
                xlabel = "LIME local surrogate coefficient"
                title = "LIME"
            ax.set_xlabel(xlabel, fontsize=5.0, color=INK, labelpad=2)
            ax.set_title(title, fontsize=5.7, color=MID, pad=3, weight="bold")
            ax.tick_params(axis="x", labelsize=5.0, length=1.8, width=0.35, pad=1)
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color("#000000")
            ax.spines["bottom"].set_linewidth(0.4)
            for yi, value in enumerate(values):
                x = value + (0.060 * vmax if value >= 0 else -0.060 * vmax)
                ha = "left" if value >= 0 else "right"
                ax.text(
                    x,
                    yi,
                    f"{value:+.3f}",
                    ha=ha,
                    va="center",
                    fontsize=5.0,
                    color=INK,
                    clip_on=True,
                )
        if len(methods) == 2:
            ax = axes["consensus"]
            consensus = (
                pd.to_numeric(merged["consensus"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            colors = [
                positive_color if value >= 0 else negative_color for value in consensus
            ]
            ax.barh(y, consensus, height=0.54, color=colors, edgecolor="none", zorder=2)
            ax.axvline(0.0, color="#000000", lw=0.5, zorder=3)
            ax.set_xlim(-2.80, 2.80)
            ax.set_xticks([-1.0, 0.0, 1.0])
            ax.set_xlabel(
                "Signed within-method rank support", fontsize=5.0, color=INK, labelpad=2
            )
            ax.set_title("Cross-method", fontsize=5.7, color=MID, pad=3, weight="bold")
            ax.tick_params(axis="x", labelsize=5.0, length=1.8, width=0.35, pad=1)
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color("#000000")
            ax.spines["bottom"].set_linewidth(0.4)
            for yi, row in merged.iterrows():
                value = float(row["consensus"])
                support_count = int(row["support_count"])
                x = value + (0.065 if value >= 0 else -0.065)
                ha = "left" if value >= 0 else "right"
                label = f"{value:+.3f} · {support_count}/2"
                ax.text(
                    x,
                    yi,
                    label,
                    ha=ha,
                    va="center",
                    fontsize=5.0,
                    color=INK,
                    clip_on=True,
                )
    save_all(fig, out_stem)
    plt.close(fig)
    return out_stem.with_suffix(".svg")
