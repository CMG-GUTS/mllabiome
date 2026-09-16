from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:
    import networkx as nx
except Exception:
    nx = None


MM = 1.0 / 25.4
COL_W_2 = 180 * MM
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
    pass
    try:
        v = float(value)
    except Exception:
        return INK
    rgba = SUPPORT_CMAP(float(np.clip(v, 0.0, 1.0)))

    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "#ffffff" if lum < 0.48 else INK


RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Inter",
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "DejaVu Sans",
    ],
    "font.size": 7.8,
    "axes.linewidth": 0.45,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlesize": 8.2,
    "axes.labelsize": 7.6,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_style() -> None:
    mpl.rcParams.update(RC)


def save_all(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)


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


def _plain_taxon_label(feature_name: str, max_len: int = 32) -> str:
    s = str(feature_name)
    last = s.split("___")[-1]
    rank = ""
    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if last.startswith(pfx):
            rank = pfx[0] + ". "
            last = last[len(pfx) :]
            break
    last = last.replace("_", " ").strip() or s.replace("_", " ")
    out = f"{rank}{last}"
    if len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


def _net_short_label(feature_name: str) -> str:
    last = str(feature_name).split("___")[-1]
    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if last.startswith(pfx):
            name = last[len(pfx) :].replace("_", " ").strip()
            return f"{pfx[0]}. {name}"
    return last.replace("_", " ").strip()


def _net_italic(label: str) -> str:
    parts = str(label).split(". ", 1)
    if len(parts) < 2:
        return label
    prefix, name = parts
    return rf"$\mathit{{{prefix}.}}$ {name.replace('$', '')}"


def _feature_label(feature_name: str) -> str:
    try:
        return _net_italic(_net_short_label(feature_name))
    except Exception:
        return _plain_taxon_label(str(feature_name), 42)


def _mean_support_column(top: pd.DataFrame) -> np.ndarray:
    if "consensus" in top.columns:
        return (
            pd.to_numeric(top["consensus"], errors="coerce").fillna(0).to_numpy(float)
        )
    method_cols = [
        c for c in ("SHAP", "LIME", "Permutation", "ALE") if c in top.columns
    ]
    if not method_cols:
        return np.zeros(len(top), dtype=float)
    return (
        top[method_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .mean(axis=1)
        .to_numpy(float)
    )


def _deduplicate_top_features(top: pd.DataFrame, max_features: int) -> pd.DataFrame:
    if "rank" in top.columns:
        top = top.sort_values("rank", ascending=True)
    top = top.drop_duplicates("feature", keep="first").head(max_features).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    return top


def plot_feature_support(
    top_features: pd.DataFrame,
    stats: pd.DataFrame,
    out_stem: Path,
    top_k: int,
    class_labels: Sequence[str] | None = None,
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
    mean_support = np.clip(_mean_support_column(top), 0, 1)

    fig_h_mm = max(62.0, 3.85 * n + 27.0)

    fig_w = (108.0 * MM) if solo_method else COL_W_2
    fig = plt.figure(figsize=(fig_w, fig_h_mm * MM))
    fig.patch.set_facecolor(BG)
    panel = [0.045, 0.075, 0.910, 0.840]

    if solo_method:
        ax_lab = fig.add_axes(_bbox(panel, 0.004, 0.050, 0.440, 0.670), zorder=5)
        ax_dir = fig.add_axes(_bbox(panel, 0.462, 0.050, 0.205, 0.670), zorder=5)
        ax_hm = fig.add_axes(_bbox(panel, 0.750, 0.050, 0.090, 0.670), zorder=5)
        ax_bar = None
        bracket_lab = (0.020, 0.444)
        bracket_dir = (0.462, 0.667)
        bracket_hm = (0.750, 0.840)
    else:
        ax_lab = fig.add_axes(_bbox(panel, 0.004, 0.050, 0.385, 0.670), zorder=5)
        ax_dir = fig.add_axes(_bbox(panel, 0.420, 0.050, 0.148, 0.670), zorder=5)
        ax_hm = fig.add_axes(_bbox(panel, 0.635, 0.050, 0.195, 0.670), zorder=5)
        ax_bar = fig.add_axes(_bbox(panel, 0.872, 0.050, 0.112, 0.670), zorder=5)
        bracket_lab = (0.020, 0.385)
        bracket_dir = (0.420, 0.568)
        bracket_hm = (0.635, 0.830)

    y = np.arange(n)
    axes_to_style = (
        (ax_lab, ax_dir, ax_hm) if ax_bar is None else (ax_lab, ax_dir, ax_hm, ax_bar)
    )
    for ax in axes_to_style:
        ax.set_facecolor(BG)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for yi in np.arange(n + 1) - 0.5:
            ax.axhline(yi, color=TRACK, lw=0.22, zorder=0)

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
            fontsize=4.65,
            color=DIM,
            clip_on=False,
        )
        ax_lab.text(
            0.120,
            i,
            _feature_label(feat),
            ha="left",
            va="center",
            fontsize=4.95,
            color=INK,
            path_effects=[mpe.withStroke(linewidth=1.65, foreground="white")],
            clip_on=False,
        )

    lfc = []
    for feat in features:
        row = stat_map.get(feat, {})
        try:
            v = float(row.get("log2_case_vs_control_mean", np.nan))
        except Exception:
            v = np.nan
        lfc.append(v)
    lfc_clip = np.clip(np.nan_to_num(np.asarray(lfc, dtype=float), nan=0.0), -2.0, 2.0)
    ax_dir.set_xlim(-2.75, 2.75)
    ax_dir.axvline(0, color="#000000", lw=0.38, zorder=1, alpha=0.75)
    for i, v in enumerate(lfc_clip):
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
    ax_dir.set_xticklabels(["Control", "0", "Case"], fontsize=3.35, color="#000000")
    for lab in ax_dir.get_xticklabels():
        lab.set_clip_on(False)
    _style_black_bottom_axis(ax_dir, show_left=False)

    if method_cols:
        M = (
            top[method_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(float)
        )
        M = np.clip(M, 0, 1)
        ax_hm.imshow(
            M, aspect="auto", interpolation="nearest", cmap=SUPPORT_CMAP, vmin=0, vmax=1
        )
        ax_hm.set_xticks(np.arange(len(method_cols)))
        labels = [m.replace("Permutation", "Perm.") for m in method_cols]
        ax_hm.set_xticklabels(labels, fontsize=4.05, color="#000000", rotation=0)
        ax_hm.tick_params(axis="x", length=0, pad=2, colors="#000000")
        for lab in ax_hm.get_xticklabels():
            lab.set_clip_on(False)
        for x in np.arange(-0.5, len(method_cols) + 0.5, 1):
            ax_hm.axvline(x, color="white", lw=0.45)
        for yline in np.arange(-0.5, n + 0.5, 1):
            ax_hm.axhline(yline, color="white", lw=0.35)
        if solo_method:
            for yi in range(n):
                val = float(M[yi, 0]) if M.shape[1] else 0.0
                ax_hm.text(
                    0,
                    yi,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.65,
                    color=_support_text_color(val),
                    zorder=5,
                )
    else:
        _soft_missing(ax_hm, "no method\nscores")

    if ax_bar is not None:
        ax_bar.set_xlim(0, 1.0)
        ax_bar.set_xticks([0, 1])
        ax_bar.set_xticklabels(["0", "1"], fontsize=4.25, color="#000000")
        _style_black_bottom_axis(ax_bar, show_left=False)
        for i, v in enumerate(mean_support):
            ax_bar.plot(
                [0, 1], [i, i], color=TRACK, lw=3.0, solid_capstyle="round", zorder=1
            )
            ax_bar.plot(
                [0, v], [i, i], color=ACC_L, lw=3.0, solid_capstyle="round", zorder=2
            )
            ax_bar.plot(
                [v, v],
                [i - 0.16, i + 0.16],
                color=ACC_D,
                lw=0.55,
                zorder=3,
                clip_on=False,
            )

    _bracket(fig, panel, bracket_lab[0], bracket_lab[1], 0.765, "Ranked feature")
    _bracket(fig, panel, bracket_dir[0], bracket_dir[1], 0.765, "Class shift")
    _bracket(fig, panel, bracket_hm[0], bracket_hm[1], 0.765, "Method support")
    if not solo_method:
        _bracket(fig, panel, 0.872, 0.984, 0.765, "Mean")
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


def _net_node_color(c_ctrl: float, c_case: float):
    pseudo = 1e-3
    lfc_max = 2.0
    lfc = np.log2((float(c_case) + pseudo) / (float(c_ctrl) + pseudo))
    neut = np.array(mcolors.to_rgba(NET_NEUT))
    if lfc >= 0:
        t = min(lfc / lfc_max, 1.0)
        target = np.array(mcolors.to_rgba(CLASS_CASE))
    else:
        t = min(-lfc / lfc_max, 1.0)
        target = np.array(mcolors.to_rgba(CLASS_CTRL))
    return tuple((1.0 - t) * neut + t * target)


def _net_node_radius(c_ctrl: float, c_case: float, is_hub: bool = False) -> float:
    mean = max((float(c_ctrl) + float(c_case)) / 2.0, 0.05)
    lo, hi = np.log10(0.05), np.log10(60.0)
    norm = np.clip((np.log10(mean) - lo) / (hi - lo), 0, 1)
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
                    c0=_get_abundance(stat_map, feat, "control_mean_pct"),
                    c1=_get_abundance(stat_map, feat, "case_mean_pct"),
                )
        if G.has_edge(f1, f2):
            if w > G[f1][f2]["weight"]:
                G[f1][f2].update(weight=w, rank=row_i + 1)
        else:
            G.add_edge(f1, f2, weight=w, rank=row_i + 1)
    if len(G.edges()) == 0:
        raise RuntimeError("no plottable interaction edges")
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
    hub_pos = np.array(pos[hub])
    strengths = [G[u][v]["weight"] for u, v in G.edges()]
    s_min, s_max = float(min(strengths)), float(max(strengths))

    def _ns(s):
        return (float(s) - s_min) / (s_max - s_min) if s_max > s_min else 0.6

    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    edge_list = sorted(G.edges(data=True), key=lambda e: e[2]["weight"])
    badge_xy = []
    for idx, (u, v, ed) in enumerate(edge_list):
        p1, p2 = pos[u], pos[v]
        n = _ns(ed["weight"])
        curv = 0.11 * (1 if idx % 2 == 0 else -1)
        xe, ye, _, _ = _net_bezier(p1, p2, curv=curv)
        lw = _net_edge_linewidth(n)
        col = NET_EDGE_CMAP(0.12 + n * 0.88)
        ax.plot(
            xe,
            ye,
            color=col,
            linewidth=lw,
            alpha=0.82,
            zorder=1,
            solid_capstyle="round",
        )
    badge_np = np.array(badge_xy) if badge_xy else np.zeros((0, 2))

    for node in G.nodes():
        nd = G.nodes[node]
        is_hub = bool(node == hub and hub_deg >= 4)
        c = _net_node_color(nd["c0"], nd["c1"])
        r = _net_node_radius(nd["c0"], nd["c1"], is_hub=is_hub)
        x, y = pos[node]
        ax.add_patch(
            plt.Circle(
                (x, y),
                r,
                facecolor=c,
                edgecolor=INK,
                linewidth=0.8,
                zorder=3,
                alpha=0.96,
            )
        )

    char_w = 0.058
    line_h = 0.26
    node_list = list(G.nodes())
    nl = len(node_list)
    node_xy = np.array([pos[n] for n in node_list], dtype=float)
    node_r_arr = np.array(
        [
            _net_node_radius(
                G.nodes[n]["c0"], G.nodes[n]["c1"], is_hub=(n == hub and hub_deg >= 4)
            )
            for n in node_list
        ],
        dtype=float,
    )
    lbl_pos = np.zeros((nl, 2), dtype=float)
    lbl_ideal = np.zeros((nl, 2), dtype=float)
    lbl_anch = np.zeros((nl, 2), dtype=float)
    lbl_text: list[str] = []
    lbl_fs: list[float] = []
    lbl_fw: list[str] = []
    lbl_hw = np.zeros(nl, dtype=float)
    lbl_hh = np.zeros(nl, dtype=float)

    for ni, node in enumerate(node_list):
        nd = G.nodes[node]
        is_hub = bool(node == hub and hub_deg >= 4)
        r = node_r_arr[ni]
        x, y = pos[node]
        raw = str(nd["label"])
        lbl = _net_italic(raw)
        fs = float(np.clip(5.8 + (r - 0.08) / max(0.32 - 0.08, 1e-9) * 1.6, 5.8, 7.4))
        fw = "bold" if is_hub else "normal"
        hw = len(raw) * char_w * (fs / 7.5) / 2 + 0.04
        hh = line_h * (fs / 7.5) / 2
        lbl_text.append(lbl)
        lbl_fs.append(fs)
        lbl_fw.append(fw)
        lbl_hw[ni] = hw
        lbl_hh[ni] = hh
        if is_hub:
            nat_angle = np.pi / 2
        else:
            dv = np.array([x, y]) - hub_pos
            nat_angle = (
                np.arctan2(dv[1], dv[0]) if np.linalg.norm(dv) > 0.01 else np.pi / 2
            )
        pad = r + (0.65 if is_hub else 0.52)
        best_score, best_angle = 1e9, nat_angle
        for angle in np.linspace(0, 2 * np.pi, 13)[:-1]:
            lx = x + np.cos(angle) * pad
            ly = y + np.sin(angle) * pad
            score = 0.0
            for pj in range(ni):
                odx = abs(lx - lbl_pos[pj, 0]) - (hw + lbl_hw[pj] + 0.08)
                ody = abs(ly - lbl_pos[pj, 1]) - (hh + lbl_hh[pj] + 0.05)
                if odx < 0 and ody < 0:
                    score += 5 + (-odx) * (-ody) * 3
            for nj in range(nl):
                dist = np.hypot(lx - node_xy[nj, 0], ly - node_xy[nj, 1])
                if dist < node_r_arr[nj] + max(hw, hh) + 0.12:
                    score += 3
            angle_diff = abs(((angle - nat_angle) + np.pi) % (2 * np.pi) - np.pi)
            score += angle_diff * 0.12
            if score < best_score:
                best_score, best_angle = score, angle
        lx0 = x + np.cos(best_angle) * pad
        ly0 = y + np.sin(best_angle) * pad
        lbl_pos[ni] = [lx0, ly0]
        lbl_ideal[ni] = [lx0, ly0]
        lbl_anch[ni] = [
            x + np.cos(best_angle) * (r + 0.05),
            y + np.sin(best_angle) * (r + 0.05),
        ]

    k_s, k_l, k_n, k_b = 0.18, 0.70, 0.45, 0.35
    badge_r, dt, lims = 0.18, 0.06, 4.85
    for _ in range(1200):
        forces = k_s * (lbl_ideal - lbl_pos)
        for i in range(nl):
            for j in range(nl):
                if i == j:
                    continue
                dx = lbl_pos[i, 0] - lbl_pos[j, 0]
                dy = lbl_pos[i, 1] - lbl_pos[j, 1]
                px = (lbl_hw[i] + lbl_hw[j] + 0.10) - abs(dx)
                py = (lbl_hh[i] + lbl_hh[j] + 0.06) - abs(dy)
                if px > 0 and py > 0:
                    if px <= py:
                        forces[i, 0] += k_l * px * (1 if dx >= 0 else -1)
                    else:
                        forces[i, 1] += k_l * py * (1 if dy >= 0 else -1)
            for nj in range(nl):
                dx = lbl_pos[i, 0] - node_xy[nj, 0]
                dy = lbl_pos[i, 1] - node_xy[nj, 1]
                dist = np.hypot(dx, dy) + 1e-9
                md = node_r_arr[nj] + max(lbl_hw[i], lbl_hh[i]) + 0.13
                if dist < md:
                    forces[i] += k_n * (md - dist) / dist * np.array([dx, dy])
            for bi in range(len(badge_np)):
                dx = lbl_pos[i, 0] - badge_np[bi, 0]
                dy = lbl_pos[i, 1] - badge_np[bi, 1]
                dist = np.hypot(dx, dy) + 1e-9
                md = badge_r + max(lbl_hw[i], lbl_hh[i]) + 0.06
                if dist < md:
                    forces[i] += k_b * (md - dist) / dist * np.array([dx, dy])
        lbl_pos += dt * forces
        lbl_pos[:, 0] = np.clip(lbl_pos[:, 0], -lims, lims)
        lbl_pos[:, 1] = np.clip(lbl_pos[:, 1], -lims, lims)

    for k in range(nl):
        ax.plot(
            [lbl_anch[k, 0], lbl_pos[k, 0]],
            [lbl_anch[k, 1], lbl_pos[k, 1]],
            color=INK,
            linewidth=0.34,
            alpha=0.35,
            linestyle=(0, (2, 3)),
            solid_capstyle="round",
            zorder=4,
        )
        ax.text(
            lbl_pos[k, 0],
            lbl_pos[k, 1],
            lbl_text[k],
            ha="center",
            va="center",
            fontsize=lbl_fs[k],
            color=INK,
            fontweight=lbl_fw[k],
            path_effects=[mpe.withStroke(linewidth=2.2, foreground="white")],
            zorder=6,
        )
    ax.set_xlim(-5.0, 5.0)
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
        f"{s_max:.3g}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="black",
    )
    ax.text(
        xc,
        fy0 - 0.040,
        f"{s_min:.3g}",
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
    fig = plt.figure(figsize=(COL_W_2, 125 * MM))
    fig.patch.set_facecolor(BG)
    ax_net = fig.add_axes([0.02, 0.06, 0.69, 0.88], zorder=4)
    ax_leg1 = fig.add_axes([0.73, 0.53, 0.25, 0.37], zorder=12)
    ax_leg2 = fig.add_axes([0.75, 0.17, 0.20, 0.28], zorder=12)
    try:
        s_min, s_max = _draw_network(ax_net, tab, stats, top_k, layout=layout)
        _legend_size_colour_net(ax_leg1, ctrl_text="controls", case_text="cases")
        _legend_edge_net(ax_leg2, s_min, s_max)
    except Exception as exc:
        ax_net.clear()
        _soft_missing(ax_net, f"Interaction network unavailable\n{exc}")
        ax_leg1.axis("off")
        ax_leg2.axis("off")
    save_all(fig, out_stem)
    plt.close(fig)
    return True
