from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

MM = 1.0 / 25.4
COL_W_1 = 89 * MM
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

# Blue family used by the representation-impact panels.
C_NAVY = "#0C4A6E"
C_DARK = "#1565A8"
C_MID = "#2E9DC8"
C_BRIGHT = "#45C1EE"
C_SKY = "#7DCFEA"
C_PALE = "#B8E3F4"
C_LIGHT = "#DEF1FA"
C_TEAL = "#1D5288"
C_HILITE = "#3277EF"

METHOD_COLORS = {
    "SHAP": "#3B6B8F",
    "LIME": "#7B5EA7",
    "ALE": "#4A8C6F",
    "Permutation": "#8F5B3B",
    "consensus": INK,
}

UC_CTRL = "#8BAABF"
UC_CASE = "#B499C2"
MINDSET_CTRL = "#B8B8B8"
MINDSET_CASE = "#2FA7D6"
NET_NEUT = "#F7F7F7"

HMAP_CMAP = LinearSegmentedColormap.from_list(
    "blue_hmap", [C_LIGHT, C_PALE, C_SKY, C_BRIGHT, C_MID, C_DARK, C_NAVY], N=256
)
CORR_CMAP = LinearSegmentedColormap.from_list(
    "silver_blue",
    ["#F0F4F8", "#C8D8E8", "#8BAFC8", "#4E7FA8", "#1D5288", "#0C3260"],
    N=256,
)
NET_EDGE_CMAP = LinearSegmentedColormap.from_list(
    "net_edge", ["#f7f9fc", "#DBE2E9"], N=256
)
SUPPORT_CMAP = LinearSegmentedColormap.from_list(
    "support", ["#ffffff", "#eff6ff", "#bfdbfe", "#60a5fa", "#1d4ed8"], N=256
)

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
    "savefig.dpi": 300,
}

REPRESENTATION_RC = dict(RC)
REPRESENTATION_RC.update(
    {
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "font.size": 7.8,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "legend.fontsize": 6.9,
    }
)


def apply() -> None:
    mpl.rcParams.update(RC)


def save_all(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)
