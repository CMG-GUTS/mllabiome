from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

MM = 1.0 / 25.4
COL_W_1 = 89 * MM
COL_W_1_5 = 120 * MM
COL_W_2 = 183 * MM
MAX_FIG_H = 170 * MM

INK = "#0f172a"
MID = "#64748b"
DIM = "#94a3b8"
TRACK = "#e2e8f0"
BG = "#ffffff"
PANEL_BG = "#f8fafc"
ACC = "#2563eb"
ACC_D = "#1d4ed8"
ACC_L = "#dbeafe"


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
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 7.0,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.bbox": None,
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
        "font.size": 7.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
    }
)


def apply() -> None:
    mpl.rcParams.update(RC)


def _is_panel_label(text_artist) -> bool:
    text = str(text_artist.get_text()).strip()
    weight = text_artist.get_fontweight()
    bold = str(weight).lower() in {"bold", "heavy", "semibold", "demibold"}
    try:
        bold = bold or float(weight) >= 600
    except Exception:
        pass
    return len(text) == 1 and text.isalpha() and text.islower() and bold


def enforce_nature_figure(fig) -> None:

    width_in, height_in = fig.get_size_inches()
    width_mm = float(width_in) / MM
    height_mm = float(height_in) / MM
    if width_mm <= 89.5:
        target_w_mm = 89.0
    elif width_mm <= 136.0:
        target_w_mm = min(136.0, max(120.0, width_mm))
    else:
        target_w_mm = 183.0
    scale = target_w_mm / max(width_mm, 1e-9)
    target_h_mm = height_mm * scale
    if target_h_mm > 170.0:
        height_scale = 170.0 / target_h_mm
        target_w_mm *= height_scale
        target_h_mm = 170.0
    fig.set_size_inches(target_w_mm * MM, target_h_mm * MM, forward=True)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    for artist in fig.findobj(match=Text):
        size = float(artist.get_fontsize())
        artist.set_fontsize(
            8.0 if _is_panel_label(artist) else min(7.0, max(5.0, size))
        )
    for line in fig.findobj(match=Line2D):
        lw = float(line.get_linewidth())
        if lw > 0:
            line.set_linewidth(min(1.0, max(0.25, lw)))
    for patch in fig.findobj(match=Patch):
        lw = float(patch.get_linewidth())
        if lw > 0:
            patch.set_linewidth(min(1.0, max(0.25, lw)))


def compact_svg(path: Path) -> Path:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<metadata\b[^>]*>.*?</metadata>", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r">\s+<", "><", text)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def save_svg(fig, path: Path, *, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enforce_nature_figure(fig)
    fig.savefig(
        path,
        dpi=max(300, int(dpi)),
        metadata={"Date": None},
        bbox_inches=None,
        pad_inches=0,
    )
    return compact_svg(path)


def save_all(fig, stem: Path) -> None:
    save_svg(fig, stem.with_suffix(".svg"), dpi=300)
