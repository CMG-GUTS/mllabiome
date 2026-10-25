from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .traceability import Schaefer400Provenance


def render_schaefer_scores(
    regional_scores: Mapping[int, float] | pd.DataFrame,
    provenance: Schaefer400Provenance,
    output: str | Path,
    *,
    label: str = "Normalized regional sensitivity",
    format: str = "both",
    top_labels: int = 12,
) -> tuple[Path, ...]:
    if provenance.atlas_path is None:
        raise ValueError(
            "A spatially verified Schaefer atlas path is required for anatomical plotting"
        )
    try:
        import nibabel as nib
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise ImportError("Brain rendering requires nibabel and matplotlib") from exc
    if isinstance(regional_scores, pd.DataFrame):
        if not {"region_index", "mean_absolute_delta"}.issubset(
            regional_scores.columns
        ):
            raise ValueError("Expected region_index and mean_absolute_delta columns")
        if regional_scores["region_index"].duplicated().any():
            raise ValueError("Repeated ROI indices")
        scores = dict(
            zip(
                regional_scores["region_index"].astype(int),
                regional_scores["mean_absolute_delta"].astype(float),
            )
        )
    else:
        scores = dict(regional_scores)
    n_regions = len(provenance.labels)
    if n_regions not in {100, 200, 400}:
        raise ValueError("Unsupported Schaefer parcellation")
    if not scores or len(scores) > n_regions:
        raise ValueError("Supply one or more Schaefer ROI scores")
    if any(
        not isinstance(k, (int, np.integer)) or not 0 <= k < n_regions for k in scores
    ):
        raise ValueError(
            f"Schaefer indices are zero based and must be 0..{n_regions - 1}"
        )
    if any(not np.isfinite(v) or v < 0 for v in scores.values()):
        raise ValueError("Regional magnitudes must be finite and nonnegative")
    if format not in {"png", "svg", "both"} or top_labels < 0:
        raise ValueError("Unsupported figure settings")
    image = nib.load(str(provenance.atlas_path))
    if len(image.shape) != 3:
        raise ValueError("Atlas must be a 3D integer-labeled NIfTI volume")
    labels = np.asanyarray(image.dataobj)
    if not np.isfinite(labels).all() or not np.array_equal(labels, np.rint(labels)):
        raise ValueError("Atlas contains invalid parcel labels")
    labels = labels.astype(np.int16)
    if not set(range(1, n_regions + 1)).issubset(set(np.unique(labels))) or np.any(
        (labels < 0) | (labels > n_regions)
    ):
        raise ValueError(
            f"Atlas must contain integer labels 1..{n_regions} and optional background 0"
        )
    affine = np.asarray(image.affine, dtype=float)
    if not np.allclose(
        affine[:3, :3], np.diag(np.diag(affine[:3, :3])), atol=1e-5
    ) or np.any(np.diag(affine[:3, :3]) == 0):
        raise ValueError(
            "Canonicalize or resample the atlas to an axis-aligned anatomical grid"
        )
    vmax = max(scores.values())
    normalized = {i: value / vmax if vmax else 0.0 for i, value in scores.items()}
    cmap = LinearSegmentedColormap.from_list(
        "mllabiome_brain", ["#838F9F", "#FFFFFF", "#3D79EA"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), dpi=160, facecolor="white")
    projections = (
        (0, "Sagittal maximum projection"),
        (1, "Coronal maximum projection"),
        (2, "Axial maximum projection"),
    )
    for ax, (axis, title) in zip(axes, projections):
        remaining = [i for i in range(3) if i != axis]
        x, y = remaining
        extent = (
            float(affine[x, 3] - affine[x, x] / 2),
            float(affine[x, 3] + affine[x, x] * (labels.shape[x] - 0.5)),
            float(affine[y, 3] - affine[y, y] / 2),
            float(affine[y, 3] + affine[y, y] * (labels.shape[y] - 0.5)),
        )
        outline = np.any(labels != 0, axis=axis).T
        background = np.full((*outline.shape, 4), 1.0, dtype=float)
        background[outline, :3] = (0.84, 0.85, 0.86)
        background[~outline, 3] = 0
        ax.imshow(background, origin="lower", extent=extent, interpolation="nearest")
        for index, score in sorted(normalized.items(), key=lambda item: item[1]):
            mask = np.any(labels == (index + 1), axis=axis).T
            rgba = np.zeros((*mask.shape, 4), dtype=float)
            rgba[mask] = cmap(score)
            rgba[mask, 3] = 0.88
            ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title, fontsize=11, color="#1B2639", pad=13)
        ax.set_aspect("equal")
        ax.axis("off")
    prominent = sorted(scores, key=lambda index: (-scores[index], index))[:top_labels]
    legend = [f"{index + 1:03d} · {provenance.labels[index]}" for index in prominent]
    if legend:
        fig.text(
            0.5,
            0.115,
            "\n".join(legend),
            ha="center",
            va="top",
            fontsize=7.5,
            color="#1B2639",
            linespacing=1.45,
        )
    fig.text(0.5, 0.94, label, ha="center", va="center", fontsize=14, color="#1B2639")
    fig.text(
        0.5,
        0.015,
        f"Schaefer-{n_regions} atlas · Maximum projections · 2-D parcel footprints may overlap · Values normalized within this figure",
        ha="center",
        fontsize=8.2,
        color="#657791",
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.86, bottom=0.30, wspace=0.01)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        target.with_suffix("." + ext)
        for ext in (("png", "svg") if format == "both" else (format,))
    )
    try:
        for path in paths:
            fig.savefig(
                path, bbox_inches="tight", pad_inches=0.16, facecolor="white", dpi=230
            )
    finally:
        plt.close(fig)
    return paths
