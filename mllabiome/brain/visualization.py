from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion

from .attributions import (
    BrainAttribution,
    _decode,
    example_brain_attributions,
    available_brain_regions,
    read_brain_attributions,
    validate_attributions,
)


@dataclass(frozen=True, slots=True)
class _Surface:
    vertices: np.ndarray
    faces: np.ndarray
    sulcal: np.ndarray
    parcels: np.ndarray


@dataclass(slots=True)
class _Panel:
    name: str
    hemisphere: str
    rectangle: tuple[float, float, float, float]
    axes: object
    collection: object
    face_ids: np.ndarray
    requested: set[int]
    masks: dict[int, np.ndarray] | None = None


@dataclass(frozen=True, slots=True)
class _Callout:
    attribution: BrainAttribution
    panel: str
    rectangle: tuple[float, float, float, float]
    side: str
    target_y: float
    boundary: np.ndarray


_COLORS = ("#838F9F", "#FFFFFF", "#3D79EA")
_INK = "#1B2639"
_MUTED = "#657791"
_RULE = "#98ABC4"


def _resources(include_subcortex: bool):
    try:
        from nilearn import datasets, image, surface
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Brain attribution plots require nilearn and nibabel"
        ) from exc
    mesh = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    atlas = datasets.fetch_atlas_surf_destrieux()
    regions = tuple(_decode(label) for label in atlas.labels)
    surfaces = {}
    for hemisphere in ("left", "right"):
        vertices, faces = surface.load_surf_mesh(getattr(mesh, f"pial_{hemisphere}"))
        sulcal = surface.load_surf_data(getattr(mesh, f"sulc_{hemisphere}"))
        parcels = np.asarray(atlas[f"map_{hemisphere}"], dtype=int)
        if not len(vertices) == len(sulcal) == len(parcels):
            raise ValueError(
                "The fsaverage5 and Destrieux vertex arrays are incompatible"
            )
        surfaces[hemisphere] = _Surface(
            np.asarray(vertices, dtype=float),
            np.asarray(faces, dtype=np.int32),
            np.asarray(sulcal, dtype=float),
            parcels,
        )
    deep = None
    if include_subcortex:
        ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
        segmentation = nib.as_closest_canonical(image.load_img(ho.maps))
        template = datasets.load_mni152_template(resolution=2)
        anatomy = image.resample_to_img(
            template, segmentation, interpolation="continuous"
        )
        labels = np.asanyarray(segmentation.dataobj).astype(np.int16)
        intensity = np.asarray(anatomy.get_fdata(dtype=np.float32))
        if labels.shape != intensity.shape:
            raise ValueError("Harvard–Oxford and MNI152 image grids do not match")
        deep = (
            labels,
            segmentation.affine,
            tuple(_decode(x) for x in ho.labels),
            intensity,
        )
    return surfaces, regions, deep


def _lookup(attributions, cortex_labels, deep):
    subcortex_labels = deep[2] if deep is not None else ()
    indices = {}
    for item in attributions:
        labels = cortex_labels if item.kind == "cortex" else subcortex_labels
        if item.region not in labels:
            raise ValueError(
                f"{item.region!r} is not a {item.kind} atlas label; "
                "use available_brain_regions() for exact names"
            )
        region_id = labels.index(item.region)
        if region_id == 0:
            raise ValueError(
                "Background or unknown atlas parcels cannot receive anatomical attributions"
            )
        indices[item.key] = region_id
    return indices


def _face_ids(parcel_vertices):
    return np.where(
        parcel_vertices[:, 0] == parcel_vertices[:, 1],
        parcel_vertices[:, 0],
        np.where(
            parcel_vertices[:, 1] == parcel_vertices[:, 2],
            parcel_vertices[:, 1],
            np.where(
                parcel_vertices[:, 0] == parcel_vertices[:, 2],
                parcel_vertices[:, 0],
                0,
            ),
        ),
    )


def _surface_artist(surface, hemisphere, view, scores, colormap):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    vertices, faces, sulcal, parcels = (
        surface.vertices,
        surface.faces,
        surface.sulcal,
        surface.parcels,
    )
    projected = np.column_stack((-vertices[:, 1], vertices[:, 2]))
    triangles = vertices[faces]
    polygon_depth = triangles[:, :, 0].mean(axis=1)
    sulcal_depth = sulcal[faces].mean(axis=1)
    lower, upper = np.percentile(sulcal_depth, (3, 97))
    grooves = np.clip((sulcal_depth - lower) / max(upper - lower, 1e-10), 0, 1)
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
    light = np.array((-0.7, -0.25, 0.67))
    light /= np.linalg.norm(light)
    shade = np.clip(
        0.70 + 0.14 * grooves + 0.10 * np.clip(normal @ light, 0, 1), 0.57, 0.92
    )
    rgba = np.column_stack((np.repeat(shade[:, None], 3, axis=1), np.ones(len(shade))))
    labels = _face_ids(parcels[faces])
    for region_id, score in scores.items():
        selected = np.count_nonzero(parcels[faces] == region_id, axis=1) >= 2
        rgba[selected, :3] = 0.12 * rgba[selected, :3] + 0.88 * np.asarray(
            to_rgb(colormap(score))
        )
    far_to_near = np.argsort(polygon_depth)
    if (hemisphere == "left" and view == "lateral") or (
        hemisphere == "right" and view == "medial"
    ):
        far_to_near = far_to_near[::-1]
    collection = PolyCollection(
        projected[faces][far_to_near],
        facecolors=rgba[far_to_near],
        edgecolors=(0.43, 0.47, 0.52, 0.055),
        linewidths=0.07,
        antialiased=True,
        zorder=2,
    )
    return collection, labels[far_to_near], projected


def _figure_pixels(mask):
    height, width = mask.shape
    y, x = np.where(mask)
    return np.column_stack(((x + 0.5) / width, 1.0 - (y + 0.5) / height))


def _visible_regions(figure, panels):
    originals = []
    for panel in panels:
        collection = panel.collection
        originals.append(
            (
                collection,
                collection.get_facecolors().copy(),
                collection.get_linewidths().copy(),
            )
        )
        if panel.face_ids.min() < -19 or panel.face_ids.max() > 235:
            raise ValueError(
                "Surface atlas indices cannot be represented in the visibility mask"
            )
        color = np.zeros((len(panel.face_ids), 4), dtype=float)
        color[:, 0] = (panel.face_ids + 20) / 255.0
        color[:, 3] = 1
        collection.set_facecolors(color)
        collection.set_linewidth(0)
        collection.set_antialiased(False)
    try:
        figure.canvas.draw()
        pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
        height, width = pixels.shape[:2]
        for panel in panels:
            bounds = panel.axes.get_window_extent()
            x0 = max(0, int(np.floor(bounds.x0)))
            x1 = min(width, int(np.ceil(bounds.x1)))
            y0 = max(0, height - int(np.ceil(bounds.y1)))
            y1 = min(height, height - int(np.floor(bounds.y0)))
            if x0 >= x1 or y0 >= y1:
                raise RuntimeError(f"Invalid axes coordinates for {panel.name}")
            portion = pixels[y0:y1, x0:x1]
            masks = {}
            for region_id in panel.requested:
                target = np.array((region_id + 20, 0, 0), dtype=np.uint8)
                visible = np.all(portion == target, axis=2)
                if visible.any():
                    mask = np.zeros((height, width), dtype=bool)
                    mask[y0:y1, x0:x1] = visible
                    masks[region_id] = mask
            panel.masks = masks
    finally:
        for panel, (collection, colors, widths) in zip(panels, originals):
            collection.set_facecolors(colors)
            collection.set_linewidths(widths)
            collection.set_antialiased(True)
        figure.canvas.draw()


def _surface_callouts(figure, surfaces, attributions, indices, colormap, has_subcortex):
    rows = (
        ((0.665, 0.270), (0.355, 0.270))
        if has_subcortex
        else ((0.52, 0.405), (0.105, 0.405))
    )
    panels = []
    for hemisphere, column in (("left", 0.020), ("right", 0.510)):
        for index, (bottom, height) in enumerate(rows):
            view = "lateral" if index == 0 else "medial"
            bounds = (column, bottom, 0.47, height)
            axes = figure.add_axes(
                (column + 0.115, bottom + 0.045, 0.24, height - 0.080)
            )
            scores = {
                indices[item.key]: item.score
                for item in attributions
                if item.kind == "cortex" and item.hemisphere == hemisphere
            }
            artist, face_ids, coordinates = _surface_artist(
                surfaces[hemisphere], hemisphere, view, scores, colormap
            )
            axes.add_collection(artist)
            axes.set_xlim(coordinates[:, 0].min() - 6, coordinates[:, 0].max() + 6)
            axes.set_ylim(coordinates[:, 1].min() - 5, coordinates[:, 1].max() + 5)
            axes.set_aspect("equal")
            axes.axis("off")
            panels.append(
                _Panel(
                    f"{hemisphere}_{view}",
                    hemisphere,
                    bounds,
                    axes,
                    artist,
                    face_ids,
                    set(scores),
                )
            )
            figure.text(
                column + 0.235,
                bottom + height - 0.020,
                f"{hemisphere.upper()}  ·  {view.upper()}",
                ha="center",
                va="center",
                fontsize=9.1,
                color=_MUTED,
            )
    figure.canvas.draw()
    _visible_regions(figure, panels)
    callouts = []
    for item in attributions:
        if item.kind != "cortex":
            continue
        region_id = indices[item.key]
        possible = [panel for panel in panels if panel.hemisphere == item.hemisphere]
        panel = max(
            possible, key=lambda p: np.count_nonzero(p.masks.get(region_id, ()))
        )
        mask = panel.masks.get(region_id)
        if mask is None or np.count_nonzero(mask) < 12:
            raise ValueError(
                f"Cortical ROI {item.key!r} is not visibly exposed in lateral or medial views"
            )
        pixel_positions = _figure_pixels(mask)
        center = np.median(pixel_positions, axis=0)
        side = "left" if center[0] < panel.rectangle[0] + 0.235 else "right"
        perimeter = mask & ~binary_erosion(mask, iterations=1)
        callouts.append(
            _Callout(
                item,
                panel.name,
                panel.rectangle,
                side,
                float(center[1]),
                _figure_pixels(perimeter),
            )
        )
    return callouts


def _subcortical_callouts(figure, deep, attributions, indices, colormap):
    from matplotlib.colors import to_rgb

    atlas, affine, _, image = deep
    if not np.allclose(affine[:3, :3], np.diag(np.diag(affine)[:3]), atol=1e-4):
        raise ValueError("The canonical subcortical atlas must be axis-aligned")
    positions = [
        affine[k, k] * np.arange(atlas.shape[k]) + affine[k, 3] for k in range(3)
    ]
    selected = [item for item in attributions if item.kind == "subcortex"]
    masks = {indices[item.key]: atlas == indices[item.key] for item in selected}
    if any(not masks[indices[item.key]].any() for item in selected):
        raise ValueError(
            "One or more requested Harvard–Oxford atlas regions contains no voxels"
        )
    callouts = []
    for title, axis, column in (
        ("CORONAL MAXIMUM PROJECTION", 1, 0.020),
        ("AXIAL MAXIMUM PROJECTION", 2, 0.510),
    ):
        rectangle = (column, 0.100, 0.47, 0.228)
        axes = figure.add_axes((column + 0.117, 0.120, 0.237, 0.172))
        kept = [dimension for dimension in range(3) if dimension != axis]
        xcoords, ycoords = (positions[k] for k in kept)
        dx, dy = xcoords[1] - xcoords[0], ycoords[1] - ycoords[0]
        extent = (
            xcoords[0] - dx / 2,
            xcoords[-1] + dx / 2,
            ycoords[0] - dy / 2,
            ycoords[-1] + dy / 2,
        )
        anatomical_projection = np.max(image, axis=axis).T
        positive = anatomical_projection[anatomical_projection > 0]
        if positive.size == 0:
            raise ValueError("The subcortical MNI152 template is empty")
        low, high = np.percentile(positive, (5, 99))
        contrast = np.clip((anatomical_projection - low) / max(high - low, 1e-6), 0, 1)
        gray = 0.73 + 0.23 * contrast
        background = np.stack(
            (
                gray,
                gray,
                gray,
                (anatomical_projection > max(low, 0.025 * positive.max())).astype(
                    float
                ),
            ),
            axis=-1,
        )
        axes.imshow(
            background,
            extent=extent,
            origin="lower",
            interpolation="bilinear",
            zorder=1,
        )
        visible_ids = np.zeros(anatomical_projection.shape, dtype=np.int32)
        for item in sorted(selected, key=lambda entry: entry.score):
            region_id = indices[item.key]
            footprint = np.any(masks[region_id], axis=axis).T
            overlay = np.zeros((*footprint.shape, 4), dtype=float)
            overlay[footprint, :3] = np.asarray(to_rgb(colormap(item.score)))
            overlay[footprint, 3] = 0.83
            axes.imshow(
                overlay,
                extent=extent,
                origin="lower",
                interpolation="nearest",
                zorder=2,
            )
            visible_ids[footprint] = region_id
        axes.set_xlim(extent[:2])
        axes.set_ylim(extent[2:])
        axes.set_aspect("equal")
        axes.axis("off")
        figure.canvas.draw()
        figure.text(
            column + 0.235,
            rectangle[1] + rectangle[3] - 0.012,
            title,
            ha="center",
            va="center",
            color=_MUTED,
            fontsize=9.1,
        )
        for item in selected:
            footprint = visible_ids == indices[item.key]
            if not footprint.any():
                continue
            rows, cols = np.where(footprint)
            centers = figure.transFigure.inverted().transform(
                axes.transData.transform(
                    np.column_stack((xcoords[cols], ycoords[rows]))
                )
            )
            median = np.median(centers, axis=0)
            perimeter = footprint & ~binary_erosion(footprint, iterations=1)
            rows, cols = np.where(perimeter)
            boundary = figure.transFigure.inverted().transform(
                axes.transData.transform(
                    np.column_stack((xcoords[cols], ycoords[rows]))
                )
            )
            side = "left" if median[0] < column + 0.235 else "right"
            callouts.append(
                (
                    _Callout(item, title, rectangle, side, float(median[1]), boundary),
                    int(footprint.sum()),
                )
            )
    chosen = {}
    for item, area in callouts:
        key = item.attribution.key
        if key not in chosen or area > chosen[key][1]:
            chosen[key] = (item, area)
    if len(chosen) != len(selected):
        missing = [item.region for item in selected if item.key not in chosen]
        raise ValueError(
            f"Subcortical ROIs are not visibly exposed in either maximum projection: {missing}"
        )
    return [item for item, _ in chosen.values()]


def _anchor(boundary, side, target_y):
    if not len(boundary):
        raise ValueError("Cannot position an annotation at an empty ROI boundary")
    edge = np.quantile(boundary[:, 0], 0.05 if side == "left" else 0.95)
    span = max(float(np.ptp(boundary[:, 0])), 0.001)
    distance = (boundary[:, 0] - edge) / span
    horizontal = (
        1.3 * np.maximum(distance, 0)
        if side == "left"
        else -1.3 * np.minimum(distance, 0)
    )
    ranking = horizontal + 6.0 * np.abs(boundary[:, 1] - target_y)
    return boundary[int(np.argmin(ranking))]


def _label_figure(figure, callouts, *, max_per_side, draw_leaders):
    from matplotlib.lines import Line2D

    groups = {}
    for item in callouts:
        groups.setdefault((item.panel, item.side), []).append(item)
    placed = []
    for (name, side), members in groups.items():
        if len(members) > max_per_side:
            raise ValueError(f"Too many labels on {name}, {side}: {len(members)}")
        members.sort(key=lambda entry: entry.target_y, reverse=True)
        rectangle = members[0].rectangle
        bottom = rectangle[1] + 0.037
        top = rectangle[1] + rectangle[3] - 0.036
        if top - bottom < (len(members) - 1) * 0.029:
            raise ValueError(
                f"The {name} panel cannot accommodate {len(members)} labels"
            )
        positions = np.array([np.clip(item.target_y, bottom, top) for item in members])
        for index in range(1, len(positions)):
            positions[index] = min(positions[index], positions[index - 1] - 0.029)
        if positions[-1] < bottom:
            positions += bottom - positions[-1]
        if positions[0] > top:
            positions -= positions[0] - top
        for item, y in zip(members, positions):
            x = (
                rectangle[0] + 0.103
                if side == "left"
                else rectangle[0] + rectangle[2] - 0.103
            )
            label = figure.text(
                x,
                y,
                item.attribution.display_label,
                fontsize=8.9,
                color=_INK,
                ha="right" if side == "left" else "left",
                va="center",
                zorder=9,
            )
            placed.append((item, label, side, float(y)))
    if not draw_leaders:
        return
    figure.canvas.draw()
    for item, label, side, y in placed:
        bounds = label.get_window_extent(figure.canvas.get_renderer()).transformed(
            figure.transFigure.inverted()
        )
        x = bounds.x1 + 0.005 if side == "left" else bounds.x0 - 0.005
        target = _anchor(item.boundary, side, y)
        left, bottom, width, height = item.rectangle
        tolerance = 0.008
        if not (
            left - tolerance <= x <= left + width + tolerance
            and bottom - tolerance <= y <= bottom + height + tolerance
            and left - tolerance <= target[0] <= left + width + tolerance
            and bottom - tolerance <= target[1] <= bottom + height + tolerance
        ):
            raise ValueError(
                f"Anatomical annotation crosses its own panel boundary: {item.attribution.key}"
            )
        figure.add_artist(
            Line2D(
                (x, target[0]),
                (y, target[1]),
                color=_RULE,
                linewidth=0.75,
                linestyle=(0, (1.5, 3.3)),
                alpha=0.72,
                transform=figure.transFigure,
                zorder=8,
            )
        )


def _legend(figure, colormap, has_subcortex, title):
    bottom = 0.041 if has_subcortex else 0.042
    axes = figure.add_axes((0.345, bottom + 0.018, 0.310, 0.016))
    axes.imshow(
        np.linspace(0, 1, 500)[None, :],
        aspect="auto",
        cmap=colormap,
        extent=(0, 1, 0, 1),
    )
    axes.axis("off")
    figure.text(
        0.345, bottom + 0.039, "0", color=_MUTED, ha="left", va="bottom", fontsize=9.5
    )
    figure.text(
        0.655, bottom + 0.039, "1", color=_MUTED, ha="right", va="bottom", fontsize=9.5
    )
    figure.text(0.5, bottom - 0.009, title, fontsize=11.8, color=_MUTED, ha="center")


def render_brain_attributions(
    attributions: Sequence[BrainAttribution],
    output: str | Path,
    *,
    format: str = "both",
    legend_label: str = "Pre-normalized regional attribution",
    max_labels_per_side: int = 7,
    label_lines: bool = True,
) -> tuple[Path, ...]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    values = validate_attributions(attributions)
    if format not in {"png", "svg", "both"}:
        raise ValueError("format must be png, svg or both")
    if not isinstance(max_labels_per_side, int) or max_labels_per_side < 1:
        raise ValueError("max_labels_per_side must be a positive integer")
    has_subcortex = any(item.kind == "subcortex" for item in values)
    surfaces, labels, subcortex = _resources(has_subcortex)
    indices = _lookup(values, labels, subcortex)
    colormap = LinearSegmentedColormap.from_list("mllabiome_brain", _COLORS, N=256)
    figure = Figure(
        figsize=(19, 14.5 if has_subcortex else 10.4), dpi=155, facecolor="white"
    )
    FigureCanvasAgg(figure)
    cortical = _surface_callouts(
        figure, surfaces, values, indices, colormap, has_subcortex
    )
    deep = (
        _subcortical_callouts(figure, subcortex, values, indices, colormap)
        if has_subcortex
        else []
    )
    _label_figure(
        figure,
        [*cortical, *deep],
        max_per_side=max_labels_per_side,
        draw_leaders=label_lines,
    )
    _legend(figure, colormap, has_subcortex, legend_label)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        destination.with_suffix(f".{suffix}")
        for suffix in (("png", "svg") if format == "both" else (format,))
    )
    for path in paths:
        figure.savefig(
            path, dpi=240, bbox_inches="tight", pad_inches=0.14, facecolor="white"
        )
    figure.clear()
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot atlas-verified anatomical ROI attributions"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--input-csv", type=Path)
    action.add_argument("--demo", action="store_true")
    action.add_argument("--list-regions", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("brain_attributions"))
    parser.add_argument("--format", choices=("png", "svg", "both"), default="both")
    parser.add_argument("--legend-label", default="Pre-normalized regional attribution")
    parser.add_argument("--max-labels-per-side", type=int, default=7)
    parser.add_argument("--no-label-lines", action="store_true")
    args = parser.parse_args(argv)
    if args.list_regions:
        for compartment, regions in available_brain_regions().items():
            print(f"{compartment.upper()}: {len(regions)} labels")
            for region in regions:
                print(region)
        return
    if args.demo:
        print("SIMULATED EXAMPLE SCORES; NOT MEASURED ATTRIBUTIONS", file=sys.stderr)
    rois = (
        example_brain_attributions()
        if args.demo
        else read_brain_attributions(args.input_csv)
    )
    paths = render_brain_attributions(
        rois,
        args.output,
        format=args.format,
        legend_label=args.legend_label,
        max_labels_per_side=args.max_labels_per_side,
        label_lines=not args.no_label_lines,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
