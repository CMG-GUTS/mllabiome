from __future__ import annotations

import re
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd


def _require_anatomical_visualization():
    packages = (
        "nilearn",
        "nibabel",
        "templateflow",
        "netneurotools",
        "torch",
        "brainmarks",
    )
    missing = tuple(package for package in packages if find_spec(package) is None)
    if missing:
        raise ImportError(
            f"AnatomicalTraceability requires {', '.join(missing)}. "
            "Install all anatomical traceability dependencies with: uv sync --extra anatomical-visualization"
        )


def _annotation_files(annotations):
    if annotations is None:
        raise ValueError(
            "Schaefer-400 7Networks fsaverage5 annotations were not returned by netneurotools"
        )
    for left_key, right_key in (("L", "R"), ("lh", "rh")):
        if isinstance(annotations, dict):
            left, right = annotations.get(left_key), annotations.get(right_key)
        else:
            left, right = (
                getattr(annotations, left_key, None),
                getattr(annotations, right_key, None),
            )
        if left is not None and right is not None:
            break
    else:
        raise ValueError(
            f"Unsupported Schaefer surface annotation type: {type(annotations).__name__}; expected L/R or lh/rh fields"
        )
    paths = (Path(left).expanduser(), Path(right).expanduser())
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(
            f"Downloaded Schaefer-400 fsaverage5 surface annotations are missing: {', '.join(absent)}"
        )
    if paths[0] == paths[1]:
        raise ValueError(
            "Left and right Schaefer surface annotations must be different files"
        )
    return paths


def _surface_atlas(provenance):
    _require_anatomical_visualization()
    try:
        import nibabel as nib
        from netneurotools.datasets import fetch_schaefer2018
        from nilearn import datasets, surface
    except ImportError as exc:
        raise ImportError(
            "Cannot import Schaefer surface dependencies. "
            "Install or repair them with: uv sync --extra anatomical-visualization"
        ) from exc
    meshes = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    atlas = fetch_schaefer2018(version="fsaverage5")
    lookup = {
        re.sub(r"[^a-z0-9]", "", str(name).casefold()): i
        for i, name in enumerate(provenance.labels)
    }
    if len(lookup) != 400:
        raise ValueError("Non-unique canonical Schaefer atlas labels")
    key = "400Parcels7Networks"
    if key not in atlas:
        raise ValueError(
            f"Schaefer surface atlas does not contain {key}; available keys: {sorted(atlas)}"
        )
    left, right = _annotation_files(atlas[key])
    out = {}
    matched = set()
    for hemisphere, path in (("left", left), ("right", right)):
        labels, _, names = nib.freesurfer.read_annot(str(path), orig_ids=False)
        mapped = np.full(len(labels), -1, dtype=np.int32)
        for index, name in enumerate(names):
            text = name.decode("utf-8") if isinstance(name, bytes) else str(name)
            canonical = re.sub(r"[^a-z0-9]", "", text.casefold())
            if canonical in lookup:
                mapped[labels == index] = lookup[canonical]
                matched.add(lookup[canonical])
        coords, faces = surface.load_surf_mesh(getattr(meshes, f"pial_{hemisphere}"))
        sulc = surface.load_surf_data(getattr(meshes, f"sulc_{hemisphere}"))
        if len(mapped) != len(coords) or len(sulc) != len(coords):
            raise ValueError(
                f"Schaefer {hemisphere} annotation and fsaverage5 surface vertices are incompatible"
            )
        out[hemisphere] = (
            np.asarray(coords, float),
            np.asarray(faces, int),
            np.asarray(sulc, float),
            mapped,
        )
    if matched != set(range(400)):
        missing = [
            provenance.labels[index] for index in sorted(set(range(400)) - matched)
        ]
        raise ValueError(
            f"Schaefer fsaverage5 surface and volumetric atlas labels disagree: {len(matched)}/400 matched; first unmatched: {missing[:8]}"
        )
    return out


def _view_collection(ax, mesh, hemisphere, view, chosen, *, autoscale=True):
    from matplotlib.collections import PolyCollection

    coords, faces, sulc, parcel = mesh
    if view == "superior":
        xy = np.column_stack((coords[:, 0], -coords[:, 1]))
        depth = coords[:, 2]
        near_low = False
    else:
        xy = np.column_stack((-coords[:, 1], coords[:, 2]))
        depth = coords[:, 0]
        near_low = (hemisphere == "left" and view == "lateral") or (
            hemisphere == "right" and view == "medial"
        )
    triangle_ids = parcel[faces]
    face_ids = np.where(
        triangle_ids[:, 0] == triangle_ids[:, 1],
        triangle_ids[:, 0],
        np.where(
            triangle_ids[:, 1] == triangle_ids[:, 2],
            triangle_ids[:, 1],
            triangle_ids[:, 2],
        ),
    )
    order = np.argsort(depth[faces].mean(axis=1))
    if near_low:
        order = order[::-1]
    shade = sulc[faces].mean(axis=1)
    low, high = np.percentile(shade, (4, 96))
    neutral = np.clip(0.76 + 0.11 * (shade - low) / max(high - low, 1e-8), 0.70, 0.89)
    rgba = np.column_stack((neutral, neutral, neutral, np.ones_like(neutral)))
    selected = np.isin(face_ids, tuple(chosen))
    rgba[selected, :3] = 0.23 * rgba[selected, :3] + 0.77 * np.asarray(
        (0.686, 0.78, 0.949)
    )
    artist = PolyCollection(
        xy[faces[order]],
        facecolors=rgba[order],
        edgecolors=(0.32, 0.39, 0.48, 0.07),
        linewidths=0.065,
        antialiased=True,
        zorder=2,
    )
    ax.add_collection(artist)
    if autoscale:
        _fit_mesh_axes(ax, xy)
    return artist, face_ids[order], xy


def _fit_mesh_axes(ax, xy):
    x_range = max(float(np.ptp(xy[:, 0])), 1e-5)
    y_range = max(float(np.ptp(xy[:, 1])), 1e-5)
    ax.set_xlim(
        float(xy[:, 0].min()) - 0.045 * x_range, float(xy[:, 0].max()) + 0.045 * x_range
    )
    ax.set_ylim(
        float(xy[:, 1].min()) - 0.045 * y_range, float(xy[:, 1].max()) + 0.045 * y_range
    )
    ax.set_aspect("equal")
    ax.axis("off")


def _visible_positions(fig, surfaces, regions):
    originals = []
    for index, panel in enumerate(surfaces):
        artist = panel["artist"]
        originals.append(
            (artist, artist.get_facecolors().copy(), artist.get_linewidths().copy())
        )
        labels = np.asarray(panel["ids"], dtype=int)
        encoded = (index + 1) * 512 + labels + 1
        colors = np.zeros((len(labels), 4), dtype=float)
        colors[:, 0] = (encoded % 256) / 255
        colors[:, 1] = (encoded // 256) / 255
        colors[:, 3] = 1
        artist.set_facecolors(colors)
        artist.set_linewidth(0)
        artist.set_antialiased(False)
    try:
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        height, width = image.shape[:2]
        for index, panel in enumerate(surfaces):
            limits = panel["axes"].get_window_extent()
            x0, x1 = max(0, int(limits.x0)), min(width, int(np.ceil(limits.x1)))
            y0, y1 = (
                max(0, height - int(np.ceil(limits.y1))),
                min(height, height - int(limits.y0)),
            )
            image_part = image[y0:y1, x0:x1]
            selected = {}
            for rid in regions:
                code = (index + 1) * 512 + int(rid) + 1
                mask = (
                    (image_part[:, :, 0] == code % 256)
                    & (image_part[:, :, 1] == code // 256)
                    & (image_part[:, :, 2] == 0)
                )
                ys, xs = np.where(mask)
                if len(xs) >= 12:
                    left_band = xs <= np.quantile(xs, 0.20)
                    anchor_y = (
                        float(np.median(ys[left_band]))
                        if left_band.any()
                        else float(np.median(ys))
                    )
                    anchor_x = (
                        float(np.median(xs[left_band]))
                        if left_band.any()
                        else float(np.median(xs))
                    )
                    selected[int(rid)] = (
                        (x0 + anchor_x + 0.5) / width,
                        1 - (y0 + anchor_y + 0.5) / height,
                        int(len(xs)),
                    )
            panel["positions"] = selected
    finally:
        for artist, colors, widths in originals:
            artist.set_facecolors(colors)
            artist.set_linewidths(widths)
            artist.set_antialiased(True)
        fig.canvas.draw()


def _interaction_nodes(directory, features):
    import warnings
    from ..storage import read_table, table_exists

    selected = {f"brain_brainharmonix_f__{feature}" for feature, _, _ in features}
    recent = directory / "feature_importance.parquet"
    candidates = (
        directory / "ale_interactions.parquet",
        directory / "feature_interactions_current.parquet",
    )
    edges_by_pair = {}
    source = None
    feature_pattern = re.compile(r"(?:^|__)bhf_([0-9]{3})$")

    def normalized(value):
        name = str(value)
        match = feature_pattern.search(name)
        return (
            f"brain_brainharmonix_f__bhf_{int(match.group(1)):03d}" if match else name
        )

    for filename in candidates:
        if not table_exists(filename):
            continue
        if (
            recent.is_file()
            and filename.is_file()
            and filename.stat().st_mtime + 2 < recent.stat().st_mtime
        ):
            warnings.warn(
                f"Ignoring stale anatomical interaction table: {filename}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        table = read_table(filename)
        first = next(
            (key for key in ("feature_1", "feature1") if key in table.columns), None
        )
        second = next(
            (key for key in ("feature_2", "feature2") if key in table.columns), None
        )
        strength = next(
            (
                key
                for key in ("interaction_strength_mean", "interaction_strength")
                if key in table.columns
            ),
            None,
        )
        if first is None or second is None or strength is None:
            raise ValueError(f"Invalid interaction table schema: {filename}")
        for a, b, value in table[[first, second, strength]].itertuples(
            index=False, name=None
        ):
            if pd.isna(a) or pd.isna(b):
                continue
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            a, b = normalized(a), normalized(b)
            if not np.isfinite(weight) or weight < 0 or a == b:
                continue
            key = tuple(sorted((a, b)))
            edges_by_pair[key] = max(weight, edges_by_pair.get(key, -np.inf))
        source = filename
        break
    edges = [(a, b, w) for (a, b), w in edges_by_pair.items()]
    edges.sort(key=lambda edge: (-edge[2], edge[0], edge[1]))
    names = set(selected)
    for a, b, _ in edges:
        names.update((a, b))
    if not edges:
        warnings.warn(
            "No current ALE interaction table found. Anatomical feature-to-region links can be drawn, but a computed interaction network cannot be shown. Enable mll.ALEInteractions() and rerun the explain stage.",
            RuntimeWarning,
            stacklevel=2,
        )
    return sorted(names), edges, source


def _network_positions(names, features):
    selected_names = []
    for feature, _, _ in features:
        matching = [
            name for name in names if name == feature or name.endswith("__" + feature)
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Expected one graph node for selected embedding {feature}, found {len(matching)}"
            )
        selected_names.extend(matching)
    other_names = sorted(set(names) - set(selected_names))
    count = len(names)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    right_slots = sorted(
        range(count),
        key=lambda i: (abs(np.arctan2(np.sin(angles[i]), np.cos(angles[i]))), i),
    )[: len(selected_names)]
    right_slots = sorted(right_slots, key=lambda i: -np.sin(angles[i]))
    left_slots = sorted(
        set(range(count)) - set(right_slots), key=lambda i: -np.sin(angles[i])
    )
    assignment = list(zip(selected_names, right_slots)) + list(
        zip(other_names, left_slots)
    )
    return {
        name: (0.64 * np.cos(angles[i]), 0.64 * np.sin(angles[i]))
        for name, i in assignment
    }, selected_names


def _pack_labels(coords, names):
    top, bottom = 0.91, -0.91
    ordered = sorted(names, key=lambda name: -coords[name][1])
    if not ordered:
        return {}
    gap = min(0.094, (top - bottom) / max(len(ordered), 1))
    rows = []
    for index, name in enumerate(ordered):
        value = float(np.clip(coords[name][1], bottom, top))
        if index:
            value = min(value, rows[-1] - gap)
        rows.append(value)
    if rows[-1] < bottom:
        rows = [y + bottom - rows[-1] for y in rows]
    if rows[0] > top:
        rows = [y - (rows[0] - top) for y in rows]
    if len(rows) > 1 and rows[-1] < bottom - 0.005:
        raise ValueError(
            "Too many network feature labels for a legible anatomical figure"
        )
    return dict(zip(ordered, rows))


def render_anatomical_traceability(
    summary, features, provenance, directory, output, *, top_regions=2
):
    import json
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    from ..explainability_visuals import INK, MID, NET_EDGE_CMAP

    if top_regions < 1 or not features:
        raise ValueError(
            "At least one ranked embedding and anatomical region are required"
        )
    names, edges, network_source = _interaction_nodes(directory, features)
    network_coords, selected_names = _network_positions(names, features)
    regions_by_feature = {}
    chosen = set()
    for feature, dimension, _ in features:
        rows = (
            summary.loc[summary["feature"].eq(feature)]
            .sort_values(
                ["mean_absolute_delta", "region_index"], ascending=[False, True]
            )
            .head(top_regions)
        )
        indices = [
            (int(row.region_index), float(row.mean_absolute_delta))
            for row in rows.itertuples()
            if float(row.mean_absolute_delta) > 0
        ]
        regions_by_feature[feature] = indices
        chosen.update(rid for rid, _ in indices)
    if not chosen:
        raise ValueError(
            "All measured regional sensitivity scores are zero; no anatomical connections can be drawn"
        )
    meshes = _surface_atlas(provenance)
    figure = plt.figure(figsize=(180 / 25.4, 128 / 25.4), dpi=180, facecolor="white")
    network_ax = figure.add_axes([0.018, 0.13, 0.555, 0.83])
    network_ax.set_xlim(-1.16, 1.16)
    network_ax.set_ylim(-1.04, 1.04)
    network_ax.set_aspect("equal")
    network_ax.axis("off")
    maxima = max((w for _, _, w in edges), default=1.0)
    minima = min((w for _, _, w in edges), default=0.0)
    for a, b, value in sorted(edges, key=lambda row: row[2]):
        x1, y1 = network_coords[a]
        x2, y2 = network_coords[b]
        t = np.linspace(0, 1, 45)
        curve = 0.055 * np.sin(t * np.pi)
        x = x1 * (1 - t) + x2 * t + curve * (-y2 + y1)
        y = y1 * (1 - t) + y2 * t + curve * (x2 - x1)
        v = 0.5 if maxima == minima else (value - minima) / max(maxima - minima, 1e-12)
        network_ax.plot(
            x,
            y,
            lw=0.40 + 1.18 * v,
            color=NET_EDGE_CMAP(0.12 + 0.78 * v),
            alpha=0.84,
            solid_capstyle="round",
            zorder=1,
        )
    selected_to_node = {
        feature: next(
            name
            for name in selected_names
            if name == feature or name.endswith("__" + feature)
        )
        for feature, _, _ in features
    }
    left_names = [name for name in names if network_coords[name][0] < 0]
    right_names = [name for name in names if network_coords[name][0] >= 0]
    packed = {
        **_pack_labels(network_coords, left_names),
        **_pack_labels(network_coords, right_names),
    }
    label_artists = {}
    for name in names:
        x, y = network_coords[name]
        selected = name in selected_names
        network_ax.add_patch(
            plt.Circle(
                (x, y),
                0.033 if selected else 0.027,
                facecolor="#B6CAF3" if selected else "#DFE5EC",
                edgecolor=INK,
                linewidth=0.83,
                zorder=3,
            )
        )
        right = x >= 0
        rail = 0.81 if right else -0.81
        tip = rail - 0.012 if right else rail + 0.012
        network_ax.plot(
            [x + (0.036 if right else -0.036), tip],
            [y, packed[name]],
            color="#98ABC4",
            lw=0.47,
            ls=(0, (1.6, 2.3)),
            alpha=0.82,
            zorder=2,
        )
        feature_name = next(
            (f for f, node in selected_to_node.items() if node == name), None
        )
        display = feature_name if feature_name is not None else name.split("__")[-1]
        if len(display) > 27:
            display = display[:26] + "…"
        label_artists[name] = network_ax.text(
            rail,
            packed[name],
            display,
            ha="left" if right else "right",
            va="center",
            fontsize=5.9 if len(names) > 20 else 6.3,
            color=INK,
            zorder=4,
        )
    surfaces = []
    for row, view in enumerate(("lateral", "medial")):
        for col, hemisphere in enumerate(("left", "right")):
            axes = figure.add_axes(
                [0.635 + 0.18 * col, 0.665 - 0.29 * row, 0.174, 0.258]
            )
            artist, ids, xy = _view_collection(
                axes, meshes[hemisphere], hemisphere, view, chosen
            )
            surfaces.append(
                {
                    "axes": axes,
                    "artist": artist,
                    "ids": ids,
                    "hemisphere": hemisphere,
                    "view": view,
                }
            )
    dorsal = figure.add_axes([0.647, 0.055, 0.335, 0.275])
    projected = []
    for hemisphere in ("left", "right"):
        artist, ids, xy = _view_collection(
            dorsal, meshes[hemisphere], hemisphere, "superior", chosen, autoscale=False
        )
        surfaces.append(
            {
                "axes": dorsal,
                "artist": artist,
                "ids": ids,
                "hemisphere": hemisphere,
                "view": "superior",
            }
        )
        projected.append(xy)
    _fit_mesh_axes(dorsal, np.concatenate(projected, axis=0))
    figure.canvas.draw()
    _visible_positions(figure, surfaces, chosen)
    anchors = {}
    for region in chosen:
        hemisphere = (
            "left"
            if "_LH_" in provenance.labels[region]
            else "right"
            if "_RH_" in provenance.labels[region]
            else None
        )
        if hemisphere is None:
            raise ValueError(
                f"Cannot determine hemisphere of Schaefer parcel {region}: {provenance.labels[region]}"
            )
        candidates = [
            (panel["positions"][region][2], panel["positions"][region][:2], panel)
            for panel in surfaces
            if panel["hemisphere"] == hemisphere and region in panel["positions"]
        ]
        if not candidates:
            raise RuntimeError(
                f"Schaefer parcel {region} ({provenance.labels[region]}) has no verified visible pixels in any of the three views"
            )
        _, coords, chosen_panel = max(candidates, key=lambda item: item[0])
        anchors[region] = (coords, chosen_panel)
    renderer = figure.canvas.get_renderer()
    invert = figure.transFigure.inverted()
    rendered_links = []
    for feature, regions in regions_by_feature.items():
        label = label_artists[selected_to_node[feature]]
        bounds = label.get_window_extent(renderer)
        sx, sy = invert.transform((bounds.x1 + 3, (bounds.y0 + bounds.y1) / 2))
        for region, strength in regions:
            (tx, ty), panel = anchors[region]
            control_x = sx + 0.46 * (tx - sx)
            vertices = [(sx, sy), (control_x, sy), (control_x, ty), (tx, ty)]
            artist = PathPatch(
                MplPath(vertices, [MplPath.MOVETO, *([MplPath.CURVE4] * 3)]),
                transform=figure.transFigure,
                fill=False,
                linewidth=0.50,
                edgecolor="#7D9FCE",
                alpha=0.68,
                capstyle="round",
                zorder=20,
                clip_on=False,
            )
            figure.add_artist(artist)
            figure.add_artist(
                plt.Circle(
                    (tx, ty),
                    0.0030,
                    transform=figure.transFigure,
                    facecolor="#6D95D6",
                    edgecolor="white",
                    linewidth=0.20,
                    zorder=22,
                    clip_on=False,
                )
            )
            rendered_links.append(
                {
                    "feature": feature,
                    "region_index": region,
                    "region_name": provenance.labels[region],
                    "hemisphere": panel["hemisphere"],
                    "view": panel["view"],
                    "mean_absolute_delta": strength,
                    "anchor_figure_x": float(tx),
                    "anchor_figure_y": float(ty),
                }
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = (output.with_suffix(".svg"), output.with_suffix(".png"))
    old_svg = plt.rcParams["svg.fonttype"]
    plt.rcParams["svg.fonttype"] = "none"
    try:
        figure.savefig(
            paths[0], facecolor="white", bbox_inches="tight", pad_inches=0.045
        )
        figure.savefig(
            paths[1], facecolor="white", dpi=400, bbox_inches="tight", pad_inches=0.045
        )
    finally:
        plt.rcParams["svg.fonttype"] = old_svg
        plt.close(figure)
    (output.parent / "network_to_anatomy_links.tsv").write_text(
        pd.DataFrame(rendered_links).to_csv(sep="\t", index=False), encoding="utf-8"
    )
    (output.parent / "network_to_anatomy_network.json").write_text(
        json.dumps(
            {
                "interaction_table": str(network_source) if network_source else None,
                "interaction_edges": len(edges),
                "network_nodes": len(names),
                "features_traced": len(features),
                "region_links": len(rendered_links),
                "graph_complete_for_available_table": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def render_saved_anatomical_traceability(
    experiment_dir, *, target="mpma_b", top_regions_per_feature=2
):
    import json
    from .traceability import Schaefer400Provenance

    if top_regions_per_feature < 1:
        raise ValueError("top_regions_per_feature must be positive")
    _require_anatomical_visualization()
    directory = Path(experiment_dir) / "explainability" / target
    output = directory / "anatomical_traceability"
    files = {
        name: output / name
        for name in (
            "provenance.json",
            "selected_features.tsv",
            "feature_region_sensitivity.tsv",
        )
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved anatomical traceability inputs: {', '.join(missing)}"
        )
    metadata = json.loads(files["provenance.json"].read_text(encoding="utf-8"))
    if (
        metadata.get("schema") != "mllabiome.anatomical_traceability.v1"
        or metadata.get("method") != "roi_occlusion"
    ):
        raise ValueError("Saved anatomical traceability provenance is incompatible")
    atlas_path = Path(metadata["atlas_path"])
    if not atlas_path.is_file():
        raise FileNotFoundError(f"Original anatomical atlas is missing: {atlas_path}")
    summary = pd.read_csv(files["feature_region_sensitivity.tsv"], sep="\t")
    selected = pd.read_csv(files["selected_features.tsv"], sep="\t")
    required_summary = {
        "feature",
        "dimension",
        "region_index",
        "region_name",
        "mean_absolute_delta",
    }
    required_selected = {"feature", "dimension", "rank_score"}
    if not required_summary.issubset(summary.columns) or not required_selected.issubset(
        selected.columns
    ):
        raise ValueError("Saved anatomical sensitivity tables have incomplete schemas")
    features = [
        (str(row.feature), int(row.dimension), float(row.rank_score))
        for row in selected.itertuples()
    ]
    regions = {int(index) for index in metadata["region_indices"]}
    pairs = {(str(row.feature), int(row.region_index)) for row in summary.itertuples()}
    expected = {(feature, index) for feature, _, _ in features for index in regions}
    if (
        not features
        or not regions
        or len(pairs) != len(summary)
        or pairs != expected
        or summary["mean_absolute_delta"].isna().any()
        or not np.isfinite(summary["mean_absolute_delta"]).all()
        or (summary["mean_absolute_delta"] < 0).any()
        or len({feature for feature, _, _ in features}) != len(features)
        or len({dimension for _, dimension, _ in features}) != len(features)
        or not all(0 <= dimension < 768 for _, dimension, _ in features)
        or not all(0 <= index < 400 for index in regions)
    ):
        raise ValueError(
            "Saved anatomical sensitivity tables do not match their provenance"
        )
    provenance = Schaefer400Provenance.from_nilearn(atlas_path)
    if any(
        str(row.region_name) != provenance.labels[int(row.region_index)]
        for row in summary.itertuples()
    ):
        raise ValueError(
            "Saved Schaefer region labels do not match the anatomical provenance"
        )
    if any(feature != f"bhf_{dimension:03d}" for feature, dimension, _ in features):
        raise ValueError(
            "Saved embedding feature identities do not match their dimensions"
        )
    return render_anatomical_traceability(
        summary,
        features,
        provenance,
        directory,
        output / "network_to_anatomy",
        top_regions=top_regions_per_feature,
    )
