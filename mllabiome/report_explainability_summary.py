from __future__ import annotations

import base64
import copy
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .explainability_visuals import (
    plot_feature_support,
    plot_interaction_network,
    plot_local_attributions,
)
from .report_explainability import (
    _read_json,
    _read_table,
    _target_dirs_by_label,
    _xai_class_slug,
    _xai_target_metadata,
)


_COMPARATORS = ("MPMA-B", "Baseline RF")
_SECTIONS = (
    ("a", "Feature importance and stability across explanation methods", "support"),
    (
        "b",
        "Interaction network of features based on accumulated local effects",
        "interactions",
    ),
    ("c", "Sample-level feature contributions to predictions", "local"),
)
_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", _SVG_NS)
ET.register_namespace("xlink", _XLINK_NS)


def _class_label_for_pair(directories: dict[str, Path]) -> str | None:
    common: set[str] | None = None
    first_order: list[str] = []
    for label in _COMPARATORS:
        _methods, classes = _xai_target_metadata(directories[label])
        labels = [str(class_label) for _, class_label in classes]
        if label == _COMPARATORS[0]:
            first_order = labels
        common = set(labels) if common is None else common.intersection(labels)
    return next((label for label in first_order if label in (common or set())), None)


def _figure_paths(
    directories: dict[str, Path], class_label: str
) -> dict[str, dict[str, Path | None]]:
    slug = _xai_class_slug(class_label)
    paths: dict[str, dict[str, Path | None]] = {}
    for label, target_dir in directories.items():
        figures = target_dir / "figures"
        candidate = {
            "support": figures / f"feature_support__{slug}.svg",
            "interactions": figures / f"interaction_network_current__{slug}.svg",
            "local": figures / "local_explanations.svg",
            "ale_curves": figures / f"ale_curves__{slug}.svg",
        }
        paths[label] = {
            key: path if path.is_file() else None for key, path in candidate.items()
        }
    return paths


def _compact_figure_paths(
    directories: dict[str, Path],
    class_label: str,
    report_dir: Path,
) -> dict[str, dict[str, Path]]:
    import pandas as pd

    slug = _xai_class_slug(class_label)
    output: dict[str, dict[str, Path]] = {}
    for label in _COMPARATORS:
        target = directories[label]
        stem_root = (
            report_dir
            / "compact_explainability_assets"
            / ("mpma_b" if label == "MPMA-B" else "baseline_rf")
        )
        stem_root.mkdir(parents=True, exist_ok=True)
        top = _read_table(target / "top_features.parquet")
        if "class_label" in top.columns:
            top = top[top["class_label"].astype(str).eq(class_label)].copy()
        elif "class_index" in top.columns:
            meta = _read_json(target / "explained_unit.json")
            pairs = zip(
                meta.get("explained_class_indices", []),
                meta.get("explained_class_labels", []),
            )
            matching = [int(i) for i, name in pairs if str(name) == class_label]
            if matching:
                top = top[
                    pd.to_numeric(top["class_index"], errors="coerce").isin(matching)
                ].copy()
        top = (
            top.sort_values("rank", ascending=True).head(8)
            if "rank" in top.columns
            else top.head(8)
        )
        stats = _read_table(target / "feature_distribution_stats.parquet")
        stability = _read_table(target / "feature_stability.parquet")
        support_stem = stem_root / f"support__{slug}"
        if top.empty or not plot_feature_support(
            top,
            stats,
            support_stem,
            8,
            ["Control", class_label],
            stability,
            compact=True,
        ):
            raise ValueError(f"Compact support results unavailable for {label}")
        interactions = _read_table(target / "feature_interactions_current.parquet")
        if "class_label" in interactions.columns:
            interactions = interactions[
                interactions["class_label"].astype(str).eq(class_label)
            ].copy()
        elif (
            "class_index" in interactions.columns
            and "class_index" in top.columns
            and not top.empty
        ):
            idx = pd.to_numeric(top["class_index"], errors="coerce").dropna()
            if not idx.empty:
                interactions = interactions[
                    pd.to_numeric(interactions["class_index"], errors="coerce").eq(
                        int(idx.iloc[0])
                    )
                ].copy()
        if "interaction_strength" in interactions.columns:
            interactions = interactions.sort_values(
                "interaction_strength", ascending=False
            ).head(9)
        net_stem = stem_root / f"interactions__{slug}"
        if interactions.empty or not plot_interaction_network(
            interactions, stats, net_stem, 9, ["Control", class_label], compact=True
        ):
            raise ValueError(f"Compact interaction results unavailable for {label}")
        local = _read_table(target / "local_explanations.parquet")
        if "class_label" in local.columns:
            local = local[local["class_label"].astype(str).eq(class_label)].copy()
        local_stem = stem_root / f"local__{slug}"
        cohort = _read_table(target / "local_cohort_context.parquet")
        result = plot_local_attributions(
            local,
            local_stem,
            task="classification",
            top_n=4,
            cohort_context=cohort,
            compact=True,
        )
        if result is None:
            raise ValueError(f"Compact sample explanations unavailable for {label}")
        output[label] = {
            "support": support_stem.with_suffix(".svg"),
            "interactions": net_stem.with_suffix(".svg"),
            "local": local_stem.with_suffix(".svg"),
        }
    return output


def _svg_tag(name: str) -> str:
    return f"{{{_SVG_NS}}}{name}"


def _svg_text(root: ET.Element, x: float, y: float, value: str, size: float) -> None:
    element = ET.SubElement(
        root,
        _svg_tag("text"),
        {
            "x": f"{x:.4f}",
            "y": f"{y:.4f}",
            "font-size": f"{size:.1f}",
            "font-family": "Arial, Helvetica, sans-serif",
            "font-weight": "700",
            "fill": "#121c2e",
        },
    )
    element.text = value


def _fit_local_header_text(source_root: ET.Element, width: float) -> None:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    for element in source_root.iter():
        if element.tag != _svg_tag("text"):
            continue
        value = "".join(element.itertext()).strip()
        if not value.startswith(("Observed:", "Out-of-fold:")):
            continue
        x = float(element.attrib.get("x", "0"))
        available = width - x - 8.0
        if available <= 0:
            continue
        style = element.attrib.get("style", "")
        match = re.search(r"font-size:\s*([0-9.]+)px", style)
        if match is None:
            continue
        size = float(match.group(1))
        weight = (
            "bold" if re.search(r"font-weight:\s*(?:bold|700)", style) else "normal"
        )
        font = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        measured = TextPath((0, 0), value, prop=font).get_extents().width
        if measured <= available:
            continue
        size = max(5.0, min(size, size * available / measured))
        style = style[: match.start(1)] + f"{size:.2f}" + style[match.end(1) :]
        element.set("style", style)
        font.set_size(size)
        measured = TextPath((0, 0), value, prop=font).get_extents().width
        if measured > available:
            element.set("textLength", f"{available:.3f}")
            element.set("lengthAdjust", "spacingAndGlyphs")


def _nest_svg(
    root: ET.Element,
    source: Path,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
    prefix: str,
) -> None:
    source_root = ET.parse(source).getroot()
    if source_root.tag != _svg_tag("svg"):
        raise ValueError(f"Not an SVG figure: {source}")
    coordinates = re.split(r"[\s,]+", source_root.attrib.get("viewBox", "").strip())
    if len(coordinates) != 4:
        raise ValueError(f"Missing SVG viewBox: {source}")
    sx, sy, sw, sh = (float(value) for value in coordinates)
    if sw <= 0 or sh <= 0:
        raise ValueError(f"Invalid SVG dimensions: {source}")
    if "local__" in source.name:
        _fit_local_header_text(source_root, sx + sw)
    scale = min(max_width / sw, max_height / sh, 1.0)
    nested = ET.SubElement(
        root,
        _svg_tag("svg"),
        {
            "x": f"{x:.4f}",
            "y": f"{y:.4f}",
            "width": f"{sw * scale:.4f}",
            "height": f"{sh * scale:.4f}",
            "viewBox": f"{sx:g} {sy:g} {sw:g} {sh:g}",
            "preserveAspectRatio": "xMinYMin meet",
        },
    )
    ids = {}
    children = [copy.deepcopy(child) for child in source_root]
    for element in children:
        for child in element.iter():
            identifier = child.attrib.get("id")
            if identifier:
                ids[identifier] = f"{prefix}_{identifier}"
                child.set("id", ids[identifier])
    for element in children:
        for child in element.iter():
            for key, value in tuple(child.attrib.items()):
                if "url(#" in value:
                    value = re.sub(
                        r"url\(#([^)]+)\)",
                        lambda match: (
                            f"url(#{ids.get(match.group(1), match.group(1))})"
                        ),
                        value,
                    )
                if value.startswith("#") and value[1:] in ids:
                    value = f"#{ids[value[1:]]}"
                child.set(key, value)
            if child.tag == _svg_tag("style") and child.text:
                child.text = re.sub(
                    r"url\(#([^)]+)\)",
                    lambda match: f"url(#{ids.get(match.group(1), match.group(1))})",
                    child.text,
                )
        nested.append(element)


def _write_native_composite(
    svg_path: Path, figures: dict[str, dict[str, Path]]
) -> None:
    mm = 72.0 / 25.4
    width = 183.0 * mm
    height = 170.0 * mm
    margin = 1.0 * mm
    gap = 3.0 * mm
    column_width = (width - 2.0 * margin - gap) / 2.0
    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": "183mm",
            "height": "170mm",
            "viewBox": f"0 0 {width:.4f} {height:.4f}",
            "version": "1.1",
            "role": "img",
            "aria-label": "MPMA-B versus Baseline RF explainability comparison",
        },
    )
    ET.SubElement(
        root,
        _svg_tag("rect"),
        {
            "x": "0",
            "y": "0",
            "width": f"{width:.4f}",
            "height": f"{height:.4f}",
            "fill": "#ffffff",
        },
    )
    rows = (
        ("support", 2.0, 39.0),
        ("interactions", 51.0, 42.0),
        ("local", 103.0, 59.0),
    )
    for index, ((letter, title, key), (_, top_mm, image_mm)) in enumerate(
        zip(_SECTIONS, rows)
    ):
        top = top_mm * mm
        _svg_text(root, margin, top + 7.0, letter, 8.1)
        _svg_text(root, margin + 10.0, top + 7.0, title, 7.2)
        for column, label in enumerate(_COMPARATORS):
            x = margin + column * (column_width + gap)
            _svg_text(root, x + 2.0, top + 17.3, label, 6.2)
            dest_y = top + 21.0
            _nest_svg(
                root,
                figures[label][key],
                x,
                dest_y,
                column_width,
                image_mm * mm,
                f"p{index}_{column}",
            )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)


def _svg_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def _composite_embed(path: Path, report_dir: Path) -> str:
    uri = _svg_data_uri(path)
    return (
        '<figure class="xai-svg-figure">'
        f'<a href="{uri}" download="explainability_composite.svg" aria-label="Download publication figure">'
        f'<img class="xai-svg-embed" src="{uri}" alt="MPMA-B versus Baseline RF explainability comparison" loading="lazy">'
        "</a></figure>"
        '<div class="xai-svg-download">'
        f'<a href="{uri}" download="explainability_composite.svg">Download publication figure (SVG)</a>'
        "</div>"
    )


def _detailed_gallery(
    figures: dict[str, dict[str, Path | None]], report_dir: Path
) -> str:
    sections = (
        *_SECTIONS,
        ("d", "Feature-specific effects on predicted class probability", "ale_curves"),
    )
    parts = ['<div class="xai-detailed-figures">']
    for _, heading, kind in sections:
        cells = []
        for label in _COMPARATORS:
            path = figures.get(label, {}).get(kind)
            if path is None:
                continue
            uri = _svg_data_uri(path)
            caption = html.escape(f"{label}: {heading}", quote=True)
            filename = html.escape(path.name, quote=True)
            cells.append(
                '<div class="compare-cell">'
                f"<h4>{html.escape(label)}</h4>"
                f'<a href="{uri}" download="{filename}" aria-label="Download {caption}">'
                f'<img class="xai-detailed-svg" src="{uri}" alt="{caption}" loading="lazy"></a>'
                f'<p><a href="{uri}" download="{filename}">Download SVG</a></p>'
                "</div>"
            )
        if cells:
            parts.append(f"<h4>{html.escape(heading)}</h4>")
            parts.append('<div class="compare-grid" style="--compare-columns:2">')
            parts.extend(cells)
            parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def explainability_summary_block(root: Path, report_dir: Path) -> str:
    report_dir = Path(report_dir)
    composite = report_dir / "explainability_composite.svg"
    old_outputs = (
        "explainability_summary.pdf",
        "explainability_summary.png",
        "explainability_composite.pdf",
        "explainability_composite.png",
        "explainability_print_width.pdf",
    )
    for name in old_outputs:
        (report_dir / name).unlink(missing_ok=True)
    available = _target_dirs_by_label(Path(root))
    if not available:
        composite.unlink(missing_ok=True)
        return "<p>No explainability figures are available yet.</p>"
    if all(label in available for label in _COMPARATORS):
        directories = {label: available[label] for label in _COMPARATORS}
        class_label = _class_label_for_pair(directories)
    else:
        directories = {
            label: available[label] for label in _COMPARATORS if label in available
        }
        first = next(iter(directories.values()))
        _methods, classes = _xai_target_metadata(first)
        class_label = str(classes[0][1]) if classes else None
    if class_label is None:
        composite.unlink(missing_ok=True)
        return (
            "<p>No shared class-specific explainability figures are available yet.</p>"
        )
    figures = _figure_paths(directories, class_label)
    gallery = _detailed_gallery(figures, report_dir)
    if len(directories) == len(_COMPARATORS):
        try:
            compact_figures = _compact_figure_paths(
                directories, class_label, report_dir
            )
            _write_native_composite(composite, compact_figures)
        except (ImportError, OSError, RuntimeError, ValueError, KeyError):
            composite.unlink(missing_ok=True)
    else:
        composite.unlink(missing_ok=True)
    return gallery
