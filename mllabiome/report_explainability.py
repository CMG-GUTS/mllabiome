from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .explainability_visuals import plot_feature_support, plot_local_attributions
from .storage import glob_tables, read_table, table_exists
from .utils import feature_tail_ellipsis as feature_tail_ellipsis


def _asset_uri(path: Path) -> str:

    suffix = path.suffix.lower()

    mime = {".png": "image/png", ".pdf": "application/pdf"}.get(
        suffix, "application/octet-stream"
    )

    payload = base64.b64encode(path.read_bytes()).decode("ascii")

    return f"data:{mime};base64,{payload}"


def _inline_svg(path: Path) -> str:

    text = path.read_text(encoding="utf-8")

    start = text.find("<svg")

    end = text.rfind("</svg>")

    if start < 0 or end < 0:
        return ""

    text = text[start : end + 6]

    text = re.sub(r"<metadata\b[^>]*>.*?</metadata>", "", text, flags=re.DOTALL)

    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    text = re.sub(r">\s+<", "><", text)

    return text.strip()


def _fig(stem_or_path: Path, report_dir: Path, caption: str = "") -> str:

    path = stem_or_path

    if path.suffix.lower() not in {".svg", ".png", ".pdf"}:
        path = stem_or_path.with_suffix(".svg")

    if not path.exists():
        png = path.with_suffix(".png")

        if png.exists():
            path = png

        else:
            return ""

    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""

    alt = html.escape(caption or path.stem)

    if path.suffix.lower() == ".svg":
        svg = _inline_svg(path)

        if not svg:
            return ""

        return f'<figure><div class="embedded-svg" role="img" aria-label="{alt}">{svg}</div>{cap}</figure>'

    uri = _asset_uri(path)

    if path.suffix.lower() == ".png":
        return f'<figure><img src="{uri}" alt="{alt}" loading="lazy">{cap}</figure>'

    return f'<figure><p><a href="{uri}">{alt}</a></p>{cap}</figure>'


def _read_table(path: Path) -> pd.DataFrame:

    try:
        return read_table(path)

    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))

    except Exception:
        return {}


def _target_dirs_by_label(root: Path) -> dict[str, Path]:

    exp_root = root / "explainability"

    if not exp_root.exists():
        return {}

    aliases = (
        ("MPMA-E", ("mpma_e", "ensemble", "best_mpmas_ensemble", "mpma-e")),
        ("MPMA-B", ("mpma_b", "best_individual", "best_mpma", "mpma-b")),
        ("Baseline RF", ("baseline_rf", "baseline", "rf_baseline", "baseline-rf")),
    )

    out: dict[str, Path] = {}

    for label, slugs in aliases:
        for slug in slugs:
            d = exp_root / slug

            if d.exists() and d.is_dir():
                out[label] = d

                break

    return out


def _xai_class_slug(label: Any) -> str:

    text = str(label).strip().lower()

    out = "".join(ch if ch.isalnum() else "_" for ch in text)

    while "__" in out:
        out = out.replace("__", "_")

    return out.strip("_") or "class"


def _xai_method_display(method: str) -> str:

    return {
        "shap": "SHAP",
        "permutation": "Permutation",
        "ale": "ALE",
        "lime": "LIME",
        "interactions": "ALE interactions",
    }.get(str(method).strip().lower(), str(method))


def _xai_coordinate_column(target_dir: Path) -> str:

    table = _read_table(target_dir / "coordinate_metadata.parquet")

    if table.empty or "exact_feature_identity" not in table.columns:
        return "Feature"

    values = table["exact_feature_identity"].astype(str).str.strip().str.lower()

    exact = values.isin({"true", "1", "yes"})

    return "Model coordinate" if bool((~exact).any()) else "Feature"


def _xai_target_metadata(target_dir: Path) -> tuple[list[str], list[tuple[int, str]]]:

    meta = _read_json(target_dir / "explained_unit.json")

    methods = [
        str(x).strip().lower() for x in meta.get("methods", []) if str(x).strip()
    ]

    for path in glob_tables(target_dir, "feature_stability_*"):
        name = path.stem.removeprefix("feature_stability_").strip().lower()

        if name and name not in methods:
            methods.append(name)

    if (
        table_exists(target_dir / "feature_interactions_current.parquet")
        and "interactions" not in methods
    ):
        methods.append("interactions")

    indices = list(meta.get("explained_class_indices", []))

    labels = list(meta.get("explained_class_labels", []))

    classes: list[tuple[int, str]] = []

    for pos, label in enumerate(labels):
        try:
            idx = int(indices[pos]) if pos < len(indices) else int(pos)

        except Exception:
            idx = int(pos)

        classes.append((idx, str(label)))

    if not classes:
        for path in [
            target_dir / "feature_stability.parquet",
            target_dir / "top_features.parquet",
        ]:
            tab = _read_table(path)

            if tab.empty or "class_index" not in tab.columns:
                continue

            lab_col = "class_label" if "class_label" in tab.columns else None

            seen: set[int] = set()

            for _, row in tab.sort_values("class_index").iterrows():
                try:
                    idx = int(row.get("class_index"))

                except Exception:
                    continue

                if idx in seen:
                    continue

                seen.add(idx)

                label = (
                    str(row.get(lab_col, f"class_{idx}")) if lab_col else f"class_{idx}"
                )

                classes.append((idx, label))

            if classes:
                break

    return methods, classes


def _refresh_xai_support_figures(target_dir: Path, fallback_top_k: int) -> None:

    top = _read_table(target_dir / "top_features.parquet")

    if top.empty or "feature" not in top.columns:
        return

    meta = _read_json(target_dir / "explained_unit.json")

    try:
        top_k = max(1, int(meta.get("top_k", fallback_top_k)))

    except Exception:
        top_k = max(1, int(fallback_top_k))

    stats = _read_table(target_dir / "feature_distribution_stats.parquet")

    stability = _read_table(target_dir / "feature_stability.parquet")

    figures = target_dir / "figures"

    figures.mkdir(parents=True, exist_ok=True)

    if "class_index" in top.columns:
        groups = top.groupby("class_index", sort=True)

    else:
        groups = [(0, top)]

    for class_index, class_top in groups:
        label = (
            str(class_top["class_label"].dropna().iloc[0])
            if "class_label" in class_top.columns
            and not class_top["class_label"].dropna().empty
            else f"class_{int(class_index)}"
        )

        class_stats = stats

        if not stats.empty and "class_index" in stats.columns:
            idx = pd.to_numeric(stats["class_index"], errors="coerce")

            class_stats = stats[idx.eq(int(class_index))].copy()

        plot_feature_support(
            class_top,
            class_stats,
            figures / f"feature_support__{_xai_class_slug(label)}",
            top_k,
            [label],
            stability,
        )

    methods = [
        str(x).strip().lower()
        for x in meta.get("methods", [])
        if str(x).strip() and str(x).strip().lower() != "interactions"
    ]

    for method in methods:
        method_top = _read_table(target_dir / f"top_features_{method}.parquet")

        if method_top.empty or "feature" not in method_top.columns:
            continue

        method_stability = _read_table(
            target_dir / f"feature_stability_{method}.parquet"
        )

        if "class_index" in method_top.columns:
            method_groups = method_top.groupby("class_index", sort=True)

        else:
            method_groups = [(0, method_top)]

        for class_index, class_top in method_groups:
            label = (
                str(class_top["class_label"].dropna().iloc[0])
                if "class_label" in class_top.columns
                and not class_top["class_label"].dropna().empty
                else f"class_{int(class_index)}"
            )

            slug = _xai_class_slug(label)

            class_stats = _read_table(
                target_dir / f"feature_distribution_stats_{method}__{slug}.parquet"
            )

            plot_feature_support(
                class_top,
                class_stats,
                figures / f"feature_importance_{method}__{slug}",
                top_k,
                [label],
                method_stability,
            )


def _xai_figure_for_class(
    target_dir: Path, stem: str, class_label: str, report_dir: Path, caption: str
) -> str:

    slug = _xai_class_slug(class_label)

    return _fig(target_dir / "figures" / f"{stem}__{slug}", report_dir, caption)


def _xai_method_global_text(method: str) -> str:

    return {
        "shap": "Aggregated absolute out-of-fold SHAP attribution across held-out samples for the class probability. Signed local SHAP values are retained separately for local explanations and sign-stability summaries.",
        "permutation": "Out-of-fold predictive importance measured by degradation in class-specific loss after disrupting one feature in held-out data.",
        "ale": "Global accumulated local effects for the class probability. Importance summarizes the magnitude of the centered ALE effect; curves show effect shape across the observed feature distribution.",
        "lime": "Aggregated absolute out-of-fold LIME local-surrogate coefficients across held-out samples. Signed local coefficients are retained separately when local explanations are requested.",
    }.get(str(method).strip().lower(), "Global out-of-fold feature explanation.")


def _xai_local_mode(target_dir: Path) -> str:

    meta = _read_json(target_dir / "explained_unit.json")

    cfg = meta.get("explainability_config", {}) if isinstance(meta, dict) else {}

    value = str(cfg.get("local_explanations", "")).strip()

    if value:
        return value

    representative = bool(cfg.get("representative_instances", False))

    requested = bool(cfg.get("instance_sample_ids", []))

    if representative and requested:
        return "representative_and_requested"

    if representative:
        return "representative"

    if requested:
        return "requested"

    return "none"


def _xai_local_figure(target_dir: Path, report_dir: Path, label: str, task: str) -> str:

    table = _read_table(target_dir / "local_explanations.parquet")

    if table.empty or "method" not in table.columns:
        return ""

    methods = {str(x).strip().lower() for x in table["method"].dropna().tolist()}

    available = [name for name in ("shap", "lime") if name in methods]

    if not available:
        return ""

    meta = _read_json(target_dir / "explained_unit.json")

    cfg = meta.get("explainability_config", {}) if isinstance(meta, dict) else {}

    local_cfg = cfg.get("local", {}) if isinstance(cfg, dict) else {}

    displayed = local_cfg.get(
        "displayed_features",
        meta.get("top_instance_features", cfg.get("top_instance_features", 8))
        if isinstance(meta, dict)
        else 8,
    )

    try:
        top_n = max(1, int(displayed))

    except (TypeError, ValueError):
        top_n = 8

    stem = target_dir / "figures" / "local_explanations"

    context = _read_table(target_dir / "local_cohort_context.parquet")

    plot_local_attributions(
        table,
        stem,
        task=str(task),
        top_n=top_n,
        cohort_context=context,
    )

    if len(available) == 2:
        caption = f"{label}: representative out-of-fold local SHAP, LIME, and cross-method explanations"

    else:
        caption = f"{label}: representative out-of-fold local {available[0].upper()} explanations"

    return _fig(stem, report_dir, caption)


def _explainability_report_blocks(
    root: Path, report_dir: Path, top_n: int = 15
) -> tuple[str, int]:

    target_dirs = _target_dirs_by_label(root)

    if not target_dirs:
        return "", 0

    parts: list[str] = []

    blocks = 0

    for label in ("MPMA-B", "MPMA-E", "Baseline RF"):
        target_dir = target_dirs.get(label)

        if target_dir is None:
            continue

        methods, classes = _xai_target_metadata(target_dir)

        if not methods and not classes:
            continue

        blocks += 1

        method_text = (
            ", ".join(_xai_method_display(x) for x in methods)
            if methods
            else "available methods"
        )

        _refresh_xai_support_figures(target_dir, top_n)

        local_mode = _xai_local_mode(target_dir)

        coordinate_mode = _xai_coordinate_column(target_dir) == "Model coordinate"

        unit_singular = "model coordinate" if coordinate_mode else "feature"

        unit_plural = "model coordinates" if coordinate_mode else "features"

        parts.append(f'<section class="xai-target"><h3>{html.escape(label)}</h3>')

        parts.append(
            f"<p>Cross-fitted out-of-fold explanations of the final selected specification. Methods: {html.escape(method_text)}. Global explanation with fold stability and local sample explanation are reported as distinct layers.</p>"
        )

        parts.append("<h4>Global explanations</h4>")

        if len(classes) == 1:
            target_class = html.escape(str(classes[0][1]))

            parts.append(
                f"<p>Global results summarize held-out predictions across samples. SHAP and LIME global importance are aggregations of local out-of-fold attributions. Permutation importance and ALE are population-level quantities by construction. The reported units are {html.escape(unit_plural)}. The attribution target is {target_class}.</p>"
            )

        else:
            parts.append(
                f"<p>Global results summarize held-out predictions across samples. SHAP and LIME global importance are aggregations of local out-of-fold attributions. Permutation importance and ALE are population-level quantities by construction. The reported units are {html.escape(unit_plural)}.</p>"
            )

        for class_index, class_label in classes:
            if len(classes) == 1:
                parts.append('<section class="xai-class">')

            else:
                parts.append(
                    f'<section class="xai-class"><h5>Target class: {html.escape(class_label)}</h5>'
                )

            consensus_fig = _xai_figure_for_class(
                target_dir,
                "feature_support",
                class_label,
                report_dir,
                f"{label} · {class_label}: cross-method top-k support and fold top-k frequency",
            )

            if consensus_fig:
                support_methods = [x for x in methods if x != "interactions"]

                if len(support_methods) <= 1:
                    parts.append("<h6>Top-k support and fold top-k frequency</h6>")

                    parts.append(
                        f"<p>Top-k support is a within-method rank score for each {html.escape(unit_singular)}. Rank 1 scores 1, rank k scores 1/k, and ranks below k score 0. Fold top-k frequency is the fraction of estimable outer folds in which that {html.escape(unit_singular)} ranks within the method-specific top k. Both quantities are scale-free from 0 to 1.</p>"
                    )

                else:
                    parts.append(
                        "<h6>Cross-method support and fold top-k frequency</h6>"
                    )

                    parts.append(
                        f"<p>Top-k support is a within-method rank score for each {html.escape(unit_singular)}. Rank 1 scores 1, rank k scores 1/k, and ranks below k score 0. Mean support is the arithmetic mean of the available method-specific top-k support scores. Fold top-k frequency is the fraction of estimable outer folds in which that {html.escape(unit_singular)} ranks within the method-specific top k. These are scale-free 0 to 1 summaries. Method-specific effect magnitudes are not compared across methods.</p>"
                    )

                parts.append(consensus_fig)

            for method in methods:
                if method == "interactions":
                    continue

                display = _xai_method_display(method)

                global_fig = _xai_figure_for_class(
                    target_dir,
                    f"feature_importance_{method}",
                    class_label,
                    report_dir,
                    f"{label} · {class_label}: global {display} explanation",
                )

                ale_curve = (
                    _xai_figure_for_class(
                        target_dir,
                        "ale_curves",
                        class_label,
                        report_dir,
                        f"{label} · {class_label}: ALE effect curves",
                    )
                    if method == "ale"
                    else ""
                )

                if not global_fig and not ale_curve:
                    continue

                parts.append(f"<h6>{html.escape(display)}</h6>")

                parts.append(f"<p>{html.escape(_xai_method_global_text(method))}</p>")

                if global_fig:
                    parts.append(global_fig)

                if ale_curve:
                    parts.append(
                        f"<p>Thin pale lines show the outer-fold ALE curves. When multiple displayed outer-fold curves have a shared feature-value range, the thick blue line shows their pointwise median only over that common-support range and the shaded band shows the corresponding interquartile range across folds. Individual fold curves may extend beyond the common-support range. If no shared range exists, no cross-fold median or interquartile band is drawn. If only one ALE curve is available, it is shown without an interquartile band. The dashed horizontal line marks zero centered ALE effect: values above or below it indicate feature regions associated with higher or lower predicted P({html.escape(class_label)}), respectively. ALE describes model behavior and should not be interpreted as a causal effect.</p>"
                    )

                    parts.append(ale_curve)

            if "interactions" in methods:
                interaction_figs_class = []

                for stem, caption in [
                    ("interaction_network_current", "2D ALE interaction network"),
                ]:
                    block = _xai_figure_for_class(
                        target_dir,
                        stem,
                        class_label,
                        report_dir,
                        f"{label} · {class_label}: {caption}",
                    )

                    if block:
                        interaction_figs_class.append(block)

                if interaction_figs_class:
                    parts.append("<h6>ALE interactions</h6>")

                    parts.append(
                        "<p>Exploratory class-specific out-of-fold 2D ALE interaction strengths. Edge width and shade represent interaction magnitude. Node area encodes cohort mean relative abundance on a log scale; for log-ratio and balance coordinates, abundance is the absolute-coefficient-weighted mean relative abundance of the underlying component taxa. Node colour represents the standardized target-versus-reference shift in the model coordinate.</p>"
                    )

                    parts.extend(interaction_figs_class)

            parts.append("</section>")

        interaction_figs = []

        for stem, caption in [
            ("interaction_network_current", "2D ALE interaction network"),
        ]:
            block = _fig(
                target_dir / "figures" / stem, report_dir, f"{label}: {caption}"
            )

            if block:
                interaction_figs.append(block)

        if interaction_figs:
            parts.append("<h5>Global interactions</h5>")

            parts.append(
                "<p>Interaction outputs are exploratory population-level 2D ALE summaries and are not local sample explanations.</p>"
            )

            parts.extend(interaction_figs)

        parts.append("<h4>Local explanations</h4>")

        if local_mode == "none":
            parts.append(
                "<p>Local sample-level reporting was not requested for this run.</p>"
            )

        else:
            local_figure = _xai_local_figure(
                target_dir, report_dir, label, "classification"
            )

            if local_figure:
                parts.append(
                    "<p>Representative held-out samples are selected from out-of-fold predictions, with one representative per observed class. The cohort relative abundance panel shows the raw cohort relative abundance of each displayed taxon, with samples colored by observed class and the explained sample outlined. Each sample is explained only by outer-fold models that did not train on that sample. SHAP attributions and LIME local-surrogate coefficients are shown side by side when both are available. Positive contributions are colored by the explained class. Negative contributions use the alternate class color for binary tasks and a neutral comparison color for multiclass tasks. The cross-method panel combines attribution direction with within-method reciprocal-rank support and does not average raw SHAP and LIME magnitudes. The support count indicates how many local methods place the feature within the displayed top set.</p>"
                )

                parts.append(local_figure)

                parts.append(
                    "<p>Sample-wise local attribution results are retained in <code>local_explanations.parquet</code>.</p>"
                )

            else:
                parts.append(
                    "<p>No representative local SHAP/LIME visualization is available for this explained unit.</p>"
                )

        parts.append("</section>")

    return "".join(parts), blocks
