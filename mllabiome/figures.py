from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import metric_is_loss
from .storage import read_table
from .transformations import TRANSFORMATION_SPACE, transformation_label
from .utils import TAXONOMIC_LEVELS

_TRANSFORM_DISPLAY = {
    "identity": "identity",
    "relative_abundance": "RA",
    "presence_absence": "P/A",
    "hellinger": "Hellinger",
    "arcsine_sqrt": "arcsin√RA",
    "log10_relative_abundance_half_min_pseudocount": "log10(RA + ε)",
    "centered_log_ratio_multiplicative_replacement": "CLR",
    "additive_log_ratio_first_reference_multiplicative_replacement": "ALR",
    "isometric_log_ratio_egozcue_multiplicative_replacement": "ILR",
    "standardize": "z-score",
    "robust_scale": "Robust scale",
    "power_yeo_johnson": "YJ",
    "quantile_normal_numeric": "QN",
    "standardized_centered_log_ratio_multiplicative_replacement": "CLR + z",
    "yeo_johnson_relative_abundance": "YJ(RA)",
    "quantile_normal_relative_abundance": "QN(RA)",
    "robust_scaled_relative_abundance": "Robust(RA)",
    "within_sample_fractional_rank": "Fractional rank",
    "training_ecdf_rank": "ECDF rank",
    "prevalence_weighted_relative_abundance": "PWRA",
}


_COMPOSITION_BASES = {
    "relative_abundance",
    "hellinger",
    "arcsine_sqrt",
    "log10_relative_abundance_half_min_pseudocount",
    "centered_log_ratio_multiplicative_replacement",
    "additive_log_ratio_first_reference_multiplicative_replacement",
    "isometric_log_ratio_egozcue_multiplicative_replacement",
    "standardized_centered_log_ratio_multiplicative_replacement",
    "yeo_johnson_relative_abundance",
    "quantile_normal_relative_abundance",
    "robust_scaled_relative_abundance",
    "within_sample_fractional_rank",
    "prevalence_weighted_relative_abundance",
}


def _representation_metric_name(metric_col: str, df: pd.DataFrame) -> str:

    if metric_col in df.columns:
        return metric_col

    if f"{metric_col}_mean" in df.columns:
        return f"{metric_col}_mean"

    for candidate in ("log_loss", "AUROC", "AUROC_macro", "MCC", "BalAcc", "Accuracy"):
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "No supported metric column is available for the representation-impact figure."
    )


def _resolution_parts(value: Any) -> tuple[str, str, str] | None:

    text = str(value).strip().lower().replace("→", "->")

    pos = {r: i for i, r in enumerate(TAXONOMIC_LEVELS)}

    for sep in ("->", "-", "+"):
        if sep not in text:
            continue

        parts = [part.strip() for part in text.split(sep) if part.strip()]

        if len(parts) >= 2 and parts[0] in pos and parts[-1] in pos:
            return parts[0], parts[-1], sep

    return None


def _resolution_display(value: Any) -> str:

    text = str(value).strip()

    parsed = _resolution_parts(text)

    if parsed is None:
        return text

    left, right, sep = parsed

    if sep in {"->", "-"}:
        return f"{left}→{right}"

    return "+".join(part.strip() for part in text.split("+") if part.strip())


def _resolution_order_key(x: Any) -> tuple[int, int, int, str]:

    lab = str(x).strip().lower()

    pos = {r: i for i, r in enumerate(TAXONOMIC_LEVELS)}

    if lab in pos:
        return (0, pos[lab], -1, lab)

    parsed = _resolution_parts(lab)

    if parsed is not None:
        left, right, sep = parsed

        return (1 if sep in {"->", "-"} else 2, pos[left], pos[right], lab)

    return (9, 9, 9, lab)


def _taxonomic_block_count(levels: Any, resolution: Any, inferred: Any = None) -> int:

    if inferred is not None and not pd.isna(inferred):
        try:
            value = int(float(inferred))

        except (TypeError, ValueError):
            value = 0

        if value > 0:
            return value

    text = "" if pd.isna(levels) else str(levels).strip().lower()

    parts = tuple(x.strip() for x in text.split(",") if x.strip())

    resolved = tuple(x for x in parts if x in TAXONOMIC_LEVELS)

    if resolved:
        return len(dict.fromkeys(resolved))

    label = str(resolution).strip().lower()

    if label in TAXONOMIC_LEVELS:
        return 1

    parsed = _resolution_parts(label)

    if parsed is not None:
        left, right, sep = parsed

        if sep in {"->", "-"}:
            a, b = TAXONOMIC_LEVELS.index(left), TAXONOMIC_LEVELS.index(right)

            if a > b:
                a, b = b, a

            resolved_tokens = tuple(TAXONOMIC_LEVELS[a : b + 1])

        else:
            resolved_tokens = tuple(
                x.strip() for x in label.split("+") if x.strip() in TAXONOMIC_LEVELS
            )

    else:
        resolved_tokens = ()

    if resolved_tokens:
        return len(dict.fromkeys(resolved_tokens))

    return 0


def _split_transform_filter_identity(value: Any) -> tuple[str, str | None]:

    text = str(value).strip()

    marker = "|filter="

    if marker not in text:
        return text, None

    transformation, feature_filter = text.split(marker, 1)

    return transformation.strip(), feature_filter.strip() or None


def _parse_transform_identity(value: Any) -> tuple[str, str | None]:

    text, _ = _split_transform_filter_identity(value)

    if "@" in text:
        raw, scope = text.rsplit("@", 1)

        scope = scope.strip().lower()

        if scope in {"rank-wise", "joint"}:
            return transformation_label(raw).key, scope

    return transformation_label(text).key, None


def _canonical_representation_transform(value: Any, multirank: bool) -> str:

    base, scope = _parse_transform_identity(value)

    _, feature_filter = _split_transform_filter_identity(value)

    if base in _COMPOSITION_BASES and multirank:
        identity = f"{base}@{scope or 'rank-wise'}"

    else:
        identity = base

    if feature_filter is not None:
        identity = f"{identity}|filter={feature_filter}"

    return identity


def _filter_sort_key(value: Any) -> tuple[int, float, float, str]:

    _, feature_filter = _split_transform_filter_identity(value)

    if feature_filter is None:
        return (0, 0.0, 0.0, "")

    values = {}

    for token in feature_filter.split(","):
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        try:
            values[key.strip()] = float(raw)
        except ValueError:
            continue

    return (
        1,
        float(values.get("prevalence", float("inf"))),
        float(values.get("detection", float("inf"))),
        feature_filter,
    )


def _transform_sort_key(
    value: Any,
) -> tuple[int, int, tuple[int, float, float, str], str]:

    base, scope = _parse_transform_identity(value)

    order = {item.key: i for i, item in enumerate(TRANSFORMATION_SPACE)}

    scope_order = {None: 0, "rank-wise": 1, "joint": 2}

    return (
        order.get(base, len(order)),
        scope_order.get(scope, 3),
        _filter_sort_key(value),
        str(value),
    )


def _filter_display(value: Any) -> str | None:

    _, feature_filter = _split_transform_filter_identity(value)

    if feature_filter is None:
        return None

    values = {}

    for token in feature_filter.split(","):
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        try:
            values[key.strip()] = float(raw)
        except ValueError:
            continue

    prevalence = values.get("prevalence")

    detection = values.get("detection")

    if prevalence is None:
        return feature_filter

    label = f"prev≥{prevalence * 100:g}%"

    if detection is not None and detection > 0.0:
        label = f"{label}, d>{detection:g}"

    return label


def _transform_display(value: Any, include_scope: bool) -> str:

    text = str(value).strip()

    modality_parts = text.split("+")

    if len(modality_parts) > 1 and all(":" in part for part in modality_parts):
        rendered = []
        for part in modality_parts:
            modality, transformation = part.split(":", 1)
            label = _transform_display(transformation, include_scope).replace("\n", " ")
            rendered.append(f"{modality.strip()}: {label}")
        return "\n".join(rendered)

    base, scope = _parse_transform_identity(value)

    label = _TRANSFORM_DISPLAY.get(base, base.replace("_", " ").strip().title())

    parts = [label]

    if include_scope and scope in {"rank-wise", "joint"}:
        parts.append(f"({scope})")

    feature_filter = _filter_display(value)

    if feature_filter is not None:
        parts.append(feature_filter)

    return "\n".join(parts)


def _semantic_outer_results(df: pd.DataFrame, metric: str) -> pd.DataFrame:

    out = df.copy()

    if "levels" not in out.columns:
        out["levels"] = out["resolution"]

    if "learner" not in out.columns:
        out["learner"] = "learner"

    if "split_key" not in out.columns:
        out["split_key"] = np.arange(len(out)).astype(str)

    inferred_counts = (
        out["taxonomic_block_count"]
        if "taxonomic_block_count" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )

    out["_block_count"] = [
        _taxonomic_block_count(levels, resolution, inferred)
        for levels, resolution, inferred in zip(
            out["levels"], out["resolution"], inferred_counts, strict=False
        )
    ]

    unresolved_raw = out["resolution"].astype(str).str.casefold().isin(
        {"raw", "all"}
    ) & out["_block_count"].eq(0)

    if unresolved_raw.any():
        for resolution in out.loc[unresolved_raw, "resolution"].astype(str).unique():
            mask = unresolved_raw & out["resolution"].astype(str).eq(resolution)

            identities = [
                _parse_transform_identity(value)
                for value in out.loc[mask, "count_transformation"].astype(str)
            ]

            scopes_by_base: dict[str, set[str]] = {}

            for base, scope in identities:
                if scope in {"rank-wise", "joint"}:
                    scopes_by_base.setdefault(base, set()).add(scope)

            inferred_multirank = any(
                "joint" in scopes for scopes in scopes_by_base.values()
            )

            out.loc[mask, "_block_count"] = 2 if inferred_multirank else 1

    out["_multirank"] = out["_block_count"].gt(1)

    out["_canonical_transform"] = [
        _canonical_representation_transform(value, bool(multirank))
        for value, multirank in zip(
            out["count_transformation"], out["_multirank"], strict=False
        )
    ]

    keys = ["split_key", "resolution", "_canonical_transform", "learner"]

    grouped = (
        out.groupby(keys, dropna=False, as_index=False)
        .agg(
            **{
                metric: (metric, "mean"),
                "levels": ("levels", "first"),
                "_block_count": ("_block_count", "max"),
                "_multirank": ("_multirank", "max"),
            }
        )
        .reset_index(drop=True)
    )

    return grouped


def _least_squares_sse(y: np.ndarray, X: np.ndarray) -> float:

    if X.shape[1] == 0:
        resid = y - float(np.mean(y))

        return float(np.dot(resid, resid))

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ coef

    return float(np.dot(resid, resid))


def _interaction_key(left: str, right: str) -> str:

    return f"{left}×{right}"


def _partial_eta2_blocked(
    df: pd.DataFrame,
    metric: str,
    factors: tuple[str, ...],
    block: str,
    interactions: tuple[tuple[str, str], ...] = (),
) -> dict[str, float]:

    cols = list(dict.fromkeys([metric, block, *factors]))

    d = df[cols].dropna().copy()

    terms = [*factors, *(_interaction_key(*pair) for pair in interactions)]

    if d.empty:
        return {term: np.nan for term in terms}

    y = d[metric].astype(float).to_numpy()

    pieces = [np.ones((len(d), 1), dtype=float)]

    term_columns: dict[str, np.ndarray] = {}

    block_dummies = pd.get_dummies(
        d[block].astype(str), prefix="block", drop_first=True, dtype=float
    ).to_numpy(dtype=float)

    if block_dummies.shape[1]:
        pieces.append(block_dummies)

    offset = sum(piece.shape[1] for piece in pieces)

    factor_designs: dict[str, np.ndarray] = {}

    for factor in factors:
        design = pd.get_dummies(
            d[factor].astype(str), prefix=factor, drop_first=True, dtype=float
        ).to_numpy(dtype=float)

        factor_designs[factor] = design

        term_columns[factor] = np.arange(offset, offset + design.shape[1])

        if design.shape[1]:
            pieces.append(design)

        offset += design.shape[1]

    for left, right in interactions:
        left_design = factor_designs[left]
        right_design = factor_designs[right]

        if left_design.shape[1] and right_design.shape[1]:
            design = (left_design[:, :, None] * right_design[:, None, :]).reshape(
                len(d), -1
            )
        else:
            design = np.empty((len(d), 0), dtype=float)

        key = _interaction_key(left, right)

        term_columns[key] = np.arange(offset, offset + design.shape[1])

        if design.shape[1]:
            pieces.append(design)

        offset += design.shape[1]

    X = np.column_stack(pieces)

    rank_full = int(np.linalg.matrix_rank(X))

    sse_full = _least_squares_sse(y, X)

    values: dict[str, float] = {}

    interaction_keys = {pair: _interaction_key(*pair) for pair in interactions}

    for term in terms:
        remove = [term_columns[term]]

        if term in factors:
            remove.extend(
                term_columns[key]
                for pair, key in interaction_keys.items()
                if term in pair
            )

        target = (
            np.unique(np.concatenate([columns for columns in remove if columns.size]))
            if any(columns.size for columns in remove)
            else np.empty(0, dtype=int)
        )

        if not target.size:
            values[term] = np.nan

            continue

        keep = np.ones(X.shape[1], dtype=bool)

        keep[target] = False

        X_reduced = X[:, keep]

        if rank_full <= int(np.linalg.matrix_rank(X_reduced)):
            values[term] = np.nan

            continue

        sse_reduced = _least_squares_sse(y, X_reduced)

        ss_effect = max(0.0, sse_reduced - sse_full)

        denom = ss_effect + sse_full

        values[term] = np.nan if denom <= 1e-12 else float(ss_effect / denom)

    return values


def _config_performance(df: pd.DataFrame, metric: str) -> pd.DataFrame:

    keys = ["resolution", "_canonical_transform", "learner"]

    return (
        df.groupby(keys, dropna=False, as_index=False)
        .agg(
            **{
                metric: (metric, "mean"),
                "_block_count": ("_block_count", "max"),
                "_multirank": ("_multirank", "max"),
            }
        )
        .reset_index(drop=True)
    )


def _heat_table(
    config_perf: pd.DataFrame,
    metric: str,
    multirank: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:

    d = config_perf[config_perf["_multirank"].eq(multirank)].copy()

    if d.empty:
        return pd.DataFrame(), [], []

    if multirank:
        d["_heat_transform"] = d["_canonical_transform"]

    else:
        d["_heat_transform"] = d["_canonical_transform"].map(
            lambda value: _canonical_representation_transform(value, False)
        )

    heat = (
        d.groupby(["resolution", "_heat_transform"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    rows = sorted(heat["resolution"].astype(str).unique(), key=_resolution_order_key)

    cols = sorted(heat["_heat_transform"].astype(str).unique(), key=_transform_sort_key)

    pivot = heat.pivot_table(
        index="resolution",
        columns="_heat_transform",
        values=metric,
        aggfunc="mean",
    ).reindex(index=rows, columns=cols)

    labels = [_transform_display(value, include_scope=multirank) for value in cols]

    return pivot, rows, labels


def _write_representation_impact_figure(
    root: Path, metric_col: str = "log_loss"
) -> None:

    result_path = root / "results" / "outer_results.parquet"

    if not result_path.exists() or result_path.stat().st_size == 0:
        return

    df = read_table(result_path)

    if df.empty or "ok" not in df.columns:
        return

    configs_path = root / "configs.parquet"

    configs_db = root / "configs.db"

    configs = pd.DataFrame()

    if configs_path.exists():
        configs = read_table(configs_path)

    elif configs_db.exists():
        conn = sqlite3.connect(str(configs_db))

        try:
            configs = pd.read_sql_query("SELECT * FROM configs", conn)

        finally:
            conn.close()

    if not configs.empty and "config_id" in df.columns:
        metadata_columns = [
            column
            for column in (
                "config_id",
                "taxonomic_blocks",
                "taxonomic_block_count",
                "representation_scope",
            )
            if column in configs.columns
        ]

        if "config_id" in metadata_columns:
            metadata = configs[metadata_columns].drop_duplicates("config_id")

            df = df.merge(
                metadata, on="config_id", how="left", suffixes=("", "_config")
            )

    df = df[df["ok"].eq(1)].copy()

    if df.empty:
        return

    metric = _representation_metric_name(metric_col, df)

    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df[np.isfinite(df[metric])].copy()

    if df.empty:
        return

    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.transforms as mtransforms

    from .style import (
        C_DARK,
        C_MID,
        C_SKY,
        COL_W_2,
        HMAP_CMAP,
        NATURE_PANEL_LABEL_PT,
        NATURE_TEXT_PT,
        REPRESENTATION_RC,
        save_svg,
    )

    mpl.rcParams.update(REPRESENTATION_RC)

    is_modality_sweep = "candidate_family" in df.columns or "modalities" in df.columns

    representation_label = (
        "Modality representation" if is_modality_sweep else "Taxonomic representation"
    )

    learner_label = "Learner" if is_modality_sweep else "Learner"

    transformation_label_text = "Transformation"

    if is_modality_sweep:
        if "split_key" not in df.columns:
            df["split_key"] = np.arange(len(df)).astype(str)

        if "learner" not in df.columns:
            df["learner"] = "learner"

        df["_canonical_transform"] = df["count_transformation"].astype(str)

        df["_multirank"] = False

        df["_block_count"] = 1

        semantic = df.groupby(
            ["split_key", "resolution", "_canonical_transform", "learner"],
            dropna=False,
            as_index=False,
        )[metric].mean()

        semantic["_multirank"] = False

        semantic["_block_count"] = 1

    else:
        semantic = _semantic_outer_results(df, metric)

    config_perf = _config_performance(semantic, metric)

    res_order = sorted(
        config_perf["resolution"].astype(str).unique(), key=_resolution_order_key
    )

    single_pivot, single_rows, single_labels = _heat_table(config_perf, metric, False)

    multi_pivot, multi_rows, multi_labels = _heat_table(config_perf, metric, True)

    if is_modality_sweep:
        single_pivot = (
            config_perf.groupby(["resolution", "_canonical_transform"], dropna=False)[
                metric
            ]
            .mean()
            .reset_index()
            .pivot_table(
                index="resolution",
                columns="_canonical_transform",
                values=metric,
                aggfunc="mean",
            )
        )

        single_rows = sorted(single_pivot.index.astype(str), key=_resolution_order_key)

        single_pivot = single_pivot.reindex(index=single_rows)

        single_cols = sorted(single_pivot.columns.astype(str), key=_transform_sort_key)

        single_pivot = single_pivot.reindex(columns=single_cols)

        single_labels = [_transform_display(value, False) for value in single_cols]

        multi_pivot = pd.DataFrame()

        multi_rows = []

        multi_labels = []

    matrices = [
        pivot.to_numpy(dtype=float)
        for pivot in (single_pivot, multi_pivot)
        if not pivot.empty
    ]

    finite_values = [matrix[np.isfinite(matrix)] for matrix in matrices]

    finite_values = [values for values in finite_values if values.size]

    if not finite_values:
        return

    all_values = np.concatenate(finite_values)

    vmin = float(np.min(all_values))

    vmax = float(np.max(all_values))

    if abs(vmax - vmin) < 1e-12:
        pad = max(abs(vmin) * 0.01, 0.01)

        vmin -= pad

        vmax += pad

    bottom_rows = max(len(single_rows), len(multi_rows), 1)

    height_mm = min(150.0, max(114.0, 88.0 + 4.3 * bottom_rows))

    fig = plt.figure(figsize=(COL_W_2, height_mm / 25.4), facecolor="white")

    outer = gridspec.GridSpec(
        2,
        1,
        height_ratios=[0.82, max(1.0, 0.9 + bottom_rows * 0.055)],
        hspace=0.70,
        left=0.068,
        right=0.965,
        bottom=0.225,
        top=0.925,
        figure=fig,
    )

    top = outer[0].subgridspec(
        1,
        3,
        width_ratios=[3.15, 0.72, 1.13],
        wspace=0.02,
    )

    ax_a = fig.add_subplot(top[0, 0])

    ax_b = fig.add_subplot(top[0, 2])

    if is_modality_sweep or multi_pivot.empty:
        bottom = outer[1].subgridspec(1, 1)

        ax_c = fig.add_subplot(bottom[0, 0])

        ax_d = None

    elif single_pivot.empty:
        bottom = outer[1].subgridspec(1, 1)

        ax_c = None

        ax_d = fig.add_subplot(bottom[0, 0])

    else:
        bottom = outer[1].subgridspec(
            1,
            3,
            width_ratios=[2.0, 0.82, 3.15],
            wspace=0.04,
        )

        ax_c = fig.add_subplot(bottom[0, 0])

        ax_d = fig.add_subplot(bottom[0, 2])

    panel_heading_y = 1.055

    panel_label_gap_pt = 7.0

    def tag(ax, label):

        transform = ax.transAxes + mtransforms.ScaledTranslation(
            -panel_label_gap_pt / 72.0,
            0.0,
            fig.dpi_scale_trans,
        )

        ax.text(
            0.0,
            panel_heading_y,
            label,
            transform=transform,
            fontsize=NATURE_PANEL_LABEL_PT,
            fontweight="bold",
            fontstyle="normal",
            va="baseline",
            ha="right",
            clip_on=False,
        )

    def set_panel_title(ax, title):

        ax.text(
            0.0,
            panel_heading_y,
            title,
            transform=ax.transAxes,
            fontsize=NATURE_TEXT_PT,
            fontweight="bold",
            fontstyle="normal",
            va="baseline",
            ha="left",
            clip_on=False,
        )

    def trim(ax):

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

    split_perf = (
        semantic.groupby(["resolution", "split_key"], dropna=False, as_index=False)[
            metric
        ]
        .mean()
        .reset_index(drop=True)
    )

    split_perf["resolution"] = split_perf["resolution"].astype(str)

    xs = np.arange(len(res_order), dtype=float)

    box_values = []

    box_positions = []

    point_values = []

    for xi, resolution in enumerate(res_order):
        values = (
            split_perf.loc[split_perf["resolution"].eq(str(resolution)), metric]
            .astype(float)
            .to_numpy()
        )

        values = values[np.isfinite(values)]

        if not values.size:
            continue

        box_values.append(values)

        box_positions.append(float(xi))

        point_values.append((xi, values))

    if box_values:
        ax_a.boxplot(
            box_values,
            positions=box_positions,
            widths=0.14,
            whis=1.5,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": C_DARK, "linewidth": 0.45},
            medianprops={"color": C_DARK, "linewidth": 0.62},
            whiskerprops={"color": C_DARK, "linewidth": 0.45},
            capprops={"color": C_DARK, "linewidth": 0.45},
            capwidths=0.055,
            zorder=2,
        )

    for xi, values in point_values:
        offsets = (
            np.zeros(1) if values.size == 1 else np.linspace(-0.045, 0.045, values.size)
        )

        ax_a.scatter(
            xi + offsets,
            values,
            s=4.2,
            color=C_SKY,
            alpha=0.62,
            linewidths=0,
            zorder=3,
        )

    ax_a.set_xticks(xs)

    rotation = 55 if len(res_order) > 12 else 42

    ax_a.set_xticklabels(
        [_resolution_display(x) for x in res_order],
        rotation=rotation,
        ha="right",
        rotation_mode="anchor",
    )

    ax_a.set_xlabel(representation_label, labelpad=4)

    ax_a.set_ylabel(metric.replace("_", "-"), labelpad=4)

    ax_a.margins(x=0.025)

    set_panel_title(ax_a, f"Performance by {representation_label.lower()}")

    tag(ax_a, "a")

    trim(ax_a)

    factors = ("resolution", "_canonical_transform", "learner")

    interactions = (
        ("resolution", "learner"),
        ("_canonical_transform", "learner"),
    )

    effects = _partial_eta2_blocked(
        semantic,
        metric,
        factors,
        "split_key",
        interactions,
    )

    rep_interaction = _interaction_key("resolution", "learner")

    transform_interaction = _interaction_key("_canonical_transform", "learner")

    effect_terms = [
        "resolution",
        "_canonical_transform",
        "learner",
        rep_interaction,
        transform_interaction,
    ]

    factor_labels = [
        "Taxonomy" if not is_modality_sweep else representation_label,
        transformation_label_text,
        learner_label,
        "Taxonomy × learner" if not is_modality_sweep else "Representation × learner",
        "Transformation × learner",
    ]

    eta_vals = np.asarray(
        [effects.get(term, np.nan) for term in effect_terms],
        dtype=float,
    )

    plot_vals = np.where(np.isfinite(eta_vals), eta_vals, 0.0)

    ypos = np.array([0.0, 1.0, 2.0, 3.35, 4.35])

    bars = ax_b.barh(
        ypos,
        plot_vals,
        color=[C_DARK, C_MID, C_SKY, C_DARK, C_MID],
        height=0.58,
        zorder=3,
    )

    ax_b.set_yticks(ypos)

    ax_b.set_yticklabels(factor_labels)

    ax_b.set_xlim(0, 1)

    ax_b.xaxis.set_major_locator(mticker.MultipleLocator(0.25))

    ax_b.set_xlabel("Split-adjusted partial η²", labelpad=4)

    ax_b.invert_yaxis()

    for bar, value in zip(bars, eta_vals, strict=False):
        y = bar.get_y() + bar.get_height() / 2

        if not np.isfinite(value):
            ax_b.text(
                0.025,
                y,
                "n/a",
                ha="left",
                va="center",
                fontsize=NATURE_TEXT_PT,
                color="#64748b",
            )

            continue

        label = "<0.01" if 0.0 < value < 0.01 else f"{value:.2f}"

        if value >= 0.82:
            ax_b.text(
                max(0.02, value - 0.03),
                y,
                label,
                ha="right",
                va="center",
                fontsize=NATURE_TEXT_PT,
                color="white",
            )

        else:
            ax_b.text(
                min(0.96, value + 0.025),
                y,
                label,
                ha="left",
                va="center",
                fontsize=NATURE_TEXT_PT,
                color="#0f172a",
            )

    set_panel_title(ax_b, "Hierarchical factor effects")

    tag(ax_b, "b")

    trim(ax_b)

    cmap = HMAP_CMAP.copy()

    cmap.set_bad("#f1f5f9")

    if metric_is_loss(metric):
        cmap = cmap.reversed()

        cmap.set_bad("#f1f5f9")

    def draw_heatmap(ax, pivot, rows, labels, panel, xlabel, title):

        if ax is None or pivot.empty:
            return None

        mat = pivot.to_numpy(dtype=float)

        image = ax.imshow(
            np.ma.masked_invalid(mat),
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        span = max(vmax - vmin, 1e-12)

        ndp = 3 if span < 0.08 else 2

        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                value = mat[ri, ci]

                if not np.isfinite(value):
                    continue

                normalized = (value - vmin) / span

                color = "white" if normalized > 0.58 else "#0f172a"

                ax.text(
                    ci,
                    ri,
                    f"{value:.{ndp}f}",
                    ha="center",
                    va="center",
                    fontsize=NATURE_TEXT_PT,
                    color=color,
                )

        ax.set_yticks(np.arange(len(rows)))

        ax.set_yticklabels([_resolution_display(row) for row in rows])

        ax.set_xticks(np.arange(len(labels)))

        rotation = 58 if len(labels) > 8 else 45

        ax.set_xticklabels(
            labels,
            rotation=rotation,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )

        ax.tick_params(
            axis="x",
            which="both",
            top=False,
            bottom=True,
            labeltop=False,
            labelbottom=True,
            length=0,
            pad=3,
        )

        ax.tick_params(axis="y", which="both", length=0, pad=3)

        ax.set_xlabel(xlabel, labelpad=5)

        ax.set_ylabel("")

        set_panel_title(ax, title)

        tag(ax, panel)

        for spine in ax.spines.values():
            spine.set_visible(False)

        return image

    image_c = draw_heatmap(
        ax_c,
        single_pivot,
        single_rows,
        single_labels,
        "c",
        "Transformation",
        "Single-rank representations"
        if not is_modality_sweep
        else representation_label,
    )

    image_d = draw_heatmap(
        ax_d,
        multi_pivot,
        multi_rows,
        multi_labels,
        "d",
        "Transformation and scope",
        "Multi-rank representations",
    )

    out_base = root / "figures" / "representation_impact"

    out_base.parent.mkdir(parents=True, exist_ok=True)

    save_svg(fig, out_base.with_suffix(".svg"), dpi=300)

    plt.close(fig)
