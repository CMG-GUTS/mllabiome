from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .storage import read_table
from . import report as _report_module
from .console import console, path_table, stage, success
from .final_models import load_final_models

_PERCENT_METRICS = {
    "AUC",
    "AUC_macro",
    "AUC_weighted",
    "PR_AUC",
    "PR_AUC_macro",
    "nMCC",
    "Accuracy",
    "BalAcc",
    "F1",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
}
_PRIMARY_METRIC_ORDER = (
    "AUC",
    "PR_AUC",
    "nMCC",
    "F1w",
    "Precision",
    "Recall",
    "BalAcc",
    "Accuracy",
)
_MULTICLASS_METRIC_ORDER = (
    "AUC_macro",
    "AUC_weighted",
    "PR_AUC_macro",
    "F1_macro",
)
_PROBABILITY_METRIC_ORDER = (
    "Brier",
    "Brier_multiclass",
    "LogLoss",
)
_CONTRAST_METRIC_ORDER = (
    "AUC",
    "AUC_macro",
    "AUC_weighted",
    "PR_AUC",
    "PR_AUC_macro",
    "nMCC",
    "F1w",
    "F1_macro",
    "Precision",
    "Recall",
    "BalAcc",
    "Accuracy",
    "Brier",
    "Brier_multiclass",
    "LogLoss",
)
_CALIBRATION_METRIC_ORDER = (
    "CalibrationInTheLarge",
    "CalibrationIntercept",
    "CalibrationSlope",
)
_METRIC_LABELS = {
    "AUC": "ROC-AUC",
    "AUC_macro": "ROC-AUC macro",
    "AUC_weighted": "ROC-AUC weighted",
    "PR_AUC": "PR-AUC (AP)",
    "PR_AUC_macro": "PR-AUC macro",
    "nMCC": "nMCC",
    "Accuracy": "Accuracy",
    "BalAcc": "Balanced accuracy",
    "F1": "F1",
    "F1w": "F1w",
    "F1_macro": "F1 macro",
    "Precision": "Precision",
    "Recall": "Recall",
    "Brier": "Brier score",
    "Brier_multiclass": "Multiclass Brier score",
    "LogLoss": "Log loss",
    "CalibrationInTheLarge": "Calibration-in-the-large",
    "CalibrationIntercept": "Calibration intercept",
    "CalibrationSlope": "Calibration slope",
}
_ESTIMAND_LABELS = {
    "mean_repeat_pooled_oof": "Pooled OOF across outer folds; mean across repeats",
    "pooled_sample_weighted": "Pooled OOF, sample-weighted",
    "cohort_macro_equal_weight": "Pooled OOF, equal-cohort weighting",
}
_SECTION_START = '<section id="oof-inference">'
_SECTION_END = "</section>"
_STATISTICS_ANCHOR = '<h3 id="statistics">Statistical comparisons</h3>'
_COMPUTE_ANCHOR = '<h3 id="compute">Computational resources</h3>'
_NAV_ANCHOR = '<a href="#statistics">Statistics</a>'
_NAV_LINK = '<a href="#oof-performance">OOF inference</a>'
_FOOTER_ANCHOR = '<p class="report-footer">'
_PROCEDURE_ANCHOR = '<h2 id="procedure" class="first-section">Evaluation procedure</h2>'
_PERFORMANCE_ANCHOR = '<h2 id="performance">Task performance summary</h2>'
_MPMA_B_SECTION_START = '<section id="mpma-b-composition">'
_MPMA_B_SECTION_END = "</section>"


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def _ensemble_summary_from_selected(root: Path) -> pd.DataFrame:
    selected = _read_json(root / "ensembling" / "selected_unit.json")
    ens = (
        selected.get("inner_val_best_mpmas_ensemble")
        or selected.get("inner_val_best_ensemble")
        or {}
    )
    if not isinstance(ens, dict) or not ens:
        return pd.DataFrame()
    members = _parse_list(ens.get("members", ens.get("member_config_ids")))
    weights = _parse_list(ens.get("weights"))
    effective = 0
    if weights and len(weights) == len(members):
        for weight in weights:
            value = _safe_float(weight)
            if np.isfinite(value) and value > 1e-12:
                effective += 1
    if effective == 0:
        value = _safe_float(ens.get("effective_member_count"))
        if np.isfinite(value) and value > 0:
            effective = int(value)
    if effective == 0:
        value = _safe_float(ens.get("member_count"))
        if np.isfinite(value) and value > 0:
            effective = int(value)
    if effective == 0:
        value = _safe_float(ens.get("ensemble_size"))
        if np.isfinite(value) and value > 0:
            effective = int(value)
    if effective == 0:
        effective = len(members)
    metric = str(ens.get("selection_metric", ens.get("optimize_metric", ""))).strip()
    row = {
        "Strategy": "MPMA-E",
        "Selection": str(ens.get("selection_strategy", ens.get("method", ""))).strip(),
        "Aggregation": str(ens.get("aggregation_strategy", "")).strip(),
        "Max size": int(_safe_float(ens.get("max_size")))
        if np.isfinite(_safe_float(ens.get("max_size")))
        else int(effective),
        "Members": int(effective),
        "Selection metric": metric,
    }
    return pd.DataFrame([row])


def _format_ci_cell(row: pd.Series, metric: str) -> str:
    estimate = _safe_float(row.get("estimate"))
    low = _safe_float(row.get("ci_low"))
    high = _safe_float(row.get("ci_high"))
    if not np.isfinite(estimate):
        return "—"
    if metric in _PERCENT_METRICS:
        body = f"{100.0 * estimate:.2f}"
        if np.isfinite(low) and np.isfinite(high):
            body += f" [{100.0 * low:.2f}, {100.0 * high:.2f}]"
        return body
    body = f"{estimate:.4f}"
    if np.isfinite(low) and np.isfinite(high):
        body += f" [{low:.4f}, {high:.4f}]"
    return body


def _wide_metric_table(
    performance: pd.DataFrame, metric_order: tuple[str, ...]
) -> pd.DataFrame:
    required = {"Strategy", "estimand", "metric", "estimate", "ci_low", "ci_high"}
    if performance.empty or not required.issubset(performance.columns):
        return pd.DataFrame()
    available = set(performance["metric"].astype(str))
    metrics = [metric for metric in metric_order if metric in available]
    if not metrics:
        return pd.DataFrame()
    estimands = [
        str(x)
        for x in performance["estimand"].dropna().astype(str).unique().tolist()
        if str(x)
    ]
    show_estimand = len(set(estimands)) > 1
    rows: list[dict[str, Any]] = []
    for (strategy, estimand), group in performance.groupby(
        ["Strategy", "estimand"], sort=False
    ):
        row: dict[str, Any] = {"Strategy": str(strategy)}
        if show_estimand:
            row["Estimand"] = _ESTIMAND_LABELS.get(str(estimand), str(estimand))
        for metric in metrics:
            sub = group[group["metric"].astype(str).eq(metric)]
            label = _METRIC_LABELS.get(metric, metric)
            row[label] = "—" if sub.empty else _format_ci_cell(sub.iloc[0], metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _performance_display(performance: pd.DataFrame) -> pd.DataFrame:
    return _wide_metric_table(performance, _PRIMARY_METRIC_ORDER)


def _multiclass_display(
    performance: pd.DataFrame, n_classes: int | None
) -> pd.DataFrame:
    if n_classes is None or int(n_classes) <= 2:
        return pd.DataFrame()
    return _wide_metric_table(performance, _MULTICLASS_METRIC_ORDER)


def _probability_display(
    performance: pd.DataFrame, n_classes: int | None
) -> pd.DataFrame:
    if n_classes is not None and int(n_classes) <= 2:
        order = ("Brier", "LogLoss")
    elif n_classes is not None and int(n_classes) > 2:
        order = ("Brier_multiclass", "LogLoss")
    else:
        order = _PROBABILITY_METRIC_ORDER
    return _wide_metric_table(performance, order)


def _calibration_display(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty or "metric" not in performance.columns:
        return pd.DataFrame()
    calibration = performance[
        performance["metric"].astype(str).isin(_CALIBRATION_METRIC_ORDER)
    ].copy()
    return _wide_metric_table(calibration, _CALIBRATION_METRIC_ORDER)


def _html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    cols = list(df.columns)
    parts = ['<div class="table-wrap"><table><thead><tr>']
    parts.extend(f"<th>{html.escape(str(c))}</th>" for c in cols)
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        parts.append("<tr>")
        for col in cols:
            value = row.get(col, "")
            text = "" if pd.isna(value) else html.escape(str(value))
            parts.append(f"<td>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _methodology_html(manifest: dict[str, Any]) -> str:
    protocol = str(manifest.get("protocol", "")).strip().lower()
    n_bootstrap = manifest.get("n_bootstrap", "")
    try:
        bootstrap = f"{int(n_bootstrap):,}"
    except (TypeError, ValueError):
        bootstrap = ""
    if protocol in {"lodo", "leave_one_dataset_out"}:
        method = "two-stage cohort-and-subject bootstrap"
        estimand = "sample-weighted and equal-cohort held-out performance"
    else:
        design = manifest.get("observed_oof_design", {})
        repeats = []
        if isinstance(design, dict):
            for value in design.values():
                if isinstance(value, dict):
                    try:
                        repeats.append(int(value.get("n_repeats")))
                    except (TypeError, ValueError):
                        pass
        method = (
            "subject-cluster bootstrap"
            if repeats and max(repeats) == 1
            else "subject-cluster and repeat bootstrap"
        )
        estimand = "pooled outer-fold out-of-fold performance"
    suffix = f" using {bootstrap} replicates" if bootstrap else ""
    return (
        f"<p>Estimates summarize {html.escape(estimand)}. "
        f"Uncertainty is reported as 95% percentile bootstrap confidence intervals "
        f"from a {html.escape(method)}{html.escape(suffix)}.</p>"
    )


def _probability_semantics_html(manifest: dict[str, Any]) -> str:
    semantics = manifest.get("probability_semantics", {})
    if not isinstance(semantics, dict):
        return ""
    excluded: list[str] = []
    for strategy, info in semantics.items():
        if not isinstance(info, dict) or bool(info.get("valid", False)):
            continue
        reason = str(info.get("reason", "")).lower()
        if "rank_mean" in reason:
            detail = "rank aggregation"
        elif "siamcat" in reason:
            detail = "score-valued output"
        elif "ridge" in reason:
            detail = "decision-score output"
        else:
            detail = "non-probability output"
        excluded.append(
            f"<strong>{html.escape(str(strategy))}</strong> ({html.escape(detail)})"
        )
    if not excluded:
        return ""
    return (
        "<p>Brier score, log loss, and calibration summaries are shown for "
        "probability-valued predictions. Probability-based summaries are not shown for "
        + "; ".join(excluded)
        + ".</p>"
    )


def _format_contrast_value(value: Any, metric: str) -> str:
    v = _safe_float(value)
    if not np.isfinite(v):
        return "—"
    if metric in _PERCENT_METRICS:
        return f"{100.0 * v:.2f}"
    return f"{v:.4f}"


def _contrast_display(
    contrasts: pd.DataFrame, n_classes: int | None = None
) -> pd.DataFrame:
    required = {
        "strategy_a",
        "strategy_b",
        "estimand",
        "metric",
        "estimate_a",
        "estimate_b",
        "advantage_a_over_b",
        "advantage_ci_low",
        "advantage_ci_high",
    }
    if contrasts.empty or not required.issubset(contrasts.columns):
        return pd.DataFrame()
    shown = contrasts.copy()
    metric_order = list(_CONTRAST_METRIC_ORDER)
    if n_classes is not None and int(n_classes) <= 2:
        excluded = {
            "AUC_macro",
            "AUC_weighted",
            "PR_AUC_macro",
            "F1_macro",
            "Brier_multiclass",
        }
        metric_order = [metric for metric in metric_order if metric not in excluded]
    metric_rank = {metric: index for index, metric in enumerate(metric_order)}
    shown = shown[shown["metric"].astype(str).isin(metric_order)].copy()
    shown["_metric_rank"] = shown["metric"].astype(str).map(metric_rank).fillna(999)
    shown = shown.sort_values(
        ["_metric_rank", "strategy_a", "strategy_b", "estimand"],
        kind="mergesort",
    )
    estimands = [
        str(x)
        for x in shown["estimand"].dropna().astype(str).unique().tolist()
        if str(x)
    ]
    show_estimand = len(set(estimands)) > 1
    rows: list[dict[str, Any]] = []
    for _, row in shown.iterrows():
        metric = str(row.get("metric", ""))
        metric_label = _METRIC_LABELS.get(metric, metric)
        advantage = _safe_float(row.get("advantage_a_over_b"))
        low = _safe_float(row.get("advantage_ci_low"))
        high = _safe_float(row.get("advantage_ci_high"))
        advantage_text = _format_contrast_value(advantage, metric)
        if np.isfinite(low) and np.isfinite(high):
            advantage_text += (
                f" [{_format_contrast_value(low, metric)}, "
                f"{_format_contrast_value(high, metric)}]"
            )
        item: dict[str, Any] = {
            "Strategy A": str(row.get("strategy_a", "")),
            "Strategy B": str(row.get("strategy_b", "")),
        }
        if show_estimand:
            item["Estimand"] = _ESTIMAND_LABELS.get(
                str(row.get("estimand", "")), str(row.get("estimand", ""))
            )
        item.update(
            {
                "Metric": metric_label,
                "A": _format_contrast_value(row.get("estimate_a"), metric),
                "B": _format_contrast_value(row.get("estimate_b"), metric),
                "Effect favoring A (95% CI)": advantage_text,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _design_display(manifest: dict[str, Any]) -> pd.DataFrame:
    design = manifest.get("observed_oof_design", {})
    if not isinstance(design, dict) or not design:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for strategy, values in design.items():
        if not isinstance(values, dict):
            continue
        row: dict[str, Any] = {"Strategy": str(strategy)}
        mapping = (
            ("n_unique_samples", "Subjects"),
            ("n_outer_units", "Outer units"),
            ("n_repeats", "Repeats"),
            ("n_cohorts", "Held-out cohorts"),
        )
        for source, label in mapping:
            value = values.get(source)
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                row[label] = number
        rows.append(row)
    return pd.DataFrame(rows)


def _links_html(tables_dir: Path) -> str:
    names = [
        ("strategy_oof_performance.parquet", "Long-form OOF estimates"),
        ("strategy_oof_performance_table.parquet", "Wide OOF table"),
        ("strategy_oof_calibration.parquet", "Reliability-curve data"),
        ("strategy_oof_pairwise_contrasts.parquet", "Paired OOF contrasts"),
        ("strategy_oof_statistics_manifest.json", "OOF methods manifest"),
        ("strategy_oof_coverage.parquet", "OOF coverage"),
        ("strategy_oof_pairwise_coverage.parquet", "Paired OOF coverage"),
    ]
    links = []
    for filename, label in names:
        path = tables_dir / filename
        if path.exists():
            links.append(
                f'<a href="tables/{html.escape(filename)}">{html.escape(label)}</a>'
            )
    if not links:
        return ""
    return '<p class="oof-downloads">' + " · ".join(links) + "</p>"


def _procedure_grid_html(procedure: pd.DataFrame) -> str:
    return _report_module._procedure_grid_html(procedure)


def _human_token(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "—"
    replacements = {
        "relative_abundance": "relative abundance",
        "arcsine_sqrt": "arcsin√x",
        "log1p": "ln(1+x)",
        "log10p": "log10(1+x)",
        "log2p": "log2(1+x)",
    }
    return replacements.get(text, text.replace("_", " "))


def _mpma_b_composition(root: Path) -> pd.DataFrame:
    final = load_final_models(root)["MPMA-B"]
    representation = final.get("resolution", final.get("levels", ""))
    return pd.DataFrame(
        [
            {
                "Data representation": _human_token(representation),
                "Transformation": _human_token(final.get("count_transformation", "")),
                "Learner": str(final.get("learner", "")).strip() or "—",
            }
        ]
    )


def _remove_mpma_b_section(text: str) -> str:
    start = text.find(_MPMA_B_SECTION_START)
    if start == -1:
        return text
    end = text.find(_MPMA_B_SECTION_END, start)
    if end == -1:
        raise RuntimeError("Malformed MPMA-B composition section in report/index.html")
    end += len(_MPMA_B_SECTION_END)
    return text[:start] + text[end:]


def _replace_procedure_layout(text: str, report_dir: Path) -> str:
    procedure = _read_table(report_dir / "tables" / "evaluation_procedure.parquet")
    grid = _procedure_grid_html(procedure)
    if not grid:
        return text
    heading = text.find(_PROCEDURE_ANCHOR)
    if heading == -1:
        return text
    next_h2 = text.find("<h2 ", heading + len(_PROCEDURE_ANCHOR))
    grid_start = text.find(
        '<div class="procedure-grid">', heading + len(_PROCEDURE_ANCHOR)
    )
    if grid_start != -1 and (next_h2 == -1 or grid_start < next_h2):
        return text
    table_start = text.find(
        '<div class="table-wrap">', heading + len(_PROCEDURE_ANCHOR)
    )
    if table_start == -1:
        return text
    table_end = text.find("</div>", table_start)
    if table_end == -1:
        raise RuntimeError("Malformed Evaluation procedure table in report/index.html")
    table_end += len("</div>")
    return text[:table_start] + grid + text[table_end:]


def _insert_mpma_b_composition(text: str, report_dir: Path) -> str:
    text = _remove_mpma_b_section(text)
    table = _mpma_b_composition(report_dir.parent)
    if table.empty:
        return text
    heading = text.find(_PERFORMANCE_ANCHOR)
    if heading == -1:
        return text
    table_start = text.find(
        '<div class="table-wrap">', heading + len(_PERFORMANCE_ANCHOR)
    )
    if table_start == -1:
        return text
    table_end = text.find("</div>", table_start)
    if table_end == -1:
        raise RuntimeError("Malformed Task performance table in report/index.html")
    table_end += len("</div>")
    section = "\n".join(
        [
            _MPMA_B_SECTION_START,
            "<h3>Final MPMA-B specification</h3>",
            _html_table(table),
            _MPMA_B_SECTION_END,
        ]
    )
    return text[:table_end] + "\n" + section + text[table_end:]


def _inject_compact_report_css(text: str) -> str:
    if ".procedure-grid {" in text:
        return text
    style_end = text.find("</style>")
    if style_end == -1:
        return text
    return (
        text[:style_end] + _report_module._publication_layout_css() + text[style_end:]
    )


def _enhance_compact_layout(text: str, report_dir: Path) -> str:
    text = _replace_procedure_layout(text, report_dir)
    text = _insert_mpma_b_composition(text, report_dir)
    return _inject_compact_report_css(text)


def _task_class_count(sweep: Any) -> int | None:
    data = getattr(sweep, "data", None)
    labels = getattr(data, "class_labels", None)
    if labels is not None:
        try:
            count = len(labels)
        except TypeError:
            count = 0
        if count > 0:
            return int(count)
    label_map = getattr(data, "label_map", None)
    if isinstance(label_map, dict) and label_map:
        values = {str(value) for value in label_map.values()}
        if values:
            return int(len(values))
    return None


def oof_section_html(
    report_dir: Path | str,
    n_classes: int | None = None,
    heading_level: int = 2,
    id_prefix: str = "",
    include_downloads: bool = True,
) -> str:
    report_dir = Path(report_dir)
    tables_dir = report_dir / "tables"
    performance = _read_table(tables_dir / "strategy_oof_performance.parquet")
    calibration_curve = _read_table(tables_dir / "strategy_oof_calibration.parquet")
    contrasts = _read_table(tables_dir / "strategy_oof_pairwise_contrasts.parquet")
    manifest = _read_json(tables_dir / "strategy_oof_statistics_manifest.json")
    if n_classes is None:
        try:
            manifest_classes = int(manifest.get("n_classes", 0))
        except (TypeError, ValueError):
            manifest_classes = 0
        if manifest_classes > 0:
            n_classes = manifest_classes
    if performance.empty:
        return ""
    primary_table = _performance_display(performance)
    multiclass_table = _multiclass_display(performance, n_classes)
    probability_table = _probability_display(performance, n_classes)
    calibration_table = _calibration_display(performance)
    contrast_table = _contrast_display(contrasts, n_classes)
    design_table = _design_display(manifest)
    level = max(1, min(5, int(heading_level)))
    sublevel = min(6, level + 1)
    heading = f"h{level}"
    subheading = f"h{sublevel}"
    prefix = f"{id_prefix}-" if id_prefix else ""
    parts = [
        f'<section id="{prefix}oof-inference">',
        f'<{heading} id="{prefix}oof-performance">Out-of-fold statistical inference</{heading}>',
        "<p>Held-out predictions are pooled at the protocol-defined inference unit. "
        "Tables report point estimates and 95% confidence intervals.</p>",
        _methodology_html(manifest),
    ]
    if not design_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-design">Inference design</{subheading}>',
                _html_table(design_table),
            ]
        )
    if not primary_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-primary-performance">Performance</{subheading}>',
                _html_table(primary_table),
            ]
        )
    if not multiclass_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-multiclass-performance">Multiclass summaries</{subheading}>',
                _html_table(multiclass_table),
            ]
        )
    if not probability_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-probability-quality">Probability quality</{subheading}>',
                _html_table(probability_table),
                _probability_semantics_html(manifest),
            ]
        )
    if not contrast_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-contrasts">Paired strategy contrasts</{subheading}>',
                "<p>Contrasts use matched held-out predictions and the same bootstrap "
                "draws for both strategies. Positive effects favor Strategy A after "
                "accounting for metric direction.</p>",
                _html_table(contrast_table),
            ]
        )
    if not calibration_table.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-calibration-summary">Calibration</{subheading}>',
                "<p>Calibration-in-the-large and intercept are referenced to 0; "
                "calibration slope is referenced to 1.</p>",
                _html_table(calibration_table),
            ]
        )
    elif not calibration_curve.empty:
        parts.extend(
            [
                f'<{subheading} id="{prefix}oof-calibration-summary">Calibration</{subheading}>',
                "<p>One-vs-rest reliability-curve data are available in "
                "<code>strategy_oof_calibration.parquet</code>.</p>",
            ]
        )
    if include_downloads:
        parts.append(_links_html(tables_dir))
    parts.append(_SECTION_END)
    return "\n".join(part for part in parts if part)


def _section_html(report_dir: Path, n_classes: int | None = None) -> str:
    return oof_section_html(report_dir, n_classes=n_classes)


def _remove_existing_section(text: str) -> str:
    start = text.find(_SECTION_START)
    if start == -1:
        return text
    end = text.find(_SECTION_END, start + len(_SECTION_START))
    if end == -1:
        raise RuntimeError("Malformed existing pooled-OOF section in report/index.html")
    end += len(_SECTION_END)
    return text[:start] + text[end:]


def _statistics_section_end(text: str) -> int:
    start = text.find(_STATISTICS_ANCHOR)
    if start == -1:
        raise RuntimeError(
            "Cannot verify report layout: the statistical-comparisons HTML anchor was not found. The base report template has changed and the integration must be updated explicitly."
        )
    search_from = start + len(_STATISTICS_ANCHOR)
    candidates = [
        pos
        for pos in (text.find("<h2 ", search_from), text.find("<h3 ", search_from))
        if pos != -1
    ]
    if candidates:
        return min(candidates)
    footer = text.find(_FOOTER_ANCHOR, search_from)
    if footer != -1:
        return footer
    body_close = text.rfind("</body>")
    return body_close if body_close != -1 else len(text)


def _extract_compute_block(text: str) -> tuple[str, str]:
    start = text.find(_COMPUTE_ANCHOR)
    if start == -1:
        raise RuntimeError(
            "Cannot locate Computational resources. The base report template has changed and the integration must be updated explicitly."
        )
    next_h2 = text.find("<h2 ", start + len(_COMPUTE_ANCHOR))
    footer = text.find(_FOOTER_ANCHOR, start + len(_COMPUTE_ANCHOR))
    candidates = [x for x in (next_h2, footer) if x != -1]
    end = min(candidates) if candidates else len(text)
    block = text[start:end].strip()
    return text[:start] + text[end:], block


def enhance_html_report(report_dir: Path | str, n_classes: int | None = None) -> bool:
    report_dir = Path(report_dir)
    index_path = report_dir / "index.html"
    if not index_path.exists():
        raise FileNotFoundError(f"Report HTML does not exist: {index_path}")
    section = _section_html(report_dir, n_classes=n_classes)
    text = _enhance_compact_layout(
        index_path.read_text(encoding="utf-8"),
        report_dir,
    )
    text = _remove_existing_section(text)
    if section:
        insertion_pos = _statistics_section_end(text)
        text = text[:insertion_pos] + section + "\n" + text[insertion_pos:]
    text, compute_block = _extract_compute_block(text)
    footer_pos = text.find(_FOOTER_ANCHOR)
    if footer_pos == -1:
        body_close = text.rfind("</body>")
        footer_pos = body_close if body_close != -1 else len(text)
    if compute_block:
        text = text[:footer_pos] + compute_block + "\n" + text[footer_pos:]
    text = text.replace(_NAV_LINK + "\n", "").replace("\n" + _NAV_LINK, "")
    if section and _NAV_ANCHOR in text:
        text = text.replace(_NAV_ANCHOR, _NAV_ANCHOR + "\n" + _NAV_LINK, 1)
    text = text.replace("Learner family", "Learner type")
    index_path.write_text(text, encoding="utf-8")
    return bool(section)


def _terminal_oof_summary(report_dir: Path) -> None:
    performance = _read_table(
        report_dir / "tables" / "strategy_oof_performance.parquet"
    )
    if performance.empty:
        return
    display = _performance_display(performance)
    if display.empty:
        return
    preferred = [
        "Strategy",
        "Estimand",
        "ROC-AUC",
        "PR-AUC (AP)",
        "nMCC",
        "Brier score",
        "Log loss",
    ]
    cols = [c for c in preferred if c in display.columns]
    if not cols:
        cols = list(display.columns[:8])
    from rich.table import Table

    table = Table(title="Pooled OOF performance (95% CI)", show_lines=False)
    for col in cols:
        table.add_column(str(col), overflow="fold", no_wrap=False)
    for _, row in display[cols].iterrows():
        table.add_row(*[str(row.get(col, "")) for col in cols])
    console.print(table)


def _member_table_with_learner_type(
    table: pd.DataFrame,
) -> pd.DataFrame:
    if table.empty:
        return table
    return table.rename(columns={"Learner family": "Learner type"})


def _procedure_table_publication(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or not {"Field", "Value"}.issubset(table.columns):
        return table
    out = table.copy()
    mask = out["Field"].astype(str).eq("Selection rule")
    out.loc[
        mask,
        "Value",
    ] = (
        "Model and ensemble selection use inner-validation performance; "
        "reported performance uses held-out outer evaluation predictions."
    )
    return out


def write_report(sweep: Any) -> dict[str, Path]:
    root = Path(sweep.root())
    original_summary = _report_module._ensemble_summary_table
    original_procedure = _report_module._procedure_table
    original_terminal = _report_module._terminal_table

    def procedure_table(report_sweep: Any, report_root: Path) -> pd.DataFrame:
        return _procedure_table_publication(
            original_procedure(report_sweep, report_root)
        )

    def terminal_table(
        title: str,
        table: pd.DataFrame,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return original_terminal(
            title,
            _member_table_with_learner_type(table),
            *args,
            **kwargs,
        )

    _report_module._ensemble_summary_table = _ensemble_summary_from_selected
    _report_module._procedure_table = procedure_table
    _report_module._terminal_table = terminal_table
    try:
        outputs = dict(_report_module.write_report(sweep))
    finally:
        _report_module._ensemble_summary_table = original_summary
        _report_module._procedure_table = original_procedure
        _report_module._terminal_table = original_terminal
    report_dir = root / "report"
    tables_dir = report_dir / "tables"
    performance_path = tables_dir / "strategy_oof_performance.parquet"
    if _read_table(performance_path).empty:
        raise RuntimeError(
            "Report statistics did not produce strategy_oof_performance.parquet."
        )
    inserted = enhance_html_report(
        report_dir,
        n_classes=_task_class_count(sweep),
    )
    if inserted:
        _terminal_oof_summary(report_dir)
    new_outputs = {
        "strategy_oof_performance": tables_dir / "strategy_oof_performance.parquet",
        "strategy_oof_performance_table": tables_dir
        / "strategy_oof_performance_table.parquet",
        "strategy_oof_calibration": tables_dir / "strategy_oof_calibration.parquet",
        "strategy_oof_pairwise_contrasts": tables_dir
        / "strategy_oof_pairwise_contrasts.parquet",
        "strategy_oof_statistics_manifest": tables_dir
        / "strategy_oof_statistics_manifest.json",
        "strategy_oof_coverage": tables_dir / "strategy_oof_coverage.parquet",
        "strategy_oof_pairwise_coverage": tables_dir
        / "strategy_oof_pairwise_coverage.parquet",
    }
    existing = {name: path for name, path in new_outputs.items() if path.exists()}
    outputs.update(existing)
    if existing:
        stage("Out-of-fold inference", str(report_dir))
        path_table("OOF report outputs", existing)
    if inserted:
        success("Out-of-fold inference added to index.html")
    return outputs
