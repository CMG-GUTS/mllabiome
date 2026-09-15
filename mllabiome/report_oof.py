from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .console import console, path_table, stage, success
from .report import write_report as _write_base_report

_PERCENT_METRICS = {
    "AUC",
    "PR_AUC",
    "nMCC",
    "Accuracy",
    "BalAcc",
    "F1",
    "F1w",
    "Precision",
    "Recall",
}
_PERFORMANCE_METRIC_ORDER = (
    "AUC",
    "PR_AUC",
    "nMCC",
    "Accuracy",
    "BalAcc",
    "F1w",
    "Precision",
    "Recall",
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
    "PR_AUC": "PR-AUC (AP)",
    "nMCC": "nMCC",
    "Accuracy": "Accuracy",
    "BalAcc": "Balanced accuracy",
    "F1": "F1",
    "F1w": "F1w",
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


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
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
    group_cols = ["Strategy", "estimand"]
    for (strategy, estimand), group in performance.groupby(group_cols, sort=False):
        row: dict[str, Any] = {"Strategy": str(strategy)}
        if show_estimand:
            row["Estimand"] = _ESTIMAND_LABELS.get(str(estimand), str(estimand))
        for metric in metrics:
            sub = group[group["metric"].astype(str).eq(metric)]
            if sub.empty:
                row[_METRIC_LABELS.get(metric, metric)] = "—"
                continue
            label = _METRIC_LABELS.get(metric, metric)
            row[label] = _format_ci_cell(sub.iloc[0], metric)
        rows.append(row)
    return pd.DataFrame(rows)


def _performance_display(performance: pd.DataFrame) -> pd.DataFrame:
    return _wide_metric_table(performance, _PERFORMANCE_METRIC_ORDER)


def _calibration_display(performance: pd.DataFrame) -> pd.DataFrame:
    if performance.empty or "metric" not in performance.columns:
        return pd.DataFrame()
    calibration = performance[
        performance["metric"].astype(str).isin(_CALIBRATION_METRIC_ORDER)
    ].copy()
    return _wide_metric_table(calibration, _CALIBRATION_METRIC_ORDER)


def _coverage_display(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty or "Strategy" not in coverage.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in coverage.iterrows():
        item: dict[str, Any] = {"Strategy": str(row.get("Strategy", ""))}
        n_complete = _safe_float(row.get("n_samples_complete_repeats"))
        n_unique = _safe_float(row.get("n_unique_samples"))
        n_samples = n_complete if np.isfinite(n_complete) else n_unique
        item["Subjects"] = int(n_samples) if np.isfinite(n_samples) else "—"
        n_repeats = _safe_float(row.get("n_repeats"))
        if np.isfinite(n_repeats) and n_repeats > 0:
            item["Repeats"] = int(n_repeats)
        n_cohorts = _safe_float(row.get("n_cohorts"))
        if np.isfinite(n_cohorts) and n_cohorts > 0:
            item["Held-out cohorts"] = int(n_cohorts)
        coverage_fraction = _safe_float(row.get("complete_repeat_coverage_fraction"))
        if np.isfinite(coverage_fraction):
            item["Complete OOF coverage"] = f"{100.0 * coverage_fraction:.1f}%"
        inference_population = str(row.get("inference_population", "")).strip()
        if inference_population:
            item["Inference population"] = inference_population.replace("_", " ")
        status = str(row.get("status", "")).strip()
        if status:
            item["Status"] = status
        reason = str(row.get("reason", "")).strip()
        if reason and reason.lower() != "nan":
            item["Reason"] = reason
        rows.append(item)
    order = [
        "Strategy",
        "Subjects",
        "Repeats",
        "Held-out cohorts",
        "Complete OOF coverage",
        "Inference population",
        "Status",
        "Reason",
    ]
    out = pd.DataFrame(rows)
    return out[[c for c in order if c in out.columns]]


def _html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    cols = list(df.columns)
    parts = ['<div class="table-wrap"><table><thead><tr>']
    parts.extend((f"<th>{html.escape(str(c))}</th>" for c in cols))
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
    bootstrap = f"{int(n_bootstrap):,}" if str(n_bootstrap).isdigit() else ""
    if protocol in {"lodo", "leave_one_dataset_out"}:
        text = "Pooled held-out estimates use a two-stage cohort-and-subject bootstrap"
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
        text = (
            "Pooled held-out estimates use a subject-cluster bootstrap"
            if repeats and max(repeats) == 1
            else "Pooled held-out estimates use subject-cluster and repeat resampling"
        )
    if bootstrap:
        text += f" ({bootstrap} replicates)"
    return f"<p>{html.escape(text)}. Brackets denote 95% confidence intervals.</p>"


def _probability_semantics_html(manifest: dict[str, Any]) -> str:
    semantics = manifest.get("probability_semantics", {})
    if not isinstance(semantics, dict):
        return ""
    omitted: list[str] = []
    for strategy, info in semantics.items():
        if not isinstance(info, dict) or bool(info.get("valid", False)):
            continue
        aggregations = info.get("aggregation_strategies", [])
        reason = str(info.get("reason", "")).lower()
        if isinstance(aggregations, list) and aggregations:
            detail = "outer-fold aggregation: " + ", ".join(
                (str(x) for x in aggregations)
            )
        elif "rank_mean" in reason:
            detail = "rank_mean aggregation"
        elif "siamcat" in reason:
            detail = "score output"
        elif "ridge" in reason:
            detail = "decision-score output"
        else:
            detail = "probability semantics not verified"
        omitted.append(
            f"<strong>{html.escape(str(strategy))}</strong> ({html.escape(detail)})"
        )
    if not omitted:
        return ""
    return (
        "<p>Proper scoring and calibration metrics are reported only for strategies with verified probabilistic outputs. Omitted: "
        + "; ".join(omitted)
        + ".</p>"
    )


def _format_contrast_value(value: Any, metric: str) -> str:
    v = _safe_float(value)
    if not np.isfinite(v):
        return "—"
    if metric in _PERCENT_METRICS:
        return f"{100.0 * v:.2f}"
    return f"{v:.4f}"


def _contrast_display(contrasts: pd.DataFrame) -> pd.DataFrame:
    required = {
        "strategy_a",
        "strategy_b",
        "estimand",
        "metric",
        "is_selection_metric",
        "estimate_a",
        "estimate_b",
        "advantage_a_over_b",
        "advantage_ci_low",
        "advantage_ci_high",
    }
    if contrasts.empty or not required.issubset(contrasts.columns):
        return pd.DataFrame()
    selection_metrics = set(
        contrasts.loc[
            contrasts["is_selection_metric"].fillna(False).astype(bool), "metric"
        ].astype(str)
    )
    preferred = selection_metrics | {
        "AUC",
        "PR_AUC",
        "Brier",
        "Brier_multiclass",
        "LogLoss",
    }
    shown = contrasts[contrasts["metric"].astype(str).isin(preferred)].copy()
    if shown.empty:
        shown = contrasts.copy()
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
            advantage_text += f" [{_format_contrast_value(low, metric)}, {_format_contrast_value(high, metric)}]"
        if np.isfinite(low) and low > 0:
            interpretation = "A favored"
        elif np.isfinite(high) and high < 0:
            interpretation = "B favored"
        else:
            interpretation = "CI includes 0"
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
                "Advantage A over B (95% CI)": advantage_text,
                "Interpretation": interpretation,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _links_html(tables_dir: Path) -> str:
    names = [
        ("strategy_oof_performance.tsv", "Long-form OOF estimates"),
        ("strategy_oof_performance_table.tsv", "Wide OOF table"),
        ("strategy_oof_calibration.tsv", "Reliability-curve data"),
        ("strategy_oof_pairwise_contrasts.tsv", "Paired OOF contrasts"),
        ("strategy_oof_statistics_manifest.json", "OOF methods manifest"),
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


def _section_html(report_dir: Path) -> str:
    tables_dir = report_dir / "tables"
    performance = _read_tsv(tables_dir / "strategy_oof_performance.tsv")
    calibration_curve = _read_tsv(tables_dir / "strategy_oof_calibration.tsv")
    contrasts = _read_tsv(tables_dir / "strategy_oof_pairwise_contrasts.tsv")
    manifest = _read_json(tables_dir / "strategy_oof_statistics_manifest.json")
    if performance.empty:
        return ""
    performance_table = _performance_display(performance)
    calibration_table = _calibration_display(performance)
    contrast_table = _contrast_display(contrasts)
    parts = [
        _SECTION_START,
        '<h2 id="oof-performance">Pooled out-of-fold performance</h2>',
        "<p>Performance is calculated from held-out predictions pooled across outer test folds. Discrimination and classification metrics are percentages; proper scoring rules and calibration statistics are shown on their natural scales.</p>",
        _methodology_html(manifest),
        _html_table(performance_table),
        _probability_semantics_html(manifest),
    ]
    if not contrast_table.empty:
        parts.extend(
            [
                '<h3 id="oof-contrasts">Paired strategy contrasts</h3>',
                "<p>Paired differences use matched held-out observations and the same resampling structure as the pooled estimates. Positive values favor Strategy A. Brackets denote 95% confidence intervals.</p>",
                _html_table(contrast_table),
            ]
        )
    if not calibration_table.empty:
        parts.extend(
            [
                '<h3 id="oof-calibration-summary">Calibration</h3>',
                "<p>Ideal values are 0 for calibration-in-the-large and intercept, and 1 for slope.</p>",
                _html_table(calibration_table),
            ]
        )
    elif not calibration_curve.empty:
        parts.extend(
            [
                '<h3 id="oof-calibration-summary">Calibration</h3>',
                "<p>One-vs-rest reliability data are available in <code>strategy_oof_calibration.tsv</code>.</p>",
            ]
        )
    parts.append(_links_html(tables_dir))
    parts.append(_SECTION_END)
    return "\n".join((part for part in parts if part))


def _remove_existing_section(text: str) -> str:
    start = text.find(_SECTION_START)
    end = text.find(_SECTION_END)
    if start == -1 and end == -1:
        return text
    if start == -1 or end == -1 or end < start:
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
    return (text[:start] + text[end:], block)


def enhance_html_report(report_dir: Path | str) -> bool:
    report_dir = Path(report_dir)
    index_path = report_dir / "index.html"
    if not index_path.exists():
        raise FileNotFoundError(f"Report HTML does not exist: {index_path}")
    section = _section_html(report_dir)
    if not section:
        return False
    original = index_path.read_text(encoding="utf-8")
    text = _remove_existing_section(original)
    insertion_pos = _statistics_section_end(text)
    marker = '<span id="mllabiome-oof-insertion-slot"></span>'
    text = text[:insertion_pos] + marker + text[insertion_pos:]
    text, compute_block = _extract_compute_block(text)
    if marker not in text:
        raise RuntimeError(
            "OOF insertion slot was lost while normalizing report layout"
        )
    text = text.replace(marker, section + "\n", 1)
    footer_pos = text.find(_FOOTER_ANCHOR)
    if footer_pos == -1:
        body_close = text.rfind("</body>")
        footer_pos = body_close if body_close != -1 else len(text)
    text = text[:footer_pos] + compute_block + "\n" + text[footer_pos:]
    text = text.replace(_NAV_LINK + "\n", "").replace("\n" + _NAV_LINK, "")
    if _NAV_ANCHOR in text:
        text = text.replace(_NAV_ANCHOR, _NAV_ANCHOR + "\n" + _NAV_LINK, 1)
    index_path.write_text(text, encoding="utf-8")
    return True


def _terminal_oof_summary(report_dir: Path) -> None:
    performance = _read_tsv(report_dir / "tables" / "strategy_oof_performance.tsv")
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


def write_report(sweep: Any) -> dict[str, Path]:
    outputs = dict(_write_base_report(sweep))
    report_dir = Path(sweep.root()) / "report"
    tables_dir = report_dir / "tables"
    inserted = enhance_html_report(report_dir)
    if not inserted:
        return outputs
    _terminal_oof_summary(report_dir)
    new_outputs = {
        "strategy_oof_performance": tables_dir / "strategy_oof_performance.tsv",
        "strategy_oof_performance_table": tables_dir
        / "strategy_oof_performance_table.tsv",
        "strategy_oof_calibration": tables_dir / "strategy_oof_calibration.tsv",
        "strategy_oof_pairwise_contrasts": tables_dir
        / "strategy_oof_pairwise_contrasts.tsv",
        "strategy_oof_statistics_manifest": tables_dir
        / "strategy_oof_statistics_manifest.json",
    }
    existing = {name: path for name, path in new_outputs.items() if path.exists()}
    outputs.update(existing)
    if existing:
        stage("Pooled OOF inference", str(report_dir))
        path_table("OOF report outputs", existing)
    success("Pooled OOF section added to index.html")
    return outputs
