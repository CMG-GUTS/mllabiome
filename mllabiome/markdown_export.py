from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .storage import read_table
from .utils import METRIC_COLUMNS


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_text(value) for value in row) + " |")
    return "\n".join(lines)


def _score(frame: pd.DataFrame, metric: str) -> str:
    if frame.empty or metric not in frame.columns:
        return "—"
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    valid = values[np.isfinite(values)]
    if not len(valid):
        return "—"
    mean = float(valid.mean())
    if len(valid) == 1:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {float(valid.std(ddof=1)):.3f}"


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "ok" not in frame.columns:
        return frame
    return frame[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]


def _folds(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "0/0"
    return f"{len(_valid(frame))}/{len(frame)}"


def _status(inner: pd.DataFrame, outer: pd.DataFrame) -> str:
    total = len(inner) + len(outer)
    passed = len(_valid(inner)) + len(_valid(outer))
    if not total:
        return "not run"
    if not passed:
        return "failed"
    if passed < total:
        return "partial failure"
    if not len(outer):
        return "inner only"
    if not len(inner):
        return "outer only"
    return "evaluated"


def _errors(inner: pd.DataFrame, outer: pd.DataFrame) -> str:
    errors = []
    for frame in (inner, outer):
        if "error" not in frame.columns:
            continue
        failed = (
            frame
            if "ok" not in frame.columns
            else frame[~pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
        )
        for item in failed["error"].dropna():
            value = str(item).strip()
            if value and value not in errors:
                errors.append(value)
    return "; ".join(errors[:2])[:240]


def _groups(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty or "config_id" not in frame.columns:
        return {}
    return {
        str(config_id): group
        for config_id, group in frame.groupby(
            frame["config_id"].astype(str), sort=False
        )
    }


def export_markdown_results(root: Path) -> Path:
    root = Path(root)
    configs = read_table(root / "configs.parquet")
    inner = read_table(root / "inner_results" / "inner_results.parquet")
    outer = read_table(root / "results" / "outer_results.parquet")
    if inner.empty and outer.empty:
        raise FileNotFoundError(
            f"No evaluation result rows found in {root}. Run --stage evaluate first."
        )
    metadata = {}
    if not configs.empty and "config_id" in configs.columns:
        for row in configs.to_dict("records"):
            metadata[str(row["config_id"])] = row
    inner_by_id = _groups(inner)
    outer_by_id = _groups(outer)
    all_ids = sorted(set(metadata) | set(inner_by_id) | set(outer_by_id))
    metrics = [
        metric
        for metric in METRIC_COLUMNS
        if any(
            metric in frame.columns
            and not _valid(frame).empty
            and pd.to_numeric(_valid(frame)[metric], errors="coerce").notna().any()
            for frame in (inner, outer)
        )
    ]
    key_metrics = [
        metric
        for metric in ("log_loss", "AUROC", "AUCPR", "AP", "MCC")
        if metric in metrics
    ]
    summary_headers = [
        "Config ID",
        "Active",
        "Resolution",
        "Transformation",
        "Learner",
        "Status",
        "Inner folds (OK/all)",
        "Outer folds (OK/all)",
    ] + [f"{phase} {metric}" for metric in key_metrics for phase in ("Inner", "Outer")]
    summary_headers.append("Error")
    summary = []
    detail_headers = [
        "Config ID",
        "Active",
        "Resolution",
        "Transformation",
        "Learner",
        "Status",
        "Folds (OK/all)",
    ] + metrics
    inner_rows = []
    outer_rows = []
    for config_id in all_ids:
        data = metadata.get(config_id, {})
        inner_group = inner_by_id.get(config_id, pd.DataFrame())
        outer_group = outer_by_id.get(config_id, pd.DataFrame())
        fallback = next(
            (frame.iloc[0] for frame in (inner_group, outer_group) if not frame.empty),
            {},
        )

        def field(name: str) -> object:
            value = data.get(name)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return fallback.get(name, "")
            return value

        active = field("active")
        active_label = (
            "yes"
            if pd.notna(active) and str(active) in {"1", "1.0", "True"}
            else "no"
            if pd.notna(active) and str(active) in {"0", "0.0", "False"}
            else "unknown"
        )
        display = field("learner")
        base = [
            config_id,
            active_label,
            field("resolution"),
            field("count_transformation"),
            display,
            _status(inner_group, outer_group),
        ]
        summary.append(
            base
            + [_folds(inner_group), _folds(outer_group)]
            + [
                _score(_valid(group), metric)
                for metric in key_metrics
                for group in (inner_group, outer_group)
            ]
            + [_errors(inner_group, outer_group)]
        )
        for rows, group in ((inner_rows, inner_group), (outer_rows, outer_group)):
            rows.append(
                base
                + [_folds(group)]
                + [_score(_valid(group), metric) for metric in metrics]
            )
    evaluated_count = len(set(inner_by_id) | set(outer_by_id))
    content = [
        "# Evaluated configurations",
        "",
        f"Experiment: `{root}`",
        "",
        f"Configurations with recorded results: {evaluated_count}. "
        f"Configurations listed: {len(all_ids)}.",
        "",
        "Metrics are mean ± sample standard deviation across successful recorded fold rows, "
        "on the original 0–1 scale when applicable. This export summarizes stored artifacts "
        "and does not rerun evaluation. Active indicates the last registered configuration "
        "set, not necessarily the source of the recorded results.",
        "",
        "## All configurations",
        "",
        _markdown_table(summary_headers, summary),
        "",
        "## All inner metrics",
        "",
        _markdown_table(detail_headers, inner_rows),
        "",
        "## All outer metrics",
        "",
        _markdown_table(detail_headers, outer_rows),
        "",
    ]
    output = root / "exports" / "markdown" / "all_configurations.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(content), encoding="utf-8")
    return output
