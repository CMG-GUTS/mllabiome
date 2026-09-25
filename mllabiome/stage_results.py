from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .configs_sweep import Sweep, target_sweeps
from .console import summary_table, warn
from .storage import read_table, table_exists
from .utils import tail_ellipsis


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _children(sweep: Sweep) -> list[Sweep]:
    if getattr(sweep, "uses_modalities", False):
        return [sweep]
    children = target_sweeps(sweep)
    return children if children else [sweep]


def _target_name(sweep: Sweep) -> str:
    if getattr(sweep, "uses_modalities", False):
        samples = getattr(sweep, "samples", None)
        return str(getattr(samples, "target_col", "target"))
    data = getattr(sweep, "data", None)
    return str(getattr(data, "target_col", "target"))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_text(summary: dict[str, Any], metric: str) -> str:
    mean = _finite(summary.get(f"outer_{metric}_mean"))
    std = _finite(summary.get(f"outer_{metric}_std"))
    count = summary.get(f"outer_{metric}_count", summary.get("n_outer_folds"))
    if mean is None:
        return f"{metric} unavailable"
    value = f"{metric}={mean:.4f}"
    if std is not None:
        value += f" ± {std:.4f}"
    try:
        folds = int(count)
    except (TypeError, ValueError):
        folds = 0
    if folds > 0:
        value += f" · {folds} outer folds"
    return value


def _selection_score(candidate: dict[str, Any]) -> str:
    score = _finite(candidate.get("inner_score"))
    if score is None:
        return "inner selection score unavailable"
    metric = str(
        candidate.get("selection_metric", candidate.get("optimize_metric", ""))
    ).strip()
    value = f"inner {metric}={score:.4f}" if metric else f"inner score={score:.4f}"
    std = _finite(candidate.get("inner_score_std"))
    if std is not None:
        value += f" ± {std:.4f}"
    try:
        folds = int(candidate.get("n_inner_folds", 0))
    except (TypeError, ValueError):
        folds = 0
    if folds > 0:
        value += f" · {folds} inner folds"
    return value


def _model_spec(candidate: dict[str, Any]) -> str:
    parts = []
    identifier = str(candidate.get("config_id", "")).strip()
    if identifier:
        parts.append(identifier)
    fields = (
        ("learner", "learner"),
        ("count_transformation", "transform"),
        ("resolution", "resolution"),
        ("levels", "levels"),
        ("mpdr_id", "MPDR"),
    )
    for key, label in fields:
        value = candidate.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            parts.append(f"{label}={text}")
    return " · ".join(parts) if parts else "final model specification unavailable"


def _ensemble_spec(candidate: dict[str, Any]) -> str:
    parts = []
    identifier = str(candidate.get("ensemble_config_id", "")).strip()
    if identifier:
        parts.append(identifier)
    for key, label in (
        ("selection_strategy", "selection"),
        ("aggregation_strategy", "aggregation"),
    ):
        value = str(candidate.get(key, "")).strip()
        if value:
            parts.append(f"{label}={value}")
    try:
        members = int(candidate.get("member_count", candidate.get("ensemble_size", 0)))
    except (TypeError, ValueError):
        members = 0
    if members > 0:
        parts.append(f"members={members}")
    try:
        effective = int(candidate.get("effective_member_count", 0))
    except (TypeError, ValueError):
        effective = 0
    if effective > 0 and effective != members:
        parts.append(f"effective={effective}")
    try:
        max_size = int(candidate.get("max_size", 0))
    except (TypeError, ValueError):
        max_size = 0
    if max_size > 0:
        parts.append(f"max={max_size}")
    return " · ".join(parts) if parts else "final ensemble specification unavailable"


def _ensemble_members(root: Path, candidate: dict[str, Any]) -> list[str]:
    models = _read_json(root / "final_models.json")
    unit = models.get("MPMA-E") if isinstance(models.get("MPMA-E"), dict) else {}
    members = unit.get("members", []) if isinstance(unit, dict) else []
    out = []
    if isinstance(members, list):
        for index, member in enumerate(members, start=1):
            if not isinstance(member, dict):
                continue
            parts = [str(member.get("config_id", f"member_{index}"))]
            for key, label in (
                ("learner", "learner"),
                ("count_transformation", "transform"),
                ("resolution", "resolution"),
                ("levels", "levels"),
            ):
                value = member.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text and text.lower() not in {"nan", "none"}:
                    parts.append(f"{label}={text}")
            weight = _finite(member.get("aggregation_weight", member.get("weight")))
            if weight is not None:
                parts.append(f"weight={weight:.4f}")
            out.append(f"{index}. " + " · ".join(parts))
    if out:
        return out
    raw_members = candidate.get("members", [])
    raw_weights = candidate.get("weights", [])
    if not isinstance(raw_members, list):
        return []
    weights = raw_weights if isinstance(raw_weights, list) else []
    for index, member in enumerate(raw_members, start=1):
        text = f"{index}. {member}"
        if index <= len(weights):
            weight = _finite(weights[index - 1])
            if weight is not None:
                text += f" · weight={weight:.4f}"
        out.append(text)
    return out


def _hardware_rows(root: Path, target: str) -> list[tuple[str, object]]:
    summary = _read_json(root / "run_summary.json")
    machine = (
        summary.get("machine", {}) if isinstance(summary.get("machine"), dict) else {}
    )
    rows: list[tuple[str, object]] = []
    cpu_model = str(machine.get("cpu_model", "")).strip()
    physical = machine.get("physical_cpus")
    logical = machine.get("logical_cpus")
    cpu_bits = [x for x in [cpu_model] if x]
    if physical not in (None, "") or logical not in (None, ""):
        cpu_bits.append(
            f"{physical if physical not in (None, '') else '?'} physical / "
            f"{logical if logical not in (None, '') else '?'} logical CPUs"
        )
    if cpu_bits:
        rows.append((f"{target} · hardware", " · ".join(cpu_bits)))

    def gib(key: str) -> float | None:
        value = _finite(machine.get(key))
        return value / (1024**3) if value is not None else None

    available = gib("memory_available_bytes")
    total = gib("memory_total_bytes")
    if available is not None or total is not None:
        parts = []
        if available is not None:
            parts.append(f"available={available:.3f} GiB")
        if total is not None:
            parts.append(f"total={total:.3f} GiB")
        rows.append((f"{target} · memory", " · ".join(parts)))
    workers = summary.get("workers")
    threads = summary.get("threads_per_worker")
    if workers not in (None, "") or threads not in (None, ""):
        rows.append(
            (
                f"{target} · parallelism",
                f"{workers if workers not in (None, '') else '?'} workers × "
                f"{threads if threads not in (None, '') else '?'} threads/worker",
            )
        )
    return rows


def _evaluation_rows(sweep: Sweep) -> list[tuple[str, object]]:
    rows = []
    for child in _children(sweep):
        root = Path(child.root())
        target = _target_name(child)
        summary = _read_json(root / "tables" / "mpma_b_strategy_summary.json")
        candidate = _read_json(root / "tables" / "mpma_b_final_candidate.json")
        metric = str(child.evaluation.optimize_metric)
        rows.extend(
            [
                (
                    f"{target} · nested performance",
                    f"MPMA-B · {_metric_text(summary, metric)}",
                ),
                (f"{target} · selected model", _model_spec(candidate)),
                (f"{target} · selection score", _selection_score(candidate)),
            ]
        )
        rows.extend(_hardware_rows(root, target))
    return rows


def _ensemble_rows(sweep: Sweep) -> list[tuple[str, object]]:
    rows = []
    for child in _children(sweep):
        root = Path(child.root())
        target = _target_name(child)
        summary = _read_json(root / "ensembling" / "mpma_e_strategy_summary.json")
        candidate = _read_json(root / "ensembling" / "mpma_e_final_candidate.json")
        metric = str(
            candidate.get(
                "selection_metric",
                getattr(child.ensemble, "optimize_metric", "log_loss"),
            )
        )
        rows.extend(
            [
                (
                    f"{target} · nested performance",
                    f"MPMA-E · {_metric_text(summary, metric)}",
                ),
                (f"{target} · selected ensemble", _ensemble_spec(candidate)),
                (f"{target} · selection score", _selection_score(candidate)),
            ]
        )
        for index, member in enumerate(_ensemble_members(root, candidate), start=1):
            rows.append((f"{target} · member {index}", member))
    return rows


def _feature_column(frame: pd.DataFrame) -> str | None:
    for column in ("feature", "representation_feature", "coordinate", "taxon"):
        if column in frame.columns:
            return column
    return None


def _feature_order(frame: pd.DataFrame) -> tuple[str | None, bool]:
    for column, ascending in (
        ("rank", True),
        ("consensus", False),
        ("mean_rank", True),
        ("importance_mean", False),
        ("gross_member_support", False),
        ("top_k_frequency", False),
    ):
        if column in frame.columns:
            return column, ascending
    return None, True


def _feature_score(row: pd.Series) -> str:
    for column, label in (
        ("consensus", "consensus"),
        ("importance_mean", "importance"),
        ("mean_rank", "mean rank"),
        ("gross_member_support", "support"),
        ("top_k_frequency", "top-k frequency"),
    ):
        if column not in row.index:
            continue
        value = _finite(row[column])
        if value is not None:
            return f"{label}={value:.4f}"
    return ""


def _top_feature_rows(target_dir: Path, top_n: int = 5) -> list[tuple[str, str]]:
    candidates = [
        target_dir / "top_features.parquet",
        target_dir / "feature_importance.parquet",
        target_dir / "feature_stability.parquet",
    ]
    path = next(
        (candidate for candidate in candidates if table_exists(candidate)), None
    )
    if path is None:
        return []
    frame = read_table(path)
    if frame.empty:
        return []
    feature_col = _feature_column(frame)
    if feature_col is None:
        return []
    frame = frame.copy()
    frame[feature_col] = frame[feature_col].astype(str)
    groups: list[tuple[str, pd.DataFrame]] = []
    if "class_label" in frame.columns and frame["class_label"].notna().any():
        for label, group in frame.groupby("class_label", sort=True, dropna=False):
            groups.append((str(label), group.copy()))
    elif "class_index" in frame.columns and frame["class_index"].notna().any():
        for label, group in frame.groupby("class_index", sort=True, dropna=False):
            groups.append((f"class {label}", group.copy()))
    else:
        groups.append(("overall", frame))
    rows = []
    for label, group in groups:
        order_col, ascending = _feature_order(group)
        if order_col is not None:
            group[order_col] = pd.to_numeric(group[order_col], errors="coerce")
            group = group.sort_values(
                [order_col, feature_col],
                ascending=[ascending, True],
                na_position="last",
            )
        else:
            group = group.sort_values(feature_col)
        seen = set()
        selected = []
        for _, item in group.iterrows():
            feature = str(item[feature_col]).strip()
            if not feature or feature in seen:
                continue
            seen.add(feature)
            display_feature = tail_ellipsis(feature, 80)
            score = _feature_score(item)
            selected.append(
                f"{display_feature} ({score})" if score else display_feature
            )
            if len(selected) >= int(top_n):
                break
        if selected:
            rows.append((label, " · ".join(selected)))
    return rows


def _explainability_rows(sweep: Sweep) -> list[tuple[str, object]]:
    rows = []
    for child in _children(sweep):
        root = Path(child.root())
        target = _target_name(child)
        metadata = sorted((root / "explainability").glob("*/explained_unit.json"))
        for path in metadata:
            meta = _read_json(path)
            methods = meta.get("methods", [])
            method_text = (
                ", ".join(map(str, methods))
                if isinstance(methods, list)
                else str(methods)
            )
            folds = meta.get("n_outer_folds_explained", meta.get("outer_folds", ""))
            samples = meta.get("sample_count", "")
            classes = meta.get("explained_class_labels", [])
            coverage = []
            try:
                n_folds = int(folds)
            except (TypeError, ValueError):
                n_folds = 0
            if n_folds > 0:
                coverage.append(f"{n_folds} OOF folds")
            try:
                n_samples = int(samples)
            except (TypeError, ValueError):
                n_samples = 0
            if n_samples > 0:
                coverage.append(f"{n_samples} samples")
            if isinstance(classes, list) and classes:
                coverage.append(f"{len(classes)} classes")
            label = str(meta.get("target_label", meta.get("unit", path.parent.name)))
            prefix = f"{target} · {label}"
            rows.append((f"{prefix} · methods", method_text or "methods recorded"))
            rows.append(
                (
                    f"{prefix} · coverage",
                    " · ".join(coverage) or "OOF metadata available",
                )
            )
            for class_label, features in _top_feature_rows(path.parent, top_n=5):
                key = f"{prefix} · top features"
                if class_label != "overall":
                    key += f" · {class_label}"
                rows.append((key, features))
    return rows


def _robustness_rows(sweep: Sweep) -> list[tuple[str, object]]:
    rows = []
    for child in _children(sweep):
        root = Path(child.root()) / "robustness"
        target = _target_name(child)
        for filename, label in (
            ("covariate_balance.parquet", "covariates audited"),
            ("subgroup_performance.parquet", "subgroup rows"),
            ("important_feature_robustness.parquet", "feature-stability rows"),
        ):
            path = root / filename
            if table_exists(path):
                frame = read_table(path)
                rows.append((f"{target} · {label}", int(len(frame))))
    return rows


def _report_rows(sweep: Sweep) -> list[tuple[str, object]]:
    index = Path(sweep.root()) / "report" / "index.html"
    return [("Status", "HTML report generated")] if index.exists() else []


def print_stage_results(sweep: Sweep, stage_name: str) -> None:
    try:
        if stage_name == "evaluate":
            rows = _evaluation_rows(sweep)
            title = "Evaluation results ready"
        elif stage_name == "ensemble":
            rows = _ensemble_rows(sweep)
            title = "Ensemble results ready"
        elif stage_name == "explain":
            rows = _explainability_rows(sweep)
            title = "Explainability results ready"
        elif stage_name == "robustness":
            rows = _robustness_rows(sweep)
            title = "Robustness results ready"
        elif stage_name == "report":
            rows = _report_rows(sweep)
            title = "Report results ready"
        else:
            return
        if rows:
            summary_table(title, rows)
    except Exception as exc:
        warn(f"Stage results summary unavailable · {exc}")
