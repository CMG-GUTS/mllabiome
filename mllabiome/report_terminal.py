from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .configs_sweep import Sweep
from .console import console, stage
from .report_explainability import _read_table, _target_dirs_by_label
from .storage import table_exists
from .utils import feature_tail_ellipsis


def _html_inline(value: Any) -> str:

    text = str(value)

    replacements = {
        r"$\arcsin\sqrt{x}$": "arcsin√x",
        r"$\sqrt{x}$": "√x",
        r"$\ln(1+x)$": "ln(1+x)",
        r"$\log_{10}(1+x)$": "log10(1+x)",
        r"$\log_2(1+x)$": "log2(1+x)",
        r"CLR-$\epsilon$+z": "CLR-ε+z",
        r"CLR-$\epsilon$": "CLR-ε",
        r"$10^{-10}$": "10⁻¹⁰",
        r"F1$_{w}$": "F1w",
        r"F1$_w$": "F1w",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace(r"\textsubscript{w}", "w")

    text = text.replace(r"\epsilon", "ε")

    text = text.replace(r"\arcsin", "arcsin")

    text = text.replace(r"\sqrt{x}", "√x")

    text = text.replace(r"\ln", "ln")

    text = text.replace(r"\log_{10}", "log10")

    text = text.replace(r"\log_2", "log2")

    text = text.replace("$", "")

    return text


def _strip_cell_markup(value: Any) -> str:

    text = _html_inline(value)

    text = text.replace("<strong>", "").replace("</strong>", "")

    text = text.replace(r"\textbf{", "").replace("}", "")

    text = text.replace(r"$\pm$", " ± ").replace(r"\pm", "±")

    text = text.replace("F1$_w$", "F1w").replace("F1$_{w}$", "F1w")

    return text


def _terminal_table(
    title: str, df: pd.DataFrame, *, max_rows: int | None = None
) -> None:

    if df.empty:
        return

    from rich.table import Table

    tab = Table(title=title, show_lines=False)

    for col in df.columns:
        tab.add_column(str(col), overflow="fold", no_wrap=False)

    table_preview = df.head(max_rows).copy() if max_rows is not None else df.copy()

    for _, row in table_preview.iterrows():
        tab.add_row(
            *[_strip_cell_markup(row.get(col, "")) for col in table_preview.columns]
        )

    console.print(tab)


def _compact_procedure_for_terminal(procedure: pd.DataFrame) -> pd.DataFrame:

    if procedure.empty:
        return procedure

    keep = {
        "Task",
        "Procedure",
        "Samples",
        "Classes",
        "Outer folds",
        "Inner folds",
        "Repeats",
        "Selection metric",
        "Qualification gate",
    }

    out = procedure[procedure["Field"].isin(keep)].copy()

    order = {
        k: i
        for i, k in enumerate(
            [
                "Task",
                "Procedure",
                "Samples",
                "Classes",
                "Outer folds",
                "Inner folds",
                "Repeats",
                "Selection metric",
                "Qualification gate",
            ]
        )
    }

    out["_order"] = out["Field"].map(order).fillna(999)

    return out.sort_values("_order").drop(columns="_order")


def _terminal_feature_label(value: Any) -> str:

    text = str(value).split("___")[-1].split("|")[-1]

    rank = ""

    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if text.startswith(pfx):
            rank = f"{pfx[0]}. "

            text = text[len(pfx) :]

            break

    text = text.replace("_", " ").strip()

    return f"{rank}{text}" if text else str(value).replace("_", " ")


def _short_feature_label(feature: Any, max_len: int = 64) -> str:

    text = str(feature)

    if text.startswith("ALR[") and text.endswith("]"):
        body = text[4:-1]

        if "/" in body:
            numerator, reference = (part.strip() for part in body.split("/", 1))

            label = f"ALR[{_terminal_feature_label(numerator)} / {_terminal_feature_label(reference)}]"

        else:
            label = text

    elif text.startswith("ILR_"):
        parts = text.split("_", 2)

        if len(parts) == 3 and parts[1].isdigit():
            label = f"ILR balance {int(parts[1])} · {parts[2][:10]}"

        else:
            label = text.replace("_", " ")

    else:
        label = _terminal_feature_label(text)

    return feature_tail_ellipsis(label, max_len)


def _target_dirs_for_terminal(root: Path) -> list[tuple[str, Path]]:

    dirs = _target_dirs_by_label(root)

    ordered = [
        (label, dirs[label])
        for label in ("MPMA-E", "MPMA-B", "Baseline RF")
        if label in dirs
    ]

    seen = {path.resolve() for _, path in ordered}

    explainability_root = root / "explainability"

    if explainability_root.exists():
        for path in sorted(x for x in explainability_root.iterdir() if x.is_dir()):
            if path.resolve() in seen:
                continue

            meta = {}

            meta_path = path / "explained_unit.json"

            if meta_path.exists():
                try:
                    obj = json.loads(meta_path.read_text(encoding="utf-8"))

                    if isinstance(obj, dict):
                        meta = obj

                except Exception:
                    meta = {}

            label = str(
                meta.get(
                    "target_label",
                    meta.get("type", meta.get("unit", path.name.replace("_", " "))),
                )
            ).strip()

            ordered.append((label or path.name.replace("_", " "), path))

            seen.add(path.resolve())

    return ordered


def _feature_support_table(root: Path, *, top_n: int = 8) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    for label, target_dir in _target_dirs_for_terminal(root):
        candidates = (
            target_dir / "top_features.parquet",
            target_dir / "top_features_shap.parquet",
            target_dir / "feature_importance.parquet",
            target_dir / "feature_stability.parquet",
        )

        tab = pd.DataFrame()

        for path in candidates:
            if table_exists(path):
                tab = _read_table(path)

                if not tab.empty and "feature" in tab.columns:
                    break

        if tab.empty or "feature" not in tab.columns:
            continue

        if "rank" in tab.columns:
            tab = tab.sort_values("rank", ascending=True)

        elif "mean_rank" in tab.columns:
            tab = tab.sort_values(["mean_rank", "feature"], ascending=[True, True])

        elif "importance_mean" in tab.columns:
            tab = tab.sort_values(
                ["importance_mean", "feature"], ascending=[False, True]
            )

        methods = [
            c
            for c in (
                "SHAP",
                "LIME",
                "Permutation",
                "ALE",
                "shap",
                "lime",
                "permutation",
                "ale",
                "consensus",
            )
            if c in tab.columns
        ]

        for local_rank, (_, row) in enumerate(tab.head(top_n).iterrows(), start=1):
            support = ""

            if "mean_rank" in tab.columns:
                value = pd.to_numeric(
                    pd.Series([row.get("mean_rank")]), errors="coerce"
                ).iloc[0]

                if pd.notna(value):
                    support = f"mean rank {float(value):.2f}"

            if not support and methods:
                vals = pd.to_numeric(
                    pd.Series([row.get(c) for c in methods]), errors="coerce"
                ).dropna()

                if not vals.empty:
                    support = f"{float(vals.mean()):.3f}"

            if not support and "top_k_frequency" in tab.columns:
                value = pd.to_numeric(
                    pd.Series([row.get("top_k_frequency")]), errors="coerce"
                ).iloc[0]

                if pd.notna(value):
                    support = f"top-k {float(value):.3f}"

            if not support and "importance_mean" in tab.columns:
                value = pd.to_numeric(
                    pd.Series([row.get("importance_mean")]), errors="coerce"
                ).iloc[0]

                if pd.notna(value):
                    support = f"importance {float(value):.3g}"

            raw_rank = row.get("rank", local_rank)

            try:
                rank = int(float(raw_rank)) if pd.notna(raw_rank) else local_rank

            except (TypeError, ValueError):
                rank = local_rank

            rows.append(
                {
                    "Strategy": label,
                    "Rank": rank,
                    "Feature": _short_feature_label(row.get("feature", "")),
                    "Support": support,
                }
            )

    return pd.DataFrame(rows, columns=["Strategy", "Rank", "Feature", "Support"])


def _feature_support_terminal(root: Path, *, top_n: int = 8) -> None:

    table = _feature_support_table(root, top_n=top_n)

    if not table.empty:
        _terminal_table("Top explainability features", table, max_rows=len(table))


def _hardware_summary_table(environment: dict[str, Any] | None) -> pd.DataFrame:

    env = environment or {}

    rows: list[dict[str, str]] = []

    def add(field: str, value: Any) -> None:

        if value is None:
            return

        text = str(value).strip()

        if text and text.lower() not in {"none", "nan"}:
            rows.append({"Field": field, "Value": text})

    add("CPU model", env.get("cpu_model"))

    physical = env.get("physical_cpus")

    logical = env.get("logical_cpus")

    if physical not in (None, "") or logical not in (None, ""):
        ptxt = str(physical) if physical not in (None, "") else "?"

        ltxt = str(logical) if logical not in (None, "") else "?"

        add("CPU topology", f"{ptxt} physical / {ltxt} logical CPUs")

    for key, label in (
        ("memory_available_bytes", "Available memory"),
        ("memory_total_bytes", "Total memory"),
    ):
        value = env.get(key)

        try:
            number = float(value)

        except (TypeError, ValueError):
            number = float("nan")

        if np.isfinite(number):
            add(label, f"{number / (1024**3):.3f} GiB")

    workers = env.get("evaluation_workers")

    threads = env.get("threads_per_worker")

    if workers not in (None, "") or threads not in (None, ""):
        wtxt = str(workers) if workers not in (None, "") else "?"

        ttxt = str(threads) if threads not in (None, "") else "?"

        add("Evaluation parallelism", f"{wtxt} workers × {ttxt} threads/worker")

    add("Platform", env.get("platform"))

    add("Python", env.get("python_version"))

    return pd.DataFrame(rows, columns=["Field", "Value"])


def _print_report_summary(
    sweep: Sweep,
    root: Path,
    procedure: pd.DataFrame,
    strategy_html: pd.DataFrame,
    top_mpmas: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
    ensemble_members: pd.DataFrame,
    *,
    primary_pairwise: pd.DataFrame | None = None,
    compute_display: pd.DataFrame | None = None,
    compute_environment: dict[str, Any] | None = None,
) -> None:

    stage("Run summary", str(root))

    _terminal_table(
        "Task definition and evaluation procedure",
        _compact_procedure_for_terminal(procedure),
    )

    _terminal_table("Task performance", strategy_html)

    if primary_pairwise is not None and not primary_pairwise.empty:
        _terminal_table("Inferential comparisons", primary_pairwise, max_rows=10)

    if not top_mpmas.empty:
        identity_priority = [
            "Rank",
            "Family",
            "Modalities",
            "Integration",
            "Representation",
            "Taxonomic representation",
            "Resolution",
            "Transformation",
            "Count transformation",
            "Learner",
        ]

        metric_cols = [
            c
            for c in top_mpmas.columns
            if c.startswith("Inner") or c.startswith("Outer")
        ]

        cols = [c for c in identity_priority if c in top_mpmas.columns]

        cols.extend([c for c in metric_cols if c not in cols][:4])

        _terminal_table(
            "Top MPMA-B configurations",
            top_mpmas[cols] if cols else top_mpmas,
            max_rows=5,
        )

    _terminal_table("Final MPMA-E specification", ensemble_summary)

    if not ensemble_members.empty:
        cols = [
            c
            for c in [
                "Member",
                "Taxonomic representation",
                "Resolution",
                "Count transformation",
                "Learner type",
                "count_transformation",
                "learner",
            ]
            if c in ensemble_members.columns
        ]

        _terminal_table(
            "Final MPMA-E members", ensemble_members[cols] if cols else ensemble_members
        )

    _feature_support_terminal(root, top_n=8)

    if compute_display is not None and not compute_display.empty:
        _terminal_table("Computational resources", compute_display)

    hardware = _hardware_summary_table(compute_environment)

    if not hardware.empty:
        _terminal_table("Hardware and runtime", hardware)
