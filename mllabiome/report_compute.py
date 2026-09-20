from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .storage import read_table, write_table, table_exists
from .utils import dump_json_standard


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _eligible_config_ids(configs: pd.DataFrame, plan: Any) -> set[str]:
    if configs.empty or "config_id" not in configs.columns:
        return set()
    frame = configs.copy()
    if not bool(getattr(plan, "include_inactive", False)) and "active" in frame.columns:
        active = pd.to_numeric(frame["active"], errors="coerce").fillna(0).astype(int)
        frame = frame[active.eq(1)]
    mask = pd.Series(False, index=frame.index, dtype=bool)
    ids = {str(x) for x in getattr(plan, "exclude_config_ids", ()) if str(x)}
    if ids:
        mask |= frame["config_id"].astype(str).isin(ids)
    fields = (
        ("learner", "exclude_learners"),
        ("resolution", "exclude_resolutions"),
        ("count_transformation", "exclude_transformations"),
    )
    for column, attr in fields:
        values = {str(x).casefold() for x in getattr(plan, attr, ()) if str(x)}
        if values and column in frame.columns:
            mask |= frame[column].astype(str).str.casefold().isin(values)
    return set(frame.loc[~mask, "config_id"].astype(str))


def _outer_keys(root: Path, resources: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    if not resources.empty and "split_key" in resources.columns:
        keys |= set(resources["split_key"].dropna().astype(str))
    for path in (
        root / "results" / "outer_results.parquet",
        root / "tables" / "qualification_gate.parquet",
    ):
        frame = _read_table(path)
        if not frame.empty and "split_key" in frame.columns:
            keys |= set(frame["split_key"].dropna().astype(str))
    return keys


def _resource_summary(
    resources: pd.DataFrame, config_ids: set[str], outer_keys: set[str]
) -> dict[str, Any]:
    if resources.empty or not config_ids:
        return {}
    frame = resources.copy()
    frame["config_id"] = frame["config_id"].astype(str)
    frame["split_key"] = frame["split_key"].astype(str)
    frame = frame[frame["config_id"].isin(config_ids)]
    if outer_keys:
        frame = frame[frame["split_key"].isin(outer_keys)]
    if frame.empty:
        return {}
    pairs = frame[["config_id", "split_key"]].drop_duplicates()
    expected = len(config_ids) * len(outer_keys) if outer_keys else len(pairs)
    coverage = float(len(pairs) / expected) if expected else float("nan")
    cpu = pd.to_numeric(frame.get("cpu_core_hours"), errors="coerce").fillna(0.0)
    wall = pd.to_numeric(frame.get("wall_time_s"), errors="coerce").fillna(0.0)
    fits = pd.to_numeric(frame.get("fits"), errors="coerce").fillna(0).astype(int)
    rss = pd.to_numeric(frame.get("peak_rss_gib"), errors="coerce")
    return {
        "candidate_configs": int(len(config_ids)),
        "outer_units": int(len(outer_keys)),
        "resource_jobs": int(len(pairs)),
        "expected_jobs": int(expected),
        "coverage_fraction": coverage,
        "model_fits": int(fits.sum()),
        "evaluation_cpu_core_hours": float(cpu.sum()),
        "evaluation_job_wall_hours_sum": float(wall.sum() / 3600.0),
        "peak_job_rss_gib": float(rss.max()) if rss.notna().any() else np.nan,
    }


def _overhead(path: Path) -> dict[str, float]:
    data = _read_json(path)
    return {
        "cpu_core_hours": float(data.get("cpu_core_hours", 0.0) or 0.0),
        "wall_time_s": float(data.get("wall_time_s", 0.0) or 0.0),
        "peak_rss_gib": float(data.get("peak_rss_gib", 0.0) or 0.0),
    }


def _strategy_config_ids(strategy_rows: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for row in strategy_rows:
        strategy = str(row.get("Strategy", "")).strip()
        config_id = str(row.get("config_id", "")).strip()
        if strategy and config_id:
            out[strategy] = config_id
    return out


def compute_display(frame: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame()
    if not frame.empty:
        display = frame[
            [
                "Strategy",
                "attributable_cpu_core_hours",
                "model_fits",
                "peak_rss_gib",
                "coverage_fraction",
            ]
        ].copy()
        display = display.rename(
            columns={
                "attributable_cpu_core_hours": "CPU core-hours",
                "model_fits": "Model fits",
                "peak_rss_gib": "Peak job RAM (GiB)",
                "coverage_fraction": "Accounting coverage",
            }
        )
        display["CPU core-hours"] = pd.to_numeric(
            display["CPU core-hours"], errors="coerce"
        ).map(lambda x: f"{x:.3f}" if np.isfinite(x) else "")
        display["Peak job RAM (GiB)"] = pd.to_numeric(
            display["Peak job RAM (GiB)"], errors="coerce"
        ).map(lambda x: f"{x:.3f}" if np.isfinite(x) else "")
        display["Accounting coverage"] = pd.to_numeric(
            display["Accounting coverage"], errors="coerce"
        ).map(lambda x: f"{100 * x:.3f}%" if np.isfinite(x) else "")
    return display


def run_compute_accounting(
    root: Path, sweep: Any, strategy_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    tables_dir = root / "report" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    resources = _read_table(root / "tables" / "job_resources.parquet")
    configs = _read_table(root / "configs.parquet")
    if resources.empty or configs.empty:
        empty = pd.DataFrame()
        compute_path = tables_dir / "strategy_compute.parquet"
        write_table(compute_path, empty)
        manifest_path = tables_dir / "compute_accounting_manifest.json"
        dump_json_standard(
            {
                "available": False,
                "reason": "No per-job resource accounting is available for this run.",
            },
            manifest_path,
        )
        return {
            "strategy_compute": empty,
            "display": empty,
            "compute_path": compute_path,
            "manifest_path": manifest_path,
            "environment": {},
        }
    outer_keys = _outer_keys(root, resources)
    pool_ids = _eligible_config_ids(configs, sweep.ensemble)
    fixed_ids = _strategy_config_ids(strategy_rows)
    mpma_b_overhead = _overhead(root / "tables" / "mpma_b_selection_resources.json")
    mpma_e_overhead = _overhead(root / "ensembling" / "mpma_e_selection_resources.json")
    rows = []
    for strategy in ("MPMA-E", "MPMA-B", "AutoML", "Baseline RF", "SIAMCAT"):
        if strategy == "MPMA-E":
            if not table_exists(root / "ensembling" / "ensemble_predictions.parquet"):
                continue
            ids = pool_ids
            overhead = mpma_e_overhead
            scope = "eligible MPMA search pool + ensemble selection"
        elif strategy == "MPMA-B":
            ids = pool_ids
            overhead = mpma_b_overhead
            scope = "eligible MPMA search pool + MPMA-B selection"
        else:
            config_id = fixed_ids.get(strategy, "")
            if not config_id:
                continue
            ids = {config_id}
            overhead = {"cpu_core_hours": 0.0, "wall_time_s": 0.0, "peak_rss_gib": 0.0}
            scope = "reported comparator configuration"
        base = _resource_summary(resources, ids, outer_keys)
        if not base:
            continue
        overhead_cpu = float(overhead.get("cpu_core_hours", 0.0))
        overhead_wall = float(overhead.get("wall_time_s", 0.0)) / 3600.0
        overhead_rss = float(overhead.get("peak_rss_gib", 0.0))
        base_rss = float(base.get("peak_job_rss_gib", np.nan))
        peak = (
            np.nanmax([base_rss, overhead_rss])
            if np.isfinite(base_rss) or overhead_rss > 0
            else np.nan
        )
        rows.append(
            {
                "Strategy": strategy,
                "accounting_scope": scope,
                **base,
                "selection_cpu_core_hours": overhead_cpu,
                "selection_wall_hours": overhead_wall,
                "attributable_cpu_core_hours": float(
                    base["evaluation_cpu_core_hours"] + overhead_cpu
                ),
                "attributable_job_wall_hours_sum": float(
                    base["evaluation_job_wall_hours_sum"] + overhead_wall
                ),
                "peak_rss_gib": float(peak) if np.isfinite(peak) else np.nan,
                "complete_accounting": bool(
                    np.isfinite(base["coverage_fraction"])
                    and base["coverage_fraction"] >= 0.999999
                ),
            }
        )
    frame = pd.DataFrame(rows)
    compute_path = tables_dir / "strategy_compute.parquet"
    write_table(compute_path, frame)
    run_summary = _read_json(root / "run_summary.json")
    environment = (
        run_summary.get("machine", {})
        if isinstance(run_summary.get("machine"), dict)
        else {}
    )
    environment = {
        **environment,
        "evaluation_workers": run_summary.get("workers"),
        "threads_per_worker": run_summary.get("threads_per_worker"),
        "latest_evaluation_invocation_wall_time_s": run_summary.get("elapsed_s"),
    }
    manifest = {
        "available": not frame.empty,
        "primary_compute_measure": "CPU core-hours = summed process and child-process CPU time / 3600",
        "wall_time_measure": "Per-job wall times are summed only as job-hours; they are not elapsed wall-clock time under parallel execution.",
        "memory_measure": "Peak resident set size is sampled for each Python worker process tree, including child processes such as R when observable.",
        "flops_policy": "FLOPs are not reported for mixed classical-ML workflows because tree induction, branching, comparisons, memory operations, and external R routines are not represented faithfully by a portable floating-point-operation count.",
        "overlap_policy": "MPMA-B and MPMA-E share the eligible MPMA search pool. Their attributable compute values overlap and must not be summed.",
        "hardware": environment,
    }
    manifest_path = tables_dir / "compute_accounting_manifest.json"
    dump_json_standard(manifest, manifest_path)
    display = compute_display(frame)
    return {
        "strategy_compute": frame,
        "display": display,
        "compute_path": compute_path,
        "manifest_path": manifest_path,
        "environment": environment,
    }
