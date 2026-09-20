from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import report as _report
from .configs_sweep import Sweep, _normalise_sweep_task, _target_task, target_sweeps
from .console import info, path_table, phase_progress, stage, success
from .final_models import build_final_models
from .metrics import metric_is_loss
from .report_compute import run_compute_accounting
from .report_oof import _mpma_b_composition, oof_section_html, write_report
from .regression_explainability import _write_regression_explainability_figures
from .utils import REGRESSION_METRIC_COLUMNS, dump_json_standard
from .storage import read_table, write_table, table_exists


_REGRESSION_DISPLAY_METRICS = (
    "RMSE",
    "MAE",
    "R2",
    "PearsonR",
    "SpearmanR",
)


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


def _finite(values: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _bootstrap_mean(
    values: np.ndarray, seed: int, n_bootstrap: int = 2000
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(arr, size=(int(n_bootstrap), len(arr)), replace=True).mean(
        axis=1
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _holm_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    out = np.full(len(p), np.nan, dtype=float)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        adjusted = min(1.0, float(p[index]) * float(m - rank))
        running = max(running, adjusted)
        out[index] = running
    return [float(x) for x in out]


def _regression_strategy_frames(root: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    mpma_b = _read_table(root / "results" / "mpma_b_outer_results.parquet")
    if not mpma_b.empty:
        out["MPMA-B"] = mpma_b
    mpma_e = _read_table(root / "ensembling" / "mpma_e_outer_results.parquet")
    if not mpma_e.empty:
        out["MPMA-E"] = mpma_e
    return out


def _regression_statistics(root: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = _regression_strategy_frames(root)
    rows: list[dict[str, Any]] = []
    for strategy, frame in frames.items():
        for metric in REGRESSION_METRIC_COLUMNS:
            if metric not in frame.columns:
                continue
            values = _finite(frame[metric])
            if len(values) == 0:
                continue
            low, high = _bootstrap_mean(values, seed + len(rows) * 101)
            rows.append(
                {
                    "Strategy": strategy,
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci_low": low,
                    "ci_high": high,
                    "n_outer_units": int(len(values)),
                }
            )
    summary = pd.DataFrame(rows)
    pair_rows: list[dict[str, Any]] = []
    strategies = list(frames)
    for i, strategy_a in enumerate(strategies):
        for strategy_b in strategies[i + 1 :]:
            a = frames[strategy_a].copy()
            b = frames[strategy_b].copy()
            key = (
                "outer_split_key"
                if "outer_split_key" in a.columns and "outer_split_key" in b.columns
                else None
            )
            for metric in REGRESSION_METRIC_COLUMNS:
                if metric not in a.columns or metric not in b.columns:
                    continue
                if key:
                    aa = a[[key, metric]].rename(columns={metric: "a"})
                    bb = b[[key, metric]].rename(columns={metric: "b"})
                    matched = aa.merge(bb, on=key, how="inner")
                    va = pd.to_numeric(matched["a"], errors="coerce").to_numpy(
                        dtype=float
                    )
                    vb = pd.to_numeric(matched["b"], errors="coerce").to_numpy(
                        dtype=float
                    )
                else:
                    va = _finite(a[metric])
                    vb = _finite(b[metric])
                    n = min(len(va), len(vb))
                    va, vb = va[:n], vb[:n]
                mask = np.isfinite(va) & np.isfinite(vb)
                va, vb = va[mask], vb[mask]
                if len(va) == 0:
                    continue
                diff = va - vb
                low, high = _bootstrap_mean(diff, seed + 5000 + len(pair_rows) * 101)
                if len(diff) > 1 and np.any(np.abs(diff) > 0):
                    try:
                        p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
                    except Exception:
                        p_value = float("nan")
                else:
                    p_value = float("nan")
                pair_rows.append(
                    {
                        "metric": metric,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "difference_a_minus_b": float(np.mean(diff)),
                        "difference_ci_low": low,
                        "difference_ci_high": high,
                        "n_matched_outer_units": int(len(diff)),
                        "test": "paired Wilcoxon signed-rank",
                        "p_value": p_value,
                    }
                )
    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty:
        pairwise["p_holm"] = np.nan
        for metric, idx in pairwise.groupby("metric").groups.items():
            indices = list(idx)
            vals = [float(pairwise.at[i, "p_value"]) for i in indices]
            finite_positions = [j for j, value in enumerate(vals) if np.isfinite(value)]
            adjusted = _holm_adjust([vals[j] for j in finite_positions])
            for pos, value in zip(finite_positions, adjusted):
                pairwise.at[indices[pos], "p_holm"] = value
        pairwise["significant_holm_0_05"] = pd.to_numeric(
            pairwise["p_holm"], errors="coerce"
        ).lt(0.05)
    return summary, pairwise


def _format_metric(
    metric: str, estimate: Any, std: Any, low: Any, high: Any, bold: bool = False
) -> str:
    values = []
    for value in (estimate, std, low, high):
        try:
            value = float(value)
        except Exception:
            value = float("nan")
        values.append(value)
    estimate_f, std_f, low_f, high_f = values
    if not np.isfinite(estimate_f):
        return "—"
    body = f"{estimate_f:.3f}"
    if np.isfinite(std_f):
        body += f" ± {std_f:.3f}"
    if np.isfinite(low_f) and np.isfinite(high_f):
        body += f" [{low_f:.3f}, {high_f:.3f}]"
    return f"<strong>{body}</strong>" if bold else body


def _regression_performance_table(statistics: pd.DataFrame) -> pd.DataFrame:
    if statistics.empty:
        return pd.DataFrame()
    strategies = statistics["Strategy"].dropna().astype(str).drop_duplicates().tolist()
    best: dict[str, str] = {}
    for metric in _REGRESSION_DISPLAY_METRICS:
        sub = statistics[statistics["metric"].astype(str).eq(metric)].copy()
        sub["estimate"] = pd.to_numeric(sub["estimate"], errors="coerce")
        sub = sub[np.isfinite(sub["estimate"].to_numpy(dtype=float))]
        if sub.empty:
            continue
        sub = sub.sort_values("estimate", ascending=metric_is_loss(metric))
        best[metric] = str(sub.iloc[0]["Strategy"])
    rows = []
    for strategy in strategies:
        row: dict[str, Any] = {"Strategy": strategy}
        for metric in _REGRESSION_DISPLAY_METRICS:
            sub = statistics[
                statistics["Strategy"].astype(str).eq(strategy)
                & statistics["metric"].astype(str).eq(metric)
            ]
            if sub.empty:
                row[metric] = "—"
                continue
            item = sub.iloc[0]
            row[metric] = _format_metric(
                metric,
                item.get("estimate"),
                item.get("std"),
                item.get("ci_low"),
                item.get("ci_high"),
                bold=best.get(metric) == strategy,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _regression_mean_std_cell(mean: Any, std: Any) -> str:
    try:
        mean_f = float(mean)
    except Exception:
        mean_f = float("nan")
    try:
        std_f = float(std)
    except Exception:
        std_f = float("nan")
    if not np.isfinite(mean_f):
        return "—"
    if np.isfinite(std_f):
        return f"{mean_f:.3f} ± {std_f:.3f}"
    return f"{mean_f:.3f}"


def _regression_top_mpmas(root: Path, metric: str, n: int = 5) -> pd.DataFrame:
    inner = _read_table(root / "inner_results" / "inner_results.parquet")
    outer = _read_table(root / "results" / "outer_results.parquet")
    if inner.empty:
        return pd.DataFrame()
    if "ok" in inner.columns:
        inner = inner[
            pd.to_numeric(inner["ok"], errors="coerce").fillna(0).astype(int).eq(1)
        ]
    if "ok" in outer.columns and not outer.empty:
        outer = outer[
            pd.to_numeric(outer["ok"], errors="coerce").fillna(0).astype(int).eq(1)
        ]
    ids = [
        c
        for c in (
            "config_id",
            "candidate_family",
            "modalities",
            "integration",
            "integration_n_components",
            "count_transformation",
            "resolution",
            "learner",
        )
        if c in inner.columns
    ]
    available = [c for c in _REGRESSION_DISPLAY_METRICS if c in inner.columns]
    if metric not in available:
        metric = (
            "RMSE" if "RMSE" in available else (available[0] if available else metric)
        )
    metric_order = [metric] + [c for c in available if c != metric]
    inner_agg = (
        inner.groupby(ids, dropna=False)[metric_order]
        .agg(["mean", "std"])
        .reset_index()
    )
    inner_agg.columns = [
        c if isinstance(c, str) else c[0] if not c[1] else f"inner_{c[0]}_{c[1]}"
        for c in inner_agg.columns
    ]
    if not outer.empty:
        outer_metrics = [c for c in metric_order if c in outer.columns]
        outer_agg = (
            outer.groupby("config_id", dropna=False)[outer_metrics]
            .agg(["mean", "std"])
            .reset_index()
        )
        outer_agg.columns = [
            c if isinstance(c, str) else c[0] if not c[1] else f"outer_{c[0]}_{c[1]}"
            for c in outer_agg.columns
        ]
        inner_agg = inner_agg.merge(outer_agg, on="config_id", how="left")
    sort_col = f"inner_{metric}_mean"
    if sort_col in inner_agg.columns:
        inner_agg = inner_agg.sort_values(sort_col, ascending=metric_is_loss(metric))
    rows = []
    labels = {"R2": "R²", "PearsonR": "Pearson r", "SpearmanR": "Spearman ρ"}
    for rank, (_, item) in enumerate(inner_agg.head(int(n)).iterrows(), start=1):
        row: dict[str, Any] = {"Rank": rank}
        if "candidate_family" in inner_agg.columns:
            row["Family"] = item.get("candidate_family", "")
            row["Modalities"] = item.get("modalities", "")
            row["Integration"] = item.get("integration", "")
            components = item.get("integration_n_components", "")
            row["Components"] = "" if pd.isna(components) else components
            row["Representation"] = item.get("resolution", "")
            row["Transformation"] = item.get("count_transformation", "")
        else:
            row["Resolution"] = item.get("resolution", "")
            row["Count transformation"] = item.get("count_transformation", "")
        row["Learner"] = item.get("learner", "")
        for name in metric_order:
            label = labels.get(name, name)
            row[f"Inner {label}"] = _regression_mean_std_cell(
                item.get(f"inner_{name}_mean"), item.get(f"inner_{name}_std")
            )
            row[f"Outer {label}"] = _regression_mean_std_cell(
                item.get(f"outer_{name}_mean"), item.get(f"outer_{name}_std")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _regression_procedure(sweep: Sweep, root: Path) -> pd.DataFrame:
    manifest = _read_json(root / "manifest.json")
    ev = sweep.evaluation
    gate = sweep.gate
    target = (
        sweep.samples.target_col
        if getattr(sweep, "uses_modalities", False)
        else sweep.data.target_col
    )
    rows = [
        ["Task", sweep.title],
        ["Task type", "Regression"],
        ["Target", str(target)],
        ["Experiment directory", str(root)],
        ["Samples", str(manifest.get("n_samples", ""))],
    ]
    if getattr(sweep, "uses_modalities", False):
        rows.extend(
            [
                [
                    "Data format",
                    "multimodal" if len(sweep.modalities) > 1 else "single-modality",
                ],
                ["Primary modality", str(manifest.get("primary_modality", ""))],
                [
                    "Modalities",
                    ", ".join(str(x) for x in manifest.get("modalities", {}).keys()),
                ],
            ]
        )
    rows.extend(
        [
            ["Procedure", ev.protocol],
            [
                "Outer folds",
                ev.outer_folds
                if ev.protocol not in {"lodo", "leave_one_dataset_out"}
                else "held-out datasets",
            ],
            [
                "Inner folds",
                ev.inner_folds
                if ev.protocol not in {"lodo", "leave_one_dataset_out"}
                else "leave-one-dataset-out across outer-training datasets",
            ],
            ["Repeats", ev.repeats],
            ["Selection metric", ev.optimize_metric],
            ["Random seed", ev.random_state],
            [
                "Qualification gate",
                f"on ({gate.metric} threshold {gate.threshold})"
                if gate.enabled
                else "off",
            ],
            [
                "Selection rule",
                "Model and ensemble selection use inner-validation performance; reported performance uses held-out outer evaluation predictions.",
            ],
        ]
    )
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _regression_outer_unit_table(root: Path) -> pd.DataFrame:
    frames = []
    for strategy, frame in _regression_strategy_frames(root).items():
        if frame.empty:
            continue
        out = frame.copy()
        if "outer_split_key" not in out.columns and "split_key" in out.columns:
            out["outer_split_key"] = out["split_key"].astype(str)
        out.insert(0, "Strategy", strategy)
        keep = [
            c
            for c in ("Strategy", "outer_split_key", *REGRESSION_METRIC_COLUMNS)
            if c in out.columns
        ]
        frames.append(out[keep])
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


def _regression_fixed(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{int(digits)}f}"


def _regression_numeric_table(table: pd.DataFrame, limit: int) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    out = table.head(int(limit)).copy()
    rename = {}
    for column in out.columns:
        text = str(column).replace("_", " ").strip()
        rename[column] = text[:1].upper() + text[1:] if text else str(column)
    out = out.rename(columns=rename)
    for column in out.columns:
        if column.casefold() == "feature":
            out[column] = out[column].map(_report._short_feature_label)
            continue
        original = next((src for src, dst in rename.items() if dst == column), column)
        numeric = (
            pd.to_numeric(table[original].head(int(limit)), errors="coerce")
            if original in table.columns
            else pd.Series(dtype=float)
        )
        if len(numeric) == len(out) and numeric.notna().any():
            if numeric.dropna().map(lambda x: float(x).is_integer()).all():
                out[column] = numeric.map(lambda x: "" if pd.isna(x) else str(int(x)))
            else:
                out[column] = numeric.map(lambda x: _regression_fixed(x, 3))
    return out


def _regression_explainability_blocks(
    root: Path, report_dir: Path, top_k: int
) -> tuple[str, int]:
    parts: list[str] = []
    count = 0
    for slug, label in (("mpma_b", "MPMA-B"), ("mpma_e", "MPMA-E")):
        target_dir = root / "explainability" / slug
        stability = _read_table(target_dir / "feature_stability.parquet")
        combined = _read_table(target_dir / "feature_importance.parquet")
        meta = _read_json(target_dir / "explained_unit.json")
        if not target_dir.exists() or (stability.empty and combined.empty):
            continue
        count += 1
        methods = [str(x) for x in meta.get("methods", []) if str(x)]
        parts.append(f'<section class="xai-target"><h3>{html.escape(label)}</h3>')
        if methods:
            parts.append(
                f"<p>Cross-fitted OOF regression explanations of the final selected specification. Methods: {html.escape(', '.join(methods))}.</p>"
            )
        _write_regression_explainability_figures(target_dir, stability, int(top_k))
        figure = _report._fig(
            target_dir / "figures" / "feature_support",
            report_dir,
            f"{label}: cross-method top-k support and fold stability",
        )
        if figure:
            parts.append("<h4>Global explanations</h4>")
            parts.append(
                "<p>Top-k support is a within-method rank score: rank 1 scores 1, rank k scores 1/k, and ranks below k score 0. Fold stability is the fraction of estimable outer folds in which a feature ranks within the method-specific top k. Both are scale-free 0–1 summaries; method-specific effect magnitudes are not compared across methods.</p>"
            )
            parts.append(figure)
        curves = _read_table(target_dir / "ale_curves.parquet")
        if not curves.empty:
            parts.append("<h4>ALE</h4>")
            parts.append(
                "<p>Accumulated local effects summarize the shape of the fitted regression response over the observed feature range.</p>"
            )
            curve_figure = _report._fig(
                target_dir / "figures" / "ale_curves",
                report_dir,
                f"{label}: ALE effect curves",
            )
            if curve_figure:
                parts.append(curve_figure)
            parts.append(
                "<p>The x-axis is the model-input feature coordinate and the y-axis is the centered ALE effect in target units. Positive values indicate predictions above the feature's average local effect and negative values indicate predictions below it. The thick curve is the cross-fold median; the band shows the interquartile range across outer folds where supported.</p>"
            )
            parts.append(
                "<p>Per-fold numerical ALE coordinates are retained in the technical artifact <code>ale_curves.parquet</code>.</p>"
            )
        interactions = _read_table(target_dir / "ale_interactions.parquet")
        if not interactions.empty:
            parts.append("<h4>ALE interactions</h4>")
            parts.append(
                "<p>Exploratory OOF 2D ALE interaction strengths. Edge weight represents interaction magnitude.</p>"
            )
            for stem, caption in (
                ("interaction_network_current", "2D ALE interaction network"),
            ):
                block = _report._fig(
                    target_dir / "figures" / stem, report_dir, f"{label}: {caption}"
                )
                if block:
                    parts.append(block)
            parts.append("<h5>Interaction strengths</h5>")
            parts.append(
                _report._html_table(_regression_numeric_table(interactions, int(top_k)))
            )
        local_mode = str(meta.get("local_explanations", "none")).strip() or "none"
        parts.append("<h4>Local explanations</h4>")
        if local_mode == "none":
            parts.append(
                "<p>Local sample-level reporting was not requested for this run.</p>"
            )
        else:
            local_figure = _report._xai_local_figure(
                target_dir, report_dir, label, "regression"
            )
            if local_figure:
                parts.append(
                    "<p>Representative held-out samples are selected from the OOF prediction distribution at the lower, central, and upper response ranges. Each sample is explained only by outer-fold model(s) that did not train on that sample. SHAP attributions and LIME local-surrogate coefficients are shown side by side when both are available. The cross-method panel combines attribution direction with within-method reciprocal-rank support and does not average raw SHAP and LIME magnitudes. The support count indicates how many local methods place the feature within the displayed top set.</p>"
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
    return "".join(parts), count


def write_regression_report(sweep: Sweep) -> dict[str, Path]:
    root = Path(sweep.root())
    source = root / "results" / "outer_results.parquet"
    if not table_exists(source):
        raise FileNotFoundError("Run evaluate(sweep) before reporting.")
    if (
        not (root / "final_models.json").exists()
        and (root / "tables" / "mpma_b_final_candidate.json").exists()
    ):
        build_final_models(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stage("Report", str(report_dir))
    with phase_progress("Regression report analysis", 5) as phase:
        phase.phase("procedure and statistics")
        metric = str(sweep.evaluation.optimize_metric)
        procedure = _regression_procedure(sweep, root)
        statistics, pairwise = _regression_statistics(
            root, int(sweep.evaluation.random_state)
        )
        performance = _regression_performance_table(statistics)
        phase.phase("rankings and ensemble summaries")
        top_mpmas = _regression_top_mpmas(root, metric, 5)
        ensemble_summary = _report._ensemble_summary_table(root)
        ensemble_members = _report._ensemble_members_table(root)
        strategy_rows = [{"Strategy": x} for x in _regression_strategy_frames(root)]
        phase.phase("compute accounting")
        compute = run_compute_accounting(root, sweep, strategy_rows)
        compute_display = compute.get("display", pd.DataFrame())
        phase.phase("outer-unit summaries")
        outer_units = _regression_outer_unit_table(root)
        phase.phase("explainability visuals")
        explainability_html, explainability_count = _regression_explainability_blocks(
            root, report_dir, int(getattr(sweep.explainability, "top_k", 15))
        )
    primary_pairwise = (
        pairwise[pairwise["metric"].astype(str).eq(metric)].copy()
        if not pairwise.empty and "metric" in pairwise.columns
        else pd.DataFrame()
    )
    if not primary_pairwise.empty:
        primary_pairwise = primary_pairwise.rename(
            columns={
                "strategy_a": "Strategy A",
                "strategy_b": "Strategy B",
                "difference_a_minus_b": "Difference A-B",
                "difference_ci_low": "95% CI low",
                "difference_ci_high": "95% CI high",
                "n_matched_outer_units": "Matched outer units",
                "test": "Test",
                "p_value": "p",
                "p_holm": "Holm p",
                "significant_holm_0_05": "Holm p<0.05",
            }
        )
        for column in ("Difference A-B", "95% CI low", "95% CI high"):
            if column in primary_pairwise.columns:
                primary_pairwise[column] = pd.to_numeric(
                    primary_pairwise[column], errors="coerce"
                ).map(lambda x: f"{x:.3f}" if np.isfinite(x) else "")
        for column in ("p", "Holm p"):
            if column in primary_pairwise.columns:
                primary_pairwise[column] = pd.to_numeric(
                    primary_pairwise[column], errors="coerce"
                ).map(lambda x: _report._format_p_value(x))
    procedure_path = tables_dir / "evaluation_procedure.parquet"
    statistics_path = tables_dir / "strategy_metrics_bootstrap.parquet"
    pairwise_path = tables_dir / "strategy_pairwise_tests.parquet"
    performance_path = tables_dir / "task_strategy_performance.parquet"
    top_path = tables_dir / "top5_mpma_inner_outer_performance.parquet"
    selection_path = tables_dir / "mpma_e_selection.parquet"
    members_path = tables_dir / "mpma_e_members.parquet"
    primary_pairwise_path = tables_dir / "strategy_pairwise_primary_metric.parquet"
    outer_units_path = tables_dir / "strategy_outer_unit_metrics.parquet"
    compute_display_path = tables_dir / "strategy_compute_display.parquet"
    statistics_manifest_path = tables_dir / "strategy_statistics_manifest.json"
    with phase_progress("Regression report outputs", 3) as phase:
        phase.phase("parquet tables")
        write_table(procedure_path, procedure)
        write_table(statistics_path, statistics)
        write_table(pairwise_path, pairwise)
        write_table(performance_path, performance)
        phase.phase("report tables")
        write_table(top_path, top_mpmas)
        write_table(selection_path, ensemble_summary)
        write_table(members_path, ensemble_members)
        write_table(primary_pairwise_path, primary_pairwise)
        write_table(outer_units_path, outer_units)
        write_table(compute_display_path, compute_display)
        phase.phase("statistics manifest")
        dump_json_standard(
            {
                "task": "regression",
                "primary_metric": metric,
                "display_metrics": list(_REGRESSION_DISPLAY_METRICS),
                "outer_unit_metrics": outer_units_path,
                "summary": statistics_path,
                "pairwise": pairwise_path,
            },
            statistics_manifest_path,
        )
    figs = [
        _report._fig(
            root / "figures" / "representation_impact",
            report_dir,
            "Modality representation impact overview"
            if getattr(sweep, "uses_modalities", False)
            else "MPDR representation impact overview",
        ),
        _report._fig(
            root / "figures" / "mpma_e", report_dir, "Selected MPMA-E schematic"
        ),
    ]
    figs = [x for x in figs if x]
    css = _report._report_css()
    raw_metric_cols = {c for c in performance.columns if c != "Strategy"}
    mpma_b = _mpma_b_composition(root)
    ensemble_html = (
        _report._html_table(ensemble_summary) + _report._html_table(ensemble_members)
        if not ensemble_summary.empty or not ensemble_members.empty
        else "<p>Ensemble results are not available yet. Run the ensemble stage and regenerate the report.</p>"
    )
    pairwise_html = (
        _report._html_table(primary_pairwise)
        if not primary_pairwise.empty
        else "<p>Pairwise strategy comparisons require at least two evaluated strategies.</p>"
    )
    html_text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>mllabiome report</title><link rel="icon" type="image/svg+xml" href="{_report._favicon_href()}"><style>{css}</style></head>
<body><header class="report-nav"><div class="report-nav-inner"><a class="brand" href="#top" aria-label="mllabiome report"><span class="brand-mark">mll</span><span>mllabiome</span></a><nav class="report-links" aria-label="Report navigation"><a href="#procedure">Evaluation</a><a href="#performance">Performance</a><a href="#mpma-b-composition">MPMA-B</a><a href="#mpma-e">MPMA-E</a><a href="#explainability-comparison">Explainability</a><a href="#figures">Figures</a><a href="#statistics">Statistics</a></nav></div></header>
<div class="report-shell"><main id="top" class="report-content">
<h2 id="procedure" class="first-section">Evaluation procedure</h2>{_report._procedure_grid_html(procedure)}
<h2 id="performance">Task performance summary</h2><p>Held-out strategy performance is reported as mean ± SD across outer evaluation units with 95% bootstrap confidence intervals. Lower values are better for loss metrics; higher values are better for R² and correlation metrics.</p>{_report._html_table(performance, raw_html_cols=raw_metric_cols)}
<section id="mpma-b-composition"><h3>Final MPMA-B specification</h3>{_report._html_table(mpma_b)}</section>
<section id="mpma-e-specification"><h3 id="mpma-e">Final MPMA-E specification</h3>{ensemble_html}</section>
<h3 id="statistics">Statistical comparisons</h3><p>Pairwise differences are Strategy A minus Strategy B on matched held-out outer units. Holm adjustment is applied across strategy pairs within each metric.</p>{pairwise_html}
<h2 id="top-mpmas">Top 5 MPMA-B configurations</h2>{_report._html_table(top_mpmas)}
<h2 id="explainability-comparison">Explainability</h2>{explainability_html if explainability_html else "<p>No explainability artefacts are available yet.</p>"}
<h2 id="figures">Global figures</h2>{"".join(figs) if figs else "<p>No figure artefacts found yet.</p>"}
<h3 id="compute">Computational resources</h3><p>Compute is summarized by additive CPU core-hours, model-fit count, and peak resident memory for the worker process tree. MPMA-B and MPMA-E share the MPMA search pool, so their compute totals overlap.</p>{_report._html_table(compute_display)}
<p class="report-footer">mllabiome · generated report</p></main></div></body></html>'''
    html_path = report_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    manifest_path = report_dir / "report_manifest.json"
    dump_json_standard(
        {
            "report_dir": report_dir,
            "task": "regression",
            "target": str(
                sweep.samples.target_col
                if getattr(sweep, "uses_modalities", False)
                else sweep.data.target_col
            ),
            "figures_embedded": len(figs),
            "explainability_targets": explainability_count,
        },
        manifest_path,
    )
    outputs = {
        "html_report": html_path,
        "top5_mpmas": top_path,
        "mpma_e_selection": selection_path,
        "mpma_e_members": members_path,
        "strategy_outer_unit_metrics": outer_units_path,
        "strategy_metrics_bootstrap": statistics_path,
        "strategy_pairwise_tests": pairwise_path,
        "strategy_statistics_manifest": statistics_manifest_path,
        "strategy_compute": compute.get(
            "compute_path", tables_dir / "strategy_compute.parquet"
        ),
        "compute_accounting_manifest": compute.get(
            "manifest_path", tables_dir / "compute_accounting_manifest.json"
        ),
    }
    success("Report completed")
    path_table("Report outputs", outputs)
    return outputs


def _target_primary_summary(child: Sweep) -> dict[str, Any]:
    root = child.root()
    target = str(child.data.target_col)
    task = str(child.data.task)
    metric = str(child.evaluation.optimize_metric)
    rows: dict[str, Any] = {"Target": target, "Task": task, "Primary metric": metric}
    files = {
        "MPMA-B": root / "results" / "mpma_b_outer_results.parquet",
        "MPMA-E": root / "ensembling" / "mpma_e_outer_results.parquet",
    }
    for strategy, path in files.items():
        frame = _read_table(path)
        if frame.empty or metric not in frame.columns:
            rows[strategy] = "—"
            continue
        values = _finite(frame[metric])
        rows[strategy] = (
            "—"
            if len(values) == 0
            else f"{float(np.mean(values)):.3f} ± {float(np.std(values, ddof=1)) if len(values) > 1 else 0.0:.3f}"
        )
    return rows


def _section_slug(value: str) -> str:
    return (
        "".join(ch if ch.isalnum() else "-" for ch in str(value).lower()).strip("-")
        or "target"
    )


def _target_report_section(child: Sweep, parent_report_dir: Path) -> str:
    root = Path(child.root())
    target = str(child.data.target_col)
    task = _target_task(child.data, target)
    section_id = f"target-{_section_slug(target)}"
    tables_dir = root / "report" / "tables"
    procedure = _read_table(tables_dir / "evaluation_procedure.parquet")
    performance = _read_table(tables_dir / "task_strategy_performance.parquet")
    selection = _read_table(tables_dir / "mpma_e_selection.parquet")
    members = _read_table(tables_dir / "mpma_e_members.parquet")
    pairwise = _read_table(tables_dir / "strategy_pairwise_primary_metric.parquet")
    top_mpmas = _read_table(tables_dir / "top5_mpma_inner_outer_performance.parquet")
    compute = _read_table(tables_dir / "strategy_compute_display.parquet")
    if compute.empty:
        compute = _read_table(tables_dir / "strategy_compute.parquet")
    if task == "regression":
        xai, _ = _regression_explainability_blocks(
            root, parent_report_dir, int(getattr(child.explainability, "top_k", 15))
        )
    else:
        xai, _ = _report._explainability_report_blocks(
            root,
            parent_report_dir,
            top_n=int(getattr(child.explainability, "top_k", 15)),
        )
    figures = [
        _report._fig(
            root / "figures" / "representation_impact",
            parent_report_dir,
            f"{target}: MPDR representation impact overview",
        ),
        _report._fig(
            root / "figures" / "mpma_e",
            parent_report_dir,
            f"{target}: selected MPMA-E schematic",
        ),
    ]
    figures = [x for x in figures if x]
    performance_raw = {c for c in performance.columns if c != "Strategy"}
    top_raw = {
        c for c in top_mpmas.columns if c.startswith("Inner ") or c.startswith("Outer ")
    }
    mpma_b = _mpma_b_composition(root)
    oof = ""
    if task != "regression":
        labels = getattr(child.data, "class_labels", None)
        try:
            n_classes = len(labels) if labels is not None else None
        except TypeError:
            n_classes = None
        oof = oof_section_html(
            root / "report",
            n_classes=n_classes,
            heading_level=3,
            id_prefix=section_id,
            include_downloads=False,
        )
    return f'''<section id="{section_id}" class="xai-target"><h2>{html.escape(target)}</h2>
<h3>Evaluation procedure</h3>{_report._procedure_grid_html(procedure)}
<h3>Task performance summary</h3>{_report._html_table(performance, raw_html_cols=performance_raw)}
<h3>Final MPMA-B specification</h3>{_report._html_table(mpma_b)}
<h3>Final MPMA-E specification</h3>{_report._html_table(selection)}{_report._html_table(members)}
{oof}
<h3>Statistical comparisons</h3>{_report._html_table(pairwise)}
<h3>Top 5 MPMA-B configurations</h3>{_report._html_table(top_mpmas, raw_html_cols=top_raw)}
<h3>Explainability</h3>{xai if xai else "<p>No explainability artefacts are available yet.</p>"}
<h3>Global figures</h3>{"".join(figures) if figures else "<p>No figure artefacts found yet.</p>"}
<h3>Computational resources</h3>{_report._html_table(compute)}</section>'''


def _multilabel_summary(children: list[Sweep], tables_dir: Path) -> pd.DataFrame:
    rows = []
    for child in children:
        frame = _read_table(child.root() / "results" / "mpma_b_outer_results.parquet")
        if frame.empty:
            continue
        row: dict[str, Any] = {"target": str(child.data.target_col)}
        for metric in (
            "AUC",
            "PR_AUC",
            "nMCC",
            "F1w",
            "BalAcc",
            "Accuracy",
            "log_loss",
            "brier",
        ):
            if metric in frame.columns:
                values = _finite(frame[metric])
                row[metric] = float(np.mean(values)) if len(values) else np.nan
        rows.append(row)
    label_summary = pd.DataFrame(rows)
    write_table(tables_dir / "multilabel_per_label_performance.parquet", label_summary)
    if label_summary.empty:
        return pd.DataFrame()
    numeric = [c for c in label_summary.columns if c != "target"]
    macro = pd.DataFrame(
        [
            {
                "Summary": "Macro across labels",
                **{
                    c: float(pd.to_numeric(label_summary[c], errors="coerce").mean())
                    for c in numeric
                },
            }
        ]
    )
    write_table(tables_dir / "multilabel_macro_performance.parquet", macro)
    return macro


def write_multi_target_report(
    sweep: Sweep, children: list[Sweep] | None = None
) -> dict[str, Path]:
    children = target_sweeps(sweep) if children is None else list(children)
    root = Path(sweep.root())
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stage("Report", str(report_dir))
    with phase_progress("Combined report", 3) as phase:
        phase.phase("target overview")
        overview = pd.DataFrame([_target_primary_summary(child) for child in children])
        overview_path = tables_dir / "target_performance_overview.parquet"
        write_table(overview_path, overview)
        target_manifest = pd.DataFrame(
            [
                {
                    "target": str(child.data.target_col),
                    "task": _target_task(child.data, str(child.data.target_col)),
                    "experiment_dir": str(child.root()),
                    "primary_metric": str(child.evaluation.optimize_metric),
                    "report": str(child.root() / "report" / "index.html"),
                }
                for child in children
            ]
        )
        targets_path = tables_dir / "targets.parquet"
        write_table(targets_path, target_manifest)
        phase.phase("task summaries")
        task = _normalise_sweep_task(sweep.data.task)
        macro = (
            _multilabel_summary(children, tables_dir)
            if task == "multilabel"
            else pd.DataFrame()
        )
        procedure = pd.DataFrame(
            [
                ["Task", sweep.title],
                [
                    "Task type",
                    "Multilabel"
                    if task == "multilabel"
                    else "Heterogeneous multi-output",
                ],
                [
                    "Targets",
                    ", ".join(str(child.data.target_col) for child in children),
                ],
                ["Target count", len(children)],
                ["Procedure", sweep.evaluation.protocol],
                ["Outer folds", sweep.evaluation.outer_folds],
                ["Inner folds", sweep.evaluation.inner_folds],
                ["Repeats", sweep.evaluation.repeats],
                ["Random seed", sweep.evaluation.random_state],
                [
                    "Selection rule",
                    "Each target is selected only from its inner-validation results. Held-out outer predictions are used for reported performance and statistical comparisons.",
                ],
            ],
            columns=["Field", "Value"],
        )
        procedure_path = tables_dir / "evaluation_procedure.parquet"
        write_table(procedure_path, procedure)
        phase.phase("target report sections")
        sections = "".join(
            _target_report_section(child, report_dir) for child in children
        )
    nav_targets = "".join(
        f'<a href="#target-{_section_slug(str(child.data.target_col))}">{html.escape(str(child.data.target_col))}</a>'
        for child in children
    )
    css = _report._report_css()
    html_text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>mllabiome report</title><link rel="icon" type="image/svg+xml" href="{_report._favicon_href()}"><style>{css}</style></head>
<body><header class="report-nav"><div class="report-nav-inner"><a class="brand" href="#top" aria-label="mllabiome report"><span class="brand-mark">mll</span><span>mllabiome</span></a><nav class="report-links" aria-label="Report navigation"><a href="#procedure">Evaluation</a><a href="#overview">Overview</a>{nav_targets}</nav></div></header>
<div class="report-shell"><main id="top" class="report-content"><h2 id="procedure" class="first-section">Evaluation procedure</h2>{_report._procedure_grid_html(procedure)}
<h2 id="overview">Target performance overview</h2><p>Every target retains its task-appropriate primary metric. Values are mean ± SD across held-out outer evaluation units. Classification and regression metrics are intentionally not collapsed into a single heterogeneous score.</p>{_report._html_table(overview)}{_report._html_table(macro) if not macro.empty else ""}
{sections}<p class="report-footer">mllabiome · generated report</p></main></div></body></html>'''
    html_path = report_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    manifest_path = report_dir / "report_manifest.json"
    dump_json_standard(
        {
            "report_dir": report_dir,
            "task": task,
            "targets": [
                {
                    "target": str(child.data.target_col),
                    "task": _target_task(child.data, str(child.data.target_col)),
                    "report": child.root() / "report" / "index.html",
                }
                for child in children
            ],
            "root_report": html_path,
        },
        manifest_path,
    )
    outputs = {
        "html_report": html_path,
        "target_overview": overview_path,
        "targets": targets_path,
        "evaluation_procedure": procedure_path,
        "manifest": manifest_path,
    }
    if table_exists(tables_dir / "multilabel_per_label_performance.parquet"):
        outputs["multilabel_per_label_performance"] = (
            tables_dir / "multilabel_per_label_performance.parquet"
        )
    if table_exists(tables_dir / "multilabel_macro_performance.parquet"):
        outputs["multilabel_macro_performance"] = (
            tables_dir / "multilabel_macro_performance.parquet"
        )
    success("Report completed")
    path_table("Report outputs", outputs)
    return outputs


def write_task_report(sweep: Sweep) -> dict[str, Any]:
    children = target_sweeps(sweep)
    if len(children) > 1 or children[0] is not sweep:
        target_outputs: dict[str, Any] = {}
        for index, child in enumerate(children, start=1):
            info(f"Report · target {index}/{len(children)} · {child.data.target_col}")
            if (child.root() / "ensembling" / "selected_unit.json").exists():
                build_final_models(child.root())
            task = _target_task(child.data, str(child.data.target_col))
            target_outputs[str(child.data.target_col)] = (
                write_regression_report(child)
                if task == "regression"
                else write_report(child)
            )
        combined = write_multi_target_report(sweep, children)
        return {"targets": target_outputs, "combined": combined}
    child = children[0]
    if (child.root() / "ensembling" / "selected_unit.json").exists():
        build_final_models(child.root())
    task = _target_task(child.data, str(child.data.target_col))
    return (
        write_regression_report(child) if task == "regression" else write_report(child)
    )
