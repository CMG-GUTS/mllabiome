from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import report as _report
from .console import path_table, phase_progress, stage, success
from .regression_explainability import _write_regression_explainability_figures
from .task_reports import _format_metric
from .storage import parquet_engine_available, read_table, write_table
from .utils import dump_json_standard, prepare_report_html_path


_MAXIMIZE = frozenset({"f1_micro", "f1_macro", "f1_samples", "subset_accuracy", "AUROC_macro", "AP_macro", "R2"})


def configuration_scores(root, summary):
    if summary.get("configuration_scores"):
        return summary["configuration_scores"]
    source = Path(root) / "tables" / "joint_configuration_scores.tsv"
    if source.exists():
        return pd.read_csv(source, sep="\t").to_dict("records")
    inner_path = Path(root) / "tables" / "joint_inner_scores.tsv"
    outer_path = Path(root) / "tables" / "joint_outer_metrics.tsv"
    experiment_path = Path(root) / "joint_experiment.json"
    if not inner_path.exists() or not experiment_path.exists():
        return []
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    inner = pd.read_csv(inner_path, sep="\t")
    outer = pd.read_csv(outer_path, sep="\t") if outer_path.exists() else pd.DataFrame()
    metric = summary["optimize_metric"]
    means = inner.groupby(["config_id", "split_key"], sort=False)[metric].mean()
    groups = means.groupby(level="config_id")
    selected = outer["config_id"].value_counts().to_dict() if "config_id" in outer else {}
    records = []
    for spec in experiment.get("configurations", ()):
        identifier = spec["config_id"]
        values = groups.get_group(identifier).to_numpy(dtype=float)
        records.append({
            "config_id": identifier,
            "learner": spec["learner"],
            "resolution": spec["resolution"],
            "transformation": spec["transformation"],
            "inner_mean": float(np.mean(values)),
            "inner_std_across_outer_splits": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "outer_splits_scored": len(values),
            "selected_outer_splits": int(selected.get(identifier, 0)),
        })
    records.sort(key=lambda r: (-r["inner_mean"] if metric in _MAXIMIZE else r["inner_mean"], r["config_id"]))
    return records


def _table(root, stem):
    path = Path(root) / "tables" / f"joint_{stem}.tsv"
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()


def _standard_table(path, frame):
    path = Path(path)
    if parquet_engine_available():
        write_table(path, frame)
        return path
    alternative = path.with_suffix(".tsv")
    frame.to_csv(alternative, sep="\t", index=False)
    return alternative


_REGRESSION_ORDER = ("normalized_rmse", "RMSE", "MAE", "R2")
_MULTILABEL_ORDER = ("log_loss", "AUROC_macro", "AP_macro", "f1_macro", "f1_micro", "subset_accuracy", "brier")


def _metric_label(name):
    labels = {
        "normalized_rmse": "Normalized RMSE",
        "AUROC_macro": "AUROC macro",
        "AP_macro": "Average precision macro",
        "f1": "F1",
        "f1_macro": "F1 macro",
        "f1_micro": "F1 micro",
        "f1_samples": "F1 samples",
        "subset_accuracy": "Subset accuracy",
        "hamming_loss": "Hamming loss",
    }
    return labels.get(name, _report._metric_display_label(name))


def _metric_is_fraction(name):
    return name in {"AUROC_macro", "AP_macro", "AUROC", "AP", "f1", "f1_macro", "f1_micro", "f1_samples", "subset_accuracy", "precision", "recall"}


def _metric_cell(name, estimate, sd=np.nan, low=np.nan, high=np.nan, bold=False):
    if _metric_is_fraction(name):
        return _report._pct_ci_cell(estimate, sd, low, high, bold=bold, html_mode=True)
    return _format_metric(name, estimate, sd, low, high, bold=bold)


def _hierarchical_estimate(frame, metric, repetitions=2000):
    if frame.empty or metric not in frame:
        return np.nan, np.nan, np.nan, np.nan
    working = frame[["repeat", metric]].copy() if "repeat" in frame else frame[[metric]].assign(repeat=0)
    working[metric] = pd.to_numeric(working[metric], errors="coerce")
    working = working[np.isfinite(working[metric].to_numpy(dtype=float))]
    if working.empty:
        return np.nan, np.nan, np.nan, np.nan
    values = working[metric].to_numpy(dtype=float)
    groups = [group[metric].to_numpy(dtype=float) for _, group in working.groupby("repeat", sort=True)]
    rng = np.random.default_rng(42)
    bootstrap = np.empty(int(repetitions), dtype=float)
    for b in range(int(repetitions)):
        chosen = rng.integers(0, len(groups), size=len(groups))
        sampled = [groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))] for i in chosen]
        bootstrap[b] = float(np.mean(np.concatenate(sampled)))
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else np.nan, float(low), float(high)


def _strategy_performance(root, summary):
    base = _table(root, "outer_metrics")
    ensembles = _table(root, "ensemble_outer_metrics")
    frames = [("MPMA-B", base)]
    if not ensembles.empty:
        frames.extend(("MPMA-E · " + str(name), group) for name, group in ensembles.groupby("strategy", sort=True))
    metrics = _MULTILABEL_ORDER if summary["task"] == "multilabel" else _REGRESSION_ORDER
    metrics = [m for m in metrics if any(m in f.columns for _, f in frames)]
    primary = str(summary["optimize_metric"])
    if primary not in metrics and any(primary in f.columns for _, f in frames):
        metrics = [primary] + metrics
    results = []
    for label, frame in frames:
        if frame.empty:
            continue
        row = {"Strategy": label}
        for metric in metrics:
            row[_metric_label(metric)] = _metric_cell(metric, *_hierarchical_estimate(frame, metric))
        results.append(row)
    return pd.DataFrame(results), frames, metrics


def _target_performance(root, summary):
    source = _table(root, "target_metrics")
    if source.empty:
        return pd.DataFrame()
    metrics = ("log_loss", "AUROC", "AP", "f1") if summary["task"] == "multilabel" else ("normalized_rmse", "RMSE", "MAE", "R2")
    rows = []
    for target in summary["targets"]:
        part = source.loc[source["target"].astype(str).eq(str(target))]
        if part.empty:
            continue
        row = {"Target": target}
        for metric in metrics:
            row[_metric_label(metric)] = _metric_cell(metric, *_hierarchical_estimate(part, metric))
        rows.append(row)
    return pd.DataFrame(rows)


def _ranked_configurations(root, summary, limit=5):
    scores = pd.DataFrame(configuration_scores(root, summary))
    if scores.empty:
        return pd.DataFrame()
    outer = _table(root, "outer_metrics")
    primary = str(summary["optimize_metric"])
    rows = []
    for index, (_, record) in enumerate(scores.head(limit).iterrows(), 1):
        selected = outer.loc[outer["config_id"].astype(str).eq(str(record["config_id"]))] if not outer.empty else pd.DataFrame()
        value = float(record["inner_mean"])
        sd = float(record.get("inner_std_across_outer_splits", np.nan))
        outer_stats = _hierarchical_estimate(selected, primary) if not selected.empty else (np.nan,) * 4
        rows.append({
            "Rank": index,
            "Taxonomic representation": _report._display_token(record["resolution"]),
            "Count transformation": _report._display_token(record["transformation"]),
            "Learner": _report._display_learner(record["learner"]),
            "Inner " + _metric_label(primary): (f"{value:.3f} ± {sd:.3f}" if np.isfinite(sd) else f"{value:.3f}") if not _metric_is_fraction(primary) else _report._pct_cell(value, sd, html_mode=True),
            "Outer " + _metric_label(primary): _metric_cell(primary, outer_stats[0], outer_stats[1]) if not selected.empty else "NA",
        })
    return pd.DataFrame(rows)


def _ensemble_specification(root, summary):
    metrics = _table(root, "ensemble_outer_metrics")
    members = _table(root, "ensemble_members")
    if metrics.empty:
        return ""
    specs = []
    for name, frame in metrics.groupby("strategy", sort=True):
        first = frame.iloc[0]
        counts = pd.to_numeric(frame["n_members"], errors="coerce").dropna()
        specs.append({
            "Strategy": "MPMA-E · " + str(name),
            "Selection": str(first["selection"]),
            "Aggregation": str(first["aggregation"]),
            "Max size": int(first["max_size"]),
            "Members": str(int(counts.min())) if len(counts) and counts.min() == counts.max() else (f"{int(counts.min())}–{int(counts.max())}" if len(counts) else ""),
            "Selection metric": summary["optimize_metric"],
        })
    details = ""
    if not members.empty:
        registry = pd.DataFrame(configuration_scores(root, summary))
        readable = members.merge(registry[["config_id", "resolution", "transformation", "learner"]], on="config_id", how="left")
        member_rows = []
        for (strategy, identifier), part in readable.groupby(["strategy", "config_id"], sort=True):
            item = part.iloc[0]
            member_rows.append({
                "Strategy": "MPMA-E · " + str(strategy),
                "Representation": _report._display_token(item.get("resolution", "")),
                "Transformation": _report._display_token(item.get("transformation", "")),
                "Learner": _report._display_learner(item.get("learner", "")),
                "Outer units": part["split_key"].nunique(),
            })
        details = _report._html_table(pd.DataFrame(member_rows))
    return '<section id="mpma-e-specification"><h3 id="mpma-e">MPMA-E specifications</h3><p>Ensemble members are selected within each outer training partition; the table describes fold-specific specifications, not a retrospectively fixed final ensemble.</p>' + _report._html_table(pd.DataFrame(specs)) + details + "</section>"


def _compute_table(root):
    inner = _table(root, "inner_scores")
    outer = _table(root, "outer_metrics")
    membership = _table(root, "ensemble_members")
    entries = []
    if not inner.empty or not outer.empty:
        entries.append({"Strategy": "MPMA-B", "CPU core-hours": np.nan, "Model fits": len(inner) + len(outer), "Peak job RAM (GiB)": np.nan, "Accounting coverage": "Model fits only"})
    if not membership.empty:
        count = len(membership[["split_key", "config_id"]].drop_duplicates())
        entries.append({"Strategy": "MPMA-E", "CPU core-hours": np.nan, "Model fits": count, "Peak job RAM (GiB)": np.nan, "Accounting coverage": "Model fits only"})
    return pd.DataFrame(entries)

def _joint_generalisation(root, report_dir, summary):
    from .generalisation_figure import _learner_family, _learned_representation, _transformation, render_generalisation_figure
    inner = _table(root, "inner_scores")
    outer = _table(root, "outer_metrics")
    metric = str(summary["optimize_metric"])
    if inner.empty or outer.empty or metric not in inner or metric not in outer:
        return ""
    groups = inner.groupby(["config_id", "split_key"], as_index=False)[metric].mean().rename(columns={metric: "inner"})
    selected = outer[["config_id", "split_key", metric]].rename(columns={metric: "outer"})
    matched = groups.merge(selected, on=["config_id", "split_key"], how="inner", validate="one_to_one")
    if matched.empty:
        return ""
    registry = pd.DataFrame(configuration_scores(root, summary))
    if registry.empty:
        return ""
    meta = registry[["config_id", "learner", "resolution", "transformation"]].drop_duplicates("config_id")
    matched = matched.merge(meta, on="config_id", how="left", validate="many_to_one")
    scores = matched.groupby("config_id", as_index=False).agg(inner=("inner", "mean"), outer=("outer", "mean"), matched_outer_splits=("split_key", "nunique"))
    scores = scores.merge(meta, on="config_id", how="left", validate="one_to_one")
    scores["learner_class"] = scores["learner"].astype(str)
    scores["count_transformation"] = scores["transformation"]
    scores["learner_family"] = scores.apply(_learner_family, axis=1)
    scores["learned_representation"] = scores["count_transformation"].map(_learned_representation)
    scores["transformation_family"] = scores["count_transformation"].map(_transformation)
    scores["generalisation_drop"] = scores["outer"] - scores["inner"] if metric not in _MAXIMIZE else scores["inner"] - scores["outer"]
    dest = Path(root) / "figures" / "inner_outer_generalisation_joint.svg"
    render_generalisation_figure(scores, dest, metric)
    figure = _report._fig(dest.with_suffix(""), report_dir, "Inner validation versus outer held-out performance for configurations selected in at least one outer fold.")
    if not figure:
        return ""
    return '<h2 id="representation-impact">Data representations and learner impact on performance</h2><h3>Inner validation versus outer held-out performance</h3><p>Each point represents a configuration selected in at least one outer fold. Inner and outer scores are matched by fold, and unselected configurations have no outer estimate. These comparisons are descriptive, not independent estimates of deployment performance.</p><h4>Optimisation metric: ' + html.escape(metric) + "</h4>" + figure

def _explainability_sections(root, report_dir, summary, top_k):
    root = Path(root)
    summaries = ['<h4>Feature importance and stability across explanation methods</h4>']
    details = []
    features = []
    locations = [("MPMA-B", target, root / "explainability" / "mpma_b" / str(target)) for target in summary["targets"]]
    for strategy_dir in sorted((root / "explainability" / "mpma_e").glob("*")):
        if strategy_dir.is_dir():
            locations.extend(("MPMA-E · " + strategy_dir.name, target, strategy_dir / str(target)) for target in summary["targets"])
    found = False
    previous_unit = None
    for unit, target, target_dir in locations:
        meta_path = target_dir / "explained_unit.json"
        if not meta_path.exists():
            continue
        found = True
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        importance = read_table(target_dir / "feature_stability.parquet")
        _write_regression_explainability_figures(target_dir, importance, int(top_k), probability_output=summary["task"] == "multilabel")
        figure = _report._fig(target_dir / "figures" / "feature_support", report_dir, f"{target}: feature importance and cross-fold support")
        summaries.append(f'<h5>{html.escape(str(unit))} · {html.escape(str(target))}</h5>')
        if figure:
            summaries.append(figure)
        if unit != previous_unit:
            details.append(f'<h3>{html.escape(str(unit))}</h3>')
            previous_unit = unit
        details.append(f'<section class="xai-target"><h4>{html.escape(str(target))}</h4><h5>Global explanations</h5>')
        details.append(f'<p>Attributions describe the jointly fitted predictor for {html.escape(str(target))}; they are not causal effects.</p>')
        top = importance.sort_values(["method", "rank_within_method"]).groupby("method", sort=False).head(int(top_k)) if not importance.empty else pd.DataFrame()
        if not top.empty:
            display = top[["method", "feature", "importance_mean", "importance_sd", "top_k_frequency", "n_estimable_folds"]].rename(columns={
                "method": "Method", "feature": "Feature", "importance_mean": "Mean importance", "importance_sd": "SD", "top_k_frequency": "Top-k frequency", "n_estimable_folds": "Fold coverage"
            })
            details.append(_report._html_table(display))
            top_features = top.copy()
            top_features.insert(0, "Target", target)
            top_features.insert(0, "Strategy", unit)
            features.append(top_features)
        for stem, heading in (("ale_curves", "ALE curves"), ("ale_interactions", "ALE interactions")):
            table = read_table(target_dir / f"{stem}.parquet")
            if table.empty:
                continue
            details.append(f"<h5>{html.escape(heading)}</h5>")
            fig_stem = "ale_curves" if stem == "ale_curves" else "interaction_network_current"
            curve = _report._fig(target_dir / "figures" / fig_stem, report_dir, f"{target}: {heading}")
            if curve:
                details.append(curve)
        local = read_table(target_dir / "local_explanations.parquet")
        if not local.empty:
            details.append("<h5>Local explanations</h5>")
            local_figure = _report._fig(target_dir / "figures" / "local_explanations", report_dir, f"{target}: selected held-out sample explanations")
            if local_figure:
                details.append(local_figure)
        details.append("</section>")
    if not found:
        return "<p>Explanations have not been generated.</p>", "<p>Explanations have not been generated.</p>", pd.DataFrame()
    return "".join(summaries), "".join(details), pd.concat(features, ignore_index=True) if features else pd.DataFrame()

def _joint_ensemble_comparison(root, summary, metrics):
    if metrics.empty:
        return pd.DataFrame()
    baseline = _table(root, "outer_metrics")
    primary = summary["optimize_metric"]
    joined = metrics[["split_key", "strategy", primary]].merge(baseline[["split_key", primary]], on="split_key", how="inner", suffixes=("_ensemble", "_individual"), validate="many_to_one")
    if joined.empty:
        return pd.DataFrame()
    rows = []
    for name, frame in joined.groupby("strategy", sort=True):
        differences = frame[f"{primary}_ensemble"].to_numpy(dtype=float) - frame[f"{primary}_individual"].to_numpy(dtype=float)
        rows.append({"Strategy A": name, "Strategy B": "Selected joint model", "Metric": primary, "Mean paired outer-fold difference": float(np.mean(differences)), "SD across outer folds": float(np.std(differences, ddof=1)) if len(differences) > 1 else np.nan, "Matched outer folds": len(differences)})
    return pd.DataFrame(rows)


def write_joint_html(root, title, summary, top_k=30, include_representation_impact=True):
    root = Path(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stage("Report", str(report_dir))
    with phase_progress("Combined report", 4) as phase:
        phase.phase("evaluation procedure")
        study = json.loads((root / "joint_experiment.json").read_text(encoding="utf-8")) if (root / "joint_experiment.json").exists() else {}
        evaluation = study.get("evaluation", {})
        procedure = pd.DataFrame([
            ("Task", title),
            ("Experiment directory", str(root)),
            ("Data format", "Joint microbiome"),
            ("Samples", summary["n_samples"]),
            ("Targets", ", ".join(map(str, summary["targets"]))),
            ("Procedure", evaluation.get("protocol", summary.get("split_strategy", "nested_cv"))),
            ("Benchmark", ""),
            ("Stratification", summary.get("split_strategy", "")),
            ("Outer folds", evaluation.get("outer_folds", "")),
            ("Inner folds", evaluation.get("inner_folds", "")),
            ("Repeats", summary["n_repeats"]),
            ("Selection metric", summary["optimize_metric"]),
            ("Random seed", evaluation.get("random_state", "")),
            ("Qualification gate", "off"),
            ("Selection rule", "Reported performance uses held-out outer evaluation predictions."),
        ], columns=["Field", "Value"])
        phase.phase("held-out performance")
        performance, frames, metrics = _strategy_performance(root, summary)
        per_target = _target_performance(root, summary)
        configs_full = pd.DataFrame(configuration_scores(root, summary))
        top = _ranked_configurations(root, summary)
        ensemble_spec = _ensemble_specification(root, summary)
        ensemble_metrics = _table(root, "ensemble_outer_metrics")
        comparison = _joint_ensemble_comparison(root, summary, ensemble_metrics)
        phase.phase("explainability")
        xai_summary, xai_detail, features = _explainability_sections(root, report_dir, summary, top_k)
        phase.phase("report artifacts")
        saved = {}
        for name, data in (("procedure", procedure), ("heldout_performance", performance), ("target_performance", per_target), ("configuration_rankings", configs_full), ("top_configurations", top), ("strategy_pairwise_comparisons", comparison), ("important_features", features)):
            if not data.empty:
                saved[name] = _standard_table(tables_dir / f"{name}.parquet", data)
        methodology = '<p>Held-out strategy performance is summarized across outer test folds as mean ± SD. The 95% confidence intervals use 2,000 hierarchical bootstrap replicates resampling repeats and then outer folds. Joint learning shares a fitted estimator across targets; each target is evaluated separately and macro metrics summarize targets.</p>'
        performance_html = methodology + _report._html_table(performance, raw_html_cols=set(performance.columns) - {"Strategy"})
        if not configs_full.empty:
            item = configs_full.iloc[0]
            specification = pd.DataFrame([{"Data representation": _report._display_token(item["resolution"]), "Transformation": _report._display_token(item["transformation"]), "Learner": _report._display_learner(item["learner"])}])
            performance_html += '<section><h3>Final MPMA-B specification</h3><p>This specification is selected retrospectively from accumulated inner validation evidence. Reported nested-CV performance is for fold-specific selections, not independent validation of this fixed specification.</p>' + _report._html_table(specification) + "</section>"
        if not per_target.empty:
            performance_html += '<h3>Per-target held-out performance</h3>' + _report._html_table(per_target, raw_html_cols=set(per_target.columns) - {"Target"})
        exploration_parts = []
        for target in summary["targets"]:
            content = _report._exploration_report_html(root / "targets" / str(target), report_dir)
            if content:
                content = content.replace('<h2 id="data-exploration">Data exploration</h2>', '')
                exploration_parts.append(f'<section class="explore-target"><h3>{html.escape(str(target))}</h3>{content}</section>')
        exploration = '<h2 id="data-exploration">Data exploration</h2>' + "".join(exploration_parts) if exploration_parts else ""
        representation = _joint_generalisation(root, report_dir, summary) if include_representation_impact else ""
        top_html = '<h2 id="top-mpmas">' + _report._top_mpma_heading(top) + '</h2><p>All evaluated configurations, including those no longer scheduled in the active sweep, remain in the experiment registry. The displayed ranking uses inner validation; outer columns describe only the folds in which that candidate was selected.</p>' + _report._html_table(top, raw_html_cols={name for name in top if str(name).startswith(("Inner ", "Outer "))})
        compare_html = '<h3 id="statistics">Outer-unit strategy comparisons</h3>'
        if not comparison.empty:
            rows = []
            for _, record in comparison.iterrows():
                rows.append({"Strategy A": "MPMA-E · " + str(record["Strategy A"]), "Strategy B": "MPMA-B", "Difference A-B": record["Mean paired outer-fold difference"], "95% CI low": np.nan, "95% CI high": np.nan, "Matched outer units": int(record["Matched outer folds"]), "Test": "Descriptive", "p": np.nan, "Holm p": np.nan, "Holm p<0.05": ""})
            compare_html += '<p>Paired differences are descriptive and use matched outer test folds. Inferential p-values and confidence intervals are not estimated for these joint comparisons.</p>' + _report._html_table(pd.DataFrame(rows))
        else:
            compare_html += '<p>No matched ensemble comparisons are available.</p>'
        computation = _compute_table(root)
        computation_html = '<p>Model fits are counted from persisted inner, outer, and unique ensemble-member fit records. CPU core-hours and peak resident memory are not available from the joint evaluation records and are not estimated.</p>' + _report._html_table(computation)
        environment_path = root / "joint_environment.json"
        environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
        hardware_frame = _report._hardware_summary_table(environment)
        hardware = _report._html_table(hardware_frame) if not hardware_frame.empty else '<p>Hardware details are unavailable for this run.</p>'
        if not hardware_frame.empty:
            saved["hardware_environment"] = _standard_table(tables_dir / "hardware_environment.parquet", hardware_frame)
        text = _report._render_standard_report_sections(
            procedure=_report._procedure_grid_html(procedure),
            exploration=exploration,
            performance=performance_html,
            explainability_summary=xai_summary,
            representation=representation,
            final_specification=ensemble_spec,
            top_configurations=top_html,
            comparisons=compare_html,
            explainability=xai_detail,
            robustness="",
            compute=computation_html,
            hardware=hardware,
        )
        destination = prepare_report_html_path(root)
        destination.write_text(text, encoding="utf-8")
        manifest = report_dir / "report_manifest.json"
        dump_json_standard({
            "report_dir": report_dir,
            "task": summary["task"],
            "report_template": _report.REPORT_TEMPLATE_VERSION,
            "report_layers": list(_report.REPORT_LAYERS),
            "targets": [{"target": target, "task": "classification" if summary["task"] == "multilabel" else "regression"} for target in summary["targets"]],
            "root_report": destination,
        }, manifest)
    _report._terminal_table("Task definition and evaluation procedure", _report._compact_procedure_for_terminal(procedure))
    _report._terminal_table("Task performance summary", performance)
    _report._terminal_table("Per-target held-out performance", per_target)
    _report._terminal_table(_report._top_mpma_heading(top), top)
    outputs = {"html_report": destination, "manifest": manifest, **saved}
    success("Report completed")
    path_table("Report outputs", outputs)
    return outputs
