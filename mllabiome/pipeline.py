from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .configs_sweep import Sweep, _target_task, evaluate, sweep_task, target_sweeps
from .console import info
from .ensemble_sweep import sweep_ensemble
from .explore import run_explore as execute_explore
from .final_explainability import explain
from .final_models import build_final_models
from .inference import run_inference as execute_inference
from .report import write_report
from .robustness import run_robustness as execute_robustness
from .stage_results import print_stage_results
from .task_reports import write_regression_report, write_task_report
from .utils import dump_json_standard


def _is_multi_target(sweep: Sweep, children: list[Sweep]) -> bool:
    return len(children) > 1 or children[0] is not sweep


def _target_record(child: Sweep, outputs: dict[str, Any]) -> dict[str, Any]:
    target = str(child.data.target_col)
    return {
        "target": target,
        "task": _target_task(child.data, target),
        "experiment_dir": child.root(),
        "outputs": outputs,
    }


def _write_stage_manifest(
    sweep: Sweep, stage_name: str, records: list[dict[str, Any]]
) -> Path:
    root = Path(sweep.root())
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stage_name}_manifest.json"
    dump_json_standard({"stage": stage_name, "targets": records}, path)
    return path


def _write_explore_index(sweep: Sweep, children: list[Sweep]) -> Path:
    root = Path(sweep.root())
    path = root / "explore" / "index.html"
    items = []
    for child in children:
        target = str(child.data.target_col)
        report = Path(child.root()) / "explore" / "index.html"
        rel = report.relative_to(root).as_posix()
        items.append(
            f'<a class="item" href="../{html.escape(rel)}"><strong>{html.escape(target)}</strong><span>Open exploratory microbiome report</span></a>'
        )
    css = "body{font-family:Arial,Helvetica,sans-serif;color:#0f172a;margin:0;background:#fff}main{max-width:860px;margin:0 auto;padding:48px 32px}h1{font-size:32px;letter-spacing:-.02em;margin:8px 0 28px}.k{font-size:12px;color:#1565A8;text-transform:uppercase;letter-spacing:.12em;font-weight:700}.grid{display:grid;gap:12px}.item{display:flex;justify-content:space-between;gap:24px;text-decoration:none;color:#0f172a;border:1px solid #e2e8f0;border-radius:13px;padding:18px;background:#f8fafc}.item span{color:#64748b;font-size:13px}"
    content = f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(sweep.title)} · Explore</title><style>{css}</style></head><body><main><div class="k">mllabiome exploratory microbiome analysis</div><h1>{html.escape(sweep.title)}</h1><div class="grid">{"".join(items)}</div></main></body></html>'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def run_explore(sweep: Sweep) -> dict[str, Any]:
    if getattr(sweep, "uses_modalities", False):
        outputs = execute_explore(sweep)
        print_stage_results(sweep, "explore")
        return outputs
    children = target_sweeps(sweep)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for index, child in enumerate(children, start=1):
        if _is_multi_target(sweep, children):
            info(f"Explore · target {index}/{len(children)} · {child.data.target_col}")
        child_outputs = execute_explore(child)
        target = str(child.data.target_col)
        outputs[target] = child_outputs
        records.append(_target_record(child, child_outputs))
    if _is_multi_target(sweep, children):
        outputs["manifest"] = _write_stage_manifest(sweep, "explore", records)
        outputs["report"] = _write_explore_index(sweep, children)
    result = (
        outputs
        if _is_multi_target(sweep, children)
        else outputs[str(children[0].data.target_col)]
    )
    print_stage_results(sweep, "explore")
    return result


def run_evaluate(sweep: Sweep) -> dict[str, Any]:
    outputs = evaluate(sweep)
    print_stage_results(sweep, "evaluate")
    return outputs


def run_ensemble(sweep: Sweep) -> dict[str, Any]:
    if getattr(sweep, "uses_modalities", False):
        outputs = sweep_ensemble(sweep)
        build_final_models(sweep.root())
        result = {
            "ensemble": outputs,
            "final_models": sweep.root() / "final_models.json",
        }
        print_stage_results(sweep, "ensemble")
        return result
    children = target_sweeps(sweep)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for index, child in enumerate(children, start=1):
        if _is_multi_target(sweep, children):
            info(f"Ensemble · target {index}/{len(children)} · {child.data.target_col}")
        child_outputs = sweep_ensemble(child)
        build_final_models(child.root())
        target = str(child.data.target_col)
        payload = {
            "ensemble": child_outputs,
            "final_models": child.root() / "final_models.json",
        }
        outputs[target] = payload
        records.append(_target_record(child, payload))
    if _is_multi_target(sweep, children):
        outputs["manifest"] = _write_stage_manifest(sweep, "ensemble", records)
    result = (
        outputs
        if _is_multi_target(sweep, children)
        else outputs[str(children[0].data.target_col)]
    )
    print_stage_results(sweep, "ensemble")
    return result


def run_inference(sweep: Sweep) -> dict[str, Any]:
    if getattr(sweep, "inference", None) is None:
        raise ValueError("No INFERENCE configuration is attached to this sweep.")
    if getattr(sweep, "uses_modalities", False):
        outputs = execute_inference(sweep)
        print_stage_results(sweep, "inference")
        return outputs
    children = target_sweeps(sweep)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for index, child in enumerate(children, start=1):
        if _is_multi_target(sweep, children):
            info(
                f"Inference · target {index}/{len(children)} · {child.data.target_col}"
            )
        if "mpma_e" in child.inference.targets:
            sweep_ensemble(child)
        models = build_final_models(
            child.root(), include_mpma_e="mpma_e" in child.inference.targets
        )
        child_outputs = execute_inference(child, models=models)
        target = str(child.data.target_col)
        outputs[target] = child_outputs
        records.append(_target_record(child, child_outputs))
    if _is_multi_target(sweep, children):
        outputs["manifest"] = _write_stage_manifest(sweep, "inference", records)
    result = (
        outputs
        if _is_multi_target(sweep, children)
        else outputs[str(children[0].data.target_col)]
    )
    print_stage_results(sweep, "inference")
    return result


def run_explain(sweep: Sweep) -> dict[str, Any]:
    if getattr(sweep, "uses_modalities", False):
        outputs = explain(sweep)
        print_stage_results(sweep, "explain")
        return outputs
    children = target_sweeps(sweep)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for index, child in enumerate(children, start=1):
        if _is_multi_target(sweep, children):
            info(
                f"Explainability · target {index}/{len(children)} · {child.data.target_col}"
            )
        child_outputs = explain(child)
        target = str(child.data.target_col)
        outputs[target] = child_outputs
        records.append(_target_record(child, child_outputs))
    if _is_multi_target(sweep, children):
        outputs["manifest"] = _write_stage_manifest(sweep, "explain", records)
    result = (
        outputs
        if _is_multi_target(sweep, children)
        else outputs[str(children[0].data.target_col)]
    )
    print_stage_results(sweep, "explain")
    return result


def run_robustness(sweep: Sweep) -> dict[str, Any]:
    if getattr(sweep, "robustness", None) is None:
        return {}
    if getattr(sweep, "uses_modalities", False):
        outputs = execute_robustness(sweep)
        print_stage_results(sweep, "robustness")
        return outputs
    children = target_sweeps(sweep)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for index, child in enumerate(children, start=1):
        if _is_multi_target(sweep, children):
            info(
                f"Robustness · target {index}/{len(children)} · {child.data.target_col}"
            )
        child_outputs = execute_robustness(child)
        target = str(child.data.target_col)
        outputs[target] = child_outputs
        records.append(_target_record(child, child_outputs))
    if _is_multi_target(sweep, children):
        outputs["manifest"] = _write_stage_manifest(sweep, "robustness", records)
    result = (
        outputs
        if _is_multi_target(sweep, children)
        else outputs[str(children[0].data.target_col)]
    )
    print_stage_results(sweep, "robustness")
    return result


def run_report(sweep: Sweep) -> dict[str, Any]:
    if getattr(sweep, "uses_modalities", False):
        task = sweep_task(sweep)
        outputs = (
            write_regression_report(sweep)
            if task == "regression"
            else write_report(sweep)
        )
    else:
        outputs = write_task_report(sweep)
    print_stage_results(sweep, "report")
    return outputs


def run_all(sweep: Sweep) -> dict[str, Any]:
    explore_output: dict[str, Any] = {}
    if not getattr(sweep, "uses_modalities", False):
        info("Pipeline · explore")
        explore_output = run_explore(sweep)
    info("Pipeline · evaluate")
    evaluate_output = run_evaluate(sweep)
    info("Pipeline · ensemble")
    ensemble_output = run_ensemble(sweep)
    inference_output: dict[str, Any] = {}
    if getattr(sweep, "inference", None) is not None:
        info("Pipeline · inference")
        inference_output = run_inference(sweep)
    info("Pipeline · explain")
    explain_output = run_explain(sweep)
    robustness_output: dict[str, Any] = {}
    if getattr(sweep, "robustness", None) is not None:
        info("Pipeline · robustness")
        robustness_output = run_robustness(sweep)
    info("Pipeline · report")
    report_output = run_report(sweep)
    outputs = {
        "evaluate": evaluate_output,
        "ensemble": ensemble_output,
        "explain": explain_output,
        "report": report_output,
    }
    if explore_output:
        outputs["explore"] = explore_output
    if getattr(sweep, "robustness", None) is not None:
        outputs["robustness"] = robustness_output
    if inference_output:
        outputs["inference"] = inference_output
    return outputs


def run_stage(sweep: Sweep, stage_name: str) -> dict[str, Any]:
    if stage_name == "all":
        return run_all(sweep)
    if stage_name == "explore":
        return run_explore(sweep)
    if stage_name == "evaluate":
        return run_evaluate(sweep)
    if stage_name == "ensemble":
        return run_ensemble(sweep)
    if stage_name == "inference":
        return run_inference(sweep)
    if stage_name == "explain":
        return run_explain(sweep)
    if stage_name == "robustness":
        return run_robustness(sweep)
    if stage_name == "report":
        return run_report(sweep)
    raise ValueError(f"Unsupported stage {stage_name!r}.")
