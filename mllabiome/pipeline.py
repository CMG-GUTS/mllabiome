from __future__ import annotations

from pathlib import Path
from typing import Any

from .configs_sweep import Sweep, _target_task, evaluate, sweep_task, target_sweeps
from .console import info
from .ensemble_sweep import sweep_ensemble
from .final_explainability import explain
from .final_models import build_final_models
from .report import write_report
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
    info("Pipeline · evaluate")
    evaluate_output = run_evaluate(sweep)
    info("Pipeline · ensemble")
    ensemble_output = run_ensemble(sweep)
    info("Pipeline · explain")
    explain_output = run_explain(sweep)
    info("Pipeline · report")
    report_output = run_report(sweep)
    return {
        "evaluate": evaluate_output,
        "ensemble": ensemble_output,
        "explain": explain_output,
        "report": report_output,
    }


def run_stage(sweep: Sweep, stage_name: str) -> dict[str, Any]:
    if stage_name == "all":
        return run_all(sweep)
    if stage_name == "evaluate":
        return run_evaluate(sweep)
    if stage_name == "ensemble":
        return run_ensemble(sweep)
    if stage_name == "explain":
        return run_explain(sweep)
    if stage_name == "report":
        return run_report(sweep)
    raise ValueError(f"Unsupported stage {stage_name!r}.")
