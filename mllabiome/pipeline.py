from __future__ import annotations
from pathlib import Path
import hashlib
import json
from dataclasses import asdict
from typing import Any
from .configs_sweep import Sweep, _target_task, evaluate, sweep_task, target_sweeps
from .console import info, success, SelectionStatus
from . import cooperative_store
from .ensemble_sweep import sweep_ensemble
from .explore import run_explore as execute_explore
from .final_explainability import explain
from .final_models import build_final_models
from .inference import run_inference as execute_inference
from .inference_progress import inference_phase, inference_progress
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


def run_explore(sweep: Sweep) -> dict[str, Any]:
    from .joint_sweep import is_joint_sweep

    if is_joint_sweep(sweep):
        from dataclasses import replace
        from .joint_sweep import _targets
        from .sweep_types import _normalise_sweep_task

        targets = _targets(sweep)
        classification = _normalise_sweep_task(sweep.data.task) == "multilabel"
        target_task = "classification" if classification else "regression"
        default_metric = "log_loss" if classification else "RMSE"
        records = []
        outputs = {}
        for index, target in enumerate(targets, start=1):
            info(f"Explore · target {index}/{len(targets)} · {target}")
            data = replace(
                sweep.data,
                task=target_task,
                target_col=target,
                joint=False,
                metadata_cols=tuple(
                    dict.fromkeys(
                        (
                            *sweep.data.metadata_cols,
                            *(name for name in targets if name != target),
                        )
                    )
                ),
            )
            child = replace(
                sweep,
                data=data,
                experiment_dir=Path(sweep.root()) / "targets" / target,
                evaluation=replace(sweep.evaluation, optimize_metric=default_metric),
                ensemble=replace(sweep.ensemble, optimize_metric=default_metric),
            )
            result = execute_explore(child, redo=bool(sweep.evaluation.redo))
            records.append(_target_record(child, result))
            outputs[target] = result
        outputs["manifest"] = _write_stage_manifest(sweep, "explore", records)
        print_stage_results(sweep, "explore")
        return outputs
    legacy_report = Path(sweep.root()) / "explore" / "index.html"
    if legacy_report.exists():
        legacy_report.unlink()
    redo = bool(getattr(sweep.evaluation, "redo", False))
    if getattr(sweep, "uses_modalities", False):
        outputs = execute_explore(sweep, redo=redo)
        print_stage_results(sweep, "explore")
        return outputs
    children = target_sweeps(sweep)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    for index, child in enumerate(children, start=1):
        if _is_multi_target(sweep, children):
            info(f"Explore · target {index}/{len(children)} · {child.data.target_col}")
        child_outputs = execute_explore(child, redo=redo)
        target = str(child.data.target_col)
        outputs[target] = child_outputs
        records.append(_target_record(child, child_outputs))
    if _is_multi_target(sweep, children):
        outputs["manifest"] = _write_stage_manifest(sweep, "explore", records)
    result = (
        outputs
        if _is_multi_target(sweep, children)
        else outputs[str(children[0].data.target_col)]
    )
    print_stage_results(sweep, "explore")
    return result


def run_evaluate(sweep: Sweep) -> dict[str, Any]:
    outputs = evaluate(sweep)

    if not cooperative_store.enabled(sweep.root()):
        print_stage_results(sweep, "evaluate")
    return outputs


def run_base_selection(sweep: Sweep) -> dict[str, Any]:
    from ._config_store import indexed_config_ids, _artifact_path, _digest, split_digest
    from .configs_sweep import _write_selection_artifacts, _existing_outputs
    from .joint_sweep import is_joint_sweep

    if is_joint_sweep(sweep) or getattr(sweep, "uses_modalities", False):
        raise NotImplementedError(
            "Independent base-selection is implemented for standard classification/regression sweeps; "
            "joint and multimodal evaluation still use the legacy orchestration."
        )
    info("MPMA-B · preparing target sweeps")
    children = target_sweeps(sweep)
    results: dict[str, Any] = {}
    for child in children:
        root = child.root()
        with SelectionStatus() as status, cooperative_store.reporting(status):
            status.phase(f"Target {child.data.target_col} · experiment {root}")
            status.phase("Waiting for experiment initialization lock")
            with cooperative_store.lock(root, "initialize"):
                if not (root / "manifest.json").is_file():
                    raise FileNotFoundError(
                        f"Cannot select MPMA-B without an evaluation manifest: {root}"
                    )
                cooperative_store.initialize(root)
            status.phase("Waiting for exclusive base-selection lock")
            with cooperative_store.lock(root, "base-selection"):
                status.phase("Discovering completed configurations")
                ids = indexed_config_ids(root)
                if not ids:
                    raise FileNotFoundError(
                        f"No completed MPMA artifacts found under {root / 'configs'}"
                    )
                info(f"MPMA-B · found {len(ids)} completed MPMAs")
                with cooperative_store.snapshot(root, ids):
                    manifest = json.loads(
                        (root / "manifest.json").read_text(encoding="utf-8")
                    )
                    class_labels = tuple(manifest.get("class_labels") or ())
                    status.phase("Computing MPMA-B selection and reports")
                    _write_selection_artifacts(root, child, class_labels)
                    status.phase("Validating selection outputs")
                    outputs = {
                        label: location
                        for label, location in _existing_outputs(root).items()
                        if label.startswith("mpma_b") and Path(location).exists()
                    }
                    if (
                        "mpma_b_final_candidate" not in outputs
                        or "mpma_b_selection" not in outputs
                    ):
                        raise RuntimeError(
                            "MPMA-B selection did not produce a final candidate and fold selection. "
                            "Check that sufficient complete inner and outer predictions exist."
                        )
                    status.phase("Recording source artifact identities")
                    digests = {}
                    for number, cid in enumerate(ids, start=1):
                        digests[cid] = _digest(_artifact_path(root, cid))
                        cooperative_store.report_progress(
                            "Hashing input artifacts", number, len(ids)
                        )
                    snapshot = {
                        "schema": "mllabiome-base-selection-snapshot-v1",
                        "candidate_config_ids": ids,
                        "candidate_digests": digests,
                        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
                        "split_digest": split_digest(root),
                        "optimize_metric": str(child.evaluation.optimize_metric),
                        "base_selection": asdict(child.base_selection)
                        if child.base_selection
                        else None,
                        "ensemble_selection_parameters": asdict(child.ensemble)
                        if child.ensemble
                        else None,
                        "outputs": {
                            key: str(value.relative_to(root))
                            for key, value in outputs.items()
                        },
                    }
                    fingerprint = hashlib.sha256(
                        json.dumps(snapshot, sort_keys=True, default=str).encode()
                    ).hexdigest()[:24]
                    snapshot["selection_id"] = fingerprint
                    folder = root / "selections" / "base" / fingerprint
                    folder.mkdir(parents=True, exist_ok=True)
                    cooperative_store.atomic_bytes(
                        folder / "manifest.json",
                        json.dumps(
                            snapshot, sort_keys=True, default=str, indent=2
                        ).encode("utf-8"),
                    )
                    for source, target in (
                        (
                            root / "tables" / "mpma_b_outer_selection.parquet",
                            folder / "outer_selection.parquet",
                        ),
                        (
                            root / "tables" / "mpma_b_final_candidate.json",
                            folder / "final_candidate.json",
                        ),
                    ):
                        if source.is_file() and not target.is_file():
                            cooperative_store.atomic_bytes(target, source.read_bytes())
                    results[str(child.data.target_col)] = {
                        "selection_id": fingerprint,
                        "snapshot": folder / "manifest.json",
                        **outputs,
                    }
                    info(
                        f"MPMA-B · snapshot {fingerprint} · {len(ids)} completed configurations"
                    )
                    final = json.loads(
                        (root / "tables" / "mpma_b_final_candidate.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    if final.get("config_id"):
                        success(
                            f"MPMA-B · selected {final['config_id']} · {final.get('learner', 'unknown learner')} · {final.get('resolution', 'unknown resolution')}"
                        )
    return results if len(children) > 1 else next(iter(results.values()))


def run_ensemble(sweep: Sweep) -> dict[str, Any]:
    from .joint_sweep import is_joint_sweep

    if is_joint_sweep(sweep):
        from .joint_ensemble import ensemble_joint

        outputs = ensemble_joint(sweep)
        print_stage_results(sweep, "ensemble")
        return outputs
    if getattr(sweep, "uses_modalities", False):
        outputs = sweep_ensemble(sweep)
        info("Validating and writing final MPMA model specifications")
        build_final_models(sweep.root())
        success("Final MPMA model specifications saved")
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
        info("Validating and writing final MPMA model specifications")
        build_final_models(child.root())
        success("Final MPMA model specifications saved")
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
    with inference_phase("Resolve target configuration"):
        children = target_sweeps(sweep)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    with inference_progress(len(children)) as tracker:
        for index, child in enumerate(children, start=1):
            target = str(child.data.target_col)
            if tracker is not None:
                tracker.target(index, target)
            else:
                info(f"Inference · target {index}/{len(children)} · {target}")
            include_mpma_e = "mpma_e" in child.inference.targets
            with inference_phase(f"Target {target} · rebuild final model selection"):
                models = build_final_models(child.root(), include_mpma_e=include_mpma_e)
            if tracker is not None:
                tracker.workflow_step("final model selection resolved")
            with inference_phase(f"Target {target} · deploy selected models"):
                child_outputs = execute_inference(child, models=models)
            outputs[target] = child_outputs
            records.append(_target_record(child, child_outputs))
            if tracker is not None:
                tracker.finish_target()
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
    from .joint_sweep import is_joint_sweep

    if is_joint_sweep(sweep):
        from .joint_explainability import explain_joint

        outputs = explain_joint(sweep)
        print_stage_results(sweep, "explain")
        return outputs
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
    from .joint_sweep import is_joint_sweep, report_joint

    if is_joint_sweep(sweep):
        outputs = report_joint(sweep)
        print_stage_results(sweep, "report")
        return outputs
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
    from .joint_sweep import is_joint_sweep

    if is_joint_sweep(sweep) and (
        sweep.inference is not None or sweep.robustness is not None
    ):
        raise NotImplementedError(
            "Joint-model inference and robustness require task-aware implementations before running all stages."
        )
    explore_output: dict[str, Any] = {}
    if not getattr(sweep, "uses_modalities", False):
        info("Pipeline · explore")
        explore_output = run_explore(sweep)
    info("Pipeline · evaluate")
    evaluate_output = run_evaluate(sweep)

    has_explicit_base = not is_joint_sweep(sweep) and not getattr(
        sweep, "uses_modalities", False
    )
    base_selection_output: dict[str, Any] = {}
    if has_explicit_base:
        info("Pipeline · base-selection")
        base_selection_output = run_base_selection(sweep)
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
    if base_selection_output:
        outputs["base-selection"] = base_selection_output
    if explore_output:
        outputs["explore"] = explore_output
    if getattr(sweep, "robustness", None) is not None:
        outputs["robustness"] = robustness_output
    if inference_output:
        outputs["inference"] = inference_output
    return outputs


def run_stage(sweep: Sweep, stage_name: str) -> dict[str, Any]:
    from .joint_sweep import is_joint_sweep

    if is_joint_sweep(sweep) and stage_name not in {
        "all",
        "explore",
        "evaluate",
        "base-selection",
        "ensemble",
        "explain",
        "report",
    }:
        raise NotImplementedError(
            f"Stage {stage_name!r} does not yet support joint multi-target models."
        )
    if stage_name == "all":
        return run_all(sweep)
    if stage_name == "explore":
        return run_explore(sweep)
    if stage_name == "federated-evaluate":
        from .federated.stage import federated_evaluate

        return federated_evaluate(sweep)
    if stage_name == "evaluate":
        return run_evaluate(sweep)
    if stage_name == "base-selection":
        return run_base_selection(sweep)
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
