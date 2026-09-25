from __future__ import annotations

from dataclasses import replace
from typing import Any

from .data import Data
from .explainability_methods import Permutation
from .sweep_types import Explainability, Sweep, _normalise_sweep_task


def _target_columns(data: Data) -> tuple[str, ...]:
    if isinstance(data.target_col, str):
        return (data.target_col,)
    return tuple(str(x) for x in data.target_col)


def _target_task(data: Data, target: str) -> str:
    root_task = _normalise_sweep_task(data.task)
    if root_task == "multilabel":
        return "classification"
    if root_task == "multioutput":
        if not data.target_tasks or target not in data.target_tasks:
            raise ValueError(
                f"Data.target_tasks must define task for multioutput target {target!r}."
            )
        value = _normalise_sweep_task(data.target_tasks[target])
        if value not in {"classification", "regression"}:
            raise ValueError(
                f"Multioutput target {target!r} must be classification or regression."
            )
        return value
    if data.target_tasks and target in data.target_tasks:
        value = _normalise_sweep_task(data.target_tasks[target])
        if value in {"classification", "regression"}:
            return value
    return root_task


def _learners_for_target(sweep: Sweep, target: str, task: str):
    learners = sweep.learners
    if not isinstance(learners, dict):
        return learners
    for key in (target, task, "default"):
        if key in learners:
            return learners[key]
    raise ValueError(f"No learners configured for target {target!r} ({task}).")


def _metric_for_target(value: Any, target: str, task: str, fallback: str) -> str:
    if isinstance(value, dict):
        for key in (target, task, "default"):
            if key in value:
                return str(value[key])
        return fallback
    return str(value)


def _explainability_for_target(
    plan: Explainability, task: str, optimize_metric: str
) -> Explainability:
    methods = []
    regression_scores = {
        "r2",
        "mae",
        "mse",
        "rmse",
        "medae",
        "explainedvariance",
        "explained_variance",
        "pearsonr",
        "spearmanr",
    }
    for method in plan.methods:
        if isinstance(method, Permutation):
            scoring = str(method.scoring).strip().casefold()
            if task == "regression":
                if scoring not in regression_scores:
                    fallback = str(optimize_metric).strip().casefold()
                    scoring = fallback if fallback in regression_scores else "rmse"
            elif scoring not in {"log_loss", "brier"}:
                scoring = "log_loss"
            method = replace(method, scoring=scoring)
        methods.append(method)
    return replace(plan, methods=tuple(methods))


def target_sweeps(sweep: Sweep) -> list[Sweep]:
    targets = _target_columns(sweep.data)
    if len(targets) == 1 and _normalise_sweep_task(sweep.data.task) not in {
        "multilabel",
        "multioutput",
    }:
        return [sweep]
    out = []
    for target in targets:
        task = _target_task(sweep.data, target)
        labels = sweep.data.class_labels
        if sweep.data.target_class_labels and target in sweep.data.target_class_labels:
            labels = tuple(sweep.data.target_class_labels[target])
        positive = sweep.data.positive_class
        if (
            sweep.data.target_positive_classes
            and target in sweep.data.target_positive_classes
        ):
            positive = sweep.data.target_positive_classes[target]
        extra_metadata = tuple(
            dict.fromkeys(
                (*sweep.data.metadata_cols, *[x for x in targets if x != target])
            )
        )
        child_data = replace(
            sweep.data,
            target_col=target,
            task=task,
            class_labels=labels,
            positive_class=positive,
            metadata_cols=extra_metadata,
        )
        default_metric = "RMSE" if task == "regression" else "log_loss"
        child_evaluation = replace(
            sweep.evaluation,
            optimize_metric=_metric_for_target(
                sweep.evaluation.optimize_metric, target, task, default_metric
            ),
        )
        child_ensemble = replace(
            sweep.ensemble,
            optimize_metric=_metric_for_target(
                sweep.ensemble.optimize_metric, target, task, default_metric
            ),
        )
        child_explainability = _explainability_for_target(
            sweep.explainability, task, str(child_evaluation.optimize_metric)
        )
        child = replace(
            sweep,
            data=child_data,
            experiment_dir=sweep.root() / "targets" / target,
            learners=_learners_for_target(sweep, target, task),
            evaluation=child_evaluation,
            ensemble=child_ensemble,
            explainability=child_explainability,
            title=f"{sweep.title} · {target}",
        )
        out.append(child)
    return out
