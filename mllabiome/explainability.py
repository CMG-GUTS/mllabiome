from __future__ import annotations

import inspect
import hashlib
import json
import logging
import math
import shutil
from contextlib import contextmanager
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from threading import Event, Thread
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_limits

from .configs_sweep import (
    Sweep,
    _effective_local_explanations_mode,
    _groups_from_metadata,
    _outer_splits,
    _strata_from_metadata,
)
from .console import info, path_table, progress, stage, success, summary_table
from .data import load_dataset
from .explainability_visuals import plot_feature_support as _plot_feature_support_visual
from .explainability_support import top_k_rank_support
from .explainability_methods import (
    ALE,
    ALEInteractions,
    LIME,
    Permutation,
    SHAP,
    coerce_method,
    method_has_global,
    method_has_local,
    method_name,
    method_to_dict,
)
from .explainability_visuals import (
    plot_interaction_network as _plot_interaction_network_visual,
    plot_local_attributions,
)
from .learners import _learner_factory
from .metrics import _predict_proba_aligned
from .resolutions import materialize_mpdr
from .runtime import (
    configure_estimator_threads,
    iter_parallel_tasks,
    resolve_execution_plan,
    thread_environment,
)
from .style import ACC_D, ACC_L, BG, COL_W_2, DIM, INK, MID, TRACK
from .style import apply as apply_style
from .style import save_all
from .transformations import CountTransformationAdapter, _count_transformation_factory
from .utils import _as_float_matrix, dump_json_standard, feature_tail_ellipsis
from .storage import read_table, write_table, table_exists, glob_tables


for _logger_name in ("PyALE", "PyALE._ALE_generic"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
    logging.getLogger(_logger_name).propagate = False


@contextmanager
def _quiet_pyale_info():
    saved_disable = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(saved_disable)


@contextmanager
class ExplainabilityConfigurationError(RuntimeError):
    pass


class ExplainabilityDependencyError(RuntimeError):
    pass


_EXPLAINABILITY_METHODS = {"shap", "lime", "ale", "permutation", "interactions"}


def _normalise_explainability_method_specs(methods: Sequence[Any]) -> tuple[Any, ...]:
    if not methods:
        raise ExplainabilityConfigurationError(
            "Explainability.methods must contain at least one explicit method."
        )
    specs: list[Any] = []
    unsupported: list[str] = []
    for method in methods:
        try:
            spec = coerce_method(method)
        except Exception:
            unsupported.append(str(method))
            continue
        if method_name(spec) not in _EXPLAINABILITY_METHODS:
            unsupported.append(str(method))
            continue
        specs.append(spec)
    if unsupported:
        raise ExplainabilityConfigurationError(
            "Unsupported explainability method(s): "
            + ", ".join(unsupported)
            + ". Supported methods are: "
            + ", ".join(sorted(_EXPLAINABILITY_METHODS))
            + "."
        )
    names = [method_name(x) for x in specs]
    if len(set(names)) != len(names):
        raise ExplainabilityConfigurationError(
            "Explainability.methods must not contain duplicate method types."
        )
    return tuple(specs)


def _normalise_explainability_methods(methods: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        method_name(x) for x in _normalise_explainability_method_specs(methods)
    )


def _method_spec(methods: Sequence[Any], name: str) -> Any:
    target = str(name)
    for spec in _normalise_explainability_method_specs(methods):
        if method_name(spec) == target:
            return spec
    raise ExplainabilityConfigurationError(
        f"Explainability method {target!r} is not configured."
    )


def _resolve_explainability_classes(dataset: Any, configured: Any) -> tuple[int, ...]:
    n_classes = len(dataset.class_labels)
    if isinstance(configured, str):
        token = configured.strip().lower()
        if token in {"", "auto"}:
            if n_classes == 2:
                value = getattr(dataset, "positive_class", None)
                return (int(1 if value is None else value),)
            return tuple(range(n_classes))
        if token == "all":
            return tuple(range(n_classes))
        configured = (configured,)
    indices: list[int] = []
    labels = [str(x) for x in dataset.class_labels]
    for value in configured:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
        else:
            text = str(value).strip()
            matches = [
                i
                for i, label in enumerate(labels)
                if label.casefold() == text.casefold()
            ]
            if len(matches) == 1:
                idx = int(matches[0])
            else:
                try:
                    idx = int(text)
                except ValueError as exc:
                    raise ExplainabilityConfigurationError(
                        f"Unknown explainability class {value!r}. Available classes: {labels!r}."
                    ) from exc
        if idx < 0 or idx >= n_classes:
            raise ExplainabilityConfigurationError(
                f"Explainability class index {idx} is outside 0..{n_classes - 1}."
            )
        if idx not in indices:
            indices.append(idx)
    if not indices:
        raise ExplainabilityConfigurationError(
            "No explainability classes were selected."
        )
    return tuple(indices)


def _class_slug(label: Any) -> str:
    text = str(label).strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "class"


def _auto_ale_bins(n_samples: int, spec: ALE | ALEInteractions) -> int:
    if isinstance(spec.bins, (int, np.integer)):
        return max(2, int(spec.bins))
    value = int(np.floor(np.sqrt(max(1, int(n_samples)))))
    return int(max(int(spec.min_bins), min(int(spec.max_bins), value)))


def _preflight_explainability_dependencies(methods: Sequence[str]) -> None:
    missing: list[str] = []
    if "shap" in methods:
        try:
            import shap
        except Exception as exc:
            missing.append(f"shap ({exc})")
    if "lime" in methods:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except Exception as exc:
            missing.append(f"lime ({exc})")
    if "ale" in methods or "interactions" in methods:
        try:
            from PyALE import ale as _pyale_preflight
        except Exception as exc:
            missing.append(f"PyALE ({exc})")
    if "interactions" in methods:
        try:
            import networkx as nx
        except Exception as exc:
            missing.append(f"networkx ({exc})")
    if missing:
        raise ExplainabilityDependencyError(
            "Strict explainability cannot start because requested method dependencies are missing: "
            + "; ".join(missing)
            + ". Install the required package dependencies or explicitly remove the corresponding method(s). No fallback method will be used."
        )


def _configured_count_transformation_factory(
    sweep: Sweep, key: str
) -> Callable[[], CountTransformationAdapter]:
    factories = dict(
        _count_transformation_factory(x, random_state=sweep.explainability.random_state)
        for x in sweep.count_transformations
    )
    if key not in factories:
        raise ExplainabilityConfigurationError(
            f"The selected MPMA uses count transformation {key!r}, but that exact transformation "
            "is not present in sweep.count_transformations. Explainability is strict and will not "
            "reconstruct or substitute transformations by name."
        )
    return factories[key]


def _configured_learner_factory(sweep: Sweep, key: str) -> Callable[[], BaseEstimator]:
    factories = dict(_learner_factory(x) for x in sweep.learners)
    if key not in factories:
        raise ExplainabilityConfigurationError(
            f"The selected MPMA uses learner {key!r}, but that exact learner is not present in "
            "sweep.learners. Explainability is strict and will not reconstruct or substitute learners by name."
        )
    return factories[key]


class _FittedMpmaEnsemble(BaseEstimator):
    def __init__(self, members: list[dict[str, Any]], aggregation: str = "mean_proba"):
        self.members = members
        self.aggregation = aggregation
        self.classes_ = (
            np.asarray(members[0]["classes"], dtype=int)
            if members
            else np.array([0, 1], dtype=int)
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_float_matrix(X)
        probs = []
        for member in self.members:
            sl = member["slice"]
            clf = member["estimator"]
            probs.append(_predict_proba_aligned(clf, X[:, sl], self.classes_))
        stack = np.stack(probs, axis=0)
        return _aggregate_member_proba(stack, self.aggregation)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _aggregate_member_proba(stack: np.ndarray, aggregation: str) -> np.ndarray:
    aggregation = str(aggregation or "mean_proba")
    if aggregation == "median_proba":
        proba = np.median(stack, axis=0)
    elif aggregation == "rank_mean":
        ranks = np.empty_like(stack, dtype=float)
        for m in range(stack.shape[0]):
            for i in range(stack.shape[1]):
                ranks[m, i, :] = rankdata(stack[m, i, :], method="average")
        proba = ranks.mean(axis=0)
    elif aggregation == "majority_vote":
        votes = np.argmax(stack, axis=2)
        proba = np.zeros((stack.shape[1], stack.shape[2]), dtype=float)
        for i in range(votes.shape[1]):
            counts = np.bincount(votes[:, i], minlength=stack.shape[2]).astype(float)
            proba[i] = counts / max(float(counts.sum()), 1.0)
    else:
        proba = stack.mean(axis=0)
    proba = np.asarray(proba, dtype=float)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return proba / row_sums


def _explain_predict_proba(
    clf: BaseEstimator, classes: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X_: _predict_proba_aligned(clf, _as_float_matrix(X_), classes)


def _explain_predict_class_probability(
    clf: BaseEstimator, classes: np.ndarray, class_index: int
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X_: _predict_proba_aligned(clf, _as_float_matrix(X_), classes)[
        :, int(class_index)
    ]


def _sample_rows(X: np.ndarray, *, max_rows: int, random_state: int) -> np.ndarray:
    X = _as_float_matrix(X)
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return np.arange(X.shape[0])
    rng = np.random.default_rng(int(random_state))
    return np.sort(rng.choice(np.arange(X.shape[0]), size=int(max_rows), replace=False))


def _xai_execution_plan(sweep: Sweep, task_count: int):
    configured = getattr(sweep.explainability, "n_jobs", None)
    n_jobs = (
        getattr(sweep.evaluation, "n_jobs", 1) if configured is None else configured
    )
    backend = getattr(sweep.explainability, "parallel_backend", None) or getattr(
        sweep.evaluation, "parallel_backend", "loky"
    )
    return resolve_execution_plan(
        n_jobs,
        max(1, int(task_count)),
        backend=str(backend),
        memory_fraction=float(getattr(sweep.evaluation, "memory_fraction", 0.80)),
        min_worker_memory_gib=float(
            getattr(sweep.evaluation, "min_worker_memory_gib", 1.0)
        ),
    )


def _run_xai_task(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    threads_per_worker: int,
) -> Any:
    with (
        thread_environment(threads_per_worker),
        threadpool_limits(limits=max(1, int(threads_per_worker))),
    ):
        return fn(*args, **kwargs)


def _xai_task_iterator(
    tasks: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
    execution: Any,
):
    payloads = [
        (
            _run_xai_task,
            (fn, args, kwargs, int(execution.threads_per_worker)),
            {},
        )
        for fn, args, kwargs in tasks
    ]
    yield from iter_parallel_tasks(payloads, execution)


def _progress_callback(
    prog: Any, task_id: Any, prefix: str
) -> Callable[[int, int, str], None]:
    def update(completed: int, total: int, detail: str) -> None:
        total_i = max(1, int(total))
        prog.update(
            task_id,
            total=total_i,
            completed=min(max(0, int(completed)), total_i),
            description=f"{prefix} · {detail}",
        )

    return update


def _queued_progress_callback(
    progress_queue: Any, fold_no: int, every: int = 5
) -> Callable[[int, int, str], None]:
    last_sent = -1

    def update(completed: int, total: int, detail: str) -> None:
        nonlocal last_sent
        completed_i = max(0, int(completed))
        total_i = max(1, int(total))
        if (
            completed_i == 0
            or completed_i >= total_i
            or completed_i - last_sent >= max(1, int(every))
        ):
            progress_queue.put((int(fold_no), completed_i, total_i, str(detail)))
            last_sent = completed_i

    return update


def _parallel_progress_results(
    tasks: Sequence[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]],
    execution: Any,
    progress_queue: Any,
    *,
    label: str,
    unit_label: str,
    fold_totals: dict[int, int],
) -> list[Any]:
    total_work = max(1, sum(max(1, int(v)) for v in fold_totals.values()))
    fold_progress = {int(k): 0 for k in fold_totals}
    stop_monitor = Event()
    results: list[Any] = []
    with progress() as prog:
        task = prog.add_task(
            f"{label} · 0/{total_work:,} {unit_label} · {execution.workers} workers",
            total=total_work,
        )

        def monitor() -> None:
            while not stop_monitor.is_set():
                try:
                    fold_no, completed, total, detail = progress_queue.get(timeout=0.25)
                except Empty:
                    continue
                fold_no = int(fold_no)
                cap = max(1, int(fold_totals.get(fold_no, total)))
                fold_progress[fold_no] = max(
                    fold_progress.get(fold_no, 0),
                    min(max(0, int(completed)), cap),
                )
                overall = min(total_work, sum(fold_progress.values()))
                prog.update(
                    task,
                    completed=overall,
                    description=(
                        f"{label} · {overall:,}/{total_work:,} {unit_label} · "
                        f"fold {fold_no}/{len(fold_totals)} · {detail}"
                    ),
                )

        monitor_thread = Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            for result in _xai_task_iterator(tasks, execution):
                results.append(result)
                fold_no = int(result[0])
                fold_progress[fold_no] = max(1, int(fold_totals.get(fold_no, 1)))
                overall = min(total_work, sum(fold_progress.values()))
                prog.update(
                    task,
                    completed=overall,
                    description=(
                        f"{label} · {overall:,}/{total_work:,} {unit_label} · "
                        f"completed fold {fold_no}/{len(fold_totals)}"
                    ),
                )
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=2.0)
        prog.update(
            task,
            completed=total_work,
            description=f"{label} · completed {total_work:,}/{total_work:,} {unit_label}",
        )
    return results


def _permutation_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    *,
    spec: Permutation,
    random_state: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    classes = np.arange(len(class_labels), dtype=int)
    X = _as_float_matrix(X)
    y = np.asarray(y, dtype=int)
    scoring_name = str(spec.scoring).strip().lower()
    if scoring_name not in {"log_loss", "brier"}:
        raise ExplainabilityConfigurationError(
            "Permutation.scoring must be 'log_loss' or 'brier'."
        )
    n_samples, n_features = X.shape
    n_repeats = int(spec.n_repeats)
    if isinstance(spec.max_samples, float):
        n_draw = max(1, min(n_samples, int(round(float(spec.max_samples) * n_samples))))
    else:
        n_draw = max(1, min(n_samples, int(spec.max_samples)))
    seed_rng = np.random.RandomState(int(random_state))
    random_seed = int(seed_rng.randint(np.iinfo(np.int32).max + 1))
    full_baseline_proba = _predict_proba_aligned(clf, X, classes)
    values = np.full((len(class_indices), n_features, n_repeats), np.nan, dtype=float)
    total = max(1, n_features * n_repeats)
    done = 0
    if progress_callback is not None:
        progress_callback(0, total, f"0/{n_features} features")

    def scores(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
        out = np.empty(len(class_indices), dtype=float)
        eps = np.finfo(float).eps
        for pos, class_index in enumerate(class_indices):
            c = int(class_index)
            target = (np.asarray(y_true, dtype=int) == c).astype(float)
            p = np.clip(np.asarray(proba, dtype=float)[:, c], eps, 1.0 - eps)
            if scoring_name == "brier":
                out[pos] = float(np.mean((target - p) ** 2))
            else:
                out[pos] = float(
                    -np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
                )
        return out

    baseline = scores(y, full_baseline_proba)
    for feature_index, feature in enumerate(feature_names):
        feature_rng = np.random.RandomState(random_seed)
        if n_draw < n_samples:
            idx = np.sort(feature_rng.choice(n_samples, size=n_draw, replace=False))
            X_eval = X[idx].copy()
            y_eval = y[idx]
        else:
            X_eval = X.copy()
            y_eval = y
        shuffling_idx = np.arange(len(X_eval))
        for repeat_index in range(n_repeats):
            feature_rng.shuffle(shuffling_idx)
            X_eval[:, feature_index] = X_eval[shuffling_idx, feature_index]
            permuted = _predict_proba_aligned(clf, X_eval, classes)
            values[:, feature_index, repeat_index] = scores(y_eval, permuted) - baseline
            done += 1
            if progress_callback is not None:
                progress_callback(
                    done,
                    total,
                    f"feature {feature_index + 1}/{n_features} · repeat {repeat_index + 1}/{n_repeats} · {feature}",
                )

    rows: list[pd.DataFrame] = []
    for pos, class_index in enumerate(class_indices):
        c = int(class_index)
        arr = values[pos]
        rows.append(
            pd.DataFrame(
                {
                    "method": "permutation",
                    "class_index": c,
                    "class_label": str(class_labels[c]),
                    "feature": list(feature_names),
                    "importance_mean": np.nanmean(arr, axis=1),
                    "within_fold_importance_sd": np.nanstd(arr, axis=1, ddof=1)
                    if n_repeats > 1
                    else np.zeros(n_features, dtype=float),
                    "scoring": f"increase_in_one_vs_rest_{scoring_name}",
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["class_index", "importance_mean", "feature"],
        ascending=[True, False, True],
    )


class _AleModelWrapper:
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self._fn = fn

    def predict(self, X: Any) -> np.ndarray:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.asarray(self._fn(arr), dtype=float)


def _require_pyale():
    try:
        from PyALE import ale as pyale
    except Exception as exc:
        raise ExplainabilityDependencyError(
            "Explainability.methods includes 'ale' or 'interactions', but the 'PyALE' package is not importable. "
            "Install PyALE or remove ALE-based methods from Explainability.methods. No fallback method will be used."
        ) from exc
    return pyale


def _ale_result_values(
    result: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    x1 = x2 = None
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        if "eff" in df.columns:
            vals = df["eff"].to_numpy(dtype=float)
        else:
            num = df.select_dtypes(include=[np.number])
            drop_cols = [
                c
                for c in num.columns
                if str(c).lower() in {"size", "count", "counts", "n"}
                or "ci" in str(c).lower()
            ]
            if drop_cols:
                num = num.drop(columns=drop_cols, errors="ignore")
            vals = num.to_numpy(dtype=float).ravel()
        if isinstance(df.index, pd.MultiIndex) and len(df.index) == len(vals):
            x1 = np.asarray([idx[0] for idx in df.index], dtype=float)
            x2 = np.asarray([idx[1] for idx in df.index], dtype=float)
        elif not isinstance(df.index, pd.MultiIndex) and len(df.index) == len(vals):
            try:
                x1 = df.index.to_numpy(dtype=float)
            except Exception:
                x1 = None
    elif isinstance(result, pd.Series):
        vals = result.to_numpy(dtype=float)
        if isinstance(result.index, pd.MultiIndex) and len(result.index) == len(vals):
            x1 = np.asarray([idx[0] for idx in result.index], dtype=float)
            x2 = np.asarray([idx[1] for idx in result.index], dtype=float)
        elif len(result.index) == len(vals):
            try:
                x1 = result.index.to_numpy(dtype=float)
            except Exception:
                x1 = None
    else:
        vals = np.asarray(result, dtype=float).ravel()
    vals = np.asarray(vals, dtype=float).ravel()
    finite = np.isfinite(vals)
    if x1 is not None and x2 is not None:
        finite = finite & np.isfinite(x1) & np.isfinite(x2)
        return vals[finite], x1[finite], x2[finite]
    if x1 is not None and x2 is None and len(x1) == len(vals):
        finite = finite & np.isfinite(x1)
        return vals[finite], x1[finite], None
    return vals[finite], None, None


def _ale_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    *,
    spec: ALE,
    top_features: Sequence[str] | None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    pyale = _require_pyale()
    X = _as_float_matrix(X)
    feature_names = list(feature_names)
    name_to_index = {str(name): i for i, name in enumerate(feature_names)}
    features = list(top_features) if top_features else feature_names
    features = [str(f) for f in features if str(f) in name_to_index]
    if not features:
        raise ExplainabilityConfigurationError(
            "ALE was requested, but no candidate features were available."
        )
    df_X = pd.DataFrame(X, columns=feature_names)
    n_bins = _auto_ale_bins(X.shape[0], spec)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    classes = np.arange(len(class_labels), dtype=int)
    total = max(1, len(class_indices) * len(features))
    done = 0
    if progress_callback is not None:
        progress_callback(
            0, total, f"0/{len(features)} features · {len(class_indices)} classes"
        )
    for class_index in class_indices:
        c = int(class_index)
        wrapper = _AleModelWrapper(_explain_predict_class_probability(clf, classes, c))
        for fname in features:
            try:
                col = X[:, name_to_index[fname]]
                finite_col = col[np.isfinite(col)]
                n_distinct = int(np.unique(finite_col).size)
                if n_distinct < 2:
                    skipped.append(
                        {
                            "class_index": c,
                            "class_label": str(class_labels[c]),
                            "feature": fname,
                            "reason": "fewer_than_two_distinct_finite_values",
                            "n_finite": int(finite_col.size),
                            "n_distinct": n_distinct,
                        }
                    )
                    continue
                try:
                    with _quiet_pyale_info():
                        result = pyale(
                            df_X,
                            wrapper,
                            [fname],
                            grid_size=int(n_bins),
                            include_CI=False,
                            plot=False,
                        )
                    vals, grid, _ = _ale_result_values(result)
                except Exception as exc:
                    skipped.append(
                        {
                            "class_index": c,
                            "class_label": str(class_labels[c]),
                            "feature": fname,
                            "reason": "pyale_failed",
                            "error": str(exc)[:500],
                            "n_finite": int(finite_col.size),
                            "n_distinct": n_distinct,
                        }
                    )
                    continue
                if vals.size == 0:
                    skipped.append(
                        {
                            "class_index": c,
                            "class_label": str(class_labels[c]),
                            "feature": fname,
                            "reason": "no_finite_ale_effect_values",
                            "n_finite": int(finite_col.size),
                            "n_distinct": n_distinct,
                        }
                    )
                    continue
                centred = vals - float(np.mean(vals))
                strength = float(np.sqrt(np.mean(centred**2)))
                rows.append(
                    {
                        "method": "ale",
                        "class_index": c,
                        "class_label": str(class_labels[c]),
                        "feature": fname,
                        "importance_mean": strength,
                        "within_fold_importance_sd": float(np.std(centred, ddof=1))
                        if len(centred) > 1
                        else 0.0,
                        "scoring": "rms_centered_class_probability_ale",
                    }
                )
                if grid is not None and len(grid) == len(vals):
                    for x_value, effect in zip(grid, vals):
                        curves.append(
                            {
                                "class_index": c,
                                "class_label": str(class_labels[c]),
                                "feature": fname,
                                "grid_value": float(x_value),
                                "ale_effect": float(effect),
                            }
                        )
            finally:
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total, f"{class_labels[c]} · {fname}")
    frame = pd.DataFrame(rows)
    frame.attrs["curves"] = pd.DataFrame(curves)
    frame.attrs["skipped_features"] = pd.DataFrame(skipped)
    return (
        frame.sort_values(
            ["class_index", "importance_mean", "feature"],
            ascending=[True, False, True],
        )
        if not frame.empty
        else frame
    )


def _same_lineage_pair(f1: str, f2: str) -> bool:
    return str(f1).startswith(str(f2)) or str(f2).startswith(str(f1))


def _candidate_pairs_from_scores(
    X: np.ndarray, feature_names: Sequence[str], scores: pd.Series, max_pairs: int
) -> list[tuple[int, int]]:
    X = _as_float_matrix(X)
    if X.shape[1] < 2 or max_pairs <= 0:
        return []
    name_to_score = {str(k): float(v) for k, v in scores.items()}

    def usable(i: int) -> bool:
        x = X[:, i]
        return (
            np.isfinite(x).sum() >= 3
            and np.unique(x[np.isfinite(x)]).size >= 2
            and not str(feature_names[i]).startswith("ILR")
        )

    idx = [i for i in range(X.shape[1]) if usable(i)]
    if len(idx) < 2:
        return []
    order = sorted(
        idx,
        key=lambda i: name_to_score.get(
            str(feature_names[i]), float(np.nanvar(X[:, i]))
        ),
        reverse=True,
    )

    frontier = order[
        : min(len(order), max(4, int(math.ceil(math.sqrt(max_pairs * 2))) + 6))
    ]
    cand: list[tuple[float, int, int]] = []
    for a, i in enumerate(frontier):
        for j in frontier[a + 1 :]:
            if _same_lineage_pair(str(feature_names[i]), str(feature_names[j])):
                continue
            si = name_to_score.get(str(feature_names[i]), float(np.nanvar(X[:, i])))
            sj = name_to_score.get(str(feature_names[j]), float(np.nanvar(X[:, j])))
            cand.append((float(si + sj), i, j))
    cand.sort(key=lambda z: z[0], reverse=True)
    return [(i, j) for _, i, j in cand[:max_pairs]]


def _interp_unique(x: np.ndarray, y: np.ndarray, xp: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        raise ValueError("not enough finite support")
    ser = pd.Series(y, index=x).groupby(level=0).mean().sort_index()
    if len(ser) < 2:
        raise ValueError("not enough unique support")
    return np.interp(xp, ser.index.to_numpy(dtype=float), ser.to_numpy(dtype=float))


def _ale_interactions(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_index: int,
    *,
    spec: ALEInteractions,
    scores: pd.Series,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    pyale = _require_pyale()
    X = _as_float_matrix(X)
    df_X = pd.DataFrame(X, columns=list(feature_names))
    c = int(class_index)
    wrapper = _AleModelWrapper(
        _explain_predict_class_probability(
            clf, np.arange(len(class_labels), dtype=int), c
        )
    )
    n_bins = _auto_ale_bins(X.shape[0], spec)
    pairs = _candidate_pairs_from_scores(
        X, feature_names, scores, max_pairs=int(spec.top_k)
    )
    if not pairs:
        raise ExplainabilityConfigurationError(
            "Interactions were requested, but no usable feature pairs were available."
        )
    raw_rows = []
    total = len(pairs)
    if progress_callback is not None:
        progress_callback(0, total, f"0/{total} pairs")
    for pair_no, (i, j) in enumerate(pairs, start=1):
        f1, f2 = str(feature_names[i]), str(feature_names[j])
        try:
            try:
                with _quiet_pyale_info():
                    result2d = pyale(
                        df_X,
                        wrapper,
                        [f1, f2],
                        grid_size=int(n_bins),
                        include_CI=False,
                        plot=False,
                    )
                vals2d, x1, x2 = _ale_result_values(result2d)
            except Exception:
                continue
            if vals2d.size == 0:
                continue
            centred = vals2d - float(np.mean(vals2d))
            current = float(np.sqrt(np.mean(centred**2)))
            corrected = float("nan")
            correction_applied = False
            correction_error = ""
            if x1 is None or x2 is None:
                correction_error = "2D PyALE result does not expose coordinates for marginal correction"
            else:
                try:
                    with _quiet_pyale_info():
                        res1 = pyale(
                            df_X,
                            wrapper,
                            [f1],
                            grid_size=int(n_bins),
                            include_CI=False,
                            plot=False,
                        )
                        res2 = pyale(
                            df_X,
                            wrapper,
                            [f2],
                            grid_size=int(n_bins),
                            include_CI=False,
                            plot=False,
                        )
                    vals1, g1, _ = _ale_result_values(res1)
                    vals2, g2, _ = _ale_result_values(res2)
                    if g1 is None or g2 is None:
                        raise ValueError("1D PyALE result does not expose grids")
                    comp = (
                        vals2d
                        - _interp_unique(g1, vals1, x1)
                        - _interp_unique(g2, vals2, x2)
                    )
                    comp = comp - float(np.mean(comp))
                    corrected = float(np.sqrt(np.mean(comp**2)))
                    correction_applied = True
                except Exception as exc:
                    correction_error = str(exc)[:240]
            raw_rows.append(
                {
                    "class_index": c,
                    "class_label": str(class_labels[c]),
                    "feature_1": f1,
                    "feature_2": f2,
                    "current": current,
                    "corrected": corrected,
                    "fixed_pairs": current,
                    "corrected_fixed": corrected,
                    "n_ale_cells": int(vals2d.size),
                    "n_bins": int(n_bins),
                    "correction_applied": bool(correction_applied),
                    "correction_error": correction_error,
                }
            )
        finally:
            if progress_callback is not None:
                progress_callback(
                    pair_no, total, f"pair {pair_no}/{total} · {f1} × {f2}"
                )
    if not raw_rows:
        raise ExplainabilityConfigurationError(
            "ALE interactions were requested, but no candidate pair produced a finite 2D ALE surface."
        )
    raw = pd.DataFrame(raw_rows)
    out: dict[str, pd.DataFrame] = {}
    for method in ("current", "corrected", "fixed_pairs", "corrected_fixed"):
        d = raw[
            [
                "class_index",
                "class_label",
                "feature_1",
                "feature_2",
                method,
                "n_ale_cells",
                "n_bins",
                "correction_applied",
                "correction_error",
            ]
        ].copy()
        d = d.rename(columns={method: "interaction_strength"})
        d["method"] = method
        d = d.sort_values("interaction_strength", ascending=False, na_position="last")
        out[method] = d
    out["all_methods"] = raw
    return out


def _collapse_duplicate_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "feature" not in frame.columns:
        return frame.copy()
    keys = ["feature"]
    if "class_index" in frame.columns:
        keys = ["class_index", "feature"]
    if not frame.duplicated(keys).any():
        return frame.copy()
    d = frame.copy()
    d["feature"] = d["feature"].astype(str)
    d["importance_mean"] = pd.to_numeric(
        d.get("importance_mean", np.nan), errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    if "importance_sd" not in d.columns:
        d["importance_sd"] = np.nan
    agg: dict[str, Any] = {
        "importance_mean": lambda x: pd.to_numeric(x, errors="coerce").sum(min_count=1),
        "importance_sd": lambda x: float(
            np.sqrt(
                np.nansum(
                    np.square(pd.to_numeric(x, errors="coerce").to_numpy(dtype=float))
                )
            )
        ),
    }
    for column in (
        "method",
        "class_label",
        "scoring",
        "n_methods",
        "n_methods_total",
        "method_coverage",
        "n_estimable_folds",
        "n_outer_folds_total",
        "fold_coverage",
        "importance_median",
        "importance_q25",
        "importance_q75",
        "mean_rank",
        "median_rank",
        "rank_iqr",
        "top_k_frequency",
        "signed_importance_mean",
        "sign_positive_fraction",
        "sign_negative_fraction",
        "sign_consistency",
    ):
        if column in d.columns:
            agg[column] = "first"
    return d.groupby(keys, as_index=False, sort=False).agg(agg)


def _rank_support_from_importance(
    frame: pd.DataFrame, top_k: int | None = None
) -> pd.Series:
    d = _collapse_duplicate_feature_importance(frame).copy()
    if "class_index" not in d.columns:
        d["class_index"] = 0
    pieces: list[pd.Series] = []
    for class_index, sub in d.groupby("class_index", sort=True):
        values = pd.to_numeric(sub["importance_mean"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = values.notna()
        if not valid.any():
            continue
        ranked = values.loc[valid].rank(ascending=False, method="average")
        n = int(valid.sum())
        if top_k is None:
            support = (
                pd.Series(1.0, index=ranked.index, dtype=float)
                if n == 1
                else 1.0 - (ranked - 1.0) / float(n - 1)
            )
        else:
            support = top_k_rank_support(values.loc[valid], int(top_k))
        index = pd.MultiIndex.from_arrays(
            [
                np.full(len(support), int(class_index), dtype=int),
                sub.loc[valid, "feature"].astype(str).to_numpy(),
            ],
            names=["class_index", "feature"],
        )
        pieces.append(
            pd.Series(support.to_numpy(dtype=float), index=index, dtype=float)
        )
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def _combine_feature_importance(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        return _collapse_duplicate_feature_importance(frames[0]).sort_values(
            ["class_index", "importance_mean", "feature"]
            if "class_index" in frames[0].columns
            else ["importance_mean", "feature"],
            ascending=[True, False, True]
            if "class_index" in frames[0].columns
            else [False, True],
            na_position="last",
        )
    long: list[pd.DataFrame] = []
    method_names: list[str] = []
    labels: dict[int, str] = {}
    for frame in frames:
        collapsed = _collapse_duplicate_feature_importance(frame)
        method = str(collapsed["method"].iloc[0])
        method_names.append(method)
        if {"class_index", "class_label"}.issubset(collapsed.columns):
            for _, row in (
                collapsed[["class_index", "class_label"]].drop_duplicates().iterrows()
            ):
                labels[int(row["class_index"])] = str(row["class_label"])
        support = _rank_support_from_importance(collapsed)
        if support.empty:
            continue
        d = support.rename("rank_support").reset_index()
        d["method"] = method
        d["normalized_rank"] = 1.0 - d["rank_support"]
        long.append(d)
    if not long:
        raise ExplainabilityConfigurationError(
            "No finite feature-importance values were available for consensus aggregation."
        )
    pooled = pd.concat(long, ignore_index=True)
    grouped = pooled.groupby(["class_index", "feature"], as_index=False).agg(
        importance_mean=("rank_support", "mean"),
        importance_sd=("rank_support", "std"),
        mean_rank=("normalized_rank", "mean"),
        n_methods=("method", "nunique"),
    )
    grouped["importance_sd"] = grouped["importance_sd"].fillna(0.0)
    grouped["consensus_score"] = grouped["importance_mean"]
    grouped["consensus_sd"] = grouped["importance_sd"]
    grouped["n_methods_total"] = int(len(dict.fromkeys(method_names)))
    grouped["method_coverage"] = grouped["n_methods"] / grouped["n_methods_total"]
    grouped["method"] = "consensus"
    grouped["scoring"] = "mean_within_method_rank_support_available_methods"
    grouped["class_label"] = (
        grouped["class_index"]
        .map(labels)
        .fillna(grouped["class_index"].map(lambda x: f"class_{int(x)}"))
    )
    grouped = grouped.sort_values(
        ["class_index", "importance_mean", "n_methods", "feature"],
        ascending=[True, False, False, True],
    )
    return grouped[
        [
            "method",
            "class_index",
            "class_label",
            "feature",
            "importance_mean",
            "importance_sd",
            "consensus_score",
            "consensus_sd",
            "scoring",
            "mean_rank",
            "n_methods",
            "n_methods_total",
            "method_coverage",
        ]
    ]


def _single_method_support_table(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    frame = _collapse_duplicate_feature_importance(frame)
    if "class_index" not in frame.columns:
        frame["class_index"] = 0
        frame["class_label"] = "class_0"
    method = _method_display(str(frame["method"].iloc[0]))
    support = _rank_support_from_importance(frame, top_k=top_k)
    pieces: list[pd.DataFrame] = []
    for class_index, sub in frame.groupby("class_index", sort=True):
        top = (
            sub.sort_values("importance_mean", ascending=False, na_position="last")
            .head(int(top_k))
            .copy()
        )
        top["rank"] = np.arange(1, len(top) + 1, dtype=int)
        keys = pd.MultiIndex.from_arrays(
            [
                np.full(len(top), int(class_index), dtype=int),
                top["feature"].astype(str).to_numpy(),
            ]
        )
        top[method] = support.reindex(keys).to_numpy(dtype=float)
        top["consensus"] = top[method].astype(float)
        pieces.append(top)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    keep = [
        "class_index",
        "class_label",
        "rank",
        "feature",
        method,
        "consensus",
        "importance_mean",
        "importance_sd",
        "mean_rank",
        "median_rank",
        "rank_iqr",
        "top_k_frequency",
        "fold_coverage",
        "sign_consistency",
    ]
    return out[[c for c in keep if c in out.columns]]


def _write_fold_feature_importance(
    method: str, frames: Sequence[pd.DataFrame], target_dir: Path
) -> tuple[Path, pd.DataFrame]:
    prepared: list[pd.DataFrame] = []
    for fold_no, frame in enumerate(frames, start=1):
        d = frame.copy()
        d.attrs = {}
        d["method"] = str(method)
        if "fold_no" not in d.columns:
            d["fold_no"] = int(fold_no)
        if "fold_key" not in d.columns:
            d["fold_key"] = str(fold_no)
        prepared.append(d)
    out = pd.concat(prepared, ignore_index=True) if prepared else pd.DataFrame()
    path = target_dir / f"feature_importance_{method}_by_outer_fold.parquet"
    write_table(path, out)
    return path, out


def _write_method_outputs(
    frame: pd.DataFrame,
    *,
    target_dir: Path,
    figures_dir: Path,
    X_base: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_labels: list[str],
    top_k: int,
) -> dict[str, Path]:
    method = str(frame["method"].iloc[0]).strip().lower()
    table_path = target_dir / f"feature_importance_{method}.parquet"
    write_table(table_path, frame)
    stability_columns = [
        "method",
        "class_index",
        "class_label",
        "feature",
        "importance_mean",
        "importance_sd",
        "importance_median",
        "importance_q25",
        "importance_q75",
        "mean_rank",
        "median_rank",
        "rank_iqr",
        "top_k_frequency",
        "n_estimable_folds",
        "n_outer_folds_total",
        "fold_coverage",
        "signed_importance_mean",
        "sign_positive_fraction",
        "sign_negative_fraction",
        "sign_consistency",
    ]
    stability_path = target_dir / f"feature_stability_{method}.parquet"
    write_table(
        stability_path, frame[[c for c in stability_columns if c in frame.columns]]
    )
    top = _single_method_support_table(frame, top_k=top_k)
    top_path = target_dir / f"top_features_{method}.parquet"
    write_table(top_path, top)
    outputs: dict[str, Path] = {
        f"importance_{method}": table_path,
        f"stability_{method}": stability_path,
        f"top_features_{method}": top_path,
    }
    if top.empty:
        return outputs
    for class_index, class_top in top.groupby("class_index", sort=True):
        c = int(class_index)
        label = str(class_labels[c]) if 0 <= c < len(class_labels) else f"class_{c}"
        slug = _class_slug(label)
        dist = _feature_distribution_stats(
            class_top["feature"].astype(str).tolist(),
            feature_names,
            X_base,
            y,
            class_labels,
        )
        dist_path = target_dir / f"feature_distribution_stats_{method}__{slug}.parquet"
        write_table(dist_path, dist)
        importance_stem = figures_dir / f"feature_importance_{method}__{slug}"
        _plot_feature_importance(
            class_top, dist, importance_stem, top_k, class_labels, frame
        )
        outputs[f"feature_distribution_{method}_{slug}"] = dist_path
        outputs[f"figure_{method}_{slug}"] = importance_stem.with_suffix(".svg")
    return outputs


def _plot_ale_curves(
    curves: pd.DataFrame,
    top_features: Sequence[str],
    out_stem: Path,
    *,
    max_panels: int = 12,
) -> None:
    apply_style()
    import math as _math

    import matplotlib.patheffects as mpe
    import matplotlib.pyplot as plt

    from .style import ACC_D, ACC_L, BG, INK, MID, MM, TRACK, save_all

    if (
        curves is None
        or curves.empty
        or not {"feature", "grid", "ale_effect"}.issubset(curves.columns)
    ):
        fig = plt.figure(figsize=(120 * MM, 40 * MM))
        ax = fig.add_axes([0.06, 0.15, 0.88, 0.70])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "ALE curves unavailable",
            ha="center",
            va="center",
            fontsize=7,
            color=MID,
        )
        save_all(fig, out_stem)
        plt.close(fig)
        return
    order = [
        str(f) for f in top_features if str(f) in set(curves["feature"].astype(str))
    ]
    if not order:
        order = list(dict.fromkeys(curves["feature"].astype(str).tolist()))
    order = order[: max(1, int(max_panels))]
    n = len(order)
    n_cols = 3 if n > 4 else min(2, n)
    n_rows = int(_math.ceil(n / max(n_cols, 1)))
    fig_w = 150 if n_cols == 3 else 112
    fig_h = max(44, 31 * n_rows + 10)
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))
    fig.patch.set_facecolor(BG)
    left, right, bottom, top = 0.075, 0.035, 0.095, 0.055
    gap_x, gap_y = 0.055, 0.115
    cell_w = (1 - left - right - gap_x * (n_cols - 1)) / n_cols
    cell_h = (1 - bottom - top - gap_y * (n_rows - 1)) / n_rows
    for idx, feat in enumerate(order):
        row, col = divmod(idx, n_cols)
        x0 = left + col * (cell_w + gap_x)
        y0 = 1 - top - (row + 1) * cell_h - row * gap_y
        ax = fig.add_axes([x0, y0, cell_w, cell_h])
        d = curves[curves["feature"].astype(str).eq(feat)].copy()
        d["grid"] = pd.to_numeric(d["grid"], errors="coerce")
        d["ale_effect"] = pd.to_numeric(d["ale_effect"], errors="coerce")
        d = (
            d.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["grid", "ale_effect"])
            .sort_values("grid")
        )
        ax.set_facecolor(BG)
        if d.empty:
            ax.text(
                0.5,
                0.5,
                "unavailable",
                ha="center",
                va="center",
                fontsize=5.5,
                color=MID,
                transform=ax.transAxes,
            )
        else:
            fold_col = next(
                (c for c in ("fold_key", "fold_no", "outer_fold") if c in d.columns),
                None,
            )
            grouped = (
                list(d.groupby(fold_col, sort=False, dropna=False))
                if fold_col is not None
                else [("all", d)]
            )
            fold_curves: list[tuple[np.ndarray, np.ndarray]] = []
            ax.axhline(0, color=TRACK, lw=0.55, zorder=1)
            for _, fold_frame in grouped:
                fold_frame = (
                    fold_frame.groupby("grid", as_index=False)["ale_effect"]
                    .mean()
                    .sort_values("grid")
                )
                x_fold = fold_frame["grid"].to_numpy(float)
                y_fold = fold_frame["ale_effect"].to_numpy(float)
                finite = np.isfinite(x_fold) & np.isfinite(y_fold)
                x_fold = x_fold[finite]
                y_fold = y_fold[finite]
                if not len(x_fold):
                    continue
                y_fold = y_fold - float(np.mean(y_fold))
                fold_curves.append((x_fold, y_fold))
                if len(grouped) > 1:
                    ax.plot(
                        x_fold,
                        y_fold,
                        color=TRACK,
                        lw=0.55,
                        alpha=0.85,
                        zorder=2,
                        solid_capstyle="round",
                    )
            if len(fold_curves) == 1:
                x, y = fold_curves[0]
                ax.plot(
                    x,
                    y,
                    color=ACC_D,
                    lw=0.95,
                    zorder=4,
                    solid_capstyle="round",
                )
                ax.fill_between(x, 0, y, color=ACC_L, alpha=0.42, zorder=3, linewidth=0)
            elif len(fold_curves) > 1:
                lower = max(float(np.min(x)) for x, _ in fold_curves)
                upper = min(float(np.max(x)) for x, _ in fold_curves)
                if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                    lower = min(float(np.min(x)) for x, _ in fold_curves)
                    upper = max(float(np.max(x)) for x, _ in fold_curves)
                if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                    grid_size = max(
                        32, min(128, max(len(x) for x, _ in fold_curves) * 4)
                    )
                    x_common = np.linspace(lower, upper, grid_size)
                    stack = np.full((len(fold_curves), grid_size), np.nan, dtype=float)
                    for fold_index, (x_fold, y_fold) in enumerate(fold_curves):
                        supported = (x_common >= float(np.min(x_fold))) & (
                            x_common <= float(np.max(x_fold))
                        )
                        if np.any(supported):
                            stack[fold_index, supported] = np.interp(
                                x_common[supported], x_fold, y_fold
                            )
                    required = min(2, len(fold_curves))
                    supported_columns = np.sum(np.isfinite(stack), axis=0) >= required
                    if np.any(supported_columns):
                        x_plot = x_common[supported_columns]
                        values = stack[:, supported_columns]
                        median = np.nanmedian(values, axis=0)
                        q25 = np.nanquantile(values, 0.25, axis=0)
                        q75 = np.nanquantile(values, 0.75, axis=0)
                        ax.fill_between(
                            x_plot,
                            q25,
                            q75,
                            color=ACC_L,
                            alpha=0.52,
                            zorder=3,
                            linewidth=0,
                        )
                        ax.plot(
                            x_plot,
                            median,
                            color=ACC_D,
                            lw=1.0,
                            zorder=4,
                            solid_capstyle="round",
                        )
        ax.text(
            0.0,
            1.055,
            _plain_taxon_label(feat, 46),
            ha="left",
            va="bottom",
            fontsize=5.2,
            color=INK,
            transform=ax.transAxes,
            path_effects=[mpe.withStroke(linewidth=1.0, foreground="white")],
        )
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color("#000000")
        ax.spines["bottom"].set_linewidth(0.45)
        ax.tick_params(
            axis="x", length=2.0, width=0.4, labelsize=5.0, pad=1, colors="#000000"
        )
        ax.tick_params(axis="y", length=0, labelsize=5.0, pad=1, colors=MID)
        ax.yaxis.set_ticks_position("none")
        try:
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
        except Exception:
            pass
    save_all(fig, out_stem)
    plt.close(fig)


def _explainability_config_payload(explainability: Any) -> dict[str, Any]:
    return {
        "profile": str(getattr(explainability, "profile", "standard")),
        "methods": [
            method_to_dict(x)
            for x in _normalise_explainability_method_specs(explainability.methods)
        ],
        "classes": explainability.classes,
        "top_k": int(explainability.top_k),
        "random_state": int(explainability.random_state),
        "local": {
            "representatives": bool(explainability.local.representatives),
            "sample_ids": list(explainability.local.sample_ids),
            "stored_features": int(explainability.local.stored_features),
            "displayed_features": int(explainability.local.displayed_features),
            "regression_quantiles": [
                float(x) for x in explainability.local.regression_quantiles
            ],
        },
        "local_explanations": _effective_local_explanations_mode(explainability),
        "effective_local_explanations": _effective_local_explanations_mode(
            explainability
        ),
        "representative_instances": bool(explainability.local.representatives),
        "instance_sample_ids": list(explainability.local.sample_ids),
        "local_top_k": int(explainability.local.stored_features),
        "top_instance_features": int(explainability.local.displayed_features),
        "representative_quantiles": [
            float(x) for x in explainability.local.regression_quantiles
        ],
    }


def _explainability_config_signature(explainability: Any) -> str:
    payload = _explainability_config_payload(explainability)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _signature_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_signature_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_signature_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _signature_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _signature_hash(payload: Any) -> str:
    raw = json.dumps(
        _signature_value(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _explainability_source_signature(
    target_slug: str,
    row: pd.Series,
    oof_folds: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    class_labels: Sequence[Any],
) -> str:
    config_keys = (
        "unit",
        "config_id",
        "resolution",
        "levels",
        "count_transformation",
        "learner",
        "members",
        "aggregation_strategy",
    )
    config = {key: row.get(key) for key in config_keys if key in row.index}
    return _signature_hash(
        {
            "target": str(row.get("unit", row.get("config_id", target_slug))),
            "config": config,
            "features": list(feature_names),
            "classes": [str(x) for x in class_labels],
            "folds": [
                {
                    "split_key": str(fold.get("split_key", "")),
                    "train_idx": np.asarray(fold.get("train_idx", []), dtype=int),
                    "test_idx": np.asarray(fold.get("test_idx", []), dtype=int),
                }
                for fold in oof_folds
            ],
        }
    )


def _method_cache_signature(
    explainability: Any,
    spec: Any,
    source_signature: str,
    class_indices: Sequence[int],
) -> str:
    name = method_name(spec)
    payload: dict[str, Any] = {
        "method": name,
        "parameters": method_to_dict(spec),
        "source_signature": str(source_signature),
        "classes": [int(x) for x in class_indices],
        "random_state": int(explainability.random_state),
    }
    return _signature_hash(payload)


def _method_cache_files_complete(target_dir: Path, method: str) -> bool:
    if method == "interactions":
        return table_exists(target_dir / "feature_interactions_current.parquet")
    required = [
        target_dir / f"feature_importance_{method}.parquet",
        target_dir / f"feature_stability_{method}.parquet",
        target_dir / f"feature_importance_{method}_by_outer_fold.parquet",
        target_dir / f"top_features_{method}.parquet",
    ]
    return all(table_exists(path) for path in required)


def _source_config_compatible(meta: dict[str, Any], row: pd.Series) -> bool:
    old = meta.get("config", {})
    keys = (
        "unit",
        "config_id",
        "resolution",
        "levels",
        "count_transformation",
        "learner",
        "members",
        "aggregation_strategy",
    )
    compared = False
    for key in keys:
        if key not in old or key not in row.index:
            continue
        a = _signature_value(old.get(key))
        b = _signature_value(row.get(key))
        if a is None or b is None:
            continue
        compared = True
        if a != b:
            return False
    return compared


def _legacy_method_cache_valid(
    meta: dict[str, Any],
    row: pd.Series,
    explainability: Any,
    spec: Any,
    class_indices: Sequence[int],
) -> bool:
    if not meta or not _source_config_compatible(meta, row):
        return False
    old_classes = [int(x) for x in meta.get("explained_class_indices", [])]
    if old_classes != [int(x) for x in class_indices]:
        return False
    params = meta.get("method_parameters", [])
    names = [str(x).strip().lower() for x in meta.get("methods", [])]
    current = _signature_value(method_to_dict(spec))
    matched = False
    for index, item in enumerate(params):
        try:
            old_name = names[index] if index < len(names) else ""
            if not old_name:
                keys = set(item)
                if {"algorithm", "masker"}.issubset(keys):
                    old_name = "shap"
                elif {"n_repeats", "scoring", "max_samples"}.issubset(keys):
                    old_name = "permutation"
                elif {"num_samples", "feature_selection"}.issubset(keys):
                    old_name = "lime"
                elif "top_k" in keys and "bins" in keys:
                    old_name = "interactions"
                elif "bins" in keys:
                    old_name = "ale"
            if old_name == method_name(spec) and _signature_value(item) == current:
                matched = True
                break
        except Exception:
            continue
    if not matched:
        return False
    old_cfg = meta.get("explainability_config", {})
    if int(old_cfg.get("random_state", explainability.random_state)) != int(
        explainability.random_state
    ):
        return False
    return True


def _method_cache_entry_status(
    entry: dict[str, Any],
    signature: str,
    source_signature: str,
    spec: Any,
    class_indices: Sequence[int],
) -> tuple[bool, str]:
    if not entry:
        return False, "no per-method provenance"
    if str(entry.get("source_signature", "")) != str(source_signature):
        return False, "selected model/data/splits changed"
    if _signature_value(entry.get("parameters", {})) != _signature_value(
        method_to_dict(spec)
    ):
        return False, "method parameters changed"
    if [int(x) for x in entry.get("class_indices", [])] != [
        int(x) for x in class_indices
    ]:
        return False, "explained classes changed"
    if str(entry.get("signature", "")) != str(signature):
        return False, "method random state or cache schema changed"
    return True, "cache hit"


def _method_cache_entry(
    signature: str,
    source_signature: str,
    spec: Any,
    class_indices: Sequence[int],
    top_k: int,
) -> dict[str, Any]:
    return {
        "cache_version": 4,
        "signature": str(signature),
        "source_signature": str(source_signature),
        "parameters": method_to_dict(spec),
        "class_indices": [int(x) for x in class_indices],
        "top_k": int(top_k),
    }


def _method_cache_sidecar_path(target_dir: Path, name: str) -> Path:
    return target_dir / ".method_cache" / f"{_safe_cache_name(name)}.json"


def _load_method_cache_entry(
    target_dir: Path,
    name: str,
    shared_cache: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = _method_cache_sidecar_path(target_dir, name)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload, "sidecar"
        except Exception:
            pass
    entry = shared_cache.get(str(name), {})
    if isinstance(entry, dict) and entry:
        return dict(entry), "shared metadata"
    return {}, "none"


def _persist_method_cache_entry(
    meta_path: Path,
    previous_meta: dict[str, Any],
    name: str,
    entry: dict[str, Any],
) -> None:
    sidecar = _method_cache_sidecar_path(meta_path.parent, name)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    dump_json_standard(dict(entry), sidecar)
    current = dict(previous_meta)
    if meta_path.exists():
        try:
            disk = json.loads(meta_path.read_text())
            if isinstance(disk, dict):
                current.update(disk)
        except Exception:
            pass
    cache = dict(current.get("method_cache", {}))
    cache[str(name)] = dict(entry)
    current["method_cache"] = cache
    dump_json_standard(current, meta_path)
    previous_meta.clear()
    previous_meta.update(current)


def _cached_method_frame(
    target_dir: Path, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_table(target_dir / f"feature_importance_{method}.parquet")
    fold = read_table(target_dir / f"feature_importance_{method}_by_outer_fold.parquet")
    return frame, fold


def _existing_method_outputs(target_dir: Path, method: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    mapping = {
        f"importance_{method}": target_dir / f"feature_importance_{method}.parquet",
        f"stability_{method}": target_dir / f"feature_stability_{method}.parquet",
        f"top_features_{method}": target_dir / f"top_features_{method}.parquet",
        f"{method}_by_outer_fold": target_dir
        / f"feature_importance_{method}_by_outer_fold.parquet",
    }
    for key, path in mapping.items():
        if table_exists(path):
            outputs[key] = path
    for path in glob_tables(target_dir, f"feature_distribution_stats_{method}__*"):
        outputs[path.stem] = path
    figures = target_dir / "figures"
    for pattern in (f"feature_importance_{method}__*.svg",):
        for path in sorted(figures.glob(pattern)):
            outputs[path.stem] = path
    if method == "ale":
        curve_table = target_dir / "ale_curves.parquet"
        if table_exists(curve_table):
            outputs["ale_curves"] = curve_table
        for path in sorted(figures.glob("ale_curves__*.svg")):
            outputs[path.stem] = path
    if method == "interactions":
        for path in glob_tables(target_dir, "feature_interactions_*"):
            outputs[path.stem] = path
        for path in sorted(figures.glob("interaction_network_*.svg")):
            outputs[path.stem] = path
    if method in {"shap", "lime"}:
        path = target_dir / f"instance_explanations_{method}_top_features.parquet"
        if table_exists(path):
            outputs[f"instance_explanations_{method}"] = path
    return outputs


def _visual_class_labels(sweep: Sweep, target_dir: Path) -> tuple[str, ...]:
    labels = getattr(sweep.data, "class_labels", None)
    if labels is not None:
        try:
            values = tuple(str(x) for x in labels)
        except TypeError:
            values = ()
        if values:
            return values
    label_map = getattr(sweep.data, "label_map", None)
    if isinstance(label_map, dict) and label_map:
        try:
            return tuple(
                str(value)
                for _, value in sorted(label_map.items(), key=lambda item: int(item[0]))
            )
        except Exception:
            return tuple(str(value) for value in label_map.values())
    meta_path = target_dir / "explained_unit.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        labels = (
            meta.get("explained_class_labels", []) if isinstance(meta, dict) else []
        )
        if isinstance(labels, list) and labels:
            return tuple(str(x) for x in labels)
    return ()


def _refresh_target_visuals(target_dir: Path, sweep: Sweep) -> dict[str, Path]:
    if not target_dir.exists():
        return {}
    class_labels = _visual_class_labels(sweep, target_dir)
    top_k = int(getattr(sweep.explainability, "top_k", 15))
    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    def render(
        top_path: Path, stats_path_for: Callable[[str], Path], stem_prefix: str
    ) -> None:
        if not table_exists(top_path):
            return
        try:
            top = read_table(top_path)
        except Exception:
            return
        if top.empty or "feature" not in top.columns:
            return
        if "class_index" in top.columns:
            groups = list(top.groupby("class_index", sort=True))
        else:
            groups = [(0, top)]
        for class_index, class_top in groups:
            try:
                c = int(class_index)
            except Exception:
                c = 0
            if class_labels and 0 <= c < len(class_labels):
                label = class_labels[c]
            elif "class_label" in class_top.columns and not class_top.empty:
                label = str(class_top.iloc[0].get("class_label", f"class_{c}"))
            else:
                label = f"class_{c}"
            slug = _class_slug(label)
            stats_path = stats_path_for(slug)
            if not table_exists(stats_path):
                continue
            try:
                stats = read_table(stats_path)
            except Exception:
                continue
            if "class_index" in stats.columns:
                selected = stats[
                    pd.to_numeric(stats["class_index"], errors="coerce").eq(c)
                ]
                if not selected.empty:
                    stats = selected
            kinds = ("feature_support",) if not stem_prefix else ("feature_importance",)
            for kind in kinds:
                stem = figures_dir / f"{kind}{stem_prefix}__{slug}"
                _plot_feature_importance(
                    class_top, stats, stem, top_k, class_labels or (label,)
                )
                path = stem.with_suffix(".svg")
                if path.exists():
                    outputs[path.stem + "_svg"] = path

    render(
        target_dir / "top_features.parquet",
        lambda slug: target_dir / "feature_distribution_stats.parquet",
        "",
    )
    for top_path in glob_tables(target_dir, "top_features_*"):
        method = top_path.stem[len("top_features_") :]
        if not method:
            continue
        render(
            top_path,
            lambda slug, method=method: (
                target_dir / f"feature_distribution_stats_{method}__{slug}.parquet"
            ),
            f"_{method}",
        )
    return outputs


def refresh_explainability_visuals(root: Path | str, sweep: Sweep) -> dict[str, Path]:
    exp_root = Path(root) / "explainability"
    outputs: dict[str, Path] = {}
    if not exp_root.exists():
        return outputs
    for slug in ("mpma_b", "mpma_e", "baseline_rf"):
        target_dir = exp_root / slug
        for key, path in _refresh_target_visuals(target_dir, sweep).items():
            outputs[f"{slug}_{key}"] = path
    return outputs


def _safe_cache_name(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _explainability_cache_complete(target_dir: Path, explainability: Any) -> bool:
    meta_path = target_dir / "explained_unit.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if str(
        meta.get("explainability_config_signature", "")
    ) != _explainability_config_signature(explainability):
        return False
    specs = _normalise_explainability_method_specs(explainability.methods)
    global_methods = {method_name(spec) for spec in specs if method_has_global(spec)}
    local_methods = {method_name(spec) for spec in specs if method_has_local(spec)}
    if global_methods:
        if not table_exists(target_dir / "feature_importance.parquet"):
            return False
        if not table_exists(target_dir / "top_features.parquet"):
            return False
        figs = target_dir / "figures"
        if not (
            (figs / "feature_support.svg").exists()
            or (figs / "feature_support.png").exists()
            or list(figs.glob("feature_support__*.svg"))
            or list(figs.glob("feature_support__*.png"))
        ):
            return False
        for method in ("shap", "lime", "ale", "permutation"):
            if method in global_methods and not table_exists(
                target_dir / f"feature_importance_{method}.parquet"
            ):
                return False
        if "interactions" in global_methods and not any(
            (target_dir / name).exists()
            for name in (
                "feature_interactions_current.parquet",
                "feature_interactions_corrected.parquet",
                "feature_interactions_fixed_pairs.parquet",
                "feature_interactions_corrected_fixed.parquet",
            )
        ):
            return False
    if local_methods and _effective_local_explanations_mode(explainability) != "none":
        if not table_exists(target_dir / "local_explanations.parquet"):
            return False
    return True


def _existing_explainability_outputs(target_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {"explainability_dir": target_dir}
    candidates = {
        "importance": target_dir / "feature_importance.parquet",
        "stability": target_dir / "feature_stability.parquet",
        "local_explanations": target_dir / "local_explanations.parquet",
        "local_explanations_figure": target_dir / "figures" / "local_explanations.svg",
    }
    for key, path in candidates.items():
        if path.exists():
            outputs[key] = path
    return outputs


def _copy_explainability_cache(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _ensure_mpma_member_explanations(
    sweep: Sweep, rankings: pd.DataFrame, members: Sequence[str]
) -> None:
    if not members:
        return
    methods = _normalise_explainability_methods(sweep.explainability.methods)
    total = len(members)
    info(f"Preparing cached member explainability for MPMA-E · members={total}")
    for i, member in enumerate(members, start=1):
        matches = rankings[rankings["config_id"].astype(str).eq(str(member))]
        if matches.empty:
            matches = rankings[
                rankings["config_id"]
                .astype(str)
                .map(lambda x: x.startswith(str(member)) or str(member).startswith(x))
            ]
        cid = str(matches.iloc[0]["config_id"]) if not matches.empty else str(member)
        cache_dir = sweep.root() / "explainability" / "cache" / _safe_cache_name(cid)
        if _explainability_cache_complete(cache_dir, sweep.explainability):
            info(f"MPMA-E member {i}/{total} · config={cid} · cached")
            continue
        if matches.empty:
            raise ExplainabilityConfigurationError(
                f"MPMA-E member {member!r} cannot be matched to mpma_rankings.parquet for cached explanation."
            )
        row = matches.iloc[0]
        desc = " · ".join(
            [
                str(row.get("resolution", row.get("levels", "MPDR"))),
                str(row.get("count_transformation", "transformation")),
                str(row.get("learner", "learner")),
            ]
        )
        info(f"MPMA-E member {i}/{total} · config={cid} · {desc}")
        _explain_one(
            sweep,
            target_override=cid,
            output_slug_override=f"cache/{_safe_cache_name(cid)}",
            display_label_override=f"MPMA-E member {i}/{total}",
            allow_member_cache=False,
        )


def _configured_explainability_targets(explainability: Any) -> str | tuple[str, ...]:
    return getattr(explainability, "targets", "auto")


def _single_configured_explainability_target(explainability: Any) -> str:
    configured = _configured_explainability_targets(explainability)
    if isinstance(configured, str):
        return configured
    values = tuple(str(x) for x in configured)
    if len(values) != 1:
        raise ExplainabilityConfigurationError(
            "Internal error: _explain_one requires one explainability target. "
            "Use explain(sweep) to process multiple targets."
        )
    return values[0]


def _shap_values_for_data(
    clf: BaseEstimator,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    random_state: int,
    spec: SHAP,
    force_explain_rows: Sequence[int] = (),
    show_progress: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import shap
    except Exception as exc:
        raise ExplainabilityDependencyError(
            "Explainability.methods includes SHAP, but the 'shap' package is not importable."
        ) from exc
    X_background = _as_float_matrix(X_background)
    X_explain = _as_float_matrix(X_explain)
    rows_bg = _sample_rows(
        X_background, max_rows=int(spec.background_size), random_state=random_state
    )
    rows_ex = _sample_rows(
        X_explain, max_rows=int(spec.max_explain), random_state=random_state + 13
    )
    if force_explain_rows:
        forced = np.asarray(
            [int(i) for i in force_explain_rows if 0 <= int(i) < X_explain.shape[0]],
            dtype=int,
        )
        if forced.size:
            rows_ex = np.array(
                sorted(set(rows_ex.tolist()) | set(forced.tolist())), dtype=int
            )
    background = X_background[rows_bg]
    X_selected = X_explain[rows_ex]
    total_samples = max(1, len(X_selected))
    if progress_callback is not None:
        progress_callback(0, total_samples, f"0/{len(X_selected)} samples")

    def normalize_values(values: Any) -> np.ndarray:
        raw = getattr(values, "values", values)
        if isinstance(raw, list):
            arr = np.stack([np.asarray(x, dtype=float) for x in raw], axis=-1)
        else:
            arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2:
            if len(class_labels) == 2:
                arr = arr[:, :, None]
            else:
                raise ExplainabilityConfigurationError(
                    f"Unexpected multiclass SHAP value shape {arr.shape}."
                )
        if arr.ndim != 3 or arr.shape[1] != len(feature_names):
            raise ExplainabilityConfigurationError(
                f"Unexpected SHAP value shape {arr.shape}; expected n_samples × n_features × n_classes."
            )
        if arr.shape[2] not in {1, len(class_labels)}:
            raise ExplainabilityConfigurationError(
                f"Unexpected SHAP output dimension {arr.shape[2]} for {len(class_labels)} classes."
            )
        return arr

    def evaluate_in_batches(
        call: Callable[[np.ndarray], Any], backend: str
    ) -> np.ndarray:
        if progress_callback is None or len(X_selected) <= 1:
            return normalize_values(call(X_selected))
        batch_size = max(1, min(25, len(X_selected)))
        chunks: list[np.ndarray] = []
        for start in range(0, len(X_selected), batch_size):
            stop = min(len(X_selected), start + batch_size)
            chunks.append(normalize_values(call(X_selected[start:stop])))
            progress_callback(
                stop,
                total_samples,
                f"sample {stop}/{len(X_selected)} · backend={backend}",
            )
        return np.concatenate(chunks, axis=0)

    requested_algorithm = str(spec.algorithm).strip().lower()
    if requested_algorithm in {"auto", "tree"}:
        try:
            explainer = shap.TreeExplainer(
                clf,
                data=background,
                model_output="probability",
                feature_perturbation="interventional",
                feature_names=list(feature_names),
            )
            arr = evaluate_in_batches(explainer, "tree")
            if progress_callback is not None and len(X_selected) <= 1:
                progress_callback(
                    len(X_selected),
                    total_samples,
                    f"sample {len(X_selected)}/{len(X_selected)} · backend=tree",
                )
            return arr, rows_ex, "tree"
        except Exception as exc:
            if requested_algorithm == "tree":
                raise ExplainabilityConfigurationError(
                    "TreeSHAP was requested but the fitted estimator is not supported in probability space."
                ) from exc
    classes = np.arange(len(class_labels), dtype=int)
    model_fn = _explain_predict_proba(clf, classes)
    masker_name = str(spec.masker).strip().lower()
    if masker_name == "independent":
        masker = shap.maskers.Independent(background, max_samples=len(background))
    elif masker_name == "partition":
        masker = shap.maskers.Partition(
            background, max_samples=len(background), clustering="correlation"
        )
    else:
        raise ExplainabilityConfigurationError(
            f"Unsupported SHAP masker {spec.masker!r}. Use 'independent' or 'partition'."
        )
    algorithm = "permutation" if requested_algorithm == "auto" else requested_algorithm
    minimum = 2 * len(feature_names) + 1
    max_evals = max(minimum, minimum * max(1, int(spec.permutation_rounds)))
    try:
        explainer = shap.Explainer(
            model_fn,
            masker,
            algorithm=algorithm,
            feature_names=list(feature_names),
            output_names=list(class_labels),
            seed=int(random_state),
        )

        def call_explainer(batch: np.ndarray) -> Any:
            try:
                return explainer(
                    batch,
                    silent=not bool(show_progress)
                    if progress_callback is None
                    else True,
                    max_evals=max_evals,
                )
            except TypeError:
                try:
                    return explainer(batch, max_evals=max_evals)
                except TypeError:
                    return explainer(batch)

        arr = evaluate_in_batches(call_explainer, algorithm)
        if progress_callback is not None and len(X_selected) <= 1:
            progress_callback(
                len(X_selected),
                total_samples,
                f"sample {len(X_selected)}/{len(X_selected)} · backend={algorithm}",
            )
    except Exception as exc:
        raise ExplainabilityConfigurationError(
            "SHAP failed for an outer-test fold."
        ) from exc
    return arr, rows_ex, algorithm


def _value_frame_for_fold(
    method: str,
    values: np.ndarray,
    feature_names: Sequence[str],
    class_indices: Sequence[int],
    class_labels: Sequence[str],
    scoring: str,
    positive_class: int | None = None,
) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[1] != len(feature_names):
        raise ExplainabilityConfigurationError(
            f"Unexpected {method.upper()} value shape {arr.shape}."
        )
    if arr.shape[2] == len(class_labels):
        selected = arr[:, :, list(class_indices)]
    elif arr.shape[2] == 1 and len(class_labels) == 2:
        positive = 1 if positive_class is None else int(positive_class)
        selected = np.stack(
            [
                arr[:, :, 0] if int(class_index) == positive else -arr[:, :, 0]
                for class_index in class_indices
            ],
            axis=2,
        )
    elif arr.shape[2] == len(class_indices):
        selected = arr
    elif arr.shape[2] == 1 and len(class_indices) == 1:
        selected = arr
    else:
        raise ExplainabilityConfigurationError(
            f"{method.upper()} output classes do not match configured explanation classes."
        )
    rows: list[dict[str, Any]] = []
    for pos, class_index in enumerate(class_indices):
        vals = selected[:, :, pos]
        abs_vals = np.abs(vals)
        mean_abs = np.nanmean(abs_vals, axis=0)
        signed = np.nanmean(vals, axis=0)
        within_sd = (
            np.nanstd(abs_vals, axis=0, ddof=1)
            if vals.shape[0] > 1
            else np.zeros(vals.shape[1], dtype=float)
        )
        for j, feature in enumerate(feature_names):
            rows.append(
                {
                    "method": method,
                    "class_index": int(class_index),
                    "class_label": str(class_labels[int(class_index)]),
                    "feature": str(feature),
                    "importance_mean": float(mean_abs[j]),
                    "signed_importance_mean": float(signed[j]),
                    "within_fold_importance_sd": float(within_sd[j]),
                    "scoring": scoring,
                }
            )
    return pd.DataFrame(rows)


def _lime_values_for_data(
    clf: BaseEstimator,
    X_training: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    *,
    random_state: int,
    spec: LIME,
    force_explain_rows: Sequence[int] = (),
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        raise ExplainabilityDependencyError(
            "Explainability.methods includes LIME, but the 'lime' package is not importable."
        ) from exc

    X_training = _as_float_matrix(X_training)
    X_explain = _as_float_matrix(X_explain)
    rows = _sample_rows(
        X_explain, max_rows=int(spec.max_explain), random_state=random_state + 29
    )
    if force_explain_rows:
        forced = np.asarray(
            [int(i) for i in force_explain_rows if 0 <= int(i) < X_explain.shape[0]],
            dtype=int,
        )
        if forced.size:
            rows = np.array(
                sorted(set(rows.tolist()) | set(forced.tolist())), dtype=int
            )
    X_selected = X_explain[rows]
    predict_fn = _explain_predict_proba(clf, np.arange(len(class_labels), dtype=int))
    kwargs: dict[str, Any] = {
        "training_data": X_training,
        "feature_names": list(feature_names),
        "class_names": list(class_labels),
        "mode": "classification",
        "feature_selection": str(spec.feature_selection),
        "discretize_continuous": bool(spec.discretize_continuous),
        "random_state": int(random_state),
    }
    if spec.kernel_width is not None:
        kwargs["kernel_width"] = float(spec.kernel_width)
    total = max(1, len(X_selected))
    if progress_callback is not None:
        progress_callback(0, total, f"0/{len(X_selected)} samples")
    try:
        explainer = LimeTabularExplainer(**kwargs)
        coeffs = np.zeros(
            (X_selected.shape[0], len(feature_names), len(class_indices)), dtype=float
        )
        labels = tuple(int(x) for x in class_indices)
        explain_method = explainer.explain_instance
        try:
            signature = inspect.signature(explain_method)
            parameters = signature.parameters
            accepts_sampling_method = "sampling_method" in parameters or any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in parameters.values()
            )
        except (TypeError, ValueError):
            accepts_sampling_method = False
        sampling_method = str(spec.sampling_method)
        if not accepts_sampling_method and sampling_method != "gaussian":
            raise ExplainabilityConfigurationError(
                f"Installed LIME does not support sampling_method={sampling_method!r}; use 'gaussian' or upgrade LIME."
            )
        for i, row in enumerate(X_selected):
            explain_kwargs: dict[str, Any] = {
                "predict_fn": predict_fn,
                "num_features": len(feature_names),
                "num_samples": int(spec.num_samples),
                "labels": labels,
            }
            if accepts_sampling_method:
                explain_kwargs["sampling_method"] = sampling_method
            exp = explain_method(row, **explain_kwargs)
            for class_pos, label in enumerate(labels):
                for fi, coef in exp.local_exp.get(label, []):
                    if 0 <= int(fi) < len(feature_names):
                        coeffs[i, int(fi), class_pos] = float(coef)
            if progress_callback is not None:
                progress_callback(i + 1, total, f"sample {i + 1}/{len(X_selected)}")
    except ExplainabilityConfigurationError:
        raise
    except Exception as exc:
        raise ExplainabilityConfigurationError(
            "LIME failed for an outer-test fold."
        ) from exc
    return coeffs, rows


def _aggregate_fold_feature_importance(
    method: str,
    frames: Sequence[pd.DataFrame],
    feature_names: Sequence[str],
    class_indices: Sequence[int],
    class_labels: Sequence[str],
    scoring: str,
    top_k: int,
) -> pd.DataFrame:
    if not frames:
        raise ExplainabilityConfigurationError(
            f"{method.upper()} produced no outer-test fold results."
        )
    n_folds_total = len(frames)
    rows: list[dict[str, Any]] = []
    for class_index in class_indices:
        class_label = str(class_labels[int(class_index)])
        fold_tables: list[pd.DataFrame] = []
        for fold_no, frame in enumerate(frames, start=1):
            sub = frame[
                pd.to_numeric(frame.get("class_index", -1), errors="coerce")
                .fillna(-1)
                .astype(int)
                .eq(int(class_index))
            ].copy()
            sub = sub.set_index("feature").reindex(list(feature_names))
            vals = pd.to_numeric(sub.get("importance_mean", np.nan), errors="coerce")
            signed = (
                pd.to_numeric(sub["signed_importance_mean"], errors="coerce")
                if "signed_importance_mean" in sub.columns
                else pd.Series(np.nan, index=sub.index, dtype=float)
            )
            tab = pd.DataFrame(
                {
                    "feature": list(feature_names),
                    "importance": vals.to_numpy(dtype=float),
                    "signed": signed.to_numpy(dtype=float),
                    "fold_no": int(fold_no),
                }
            )
            valid = np.isfinite(tab["importance"].to_numpy(dtype=float))
            tab["rank"] = np.nan
            if np.any(valid):
                tab.loc[valid, "rank"] = tab.loc[valid, "importance"].rank(
                    ascending=False, method="average"
                )
                k = min(max(1, int(top_k)), int(valid.sum()))
                tab["top_k"] = False
                tab.loc[valid, "top_k"] = tab.loc[valid, "rank"] <= k
            else:
                tab["top_k"] = False
            fold_tables.append(tab)
        long = pd.concat(fold_tables, ignore_index=True)
        for feature in feature_names:
            sub = long[long["feature"].astype(str).eq(str(feature))].copy()
            imp = pd.to_numeric(sub["importance"], errors="coerce")
            valid = imp.notna() & np.isfinite(imp)
            impv = imp[valid].to_numpy(dtype=float)
            rankv = (
                pd.to_numeric(sub.loc[valid, "rank"], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            signed = pd.to_numeric(sub.loc[valid, "signed"], errors="coerce")
            signed = signed[np.isfinite(signed)].to_numpy(dtype=float)
            n = int(len(impv))
            if n == 0:
                rows.append(
                    {
                        "method": method,
                        "class_index": int(class_index),
                        "class_label": class_label,
                        "feature": str(feature),
                        "importance_mean": np.nan,
                        "importance_sd": np.nan,
                        "importance_median": np.nan,
                        "importance_q25": np.nan,
                        "importance_q75": np.nan,
                        "mean_rank": np.nan,
                        "median_rank": np.nan,
                        "rank_iqr": np.nan,
                        "top_k_frequency": np.nan,
                        "n_estimable_folds": 0,
                        "n_outer_folds_total": int(n_folds_total),
                        "fold_coverage": 0.0,
                        "signed_importance_mean": np.nan,
                        "sign_positive_fraction": np.nan,
                        "sign_negative_fraction": np.nan,
                        "sign_consistency": np.nan,
                        "scoring": scoring,
                    }
                )
                continue
            q25, med, q75 = np.quantile(impv, [0.25, 0.5, 0.75])
            if rankv.size:
                rq25, rmed, rq75 = np.quantile(rankv, [0.25, 0.5, 0.75])
            else:
                rq25 = rmed = rq75 = np.nan
            top_freq = float(sub.loc[valid, "top_k"].astype(float).mean())
            if signed.size:
                pos = float(np.mean(signed > 0.0))
                neg = float(np.mean(signed < 0.0))
                consistency = float(max(pos, neg))
                signed_mean = float(np.mean(signed))
            else:
                pos = neg = consistency = signed_mean = np.nan
            rows.append(
                {
                    "method": method,
                    "class_index": int(class_index),
                    "class_label": class_label,
                    "feature": str(feature),
                    "importance_mean": float(np.mean(impv)),
                    "importance_sd": float(np.std(impv, ddof=1)) if n > 1 else 0.0,
                    "importance_median": float(med),
                    "importance_q25": float(q25),
                    "importance_q75": float(q75),
                    "mean_rank": float(np.mean(rankv)) if rankv.size else np.nan,
                    "median_rank": float(rmed),
                    "rank_iqr": float(rq75 - rq25) if rankv.size else np.nan,
                    "top_k_frequency": top_freq,
                    "n_estimable_folds": int(n),
                    "n_outer_folds_total": int(n_folds_total),
                    "fold_coverage": float(n / n_folds_total),
                    "signed_importance_mean": signed_mean,
                    "sign_positive_fraction": pos,
                    "sign_negative_fraction": neg,
                    "sign_consistency": consistency,
                    "scoring": scoring,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["class_index", "importance_mean", "feature"],
        ascending=[True, False, True],
        na_position="last",
    )


def _explainability_outer_splits(sweep: Sweep, dataset: Any) -> list[dict[str, Any]]:
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    y = np.asarray(dataset.y, dtype=int)
    strata = _strata_from_metadata(dataset.metadata, y, sweep.data.stratify_col)
    splits = _outer_splits(sweep.evaluation, y, groups, strata, sweep.data.stratify_col)
    if not splits:
        raise ExplainabilityConfigurationError(
            "No outer splits are available for out-of-fold explainability."
        )
    return splits


def _ordered_member_rows(
    root: Path, rankings: pd.DataFrame, members: Sequence[str]
) -> pd.DataFrame:
    config_path = root / "configs.parquet"
    configs = read_table(config_path) if table_exists(config_path) else rankings
    configs = configs.copy()
    configs["config_id"] = configs["config_id"].astype(str)
    rows: list[pd.Series] = []
    for member in members:
        m = str(member)
        match = configs[configs["config_id"].eq(m)]
        if match.empty:
            match = configs[
                configs["config_id"].map(
                    lambda x: str(x).startswith(m) or m.startswith(str(x))
                )
            ]
        if match.empty:
            raise ExplainabilityConfigurationError(
                f"MPMA-E member {member!r} cannot be matched to configs.parquet."
            )
        rows.append(match.iloc[0])
    return pd.DataFrame(rows).drop_duplicates("config_id")


def _fit_oof_single_fold_task(
    split_no: int,
    split: dict[str, Any],
    X_base: np.ndarray,
    y: np.ndarray,
    class_count: int,
    ct_factory: Callable[[], Any],
    learner_factory: Callable[[], BaseEstimator],
    threads_per_worker: int,
) -> tuple[int, dict[str, Any] | None]:
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    if len(test_idx) == 0 or len(np.unique(y[train_idx])) < 2:
        return int(split_no), None
    ct = ct_factory()
    X_train, X_test = ct.apply_pair(X_base[train_idx], X_base[test_idx])
    clf = configure_estimator_threads(learner_factory(), threads_per_worker)
    clf.fit(X_train, y[train_idx])
    proba = _predict_proba_aligned(clf, X_test, np.arange(int(class_count), dtype=int))
    return int(split_no), {
        "split_key": str(split["split_key"]),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y[test_idx],
        "estimator": clf,
        "proba": proba,
    }


def _fit_oof_single_for_explainability(
    sweep: Sweep,
    row: pd.Series,
) -> dict[str, Any]:
    if getattr(sweep, "uses_modalities", False):
        from .multimodal_sweep import fit_modality_candidate_oof_for_explainability

        return fit_modality_candidate_oof_for_explainability(sweep, row)
    levels = tuple(str(row["levels"]).split(","))
    dataset = load_dataset(sweep.data, levels)
    X_base, feature_names = materialize_mpdr(dataset, levels)
    splits = _explainability_outer_splits(sweep, dataset)
    transformation_key = str(row["count_transformation"])
    learner_key = str(row["learner"])
    ct_factory = _configured_count_transformation_factory(sweep, transformation_key)
    learner_factory = _configured_learner_factory(sweep, learner_key)
    execution = _xai_execution_plan(sweep, len(splits))
    tasks = [
        (
            _fit_oof_single_fold_task,
            (
                split_no,
                split,
                X_base,
                dataset.y,
                len(dataset.class_labels),
                ct_factory,
                learner_factory,
                int(execution.threads_per_worker),
            ),
            {},
        )
        for split_no, split in enumerate(splits, start=1)
    ]
    folds_by_no: dict[int, dict[str, Any]] = {}
    with progress() as prog:
        task = prog.add_task(
            f"Fitting OOF explanation models · {execution.workers} workers · {execution.threads_per_worker} threads/worker",
            total=len(tasks),
        )
        for split_no, fold in _xai_task_iterator(tasks, execution):
            if fold is not None:
                folds_by_no[int(split_no)] = fold
            prog.advance(task)
    folds = [folds_by_no[i] for i in sorted(folds_by_no)]
    if not folds:
        raise ExplainabilityConfigurationError(
            "No outer fold could be fitted for out-of-fold explainability."
        )
    ct_reference = ct_factory()
    X_reference, _ = ct_reference.apply_pair(X_base, X_base)
    coordinate_metadata = ct_reference.coordinate_metadata(list(feature_names))
    transformed_feature_names = [str(item.name) for item in coordinate_metadata]
    if X_reference.shape[1] != len(transformed_feature_names):
        raise ExplainabilityConfigurationError(
            f"Transformation {transformation_key!r} produced feature metadata inconsistent with its transformed matrix."
        )
    for fold in folds:
        if fold["X_train"].shape[1] != len(transformed_feature_names) or fold[
            "X_test"
        ].shape[1] != len(transformed_feature_names):
            raise ExplainabilityConfigurationError(
                f"Transformation {transformation_key!r} produced inconsistent feature coordinates across outer folds."
            )
    return {
        "dataset": dataset,
        "X_base": np.asarray(X_reference, dtype=float),
        "feature_names": transformed_feature_names,
        "coordinate_metadata": coordinate_metadata,
        "folds": folds,
        "execution": execution,
    }


def _mpma_e_reference_and_folds(
    sweep: Sweep,
    rankings: pd.DataFrame,
) -> tuple[Any, np.ndarray, list[str], list[dict[str, Any]], pd.Series]:
    root = sweep.root()
    members, aggregation = _selected_ensemble_members(root)
    member_rows = _ordered_member_rows(root, rankings, members)
    all_levels: list[str] = []
    for _, r in member_rows.iterrows():
        for lv in _row_levels(r):
            if lv not in all_levels:
                all_levels.append(lv)
    if not all_levels:
        all_levels = ["all"]
    dataset = load_dataset(sweep.data, tuple(all_levels))
    splits = _explainability_outer_splits(sweep, dataset)

    feature_names: list[str] = []
    reference_blocks: list[np.ndarray] = []
    member_materialized: list[dict[str, Any]] = []
    for member_i, (_, r) in enumerate(member_rows.iterrows(), start=1):
        levels = _row_levels(r) or ("all",)
        X_base_member, names_member = materialize_mpdr(dataset, levels)
        transformation_key = str(r["count_transformation"])
        learner_key = str(r["learner"])
        label_prefix = (
            f"{str(r.get('resolution', '+'.join(levels)))}"
            f"|{transformation_key}"
            f"|{learner_key}"
            f"|{str(r.get('config_id', member_i))}"
        )
        ct_ref = _configured_count_transformation_factory(sweep, transformation_key)()
        X_ref_member, _ = ct_ref.apply_pair(X_base_member, X_base_member)
        transformed_names = ct_ref.get_feature_names_out(list(names_member))
        if X_ref_member.shape[1] != len(transformed_names):
            raise ExplainabilityConfigurationError(
                f"Transformation {transformation_key!r} produced feature metadata inconsistent with its transformed matrix."
            )
        feature_names.extend([f"{label_prefix}|{name}" for name in transformed_names])
        reference_blocks.append(X_ref_member)
        member_materialized.append(
            {
                "row": r,
                "levels": levels,
                "X_base": X_base_member,
                "n_features": X_base_member.shape[1],
                "transformation_key": transformation_key,
                "learner_key": learner_key,
            }
        )

    X_reference = (
        np.concatenate(reference_blocks, axis=1)
        if len(reference_blocks) > 1
        else reference_blocks[0]
    )
    folds: list[dict[str, Any]] = []
    total_members = len(member_materialized)
    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
            continue
        X_train_blocks: list[np.ndarray] = []
        X_test_blocks: list[np.ndarray] = []
        fitted_members: list[dict[str, Any]] = []
        start = 0
        for member_i, spec in enumerate(member_materialized, start=1):
            r = spec["row"]
            info(
                "Fitting MPMA-E OOF member "
                f"{member_i}/{total_members} · split={str(split['split_key'])} · "
                f"config={str(r.get('config_id', ''))} · "
                f"{str(r.get('resolution', r.get('levels', 'MPDR')))} · "
                f"{spec['transformation_key']} · {spec['learner_key']}"
            )
            ct = _configured_count_transformation_factory(
                sweep, spec["transformation_key"]
            )()
            X_train_member, X_test_member = ct.apply_pair(
                spec["X_base"][train_idx], spec["X_base"][test_idx]
            )
            clf_member = _configured_learner_factory(sweep, spec["learner_key"])()
            clf_member.fit(X_train_member, dataset.y[train_idx])
            stop = start + X_train_member.shape[1]
            X_train_blocks.append(X_train_member)
            X_test_blocks.append(X_test_member)
            fitted_members.append(
                {
                    "config_id": str(r["config_id"]),
                    "slice": slice(start, stop),
                    "estimator": clf_member,
                    "classes": np.arange(len(dataset.class_labels), dtype=int),
                }
            )
            start = stop
        X_train = np.concatenate(X_train_blocks, axis=1)
        X_test = np.concatenate(X_test_blocks, axis=1)
        ensemble = _FittedMpmaEnsemble(fitted_members, aggregation=aggregation)
        proba = _predict_proba_aligned(
            ensemble, X_test, np.arange(len(dataset.class_labels), dtype=int)
        )
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": X_train,
                "X_test": X_test,
                "y_test": dataset.y[test_idx],
                "estimator": ensemble,
                "proba": proba,
            }
        )
    if not folds:
        raise ExplainabilityConfigurationError(
            "No outer fold could be fitted for MPMA-E out-of-fold explainability."
        )
    row = pd.Series(
        {
            "unit": "MPMA-E",
            "members": ",".join(member_rows["config_id"].astype(str).tolist()),
            "aggregation_strategy": aggregation,
            "levels": ",".join(all_levels),
            "count_transformation": "ensemble",
            "learner": "MPMA-E",
        }
    )
    return dataset, X_reference, feature_names, folds, row


def _aggregate_oof_predictions(
    dataset: Any, folds: Sequence[dict[str, Any]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        proba = np.asarray(fold.get("proba"), dtype=float)
        pred = (
            np.argmax(proba, axis=1)
            if proba.ndim == 2 and proba.shape[0] == len(test_idx)
            else np.full(len(test_idx), -1)
        )
        for local_i, sample_index in enumerate(test_idx):
            rec = {
                "split_key": str(fold.get("split_key", "")),
                "sample_id": str(dataset.sample_ids[int(sample_index)]),
                "sample_index": int(sample_index),
                "y_true": int(dataset.y[int(sample_index)]),
                "y_pred": int(pred[local_i]),
            }
            if proba.ndim == 2 and local_i < proba.shape[0]:
                for j, label in enumerate(dataset.class_labels):
                    if j < proba.shape[1]:
                        rec[f"proba_{label}"] = float(proba[local_i, j])
                if proba.shape[1] > 1:
                    rec["y_proba_pos"] = float(proba[local_i, 1])
            rows.append(rec)
    return pd.DataFrame(rows)


def _write_oof_prediction_summary(
    dataset: Any, folds: Sequence[dict[str, Any]], target_dir: Path
) -> Path:
    pred_df = _aggregate_oof_predictions(dataset, folds)
    path = target_dir / "oof_predictions.parquet"
    write_table(path, pred_df)
    return path


def _select_local_sample_pairs(
    dataset: Any,
    folds: Sequence[dict[str, Any]],
    *,
    sample_ids: Sequence[str],
    representative: bool,
) -> list[tuple[int, str]]:
    sid_to_idx = {str(sid): i for i, sid in enumerate(dataset.sample_ids)}
    selected: list[tuple[int, str]] = []
    for sid in sample_ids:
        if str(sid) not in sid_to_idx:
            raise ExplainabilityConfigurationError(
                f"Requested instance sample_id {sid!r} was not found in the loaded dataset."
            )
        selected.append((int(sid_to_idx[str(sid)]), f"requested:{sid}"))
    if representative:
        records: list[dict[str, Any]] = []
        for fold in folds:
            test_idx = np.asarray(fold["test_idx"], dtype=int)
            proba = np.asarray(fold.get("proba"), dtype=float)
            if proba.ndim != 2 or proba.shape[0] != len(test_idx):
                continue
            for local_i, global_i in enumerate(test_idx):
                cls = int(dataset.y[int(global_i)])
                if 0 <= cls < proba.shape[1]:
                    records.append(
                        {
                            "sample_index": int(global_i),
                            "true_class": cls,
                            "p_own_class": float(proba[int(local_i), cls]),
                        }
                    )
        pred = pd.DataFrame(records)
        if not pred.empty:
            pred = pred.groupby(["sample_index", "true_class"], as_index=False).agg(
                p_own_class=("p_own_class", "mean")
            )
            for cls in sorted(set(int(x) for x in dataset.y.tolist())):
                sub = pred[pred["true_class"].eq(cls)].copy()
                if sub.empty:
                    continue
                centre = float(
                    pd.to_numeric(sub["p_own_class"], errors="coerce").median()
                )
                sub["_dist"] = (
                    pd.to_numeric(sub["p_own_class"], errors="coerce") - centre
                ).abs()
                row = sub.sort_values(["_dist", "sample_index"]).iloc[0]
                selected.append(
                    (int(row["sample_index"]), f"representative_class_{cls}")
                )
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for idx, role in selected:
        if idx not in seen:
            seen.add(idx)
            out.append((idx, role))
    return out


def _local_explanations_signature(
    source_signature: str,
    explainability: Any,
    specs_by_name: dict[str, Any],
    methods: Sequence[str],
    selected_pairs: Sequence[tuple[int, str]],
) -> str:
    local_methods = [
        name
        for name in ("shap", "lime")
        if name in methods and method_has_local(specs_by_name[name])
    ]
    payload = {
        "source_signature": str(source_signature),
        "methods": {
            name: method_to_dict(specs_by_name[name]) for name in local_methods
        },
        "random_state": int(explainability.random_state),
        "mode": _effective_local_explanations_mode(explainability),
        "selected_samples": [[int(i), str(role)] for i, role in selected_pairs],
        "local_top_k": int(explainability.local.stored_features),
        "top_instance_features": int(explainability.local.displayed_features),
    }
    return _signature_hash(payload)


def _compute_local_method_rows(
    method: str,
    spec: Any,
    folds: Sequence[dict[str, Any]],
    dataset: Any,
    feature_names: Sequence[str],
    random_state: int,
    threads_per_worker: int,
) -> list[dict[str, Any]]:
    class_indices = tuple(range(len(dataset.class_labels)))
    rows: list[dict[str, Any]] = []
    for fold_no, fold in enumerate(folds, start=1):
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        if len(test_idx) == 0:
            continue
        estimator = configure_estimator_threads(
            fold["estimator"], int(threads_per_worker)
        )
        force_rows = tuple(range(len(test_idx)))
        if method == "shap":
            values, rows_ex, _ = _shap_values_for_data(
                estimator,
                fold["X_train"],
                fold["X_test"],
                feature_names,
                dataset.class_labels,
                random_state=int(random_state) + fold_no * 997,
                spec=spec,
                force_explain_rows=force_rows,
                show_progress=False,
            )
        elif method == "lime":
            values, rows_ex = _lime_values_for_data(
                estimator,
                fold["X_train"],
                fold["X_test"],
                feature_names,
                dataset.class_labels,
                class_indices,
                random_state=int(random_state) + fold_no * 997,
                spec=spec,
                force_explain_rows=force_rows,
                progress_callback=None,
            )
        else:
            continue
        rows.extend(
            _local_value_rows_for_fold(
                fold,
                values,
                rows_ex,
                class_indices,
                dataset.class_labels,
                dataset.sample_ids,
                dataset.y,
                positive_class=dataset.positive_class,
            )
        )
    return rows


def _compact_local_explanations(
    method: str,
    dataset: Any,
    feature_names: Sequence[str],
    rows: Sequence[dict[str, Any]],
    selected_pairs: Sequence[tuple[int, str]],
    local_top_k: int,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    selected = {int(i): str(role) for i, role in selected_pairs}
    records: list[dict[str, Any]] = []
    k = max(1, int(local_top_k))
    for row in rows:
        sample_index = int(row["sample_index"])
        true_class = int(row["true_class"])
        if int(row["class_index"]) != true_class:
            continue
        values = np.asarray(row["values"], dtype=float).reshape(-1)
        feature_values = np.asarray(
            row.get("feature_values", np.full(len(values), np.nan)), dtype=float
        ).reshape(-1)
        if len(values) != len(feature_names):
            raise ExplainabilityConfigurationError(
                f"Local {method.upper()} attribution width {len(values)} does not match {len(feature_names)} feature names."
            )
        is_selected = sample_index in selected
        if is_selected:
            chosen = np.argsort(-np.abs(values))
        else:
            chosen = np.argsort(-np.abs(values))[: min(k, len(values))]
        predicted_class = int(row.get("predicted_class", -1))
        predicted_label = (
            str(dataset.class_labels[predicted_class])
            if 0 <= predicted_class < len(dataset.class_labels)
            else ""
        )
        for rank, feature_index in enumerate(chosen, start=1):
            j = int(feature_index)
            value = float(values[j])
            records.append(
                {
                    "method": str(method),
                    "sample_id": str(row["sample_id"]),
                    "sample_index": sample_index,
                    "split_key": str(row.get("split_key", "")),
                    "selection_role": selected.get(sample_index, ""),
                    "selected_for_report": bool(is_selected),
                    "true_class": true_class,
                    "true_class_label": str(dataset.class_labels[true_class]),
                    "predicted_class": predicted_class,
                    "predicted_class_label": predicted_label,
                    "class_index": true_class,
                    "class_label": str(dataset.class_labels[true_class]),
                    "prediction": float(row.get("p_class", np.nan)),
                    "feature": str(feature_names[j]),
                    "feature_value": float(feature_values[j])
                    if j < len(feature_values)
                    else np.nan,
                    "attribution": value,
                    "abs_attribution": abs(value),
                    "local_rank": int(rank),
                    "retained_scope": "representative_full" if is_selected else "top_k",
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["method", "sample_index", "split_key", "local_rank"],
        kind="stable",
    )


def _ensure_local_explanation_outputs(
    target_dir: Path,
    source_signature: str,
    sweep: Any,
    specs_by_name: dict[str, Any],
    methods: Sequence[str],
    folds: Sequence[dict[str, Any]],
    dataset: Any,
    feature_names: Sequence[str],
    threads_per_worker: int,
    precomputed_rows: dict[str, Sequence[dict[str, Any]]] | None = None,
) -> dict[str, Path]:
    mode = _effective_local_explanations_mode(sweep.explainability)
    if mode == "none":
        return {}
    include_representative = mode in {"representative", "representative_and_requested"}
    include_requested = mode in {"requested", "representative_and_requested"}
    selected_pairs = _select_local_sample_pairs(
        dataset,
        folds,
        sample_ids=sweep.explainability.local.sample_ids if include_requested else (),
        representative=include_representative,
    )
    local_methods = [
        name
        for name in ("shap", "lime")
        if name in methods and method_has_local(specs_by_name[name])
    ]
    if not local_methods:
        return {}
    signature = _local_explanations_signature(
        source_signature, sweep.explainability, specs_by_name, methods, selected_pairs
    )
    local_path = target_dir / "local_explanations.parquet"
    cache_path = target_dir / "local_explanations_cache.json"
    try:
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    except Exception:
        cache = {}
    if str(cache.get("signature", "")) == signature and table_exists(local_path):
        local_table = read_table(local_path)
        info("Reusing cached sample-wise OOF explanations")
    else:
        frames: list[pd.DataFrame] = []
        supplied = precomputed_rows or {}
        for method in local_methods:
            rows = list(supplied.get(method, ()))
            if not rows:
                info(
                    f"Computing sample-wise OOF {method.upper()} explanations · all held-out samples"
                )
                rows = _compute_local_method_rows(
                    method,
                    specs_by_name[method],
                    folds,
                    dataset,
                    feature_names,
                    sweep.explainability.random_state,
                    threads_per_worker,
                )
            frame = _compact_local_explanations(
                method,
                dataset,
                feature_names,
                rows,
                selected_pairs,
                int(sweep.explainability.local.stored_features),
            )
            if not frame.empty:
                frames.append(frame)
        local_table = (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )
        write_table(local_path, local_table)
        dump_json_standard(
            {
                "signature": signature,
                "mode": mode,
                "selected_samples": [
                    {"sample_id": str(dataset.sample_ids[int(i)]), "role": str(role)}
                    for i, role in selected_pairs
                ],
            },
            cache_path,
        )
    outputs: dict[str, Path] = {"local_explanations": local_path}
    if not local_table.empty and selected_pairs:
        figure_path = plot_local_attributions(
            local_table,
            target_dir / "figures" / "local_explanations",
            task="classification",
            top_n=int(sweep.explainability.local.displayed_features),
        )
        if figure_path is not None:
            outputs["local_explanations_figure"] = figure_path
    return outputs


def _aggregate_oof_interactions(
    tables_by_fold: Sequence[pd.DataFrame], method: str
) -> pd.DataFrame:
    if not tables_by_fold:
        return pd.DataFrame()
    raw = pd.concat(tables_by_fold, ignore_index=True)
    raw["interaction_strength"] = pd.to_numeric(
        raw["interaction_strength"], errors="coerce"
    )
    keys = ["class_index", "class_label", "feature_1", "feature_2"]
    out = raw.groupby(keys, as_index=False).agg(
        interaction_strength=("interaction_strength", "mean"),
        interaction_strength_sd=("interaction_strength", "std"),
        n_outer_folds=("fold_key", "nunique"),
        n_evaluations=("interaction_strength", "count"),
        n_ale_cells=("n_ale_cells", "mean"),
        correction_applied=("correction_applied", "max"),
        correction_error=(
            "correction_error",
            lambda x: "; ".join([str(v) for v in x if str(v) and str(v) != "nan"])[
                :240
            ],
        ),
    )
    out["method"] = method
    return out.sort_values(
        ["class_index", "interaction_strength"],
        ascending=[True, False],
        na_position="last",
    )


def _shap_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    sample_ids: Sequence[Any],
    y: np.ndarray,
    positive_class: int | None,
    collect_local: bool,
    random_state: int,
    spec: SHAP,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame, list[dict[str, Any]], int, str]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    force_rows = list(range(len(test_idx))) if collect_local else []
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    vals, rows_ex, backend = _shap_values_for_data(
        estimator,
        fold["X_train"],
        fold["X_test"],
        feature_names,
        class_labels,
        random_state=int(random_state),
        spec=spec,
        force_explain_rows=force_rows,
        show_progress=False,
        progress_callback=callback,
    )
    fold_frame = _value_frame_for_fold(
        "shap",
        vals,
        feature_names,
        class_indices,
        class_labels,
        "mean_abs_probability_shap_within_outer_fold",
        positive_class=positive_class,
    )
    fold_frame["fold_key"] = str(fold.get("split_key", ""))
    fold_frame["fold_no"] = int(fold_no)
    fold_frame["shap_backend"] = str(backend)
    rows = (
        _local_value_rows_for_fold(
            fold,
            vals,
            rows_ex,
            tuple(range(len(class_labels))),
            class_labels,
            sample_ids,
            y,
            positive_class=positive_class,
        )
        if collect_local
        else []
    )
    return int(fold_no), fold_frame, rows, int(len(rows_ex)), str(backend)


def _local_value_rows_for_fold(
    fold: dict[str, Any],
    values: np.ndarray,
    rows_ex: Sequence[int],
    class_indices: Sequence[int],
    class_labels: Sequence[str],
    sample_ids: Sequence[Any],
    y: np.ndarray,
    *,
    positive_class: int | None = None,
) -> list[dict[str, Any]]:
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    X_test = np.asarray(fold["X_test"], dtype=float)
    proba = np.asarray(fold.get("proba"), dtype=float)
    pred = (
        np.argmax(proba, axis=1)
        if proba.ndim == 2 and proba.shape[0] == len(test_idx)
        else np.full(len(test_idx), -1)
    )
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    rows: list[dict[str, Any]] = []
    for local_value_row, local_test_row in enumerate(rows_ex):
        global_i = int(test_idx[int(local_test_row)])
        for class_pos, class_index in enumerate(class_indices):
            if arr.shape[2] == len(class_labels):
                local_values = arr[local_value_row, :, int(class_index)]
            elif arr.shape[2] == len(class_indices):
                local_values = arr[local_value_row, :, class_pos]
            elif arr.shape[2] == 1 and len(class_labels) == 2:
                positive = 1 if positive_class is None else int(positive_class)
                local_values = arr[local_value_row, :, 0]
                if int(class_index) != positive:
                    local_values = -local_values
            elif arr.shape[2] == 1 and len(class_indices) == 1:
                local_values = arr[local_value_row, :, 0]
            else:
                raise ExplainabilityConfigurationError(
                    f"Local attribution output shape {arr.shape} does not match configured classes."
                )
            p_class = (
                float(proba[int(local_test_row), int(class_index)])
                if proba.ndim == 2 and int(class_index) < proba.shape[1]
                else float("nan")
            )
            rows.append(
                {
                    "split_key": str(fold.get("split_key", "")),
                    "sample_id": str(sample_ids[global_i]),
                    "sample_index": global_i,
                    "true_class": int(y[global_i]),
                    "predicted_class": int(pred[int(local_test_row)])
                    if int(local_test_row) < len(pred)
                    else -1,
                    "class_index": int(class_index),
                    "class_label": str(class_labels[int(class_index)]),
                    "p_class": p_class,
                    "feature_values": X_test[int(local_test_row)].copy(),
                    "values": np.asarray(local_values, dtype=float).copy(),
                }
            )
    return rows


def _lime_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    sample_ids: Sequence[Any],
    y: np.ndarray,
    positive_class: int | None,
    collect_local: bool,
    random_state: int,
    spec: LIME,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame, list[dict[str, Any]]]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    force_rows = list(range(len(test_idx))) if collect_local else []
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    coeffs, rows_ex = _lime_values_for_data(
        estimator,
        fold["X_train"],
        fold["X_test"],
        feature_names,
        class_labels,
        tuple(range(len(class_labels))) if collect_local else class_indices,
        random_state=int(random_state),
        spec=spec,
        force_explain_rows=force_rows,
        progress_callback=callback,
    )
    frame = _value_frame_for_fold(
        "lime",
        coeffs,
        feature_names,
        class_indices,
        class_labels,
        "mean_abs_lime_coefficient_within_outer_fold",
    )
    frame["fold_key"] = str(fold.get("split_key", ""))
    frame["fold_no"] = int(fold_no)
    rows = (
        _local_value_rows_for_fold(
            fold,
            coeffs,
            rows_ex,
            tuple(range(len(class_labels))) if collect_local else class_indices,
            class_labels,
            sample_ids,
            y,
            positive_class=positive_class,
        )
        if collect_local
        else []
    )
    return int(fold_no), frame, rows


def _permutation_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    random_state: int,
    spec: Permutation,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    frame = _permutation_feature_importance(
        estimator,
        fold["X_test"],
        fold["y_test"],
        feature_names,
        class_labels,
        class_indices,
        spec=spec,
        random_state=int(random_state),
        progress_callback=callback,
    )
    frame["fold_key"] = str(fold.get("split_key", ""))
    frame["fold_no"] = int(fold_no)
    return int(fold_no), frame


def _ale_fold_parallel_task(
    fold_no: int,
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    spec: ALE,
    threads_per_worker: int,
    progress_queue: Any | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    estimator = configure_estimator_threads(fold["estimator"], threads_per_worker)
    callback = (
        _queued_progress_callback(progress_queue, int(fold_no))
        if progress_queue is not None
        else None
    )
    frame = _ale_feature_importance(
        estimator,
        fold["X_test"],
        feature_names,
        class_labels,
        class_indices,
        spec=spec,
        top_features=None,
        progress_callback=callback,
    )
    skipped = frame.attrs.get("skipped_features")
    skipped_out = None
    if isinstance(skipped, pd.DataFrame) and not skipped.empty:
        skipped_out = skipped.copy()
        skipped_out["fold_key"] = str(fold.get("split_key", ""))
        skipped_out["fold_no"] = int(fold_no)
    if frame.empty:
        return int(fold_no), frame, None, skipped_out
    curves = frame.attrs.get("curves")
    frame = frame.copy()
    frame["fold_key"] = str(fold.get("split_key", ""))
    frame["fold_no"] = int(fold_no)
    curves_out = None
    if isinstance(curves, pd.DataFrame) and not curves.empty:
        curves_out = curves.copy()
        curves_out["fold_key"] = str(fold.get("split_key", ""))
        curves_out["fold_no"] = int(fold_no)
    return int(fold_no), frame, curves_out, skipped_out


def _explain_one(
    sweep: Sweep,
    target_override: str | None = None,
    *,
    output_slug_override: str | None = None,
    display_label_override: str | None = None,
    allow_member_cache: bool = True,
) -> dict[str, Path]:
    method_specs = _normalise_explainability_method_specs(sweep.explainability.methods)
    methods = tuple(method_name(x) for x in method_specs)
    global_methods = tuple(method_name(x) for x in method_specs if method_has_global(x))
    local_methods = tuple(method_name(x) for x in method_specs if method_has_local(x))
    specs_by_name = {method_name(x): x for x in method_specs}
    _preflight_explainability_dependencies(methods)
    root = sweep.root()
    stage("Explainability", str(root))
    summary_table(
        "Explainability suite",
        {
            "target": target_override
            or _configured_explainability_targets(sweep.explainability),
            "methods": methods,
            "profile": str(getattr(sweep.explainability, "profile", "standard")),
            "top features": sweep.explainability.top_k,
            "local explanations": _effective_local_explanations_mode(
                sweep.explainability
            ),
            "interaction pairs": int(specs_by_name["interactions"].top_k)
            if "interactions" in methods
            else "not requested",
            "fallbacks": "off",
        },
    )
    rankings_path = root / "tables" / "mpma_rankings.parquet"
    if not table_exists(rankings_path):
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = read_table(rankings_path)
    if rankings.empty:
        raise RuntimeError("No ranked MPMA is available for explainability.")
    target = target_override or _single_configured_explainability_target(
        sweep.explainability
    )
    ensemble_explain = False
    if target in {"best", "best_individual", "best_mpma", "mpma_b", "MPMA-B"}:
        row = rankings.iloc[0]
        target_label = "MPMA-B"
        target_slug = "mpma_b"
    elif target in {"ensemble", "mpma_e", "MPMA-E"}:
        ensemble_explain = True
        row = pd.Series({"unit": "MPMA-E"})
        target_label = "MPMA-E"
        target_slug = "mpma_e"
    elif target in {"baseline_rf", "Baseline RF", "baseline-rf"}:
        row = _resolve_baseline_rf_row(rankings)
        if row is None:
            raise ExplainabilityConfigurationError(
                "Baseline RF explainability was requested, but no completed MPMA matches "
                "RF_1000_msl5 + arcsin_sqrt + deepest available single rank."
            )
        target_label = "Baseline RF"
        target_slug = "baseline_rf"
    else:
        matches = rankings[rankings["config_id"].astype(str).eq(str(target))]
        if matches.empty:
            raise ValueError(f"No MPMA with config_id={target!r} in rankings.")
        row = matches.iloc[0]
        target_label = "MPMA"
        target_slug = str(target)

    if output_slug_override:
        target_slug = output_slug_override.strip("/")
    if display_label_override:
        target_label = display_label_override

    if ensemble_explain and allow_member_cache:
        members_for_cache, _ = _selected_ensemble_members(root)
        _ensure_mpma_member_explanations(sweep, rankings, members_for_cache)

    config_id_for_cache = None
    if (
        not ensemble_explain
        and "config_id" in row.index
        and not pd.isna(row.get("config_id"))
    ):
        config_id_for_cache = str(row.get("config_id"))
    target_dir = root / "explainability" / target_slug
    cache_dir = (
        root / "explainability" / "cache" / _safe_cache_name(config_id_for_cache)
        if config_id_for_cache
        else None
    )
    if (
        allow_member_cache
        and target_dir.exists()
        and _explainability_cache_complete(target_dir, sweep.explainability)
    ):
        info(f"Reusing existing explainability for {target_label}")
        return _existing_explainability_outputs(target_dir)
    if (
        not ensemble_explain
        and allow_member_cache
        and cache_dir is not None
        and cache_dir.exists()
        and _explainability_cache_complete(cache_dir, sweep.explainability)
    ):
        if target_dir != cache_dir:
            _copy_explainability_cache(cache_dir, target_dir)
        info(
            f"Reusing cached explainability for {target_label} · config={config_id_for_cache}"
        )
        return _existing_explainability_outputs(target_dir)

    coordinate_metadata: list[Any] = []
    if ensemble_explain:
        info("Preparing selected MPMA-E outer-fold units for OOF explanation")
        dataset, X_base, feature_names, oof_folds, row = _mpma_e_reference_and_folds(
            sweep, rankings
        )
    else:
        info("Preparing selected MPMA outer-fold units for OOF explanation")
        oof_bundle = _fit_oof_single_for_explainability(sweep, row)
        dataset = oof_bundle["dataset"]
        X_base = oof_bundle["X_base"]
        feature_names = oof_bundle["feature_names"]
        coordinate_metadata = list(oof_bundle.get("coordinate_metadata", []))
        oof_folds = oof_bundle["folds"]

    class_indices = _resolve_explainability_classes(
        dataset, sweep.explainability.classes
    )
    execution = _xai_execution_plan(sweep, len(oof_folds))
    memory_gib = (
        "unknown"
        if execution.memory_bytes is None
        else f"{execution.memory_bytes / (1024**3):.1f} GiB"
    )
    summary_table(
        "Explainability workload",
        {
            "outer folds": len(oof_folds),
            "features": len(feature_names),
            "classes explained": len(class_indices),
            "class labels": [str(dataset.class_labels[int(i)]) for i in class_indices],
            "CPU logical/physical": f"{execution.logical_cpus}/{execution.physical_cpus}",
            "available memory": memory_gib,
            "workers": execution.workers,
            "threads per worker": execution.threads_per_worker,
            "parallel backend": execution.backend,
        },
    )
    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    coordinate_metadata_path: Path | None = None
    if coordinate_metadata:
        coordinate_frame = pd.DataFrame(
            [
                {
                    "coordinate": str(item.name),
                    "coordinate_type": str(item.coordinate_type),
                    "anchor_feature": ""
                    if item.anchor_feature is None
                    else str(item.anchor_feature),
                    "exact_feature_identity": bool(item.exact_feature_identity),
                    "components": json.dumps(
                        list(item.components), separators=(",", ":")
                    ),
                    "coefficients": json.dumps(
                        [float(x) for x in item.coefficients], separators=(",", ":")
                    ),
                }
                for item in coordinate_metadata
            ]
        )
        coordinate_metadata_path = target_dir / "coordinate_metadata.parquet"
        write_table(coordinate_metadata_path, coordinate_frame)

    source_signature = _explainability_source_signature(
        target_slug, row, oof_folds, feature_names, dataset.class_labels
    )
    previous_meta: dict[str, Any] = {}
    meta_path = target_dir / "explained_unit.json"
    if meta_path.exists():
        try:
            previous_meta = json.loads(meta_path.read_text())
        except Exception:
            previous_meta = {}
    previous_method_cache = dict(previous_meta.get("method_cache", {}))
    current_method_signatures = {
        name: _method_cache_signature(
            sweep.explainability, specs_by_name[name], source_signature, class_indices
        )
        for name in global_methods
    }
    current_method_entries = {
        name: _method_cache_entry(
            current_method_signatures[name],
            source_signature,
            specs_by_name[name],
            class_indices,
            sweep.explainability.top_k,
        )
        for name in current_method_signatures
    }
    reusable_methods: set[str] = set()
    for name, signature in current_method_signatures.items():
        if not _method_cache_files_complete(target_dir, name):
            info(
                f"Cache miss {name.upper()} · required method artifacts are incomplete"
            )
            continue
        entry, provenance_source = _load_method_cache_entry(
            target_dir, name, previous_method_cache
        )
        compatible, reason = _method_cache_entry_status(
            entry, signature, source_signature, specs_by_name[name], class_indices
        )
        legacy = not compatible and _legacy_method_cache_valid(
            previous_meta, row, sweep.explainability, specs_by_name[name], class_indices
        )
        if compatible or legacy:
            reusable_methods.add(name)
            _persist_method_cache_entry(
                meta_path, previous_meta, name, current_method_entries[name]
            )
            previous_method_cache[name] = current_method_entries[name]
            suffix = (
                "migrated from legacy metadata"
                if legacy and not compatible
                else provenance_source
            )
            info(
                f"Cache hit {name.upper()} · reusing completed global explainability · {suffix}"
            )
        else:
            info(f"Cache miss {name.upper()} · {reason}")

    method_frames: list[pd.DataFrame] = []
    fold_importance_frames: list[pd.DataFrame] = []
    method_outputs: dict[str, Path] = {}
    interaction_outputs: dict[str, Path] = {}
    resolved_shap_backends: set[str] = set()
    for name in global_methods:
        if name not in reusable_methods:
            continue
        if name == "interactions":
            interaction_outputs.update(_existing_method_outputs(target_dir, name))
            continue
        _, fold_long = _cached_method_frame(target_dir, name)
        fold_frames = (
            [frame.copy() for _, frame in fold_long.groupby("fold_no", sort=True)]
            if "fold_no" in fold_long.columns
            else [fold_long.copy()]
        )
        scoring_values = (
            fold_long.get("scoring", pd.Series(dtype=object)).dropna().astype(str)
        )
        scoring = scoring_values.iloc[0] if not scoring_values.empty else name
        frame = _aggregate_fold_feature_importance(
            name,
            fold_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            scoring,
            sweep.explainability.top_k,
        )
        method_frames.append(frame)
        fold_importance_frames.append(fold_long)
        if name == "shap" and "shap_backend" in fold_long.columns:
            resolved_shap_backends.update(
                str(x) for x in fold_long["shap_backend"].dropna().astype(str).unique()
            )
        method_outputs.update(
            _write_method_outputs(
                frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        method_outputs.update(_existing_method_outputs(target_dir, name))
    oof_pred_path = _write_oof_prediction_summary(dataset, oof_folds, target_dir)
    method_outputs["oof_predictions"] = oof_pred_path
    local_mode = _effective_local_explanations_mode(sweep.explainability)
    local_enabled = local_mode != "none" and bool(local_methods)

    shap_oof_rows: list[dict[str, Any]] = []
    lime_oof_rows: list[dict[str, Any]] = []
    shap_local_enabled = local_enabled and "shap" in local_methods
    if "shap" in global_methods and "shap" not in reusable_methods:
        spec = specs_by_name["shap"]
        info("Running global class-specific OOF SHAP feature attribution")
        shap_fold_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            with progress() as prog:
                for fold_no, fold in enumerate(oof_folds, start=1):
                    test_idx = np.asarray(fold["test_idx"], dtype=int)
                    expected_rows = max(
                        1,
                        len(test_idx)
                        if shap_local_enabled
                        else min(int(spec.max_explain), len(test_idx)),
                    )
                    prefix = (
                        f"SHAP fold {fold_no}/{len(oof_folds)} · "
                        f"{expected_rows} samples · {len(feature_names)} features · {len(class_indices)} classes"
                    )
                    task = prog.add_task(f"{prefix} · preparing", total=expected_rows)
                    vals, rows_ex, backend = _shap_values_for_data(
                        configure_estimator_threads(
                            fold["estimator"], int(execution.threads_per_worker)
                        ),
                        fold["X_train"],
                        fold["X_test"],
                        feature_names,
                        dataset.class_labels,
                        random_state=sweep.explainability.random_state + fold_no * 997,
                        spec=spec,
                        force_explain_rows=list(range(len(test_idx)))
                        if shap_local_enabled
                        else [],
                        show_progress=False,
                        progress_callback=_progress_callback(prog, task, prefix),
                    )
                    fold_frame = _value_frame_for_fold(
                        "shap",
                        vals,
                        feature_names,
                        class_indices,
                        dataset.class_labels,
                        "mean_abs_probability_shap_within_outer_fold",
                        positive_class=dataset.positive_class,
                    )
                    fold_frame["fold_key"] = str(fold.get("split_key", ""))
                    fold_frame["fold_no"] = int(fold_no)
                    fold_frame["shap_backend"] = str(backend)
                    shap_fold_frames.append(fold_frame)
                    if shap_local_enabled:
                        shap_oof_rows.extend(
                            _local_value_rows_for_fold(
                                fold,
                                vals,
                                rows_ex,
                                tuple(range(len(dataset.class_labels))),
                                dataset.class_labels,
                                dataset.sample_ids,
                                dataset.y,
                                positive_class=dataset.positive_class,
                            )
                        )
        else:
            fold_totals = {
                fold_no: max(
                    1,
                    len(np.asarray(fold["test_idx"], dtype=int))
                    if shap_local_enabled
                    else min(
                        int(spec.max_explain),
                        len(np.asarray(fold["test_idx"], dtype=int)),
                    ),
                )
                for fold_no, fold in enumerate(oof_folds, start=1)
            }
            info(
                f"SHAP workload · {sum(fold_totals.values()):,} explained samples · "
                f"{len(oof_folds)} folds × up to {int(spec.max_explain)} samples"
            )
            with Manager() as manager:
                progress_queue = manager.Queue()
                shap_tasks = [
                    (
                        _shap_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            dataset.sample_ids,
                            dataset.y,
                            dataset.positive_class,
                            shap_local_enabled,
                            sweep.explainability.random_state + fold_no * 997,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                parallel_results = _parallel_progress_results(
                    shap_tasks,
                    execution,
                    progress_queue,
                    label="SHAP",
                    unit_label="samples",
                    fold_totals=fold_totals,
                )
            fold_results: dict[
                int, tuple[pd.DataFrame, list[dict[str, Any]], int, str]
            ] = {}
            for fold_no, fold_frame, rows, explained_count, backend in parallel_results:
                fold_results[int(fold_no)] = (
                    fold_frame,
                    rows,
                    int(explained_count),
                    str(backend),
                )
            for fold_no in sorted(fold_results):
                fold_frame, rows, _, _ = fold_results[fold_no]
                shap_fold_frames.append(fold_frame)
                shap_oof_rows.extend(rows)
        backend_counts: dict[str, int] = {}
        for frame in shap_fold_frames:
            if "shap_backend" not in frame.columns or frame.empty:
                continue
            backend = str(frame["shap_backend"].iloc[0])
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        if backend_counts:
            resolved_shap_backends.update(backend_counts)
            info(
                "SHAP backend routing · "
                + ", ".join(f"{k}={v}" for k, v in sorted(backend_counts.items()))
            )
        info("Aggregating SHAP global importance and cross-fold stability")
        shap_frame = _aggregate_fold_feature_importance(
            "shap",
            shap_fold_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            "outer_fold_mean_abs_probability_shap",
            sweep.explainability.top_k,
        )
        if backend_counts:
            shap_frame["shap_backend"] = ",".join(sorted(backend_counts))
        method_frames.append(shap_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "shap", shap_fold_frames, target_dir
        )
        method_outputs["shap_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                shap_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path, previous_meta, "shap", current_method_entries["shap"]
        )
        previous_method_cache["shap"] = current_method_entries["shap"]
        success("Class-specific OOF SHAP completed")

    lime_local_enabled = local_enabled and "lime" in local_methods
    if "lime" in global_methods and "lime" not in reusable_methods:
        spec = specs_by_name["lime"]
        info("Running global class-specific OOF LIME feature attribution")
        lime_fold_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            prog = progress()
            prog.start()
            for fold_no, fold in enumerate(oof_folds, start=1):
                prefix = f"LIME fold {fold_no}/{len(oof_folds)} · {len(feature_names)} features · {len(class_indices)} classes"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                force_rows = (
                    list(range(len(np.asarray(fold["test_idx"], dtype=int))))
                    if lime_local_enabled
                    else []
                )
                local_class_indices = (
                    tuple(range(len(dataset.class_labels)))
                    if lime_local_enabled
                    else class_indices
                )
                coeffs, rows_ex = _lime_values_for_data(
                    configure_estimator_threads(
                        fold["estimator"], int(execution.threads_per_worker)
                    ),
                    fold["X_train"],
                    fold["X_test"],
                    feature_names,
                    dataset.class_labels,
                    local_class_indices,
                    random_state=sweep.explainability.random_state + fold_no * 997,
                    spec=spec,
                    force_explain_rows=force_rows,
                    progress_callback=_progress_callback(prog, task, prefix),
                )
                fold_frame = _value_frame_for_fold(
                    "lime",
                    coeffs,
                    feature_names,
                    class_indices,
                    dataset.class_labels,
                    "mean_abs_lime_coefficient_within_outer_fold",
                )
                fold_frame["fold_key"] = str(fold.get("split_key", ""))
                fold_frame["fold_no"] = int(fold_no)
                lime_fold_frames.append(fold_frame)
                if lime_local_enabled:
                    lime_oof_rows.extend(
                        _local_value_rows_for_fold(
                            fold,
                            coeffs,
                            rows_ex,
                            local_class_indices,
                            dataset.class_labels,
                            dataset.sample_ids,
                            dataset.y,
                            positive_class=dataset.positive_class,
                        )
                    )
            prog.stop()
        else:
            fold_totals = {
                fold_no: max(
                    1,
                    len(np.asarray(fold["test_idx"], dtype=int))
                    if lime_local_enabled
                    else min(
                        int(spec.max_explain),
                        len(np.asarray(fold["test_idx"], dtype=int)),
                    ),
                )
                for fold_no, fold in enumerate(oof_folds, start=1)
            }
            info(
                f"LIME workload · {sum(fold_totals.values()):,} explained samples · "
                f"{len(oof_folds)} folds × up to {int(spec.max_explain)} samples"
            )
            with Manager() as manager:
                progress_queue = manager.Queue()
                tasks = [
                    (
                        _lime_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            dataset.sample_ids,
                            dataset.y,
                            dataset.positive_class,
                            lime_local_enabled,
                            sweep.explainability.random_state + fold_no * 997,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                parallel_results = _parallel_progress_results(
                    tasks,
                    execution,
                    progress_queue,
                    label="LIME",
                    unit_label="samples",
                    fold_totals=fold_totals,
                )
            results: dict[int, tuple[pd.DataFrame, list[dict[str, Any]]]] = {}
            for fold_no, frame, rows in parallel_results:
                results[int(fold_no)] = (frame, rows)
            lime_fold_frames = [results[i][0] for i in sorted(results)]
            for i in sorted(results):
                lime_oof_rows.extend(results[i][1])
        info("Aggregating LIME global importance and cross-fold stability")
        lime_frame = _aggregate_fold_feature_importance(
            "lime",
            lime_fold_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            "outer_fold_mean_abs_lime_coefficient",
            sweep.explainability.top_k,
        )
        method_frames.append(lime_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "lime", lime_fold_frames, target_dir
        )
        method_outputs["lime_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                lime_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path, previous_meta, "lime", current_method_entries["lime"]
        )
        previous_method_cache["lime"] = current_method_entries["lime"]
        success("Class-specific OOF LIME completed")

    if "permutation" in methods and "permutation" not in reusable_methods:
        spec = specs_by_name["permutation"]
        info("Running global class-specific OOF permutation importance")
        perm_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            prog = progress()
            prog.start()
            for fold_no, fold in enumerate(oof_folds, start=1):
                prefix = f"Permutation fold {fold_no}/{len(oof_folds)} · {len(feature_names)} features · {len(class_indices)} classes"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                frame = _permutation_feature_importance(
                    configure_estimator_threads(
                        fold["estimator"], int(execution.threads_per_worker)
                    ),
                    fold["X_test"],
                    fold["y_test"],
                    feature_names,
                    dataset.class_labels,
                    class_indices,
                    spec=spec,
                    random_state=sweep.explainability.random_state + fold_no * 997,
                    progress_callback=_progress_callback(prog, task, prefix),
                )
                frame["fold_key"] = str(fold.get("split_key", ""))
                frame["fold_no"] = int(fold_no)
                perm_frames.append(frame)
            prog.stop()
        else:
            total_per_fold = max(1, len(feature_names) * int(spec.n_repeats))
            fold_totals = {
                fold_no: total_per_fold for fold_no, _ in enumerate(oof_folds, start=1)
            }
            info(
                f"Permutation workload · {sum(fold_totals.values()):,} feature-repeat jobs · "
                f"{len(feature_names):,} features × {int(spec.n_repeats)} repeats × {len(oof_folds)} folds"
            )
            with Manager() as manager:
                progress_queue = manager.Queue()
                tasks = [
                    (
                        _permutation_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            sweep.explainability.random_state + fold_no * 997,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                parallel_results = _parallel_progress_results(
                    tasks,
                    execution,
                    progress_queue,
                    label="Permutation",
                    unit_label="feature-repeat jobs",
                    fold_totals=fold_totals,
                )
            results: dict[int, pd.DataFrame] = {}
            for fold_no, frame in parallel_results:
                results[int(fold_no)] = frame
            perm_frames = [results[i] for i in sorted(results)]
        info("Aggregating permutation global importance and cross-fold stability")
        permutation_frame = _aggregate_fold_feature_importance(
            "permutation",
            perm_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            f"outer_fold_increase_in_one_vs_rest_{spec.scoring}",
            sweep.explainability.top_k,
        )
        method_frames.append(permutation_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "permutation", perm_frames, target_dir
        )
        method_outputs["permutation_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                permutation_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path,
            previous_meta,
            "permutation",
            current_method_entries["permutation"],
        )
        previous_method_cache["permutation"] = current_method_entries["permutation"]
        success("Class-specific OOF permutation importance completed")

    if "ale" in methods and "ale" not in reusable_methods:
        spec = specs_by_name["ale"]
        info("Running global class-specific OOF ALE feature effects")
        ale_frames: list[pd.DataFrame] = []
        curve_frames: list[pd.DataFrame] = []
        skipped_frames: list[pd.DataFrame] = []
        if execution.workers == 1:
            prog = progress()
            prog.start()
            for fold_no, fold in enumerate(oof_folds, start=1):
                prefix = f"ALE fold {fold_no}/{len(oof_folds)} · {len(feature_names)} features · {len(class_indices)} classes"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                frame = _ale_feature_importance(
                    configure_estimator_threads(
                        fold["estimator"], int(execution.threads_per_worker)
                    ),
                    fold["X_test"],
                    feature_names,
                    dataset.class_labels,
                    class_indices,
                    spec=spec,
                    top_features=None,
                    progress_callback=_progress_callback(prog, task, prefix),
                )
                skipped = frame.attrs.get("skipped_features")
                if isinstance(skipped, pd.DataFrame) and not skipped.empty:
                    skipped = skipped.copy()
                    skipped["fold_key"] = str(fold.get("split_key", ""))
                    skipped["fold_no"] = int(fold_no)
                    skipped_frames.append(skipped)
                if frame.empty:
                    continue
                curves = frame.attrs.get("curves")
                frame = frame.copy()
                frame["fold_key"] = str(fold.get("split_key", ""))
                frame["fold_no"] = int(fold_no)
                ale_frames.append(frame)
                if isinstance(curves, pd.DataFrame) and not curves.empty:
                    curves = curves.copy()
                    curves["fold_key"] = str(fold.get("split_key", ""))
                    curves["fold_no"] = int(fold_no)
                    curve_frames.append(curves)
            prog.stop()
        else:
            total_per_fold = max(1, len(feature_names) * len(class_indices))
            total_work = max(1, len(oof_folds) * total_per_fold)
            info(
                f"ALE workload · {total_work:,} feature-class jobs · "
                f"{len(feature_names):,} features × {len(class_indices)} classes × {len(oof_folds)} folds"
            )
            results: dict[
                int, tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]
            ] = {}
            with Manager() as manager:
                progress_queue = manager.Queue()
                tasks = [
                    (
                        _ale_fold_parallel_task,
                        (
                            fold_no,
                            fold,
                            feature_names,
                            dataset.class_labels,
                            class_indices,
                            spec,
                            int(execution.threads_per_worker),
                            progress_queue,
                        ),
                        {},
                    )
                    for fold_no, fold in enumerate(oof_folds, start=1)
                ]
                fold_progress = {fold_no: 0 for fold_no in range(1, len(oof_folds) + 1)}
                stop_monitor = Event()
                with progress() as prog:
                    task = prog.add_task(
                        f"ALE · 0/{total_work:,} feature-class jobs · {execution.workers} workers",
                        total=total_work,
                    )

                    def monitor() -> None:
                        while not stop_monitor.is_set():
                            try:
                                fold_no, completed, total, detail = progress_queue.get(
                                    timeout=0.25
                                )
                            except Empty:
                                continue
                            fold_no = int(fold_no)
                            fold_progress[fold_no] = max(
                                fold_progress.get(fold_no, 0),
                                min(max(0, int(completed)), max(1, int(total))),
                            )
                            overall = min(total_work, sum(fold_progress.values()))
                            prog.update(
                                task,
                                completed=overall,
                                description=(
                                    f"ALE · {overall:,}/{total_work:,} feature-class jobs · "
                                    f"fold {fold_no}/{len(oof_folds)} · {detail}"
                                ),
                            )

                    monitor_thread = Thread(target=monitor, daemon=True)
                    monitor_thread.start()
                    try:
                        for fold_no, frame, curves, skipped in _xai_task_iterator(
                            tasks, execution
                        ):
                            results[int(fold_no)] = (frame, curves, skipped)
                            fold_progress[int(fold_no)] = total_per_fold
                            overall = min(total_work, sum(fold_progress.values()))
                            prog.update(
                                task,
                                completed=overall,
                                description=(
                                    f"ALE · {overall:,}/{total_work:,} feature-class jobs · "
                                    f"completed fold {fold_no}/{len(oof_folds)}"
                                ),
                            )
                    finally:
                        stop_monitor.set()
                        monitor_thread.join(timeout=2.0)
                    prog.update(
                        task,
                        completed=total_work,
                        description=f"ALE · completed {total_work:,}/{total_work:,} feature-class jobs",
                    )
            for fold_no in sorted(results):
                frame, curves, skipped = results[fold_no]
                if skipped is not None and not skipped.empty:
                    skipped_frames.append(skipped)
                if not frame.empty:
                    ale_frames.append(frame)
                if curves is not None and not curves.empty:
                    curve_frames.append(curves)
        info("Aggregating ALE global effects and cross-fold stability")
        if skipped_frames:
            write_table(
                target_dir / "ale_skipped_features.parquet",
                pd.concat(skipped_frames, ignore_index=True),
            )
        if not ale_frames:
            raise ExplainabilityConfigurationError(
                "ALE was requested, but no candidate feature was estimable in any outer-test fold."
            )
        ale_frame = _aggregate_fold_feature_importance(
            "ale",
            ale_frames,
            feature_names,
            class_indices,
            dataset.class_labels,
            "outer_fold_rms_centered_class_probability_ale",
            sweep.explainability.top_k,
        )
        if curve_frames:
            curves_all = pd.concat(curve_frames, ignore_index=True)
            write_table(target_dir / "ale_curves.parquet", curves_all)
            for class_index in class_indices:
                label = str(dataset.class_labels[int(class_index)])
                slug = _class_slug(label)
                class_imp = ale_frame[ale_frame["class_index"].eq(int(class_index))]
                top_features = (
                    class_imp.dropna(subset=["importance_mean"])
                    .sort_values("importance_mean", ascending=False)
                    .head(sweep.explainability.top_k)["feature"]
                    .astype(str)
                    .tolist()
                )
                class_curves = curves_all[
                    curves_all["class_index"].eq(int(class_index))
                ]
                curve_stem = figures_dir / f"ale_curves__{slug}"
                _plot_ale_curves(
                    class_curves,
                    top_features,
                    curve_stem,
                    max_panels=min(12, sweep.explainability.top_k),
                )
                method_outputs[f"ale_curves_{slug}"] = curve_stem.with_suffix(".svg")
        method_frames.append(ale_frame)
        fold_path, fold_long = _write_fold_feature_importance(
            "ale", ale_frames, target_dir
        )
        method_outputs["ale_by_outer_fold"] = fold_path
        fold_importance_frames.append(fold_long)
        method_outputs.update(
            _write_method_outputs(
                ale_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            )
        )
        _persist_method_cache_entry(
            meta_path, previous_meta, "ale", current_method_entries["ale"]
        )
        previous_method_cache["ale"] = current_method_entries["ale"]
        success("Class-specific OOF ALE completed")

    if not method_frames and not local_enabled:
        raise ExplainabilityConfigurationError(
            "No global or local explainability method was executed."
        )

    imp = (
        _combine_feature_importance(method_frames) if method_frames else pd.DataFrame()
    )
    write_table(target_dir / "feature_importance.parquet", imp)
    stability_all = (
        pd.concat(
            [
                frame.assign(method=str(frame["method"].iloc[0]))
                for frame in method_frames
                if not frame.empty
            ],
            ignore_index=True,
        )
        if method_frames
        else pd.DataFrame()
    )
    stability_columns = [
        "method",
        "class_index",
        "class_label",
        "feature",
        "importance_mean",
        "importance_sd",
        "importance_median",
        "importance_q25",
        "importance_q75",
        "mean_rank",
        "median_rank",
        "rank_iqr",
        "top_k_frequency",
        "n_estimable_folds",
        "n_outer_folds_total",
        "fold_coverage",
        "signed_importance_mean",
        "sign_positive_fraction",
        "sign_negative_fraction",
        "sign_consistency",
    ]
    write_table(
        target_dir / "feature_stability.parquet",
        stability_all[[c for c in stability_columns if c in stability_all.columns]],
    )
    method_outputs["feature_stability"] = target_dir / "feature_stability.parquet"
    fold_all = (
        pd.concat(fold_importance_frames, ignore_index=True)
        if fold_importance_frames
        else pd.DataFrame()
    )
    write_table(target_dir / "feature_importance_by_outer_fold.parquet", fold_all)
    method_outputs["feature_importance_by_outer_fold"] = (
        target_dir / "feature_importance_by_outer_fold.parquet"
    )

    if "interactions" in global_methods and "interactions" not in reusable_methods:
        spec = specs_by_name["interactions"]
        info("Running class-specific OOF ALE interaction analysis")
        by_method: dict[str, list[pd.DataFrame]] = {
            m: [] for m in ("current", "corrected", "fixed_pairs", "corrected_fixed")
        }
        all_raw: list[pd.DataFrame] = []
        prog = progress()
        prog.start()
        for class_index in class_indices:
            score_series = imp[imp["class_index"].eq(int(class_index))].set_index(
                "feature"
            )["importance_mean"]
            for fold_no, fold in enumerate(oof_folds, start=1):
                prefix = f"Interactions · {dataset.class_labels[int(class_index)]} · fold {fold_no}/{len(oof_folds)}"
                task = prog.add_task(f"{prefix} · preparing", total=1)
                try:
                    tables = _ale_interactions(
                        fold["estimator"],
                        fold["X_test"],
                        feature_names,
                        dataset.class_labels,
                        int(class_index),
                        spec=spec,
                        scores=score_series,
                        progress_callback=_progress_callback(prog, task, prefix),
                    )
                except ExplainabilityConfigurationError:
                    prog.update(
                        task, completed=1, total=1, description=f"{prefix} · skipped"
                    )
                    continue
                for name, tab in tables.items():
                    tab = tab.copy()
                    tab["fold_key"] = str(fold.get("split_key", ""))
                    if name == "all_methods":
                        all_raw.append(tab)
                    elif name in by_method:
                        by_method[name].append(tab)
        prog.stop()
        info("Aggregating interaction stability and writing outputs")
        interaction_tables = {
            name: _aggregate_oof_interactions(frames, name)
            for name, frames in by_method.items()
            if frames
        }
        if all_raw:
            all_path = target_dir / "feature_interactions_by_target_all_methods.parquet"
            write_table(all_path, pd.concat(all_raw, ignore_index=True))
            interaction_outputs["interactions_all_methods"] = all_path
        for name, tab in interaction_tables.items():
            path = target_dir / f"feature_interactions_{name}.parquet"
            write_table(path, tab)
            interaction_outputs[f"interactions_{name}"] = path
        _persist_method_cache_entry(
            meta_path,
            previous_meta,
            "interactions",
            current_method_entries["interactions"],
        )
        previous_method_cache["interactions"] = current_method_entries["interactions"]
        success("Class-specific OOF interaction analysis completed")

    if "interactions" in methods:
        interaction_outputs.update(
            _ensure_interaction_network_outputs(
                target_dir,
                figures_dir,
                X_base,
                dataset.y,
                feature_names,
                dataset.class_labels,
                class_indices,
                int(specs_by_name["interactions"].top_k),
            )
        )

    method_outputs.update(
        _ensure_local_explanation_outputs(
            target_dir,
            source_signature,
            sweep,
            specs_by_name,
            methods,
            oof_folds,
            dataset,
            feature_names,
            int(execution.threads_per_worker),
            precomputed_rows={"shap": shap_oof_rows, "lime": lime_oof_rows},
        )
    )

    top_features = (
        _method_support_table(method_frames, imp, top_k=sweep.explainability.top_k)
        if method_frames and not imp.empty
        else pd.DataFrame()
    )
    write_table(target_dir / "top_features.parquet", top_features)
    dist_frames: list[pd.DataFrame] = []
    class_figure_paths: dict[str, Path] = {}
    grouped_top_features = (
        top_features.groupby("class_index", sort=True)
        if not top_features.empty and "class_index" in top_features.columns
        else ()
    )
    for class_index, class_top in grouped_top_features:
        c = int(class_index)
        label = str(dataset.class_labels[c])
        slug = _class_slug(label)
        dist = _feature_distribution_stats(
            class_top["feature"].astype(str).tolist(),
            feature_names,
            X_base,
            dataset.y,
            dataset.class_labels,
        )
        dist.insert(0, "class_label", label)
        dist.insert(0, "class_index", c)
        dist_frames.append(dist)
        support_stem = figures_dir / f"feature_support__{slug}"
        _plot_feature_importance(
            class_top,
            dist,
            support_stem,
            sweep.explainability.top_k,
            dataset.class_labels,
            stability_all,
        )
        class_figure_paths[f"feature_support_{slug}"] = support_stem.with_suffix(".svg")
    dist_all = (
        pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
    )
    write_table(target_dir / "feature_distribution_stats.parquet", dist_all)

    dump_json_standard(
        {
            "target": target_slug,
            "target_label": target_label,
            "config": row.to_dict(),
            "methods": list(methods),
            "method_parameters": [method_to_dict(x) for x in method_specs],
            "method_cache": {
                **previous_method_cache,
                **current_method_entries,
            },
            "explanation_source_signature": source_signature,
            "explainability_config": _explainability_config_payload(
                sweep.explainability
            ),
            "explainability_config_signature": _explainability_config_signature(
                sweep.explainability
            ),
            "explained_class_indices": [int(x) for x in class_indices],
            "explained_class_labels": [
                str(dataset.class_labels[int(x)]) for x in class_indices
            ],
            "class_target": "class_probability",
            "explanation_layers": [
                *(["global", "cross_fold_stability"] if global_methods else []),
                *(["local"] if local_enabled else []),
            ],
            "local_explanations": local_mode,
            "local_methods": list(local_methods) if local_enabled else [],
            "local_target": "observed_class_probability",
            "local_storage": "top_k_per_oof_split_full_vectors_for_report_representatives",
            "local_top_k": int(sweep.explainability.local.stored_features),
            "global_methods": [x for x in global_methods if x != "interactions"],
            "interaction_scope": "global_exploratory"
            if "interactions" in global_methods
            else "not_requested",
            "shap_backends": sorted(resolved_shap_backends),
            "default_suite": ["shap", "lime", "ale", "permutation", "interactions"],
            "strict": True,
            "fallbacks": False,
            "cross_validation_explanations": "outer_test_folds",
            "out_of_fold": True,
            "n_outer_folds_explained": len(oof_folds),
            "ale_feature_scope": "all_estimable_features",
            "consensus_strategy": "mean_within_method_rank_support_available_methods",
            "consensus_missing_values": "excluded_not_zero",
            "stability_unit": "outer_fold",
            "stability_interpretation": "descriptive_cross_fit_variability_not_independent_fold_confidence_intervals",
            "stability_statistics": [
                "importance_sd",
                "rank_iqr",
                "top_k_frequency",
                "fold_coverage",
                "sign_consistency",
            ],
            "fold_level_importance_file": "feature_importance_by_outer_fold.parquet",
            "execution": {
                "workers": int(execution.workers),
                "threads_per_worker": int(execution.threads_per_worker),
                "backend": str(execution.backend),
                "logical_cpus": int(execution.logical_cpus),
                "physical_cpus": int(execution.physical_cpus),
            },
        },
        target_dir / "explained_unit.json",
    )
    if not ensemble_explain and cache_dir is not None and target_dir != cache_dir:
        _copy_explainability_cache(target_dir, cache_dir)
    success(
        f"Explainability completed · target={target_label} · methods={','.join(methods)}"
    )
    outputs = {
        "explainability_dir": target_dir,
        "importance": target_dir / "feature_importance.parquet",
        "stability": target_dir / "feature_stability.parquet",
    }
    if coordinate_metadata_path is not None:
        outputs["coordinate_metadata"] = coordinate_metadata_path
    outputs.update(class_figure_paths)
    outputs.update(method_outputs)
    outputs.update(interaction_outputs)
    path_table("Explainability outputs", outputs)
    return outputs


def _score_sort_column(df: pd.DataFrame) -> str | None:
    for col in (
        "inner_validation_score",
        "nMCC_inner_mean",
        "nMCC_mean",
        "score",
        "ROC_AUC_mean",
        "MCC_mean",
    ):
        if col in df.columns:
            return col
    numeric = [
        c
        for c in df.columns
        if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric[0] if numeric else None


def _row_levels(row: pd.Series) -> tuple[str, ...]:
    val = row.get("levels", row.get("resolution", ""))
    if pd.isna(val):
        return ()
    return tuple(x.strip() for x in str(val).replace("+", ",").split(",") if x.strip())


def _resolve_baseline_rf_row(rankings: pd.DataFrame) -> pd.Series | None:
    required = rankings.copy()
    if (
        "learner" not in required.columns
        or "count_transformation" not in required.columns
    ):
        return None
    required = required[
        required["learner"].astype(str).eq("RF_1000_msl5")
        & required["count_transformation"].astype(str).eq("arcsine_sqrt")
    ].copy()
    if required.empty:
        return None
    required["_single_rank"] = required.apply(
        lambda r: _row_levels(r)[0] if len(_row_levels(r)) == 1 else "", axis=1
    )
    for rank in ("strain", "species", "genus"):
        sub = required[required["_single_rank"].eq(rank)].copy()
        if sub.empty:
            continue
        sort_col = _score_sort_column(sub)
        if sort_col and sort_col in sub.columns:
            sub = sub.sort_values(sort_col, ascending=False)
        return sub.iloc[0].drop(labels=["_single_rank"], errors="ignore")
    return None


def _parse_members(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [x.strip() for x in text.split(",") if x.strip()]
    else:
        parsed = value
    out: list[str] = []
    if isinstance(parsed, dict):
        parsed = parsed.get("members", [])
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, dict):
                cid = item.get("config_id") or item.get("id")
            else:
                cid = item
            if cid is not None:
                out.append(str(cid))
    return out


def _selected_ensemble_members(root: Path) -> tuple[list[str], str]:
    selected_path = root / "ensembling" / "selected_unit.json"
    if not selected_path.exists():
        raise ExplainabilityConfigurationError(
            "MPMA-E explainability requires ensembling/selected_unit.json."
        )
    with open(selected_path, "r", encoding="utf-8") as fh:
        selected = json.load(fh)
    unit = selected.get(
        "inner_val_best_mpmas_ensemble", selected.get("MPMA-E", selected)
    )
    members = _parse_members(unit.get("members") if isinstance(unit, dict) else None)
    aggregation = (
        str(unit.get("aggregation_strategy", "mean_proba"))
        if isinstance(unit, dict)
        else "mean_proba"
    )
    if not members:
        raise ExplainabilityConfigurationError(
            "MPMA-E selected unit does not contain members."
        )
    return members, aggregation


def _standard_explainability_targets(sweep: Sweep, rankings: pd.DataFrame) -> list[str]:
    targets: list[str] = []
    if (sweep.root() / "ensembling" / "selected_unit.json").exists() and not getattr(
        sweep, "uses_modalities", False
    ):
        targets.append("mpma_e")
    targets.append("mpma_b")
    if _resolve_baseline_rf_row(rankings) is not None:
        targets.append("baseline_rf")
    return targets


def _automatic_explainability_targets(
    sweep: Sweep, rankings: pd.DataFrame
) -> list[str]:
    configured = getattr(sweep.explainability, "targets", "auto")
    requested = [configured] if isinstance(configured, str) else list(configured)
    standard_aliases = {"auto", "all", "comparison", "standard", "standard_suite"}
    targets: list[str] = []
    for item in requested:
        target = str(item).strip()
        if not target:
            continue
        if target in standard_aliases:
            targets.extend(_standard_explainability_targets(sweep, rankings))
        else:
            targets.append(target)
    if not targets:
        targets.extend(_standard_explainability_targets(sweep, rankings))

    return list(dict.fromkeys(targets))


def explain(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    rankings_path = root / "tables" / "mpma_rankings.parquet"
    if not table_exists(rankings_path):
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = read_table(rankings_path)
    targets = _automatic_explainability_targets(sweep, rankings)
    outputs: dict[str, Path] = {}
    for target in targets:
        out = _explain_one(sweep, target_override=target)
        key = {
            "mpma_e": "mpma_e",
            "ensemble": "mpma_e",
            "MPMA-E": "mpma_e",
            "baseline_rf": "baseline_rf",
            "Baseline RF": "baseline_rf",
            "baseline-rf": "baseline_rf",
            "mpma_b": "mpma_b",
            "best_individual": "mpma_b",
            "best": "mpma_b",
            "MPMA-B": "mpma_b",
        }.get(str(target), str(target))
        for name, path in out.items():
            outputs[f"{key}_{name}"] = path
    return outputs


def _method_display(method: str) -> str:
    m = str(method).strip().lower()
    return {
        "shap": "SHAP",
        "lime": "LIME",
        "ale": "ALE",
        "permutation": "Permutation",
        "consensus": "consensus",
    }.get(m, str(method))


def _method_support_table(
    frames: Sequence[pd.DataFrame], imp: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    if "class_index" not in imp.columns:
        imp = imp.copy()
        imp["class_index"] = 0
        imp["class_label"] = "class_0"
    pieces: list[pd.DataFrame] = []
    supports: dict[str, pd.Series] = {}
    for frame in frames:
        method = _method_display(frame["method"].iloc[0])
        supports[method] = _rank_support_from_importance(frame, top_k=top_k)
    for class_index, class_imp in imp.groupby("class_index", sort=True):
        top = class_imp.copy()
        keys = pd.MultiIndex.from_arrays(
            [
                np.full(len(top), int(class_index), dtype=int),
                top["feature"].astype(str).to_numpy(),
            ]
        )
        for method, support in supports.items():
            top[method] = support.reindex(keys).to_numpy(dtype=float)
        present_columns = [
            col for col in ("SHAP", "LIME", "ALE", "Permutation") if col in top.columns
        ]
        for col in ("SHAP", "LIME", "ALE", "Permutation"):
            if col not in top.columns:
                top[col] = np.nan
        top["consensus"] = (
            top[present_columns].mean(axis=1, skipna=True)
            if present_columns
            else np.nan
        )
        top["n_methods"] = (
            top[present_columns].notna().sum(axis=1) if present_columns else 0
        )
        top["n_methods_total"] = len(present_columns)
        top["method_coverage"] = (
            top["n_methods"] / float(len(present_columns))
            if present_columns
            else np.nan
        )
        top = top.sort_values(
            ["consensus", "n_methods", "importance_mean", "feature"],
            ascending=[False, False, False, True],
            na_position="last",
        ).head(int(top_k))
        top["rank"] = np.arange(1, len(top) + 1, dtype=int)
        pieces.append(top)
    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    cols = [
        "class_index",
        "class_label",
        "rank",
        "feature",
        "SHAP",
        "LIME",
        "ALE",
        "Permutation",
        "consensus",
        "importance_mean",
        "mean_rank",
        "n_methods",
        "n_methods_total",
        "method_coverage",
    ]
    return out[[c for c in cols if c in out.columns]]


def _terminal_taxon_label(value: str) -> str:
    text = str(value).split("___")[-1].split("|")[-1]
    rank = ""
    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if text.startswith(pfx):
            rank = pfx[0] + ". "
            text = text[len(pfx) :]
            break
    text = text.replace("_", " ").strip()
    return f"{rank}{text}" if text else str(value).replace("_", " ")


def _plain_taxon_label(feature_name: str, max_len: int = 34) -> str:
    text = str(feature_name)
    if text.startswith("ALR[") and text.endswith("]"):
        body = text[4:-1]
        if "/" in body:
            numerator, reference = (part.strip() for part in body.split("/", 1))
            out = f"ALR[{_terminal_taxon_label(numerator)} / {_terminal_taxon_label(reference)}]"
        else:
            out = text
    elif text.startswith("ILR_"):
        parts = text.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            out = f"ILR balance {int(parts[1])} · {parts[2][:6]}"
        else:
            out = text.replace("_", " ")
    else:
        out = _terminal_taxon_label(text)
    return feature_tail_ellipsis(out, max_len)


def _plot_feature_importance(
    top_features: pd.DataFrame,
    stats: pd.DataFrame,
    out_stem: Path,
    top_k: int,
    class_labels: Sequence[str],
    stability: pd.DataFrame | None = None,
) -> None:
    _plot_feature_support_visual(
        top_features, stats, out_stem, top_k, class_labels, stability
    )


def _plot_interaction_network(
    tab: pd.DataFrame,
    stats: pd.DataFrame,
    out_stem: Path,
    top_k: int,
    class_labels: Sequence[str],
    layout: str = "default",
) -> bool:
    try:
        return bool(
            _plot_interaction_network_visual(
                tab, stats, out_stem, top_k, class_labels, layout=layout
            )
        )
    except Exception as exc:
        _write_unavailable_interaction_network(out_stem, str(exc))
        return False


def _write_unavailable_interaction_network(out_stem: Path, message: str) -> None:
    apply_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(COL_W_2, 82 * (1.0 / 25.4)))
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "Interaction network unavailable\nno finite 2D-ALE interaction strengths",
        ha="center",
        va="center",
        fontsize=7,
        color=DIM,
        linespacing=1.15,
    )
    ax.text(
        0.5,
        0.38,
        str(message)[:180],
        ha="center",
        va="center",
        fontsize=5.2,
        color=MID,
        linespacing=1.12,
    )
    save_all(fig, out_stem)
    plt.close(fig)


def _standardized_group_shift(
    control: np.ndarray,
    case: np.ndarray,
) -> tuple[float, float, float, float, float]:
    control = np.asarray(control, dtype=float)
    case = np.asarray(case, dtype=float)
    control = control[np.isfinite(control)]
    case = case[np.isfinite(case)]
    control_mean = float(np.mean(control)) if control.size else np.nan
    case_mean = float(np.mean(case)) if case.size else np.nan
    difference = (
        float(case_mean - control_mean)
        if np.isfinite(control_mean) and np.isfinite(case_mean)
        else np.nan
    )
    control_sd = float(np.std(control, ddof=1)) if control.size > 1 else np.nan
    case_sd = float(np.std(case, ddof=1)) if case.size > 1 else np.nan
    pooled_sd = np.nan
    if control.size > 1 and case.size > 1:
        denominator = control.size + case.size - 2
        if denominator > 0:
            pooled_variance = (
                (control.size - 1) * control_sd**2 + (case.size - 1) * case_sd**2
            ) / float(denominator)
            if np.isfinite(pooled_variance) and pooled_variance >= 0:
                pooled_sd = float(np.sqrt(pooled_variance))
    if np.isfinite(difference) and np.isfinite(pooled_sd) and pooled_sd > 0:
        standardized = float(difference / pooled_sd)
    elif np.isfinite(difference) and difference == 0:
        standardized = 0.0
    else:
        standardized = np.nan
    return control_mean, case_mean, difference, pooled_sd, standardized


def _interaction_distribution_stats_for_class(
    features: Sequence[str],
    feature_names: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    class_index: int,
) -> pd.DataFrame:
    index = {str(feature): i for i, feature in enumerate(feature_names)}
    target_mask = np.asarray(y, dtype=int) == int(class_index)
    other_mask = ~target_mask
    rows: list[dict[str, Any]] = []
    for feature in features:
        key = str(feature)
        if key not in index:
            continue
        values = np.asarray(X[:, index[key]], dtype=float)
        control_mean, case_mean, difference, pooled_sd, standardized = (
            _standardized_group_shift(values[other_mask], values[target_mask])
        )
        rows.append(
            {
                "feature": key,
                "control_mean_coordinate": control_mean,
                "case_mean_coordinate": case_mean,
                "case_minus_control_coordinate": difference,
                "pooled_sd_coordinate": pooled_sd,
                "standardized_mean_difference": standardized,
            }
        )
    return pd.DataFrame(rows)


def _ensure_interaction_network_outputs(
    target_dir: Path,
    figures_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    top_k: int,
) -> dict[str, Path]:
    path = target_dir / "feature_interactions_current.parquet"
    if not table_exists(path):
        return {}
    table = read_table(path)
    if table.empty or "class_index" not in table.columns:
        return {}
    outputs: dict[str, Path] = {}
    for class_index in class_indices:
        c = int(class_index)
        class_table = table[table["class_index"].eq(c)].copy()
        class_table["interaction_strength"] = pd.to_numeric(
            class_table.get("interaction_strength", np.nan), errors="coerce"
        )
        class_table = class_table.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["interaction_strength"]
        )
        if class_table.empty:
            continue
        class_table = class_table.sort_values(
            "interaction_strength", ascending=False
        ).head(int(top_k))
        features = list(
            dict.fromkeys(
                class_table.get("feature_1", pd.Series(dtype=object))
                .astype(str)
                .tolist()
                + class_table.get("feature_2", pd.Series(dtype=object))
                .astype(str)
                .tolist()
            )
        )
        stats = _interaction_distribution_stats_for_class(
            features, feature_names, X, y, c
        )
        label = str(class_labels[c])
        other_labels = [str(v) for i, v in enumerate(class_labels) if i != c]
        reference_label = other_labels[0] if len(other_labels) == 1 else "Other classes"
        slug = _class_slug(label)
        stem = figures_dir / f"interaction_network_current__{slug}"
        _plot_interaction_network(
            class_table,
            stats,
            stem,
            int(top_k),
            (reference_label, label),
            layout="default",
        )
        outputs[stem.name] = stem.with_suffix(".svg")
    return outputs


def _feature_distribution_stats(
    features: list[str],
    feature_names: list[str],
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    idx = {f: i for i, f in enumerate(feature_names)}
    rows = []
    for feat in features:
        if feat not in idx:
            continue
        vals = np.asarray(X[:, idx[feat]], dtype=float)
        finite = vals[np.isfinite(vals)]
        row = {
            "feature": feat,
            "overall_mean_coordinate": float(np.mean(finite))
            if finite.size
            else np.nan,
            "overall_median_coordinate": float(np.median(finite))
            if finite.size
            else np.nan,
            "overall_sd_coordinate": float(np.std(finite, ddof=1))
            if finite.size > 1
            else np.nan,
        }
        for c, label in enumerate(labels):
            sub = vals[np.asarray(y) == c]
            sub = sub[np.isfinite(sub)]
            row[f"mean_coordinate_{label}"] = (
                float(np.mean(sub)) if sub.size else np.nan
            )
            row[f"median_coordinate_{label}"] = (
                float(np.median(sub)) if sub.size else np.nan
            )
        if len(labels) >= 2:
            control_mean, case_mean, difference, pooled_sd, standardized = (
                _standardized_group_shift(
                    vals[np.asarray(y) == 0], vals[np.asarray(y) == 1]
                )
            )
            row["control_mean_coordinate"] = control_mean
            row["case_mean_coordinate"] = case_mean
            row["case_minus_control_coordinate"] = difference
            row["pooled_sd_coordinate"] = pooled_sd
            row["standardized_mean_difference"] = standardized
        rows.append(row)
    return pd.DataFrame(rows)
