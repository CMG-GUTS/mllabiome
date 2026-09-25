from __future__ import annotations

import logging
from contextlib import contextmanager
from queue import Empty
from threading import Event, Thread
from typing import Any, Callable, Sequence

from threadpoolctl import threadpool_limits

from .console import progress
from .runtime import iter_parallel_tasks, resolve_execution_plan, thread_environment
from .sweep_types import Sweep


@contextmanager
def _quiet_pyale_info():
    saved_disable = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(saved_disable)


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
