from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, RLock, Thread
from typing import Iterator

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .console import console, info, success, warn


_CURRENT: ContextVar[InferenceProgress | None] = ContextVar(
    "mllabiome_inference_progress", default=None
)


class InferenceProgress:
    def __init__(self, total_targets: int):
        self.total_targets = max(1, int(total_targets))
        self._lock = RLock()
        self._progress: Progress | None = None
        self._token = None
        self._targets = None
        self._models = None
        self._workflow = None
        self._steps = None
        self._operation = None
        self._phases: list[str] = []
        self._completed_models: set[str] = set()
        self._planned_models: set[str] = set()
        self._current_model: str | None = None
        self._step_count = 0
        self._target_name = ""
        self._target_index = 0
        self._workflow_count = 0

    def __enter__(self) -> InferenceProgress:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=22),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
            refresh_per_second=4,
            disable=False,
            expand=False,
        )
        self._progress.__enter__()
        self._targets = self._progress.add_task("Targets", total=self.total_targets)
        self._workflow = self._progress.add_task(
            "Inference stages", total=10, visible=False
        )
        self._models = self._progress.add_task(
            "Selected models", total=1, visible=False
        )
        self._steps = self._progress.add_task("Model steps", total=7, visible=False)
        self._operation = self._progress.add_task("Starting inference", total=None)
        self._token = _CURRENT.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _CURRENT.reset(self._token)
        with self._lock:
            if self._progress is not None and self._operation is not None:
                self._progress.update(
                    self._operation,
                    description="Inference complete"
                    if exc_type is None
                    else "Inference failed",
                    visible=exc_type is not None,
                )
            if self._progress is not None:
                self._progress.__exit__(exc_type, exc, tb)
        return False

    def target(self, index: int, label: str) -> None:
        with self._lock:
            self._target_index = int(index)
            self._target_name = str(label)
            self._completed_models.clear()
            self._planned_models.clear()
            self._current_model = None
            self._step_count = 0
            self._workflow_count = 0
            if self._progress is not None:
                self._progress.update(
                    self._workflow,
                    description="Inference stages",
                    total=10,
                    completed=0,
                    visible=True,
                )
                self._progress.update(
                    self._targets,
                    description=f"Targets · {label} ({index}/{self.total_targets})",
                )
                self._progress.update(self._models, visible=False)
                self._progress.update(self._steps, visible=False)

    def workflow_step(self, label: str) -> None:
        with self._lock:
            self._workflow_count = min(10, self._workflow_count + 1)
            if self._progress is not None:
                self._progress.update(
                    self._workflow,
                    description=f"Inference stages · {label}",
                    completed=self._workflow_count,
                )

    def plan_models(self, identifiers: list[str]) -> None:
        with self._lock:
            self._planned_models = set(map(str, identifiers))
            self._completed_models.clear()
            if self._progress is not None:
                self._progress.update(
                    self._models,
                    description="Selected models fitted / predicted",
                    total=max(1, len(self._planned_models)),
                    completed=0,
                    visible=bool(self._planned_models),
                )

    def begin_member(self, identifier: str, learner: str) -> None:
        with self._lock:
            self._current_model = str(identifier)
            self._step_count = 0
            if self._progress is not None:
                self._progress.update(
                    self._steps,
                    description=f"Model {identifier} · {learner} · steps",
                    total=7,
                    completed=0,
                    visible=True,
                )

    def member_step(self, description: str) -> None:
        with self._lock:
            if self._current_model is None:
                return
            self._step_count = min(7, self._step_count + 1)
            if self._progress is not None:
                self._progress.update(
                    self._steps,
                    completed=self._step_count,
                    description=f"Model {self._current_model} · {description}",
                )

    def finish_member(self, identifier: str) -> None:
        with self._lock:
            identifier = str(identifier)
            if identifier not in self._completed_models:
                self._completed_models.add(identifier)
                if self._progress is not None:
                    self._progress.update(
                        self._models,
                        completed=len(self._completed_models),
                    )
            if self._progress is not None:
                self._progress.update(self._steps, completed=7, visible=False)
            self._current_model = None
            self._step_count = 0

    def finish_target(self) -> None:
        with self._lock:
            if self._progress is not None:
                self._progress.advance(self._targets, 1)
                self._progress.update(self._models, visible=False)
                self._progress.update(self._steps, visible=False)

    def push_phase(self, label: str) -> None:
        with self._lock:
            self._phases.append(label)
            if self._progress is not None:
                self._progress.update(self._operation, description=label)

    def pop_phase(self, label: str, duration: float, failed: bool) -> None:
        with self._lock:
            if self._phases:
                self._phases.pop()
            if self._progress is not None:
                if failed:
                    self._progress.console.print(
                        f"[red]✗[/red] {label} · failed ({duration:.1f}s)"
                    )
                self._progress.update(
                    self._operation,
                    description=self._phases[-1]
                    if self._phases
                    else f"{label} · done ({duration:.1f}s)",
                )


def current_inference_progress() -> InferenceProgress | None:
    return _CURRENT.get()


@contextmanager
def inference_progress(total_targets: int = 1) -> Iterator[InferenceProgress | None]:
    current = current_inference_progress()
    if current is not None:
        yield current
        return
    if not console.is_terminal:
        yield None
        return
    with InferenceProgress(total_targets) as tracker:
        yield tracker


@contextmanager
def inference_phase(label: str, *, heartbeat_seconds: float = 15.0) -> Iterator[None]:
    name = str(label)
    tracker = current_inference_progress()
    started = time.monotonic()
    if tracker is not None:
        tracker.push_phase(name)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            tracker.pop_phase(name, time.monotonic() - started, failed)
        return
    info(f"Inference · {name}")
    stopped = Event()

    def heartbeat() -> None:
        while not stopped.wait(heartbeat_seconds):
            elapsed = int(time.monotonic() - started)
            info(f"Inference · {name} · still running ({elapsed}s elapsed)")

    monitor = Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        yield
    except BaseException:
        warn(f"Inference · {name} · failed after {time.monotonic() - started:.1f}s")
        raise
    else:
        success(f"Inference · {name} · done ({time.monotonic() - started:.1f}s)")
    finally:
        stopped.set()
        monitor.join(timeout=1.0)
