from __future__ import annotations
from typing import Any
from threading import Event, RLock, Thread
from time import monotonic
from .console import progress


class EnsembleSearchProgress:
    def __init__(
        self, scopes: list[str], candidates_per_scope: int, *, pooled_grid: bool = False
    ):
        self.pooled_grid = bool(pooled_grid)
        self.scopes = [str(x) for x in scopes]
        self.candidates_per_scope = max(1, int(candidates_per_scope))
        self._context = None
        self._progress = None
        self._overall = None
        self._tasks: dict[str, int] = {}
        self._done = {scope: 0 for scope in self.scopes}
        self._valid = {scope: 0 for scope in self.scopes}
        self._skipped = {scope: 0 for scope in self.scopes}
        self._failures = 0
        self._phase_task = None
        self._preparation_task = None
        self._heartbeat_stop = Event()
        self._heartbeat_thread = None
        self._status_lock = RLock()
        self._activity_task = None
        self._activity_label = ""
        self._activity_since = 0.0

    def __enter__(self):
        self._context = progress()
        self._progress = self._context.__enter__()
        total = self.candidates_per_scope * len(self.scopes)
        pooled_only = self.scopes == ["__final__"] and self.candidates_per_scope == 1
        self._overall = self._progress.add_task(
            (
                f"MPMA-E search · preparing {self.candidates_per_scope} candidates"
                if self.pooled_grid
                else "MPMA-E search · 0 valid · 0 skipped · 0 failures"
            ),
            total=max(1, total),
            visible=not pooled_only,
        )
        if self.pooled_grid:
            self._preparation_task = self._progress.add_task(
                "Preparing pooled candidate library · starting",
                total=None,
            )
            self._set_activity(
                self._preparation_task, "Preparing pooled candidate library · starting"
            )
        if (
            self.scopes == ["__final__"]
            and self.candidates_per_scope == 1
            and not self.pooled_grid
        ):
            self._phase_task = self._progress.add_task(
                "Pooled Pareto · preparing selection",
                total=5,
            )
        for scope in self.scopes:
            label = "final pooled selection" if scope == "__final__" else scope
            self._tasks[scope] = self._progress.add_task(
                (
                    f"{label} · waiting for library preparation"
                    if self.pooled_grid
                    else f"{label} · queued · 0/{self.candidates_per_scope}"
                ),
                total=self.candidates_per_scope,
                visible=not pooled_only and not self.pooled_grid,
            )
        if self.pooled_grid:
            self._heartbeat_thread = Thread(target=self._heartbeat, daemon=True)
            self._heartbeat_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        if self._progress is not None and self._phase_task is not None and exc is None:
            self._progress.update(
                self._phase_task,
                completed=5,
                description="Pooled Pareto · fixed ensemble selected",
            )
        if self._progress is not None and self._overall is not None and exc is None:
            total = self.candidates_per_scope * len(self.scopes)
            self._progress.update(
                self._overall,
                completed=total,
                description=(
                    f"MPMA-E search · complete · {sum(self._valid.values())} valid · "
                    f"{sum(self._skipped.values())} skipped · {self._failures} failures"
                ),
            )
        if self._context is not None:
            return self._context.__exit__(exc_type, exc, tb)
        return False

    def _set_activity(self, task: int | None, label: str) -> None:
        with self._status_lock:
            self._activity_task = task
            self._activity_label = label
            self._activity_since = monotonic()
            if task is not None and self._progress is not None:
                self._progress.update(task, description=label)

    def _heartbeat(self) -> None:
        while not self._heartbeat_stop.wait(10.0):
            with self._status_lock:
                if self._activity_task is None or self._progress is None:
                    continue
                elapsed = int(monotonic() - self._activity_since)
                if elapsed < 10:
                    continue
                self._progress.update(
                    self._activity_task,
                    description=f"{self._activity_label} · {elapsed}s in progress",
                )

    def update(
        self,
        scope: str,
        event: str,
        spec: dict[str, Any] | None,
        index: int,
        total: int,
        detail: str = "",
    ) -> None:
        if self._progress is None or self._overall is None:
            return
        scope = str(scope)
        if scope not in self._tasks:
            return
        label = "final pooled selection" if scope == "__final__" else scope
        method = str((spec or {}).get("selection_strategy", ""))
        aggregation = str((spec or {}).get("aggregation_strategy", ""))
        max_size = (spec or {}).get("max_size", "")
        current = " · ".join(
            x
            for x in (method, aggregation, f"max {max_size}" if max_size != "" else "")
            if x
        )
        if event == "prepare":
            self._set_activity(
                self._preparation_task, f"Preparing pooled library · {detail}"
            )
            return
        if event == "ready":
            self._set_activity(None, "")
            if self._preparation_task is not None:
                self._progress.update(
                    self._preparation_task,
                    description=f"Pooled library ready · {detail}",
                    total=1,
                    completed=1,
                )
            self._progress.update(
                self._overall,
                description=f"MPMA-E search · scoring {self.candidates_per_scope} candidates",
            )
            self._progress.update(self._tasks[scope], visible=True)
            return
        if event == "phase":
            if self._phase_task is not None:
                self._progress.update(
                    self._phase_task,
                    completed=min(5, max(0, int(index) - 1)),
                    description=f"Pooled Pareto · {detail}",
                )
            return
        if event == "start":
            description = (
                f"{label} · {self._done[scope]}/{self.candidates_per_scope} scored · "
                f"candidate {int(index)}/{int(total)} · {current} · {detail or 'evaluating'}"
            )
            self._progress.update(self._tasks[scope], description=description)
            if self.pooled_grid:
                self._set_activity(self._tasks[scope], description)
            return
        if event == "done":
            if self.pooled_grid:
                self._set_activity(None, "")
            self._done[scope] += 1
            if detail == "valid":
                self._valid[scope] += 1
            elif detail == "skipped":
                self._skipped[scope] += 1
            self._progress.update(
                self._tasks[scope],
                completed=self._done[scope],
                description=(
                    f"{label} · {self._done[scope]}/{self.candidates_per_scope} searched · "
                    f"{self._valid[scope]} valid · {self._skipped[scope]} skipped · last {current}"
                ),
            )
            self._progress.update(
                self._overall,
                completed=sum(self._done.values()),
                description=(
                    f"MPMA-E search · {sum(self._valid.values())} valid · "
                    f"{sum(self._skipped.values())} skipped · {self._failures} failures"
                ),
            )
            return
        if event == "error":
            self._failures += 1
            self._progress.update(
                self._tasks[scope],
                description=f"{label} · error · {current} · {detail}",
            )
            self._progress.update(
                self._overall,
                description=(
                    f"MPMA-E search · {sum(self._valid.values())} valid · "
                    f"{sum(self._skipped.values())} skipped · {self._failures} failures"
                ),
            )
            return
        if event == "selected":
            self._progress.update(
                self._tasks[scope],
                completed=self.candidates_per_scope,
                description=f"{label} · selected · {current} · {detail}",
            )
