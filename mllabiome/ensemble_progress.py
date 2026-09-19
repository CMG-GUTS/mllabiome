from __future__ import annotations

from typing import Any

from .console import progress


class EnsembleSearchProgress:
    def __init__(self, scopes: list[str], candidates_per_scope: int):
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

    def __enter__(self):
        self._context = progress()
        self._progress = self._context.__enter__()
        total = self.candidates_per_scope * len(self.scopes)
        self._overall = self._progress.add_task(
            "MPMA-E search · 0 valid · 0 skipped · 0 failures",
            total=max(1, total),
        )
        for scope in self.scopes:
            label = "final pooled selection" if scope == "__final__" else scope
            self._tasks[scope] = self._progress.add_task(
                f"{label} · queued · 0/{self.candidates_per_scope}",
                total=self.candidates_per_scope,
            )
        return self

    def __exit__(self, exc_type, exc, tb):
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
        if event == "start":
            self._progress.update(
                self._tasks[scope],
                description=(
                    f"{label} · {self._done[scope]}/{self.candidates_per_scope} searched · "
                    f"candidate {int(index)}/{int(total)} · {current} · {detail or 'evaluating'}"
                ),
            )
            return
        if event == "done":
            self._done[scope] += 1
            if detail == "valid":
                self._valid[scope] += 1
            else:
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
