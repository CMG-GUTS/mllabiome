from __future__ import annotations

from contextvars import ContextVar, Token
from time import perf_counter
from typing import Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from ..console import console, info, success


_ACTIVE: ContextVar[BrainProgress | None] = ContextVar(
    "mllabiome_brain_progress", default=None
)


class WorkColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("running", style="cyan")
        return Text(f"{int(task.completed):,}/{int(task.total):,}", style="bold")


class RemainingColumn(TimeRemainingColumn):
    def render(self, task: Task) -> Text:
        if (
            task.total is None
            or task.completed == 0
            or not task.fields.get("estimate", False)
        ):
            return Text("—", style="dim")
        return super().render(task)


class BrainProgress:
    def __init__(self, total: int, *, output: Console | None = None) -> None:
        if total <= 0:
            raise ValueError("Representation count must be positive")
        self.total = total
        self.output = output or console
        self.interactive = bool(self.output.is_terminal)
        self.renderer = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}[/bold]", justify="left"),
            BarColumn(
                bar_width=22,
                complete_style="green",
                finished_style="green",
                pulse_style="cyan",
            ),
            WorkColumn(),
            TimeElapsedColumn(),
            RemainingColumn(),
            console=self.output,
            refresh_per_second=8,
            transient=False,
            disable=not self.interactive,
            expand=False,
        )
        self.parent: int | None = None
        self.current: int | None = None
        self._token: Token | None = None
        self._count = 0
        self._units_done = 0
        self._units_total = 0
        self._units_label = ""
        self._started = perf_counter()
        self._phase_start = self._started

    def __enter__(self) -> Self:
        self.renderer.__enter__()
        self.parent = self.renderer.add_task(
            "Brain representations", total=self.total, estimate=False
        )
        self.current = self.renderer.add_task(
            "Preparing inputs", total=None, estimate=False
        )
        self._token = _ACTIVE.set(self)
        if not self.interactive:
            info(f"Brain · starting {self.total} representation candidates")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
        if exc_type is not None and self.current is not None:
            self.renderer.update(
                self.current,
                description=f"Failed · {self._units_label or 'brain preparation'}",
            )
            if not self.interactive:
                info(f"Brain · failed during {self._units_label or 'preparation'}")
        if exc_type is None and self.current is not None:
            self.renderer.update(self.current, visible=False)
        self.renderer.__exit__(exc_type, exc, tb)
        if exc_type is None:
            success(
                f"Brain · preparation finished in {perf_counter() - self._started:.1f}s"
            )

    def phase(self, label: str) -> None:
        self._units_label = str(label)
        self._phase_start = perf_counter()
        self._units_done = 0
        self._units_total = 0
        if self.current is not None:
            self.renderer.remove_task(self.current)
        self.current = self.renderer.add_task(label, total=None, estimate=False)
        if self.interactive:
            self.output.print(f"[dim]◦[/dim] {label}")
        else:
            info(f"Brain · {label} · running")

    def begin_units(self, label: str, total: int) -> None:
        if total <= 0:
            raise ValueError(f"Cannot report progress for empty workload: {label}")
        self._units_label = str(label)
        self._phase_start = perf_counter()
        self._units_done = 0
        self._units_total = int(total)
        self.renderer.update(
            self.current,
            description=label,
            completed=0,
            total=total,
            visible=True,
            estimate=True,
        )
        if self.interactive:
            self.output.print(
                f"[dim]◦[/dim] {label} · [bold]{total:,} work units[/bold]"
            )
        else:
            info(f"Brain · {label} · 0/{total}")

    def working(self, label: str) -> None:
        if self._units_total <= 0:
            raise RuntimeError("Start a counted workload before marking a subject")
        self.renderer.update(self.current, description=f"{self._units_label} · {label}")
        if not self.interactive:
            info(
                f"Brain · {self._units_label} · processing {self._units_done + 1}/{self._units_total} · {label}"
            )

    def substep(self, label: str) -> None:
        if self._units_total <= 0:
            raise RuntimeError("Start a counted workload before describing a substep")
        self.renderer.update(self.current, description=f"{self._units_label} · {label}")
        if not self.interactive:
            info(f"Brain · {self._units_label} · {label}")

    def advance(self) -> None:
        if self._units_total <= 0 or self._units_done >= self._units_total:
            raise RuntimeError(
                "Progress advance exceeds the declared number of work units"
            )
        self._units_done += 1
        self.renderer.update(self.current, completed=self._units_done)
        if not self.interactive:
            info(
                f"Brain · {self._units_label} · {self._units_done}/{self._units_total} completed"
            )
        if self._units_done == self._units_total:
            self.renderer.update(
                self.current, description=f"{self._units_label} · complete"
            )

    def complete_representation(self, label: str) -> None:
        if self._count >= self.total:
            raise RuntimeError("Completed more representations than declared")
        self._count += 1
        self.renderer.update(
            self.parent,
            completed=self._count,
            description=f"Brain representations · {label}",
        )
        self.phase(
            f"Preparing next representation ({self._count}/{self.total} complete)"
        )
        if not self.interactive:
            success(
                f"Brain · representation {self._count}/{self.total} complete · {label}"
            )

    def complete_from_cache(self) -> None:
        self._count = self.total
        self.renderer.update(
            self.parent,
            completed=self.total,
            description="Brain representations · cache verified",
        )
        self.phase("Cached feature table verified")


def active_progress() -> BrainProgress | None:
    return _ACTIVE.get()
