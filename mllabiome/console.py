from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console(highlight=False)


def stage(title: str, subtitle: str | None = None) -> None:
    text = f"[bold]{title}[/bold]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(text, border_style="cyan", expand=False))


def info(message: str) -> None:
    console.print(f"[cyan]•[/cyan] {message}")


def success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def warn(message: str) -> None:
    console.print(f"[yellow]![/yellow] {message}")


def error(message: str) -> None:
    console.print(f"[red]✗[/red] {message}")


def summary_table(
    title: str, rows: Mapping[str, object] | Sequence[tuple[str, object]]
) -> None:
    table = Table(title=title, box=None, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value")
    items = rows.items() if isinstance(rows, Mapping) else rows
    for key, value in items:
        table.add_row(str(key), _format_value(value))
    console.print(table)


def path_table(
    title: str, rows: Mapping[str, object] | Sequence[tuple[str, object]]
) -> None:
    items = rows.items() if isinstance(rows, Mapping) else rows
    directories = []
    for _, value in items:
        if value is None:
            continue
        path = Path(value)
        directories.append(path if path.suffix == "" else path.parent)
    if not directories:
        return
    try:
        main = Path(os.path.commonpath([str(path) for path in directories]))
    except ValueError:
        main = directories[0]
    console.print(f"[bold]{title}[/bold] · {main}")


def progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


class PhaseProgress:
    def __init__(self, title: str, total: int):
        self.title = str(title)
        self.total = max(1, int(total))
        self._context = None
        self._progress = None
        self._task = None
        self._completed = 0
        self._active = False

    def __enter__(self):
        self._context = progress()
        self._progress = self._context.__enter__()
        self._task = self._progress.add_task(self.title, total=self.total)
        return self

    def phase(self, label: str) -> None:
        if self._progress is None or self._task is None:
            return
        if self._active:
            self._completed = min(self.total, self._completed + 1)
        self._active = True
        self._progress.update(
            self._task,
            completed=self._completed,
            description=f"{self.title} · {label}",
        )

    def __exit__(self, exc_type, exc, tb):
        if self._progress is not None and self._task is not None and exc is None:
            self._progress.update(
                self._task,
                completed=self.total,
                description=f"{self.title} · complete",
            )
        if self._context is not None:
            return self._context.__exit__(exc_type, exc, tb)
        return False


def phase_progress(title: str, total: int) -> PhaseProgress:
    return PhaseProgress(title, total)


def _format_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    return str(value)
