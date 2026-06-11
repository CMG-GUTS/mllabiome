from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

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


def summary_table(title: str, rows: Mapping[str, object] | Sequence[tuple[str, object]]) -> None:
    table = Table(title=title, box=None, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value")
    items = rows.items() if isinstance(rows, Mapping) else rows
    for key, value in items:
        table.add_row(str(key), _format_value(value))
    console.print(table)


def path_table(title: str, rows: Mapping[str, object] | Sequence[tuple[str, object]]) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("Output", style="bold")
    table.add_column("Path", overflow="fold")
    items = rows.items() if isinstance(rows, Mapping) else rows
    for key, value in items:
        if value is None:
            continue
        table.add_row(str(key), str(value))
    console.print(table)


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


def _format_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    return str(value)
