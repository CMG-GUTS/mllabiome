from __future__ import annotations

import os
import json
import math
import sqlite3
import zlib
from threading import Event, RLock, Thread
from collections.abc import Mapping, Sequence
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
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


def progress(*, auto_refresh: bool = True, transient: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not console.is_terminal,
        auto_refresh=auto_refresh,
        transient=transient,
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
        self._detail_task = None
        self._detail_label = None

    def __enter__(self):
        self._context = progress()
        self._progress = self._context.__enter__()
        self._task = self._progress.add_task(self.title, total=self.total)
        return self

    def _clear_detail(self) -> None:
        if self._progress is not None and self._detail_task is not None:
            self._progress.remove_task(self._detail_task)
        self._detail_task = None
        self._detail_label = None

    def phase(self, label: str) -> None:
        if self._progress is None or self._task is None:
            return
        self._clear_detail()
        if self._active:
            self._completed = min(self.total, self._completed + 1)
        self._active = True
        self._progress.update(
            self._task,
            total=self.total,
            completed=self._completed,
            description=f"{self.title} · {label}",
        )

    def detail(
        self,
        label: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        if self._progress is None:
            return
        label = str(label)
        resolved_total = int(total) if total is not None and int(total) > 1 else None
        resolved_completed = int(completed) if completed is not None else 0
        if self._detail_task is None or self._detail_label != label:
            self._clear_detail()
            self._detail_task = self._progress.add_task(
                f"{self.title} · {label}",
                total=resolved_total,
                completed=resolved_completed,
            )
            self._detail_label = label
            return
        self._progress.update(
            self._detail_task,
            total=resolved_total,
            completed=resolved_completed,
            description=f"{self.title} · {label}",
        )

    def __exit__(self, exc_type, exc, tb):
        self._clear_detail()
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


def _evaluation_names(root: Path, identifiers: Sequence[str]) -> dict[str, str]:
    import sqlite3

    path = Path(root) / "configs.db"
    if not path.exists() or not identifiers:
        return {}
    with sqlite3.connect(path) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='configs'"
        ).fetchone():
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        rows = conn.execute(
            f"SELECT config_id, learner, resolution, count_transformation FROM configs WHERE config_id IN ({placeholders})",
            tuple(identifiers),
        )
        return {
            str(cid): " · ".join(
                str(value)
                for value in (learner, resolution, transformation)
                if value
            )
            for cid, learner, resolution, transformation in rows
        }


def _evaluation_summary(
    root: Path,
    config_id: str,
    metrics: Sequence[str],
) -> dict[str, tuple[dict[str, float], int]]:
    import pandas as pd

    from ._config_store import read_configuration
    from .metrics import aggregate_validation_metric

    result = {}
    for section in ("inner_metrics", "outer_metrics"):
        frame = read_configuration(root, config_id, section)
        if "ok" in frame.columns:
            valid = pd.to_numeric(frame["ok"], errors="coerce").eq(1)
            frame = frame.loc[valid]
        key = "inner_key" if section == "inner_metrics" else "split_key"
        if key in frame.columns:
            frame = frame.drop_duplicates(key, keep="last")
        scores = {
            metric: aggregate_validation_metric(frame, metric)[0]
            for metric in metrics
        }
        result[section] = (scores, len(frame))
    return result


def _evaluation_score(value: float) -> str:
    import math

    return f"{value:.5f}" if math.isfinite(value) else "—"


def evaluation_completion(
    root: Path,
    metric: str,
    ranked: Sequence[tuple[str, float, int]],
    config_id: str,
    *,
    classification: bool = True,
) -> None:
    import pandas as pd

    from ._config_store import read_configuration
    from .metrics import aggregate_validation_metric

    position = next(
        (index for index, item in enumerate(ranked, 1) if item[0] == config_id),
        None,
    )
    names = _evaluation_names(root, [config_id])
    name = names.get(config_id, config_id)
    if position is None:
        warn(f"Configuration finished without a complete inner ranking · {name}")
        return
    metrics = [metric]
    other = ("AUROC", "F1w", "MCC") if classification else ("R2", "MAE", "RMSE")
    metrics.extend(value for value in other if value != metric)
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_lines=False)
    table.add_column("Validation", style="bold")
    table.add_column("Folds", justify="right", style="dim")
    for item in metrics:
        table.add_column(item, justify="right", no_wrap=True)
    for section, label, key in (
        ("inner_metrics", "Inner", "inner_key"),
        ("outer_metrics", "Outer", "split_key"),
    ):
        frame = read_configuration(root, config_id, section)
        if "ok" in frame.columns:
            frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
        if key in frame.columns:
            frame = frame.drop_duplicates(key, keep="last")
        values = []
        for item in metrics:
            mean, std = aggregate_validation_metric(frame, item)
            digits = 5 if item == metric else 4
            values.append(
                f"{mean:.{digits}f} ± {std:.{digits}f}"
                if math.isfinite(mean) and math.isfinite(std)
                else "—"
            )
        table.add_row(label, str(len(frame)), *values, style="bold" if label == "Inner" else None)
    console.print(
        Panel(
            table,
            title=f"[green]✓ Configuration finalized[/green] · rank {position}/{len(ranked)}",
            subtitle=name,
            border_style="green",
            expand=False,
        )
    )


class EvaluationMonitor:
    def __init__(
        self,
        root: Path,
        metric: str,
        pending_config_ids: Sequence[str],
        outer_keys: Sequence[str],
        inner_keys: Sequence[str],
        existing_inner: Sequence[Mapping[str, object]],
        existing_outer: Sequence[Mapping[str, object]],
        *,
        all_config_ids: Sequence[str],
        completed_pairs: set[tuple[str, str]],
        classification: bool = True,
    ):
        self.root = Path(root)
        self.metric = str(metric)
        self.classification = bool(classification)
        self.metrics = [self.metric]
        extras = ("AUROC", "F1w", "MCC") if self.classification else ("R2", "MAE", "RMSE")
        self.metrics.extend(item for item in extras if item != self.metric)
        self.names = _evaluation_names(self.root, pending_config_ids)
        self.config_ids = sorted(set(map(str, pending_config_ids)), key=lambda item: (self.names.get(item, item), item))
        self.expected_outer = set(map(str, outer_keys))
        self.expected_inner = set(map(str, inner_keys))
        self.all_config_ids = set(map(str, all_config_ids))
        self.completed_pairs = set(completed_pairs)
        self.total_pairs = len(self.all_config_ids) * len(self.expected_outer)
        self._reported_completed = len(self.completed_pairs)
        self.inner = {cid: {} for cid in self.config_ids}
        self.outer = {cid: {} for cid in self.config_ids}
        self.finished = set()
        self._lock = RLock()
        self._stop = Event()
        self._thread = None
        self._live = None
        for section, rows in (("inner", existing_inner), ("outer", existing_outer)):
            for row in rows:
                self._store(section, row, announce=False)

    def _store(self, section: str, row: Mapping[str, object], *, announce: bool) -> bool:
        cid = str(row.get("config_id", ""))
        if cid not in self.inner:
            return False
        key_field = "inner_key" if section == "inner" else "split_key"
        key = str(row.get(key_field, ""))
        if not key or key not in (self.expected_inner if section == "inner" else self.expected_outer):
            return False
        bucket = self.inner[cid] if section == "inner" else self.outer[cid]
        previous = bucket.get(key)
        value = dict(row)
        if previous == value:
            return False
        bucket[key] = value
        return True

    def _valid(self, rows: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
        return [dict(row) for row in rows.values() if str(row.get("ok", 0)) in {"1", "1.0", "True"}]

    def _score(self, rows: Sequence[Mapping[str, object]], metric: str) -> str:
        import pandas as pd

        from .metrics import aggregate_validation_metric

        score, _ = aggregate_validation_metric(pd.DataFrame(rows), metric)
        return f"{score:.4f}" if math.isfinite(score) else "—"

    def _counts(self, rows: Mapping[str, Mapping[str, object]], expected: set[str]) -> str:
        valid = self._valid(rows)
        errors = len(rows) - len(valid)
        count = f"{len(valid)}/{len(expected)}"
        return f"{count} [red]({errors} failed)[/red]" if errors else count

    def _render(self) -> Group:
        table = Table(
            title="Validation by configuration · live fold aggregates",
            box=box.SIMPLE_HEAD,
            header_style="bold",
            show_lines=False,
            pad_edge=False,
            expand=False,
        )
        table.add_column("Configuration", no_wrap=False, max_width=35)
        table.add_column("Level", style="dim", no_wrap=True)
        table.add_column("Folds", justify="right", no_wrap=True)
        for item in self.metrics:
            table.add_column(item, justify="right", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        for cid in self.config_ids:
            for section, title, expected in (
                ("inner", "Inner", self.expected_inner),
                ("outer", "Outer", self.expected_outer),
            ):
                bucket = self.inner[cid] if section == "inner" else self.outer[cid]
                valid = self._valid(bucket)
                status = (
                    "[green]Finalized[/green]"
                    if cid in self.finished
                    else "[cyan]Evaluating[/cyan]"
                    if self.inner[cid] or self.outer[cid]
                    else "[dim]Queued[/dim]"
                )
                table.add_row(
                    self.names.get(cid, cid) if section == "inner" else "",
                    title,
                    self._counts(bucket, expected),
                    *(self._score(valid, item) for item in self.metrics),
                    status if section == "inner" else "",
                    style="bold" if section == "outer" else None,
                    end_section=section == "outer",
                )
        progress = Progress(
            TextColumn("[bold cyan]Overall[/bold cyan]"),
            BarColumn(bar_width=34, complete_style="cyan", finished_style="green"),
            TextColumn("[bold]{task.percentage:>5.1f}%[/bold]"),
            TextColumn("[dim]{task.completed:.0f}/{task.total:.0f} pairs[/dim]"),
            console=console,
            auto_refresh=False,
            expand=False,
        )
        progress.add_task(
            "Evaluation", total=max(1, self.total_pairs),
            completed=min(len(self.completed_pairs), self.total_pairs),
        )
        return Group(
            table,
            Panel(
                progress,
                title="[bold]Overall evaluation[/bold]",
                subtitle="[dim]Resolved MPMA/outer-split pairs[/dim]",
                border_style="cyan",
                expand=False,
                padding=(0, 1),
            ),
        )

    def _update(
        self,
        events: Sequence[tuple[str, str, Mapping[str, object], int]],
        *,
        changed: bool = False,
    ) -> None:
        if not events and not changed:
            return
        with self._lock:
            if self._live is not None:
                self._live.update(self._render(), refresh=True)
            elif len(self.completed_pairs) != self._reported_completed:
                self._reported_completed = len(self.completed_pairs)
                console.print(
                    f"[cyan]Evaluation progress[/cyan] · "
                    f"{len(self.completed_pairs):,}/{self.total_pairs:,} resolved MPMA/outer-split pairs"
                )

    def _poll(self) -> None:
        if not self.config_ids:
            return
        path = self.root / "configs.db"
        if not path.exists():
            return
        try:
            with sqlite3.connect(path, timeout=2) as conn:
                found = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='inner_fold_checkpoints'"
                ).fetchone()
                if not found:
                    return
                placeholders = ",".join("?" for _ in self.config_ids)
                keys = conn.execute(
                    f"SELECT config_id, inner_key FROM inner_fold_checkpoints WHERE config_id IN ({placeholders})",
                    self.config_ids,
                ).fetchall()
                with self._lock:
                    missing = [(cid, key) for cid, key in keys if key not in self.inner.get(cid, {})]
                if not missing:
                    return
                events = []
                for cid, key in missing:
                    found_row = conn.execute(
                        "SELECT payload FROM inner_fold_checkpoints WHERE config_id=? AND inner_key=?",
                        (cid, key),
                    ).fetchone()
                    if found_row is None:
                        continue
                    payload = json.loads(zlib.decompress(found_row[0]).decode("utf-8"))
                    for item in payload.get("inner_metrics", []):
                        with self._lock:
                            if self._store("inner", item, announce=True):
                                events.append(("inner", cid, item, len(self._valid(self.inner[cid]))))
                self._update(events)
        except (sqlite3.Error, OSError, ValueError, zlib.error):
            return

    def _watch(self) -> None:
        while not self._stop.wait(0.4):
            self._poll()

    def __enter__(self):
        console.print("[dim]Fold metrics are checkpointed as completed; outer results remain held out.[/dim]")
        if console.is_terminal:
            self._live = Live(
                self._render(),
                console=console,
                auto_refresh=False,
                transient=True,
                vertical_overflow="visible",
            )
            self._live.__enter__()
        else:
            console.print(self._render())
        self._thread = Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def result(self, result: Mapping[str, object]) -> None:
        self._poll()
        events = []
        with self._lock:
            previous = len(self.completed_pairs)
            for section in ("inner", "outer"):
                for row in result.get(f"{section}_metrics", []):
                    if self._store(section, row, announce=True):
                        cid = str(row["config_id"])
                        bucket = self.inner[cid] if section == "inner" else self.outer[cid]
                        events.append((section, cid, row, len(self._valid(bucket))))
                    if section == "outer" and str(row.get("ok", 0)) in {"1", "1.0", "True"}:
                        self._complete_pair(row)
            for row in result.get("qualification", []):
                if str(row.get("qualified", 1)) in {"0", "0.0", "False"}:
                    self._complete_pair(row)
            changed = len(self.completed_pairs) != previous
        self._update(events, changed=changed)

    def _complete_pair(self, row: Mapping[str, object]) -> None:
        pair = (str(row.get("split_key", "")), str(row.get("config_id", "")))
        if pair[0] in self.expected_outer and pair[1] in self.all_config_ids:
            self.completed_pairs.add(pair)

    def finalized(self, config_id: str, ranked: Sequence[tuple[str, float, int]]) -> None:
        self._poll()
        with self._lock:
            self.finished.add(config_id)
            if self._live is not None:
                self._live.update(self._render(), refresh=True)
        evaluation_completion(
            self.root,
            self.metric,
            ranked,
            config_id,
            classification=self.classification,
        )

    def unfinished(self, config_id: str) -> None:
        warn(f"Configuration remains resumable · {self.names.get(config_id, config_id)} · incomplete or failed folds")

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._poll()
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
        return False


def evaluation_leaders(
    root: Path,
    metric: str,
    ranked: Sequence[tuple[str, float, int]],
    *,
    latest_config_id: str | None = None,
    classification: bool = True,
    limit: int = 20,
) -> None:
    if not ranked:
        info("No completed configurations are available for ranking.")
        return
    metric_names = [metric]
    additional = ("AUROC", "F1w", "MCC") if classification else ("R2", "MAE", "RMSE")
    metric_names.extend(name for name in additional if name != metric)
    displayed = list(ranked[:limit])
    position = next(
        (index for index, item in enumerate(ranked, 1) if item[0] == latest_config_id),
        None,
    )
    if position is not None and position > limit:
        displayed.append(ranked[position - 1])
    names = _evaluation_names(root, [item[0] for item in displayed])
    ranks = {item[0]: index for index, item in enumerate(ranked, 1)}
    direction = (
        "lower is better" if _evaluation_metric_is_loss(metric) else "higher is better"
    )
    table = Table(
        title=f"Configuration ranking · top {min(limit, len(ranked))} of {len(ranked)} · inner {metric} ({direction})",
        box=box.SIMPLE_HEAD,
        pad_edge=False,
        show_lines=False,
        header_style="bold",
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Configuration", no_wrap=False)
    table.add_column("Validation", no_wrap=True)
    for name in metric_names:
        table.add_column(name, justify="right", no_wrap=True)
    table.add_column("Folds", justify="right", style="dim", no_wrap=True)
    for config_id, _, _ in displayed:
        rank = ranks[config_id]
        summary = _evaluation_summary(root, config_id, metric_names)
        name = names.get(config_id, config_id)
        if config_id == latest_config_id:
            name += " [cyan]· latest[/cyan]"
        for section, label in (("inner_metrics", "Inner"), ("outer_metrics", "Outer")):
            scores, folds = summary[section]
            table.add_row(
                str(rank) if section == "inner_metrics" else "",
                name if section == "inner_metrics" else "",
                label,
                *(_evaluation_score(scores[column]) for column in metric_names),
                str(folds),
                style="bold" if section == "inner_metrics" and rank == 1 else None,
                end_section=section == "outer_metrics",
            )
    console.print(table)
    console.print(
        "[dim]Ranks use aggregated inner cross-validation scores for the optimization metric. "
        "Outer metrics describe held-out evaluation. "
        "Outer scores should not be used for model selection.[/dim]"
    )
    if position is not None:
        config_id, score, _ = ranked[position - 1]
        console.print(
            f"[cyan]Latest completed[/cyan] · rank {position}/{len(ranked)} · "
            f"{names.get(config_id, config_id)} · inner {metric}={score:.5f}"
        )


def _evaluation_metric_is_loss(metric: str) -> bool:
    from .metrics import metric_is_loss

    return metric_is_loss(metric)
