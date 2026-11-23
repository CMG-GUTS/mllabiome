from __future__ import annotations
import os
import json
import math
import time
from threading import Lock
import sqlite3
import zlib
from threading import Event, RLock, Thread
from datetime import datetime
from collections.abc import Mapping, Sequence
from pathlib import Path
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
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
_batch_lock = RLock()


def _batch_log(level: str, message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with _batch_lock:
        print(f"[{stamp}] [{level}] {message}", flush=True)


def stage(title: str, subtitle: str | None = None) -> None:
    if not console.is_terminal:
        _batch_log("STAGE", title if not subtitle else f"{title} · {subtitle}")
        return
    text = f"[bold]{title}[/bold]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(text, border_style="cyan", expand=False))


def info(message: str) -> None:
    if console.is_terminal:
        console.print(f"[cyan]•[/cyan] {message}")
    else:
        _batch_log("INFO", message)


def success(message: str) -> None:
    if console.is_terminal:
        console.print(f"[green]✓[/green] {message}")
    else:
        _batch_log("DONE", message)


def warn(message: str) -> None:
    if console.is_terminal:
        console.print(f"[yellow]![/yellow] {message}")
    else:
        _batch_log("WARN", message)


def error(message: str) -> None:
    if console.is_terminal:
        console.print(f"[red]✗[/red] {message}")
    else:
        _batch_log("ERROR", message)


def summary_table(
    title: str, rows: Mapping[str, object] | Sequence[tuple[str, object]]
) -> None:
    table = Table(title=title, box=None, show_header=False, pad_edge=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value")
    items = rows.items() if isinstance(rows, Mapping) else rows
    if not console.is_terminal:
        info(title)
        for key, value in items:
            info(f"  {key}: {_format_value(value)}")
        return
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
    if console.is_terminal:
        console.print(f"[bold]{title}[/bold] · {main}")
    else:
        info(f"{title} · {main}")


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


class SelectionStatus:
    def __init__(self, title: str = "MPMA-B", heartbeat_seconds: int = 60):
        self.title = title
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = Lock()
        self._stop = Event()
        self._start = 0.0
        self._phase = "Starting"
        self._detail = ""
        self._reported: dict[str, tuple[int, float]] = {}
        self._thread: Thread | None = None

    def __enter__(self):
        self._start = time.monotonic()
        self._stop.clear()
        info(f"{self.title} · started")
        self._thread = Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def phase(self, label: str) -> None:
        with self._lock:
            self._phase = str(label)
            self._detail = ""
        info(f"{self.title} · {label}")

    def progress(self, label: str, completed: int, total: int) -> None:
        if total <= 0:
            return
        completed = min(int(total), max(0, int(completed)))
        now = time.monotonic()
        with self._lock:
            self._detail = f"{label}: {completed}/{total}"
            old_step, old_time = self._reported.get(label, (-1, 0.0))
            step = completed * 10 // total
            log = (
                completed == total
                or old_step < 0
                or step > old_step
                or now - old_time >= 20
            )
            if log:
                self._reported[label] = (step, now)
        if log:
            info(
                f"{self.title} · {label}: {completed}/{total} ({completed / total:.0%})"
            )

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                detail = self._detail or self._phase
            elapsed = int(time.monotonic() - self._start)
            info(f"{self.title} · heartbeat · {detail} · elapsed {elapsed}s")

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        elapsed = int(time.monotonic() - self._start)
        if exc_type is None:
            success(f"{self.title} · finished · elapsed {elapsed}s")
        else:
            error(
                f"{self.title} · failed after {elapsed}s · {exc_type.__name__}: {exc}"
            )
        return False


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
    from . import cooperative_store

    if cooperative_store.enabled(Path(root)):
        from ._config_store import _artifact_path, _file_metadata

        names = {}
        for cid in identifiers:
            path = _artifact_path(root, cid)
            if path.is_file():
                meta = _file_metadata(path).get("configuration", {})
                names[str(cid)] = " · ".join(
                    str(meta.get(field, ""))
                    for field in ("learner", "resolution", "count_transformation")
                )
        return names
    path = Path(root) / "configs.db"
    if not path.exists() or not identifiers:
        return {}
    with sqlite3.connect(path) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='configs'"
        ).fetchone():
            return {}
        available = {row[1] for row in conn.execute("PRAGMA table_info(configs)")}
        fields = [
            column
            for column in (
                "learner",
                "modalities",
                "integration",
                "integration_n_components",
                "resolution",
                "count_transformation",
            )
            if column in available
        ]
        if "config_id" not in available or not fields:
            return {}
        names = {}
        for start in range(0, len(identifiers), 500):
            chunk = list(identifiers[start : start + 500])
            placeholders = ",".join("?" for _ in chunk)
            query = (
                f"SELECT config_id, {', '.join(fields)} FROM configs "
                f"WHERE config_id IN ({placeholders})"
            )
            for record in conn.execute(query, tuple(chunk)):
                values = dict(zip(fields, record[1:]))
                parts = []
                for field in ("learner", "modalities", "integration"):
                    value = values.get(field)
                    if value is not None and str(value).strip():
                        parts.append(str(value).strip())
                components = values.get("integration_n_components")
                if components is not None and str(components).strip():
                    parts.append(f"components={components}")
                for field in ("resolution", "count_transformation"):
                    value = values.get(field)
                    if value is not None and str(value).strip():
                        parts.append(str(value).strip())
                names[str(record[0])] = " · ".join(parts) or str(record[0])
        return names


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
            metric: aggregate_validation_metric(frame, metric)[0] for metric in metrics
        }
        result[section] = (scores, len(frame))
    return result


def _evaluation_score(value: float) -> str:
    import math

    return f"{value:.5f}" if math.isfinite(value) else "—"


def evaluation_completion(
    root: Path,
    metric: str,
    ranked: Sequence[tuple[str, float, int]] | None,
    config_id: str,
    *,
    classification: bool = True,
    display_name: str | None = None,
) -> None:
    import pandas as pd
    from ._config_store import read_configuration
    from .metrics import aggregate_validation_metric

    position = (
        next(
            (index for index, item in enumerate(ranked, 1) if item[0] == config_id),
            None,
        )
        if ranked is not None
        else None
    )
    name = display_name or _evaluation_names(root, [config_id]).get(
        config_id, config_id
    )
    if ranked is not None and position is None:
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
    batch_lines = []
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
        table.add_row(
            label, str(len(frame)), *values, style="bold" if label == "Inner" else None
        )
        batch_lines.append(
            f"{label} · folds={len(frame)} · "
            + " · ".join(f"{item}={value}" for item, value in zip(metrics, values))
        )
    if not console.is_terminal:
        rank_text = f" · rank {position}/{len(ranked)}" if position is not None else ""
        success(f"MPMA finalized{rank_text} · {name}")
        for line in batch_lines:
            info(f"  {line}")
        return
    title = (
        f"[green]✓ Configuration finalized[/green] · rank {position}/{len(ranked)}"
        if ranked is not None
        else "[green]✓ Configuration finalized[/green]"
    )
    console.print(
        Panel(
            table,
            title=title,
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
        display_names: Mapping[str, str] | None = None,
    ):
        self.root = Path(root)
        self.metric = str(metric)
        self.classification = bool(classification)
        self.metrics = [self.metric]
        extras = (
            ("AUROC", "F1w", "MCC") if self.classification else ("R2", "MAE", "RMSE")
        )
        self.metrics.extend(item for item in extras if item != self.metric)
        self.names = _evaluation_names(self.root, pending_config_ids)
        if display_names:
            self.names.update(
                {str(key): str(value) for key, value in display_names.items()}
            )
        self._heartbeat_interval = max(
            10, int(os.environ.get("MLLABIOME_HEARTBEAT_SECONDS", "60"))
        )
        self._last_heartbeat = time.monotonic()
        self._last_poll = self._last_heartbeat
        self._last_progress = self._last_heartbeat
        self._poll_interval = 15.0
        self.config_ids = sorted(
            set(map(str, pending_config_ids)),
            key=lambda item: (self.names.get(item, item), item),
        )
        self.expected_outer = set(map(str, outer_keys))
        self.expected_inner = set(map(str, inner_keys))
        self.all_config_ids = set(map(str, all_config_ids))
        self.completed_pairs = set(completed_pairs)
        self.total_pairs = len(self.all_config_ids) * len(self.expected_outer)
        self._reported_completed = len(self.completed_pairs)
        self.total_inner = len(self.config_ids) * len(self.expected_inner)
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
        self._reported_inner = self._inner_count()

    def _inner_count(self) -> int:
        return sum(len(rows) for rows in self.inner.values())

    def _progress_label(self) -> str:
        inner = (
            f"inner folds {self._inner_count()}/{self.total_inner} · "
            if self.total_inner
            else ""
        )
        return f"{inner}outer pairs {len(self.completed_pairs)}/{self.total_pairs}"

    def _store(
        self, section: str, row: Mapping[str, object], *, announce: bool
    ) -> bool:
        cid = str(row.get("config_id", ""))
        if cid not in self.inner:
            return False
        key_field = "inner_key" if section == "inner" else "split_key"
        key = str(row.get(key_field, ""))
        if not key or key not in (
            self.expected_inner if section == "inner" else self.expected_outer
        ):
            return False
        bucket = self.inner[cid] if section == "inner" else self.outer[cid]
        previous = bucket.get(key)
        value = dict(row)
        if previous == value:
            return False
        bucket[key] = value
        return True

    def _valid(
        self, rows: Mapping[str, Mapping[str, object]]
    ) -> list[dict[str, object]]:
        return [
            dict(row)
            for row in rows.values()
            if str(row.get("ok", 0)) in {"1", "1.0", "True"}
        ]

    def _score(self, rows: Sequence[Mapping[str, object]], metric: str) -> str:
        import pandas as pd
        from .metrics import aggregate_validation_metric

        score, _ = aggregate_validation_metric(pd.DataFrame(rows), metric)
        return f"{score:.4f}" if math.isfinite(score) else "—"

    def _counts(
        self, rows: Mapping[str, Mapping[str, object]], expected: set[str]
    ) -> str:
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
            "Evaluation",
            total=max(1, self.total_pairs),
            completed=min(len(self.completed_pairs), self.total_pairs),
        )
        return Group(
            Spinner(
                "dots",
                text="Running evaluation · waiting for completed fold checkpoints",
                style="cyan",
            ),
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
            else:
                current_inner = self._inner_count()
                current_outer = len(self.completed_pairs)
                if (
                    current_outer != self._reported_completed
                    or current_inner != self._reported_inner
                ):
                    self._reported_completed = current_outer
                    self._reported_inner = current_inner
                    self._last_progress = time.monotonic()
                    info(f"Evaluation progress · {self._progress_label()}")

    def _poll(self) -> None:
        if not self.config_ids:
            return
        from . import cooperative_store

        if cooperative_store.enabled(self.root):
            events = []

            with self._lock:
                known = {cid: set(self.inner.get(cid, {})) for cid in self.config_ids}
            for cid in self.config_ids:
                directory = self.root / "checkpoints" / cid / "inner"
                for file in directory.glob("*.json.zlib"):
                    try:
                        payload = cooperative_store._read_checkpoint(file, cid, "inner")
                    except (OSError, ValueError):
                        continue
                    for item in payload.get("inner_metrics", []):
                        if str(item.get("inner_key", "")) in known[cid]:
                            continue
                        with self._lock:
                            if self._store("inner", item, announce=True):
                                events.append(
                                    (
                                        "inner",
                                        cid,
                                        item,
                                        len(self._valid(self.inner[cid])),
                                    )
                                )
            self._update(events)
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
                    missing = [
                        (cid, key)
                        for cid, key in keys
                        if key not in self.inner.get(cid, {})
                    ]
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
                                events.append(
                                    (
                                        "inner",
                                        cid,
                                        item,
                                        len(self._valid(self.inner[cid])),
                                    )
                                )
                self._update(events)
        except (sqlite3.Error, OSError, ValueError, zlib.error):
            return

    def _watch(self) -> None:
        from . import cooperative_store

        cooperative = cooperative_store.enabled(self.root)
        while not self._stop.wait(2.0):
            now = time.monotonic()
            if console.is_terminal:
                self._poll()
            elif cooperative and now - self._last_poll >= self._poll_interval:
                self._last_poll = now
                self._poll()
            if (
                not console.is_terminal
                and now - self._last_heartbeat >= self._heartbeat_interval
            ):
                self._last_heartbeat = now
                with self._lock:
                    label = self._progress_label()
                    elapsed = int(now - self._last_progress)
                info(f"Evaluation heartbeat · {label} · no new fold for {elapsed}s")

    def __enter__(self):
        if console.is_terminal:
            console.print(
                "[dim]Fold metrics are checkpointed as completed; outer results remain held out.[/dim]"
            )
        if console.is_terminal:
            self._live = Live(
                self._render(),
                console=console,
                auto_refresh=True,
                refresh_per_second=2,
                transient=True,
                vertical_overflow="visible",
            )
            self._live.__enter__()
        else:
            info(
                f"Evaluation batch mode · {len(self.config_ids)} configurations · {self._progress_label()}"
            )
        self._thread = Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def result(self, result: Mapping[str, object]) -> None:
        if console.is_terminal:
            self._poll()
        events = []
        with self._lock:
            previous = len(self.completed_pairs)
            for section in ("inner", "outer"):
                for row in result.get(f"{section}_metrics", []):
                    if self._store(section, row, announce=True):
                        cid = str(row["config_id"])
                        bucket = (
                            self.inner[cid] if section == "inner" else self.outer[cid]
                        )
                        events.append((section, cid, row, len(self._valid(bucket))))
                    if section == "outer" and str(row.get("ok", 0)) in {
                        "1",
                        "1.0",
                        "True",
                    }:
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

    def finalized(
        self,
        config_id: str,
        ranked: Sequence[tuple[str, float, int]] | None = None,
    ) -> None:
        if console.is_terminal:
            self._poll()
        with self._lock:
            self.finished.add(config_id)
            if self._live is not None:
                self._live.update(self._render(), refresh=True)

    def unfinished(self, config_id: str) -> None:
        warn(
            f"Configuration remains resumable · {self.names.get(config_id, config_id)} · incomplete or failed folds"
        )

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if console.is_terminal:
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
    if not console.is_terminal:
        info(
            f"Configuration ranking · top {min(limit, len(ranked))} of {len(ranked)} · inner {metric} ({direction})"
        )
        for cid, score, _ in displayed:
            summary = _evaluation_summary(root, cid, metric_names)
            outer_score = summary["outer_metrics"][0].get(metric, float("nan"))
            info(
                f"rank {ranks[cid]}/{len(ranked)} · {names.get(cid, cid)} · "
                f"inner {metric}={_evaluation_score(score)} · "
                f"outer {metric}={_evaluation_score(outer_score)}"
            )
        info(
            "Ranks use inner cross-validation only; outer scores are held-out evaluation"
        )
        if position is not None:
            cid, score, _ = ranked[position - 1]
            info(
                f"Latest completed · rank {position}/{len(ranked)} · "
                f"{names.get(cid, cid)} · inner {metric}={score:.5f}"
            )
        return
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
