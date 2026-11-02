from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from rich import box
from rich.live import Live
from rich.table import Table

from .console import console, info, summary_table


class JointEvaluationMonitor:
    def __init__(self, configurations, metric, n_splits, inner_fold_counts):
        self.configurations = [dict(item) for item in configurations]
        self.metric = str(metric)
        self.n_splits = int(n_splits)
        self.inner_fold_counts = tuple(int(value) for value in inner_fold_counts)
        self.expected = sum(self.inner_fold_counts)
        self.scores = defaultdict(list)
        self.selections = defaultdict(int)
        self.completed = 0
        self.live = None

    def _render(self):
        table = Table(title="Joint evaluation · configurations", box=box.SIMPLE_HEAD, pad_edge=False)
        table.add_column("Learner", style="bold")
        table.add_column("Resolution")
        table.add_column("Transformation")
        table.add_column("Inner folds", justify="right")
        table.add_column(f"Inner {self.metric}", justify="right")
        table.add_column("Selected", justify="right")
        table.add_column("Status")
        for item in self.configurations:
            identifier = item["config_id"]
            values = self.scores[identifier]
            mean = float(np.mean(values)) if values else math.nan
            status = "[green]Complete[/green]" if len(values) == self.expected else "[cyan]Evaluating[/cyan]" if values else "[dim]Queued[/dim]"
            table.add_row(
                str(item["learner"]),
                str(item["resolution"]),
                str(item["transformation"]),
                f"{len(values)}/{self.expected}",
                f"{mean:.4f}" if math.isfinite(mean) else "—",
                str(self.selections[identifier]),
                status,
            )
        return table

    def __enter__(self):
        summary_table("Joint nested cross-validation", [
            ("Configurations", len(self.configurations)),
            ("Outer splits", self.n_splits),
            ("Inner folds per split", ", ".join(map(str, sorted(set(self.inner_fold_counts))))),
            ("Selection metric", self.metric),
            ("Inner training fits", len(self.configurations) * self.expected),
            ("Outer refits", self.n_splits),
        ])
        if console.is_terminal:
            self.live = Live(self._render(), console=console, refresh_per_second=4, transient=False)
            self.live.__enter__()
        else:
            console.print(self._render())
        return self

    def record_inner(self, identifier, score):
        self.scores[str(identifier)].append(float(score))
        if self.live is not None:
            self.live.update(self._render(), refresh=True)

    def record_outer(self, identifier, split_key):
        self.selections[str(identifier)] += 1
        self.completed += 1
        if self.live is not None:
            self.live.update(self._render(), refresh=True)
        else:
            info(f"Joint evaluation · outer split {self.completed}/{self.n_splits} complete · {split_key}")

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.update(self._render(), refresh=True)
            self.live.__exit__(exc_type, exc, tb)
        elif exc is None:
            console.print(self._render())
        return False
