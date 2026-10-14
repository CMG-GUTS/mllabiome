from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Federated:
    num_rounds: int = 10
    local_epochs: int = 2
    num_shares: int = 3
    reconstruction_threshold: int = 2
    clipping_range: float = 8.0
    held_out_study: str | None = None
    random_state: int = 42
    flower_project: str | Path | None = None
    client_cpus: float = 1.0

    def __post_init__(self) -> None:
        if self.num_rounds < 1 or self.local_epochs < 1:
            raise ValueError("num_rounds and local_epochs must be positive.")
        if self.num_shares < 3 or not 2 <= self.reconstruction_threshold <= self.num_shares:
            raise ValueError("SecAgg+ requires num_shares >= 3 and 2 <= reconstruction_threshold <= num_shares.")
        if not 0.0 < float(self.clipping_range) < float("inf"):
            raise ValueError("clipping_range must be finite and positive.")
        if not 0.0 < float(self.client_cpus) < float("inf"):
            raise ValueError("client_cpus must be finite and positive.")
