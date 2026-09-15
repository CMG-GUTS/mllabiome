from __future__ import annotations

from .configs_sweep import Sweep, evaluate
from .ensemble_sweep import sweep_ensemble
from .explainability import explain
from .report_oof import write_report


def run_all(sweep: Sweep) -> dict:
    outputs = {
        "evaluate": evaluate(sweep),
        "ensemble": sweep_ensemble(sweep),
        "explain": explain(sweep),
    }
    outputs["report"] = write_report(sweep)
    return outputs
