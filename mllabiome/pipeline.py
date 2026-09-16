from __future__ import annotations
from .configs_sweep import Sweep, evaluate
from .ensemble_sweep import sweep_ensemble
from .final_explainability import explain
from .final_models import build_final_models
from .report import write_report


def run_all(sweep: Sweep) -> dict:
    evaluate_output = evaluate(sweep)
    ensemble_output = sweep_ensemble(sweep)
    build_final_models(sweep.root())
    explain_output = explain(sweep)
    report_output = write_report(sweep)
    return {
        "evaluate": evaluate_output,
        "ensemble": ensemble_output,
        "explain": explain_output,
        "report": report_output,
        "final_models": sweep.root() / "final_models.json",
    }
