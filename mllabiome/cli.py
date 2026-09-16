from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
from rich.traceback import install as install_rich_traceback
from .configs_sweep import Sweep, build_sweep_from_module, evaluate
from .console import error, stage
from .ensemble_sweep import sweep_ensemble
from .final_explainability import explain
from .final_models import build_final_models
from .pipeline import run_all
from .report import write_report


def _load_sweep(path: Path) -> Sweep:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    spec = importlib.util.spec_from_file_location("mllabiome_user_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return build_sweep_from_module(mod)


def main(argv: list[str] | None = None) -> None:
    install_rich_traceback(show_locals=False)
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="mllabiome",
        description="Run the configured MPMA sweep, ensemble sweep, and explainability stages.",
    )
    parser.add_argument("config", type=Path, help="Python sweep config.")
    parser.add_argument(
        "--stage",
        choices=["all", "evaluate", "ensemble", "explain", "report"],
        default="all",
        help="Entry point for full execution or stage-level restart.",
    )
    parser.add_argument(
        "--redo", action="store_true", help="Recompute completed evaluation outputs."
    )
    args = parser.parse_args(argv)
    try:
        sweep = _load_sweep(args.config)
    except Exception as exc:
        error(f"Could not load config: {exc}")
        parser.exit(2)
    stage("mllabiome", f"config={args.config} · stage={args.stage}")
    if args.redo:
        sweep.evaluation.redo = True
    if args.stage == "all":
        run_all(sweep)
    elif args.stage == "evaluate":
        evaluate(sweep)
    elif args.stage == "ensemble":
        sweep_ensemble(sweep)
        build_final_models(sweep.root())
    elif args.stage == "explain":
        explain(sweep)
    elif args.stage == "report":
        build_final_models(sweep.root())
        write_report(sweep)


if __name__ == "__main__":
    main()
