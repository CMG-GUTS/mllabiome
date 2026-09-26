from __future__ import annotations

import argparse
import importlib.util
import sys
import webbrowser
from pathlib import Path

from rich.traceback import install as install_rich_traceback

from .configs_sweep import Sweep, build_sweep_from_module
from .console import error, progress, stage, success, warn
from .figure_export import export_svg_tree
from .pipeline import run_stage
from .storage import export_tsv_tree
from .utils import report_html_path


def _load_sweep(path: Path) -> Sweep:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    spec = importlib.util.spec_from_file_location("mllabiome_user_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return build_sweep_from_module(mod)


def _open_report(root: Path) -> None:
    path = report_html_path(root).resolve()
    if not path.exists():
        warn(f"Report browser · report not found: {path}")
        return
    target = path.as_uri()
    if webbrowser.open(target, new=2, autoraise=True):
        success(f"Report opened · {target}")
    else:
        warn(f"Report browser · could not open automatically: {target}")


def main(argv: list[str] | None = None) -> None:
    install_rich_traceback(show_locals=False)
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="mllabiome",
        description="Run exploratory microbiome analysis, MPMA evaluation, ensembling, deployment inference, explainability, robustness, and reporting.",
    )
    parser.add_argument("config", type=Path, help="Python sweep config.")
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "explore",
            "evaluate",
            "ensemble",
            "inference",
            "explain",
            "robustness",
            "report",
        ],
        default="all",
        help="Entry point for full execution or stage-level restart.",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Run only the exploratory microbiome analysis stage.",
    )
    parser.add_argument(
        "--redo", action="store_true", help="Recompute completed evaluation outputs."
    )
    parser.add_argument(
        "--export-tsv",
        action="store_true",
        help="After report or explore, export canonical tables to mirrored TSV files under exports/tsv/.",
    )
    parser.add_argument(
        "--export-png",
        action="store_true",
        help="After report or explore, convert canonical SVG figures to mirrored PNG files under exports/png/.",
    )
    parser.add_argument(
        "--export-pdf",
        action="store_true",
        help="After report or explore, convert canonical SVG figures to mirrored PDF files under exports/pdf/.",
    )
    args = parser.parse_args(argv)
    if args.explore:
        if args.stage != "all":
            parser.error("Use either --explore or --stage, not both.")
        args.stage = "explore"
    if (args.export_tsv or args.export_png or args.export_pdf) and args.stage not in {
        "report",
        "explore",
    }:
        parser.error(
            "Export flags are only valid with --stage report, --stage explore, or --explore."
        )
    try:
        sweep = _load_sweep(args.config)
    except Exception as exc:
        error(f"Could not load config: {exc}")
        parser.exit(2)
    stage("mllabiome", f"config={args.config} · stage={args.stage}")
    if args.redo:
        sweep.evaluation.redo = True
    run_stage(sweep, args.stage)
    if args.export_tsv:
        with progress() as prog:
            task = prog.add_task("TSV export", total=1)

            def export_progress(completed: int, total: int, detail: str) -> None:
                prog.update(
                    task,
                    total=max(1, int(total)),
                    completed=int(completed),
                    description=f"TSV export · {detail}",
                )

            exported = export_tsv_tree(sweep.root(), progress_callback=export_progress)
            exported_count = max(1, len(exported["files"]))
            prog.update(
                task,
                total=exported_count,
                completed=exported_count,
                description="TSV export · complete",
            )
        stage("TSV export", str(exported["directory"]))
    figure_formats = [
        fmt
        for enabled, fmt in ((args.export_png, "png"), (args.export_pdf, "pdf"))
        if enabled
    ]
    if figure_formats:
        label = "+".join(fmt.upper() for fmt in figure_formats)
        with progress() as prog:
            task = prog.add_task(f"{label} export", total=1)

            def figure_progress(completed: int, total: int, detail: str) -> None:
                prog.update(
                    task,
                    total=max(1, int(total)),
                    completed=int(completed),
                    description=f"Figure export · {detail}",
                )

            figure_export = export_svg_tree(
                sweep.root(), figure_formats, progress_callback=figure_progress
            )
            exported_count = max(1, len(figure_export["files"]))
            prog.update(
                task,
                total=exported_count,
                completed=exported_count,
                description=f"{label} export · complete",
            )
        stage("Figure export", str(sweep.root() / "exports"))
    if args.stage in {"report", "all"}:
        _open_report(sweep.root())


if __name__ == "__main__":
    main()
