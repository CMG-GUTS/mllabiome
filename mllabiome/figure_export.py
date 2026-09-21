from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _converter():
    try:
        import cairosvg
    except Exception as exc:
        raise RuntimeError(
            "PNG/PDF figure export requires CairoSVG. Install cairosvg before using --export-png or --export-pdf."
        ) from exc
    return cairosvg


def export_svg_tree(
    root: Path,
    formats: Iterable[str],
    progress_callback=None,
) -> dict[str, object]:
    root = Path(root)
    requested = tuple(dict.fromkeys(str(x).strip().lower() for x in formats))
    invalid = [x for x in requested if x not in {"png", "pdf"}]
    if invalid:
        raise ValueError(f"Unsupported figure export format: {invalid[0]}")
    if not requested:
        return {"directories": {}, "files": []}
    cairosvg = _converter()
    export_root = root / "exports"
    sources = [
        path for path in sorted(root.rglob("*.svg")) if export_root not in path.parents
    ]
    total = len(sources) * len(requested)
    completed = 0
    exported: list[dict[str, object]] = []
    directories = {fmt: export_root / fmt for fmt in requested}
    for fmt, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        for source in sources:
            rel = source.relative_to(root).with_suffix(f".{fmt}")
            target = directory / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "png":
                cairosvg.svg2png(url=str(source), write_to=str(target), dpi=450)
            else:
                cairosvg.svg2pdf(url=str(source), write_to=str(target))
            exported.append(
                {
                    "source": str(source.relative_to(root)),
                    "format": fmt,
                    "output": str(target.relative_to(root)),
                    "svg_bytes": int(source.stat().st_size),
                    "output_bytes": int(target.stat().st_size),
                }
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, max(1, total), f"{fmt.upper()} · {rel}")
    return {
        "directories": {fmt: path for fmt, path in directories.items()},
        "files": exported,
    }
