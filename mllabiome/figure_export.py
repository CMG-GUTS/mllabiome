from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _backend():
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PNG/PDF figure export requires PyMuPDF.") from exc
    return pymupdf


def _svg_to_pdf_bytes(path: Path) -> bytes:
    pymupdf = _backend()
    with pymupdf.open(filename=str(path), filetype="svg") as document:
        return document.convert_to_pdf()


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
    pymupdf = _backend()
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
                with pymupdf.open(filename=str(source), filetype="svg") as document:
                    document[0].get_pixmap(
                        matrix=pymupdf.Matrix(450.0 / 72.0, 450.0 / 72.0),
                        alpha=True,
                    ).save(str(target))
            else:
                target.write_bytes(_svg_to_pdf_bytes(source))
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
