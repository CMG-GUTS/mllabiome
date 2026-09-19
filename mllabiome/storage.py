from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
import pandas as pd


def _canonical_path(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return path
    if path.suffix.lower() in {".tsv", ".csv", ".txt"}:
        return path.with_suffix(".parquet")
    return (
        path.with_suffix(path.suffix + ".parquet")
        if path.suffix
        else path.with_suffix(".parquet")
    )


def _legacy_paths(path: Path) -> tuple[Path, ...]:
    path = Path(path)
    stem = path.with_suffix("") if path.suffix else path
    candidates = [stem.with_suffix(".tsv"), stem.with_suffix(".csv")]
    if path.suffix.lower() in {".tsv", ".csv"}:
        candidates.insert(0, path)
    out = []
    for candidate in candidates:
        if candidate not in out:
            out.append(candidate)
    return tuple(out)


def parquet_engine_available() -> bool:
    return (
        importlib.util.find_spec("pyarrow") is not None
        or importlib.util.find_spec("fastparquet") is not None
    )


def require_parquet_engine() -> None:
    if parquet_engine_available():
        return
    raise RuntimeError(
        "Parquet storage requires pyarrow or fastparquet. Install mllabiome with its Parquet storage dependency before running experiments."
    )


def _configs_db(path: Path) -> Path | None:
    path = Path(path)
    if path.stem != "configs":
        return None
    db = path.parent / "configs.db"
    return db if db.exists() else None


def _read_configs_db(path: Path) -> pd.DataFrame:
    db = _configs_db(path)
    if db is None:
        return pd.DataFrame()
    conn = sqlite3.connect(db)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='configs'"
        ).fetchone()
        if not exists:
            return pd.DataFrame()
        return pd.read_sql_query(
            "SELECT * FROM configs ORDER BY active DESC, count_transformation, resolution, learner",
            conn,
        )
    finally:
        conn.close()


def read_table(path: Path, *, dtype=None) -> pd.DataFrame:
    path = Path(path)
    parquet = _canonical_path(path)
    if parquet.exists() and parquet.stat().st_size > 0:
        require_parquet_engine()
        frame = pd.read_parquet(parquet)
        return frame.astype(str) if dtype is str else frame
    configs = _read_configs_db(path)
    if not configs.empty:
        return configs.astype(str) if dtype is str else configs
    for legacy in _legacy_paths(path):
        if not legacy.exists() or legacy.stat().st_size == 0:
            continue
        try:
            return pd.read_csv(
                legacy,
                sep="\t" if legacy.suffix.lower() == ".tsv" else ",",
                dtype=dtype,
            )
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def write_table(path: Path, frame: pd.DataFrame) -> Path:
    require_parquet_engine()
    parquet = _canonical_path(Path(path))
    parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_name(parquet.stem + ".tmp.parquet")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(parquet)
    return parquet


def table_exists(path: Path) -> bool:
    path = Path(path)
    parquet = _canonical_path(path)
    if parquet.exists() and parquet.stat().st_size > 0:
        return True
    if _configs_db(path) is not None:
        return True
    return any(p.exists() and p.stat().st_size > 0 for p in _legacy_paths(path))


def remove_table(path: Path) -> None:
    path = Path(path)
    for candidate in (_canonical_path(path), *_legacy_paths(path)):
        if candidate.exists():
            candidate.unlink()


def resolve_table_path(path: Path) -> Path:
    path = Path(path)
    parquet = _canonical_path(path)
    if parquet.exists():
        return parquet
    db = _configs_db(path)
    if db is not None:
        return db
    for legacy in _legacy_paths(path):
        if legacy.exists():
            return legacy
    return parquet


def canonical_table_path(path: Path) -> Path:
    return _canonical_path(Path(path))


def glob_tables(directory: Path, stem_pattern: str) -> list[Path]:
    directory = Path(directory)
    chosen: dict[str, Path] = {}
    for suffix in (".parquet", ".tsv", ".csv"):
        for path in sorted(directory.glob(stem_pattern + suffix)):
            chosen.setdefault(path.stem, path)
    return [chosen[key] for key in sorted(chosen)]


def export_tsv_tree(root: Path, output_dir: Path | None = None) -> dict[str, object]:
    require_parquet_engine()
    root = Path(root)
    output_dir = (
        Path(output_dir) if output_dir is not None else root / "exports" / "tsv"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for source in sorted(root.rglob("*.parquet")):
        if output_dir == source or output_dir in source.parents:
            continue
        rel = source.relative_to(root).with_suffix(".tsv")
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_parquet(source)
        frame.to_csv(target, sep="\t", index=False)
        exported.append(
            {
                "source": str(source.relative_to(root)),
                "tsv": str(target.relative_to(root)),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "parquet_bytes": int(source.stat().st_size),
                "tsv_bytes": int(target.stat().st_size),
            }
        )
    db_path = root / "configs.db"
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='configs'"
            ).fetchone()
            if exists:
                frame = pd.read_sql_query(
                    "SELECT * FROM configs ORDER BY active DESC, count_transformation, resolution, learner",
                    conn,
                )
                target = output_dir / "configs.tsv"
                frame.to_csv(target, sep="\t", index=False)
                exported.append(
                    {
                        "source": "configs.db::configs",
                        "tsv": str(target.relative_to(root)),
                        "rows": int(len(frame)),
                        "columns": int(len(frame.columns)),
                        "parquet_bytes": None,
                        "tsv_bytes": int(target.stat().st_size),
                    }
                )
        finally:
            conn.close()
    manifest = output_dir / "export_manifest.json"
    manifest.write_text(
        json.dumps({"format": "tsv", "files": exported}, indent=2), encoding="utf-8"
    )
    return {"directory": output_dir, "manifest": manifest, "files": exported}
