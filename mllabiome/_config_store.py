from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SECTIONS = (
    "outer_metrics",
    "inner_metrics",
    "outer_predictions",
    "inner_predictions",
    "qualification",
    "job_resources",
)
PATH_SECTIONS = {
    ("results", "outer_results"): "outer_metrics",
    ("inner_results", "inner_results"): "inner_metrics",
    ("predictions", "outer_predictions"): "outer_predictions",
    ("inner_predictions", "inner_predictions"): "inner_predictions",
    ("tables", "qualification_gate"): "qualification",
    ("tables", "job_resources"): "job_resources",
}


def section_for_path(path: Path) -> str | None:
    path = Path(path)
    if path.suffix != ".parquet":
        return None
    return PATH_SECTIONS.get((path.parent.name, path.stem))


def _artifact_path(root: Path, config_id: str) -> Path:
    value = str(config_id)
    if not value or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError("Configuration identifier must be hexadecimal.")
    return root / "configs" / f"{value}.parquet"


def _connect(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "configs.db", timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS config_artifacts (
            config_id TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            split_digest TEXT NOT NULL,
            task TEXT NOT NULL,
            metric TEXT NOT NULL,
            outer_count INTEGER NOT NULL,
            expected_outer_count INTEGER NOT NULL,
            inner_count INTEGER NOT NULL,
            score REAL,
            configuration_json TEXT NOT NULL
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifact_ranking ON config_artifacts (benchmark, metric, score)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS config_fold_scores (
            config_id TEXT NOT NULL,
            split_key TEXT NOT NULL,
            metric TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (config_id, split_key, metric)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fold_scores ON config_fold_scores (metric, config_id, split_key)"
    )
    return conn


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(path).metadata or {}
    value = metadata.get(b"mllabiome.configuration")
    if value is None:
        raise ValueError(f"Configuration artifact has no identity metadata: {path}")
    result = json.loads(value.decode("utf-8"))
    if result.get("schema") != "mllabiome-config-v1":
        raise ValueError(f"Unsupported configuration artifact format: {path}")
    if _artifact_path(path.parent.parent, str(result["config_id"])) != path:
        raise ValueError(
            f"Configuration artifact filename disagrees with its identity: {path}"
        )
    return result


def write_configuration(
    root: Path,
    config_id: str,
    frames: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .storage import _coerce_mixed_text_columns

    root = Path(root)
    path = _artifact_path(root, config_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    columns = {}
    for section in SECTIONS:
        frame = frames.get(section, pd.DataFrame())
        if frame is None:
            frame = pd.DataFrame()
        columns[section] = list(frame.columns)
        compact = frame.drop(
            columns=[
                name
                for name in (
                    "config_id",
                    "learner",
                    "resolution",
                    "levels",
                    "count_transformation",
                    "mpdr_id",
                )
                if name in frame.columns
            ],
        ).copy()
        if not compact.empty:
            compact.insert(0, "__section", section)
            chunks.append(compact)
    if not chunks:
        raise ValueError("A completed configuration must contain evaluation records")
    joined = pd.concat(chunks, ignore_index=True, sort=False)
    joined = _coerce_mixed_text_columns(joined)
    table = pa.Table.from_pandas(joined, preserve_index=False)
    payload = {
        "schema": "mllabiome-config-v1",
        "config_id": str(config_id),
        "section_columns": columns,
        **dict(metadata),
    }
    schema = table.schema.with_metadata(
        {
            b"mllabiome.configuration": json.dumps(
                payload, sort_keys=True, default=str
            ).encode("utf-8")
        }
    )
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    try:
        with pq.ParquetWriter(
            temp, schema, compression="zstd", write_statistics=True
        ) as writer:
            offset = 0
            for chunk in chunks:
                fragment = table.slice(offset, len(chunk)).replace_schema_metadata(
                    schema.metadata
                )
                writer.write_table(fragment, row_group_size=131072)
                offset += len(chunk)
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        if pq.read_metadata(temp).num_rows != len(joined):
            raise ValueError("Incomplete configuration artifact")
        os.replace(temp, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temp.exists():
            temp.unlink()
    return path


def read_configuration(root: Path, config_id: str, section: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    if section not in SECTIONS:
        raise KeyError(section)
    path = _artifact_path(Path(root), config_id)
    if not path.exists():
        return pd.DataFrame()
    metadata = _file_metadata(path)
    table = pq.read_table(path, filters=[("__section", "=", section)])
    frame = table.to_pandas()
    if "__section" in frame.columns:
        frame = frame.drop(columns="__section")
    columns = metadata.get("section_columns", {}).get(section, [])
    configuration = metadata.get("configuration", {})
    restored = {"config_id": str(config_id)}
    for name in ("learner", "resolution", "levels", "count_transformation", "mpdr_id"):
        if name in configuration:
            restored[name] = configuration[name]
    for name, value in restored.items():
        if name in columns:
            frame[name] = value
    if columns:
        return frame.reindex(columns=columns)
    return pd.DataFrame()


@lru_cache(maxsize=12)
def _split_digest_cached(path: str, size: int, mtime_ns: int) -> str:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError("Evaluation split manifest is empty")
    names = sorted(frame.columns)
    normalized = frame.loc[:, names].astype("string").fillna("<missing>")
    normalized = normalized.sort_values(names, kind="mergesort")
    data = normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def split_digest(root: Path) -> str:
    path = Path(root) / "tables" / "cv_splits.parquet"
    if not path.exists():
        return ""
    stat = path.stat()
    return _split_digest_cached(str(path), stat.st_size, stat.st_mtime_ns)


def _benchmark_identity(root: Path) -> tuple[str, str]:
    path = root / "manifest.json"
    if not path.exists():
        return "", ""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return str(manifest.get("evaluation_cache_fingerprint") or ""), str(
        manifest.get("dataset_fingerprint") or ""
    )


def _register(
    conn: sqlite3.Connection, path: Path, meta: Mapping[str, Any], digest: str
) -> None:
    stat = path.stat()
    identifier = str(meta["config_id"])
    old = conn.execute(
        "SELECT sha256 FROM config_artifacts WHERE config_id=?", (identifier,)
    ).fetchone()
    if old is not None and old[0] != digest:
        raise ValueError(
            f"Configuration {identifier} differs from the indexed artifact. Resolve the conflict before continuing."
        )
    conf = json.dumps(meta.get("configuration", {}), sort_keys=True, default=str)
    conn.execute(
        """INSERT INTO config_artifacts
        (config_id, file_size, mtime_ns, sha256, benchmark, split_digest, task, metric, outer_count, expected_outer_count, inner_count, score, configuration_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(config_id) DO UPDATE SET
        file_size=excluded.file_size, mtime_ns=excluded.mtime_ns,
        benchmark=excluded.benchmark, split_digest=excluded.split_digest,
        task=excluded.task, metric=excluded.metric, outer_count=excluded.outer_count,
        expected_outer_count=excluded.expected_outer_count,
        inner_count=excluded.inner_count, score=excluded.score,
        configuration_json=excluded.configuration_json""",
        (
            identifier,
            stat.st_size,
            stat.st_mtime_ns,
            digest,
            str(meta.get("benchmark", "")),
            str(meta.get("split_digest", "")),
            str(meta.get("task", "")),
            str(meta.get("metric", "")),
            int(meta.get("outer_count", 0)),
            int(meta.get("expected_outer_count", meta.get("outer_count", 0))),
            int(meta.get("inner_count", 0)),
            meta.get("score"),
            conf,
        ),
    )
    metric = str(meta.get("metric", ""))
    outer = read_configuration(path.parent.parent, identifier, "outer_metrics")
    if metric and not outer.empty and metric in outer.columns:
        for row in outer.to_dict(orient="records"):
            if int(row.get("ok", 1)) != 1:
                continue
            score = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.notna(score) and float("-inf") < float(score) < float("inf"):
                conn.execute(
                    "INSERT OR REPLACE INTO config_fold_scores VALUES (?, ?, ?, ?)",
                    (identifier, str(row["split_key"]), metric, float(score)),
                )
    configuration = meta.get("configuration", {})
    if configuration:
        cols = {item[1] for item in conn.execute("PRAGMA table_info(configs)")}
        if cols:
            matching = {
                key: value
                for key, value in configuration.items()
                if key in cols and key != "active"
            }
            matching["config_id"] = identifier
            matching["active"] = 0
            if matching:
                names = list(matching)
                statement = f"INSERT OR IGNORE INTO configs ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})"
                conn.execute(statement, tuple(matching[name] for name in names))


def refresh_catalog(root: Path) -> tuple[int, int]:
    root = Path(root)
    folder = root / "configs"
    if not folder.exists():
        return 0, 0
    benchmark, _ = _benchmark_identity(root)
    expected_splits = split_digest(root)
    if not benchmark or not expected_splits:
        return 0, 0
    added = 0
    rejected = 0
    with _connect(root) as conn:
        known = {
            str(row[0]): (int(row[1]), int(row[2]))
            for row in conn.execute(
                "SELECT config_id, file_size, mtime_ns FROM config_artifacts"
            )
        }
        for path in sorted(folder.glob("*.parquet")):
            if not path.stem or any(
                ch not in "0123456789abcdef" for ch in path.stem.lower()
            ):
                continue
            stat = path.stat()
            if known.get(path.stem) == (stat.st_size, stat.st_mtime_ns):
                continue
            meta = _file_metadata(path)
            if benchmark and str(meta.get("benchmark", "")) != benchmark:
                rejected += 1
                continue
            if expected_splits and str(meta.get("split_digest", "")) != expected_splits:
                rejected += 1
                continue
            _register(conn, path, meta, _digest(path))
            added += 1
        missing = [key for key in known if not _artifact_path(root, key).exists()]
        if missing:
            conn.executemany(
                "DELETE FROM config_artifacts WHERE config_id=?",
                [(key,) for key in missing],
            )
    return added, rejected


def register_configuration(root: Path, config_id: str) -> None:
    path = _artifact_path(Path(root), config_id)
    meta = _file_metadata(path)
    benchmark, _ = _benchmark_identity(Path(root))
    expected_splits = split_digest(Path(root))
    if not benchmark or not expected_splits:
        raise ValueError(
            "Benchmark and exact split identities are required before registering artifacts"
        )
    if str(meta.get("benchmark", "")) != benchmark:
        raise ValueError("Configuration does not match this benchmark")
    if str(meta.get("split_digest", "")) != expected_splits:
        raise ValueError("Configuration does not match the exact evaluation splits")
    with _connect(Path(root)) as conn:
        conn.execute(
            "DELETE FROM config_artifacts WHERE config_id=?", (str(config_id),)
        )
        _register(conn, path, meta, _digest(path))


def indexed_config_ids(root: Path) -> list[str]:
    path = Path(root) / "configs.db"
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        present = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='config_artifacts'"
        ).fetchone()
        if present is None:
            return []
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT config_id FROM config_artifacts ORDER BY config_id"
            )
        ]


def read_configurations(
    root: Path, section: str, config_ids: Sequence[str] | None = None
) -> pd.DataFrame:
    identifiers = (
        list(config_ids) if config_ids is not None else indexed_config_ids(root)
    )
    frames = []
    for config_id in identifiers:
        frame = read_configuration(root, config_id, section)
        if not frame.empty:
            frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )


@lru_cache(maxsize=8192)
def _inner_ranking_score(
    root: str,
    config_id: str,
    metric: str,
    expected_inner: int,
    file_size: int,
    mtime_ns: int,
) -> float | None:
    import math

    from .metrics import aggregate_validation_metric

    frame = read_configuration(Path(root), config_id, "inner_metrics")
    if frame.empty or metric not in frame.columns or "inner_key" not in frame.columns:
        return None
    frame = frame.drop_duplicates("inner_key", keep="last")
    if "ok" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
    frame = frame.loc[pd.to_numeric(frame[metric], errors="coerce").map(math.isfinite)]
    if len(frame) != expected_inner:
        return None
    score, _ = aggregate_validation_metric(frame, metric)
    return float(score) if math.isfinite(score) else None


def top_configs(
    root: Path, metric: str, k: int = 3, loss: bool = True
) -> list[tuple[str, float, int]]:
    root = Path(root)
    path = root / "configs.db"
    if not path.exists() or k <= 0:
        return []
    with sqlite3.connect(path) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='config_artifacts'"
        ).fetchone():
            return []
        rows = conn.execute(
            "SELECT config_id, inner_count, outer_count FROM config_artifacts "
            "WHERE metric=? AND inner_count>0",
            (str(metric),),
        ).fetchall()
    ranked = []
    for config_id, expected_inner, outer_count in rows:
        artifact = _artifact_path(root, str(config_id))
        if not artifact.is_file():
            continue
        stat = artifact.stat()
        score = _inner_ranking_score(
            str(root), str(config_id), str(metric), int(expected_inner),
            stat.st_size, stat.st_mtime_ns,
        )
        if score is not None:
            ranked.append((str(config_id), score, int(outer_count)))
    ranked.sort(key=lambda item: (item[1] if loss else -item[1], item[0]))
    return ranked[:k]


def latest_ranked_config(root: Path, metric: str) -> str | None:
    ranked_ids = {
        config_id for config_id, _, _ in top_configs(root, metric, k=2147483647)
    }
    if not ranked_ids:
        return None
    with sqlite3.connect(Path(root) / "configs.db") as conn:
        rows = conn.execute(
            "SELECT config_id FROM config_artifacts WHERE metric=? "
            "ORDER BY mtime_ns DESC, config_id",
            (str(metric),),
        )
        return next((str(row[0]) for row in rows if str(row[0]) in ranked_ids), None)


def matched_outer_difference(
    root: Path, config_id: str, comparator_id: str, metric: str
) -> tuple[float, int] | None:
    path = Path(root) / "configs.db"
    if not path.exists() or config_id == comparator_id:
        return None
    with sqlite3.connect(path) as conn:
        found = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='config_fold_scores'"
        ).fetchone()
        if found is None:
            return None
        result = conn.execute(
            """SELECT AVG(current.score - reference.score), COUNT(*)
            FROM config_fold_scores AS current
            JOIN config_fold_scores AS reference
              ON current.split_key=reference.split_key AND current.metric=reference.metric
            WHERE current.config_id=? AND reference.config_id=? AND current.metric=?""",
            (str(config_id), str(comparator_id), str(metric)),
        ).fetchone()
        if not result or not result[1]:
            return None
        return float(result[0]), int(result[1])
