#!/usr/bin/env python3
"""Migrate mllabiome experiments from commit 101db695's combined-results layout.

Requires the *new* mllabiome source on PYTHONPATH and pyarrow, pandas.
Example:
    python migrate_mllabiome_experiments.py /experiments/old_run /experiments/new_run --dry-run
    python migrate_mllabiome_experiments.py /experiments/old_run /experiments/new_run

The source experiment is never modified. Destination must not exist.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any
import zlib

SECTIONS = {
    "outer_metrics": "results/outer_results",
    "inner_metrics": "inner_results/inner_results",
    "outer_predictions": "predictions/outer_predictions",
    "inner_predictions": "inner_predictions/inner_predictions",
    "qualification": "tables/qualification_gate",
    "job_resources": "tables/job_resources",
}
DROP_KEYS = {
    "outer_metrics": ["config_id", "split_key"],
    "inner_metrics": ["config_id", "inner_key"],
    "outer_predictions": ["config_id", "split_key", "sample_id"],
    "inner_predictions": ["config_id", "split_key", "sample_id"],
    "qualification": ["config_id", "split_key"],
    "job_resources": ["config_id", "split_key"],
}
IDENTITY_COLUMNS = (
    "config_id", "learner", "resolution", "levels", "count_transformation", "mpdr_id"
)
HEX_ID = re.compile(r"[0-9a-fA-F]+\Z")


def _modules():
    try:
        import pandas as pd
        import pyarrow.parquet as pq
        from mllabiome._config_store import (
            write_configuration, read_configuration, register_configuration,
            indexed_config_ids, split_digest, refresh_catalog,
        )
        from mllabiome.metrics import canonical_metric_name
    except ImportError as exc:
        raise SystemExit(
            "Install the current mllabiome package and pyarrow in the migration environment "
            "(e.g. pip install pyarrow). Original error: " + str(exc)
        ) from exc
    return pd, pq, write_configuration, read_configuration, register_configuration, indexed_config_ids, split_digest, refresh_catalog, canonical_metric_name


def _source_table(root: Path, stem: str) -> Path | None:
    for suffix in (".parquet", ".tsv", ".csv"):
        path = root / (stem + suffix)
        if path.is_file():
            return path
    return None


def _observed_ids(path: Path, pq, pd) -> set[str]:
    ids: set[str] = set()
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        if "config_id" not in parquet.schema_arrow.names:
            raise ValueError(f"No config_id column in {path}")
        for batch in parquet.iter_batches(columns=["config_id"], batch_size=100_000):
            ids.update(str(x) for x in batch.column(0).to_pylist() if x is not None)
    else:
        for chunk in pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", usecols=["config_id"], dtype={"config_id": str}, chunksize=100_000):
            ids.update(chunk["config_id"].dropna().astype(str).tolist())
    return ids


def _read_one(path: Path | None, config_id: str, pd) -> Any:
    if path is None:
        return pd.DataFrame()
    if path.suffix == ".parquet":
        # Predicate pushdown keeps RAM bounded even for large old experiments.
        return pd.read_parquet(path, filters=[("config_id", "=", config_id)])
    chunks = []
    for chunk in pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", chunksize=100_000, dtype={"config_id": str}):
        subset = chunk.loc[chunk["config_id"].astype(str).eq(config_id)]
        if not subset.empty:
            chunks.append(subset)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _merge_tables(base, extra, key: str, pd):
    arrays = [f for f in (base, extra) if f is not None and not f.empty]
    if not arrays:
        return pd.DataFrame()
    merged = pd.concat(arrays, ignore_index=True, sort=False)
    keys = [k for k in DROP_KEYS[key] if k in merged.columns]
    return merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True) if keys else merged


def _check_identity(frame, config: dict[str, Any], section: str, pd):
    for col in IDENTITY_COLUMNS:
        if col not in frame or col not in config or pd.isna(config[col]):
            continue
        actual = frame[col].dropna().astype(str).unique().tolist()
        if len(actual) > 1 or (actual and actual[0] != str(config[col])):
            raise ValueError(f"{section}: inconsistent {col} for {config['config_id']}: {actual[:4]} != {config[col]}")


def _values(obj: Any):
    """Only for SQLite JSON checkpoint payloads. Parquet uses original data."""
    import numpy as np
    import pandas as pd
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, np.generic):
        return _values(obj.item())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_values(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _checkpoint_payloads(conn: sqlite3.Connection, config_id: str) -> dict[str, list[dict]]:
    bucket: dict[str, list[dict]] = {key: [] for key in SECTIONS}
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='evaluation_checkpoints'"
    ).fetchone()
    if not exists:
        return bucket
    for (blob,) in conn.execute(
        "SELECT payload FROM evaluation_checkpoints WHERE config_id=? ORDER BY split_key", (config_id,)
    ):
        payload = json.loads(zlib.decompress(blob).decode("utf-8"))
        for section in SECTIONS:
            bucket[section].extend(payload.get(section, []))
    return bucket


def _expected_splits(root: Path, pd) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]]]:
    path = root / "tables" / "cv_splits.parquet"
    if not path.is_file():
        raise ValueError("Missing tables/cv_splits.parquet; cannot prove fold identity")
    splits = pd.read_parquet(path)
    required = {"stage", "split_key", "inner_key", "role", "sample_id"}
    if splits.empty or not required.issubset(splits.columns):
        raise ValueError(f"Invalid split manifest: {path}")
    outer = set(splits.loc[(splits["stage"] == "outer") & (splits["role"] == "test"), "split_key"].dropna().astype(str))
    inner = set(splits.loc[(splits["stage"] == "inner") & (splits["role"] == "validation"), "inner_key"].dropna().astype(str))
    if not outer or any(str(k).rsplit("__i", 1)[0] not in outer for k in inner):
        raise ValueError("Split manifest has inconsistent outer/inner keys")
    outer_frame = splits.loc[(splits["stage"] == "outer") & (splits["role"] == "test")]
    inner_frame = splits.loc[(splits["stage"] == "inner") & (splits["role"] == "validation")]
    outer_samples = {str(k): set(group["sample_id"].astype(str)) for k, group in outer_frame.groupby("split_key")}
    inner_samples = {str(k): set(group["sample_id"].astype(str)) for k, group in inner_frame.groupby("inner_key")}
    return outer, inner, outer_samples, inner_samples


def _check_prediction_coverage(cid: str, frames: dict, inner_done: set[str], outer_done: set[str], outer_samples: dict, inner_samples: dict):
    """Prevent silently registering a completed fold lacking held-out samples."""
    for section, completed, expected in (
        ("outer_predictions", outer_done, outer_samples),
        ("inner_predictions", inner_done, inner_samples),
    ):
        frame = frames[section]
        if not completed:
            continue
        if frame.empty or not {"split_key", "sample_id"}.issubset(frame.columns):
            raise ValueError(f"{cid}: {section} absent for {len(completed)} completed folds")
        found = {str(k): set(group["sample_id"].astype(str)) for k, group in frame.groupby("split_key")}
        for key in sorted(completed):
            if found.get(key, set()) != expected.get(key, set()):
                missing = expected.get(key, set()) - found.get(key, set())
                extra = found.get(key, set()) - expected.get(key, set())
                raise ValueError(
                    f"{cid}: prediction coverage mismatch for {section}/{key}: "
                    f"missing {len(missing)}, extra {len(extra)}; manually reconcile before migration"
                )


def _outer_key(section: str, row: dict[str, Any]) -> str:
    # Predictions for inner folds use split_key=<outer key>__i<fold>.
    split = str(row.get("split_key", ""))
    if section in ("inner_metrics", "inner_predictions") and "__i" in split:
        split = split.rsplit("__i", 1)[0]
    if section == "inner_metrics" and not split:
        split = str(row.get("inner_key", "")).rsplit("__i", 1)[0]
    return split


def _write_pending_checkpoints(conn, cid: str, frames: dict, outer_keys: set[str]):
    buckets = defaultdict(lambda: {key: [] for key in SECTIONS})
    for section, frame in frames.items():
        for row in frame.to_dict(orient="records"):
            outer = _outer_key(section, row)
            if outer not in outer_keys:
                raise ValueError(f"Unrecognized outer split {outer!r} for {cid}/{section}")
            buckets[outer][section].append(_values(row))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS evaluation_checkpoints (
            config_id TEXT NOT NULL, split_key TEXT NOT NULL, payload BLOB NOT NULL,
            elapsed_s REAL, PRIMARY KEY(config_id, split_key))"""
    )
    conn.execute("DELETE FROM evaluation_checkpoints WHERE config_id=?", (cid,))
    for split_key, payload in buckets.items():
        content = zlib.compress(json.dumps(payload, allow_nan=False, default=str, separators=(",", ":")).encode("utf-8"), level=3)
        conn.execute(
            "INSERT INTO evaluation_checkpoints VALUES (?, ?, ?, ?)",
            (cid, split_key, sqlite3.Binary(content), 0.0),
        )
    return len(buckets)


def _roundtrip_verify(root, cid: str, frames: dict, pd, read_configuration):
    for section, before in frames.items():
        after = read_configuration(root, cid, section)
        if len(before) != len(after) or list(before.columns) != list(after.columns):
            raise ValueError(
                f"Roundtrip changed {cid}/{section}: {len(before)}->{len(after)} rows; "
                f"columns {list(before.columns)} -> {list(after.columns)}"
            )
        # Compare values as well as row counts; fail rather than silently altering results.
        if not before.empty:
            pd.testing.assert_frame_equal(
                before.reset_index(drop=True), after.reset_index(drop=True),
                check_dtype=False, check_categorical=False, check_exact=False,
                rtol=1e-11, atol=1e-13,
            )


def migrate(src: Path, dest: Path, *, dry_run: bool, strict: bool, verify: bool) -> dict:
    pd, pq, write_configuration, read_configuration, register_configuration, indexed_config_ids, split_digest, refresh_catalog, canonical_metric_name = _modules()
    src, dest = src.expanduser().resolve(), dest.expanduser().resolve()
    if not src.is_dir() or not (src / "manifest.json").is_file() or not (src / "configs.db").is_file():
        raise ValueError("Source must be an OLD experiment directory containing manifest.json and configs.db")
    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")
    if src in dest.parents or dest in src.parents:
        raise ValueError("Source and destination cannot be nested")
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    benchmark = manifest.get("evaluation_cache_fingerprint")
    dataset = manifest.get("dataset_fingerprint")
    if not benchmark or not dataset:
        raise ValueError("Missing evaluation_cache_fingerprint or dataset_fingerprint in legacy manifest")
    gate = bool((manifest.get("sweep", {}).get("gate") or {}).get("enabled", False))
    raw_metric = (manifest.get("sweep", {}).get("evaluation") or {}).get("optimize_metric")
    if not raw_metric:
        raise ValueError("No optimize_metric in manifest")
    metric = canonical_metric_name(str(raw_metric))
    outer_keys, inner_keys, outer_samples, inner_samples = _expected_splits(src, pd)
    old_paths = {name: _source_table(src, stem) for name, stem in SECTIONS.items()}
    with sqlite3.connect(src / "configs.db") as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='configs'").fetchone():
            raise ValueError("No configs table in source configs.db")
        configurations = {str(r["config_id"]): dict(r) for r in conn.execute("SELECT * FROM configs")}
        checkpoint_ids = set()
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evaluation_checkpoints'").fetchone():
            checkpoint_ids = {str(r[0]) for r in conn.execute("SELECT DISTINCT config_id FROM evaluation_checkpoints")}
    if not any(old_paths.values()) and not checkpoint_ids:
        raise ValueError("No old combined evaluation tables or evaluation checkpoints detected")
    observed = set(checkpoint_ids)
    for section, path in old_paths.items():
        if path is not None:
            ids = _observed_ids(path, pq, pd)
            print(f"Source {section:20} configs={len(ids):5}  {path.relative_to(src)}", flush=True)
            observed.update(ids)
    missing = observed - configurations.keys()
    if missing:
        raise ValueError(f"Evaluation rows reference missing configurations: {sorted(missing)[:10]}")
    for cid in observed:
        if not HEX_ID.fullmatch(cid):
            raise ValueError(f"Config identifier not hexadecimal: {cid!r}")
    print(f"Detected {len(configurations)} configurations; {len(observed)} with stored results; "
          f"{len(outer_keys)} outer folds; {len(inner_keys)} inner folds; metric={metric}; gate={gate}", flush=True)
    if not dry_run:
        # Snapshot SQLite through its backup API (captures committed WAL transactions).
        def skip_old_db(directory, children):
            return {c for c in children if c in {"configs.db", "configs.db-wal", "configs.db-shm"}}
        shutil.copytree(src, dest, ignore=skip_old_db)
        with sqlite3.connect(src / "configs.db") as original, sqlite3.connect(dest / "configs.db") as copy:
            original.backup(copy)
        for stem in (*SECTIONS.values(), "configs"):
            for ext in (".parquet", ".tsv", ".csv"):
                legacy = dest / (stem + ext)
                if legacy.exists():
                    legacy.unlink()
    completed, partial, skipped = [], [], []
    total_rows = defaultdict(int)
    try:
        db_root = src if dry_run else dest
        with sqlite3.connect(db_root / "configs.db", timeout=60) as conn:
            # Do not use source conn to write in dry run.
            if not dry_run:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=60000")
                conn.execute("PRAGMA synchronous=FULL")
            for count, cid in enumerate(sorted(observed), start=1):
                config = configurations[cid]
                oldcp = _checkpoint_payloads(conn, cid)
                frames = {}
                for section in SECTIONS:
                    base = _read_one(old_paths[section], cid, pd)
                    cp = pd.DataFrame(oldcp[section])
                    merged = _merge_tables(base, cp, section, pd)
                    if not merged.empty:
                        _check_identity(merged, config, section, pd)
                        if "config_id" in merged.columns and not merged["config_id"].astype(str).eq(cid).all():
                            raise ValueError(f"Bad config IDs in {section}/{cid}")
                    frames[section] = merged
                    total_rows[section] += len(merged)
                inner_df, outer_df, gate_df = frames["inner_metrics"], frames["outer_metrics"], frames["qualification"]
                def done(frame, key):
                    if frame.empty or key not in frame:
                        return set()
                    valid = frame
                    if "ok" in frame:
                        valid = frame[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
                    return set(valid[key].dropna().astype(str))
                inner_done = done(inner_df, "inner_key")
                outer_done = done(outer_df, "split_key")
                _check_prediction_coverage(cid, frames, inner_done, outer_done, outer_samples, inner_samples)
                excluded = set()
                if gate and not gate_df.empty and "qualified" in gate_df:
                    rejected = gate_df[pd.to_numeric(gate_df["qualified"], errors="coerce").eq(0)]
                    excluded = set(rejected["split_key"].dropna().astype(str))
                extras = (inner_done - inner_keys) | (outer_done - outer_keys) | (excluded - outer_keys)
                if extras:
                    raise ValueError(f"Unexpected split keys for {cid}: {sorted(extras)[:7]}")
                done_all = bool(inner_keys) and inner_keys.issubset(inner_done) and outer_keys.issubset(outer_done | excluded)
                if done_all:
                    completed.append(cid)
                    if not dry_run:
                        vals = pd.to_numeric(outer_df.get(metric, pd.Series(dtype=float)), errors="coerce")
                        valid = vals[vals.map(lambda x: pd.notna(x) and math.isfinite(float(x)))]
                        score = float(valid.mean()) if not valid.empty else None
                        meta = {
                            "benchmark": str(benchmark), "dataset": str(dataset),
                            "split_digest": split_digest(dest),
                            "task": str(manifest.get("task", "")),
                            "target": str(manifest.get("target", "")),
                            "metric": metric,
                            "outer_count": len(outer_done),
                            "expected_outer_count": len(outer_keys),
                            "inner_count": len(inner_keys),
                            "score": score,
                            "configuration": config,
                            "hardware": {"note": "Not re-measured; see original manifest"},
                            "software": manifest.get("software_provenance", {}),
                            "migration": {"source_commit": "101db6955be276114c2d40999ff36ae5e60b65ba", "schema": "legacy-combined-to-config-v1"},
                        }
                        write_configuration(dest, cid, frames, meta)
                        if verify:
                            _roundtrip_verify(dest, cid, frames, pd, read_configuration)
                        register_configuration(dest, cid)
                        # Legacy databases can have no evaluation_checkpoints table.
                        # A completed, migrated configuration does not need one;
                        # delete old checkpoint rows only when the table exists.
                        if conn.execute(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE type='table' AND name='evaluation_checkpoints'"
                        ).fetchone():
                            conn.execute(
                                "DELETE FROM evaluation_checkpoints WHERE config_id=?",
                                (cid,),
                            )
                elif any(not f.empty for f in frames.values()):
                    partial.append(cid)
                    if not dry_run:
                        _write_pending_checkpoints(conn, cid, frames, outer_keys)
                else:
                    skipped.append(cid)
                # Commit after each configuration. The new catalog writer uses
                # another SQLite connection and would otherwise hit a write lock.
                if not dry_run:
                    conn.commit()
                if count % 20 == 0 or count == len(observed):
                    print(f"Processed {count}/{len(observed)} configs (completed {len(completed)}, pending {len(partial)})", flush=True)
            if strict and partial:
                raise ValueError(f"Strict mode rejects {len(partial)} incomplete configurations")
            if not dry_run:
                conn.commit()
        report = {
            "source": str(src), "destination": str(dest),
            "source_commit": "101db6955be276114c2d40999ff36ae5e60b65ba",
            "schema": "mllabiome-config-v1", "dry_run": dry_run,
            "evaluation_cache_fingerprint": str(benchmark),
            "dataset_fingerprint": str(dataset), "metric": metric,
            "expected_outer_folds": len(outer_keys), "expected_inner_folds": len(inner_keys),
            "complete_configurations": completed,
            "resumable_partial_configurations": partial,
            "unpopulated_configurations": skipped,
            "effective_row_counts": dict(total_rows),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        if not dry_run:
            ids = set(indexed_config_ids(dest))
            if ids != set(completed):
                raise ValueError(f"Indexed configurations mismatch: expected={len(completed)} found={len(ids)}")
            with sqlite3.connect(dest / "configs.db") as conn:
                result = conn.execute("PRAGMA integrity_check").fetchone()[0]
                if result != "ok":
                    raise ValueError(f"SQLite integrity check failed: {result}")
            out = dest / "migration_report.json"
            out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        return report
    except Exception:
        if not dry_run:
            print(f"Migration failed; inspect unfinished destination: {dest}; source is untouched", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_experiment", type=Path, help="Path to old experiment output directory (not the source-code archive)")
    parser.add_argument("new_experiment", type=Path, help="New, non-existing destination directory")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without writing")
    parser.add_argument("--strict", action="store_true", help="Fail if any configuration is incomplete")
    parser.add_argument("--no-verify", action="store_true", help="Skip strict roundtrip value verification (not recommended)")
    args = parser.parse_args()
    report = migrate(args.old_experiment, args.new_experiment, dry_run=args.dry_run, strict=args.strict, verify=not args.no_verify)
    print(json.dumps({
        "completed": len(report["complete_configurations"]),
        "resumable_partial": len(report["resumable_partial_configurations"]),
        "new_location": report["destination"],
        "dry_run": report["dry_run"],
    }, indent=2))


if __name__ == "__main__":
    main()
