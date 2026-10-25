from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from collections.abc import Iterator

from filelock import FileLock
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cache import atomic_path


class RepresentationStore:
    def __init__(
        self, directory: Path | str, name: str, columns: tuple[str, ...]
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.name = name
        self.columns = columns
        self.path = self.directory / f"{name}.parquet"
        self.manifest = self.directory / f"{name}.manifest.json"
        self.pending = self.directory / ".in_progress" / name

    @contextmanager
    def locked(self) -> Iterator[None]:
        lock = self.directory / ".locks" / f"{self.name}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock), timeout=3600):
            yield

    def _digest(self) -> str:
        hasher = hashlib.sha256()
        with self.path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def existing(self) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        if not self.path.is_file():
            return {}, {}
        try:
            metadata = (
                json.loads(self.manifest.read_text(encoding="utf-8"))
                if self.manifest.is_file()
                else {}
            )
            current_sha = self._digest()
            if metadata.get("sha256") != current_sha:
                recovering = self.pending / "commit.json"
                candidate = (
                    json.loads(recovering.read_text(encoding="utf-8"))
                    if recovering.is_file()
                    else {}
                )
                if candidate.get("sha256") == current_sha:
                    metadata = candidate
                    with atomic_path(self.manifest) as temporary:
                        temporary.write_text(
                            json.dumps(metadata, sort_keys=True, indent=2),
                            encoding="utf-8",
                        )
                    shutil.rmtree(self.pending, ignore_errors=True)
            if metadata.get("schema") != "mllabiome.brain.representation.parquet.v1":
                return {}, {}
            if (
                metadata.get("columns") != list(self.columns)
                or metadata.get("sha256") != current_sha
            ):
                return {}, {}
            frame = pd.read_parquet(self.path)
            if tuple(frame.columns) != ("sample_id", *self.columns):
                return {}, {}
            ids = frame["sample_id"].astype(str)
            matrix = frame.loc[:, self.columns].to_numpy(dtype=np.float64)
            if ids.duplicated().any() or not np.isfinite(matrix).all():
                return {}, {}
            keys = metadata.get("subject_keys", {})
            if set(ids) != set(keys):
                return {}, {}
            return dict(zip(ids, matrix)), keys
        except (ValueError, KeyError, OSError, TypeError, ImportError):
            return {}, {}

    def staged(self, subject: str, key: str) -> np.ndarray | None:
        path = self.pending / f"{key}.npz"
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if (
                    str(archive["signature"].item()) != key
                    or str(archive["sample_id"].item()) != subject
                ):
                    return None
                vector = np.asarray(archive["vector"], dtype=np.float64)
            if vector.shape != (len(self.columns),) or not np.isfinite(vector).all():
                return None
            return vector
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def stage(self, subject: str, key: str, vector: np.ndarray) -> None:
        values = np.asarray(vector)
        if values.shape != (len(self.columns),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid representation values: {self.name}, {subject}")
        path = self.pending / f"{key}.npz"
        with atomic_path(path) as temporary:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    signature=np.asarray(key),
                    sample_id=np.asarray(subject),
                    vector=values,
                )

    def adopt(self, requested: Iterable[tuple[str, str]]) -> Path:
        pairs = tuple(requested)
        frame = pd.read_parquet(self.path)
        if tuple(frame.columns) != ("sample_id", *self.columns):
            raise ValueError(f"Invalid Parquet columns for {self.name}")
        subjects = frame["sample_id"].astype(str).tolist()
        if len(subjects) != len(set(subjects)) or set(subjects) != {
            sid for sid, _ in pairs
        }:
            raise ValueError(f"Invalid subject coverage for {self.name}")
        if not np.isfinite(frame[list(self.columns)].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite Parquet values for {self.name}")
        metadata = {
            "schema": "mllabiome.brain.representation.parquet.v1",
            "columns": list(self.columns),
            "sha256": self._digest(),
            "subject_keys": dict(pairs),
            "n_subjects": len(pairs),
        }
        with atomic_path(self.manifest) as temporary:
            temporary.write_text(
                json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
            )
        shutil.rmtree(self.pending, ignore_errors=True)
        return self.path

    def commit(
        self, requested: Iterable[tuple[str, str]], values: dict[str, np.ndarray]
    ) -> Path:
        pairs = tuple(requested)
        if len({sid for sid, _ in pairs}) != len(pairs):
            raise ValueError(f"Duplicate subject identifiers in {self.name}")
        if not pairs or set(values) != {sid for sid, _ in pairs}:
            raise ValueError(f"Incomplete representation: {self.name}")
        matrix = np.stack([np.asarray(values[sid]) for sid, _ in pairs])
        if (
            matrix.shape != (len(pairs), len(self.columns))
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(f"Non-finite or incompatible representation: {self.name}")
        frame = pd.DataFrame(matrix, columns=self.columns)
        frame.insert(0, "sample_id", [sid for sid, _ in pairs])
        with atomic_path(self.path) as temporary:
            frame.to_parquet(
                temporary, engine="pyarrow", compression="zstd", index=False
            )
            hasher = hashlib.sha256()
            with temporary.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            metadata = {
                "schema": "mllabiome.brain.representation.parquet.v1",
                "columns": list(self.columns),
                "sha256": hasher.hexdigest(),
                "subject_keys": dict(pairs),
                "n_subjects": len(pairs),
            }
            self.pending.mkdir(parents=True, exist_ok=True)
            with atomic_path(self.pending / "commit.json") as checkpoint:
                checkpoint.write_text(
                    json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
                )
        return self.adopt(pairs)
