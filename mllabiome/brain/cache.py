from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator
from zipfile import BadZipFile

import numpy as np
import psutil
from filelock import FileLock


SCHEMA = "mllabiome.brain.cache.v2"


def signature(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def software(foundation: bool = False) -> dict[str, str]:
    names = ("nilearn", "nibabel", "numpy", "scipy")
    if foundation:
        names += ("brainmarks", "brainharmonix", "torch")
    releases = {}
    for package in names:
        try:
            releases[package] = version(package)
        except PackageNotFoundError:
            releases[package] = "not-installed"
    return releases


def file_identity(path: Path | str) -> tuple[str, int, int]:
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns


def input_identity(run: Any) -> dict[str, Any]:
    return {
        "subject": run.subject_id,
        "bold": file_identity(run.bold),
        "confounds": file_identity(run.confounds),
        "tr_source": file_identity(run.bold_json),
        "tr": run.repetition_time,
    }


def settings_identity(settings: Any) -> Any:
    return asdict(settings) if is_dataclass(settings) else settings


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=target.parent, prefix=".tmp_", suffix=target.suffix, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


class BrainCache:
    def __init__(self, root: Path | str, *, migrate_legacy: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.legacy_roots = (
            (self.root / "v2", Path.home() / ".cache" / "mllabiome" / "brain" / "v2")
            if migrate_legacy
            else ()
        )
        self.legacy_root = self.legacy_roots[0] if self.legacy_roots else None

    def key(self, *components: Any, foundation: bool = False) -> str:
        return signature((SCHEMA, software(foundation), components))

    def path(self, kind: str, key: str) -> Path:
        return self.root / kind / key[:2] / f"{key}.npz"

    def read(self, kind: str, key: str) -> dict[str, np.ndarray] | None:
        path = self.path(kind, key)
        stored = self._read_path(path, key)
        if stored is not None or path.exists():
            return stored
        for root in self.legacy_roots:
            legacy = root / kind / key[:2] / f"{key}.npz"
            if legacy == path:
                continue
            stored = self._read_path(legacy, key)
            if stored is not None:
                self.write(kind, key, **stored)
                return stored
        return None

    @staticmethod
    def _read_path(path: Path, key: str) -> dict[str, np.ndarray] | None:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["signature"].item()) != key:
                    return None
                return {
                    name: archive[name] for name in archive.files if name != "signature"
                }
        except (ValueError, OSError, KeyError, EOFError, TypeError, BadZipFile):
            return None

    @contextmanager
    def locked(self, kind: str, key: str) -> Iterator[None]:
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=3600):
            yield

    def write(self, kind: str, key: str, **arrays: Any) -> Path:
        path = self.path(kind, key)
        with atomic_path(path) as temporary:
            with temporary.open("wb") as stream:
                np.savez(stream, signature=np.asarray(key), **arrays)
        return path


def cache_root(
    configured: str | Path | None, output_dir: str | Path | None = None
) -> Path:
    if configured is not None:
        return Path(configured).expanduser().resolve()
    if output_dir is None:
        raise ValueError("Brain representation storage requires an output directory")
    return Path(output_dir).expanduser().resolve() / "prepared"


def worker_count(n_jobs: int | str, pending: int) -> int:
    if n_jobs == "auto":
        available = max(1, int(psutil.virtual_memory().available // (2 * 1024**3)))
        count = min(4, available, max(1, (os.cpu_count() or 2) // 2))
    elif isinstance(n_jobs, int) and not isinstance(n_jobs, bool) and n_jobs >= 1:
        count = n_jobs
    else:
        raise ValueError("Brain n_jobs must be a positive integer or 'auto'")
    return max(1, min(count, max(1, pending)))
