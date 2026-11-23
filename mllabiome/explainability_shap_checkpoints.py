from __future__ import annotations
import hashlib
import inspect
import json
import os
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence
from zipfile import BadZipFile
import numpy as np


def member_shap_checkpoint_path(target_dir: Path, fold_no: int, split_key: str) -> Path:
    token = hashlib.sha256(str(split_key).encode("utf-8")).hexdigest()[:16]
    return (
        target_dir
        / "checkpoints"
        / "member_shap"
        / f"fold_{int(fold_no):03d}_{token}.npz"
    )


def member_shap_signature(
    fold: dict[str, Any],
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    spec: Any,
    seed: int,
    config_id: str,
    y: np.ndarray,
) -> str:
    estimator = fold["estimator"]
    parameters = (
        estimator.get_params(deep=True)
        if callable(getattr(estimator, "get_params", None))
        else repr(estimator)
    )
    dependencies: dict[str, str | None] = {}
    for package in ("shap", "scikit-learn", "numpy"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    payload = {
        "schema": 2,
        "config_id": str(config_id),
        "fold_key": str(fold.get("split_key", "")),
        "features": list(map(str, feature_names)),
        "classes": list(map(str, class_labels)),
        "spec": asdict(spec),
        "seed": int(seed),
        "estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "parameters": parameters,
        "projection": str(fold.get("perturbation_geometry", "")),
        "projected": fold.get("input_projector") is not None,
        "versions": dependencies,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    projector = fold.get("input_projector")
    if projector is not None:
        callable_object = getattr(projector, "__func__", projector)
        digest.update(
            f"{type(projector).__module__}.{type(projector).__qualname__}".encode(
                "utf-8"
            )
        )
        try:
            digest.update(inspect.getsource(callable_object).encode("utf-8"))
        except (OSError, TypeError):
            digest.update(str(getattr(callable_object, "__qualname__", "")).encode())
        reference = np.asarray(fold["X_train"], dtype=np.float64)
        if len(reference):
            probe = reference[: min(3, len(reference))].copy()
            if len(probe) > 1:
                probe = np.concatenate((probe, probe.mean(axis=0, keepdims=True)))
            projected = np.ascontiguousarray(
                np.asarray(projector(probe), dtype=np.float64)
            )
            digest.update(str(projected.shape).encode("ascii"))
            digest.update(projected.tobytes())
    for value in (
        fold.get("train_idx", []),
        fold.get("test_idx", []),
        fold["X_train"],
        fold["X_test"],
        fold.get("proba", []),
        y[np.asarray(fold.get("train_idx", []), dtype=int)],
        y[np.asarray(fold.get("test_idx", []), dtype=int)],
    ):
        arr = np.ascontiguousarray(np.asarray(value))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("ascii"))
        if arr.dtype.kind in "iufb":
            digest.update(arr.tobytes())
        else:
            digest.update(
                json.dumps(arr.tolist(), sort_keys=True, default=str).encode("utf-8")
            )
    return digest.hexdigest()


def load_member_shap_checkpoint(
    path: Path,
    signature: str,
    rows_ex: np.ndarray,
    rows_bg: np.ndarray,
    n_features: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"].item()) != signature:
                return None
            if not np.array_equal(data["rows_ex"], rows_ex):
                return None
            if not np.array_equal(data["rows_bg"], rows_bg):
                return None
            values = np.asarray(data["values"], dtype=np.float64)
            completed = np.asarray(data["completed"], dtype=bool)
            backend = str(data["backend"].item())
        if values.ndim != 3 or values.shape[:2] != (len(rows_ex), n_features):
            return None
        if values.shape[2] not in ({1, n_classes} if n_classes == 2 else {n_classes}):
            return None
        if completed.shape != (len(rows_ex),):
            return None
        if not np.isfinite(values[completed]).all():
            return None
        if not backend or not completed.any():
            return None
        return values, completed, backend
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return None


def save_member_shap_checkpoint(
    path: Path,
    signature: str,
    rows_ex: np.ndarray,
    rows_bg: np.ndarray,
    values: np.ndarray,
    completed: np.ndarray,
    backend: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            np.savez_compressed(
                handle,
                signature=np.asarray(signature),
                rows_ex=np.asarray(rows_ex, dtype=np.int64),
                rows_bg=np.asarray(rows_bg, dtype=np.int64),
                values=np.asarray(values, dtype=np.float64),
                completed=np.asarray(completed, dtype=bool),
                backend=np.asarray(str(backend)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
