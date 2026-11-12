from __future__ import annotations

import hashlib
import json
import os
import tempfile
from zipfile import BadZipFile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def checkpoint_path(
    out_dir: Path, fold_no: int, member_no: int, split_key: str, config_id: str
) -> Path:
    identity = json.dumps(
        [str(split_key), str(config_id), int(member_no)], separators=(",", ":")
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return (
        out_dir
        / "checkpoints"
        / "hierarchical_shap"
        / f"fold_{fold_no:03d}_member_{member_no:03d}_{suffix}.npz"
    )


def checkpoint_signature(
    member: dict[str, Any],
    dataset: Any,
    fold: dict[str, Any],
    rows_ex: np.ndarray,
    rows_bg: np.ndarray,
    spec: Any,
    seed: int,
) -> str:
    train_idx = np.asarray(fold["train_idx"], dtype=int)
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    estimator = member["estimator"]
    parameters = (
        estimator.get_params(deep=True)
        if callable(getattr(estimator, "get_params", None))
        else repr(estimator)
    )
    dependencies = {}
    for name in ("shap", "scikit-learn"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    payload = {
        "schema": 1,
        "split_key": str(fold["split_key"]),
        "config_id": str(member["config_id"]),
        "transformation": str(member["transformation_key"]),
        "learner": str(member["learner_key"]),
        "coordinates": list(member["feature_names_fold"]),
        "class_labels": [str(value) for value in dataset.class_labels],
        "positive_class": dataset.positive_class,
        "train_samples": [str(dataset.sample_ids[int(i)]) for i in train_idx],
        "test_samples": [str(dataset.sample_ids[int(i)]) for i in test_idx],
        "rows_ex": np.asarray(rows_ex, dtype=int).tolist(),
        "rows_bg": np.asarray(rows_bg, dtype=int).tolist(),
        "spec": asdict(spec),
        "seed": int(seed),
        "parameters": parameters,
        "geometry": str(member.get("perturbation_geometry")),
        "projected": member.get("input_projector") is not None,
        "dependencies": dependencies,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    for array in (
        np.asarray(member["X_train"], dtype=np.float64),
        np.asarray(member["X_test"], dtype=np.float64),
        np.asarray(dataset.y[train_idx], dtype=np.int64),
        np.asarray(member["proba"], dtype=np.float64),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_checkpoint(
    path: Path, signature: str, n_samples: int, n_features: int, n_classes: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["signature"].item()) != signature:
                return None
            values = np.asarray(archive["values"], dtype=float)
            base = np.asarray(archive["base"], dtype=float)
        if values.ndim != 3 or values.shape[:2] != (n_samples, n_features):
            return None
        if values.shape[2] not in ({1, n_classes} if n_classes == 2 else {n_classes}):
            return None
        if base.shape != (n_samples, values.shape[2]):
            return None
        if not np.isfinite(values).all() or not np.isfinite(base).all():
            return None
        return values, base
    except (OSError, ValueError, KeyError, EOFError, TypeError, BadZipFile):
        return None


def save_checkpoint(
    path: Path, signature: str, values: np.ndarray, base: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            np.savez_compressed(
                handle,
                signature=np.asarray(signature),
                values=np.asarray(values, dtype=float),
                base=np.asarray(base, dtype=float),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
