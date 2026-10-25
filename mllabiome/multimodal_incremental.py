from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .data import dataset_fingerprint
from .modalities import modality_dataset_fingerprint


def context_fingerprint(payload: dict[str, Any], digest: Callable[[Any], str]) -> str:
    return digest({"schema": "multimodal-incremental-context", **payload})


def candidate_fingerprints(
    specs: list[Any], matrices: dict[Any, np.ndarray], names: dict[Any, Any]
) -> dict[str, str]:
    inputs = {}
    for key, array in matrices.items():
        features = names[key][0]
        hasher = hashlib.sha256()
        hasher.update(json.dumps(key, sort_keys=True, default=str).encode("utf-8"))
        hasher.update(json.dumps(features, ensure_ascii=False).encode("utf-8"))
        matrix = np.ascontiguousarray(array, dtype=np.float32)
        hasher.update(str(matrix.shape).encode("ascii"))
        hasher.update(memoryview(matrix).cast("B"))
        inputs[key] = hasher.hexdigest()
    output = {}
    for spec in specs:
        pairs = [
            (path.modality, path.representation, path.selector)
            for path in spec.modality_paths
        ]
        hasher = hashlib.sha256()
        hasher.update(str(spec.config_id).encode("utf-8"))
        for key in pairs:
            hasher.update(inputs[key].encode("ascii"))
        output[str(spec.config_id)] = hasher.hexdigest()
    return output


def _previous_prefixes(root: Path, modality: str) -> tuple[str, ...]:
    database = root / "configs.db"
    if not database.is_file():
        return ()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT DISTINCT levels FROM configs").fetchall()
    prefixes = set()
    pattern = re.compile(rf"(?:^|;){re.escape(modality)}:prefix:([^;]+)")
    for (value,) in rows:
        for match in pattern.finditer(str(value)):
            prefixes.add(match.group(1))
    return tuple(sorted(prefixes))


def _original_modality_fingerprint(
    root: Path, dataset: Any, name: str, expected: str, size: int
) -> bool:
    matrix = dataset.modalities[name]
    if dataset_fingerprint(matrix.dataset) == expected:
        return True
    prefixes = _previous_prefixes(root, name)
    if not prefixes:
        return False
    columns = [
        i
        for i, feature in enumerate(matrix.feature_names)
        if any(feature.startswith(p) for p in prefixes)
    ]
    if len(columns) != size:
        return False
    reduced = replace(
        matrix.dataset,
        X_by_level={"all": np.asarray(matrix.X[:, columns], dtype=np.float32)},
        feature_names_by_level={"all": [matrix.feature_names[i] for i in columns]},
    )
    return dataset_fingerprint(reduced) == expected


def _validate_prior_experiment(
    root: Path,
    prior: dict[str, Any],
    dataset: Any,
    sources: dict[str, Any],
    evaluation_payload: dict[str, Any],
    digest: Callable[[Any], str],
) -> None:
    if prior.get("task") != dataset.task or prior.get("target") != dataset.target_name:
        raise ValueError(
            "Existing evaluation uses another task or target; choose a different experiment directory."
        )
    if (
        prior.get("n_samples") != len(dataset.sample_ids)
        or prior.get("primary_modality") != dataset.primary_modality
    ):
        raise ValueError(
            "Adding representations changed the evaluation cohort or primary modality; existing fits cannot be reused."
        )
    if prior.get("source_file_sha256", {}).get("samples") != sources["samples"]:
        raise ValueError(
            "Sample metadata changed; existing evaluation results cannot be reused safely."
        )
    previous = prior.get("modality_fingerprints", {})
    if not previous or not set(previous).issubset(dataset.modalities):
        raise ValueError(
            "Previous modalities are missing from the new configuration; cannot verify prior evaluation inputs."
        )
    for name, fingerprint in previous.items():
        size = prior.get("modalities", {}).get(name, {}).get("n_features")
        if not isinstance(size, int) or not _original_modality_fingerprint(
            root, dataset, name, fingerprint, size
        ):
            raise ValueError(
                f"Existing {name!r} features have changed or cannot be verified; use a separate experiment directory."
            )
    previous_dataset = modality_dataset_fingerprint(dataset, fingerprints=previous)
    if previous_dataset != prior.get("dataset_fingerprint"):
        raise ValueError(
            "Previously evaluated subject order or inputs differ from the current dataset."
        )
    payload = {
        **evaluation_payload,
        "dataset_fingerprint": previous_dataset,
        "modality_fingerprints": previous,
        "multimodal_inclusion": prior.get("multimodal_inclusion"),
    }
    if digest(payload) != prior.get("evaluation_fingerprint"):
        raise ValueError(
            "Evaluation splits, settings, or metadata differ from the previous experiment; use a separate directory."
        )


def validate_incremental(
    root: Path,
    evaluation_fingerprint: str,
    stable_context: str,
    inputs: dict[str, str],
    dataset: Any,
    sources: dict[str, Any],
    evaluation_payload: dict[str, Any],
    digest: Callable[[Any], str],
    redo: bool,
) -> dict[str, str]:
    path = root / "manifest.json"
    if redo or not path.exists():
        return dict(inputs)
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            "Previous evaluation manifest is unreadable; refusing to reuse unchecked results."
        ) from error
    previous_context = prior.get("evaluation_context_fingerprint")
    previous_inputs = prior.get("candidate_input_fingerprints")
    if previous_context is None or not isinstance(previous_inputs, dict):
        if prior.get("evaluation_fingerprint") != evaluation_fingerprint:
            _validate_prior_experiment(
                root, prior, dataset, sources, evaluation_payload, digest
            )
        return dict(inputs)
    if previous_context != stable_context:
        raise ValueError(
            "The benchmark cohort or scientific evaluation protocol changed; use a new experiment directory or --redo."
        )
    changed = [
        key
        for key in previous_inputs.keys() & inputs.keys()
        if previous_inputs[key] != inputs[key]
    ]
    if changed:
        raise ValueError(
            f"Previously evaluated configurations have changed input features: {changed[:5]}. Use a new experiment directory or --redo."
        )
    return {**previous_inputs, **inputs}
