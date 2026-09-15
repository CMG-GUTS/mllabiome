from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .data import Dataset
from .utils import TAXONOMIC_LEVELS

_RAW_ALIASES = {"raw", "all", "features", "asis"}
_LEVEL_ALIASES = {"kingdom": "domain"}


def _canonical_level(level: str) -> str:
    value = str(level).strip()
    return _LEVEL_ALIASES.get(value, value)


def _canonical_levels(levels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_canonical_level(level) for level in levels)


def _canonical_name(name: str) -> str:
    value = str(name).strip()
    if value in _RAW_ALIASES:
        return "raw"
    if value in TAXONOMIC_LEVELS or value in _LEVEL_ALIASES:
        return _canonical_level(value)
    for sep in ("-", "+", ","):
        if sep in value:
            parts = tuple(part.strip() for part in value.split(sep) if part.strip())
            canonical = _canonical_levels(parts)
            if parts and all(
                part in TAXONOMIC_LEVELS or part in _LEVEL_ALIASES for part in parts
            ):
                return ("+" if sep == "," else sep).join(canonical)
    return value


def _manual_range(lo: str, hi: str) -> tuple[str, ...]:
    lo = _canonical_level(lo)
    hi = _canonical_level(hi)
    a, b = TAXONOMIC_LEVELS.index(lo), TAXONOMIC_LEVELS.index(hi)
    if a > b:
        a, b = b, a
    return TAXONOMIC_LEVELS[a : b + 1]


def _normalise_levels(levels: Any) -> tuple[str, ...]:
    if isinstance(levels, str):
        return (_canonical_level(levels),)
    return _canonical_levels(tuple(str(x).strip() for x in levels))


def _validate_explicit_levels(levels: tuple[str, ...]) -> tuple[str, ...]:
    if not levels:
        raise ValueError("A taxonomic resolution must contain at least one level.")
    raw = [x for x in levels if x in _RAW_ALIASES]
    if raw:
        if len(levels) != 1:
            raise ValueError(
                "The raw resolution cannot be combined with explicit taxonomic ranks."
            )
        return ("all",)
    unknown = [x for x in levels if x not in TAXONOMIC_LEVELS]
    if unknown:
        raise ValueError(f"Unknown taxonomic level(s): {unknown}.")
    if len(set(levels)) != len(levels):
        raise ValueError(f"Duplicate taxonomic levels are not allowed: {levels!r}.")
    return levels


def _resolution_from_name(name: str) -> tuple[str, tuple[str, ...]]:
    raw_name = str(name).strip()
    name = _canonical_name(raw_name)
    if name in TAXONOMIC_LEVELS:
        return name, (name,)
    if name in _RAW_ALIASES:
        return "raw", ("all",)
    if "-" in name:
        left, right = [x.strip() for x in name.split("-", 1)]
        if left in TAXONOMIC_LEVELS and right in TAXONOMIC_LEVELS:
            return name, _manual_range(left, right)
    if "+" in name:
        levels = _validate_explicit_levels(
            tuple(x.strip() for x in name.split("+") if x.strip())
        )
        return name, levels
    raise ValueError(f"Cannot parse taxonomic resolution {raw_name!r}.")


def _range_levels(lo: str, hi: str) -> tuple[str, ...]:
    lo = _canonical_level(lo)
    hi = _canonical_level(hi)
    a, b = TAXONOMIC_LEVELS.index(lo), TAXONOMIC_LEVELS.index(hi)
    if a > b:
        a, b = b, a
    return TAXONOMIC_LEVELS[a : b + 1]


def _parse_resolution(item: Any) -> tuple[str, tuple[str, ...]]:
    if isinstance(item, tuple):
        name, levels = item
        name = _canonical_name(name)
        levels = _normalise_levels(levels)
    elif hasattr(item, "name") and hasattr(item, "levels"):
        name = _canonical_name(getattr(item, "name"))
        levels = _normalise_levels(getattr(item, "levels"))
    else:
        return _resolution_from_name(str(item))
    name_is_raw = name in _RAW_ALIASES
    raw_levels = [x for x in levels if x in _RAW_ALIASES]
    if name_is_raw or raw_levels:
        if not name_is_raw or len(levels) != 1 or levels[0] not in _RAW_ALIASES:
            raise ValueError(
                "The raw resolution cannot be combined with explicit taxonomic ranks."
            )
        return "raw", ("all",)
    return name, _validate_explicit_levels(levels)


def _validated_block(dataset: Dataset, level: str) -> tuple[np.ndarray, list[str]]:
    if level not in dataset.X_by_level or level not in dataset.feature_names_by_level:
        raise ValueError(
            f"Requested taxonomic level {level!r} is not available in the dataset."
        )
    matrix = np.asarray(dataset.X_by_level[level])
    names = list(dataset.feature_names_by_level[level])
    if matrix.ndim != 2:
        raise ValueError(f"Feature matrix for level {level!r} must be two-dimensional.")
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"Feature-name count for level {level!r} does not match the feature matrix width."
        )
    return matrix, names


def materialize_mpdr(
    dataset: Dataset, levels: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    requested = _canonical_levels(tuple(str(x).strip() for x in levels))
    if not requested:
        requested = ("all",)
    raw = [x for x in requested if x in _RAW_ALIASES]
    if raw:
        if len(requested) != 1:
            raise ValueError(
                "The raw resolution cannot be combined with explicit taxonomic ranks."
            )
        if "all" in dataset.X_by_level:
            return _validated_block(dataset, "all")
        requested = tuple(
            level for level in TAXONOMIC_LEVELS if level in dataset.X_by_level
        )
        if not requested:
            raise ValueError("No feature matrix is available for the raw resolution.")
    else:
        requested = _validate_explicit_levels(requested)
    blocks = [_validated_block(dataset, level) for level in requested]
    sample_counts = {matrix.shape[0] for matrix, _ in blocks}
    if len(sample_counts) != 1:
        raise ValueError(
            f"Selected taxonomic ranks have inconsistent sample counts: {sorted(sample_counts)}."
        )
    if len(blocks) == 1:
        return blocks[0]
    matrices = [matrix for matrix, _ in blocks]
    names = [name for _, block_names in blocks for name in block_names]
    return np.concatenate(matrices, axis=1), names
