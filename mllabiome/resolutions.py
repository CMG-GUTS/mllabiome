from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .data import Dataset
from .utils import TAXONOMIC_LEVELS

_RAW_ALIASES = {"raw", "all", "features", "asis"}

_LEVEL_ALIASES = {"kingdom": "domain"}

FeatureBlocks = tuple[tuple[str, tuple[int, ...]], ...]

_RANK_PREFIXES = {
    "domain": ("d__", "k__"),
    "phylum": ("p__",),
    "class": ("c__",),
    "order": ("o__",),
    "family": ("f__",),
    "genus": ("g__",),
    "species": ("s__",),
    "strain": ("t__", "st__"),
}


def _feature_rank_from_name(name: Any) -> str | None:

    parts = [
        part.strip().lower()
        for part in re.split(r"[|;,\t]+", str(name))
        if part.strip()
    ]

    found: list[str] = []

    for part in parts:
        for level in TAXONOMIC_LEVELS:
            if any(part.startswith(prefix) for prefix in _RANK_PREFIXES[level]):
                found.append(level)

                break

    if not found:
        return None

    return max(found, key=TAXONOMIC_LEVELS.index)


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

    normalized = value.replace("→", "->")

    for sep in ("->", "-", "+", ","):
        if sep in normalized:
            parts = tuple(
                part.strip() for part in normalized.split(sep) if part.strip()
            )

            canonical = _canonical_levels(parts)

            if parts and all(
                part in TAXONOMIC_LEVELS or part in _LEVEL_ALIASES for part in parts
            ):
                if sep in {"->", "-"} and len(canonical) == 2:
                    return "→".join(canonical)

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

    if "→" in name:
        left, right = [x.strip() for x in name.split("→", 1)]

        if left in TAXONOMIC_LEVELS and right in TAXONOMIC_LEVELS:
            return name, _manual_range(left, right)

    if "+" in name:
        levels = _validate_explicit_levels(
            tuple(x.strip() for x in name.split("+") if x.strip())
        )

        return name, levels

    raise ValueError(f"Cannot parse taxonomic resolution {raw_name!r}.")


def _parse_resolution(item: Any) -> tuple[str, tuple[str, ...]]:

    if isinstance(item, tuple):
        name, levels = item

        name = _canonical_name(name)

        levels = _normalise_levels(levels)

    elif hasattr(item, "name") and hasattr(item, "levels"):
        name = _canonical_name(item.name)

        levels = _normalise_levels(item.levels)

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


def _raw_feature_blocks(dataset: Dataset, names: Sequence[str]) -> FeatureBlocks:

    membership: dict[str, set[str]] = {}

    for level in TAXONOMIC_LEVELS:
        for name in dataset.feature_names_by_level.get(level, ()):
            membership.setdefault(str(name), set()).add(level)

    by_level: dict[str, list[int]] = {level: [] for level in TAXONOMIC_LEVELS}

    unresolved: list[int] = []

    for index, name in enumerate(names):
        levels = membership.get(str(name), set())

        if len(levels) == 1:
            by_level[next(iter(levels))].append(index)

            continue

        inferred = _feature_rank_from_name(name)

        if inferred is not None:
            by_level[inferred].append(index)

        else:
            unresolved.append(index)

    blocks = [(level, tuple(indices)) for level, indices in by_level.items() if indices]

    if unresolved:
        blocks.append(("unresolved", tuple(unresolved)))

    return tuple(blocks) if blocks else (("all", tuple(range(len(names)))),)


def materialize_mpdr_with_blocks(
    dataset: Dataset, levels: Sequence[str]
) -> tuple[np.ndarray, list[str], FeatureBlocks]:

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
            matrix, names = _validated_block(dataset, "all")

            return matrix, names, _raw_feature_blocks(dataset, names)

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

    matrices = [matrix for matrix, _ in blocks]

    names = [name for _, block_names in blocks for name in block_names]

    offsets: list[tuple[str, tuple[int, ...]]] = []

    start = 0

    for level, (matrix, _) in zip(requested, blocks):
        stop = start + int(matrix.shape[1])

        offsets.append((level, tuple(range(start, stop))))

        start = stop

    if len(matrices) == 1:
        return matrices[0], names, tuple(offsets)

    return np.concatenate(matrices, axis=1), names, tuple(offsets)


def materialize_mpdr(
    dataset: Dataset, levels: Sequence[str]
) -> tuple[np.ndarray, list[str]]:

    matrix, names, _ = materialize_mpdr_with_blocks(dataset, levels)

    return matrix, names


def mask_feature_blocks(blocks: FeatureBlocks, mask: np.ndarray) -> FeatureBlocks:

    keep = np.asarray(mask, dtype=bool).reshape(-1)

    positions = np.full(keep.shape[0], -1, dtype=int)

    positions[np.flatnonzero(keep)] = np.arange(int(keep.sum()), dtype=int)

    out: list[tuple[str, tuple[int, ...]]] = []

    for level, indices in blocks:
        mapped = tuple(
            int(positions[index])
            for index in indices
            if 0 <= int(index) < len(positions) and positions[int(index)] >= 0
        )

        if mapped:
            out.append((str(level), mapped))

    return tuple(out)
