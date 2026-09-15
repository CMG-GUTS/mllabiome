from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .data import Dataset
from .utils import TAXONOMIC_LEVELS


def _manual_range(lo: str, hi: str) -> tuple[str, ...]:
    a, b = TAXONOMIC_LEVELS.index(lo), TAXONOMIC_LEVELS.index(hi)
    if a > b:
        a, b = b, a
    return TAXONOMIC_LEVELS[a : b + 1]


def _resolution_from_name(name: str) -> tuple[str, tuple[str, ...]]:
    name = str(name).strip()
    if name in TAXONOMIC_LEVELS:
        return name, (name,)
    if "-" in name:
        left, right = [x.strip() for x in name.split("-", 1)]
        if left in TAXONOMIC_LEVELS and right in TAXONOMIC_LEVELS:
            return name, _manual_range(left, right)
    if "+" in name:
        levels = tuple(x.strip() for x in name.split("+") if x.strip())
        if levels and all(x in TAXONOMIC_LEVELS for x in levels):
            return name, levels
    if "," in name:
        levels = tuple(x.strip() for x in name.split(",") if x.strip())
        if levels and all(x in TAXONOMIC_LEVELS for x in levels):
            return "+".join(levels), levels
    if name in {"raw", "all", "features", "asis"}:
        return "raw", ("all",)
    raise ValueError(f"Cannot parse taxonomic resolution {name!r}.")


def _range_levels(lo: str, hi: str) -> tuple[str, ...]:
    a, b = TAXONOMIC_LEVELS.index(lo), TAXONOMIC_LEVELS.index(hi)
    if a > b:
        a, b = b, a
    return TAXONOMIC_LEVELS[a : b + 1]


def _parse_resolution(item: Any) -> tuple[str, tuple[str, ...]]:
    if isinstance(item, tuple):
        name, levels = item
        name = str(name).strip()
        levels = tuple(str(x).strip() for x in levels)
        if name in {"raw", "all", "features", "asis"} or any(
            x in {"raw", "all", "features", "asis"} for x in levels
        ):
            return "raw", ("all",)
        return name, levels
    if hasattr(item, "name") and hasattr(item, "levels"):
        name = str(getattr(item, "name")).strip()
        levels = tuple(str(x).strip() for x in getattr(item, "levels"))
        if name in {"raw", "all", "features", "asis"} or any(
            x in {"raw", "all", "features", "asis"} for x in levels
        ):
            return "raw", ("all",)
        return name, levels
    return _resolution_from_name(str(item))


def materialize_mpdr(
    dataset: Dataset, levels: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    requested = tuple(levels)
    if not requested or any(x in {"raw", "all", "features", "asis"} for x in requested):
        if "all" in dataset.X_by_level:
            return dataset.X_by_level["all"], dataset.feature_names_by_level["all"]
        requested = tuple(k for k in TAXONOMIC_LEVELS if k in dataset.X_by_level)
    matrices: list[np.ndarray] = []
    names: list[str] = []
    for lv in requested:
        if lv in dataset.X_by_level:
            matrices.append(dataset.X_by_level[lv])
            names.extend(dataset.feature_names_by_level[lv])
    if not matrices and "all" in dataset.X_by_level:
        return dataset.X_by_level["all"], dataset.feature_names_by_level["all"]
    if not matrices:
        raise ValueError(f"No features available for levels {requested!r}.")
    if len(matrices) == 1:
        return matrices[0], names
    return np.concatenate(matrices, axis=1), names
