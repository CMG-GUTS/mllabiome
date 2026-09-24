from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return np.asarray(matrix, dtype=float)
    matrix = np.where(np.isfinite(matrix) & (matrix >= 0), matrix, 0.0)
    totals = matrix.sum(axis=1)
    positive_totals = totals[totals > 0]
    median_total = float(np.median(positive_totals)) if positive_totals.size else 0.0
    maximum = float(np.max(matrix)) if matrix.size else 0.0
    if maximum <= 1.0 + 1e-6 and 0.85 <= median_total <= 1.15:
        return np.clip(matrix, 0.0, 1.0)
    if maximum <= 100.0 + 1e-6 and 85.0 <= median_total <= 115.0:
        return np.clip(matrix / 100.0, 0.0, 1.0)
    totals = totals.reshape(-1, 1)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def _datasets(dataset: Any) -> list[tuple[str, Any]]:
    if hasattr(dataset, "X_by_level") and hasattr(dataset, "feature_names_by_level"):
        return [("", dataset)]
    modalities = getattr(dataset, "modalities", None)
    if isinstance(modalities, dict):
        out: list[tuple[str, Any]] = []
        for name, matrix in modalities.items():
            child = getattr(matrix, "dataset", None)
            if child is not None and hasattr(child, "X_by_level"):
                out.append((str(name), child))
        return out
    return []


def _feature_candidates(feature: str) -> list[tuple[str, str]]:
    value = str(feature)
    candidates: list[tuple[str, str]] = [("", value)]
    if "::" in value:
        modality, tail = value.split("::", 1)
        candidates.insert(0, (modality, tail))
    if "|" in value:
        tail = value.rsplit("|", 1)[-1]
        candidates.append(("", tail))
        if "::" in tail:
            modality, name = tail.split("::", 1)
            candidates.insert(0, (modality, name))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _feature_source(dataset: Any, feature: str) -> tuple[Any, str, str, int] | None:
    sources = _datasets(dataset)
    for modality_hint, candidate in _feature_candidates(feature):
        ordered = sources
        if modality_hint:
            ordered = sorted(sources, key=lambda item: item[0] != modality_hint)
        for modality, child in ordered:
            if modality_hint and modality != modality_hint:
                continue
            by_level = getattr(child, "feature_names_by_level", {})
            for level, names in by_level.items():
                if level == "all":
                    continue
                index = {str(name): i for i, name in enumerate(names)}
                if candidate in index:
                    return child, str(level), candidate, int(index[candidate])
            names = [str(name) for name in by_level.get("all", [])]
            if candidate in names:
                return child, "all", candidate, int(names.index(candidate))
    return None


def build_local_relative_abundance_context(
    dataset: Any,
    features: Sequence[str],
) -> pd.DataFrame:
    unique_features = list(dict.fromkeys(str(feature) for feature in features))
    if not unique_features:
        return pd.DataFrame()
    sample_ids = [str(x) for x in getattr(dataset, "sample_ids", [])]
    y = np.asarray(getattr(dataset, "y", []))
    class_labels = [str(x) for x in getattr(dataset, "class_labels", [])]
    task = str(getattr(dataset, "task", "classification")).strip().lower()
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, str], np.ndarray] = {}
    for feature in unique_features:
        source = _feature_source(dataset, feature)
        if source is None:
            continue
        child, level, source_feature, index = source
        key = (id(child), level)
        if key not in cache:
            matrix = np.asarray(child.X_by_level[level], dtype=float)
            cache[key] = _normalise_rows(matrix)
        rel = cache[key]
        child_ids = [str(x) for x in getattr(child, "sample_ids", sample_ids)]
        if rel.ndim != 2 or index >= rel.shape[1] or len(child_ids) != rel.shape[0]:
            continue
        child_y = np.asarray(getattr(child, "y", y))
        child_labels = [str(x) for x in getattr(child, "class_labels", class_labels)]
        child_task = str(getattr(child, "task", task)).strip().lower()
        for sample_no, sample_id in enumerate(child_ids):
            class_index = np.nan
            class_label = ""
            if child_task == "classification" and sample_no < len(child_y):
                try:
                    class_index = int(child_y[sample_no])
                except (TypeError, ValueError):
                    class_index = np.nan
                if np.isfinite(class_index) and 0 <= int(class_index) < len(
                    child_labels
                ):
                    class_label = child_labels[int(class_index)]
            rows.append(
                {
                    "feature": feature,
                    "source_feature": source_feature,
                    "sample_id": sample_id,
                    "relative_abundance": float(rel[sample_no, index]),
                    "class_index": class_index,
                    "class_label": class_label,
                }
            )
    return pd.DataFrame(rows)


def _coordinate_components(
    feature: str, metadata: Any
) -> tuple[list[str], np.ndarray, str]:
    if metadata is not None:
        if isinstance(metadata, dict):
            anchor = str(metadata.get("anchor_feature", "")).strip()
            components_raw = metadata.get("components", ())
            coefficients_raw = metadata.get("coefficients", ())
            if isinstance(components_raw, str):
                try:
                    components_raw = json.loads(components_raw)
                except Exception:
                    components_raw = ()
            if isinstance(coefficients_raw, str):
                try:
                    coefficients_raw = json.loads(coefficients_raw)
                except Exception:
                    coefficients_raw = ()
        else:
            anchor = (
                ""
                if getattr(metadata, "anchor_feature", None) is None
                else str(metadata.anchor_feature).strip()
            )
            components_raw = getattr(metadata, "components", ())
            coefficients_raw = getattr(metadata, "coefficients", ())
        if anchor:
            return [anchor], np.asarray([1.0], dtype=float), "anchor_feature"
        components = [str(value) for value in list(components_raw or ()) if str(value)]
        if components:
            coefficients = np.asarray(list(coefficients_raw or ()), dtype=float)
            if coefficients.shape != (len(components),) or not np.all(
                np.isfinite(coefficients)
            ):
                coefficients = np.ones(len(components), dtype=float)
            weights = np.abs(coefficients)
            if not np.any(weights > 0):
                weights = np.ones(len(components), dtype=float)
            return components, weights, "component_weighted"
    tail = str(feature).rsplit("|", 1)[-1]
    if tail.startswith("ALR["):
        closing = tail.find("]")
        body = tail[4:closing] if closing > 4 else ""
        if "/" in body:
            numerator, reference = (part.strip() for part in body.split("/", 1))
            if numerator and reference:
                return (
                    [numerator, reference],
                    np.asarray([1.0, 1.0], dtype=float),
                    "component_weighted",
                )
    return [str(feature)], np.asarray([1.0], dtype=float), "feature"


def build_feature_relative_abundance_summary(
    dataset: Any,
    features: Sequence[str],
    coordinate_metadata: Sequence[Any] | None = None,
) -> pd.DataFrame:
    unique_features = list(dict.fromkeys(str(feature) for feature in features))
    if not unique_features:
        return pd.DataFrame()
    metadata_lookup: dict[str, Any] = {}
    for item in coordinate_metadata or ():
        if isinstance(item, dict):
            name = str(item.get("name", item.get("coordinate", ""))).strip()
        else:
            name = str(getattr(item, "name", "")).strip()
        if name:
            metadata_lookup[name] = item
    specs: dict[str, tuple[list[str], np.ndarray, str]] = {}
    requested_sources: list[str] = []
    for feature in unique_features:
        components, weights, basis = _coordinate_components(
            feature, metadata_lookup.get(feature)
        )
        specs[feature] = (components, weights, basis)
        requested_sources.extend(components)
    requested_sources = list(dict.fromkeys(requested_sources))
    context = build_local_relative_abundance_context(dataset, requested_sources)
    if context.empty:
        return pd.DataFrame(
            {
                "feature": unique_features,
                "mean_relative_abundance": np.nan,
                "abundance_basis": "unavailable",
            }
        )
    context["relative_abundance"] = pd.to_numeric(
        context["relative_abundance"], errors="coerce"
    )
    context = context.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["relative_abundance"]
    )
    means = context.groupby("feature", sort=False)["relative_abundance"].mean()
    rows: list[dict[str, Any]] = []
    for feature in unique_features:
        components, weights, basis = specs[feature]
        values: list[float] = []
        kept_weights: list[float] = []
        for component, weight in zip(components, weights):
            if component not in means.index:
                continue
            value = float(means.loc[component])
            if not np.isfinite(value) or value < 0:
                continue
            values.append(value)
            kept_weights.append(float(weight))
        if values:
            weight_array = np.asarray(kept_weights, dtype=float)
            if not np.any(np.isfinite(weight_array) & (weight_array > 0)):
                weight_array = np.ones(len(values), dtype=float)
            weight_array = np.where(
                np.isfinite(weight_array) & (weight_array > 0), weight_array, 0.0
            )
            abundance = float(
                np.average(np.asarray(values, dtype=float), weights=weight_array)
            )
        else:
            abundance = np.nan
        rows.append(
            {
                "feature": feature,
                "mean_relative_abundance": abundance,
                "abundance_basis": basis if values else "unavailable",
            }
        )
    return pd.DataFrame(rows)
