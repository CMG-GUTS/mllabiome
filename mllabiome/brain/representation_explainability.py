from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SchaeferProvenance:
    labels: tuple[str, ...]
    atlas_path: Path


def _load_index(path: Path) -> tuple[dict[str, dict], dict[int, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mllabiome.brain.representation_provenance.v1":
        raise ValueError("Invalid brain representation provenance schema")
    index = {record["feature"]: record for record in payload["records"]}
    if len(index) != len(payload["records"]):
        raise ValueError("Duplicate brain features in provenance")
    atlas_paths = {
        int(key): Path(value) for key, value in payload["atlas_paths"].items()
    }
    return index, atlas_paths


def _lookup(value: str, index: Mapping[str, dict]) -> dict | None:
    raw = str(value)
    if raw in index:
        return index[raw]
    return index.get(raw.split("::")[-1])


def annotate_brain_representation_table(
    frame: pd.DataFrame, provenance: str | Path
) -> pd.DataFrame:
    index, _ = _load_index(Path(provenance))
    output = frame.copy()
    for feature_column in ("feature", "feature_1", "feature_2"):
        if feature_column not in output:
            continue
        matches = output[feature_column].map(lambda value: _lookup(value, index))
        output[f"{feature_column}_brain_representation"] = matches.map(
            lambda record: record["representation"] if record else None
        )
        output[f"{feature_column}_brain_relationship"] = matches.map(
            lambda record: record["relationship"] if record else None
        )
        output[f"{feature_column}_brain_regions"] = matches.map(
            lambda record: json.dumps(record["region_names"]) if record else None
        )
    return output


def regional_feature_scores(
    feature_scores: Mapping[str, float], provenance: str | Path
) -> dict[str, dict[int, float]]:
    index, _ = _load_index(Path(provenance))
    regional: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for feature, value in feature_scores.items():
        record = _lookup(feature, index)
        if record is None or not record["region_indices"]:
            continue
        score = float(value)
        if not np.isfinite(score) or score < 0:
            raise ValueError(
                "Anatomical importance mapping requires finite nonnegative magnitudes"
            )
        regions = record["region_indices"]
        for region in regions:
            regional[record["representation"]][int(region)] += score / len(regions)
    return {key: dict(values) for key, values in regional.items()}


def render_brain_feature_importance(
    feature_scores: Mapping[str, float],
    provenance: str | Path,
    output_dir: str | Path,
    *,
    format: str = "both",
) -> dict[str, tuple[Path, ...]]:
    from .schaefer_figures import render_schaefer_scores

    index, paths = _load_index(Path(provenance))
    regions = regional_feature_scores(feature_scores, provenance)
    output = Path(output_dir)
    results = {}
    for rep, values in regions.items():
        record = next(item for item in index.values() if item["representation"] == rep)
        n = record["parcels"]
        if n is None or n not in paths:
            continue
        from .representations import _atlas_labels

        atlas = SchaeferProvenance(_atlas_labels(n), paths[n])
        edge = record["method"] in {"pearson_fc", "partial_fc"}
        caption = (
            "Endpoint participation in important functional connections; edge magnitude shared equally"
            if edge
            else "Regional feature importance magnitude"
        )
        results[rep] = render_schaefer_scores(
            values,
            atlas,
            output / f"{rep}_importance",
            label=f"{caption} · {rep}",
            format=format,
        )
    return results


def annotate_saved_brain_representations(sweep: object) -> dict[str, Path]:
    from ..storage import read_table, table_exists

    brain = getattr(sweep, "brain", None)
    if brain is None:
        return {}
    source = (
        Path(sweep.root()) / "brain_representations" / "brain_feature_provenance.json"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Brain provenance missing: {source}")
    root = Path(sweep.root()) / "explainability"
    if not root.is_dir():
        return {}
    index, _ = _load_index(source)
    outputs = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.parquet"))
        for path in files:
            if (
                "importance" not in path.stem
                and "interaction" not in path.stem
                and "stability" not in path.stem
            ):
                continue
            if not table_exists(path):
                continue
            frame = read_table(path)
            if not any(
                field in frame.columns
                for field in ("feature", "feature_1", "feature_2")
            ):
                continue
            annotated = annotate_brain_representation_table(frame, source)
            dest = directory / f"{path.stem}_brain_provenance.tsv"
            annotated.to_csv(dest, sep="\t", index=False)
            outputs[f"{directory.name}_{path.stem}_brain_provenance"] = dest
            if (
                "feature" not in frame
                or "importance_mean" not in frame
                or "interaction" in path.stem
            ):
                continue
            importance = pd.to_numeric(frame["importance_mean"], errors="coerce")
            scores: dict[str, float] = defaultdict(float)
            for name, value in zip(frame["feature"], importance):
                if not np.isfinite(value):
                    continue
                record = _lookup(str(name), index)
                if record and record["region_indices"]:
                    scores[record["feature"]] += abs(float(value))
            if not scores:
                continue
            figures = render_brain_feature_importance(
                scores,
                source,
                directory / "figures" / "brain" / path.stem,
                format="png",
            )
            for key, paths in figures.items():
                outputs[f"{directory.name}_{path.stem}_{key}_brain_figure"] = paths[0]
    return outputs
