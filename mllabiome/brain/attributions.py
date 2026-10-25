from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BrainAttribution:
    kind: str
    hemisphere: str
    region: str
    score: float
    label: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"cortex", "subcortex"}:
            raise ValueError(f"Unsupported anatomical compartment: {self.kind!r}")
        if self.kind == "cortex" and self.hemisphere not in {"left", "right"}:
            raise ValueError("Cortical regions require a left or right hemisphere")
        if self.kind == "subcortex" and self.hemisphere:
            raise ValueError(
                "Subcortical laterality must be encoded in the atlas region name"
            )
        if not self.region or not self.region.strip():
            raise ValueError("Atlas region name cannot be empty")
        if self.label is not None and not self.label.strip():
            raise ValueError("Display label cannot be empty")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError(
                "Anatomical attribution scores must be finite and in [0, 1]"
            )

    @property
    def key(self) -> tuple[str, str, str]:
        return self.kind, self.hemisphere, self.region

    @property
    def display_label(self) -> str:
        return self.label if self.label is not None else self.region


def validate_attributions(
    attributions: Iterable[BrainAttribution],
) -> tuple[BrainAttribution, ...]:
    values = tuple(attributions)
    if not values:
        raise ValueError("At least one anatomical attribution is required")
    keys = [item.key for item in values]
    if len(keys) != len(set(keys)):
        raise ValueError("An anatomical ROI occurs more than once")
    return values


def read_brain_attributions(path: str | Path) -> tuple[BrainAttribution, ...]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"kind", "hemisphere", "region", "score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Attribution CSV requires columns: {sorted(required)}")
        values = []
        for row in reader:
            try:
                score = float(row["score"])
                kind = row["kind"].strip().lower()
                hemisphere = row["hemisphere"].strip().lower()
                region = row["region"].strip()
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"Invalid attribution row: {row!r}") from exc
            label = row.get("label")
            values.append(
                BrainAttribution(
                    kind=kind,
                    hemisphere=hemisphere,
                    region=region,
                    score=score,
                    label=label.strip() if label and label.strip() else None,
                )
            )
    return validate_attributions(values)


def available_brain_regions() -> dict[str, tuple[str, ...]]:
    try:
        from nilearn import datasets
    except ImportError as exc:
        raise ImportError("Atlas discovery requires nilearn") from exc
    cortex = datasets.fetch_atlas_surf_destrieux()
    subcortex = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    return {
        "cortex": tuple(_decode(label) for label in cortex.labels),
        "subcortex": tuple(_decode(label) for label in subcortex.labels),
    }


def _decode(value: str | bytes) -> str:
    return (
        value.decode("utf-8", errors="replace").strip()
        if isinstance(value, bytes)
        else str(value).strip()
    )


def example_brain_attributions() -> tuple[BrainAttribution, ...]:
    return (
        BrainAttribution(
            "cortex", "left", "G_front_sup", 0.38, "Superior frontal gyrus"
        ),
        BrainAttribution("cortex", "left", "G_precentral", 0.89, "Precentral gyrus"),
        BrainAttribution(
            "cortex", "left", "G_pariet_inf-Angular", 0.96, "Angular gyrus"
        ),
        BrainAttribution("cortex", "right", "G_precuneus", 0.61, "Precuneus"),
        BrainAttribution(
            "cortex", "right", "G_cingul-Post-dorsal", 0.80, "Posterior cingulate"
        ),
        BrainAttribution("subcortex", "", "Left Hippocampus", 0.84, "Left hippocampus"),
        BrainAttribution("subcortex", "", "Right Thalamus", 0.55, "Right thalamus"),
    )
