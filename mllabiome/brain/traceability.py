from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BrainFeatureTrace:
    feature: str
    representation: str
    relationship: str
    region_indices: tuple[int, ...]
    region_names: tuple[str, ...]
    explanation: str

    def as_record(self) -> dict[str, object]:
        return {
            **asdict(self),
            "region_indices": json.dumps(self.region_indices),
            "region_names": json.dumps(self.region_names),
        }


@dataclass(frozen=True, slots=True)
class Schaefer400Provenance:
    labels: tuple[str, ...]
    atlas_path: Path | None = None

    def __post_init__(self) -> None:
        if (
            len(self.labels) != 400
            or len(set(self.labels)) != 400
            or any(not x for x in self.labels)
        ):
            raise ValueError(
                "Expected 400 unique nonempty atlas labels in verified Schaefer ROI order"
            )

    @classmethod
    def from_nilearn(
        cls, atlas_path: str | Path | None = None
    ) -> Schaefer400Provenance:
        from nilearn import datasets

        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=400, yeo_networks=7, resolution_mm=2
        )
        names = tuple(
            x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in atlas.labels
        )
        if len(names) == 401 and names[0].lower() in {"background", "background+"}:
            names = names[1:]
        return cls(
            names, Path(atlas_path).resolve() if atlas_path is not None else None
        )

    @classmethod
    def load(cls, path: str | Path) -> Schaefer400Provenance:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != "mllabiome.brain.schaefer400.v1":
            raise ValueError("Unsupported brain provenance schema")
        atlas = data.get("atlas_path")
        return cls(tuple(data["labels"]), Path(atlas) if atlas else None)

    def save(self, path: str | Path) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(
                {
                    "schema": "mllabiome.brain.schaefer400.v1",
                    "labels": self.labels,
                    "atlas_path": str(self.atlas_path) if self.atlas_path else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return dest

    def trace(self, feature: str) -> BrainFeatureTrace | None:
        name = str(feature)
        leaf = re.split(r"::|\|", name)[-1]
        if name != leaf and not any(
            part.lower().startswith("brain") for part in re.split(r"::|\|", name)[:-1]
        ):
            return None
        single = re.fullmatch(r"(alff|falff)_(\d{3})", leaf)
        pair = re.fullmatch(r"(fc|pfc)_(\d{3})_(\d{3})", leaf)
        embedding = re.fullmatch(r"bhf_(\d{3})", leaf)
        tissue = re.fullmatch(r"tissue_(gm|wm|csf)_mm3", leaf)
        if single:
            indices = (int(single.group(2)),)
            relationship = "exact_parcel"
            representation = single.group(1)
            explanation = "Feature extracted from the indicated Schaefer parcel"
        elif pair:
            a, b = int(pair.group(2)), int(pair.group(3))
            if a >= b:
                raise ValueError(
                    f"Malformed upper-triangle connectivity feature: {name}"
                )
            indices = (a, b)
            relationship = "exact_edge_endpoints"
            representation = "pearson_fc" if pair.group(1) == "fc" else "partial_fc"
            explanation = "Connection between two Schaefer parcels; not a regional importance score"
        elif embedding:
            if int(embedding.group(1)) >= 768:
                raise ValueError(
                    f"BrainHarmonix index outside the expected embedding width: {name}"
                )
            return BrainFeatureTrace(
                name,
                "brainharmonix_f",
                "requires_encoder_attribution",
                (),
                (),
                "Distributed latent coordinate; anatomical regions require encoder-aware sensitivity analysis",
            )
        elif tissue:
            return BrainFeatureTrace(
                name,
                "tissue_volumes",
                "global_tissue",
                (),
                (),
                "Whole-brain tissue volume; no justified single-ROI assignment",
            )
        elif re.fullmatch(r"PCA_\d+", leaf, re.IGNORECASE):
            return BrainFeatureTrace(
                name,
                "integrated_pca",
                "requires_fitted_transform",
                (),
                (),
                "PCA coordinate; use its fold-specific fitted component mapping and upstream attribution",
            )
        else:
            return None
        if any(i < 0 or i >= 400 for i in indices):
            raise ValueError(f"Schaefer feature index outside 0..399: {name}")
        return BrainFeatureTrace(
            name,
            representation,
            relationship,
            indices,
            tuple(self.labels[i] for i in indices),
            explanation,
        )


def annotate_brain_features(
    frame: pd.DataFrame,
    provenance: Schaefer400Provenance,
    *,
    columns: Sequence[str] = ("feature", "feature_1", "feature_2"),
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            continue
        mapping = {
            name: provenance.trace(str(name))
            for name in result[column].dropna().astype(str).unique()
        }
        result[f"{column}_brain_relationship"] = result[column].map(
            lambda value: (
                mapping.get(str(value)).relationship
                if mapping.get(str(value))
                else None
            )
        )
        result[f"{column}_brain_regions"] = result[column].map(
            lambda value: (
                json.dumps(mapping[str(value)].region_names)
                if mapping.get(str(value))
                else None
            )
        )
        result[f"{column}_brain_representation"] = result[column].map(
            lambda value: (
                mapping[str(value)].representation if mapping.get(str(value)) else None
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class RegionalSensitivity:
    sample_id: str
    region_index: int
    region_name: str
    baseline_value: float
    perturbed_value: float
    delta: float
    quantity: str
    intervention: str = "zero_standardized_roi_signal"


def _checked_observations(
    observations: Mapping[str, tuple[np.ndarray, float]], regions: Sequence[int]
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not observations:
        raise ValueError("At least one held-out observation is required")
    selected = tuple(int(i) for i in regions)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(i < 0 or i >= 400 for i in selected)
    ):
        raise ValueError("Region indices must be unique and in 0..399")
    for sample_id, (signals, tr) in observations.items():
        if (
            not sample_id
            or np.asarray(signals).ndim != 2
            or np.asarray(signals).shape[1] != 400
            or np.asarray(signals).shape[0] < 2
        ):
            raise ValueError(f"Invalid Schaefer time series for {sample_id}")
        if not np.isfinite(signals).all() or not np.isfinite(tr) or tr <= 0:
            raise ValueError(f"Invalid signal values or TR for {sample_id}")
    return tuple(observations), selected


def regional_embedding_sensitivity(
    encoder: object,
    observations: Mapping[str, tuple[np.ndarray, float]],
    provenance: Schaefer400Provenance,
    *,
    dimension: int,
    regions: Sequence[int],
) -> pd.DataFrame:
    if not 0 <= dimension < 768:
        raise ValueError("Embedding dimension must be in 0..767")
    _, selected = _checked_observations(observations, regions)
    rows = []
    for sample_id, (signals, tr) in observations.items():
        original = np.asarray(signals, dtype=np.float32)
        baseline_vector = np.asarray(encoder.encode(original, tr)[0], dtype=float)
        if baseline_vector.shape != (768,) or not np.isfinite(baseline_vector).all():
            raise ValueError(f"Invalid unperturbed embedding for {sample_id}")
        baseline = float(baseline_vector[dimension])
        for region in selected:
            ablated = original.copy()
            ablated[:, region] = 0.0
            perturbed_vector = np.asarray(encoder.encode(ablated, tr)[0], dtype=float)
            if (
                perturbed_vector.shape != (768,)
                or not np.isfinite(perturbed_vector).all()
            ):
                raise ValueError(f"Invalid perturbed embedding for {sample_id}")
            value = float(perturbed_vector[dimension])
            rows.append(
                asdict(
                    RegionalSensitivity(
                        sample_id,
                        region,
                        provenance.labels[region],
                        baseline,
                        value,
                        baseline - value,
                        f"bhf_{dimension:03d}",
                    )
                )
            )
    return pd.DataFrame.from_records(rows)


def regional_prediction_sensitivity(
    encoder: object,
    observations: Mapping[str, tuple[np.ndarray, float]],
    provenance: Schaefer400Provenance,
    *,
    predict: Callable[[str, np.ndarray], float],
    regions: Sequence[int],
) -> pd.DataFrame:
    _, selected = _checked_observations(observations, regions)
    rows = []
    for sample_id, (signals, tr) in observations.items():
        original = np.asarray(signals, dtype=np.float32)
        pooled = np.asarray(encoder.encode(original, tr)[0], dtype=float)
        if pooled.shape != (768,) or not np.isfinite(pooled).all():
            raise ValueError(f"Invalid pooled representation for {sample_id}")
        baseline = float(predict(sample_id, pooled))
        if not np.isfinite(baseline):
            raise ValueError(f"Non-finite unperturbed prediction for {sample_id}")
        for region in selected:
            ablated = original.copy()
            ablated[:, region] = 0.0
            perturbed = np.asarray(encoder.encode(ablated, tr)[0], dtype=float)
            if perturbed.shape != (768,) or not np.isfinite(perturbed).all():
                raise ValueError(f"Invalid perturbed representation for {sample_id}")
            value = float(predict(sample_id, perturbed))
            if not np.isfinite(value):
                raise ValueError(f"Non-finite perturbed prediction for {sample_id}")
            rows.append(
                asdict(
                    RegionalSensitivity(
                        sample_id,
                        region,
                        provenance.labels[region],
                        baseline,
                        value,
                        baseline - value,
                        "fitted_model_prediction",
                    )
                )
            )
    return pd.DataFrame.from_records(rows)


def regional_interaction_sensitivity(
    encoder: object,
    observations: Mapping[str, tuple[np.ndarray, float]],
    provenance: Schaefer400Provenance,
    *,
    predict: Callable[[str, np.ndarray, str], float],
    regions: Sequence[int],
) -> pd.DataFrame:
    _, selected = _checked_observations(observations, regions)
    rows = []
    for sample_id, (signals, tr) in observations.items():
        original = np.asarray(signals, dtype=np.float32)
        pooled = np.asarray(encoder.encode(original, tr)[0], dtype=float)
        if pooled.shape != (768,) or not np.isfinite(pooled).all():
            raise ValueError(f"Invalid pooled representation for {sample_id}")
        observed = float(predict(sample_id, pooled, "observed"))
        reference = float(predict(sample_id, pooled, "reference"))
        if not np.isfinite((observed, reference)).all():
            raise ValueError(f"Non-finite baseline prediction for {sample_id}")
        baseline_difference = observed - reference
        for region in selected:
            ablated = original.copy()
            ablated[:, region] = 0.0
            perturbed = np.asarray(encoder.encode(ablated, tr)[0], dtype=float)
            if perturbed.shape != (768,) or not np.isfinite(perturbed).all():
                raise ValueError(f"Invalid perturbed embedding for {sample_id}")
            observed_ablated = float(predict(sample_id, perturbed, "observed"))
            reference_ablated = float(predict(sample_id, perturbed, "reference"))
            if not np.isfinite((observed_ablated, reference_ablated)).all():
                raise ValueError(f"Non-finite perturbed prediction for {sample_id}")
            difference = observed_ablated - reference_ablated
            rows.append(
                asdict(
                    RegionalSensitivity(
                        sample_id,
                        region,
                        provenance.labels[region],
                        baseline_difference,
                        difference,
                        baseline_difference - difference,
                        "conditional_prediction_difference_in_differences",
                    )
                )
            )
    return pd.DataFrame.from_records(rows)


def summarize_regional_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sample_id",
        "region_index",
        "region_name",
        "delta",
        "quantity",
        "intervention",
    }
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Expected a nonempty regional sensitivity table")
    if frame.duplicated(["sample_id", "region_index", "quantity"]).any():
        raise ValueError("Duplicate subject-region-quantity measurements")
    if not np.isfinite(pd.to_numeric(frame["delta"], errors="coerce")).all():
        raise ValueError(
            "Regional sensitivity contains non-finite or non-numeric changes"
        )
    if frame["intervention"].nunique() != 1 or frame["quantity"].nunique() != 1:
        raise ValueError("Cannot combine different quantities or interventions")
    values = frame.assign(absolute_delta=frame["delta"].abs())
    summary = values.groupby(["region_index", "region_name"], as_index=False).agg(
        signed_mean_delta=("delta", "mean"),
        mean_absolute_delta=("absolute_delta", "mean"),
        n_subjects=("sample_id", "nunique"),
    )
    if summary["n_subjects"].nunique() != 1:
        raise ValueError(
            "Regions have unequal subject coverage; aggregation would be misleading"
        )
    return summary.sort_values("mean_absolute_delta", ascending=False).reset_index(
        drop=True
    )


def annotate_saved_explanations(sweep: object) -> dict[str, Path]:
    from ..storage import read_table, table_exists

    modalities = getattr(sweep, "modalities", ())
    identified = []
    for modality in modalities:
        if not str(modality.name).lower().startswith("brain"):
            continue
        path = Path(modality.path).parent / "brain_feature_atlas_labels.json"
        if path.is_file():
            identified.append(Schaefer400Provenance.load(path))
    if not identified:
        return {}
    if any(prov.labels != identified[0].labels for prov in identified[1:]):
        raise ValueError("Brain modalities use incompatible Schaefer ROI orders")
    provenance = identified[0]
    root = Path(sweep.root()) / "explainability"
    if not root.is_dir():
        return {}
    produced = {}
    for target_dir in root.iterdir():
        if not target_dir.is_dir():
            continue
        sources = [
            target_dir / filename
            for filename in (
                "feature_importance.parquet",
                "feature_stability.parquet",
                "ale_interactions.parquet",
                "fold_feature_importance.parquet",
                "feature_importance_by_outer_fold.parquet",
            )
        ]
        sources.extend(sorted(target_dir.glob("feature_interactions_*.parquet")))
        for source in sources:
            if not table_exists(source):
                continue
            frame = read_table(source)
            if not any(
                column in frame.columns
                for column in ("feature", "feature_1", "feature_2")
            ):
                continue
            annotated = annotate_brain_features(frame, provenance)
            output = target_dir / (source.stem + "_brain_provenance.tsv")
            annotated.to_csv(output, sep="\t", index=False)
            produced[f"{target_dir.name}_{source.stem}_brain_provenance"] = output
    return produced
