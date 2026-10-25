from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .fmriprep import discover_fmriprep_runs
from .source import BrainSource
from .timeseries import extract_schaefer400
from .traceability import (
    Schaefer400Provenance,
    regional_embedding_sensitivity,
    summarize_regional_sensitivity,
)


@dataclass(frozen=True, slots=True)
class BrainTraceResult:
    sensitivity_table: Path
    regional_summary: Path
    metadata: Path
    figures: tuple[Path, ...]


def trace_fmriprep_embedding_feature(
    source: BrainSource,
    feature: str,
    output_dir: str | Path,
    *,
    normalization_stats: str | Path,
    subject_ids: Sequence[str],
    regions: Sequence[int] = tuple(range(400)),
    device: str = "auto",
    render: bool = True,
) -> BrainTraceResult:
    if source.representation != "brainharmonix_f" or not source.roi_order_verified:
        raise ValueError(
            "Requires an explicitly ROI-order-verified BrainHarmonix-F source"
        )
    if source.atlas_path is None:
        raise ValueError("A spatially verified Schaefer-400 atlas image is required")
    match = re.fullmatch(
        r"(?:(?:[A-Za-z][A-Za-z0-9_]*::)?brain_brainharmonix_f__|(?:brain(?:harmonix)?(?:::|\|))?)bhf_(\d{3})",
        feature,
    )
    if match is None or int(match.group(1)) >= 768:
        raise ValueError(
            "Feature must identify a valid BrainHarmonix-F pooled coordinate"
        )
    selected = tuple(str(subject) for subject in subject_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(
            "Supply unique analysis subject IDs, explicitly chosen from the evaluation partition"
        )
    run_list = discover_fmriprep_runs(
        source.derivatives_dir, bids_root=source.bids_root
    )
    run_by_id = {run.subject_id: run for run in run_list}
    if not set(selected).issubset(run_by_id):
        raise ValueError(
            f"Requested subjects not present in fMRIPrep derivatives: {sorted(set(selected) - set(run_by_id))}"
        )
    from nilearn import image
    from .brainharmonix import BrainHarmonixFEncoder

    atlas = image.load_img(str(source.atlas_path))
    provenance = Schaefer400Provenance.from_nilearn(source.atlas_path)
    observations = {}
    for subject in selected:
        run = run_by_id[subject]
        time_series, _ = extract_schaefer400(
            run, preprocessing=source.preprocessing, atlas_img=atlas
        )
        observations[subject] = (time_series, run.repetition_time)
    encoder = BrainHarmonixFEncoder(device=device).load_normalization(
        normalization_stats
    )
    dimension = int(match.group(1))
    raw = regional_embedding_sensitivity(
        encoder, observations, provenance, dimension=dimension, regions=regions
    )
    regional = summarize_regional_sensitivity(raw)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"bhf_{dimension:03d}_schaefer_sensitivity"
    raw_path = output / f"{stem}_subjects.tsv"
    regional_path = output / f"{stem}_regions.tsv"
    metadata_path = output / f"{stem}.json"
    raw.to_csv(raw_path, sep="\t", index=False)
    regional.to_csv(regional_path, sep="\t", index=False)
    figures = ()
    if render:
        from .schaefer_figures import render_schaefer_scores

        figures = render_schaefer_scores(
            regional,
            provenance,
            output / stem,
            label=f"Mean absolute change in {stem.split('_schaefer')[0]} after Schaefer parcel signal replacement",
        )
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "mllabiome.brain.embedding_sensitivity.v1",
                "feature": f"bhf_{dimension:03d}",
                "source_derivatives": str(
                    Path(source.derivatives_dir).expanduser().resolve()
                ),
                "atlas_image": str(provenance.atlas_path),
                "normalization_stats": str(
                    Path(normalization_stats).expanduser().resolve()
                ),
                "subjects": list(selected),
                "regions": [int(i) for i in regions],
                "preprocessing": asdict(source.preprocessing),
                "intervention": "replace one standardized ROI time series with zeros; preserve all remaining ROIs, TR, and fitted normalization",
                "quantity": "change in pooled BrainHarmonix-F coordinate, not contribution to final learner prediction or interaction",
                "regional_aggregation": "mean across subjects of absolute within-subject coordinate changes; signed mean also reported",
                "figure_normalization": "within-figure division by the largest regional mean absolute change",
                "scientific_limitations": "ROI ablation can be out-of-distribution and is neither causal nor an exact decomposition of nonlinear embedding coordinates",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return BrainTraceResult(raw_path, regional_path, metadata_path, figures)


@dataclass(frozen=True, slots=True)
class BrainTraceabilityConfig:
    source: BrainSource
    normalization_stats: Path | str
    subject_ids: tuple[str, ...]
    regions: tuple[int, ...] = tuple(range(400))
    max_features_per_target: int = 2

    def __post_init__(self) -> None:
        if not self.subject_ids or len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError(
                "Brain tracing requires a nonempty unique evaluation-subject selection"
            )
        if self.max_features_per_target < 1:
            raise ValueError("max_features_per_target must be positive")
        if (
            not self.regions
            or len(set(self.regions)) != len(self.regions)
            or any(not 0 <= r < 400 for r in self.regions)
        ):
            raise ValueError("Schaefer regions must have unique indices in 0..399")


def trace_explained_brain_features(
    sweep: object, configuration: BrainTraceabilityConfig
) -> dict[str, Path]:
    from ..storage import read_table, table_exists

    result = {}
    base = Path(sweep.root()) / "explainability"
    if not base.is_dir():
        return result
    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue
        ranked = []
        for filename, metric in (
            ("feature_interactions_current.parquet", "interaction_strength"),
            ("ale_interactions.parquet", "interaction_strength_mean"),
            ("feature_stability.parquet", "importance_mean"),
            ("feature_importance.parquet", "importance_mean"),
        ):
            file = directory / filename
            if not table_exists(file):
                continue
            frame = read_table(file)
            if metric not in frame or frame.empty:
                continue
            values = frame.copy()
            values[metric] = pd.to_numeric(values[metric], errors="coerce")
            values = values.dropna(subset=[metric]).sort_values(metric, ascending=False)
            columns = [
                column
                for column in ("feature", "feature_1", "feature_2")
                if column in values
            ]
            for _, row in values.iterrows():
                ranked.extend(str(row[column]) for column in columns)
            if ranked:
                break
        selected = []
        for candidate in ranked:
            match = re.fullmatch(
                r"(?:(?:[A-Za-z][A-Za-z0-9_]*::)?brain_brainharmonix_f__|(?:brain(?:harmonix)?(?:::|\|))?)(bhf_\d{3})",
                candidate,
            )
            if (
                match
                and int(match.group(1).split("_")[1]) < 768
                and match.group(1) not in selected
            ):
                selected.append(match.group(1))
            if len(selected) >= configuration.max_features_per_target:
                break
        for feature in selected:
            export = trace_fmriprep_embedding_feature(
                configuration.source,
                feature,
                directory / "figures" / "brain_traceability",
                normalization_stats=configuration.normalization_stats,
                subject_ids=configuration.subject_ids,
                regions=configuration.regions,
            )
            prefix = f"{directory.name}_{feature}"
            result[f"{prefix}_regional_sensitivity"] = export.regional_summary
            result[f"{prefix}_subject_sensitivity"] = export.sensitivity_table
            result[f"{prefix}_trace_metadata"] = export.metadata
            result.update(
                {f"{prefix}_{path.suffix.lstrip('.')}": path for path in export.figures}
            )
    return result
