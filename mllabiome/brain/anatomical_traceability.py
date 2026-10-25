from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .brainharmonix import BrainHarmonixFEncoder
from .fmriprep import discover_fmriprep_runs
from .representations import _resolved_brain
from .timeseries import extract_schaefer400
from .traceability import Schaefer400Provenance


_PATTERN = re.compile(r"(?:^|__)bhf_(\d{3})$")


def _ranked_features(
    directory: Path, method: str, top_k: int
) -> list[tuple[str, int, float]]:
    from ..storage import read_table, table_exists

    sources = (
        directory / "top_features.parquet",
        directory / "feature_importance.parquet",
        directory / "feature_stability.parquet",
    )
    available = [path for path in sources if table_exists(path)]
    if not available:
        raise FileNotFoundError(
            f"No feature-ranking tables found for anatomical traceability in {directory}"
        )
    for path in available:
        frame = read_table(path)
        if frame.empty or "feature" not in frame.columns:
            continue
        frame = frame.copy()
        if path.name == "top_features.parquet":
            if not {"rank", method}.issubset(frame.columns):
                raise ValueError(f"Unsupported feature-ranking schema in {path}")
            frame["_score"] = pd.to_numeric(frame[method], errors="coerce")
            frame["_rank"] = pd.to_numeric(frame["rank"], errors="coerce")
            frame = frame.sort_values(
                ["_score", "_rank", "feature"],
                ascending=[False, True, True],
                na_position="last",
            )
        elif path.name == "feature_importance.parquet":
            if "mean_rank" not in frame.columns:
                raise ValueError(f"Regression feature ranking has no mean_rank: {path}")
            if method == "importance_mean":
                continue
            frame["_rank"] = pd.to_numeric(frame["mean_rank"], errors="coerce")
            frame["_score"] = 1.0 / frame["_rank"].where(frame["_rank"] > 0)
            frame = frame.sort_values(
                ["_rank", "feature"], ascending=[True, True], na_position="last"
            )
        else:
            if not {"method", "importance_mean"}.issubset(frame.columns):
                raise ValueError(
                    f"Regression feature stability has an unsupported schema: {path}"
                )
            methods = frame["method"].dropna().astype(str).unique()
            if method == "importance_mean" and len(methods) != 1:
                raise ValueError(
                    "importance_mean cannot be compared between distinct explainability methods; use rank_by='consensus'"
                )
            frame["_score"] = pd.to_numeric(frame["importance_mean"], errors="coerce")
            frame = frame.sort_values(
                ["_score", "feature"], ascending=[False, True], na_position="last"
            )
        matches = []
        seen = set()
        for _, row in frame.iterrows():
            match = _PATTERN.search(str(row["feature"]))
            score = row["_score"]
            if match is None or not np.isfinite(score) or int(match.group(1)) >= 768:
                continue
            feature = f"bhf_{int(match.group(1)):03d}"
            if feature not in seen:
                matches.append((feature, int(match.group(1)), float(score)))
                seen.add(feature)
            if len(matches) >= top_k:
                return matches
        if matches:
            return matches
    return []


def _select_subjects(
    frame: pd.DataFrame, runs: dict, spec, sample_id_map, samples, eligible_ids=None
) -> list[tuple[str, str]]:
    if "sample_id" not in frame.columns or frame["sample_id"].duplicated().any():
        raise ValueError(
            "Canonical BrainHarmonix representation must have unique sample_id entries"
        )
    inverse = {str(v): str(k) for k, v in (sample_id_map or {}).items()}
    if len(inverse) != len(sample_id_map or {}):
        raise ValueError("Brain sample mapping is not one-to-one")
    metadata = pd.read_csv(Path(samples.path), sep=None, engine="python", dtype=str)
    if samples.sample_id_col not in metadata or samples.target_col not in metadata:
        raise ValueError("The original analysis sample and target columns are required")
    allowed = set(
        metadata.loc[metadata[samples.target_col].notna(), samples.sample_id_col]
        .astype(str)
        .str.strip()
    )
    if eligible_ids is not None:
        allowed &= {str(sample_id) for sample_id in eligible_ids}
    matched = []
    for sample_id in frame["sample_id"].astype(str):
        source_id = inverse.get(sample_id, sample_id)
        study_id = str((sample_id_map or {}).get(source_id, sample_id))
        if source_id in runs and study_id in allowed:
            matched.append((sample_id, source_id))
    if not matched:
        raise ValueError(
            "No matching canonical BrainHarmonix samples and fMRIPrep subjects"
        )
    rng = np.random.default_rng(42)
    indices = np.sort(
        rng.choice(
            len(matched), size=min(len(matched), spec.max_subjects), replace=False
        )
    )
    return [matched[int(index)] for index in indices]


def _trace(
    encoder,
    source,
    selected,
    representations,
    features,
    provenance,
    region_indices,
    tolerance,
):
    from nilearn import image
    from ..console import console, info, progress

    atlas = image.load_img(str(provenance.atlas_path))
    rows = []
    baseline_errors = []
    expected = representations.set_index("sample_id")
    subject_count = len(selected)
    region_count = len(region_indices)
    evaluation_count = subject_count * (region_count + 1)
    with progress() as bar:
        task = bar.add_task(
            f"Anatomical traceability · subject 1/{subject_count} · preparing fMRI",
            total=evaluation_count,
        )
        for subject_number, (sample_id, run_id) in enumerate(selected, 1):
            bar.update(
                task,
                description=f"Anatomical traceability · subject {subject_number}/{subject_count} · extracting fMRI",
            )
            if not console.is_terminal:
                info(
                    f"Anatomical traceability · subject {subject_number}/{subject_count} · extracting fMRI"
                )
            signals, _ = extract_schaefer400(
                source[run_id], preprocessing=source["preprocessing"], atlas_img=atlas
            )
            tr = float(source[run_id].repetition_time)
            bar.update(
                task,
                description=f"Anatomical traceability · subject {subject_number}/{subject_count} · baseline",
            )
            reference = np.asarray(encoder.encode(signals, tr)[0], dtype=np.float64)
            if reference.shape != (768,) or not np.isfinite(reference).all():
                raise ValueError(f"Invalid BrainHarmonix output: {sample_id}")
            saved = np.asarray(
                [
                    expected.loc[sample_id, f"brain_brainharmonix_f__bhf_{k:03d}"]
                    for _, k, _ in features
                ],
                dtype=np.float64,
            )
            recomputed = np.asarray(
                [reference[k] for _, k, _ in features], dtype=np.float64
            )
            if not np.isfinite(saved).all():
                raise ValueError(f"Non-finite stored features: {sample_id}")
            error = float(np.max(np.abs(saved - recomputed)))
            baseline_errors.append(error)
            if error > tolerance:
                raise ValueError(
                    f"Stored embedding mismatch for {sample_id}: max error={error:.7g}; tolerance={tolerance}"
                )
            bar.advance(task)
            for region_number, region_index in enumerate(region_indices, 1):
                bar.update(
                    task,
                    description=f"Anatomical traceability · subject {subject_number}/{subject_count} · ROI {region_number}/{region_count}",
                )
                altered = np.asarray(signals, dtype=np.float32).copy()
                altered[:, region_index] = 0.0
                value = np.asarray(
                    encoder.encode(altered, tr, freeze_normalization=True)[0],
                    dtype=np.float64,
                )
                if value.shape != (768,) or not np.isfinite(value).all():
                    raise ValueError(
                        f"Invalid region-perturbed BrainHarmonix output: {sample_id}, {region_index}"
                    )
                for feature, dimension, rank_score in features:
                    rows.append(
                        (
                            sample_id,
                            feature,
                            dimension,
                            int(region_index),
                            provenance.labels[region_index],
                            float(reference[dimension]),
                            float(value[dimension]),
                            float(reference[dimension] - value[dimension]),
                            rank_score,
                        )
                    )
                bar.advance(task)
                if not console.is_terminal and (
                    region_number % 25 == 0 or region_number == region_count
                ):
                    completed = (
                        (subject_number - 1) * (region_count + 1) + 1 + region_number
                    )
                    info(
                        f"Anatomical traceability · {completed}/{evaluation_count} evaluations · subject {subject_number}/{subject_count} · ROI {region_number}/{region_count}"
                    )
        bar.update(
            task, description="Anatomical traceability · encoder evaluations complete"
        )
    info("Anatomical traceability · aggregating region sensitivity and uncertainty")
    frame = pd.DataFrame(
        rows,
        columns=(
            "sample_id",
            "feature",
            "dimension",
            "region_index",
            "region_name",
            "baseline_value",
            "perturbed_value",
            "delta",
            "feature_rank_score",
        ),
    )
    frame["absolute_delta"] = frame["delta"].abs()
    summary = frame.groupby(
        ["feature", "dimension", "region_index", "region_name"], as_index=False
    ).agg(
        mean_absolute_delta=("absolute_delta", "mean"),
        signed_mean_delta=("delta", "mean"),
        std_absolute_delta=("absolute_delta", "std"),
        n_subjects=("sample_id", "nunique"),
    )
    summary["std_absolute_delta"] = summary["std_absolute_delta"].fillna(0.0)
    rng = np.random.default_rng(42)
    intervals = []
    for (feature, region), part in frame.groupby(
        ["feature", "region_index"], sort=True
    ):
        values = part["absolute_delta"].to_numpy(dtype=float)
        if len(values) < 2:
            lower = upper = float(values.mean())
        else:
            draws = values[rng.integers(len(values), size=(500, len(values)))].mean(
                axis=1
            )
            lower, upper = np.quantile(draws, [0.025, 0.975])
        intervals.append((feature, region, lower, upper))
    ci = pd.DataFrame(
        intervals,
        columns=("feature", "region_index", "bootstrap_ci_low", "bootstrap_ci_high"),
    )
    summary = summary.merge(ci, on=["feature", "region_index"], validate="one_to_one")
    summary = summary.sort_values(
        ["feature", "mean_absolute_delta", "region_index"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return frame, summary, float(max(baseline_errors))


def run_anatomical_traceability(sweep, spec) -> dict[str, Path]:
    from ..storage import read_table, table_exists
    from ..console import info, success
    from .representations import Brain

    brain = getattr(sweep, "brain", None)
    if not isinstance(brain, Brain) or not any(
        rep.method == "brainharmonix_f" for rep in brain.representations
    ):
        raise ValueError(
            "AnatomicalTraceability currently supports prepared BrainHarmonix-F brain representations only"
        )
    from .anatomical_figure import _require_anatomical_visualization

    _require_anatomical_visualization()
    resolved = _resolved_brain(brain)
    atlas_path = Path(resolved.atlas_paths[400]).resolve()
    provenance = Schaefer400Provenance.from_nilearn(atlas_path)
    normalization = (
        "reference"
        if brain.normalization == "auto"
        and (
            brain.normalization_stats is not None
            or brain.reference_derivatives_dir is not None
        )
        else "per_run"
        if brain.normalization == "auto"
        else brain.normalization
    )
    stats_path = (
        Path(brain.normalization_stats)
        if brain.normalization_stats is not None
        else Path(sweep.root())
        / "brain_representations"
        / "brainharmonix_f"
        / "brainharmonix_f_normalization.npz"
    )
    if normalization == "reference" and not stats_path.exists():
        raise FileNotFoundError(
            f"Cannot reproduce original BrainHarmonix normalization: {stats_path}"
        )
    repr_path = Path(sweep.root()) / "brain_representations" / "brainharmonix_f.parquet"
    if not table_exists(repr_path):
        raise FileNotFoundError(
            f"Missing original BrainHarmonix representation: {repr_path}"
        )
    representations = read_table(repr_path)
    expected = {f"brain_brainharmonix_f__bhf_{k:03d}" for k in range(768)}
    if not expected.issubset(representations.columns):
        raise ValueError(
            "Canonical BrainHarmonix-F embeddings are incomplete or renamed"
        )
    run_map = {
        run.subject_id: run
        for run in discover_fmriprep_runs(
            brain.derivatives_dir, bids_root=brain.bids_root
        )
    }
    source = dict(run_map)
    source["preprocessing"] = brain.preprocessing
    encoder = None
    root = Path(sweep.root()) / "explainability"
    if not root.is_dir():
        raise FileNotFoundError(f"No explainability directory exists: {root}")
    produced = {}
    requested = getattr(sweep.explainability, "targets", "auto")
    target_names = (
        {str(value).strip().lower().replace("-", "_") for value in requested}
        if isinstance(requested, (list, tuple))
        else {str(requested).strip().lower().replace("-", "_")}
    )
    all_targets = bool(target_names & {"auto", "all", "best"})
    eligible_targets = 0
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "explained_unit.json").exists():
            continue
        if not all_targets and directory.name.lower() not in target_names:
            continue
        eligible_targets += 1
        features = _ranked_features(directory, spec.rank_by, spec.top_k)
        if not features:
            raise ValueError(
                f"AnatomicalTraceability found no BrainHarmonix-F embedding coordinates in the explanation of {directory.name}. Verify that the selected model uses the raw brainharmonix_f representation, not microbiota-only or transformed joint-PCA coordinates."
            )
        info(
            f"Anatomical traceability · {directory.name} · {len(features)} embedding features · {len(spec.region_indices)} Schaefer parcels"
        )
        predicted_ids = None
        candidate_paths = [directory / "oof_predictions.parquet"]
        if directory.name == "mpma_b":
            candidate_paths.insert(
                0,
                Path(sweep.root()) / "predictions" / "mpma_b_outer_predictions.parquet",
            )
        for candidate in candidate_paths:
            if table_exists(candidate):
                predictions = read_table(candidate)
                if "sample_id" not in predictions.columns:
                    raise ValueError(
                        f"Missing sample IDs in evaluated predictions: {candidate}"
                    )
                predicted_ids = set(predictions["sample_id"].astype(str))
                break
        selected = _select_subjects(
            representations,
            run_map,
            spec,
            brain.sample_id_map,
            sweep.samples,
            predicted_ids,
        )
        if encoder is None:
            info(
                f"Anatomical traceability · initializing BrainHarmonix-F · {len(selected) * (len(spec.region_indices) + 1):,} encoder evaluations planned"
            )
            encoder = BrainHarmonixFEncoder(device="auto", normalization=normalization)
            if normalization == "reference":
                encoder.load_normalization(stats_path)
            info(
                f"Anatomical traceability · BrainHarmonix-F ready · device={encoder.device}"
            )
        frame, summary, max_error = _trace(
            encoder,
            source,
            selected,
            representations,
            features,
            provenance,
            spec.region_indices,
            spec.verification_tolerance,
        )
        info(
            f"Anatomical traceability · saving {len(frame):,} subject-region feature measurements"
        )
        output = directory / "anatomical_traceability"
        output.mkdir(parents=True, exist_ok=True)
        subject_path = output / "subject_region_sensitivity.tsv"
        region_path = output / "feature_region_sensitivity.tsv"
        selected_path = output / "selected_features.tsv"
        meta_path = output / "provenance.json"
        frame.to_csv(subject_path, sep="\t", index=False)
        summary.to_csv(region_path, sep="\t", index=False)
        pd.DataFrame(features, columns=("feature", "dimension", "rank_score")).to_csv(
            selected_path, sep="\t", index=False
        )
        meta_path.write_text(
            json.dumps(
                {
                    "schema": "mllabiome.anatomical_traceability.v1",
                    "representation": "brainharmonix_f",
                    "source": str(Path(brain.derivatives_dir).resolve()),
                    "atlas_path": str(atlas_path),
                    "atlas": "Schaefer-400 7Networks",
                    "render_space": "Schaefer-400 fsaverage5 template surfaces, not individual participant anatomy",
                    "normalization": normalization,
                    "normalization_stats": str(stats_path.resolve())
                    if normalization == "reference"
                    else None,
                    "intervention": "zero_one_raw_schaefer_roi_signal_with_unperturbed_normalization_frozen",
                    "quantity": "absolute_or_signed_change_in_a_pooled_embedding_coordinate",
                    "scope": "embedding_not_outcome_or_pairwise_interaction",
                    "uncertainty": "500 subject-resampled bootstrap replicates for each feature-region mean absolute delta; 95% percentile intervals",
                    "samples": [
                        {"sample_id": a, "source_run_id": b} for a, b in selected
                    ],
                    "n_subjects": len(selected),
                    "n_available_samples": int(len(representations)),
                    "sampling_basis": "prediction_subject_ids_intersected_with_original_analysis_cohort"
                    if predicted_ids is not None
                    else "analysis_metadata_and_available_brain_representations",
                    "top_k": spec.top_k,
                    "region_indices": list(spec.region_indices),
                    "regions_tested": len(spec.region_indices),
                    "regions_total": 400,
                    "coverage": "complete"
                    if len(spec.region_indices) == 400
                    else "subset",
                    "method": spec.method,
                    "max_baseline_embedding_error": max_error,
                    "verification_tolerance": spec.verification_tolerance,
                    "limitations": "Model dependence on ROI replacement is not a biological mechanism, structural connectivity, a unique decomposition, or causal evidence. Replacing an ROI with zeros may be out of distribution. A top-feature ranking can be unstable, particularly when model generalization is weak.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        from .anatomical_figure import render_anatomical_traceability

        figure = render_anatomical_traceability(
            summary,
            features,
            provenance,
            directory,
            output / "network_to_anatomy",
            top_regions=spec.top_regions_per_feature,
        )
        for key, value in {
            "subject_sensitivity": subject_path,
            "regional_sensitivity": region_path,
            "selected_features": selected_path,
            "provenance": meta_path,
            "network_anatomy_svg": figure[0],
            "network_anatomy_png": figure[1],
        }.items():
            produced[f"{directory.name}_anatomical_{key}"] = value
        success(f"Anatomical traceability · {directory.name} · {figure[0]}")
    if not eligible_targets:
        raise FileNotFoundError(
            f"No explained target matches {sorted(target_names)} in {root}. Run explainability for the requested target first."
        )
    if not produced:
        raise RuntimeError(f"AnatomicalTraceability produced no outputs in {root}")
    return produced
