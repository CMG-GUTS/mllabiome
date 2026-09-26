from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ._version import __version__
from .compute import ResourceTracker, machine_profile
from .configs_sweep import Sweep, _scientific_digest
from .data import (
    Dataset,
    _taxonomic_rank,
    dataset_fingerprint,
    load_dataset,
    metadata_path as training_metadata_path,
)
from .ensemble_aggregation import PROBABILITY_PRESERVING_AGGREGATIONS
from .evaluation_splits import _groups_from_metadata
from .final_models import aggregate_member_predictions, build_final_models
from .learners import _learner_factory, _learner_name, fit_classifier
from .metrics import _estimator_call, _predict_proba_aligned
from .resolutions import _parse_resolution, materialize_mpdr_with_blocks
from .runtime import configure_estimator_threads
from .storage import table_exists
from .transformations import (
    _count_transformation_factory,
    _count_transformation_specs_for_blocks,
)
from .utils import TAXONOMIC_LEVELS, dump_json_standard

_DEPLOYMENT_SCHEMA = 1
_INFERENCE_SCHEMA = 1
_MATRIX_FORMATS = {"mllab", "matrix_tsv", "metaphlan", "metaphlan_tsv", "profile_tsv"}
_WIDE_FORMATS = {"csv", "wide_csv"}
_RAW_LEVELS = {"all", "features", "asis", "raw"}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _normalise_format(value: str) -> str:
    return str(value).strip().casefold().replace("-", "_")


def _training_format(sweep: Sweep) -> str:
    value = _normalise_format(sweep.data.format)
    if value != "auto":
        return value
    path = Path(sweep.data.abundance_path)
    if (
        path.suffix.lower() in {".tsv", ".txt"}
        and training_metadata_path(sweep.data) is not None
    ):
        return "mllab"
    return "wide_csv"


def _format_family(value: str) -> str:
    value = _normalise_format(value)
    if value in _MATRIX_FORMATS:
        return "matrix_tsv"
    if value in _WIDE_FORMATS:
        return "wide_csv"
    raise ValueError(f"Unsupported inference data format {value!r}.")


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_")
    return token or "class"


def _atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, sep="\t", index=False)
    temp.replace(path)


def _atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temp, compress=3)
    temp.replace(path)


def _read_optional_metadata(
    path: Path | None, sample_id_col: str, sample_ids: list[str]
) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path, sep=None, engine="python", dtype=str)
    if sample_id_col not in frame.columns:
        raise ValueError(
            f"Inference metadata is missing sample ID column {sample_id_col!r}."
        )
    frame[sample_id_col] = frame[sample_id_col].astype(str).str.strip()
    if frame[sample_id_col].duplicated().any():
        duplicates = frame.loc[
            frame[sample_id_col].duplicated(), sample_id_col
        ].tolist()
        raise ValueError(
            f"Inference metadata contains duplicate sample IDs: {duplicates[:5]!r}."
        )
    indexed = frame.set_index(sample_id_col)
    missing = [sample_id for sample_id in sample_ids if sample_id not in indexed.index]
    if missing:
        raise ValueError(
            f"Inference metadata is missing {len(missing)} profile sample(s); first IDs: {missing[:5]!r}."
        )
    return indexed.loc[sample_ids].reset_index()


def _read_matrix_profile(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    frame = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    frame.index = frame.index.astype(str).str.strip()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("Inference abundance matrix is empty.")
    if frame.index.duplicated().any():
        duplicates = frame.index[frame.index.duplicated()].tolist()
        raise ValueError(
            f"Inference abundance matrix contains duplicate feature names: {duplicates[:5]!r}."
        )
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(
            f"Inference abundance matrix contains duplicate sample IDs: {duplicates[:5]!r}."
        )
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any(axis=0)].astype(str).tolist()
        raise ValueError(
            f"Inference abundance matrix contains missing or non-numeric values; affected samples: {bad[:5]!r}."
        )
    values = numeric.T.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Inference abundance matrix contains NaN or infinite values.")
    if np.any(values < 0):
        raise ValueError(
            "Inference abundance matrix contains negative abundance values."
        )
    return values, frame.index.astype(str).tolist(), frame.columns.astype(str).tolist()


def _read_wide_profile(
    path: Path, sample_id_col: str, expected_features: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    frame = pd.read_csv(path, sep=None, engine="python")
    if sample_id_col not in frame.columns:
        raise ValueError(
            f"Inference table is missing sample ID column {sample_id_col!r}."
        )
    sample_ids = frame[sample_id_col].astype(str).str.strip().tolist()
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Inference table contains duplicate sample IDs.")
    missing = [name for name in expected_features if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Inference table is missing {len(missing)} training feature column(s); first features: {missing[:5]!r}."
        )
    numeric = frame[expected_features].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(
            "Inference feature columns contain missing or non-numeric values."
        )
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            "Inference feature matrix must contain finite non-negative values."
        )
    return values, list(expected_features), sample_ids


def _profile_for_sweep(
    sweep: Sweep, training: Dataset
) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame | None, dict[str, Any]]:
    plan = sweep.inference
    training_format = _training_format(sweep)
    inference_format = (
        training_format if plan.format == "same" else _normalise_format(plan.format)
    )
    training_family = _format_family(training_format)
    inference_family = _format_family(inference_format)
    if inference_family != training_family:
        raise ValueError(
            f"Inference data must use the same profile format family as training data: training={training_format!r}, inference={inference_format!r}."
        )
    path = Path(plan.abundance_path)
    if not path.exists():
        raise FileNotFoundError(path)
    sample_id_col = plan.sample_id_col or sweep.data.sample_id_col
    if inference_family == "matrix_tsv":
        X, features, sample_ids = _read_matrix_profile(path)
    else:
        X, features, sample_ids = _read_wide_profile(
            path, sample_id_col, training.feature_names_by_level["all"]
        )
    overlap = sorted(set(sample_ids) & set(training.sample_ids))
    if overlap and not plan.allow_training_sample_overlap:
        raise ValueError(
            f"Inference data overlap the training cohort by {len(overlap)} sample ID(s); first IDs: {overlap[:5]!r}."
        )
    metadata_path = None if plan.metadata_path is None else Path(plan.metadata_path)
    metadata = _read_optional_metadata(metadata_path, sample_id_col, sample_ids)
    audit = {
        "training_format": training_format,
        "inference_format": inference_format,
        "format_family": inference_family,
        "n_samples": len(sample_ids),
        "n_features": len(features),
        "sample_overlap_with_training": len(overlap),
        "abundance_sha256": _sha256_file(path),
        "metadata_sha256": None
        if metadata_path is None
        else _sha256_file(metadata_path),
        "metadata_used_as_predictors": False,
    }
    return X, features, sample_ids, metadata, audit


def _selected_levels(sweep: Sweep, model: dict[str, Any]) -> tuple[str, ...]:
    resolution = str(model.get("resolution", "")).strip()
    matches = []
    for item in sweep.resolutions:
        name, levels = _parse_resolution(item)
        if str(name) == resolution:
            matches.append(tuple(levels))
    if len(matches) != 1:
        raise ValueError(
            f"Cannot resolve selected resolution {resolution!r} in the current sweep configuration."
        )
    return tuple(str(level) for level in matches[0])


def _abundance_scale(X: np.ndarray) -> str:
    values = np.asarray(X, dtype=float)
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError(
            "Inference contains zero-total abundance profiles at a selected model resolution."
        )
    positive = values[values > 0]
    integer_fraction = (
        float(np.mean(np.isclose(positive, np.rint(positive), atol=1e-8)))
        if positive.size
        else 0.0
    )
    proportion_fraction = float(np.mean(np.isclose(totals, 1.0, rtol=0.03, atol=0.03)))
    percentage_fraction = float(np.mean(np.isclose(totals, 100.0, rtol=0.03, atol=2.0)))
    if proportion_fraction >= 0.80:
        return "proportion"
    if percentage_fraction >= 0.80:
        return "percentage"
    if integer_fraction >= 0.98 and float(np.median(totals)) > 1.0:
        return "counts"
    return "nonnegative_abundance"


def _profile_resolution(
    X: np.ndarray, names: list[str], levels: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    if any(level in _RAW_LEVELS for level in levels):
        if len(levels) != 1:
            raise ValueError(
                "Raw inference resolution cannot be combined with explicit taxonomic ranks."
            )
        return np.asarray(X, dtype=np.float64), list(names)
    pieces = []
    out_names = []
    for level in levels:
        indices = [
            index for index, name in enumerate(names) if _taxonomic_rank(name) == level
        ]
        if not indices:
            raise ValueError(
                f"Inference profiles contain no features at selected taxonomic rank {level!r}."
            )
        pieces.append(np.asarray(X[:, indices], dtype=np.float64))
        out_names.extend(names[index] for index in indices)
    return (
        pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
    ), out_names


def _align_resolution(
    train_X: np.ndarray,
    train_names: list[str],
    test_X: np.ndarray,
    test_names: list[str],
    *,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_set = set(train_names)
    test_set = set(test_names)
    missing = sorted(train_set - test_set)
    extra = sorted(test_set - train_set)
    if policy == "strict" and (missing or extra):
        raise ValueError(
            f"Inference feature schema differs from training at the selected resolution: {len(missing)} missing and {len(extra)} extra feature(s). Missing examples: {missing[:5]!r}; extra examples: {extra[:5]!r}."
        )
    if policy == "zero_fill" and extra:
        test_map = {name: index for index, name in enumerate(test_names)}
        aligned = np.zeros((test_X.shape[0], len(train_names)), dtype=np.float64)
        for index, name in enumerate(train_names):
            if name in test_map:
                aligned[:, index] = test_X[:, test_map[name]]
    elif policy == "zero_fill":
        test_map = {name: index for index, name in enumerate(test_names)}
        aligned = np.zeros((test_X.shape[0], len(train_names)), dtype=np.float64)
        for index, name in enumerate(train_names):
            if name in test_map:
                aligned[:, index] = test_X[:, test_map[name]]
    else:
        test_map = {name: index for index, name in enumerate(test_names)}
        aligned = test_X[:, [test_map[name] for name in train_names]]
    audit = {
        "n_training_features": len(train_names),
        "n_inference_features": len(test_names),
        "n_missing_features": len(missing),
        "n_extra_features": len(extra),
        "exact_feature_set": not missing and not extra,
        "missing_features": missing,
        "extra_features": extra,
    }
    return np.asarray(aligned, dtype=np.float64), audit


def _transformation_item(sweep: Sweep, blocks: Any, identity: str) -> Any:
    matches = [
        item
        for item in _count_transformation_specs_for_blocks(
            sweep.count_transformations, blocks
        )
        if str(item[0]) == str(identity)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot resolve selected abundance transformation {identity!r} in the current sweep configuration."
        )
    return matches[0]


def _learner_item(sweep: Sweep, name: str) -> Any:
    matches = [item for item in sweep.learners if _learner_name(item) == str(name)]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot resolve selected learner {name!r} in the current sweep configuration."
        )
    return matches[0]


def _member_signature(dataset: Dataset, model: dict[str, Any]) -> str:
    return _scientific_digest(
        {
            "schema": _DEPLOYMENT_SCHEMA,
            "package_version": __version__,
            "training_dataset_fingerprint": dataset_fingerprint(dataset),
            "model": model,
        }
    )


def _fit_member(
    sweep: Sweep,
    dataset: Dataset,
    profile_X: np.ndarray,
    profile_names: list[str],
    model: dict[str, Any],
    deployment_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    config_id = str(model["config_id"])
    signature = _member_signature(dataset, model)
    bundle_path = deployment_dir / "members" / f"{config_id}.joblib"
    bundle = None
    if bundle_path.exists():
        try:
            candidate = joblib.load(bundle_path)
            if isinstance(candidate, dict) and candidate.get("signature") == signature:
                bundle = candidate
        except Exception:
            bundle = None
    levels = _selected_levels(sweep, model)
    train_X, train_names, blocks = materialize_mpdr_with_blocks(dataset, levels)
    test_raw, test_names = _profile_resolution(profile_X, profile_names, levels)
    training_scale = _abundance_scale(np.asarray(train_X, dtype=float))
    inference_scale = _abundance_scale(np.asarray(test_raw, dtype=float))
    if training_scale != inference_scale:
        raise ValueError(
            f"Inference abundance scale differs from training for config {config_id!r}: training={training_scale!r}, inference={inference_scale!r}."
        )
    test_X, audit = _align_resolution(
        np.asarray(train_X, dtype=np.float64),
        list(train_names),
        test_raw,
        test_names,
        policy=sweep.inference.feature_policy,
    )
    if bundle is None:
        ct_item = _transformation_item(
            sweep, blocks, str(model["count_transformation"])
        )
        _, factory = _count_transformation_factory(
            ct_item,
            random_state=int(sweep.evaluation.random_state),
            feature_blocks=blocks,
        )
        transformer = factory()
        train_transformed, test_transformed = transformer.apply_pair(
            np.asarray(train_X, dtype=np.float64), test_X
        )
        learner = _learner_item(sweep, str(model["learner"]))
        _, learner_factory = _learner_factory(learner, task=dataset.task)
        estimator = configure_estimator_threads(learner_factory(), 1)
        if dataset.task == "classification":
            groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
            fit_classifier(estimator, train_transformed, dataset.y, groups)
        else:
            estimator.fit(train_transformed, dataset.y)
        transformed_names = transformer.get_feature_names_out(list(train_names))
        bundle = {
            "schema_version": _DEPLOYMENT_SCHEMA,
            "signature": signature,
            "package_version": __version__,
            "config_id": config_id,
            "resolution": str(model.get("resolution", "")),
            "levels": list(levels),
            "count_transformation": str(model.get("count_transformation", "")),
            "learner": str(model.get("learner", "")),
            "task": dataset.task,
            "class_labels": list(dataset.class_labels),
            "positive_class": dataset.positive_class,
            "raw_feature_names": list(train_names),
            "transformed_feature_names": list(transformed_names),
            "feature_filter": transformer.feature_filter_metadata(),
            "transformer": transformer,
            "estimator": estimator,
        }
        _atomic_joblib(bundle, bundle_path)
    else:
        expected_names = [str(value) for value in bundle.get("raw_feature_names", [])]
        if expected_names != list(train_names):
            raise ValueError(
                f"Cached deployment feature schema for {config_id!r} does not match the current training data."
            )
        test_transformed = bundle["transformer"].apply(test_X)
    estimator = bundle["estimator"]
    if dataset.task == "classification":
        proba = _predict_proba_aligned(estimator, test_transformed, dataset.classes)
        if (
            not np.isfinite(proba).all()
            or np.any(proba < -1e-12)
            or np.any(proba > 1.0 + 1e-12)
        ):
            raise ValueError(
                f"Deployment model {config_id!r} produced invalid probabilities."
            )
        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError(
                f"Deployment model {config_id!r} produced probabilities that do not sum to one."
            )
        predictions = np.asarray(proba, dtype=float)
    else:
        predictions = np.asarray(
            _estimator_call(estimator, "predict", test_transformed), dtype=float
        ).reshape(-1, 1)
        if not np.isfinite(predictions).all():
            raise ValueError(
                f"Deployment model {config_id!r} produced non-finite predictions."
            )
    audit.update(
        {
            "config_id": config_id,
            "resolution": str(model.get("resolution", "")),
            "levels": ",".join(levels),
            "count_transformation": str(model.get("count_transformation", "")),
            "learner": str(model.get("learner", "")),
            "training_abundance_scale": training_scale,
            "inference_abundance_scale": inference_scale,
            "deployment_signature": signature,
            "bundle_path": str(bundle_path),
        }
    )
    return bundle, predictions, audit


def _strategy_predictions(
    sweep: Sweep,
    dataset: Dataset,
    profile_X: np.ndarray,
    profile_names: list[str],
    models: dict[str, Any],
    deployment_dir: Path,
    strategy: str,
    member_cache: dict[str, tuple[dict[str, Any], np.ndarray, dict[str, Any]]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if strategy == "MPMA-B":
        model = models["MPMA-B"]
        config_id = str(model["config_id"])
        if config_id not in member_cache:
            member_cache[config_id] = _fit_member(
                sweep, dataset, profile_X, profile_names, model, deployment_dir
            )
        return member_cache[config_id][1], [member_cache[config_id][2]]
    unit = models.get("MPMA-E")
    if not isinstance(unit, dict):
        raise ValueError(
            "MPMA-E was requested for inference but no final MPMA-E specification is available."
        )
    member_predictions = []
    audits = []
    for member in unit["members"]:
        config_id = str(member["config_id"])
        if config_id not in member_cache:
            member_cache[config_id] = _fit_member(
                sweep, dataset, profile_X, profile_names, member, deployment_dir
            )
        member_predictions.append(member_cache[config_id][1])
        audits.append(member_cache[config_id][2])
    stack = np.stack(member_predictions, axis=0)
    aggregation = str(unit["aggregation_strategy"])
    weights = None
    if aggregation in {"weighted_mean_proba", "weighted_mean_prediction"}:
        weights = [
            float(member.get("aggregation_weight", member.get("weight")))
            for member in unit["members"]
        ]
    predictions = aggregate_member_predictions(stack, aggregation, weights)
    if dataset.task == "classification" and aggregation in set(
        PROBABILITY_PRESERVING_AGGREGATIONS
    ):
        if not np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("MPMA-E deployment probabilities do not sum to one.")
    return np.asarray(predictions, dtype=float), audits


def _prediction_frame(
    sample_ids: list[str], dataset: Dataset, strategy_predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    out = pd.DataFrame({"sample_id": sample_ids})
    for strategy, values in strategy_predictions.items():
        prefix = strategy.casefold().replace("-", "_")
        if dataset.task == "classification":
            predicted = np.argmax(values, axis=1).astype(int)
            out[f"{prefix}_predicted_class"] = [
                dataset.class_labels[index] for index in predicted
            ]
            used = set()
            for index, label in enumerate(dataset.class_labels):
                token = _safe_token(label)
                base = token
                suffix = 2
                while token in used:
                    token = f"{base}_{suffix}"
                    suffix += 1
                used.add(token)
                out[f"{prefix}_probability_{token}"] = values[:, index]
        else:
            out[f"{prefix}_prediction"] = values.reshape(-1)
    return out


def _deployment_manifest(
    sweep: Sweep,
    dataset: Dataset,
    models: dict[str, Any],
    targets: tuple[str, ...],
    member_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": _DEPLOYMENT_SCHEMA,
        "package_version": __version__,
        "training_dataset_fingerprint": dataset_fingerprint(dataset),
        "training_samples": len(dataset.sample_ids),
        "task": dataset.task,
        "class_labels": list(dataset.class_labels),
        "positive_class": dataset.positive_class,
        "targets": list(targets),
        "final_models": models,
        "members": member_audits,
    }


def run_inference(
    sweep: Sweep, models: dict[str, Any] | None = None
) -> dict[str, Path]:
    if sweep.inference is None:
        raise ValueError("No Inference configuration is attached to this sweep.")
    if sweep.uses_modalities:
        raise ValueError(
            "Inference currently supports microbiota Data-based sweeps only."
        )
    root = Path(sweep.root())
    required = (
        root / "configs.parquet",
        root / "predictions" / "outer_predictions.parquet",
        root / "tables" / "mpma_b_final_candidate.json",
    )
    missing = [
        str(path)
        for path in required
        if not (table_exists(path) if path.suffix == ".parquet" else path.exists())
    ]
    if missing:
        raise FileNotFoundError(
            "Inference requires completed evaluation artifacts. Run --stage evaluate first. Missing: "
            + ", ".join(missing)
        )
    models = (
        build_final_models(root, include_mpma_e="mpma_e" in sweep.inference.targets)
        if models is None
        else models
    )
    requested = tuple(
        "MPMA-B" if value == "mpma_b" else "MPMA-E" for value in sweep.inference.targets
    )
    if "MPMA-E" in requested and "MPMA-E" not in models:
        raise ValueError(
            "MPMA-E inference was requested but the final ensemble specification is unavailable."
        )
    selected_models = [models["MPMA-B"]]
    if "MPMA-E" in requested:
        selected_models.extend(models["MPMA-E"]["members"])
    levels = []
    for model in selected_models:
        levels.extend(
            level
            for level in _selected_levels(sweep, model)
            if level not in _RAW_LEVELS
        )
    training = load_dataset(sweep.data, tuple(dict.fromkeys(levels)) or ("all",))
    profile_X, profile_names, sample_ids, metadata, input_audit = _profile_for_sweep(
        sweep, training
    )
    inference_root = root / "inference"
    deployment_dir = inference_root / "deployment"
    inference_root.mkdir(parents=True, exist_ok=True)
    tracker = ResourceTracker().start()
    member_cache: dict[str, tuple[dict[str, Any], np.ndarray, dict[str, Any]]] = {}
    strategy_values: dict[str, np.ndarray] = {}
    audits: list[dict[str, Any]] = []
    try:
        for strategy in requested:
            values, rows = _strategy_predictions(
                sweep,
                training,
                profile_X,
                profile_names,
                models,
                deployment_dir,
                strategy,
                member_cache,
            )
            strategy_values[strategy] = values
            audits.extend(rows)
    finally:
        resources = tracker.stop()
    predictions = _prediction_frame(sample_ids, training, strategy_values)
    predictions_path = inference_root / "predictions.tsv"
    _atomic_tsv(predictions, predictions_path)
    metadata_predictions_path = None
    if metadata is not None:
        sample_id_col = sweep.inference.sample_id_col or sweep.data.sample_id_col
        annotation = metadata.copy()
        if sample_id_col != "sample_id":
            annotation = annotation.rename(columns={sample_id_col: "sample_id"})
        duplicate_columns = [
            column
            for column in annotation.columns
            if column != "sample_id" and column in predictions.columns
        ]
        if duplicate_columns:
            annotation = annotation.rename(
                columns={column: f"metadata_{column}" for column in duplicate_columns}
            )
        annotated = predictions.merge(
            annotation, on="sample_id", how="left", validate="one_to_one"
        )
        metadata_predictions_path = inference_root / "predictions_with_metadata.tsv"
        _atomic_tsv(annotated, metadata_predictions_path)
    unique_audits = {row["config_id"]: row for row in audits}
    feature_audit = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"missing_features", "extra_features", "bundle_path"}
            }
            for row in unique_audits.values()
        ]
    )
    feature_audit_path = inference_root / "feature_audit.tsv"
    _atomic_tsv(feature_audit, feature_audit_path)
    deployment_manifest = _deployment_manifest(
        sweep,
        training,
        models,
        requested,
        list(unique_audits.values()),
    )
    deployment_manifest_path = deployment_dir / "manifest.json"
    dump_json_standard(deployment_manifest, deployment_manifest_path)
    manifest = {
        "schema_version": _INFERENCE_SCHEMA,
        "package_version": __version__,
        "task": training.task,
        "target_name": training.target_name,
        "targets": list(requested),
        "training_samples": len(training.sample_ids),
        "inference_samples": len(sample_ids),
        "input": input_audit,
        "feature_policy": sweep.inference.feature_policy,
        "training_outcomes_used_for_full_fit": True,
        "inference_outcomes_used_for_fit_or_prediction": False,
        "inference_metadata_used_as_predictors": False,
        "predictions": str(predictions_path),
        "predictions_with_metadata": None
        if metadata_predictions_path is None
        else str(metadata_predictions_path),
        "feature_audit": str(feature_audit_path),
        "deployment_manifest": str(deployment_manifest_path),
        "resources": resources,
        "machine": machine_profile(),
        "python_version": platform.python_version(),
    }
    manifest_path = inference_root / "manifest.json"
    dump_json_standard(manifest, manifest_path)
    outputs = {
        "predictions": predictions_path,
        "feature_audit": feature_audit_path,
        "manifest": manifest_path,
        "deployment_manifest": deployment_manifest_path,
    }
    if metadata_predictions_path is not None:
        outputs["predictions_with_metadata"] = metadata_predictions_path
    return outputs
