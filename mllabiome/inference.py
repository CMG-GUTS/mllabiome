from __future__ import annotations


import gc
import hashlib
import importlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any


import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


from ._version import __version__
from .compute import ResourceTracker, machine_profile
from .conformal import external_conformal_sets
from .configs_sweep import Sweep, _scientific_digest
from .data import (
    Dataset,
    _first_nonempty_tsv_row,
    _read_feature_by_sample_tsv,
    _taxonomic_rank,
    dataset_fingerprint,
    load_dataset,
)
from .data import metadata_path as training_metadata_path
from .ensemble_aggregation import PROBABILITY_PRESERVING_AGGREGATIONS
from .evaluation_splits import _groups_from_metadata
from .final_models import (
    aggregate_member_predictions,
    build_final_models,
    hydrate_final_models,
)
from .learners import _learner_factory, _learner_name, fit_classifier
from .inference_progress import current_inference_progress, inference_phase
from .console import info
from .metrics import _estimator_call, _predict_proba_aligned
from .resolutions import (
    _normalise_levels,
    _parse_resolution,
    materialize_mpdr_with_blocks,
)
from .runtime import configure_estimator_threads
from .storage import table_exists
from .transformations import (
    PrevalenceFilter,
    Transform,
    _count_transformation_factory,
    _count_transformation_specs_for_blocks,
    _parse_transformation_identity,
    _split_transformation_filter_identity,
    transformation_label,
)
from .utils import dump_json_standard


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


def _prediction_checkpoint_path(deployment_dir: Path, config_id: str) -> Path:
    return deployment_dir / "predictions" / f"{config_id}.npy"


def _load_prediction_checkpoint(
    path: Path,
    signature: str,
    n_samples: int,
    n_outputs: int,
    *,
    require_bundle: bool,
    bundle_path: Path,
    bundle_signature: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    if require_bundle and (
        not bundle_path.is_file()
        or not _bundle_signature_path(bundle_path).is_file()
        or not _bundle_might_match(bundle_path, bundle_signature)
    ):
        return None
    try:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("signature") != signature:
            return None
        if record.get("sha256") != _sha256_file(path):
            return None
        with path.open("rb") as handle:
            values = np.load(handle, allow_pickle=False)
        if values.shape != (n_samples, n_outputs) or not np.isfinite(values).all():
            return None
        if not np.issubdtype(values.dtype, np.number):
            return None
        audit = record.get("audit")
        if not isinstance(audit, dict):
            return None
        if n_outputs > 1:
            if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
                return None
            if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
                return None
        return np.asarray(values, dtype=float), audit
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _save_prediction_checkpoint(
    path: Path,
    signature: str,
    values: np.ndarray,
    audit: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_values = path.with_suffix(".npy.tmp")
    with temp_values.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    record = {
        "schema_version": 1,
        "signature": signature,
        "sha256": _sha256_file(temp_values),
        "audit": audit,
    }
    temp_metadata = path.with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    temp_values.replace(path)
    temp_metadata.replace(path.with_suffix(".json"))


def _is_mps_oom(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return "mps" in message and "out of memory" in message


def _kumo_cpu_retry_enabled() -> bool:
    return os.getenv("MLLABIOME_INFERENCE_KUMO_CPU_FALLBACK", "").strip().casefold() in {
        "1", "true", "yes", "on"
    }


def _fit_estimator_with_kumo_fallback(
    estimator: Any,
    learner_factory: Any,
    train_X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    task: str,
    feature_names: list[str],
    config_id: str,
) -> tuple[Any, str]:
    failed_on_mps = False
    try:
        if task == "classification":
            fit_classifier(estimator, train_X, y, groups)
        else:
            estimator.fit(train_X, y)
    except RuntimeError as exc:
        if not _is_mps_oom(exc):
            raise
        if not (
            estimator.__class__.__module__ == "mllabiome.kumo"
            and estimator.__class__.__name__ == "KumoTabularClassifier"
        ):
            raise
        if not _kumo_cpu_retry_enabled():
            raise RuntimeError(
                f"Kumo member {config_id} exhausted MPS memory during its own fit. "
                "Previous ensemble members are already processed sequentially. "
                "To retry the same Kumo architecture on CPU, set "
                "MLLABIOME_INFERENCE_KUMO_CPU_FALLBACK=1. "
                "CPU execution can still require substantial system RAM."
            ) from exc
        failed_on_mps = True
    if not failed_on_mps:
        return estimator, str(getattr(estimator, "device_", "default"))
    info(f"Inference · member {config_id} · MPS fit exhausted available memory; retrying Kumo on CPU")
    del estimator
    _release_accelerator_memory(config_id)
    estimator = configure_estimator_threads(learner_factory(), 1)
    estimator.set_params(device="cpu")
    if callable(getattr(estimator, "bind_schema", None)):
        estimator.bind_schema(feature_names)
    if task == "classification":
        fit_classifier(estimator, train_X, y, groups)
    else:
        estimator.fit(train_X, y)
    return estimator, "cpu"


def _release_accelerator_memory(config_id: str) -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    gc.collect()
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            info(f"Inference · member {config_id} · MPS allocated after cleanup: {allocated:.2f} GiB")
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _bundle_signature_path(path: Path) -> Path:
    return path.with_suffix(".signature.json")


def _bundle_might_match(path: Path, signature: str) -> bool:
    signature_path = _bundle_signature_path(path)
    if not signature_path.exists():
        return True
    try:
        data = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("signature") == signature


def _write_bundle_signature(path: Path, signature: str) -> None:
    signature_path = _bundle_signature_path(path)
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    temp = signature_path.with_suffix(signature_path.suffix + ".tmp")
    temp.write_text(json.dumps({"signature": signature}, sort_keys=True), encoding="utf-8")
    temp.replace(signature_path)


def _read_profile_metadata(path: Path | None, sample_id_col: str) -> pd.DataFrame | None:
    if path is None:
        return None
    parsed_headers: dict[str, list[str]] = {}
    frame = None
    for delimiter in ("\t", ",", ";"):
        candidate = pd.read_csv(path, sep=delimiter, dtype=str, encoding="utf-8-sig")
        candidate.columns = candidate.columns.astype(str).str.strip()
        if sample_id_col in candidate.columns:
            frame = candidate
            break
        parsed_headers[repr(delimiter)] = candidate.columns.tolist()
    if frame is None:
        raise ValueError(
            f"Inference metadata {path} is missing sample ID column {sample_id_col!r}. "
            f"Parsed headers by delimiter: {parsed_headers!r}."
        )
    frame[sample_id_col] = frame[sample_id_col].astype(str).str.strip()
    if frame[sample_id_col].duplicated().any():
        duplicates = frame.loc[
            frame[sample_id_col].duplicated(), sample_id_col
        ].tolist()
        raise ValueError(
            f"Inference metadata contains duplicate sample IDs: {duplicates[:5]!r}."
        )
    return frame


def _align_profile_metadata(
    frame: pd.DataFrame | None, sample_id_col: str, sample_ids: list[str]
) -> pd.DataFrame | None:
    if frame is None:
        return None
    indexed = frame.set_index(sample_id_col)
    missing = [sample_id for sample_id in sample_ids if sample_id not in indexed.index]
    if missing:
        raise ValueError(
            f"Inference metadata is missing {len(missing)} profile sample(s); first IDs: {missing[:5]!r}."
        )
    return indexed.loc[sample_ids].reset_index()


def _read_optional_metadata(
    path: Path | None, sample_id_col: str, sample_ids: list[str]
) -> pd.DataFrame | None:
    return _align_profile_metadata(
        _read_profile_metadata(path, sample_id_col), sample_id_col, sample_ids
    )


def _read_matrix_profile(
    path: Path, metadata_sample_ids: list[str] | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    if metadata_sample_ids is None:
        first = _first_nonempty_tsv_row(path)
        if _taxonomic_rank(first[0].lstrip("#").strip()) is not None:
            first_values = pd.to_numeric(pd.Series(first[1:]), errors="coerce")
            if len(first_values) and first_values.notna().all():
                raise ValueError(
                    "Headerless inference abundance matrices require Inference.metadata_path to establish sample IDs."
                )
        frame = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
        frame.index = frame.index.astype(str).str.strip()
        frame.columns = frame.columns.astype(str).str.strip()
    else:
        frame, selected = _read_feature_by_sample_tsv(path, metadata_sample_ids)
        frame = frame.loc[:, selected]
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
        raise ValueError("Inference abundance matrix contains negative abundance values.")
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
    metadata_path = None if plan.metadata_path is None else Path(plan.metadata_path)
    metadata_frame = _read_profile_metadata(metadata_path, sample_id_col)
    if inference_family == "matrix_tsv":
        metadata_ids = (
            None
            if metadata_frame is None
            else metadata_frame[sample_id_col].tolist()
        )
        X, features, sample_ids = _read_matrix_profile(path, metadata_ids)
    else:
        X, features, sample_ids = _read_wide_profile(
            path, sample_id_col, training.feature_names_by_level["all"]
        )
    overlap = sorted(set(sample_ids) & set(training.sample_ids))
    if overlap and not plan.allow_training_sample_overlap:
        raise ValueError(
            f"Inference data overlap the training cohort by {len(overlap)} sample ID(s); first IDs: {overlap[:5]!r}."
        )
    metadata = _align_profile_metadata(metadata_frame, sample_id_col, sample_ids)
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
    stored = model.get("levels")
    if isinstance(stored, (list, tuple)):
        values = tuple(str(value).strip() for value in stored if str(value).strip())
        if values:
            return _normalise_levels(values)
    elif stored is not None:
        text = str(stored).strip()
        if text and text.casefold() not in {"none", "nan"}:
            parsed = None
            if text.startswith("["):
                try:
                    candidate = json.loads(text)
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, list):
                    parsed = tuple(
                        str(value).strip() for value in candidate if str(value).strip()
                    )
            if parsed is None:
                parsed = tuple(
                    value.strip() for value in re.split(r"[,;]", text) if value.strip()
                )
            if parsed:
                return _normalise_levels(parsed)
    resolution = str(model.get("resolution", "")).strip()
    matches = []
    for item in sweep.resolutions:
        name, levels = _parse_resolution(item)
        if str(name) == resolution:
            matches.append(tuple(levels))
    if len(matches) != 1:
        raise ValueError(
            f"Cannot resolve selected resolution {resolution!r}; the persisted model has no usable levels and the current sweep configuration does not contain exactly one matching resolution."
        )
    return tuple(str(level) for level in matches[0])


def _abundance_scale(X: np.ndarray) -> str:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"Abundance-scale detection expects a 2-D matrix, got shape {values.shape!r}."
        )
    if not np.isfinite(values).all():
        raise ValueError("Abundance-scale detection received NaN or infinite values.")
    if np.any(values < 0):
        raise ValueError("Abundance-scale detection received negative values.")
    totals = values.sum(axis=1)
    informative = totals > 0
    if not np.any(informative):
        return "zero"
    informative_values = values[informative]
    informative_totals = totals[informative]
    positive = informative_values[informative_values > 0]
    integer_fraction = float(
        np.mean(np.isclose(positive, np.rint(positive), atol=1e-8))
    )
    proportion_fraction = float(
        np.mean(np.isclose(informative_totals, 1.0, rtol=0.03, atol=0.03))
    )
    percentage_fraction = float(
        np.mean(np.isclose(informative_totals, 100.0, rtol=0.03, atol=2.0))
    )
    if proportion_fraction >= 0.80:
        return "proportion"
    if percentage_fraction >= 0.80:
        return "percentage"
    if float(np.max(informative_values)) <= 1.03 and float(
        np.mean(informative_totals <= 1.03)
    ) >= 0.80:
        return "proportion"
    if integer_fraction >= 0.98 and float(np.max(informative_values)) > 100.0:
        return "counts"
    return "nonnegative_abundance"


def _transformation_base_token(model: dict[str, Any]) -> str:

    identity = str(model.get("count_transformation", "")).strip()
    if not identity:
        return ""
    base_identity, _ = _split_transformation_filter_identity(identity)
    base, _ = _parse_transformation_identity(base_identity)
    return re.sub(r"[^0-9a-z]+", "_", str(base).casefold()).strip("_")


def _scale_validation_mode(
    model: dict[str, Any],
    training_scale: str,
    inference_scale: str,
) -> str:
    if _transformation_base_token(model) == "presence_absence":
        return "skipped_presence_absence"
    verified = {"proportion", "percentage", "counts"}
    if training_scale not in verified or inference_scale not in verified:
        return "inconclusive"
    if training_scale == inference_scale:
        return "matched"
    return "mismatch"


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


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _filter_from_model(
    model: dict[str, Any], filter_identity: str | None
) -> PrevalenceFilter | None:
    if not filter_identity or filter_identity.casefold() == "none":
        return None
    prevalence = _float_or_none(model.get("prevalence_threshold"))
    detection = _float_or_none(model.get("detection_threshold"))
    if prevalence is None:
        match = re.search(
            r"prevalence=([^,]+)(?:,detection=([^,]+))?", str(filter_identity)
        )
        if match is not None:
            prevalence = _float_or_none(match.group(1))
            if detection is None and match.group(2) is not None:
                detection = _float_or_none(match.group(2))
    if prevalence is None:
        raise ValueError(
            f"Cannot reconstruct persisted feature filter {filter_identity!r}."
        )
    return PrevalenceFilter(
        threshold=prevalence,
        detection_threshold=0.0 if detection is None else detection,
    )


def _transformation_item(sweep: Sweep, blocks: Any, model: dict[str, Any]) -> Any:
    identity = str(model.get("count_transformation", "")).strip()
    if not identity:
        raise ValueError("Persisted model has no count_transformation.")
    base_identity, filter_identity = _split_transformation_filter_identity(identity)
    base, scope = _parse_transformation_identity(base_identity)
    if transformation_label(base).category != "custom":
        feature_filter = _filter_from_model(model, filter_identity)
        spec = Transform(
            base,
            composition_scope=scope,
            feature_filter=feature_filter,
        )
        if str(spec.identity) != identity:
            raise ValueError(
                f"Persisted transformation identity {identity!r} reconstructed as {spec.identity!r}."
            )
        return (identity, spec)
    matches = [
        item
        for item in _count_transformation_specs_for_blocks(
            sweep.count_transformations, blocks
        )
        if str(item[0]) == identity
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Persisted custom abundance transformation {identity!r} cannot be reconstructed without its callable in the current sweep configuration."
        )
    return matches[0]


def _import_class(path: str) -> type[Any]:
    parts = str(path).strip().split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        attrs = parts[index:]
        try:
            value: Any = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in attrs:
            value = getattr(value, attr)
        if not isinstance(value, type):
            raise TypeError(f"Persisted learner class {path!r} is not a class.")
        return value
    raise ImportError(f"Cannot import persisted learner class {path!r}.")


def _instantiate_persisted_estimator(cls: type[Any], params: dict[str, Any]) -> Any:
    direct = {key: value for key, value in params.items() if "__" not in key}
    if issubclass(cls, Pipeline):
        steps = direct["steps"]
        direct["steps"] = [(name, estimator) for name, estimator in steps]
        for name, _ in direct["steps"]:
            direct.pop(name, None)
    return cls(**direct)


def _restore_persisted_value(value: Any) -> Any:
    if isinstance(value, str):
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
        return value
    if isinstance(value, list):
        return [_restore_persisted_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value).issuperset({"class", "params"}) and isinstance(
        value.get("params"), dict
    ):
        cls = _import_class(str(value["class"]))
        params = {
            str(key): _restore_persisted_value(item)
            for key, item in value["params"].items()
        }
        return _instantiate_persisted_estimator(cls, params)
    return {str(key): _restore_persisted_value(item) for key, item in value.items()}


def _learner_item(sweep: Sweep, model: dict[str, Any]) -> Any:
    name = str(model.get("learner", "")).strip()
    class_path = str(model.get("learner_class", "")).strip()
    raw_params = model.get("learner_params")
    if class_path and raw_params is not None:
        if isinstance(raw_params, str):
            try:
                parsed = json.loads(raw_params)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Persisted learner parameters for {name!r} are not valid JSON."
                ) from exc
        else:
            parsed = raw_params
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Persisted learner parameters for {name!r} must be a mapping."
            )
        cls = _import_class(class_path)
        params = {
            str(key): _restore_persisted_value(value) for key, value in parsed.items()
        }
        estimator = _instantiate_persisted_estimator(cls, params)
        return (name, estimator)
    matches = [item for item in sweep.learners if _learner_name(item) == name]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot reconstruct persisted learner {name!r}; learner_class/learner_params are unavailable and the current sweep configuration has no unique fallback."
        )
    return matches[0]


def _member_signature(
    dataset: Dataset,
    model: dict[str, Any],
    *,
    random_state: int,
    group_col: str | None,
) -> str:
    groups = _groups_from_metadata(dataset.metadata, group_col)
    return _scientific_digest(
        {
            "schema": _DEPLOYMENT_SCHEMA,
            "package_version": __version__,
            "training_dataset_fingerprint": dataset_fingerprint(dataset),
            "model": model,
            "random_state": int(random_state),
            "group_col": group_col,
            "training_groups": None if groups is None else [str(value) for value in groups],
        }
    )


def _fit_member(
    sweep: Sweep,
    dataset: Dataset,
    profile_X: np.ndarray,
    profile_names: list[str],
    model: dict[str, Any],
    deployment_dir: Path,
    save_weights: bool,
    *,
    input_signature: str | None = None,
) -> tuple[dict[str, Any] | None, np.ndarray, dict[str, Any]]:
    config_id = str(model["config_id"])
    tracker = current_inference_progress()
    if tracker is not None:
        tracker.begin_member(config_id, str(model.get("learner", "")))
    with inference_phase(f"Member {config_id} · validate selected model and cache fingerprint"):
        signature = _member_signature(
            dataset,
            model,
            random_state=int(sweep.evaluation.random_state),
            group_col=sweep.data.group_col,
        )
    if tracker is not None:
        tracker.member_step("fingerprint checked")
    bundle_path = deployment_dir / "members" / f"{config_id}.joblib"
    prediction_path = _prediction_checkpoint_path(deployment_dir, config_id)
    checkpoint_signature = (
        None if input_signature is None else _scientific_digest(
            {"schema": 1, "model": signature, "input": input_signature}
        )
    )
    if checkpoint_signature is not None:
        with inference_phase(f"Member {config_id} · inspect saved external probabilities"):
            cached = _load_prediction_checkpoint(
                prediction_path,
                checkpoint_signature,
                len(profile_X),
                len(dataset.classes) if dataset.task == "classification" else 1,
                require_bundle=save_weights,
                bundle_path=bundle_path,
                bundle_signature=signature,
            )
        if cached is not None:
            info(f"Inference · member {config_id} · reusing verified external prediction checkpoint")
            if tracker is not None:
                tracker.finish_member(config_id)
            return None, cached[0], cached[1]
    bundle = None
    if bundle_path.exists():
        if not _bundle_might_match(bundle_path, signature):
            info(f"Inference · member {config_id} · cache fingerprint changed; skip loading stale model")
        else:
            with inference_phase(f"Member {config_id} · inspect saved deployment bundle"):
                try:
                    candidate = joblib.load(bundle_path)
                    if isinstance(candidate, dict) and candidate.get("signature") == signature:
                        bundle = candidate
                    else:
                        del candidate
                        _release_accelerator_memory(config_id)
                except Exception:
                    bundle = None
                    _release_accelerator_memory(config_id)
    if tracker is not None:
        tracker.member_step("cached bundle inspected")
    levels = _selected_levels(sweep, model)
    info(
        f"Inference · member {config_id} · resolution={model.get('resolution', '')} "
        f"levels={','.join(levels)} · transform={model.get('count_transformation', '')} "
        f"· learner={model.get('learner', '')}"
    )
    with inference_phase(f"Member {config_id} · materialise and align representation"):
        train_X, train_names, blocks = materialize_mpdr_with_blocks(dataset, levels)
        test_raw, test_names = _profile_resolution(profile_X, profile_names, levels)
        test_X, audit = _align_resolution(
            np.asarray(train_X, dtype=np.float64),
            list(train_names),
            test_raw,
            test_names,
            policy=sweep.inference.feature_policy,
        )

    if tracker is not None:
        tracker.member_step("features aligned")
    training_scale = (
        sweep.inference.training_abundance_scale
        or _abundance_scale(np.asarray(dataset.X_by_level["all"], dtype=float))
    )
    inference_scale = (
        sweep.inference.inference_abundance_scale
        or _abundance_scale(np.asarray(profile_X, dtype=float))
    )
    scale_validation = _scale_validation_mode(model, training_scale, inference_scale)
    if scale_validation == "mismatch":
        raise ValueError(
            f"Inference abundance scale differs from training for config {config_id!r}: "
            f"training={training_scale!r}, inference={inference_scale!r}. "
            "Verify that training and external profiles use the same abundance units."
        )
    if bundle is None:
        ct_item = _transformation_item(sweep, blocks, model)
        _, factory = _count_transformation_factory(
            ct_item,
            random_state=int(sweep.evaluation.random_state),
            feature_blocks=blocks,
            feature_names=list(train_names),
        )
        transformer = factory()
        with inference_phase(f"Member {config_id} · fit transformation and transform external profiles"):
            train_transformed, test_transformed = transformer.apply_pair(
                np.asarray(train_X, dtype=np.float64), test_X
            )
        if tracker is not None:
            tracker.member_step("preprocessing fitted")
        learner = _learner_item(sweep, model)
        _, learner_factory = _learner_factory(learner, task=dataset.task)
        estimator = configure_estimator_threads(learner_factory(), 1)
        if callable(getattr(estimator, "bind_schema", None)):
            estimator.bind_schema(transformer.get_feature_names_out(list(train_names)))
        with inference_phase(f"Member {config_id} · refit {model.get('learner', '')} on full training cohort"):
            groups = (
                _groups_from_metadata(dataset.metadata, sweep.data.group_col)
                if dataset.task == "classification" else None
            )
            estimator, effective_device = _fit_estimator_with_kumo_fallback(
                estimator,
                learner_factory,
                train_transformed,
                dataset.y,
                groups,
                dataset.task,
                list(transformer.get_feature_names_out(list(train_names))),
                config_id,
            )
        if tracker is not None:
            tracker.member_step("learner fitted")
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
            "effective_fit_device": effective_device,
        }
        if save_weights:
            with inference_phase(f"Member {config_id} · save fitted model and preprocessing"):
                _atomic_joblib(bundle, bundle_path)
                _write_bundle_signature(bundle_path, signature)
        if tracker is not None:
            tracker.member_step("fitted bundle saved" if save_weights else "bundle persistence skipped")
    else:
        info(f"Inference · member {config_id} · reusing cached fitted model and preprocessing")
        expected_names = [str(value) for value in bundle.get("raw_feature_names", [])]
        if expected_names != list(train_names):
            raise ValueError(
                f"Cached deployment feature schema for {config_id!r} does not match the current training data."
            )
        with inference_phase(f"Member {config_id} · transform external profiles using cached preprocessing"):
            test_transformed = bundle["transformer"].apply(test_X)
        if tracker is not None:
            tracker.member_step("cached preprocessing applied")
            tracker.member_step("cached learner reused")
            tracker.member_step("cached weights reused")
    estimator = bundle["estimator"]
    if dataset.task == "classification":
        with inference_phase(f"Member {config_id} · predict external class probabilities"):
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
        with inference_phase(f"Member {config_id} · predict external outcomes"):
            predictions = np.asarray(
                _estimator_call(estimator, "predict", test_transformed), dtype=float
            ).reshape(-1, 1)
        if not np.isfinite(predictions).all():
            raise ValueError(
                f"Deployment model {config_id!r} produced non-finite predictions."
            )
    if tracker is not None:
        tracker.member_step("external predictions computed")
    audit.update(
        {
            "config_id": config_id,
            "resolution": str(model.get("resolution", "")),
            "levels": ",".join(levels),
            "count_transformation": str(model.get("count_transformation", "")),
            "learner": str(model.get("learner", "")),
            "training_abundance_scale": training_scale,
            "inference_abundance_scale": inference_scale,
            "abundance_scale_validation": scale_validation,
            "deployment_signature": signature,
            "bundle_path": str(bundle_path) if save_weights else None,
            "effective_fit_device": str(bundle.get("effective_fit_device", "unknown")),
        }
    )
    if checkpoint_signature is not None:
        with inference_phase(f"Member {config_id} · checkpoint external predictions"):
            _save_prediction_checkpoint(prediction_path, checkpoint_signature, predictions, audit)
    if tracker is not None:
        tracker.finish_member(config_id)
    return None, predictions, audit


def _strategy_predictions(
    sweep: Sweep,
    dataset: Dataset,
    profile_X: np.ndarray,
    profile_names: list[str],
    models: dict[str, Any],
    deployment_dir: Path,
    save_weights: bool,
    strategy: str,
    member_cache: dict[str, tuple[dict[str, Any] | None, np.ndarray, dict[str, Any]]],
    input_signature: str | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if strategy == "MPMA-B":
        model = models["MPMA-B"]
        config_id = str(model["config_id"])
        if config_id not in member_cache:
            try:
                member_cache[config_id] = _fit_member(
                    sweep,
                    dataset,
                    profile_X,
                    profile_names,
                    model,
                    deployment_dir,
                    save_weights,
                    **({} if input_signature is None else {"input_signature": input_signature}),
                )
            finally:
                _release_accelerator_memory(config_id)
        probabilities = member_cache[config_id][1]
        alpha = float(model.get("probability_softening_alpha", 0.0))
        if alpha > 0.0:
            from .mpma_vnext import soften_probabilities

            probabilities = soften_probabilities(probabilities, alpha)
        return probabilities, [member_cache[config_id][2]]
    unit = models.get("MPMA-E")
    if not isinstance(unit, dict):
        raise ValueError(
            "MPMA-E was requested for inference but no final MPMA-E specification is available."
        )
    member_predictions = []
    audits = []
    for index, member in enumerate(unit["members"], start=1):
        config_id = str(member["config_id"])
        info(f"Inference · MPMA-E member {index}/{len(unit['members'])} · {config_id}")
        if config_id not in member_cache:
            try:
                member_cache[config_id] = _fit_member(
                    sweep,
                    dataset,
                    profile_X,
                    profile_names,
                    member,
                    deployment_dir,
                    save_weights,
                    **({} if input_signature is None else {"input_signature": input_signature}),
                )
            finally:
                _release_accelerator_memory(config_id)
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
    parameters = unit.get("aggregation_parameters", {})
    if weights is not None:
        info(f"Inference · MPMA-E · selected {aggregation} weights={weights!r}")
    elif isinstance(parameters, dict) and "member_weights" in parameters:
        info(f"Inference · MPMA-E · selected {aggregation} weights={parameters['member_weights']!r}")
    else:
        info(f"Inference · MPMA-E · selected aggregation={aggregation}")
    with inference_phase(f"MPMA-E · aggregate {len(member_predictions)} members using {aggregation}"):
        predictions = aggregate_member_predictions(
            stack, aggregation, weights, parameters
        )
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
    with inference_phase("Load and verify final selected model specifications"):
        models = (
            build_final_models(root, include_mpma_e="mpma_e" in sweep.inference.targets)
            if models is None
            else models
        )
        models = hydrate_final_models(root, models)
    tracker = current_inference_progress()
    if tracker is not None:
        tracker.workflow_step("model specifications verified")
    requested = tuple(
        "MPMA-B" if value == "mpma_b" else "MPMA-E" for value in sweep.inference.targets
    )
    if "MPMA-E" in requested and "MPMA-E" not in models:
        raise ValueError(
            "MPMA-E inference was requested but the final ensemble specification is unavailable. Run --stage ensemble first."
        )
    selected_models = [models["MPMA-B"]]
    if "MPMA-E" in requested:
        selected_models.extend(models["MPMA-E"]["members"])
    tracker = current_inference_progress()
    if tracker is not None:
        predicted_models = []
        if "MPMA-B" in requested:
            predicted_models.append(models["MPMA-B"])
        if "MPMA-E" in requested:
            predicted_models.extend(models["MPMA-E"]["members"])
        tracker.plan_models(list(dict.fromkeys(str(model["config_id"]) for model in predicted_models)))
    levels = []
    for model in selected_models:
        levels.extend(
            level
            for level in _selected_levels(sweep, model)
            if level not in _RAW_LEVELS
        )
    with inference_phase("Load full training cohort and selected taxonomic ranks"):
        training = load_dataset(sweep.data, tuple(dict.fromkeys(levels)) or ("all",))
    info(f"Inference · training cohort · {len(training.sample_ids)} samples")
    if tracker is not None:
        tracker.workflow_step("training data loaded")
    with inference_phase("Read external abundance profiles and align metadata"):
        profile_X, profile_names, sample_ids, metadata, input_audit = _profile_for_sweep(
            sweep, training
        )
    info(f"Inference · external cohort · {len(sample_ids)} samples · {len(profile_names)} features")
    if tracker is not None:
        tracker.workflow_step("external profiles aligned")
    inference_root = root / "inference"
    deployment_dir = inference_root / "deployment"
    if sweep.inference.save_weights:
        info("Inference · preparing reusable deployment weights and background")
        deployment_dir.mkdir(parents=True, exist_ok=True)
        background_path = deployment_dir / "background.npz"
        fingerprint = dataset_fingerprint(training)
        replace_background = True
        if background_path.is_file():
            try:
                with np.load(background_path, allow_pickle=False) as existing:
                    replace_background = (
                        str(existing["fingerprint"].item()) != fingerprint
                    )
            except (ValueError, OSError, KeyError):
                pass
        if replace_background:
            rng = np.random.default_rng(int(sweep.evaluation.random_state))
            rows = np.sort(
                rng.choice(
                    len(training.sample_ids),
                    size=min(32, len(training.sample_ids)),
                    replace=False,
                )
            )
            temp = background_path.with_suffix(".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    X=training.X_by_level["all"][rows],
                    names=np.asarray(training.feature_names_by_level["all"], dtype=str),
                    fingerprint=np.asarray(fingerprint),
                )
            temp.replace(background_path)
    inference_root.mkdir(parents=True, exist_ok=True)
    if tracker is not None:
        tracker.workflow_step("deployment storage ready")
    resources_tracker = ResourceTracker().start()
    input_signature = _scientific_digest(
        {
            "schema": 1,
            "profile_sha256": input_audit["abundance_sha256"],
            "sample_ids": sample_ids,
            "feature_names": profile_names,
            "feature_policy": sweep.inference.feature_policy,
            "training_abundance_scale": sweep.inference.training_abundance_scale,
            "inference_abundance_scale": sweep.inference.inference_abundance_scale,
            "class_labels": list(training.class_labels),
            "task": training.task,
        }
    )
    member_cache: dict[
        str, tuple[dict[str, Any] | None, np.ndarray, dict[str, Any]]
    ] = {}
    strategy_values: dict[str, np.ndarray] = {}
    audits: list[dict[str, Any]] = []
    try:
        for index, strategy in enumerate(requested, start=1):
            info(f"Inference · strategy {index}/{len(requested)} · {strategy}")
            with inference_phase(f"Strategy {strategy} · fit or reuse selected learners and predict"):
                values, rows = _strategy_predictions(
                    sweep,
                    training,
                    profile_X,
                    profile_names,
                    models,
                    deployment_dir,
                    sweep.inference.save_weights,
                    strategy,
                    member_cache,
                    input_signature=input_signature,
                )
            strategy_values[strategy] = values
            audits.extend(rows)
    finally:
        resources = resources_tracker.stop()
    if tracker is not None:
        tracker.workflow_step("selected models predicted")
    info("Inference · preparing sample and prediction outputs")
    samples_path = inference_root / "samples.npz"
    temp = samples_path.with_suffix(".tmp")
    with inference_phase("Save external sample matrix for deployment provenance"):
        with temp.open("wb") as handle:
            np.savez_compressed(
                handle,
                X=profile_X,
                names=np.asarray(profile_names, dtype=str),
                samples=np.asarray(sample_ids, dtype=str),
                fingerprint=np.asarray(input_audit["abundance_sha256"]),
            )
        temp.replace(samples_path)
    if tracker is not None:
        tracker.workflow_step("external sample matrix saved")
    predictions = _prediction_frame(sample_ids, training, strategy_values)
    with inference_phase("Verify fitted model provenance for optional conformal prediction"):
        calibration_groups = _groups_from_metadata(training.metadata, sweep.data.group_col)
        model_signature = _scientific_digest(
            {
                "models": models,
                "dataset": dataset_fingerprint(training),
                "random_state": int(sweep.evaluation.random_state),
                "group_col": sweep.data.group_col,
                "training_groups": None if calibration_groups is None else [str(value) for value in calibration_groups],
            }
        )
    with inference_phase("Hash saved fitted learners for deployment provenance"):
        model_artifact_hashes = {
            str(row["config_id"]): _sha256_file(Path(row["bundle_path"]))
            for row in audits
            if row.get("bundle_path") is not None
        }
    conformal_report = None
    if sweep.inference.conformal_alpha is not None:
        if training.task != "classification":
            raise ValueError("Split conformal classification is unavailable for regression.")
        with inference_phase("Construct conformal prediction sets from independent calibration cohort"):
            predictions, conformal_report = external_conformal_sets(
                predictions,
                Path(sweep.inference.conformal_calibration_predictions),
                Path(sweep.inference.conformal_calibration_labels),
                class_labels=list(training.class_labels),
                strategies=requested,
                alpha=float(sweep.inference.conformal_alpha),
                training_ids=training.sample_ids,
                test_ids=sample_ids,
                model_signature=model_signature,
                model_artifact_hashes=model_artifact_hashes,
            )
    if tracker is not None:
        tracker.workflow_step("prediction provenance and calibration verified")
    predictions_path = inference_root / "predictions.tsv"
    with inference_phase("Write external predictions"):
        _atomic_tsv(predictions, predictions_path)
    metadata_predictions_path = None
    if metadata is not None:
        info("Inference · join external metadata with predictions")
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
    info("Inference · write feature alignment and transformation audit")
    _atomic_tsv(feature_audit, feature_audit_path)
    deployment_manifest_path = None
    if sweep.inference.save_weights:
        deployment_manifest = _deployment_manifest(
            sweep,
            training,
            models,
            requested,
            list(unique_audits.values()),
        )
        deployment_manifest_path = deployment_dir / "manifest.json"
        info("Inference · write deployment manifest with selected ensemble parameters")
        dump_json_standard(deployment_manifest, deployment_manifest_path)
    manifest = {
        "schema_version": _INFERENCE_SCHEMA,
        "package_version": __version__,
        "task": training.task,
        "target_name": training.target_name,
        "targets": list(requested),
        "training_samples": len(training.sample_ids),
        "inference_samples": len(sample_ids),
        "model_signature": model_signature,
        "model_artifact_hashes": model_artifact_hashes,
        "training_dataset_fingerprint": dataset_fingerprint(training),
        "conformal": conformal_report,
        "input": input_audit,
        "feature_policy": sweep.inference.feature_policy,
        "save_weights": sweep.inference.save_weights,
        "training_outcomes_used_for_full_fit": True,
        "inference_outcomes_used_for_fit_or_prediction": False,
        "inference_metadata_used_as_predictors": False,
        "predictions": str(predictions_path),
        "samples": str(samples_path),
        "predictions_with_metadata": None
        if metadata_predictions_path is None
        else str(metadata_predictions_path),
        "feature_audit": str(feature_audit_path),
        "deployment_manifest": None
        if deployment_manifest_path is None
        else str(deployment_manifest_path),
        "resources": resources,
        "machine": machine_profile(),
        "python_version": platform.python_version(),
    }
    if tracker is not None:
        tracker.workflow_step("prediction outputs and audits written")
    manifest_path = inference_root / "manifest.json"
    with inference_phase("Write final inference manifest and model provenance"):
        dump_json_standard(manifest, manifest_path)
    if tracker is not None:
        tracker.workflow_step("inference manifest saved")
    info(f"Inference · completed · predictions={predictions_path}")
    outputs = {
        "predictions": predictions_path,
        "feature_audit": feature_audit_path,
        "manifest": manifest_path,
    }
    if deployment_manifest_path is not None:
        outputs["deployment_manifest"] = deployment_manifest_path
    if metadata_predictions_path is not None:
        outputs["predictions_with_metadata"] = metadata_predictions_path
    return outputs
