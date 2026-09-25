from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .configs_sweep import _source_tree_sha256
from .explainability_config import (
    _EXPLAINABILITY_PIPELINE_SCHEMA,
    _normalise_explainability_method_specs,
)
from .explainability_methods import (
    method_has_global,
    method_has_local,
    method_name,
    method_to_dict,
)
from .explainability_reporting import _class_slug, _plot_feature_importance
from .storage import glob_tables, read_table, table_exists
from .sweep_types import Sweep, _effective_local_explanations_mode
from .utils import dump_json_standard


def _explainability_config_payload(explainability: Any) -> dict[str, Any]:
    return {
        "profile": str(getattr(explainability, "profile", "standard")),
        "methods": [
            method_to_dict(x)
            for x in _normalise_explainability_method_specs(explainability.methods)
        ],
        "classes": explainability.classes,
        "top_k": int(explainability.top_k),
        "random_state": int(explainability.random_state),
        "local": {
            "representatives": bool(explainability.local.representatives),
            "sample_ids": list(explainability.local.sample_ids),
            "stored_features": int(explainability.local.stored_features),
            "displayed_features": int(explainability.local.displayed_features),
            "regression_quantiles": [
                float(x) for x in explainability.local.regression_quantiles
            ],
        },
        "local_explanations": _effective_local_explanations_mode(explainability),
        "effective_local_explanations": _effective_local_explanations_mode(
            explainability
        ),
        "representative_instances": bool(explainability.local.representatives),
        "instance_sample_ids": list(explainability.local.sample_ids),
        "local_top_k": int(explainability.local.stored_features),
        "top_instance_features": int(explainability.local.displayed_features),
        "representative_quantiles": [
            float(x) for x in explainability.local.regression_quantiles
        ],
    }


def _explainability_config_signature(explainability: Any) -> str:
    payload = _explainability_config_payload(explainability)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _signature_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_signature_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_signature_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _signature_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _signature_hash(payload: Any) -> str:
    raw = json.dumps(
        _signature_value(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _explainability_source_signature(
    target_slug: str,
    row: pd.Series,
    oof_folds: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    class_labels: Sequence[Any],
) -> str:
    config_keys = (
        "unit",
        "config_id",
        "resolution",
        "levels",
        "count_transformation",
        "learner",
        "members",
        "aggregation_strategy",
    )
    config = {key: row.get(key) for key in config_keys if key in row.index}
    return _signature_hash(
        {
            "source_tree_sha256": _source_tree_sha256(),
            "target": str(row.get("unit", row.get("config_id", target_slug))),
            "config": config,
            "features": list(feature_names),
            "classes": [str(x) for x in class_labels],
            "folds": [
                {
                    "split_key": str(fold.get("split_key", "")),
                    "train_idx": np.asarray(fold.get("train_idx", []), dtype=int),
                    "test_idx": np.asarray(fold.get("test_idx", []), dtype=int),
                }
                for fold in oof_folds
            ],
        }
    )


def _method_cache_signature(
    explainability: Any,
    spec: Any,
    source_signature: str,
    class_indices: Sequence[int],
) -> str:
    name = method_name(spec)
    payload: dict[str, Any] = {
        "method": name,
        "parameters": method_to_dict(spec),
        "source_signature": str(source_signature),
        "classes": [int(x) for x in class_indices],
        "random_state": int(explainability.random_state),
    }
    return _signature_hash(payload)


def _method_cache_files_complete(target_dir: Path, method: str) -> bool:
    if method == "interactions":
        return table_exists(target_dir / "feature_interactions_current.parquet")
    required = [
        target_dir / f"feature_importance_{method}.parquet",
        target_dir / f"feature_stability_{method}.parquet",
        target_dir / f"feature_importance_{method}_by_outer_fold.parquet",
        target_dir / f"top_features_{method}.parquet",
    ]
    return all(table_exists(path) for path in required)


def _source_config_compatible(meta: dict[str, Any], row: pd.Series) -> bool:
    old = meta.get("config", {})
    keys = (
        "unit",
        "config_id",
        "resolution",
        "levels",
        "count_transformation",
        "learner",
        "members",
        "aggregation_strategy",
    )
    compared = False
    for key in keys:
        if key not in old or key not in row.index:
            continue
        a = _signature_value(old.get(key))
        b = _signature_value(row.get(key))
        if a is None or b is None:
            continue
        compared = True
        if a != b:
            return False
    return compared


def _legacy_method_cache_valid(
    meta: dict[str, Any],
    row: pd.Series,
    explainability: Any,
    spec: Any,
    class_indices: Sequence[int],
) -> bool:
    if not meta or not _source_config_compatible(meta, row):
        return False
    old_classes = [int(x) for x in meta.get("explained_class_indices", [])]
    if old_classes != [int(x) for x in class_indices]:
        return False
    params = meta.get("method_parameters", [])
    names = [str(x).strip().lower() for x in meta.get("methods", [])]
    current = _signature_value(method_to_dict(spec))
    matched = False
    for index, item in enumerate(params):
        try:
            old_name = names[index] if index < len(names) else ""
            if not old_name:
                keys = set(item)
                if {"algorithm", "masker"}.issubset(keys):
                    old_name = "shap"
                elif {"n_repeats", "scoring", "max_samples"}.issubset(keys):
                    old_name = "permutation"
                elif {"num_samples", "feature_selection"}.issubset(keys):
                    old_name = "lime"
                elif "top_k" in keys and "bins" in keys:
                    old_name = "interactions"
                elif "bins" in keys:
                    old_name = "ale"
            if old_name == method_name(spec) and _signature_value(item) == current:
                matched = True
                break
        except Exception:
            continue
    if not matched:
        return False
    old_cfg = meta.get("explainability_config", {})
    if int(old_cfg.get("random_state", explainability.random_state)) != int(
        explainability.random_state
    ):
        return False
    return True


def _method_cache_entry_status(
    entry: dict[str, Any],
    signature: str,
    source_signature: str,
    spec: Any,
    class_indices: Sequence[int],
) -> tuple[bool, str]:
    if not entry:
        return False, "no per-method provenance"
    if str(entry.get("source_signature", "")) != str(source_signature):
        return False, "selected model/data/splits changed"
    if _signature_value(entry.get("parameters", {})) != _signature_value(
        method_to_dict(spec)
    ):
        return False, "method parameters changed"
    if [int(x) for x in entry.get("class_indices", [])] != [
        int(x) for x in class_indices
    ]:
        return False, "explained classes changed"
    if str(entry.get("signature", "")) != str(signature):
        return False, "method random state or cache schema changed"
    return True, "cache hit"


def _method_cache_entry(
    signature: str,
    source_signature: str,
    spec: Any,
    class_indices: Sequence[int],
    top_k: int,
) -> dict[str, Any]:
    return {
        "cache_version": 4,
        "signature": str(signature),
        "source_signature": str(source_signature),
        "parameters": method_to_dict(spec),
        "class_indices": [int(x) for x in class_indices],
        "top_k": int(top_k),
    }


def _method_cache_sidecar_path(target_dir: Path, name: str) -> Path:
    return target_dir / ".method_cache" / f"{_safe_cache_name(name)}.json"


def _load_method_cache_entry(
    target_dir: Path,
    name: str,
    shared_cache: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = _method_cache_sidecar_path(target_dir, name)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return payload, "sidecar"
        except Exception:
            pass
    entry = shared_cache.get(str(name), {})
    if isinstance(entry, dict) and entry:
        return dict(entry), "shared metadata"
    return {}, "none"


def _persist_method_cache_entry(
    meta_path: Path,
    previous_meta: dict[str, Any],
    name: str,
    entry: dict[str, Any],
) -> None:
    sidecar = _method_cache_sidecar_path(meta_path.parent, name)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    dump_json_standard(dict(entry), sidecar)
    current = dict(previous_meta)
    if meta_path.exists():
        try:
            disk = json.loads(meta_path.read_text())
            if isinstance(disk, dict):
                current.update(disk)
        except Exception:
            pass
    cache = dict(current.get("method_cache", {}))
    cache[str(name)] = dict(entry)
    current["method_cache"] = cache
    dump_json_standard(current, meta_path)
    previous_meta.clear()
    previous_meta.update(current)


def _cached_method_frame(
    target_dir: Path, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_table(target_dir / f"feature_importance_{method}.parquet")
    fold = read_table(target_dir / f"feature_importance_{method}_by_outer_fold.parquet")
    return frame, fold


def _existing_method_outputs(target_dir: Path, method: str) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    mapping = {
        f"importance_{method}": target_dir / f"feature_importance_{method}.parquet",
        f"stability_{method}": target_dir / f"feature_stability_{method}.parquet",
        f"top_features_{method}": target_dir / f"top_features_{method}.parquet",
        f"{method}_by_outer_fold": target_dir
        / f"feature_importance_{method}_by_outer_fold.parquet",
    }
    for key, path in mapping.items():
        if table_exists(path):
            outputs[key] = path
    for path in glob_tables(target_dir, f"feature_distribution_stats_{method}__*"):
        outputs[path.stem] = path
    figures = target_dir / "figures"
    for pattern in (f"feature_importance_{method}__*.svg",):
        for path in sorted(figures.glob(pattern)):
            outputs[path.stem] = path
    if method == "ale":
        curve_table = target_dir / "ale_curves.parquet"
        if table_exists(curve_table):
            outputs["ale_curves"] = curve_table
        for path in sorted(figures.glob("ale_curves__*.svg")):
            outputs[path.stem] = path
    if method == "interactions":
        for path in glob_tables(target_dir, "feature_interactions_*"):
            outputs[path.stem] = path
        for path in sorted(figures.glob("interaction_network_*.svg")):
            outputs[path.stem] = path
    if method in {"shap", "lime"}:
        path = target_dir / f"instance_explanations_{method}_top_features.parquet"
        if table_exists(path):
            outputs[f"instance_explanations_{method}"] = path
    return outputs


def _visual_class_labels(sweep: Sweep, target_dir: Path) -> tuple[str, ...]:
    labels = getattr(sweep.data, "class_labels", None)
    if labels is not None:
        try:
            values = tuple(str(x) for x in labels)
        except TypeError:
            values = ()
        if values:
            return values
    label_map = getattr(sweep.data, "label_map", None)
    if isinstance(label_map, dict) and label_map:
        try:
            return tuple(
                str(value)
                for _, value in sorted(label_map.items(), key=lambda item: int(item[0]))
            )
        except Exception:
            return tuple(str(value) for value in label_map.values())
    meta_path = target_dir / "explained_unit.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        labels = (
            meta.get("explained_class_labels", []) if isinstance(meta, dict) else []
        )
        if isinstance(labels, list) and labels:
            return tuple(str(x) for x in labels)
    return ()


def _refresh_target_visuals(target_dir: Path, sweep: Sweep) -> dict[str, Path]:
    if not target_dir.exists():
        return {}
    class_labels = _visual_class_labels(sweep, target_dir)
    top_k = int(getattr(sweep.explainability, "top_k", 15))
    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    def render(
        top_path: Path, stats_path_for: Callable[[str], Path], stem_prefix: str
    ) -> None:
        if not table_exists(top_path):
            return
        try:
            top = read_table(top_path)
        except Exception:
            return
        if top.empty or "feature" not in top.columns:
            return
        if "class_index" in top.columns:
            groups = list(top.groupby("class_index", sort=True))
        else:
            groups = [(0, top)]
        for class_index, class_top in groups:
            try:
                c = int(class_index)
            except Exception:
                c = 0
            if class_labels and 0 <= c < len(class_labels):
                label = class_labels[c]
            elif "class_label" in class_top.columns and not class_top.empty:
                label = str(class_top.iloc[0].get("class_label", f"class_{c}"))
            else:
                label = f"class_{c}"
            slug = _class_slug(label)
            stats_path = stats_path_for(slug)
            if not table_exists(stats_path):
                continue
            try:
                stats = read_table(stats_path)
            except Exception:
                continue
            if "class_index" in stats.columns:
                selected = stats[
                    pd.to_numeric(stats["class_index"], errors="coerce").eq(c)
                ]
                if not selected.empty:
                    stats = selected
            kinds = ("feature_support",) if not stem_prefix else ("feature_importance",)
            for kind in kinds:
                stem = figures_dir / f"{kind}{stem_prefix}__{slug}"
                _plot_feature_importance(
                    class_top, stats, stem, top_k, class_labels or (label,)
                )
                path = stem.with_suffix(".svg")
                if path.exists():
                    outputs[path.stem + "_svg"] = path

    render(
        target_dir / "top_features.parquet",
        lambda slug: target_dir / "feature_distribution_stats.parquet",
        "",
    )
    for top_path in glob_tables(target_dir, "top_features_*"):
        method = top_path.stem[len("top_features_") :]
        if not method:
            continue
        render(
            top_path,
            lambda slug, method=method: (
                target_dir / f"feature_distribution_stats_{method}__{slug}.parquet"
            ),
            f"_{method}",
        )
    return outputs


def refresh_explainability_visuals(root: Path | str, sweep: Sweep) -> dict[str, Path]:
    exp_root = Path(root) / "explainability"
    outputs: dict[str, Path] = {}
    if not exp_root.exists():
        return outputs
    for slug in ("mpma_b", "mpma_e", "baseline_rf"):
        target_dir = exp_root / slug
        for key, path in _refresh_target_visuals(target_dir, sweep).items():
            outputs[f"{slug}_{key}"] = path
    return outputs


def _safe_cache_name(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _explainability_cache_complete(
    target_dir: Path, explainability: Any, expected_config_id: str | None = None
) -> bool:
    meta_path = target_dir / "explained_unit.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if str(meta.get("pipeline_schema", "")) != _EXPLAINABILITY_PIPELINE_SCHEMA:
        return False
    if str(
        meta.get("explainability_config_signature", "")
    ) != _explainability_config_signature(explainability):
        return False
    if expected_config_id is not None:
        config = meta.get("config")
        if not isinstance(config, dict) or str(config.get("config_id", "")) != str(
            expected_config_id
        ):
            return False
    specs = _normalise_explainability_method_specs(explainability.methods)
    global_methods = {method_name(spec) for spec in specs if method_has_global(spec)}
    local_methods = {method_name(spec) for spec in specs if method_has_local(spec)}
    if global_methods:
        if not (target_dir / "perturbation_policy.json").exists():
            return False
        if not table_exists(target_dir / "feature_importance.parquet"):
            return False
        if not table_exists(target_dir / "top_features.parquet"):
            return False
        figs = target_dir / "figures"
        if not (
            (figs / "feature_support.svg").exists()
            or (figs / "feature_support.png").exists()
            or list(figs.glob("feature_support__*.svg"))
            or list(figs.glob("feature_support__*.png"))
        ):
            return False
        for method in ("shap", "lime", "ale", "permutation"):
            if method in global_methods and not table_exists(
                target_dir / f"feature_importance_{method}.parquet"
            ):
                return False
        if "interactions" in global_methods and not any(
            (target_dir / name).exists()
            for name in (
                "feature_interactions_current.parquet",
                "feature_interactions_corrected.parquet",
                "feature_interactions_fixed_pairs.parquet",
                "feature_interactions_corrected_fixed.parquet",
            )
        ):
            return False
    if local_methods and _effective_local_explanations_mode(explainability) != "none":
        if not table_exists(target_dir / "local_explanations.parquet"):
            return False
    return True


def _existing_explainability_outputs(target_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {"explainability_dir": target_dir}
    candidates = {
        "importance": target_dir / "feature_importance.parquet",
        "stability": target_dir / "feature_stability.parquet",
        "local_explanations": target_dir / "local_explanations.parquet",
        "local_cohort_context": target_dir / "local_cohort_context.parquet",
        "local_explanations_figure": target_dir / "figures" / "local_explanations.svg",
        "perturbation_policy": target_dir / "perturbation_policy.json",
        "coordinate_metadata": target_dir / "coordinate_metadata.parquet",
        "coordinate_metadata_by_outer_fold": target_dir
        / "coordinate_metadata_by_outer_fold.parquet",
        "prediction_reproduction": target_dir / "prediction_reproduction.parquet",
    }
    for key, path in candidates.items():
        if path.exists():
            outputs[key] = path
    return outputs


def _copy_explainability_cache(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
