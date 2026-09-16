from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_SCHEMA_VERSION = 1
_PROBABILITY_PRESERVING = {"mean_proba", "weighted_mean_proba", "median_proba"}
_SUPPORTED_ENSEMBLES = _PROBABILITY_PRESERVING | {
    "rank_mean",
    "majority_vote",
    "max_proba",
    "min_proba",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    if frame.empty:
        raise ValueError(f"Expected non-empty table: {path}")
    return frame


def _parse_members(value: Any) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("members", [])
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("MPMA-E members must be a list or JSON-encoded list")
    out: list[str] = []
    for item in parsed:
        config_id = item.get("config_id") if isinstance(item, dict) else item
        if config_id is not None and str(config_id).strip():
            out.append(str(config_id).strip())
    return list(dict.fromkeys(out))


def _resolve_config_id(value: str, available: list[str]) -> str:
    value = str(value).strip()
    exact = [config_id for config_id in available if config_id == value]
    if len(exact) == 1:
        return exact[0]
    prefix = [config_id for config_id in available if config_id.startswith(value)]
    if len(prefix) == 1:
        return prefix[0]
    if not prefix:
        raise ValueError(f"Unknown config_id {value!r}")
    raise ValueError(f"Ambiguous config_id {value!r}: {prefix}")


def _core_config(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {"config_id": str(row["config_id"])}
    for key in (
        "resolution",
        "levels",
        "count_transformation",
        "transformation_abbreviation",
        "learner",
    ):
        if key in row.index and pd.notna(row[key]) and str(row[key]).strip():
            result[key] = str(row[key])
    return result


def _member_scores(root: Path, member_ids: list[str], metric: str) -> dict[str, float]:
    inner = _read_tsv(root / "inner_results" / "inner_results.tsv")
    if "config_id" not in inner.columns or metric not in inner.columns:
        raise ValueError(f"inner_results.tsv must contain config_id and {metric!r}")
    inner = inner.copy()
    inner["config_id"] = inner["config_id"].astype(str)
    if "ok" in inner.columns:
        inner = inner[pd.to_numeric(inner["ok"], errors="coerce").fillna(0).eq(1)]
    scores = pd.to_numeric(inner[metric], errors="coerce")
    inner = inner.assign(_score=scores)
    means = inner.groupby("config_id")["_score"].mean()
    out: dict[str, float] = {}
    for config_id in member_ids:
        value = means.get(config_id, np.nan)
        if not np.isfinite(value):
            raise ValueError(
                f"No finite {metric} inner-validation score for MPMA-E member {config_id}"
            )
        out[config_id] = float(value)
    return out


def _normalized_weights(scores: list[float]) -> list[float]:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("MPMA-E weights require finite member scores")
    shifted = np.clip(arr - float(np.min(arr)), 0.0, None) + 1e-8
    shifted = shifted / float(np.sum(shifted))
    return [float(value) for value in shifted]


def _build_mpma_b(root: Path, configs: pd.DataFrame) -> dict[str, Any]:
    source = _read_json(root / "tables" / "mpma_b_final_candidate.json")
    config_id = str(source.get("config_id", "")).strip()
    if not config_id:
        raise ValueError("tables/mpma_b_final_candidate.json has no config_id")
    available = configs["config_id"].astype(str).tolist()
    config_id = _resolve_config_id(config_id, available)
    row = configs[configs["config_id"].astype(str).eq(config_id)].iloc[0]
    result = _core_config(row)
    metric = str(
        source.get("selection_metric", source.get("optimize_metric", ""))
    ).strip()
    if metric:
        result["selection_metric"] = metric
    for key in ("score", "inner_validation_score"):
        if key in source:
            try:
                value = float(source[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                result["selection_score"] = value
                break
    return result


def _mpma_e_unit(root: Path) -> dict[str, Any] | None:
    selected_path = root / "ensembling" / "selected_unit.json"
    if not selected_path.exists():
        return None
    selected = _read_json(selected_path)
    unit = selected.get("inner_val_best_mpmas_ensemble", selected.get("MPMA-E"))
    if not isinstance(unit, dict):
        raise ValueError(
            "ensembling/selected_unit.json has no final MPMA-E specification"
        )
    return unit


def _stored_weights(unit: dict[str, Any], member_ids: list[str]) -> list[float] | None:
    raw = unit.get("weights")
    if isinstance(raw, dict):
        values = [raw.get(config_id) for config_id in member_ids]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        return None
    if len(values) != len(member_ids):
        raise ValueError("Stored MPMA-E weights do not match the final member count")
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all() or np.any(arr < 0) or float(arr.sum()) <= 0:
        raise ValueError(
            "Stored MPMA-E weights must be finite, non-negative, and have positive total weight"
        )
    arr = arr / float(arr.sum())
    return [float(value) for value in arr]


def _build_mpma_e(root: Path, configs: pd.DataFrame) -> dict[str, Any] | None:
    unit = _mpma_e_unit(root)
    if unit is None:
        return None
    raw_members = _parse_members(unit.get("members", unit.get("member_config_ids")))
    if not raw_members:
        raise ValueError("Final MPMA-E specification contains no members")
    available = configs["config_id"].astype(str).tolist()
    member_ids = [_resolve_config_id(config_id, available) for config_id in raw_members]
    if len(member_ids) < 2 or len(set(member_ids)) != len(member_ids):
        raise ValueError(
            "Final MPMA-E specification must contain at least two distinct members"
        )
    aggregation = str(unit.get("aggregation_strategy", "")).strip()
    if aggregation not in _SUPPORTED_ENSEMBLES:
        raise ValueError(
            f"Unsupported final MPMA-E aggregation_strategy {aggregation!r}"
        )
    selection_strategy = str(unit.get("selection_strategy", "")).strip()
    metric = (
        str(unit.get("optimize_metric", unit.get("selection_metric", "nMCC"))).strip()
        or "nMCC"
    )
    score_map = _member_scores(root, member_ids, metric)
    weights = None
    if aggregation == "weighted_mean_proba":
        weights = _stored_weights(unit, member_ids)
        if weights is None:
            weights = _normalized_weights(
                [score_map[config_id] for config_id in member_ids]
            )
    members: list[dict[str, Any]] = []
    for index, config_id in enumerate(member_ids):
        row = configs[configs["config_id"].astype(str).eq(config_id)].iloc[0]
        member = _core_config(row)
        member["selection_score"] = score_map[config_id]
        if weights is not None:
            member["weight"] = weights[index]
        members.append(member)
    result: dict[str, Any] = {
        "ensemble_config_id": str(unit.get("ensemble_config_id", "")).strip(),
        "selection_strategy": selection_strategy,
        "aggregation_strategy": aggregation,
        "selection_metric": metric,
        "members": members,
    }
    if "ensemble_size" in unit:
        try:
            result["ensemble_size"] = int(unit["ensemble_size"])
        except (TypeError, ValueError):
            pass
    if "threshold_score" in unit:
        try:
            threshold = float(unit["threshold_score"])
        except (TypeError, ValueError):
            threshold = float("nan")
        if np.isfinite(threshold):
            result["threshold_score"] = threshold
    if not result["ensemble_config_id"]:
        result.pop("ensemble_config_id")
    if not result["selection_strategy"]:
        result.pop("selection_strategy")
    return result


def _validate_outer_predictions(root: Path, models: dict[str, Any]) -> None:
    outer = _read_tsv(root / "predictions" / "outer_predictions.tsv")
    if "config_id" not in outer.columns:
        raise ValueError("outer_predictions.tsv has no config_id")
    available = set(outer["config_id"].astype(str))
    mpma_b = models["MPMA-B"]
    if str(mpma_b["config_id"]) not in available:
        raise ValueError(
            f"Final MPMA-B config_id {mpma_b['config_id']!r} has no outer predictions"
        )
    mpma_e = models.get("MPMA-E")
    if isinstance(mpma_e, dict):
        missing = [
            member["config_id"]
            for member in mpma_e["members"]
            if str(member["config_id"]) not in available
        ]
        if missing:
            raise ValueError(
                f"Final MPMA-E members have no outer predictions: {missing}"
            )


def build_final_models(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    configs = _read_tsv(root / "configs.tsv").copy()
    if "config_id" not in configs.columns:
        raise ValueError("configs.tsv has no config_id")
    configs["config_id"] = configs["config_id"].astype(str)
    if configs["config_id"].duplicated().any():
        duplicates = sorted(
            configs.loc[configs["config_id"].duplicated(keep=False), "config_id"]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"configs.tsv contains duplicate config_id values: {duplicates}"
        )
    models: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "MPMA-B": _build_mpma_b(root, configs),
    }
    mpma_e = _build_mpma_e(root, configs)
    if mpma_e is not None:
        models["MPMA-E"] = mpma_e
    _validate_outer_predictions(root, models)
    path = root / "final_models.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(models, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)
    return models


def load_final_models(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    path = root / "final_models.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Final model artifact does not exist: {path}. Run the ensemble, explain, or report stage first."
        )
    models = _read_json(path)
    if int(models.get("schema_version", -1)) != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported final_models.json schema_version: {models.get('schema_version')!r}"
        )
    if (
        not isinstance(models.get("MPMA-B"), dict)
        or not str(models["MPMA-B"].get("config_id", "")).strip()
    ):
        raise ValueError("final_models.json has no valid MPMA-B specification")
    if "MPMA-E" in models:
        unit = models["MPMA-E"]
        if (
            not isinstance(unit, dict)
            or not isinstance(unit.get("members"), list)
            or not unit["members"]
        ):
            raise ValueError("final_models.json has no valid MPMA-E specification")
    return models


def aggregate_member_predictions(
    stack: np.ndarray, aggregation: str, weights: list[float] | None = None
) -> np.ndarray:
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError(
            "Expected member predictions with shape n_members x n_samples x n_classes"
        )
    method = str(aggregation).strip()
    if method == "mean_proba":
        proba = np.mean(stack, axis=0)
    elif method == "median_proba":
        proba = np.median(stack, axis=0)
    elif method == "weighted_mean_proba":
        if weights is None or len(weights) != stack.shape[0]:
            raise ValueError(
                "weighted_mean_proba requires one stored weight per MPMA-E member"
            )
        weight_array = np.asarray(weights, dtype=float)
        if (
            not np.isfinite(weight_array).all()
            or np.any(weight_array < 0)
            or not np.isclose(weight_array.sum(), 1.0)
        ):
            raise ValueError(
                "Stored MPMA-E weights must be finite, non-negative, and sum to 1"
            )
        proba = np.tensordot(weight_array, stack, axes=(0, 0))
    elif method == "rank_mean":
        ranked = np.empty_like(stack)
        for member_index in range(stack.shape[0]):
            for class_index in range(stack.shape[2]):
                ranked[member_index, :, class_index] = rankdata(
                    stack[member_index, :, class_index]
                )
        proba = np.mean(ranked, axis=0)
    elif method == "majority_vote":
        votes = np.argmax(stack, axis=2)
        proba = np.zeros(stack.shape[1:], dtype=float)
        for sample_index in range(stack.shape[1]):
            counts = np.bincount(
                votes[:, sample_index], minlength=stack.shape[2]
            ).astype(float)
            proba[sample_index] = counts / float(counts.sum())
    elif method == "max_proba":
        proba = np.max(stack, axis=0)
    elif method == "min_proba":
        proba = np.min(stack, axis=0)
    else:
        raise ValueError(f"Unsupported MPMA-E aggregation_strategy {aggregation!r}")
    proba = np.asarray(proba, dtype=float)
    if not np.isfinite(proba).all() or np.any(proba < 0):
        raise ValueError("Aggregated MPMA-E predictions contain invalid values")
    sums = proba.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError(
            "Aggregated MPMA-E predictions contain rows with zero total score"
        )
    return proba / sums


def _outer_predictions(root: Path) -> pd.DataFrame:
    frame = _read_tsv(root / "predictions" / "outer_predictions.tsv").copy()
    if "outer_split_key" not in frame.columns and "split_key" in frame.columns:
        frame["outer_split_key"] = frame["split_key"]
    required = {"outer_split_key", "sample_id", "y_true", "config_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"outer_predictions.tsv is missing required columns: {missing}"
        )
    frame["outer_split_key"] = frame["outer_split_key"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["config_id"] = frame["config_id"].astype(str)
    return frame


def fixed_strategy_predictions(root: Path | str, strategy: str) -> pd.DataFrame:
    root = Path(root)
    models = load_final_models(root)
    outer = _outer_predictions(root)
    strategy = str(strategy)
    if strategy == "MPMA-B":
        config_id = str(models["MPMA-B"]["config_id"])
        out = outer[outer["config_id"].eq(config_id)].copy()
        if out.empty:
            raise ValueError(
                f"Final MPMA-B config_id {config_id!r} has no held-out predictions"
            )
        return out
    if strategy != "MPMA-E":
        raise ValueError(f"Unsupported final strategy {strategy!r}")
    if "MPMA-E" not in models:
        raise ValueError("No final MPMA-E specification is available")
    unit = models["MPMA-E"]
    member_ids = [str(member["config_id"]) for member in unit["members"]]
    aggregation = str(unit["aggregation_strategy"])
    weights = (
        [float(member["weight"]) for member in unit["members"]]
        if aggregation == "weighted_mean_proba"
        else None
    )
    pcols = [column for column in outer.columns if column.startswith("proba_")]
    if not pcols:
        raise ValueError("outer_predictions.tsv has no probability columns")
    rows: list[pd.DataFrame] = []
    selected = outer[outer["config_id"].isin(member_ids)].copy()
    for split_key, split in selected.groupby("outer_split_key", sort=False):
        member_frames: list[pd.DataFrame] = []
        sample_order: list[str] | None = None
        y_true_reference: np.ndarray | None = None
        for config_id in member_ids:
            member = split[split["config_id"].eq(config_id)].copy()
            if member.empty:
                raise ValueError(
                    f"Final MPMA-E member {config_id!r} is missing outer split {split_key!r}"
                )
            if member["sample_id"].duplicated().any():
                raise ValueError(
                    f"Final MPMA-E member {config_id!r} has duplicate samples in outer split {split_key!r}"
                )
            current_samples = member["sample_id"].tolist()
            if sample_order is None:
                sample_order = current_samples
            elif set(current_samples) != set(sample_order):
                raise ValueError(
                    f"Final MPMA-E members have different sample sets in outer split {split_key!r}"
                )
            member = member.set_index("sample_id").reindex(sample_order).reset_index()
            if member[pcols].isna().any().any():
                raise ValueError(
                    f"Final MPMA-E member {config_id!r} has incomplete predictions in outer split {split_key!r}"
                )
            y_true = (
                pd.to_numeric(member["y_true"], errors="raise").astype(int).to_numpy()
            )
            if y_true_reference is None:
                y_true_reference = y_true
            elif not np.array_equal(y_true_reference, y_true):
                raise ValueError(
                    f"Final MPMA-E members disagree on y_true in outer split {split_key!r}"
                )
            member_frames.append(member)
        stack = np.stack(
            [frame[pcols].to_numpy(dtype=float) for frame in member_frames], axis=0
        )
        proba = aggregate_member_predictions(stack, aggregation, weights)
        base = member_frames[0].copy()
        drop_columns = [
            column for column in base.columns if column.startswith("proba_")
        ]
        drop_columns.extend(
            [
                "config_id",
                "y_pred",
                "learner",
                "resolution",
                "levels",
                "count_transformation",
                "transformation_abbreviation",
                "mpdr_id",
            ]
        )
        base = base.drop(
            columns=[column for column in drop_columns if column in base.columns],
            errors="ignore",
        )
        base["y_pred"] = np.argmax(proba, axis=1).astype(int)
        for index, column in enumerate(pcols):
            base[column] = proba[:, index]
        if len(pcols) == 2:
            base["y_proba_pos"] = proba[:, 1]
        base["ensemble_config_id"] = str(unit.get("ensemble_config_id", ""))
        base["aggregation_strategy"] = aggregation
        base["members"] = json.dumps(member_ids)
        base["probability_valid"] = aggregation in _PROBABILITY_PRESERVING
        rows.append(base)
    if not rows:
        raise ValueError(
            "No held-out predictions could be reconstructed for final MPMA-E"
        )
    return pd.concat(rows, ignore_index=True)
