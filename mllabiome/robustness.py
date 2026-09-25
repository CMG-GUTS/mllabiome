from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .console import path_table, stage, success
from .data import load_dataset, metadata_spec
from .metrics import compute_metrics, compute_regression_metrics
from .modalities import load_modalities
from .storage import read_table, table_exists, write_table
from .utils import dump_json_standard


def _metadata(sweep: Any) -> tuple[pd.DataFrame, str, str]:
    if getattr(sweep, "uses_modalities", False):
        dataset = load_modalities(sweep.samples, sweep.modalities)
        frame = dataset.metadata.copy()
        sample_col = str(sweep.samples.sample_id_col)
        target = str(dataset.target_name)
    else:
        dataset = load_dataset(sweep.data, levels_needed=("all",))
        frame = dataset.metadata.copy()
        sample_col = str(sweep.data.sample_id_col)
        target = str(dataset.target_name)
    frame[sample_col] = frame[sample_col].astype(str)
    y = pd.DataFrame({sample_col: dataset.sample_ids, "__y__": dataset.y})
    frame = frame.merge(y, on=sample_col, how="inner", validate="one_to_one")
    return frame, sample_col, target


def _metadata_schema(sweep: Any) -> Any:
    if (
        getattr(sweep, "uses_modalities", False)
        or getattr(sweep, "data", None) is None
    ):
        return None
    return metadata_spec(sweep.data)


def _metadata_inventory(metadata: pd.DataFrame, schema: Any) -> pd.DataFrame:
    if schema is None:
        return pd.DataFrame()
    declarations: dict[str, dict[str, set[str]]] = {}
    for role, column in schema.semantic_mapping.items():
        category = "technical" if role in {"site", "batch"} else "covariate"
        entry = declarations.setdefault(
            column, {"roles": set(), "categories": set()}
        )
        entry["roles"].add(role)
        entry["categories"].add(category)
    for column in schema.covariates:
        entry = declarations.setdefault(
            column, {"roles": set(), "categories": set()}
        )
        entry["roles"].add("covariate")
        entry["categories"].add("covariate")
    for column in schema.technical:
        entry = declarations.setdefault(
            column, {"roles": set(), "categories": set()}
        )
        entry["roles"].add("technical")
        entry["categories"].add("technical")
    rows: list[dict[str, Any]] = []
    for column, declaration in declarations.items():
        if column not in metadata.columns:
            raise ValueError(
                f"Declared metadata column {column!r} is not present in metadata."
            )
        values = metadata[column]
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(values) else 0.0
        kind = (
            "numeric"
            if numeric_fraction >= 0.95 and numeric.nunique(dropna=True) > 2
            else "categorical"
        )
        rows.append(
            {
                "role": ", ".join(sorted(declaration["roles"])),
                "category": ", ".join(sorted(declaration["categories"])),
                "column": column,
                "kind": kind,
                "n": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "missing_fraction": (
                    float(values.isna().mean()) if len(values) else np.nan
                ),
                "unique": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _numeric_balance(values: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    groups = [numeric[y.eq(level)].dropna().to_numpy(dtype=float) for level in sorted(y.dropna().unique())]
    groups = [group for group in groups if group.size]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    means = [float(np.mean(group)) for group in groups]
    sds = [float(np.std(group, ddof=1)) if len(group) > 1 else 0.0 for group in groups]
    pooled = float(np.sqrt(np.mean(np.square(sds))))
    effect = float((max(means) - min(means)) / pooled) if pooled > 0 else np.nan
    return effect, min(means), max(means)


def _categorical_balance(values: pd.Series, y: pd.Series) -> tuple[float, str, str]:
    frame = pd.DataFrame({"value": values.astype("string"), "y": y}).dropna()
    if frame.empty or frame["y"].nunique() < 2:
        return np.nan, "", ""
    table = pd.crosstab(frame["y"], frame["value"], normalize="index")
    if table.empty:
        return np.nan, "", ""
    spread = table.max(axis=0) - table.min(axis=0)
    level = str(spread.idxmax())
    return float(spread.max()), level, json.dumps({str(index): float(value) for index, value in table[level].items()}, sort_keys=True)


def _balance_table(metadata: pd.DataFrame, columns: tuple[str, ...], technical: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y = metadata["__y__"]
    for column in columns:
        if column not in metadata.columns:
            raise ValueError(f"Robustness covariate {column!r} is not present in metadata.")
        values = metadata[column]
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(values) else 0.0
        if numeric_fraction >= 0.95 and numeric.nunique(dropna=True) > 2:
            effect, minimum, maximum = _numeric_balance(values, y)
            rows.append({
                "covariate": column,
                "role": "technical" if column in technical else "covariate",
                "kind": "numeric",
                "n": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "imbalance": effect,
                "group_minimum_mean": minimum,
                "group_maximum_mean": maximum,
                "most_imbalanced_level": "",
                "level_proportions": "",
            })
        else:
            effect, level, proportions = _categorical_balance(values, y)
            rows.append({
                "covariate": column,
                "role": "technical" if column in technical else "covariate",
                "kind": "categorical",
                "n": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "imbalance": effect,
                "group_minimum_mean": np.nan,
                "group_maximum_mean": np.nan,
                "most_imbalanced_level": level,
                "level_proportions": proportions,
            })
    return pd.DataFrame(rows)


def _prediction_sources(root: Path, targets: tuple[str, ...]) -> list[tuple[str, Path]]:
    sources = {
        "mpma_b": root / "predictions" / "mpma_b_outer_predictions.parquet",
        "mpma_e": root / "ensembling" / "ensemble_predictions.parquet",
    }
    return [(target, sources[target]) for target in targets if target in sources and table_exists(sources[target])]


def _subgroup_metrics(predictions: pd.DataFrame, metadata: pd.DataFrame, sample_col: str, subgroup: str, strategy: str, minimum: int) -> pd.DataFrame:
    if subgroup not in metadata.columns:
        raise ValueError(f"Robustness subgroup {subgroup!r} is not present in metadata.")
    joined = predictions.merge(metadata[[sample_col, subgroup]], left_on="sample_id", right_on=sample_col, how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    pcols = sorted([column for column in joined.columns if str(column).startswith("proba_")])
    for level, group in joined.dropna(subset=[subgroup]).groupby(subgroup, sort=True):
        sample_count = int(group["sample_id"].nunique())
        if sample_count < minimum:
            continue
        if pcols:
            y_true = pd.to_numeric(group["y_true"], errors="raise").astype(int).to_numpy()
            y_pred = pd.to_numeric(group["y_pred"], errors="raise").astype(int).to_numpy()
            proba = group[pcols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            metrics = compute_metrics(y_true, y_pred, proba, np.arange(len(pcols), dtype=int))
        else:
            y_true = pd.to_numeric(group["y_true"], errors="raise").to_numpy(dtype=float)
            y_pred = pd.to_numeric(group["y_pred"], errors="raise").to_numpy(dtype=float)
            metrics = compute_regression_metrics(y_true, y_pred)
        rows.append({"strategy": strategy.upper().replace("_", "-"), "subgroup": subgroup, "level": str(level), "n_samples": sample_count, "n_predictions": int(len(group)), **metrics})
    return pd.DataFrame(rows)


def _explained_unit(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _feature_robustness(root: Path, targets: tuple[str, ...], top_k: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for target in targets:
        directory = root / "explainability" / target
        path = directory / "feature_stability.parquet"
        if not table_exists(path):
            continue
        frame = read_table(path).copy()
        if frame.empty:
            continue
        unit = _explained_unit(directory / "explained_unit.json")
        model_id = ""
        if target == "mpma_b":
            config = unit.get("config", {}) if isinstance(unit.get("config"), dict) else {}
            model_id = str(config.get("config_id", ""))
        else:
            model_id = str(unit.get("ensemble_config_id", ""))
        frame.insert(0, "strategy", target.upper().replace("_", "-"))
        frame.insert(1, "model_or_ensemble_id", model_id)
        frame.insert(2, "explainability_scope", "outer_test_folds")
        if "rank_median" in frame.columns:
            frame = frame.sort_values([column for column in ("class_index", "rank_median", "importance_mean") if column in frame.columns], ascending=True, kind="mergesort")
        elif "importance_mean" in frame.columns:
            frame = frame.sort_values("importance_mean", ascending=False, kind="mergesort")
        group_cols = [column for column in ("class_index", "class_label") if column in frame.columns]
        if group_cols:
            frame = frame.groupby(group_cols, group_keys=False, sort=False).head(top_k)
        else:
            frame = frame.head(top_k)
        frames.append(frame.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_robustness(sweep: Any) -> dict[str, Path]:
    config = getattr(sweep, "robustness", None)
    if config is None:
        return {}
    root = Path(sweep.root())
    out_dir = root / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage("Robustness", str(out_dir))
    metadata, sample_col, target = _metadata(sweep)
    schema = _metadata_schema(sweep)
    outputs: dict[str, Path] = {}
    inventory = _metadata_inventory(metadata, schema)
    if not inventory.empty:
        path = out_dir / "metadata_inventory.parquet"
        write_table(path, inventory)
        outputs["metadata_inventory"] = path
    automatic_covariates = schema.covariate_columns if schema is not None else ()
    automatic_technical = schema.technical_columns if schema is not None else ()
    covariate_columns = tuple(
        dict.fromkeys((*automatic_covariates, *config.covariates))
    )
    technical_columns = tuple(
        dict.fromkeys((*automatic_technical, *config.technical))
    )
    balance_columns = tuple(dict.fromkeys((*covariate_columns, *technical_columns)))
    if balance_columns:
        balance = _balance_table(metadata, balance_columns, set(technical_columns))
        path = out_dir / "covariate_balance.parquet"
        write_table(path, balance)
        outputs["covariate_balance"] = path
    subgroup_frames = []
    automatic_subgroups = (
        schema.categorical_subgroup_columns if schema is not None else ()
    )
    subgroup_columns = tuple(dict.fromkeys((*automatic_subgroups, *config.subgroups)))
    for strategy, path in _prediction_sources(root, config.targets):
        predictions = read_table(path)
        for subgroup in subgroup_columns:
            result = _subgroup_metrics(
                predictions,
                metadata,
                sample_col,
                subgroup,
                strategy,
                config.min_subgroup_size,
            )
            if not result.empty:
                subgroup_frames.append(result)
    if subgroup_frames:
        subgroup = pd.concat(subgroup_frames, ignore_index=True)
        path = out_dir / "subgroup_performance.parquet"
        write_table(path, subgroup)
        outputs["subgroup_performance"] = path
    features = _feature_robustness(root, config.targets, config.top_k)
    if not features.empty:
        path = out_dir / "important_feature_robustness.parquet"
        write_table(path, features)
        outputs["important_feature_robustness"] = path
    manifest = out_dir / "robustness_manifest.json"
    dump_json_standard({
        "schema_version": 2,
        "target": target,
        "targets": list(config.targets),
        "metadata": schema.semantic_mapping if schema is not None else {},
        "covariates": list(covariate_columns),
        "technical": list(technical_columns),
        "subgroups": list(subgroup_columns),
        "top_k": int(config.top_k),
        "min_subgroup_size": int(config.min_subgroup_size),
        "feature_provenance": "strategy-specific outer-test explainability only",
        "outputs": outputs,
    }, manifest)
    outputs["manifest"] = manifest
    success("Robustness completed")
    path_table("Robustness outputs", outputs)
    return outputs
