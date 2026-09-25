from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .console import path_table, stage, success
from .data import load_dataset, metadata_spec
from .metrics import compute_metrics, compute_regression_metrics
from .modalities import load_modalities
from .oof_statistics import _calibration_binary
from .statistics_common import _LODO_PROTOCOLS, _repeat_id, _stable_seed
from .storage import read_table, table_exists, write_table
from .utils import CLASSIFICATION_METRIC_COLUMNS, REGRESSION_METRIC_COLUMNS, dump_json_standard

_CALIBRATION_METRICS = (
    "CalibrationInTheLarge",
    "CalibrationIntercept",
    "CalibrationSlope",
    "CalibrationInTheLarge_macro_OvR",
    "CalibrationIntercept_macro_OvR",
    "CalibrationSlope_macro_OvR",
)

_CLASSIFICATION_ROBUSTNESS_METRICS = tuple(
    metric
    for metric in CLASSIFICATION_METRIC_COLUMNS
    if metric not in {"subject_macro_log_loss", "cohort_macro_log_loss"}
) + _CALIBRATION_METRICS

_BINARY_CONTRAST_METRICS = ("AUROC", "AP", "log_loss", "brier", "MCC", "BalAcc")
_MULTICLASS_CONTRAST_METRICS = (
    "AUROC_macro",
    "AP_macro",
    "log_loss",
    "brier",
    "MCC",
    "BalAcc",
)
_REGRESSION_CONTRAST_METRICS = ("R2", "MAE", "RMSE", "SpearmanR")


def _metadata(sweep: Any) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if getattr(sweep, "uses_modalities", False):
        dataset = load_modalities(sweep.samples, sweep.modalities)
        frame = dataset.metadata.copy()
        spec = sweep.samples
        sample_col = str(spec.sample_id_col)
    else:
        dataset = load_dataset(sweep.data, levels_needed=("all",))
        frame = dataset.metadata.copy()
        spec = sweep.data
        sample_col = str(spec.sample_id_col)
    frame[sample_col] = frame[sample_col].astype(str)
    sample_ids = [str(value) for value in dataset.sample_ids]
    subject_ids = [str(value) for value in dataset.subject_ids]
    annotation = pd.DataFrame(
        {
            sample_col: sample_ids,
            "__subject_id__": subject_ids,
            "__y__": dataset.y,
        }
    )
    if str(dataset.task) == "classification":
        labels = [str(value) for value in dataset.class_labels]
        annotation["__y_label__"] = [labels[int(value)] for value in dataset.y]
    frame = frame.merge(annotation, on=sample_col, how="inner", validate="one_to_one")
    subject_declared = bool(getattr(spec, "subject_id_col", None))
    group_col = getattr(spec, "group_col", None)
    cluster_source = "subject_id" if subject_declared else "sample_id"
    frame["__cluster_id__"] = frame["__subject_id__"].astype(str)
    if not subject_declared and group_col is not None and str(group_col) in frame.columns:
        values = frame[str(group_col)]
        if values.notna().all() and values.astype(str).nunique() < len(frame):
            frame["__cluster_id__"] = values.astype(str)
            cluster_source = f"group_col:{group_col}"
    return frame, sample_col, {
        "target": str(dataset.target_name),
        "task": str(dataset.task),
        "class_labels": tuple(str(value) for value in dataset.class_labels),
        "positive_class": dataset.positive_class,
        "subject_id_declared": subject_declared,
        "cluster_source": cluster_source,
        "protocol": str(sweep.evaluation.protocol),
    }


def _metadata_schema(sweep: Any) -> Any:
    if getattr(sweep, "uses_modalities", False) or getattr(sweep, "data", None) is None:
        return None
    return metadata_spec(sweep.data)


def _metadata_inventory(metadata: pd.DataFrame, schema: Any, context: dict[str, Any]) -> pd.DataFrame:
    if schema is None:
        return pd.DataFrame()
    declarations: dict[str, dict[str, set[str]]] = {}
    for role, column in schema.semantic_mapping.items():
        category = "technical" if role in {"site", "batch"} else "covariate"
        entry = declarations.setdefault(column, {"roles": set(), "categories": set()})
        entry["roles"].add(role)
        entry["categories"].add(category)
    for column in schema.covariates:
        entry = declarations.setdefault(column, {"roles": set(), "categories": set()})
        entry["roles"].add("covariate")
        entry["categories"].add("covariate")
    for column in schema.technical:
        entry = declarations.setdefault(column, {"roles": set(), "categories": set()})
        entry["roles"].add("technical")
        entry["categories"].add("technical")
    rows: list[dict[str, Any]] = []
    total_subjects = int(metadata["__subject_id__"].nunique()) if bool(context["subject_id_declared"]) else np.nan
    total_clusters = int(metadata["__cluster_id__"].nunique())
    for column, declaration in declarations.items():
        if column not in metadata.columns:
            raise ValueError(f"Declared metadata column {column!r} is not present in metadata.")
        values = metadata[column]
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(values) else 0.0
        kind = "numeric" if numeric_fraction >= 0.95 and numeric.nunique(dropna=True) > 2 else "categorical"
        observed_subjects = int(metadata.loc[values.notna(), "__subject_id__"].nunique()) if bool(context["subject_id_declared"]) else np.nan
        observed_clusters = int(metadata.loc[values.notna(), "__cluster_id__"].nunique())
        rows.append(
            {
                "role": ", ".join(sorted(declaration["roles"])),
                "category": ", ".join(sorted(declaration["categories"])),
                "column": column,
                "kind": kind,
                "n_samples": int(len(values)),
                "n_subjects": total_subjects,
                "n_resampling_clusters": total_clusters,
                "observed": int(values.notna().sum()),
                "observed_subjects": observed_subjects,
                "observed_clusters": observed_clusters,
                "missing": int(values.isna().sum()),
                "missing_subjects": int(total_subjects - observed_subjects) if np.isfinite(total_subjects) and np.isfinite(observed_subjects) else np.nan,
                "missing_fraction": float(values.isna().mean()) if len(values) else np.nan,
                "unique": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _stable_within_subject(values: pd.Series, subjects: pd.Series) -> bool:
    frame = pd.DataFrame({"subject": subjects.astype(str), "value": values})
    encoded = frame["value"].astype("string").fillna("__MISSING__")
    return bool(encoded.groupby(frame["subject"], sort=False).nunique().le(1).all())


def _balance_analysis_frame(metadata: pd.DataFrame, column: str, context: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    cluster = metadata["__cluster_id__"]
    stable_target = _stable_within_subject(metadata["__y_label__"], cluster)
    stable_covariate = _stable_within_subject(metadata[column], cluster)
    if stable_target and stable_covariate and int(cluster.nunique()) < len(metadata):
        unit = "subject" if bool(context["subject_id_declared"]) else "dependence_cluster"
        return metadata.drop_duplicates("__cluster_id__", keep="first").copy(), unit
    return metadata.copy(), "sample"


def _ratio_smd(p1: float, p2: float) -> float:
    denominator = float(np.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / 2.0))
    difference = abs(float(p1) - float(p2))
    if denominator > 0.0:
        return difference / denominator
    return 0.0 if difference == 0.0 else float("inf")


def _numeric_pairwise_balance(values: pd.Series, groups: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    levels = sorted(groups.dropna().astype(str).unique())
    summaries: dict[str, dict[str, float | int]] = {}
    for level in levels:
        array = numeric[groups.astype(str).eq(level)].dropna().to_numpy(dtype=float)
        summaries[level] = {
            "n": int(len(array)),
            "mean": float(np.mean(array)) if len(array) else np.nan,
            "sd": float(np.std(array, ddof=1)) if len(array) > 1 else np.nan,
            "median": float(np.median(array)) if len(array) else np.nan,
        }
    best_smd = np.nan
    best_raw = np.nan
    best_pair = ""
    for first, second in combinations(levels, 2):
        a = numeric[groups.astype(str).eq(first)].dropna().to_numpy(dtype=float)
        b = numeric[groups.astype(str).eq(second)].dropna().to_numpy(dtype=float)
        if not len(a) or not len(b):
            continue
        raw = abs(float(np.mean(a)) - float(np.mean(b)))
        if len(a) > 1 and len(b) > 1:
            denominator_df = len(a) + len(b) - 2
            variance = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / denominator_df
            denominator = float(np.sqrt(max(float(variance), 0.0)))
            smd = raw / denominator if denominator > 0.0 else (0.0 if raw == 0.0 else float("inf"))
        else:
            smd = np.nan
        if np.isfinite(smd) or np.isinf(smd):
            if not (np.isfinite(best_smd) or np.isinf(best_smd)) or smd > best_smd:
                best_smd = float(smd)
                best_raw = float(raw)
                best_pair = f"{first} vs {second}"
    return {
        "max_pairwise_smd": best_smd,
        "max_pairwise_raw_difference": best_raw,
        "comparison": best_pair,
        "most_imbalanced_level": "",
        "group_summaries": json.dumps(summaries, sort_keys=True),
    }


def _categorical_pairwise_balance(values: pd.Series, groups: pd.Series) -> dict[str, Any]:
    observed = pd.DataFrame({"value": values.astype("string"), "group": groups.astype("string")}).dropna()
    if observed.empty or observed["group"].nunique() < 2:
        return {
            "max_pairwise_smd": np.nan,
            "max_pairwise_raw_difference": np.nan,
            "comparison": "",
            "most_imbalanced_level": "",
            "group_summaries": "{}",
        }
    table = pd.crosstab(observed["group"], observed["value"], normalize="index")
    group_levels = sorted(str(value) for value in table.index)
    value_levels = sorted(str(value) for value in table.columns)
    table = table.reindex(index=group_levels, columns=value_levels, fill_value=0.0)
    best_smd = np.nan
    best_raw = np.nan
    best_pair = ""
    best_level = ""
    for first, second in combinations(group_levels, 2):
        for level in value_levels:
            p1 = float(table.loc[first, level])
            p2 = float(table.loc[second, level])
            raw = abs(p1 - p2)
            smd = _ratio_smd(p1, p2)
            if not (np.isfinite(best_smd) or np.isinf(best_smd)) or smd > best_smd:
                best_smd = float(smd)
                best_raw = float(raw)
                best_pair = f"{first} vs {second}"
                best_level = str(level)
    summaries = {
        str(group): {str(level): float(table.loc[group, level]) for level in value_levels}
        for group in group_levels
    }
    return {
        "max_pairwise_smd": best_smd,
        "max_pairwise_raw_difference": best_raw,
        "comparison": best_pair,
        "most_imbalanced_level": best_level,
        "group_summaries": json.dumps(summaries, sort_keys=True),
    }


def _missingness_balance(values: pd.Series, groups: pd.Series) -> tuple[float, float, str]:
    levels = sorted(groups.dropna().astype(str).unique())
    missing = values.isna()
    best_smd = np.nan
    best_raw = np.nan
    best_pair = ""
    for first, second in combinations(levels, 2):
        first_mask = groups.astype(str).eq(first)
        second_mask = groups.astype(str).eq(second)
        p1 = float(missing[first_mask].mean()) if bool(first_mask.any()) else np.nan
        p2 = float(missing[second_mask].mean()) if bool(second_mask.any()) else np.nan
        if not np.isfinite(p1) or not np.isfinite(p2):
            continue
        raw = abs(p1 - p2)
        smd = _ratio_smd(p1, p2)
        if not (np.isfinite(best_smd) or np.isinf(best_smd)) or smd > best_smd:
            best_smd = float(smd)
            best_raw = float(raw)
            best_pair = f"{first} vs {second}"
    return best_smd, best_raw, best_pair


def _balance_table(metadata: pd.DataFrame, columns: tuple[str, ...], technical: set[str], context: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in metadata.columns:
            raise ValueError(f"Robustness covariate {column!r} is not present in metadata.")
        analysis, unit = _balance_analysis_frame(metadata, column, context)
        values = analysis[column]
        groups = analysis["__y_label__"]
        numeric = pd.to_numeric(values, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(values) else 0.0
        kind = "numeric" if numeric_fraction >= 0.95 and numeric.nunique(dropna=True) > 2 else "categorical"
        result = _numeric_pairwise_balance(values, groups) if kind == "numeric" else _categorical_pairwise_balance(values, groups)
        missing_smd, missing_raw, missing_pair = _missingness_balance(values, groups)
        rows.append(
            {
                "covariate": column,
                "role": "technical" if column in technical else "covariate",
                "kind": kind,
                "analysis_unit": unit,
                "n_units": int(len(analysis)),
                "n_samples": int(len(metadata)),
                "n_subjects": int(metadata["__subject_id__"].nunique()) if bool(context["subject_id_declared"]) else np.nan,
                "n_resampling_clusters": int(metadata["__cluster_id__"].nunique()),
                "observed": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "imbalance": result["max_pairwise_smd"],
                "max_pairwise_smd": result["max_pairwise_smd"],
                "max_pairwise_raw_difference": result["max_pairwise_raw_difference"],
                "comparison": result["comparison"],
                "most_imbalanced_level": result["most_imbalanced_level"],
                "missingness_smd": missing_smd,
                "missingness_difference": missing_raw,
                "missingness_comparison": missing_pair,
                "group_summaries": result["group_summaries"],
            }
        )
    return pd.DataFrame(rows)


def _prediction_sources(root: Path, targets: tuple[str, ...]) -> list[tuple[str, Path]]:
    sources = {
        "mpma_b": root / "predictions" / "mpma_b_outer_predictions.parquet",
        "mpma_e": root / "ensembling" / "ensemble_predictions.parquet",
    }
    return [(target, sources[target]) for target in targets if target in sources and table_exists(sources[target])]


def _probability_columns(frame: pd.DataFrame, class_labels: tuple[str, ...]) -> list[str]:
    expected = [f"proba_{label}" for label in class_labels]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        available = [str(column) for column in frame.columns if str(column).startswith("proba_")]
        raise ValueError(
            f"Held-out predictions do not match declared class-label order; missing={missing!r}, available={available!r}."
        )
    return expected


def _prepare_predictions(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_col: str,
    subgroup: str,
    context: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    if subgroup not in metadata.columns:
        raise ValueError(f"Robustness subgroup {subgroup!r} is not present in metadata.")
    out = predictions.copy()
    if "outer_split_key" not in out.columns:
        if "split_key" not in out.columns:
            raise ValueError("Held-out predictions must contain outer_split_key or split_key.")
        out["outer_split_key"] = out["split_key"]
    required = {"outer_split_key", "sample_id", "y_true", "y_pred"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Held-out predictions are missing required columns: {missing!r}.")
    out["sample_id"] = out["sample_id"].astype(str)
    out["outer_split_key"] = out["outer_split_key"].astype(str)
    meta = metadata[[sample_col, "__subject_id__", "__cluster_id__", subgroup]].copy()
    meta[sample_col] = meta[sample_col].astype(str)
    out = out.merge(meta, left_on="sample_id", right_on=sample_col, how="left", validate="many_to_one")
    if out[subgroup].isna().all():
        raise ValueError(f"Robustness subgroup {subgroup!r} has no observed values among held-out predictions.")
    if "subject_id" in out.columns:
        if out["subject_id"].isna().any():
            raise ValueError("Held-out predictions contain missing subject_id values.")
        mismatch = out["subject_id"].astype(str).ne(out["__subject_id__"].astype(str))
        if bool(mismatch.any()):
            raise ValueError("Held-out prediction subject_id values disagree with the data specification.")
    protocol = str(context["protocol"]).strip().lower()
    if protocol in _LODO_PROTOCOLS:
        out["_repeat"] = "r0"
        out["_cluster"] = out["outer_split_key"].astype(str)
        duplicate_keys = ["_cluster", "sample_id"]
        cluster_cohorts = out.groupby("__cluster_id__", sort=False)["_cluster"].nunique()
        if bool((cluster_cohorts > 1).any()):
            raise ValueError("A robustness resampling cluster occurs in more than one held-out LODO cohort.")
    else:
        out["_repeat"] = out["outer_split_key"].map(_repeat_id)
        out["_cluster"] = out["_repeat"]
        duplicate_keys = ["_repeat", "sample_id"]
    if out.duplicated(duplicate_keys).any():
        raise ValueError("Held-out predictions contain duplicate inference rows within an evaluation repeat or cohort.")
    task = str(context["task"])
    if task == "classification":
        pcols = _probability_columns(out, tuple(context["class_labels"]))
        out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(int)
        out["y_pred"] = pd.to_numeric(out["y_pred"], errors="raise").astype(int)
        for column in pcols:
            out[column] = pd.to_numeric(out[column], errors="raise")
        probabilities = out[pcols].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Held-out predictions contain non-finite class scores.")
        if np.any(probabilities < -1e-8) or np.any(probabilities > 1.0 + 1e-8):
            raise ValueError("Held-out prediction class scores fall outside [0, 1].")
        inferred_probability_valid = bool(
            np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
        )
        if "probability_valid" in out.columns:
            declared = out["probability_valid"].map(
                lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() in {"1", "true", "yes"}
            )
            probability_valid = bool(declared.all())
        else:
            probability_valid = inferred_probability_valid
        if probability_valid and not inferred_probability_valid:
            raise ValueError("Held-out predictions are declared probability-valued but rows do not sum to one.")
        out["__probability_valid__"] = probability_valid
    else:
        pcols = []
        out["y_true"] = pd.to_numeric(out["y_true"], errors="raise").astype(float)
        out["y_pred"] = pd.to_numeric(out["y_pred"], errors="raise").astype(float)
        if not np.isfinite(out[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
            raise ValueError("Held-out regression predictions contain non-finite values.")
    return out.reset_index(drop=True), pcols


def _classification_metrics(
    frame: pd.DataFrame,
    pcols: list[str],
    positive_class: int | None,
) -> dict[str, float]:
    n_classes = len(pcols)
    classes = np.arange(n_classes, dtype=int)
    y_true = frame["y_true"].to_numpy(dtype=int)
    y_pred = frame["y_pred"].to_numpy(dtype=int)
    proba = frame[pcols].to_numpy(dtype=float)
    values = compute_metrics(y_true, y_pred, proba, classes, positive_class=positive_class)
    out = {metric: float(values.get(metric, np.nan)) for metric in _CLASSIFICATION_ROBUSTNESS_METRICS}
    probability_valid = bool(frame["__probability_valid__"].all()) if "__probability_valid__" in frame.columns else True
    if not probability_valid:
        out["log_loss"] = np.nan
        out["brier"] = np.nan
        for metric in _CALIBRATION_METRICS:
            out[metric] = np.nan
        return out
    if n_classes == 2:
        index = int(classes[-1] if positive_class is None else positive_class)
        calibration = _calibration_binary((y_true == index).astype(float), proba[:, index])
        out["CalibrationInTheLarge"] = float(calibration[0])
        out["CalibrationIntercept"] = float(calibration[1])
        out["CalibrationSlope"] = float(calibration[2])
    else:
        calibration = [_calibration_binary((y_true == index).astype(float), proba[:, index]) for index in classes]
        for position, name in enumerate(
            (
                "CalibrationInTheLarge_macro_OvR",
                "CalibrationIntercept_macro_OvR",
                "CalibrationSlope_macro_OvR",
            )
        ):
            finite = np.asarray([value[position] for value in calibration if np.isfinite(value[position])], dtype=float)
            out[name] = float(np.mean(finite)) if len(finite) else np.nan
    return out


def _metric_values(frame: pd.DataFrame, task: str, pcols: list[str], positive_class: int | None) -> dict[str, float]:
    if frame.empty:
        metrics = _CLASSIFICATION_ROBUSTNESS_METRICS if task == "classification" else tuple(REGRESSION_METRIC_COLUMNS)
        return {metric: np.nan for metric in metrics}
    if task == "classification":
        return _classification_metrics(frame, pcols, positive_class)
    return {metric: float(value) for metric, value in compute_regression_metrics(frame["y_true"].to_numpy(dtype=float), frame["y_pred"].to_numpy(dtype=float)).items()}


def _mean_metric_dicts(values: list[dict[str, float]], metrics: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        array = np.asarray([value.get(metric, np.nan) for value in values], dtype=float)
        array = array[np.isfinite(array)]
        out[metric] = float(np.mean(array)) if len(array) else np.nan
    return out


def _point_estimands(
    frame: pd.DataFrame,
    context: dict[str, Any],
    pcols: list[str],
) -> dict[str, dict[str, float]]:
    task = str(context["task"])
    metrics = _CLASSIFICATION_ROBUSTNESS_METRICS if task == "classification" else tuple(REGRESSION_METRIC_COLUMNS)
    positive = context["positive_class"] if task == "classification" else None
    protocol = str(context["protocol"]).strip().lower()
    if protocol in _LODO_PROTOCOLS:
        cohort_values = [_metric_values(group, task, pcols, positive) for _, group in frame.groupby("_cluster", sort=True)]
        return {
            "pooled_sample_weighted": _metric_values(frame, task, pcols, positive),
            "cohort_macro_equal_weight": _mean_metric_dicts(cohort_values, metrics),
        }
    repeat_values = [_metric_values(group, task, pcols, positive) for _, group in frame.groupby("_repeat", sort=True)]
    return {"mean_repeat_pooled_oof": _mean_metric_dicts(repeat_values, metrics)}


def _resample_subject_clusters(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    clusters = [group.reset_index(drop=True) for _, group in frame.groupby("__cluster_id__", sort=True)]
    if not clusters:
        return frame.iloc[0:0].copy()
    chosen = rng.integers(0, len(clusters), size=len(clusters))
    return pd.concat([clusters[int(index)] for index in chosen], ignore_index=True)


def _bootstrap_frame(frame: pd.DataFrame, protocol: str, rng: np.random.Generator) -> pd.DataFrame:
    key = str(protocol).strip().lower()
    if key not in _LODO_PROTOCOLS:
        return _resample_subject_clusters(frame, rng)
    cohorts = [group.reset_index(drop=True) for _, group in frame.groupby("_cluster", sort=True)]
    if not cohorts:
        return frame.iloc[0:0].copy()
    chosen = rng.integers(0, len(cohorts), size=len(cohorts))
    sampled: list[pd.DataFrame] = []
    for position, index in enumerate(chosen):
        cohort = _resample_subject_clusters(cohorts[int(index)], rng)
        cohort = cohort.copy()
        cohort["_cluster"] = f"bootstrap_cohort_{position}"
        sampled.append(cohort)
    return pd.concat(sampled, ignore_index=True)


def _support_row(
    frame: pd.DataFrame,
    strategy: str,
    subgroup: str,
    level: Any,
    context: dict[str, Any],
    min_samples: int,
    min_clusters: int,
    min_class_clusters: int,
) -> dict[str, Any]:
    unique_samples = frame.drop_duplicates("sample_id", keep="first")
    n_samples = int(unique_samples["sample_id"].nunique())
    n_subjects = int(unique_samples["__subject_id__"].nunique()) if bool(context["subject_id_declared"]) else np.nan
    n_clusters = int(unique_samples["__cluster_id__"].nunique())
    eligible = n_samples >= min_samples and n_clusters >= min_clusters
    reason = ""
    if n_samples < min_samples:
        reason = f"n_samples<{min_samples}"
    elif n_clusters < min_clusters:
        reason = f"n_resampling_clusters<{min_clusters}"
    row: dict[str, Any] = {
        "strategy": strategy.upper().replace("_", "-"),
        "subgroup": subgroup,
        "level": str(level),
        "n_samples": n_samples,
        "n_subjects": n_subjects,
        "n_resampling_clusters": n_clusters,
        "n_oof_rows": int(len(frame)),
        "n_repeats": int(frame["_repeat"].nunique()),
        "resampling_unit": str(context["cluster_source"]),
        "eligible": bool(eligible),
        "support_status": "adequate" if eligible else "insufficient_size",
        "reason": reason,
        "class_counts_samples": "",
        "class_counts_clusters": "",
        "minimum_class_clusters": np.nan,
    }
    if str(context["task"]) == "classification":
        labels = tuple(context["class_labels"])
        sample_counts = {label: int((unique_samples["y_true"].astype(int) == index).sum()) for index, label in enumerate(labels)}
        cluster_counts = {
            label: int(unique_samples.loc[unique_samples["y_true"].astype(int).eq(index), "__cluster_id__"].nunique())
            for index, label in enumerate(labels)
        }
        minimum = min(cluster_counts.values()) if cluster_counts else 0
        row["class_counts_samples"] = json.dumps(sample_counts, sort_keys=True)
        row["class_counts_clusters"] = json.dumps(cluster_counts, sort_keys=True)
        row["minimum_class_clusters"] = int(minimum)
        if eligible and minimum < min_class_clusters:
            row["support_status"] = "limited_class_support"
            row["reason"] = f"minimum_class_clusters<{min_class_clusters}"
    return row


def _confidence_interval(draws: np.ndarray, confidence_level: float, minimum_valid: int) -> tuple[float, float, int, str]:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    count = int(len(finite))
    if count < int(minimum_valid):
        return np.nan, np.nan, count, "insufficient_valid_bootstrap_draws"
    alpha = (1.0 - float(confidence_level)) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high), count, "ok"


def _subgroup_analysis(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_col: str,
    subgroup: str,
    strategy: str,
    context: dict[str, Any],
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined, pcols = _prepare_predictions(predictions, metadata, sample_col, subgroup, context)
    observed = joined.dropna(subset=[subgroup]).copy()
    levels = sorted(observed[subgroup].unique(), key=lambda value: str(value))
    support_rows: list[dict[str, Any]] = []
    eligible_levels: list[Any] = []
    min_clusters = int(config.min_subgroup_clusters or config.min_subgroup_size)
    for level in levels:
        level_frame = observed[observed[subgroup].eq(level)].copy()
        support = _support_row(
            level_frame,
            strategy,
            subgroup,
            level,
            context,
            int(config.min_subgroup_size),
            min_clusters,
            int(config.min_class_clusters),
        )
        support_rows.append(support)
        if bool(support["eligible"]):
            eligible_levels.append(level)
    missing = joined[joined[subgroup].isna()].copy()
    if not missing.empty:
        missing_support = _support_row(
            missing,
            strategy,
            subgroup,
            "<missing>",
            context,
            int(config.min_subgroup_size),
            min_clusters,
            int(config.min_class_clusters),
        )
        missing_support["eligible"] = False
        missing_support["support_status"] = "missing_subgroup_value"
        missing_support["reason"] = "subgroup value missing"
        support_rows.append(missing_support)
    if not eligible_levels:
        return pd.DataFrame(), pd.DataFrame(support_rows), pd.DataFrame()
    points = {
        level: _point_estimands(observed[observed[subgroup].eq(level)], context, pcols)
        for level in eligible_levels
    }
    task = str(context["task"])
    metrics = _CLASSIFICATION_ROBUSTNESS_METRICS if task == "classification" else tuple(REGRESSION_METRIC_COLUMNS)
    estimands = tuple(next(iter(points.values())).keys())
    n_bootstrap = int(config.bootstrap_replicates)
    storage = {
        level: {
            estimand: {metric: np.full(n_bootstrap, np.nan, dtype=float) for metric in metrics}
            for estimand in estimands
        }
        for level in eligible_levels
    }
    rng = np.random.default_rng(_stable_seed(int(config.random_state), "robustness", strategy, subgroup))
    for bootstrap_index in range(n_bootstrap):
        sampled = _bootstrap_frame(observed, str(context["protocol"]), rng)
        for level in eligible_levels:
            level_frame = sampled[sampled[subgroup].eq(level)]
            if level_frame.empty:
                continue
            values = _point_estimands(level_frame, context, pcols)
            for estimand in estimands:
                for metric in metrics:
                    storage[level][estimand][metric][bootstrap_index] = values.get(estimand, {}).get(metric, np.nan)
    minimum_valid = int(np.ceil(float(config.min_bootstrap_valid_fraction) * n_bootstrap))
    performance_rows: list[dict[str, Any]] = []
    support_lookup = {row["level"]: row for row in support_rows}
    for level in eligible_levels:
        level_name = str(level)
        support = support_lookup[level_name]
        probability_valid = bool(observed.loc[observed[subgroup].eq(level), "__probability_valid__"].all()) if task == "classification" else np.nan
        for estimand, values in points[level].items():
            for metric in metrics:
                estimate = float(values.get(metric, np.nan))
                if not np.isfinite(estimate):
                    continue
                low, high, valid, status = _confidence_interval(
                    storage[level][estimand][metric],
                    float(config.confidence_level),
                    minimum_valid,
                )
                performance_rows.append(
                    {
                        "strategy": strategy.upper().replace("_", "-"),
                        "subgroup": subgroup,
                        "level": level_name,
                        "estimand": estimand,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "confidence_level": float(config.confidence_level),
                        "ci_method": "hierarchical_cluster_percentile_bootstrap" if str(context["protocol"]).strip().lower() in _LODO_PROTOCOLS else "cluster_percentile_bootstrap",
                        "n_bootstrap": n_bootstrap,
                        "n_bootstrap_valid": valid,
                        "ci_status": status,
                        "n_samples": int(support["n_samples"]),
                        "n_subjects": float(support["n_subjects"]) if np.isfinite(support["n_subjects"]) else np.nan,
                        "n_resampling_clusters": int(support["n_resampling_clusters"]),
                        "n_oof_rows": int(support["n_oof_rows"]),
                        "n_repeats": int(support["n_repeats"]),
                        "resampling_unit": str(support["resampling_unit"]),
                        "support_status": str(support["support_status"]),
                        "probability_valid": probability_valid,
                    }
                )
    contrast_metrics = (
        _REGRESSION_CONTRAST_METRICS
        if task == "regression"
        else (_BINARY_CONTRAST_METRICS if len(tuple(context["class_labels"])) == 2 else _MULTICLASS_CONTRAST_METRICS)
    )
    contrast_rows: list[dict[str, Any]] = []
    for first, second in combinations(eligible_levels, 2):
        for estimand in estimands:
            for metric in contrast_metrics:
                first_estimate = float(points[first][estimand].get(metric, np.nan))
                second_estimate = float(points[second][estimand].get(metric, np.nan))
                if not np.isfinite(first_estimate) or not np.isfinite(second_estimate):
                    continue
                draws = storage[first][estimand][metric] - storage[second][estimand][metric]
                low, high, valid, status = _confidence_interval(
                    draws,
                    float(config.confidence_level),
                    minimum_valid,
                )
                contrast_rows.append(
                    {
                        "strategy": strategy.upper().replace("_", "-"),
                        "subgroup": subgroup,
                        "level_a": str(first),
                        "level_b": str(second),
                        "estimand": estimand,
                        "metric": metric,
                        "difference_a_minus_b": first_estimate - second_estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "confidence_level": float(config.confidence_level),
                        "ci_method": "paired_hierarchical_cluster_percentile_bootstrap" if str(context["protocol"]).strip().lower() in _LODO_PROTOCOLS else "paired_cluster_percentile_bootstrap",
                        "n_bootstrap": n_bootstrap,
                        "n_bootstrap_valid": valid,
                        "ci_status": status,
                        "multiplicity_adjusted": False,
                    }
                )
    return pd.DataFrame(performance_rows), pd.DataFrame(support_rows), pd.DataFrame(contrast_rows)


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
            columns = [column for column in ("class_index", "rank_median", "importance_mean") if column in frame.columns]
            frame = frame.sort_values(columns, ascending=True, kind="mergesort")
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
    metadata, sample_col, context = _metadata(sweep)
    schema = _metadata_schema(sweep)
    outputs: dict[str, Path] = {}
    inventory = _metadata_inventory(metadata, schema, context)
    if not inventory.empty:
        path = out_dir / "metadata_inventory.parquet"
        write_table(path, inventory)
        outputs["metadata_inventory"] = path
    automatic_covariates = schema.covariate_columns if schema is not None else ()
    automatic_technical = schema.technical_columns if schema is not None else ()
    covariate_columns = tuple(dict.fromkeys((*automatic_covariates, *config.covariates)))
    technical_columns = tuple(dict.fromkeys((*automatic_technical, *config.technical)))
    balance_columns = tuple(dict.fromkeys((*covariate_columns, *technical_columns)))
    if balance_columns and str(context["task"]) == "classification":
        balance = _balance_table(metadata, balance_columns, set(technical_columns), context)
        path = out_dir / "covariate_balance.parquet"
        write_table(path, balance)
        outputs["covariate_balance"] = path
    subgroup_frames: list[pd.DataFrame] = []
    support_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    automatic_subgroups = schema.categorical_subgroup_columns if schema is not None else ()
    subgroup_columns = tuple(dict.fromkeys((*automatic_subgroups, *config.subgroups)))
    for strategy, path in _prediction_sources(root, config.targets):
        predictions = read_table(path)
        for subgroup in subgroup_columns:
            performance, support, contrasts = _subgroup_analysis(
                predictions,
                metadata,
                sample_col,
                subgroup,
                strategy,
                context,
                config,
            )
            if not performance.empty:
                subgroup_frames.append(performance)
            if not support.empty:
                support_frames.append(support)
            if not contrasts.empty:
                contrast_frames.append(contrasts)
    if subgroup_frames:
        subgroup = pd.concat(subgroup_frames, ignore_index=True)
        path = out_dir / "subgroup_performance.parquet"
        write_table(path, subgroup)
        outputs["subgroup_performance"] = path
    if support_frames:
        support = pd.concat(support_frames, ignore_index=True)
        path = out_dir / "subgroup_support.parquet"
        write_table(path, support)
        outputs["subgroup_support"] = path
    if contrast_frames:
        contrasts = pd.concat(contrast_frames, ignore_index=True)
        path = out_dir / "subgroup_contrasts.parquet"
        write_table(path, contrasts)
        outputs["subgroup_contrasts"] = path
    features = _feature_robustness(root, config.targets, config.top_k)
    if not features.empty:
        path = out_dir / "important_feature_robustness.parquet"
        write_table(path, features)
        outputs["important_feature_robustness"] = path
    manifest = out_dir / "robustness_manifest.json"
    dump_json_standard(
        {
            "schema_version": 3,
            "target": context["target"],
            "task": context["task"],
            "targets": list(config.targets),
            "evaluation_protocol": context["protocol"],
            "class_labels": list(context["class_labels"]),
            "positive_class": context["positive_class"],
            "subject_id_declared": bool(context["subject_id_declared"]),
            "robustness_resampling_unit": context["cluster_source"],
            "metadata": schema.semantic_mapping if schema is not None else {},
            "covariates": list(covariate_columns),
            "technical": list(technical_columns),
            "subgroups": list(subgroup_columns),
            "top_k": int(config.top_k),
            "min_subgroup_size": int(config.min_subgroup_size),
            "min_subgroup_clusters": int(config.min_subgroup_clusters or config.min_subgroup_size),
            "min_class_clusters": int(config.min_class_clusters),
            "bootstrap_replicates": int(config.bootstrap_replicates),
            "confidence_level": float(config.confidence_level),
            "min_bootstrap_valid_fraction": float(config.min_bootstrap_valid_fraction),
            "random_state": int(config.random_state),
            "subgroup_point_estimand": "mean of repeat-specific pooled outer-held-out metrics" if str(context["protocol"]).strip().lower() not in _LODO_PROTOCOLS else "pooled and cohort-macro outer-held-out metrics",
            "subgroup_uncertainty": "nonparametric cluster bootstrap preserving repeated observations and repeated CV structure",
            "subgroup_uncertainty_scope": "conditional on the existing outer-held-out prediction artifacts; model fitting and model selection are not rerun inside bootstrap replicates",
            "subgroup_contrasts": "exploratory pairwise metric differences with paired cluster-bootstrap confidence intervals and no multiplicity adjustment",
            "balance_estimand": "maximum pairwise standardized difference across outcome groups; dependence-cluster level when target and covariate are cluster-invariant, otherwise sample-level",
            "feature_provenance": "strategy-specific outer-test explainability only",
            "outputs": outputs,
        },
        manifest,
    )
    outputs["manifest"] = manifest
    success("Robustness completed")
    path_table("Robustness outputs", outputs)
    return outputs
