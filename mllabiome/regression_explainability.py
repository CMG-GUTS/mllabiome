from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .configs_sweep import (
    Sweep,
    _effective_local_explanations_mode,
    _groups_from_metadata,
    _lodo_feature_pair,
    _regression_outer_splits,
)
from .console import (
    info,
    path_table,
    phase_progress,
    progress,
    stage,
    success,
    summary_table,
)
from .data import load_dataset
from .explainability import (
    _AleModelWrapper,
    _ale_result_values,
    _auto_ale_bins,
    _plot_ale_curves,
    _quiet_pyale_info,
    _require_pyale,
)
from .explainability_methods import (
    ALE,
    ALEInteractions,
    LIME,
    Permutation,
    SHAP,
    method_has_global,
    method_has_local,
    method_name,
)
from .learners import _learner_factory
from .metrics import _estimator_call, compute_regression_metrics, metric_is_loss
from .regression_ensemble import aggregate_regression_predictions
from .resolutions import materialize_mpdr
from .runtime import configure_estimator_threads
from .transformations import _count_transformation_factory
from .explainability_visuals import (
    plot_interaction_network,
    plot_local_attributions,
    plot_regression_feature_support,
)
from .final_models import build_final_models
from .utils import dump_json_standard
from .storage import read_table, write_table, table_exists


_REGRESSION_METRICS = {
    "r2": "R2",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "medae": "MedAE",
    "median_absolute_error": "MedAE",
    "explainedvariance": "ExplainedVariance",
    "explained_variance": "ExplainedVariance",
    "pearsonr": "PearsonR",
    "pearson": "PearsonR",
    "spearmanr": "SpearmanR",
    "spearman": "SpearmanR",
}


class _FittedRegressionEnsemble(BaseEstimator):
    def __init__(
        self,
        members: list[dict[str, Any]],
        aggregation: str,
        weights: Sequence[float] | None = None,
    ):
        self.members = members
        self.aggregation = str(aggregation)
        self.weights = (
            None if weights is None else np.asarray(list(weights), dtype=float)
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        stack = np.vstack(
            [
                np.asarray(
                    _estimator_call(
                        member["estimator"], "predict", arr[:, member["slice"]]
                    ),
                    dtype=float,
                ).reshape(-1)
                for member in self.members
            ]
        )
        return aggregate_regression_predictions(stack, self.aggregation, self.weights)


def _row_levels(row: pd.Series) -> tuple[str, ...]:
    text = str(row.get("levels", "")).strip()
    if text:
        return tuple(x.strip() for x in text.split(",") if x.strip())
    resolution = str(row.get("resolution", "all")).strip()
    return (resolution,) if resolution else ("all",)


def _factories(sweep: Sweep) -> tuple[dict[str, Any], dict[str, Any]]:
    transforms = dict(
        _count_transformation_factory(x, random_state=sweep.explainability.random_state)
        for x in sweep.count_transformations
    )
    learners = dict(_learner_factory(x, task="regression") for x in sweep.learners)
    return transforms, learners


def _config_row(configs: pd.DataFrame, config_id: str) -> pd.Series:
    frame = configs[configs["config_id"].astype(str).eq(str(config_id))]
    if frame.empty:
        frame = configs[
            configs["config_id"]
            .astype(str)
            .map(lambda x: x.startswith(str(config_id)) or str(config_id).startswith(x))
        ]
    if frame.empty:
        raise ValueError(f"No MPMA with config_id={config_id!r} in configs.parquet.")
    return frame.iloc[0]


def _selected_models(root: Path) -> dict[str, Any]:
    candidate = root / "tables" / "mpma_b_final_candidate.json"
    if not candidate.exists():
        raise FileNotFoundError("Run evaluate(sweep) before regression explainability.")
    value = build_final_models(root)
    if not isinstance(value, dict) or not isinstance(value.get("MPMA-B"), dict):
        raise ValueError(
            "Final model specification does not contain a valid MPMA-B specification."
        )
    return value


def _targets(sweep: Sweep, models: dict[str, Any]) -> list[str]:
    configured = getattr(sweep.explainability, "targets", "auto")
    values = [configured] if isinstance(configured, str) else list(configured)
    out: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        if value in {"auto", "all", "comparison", "standard", "standard_suite"}:
            if isinstance(models.get("MPMA-E"), dict):
                out.append("mpma_e")
            out.append("mpma_b")
        elif value in {"best", "best_individual", "best_mpma", "MPMA-B"}:
            out.append("mpma_b")
        elif value in {"ensemble", "MPMA-E"}:
            out.append("mpma_e")
        else:
            out.append(value)
    return list(dict.fromkeys(out or ["mpma_b"]))


def _coordinate_metadata_rows(
    transform: Any, input_features: Sequence[str], prefix: str = ""
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in transform.coordinate_metadata(list(input_features)):
        coordinate = str(item.name)
        rows.append(
            {
                "coordinate": f"{prefix}|{coordinate}" if prefix else coordinate,
                "native_coordinate": coordinate,
                "coordinate_type": str(item.coordinate_type),
                "anchor_feature": ""
                if item.anchor_feature is None
                else str(item.anchor_feature),
                "exact_feature_identity": bool(item.exact_feature_identity),
                "components": json.dumps(list(item.components), separators=(",", ":")),
                "coefficients": json.dumps(
                    [float(x) for x in item.coefficients], separators=(",", ":")
                ),
            }
        )
    return rows


def _fit_individual_folds(
    sweep: Sweep,
    row: pd.Series,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    if getattr(sweep, "uses_modalities", False):
        from .multimodal_sweep import fit_modality_regression_candidate_folds

        dataset, folds = fit_modality_regression_candidate_folds(
            sweep, row, progress_callback=progress_callback
        )
        for fold in folds:
            objects = fold.pop("coordinate_metadata_objects", [])
            fold["coordinate_metadata"] = [
                {
                    "coordinate": str(item.name),
                    "native_coordinate": str(item.name),
                    "coordinate_type": str(item.coordinate_type),
                    "anchor_feature": ""
                    if item.anchor_feature is None
                    else str(item.anchor_feature),
                    "exact_feature_identity": bool(item.exact_feature_identity),
                    "components": json.dumps(
                        list(item.components), separators=(",", ":")
                    ),
                    "coefficients": json.dumps(
                        [float(x) for x in item.coefficients], separators=(",", ":")
                    ),
                }
                for item in objects
            ]
        return dataset, folds
    levels = _row_levels(row)
    dataset = load_dataset(sweep.data, levels)
    X_base, base_names = materialize_mpdr(dataset, levels)
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    splits = _regression_outer_splits(sweep.evaluation, len(dataset.y), groups)
    transforms, learners = _factories(sweep)
    transform_key = str(row["count_transformation"])
    learner_key = str(row["learner"])
    if transform_key not in transforms:
        raise ValueError(
            f"Configured transformation {transform_key!r} is unavailable for explainability."
        )
    if learner_key not in learners:
        raise ValueError(
            f"Configured regression learner {learner_key!r} is unavailable for explainability."
        )
    folds: list[dict[str, Any]] = []
    for fold_no, split in enumerate(splits, start=1):
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        X_train0, X_test0, mask = _lodo_feature_pair(
            X_base, train_idx, test_idx, sweep.evaluation.protocol
        )
        names = [str(name) for name, keep in zip(base_names, mask) if bool(keep)]
        transform = transforms[transform_key]()
        X_train, X_test = transform.apply_pair(X_train0, X_test0)
        coordinate_metadata = _coordinate_metadata_rows(transform, names)
        names = transform.get_feature_names_out(names)
        model = configure_estimator_threads(learners[learner_key](), 1)
        model.fit(X_train, dataset.y[train_idx])
        pred = np.asarray(
            _estimator_call(model, "predict", X_test), dtype=float
        ).reshape(-1)
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": np.asarray(X_train, dtype=float),
                "X_test": np.asarray(X_test, dtype=float),
                "feature_names": names,
                "coordinate_metadata": coordinate_metadata,
                "y_test": np.asarray(dataset.y[test_idx], dtype=float),
                "y_pred": pred,
                "estimator": model,
            }
        )
        if progress_callback is not None:
            progress_callback(fold_no, len(splits), str(split["split_key"]))
    return dataset, folds


def _fit_ensemble_folds(
    sweep: Sweep,
    configs: pd.DataFrame,
    unit: dict[str, Any],
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    if getattr(sweep, "uses_modalities", False):
        raise ValueError(
            "Regression MPMA-E explainability for modality-based candidates is not enabled in rc24. Ensemble prediction is supported, but hierarchical attribution across heterogeneous modality pipelines is intentionally not approximated."
        )
    members = [
        str(member.get("config_id") if isinstance(member, dict) else member)
        for member in unit.get("members", [])
    ]
    members = [x for x in members if x]
    if len(members) < 2:
        raise ValueError(
            "Regression MPMA-E explainability requires at least two ensemble members."
        )
    member_rows = [_config_row(configs, member) for member in members]
    all_levels: list[str] = []
    for row in member_rows:
        for level in _row_levels(row):
            if level not in all_levels:
                all_levels.append(level)
    dataset = load_dataset(sweep.data, tuple(all_levels or ["all"]))
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    splits = _regression_outer_splits(sweep.evaluation, len(dataset.y), groups)
    transforms, learners = _factories(sweep)
    materialized: list[tuple[pd.Series, np.ndarray, list[str]]] = []
    for row in member_rows:
        levels = _row_levels(row)
        X_base, names = materialize_mpdr(dataset, levels)
        materialized.append((row, np.asarray(X_base), [str(x) for x in names]))
    aggregation = str(unit.get("aggregation_strategy", "mean_prediction"))
    raw_weights = unit.get("weights")
    weights = None
    if aggregation == "weighted_mean_prediction":
        if isinstance(raw_weights, dict):
            weights = [float(raw_weights.get(member, 0.0)) for member in members]
        elif isinstance(raw_weights, (list, tuple)):
            weights = [float(x) for x in raw_weights]
        else:
            raise ValueError("Regression weighted MPMA-E is missing weights.")
    folds: list[dict[str, Any]] = []
    for fold_no, split in enumerate(splits, start=1):
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        train_blocks: list[np.ndarray] = []
        test_blocks: list[np.ndarray] = []
        feature_names: list[str] = []
        coordinate_metadata: list[dict[str, Any]] = []
        fitted: list[dict[str, Any]] = []
        start = 0
        for row, X_base, names in materialized:
            X_train0, X_test0, mask = _lodo_feature_pair(
                X_base, train_idx, test_idx, sweep.evaluation.protocol
            )
            kept_names = [name for name, keep in zip(names, mask) if bool(keep)]
            transform_key = str(row["count_transformation"])
            learner_key = str(row["learner"])
            transform = transforms[transform_key]()
            X_train_member, X_test_member = transform.apply_pair(X_train0, X_test0)
            transformed_names = transform.get_feature_names_out(kept_names)
            model = configure_estimator_threads(learners[learner_key](), 1)
            model.fit(X_train_member, dataset.y[train_idx])
            stop = start + X_train_member.shape[1]
            fitted.append({"slice": slice(start, stop), "estimator": model})
            prefix = (
                f"{row['resolution']}|{transform_key}|{learner_key}|{row['config_id']}"
            )
            feature_names.extend([f"{prefix}|{name}" for name in transformed_names])
            coordinate_metadata.extend(
                _coordinate_metadata_rows(transform, kept_names, prefix)
            )
            train_blocks.append(np.asarray(X_train_member, dtype=float))
            test_blocks.append(np.asarray(X_test_member, dtype=float))
            start = stop
        X_train = np.concatenate(train_blocks, axis=1)
        X_test = np.concatenate(test_blocks, axis=1)
        ensemble = _FittedRegressionEnsemble(fitted, aggregation, weights)
        pred = ensemble.predict(X_test)
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": X_train,
                "X_test": X_test,
                "feature_names": feature_names,
                "coordinate_metadata": coordinate_metadata,
                "y_test": np.asarray(dataset.y[test_idx], dtype=float),
                "y_pred": pred,
                "estimator": ensemble,
            }
        )
        if progress_callback is not None:
            progress_callback(fold_no, len(splits), str(split["split_key"]))
    return dataset, folds


def _metric_name(value: str, fallback: str) -> str:
    token = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    compact = token.replace("_", "")
    if token in _REGRESSION_METRICS:
        return _REGRESSION_METRICS[token]
    if compact in _REGRESSION_METRICS:
        return _REGRESSION_METRICS[compact]
    fallback_token = str(fallback).strip()
    if fallback_token in compute_regression_metrics(
        np.array([0.0, 1.0]), np.array([0.0, 1.0])
    ):
        return fallback_token
    return "RMSE"


def _sample_indices(n: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n), size=max_rows, replace=False))


def _regression_local_sample_pairs(
    dataset: Any,
    folds: Sequence[dict[str, Any]],
    mode: str,
    sample_ids: Sequence[str],
    quantiles: Sequence[float],
) -> list[tuple[int, str]]:
    sid_to_idx = {str(sid): i for i, sid in enumerate(dataset.sample_ids)}
    selected: list[tuple[int, str]] = []
    if mode in {"requested", "representative_and_requested"}:
        for sid in sample_ids:
            if str(sid) not in sid_to_idx:
                raise ValueError(
                    f"Requested instance sample_id {sid!r} was not found in the loaded dataset."
                )
            selected.append((int(sid_to_idx[str(sid)]), f"requested:{sid}"))
    if mode in {"representative", "representative_and_requested"}:
        records: list[dict[str, Any]] = []
        for fold in folds:
            test_idx = np.asarray(fold["test_idx"], dtype=int)
            pred = np.asarray(fold.get("y_pred"), dtype=float).reshape(-1)
            if len(pred) != len(test_idx):
                continue
            for local_i, sample_index in enumerate(test_idx):
                records.append(
                    {
                        "sample_index": int(sample_index),
                        "prediction": float(pred[int(local_i)]),
                    }
                )
        frame = pd.DataFrame(records)
        if not frame.empty:
            frame = frame.groupby("sample_index", as_index=False).agg(
                prediction=("prediction", "mean")
            )
            used = {int(i) for i, _ in selected}
            for quantile in quantiles:
                q = float(quantile)
                centre = float(frame["prediction"].quantile(q))
                ranked = frame.assign(
                    _dist=(
                        pd.to_numeric(frame["prediction"], errors="coerce") - centre
                    ).abs()
                ).sort_values(["_dist", "sample_index"])
                row = next(
                    (
                        r
                        for _, r in ranked.iterrows()
                        if int(r["sample_index"]) not in used
                    ),
                    ranked.iloc[0],
                )
                idx = int(row["sample_index"])
                used.add(idx)
                selected.append(
                    (idx, f"representative_prediction_q{int(round(q * 100)):02d}")
                )
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for sample_index, role in selected:
        if int(sample_index) in seen:
            continue
        seen.add(int(sample_index))
        out.append((int(sample_index), str(role)))
    return out


def _regression_local_records(
    fold: dict[str, Any],
    method: str,
    values: np.ndarray,
    rows_ex: Sequence[int],
    sample_ids: Sequence[Any],
    selected_roles: dict[int, str],
    local_top_k: int,
) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected regression local attribution shape {arr.shape}.")
    test_idx = np.asarray(fold["test_idx"], dtype=int)
    X_test = np.asarray(fold["X_test"], dtype=float)
    names = [str(x) for x in fold["feature_names"]]
    y_test = np.asarray(fold["y_test"], dtype=float).reshape(-1)
    y_pred = np.asarray(fold["y_pred"], dtype=float).reshape(-1)
    records: list[dict[str, Any]] = []
    k = max(1, int(local_top_k))
    for value_row, local_test_row in enumerate(rows_ex):
        local_i = int(local_test_row)
        global_i = int(test_idx[local_i])
        vals = arr[int(value_row)]
        is_selected = global_i in selected_roles
        chosen = np.argsort(-np.abs(vals))
        if not is_selected:
            chosen = chosen[: min(k, len(chosen))]
        for rank, feature_index in enumerate(chosen, start=1):
            j = int(feature_index)
            value = float(vals[j])
            records.append(
                {
                    "method": str(method),
                    "sample_id": str(sample_ids[global_i]),
                    "sample_index": global_i,
                    "split_key": str(fold.get("split_key", "")),
                    "selection_role": selected_roles.get(global_i, ""),
                    "selected_for_report": bool(is_selected),
                    "observed_response": float(y_test[local_i]),
                    "prediction": float(y_pred[local_i]),
                    "feature": names[j],
                    "feature_value": float(X_test[local_i, j]),
                    "attribution": value,
                    "abs_attribution": abs(value),
                    "local_rank": int(rank),
                    "retained_scope": "representative_full" if is_selected else "top_k",
                }
            )
    return pd.DataFrame(records)


def _permutation_importance(
    fold: dict[str, Any],
    spec: Permutation,
    fallback_metric: str,
    seed: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    X = np.asarray(fold["X_test"], dtype=float)
    y = np.asarray(fold["y_test"], dtype=float)
    names = list(fold["feature_names"])
    metric = _metric_name(str(spec.scoring), fallback_metric)
    if isinstance(spec.max_samples, float):
        n = max(1, min(len(X), int(np.ceil(len(X) * float(spec.max_samples)))))
    else:
        n = max(1, min(len(X), int(spec.max_samples)))
    indices = _sample_indices(len(X), n, seed)
    X_eval = X[indices]
    y_eval = y[indices]
    baseline_pred = np.asarray(
        _estimator_call(fold["estimator"], "predict", X_eval), dtype=float
    ).reshape(-1)
    baseline = compute_regression_metrics(y_eval, baseline_pred)[metric]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        values: list[float] = []
        for _ in range(int(spec.n_repeats)):
            permuted = X_eval.copy()
            permuted[:, j] = permuted[rng.permutation(len(permuted)), j]
            pred = np.asarray(
                _estimator_call(fold["estimator"], "predict", permuted), dtype=float
            ).reshape(-1)
            score = compute_regression_metrics(y_eval, pred)[metric]
            values.append(
                float(score - baseline if metric_is_loss(metric) else baseline - score)
            )
        rows.append(
            {
                "method": "permutation",
                "feature": str(name),
                "importance_mean": float(np.mean(values)),
                "signed_importance_mean": float(np.mean(values)),
                "within_fold_importance_sd": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "scoring": f"increase_in_{metric}"
                if metric_is_loss(metric)
                else f"decrease_in_{metric}",
            }
        )
        if progress_callback is not None:
            progress_callback(j + 1, len(names), str(name))
    return pd.DataFrame(rows)


def _shap_importance(
    fold: dict[str, Any],
    spec: SHAP,
    seed: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    local_sample_ids: Sequence[Any] | None = None,
    selected_roles: dict[int, str] | None = None,
    local_top_k: int | None = None,
) -> pd.DataFrame:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("SHAP is required for regression explainability.") from exc
    X_train = np.asarray(fold["X_train"], dtype=float)
    X_test = np.asarray(fold["X_test"], dtype=float)
    names = list(fold["feature_names"])
    bg_rows = _sample_indices(len(X_train), int(spec.background_size), seed)
    ex_rows = (
        np.arange(len(X_test), dtype=int)
        if local_top_k is not None
        else _sample_indices(len(X_test), int(spec.max_explain), seed + 17)
    )
    background = X_train[bg_rows]
    X_explain = X_test[ex_rows]
    if progress_callback is not None:
        progress_callback(
            1, 4, f"background {len(background)} · explain {len(X_explain)}"
        )
    model = fold["estimator"]
    requested = str(spec.algorithm).strip().casefold()
    values = None
    backend = "permutation"
    if requested in {"auto", "tree"} and not isinstance(
        model, _FittedRegressionEnsemble
    ):
        try:
            explainer = shap.TreeExplainer(
                model,
                data=background,
                model_output="raw",
                feature_perturbation="interventional",
                feature_names=names,
            )
            values = explainer(X_explain).values
            backend = "tree"
            if progress_callback is not None:
                progress_callback(3, 4, "tree attributions")
        except Exception:
            if requested == "tree":
                raise
    if values is None:
        if str(spec.masker).strip().casefold() == "partition":
            masker = shap.maskers.Partition(
                background, max_samples=len(background), clustering="correlation"
            )
        else:
            masker = shap.maskers.Independent(background, max_samples=len(background))
        algorithm = "permutation" if requested == "auto" else requested
        minimum = 2 * len(names) + 1
        max_evals = max(minimum, minimum * max(1, int(spec.permutation_rounds)))
        explainer = shap.Explainer(
            lambda x: np.asarray(
                _estimator_call(model, "predict", np.asarray(x, dtype=float)),
                dtype=float,
            ),
            masker,
            algorithm=algorithm,
            feature_names=names,
            seed=int(seed),
        )
        if progress_callback is not None:
            progress_callback(2, 4, f"{algorithm} explainer")
        try:
            values = explainer(X_explain, max_evals=max_evals, silent=True).values
        except TypeError:
            values = explainer(X_explain, max_evals=max_evals).values
        backend = algorithm
        if progress_callback is not None:
            progress_callback(3, 4, f"{algorithm} attributions")
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(f"Unexpected regression SHAP shape {arr.shape}.")
    rows = []
    for j, name in enumerate(names):
        vals = arr[:, j]
        rows.append(
            {
                "method": "shap",
                "feature": str(name),
                "importance_mean": float(np.nanmean(np.abs(vals))),
                "signed_importance_mean": float(np.nanmean(vals)),
                "within_fold_importance_sd": float(np.nanstd(np.abs(vals), ddof=1))
                if len(vals) > 1
                else 0.0,
                "scoring": "mean_abs_regression_shap",
                "backend": backend,
            }
        )
    if progress_callback is not None:
        progress_callback(4, 4, f"{len(names)} features")
    frame = pd.DataFrame(rows)
    if local_top_k is not None and local_sample_ids is not None:
        frame.attrs["local_explanations"] = _regression_local_records(
            fold,
            "shap",
            arr,
            ex_rows,
            local_sample_ids,
            selected_roles or {},
            int(local_top_k),
        )
    return frame


def _lime_importance(
    fold: dict[str, Any],
    spec: LIME,
    seed: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    local_sample_ids: Sequence[Any] | None = None,
    selected_roles: dict[int, str] | None = None,
    local_top_k: int | None = None,
) -> pd.DataFrame:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        raise RuntimeError("LIME is required for regression explainability.") from exc
    X_train = np.asarray(fold["X_train"], dtype=float)
    X_test = np.asarray(fold["X_test"], dtype=float)
    names = list(fold["feature_names"])
    rows = (
        np.arange(len(X_test), dtype=int)
        if local_top_k is not None
        else _sample_indices(len(X_test), int(spec.max_explain), seed + 23)
    )
    kwargs: dict[str, Any] = {
        "training_data": X_train,
        "feature_names": names,
        "mode": "regression",
        "discretize_continuous": bool(spec.discretize_continuous),
        "feature_selection": str(spec.feature_selection),
        "random_state": int(seed),
    }
    if spec.kernel_width is not None:
        kwargs["kernel_width"] = float(spec.kernel_width)
    explainer = LimeTabularExplainer(**kwargs)
    values: list[np.ndarray] = []
    model = fold["estimator"]
    for sample_no, index in enumerate(rows, start=1):
        call_kwargs: dict[str, Any] = {
            "num_features": len(names),
            "num_samples": int(spec.num_samples),
        }
        if str(spec.sampling_method) != "gaussian":
            call_kwargs["sampling_method"] = str(spec.sampling_method)
        exp = explainer.explain_instance(
            X_test[int(index)],
            lambda x: np.asarray(
                _estimator_call(model, "predict", np.asarray(x, dtype=float)),
                dtype=float,
            ),
            **call_kwargs,
        )
        mapping = exp.as_map()
        key = 1 if 1 in mapping else next(iter(mapping))
        coeff = np.zeros(len(names), dtype=float)
        for feature_index, weight in mapping[key]:
            if 0 <= int(feature_index) < len(coeff):
                coeff[int(feature_index)] = float(weight)
        values.append(coeff)
        if progress_callback is not None:
            progress_callback(sample_no, len(rows), f"sample {int(index)}")
    arr = np.vstack(values) if values else np.zeros((0, len(names)), dtype=float)
    frame = pd.DataFrame(
        [
            {
                "method": "lime",
                "feature": str(name),
                "importance_mean": float(np.mean(np.abs(arr[:, j])))
                if len(arr)
                else np.nan,
                "signed_importance_mean": float(np.mean(arr[:, j]))
                if len(arr)
                else np.nan,
                "within_fold_importance_sd": float(np.std(np.abs(arr[:, j]), ddof=1))
                if len(arr) > 1
                else 0.0,
                "scoring": "mean_abs_regression_lime_coefficient",
            }
            for j, name in enumerate(names)
        ]
    )
    if local_top_k is not None and local_sample_ids is not None:
        frame.attrs["local_explanations"] = _regression_local_records(
            fold,
            "lime",
            arr,
            rows,
            local_sample_ids,
            selected_roles or {},
            int(local_top_k),
        )
    return frame


def _ale_importance(
    fold: dict[str, Any],
    spec: ALE,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pyale = _require_pyale()
    X = np.asarray(fold["X_test"], dtype=float)
    names = list(fold["feature_names"])
    frame = pd.DataFrame(X, columns=names)
    wrapper = _AleModelWrapper(
        lambda x: np.asarray(
            _estimator_call(fold["estimator"], "predict", np.asarray(x, dtype=float)),
            dtype=float,
        )
    )
    bins = _auto_ale_bins(len(X), spec)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for feature_no, name in enumerate(names, start=1):
        col = frame[name].to_numpy(dtype=float)
        finite = col[np.isfinite(col)]
        if len(np.unique(finite)) < 2:
            if progress_callback is not None:
                progress_callback(feature_no, len(names), str(name))
            continue
        try:
            with _quiet_pyale_info():
                result = pyale(
                    frame,
                    wrapper,
                    [name],
                    grid_size=int(bins),
                    include_CI=False,
                    plot=False,
                )
            vals, grid, _ = _ale_result_values(result)
        except Exception:
            if progress_callback is not None:
                progress_callback(feature_no, len(names), str(name))
            continue
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            if progress_callback is not None:
                progress_callback(feature_no, len(names), str(name))
            continue
        centred = vals - float(np.mean(vals))
        rows.append(
            {
                "method": "ale",
                "feature": str(name),
                "importance_mean": float(np.sqrt(np.mean(centred**2))),
                "signed_importance_mean": float(np.mean(vals)),
                "within_fold_importance_sd": float(np.std(centred, ddof=1))
                if len(centred) > 1
                else 0.0,
                "scoring": "rms_centered_regression_ale",
            }
        )
        if grid is not None and len(grid) == len(vals):
            for x, effect in zip(grid, vals):
                curves.append(
                    {
                        "feature": str(name),
                        "grid_value": float(x),
                        "ale_effect": float(effect),
                    }
                )
        if progress_callback is not None:
            progress_callback(feature_no, len(names), str(name))
    return pd.DataFrame(rows), pd.DataFrame(curves)


def _interaction_importance(
    fold: dict[str, Any],
    spec: ALEInteractions,
    seed_scores: pd.DataFrame,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    pyale = _require_pyale()
    X = np.asarray(fold["X_test"], dtype=float)
    names = list(fold["feature_names"])
    if len(names) < 2:
        return pd.DataFrame()
    score_map = (
        {
            str(row["feature"]): float(row["importance_mean"])
            for _, row in seed_scores.iterrows()
        }
        if not seed_scores.empty
        else {}
    )
    ranked = sorted(
        range(len(names)),
        key=lambda i: score_map.get(names[i], float(np.nanvar(X[:, i]))),
        reverse=True,
    )
    frontier = ranked[
        : min(len(ranked), max(4, int(np.ceil(np.sqrt(int(spec.top_k) * 2))) + 6))
    ]
    candidates: list[tuple[float, int, int]] = []
    for pos, i in enumerate(frontier):
        for j in frontier[pos + 1 :]:
            score = score_map.get(names[i], float(np.nanvar(X[:, i]))) + score_map.get(
                names[j], float(np.nanvar(X[:, j]))
            )
            candidates.append((float(score), i, j))
    candidates.sort(reverse=True)
    frame = pd.DataFrame(X, columns=names)
    wrapper = _AleModelWrapper(
        lambda x: np.asarray(
            _estimator_call(fold["estimator"], "predict", np.asarray(x, dtype=float)),
            dtype=float,
        )
    )
    bins = _auto_ale_bins(len(X), spec)
    rows: list[dict[str, Any]] = []
    selected = candidates[: int(spec.top_k)]
    for pair_no, (_, i, j) in enumerate(selected, start=1):
        f1, f2 = names[i], names[j]
        try:
            with _quiet_pyale_info():
                result = pyale(
                    frame,
                    wrapper,
                    [f1, f2],
                    grid_size=int(bins),
                    include_CI=False,
                    plot=False,
                )
            vals, _, _ = _ale_result_values(result)
        except Exception:
            if progress_callback is not None:
                progress_callback(pair_no, len(selected), f"{f1} × {f2}")
            continue
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            if progress_callback is not None:
                progress_callback(pair_no, len(selected), f"{f1} × {f2}")
            continue
        centred = vals - float(np.mean(vals))
        rows.append(
            {
                "feature_1": f1,
                "feature_2": f2,
                "interaction_strength": float(np.sqrt(np.mean(centred**2))),
                "scoring": "rms_centered_regression_2d_ale",
            }
        )
        if progress_callback is not None:
            progress_callback(pair_no, len(selected), f"{f1} × {f2}")
    return (
        pd.DataFrame(rows).sort_values("interaction_strength", ascending=False)
        if rows
        else pd.DataFrame()
    )


def _aggregate_long_importance(
    long: pd.DataFrame, n_folds: int, top_k: int
) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    d = long.copy()
    d["method"] = d["method"].astype(str)
    d["feature"] = d["feature"].astype(str)
    d["_importance"] = pd.to_numeric(d["importance_mean"], errors="coerce")
    fold_col = "fold_key" if "fold_key" in d.columns else "fold_no"
    if fold_col not in d.columns:
        d["fold_no"] = 1
        fold_col = "fold_no"
    d["_rank"] = d.groupby(["method", fold_col], sort=False)["_importance"].rank(
        ascending=False, method="average"
    )
    d["_top_k"] = d["_rank"].le(max(1, int(top_k)))
    rows: list[dict[str, Any]] = []
    for (method, feature), group in d.groupby(["method", "feature"], sort=True):
        valid = group["_importance"].notna() & np.isfinite(
            group["_importance"].to_numpy(dtype=float)
        )
        values = group.loc[valid, "_importance"].to_numpy(dtype=float)
        ranks = (
            pd.to_numeric(group.loc[valid, "_rank"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        signed = (
            pd.to_numeric(
                group.loc[valid, "signed_importance_mean"]
                if "signed_importance_mean" in group.columns
                else pd.Series(dtype=float),
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        if not len(values):
            continue
        if ranks.size:
            rq25, rmed, rq75 = np.quantile(ranks, [0.25, 0.5, 0.75])
        else:
            rq25 = rmed = rq75 = np.nan
        rows.append(
            {
                "method": str(method),
                "feature": str(feature),
                "importance_mean": float(np.mean(values)),
                "importance_sd": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "importance_median": float(np.median(values)),
                "signed_importance_mean": float(np.mean(signed))
                if len(signed)
                else np.nan,
                "mean_rank": float(np.mean(ranks)) if ranks.size else np.nan,
                "median_rank": float(rmed),
                "rank_iqr": float(rq75 - rq25) if ranks.size else np.nan,
                "top_k_frequency": float(
                    group.loc[valid, "_top_k"].astype(float).mean()
                ),
                "n_estimable_folds": int(len(values)),
                "n_outer_folds_total": int(n_folds),
                "fold_coverage": float(len(values) / max(1, int(n_folds))),
                "scoring": str(group["scoring"].dropna().iloc[0])
                if "scoring" in group.columns and not group["scoring"].dropna().empty
                else "",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_within_method"] = out.groupby("method")["importance_mean"].rank(
        ascending=False, method="average"
    )
    out["top_k"] = out["rank_within_method"] <= int(top_k)
    return out.sort_values(
        ["method", "importance_mean", "feature"], ascending=[True, False, True]
    )


def _aggregate_frames(
    frames: list[pd.DataFrame], n_folds: int, top_k: int
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True, sort=False)
    return _aggregate_long_importance(long, int(n_folds), int(top_k))


def _combined_feature_table(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    pivot = importance.pivot_table(
        index="feature", columns="method", values="importance_mean", aggfunc="first"
    )
    ranks = pivot.rank(axis=0, ascending=False, method="average")
    out = pivot.copy()
    out["mean_rank"] = ranks.mean(axis=1, skipna=True)
    out["methods_available"] = pivot.notna().sum(axis=1)
    return out.reset_index().sort_values(
        ["mean_rank", "feature"], ascending=[True, True]
    )


def _write_feature_support_figure(
    target_dir: Path, importance: pd.DataFrame, top_k: int
) -> dict[str, Path]:
    stem = target_dir / "figures" / "feature_support"
    if not plot_regression_feature_support(importance, stem, int(top_k)):
        return {}
    return {"feature_support_svg": stem.with_suffix(".svg")}


def _write_regression_ale_figure(
    target_dir: Path, importance: pd.DataFrame, curves: pd.DataFrame, top_k: int
) -> dict[str, Path]:
    if curves.empty:
        return {}
    curve_table = curves.rename(columns={"grid_value": "grid"}).copy()
    if not {"feature", "grid", "ale_effect"}.issubset(curve_table.columns):
        return {}
    ale = (
        importance[
            importance.get("method", pd.Series(dtype=object)).astype(str).eq("ale")
        ].copy()
        if not importance.empty and "method" in importance.columns
        else pd.DataFrame()
    )
    if not ale.empty and "importance_mean" in ale.columns:
        ale["importance_mean"] = pd.to_numeric(ale["importance_mean"], errors="coerce")
        top_features = (
            ale.dropna(subset=["importance_mean"])
            .sort_values("importance_mean", ascending=False)
            .head(int(top_k))["feature"]
            .astype(str)
            .tolist()
        )
    else:
        top_features = list(dict.fromkeys(curve_table["feature"].astype(str).tolist()))[
            : int(top_k)
        ]
    stem = target_dir / "figures" / "ale_curves"
    _plot_ale_curves(curve_table, top_features, stem, max_panels=min(12, int(top_k)))
    return {"ale_curves_svg": stem.with_suffix(".svg")}


def _write_regression_interaction_figures(
    target_dir: Path, interactions: pd.DataFrame, top_k: int
) -> dict[str, Path]:
    if interactions.empty:
        return {}
    table = interactions.copy()
    if (
        "interaction_strength" not in table.columns
        and "interaction_strength_mean" in table.columns
    ):
        table = table.rename(
            columns={"interaction_strength_mean": "interaction_strength"}
        )
    if not {"feature_1", "feature_2", "interaction_strength"}.issubset(table.columns):
        return {}
    table["interaction_strength"] = pd.to_numeric(
        table["interaction_strength"], errors="coerce"
    )
    table = (
        table.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["interaction_strength"])
        .sort_values("interaction_strength", ascending=False)
    )
    if table.empty:
        return {}
    figures = target_dir / "figures"
    stem = figures / "interaction_network_current"
    plot_interaction_network(
        table, pd.DataFrame(), stem, int(top_k), None, layout="default"
    )
    return {"interaction_network_current_svg": stem.with_suffix(".svg")}


def _write_regression_explainability_figures(
    target_dir: Path, importance: pd.DataFrame | None = None, top_k: int = 15
) -> dict[str, Path]:
    frame = importance if importance is not None else pd.DataFrame()
    if frame.empty:
        path = target_dir / "feature_stability.parquet"
        if table_exists(path) and path.stat().st_size:
            frame = read_table(path)
    if not frame.empty and "top_k_frequency" not in frame.columns:
        fold_path = target_dir / "fold_feature_importance.parquet"
        if table_exists(fold_path) and fold_path.stat().st_size:
            fold_table = read_table(fold_path)
            if not fold_table.empty:
                fold_col = "fold_key" if "fold_key" in fold_table.columns else "fold_no"
                n_folds = (
                    int(fold_table[fold_col].astype(str).nunique())
                    if fold_col in fold_table.columns
                    else 1
                )
                frame = _aggregate_long_importance(
                    fold_table, max(1, n_folds), int(top_k)
                )
    outputs = (
        _write_feature_support_figure(target_dir, frame, int(top_k))
        if not frame.empty
        else {}
    )
    curve_path = target_dir / "ale_curves.parquet"
    if table_exists(curve_path):
        outputs.update(
            _write_regression_ale_figure(
                target_dir, frame, read_table(curve_path), int(top_k)
            )
        )
    interaction_path = target_dir / "ale_interactions.parquet"
    if table_exists(interaction_path):
        outputs.update(
            _write_regression_interaction_figures(
                target_dir, read_table(interaction_path), int(top_k)
            )
        )
    return outputs


def _explain_target(
    sweep: Sweep, configs: pd.DataFrame, models: dict[str, Any], target: str
) -> dict[str, Path]:
    root = Path(sweep.root())
    if target == "mpma_b":
        row = _config_row(configs, str(models["MPMA-B"]["config_id"]))
        slug = "mpma_b"
        unit = {"type": "MPMA-B", "config_id": str(row["config_id"])}
    elif target == "mpma_e":
        if not isinstance(models.get("MPMA-E"), dict):
            raise ValueError(
                "MPMA-E regression explainability was requested, but no final MPMA-E exists."
            )
        slug = "mpma_e"
        unit = {
            "type": "MPMA-E",
            "ensemble_config_id": str(models["MPMA-E"].get("ensemble_config_id", "")),
            "aggregation_strategy": str(
                models["MPMA-E"].get("aggregation_strategy", "")
            ),
        }
    else:
        row = _config_row(configs, target)
        slug = str(target)
        unit = {"type": "MPMA", "config_id": str(row["config_id"])}
    with progress() as prog:
        refit_task = prog.add_task(f"{slug} · refitting outer folds", total=1)

        def refit_progress(completed: int, total: int, detail: str) -> None:
            prog.update(
                refit_task,
                total=max(1, int(total)),
                completed=int(completed),
                description=f"{slug} · refit {int(completed)}/{int(total)} · {detail}",
            )

        if target == "mpma_e":
            dataset, folds = _fit_ensemble_folds(
                sweep,
                configs,
                models["MPMA-E"],
                progress_callback=refit_progress,
            )
        else:
            dataset, folds = _fit_individual_folds(
                sweep, row, progress_callback=refit_progress
            )
        prog.update(
            refit_task,
            total=max(1, len(folds)),
            completed=max(1, len(folds)),
            description=f"{slug} · refit complete · {len(folds)} outer folds",
        )
    target_dir = root / "explainability" / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    coordinate_rows: list[dict[str, Any]] = []
    for fold_no, fold in enumerate(folds, start=1):
        for item in fold.get("coordinate_metadata", []):
            coordinate_rows.append(
                {
                    "fold_no": int(fold_no),
                    "split_key": str(fold.get("split_key", "")),
                    **item,
                }
            )
    coordinate_path: Path | None = None
    if coordinate_rows:
        coordinate_path = target_dir / "coordinate_metadata.parquet"
        write_table(coordinate_path, pd.DataFrame(coordinate_rows).drop_duplicates())
    methods = tuple(method_name(method) for method in sweep.explainability.methods)
    global_methods = tuple(
        method_name(method)
        for method in sweep.explainability.methods
        if method_has_global(method)
    )
    local_methods = tuple(
        method_name(method)
        for method in sweep.explainability.methods
        if method_has_local(method)
    )
    local_mode = _effective_local_explanations_mode(sweep.explainability)
    local_enabled = local_mode != "none" and bool(local_methods)
    selected_pairs = (
        _regression_local_sample_pairs(
            dataset,
            folds,
            local_mode,
            sweep.explainability.local.sample_ids,
            sweep.explainability.local.regression_quantiles,
        )
        if local_enabled
        else []
    )
    selected_roles = {int(i): str(role) for i, role in selected_pairs}
    frames: list[pd.DataFrame] = []
    local_frames: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    interaction_frames: list[pd.DataFrame] = []
    fallback_metric = str(sweep.evaluation.optimize_metric)
    method_specs = tuple(sweep.explainability.methods)
    with progress() as prog:
        total_methods = max(1, len(folds) * len(method_specs))
        overall_task = prog.add_task(
            f"{slug} · explainability methods", total=total_methods
        )
        fold_task = prog.add_task("outer fold", total=max(1, len(method_specs)))
        method_task = prog.add_task("method", total=1)
        completed_methods = 0
        for fold_no, fold in enumerate(folds, start=1):
            split_key = str(fold["split_key"])
            prog.update(
                fold_task,
                total=max(1, len(method_specs)),
                completed=0,
                description=f"{slug} · fold {fold_no}/{len(folds)} · {split_key}",
            )
            fold_seed = int(sweep.explainability.random_state) + fold_no * 1009
            seed_scores = pd.DataFrame()
            for method_no, spec in enumerate(method_specs, start=1):
                name = method_name(spec)
                global_method = method_has_global(spec)
                local_method = local_enabled and method_has_local(spec)
                prog.update(
                    method_task,
                    total=1,
                    completed=0,
                    description=f"{slug} · fold {fold_no}/{len(folds)} · {name} · starting",
                )

                def method_progress(
                    completed: int,
                    total: int,
                    detail: str,
                    method_name_value: str = name,
                    fold_number: int = fold_no,
                ) -> None:
                    total_value = max(1, int(total))
                    completed_value = min(int(completed), total_value)
                    stride = max(1, total_value // 100)
                    if completed_value < total_value and completed_value % stride:
                        return
                    prog.update(
                        method_task,
                        total=total_value,
                        completed=completed_value,
                        description=(
                            f"{slug} · fold {fold_number}/{len(folds)} · "
                            f"{method_name_value} · {detail}"
                        ),
                    )

                if name == "permutation":
                    frame = _permutation_importance(
                        fold,
                        spec,
                        fallback_metric,
                        fold_seed,
                        progress_callback=method_progress,
                    )
                elif name == "shap":
                    frame = _shap_importance(
                        fold,
                        spec,
                        fold_seed,
                        progress_callback=method_progress,
                        local_sample_ids=dataset.sample_ids if local_method else None,
                        selected_roles=selected_roles if local_method else None,
                        local_top_k=int(sweep.explainability.local.stored_features)
                        if local_method
                        else None,
                    )
                elif name == "lime":
                    frame = _lime_importance(
                        fold,
                        spec,
                        fold_seed,
                        progress_callback=method_progress,
                        local_sample_ids=dataset.sample_ids if local_method else None,
                        selected_roles=selected_roles if local_method else None,
                        local_top_k=int(sweep.explainability.local.stored_features)
                        if local_method
                        else None,
                    )
                elif name == "ale":
                    frame, curve = _ale_importance(
                        fold, spec, progress_callback=method_progress
                    )
                    if not curve.empty:
                        curve["fold_no"] = fold_no
                        curve["fold_key"] = split_key
                        curves.append(curve)
                elif name == "interactions":
                    frame = pd.DataFrame()
                    interactions = _interaction_importance(
                        fold, spec, seed_scores, progress_callback=method_progress
                    )
                    if not interactions.empty:
                        interactions["fold_no"] = fold_no
                        interactions["fold_key"] = split_key
                        interaction_frames.append(interactions)
                else:
                    raise ValueError(
                        f"Unsupported regression explainability method {name!r}."
                    )
                if not frame.empty:
                    local_frame = frame.attrs.get("local_explanations")
                    if isinstance(local_frame, pd.DataFrame) and not local_frame.empty:
                        local_frames.append(local_frame.copy())
                    if global_method:
                        frame = frame.copy()
                        frame["fold_no"] = fold_no
                        frame["fold_key"] = split_key
                        frames.append(frame)
                        if seed_scores.empty or name in {"shap", "permutation"}:
                            seed_scores = frame
                prog.update(
                    method_task,
                    total=1,
                    completed=1,
                    description=f"{slug} · fold {fold_no}/{len(folds)} · {name} · complete",
                )
                prog.update(fold_task, completed=method_no)
                completed_methods += 1
                prog.update(
                    overall_task,
                    completed=completed_methods,
                    description=(
                        f"{slug} · explainability · {completed_methods}/{total_methods} "
                        f"fold-method units"
                    ),
                )
    with phase_progress(f"{slug} explainability outputs", 4) as phase:
        phase.phase("aggregate feature attribution")
        importance = _aggregate_frames(
            frames, len(folds), int(sweep.explainability.top_k)
        )
        combined = _combined_feature_table(importance)
        importance_path = target_dir / "feature_stability.parquet"
        combined_path = target_dir / "feature_importance.parquet"
        phase.phase("write fold and stability tables")
        write_table(importance_path, importance)
        write_table(combined_path, combined)
        fold_path = target_dir / "fold_feature_importance.parquet"
        write_table(
            fold_path,
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame(),
        )
        outputs: dict[str, Path] = {
            "explainability_dir": target_dir,
            "importance": combined_path,
            "stability": importance_path,
            "fold_importance": fold_path,
        }
        if coordinate_path is not None:
            outputs["coordinate_metadata"] = coordinate_path
        local_table = (
            pd.concat(local_frames, ignore_index=True, sort=False)
            if local_frames
            else pd.DataFrame()
        )
        if local_enabled:
            local_path = target_dir / "local_explanations.parquet"
            write_table(local_path, local_table)
            outputs["local_explanations"] = local_path
        phase.phase("write ALE and interaction tables")
        if curves:
            curve_path = target_dir / "ale_curves.parquet"
            write_table(curve_path, pd.concat(curves, ignore_index=True, sort=False))
            outputs["ale_curves"] = curve_path
        if interaction_frames:
            interactions = pd.concat(interaction_frames, ignore_index=True, sort=False)
            summary = interactions.groupby(
                ["feature_1", "feature_2"], as_index=False
            ).agg(
                interaction_strength_mean=("interaction_strength", "mean"),
                interaction_strength_sd=("interaction_strength", "std"),
                n_folds=("interaction_strength", "count"),
            )
            interaction_path = target_dir / "ale_interactions.parquet"
            write_table(
                interaction_path,
                summary.sort_values("interaction_strength_mean", ascending=False),
            )
            outputs["interactions"] = interaction_path
        phase.phase("render explainability figures")
        outputs.update(
            _write_regression_explainability_figures(
                target_dir, importance, int(sweep.explainability.top_k)
            )
        )
        if local_enabled and not local_table.empty and selected_pairs:
            local_figure = plot_local_attributions(
                local_table,
                target_dir / "figures" / "local_explanations",
                task="regression",
                top_n=int(sweep.explainability.local.displayed_features),
            )
            if local_figure is not None:
                outputs["local_explanations_figure"] = local_figure
    target_col = (
        sweep.samples.target_col
        if getattr(sweep, "uses_modalities", False)
        else sweep.data.target_col
    )
    explanation_spaces = sorted(
        {str(fold.get("explanation_space", "model_coordinates")) for fold in folds}
    )
    explained = {
        "task": "regression",
        "target": str(target_col),
        "unit": unit,
        "methods": list(methods),
        "outer_folds": int(len(folds)),
        "sample_count": int(len(dataset.y)),
        "top_k": int(sweep.explainability.top_k),
        "local_explanations": local_mode,
        "local_methods": list(local_methods) if local_enabled else [],
        "global_methods": list(global_methods),
        "local_top_k": int(sweep.explainability.local.stored_features),
        "local_target": "predicted_response",
        "local_storage": "top_k_per_oof_split_full_vectors_for_report_representatives",
        "top_instance_features": int(sweep.explainability.local.displayed_features),
        "representative_quantiles": [
            float(x) for x in sweep.explainability.local.regression_quantiles
        ],
        "explanation_space": explanation_spaces[0]
        if len(explanation_spaces) == 1
        else explanation_spaces,
    }
    meta_path = target_dir / "explained_unit.json"
    dump_json_standard(explained, meta_path)
    outputs["metadata"] = meta_path
    return outputs


def explain_regression(sweep: Sweep) -> dict[str, Path]:
    root = Path(sweep.root())
    stage("Regression explainability", str(root))
    models = _selected_models(root)
    configs_path = root / "configs.parquet"
    if not table_exists(configs_path):
        raise FileNotFoundError("Run evaluate(sweep) before regression explainability.")
    configs = read_table(configs_path)
    targets = _targets(sweep, models)
    summary_table(
        "Regression explainability suite",
        {
            "targets": targets,
            "methods": [method_name(method) for method in sweep.explainability.methods],
            "top features": int(sweep.explainability.top_k),
            "local explanations": _effective_local_explanations_mode(
                sweep.explainability
            ),
        },
    )
    outputs: dict[str, Path] = {}
    for target_no, target in enumerate(targets, start=1):
        info(
            f"Regression explainability · target {target_no}/{len(targets)} · {target}"
        )
        result = _explain_target(sweep, configs, models, target)
        for key, path in result.items():
            outputs[f"{target}_{key}"] = path
    success("Regression explainability completed")
    path_table("Regression explainability outputs", outputs)
    return outputs
