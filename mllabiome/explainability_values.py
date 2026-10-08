from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .explainability_config import (
    ExplainabilityConfigurationError,
    ExplainabilityDependencyError,
)
from .explainability_methods import LIME, SHAP
from .explainability_model import _explain_predict_proba, _project_input, _sample_rows
from .utils import _as_float_matrix


def _shap_values_for_data(
    clf: BaseEstimator,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    random_state: int,
    spec: SHAP,
    force_explain_rows: Sequence[int] = (),
    show_progress: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    input_projector: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import shap
    except Exception as exc:
        raise ExplainabilityDependencyError(
            "Explainability.methods includes SHAP, but the 'shap' package is not importable."
        ) from exc
    X_background = _as_float_matrix(X_background)
    X_explain = _as_float_matrix(X_explain)
    rows_bg = _sample_rows(
        X_background, max_rows=int(spec.background_size), random_state=random_state
    )
    rows_ex = _sample_rows(
        X_explain, max_rows=int(spec.max_explain), random_state=random_state + 13
    )
    if force_explain_rows:
        forced = np.asarray(
            [int(i) for i in force_explain_rows if 0 <= int(i) < X_explain.shape[0]],
            dtype=int,
        )
        if forced.size:
            rows_ex = np.array(
                sorted(set(rows_ex.tolist()) | set(forced.tolist())), dtype=int
            )
    background = X_background[rows_bg]
    X_selected = X_explain[rows_ex]
    total_samples = max(1, len(X_selected))
    if progress_callback is not None:
        progress_callback(0, total_samples, f"0/{len(X_selected)} samples")

    def normalize_values(values: Any) -> np.ndarray:
        raw = getattr(values, "values", values)
        if isinstance(raw, list):
            arr = np.stack([np.asarray(x, dtype=float) for x in raw], axis=-1)
        else:
            arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2:
            if len(class_labels) == 2:
                arr = arr[:, :, None]
            else:
                raise ExplainabilityConfigurationError(
                    f"Unexpected multiclass SHAP value shape {arr.shape}."
                )
        if arr.ndim != 3 or arr.shape[1] != len(feature_names):
            raise ExplainabilityConfigurationError(
                f"Unexpected SHAP value shape {arr.shape}; expected n_samples × n_features × n_classes."
            )
        if arr.shape[2] not in {1, len(class_labels)}:
            raise ExplainabilityConfigurationError(
                f"Unexpected SHAP output dimension {arr.shape[2]} for {len(class_labels)} classes."
            )
        return arr

    def evaluate_in_batches(
        call: Callable[[np.ndarray], Any], backend: str
    ) -> np.ndarray:
        if progress_callback is None or len(X_selected) <= 1:
            return normalize_values(call(X_selected))
        batch_size = max(1, min(25, len(X_selected)))
        chunks: list[np.ndarray] = []
        for start in range(0, len(X_selected), batch_size):
            stop = min(len(X_selected), start + batch_size)
            chunks.append(normalize_values(call(X_selected[start:stop])))
            progress_callback(
                stop,
                total_samples,
                f"sample {stop}/{len(X_selected)} · backend={backend}",
            )
        return np.concatenate(chunks, axis=0)

    requested_algorithm = str(spec.algorithm).strip().lower()
    if requested_algorithm in {"auto", "tree"} and input_projector is None:
        try:
            explainer = shap.TreeExplainer(
                clf,
                data=background,
                model_output="probability",
                feature_perturbation="interventional",
                feature_names=list(feature_names),
            )
            arr = evaluate_in_batches(explainer, "tree")
            if progress_callback is not None and len(X_selected) <= 1:
                progress_callback(
                    len(X_selected),
                    total_samples,
                    f"sample {len(X_selected)}/{len(X_selected)} · backend=tree",
                )
            return arr, rows_ex, "tree"
        except Exception as exc:
            if requested_algorithm == "tree":
                raise ExplainabilityConfigurationError(
                    "TreeSHAP was requested but the fitted estimator is not supported in probability space."
                ) from exc
    if requested_algorithm == "tree" and input_projector is not None:
        raise ExplainabilityConfigurationError(
            "TreeSHAP cannot enforce the fitted compositional input constraint. Use SHAP algorithm='auto' or 'permutation' for this transformation."
        )
    classes = np.arange(len(class_labels), dtype=int)
    raw_model_fn = _explain_predict_proba(clf, classes)
    model_fn = (
        raw_model_fn
        if input_projector is None
        else lambda values: raw_model_fn(_project_input(values, input_projector))
    )
    masker_name = str(spec.masker).strip().lower()
    if masker_name == "independent":
        masker = shap.maskers.Independent(background, max_samples=len(background))
    elif masker_name == "partition":
        masker = shap.maskers.Partition(
            background, max_samples=len(background), clustering="correlation"
        )
    else:
        raise ExplainabilityConfigurationError(
            f"Unsupported SHAP masker {spec.masker!r}. Use 'independent' or 'partition'."
        )
    algorithm = "permutation" if requested_algorithm == "auto" else requested_algorithm
    minimum = 2 * len(feature_names) + 1
    max_evals = (
        max(minimum, 2 ** len(feature_names))
        if algorithm == "exact"
        else max(minimum, minimum * max(1, int(spec.permutation_rounds)))
    )
    try:
        explainer = shap.Explainer(
            model_fn,
            masker,
            algorithm=algorithm,
            feature_names=list(feature_names),
            output_names=list(class_labels),
            seed=int(random_state),
        )

        def call_explainer(batch: np.ndarray) -> Any:
            try:
                return explainer(
                    batch,
                    silent=not bool(show_progress)
                    if progress_callback is None
                    else True,
                    max_evals=max_evals,
                )
            except TypeError:
                try:
                    return explainer(batch, max_evals=max_evals)
                except TypeError:
                    return explainer(batch)

        backend = f"{algorithm}_projected" if input_projector is not None else algorithm
        arr = evaluate_in_batches(call_explainer, backend)
        if progress_callback is not None and len(X_selected) <= 1:
            progress_callback(
                len(X_selected),
                total_samples,
                f"sample {len(X_selected)}/{len(X_selected)} · backend={backend}",
            )
    except Exception as exc:
        raise ExplainabilityConfigurationError(
            "SHAP failed for an outer-test fold."
        ) from exc
    return arr, rows_ex, backend


def _value_frame_for_fold(
    method: str,
    values: np.ndarray,
    feature_names: Sequence[str],
    class_indices: Sequence[int],
    class_labels: Sequence[str],
    scoring: str,
    positive_class: int | None = None,
) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[1] != len(feature_names):
        raise ExplainabilityConfigurationError(
            f"Unexpected {method.upper()} value shape {arr.shape}."
        )
    if arr.shape[2] == len(class_labels):
        selected = arr[:, :, list(class_indices)]
    elif arr.shape[2] == 1 and len(class_labels) == 2:
        positive = 1 if positive_class is None else int(positive_class)
        selected = np.stack(
            [
                arr[:, :, 0] if int(class_index) == positive else -arr[:, :, 0]
                for class_index in class_indices
            ],
            axis=2,
        )
    elif (
        arr.shape[2] == len(class_indices)
        or arr.shape[2] == 1
        and len(class_indices) == 1
    ):
        selected = arr
    else:
        raise ExplainabilityConfigurationError(
            f"{method.upper()} output classes do not match configured explanation classes."
        )
    rows: list[dict[str, Any]] = []
    for pos, class_index in enumerate(class_indices):
        vals = selected[:, :, pos]
        abs_vals = np.abs(vals)
        mean_abs = np.nanmean(abs_vals, axis=0)
        signed = np.nanmean(vals, axis=0)
        within_sd = (
            np.nanstd(abs_vals, axis=0, ddof=1)
            if vals.shape[0] > 1
            else np.zeros(vals.shape[1], dtype=float)
        )
        for j, feature in enumerate(feature_names):
            rows.append(
                {
                    "method": method,
                    "class_index": int(class_index),
                    "class_label": str(class_labels[int(class_index)]),
                    "feature": str(feature),
                    "importance_mean": float(mean_abs[j]),
                    "signed_importance_mean": float(signed[j]),
                    "within_fold_importance_sd": float(within_sd[j]),
                    "scoring": scoring,
                }
            )
    return pd.DataFrame(rows)


def _lime_values_for_data(
    clf: BaseEstimator,
    X_training: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    class_indices: Sequence[int],
    *,
    random_state: int,
    spec: LIME,
    force_explain_rows: Sequence[int] = (),
    progress_callback: Callable[[int, int, str], None] | None = None,
    input_projector: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        raise ExplainabilityDependencyError(
            "Explainability.methods includes LIME, but the 'lime' package is not importable."
        ) from exc

    X_training = _as_float_matrix(X_training)
    X_explain = _as_float_matrix(X_explain)
    rows = _sample_rows(
        X_explain, max_rows=int(spec.max_explain), random_state=random_state + 29
    )
    if force_explain_rows:
        forced = np.asarray(
            [int(i) for i in force_explain_rows if 0 <= int(i) < X_explain.shape[0]],
            dtype=int,
        )
        if forced.size:
            rows = np.array(
                sorted(set(rows.tolist()) | set(forced.tolist())), dtype=int
            )
    X_selected = X_explain[rows]
    raw_predict_fn = _explain_predict_proba(
        clf, np.arange(len(class_labels), dtype=int)
    )
    predict_fn = (
        raw_predict_fn
        if input_projector is None
        else lambda values: raw_predict_fn(_project_input(values, input_projector))
    )
    kwargs: dict[str, Any] = {
        "training_data": X_training,
        "feature_names": list(feature_names),
        "class_names": list(class_labels),
        "mode": "classification",
        "feature_selection": str(spec.feature_selection),
        "discretize_continuous": bool(spec.discretize_continuous),
        "random_state": int(random_state),
    }
    if spec.kernel_width is not None:
        kwargs["kernel_width"] = float(spec.kernel_width)
    total = max(1, len(X_selected))
    if progress_callback is not None:
        progress_callback(0, total, f"0/{len(X_selected)} samples")
    try:
        explainer = LimeTabularExplainer(**kwargs)
        coeffs = np.zeros(
            (X_selected.shape[0], len(feature_names), len(class_indices)), dtype=float
        )
        labels = tuple(int(x) for x in class_indices)
        explain_method = explainer.explain_instance
        try:
            signature = inspect.signature(explain_method)
            parameters = signature.parameters
            accepts_sampling_method = "sampling_method" in parameters or any(
                item.kind is inspect.Parameter.VAR_KEYWORD
                for item in parameters.values()
            )
        except (TypeError, ValueError):
            accepts_sampling_method = False
        sampling_method = str(spec.sampling_method)
        if not accepts_sampling_method and sampling_method != "gaussian":
            raise ExplainabilityConfigurationError(
                f"Installed LIME does not support sampling_method={sampling_method!r}; use 'gaussian' or upgrade LIME."
            )
        for i, row in enumerate(X_selected):
            explain_kwargs: dict[str, Any] = {
                "predict_fn": predict_fn,
                "num_features": len(feature_names),
                "num_samples": int(spec.num_samples),
                "labels": labels,
            }
            if accepts_sampling_method:
                explain_kwargs["sampling_method"] = sampling_method
            exp = explain_method(row, **explain_kwargs)
            for class_pos, label in enumerate(labels):
                for fi, coef in exp.local_exp.get(label, []):
                    if 0 <= int(fi) < len(feature_names):
                        coeffs[i, int(fi), class_pos] = float(coef)
            if progress_callback is not None:
                progress_callback(i + 1, total, f"sample {i + 1}/{len(X_selected)}")
    except ExplainabilityConfigurationError:
        raise
    except Exception as exc:
        raise ExplainabilityConfigurationError(
            "LIME failed for an outer-test fold."
        ) from exc
    return coeffs, rows


def _aggregate_fold_feature_importance(
    method: str,
    frames: Sequence[pd.DataFrame],
    feature_names: Sequence[str],
    class_indices: Sequence[int],
    class_labels: Sequence[str],
    scoring: str,
    top_k: int,
) -> pd.DataFrame:
    if not frames:
        raise ExplainabilityConfigurationError(
            f"{method.upper()} produced no outer-test fold results."
        )
    n_folds_total = len(frames)
    rows: list[dict[str, Any]] = []
    for class_index in class_indices:
        class_label = str(class_labels[int(class_index)])
        fold_tables: list[pd.DataFrame] = []
        for fold_no, frame in enumerate(frames, start=1):
            sub = frame[
                pd.to_numeric(frame.get("class_index", -1), errors="coerce")
                .fillna(-1)
                .astype(int)
                .eq(int(class_index))
            ].copy()
            sub = sub.set_index("feature").reindex(list(feature_names))
            vals = pd.to_numeric(sub.get("importance_mean", np.nan), errors="coerce")
            signed = (
                pd.to_numeric(sub["signed_importance_mean"], errors="coerce")
                if "signed_importance_mean" in sub.columns
                else pd.Series(np.nan, index=sub.index, dtype=float)
            )
            tab = pd.DataFrame(
                {
                    "feature": list(feature_names),
                    "importance": vals.to_numpy(dtype=float),
                    "signed": signed.to_numpy(dtype=float),
                    "fold_no": int(fold_no),
                }
            )
            valid = np.isfinite(tab["importance"].to_numpy(dtype=float))
            tab["rank"] = np.nan
            if np.any(valid):
                tab.loc[valid, "rank"] = tab.loc[valid, "importance"].rank(
                    ascending=False, method="average"
                )
                k = min(max(1, int(top_k)), int(valid.sum()))
                tab["top_k"] = False
                tab.loc[valid, "top_k"] = tab.loc[valid, "rank"] <= k
            else:
                tab["top_k"] = False
            fold_tables.append(tab)
        long = pd.concat(fold_tables, ignore_index=True)
        for feature in feature_names:
            sub = long[long["feature"].astype(str).eq(str(feature))].copy()
            imp = pd.to_numeric(sub["importance"], errors="coerce")
            valid = imp.notna() & np.isfinite(imp)
            impv = imp[valid].to_numpy(dtype=float)
            rankv = (
                pd.to_numeric(sub.loc[valid, "rank"], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            signed = pd.to_numeric(sub.loc[valid, "signed"], errors="coerce")
            signed = signed[np.isfinite(signed)].to_numpy(dtype=float)
            n = len(impv)
            if n == 0:
                rows.append(
                    {
                        "method": method,
                        "class_index": int(class_index),
                        "class_label": class_label,
                        "feature": str(feature),
                        "importance_mean": np.nan,
                        "importance_sd": np.nan,
                        "importance_median": np.nan,
                        "importance_q25": np.nan,
                        "importance_q75": np.nan,
                        "mean_rank": np.nan,
                        "median_rank": np.nan,
                        "rank_iqr": np.nan,
                        "top_k_frequency": np.nan,
                        "n_estimable_folds": 0,
                        "n_outer_folds_total": int(n_folds_total),
                        "fold_coverage": 0.0,
                        "signed_importance_mean": np.nan,
                        "sign_positive_fraction": np.nan,
                        "sign_negative_fraction": np.nan,
                        "sign_consistency": np.nan,
                        "scoring": scoring,
                    }
                )
                continue
            q25, med, q75 = np.quantile(impv, [0.25, 0.5, 0.75])
            if rankv.size:
                rq25, rmed, rq75 = np.quantile(rankv, [0.25, 0.5, 0.75])
            else:
                rq25 = rmed = rq75 = np.nan
            top_freq = float(sub.loc[valid, "top_k"].astype(float).mean())
            if signed.size:
                pos = float(np.mean(signed > 0.0))
                neg = float(np.mean(signed < 0.0))
                consistency = float(max(pos, neg))
                signed_mean = float(np.mean(signed))
            else:
                pos = neg = consistency = signed_mean = np.nan
            rows.append(
                {
                    "method": method,
                    "class_index": int(class_index),
                    "class_label": class_label,
                    "feature": str(feature),
                    "importance_mean": float(np.mean(impv)),
                    "importance_sd": float(np.std(impv, ddof=1)) if n > 1 else 0.0,
                    "importance_median": float(med),
                    "importance_q25": float(q25),
                    "importance_q75": float(q75),
                    "mean_rank": float(np.mean(rankv)) if rankv.size else np.nan,
                    "median_rank": float(rmed),
                    "rank_iqr": float(rq75 - rq25) if rankv.size else np.nan,
                    "top_k_frequency": top_freq,
                    "n_estimable_folds": int(n),
                    "n_outer_folds_total": int(n_folds_total),
                    "fold_coverage": float(n / n_folds_total),
                    "signed_importance_mean": signed_mean,
                    "sign_positive_fraction": pos,
                    "sign_negative_fraction": neg,
                    "sign_consistency": consistency,
                    "scoring": scoring,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["class_index", "importance_mean", "feature"],
        ascending=[True, False, True],
        na_position="last",
    )
