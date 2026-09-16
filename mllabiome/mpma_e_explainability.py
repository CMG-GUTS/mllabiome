from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import explainability as _core
from .configs_sweep import _lodo_feature_pair
from .data import load_dataset
from .ensemble_aggregation import (
    LINEAR_PROBABILITY_AGGREGATIONS,
    aggregate_member_predictions,
    effective_aggregation_weights,
)
from .final_models import build_final_models
from .resolutions import materialize_mpdr
from .utils import dump_json_standard


def _member_rows_for_final_model(
    root: Path, rankings: pd.DataFrame, mpma_e: dict[str, Any]
) -> pd.DataFrame:
    member_ids = [str(member["config_id"]) for member in mpma_e["members"]]
    rows = _core._ordered_member_rows(root, rankings, member_ids)
    by_id = {str(row["config_id"]): row for _, row in rows.iterrows()}
    ordered: list[pd.Series] = []
    for config_id in member_ids:
        if config_id not in by_id:
            raise _core.ExplainabilityConfigurationError(
                f"Final MPMA-E member {config_id!r} is not present in configs.tsv."
            )
        ordered.append(by_id[config_id])
    return pd.DataFrame(ordered)


def _linear_weights(mpma_e: dict[str, Any]) -> np.ndarray | None:
    aggregation = str(mpma_e["aggregation_strategy"])
    raw = None
    if aggregation == "weighted_mean_proba":
        raw = [
            float(member.get("aggregation_weight", member.get("weight", np.nan)))
            for member in mpma_e["members"]
        ]
    return effective_aggregation_weights(aggregation, len(mpma_e["members"]), raw)


def _prepare_member_specs(
    sweep: Any,
    dataset: Any,
    member_rows: pd.DataFrame,
    mpma_e: dict[str, Any],
) -> list[dict[str, Any]]:
    final_by_id = {str(x["config_id"]): x for x in mpma_e["members"]}
    specs: list[dict[str, Any]] = []
    for _, row in member_rows.iterrows():
        config_id = str(row["config_id"])
        levels = _core._row_levels(row) or ("all",)
        X_base, feature_names = materialize_mpdr(dataset, levels)
        specs.append(
            {
                "config_id": config_id,
                "row": row,
                "levels": levels,
                "X_base": np.asarray(X_base, dtype=float),
                "feature_names": list(feature_names),
                "transformation_key": str(row["count_transformation"]),
                "learner_key": str(row["learner"]),
                "final_member": final_by_id[config_id],
            }
        )
    return specs


def _stored_member_outer_predictions(root: Path) -> pd.DataFrame | None:
    path = root / "predictions" / "outer_predictions.tsv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, sep="\t")
    if "outer_split_key" not in frame.columns and "split_key" in frame.columns:
        frame["outer_split_key"] = frame["split_key"]
    required = {"outer_split_key", "sample_id", "config_id"}
    if not required.issubset(frame.columns):
        return None
    frame = frame.copy()
    frame["outer_split_key"] = frame["outer_split_key"].astype(str)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["config_id"] = frame["config_id"].astype(str)
    return frame


def _prediction_reproduction_error(
    stored: pd.DataFrame | None,
    *,
    split_key: str,
    config_id: str,
    sample_ids: Sequence[str],
    proba: np.ndarray,
) -> float | None:
    if stored is None:
        return None
    pcols = [column for column in stored.columns if column.startswith("proba_")]
    if not pcols or len(pcols) != proba.shape[1]:
        return None
    sub = stored[
        stored["outer_split_key"].eq(str(split_key))
        & stored["config_id"].eq(str(config_id))
    ].copy()
    if sub.empty or sub["sample_id"].duplicated().any():
        return None
    sub = sub.set_index("sample_id").reindex([str(x) for x in sample_ids])
    if sub[pcols].isna().any().any():
        return None
    expected = sub[pcols].to_numpy(dtype=float)
    return float(np.max(np.abs(expected - np.asarray(proba, dtype=float))))


def _fit_oof_members(
    sweep: Any,
    rankings: pd.DataFrame,
    mpma_e: dict[str, Any],
) -> dict[str, Any]:
    root = Path(sweep.root())
    member_rows = _member_rows_for_final_model(root, rankings, mpma_e)
    all_levels: list[str] = []
    for _, row in member_rows.iterrows():
        for level in _core._row_levels(row):
            if level not in all_levels:
                all_levels.append(level)
    if not all_levels:
        all_levels = ["all"]
    dataset = load_dataset(sweep.data, tuple(all_levels))
    specs = _prepare_member_specs(sweep, dataset, member_rows, mpma_e)
    splits = _core._explainability_outer_splits(sweep, dataset)
    stored = _stored_member_outer_predictions(root)

    aggregation = str(mpma_e["aggregation_strategy"])
    linear_weights = _linear_weights(mpma_e)
    weighted_input = (
        linear_weights.tolist() if aggregation == "weighted_mean_proba" else None
    )

    folds: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []
    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
            continue
        fitted_members: list[dict[str, Any]] = []
        member_proba: list[np.ndarray] = []
        for member_no, spec in enumerate(specs, start=1):
            X_train_raw, X_test_raw, mask = _lodo_feature_pair(
                spec["X_base"],
                train_idx,
                test_idx,
                str(sweep.evaluation.protocol),
            )
            names = [
                name
                for name, keep in zip(
                    spec["feature_names"], np.asarray(mask, dtype=bool)
                )
                if bool(keep)
            ]
            ct = _core._configured_count_transformation_factory(
                sweep, spec["transformation_key"]
            )()
            X_train, X_test = ct.apply_pair(X_train_raw, X_test_raw)
            clf = _core._configured_learner_factory(sweep, spec["learner_key"])()
            clf.fit(X_train, dataset.y[train_idx])
            proba = _core._predict_proba_aligned(
                clf, X_test, np.arange(len(dataset.class_labels), dtype=int)
            )
            member_proba.append(proba)
            sample_ids = [str(dataset.sample_ids[int(i)]) for i in test_idx]
            max_error = _prediction_reproduction_error(
                stored,
                split_key=str(split["split_key"]),
                config_id=spec["config_id"],
                sample_ids=sample_ids,
                proba=proba,
            )
            reproduction_rows.append(
                {
                    "split_key": str(split["split_key"]),
                    "config_id": spec["config_id"],
                    "max_abs_probability_error": max_error,
                    "verified_against_stored_outer_predictions": int(
                        max_error is not None
                    ),
                }
            )
            fitted_members.append(
                {
                    **spec,
                    "member_no": int(member_no),
                    "feature_names_fold": names,
                    "X_train": np.asarray(X_train, dtype=float),
                    "X_test": np.asarray(X_test, dtype=float),
                    "estimator": clf,
                    "proba": np.asarray(proba, dtype=float),
                }
            )
        stack = np.stack(member_proba, axis=0)
        ensemble_proba = aggregate_member_predictions(
            stack, aggregation, weighted_input
        )
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "members": fitted_members,
                "member_stack": stack,
                "proba": ensemble_proba,
            }
        )

    if not folds:
        raise _core.ExplainabilityConfigurationError(
            "No outer fold could be fitted for MPMA-E explainability."
        )

    reproduction = pd.DataFrame(reproduction_rows)
    verified = reproduction[
        reproduction["verified_against_stored_outer_predictions"].eq(1)
    ]
    if not verified.empty:
        worst = pd.to_numeric(
            verified["max_abs_probability_error"], errors="coerce"
        ).max()
        if np.isfinite(worst) and float(worst) > 1e-5:
            raise _core.ExplainabilityConfigurationError(
                "MPMA-E explainability refits do not reproduce the stored evaluated member "
                f"probabilities (worst absolute difference={float(worst):.3g}). "
                "The evaluated pipeline must be reproduced exactly before explanation."
            )
    return {
        "dataset": dataset,
        "member_rows": member_rows,
        "specs": specs,
        "folds": folds,
        "reproduction": reproduction,
        "linear_weights": linear_weights,
    }


def _shap_member_values(
    member: dict[str, Any],
    class_labels: Sequence[str],
    *,
    rows_ex: np.ndarray,
    rows_bg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import shap
    except Exception as exc:
        raise _core.ExplainabilityDependencyError(
            "MPMA-E hierarchical SHAP requires the 'shap' package."
        ) from exc

    X_train = np.asarray(member["X_train"], dtype=float)
    X_test = np.asarray(member["X_test"], dtype=float)
    background = X_train[np.asarray(rows_bg, dtype=int)]
    selected = X_test[np.asarray(rows_ex, dtype=int)]
    classes = np.arange(len(class_labels), dtype=int)
    model_fn = _core._explain_predict_proba(member["estimator"], classes)
    min_max_evals = max(500, 2 * X_train.shape[1] + 1)
    try:
        with _core._quiet_external_progress():
            explainer = shap.Explainer(
                model_fn,
                background,
                feature_names=list(member["feature_names_fold"]),
            )
            try:
                explanation = explainer(selected, silent=True, max_evals=min_max_evals)
            except TypeError:
                try:
                    explanation = explainer(selected, max_evals=min_max_evals)
                except TypeError:
                    explanation = explainer(selected)
    except Exception as exc:
        raise _core.ExplainabilityConfigurationError(
            f"SHAP failed for MPMA-E member {member['config_id']!r}."
        ) from exc

    values = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float)
    n_classes = len(class_labels)
    if values.ndim == 2:
        if n_classes != 2:
            raise _core.ExplainabilityConfigurationError(
                f"Unexpected SHAP shape {values.shape} for {n_classes} classes."
            )
        values = values[:, :, None]
        if base.ndim == 0:
            base = np.full((len(rows_ex), 1), float(base))
        elif base.ndim == 1:
            base = (
                base.reshape(-1, 1)
                if len(base) == len(rows_ex)
                else np.tile(base[None, :], (len(rows_ex), 1))
            )

        return values, base
    if values.ndim != 3 or values.shape[2] != n_classes:
        raise _core.ExplainabilityConfigurationError(
            f"Unexpected SHAP value shape {values.shape}; expected n_samples × n_features × n_classes."
        )
    if base.ndim == 1:
        if len(base) == n_classes:
            base = np.tile(base[None, :], (len(rows_ex), 1))
        elif len(base) == len(rows_ex) and n_classes == 1:
            base = base[:, None]
    if base.ndim != 2 or base.shape[0] != len(rows_ex):
        raise _core.ExplainabilityConfigurationError(
            f"Unexpected SHAP base-value shape {base.shape}."
        )
    return values, base


def _write_prediction_tables(
    bundle: dict[str, Any],
    mpma_e: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    dataset = bundle["dataset"]
    aggregation = str(mpma_e["aggregation_strategy"])
    linear_weights = bundle["linear_weights"]
    ensemble_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []

    for fold in bundle["folds"]:
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        stack = np.asarray(fold["member_stack"], dtype=float)
        ensemble = np.asarray(fold["proba"], dtype=float)
        for local_i, global_i in enumerate(test_idx):
            base = {
                "split_key": str(fold["split_key"]),
                "sample_id": str(dataset.sample_ids[int(global_i)]),
                "sample_index": int(global_i),
                "y_true": int(dataset.y[int(global_i)]),
                "y_pred": int(np.argmax(ensemble[local_i])),
            }
            for c, label in enumerate(dataset.class_labels):
                base[f"proba_{label}"] = float(ensemble[local_i, c])
            ensemble_rows.append(base)

        for member_index, member in enumerate(fold["members"]):
            member_proba = np.asarray(member["proba"], dtype=float)
            for local_i, global_i in enumerate(test_idx):
                for class_index, label in enumerate(dataset.class_labels):
                    rec = {
                        "split_key": str(fold["split_key"]),
                        "sample_id": str(dataset.sample_ids[int(global_i)]),
                        "sample_index": int(global_i),
                        "config_id": str(member["config_id"]),
                        "member_no": int(member_index + 1),
                        "class_index": int(class_index),
                        "class_label": str(label),
                        "member_probability": float(member_proba[local_i, class_index]),
                    }
                    if linear_weights is not None:
                        rec["aggregation_weight"] = float(linear_weights[member_index])
                        rec["weighted_probability_contribution"] = float(
                            linear_weights[member_index]
                            * member_proba[local_i, class_index]
                        )
                    member_rows.append(rec)

        if stack.shape[0] > 1:
            for member_index, member in enumerate(fold["members"]):
                reduced_stack = np.delete(stack, member_index, axis=0)
                reduced_weights = None
                if aggregation == "weighted_mean_proba":
                    assert linear_weights is not None
                    reduced_weights = np.delete(linear_weights, member_index)
                reduced = aggregate_member_predictions(
                    reduced_stack, aggregation, reduced_weights
                )
                delta = ensemble - reduced
                for local_i, global_i in enumerate(test_idx):
                    for class_index, label in enumerate(dataset.class_labels):
                        influence_rows.append(
                            {
                                "split_key": str(fold["split_key"]),
                                "sample_id": str(dataset.sample_ids[int(global_i)]),
                                "sample_index": int(global_i),
                                "config_id": str(member["config_id"]),
                                "member_no": int(member_index + 1),
                                "class_index": int(class_index),
                                "class_label": str(label),
                                "leave_one_member_out_delta": float(
                                    delta[local_i, class_index]
                                ),
                                "abs_leave_one_member_out_delta": float(
                                    abs(delta[local_i, class_index])
                                ),
                            }
                        )

    outputs: dict[str, Path] = {}
    ensemble_df = pd.DataFrame(ensemble_rows)
    p = out_dir / "oof_predictions.tsv"
    ensemble_df.to_csv(p, sep="\t", index=False)
    outputs["oof_predictions"] = p

    member_df = pd.DataFrame(member_rows)
    p = out_dir / "member_probability_decomposition.tsv"
    member_df.to_csv(p, sep="\t", index=False)
    outputs["member_probability_decomposition"] = p

    influence_df = pd.DataFrame(influence_rows)
    p = out_dir / "member_aggregation_influence.tsv"
    influence_df.to_csv(p, sep="\t", index=False)
    outputs["member_aggregation_influence"] = p
    if not influence_df.empty:
        summary = (
            influence_df.groupby(
                ["config_id", "member_no", "class_index", "class_label"], as_index=False
            )
            .agg(
                mean_abs_influence=("abs_leave_one_member_out_delta", "mean"),
                sd_abs_influence=("abs_leave_one_member_out_delta", "std"),
                mean_signed_influence=("leave_one_member_out_delta", "mean"),
                n_oof_rows=("leave_one_member_out_delta", "size"),
            )
            .sort_values("mean_abs_influence", ascending=False)
        )
        p = out_dir / "member_aggregation_influence_summary.tsv"
        summary.to_csv(p, sep="\t", index=False)
        outputs["member_aggregation_influence_summary"] = p
    return outputs


def _run_hierarchical_shap(
    sweep: Any,
    bundle: dict[str, Any],
    mpma_e: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    dataset = bundle["dataset"]
    linear_weights = bundle["linear_weights"]
    linear = linear_weights is not None
    class_labels = list(dataset.class_labels)

    representation_records: list[dict[str, Any]] = []
    taxon_records: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for fold_no, fold in enumerate(bundle["folds"], start=1):
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        dummy_train = np.zeros((len(train_idx), 1), dtype=float)
        dummy_test = np.zeros((len(test_idx), 1), dtype=float)
        seed = int(sweep.explainability.random_state) + fold_no * 997
        rows_bg = _core._sample_rows(
            dummy_train,
            max_rows=int(sweep.explainability.shap_background),
            random_state=seed,
        )
        rows_ex = _core._sample_rows(
            dummy_test,
            max_rows=int(sweep.explainability.shap_max_samples),
            random_state=seed + 13,
        )
        requested = {str(x) for x in sweep.explainability.instance_sample_ids}
        if requested:
            forced = [
                local_i
                for local_i, global_i in enumerate(test_idx)
                if str(dataset.sample_ids[int(global_i)]) in requested
            ]
            rows_ex = np.asarray(
                sorted(set(rows_ex.tolist()) | set(int(x) for x in forced)), dtype=int
            )

        taxon_accumulator: dict[tuple[int, int, str], float] = {}
        taxon_gross: dict[tuple[int, int, str], float] = {}
        ensemble_base = None
        ensemble_sum_phi = None
        covered_ensemble_classes: set[int] = set()

        for member_index, member in enumerate(fold["members"]):
            values, base = _shap_member_values(
                member,
                class_labels,
                rows_ex=rows_ex,
                rows_bg=rows_bg,
            )
            proba = np.asarray(member["proba"], dtype=float)[rows_ex]
            if values.shape[2] == 1 and len(class_labels) == 2:
                class_indices = [1]
            else:
                class_indices = list(range(values.shape[2]))
            weight = float(linear_weights[member_index]) if linear else float("nan")
            prefix = "|".join(
                [
                    str(member["row"].get("resolution", "+".join(member["levels"]))),
                    str(member["transformation_key"]),
                    str(member["learner_key"]),
                    str(member["config_id"]),
                ]
            )

            for out_class_pos, class_index in enumerate(class_indices):
                local_values = values[:, :, out_class_pos]
                local_base = base[:, out_class_pos]
                member_residual = (
                    local_base + local_values.sum(axis=1) - proba[:, class_index]
                )
                for row_pos, local_test_row in enumerate(rows_ex):
                    diagnostic_rows.append(
                        {
                            "split_key": str(fold["split_key"]),
                            "sample_id": str(
                                dataset.sample_ids[int(test_idx[int(local_test_row)])]
                            ),
                            "config_id": str(member["config_id"]),
                            "class_index": int(class_index),
                            "class_label": str(class_labels[class_index]),
                            "member_shap_additivity_residual": float(
                                member_residual[row_pos]
                            ),
                        }
                    )

                propagated = local_values * weight if linear else local_values
                for feature_index, biological_feature in enumerate(
                    member["feature_names_fold"]
                ):
                    representation_feature = f"{prefix}|{biological_feature}"
                    for row_pos, local_test_row in enumerate(rows_ex):
                        global_i = int(test_idx[int(local_test_row)])
                        raw_value = float(local_values[row_pos, feature_index])
                        propagated_value = float(propagated[row_pos, feature_index])
                        representation_records.append(
                            {
                                "split_key": str(fold["split_key"]),
                                "sample_id": str(dataset.sample_ids[global_i]),
                                "sample_index": global_i,
                                "config_id": str(member["config_id"]),
                                "member_no": int(member_index + 1),
                                "class_index": int(class_index),
                                "class_label": str(class_labels[class_index]),
                                "biological_feature": str(biological_feature),
                                "representation_feature": representation_feature,
                                "member_shap": raw_value,
                                "aggregation_weight": weight if linear else np.nan,
                                "propagated_shap": propagated_value
                                if linear
                                else np.nan,
                            }
                        )
                        if linear:
                            key = (global_i, int(class_index), str(biological_feature))
                            taxon_accumulator[key] = (
                                taxon_accumulator.get(key, 0.0) + propagated_value
                            )
                            taxon_gross[key] = taxon_gross.get(key, 0.0) + abs(
                                propagated_value
                            )

                if linear:
                    if ensemble_base is None:
                        ensemble_base = np.zeros(
                            (len(rows_ex), len(class_labels)), dtype=float
                        )
                        ensemble_sum_phi = np.zeros_like(ensemble_base)
                    covered_ensemble_classes.add(int(class_index))
                    ensemble_base[:, class_index] += weight * local_base
                    ensemble_sum_phi[:, class_index] += weight * local_values.sum(
                        axis=1
                    )

        if linear and ensemble_base is not None and ensemble_sum_phi is not None:
            ensemble_proba = np.asarray(fold["proba"], dtype=float)[rows_ex]
            for row_pos, local_test_row in enumerate(rows_ex):
                global_i = int(test_idx[int(local_test_row)])
                for class_index in sorted(covered_ensemble_classes):
                    label = class_labels[class_index]
                    residual = (
                        ensemble_base[row_pos, class_index]
                        + ensemble_sum_phi[row_pos, class_index]
                        - ensemble_proba[row_pos, class_index]
                    )
                    diagnostic_rows.append(
                        {
                            "split_key": str(fold["split_key"]),
                            "sample_id": str(dataset.sample_ids[global_i]),
                            "config_id": "__MPMA_E__",
                            "class_index": int(class_index),
                            "class_label": str(label),
                            "ensemble_shap_additivity_residual": float(residual),
                            "ensemble_base_value": float(
                                ensemble_base[row_pos, class_index]
                            ),
                            "ensemble_probability": float(
                                ensemble_proba[row_pos, class_index]
                            ),
                        }
                    )

        if linear:
            for (
                global_i,
                class_index,
                biological_feature,
            ), net_value in taxon_accumulator.items():
                gross_value = float(
                    taxon_gross[(global_i, class_index, biological_feature)]
                )
                taxon_records.append(
                    {
                        "split_key": str(fold["split_key"]),
                        "sample_id": str(dataset.sample_ids[int(global_i)]),
                        "sample_index": int(global_i),
                        "class_index": int(class_index),
                        "class_label": str(class_labels[class_index]),
                        "feature": str(biological_feature),
                        "net_propagated_shap": float(net_value),
                        "gross_propagated_shap": gross_value,
                        "cancellation_fraction": float(
                            1.0 - abs(net_value) / gross_value
                        )
                        if gross_value > 0.0
                        else 0.0,
                    }
                )

    raw = pd.DataFrame(representation_records)
    outputs: dict[str, Path] = {}
    raw_path = out_dir / "shap_member_attributions.tsv.gz"
    raw.to_csv(raw_path, sep="\t", index=False, compression="gzip")
    outputs["shap_member_attributions"] = raw_path

    if raw.empty:
        raise _core.ExplainabilityConfigurationError(
            "MPMA-E hierarchical SHAP produced no attributions."
        )

    member_summary = (
        raw.assign(abs_member_shap=raw["member_shap"].abs())
        .groupby(
            [
                "config_id",
                "member_no",
                "class_index",
                "class_label",
                "biological_feature",
                "representation_feature",
            ],
            as_index=False,
        )
        .agg(
            importance_mean=("abs_member_shap", "mean"),
            signed_shap_mean=("member_shap", "mean"),
            importance_sd=("abs_member_shap", "std"),
            n_oof_explanations=("member_shap", "size"),
        )
        .sort_values("importance_mean", ascending=False)
    )
    p = out_dir / "feature_importance_member_shap.tsv"
    member_summary.to_csv(p, sep="\t", index=False)
    outputs["feature_importance_member_shap"] = p

    if linear:
        linear_raw = raw[
            np.isfinite(pd.to_numeric(raw["propagated_shap"], errors="coerce"))
        ].copy()
        linear_raw["abs_propagated_shap"] = linear_raw["propagated_shap"].abs()
        representation_summary = (
            linear_raw.groupby(
                [
                    "config_id",
                    "member_no",
                    "class_index",
                    "class_label",
                    "biological_feature",
                    "representation_feature",
                ],
                as_index=False,
            )
            .agg(
                importance_mean=("abs_propagated_shap", "mean"),
                signed_importance_mean=("propagated_shap", "mean"),
                importance_sd=("abs_propagated_shap", "std"),
                aggregation_weight=("aggregation_weight", "first"),
                n_oof_explanations=("propagated_shap", "size"),
            )
            .sort_values("importance_mean", ascending=False)
        )
        p = out_dir / "feature_importance_mpdr_propagated_shap.tsv"
        representation_summary.to_csv(p, sep="\t", index=False)
        outputs["feature_importance_mpdr_propagated_shap"] = p

        taxon_raw = pd.DataFrame(taxon_records)
        p = out_dir / "shap_taxon_net_oof.tsv.gz"
        taxon_raw.to_csv(p, sep="\t", index=False, compression="gzip")
        outputs["shap_taxon_net_oof"] = p
        taxon_summary = (
            taxon_raw.assign(
                abs_net=taxon_raw["net_propagated_shap"].abs(),
                abs_gross=taxon_raw["gross_propagated_shap"].abs(),
            )
            .groupby(["class_index", "class_label", "feature"], as_index=False)
            .agg(
                importance_mean=("abs_net", "mean"),
                gross_member_support=("abs_gross", "mean"),
                signed_importance_mean=("net_propagated_shap", "mean"),
                importance_sd=("abs_net", "std"),
                cancellation_fraction=("cancellation_fraction", "mean"),
                n_oof_explanations=("net_propagated_shap", "size"),
            )
            .sort_values("importance_mean", ascending=False)
        )
        p = out_dir / "feature_importance_taxon_net_shap.tsv"
        taxon_summary.to_csv(p, sep="\t", index=False)
        outputs["feature_importance_taxon_net_shap"] = p

        if len(class_labels) == 2:
            compat = taxon_summary[taxon_summary["class_index"].eq(1)].copy()
        else:
            compat = taxon_summary.copy()
        compat.insert(0, "method", "propagated_member_shap")
        compat["scoring"] = "mean_abs_net_weighted_member_shap"
        p = out_dir / "feature_importance.tsv"
        compat.to_csv(p, sep="\t", index=False)
        outputs["importance"] = p
    else:
        compat = member_summary.copy()
        if len(class_labels) == 2:
            compat = compat[compat["class_index"].eq(1)].copy()
        compat.insert(0, "method", "member_shap_not_exact_ensemble")
        compat["feature"] = compat["representation_feature"]
        compat["scoring"] = "constituent_member_mean_abs_shap"
        p = out_dir / "feature_importance.tsv"
        compat.to_csv(p, sep="\t", index=False)
        outputs["importance"] = p

    diagnostics = pd.DataFrame(diagnostic_rows)
    p = out_dir / "shap_additivity_diagnostics.tsv"
    diagnostics.to_csv(p, sep="\t", index=False)
    outputs["shap_additivity_diagnostics"] = p
    return outputs


def explain_mpma_e(sweep: Any, rankings: pd.DataFrame | None = None) -> dict[str, Path]:
    root = Path(sweep.root())
    models = build_final_models(root)
    mpma_e = models.get("MPMA-E")
    if not isinstance(mpma_e, dict):
        raise _core.ExplainabilityConfigurationError(
            "MPMA-E explainability was requested, but no final MPMA-E specification is available."
        )
    if rankings is None:
        rankings_path = root / "tables" / "mpma_rankings.tsv"
        if not rankings_path.exists():
            raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
        rankings = pd.read_csv(rankings_path, sep="\t")

    out_dir = root / "explainability" / "mpma_e"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = _core._normalise_explainability_methods(sweep.explainability.methods)
    bundle = _fit_oof_members(sweep, rankings, mpma_e)

    outputs = _write_prediction_tables(bundle, mpma_e, out_dir)
    reproduction_path = out_dir / "prediction_reproduction_diagnostics.tsv"
    bundle["reproduction"].to_csv(reproduction_path, sep="\t", index=False)
    outputs["prediction_reproduction_diagnostics"] = reproduction_path

    if "shap" in methods:
        outputs.update(_run_hierarchical_shap(sweep, bundle, mpma_e, out_dir))

    linear_exact = (
        str(mpma_e["aggregation_strategy"]) in LINEAR_PROBABILITY_AGGREGATIONS
    )
    metadata = {
        "schema_version": 2,
        "unit": "MPMA-E",
        "ensemble_config_id": mpma_e.get("ensemble_config_id", ""),
        "selection_strategy": mpma_e.get("selection_strategy", ""),
        "aggregation_strategy": mpma_e.get("aggregation_strategy", ""),
        "max_size": mpma_e.get("max_size", mpma_e.get("member_count")),
        "member_count": mpma_e.get("member_count", len(mpma_e.get("members", []))),
        "members": mpma_e.get("members", []),
        "explanation_architecture": "member_native_then_aggregation",
        "exact_feature_attribution_propagation": bool(
            linear_exact and "shap" in methods
        ),
        "exact_feature_attribution_condition": (
            "Available because MPMA-E is a fixed linear mean/weighted mean of member probabilities. "
            "Member SHAP decompositions are propagated using the exact aggregation weights."
            if linear_exact
            else "Unavailable: the selected aggregation is non-linear. Member-native SHAP and leave-one-member-out aggregation influence are reported instead."
        ),
        "pseudo_concatenated_mpdr_feature_space_used": False,
        "cross_validation_explanations": "outer_test_folds_of_final_selected_specification",
        "selection_independence_note": (
            "The final MPMA-E specification is selected from all inner-OOF evidence and then cross-fitted over outer folds for descriptive final-model explanation; this is not strict fold-local nested strategy attribution."
        ),
    }
    meta_path = out_dir / "explained_unit.json"
    dump_json_standard(metadata, meta_path)
    outputs["explained_unit"] = meta_path
    return outputs
