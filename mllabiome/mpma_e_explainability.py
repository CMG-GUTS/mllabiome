from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import explainability as _core
from .configs_sweep import _lodo_feature_pair
from .console import info, progress, success, summary_table
from .data import load_dataset
from .ensemble_aggregation import (
    LINEAR_PROBABILITY_AGGREGATIONS,
    aggregate_member_predictions,
    effective_aggregation_weights,
)
from .final_models import build_final_models
from .resolutions import mask_feature_blocks, materialize_mpdr_with_blocks
from .utils import dump_json_standard
from .storage import read_table, write_table, table_exists


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
                f"Final MPMA-E member {config_id!r} is not present in configs.parquet."
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
        X_base, feature_names, feature_blocks = materialize_mpdr_with_blocks(
            dataset, levels
        )
        specs.append(
            {
                "config_id": config_id,
                "row": row,
                "levels": levels,
                "X_base": np.asarray(X_base, dtype=float),
                "feature_names": list(feature_names),
                "feature_blocks": feature_blocks,
                "transformation_key": str(row["count_transformation"]),
                "learner_key": str(row["learner"]),
                "final_member": final_by_id[config_id],
            }
        )
    return specs


def _stored_member_outer_predictions(root: Path) -> pd.DataFrame | None:
    path = root / "predictions" / "outer_predictions.parquet"
    if not table_exists(path):
        return None
    frame = read_table(path)
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


def _fit_oof_members_fold_task(
    split_no: int,
    split: dict[str, Any],
    specs: list[dict[str, Any]],
    sweep: Any,
    dataset: Any,
    stored: pd.DataFrame,
    aggregation: str,
    weighted_input: list[float] | None,
    threads_per_worker: int,
) -> tuple[int, dict[str, Any] | None, list[dict[str, Any]]]:
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
        return int(split_no), None, []
    groups = _core._groups_from_metadata(dataset.metadata, sweep.data.group_col)
    fitted_members: list[dict[str, Any]] = []
    member_proba: list[np.ndarray] = []
    reproduction_rows: list[dict[str, Any]] = []
    for member_no, spec in enumerate(specs, start=1):
        X_train_raw, X_test_raw, mask = _lodo_feature_pair(
            spec["X_base"], train_idx, test_idx, str(sweep.evaluation.protocol)
        )
        feature_blocks = mask_feature_blocks(spec["feature_blocks"], mask)
        names = [
            name
            for name, keep in zip(spec["feature_names"], np.asarray(mask, dtype=bool))
            if bool(keep)
        ]
        ct = _core._configured_count_transformation_factory(
            sweep, spec["transformation_key"], feature_blocks
        )()
        X_train, X_test = ct.apply_pair(X_train_raw, X_test_raw)
        coordinate_metadata = ct.coordinate_metadata(names)
        transformed_names = [str(item.name) for item in coordinate_metadata]
        if (
            len(transformed_names) != X_train.shape[1]
            or X_train.shape[1] != X_test.shape[1]
        ):
            raise _core.ExplainabilityConfigurationError(
                f"Transformation {spec['transformation_key']!r} produced feature metadata inconsistent with its transformed matrix."
            )
        clf = _core.configure_estimator_threads(
            _core._configured_learner_factory(sweep, spec["learner_key"])(),
            threads_per_worker,
        )
        _core.fit_classifier(
            clf,
            X_train,
            dataset.y[train_idx],
            None if groups is None else groups[train_idx],
        )
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
                "verified_against_stored_outer_predictions": int(max_error is not None),
            }
        )
        fitted_members.append(
            {
                **spec,
                "member_no": int(member_no),
                "feature_names_fold": transformed_names,
                "coordinate_metadata_fold": coordinate_metadata,
                "X_train": np.asarray(X_train, dtype=float),
                "X_test": np.asarray(X_test, dtype=float),
                "estimator": clf,
                "proba": np.asarray(proba, dtype=float),
            }
        )
    stack = np.stack(member_proba, axis=0)
    ensemble_proba = aggregate_member_predictions(stack, aggregation, weighted_input)
    fold = {
        "split_key": str(split["split_key"]),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "members": fitted_members,
        "member_stack": stack,
        "proba": ensemble_proba,
    }
    return int(split_no), fold, reproduction_rows


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
    execution = _core._xai_execution_plan(sweep, len(splits))
    tasks = [
        (
            _fit_oof_members_fold_task,
            (
                split_no,
                split,
                specs,
                sweep,
                dataset,
                stored,
                aggregation,
                weighted_input,
                int(execution.threads_per_worker),
            ),
            {},
        )
        for split_no, split in enumerate(splits, start=1)
    ]
    folds_by_no: dict[int, dict[str, Any]] = {}
    reproduction_rows: list[dict[str, Any]] = []
    with progress() as prog:
        task = prog.add_task(
            f"Fitting MPMA-E OOF folds · {execution.workers} workers · {execution.threads_per_worker} threads/worker",
            total=len(tasks),
        )
        for split_no, fold, reproduction in _core._xai_task_iterator(tasks, execution):
            if fold is not None:
                folds_by_no[int(split_no)] = fold
            reproduction_rows.extend(reproduction)
            prog.update(
                task,
                advance=1,
                description=f"Fitting MPMA-E OOF folds · completed {len(folds_by_no)}/{len(tasks)} · last fold {split_no}",
            )
    folds = [folds_by_no[i] for i in sorted(folds_by_no)]
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
    success("MPMA-E outer-fold member refits completed")
    return {
        "dataset": dataset,
        "member_rows": member_rows,
        "specs": specs,
        "folds": folds,
        "reproduction": reproduction,
        "linear_weights": linear_weights,
        "execution": execution,
    }


def _shap_member_values(
    member: dict[str, Any],
    class_labels: Sequence[str],
    *,
    rows_ex: np.ndarray,
    rows_bg: np.ndarray,
    spec: Any,
    random_state: int,
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
    masker_name = str(spec.masker).strip().lower()
    if masker_name == "independent":
        masker = shap.maskers.Independent(background, max_samples=len(background))
    elif masker_name == "partition":
        masker = shap.maskers.Partition(
            background, max_samples=len(background), clustering="correlation"
        )
    else:
        raise _core.ExplainabilityConfigurationError(
            f"Unsupported SHAP masker {spec.masker!r}."
        )
    minimum = 2 * X_train.shape[1] + 1
    max_evals = max(minimum, minimum * max(1, int(spec.permutation_rounds)))
    try:
        explainer = shap.Explainer(
            model_fn,
            masker,
            algorithm=str(spec.algorithm),
            feature_names=list(member["feature_names_fold"]),
            output_names=list(class_labels),
            seed=int(random_state),
        )
        try:
            explanation = explainer(selected, silent=False, max_evals=max_evals)
        except TypeError:
            try:
                explanation = explainer(selected, max_evals=max_evals)
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
    p = out_dir / "oof_predictions.parquet"
    write_table(p, ensemble_df)
    outputs["oof_predictions"] = p

    member_df = pd.DataFrame(member_rows)
    p = out_dir / "member_probability_decomposition.parquet"
    write_table(p, member_df)
    outputs["member_probability_decomposition"] = p

    influence_df = pd.DataFrame(influence_rows)
    p = out_dir / "member_aggregation_influence.parquet"
    write_table(p, influence_df)
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
        p = out_dir / "member_aggregation_influence_summary.parquet"
        write_table(p, summary)
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
    class_indices = _core._resolve_explainability_classes(
        dataset, sweep.explainability.classes
    )
    spec = _core._method_spec(sweep.explainability.methods, "shap")

    representation_records: list[dict[str, Any]] = []
    exact_taxon_records: list[dict[str, Any]] = []
    participation_records: list[dict[str, Any]] = []
    coordinate_records: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    has_non_exact_coordinates = False
    for fold_no, fold in enumerate(bundle["folds"], start=1):
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        train_idx = np.asarray(fold["train_idx"], dtype=int)
        dummy_train = np.zeros((len(train_idx), 1), dtype=float)
        dummy_test = np.zeros((len(test_idx), 1), dtype=float)
        seed = int(sweep.explainability.random_state) + fold_no * 997
        rows_bg = _core._sample_rows(
            dummy_train,
            max_rows=int(spec.background_size),
            random_state=seed,
        )
        rows_ex = _core._sample_rows(
            dummy_test,
            max_rows=int(spec.max_explain),
            random_state=seed + 13,
        )
        requested = (
            {str(x) for x in sweep.explainability.local.sample_ids}
            if _core.method_has_local(spec)
            else set()
        )
        if requested:
            forced = [
                local_i
                for local_i, global_i in enumerate(test_idx)
                if str(dataset.sample_ids[int(global_i)]) in requested
            ]
            rows_ex = np.asarray(
                sorted(set(rows_ex.tolist()) | set(int(x) for x in forced)), dtype=int
            )

        exact_taxon_accumulator: dict[tuple[int, int, str], float] = {}
        exact_taxon_gross: dict[tuple[int, int, str], float] = {}
        participation_accumulator: dict[tuple[int, int, str], float] = {}
        ensemble_base = None
        ensemble_sum_phi = None
        covered_ensemble_classes: set[int] = set()

        for member_index, member in enumerate(fold["members"]):
            info(
                f"MPMA-E SHAP · fold {fold_no}/{len(bundle['folds'])} · member {member_index + 1}/{len(fold['members'])} · "
                f"{member['config_id']} · {len(rows_ex)} samples · {len(member['feature_names_fold'])} features"
            )
            values, base = _shap_member_values(
                member,
                class_labels,
                rows_ex=rows_ex,
                rows_bg=rows_bg,
                spec=spec,
                random_state=seed + member_index * 101,
            )
            success(
                f"MPMA-E SHAP fold {fold_no}/{len(bundle['folds'])} · member {member_index + 1}/{len(fold['members'])} completed"
            )
            proba = np.asarray(member["proba"], dtype=float)[rows_ex]
            if values.shape[2] == len(class_labels):
                output_pairs = [(int(c), int(c)) for c in class_indices]
            elif values.shape[2] == 1 and len(class_labels) == 2:
                positive = int(
                    1 if dataset.positive_class is None else dataset.positive_class
                )
                if positive not in class_indices:
                    raise _core.ExplainabilityConfigurationError(
                        "Binary SHAP returned one output that does not match the configured explained class."
                    )
                output_pairs = [(0, positive)]
            else:
                raise _core.ExplainabilityConfigurationError(
                    f"Unexpected SHAP output shape {values.shape}."
                )
            weight = float(linear_weights[member_index]) if linear else float("nan")
            prefix = "|".join(
                [
                    str(member["row"].get("resolution", "+".join(member["levels"]))),
                    str(member["transformation_key"]),
                    str(member["learner_key"]),
                    str(member["config_id"]),
                ]
            )
            coordinates = list(member["coordinate_metadata_fold"])
            if len(coordinates) != values.shape[1]:
                raise _core.ExplainabilityConfigurationError(
                    f"Transformation coordinate metadata for {member['config_id']!r} has {len(coordinates)} entries but SHAP returned {values.shape[1]} features."
                )
            for coordinate in coordinates:
                if not bool(coordinate.exact_feature_identity):
                    has_non_exact_coordinates = True
                coordinate_records.append(
                    {
                        "split_key": str(fold["split_key"]),
                        "fold_no": int(fold_no),
                        "config_id": str(member["config_id"]),
                        "member_no": int(member_index + 1),
                        "transformation": str(member["transformation_key"]),
                        "coordinate": str(coordinate.name),
                        "coordinate_type": str(coordinate.coordinate_type),
                        "anchor_feature": ""
                        if coordinate.anchor_feature is None
                        else str(coordinate.anchor_feature),
                        "exact_feature_identity": bool(
                            coordinate.exact_feature_identity
                        ),
                        "components": json.dumps(
                            list(coordinate.components), separators=(",", ":")
                        ),
                        "coefficients": json.dumps(
                            [float(x) for x in coordinate.coefficients],
                            separators=(",", ":"),
                        ),
                        "representation_feature": f"{prefix}|{coordinate.name}",
                    }
                )

            for out_class_pos, class_index in output_pairs:
                local_values = values[:, :, out_class_pos]
                local_base = base[:, out_class_pos]
                member_residual = (
                    local_base + local_values.sum(axis=1) - proba[:, class_index]
                )
                for row_pos, local_test_row in enumerate(rows_ex):
                    diagnostic_rows.append(
                        {
                            "split_key": str(fold["split_key"]),
                            "fold_no": int(fold_no),
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
                for feature_index, coordinate in enumerate(coordinates):
                    coordinate_name = str(coordinate.name)
                    anchor_feature = (
                        ""
                        if coordinate.anchor_feature is None
                        else str(coordinate.anchor_feature)
                    )
                    representation_feature = f"{prefix}|{coordinate_name}"
                    coefficients = np.asarray(coordinate.coefficients, dtype=float)
                    coefficient_mass = float(np.abs(coefficients).sum())
                    if coefficient_mass <= 0.0 or len(coordinate.components) != len(
                        coefficients
                    ):
                        raise _core.ExplainabilityConfigurationError(
                            f"Invalid coordinate metadata for {representation_feature!r}."
                        )
                    component_weights = np.abs(coefficients) / coefficient_mass
                    for row_pos, local_test_row in enumerate(rows_ex):
                        global_i = int(test_idx[int(local_test_row)])
                        raw_value = float(local_values[row_pos, feature_index])
                        propagated_value = float(propagated[row_pos, feature_index])
                        representation_records.append(
                            {
                                "split_key": str(fold["split_key"]),
                                "fold_no": int(fold_no),
                                "sample_id": str(dataset.sample_ids[global_i]),
                                "sample_index": global_i,
                                "config_id": str(member["config_id"]),
                                "member_no": int(member_index + 1),
                                "class_index": int(class_index),
                                "class_label": str(class_labels[class_index]),
                                "coordinate": coordinate_name,
                                "coordinate_type": str(coordinate.coordinate_type),
                                "anchor_feature": anchor_feature,
                                "exact_feature_identity": bool(
                                    coordinate.exact_feature_identity
                                ),
                                "biological_feature": anchor_feature,
                                "representation_feature": representation_feature,
                                "member_shap": raw_value,
                                "aggregation_weight": weight if linear else np.nan,
                                "propagated_shap": propagated_value
                                if linear
                                else np.nan,
                            }
                        )
                        if (
                            linear
                            and bool(coordinate.exact_feature_identity)
                            and anchor_feature
                        ):
                            key = (global_i, int(class_index), anchor_feature)
                            exact_taxon_accumulator[key] = (
                                exact_taxon_accumulator.get(key, 0.0) + propagated_value
                            )
                            exact_taxon_gross[key] = exact_taxon_gross.get(
                                key, 0.0
                            ) + abs(propagated_value)
                        magnitude = abs(propagated_value) if linear else abs(raw_value)
                        for component, component_weight in zip(
                            coordinate.components, component_weights
                        ):
                            key = (global_i, int(class_index), str(component))
                            participation_accumulator[key] = (
                                participation_accumulator.get(key, 0.0)
                                + magnitude * float(component_weight)
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
                    residual = (
                        ensemble_base[row_pos, class_index]
                        + ensemble_sum_phi[row_pos, class_index]
                        - ensemble_proba[row_pos, class_index]
                    )
                    diagnostic_rows.append(
                        {
                            "split_key": str(fold["split_key"]),
                            "fold_no": int(fold_no),
                            "sample_id": str(dataset.sample_ids[global_i]),
                            "config_id": "__MPMA_E__",
                            "class_index": int(class_index),
                            "class_label": str(class_labels[class_index]),
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
                feature,
            ), net_value in exact_taxon_accumulator.items():
                gross_value = float(exact_taxon_gross[(global_i, class_index, feature)])
                exact_taxon_records.append(
                    {
                        "split_key": str(fold["split_key"]),
                        "fold_no": int(fold_no),
                        "sample_id": str(dataset.sample_ids[int(global_i)]),
                        "sample_index": int(global_i),
                        "class_index": int(class_index),
                        "class_label": str(class_labels[class_index]),
                        "feature": str(feature),
                        "net_propagated_shap": float(net_value),
                        "gross_propagated_shap": gross_value,
                        "cancellation_fraction": float(
                            1.0 - abs(net_value) / gross_value
                        )
                        if gross_value > 0.0
                        else 0.0,
                    }
                )
        for (
            global_i,
            class_index,
            feature,
        ), value in participation_accumulator.items():
            participation_records.append(
                {
                    "split_key": str(fold["split_key"]),
                    "fold_no": int(fold_no),
                    "sample_id": str(dataset.sample_ids[int(global_i)]),
                    "sample_index": int(global_i),
                    "class_index": int(class_index),
                    "class_label": str(class_labels[class_index]),
                    "feature": str(feature),
                    "participation": float(value),
                    "scope": "linear_ensemble_abs_propagated_shap"
                    if linear
                    else "constituent_member_abs_shap",
                }
            )

    info("Aggregating MPMA-E member SHAP attributions and writing outputs")
    raw = pd.DataFrame(representation_records)
    outputs: dict[str, Path] = {}
    raw_path = out_dir / "shap_member_attributions.parquet"
    write_table(raw_path, raw)
    outputs["shap_member_attributions"] = raw_path
    if raw.empty:
        raise _core.ExplainabilityConfigurationError(
            "MPMA-E hierarchical SHAP produced no attributions."
        )

    coordinate_frame = pd.DataFrame(coordinate_records).drop_duplicates()
    p = out_dir / "coordinate_metadata.parquet"
    write_table(p, coordinate_frame)
    outputs["coordinate_metadata"] = p

    member_summary = (
        raw.assign(abs_member_shap=raw["member_shap"].abs())
        .groupby(
            [
                "config_id",
                "member_no",
                "class_index",
                "class_label",
                "coordinate",
                "coordinate_type",
                "anchor_feature",
                "exact_feature_identity",
                "representation_feature",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            importance_mean=("abs_member_shap", "mean"),
            signed_shap_mean=("member_shap", "mean"),
            importance_sd=("abs_member_shap", "std"),
            n_oof_explanations=("member_shap", "size"),
        )
        .sort_values(["class_index", "importance_mean"], ascending=[True, False])
    )
    p = out_dir / "feature_importance_member_shap.parquet"
    write_table(p, member_summary)
    outputs["feature_importance_member_shap"] = p

    participation_raw = pd.DataFrame(participation_records)
    p = out_dir / "shap_taxon_participation_oof.parquet"
    write_table(p, participation_raw)
    outputs["shap_taxon_participation_oof"] = p
    participation_frames: list[pd.DataFrame] = []
    if not participation_raw.empty:
        for split_key, fold_raw in participation_raw.groupby("split_key", sort=False):
            fold_frame = fold_raw.groupby(
                ["class_index", "class_label", "feature"], as_index=False
            ).agg(importance_mean=("participation", "mean"))
            fold_frame["fold_key"] = str(split_key)
            participation_frames.append(fold_frame)
        participation_features = sorted(
            participation_raw["feature"].astype(str).unique().tolist()
        )
        participation_scoring = (
            "outer_fold_mean_component_weighted_abs_propagated_shap_participation"
            if linear
            else "outer_fold_mean_component_weighted_constituent_abs_shap_participation"
        )
        participation_summary = _core._aggregate_fold_feature_importance(
            "taxon_participation",
            participation_frames,
            participation_features,
            class_indices,
            class_labels,
            participation_scoring,
            sweep.explainability.top_k,
        )
        participation_summary["interpretation"] = (
            "unsigned_nonadditive_logcontrast_component_participation"
        )
        p = out_dir / "feature_importance_taxon_participation.parquet"
        write_table(p, participation_summary)
        outputs["feature_importance_taxon_participation"] = p

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
                    "coordinate",
                    "coordinate_type",
                    "anchor_feature",
                    "exact_feature_identity",
                    "representation_feature",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                importance_mean=("abs_propagated_shap", "mean"),
                signed_importance_mean=("propagated_shap", "mean"),
                importance_sd=("abs_propagated_shap", "std"),
                aggregation_weight=("aggregation_weight", "first"),
                n_oof_explanations=("propagated_shap", "size"),
            )
            .sort_values(["class_index", "importance_mean"], ascending=[True, False])
        )
        p = out_dir / "feature_importance_mpdr_propagated_shap.parquet"
        write_table(p, representation_summary)
        outputs["feature_importance_mpdr_propagated_shap"] = p

        coordinate_fold_frames: list[pd.DataFrame] = []
        for split_key, fold_raw in linear_raw.groupby("split_key", sort=False):
            fold_frame = (
                fold_raw.assign(abs_value=fold_raw["propagated_shap"].abs())
                .groupby(
                    ["class_index", "class_label", "representation_feature"],
                    as_index=False,
                )
                .agg(
                    importance_mean=("abs_value", "mean"),
                    signed_importance_mean=("propagated_shap", "mean"),
                )
                .rename(columns={"representation_feature": "feature"})
            )
            fold_frame["fold_key"] = str(split_key)
            coordinate_fold_frames.append(fold_frame)
        representation_features = sorted(
            linear_raw["representation_feature"].astype(str).unique().tolist()
        )
        coordinate_fold_path, _ = _core._write_fold_feature_importance(
            "propagated_member_shap_coordinate", coordinate_fold_frames, out_dir
        )
        outputs["shap_coordinate_by_outer_fold"] = coordinate_fold_path
        coordinate_stability = _core._aggregate_fold_feature_importance(
            "propagated_member_shap_coordinate",
            coordinate_fold_frames,
            representation_features,
            class_indices,
            class_labels,
            "outer_fold_mean_abs_weighted_member_coordinate_shap",
            sweep.explainability.top_k,
        )
        p = out_dir / "feature_stability_coordinate.parquet"
        write_table(p, coordinate_stability)
        outputs["feature_stability_coordinate"] = p

        exact_taxon_raw = pd.DataFrame(exact_taxon_records)
        exact_taxon_summary = pd.DataFrame()
        if not exact_taxon_raw.empty:
            p = out_dir / "shap_taxon_net_oof.parquet"
            write_table(p, exact_taxon_raw)
            outputs["shap_taxon_net_oof"] = p
            fold_frames: list[pd.DataFrame] = []
            for split_key, fold_raw in exact_taxon_raw.groupby("split_key", sort=False):
                fold_frame = (
                    fold_raw.assign(abs_net=fold_raw["net_propagated_shap"].abs())
                    .groupby(["class_index", "class_label", "feature"], as_index=False)
                    .agg(
                        importance_mean=("abs_net", "mean"),
                        signed_importance_mean=("net_propagated_shap", "mean"),
                    )
                )
                fold_frame["fold_key"] = str(split_key)
                fold_frames.append(fold_frame)
            biological_features = sorted(
                exact_taxon_raw["feature"].astype(str).unique().tolist()
            )
            fold_path, _ = _core._write_fold_feature_importance(
                "propagated_member_shap", fold_frames, out_dir
            )
            outputs["shap_taxon_by_outer_fold"] = fold_path
            exact_taxon_summary = _core._aggregate_fold_feature_importance(
                "propagated_member_shap",
                fold_frames,
                biological_features,
                class_indices,
                class_labels,
                "outer_fold_mean_abs_net_weighted_member_shap",
                sweep.explainability.top_k,
            )
            support = (
                exact_taxon_raw.assign(
                    abs_gross=exact_taxon_raw["gross_propagated_shap"].abs()
                )
                .groupby(["class_index", "class_label", "feature"], as_index=False)
                .agg(
                    gross_member_support=("abs_gross", "mean"),
                    cancellation_fraction=("cancellation_fraction", "mean"),
                    n_oof_explanations=("net_propagated_shap", "size"),
                )
            )
            exact_taxon_summary = exact_taxon_summary.merge(
                support,
                on=["class_index", "class_label", "feature"],
                how="left",
            )
            exact_taxon_summary["scope"] = (
                "exact_feature_identity_members_only"
                if has_non_exact_coordinates
                else "all_members"
            )
            p = out_dir / "feature_importance_taxon_net_shap.parquet"
            write_table(p, exact_taxon_summary)
            outputs["feature_importance_taxon_net_shap"] = p

        if has_non_exact_coordinates:
            compat = coordinate_stability.copy()
            compat["scoring"] = "outer_fold_mean_abs_weighted_member_coordinate_shap"
            p = out_dir / "feature_stability.parquet"
            write_table(p, compat)
            outputs["stability"] = p
            p = out_dir / "feature_importance.parquet"
            write_table(p, compat)
            outputs["importance"] = p
        elif not exact_taxon_summary.empty:
            stability_cols = [
                "method",
                "class_index",
                "class_label",
                "feature",
                "importance_mean",
                "importance_sd",
                "importance_median",
                "importance_q25",
                "importance_q75",
                "mean_rank",
                "median_rank",
                "rank_iqr",
                "top_k_frequency",
                "n_estimable_folds",
                "n_outer_folds_total",
                "fold_coverage",
                "signed_importance_mean",
                "sign_positive_fraction",
                "sign_negative_fraction",
                "sign_consistency",
            ]
            p = out_dir / "feature_stability.parquet"
            write_table(
                p,
                exact_taxon_summary[
                    [c for c in stability_cols if c in exact_taxon_summary.columns]
                ],
            )
            outputs["stability"] = p
            compat = exact_taxon_summary.copy()
            compat["scoring"] = "outer_fold_mean_abs_net_weighted_member_shap"
            p = out_dir / "feature_importance.parquet"
            write_table(p, compat)
            outputs["importance"] = p
        else:
            compat = coordinate_stability.copy()
            p = out_dir / "feature_stability.parquet"
            write_table(p, compat)
            outputs["stability"] = p
            p = out_dir / "feature_importance.parquet"
            write_table(p, compat)
            outputs["importance"] = p
    else:
        compat = member_summary[
            member_summary["class_index"].isin([int(x) for x in class_indices])
        ].copy()
        compat.insert(0, "method", "member_shap_not_exact_ensemble")
        compat["feature"] = compat["representation_feature"]
        compat["scoring"] = "constituent_member_mean_abs_shap"
        p = out_dir / "feature_importance.parquet"
        write_table(p, compat)
        outputs["importance"] = p

    diagnostics = pd.DataFrame(diagnostic_rows)
    p = out_dir / "shap_additivity_diagnostics.parquet"
    write_table(p, diagnostics)
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
        rankings_path = root / "tables" / "mpma_rankings.parquet"
        if not rankings_path.exists():
            raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
        rankings = read_table(rankings_path)

    out_dir = root / "explainability" / "mpma_e"
    out_dir.mkdir(parents=True, exist_ok=True)
    method_specs = _core._normalise_explainability_method_specs(
        sweep.explainability.methods
    )
    methods = tuple(_core.method_name(x) for x in method_specs)
    global_methods = tuple(
        _core.method_name(x) for x in method_specs if _core.method_has_global(x)
    )
    info("Preparing final MPMA-E member refits for OOF explanation")
    bundle = _fit_oof_members(sweep, rankings, mpma_e)
    dataset = bundle["dataset"]
    class_indices = _core._resolve_explainability_classes(
        dataset, sweep.explainability.classes
    )
    summary_table(
        "MPMA-E explainability workload",
        {
            "outer folds": len(bundle.get("folds", [])),
            "members": len(bundle.get("specs", mpma_e.get("members", []))),
            "classes explained": len(class_indices),
            "class labels": [str(dataset.class_labels[int(i)]) for i in class_indices],
        },
    )

    outputs = _write_prediction_tables(bundle, mpma_e, out_dir)
    reproduction_path = out_dir / "prediction_reproduction_diagnostics.parquet"
    write_table(reproduction_path, bundle["reproduction"])
    outputs["prediction_reproduction_diagnostics"] = reproduction_path

    if "shap" in global_methods:
        outputs.update(_run_hierarchical_shap(sweep, bundle, mpma_e, out_dir))

    linear_exact = (
        str(mpma_e["aggregation_strategy"]) in LINEAR_PROBABILITY_AGGREGATIONS
    )
    metadata = {
        "schema_version": 3,
        "unit": "MPMA-E",
        "ensemble_config_id": mpma_e.get("ensemble_config_id", ""),
        "selection_strategy": mpma_e.get("selection_strategy", ""),
        "aggregation_strategy": mpma_e.get("aggregation_strategy", ""),
        "max_size": mpma_e.get("max_size", mpma_e.get("member_count")),
        "member_count": mpma_e.get("member_count", len(mpma_e.get("members", []))),
        "members": mpma_e.get("members", []),
        "explanation_architecture": "member_native_then_aggregation",
        "exact_feature_attribution_propagation": bool(
            linear_exact and "shap" in global_methods
        ),
        "exact_feature_attribution_condition": (
            "Exact at model-coordinate level because MPMA-E is a fixed linear mean/weighted mean of member probabilities and member SHAP decompositions are propagated using the exact aggregation weights. Taxon-level signed aggregation is restricted to coordinates with exact one-to-one feature identity."
            if linear_exact
            else "Unavailable at ensemble level because the selected aggregation is non-linear. Member-native coordinate SHAP and leave-one-member-out aggregation influence are reported instead."
        ),
        "taxon_level_attribution_policy": "Signed taxon SHAP is emitted only for one-to-one feature coordinates. ALR and ILR remain exact model-coordinate explanations and are not back-projected as signed taxon SHAP.",
        "taxon_participation_policy": "Unsigned non-additive taxon participation is derived from absolute coordinate SHAP weighted by normalized absolute log-contrast coefficients and is reported separately from SHAP attribution.",
        "pseudo_concatenated_mpdr_feature_space_used": False,
        "methods": list(methods),
        "method_parameters": [_core.method_to_dict(x) for x in method_specs],
        "explainability_config": _core._explainability_config_payload(
            sweep.explainability
        ),
        "explainability_config_signature": _core._explainability_config_signature(
            sweep.explainability
        ),
        "explained_class_indices": [int(x) for x in class_indices],
        "explained_class_labels": [
            str(dataset.class_labels[int(x)]) for x in class_indices
        ],
        "class_target": "class_probability",
        "stability_unit": "outer_fold",
        "stability_interpretation": "descriptive_cross_fit_variability_not_independent_fold_confidence_intervals",
        "stability_statistics": [
            "importance_sd",
            "rank_iqr",
            "top_k_frequency",
            "fold_coverage",
            "sign_consistency",
        ],
        "ensemble_level_method_scope": {
            "shap": "exact_for_linear_probability_aggregation",
            "lime": "member_native_only",
            "ale": "member_native_only",
            "permutation": "member_native_only",
            "interactions": "member_native_only",
        },
        "cross_validation_explanations": "outer_test_folds_of_final_selected_specification",
        "selection_independence_note": (
            "The final MPMA-E specification is selected from all inner-OOF evidence and then cross-fitted over outer folds for descriptive final-model explanation; this is not strict fold-local nested strategy attribution."
        ),
    }
    meta_path = out_dir / "explained_unit.json"
    dump_json_standard(metadata, meta_path)
    outputs["explained_unit"] = meta_path
    return outputs
