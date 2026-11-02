from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .console import info, path_table, progress, stage, success, summary_table
from .explainability import _projection_required
from .explainability_methods import method_has_global, method_has_local, method_name
from .joint_sweep import _configurations, _load_joint_dataset, _predict, _targets, _transform
from .explainability_visuals import plot_local_attributions
from .regression_explainability import (
    _aggregate_frames,
    _ale_importance,
    _combined_feature_table,
    _interaction_importance,
    _lime_importance,
    _permutation_importance,
    _regression_local_sample_pairs,
    _regression_local_records,
    _sample_indices,
    _shap_importance,
    _write_regression_explainability_figures,
)
from .resolutions import _parse_resolution, materialize_mpdr_with_blocks
from .storage import write_table
from .utils import dump_json_standard


class _JointOutputView:
    def __init__(self, model, target_index, classification, target_count):
        self.model = model
        self.target_index = int(target_index)
        self.classification = bool(classification)
        self.target_count = int(target_count)

    def predict(self, X):
        return _predict(self.model, X, self.classification, self.target_count)[:, self.target_index]



def _joint_tree_shap(fold, spec, seed, sample_ids=None, selected_roles=None, local_top_k=None):
    import shap

    if fold.get("input_projector") is not None:
        raise ValueError("Tree SHAP cannot preserve a constrained model-input geometry; use SHAP(algorithm='permutation').")
    model = fold["joint_estimator"]
    train = np.asarray(fold["X_train"], dtype=float)
    test = np.asarray(fold["X_test"], dtype=float)
    indexes = np.arange(len(test), dtype=int) if local_top_k is not None else _sample_indices(len(test), int(spec.max_explain), seed + 17)
    background = train[_sample_indices(len(train), int(spec.background_size), seed)]
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional", model_output="raw")
    raw = np.asarray(explainer(test[indexes]).values, dtype=float)
    target_count = int(fold["target_count"])
    target_index = int(fold["target_index"])
    if raw.ndim == 2 and target_count == 1:
        values = raw
    elif raw.ndim == 3 and not fold["classification"] and raw.shape[-1] == target_count:
        values = raw[:, :, target_index]
    elif raw.ndim == 3 and fold["classification"]:
        classes = getattr(model, "classes_", None)
        if not isinstance(classes, (list, tuple)) or len(classes) != target_count:
            raise ValueError("Joint Tree SHAP requires per-target class labels from the native multilabel model.")
        offsets = [len(np.asarray(item)) for item in classes]
        if raw.shape[-1] != sum(offsets):
            raise ValueError("The Tree SHAP output columns do not match the native multilabel class distribution.")
        matches = np.flatnonzero(np.asarray(classes[target_index]) == 1)
        values = raw[:, :, sum(offsets[:target_index]) + int(matches[0])] if len(matches) else np.zeros(raw.shape[:2])
    else:
        raise ValueError(f"Unexpected joint Tree SHAP output shape {raw.shape}.")
    names = list(fold["feature_names"])
    if values.shape != (len(indexes), len(names)):
        raise ValueError("Joint Tree SHAP did not return one attribution per model coordinate.")
    rows = []
    for j, name in enumerate(names):
        score = values[:, j]
        rows.append({
            "method": "shap",
            "feature": str(name),
            "importance_mean": float(np.nanmean(np.abs(score))),
            "signed_importance_mean": float(np.nanmean(score)),
            "within_fold_importance_sd": float(np.nanstd(np.abs(score), ddof=1)) if len(score) > 1 else 0.0,
            "scoring": "mean_abs_positive_probability_shap" if fold["classification"] else "mean_abs_regression_shap",
            "backend": "native_joint_tree",
        })
    output = pd.DataFrame(rows)
    if local_top_k is not None and sample_ids is not None:
        output.attrs["local_explanations"] = _regression_local_records(
            fold, "shap", values, indexes, sample_ids, selected_roles or {}, int(local_top_k)
        )
    return output


def _classification_permutation(fold, spec, seed):
    X = np.asarray(fold["X_test"], dtype=float)
    y = np.asarray(fold["y_test"], dtype=int)
    names = list(fold["feature_names"])
    if isinstance(spec.max_samples, float):
        n = min(len(X), max(1, int(np.ceil(len(X) * float(spec.max_samples)))))
    else:
        n = min(len(X), int(spec.max_samples))
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(X), size=n, replace=False))
    X, y = X[index], y[index]
    projector = fold.get("input_projector")
    metric = str(spec.scoring).casefold()
    if metric not in {"auto", "log_loss", "brier"}:
        raise ValueError("Multilabel permutation scoring supports 'auto', 'log_loss', and 'brier'.")
    metric = "log_loss" if metric == "auto" else metric

    def loss(input_matrix):
        matrix = input_matrix if projector is None else projector(input_matrix)
        probability = np.clip(fold["estimator"].predict(matrix), 1e-15, 1 - 1e-15)
        if metric == "brier":
            return float(np.mean((y - probability) ** 2))
        return float(-np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability)))

    baseline = loss(X)
    rows = []
    for j, name in enumerate(names):
        differences = []
        for _ in range(spec.n_repeats):
            altered = X.copy()
            altered[:, j] = altered[rng.permutation(len(X)), j]
            differences.append(loss(altered) - baseline)
        rows.append({
            "method": "permutation",
            "feature": str(name),
            "importance_mean": float(np.mean(differences)),
            "signed_importance_mean": float(np.mean(differences)),
            "within_fold_importance_sd": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0,
            "scoring": f"increase_in_{metric}",
            "perturbation_projection": "fitted_model_input_geometry" if projector is not None else "none",
        })
    return pd.DataFrame(rows)


def _selected_splits(root):
    path = root / "tables" / "joint_cv_splits.json"
    if not path.exists():
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    return json.loads(path.read_text(encoding="utf-8"))["splits"]


def _selected_configurations(root):
    path = root / "tables" / "joint_outer_metrics.tsv"
    if not path.exists():
        raise FileNotFoundError("Joint outer-fold model selections are missing; run evaluate(sweep).")
    records = pd.read_csv(path, sep="\t")
    if records.duplicated("split_key").any():
        raise ValueError("Exactly one selected joint configuration is required per outer split.")
    return records.set_index("split_key")["config_id"].astype(str).to_dict()


def _target_local_pairs(dataset, folds, sweep, classification):
    from .sweep_types import _effective_local_explanations_mode

    mode = _effective_local_explanations_mode(sweep.explainability)
    if mode == "none":
        return mode, {}
    if classification:
        indices = []
        if mode in {"representative", "representative_and_requested"}:
            labels = np.concatenate([fold["y_test"] for fold in folds])
            positions = np.concatenate([fold["test_idx"] for fold in folds])
            for label in (0, 1):
                candidates = positions[labels == label]
                if len(candidates):
                    indices.append((int(candidates[0]), f"label_{label}"))
        if mode in {"requested", "representative_and_requested"}:
            lookup = {str(sample_id): i for i, sample_id in enumerate(dataset.sample_ids)}
            for value in sweep.explainability.local.sample_ids:
                if str(value) not in lookup:
                    raise ValueError(f"Requested explanation sample {value!r} is not in the dataset.")
                indices.append((lookup[str(value)], "requested"))
        return mode, dict(indices)
    indices = _regression_local_sample_pairs(
        dataset,
        folds,
        mode,
        sweep.explainability.local.sample_ids,
        sweep.explainability.local.regression_quantiles,
    )
    return mode, dict(indices)


def explain_joint(sweep):
    root = Path(sweep.root())
    dataset, y, targets, classification = _load_joint_dataset(sweep)
    splits = _selected_splits(root)
    selected = _selected_configurations(root)
    if set(selected) != {str(row["key"]) for row in splits}:
        raise ValueError("Joint explanation splits do not match the saved evaluation selections.")
    candidates = {row[0]["config_id"]: row for row in _configurations(sweep, dataset)}
    methods = tuple(sweep.explainability.methods)
    names = tuple(method_name(method) for method in methods)
    if sweep.explainability.anatomical_traceability is not None:
        raise ValueError("Joint anatomical traceability is not supported for this data modality.")
    requested = sweep.explainability.targets
    requested = (requested,) if isinstance(requested, str) else tuple(requested)
    allowed = {"auto", "all", "mpma_b", "MPMA-B", "best", "best_individual", "best_mpma", "comparison", "standard", "standard_suite"}
    if any(str(value) not in allowed and str(value) != "mpma_e" and not str(value).startswith("ensemble:") for value in requested):
        raise ValueError("Joint explanations support the selected joint model and named ensemble strategies.")
    target_dirs = {target: root / "explainability" / "mpma_b" / target for target in targets}
    global_records = {target: [] for target in targets}
    local_records = {target: [] for target in targets}
    curve_records = {target: [] for target in targets}
    interaction_records = {target: [] for target in targets}
    folds_by_target = {target: [] for target in targets}
    saved_predictions = pd.read_csv(root / "tables" / "joint_outer_predictions.tsv", sep="\t").set_index(["split_key", "sample_id"])
    stage("Joint explainability", str(root))
    summary_table("Joint explainability suite", {
        "joint targets": len(targets),
        "outer splits": len(splits),
        "methods": names,
        "top features": sweep.explainability.top_k,
    })
    with progress() as prog:
        task = prog.add_task("Joint explainability · refit selected models", total=len(splits))
        for fold_no, split in enumerate(splits, start=1):
            key = str(split["key"])
            config, X, transform_factory, model_factory = candidates[selected[key]]
            train_idx = np.asarray(split["train"], dtype=int)
            test_idx = np.asarray(split["test"], dtype=int)
            transformer = transform_factory()
            X_train = np.asarray(transformer.fit_apply(X[train_idx]), dtype=float)
            X_test = np.asarray(transformer.apply(X[test_idx]), dtype=float)
            model = model_factory()
            model.fit(X_train, y[train_idx])
            predictions = _predict(model, X_test, classification, len(targets))
            if not np.allclose(
                predictions,
                saved_predictions.loc[[(key, str(dataset.sample_ids[i])) for i in test_idx], [f"pred__{target}" for target in targets]]
                .to_numpy(dtype=float),
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError(f"Refitted joint model differs from saved outer-fold predictions for {key}.")
            resolution = next(item for item in sweep.resolutions if _parse_resolution(item)[0] == config["resolution"])
            _, raw_names, _ = materialize_mpdr_with_blocks(dataset, _parse_resolution(resolution)[1])
            feature_names = list(transformer.get_feature_names_out(raw_names))
            if len(feature_names) != X_train.shape[1]:
                raise ValueError("Joint explainability feature names do not match transformed features.")
            geometry = transformer.perturbation_geometry()
            projector = transformer.project_model_input if _projection_required(geometry) else None
            for target_idx, target in enumerate(targets):
                fold = {
                    "split_key": key,
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                    "X_train": X_train,
                    "X_test": X_test,
                    "feature_names": feature_names,
                    "y_test": y[test_idx, target_idx],
                    "y_pred": predictions[:, target_idx],
                    "estimator": _JointOutputView(model, target_idx, classification, len(targets)),
                    "joint_estimator": model,
                    "target_index": target_idx,
                    "target_count": len(targets),
                    "classification": classification,
                    "input_projector": projector,
                    "perturbation_geometry": geometry,
                    "explanation_backend": "joint_selected_outer_model",
                }
                folds_by_target[target].append(fold)
            prog.update(task, completed=fold_no, description=f"Joint explainability · refit {fold_no}/{len(splits)}")
    outputs = _write_joint_explanation_folds(sweep, dataset, targets, classification, folds_by_target)
    ensemble_targets = [str(value) for value in requested if str(value).startswith("ensemble:") or str(value) == "mpma_e"]
    for spec in ensemble_targets:
        outputs.update(_explain_joint_ensemble(sweep, dataset, y, targets, classification, spec))
    manifest = root / "explainability" / "explainability_manifest.json"
    dump_json_standard({"unit": "joint_outer_selected", "targets": list(targets), "methods": list(names), "outputs": {key: str(path.relative_to(root)) for key, path in outputs.items()}}, manifest)
    outputs["manifest"] = manifest
    success("Joint explainability completed")
    path_table("Joint explainability outputs", outputs)
    return outputs


def _write_joint_explanation_folds(sweep, dataset, targets, classification, folds_by_target, unit="mpma_b", strategy=None):
    root = Path(sweep.root())
    splits = _selected_splits(root)
    methods = tuple(sweep.explainability.methods)
    names = tuple(method_name(method) for method in methods)
    target_dirs = {target: root / "explainability" / unit / (str(strategy) if strategy else "") / target for target in targets}
    global_records = {target: [] for target in targets}
    local_records = {target: [] for target in targets}
    curve_records = {target: [] for target in targets}
    interaction_records = {target: [] for target in targets}
    output = {}
    for target_idx, target in enumerate(targets):
        target_dir = target_dirs[target]
        target_dir.mkdir(parents=True, exist_ok=True)
        folds = folds_by_target[target]
        local_mode, selected_roles = _target_local_pairs(dataset, folds, sweep, classification)
        for fold_no, fold in enumerate(folds, start=1):
            seed = int(sweep.explainability.random_state) + fold_no * 1009 + target_idx * 100000
            seed_scores = pd.DataFrame()
            for method in methods:
                name = method_name(method)
                local = local_mode != "none" and method_has_local(method)
                if name == "permutation":
                    frame = _classification_permutation(fold, method, seed) if classification else _permutation_importance(fold, method, str(sweep.evaluation.optimize_metric), seed)
                elif name == "shap":
                    kwargs = {
                        "local_sample_ids": dataset.sample_ids if local else None,
                        "selected_roles": selected_roles if local else None,
                        "local_top_k": int(sweep.explainability.local.stored_features) if local else None,
                    }
                    if method.algorithm in {"tree", "auto"} and hasattr(fold["joint_estimator"], "estimators_") and fold.get("input_projector") is None:
                        try:
                            frame = _joint_tree_shap(fold, method, seed, **{
                                "sample_ids": kwargs["local_sample_ids"],
                                "selected_roles": kwargs["selected_roles"],
                                "local_top_k": kwargs["local_top_k"],
                            })
                        except Exception:
                            if method.algorithm == "tree":
                                raise
                            frame = _shap_importance(fold, method, seed, **kwargs)
                    elif method.algorithm == "tree":
                        raise ValueError("Native joint Tree SHAP requires a compatible tree model and unconstrained model-input geometry.")
                    else:
                        frame = _shap_importance(fold, method, seed, **kwargs)
                elif name == "lime":
                    frame = _lime_importance(
                        fold, method, seed,
                        local_sample_ids=dataset.sample_ids if local else None,
                        selected_roles=selected_roles if local else None,
                        local_top_k=int(sweep.explainability.local.stored_features) if local else None,
                    )
                elif name == "ale":
                    frame, curve = _ale_importance(fold, method)
                    if not curve.empty:
                        curve_records[target].append(curve.assign(fold_no=fold_no, fold_key=fold["split_key"]))
                elif name == "interactions":
                    frame = pd.DataFrame()
                    interactions = _interaction_importance(fold, method, seed_scores)
                    if not interactions.empty:
                        interaction_records[target].append(interactions.assign(fold_no=fold_no, fold_key=fold["split_key"]))
                else:
                    raise ValueError(f"Unsupported joint explainability method {name!r}.")
                if not frame.empty:
                    local_frame = frame.attrs.get("local_explanations")
                    if local and isinstance(local_frame, pd.DataFrame) and not local_frame.empty:
                        local_records[target].append(local_frame)
                    if method_has_global(method):
                        frame = frame.copy()
                        frame.attrs.clear()
                        frame = frame.assign(fold_no=fold_no, fold_key=fold["split_key"])
                        global_records[target].append(frame)
                        if seed_scores.empty or name in {"permutation", "shap"}:
                            seed_scores = frame
        importance = _aggregate_frames(global_records[target], len(splits), int(sweep.explainability.top_k))
        combined = _combined_feature_table(importance)
        fold_table = pd.concat(global_records[target], ignore_index=True, sort=False) if global_records[target] else pd.DataFrame()
        tables = {
            "feature_stability": importance,
            "feature_importance": combined,
            "fold_feature_importance": fold_table,
        }
        if local_records[target]:
            local_table = pd.concat(local_records[target], ignore_index=True, sort=False)
            if classification:
                local_table["class_label"] = "1"
                local_table["true_class_label"] = local_table["observed_response"].astype(int).astype(str)
                local_table["predicted_class_label"] = (local_table["prediction"] >= 0.5).astype(int).astype(str)
                local_table["class_index"] = 1
            tables["local_explanations"] = local_table
        if curve_records[target]:
            tables["ale_curves"] = pd.concat(curve_records[target], ignore_index=True, sort=False)
        if interaction_records[target]:
            tables["ale_interactions"] = pd.concat(interaction_records[target], ignore_index=True, sort=False)
        for stem, frame in tables.items():
            location = target_dir / f"{stem}.parquet"
            write_table(location, frame)
            output[f"{target}_{stem}" if unit == "mpma_b" else f"{unit}_{strategy}_{target}_{stem}"] = location
        figures = _write_regression_explainability_figures(target_dir, importance, int(sweep.explainability.top_k), probability_output=classification)
        if "local_explanations" in tables:
            figure = plot_local_attributions(
                tables["local_explanations"],
                target_dir / "figures" / "local_explanations",
                task="classification" if classification else "regression",
                top_n=int(sweep.explainability.local.displayed_features),
            )
            if figure is not None:
                figures["local_explanations_figure"] = figure
        output.update({f"{target}_{name}" if unit == "mpma_b" else f"{unit}_{strategy}_{target}_{name}": path for name, path in figures.items()})
        metadata = target_dir / "explained_unit.json"
        dump_json_standard({
            "task": "multilabel" if classification else "multioutput_regression",
            "target": target,
            "joint_targets": list(targets),
            "unit": {"type": "joint_outer_selected" if unit == "mpma_b" else "joint_outer_ensemble", "selection": "nested_inner_cv", "strategy": strategy},
            "methods": list(names),
            "outer_folds": len(splits),
            "top_k": int(sweep.explainability.top_k),
            "local_explanations": local_mode,
            "local_methods": [method_name(m) for m in methods if method_has_local(m)],
            "global_methods": [method_name(m) for m in methods if method_has_global(m)],
            "explanation_space": "transformed_joint_model_coordinates" if unit == "mpma_b" else "raw_shared_feature_coordinates",
            "attribution_output": "positive_label_probability" if classification else "continuous_target_prediction",
            "perturbation_interpretation": "held-out model sensitivity, not causal feature effects",
        }, metadata)
        output[f"{target}_metadata" if unit == "mpma_b" else f"{unit}_{strategy}_{target}_metadata"] = metadata
        info(f"Joint explainability · {target} · {len(importance)} feature-method summaries")
    return output



class _JointEnsemblePredictor:
    def __init__(self, members, weights, aggregation, classification, target_count):
        self.members = members
        self.weights = np.asarray(weights, dtype=float)
        self.aggregation = aggregation
        self.classification = classification
        self.target_count = target_count

    def predict(self, X):
        from .regression_ensemble import aggregate_regression_predictions

        raw = np.asarray(X, dtype=float)
        stack = []
        for item in self.members:
            data = np.asarray(item["transformer"].apply(raw[:, item["columns"]]), dtype=float)
            stack.append(_predict(item["model"], data, self.classification, self.target_count))
        prediction = aggregate_regression_predictions(np.stack(stack), self.aggregation, self.weights)
        return np.clip(prediction, 0.0, 1.0) if self.classification else prediction

    def predict_proba(self, X):
        return self.predict(X)


def _explain_joint_ensemble(sweep, dataset, y, targets, classification, requested):
    root = Path(sweep.root())
    manifest = root / "joint_ensemble.json"
    if not manifest.exists():
        raise FileNotFoundError("Run the ensemble stage before explaining an ensemble strategy.")
    config = json.loads(manifest.read_text(encoding="utf-8"))
    available = tuple(config["strategies"])
    if requested == "mpma_e":
        if len(available) != 1:
            raise ValueError("Specify a named ensemble strategy with Explainability(targets=('ensemble:<strategy>',)); multiple strategies are available.")
        strategy = available[0]
    else:
        strategy = requested.partition(":")[2]
        if strategy not in available:
            raise ValueError(f"Unknown joint ensemble strategy {strategy!r}; available strategies: {available!r}.")
    methods = tuple(sweep.explainability.methods)
    if any(method_name(method) not in {"permutation", "shap"} for method in methods):
        raise ValueError("Ensemble-level joint explanations currently support permutation and model-agnostic SHAP only.")
    if any(method_name(method) == "shap" and str(method.algorithm).casefold() == "tree" for method in methods):
        raise ValueError("Tree SHAP does not directly explain a heterogeneous joint ensemble; use SHAP(algorithm='permutation').")
    raw = np.asarray(dataset.X_by_level["all"], dtype=float)
    raw_names = list(map(str, dataset.feature_names_by_level["all"]))
    if len(raw_names) != len(set(raw_names)):
        raise ValueError("Ensemble explanations require unambiguous feature names in the raw representation.")
    column_lookup = {name: i for i, name in enumerate(raw_names)}
    candidates = {row[0]["config_id"]: row for row in _configurations(sweep, dataset)}
    records = {str(row["split_key"]): row for row in config["records"] if row["strategy"] == strategy}
    splits = _selected_splits(root)
    predictions = pd.read_csv(root / "tables" / "joint_ensemble_outer_predictions.tsv", sep="\t")
    predictions = predictions[predictions["strategy"] == strategy].set_index(["split_key", "sample_id"])
    if set(records) != {str(item["key"]) for item in splits}:
        raise ValueError("Ensemble explanation records do not match the evaluated outer splits.")
    folds_by_target = {target: [] for target in targets}
    for split in splits:
        key = str(split["key"])
        train = np.asarray(split["train"], dtype=int)
        test = np.asarray(split["test"], dtype=int)
        recipe = records[key]
        represented_resolutions = {candidates[identifier][0]["resolution"] for identifier in recipe["members"]}
        if len(represented_resolutions) > 1:
            raise NotImplementedError("Ensemble feature attribution across different taxonomic resolutions requires coherent hierarchical perturbations and is not yet supported.")
        fitted = []
        for identifier in recipe["members"]:
            config_row, X, transform_factory, estimator_factory = candidates[identifier]
            native_names = next(materialize_mpdr_with_blocks(dataset, _parse_resolution(item)[1])[1] for item in sweep.resolutions if _parse_resolution(item)[0] == config_row["resolution"])
            try:
                columns = [column_lookup[name] for name in native_names]
            except KeyError as exc:
                raise ValueError("An ensemble feature is not part of the shared raw coordinate system.") from exc
            transformer = transform_factory()
            x_train = np.asarray(transformer.fit_apply(X[train]), dtype=float)
            model = estimator_factory()
            model.fit(x_train, y[train])
            fitted.append({"model": model, "transformer": transformer, "columns": columns})
        model = _JointEnsemblePredictor(fitted, recipe["weights"], recipe["aggregation"], classification, len(targets))
        prediction = model.predict(raw[test])
        saved = predictions.loc[[(key, str(dataset.sample_ids[i])) for i in test], [f"pred__{target}" for target in targets]].to_numpy(dtype=float)
        if not np.allclose(prediction, saved, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Refitted joint ensemble differs from saved held-out predictions for {key}.")
        for i, target in enumerate(targets):
            folds_by_target[target].append({
                "split_key": key,
                "train_idx": train,
                "test_idx": test,
                "X_train": raw[train],
                "X_test": raw[test],
                "feature_names": raw_names,
                "y_test": y[test, i],
                "y_pred": prediction[:, i],
                "estimator": _JointOutputView(model, i, classification, len(targets)),
                "joint_estimator": model,
                "target_index": i,
                "target_count": len(targets),
                "classification": classification,
                "input_projector": None,
                "explanation_backend": "joint_ensemble_outer_fit",
            })
    return _write_joint_explanation_folds(sweep, dataset, targets, classification, folds_by_target, unit="mpma_e", strategy=strategy)
