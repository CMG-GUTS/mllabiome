from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold

from .data import dataset_fingerprint, load_dataset
from .evaluation_splits import _regression_inner_splits, _regression_outer_splits, _subject_safe_groups
from .learners import _learner_factory, _learner_name
from .joint_progress import JointEvaluationMonitor
from .resolutions import _parse_resolution, materialize_mpdr_with_blocks
from .sweep_types import _normalise_sweep_task
from .transformations import _count_transformation_factory, _count_transformation_specs_for_blocks
from .utils import dump_json_standard


_CLASSIFICATION_METRICS = frozenset({"log_loss", "brier", "f1_micro", "f1_macro", "f1_samples", "subset_accuracy", "hamming_loss", "AUROC_macro", "AP_macro"})
_REGRESSION_METRICS = frozenset({"normalized_rmse", "RMSE", "MAE", "MSE", "R2"})
_MAXIMIZE = frozenset({"f1_micro", "f1_macro", "f1_samples", "subset_accuracy", "AUROC_macro", "AP_macro", "R2"})


def is_joint_sweep(sweep):
    if sweep.uses_modalities or sweep.data is None:
        return False
    task = _normalise_sweep_task(sweep.data.task)
    return task == "multilabel" or (task == "multioutput" and bool(sweep.data.joint))


def _targets(sweep):
    value = sweep.data.target_col
    targets = (value,) if isinstance(value, str) else tuple(str(v) for v in value)
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise ValueError("Joint evaluation requires at least two unique targets.")
    return targets


def _binary_target(series, name):
    values = series.astype("string").str.strip().str.casefold()
    mapping = {"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1, "absent": 0, "present": 1}
    encoded = values.map(mapping)
    if encoded.isna().any():
        raise ValueError(f"Multilabel target {name!r} contains missing or non-binary values.")
    result = encoded.to_numpy(dtype=np.int64)
    if np.unique(result).size != 2:
        raise ValueError(f"Multilabel target {name!r} has only one class in the full dataset.")
    return result


def _load_joint_dataset(sweep):
    targets = _targets(sweep)
    classification = _normalise_sweep_task(sweep.data.task) == "multilabel"
    if sweep.data.feature_cols is not None:
        features = (sweep.data.feature_cols,) if isinstance(sweep.data.feature_cols, str) else sweep.data.feature_cols
        if set(features) & set(targets):
            raise ValueError("Target columns cannot be used as features.")
    spec = replace(
        sweep.data,
        target_col=targets[0],
        task="classification" if classification else "regression",
        metadata_cols=tuple(dict.fromkeys((*sweep.data.metadata_cols, *targets[1:]))),
    )
    levels = tuple(dict.fromkeys(level for resolution in sweep.resolutions for level in _parse_resolution(resolution)[1]))
    dataset = load_dataset(spec, levels_needed=levels)
    missing = [name for name in targets if name not in dataset.metadata.columns]
    if missing:
        raise ValueError(f"Dataset is missing target columns {missing!r}.")
    if classification:
        y = np.column_stack([_binary_target(dataset.metadata[name], name) for name in targets])
    else:
        frame = dataset.metadata.loc[:, targets].apply(pd.to_numeric, errors="coerce")
        y = frame.to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("All joint regression target values must be finite and numeric.")
        if np.any(np.std(y, axis=0) < 1e-12):
            raise ValueError("Joint regression targets must have non-zero variance.")
    return dataset, y, targets, classification


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan")


def _safe_ap(y, p):
    return float(average_precision_score(y, p)) if np.unique(y).size > 1 else float("nan")


def _scores(y, pred, targets, classification, train_y):
    y = np.asarray(y)
    pred = np.asarray(pred, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 2 or not np.isfinite(pred).all():
        raise ValueError("Joint model prediction must be a finite (n_samples, n_targets) matrix.")
    rows = []
    if classification:
        if np.any((pred < 0) | (pred > 1)):
            raise ValueError("Multilabel probabilities must lie in [0, 1].")
        probabilities = np.clip(pred, 1e-15, 1 - 1e-15)
        decisions = (pred >= 0.5).astype(int)
        for index, target in enumerate(targets):
            actual = y[:, index]
            proba = probabilities[:, index]
            labels = decisions[:, index]
            rows.append({
                "target": target,
                "log_loss": float(-np.mean(actual * np.log(proba) + (1 - actual) * np.log1p(-proba))),
                "brier": float(np.mean((actual - proba) ** 2)),
                "f1": float(f1_score(actual, labels, zero_division=0)),
                "precision": float(precision_score(actual, labels, zero_division=0)),
                "recall": float(recall_score(actual, labels, zero_division=0)),
                "support": int(np.sum(actual)),
                "AUROC": _safe_auc(actual, proba),
                "AP": _safe_ap(actual, proba),
                "prevalence": float(np.mean(actual)),
            })
        metrics = {
            "log_loss": float(np.mean([r["log_loss"] for r in rows])),
            "brier": float(np.mean([r["brier"] for r in rows])),
            "f1_micro": float(f1_score(y, decisions, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(y, decisions, average="macro", zero_division=0)),
            "f1_samples": float(f1_score(y, decisions, average="samples", zero_division=0)),
            "subset_accuracy": float(accuracy_score(y, decisions)),
            "hamming_loss": float(hamming_loss(y, decisions)),
            "AUROC_macro": _nanmean([r["AUROC"] for r in rows]),
            "AP_macro": _nanmean([r["AP"] for r in rows]),
            "n_auroc_defined": sum(np.isfinite(r["AUROC"]) for r in rows),
            "n_ap_defined": sum(np.isfinite(r["AP"]) for r in rows),
        }
    else:
        reference_scale = np.std(train_y, axis=0, ddof=0)
        reference_scale = np.where(reference_scale < 1e-12, 1.0, reference_scale)
        for index, target in enumerate(targets):
            actual, predicted = y[:, index], pred[:, index]
            mse = float(mean_squared_error(actual, predicted))
            rows.append({
                "target": target,
                "RMSE": float(np.sqrt(mse)),
                "MAE": float(mean_absolute_error(actual, predicted)),
                "MSE": mse,
                "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
                "normalized_rmse": float(np.sqrt(mse) / reference_scale[index]),
            })
        metrics = {name: _nanmean([r[name] for r in rows]) for name in ("RMSE", "MAE", "MSE", "R2", "normalized_rmse")}
    return metrics, rows


def _nanmean(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _predict(model, X, classification, n_targets):
    if not classification:
        output = np.asarray(model.predict(X), dtype=float)
    else:
        raw = model.predict_proba(X)
        if isinstance(raw, (list, tuple)):
            if len(raw) != n_targets:
                raise ValueError("Multilabel predict_proba has the wrong number of outputs.")
            classes = getattr(model, "classes_", None)
            columns = []
            for i, item in enumerate(raw):
                matrix = np.asarray(item, dtype=float)
                known = np.asarray(classes[i]) if isinstance(classes, (list, tuple)) else np.arange(matrix.shape[1])
                if matrix.ndim != 2 or matrix.shape[1] != len(known):
                    raise ValueError("Invalid multilabel class probabilities.")
                indices = np.flatnonzero(known == 1)
                columns.append(matrix[:, indices[0]] if len(indices) else np.zeros(len(matrix)))
            output = np.column_stack(columns)
        else:
            output = np.asarray(raw, dtype=float)
    if output.shape != (len(X), n_targets):
        raise ValueError(f"Joint model returned predictions of shape {output.shape}; expected {(len(X), n_targets)}.")
    return output


def _multilabel_splits(y, n_splits, random_state):
    labels = np.asarray(y, dtype=np.int64)
    n_samples = len(labels)
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError("Invalid number of multilabel CV folds.")
    rng = np.random.default_rng(random_state)
    capacities = np.full(n_splits, n_samples // n_splits, dtype=int)
    extras = rng.permutation(n_splits)[: n_samples % n_splits]
    capacities[extras] += 1
    expected = capacities[:, None] * (labels.sum(axis=0) / n_samples)
    assignments = np.full(n_samples, -1, dtype=int)
    remaining = labels.sum(axis=0).astype(int)
    while np.any(remaining > 0):
        rarest = np.flatnonzero(remaining == remaining[remaining > 0].min())
        label = int(rng.choice(rarest))
        candidates = np.flatnonzero((assignments < 0) & (labels[:, label] == 1))
        rng.shuffle(candidates)
        candidates = candidates[np.argsort(-labels[candidates].sum(axis=1), kind="stable")]
        for row in candidates:
            possible = np.flatnonzero(capacities > 0)
            if not len(possible):
                raise ValueError("Multilabel stratification exhausted fold capacity.")
            demand = expected[possible, label]
            possible = possible[np.isclose(demand, np.max(demand))]
            if len(possible) > 1:
                available = capacities[possible]
                possible = possible[available == np.max(available)]
            fold = int(rng.choice(possible))
            assignments[row] = fold
            capacities[fold] -= 1
            expected[fold] -= labels[row]
            remaining -= labels[row]
    unassigned = np.flatnonzero(assignments < 0)
    rng.shuffle(unassigned)
    for row in unassigned:
        possible = np.flatnonzero(capacities == capacities.max())
        fold = int(rng.choice(possible))
        assignments[row] = fold
        capacities[fold] -= 1
    return [(np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)) for fold in range(n_splits)]


def _make_splits(sweep, dataset, y, classification):
    plan = sweep.evaluation
    if sweep.data.stratify_col is not None:
        raise NotImplementedError("Joint evaluation does not yet support explicit stratify_col.")
    if plan.split_manifest is not None:
        raise NotImplementedError("External split_manifest imports are not supported for joint sweeps.")
    group_values, group_col = _subject_safe_groups(
        plan,
        dataset,
        None if not sweep.data.group_col else _group_values(dataset, sweep.data.group_col),
        sweep.data.group_col,
    )
    subjects = np.asarray(dataset.subject_ids, dtype=object)
    outer = _regression_outer_splits(plan, len(y), group_values)
    strategy = "group_kfold" if group_values is not None else "kfold"
    if classification and group_values is None and plan.protocol not in {"lodo", "leave_one_dataset_out", "hierarchical_lodo"}:
        strategy = "greedy_multilabel_marginal_stratification"
        outer = []
        for repeat in range(plan.repeats if plan.protocol == "repeated_nested_cv" else 1):
            for fold, (train, test) in enumerate(_multilabel_splits(y, min(plan.outer_folds, len(y)), plan.random_state + repeat)):
                outer.append({"split_key": f"r{repeat}_o{fold}", "repeat": repeat, "outer_fold": fold, "train_idx": train, "test_idx": test})
    inner = {}
    for split in outer:
        train = np.asarray(split["train_idx"], dtype=int)
        pairs = _regression_inner_splits(plan, train, group_values, split, subject_groups=subjects)
        if classification and group_values is None and strategy == "greedy_multilabel_marginal_stratification":
            pairs = _multilabel_splits(y[train], min(plan.inner_folds, len(train)), plan.random_state + 1009 * int(split.get("repeat", 0)) + int(split.get("outer_fold", 0)))
        if not pairs:
            raise ValueError(f"No valid inner folds for {split['split_key']}.")
        inner[split["split_key"]] = pairs
    for split in outer:
        train, test = np.asarray(split["train_idx"]), np.asarray(split["test_idx"])
        if set(subjects[train]) & set(subjects[test]):
            raise ValueError("An outer fold separates repeated measurements from the same subject.")
        for subtrain, valid in inner[split["split_key"]]:
            if set(subjects[train[subtrain]]) & set(subjects[train[valid]]):
                raise ValueError("An inner fold separates repeated measurements from the same subject.")
    return outer, inner, strategy, group_col


def _group_values(dataset, name):
    if name not in dataset.metadata:
        raise ValueError(f"Grouping column {name!r} not present in metadata.")
    values = dataset.metadata[name]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"Grouping column {name!r} contains missing values.")
    return values.astype(str).to_numpy()


def _transform(factory, X_train, X_valid):
    transformer = factory()
    train = np.asarray(transformer.fit_apply(X_train), dtype=np.float32)
    valid = np.asarray(transformer.apply(X_valid), dtype=np.float32)
    if train.ndim != 2 or valid.ndim != 2 or train.shape[1] != valid.shape[1] or train.shape[1] == 0:
        raise ValueError("Transformation produced incompatible or empty feature matrices.")
    if not np.isfinite(train).all() or not np.isfinite(valid).all():
        raise ValueError("Transformation produced non-finite features.")
    return train, valid


def _configurations(sweep, dataset):
    models = sweep.learners
    if isinstance(models, dict):
        models = models.get("joint", models.get("default", ()))
    manifest = Path(sweep.root()) / "joint_experiment.json"
    try:
        historical = json.loads(manifest.read_text(encoding="utf-8")).get("configurations", ()) if manifest.is_file() else ()
    except (ValueError, OSError):
        historical = ()
    historical = list(historical)
    used_ids = {spec["config_id"] for spec in historical}
    names = set()
    items = []
    for item in models:
        name, estimator_factory = _learner_factory(item, task="regression")
        if name in names:
            raise ValueError(f"Duplicate learner name {name!r}.")
        names.add(name)
        for resolution in sweep.resolutions:
            resolution_name, levels = _parse_resolution(resolution)
            if name in sweep.learner_resolutions:
                requested = sweep.learner_resolutions[name]
                allowed = {_parse_resolution(value)[0] for value in ((requested,) if isinstance(requested, str) else requested)}
                if resolution_name not in allowed:
                    continue
            X, feature_names, blocks = materialize_mpdr_with_blocks(dataset, levels)
            transformations = _count_transformation_specs_for_blocks(sweep.count_transformations, blocks, resolution_name)
            for transform_name, transform_spec in transformations:
                _, factory = _count_transformation_factory((transform_name, transform_spec), random_state=sweep.evaluation.random_state, feature_blocks=blocks, feature_names=feature_names)
                configuration = {"learner": name, "resolution": resolution_name, "transformation": transform_name}
                estimator = estimator_factory()
                spec = {
                    **configuration,
                    "resolution_levels": list(levels),
                    "estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
                    "parameters": repr(estimator.get_params(deep=True)),
                }
                matches = [previous for previous in historical if all(previous.get(key) == value for key, value in spec.items())]
                if not matches:
                    matches = [previous for previous in historical if all(previous.get(key) == value for key, value in spec.items() if key != "resolution_levels") and "resolution_levels" not in previous and ((resolution_name == "raw" and list(levels) == ["all"]) or (resolution_name in ("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain") and list(levels) == [resolution_name]))]
                if matches:
                    identifier = matches[0]["config_id"]
                else:
                    identifier = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
                    if identifier in used_ids:
                        identifier = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
                        if identifier in used_ids:
                            raise ValueError("A distinct joint configuration produced an existing configuration identifier.")
                configuration["config_id"] = identifier
                used_ids.add(identifier)
                items.append((configuration, np.asarray(X), factory, estimator_factory))
    if not items:
        raise ValueError("No joint model configurations were generated.")
    return items


def _split_manifest(dataset, y, outer, inner, strategy, group_col):
    signature = hashlib.sha256()
    signature.update(dataset_fingerprint(dataset).encode())
    signature.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    return {
        "data_signature": signature.hexdigest(),
        "strategy": strategy,
        "group_col": group_col,
        "splits": [{
            "key": split["split_key"],
            "repeat": int(split.get("repeat", 0)),
            "outer_fold": int(split.get("outer_fold", 0)),
            "train": np.asarray(split["train_idx"], dtype=int).tolist(),
            "test": np.asarray(split["test_idx"], dtype=int).tolist(),
            "inner": [[np.asarray(a, dtype=int).tolist(), np.asarray(b, dtype=int).tolist()] for a, b in inner[split["split_key"]]],
        } for split in outer],
    }


def _joint_cache(path, old_plan, new_plan, new_splits, redo):
    if redo or old_plan is None:
        return {}
    original = path / "tables" / "joint_cv_splits.json"
    if not original.is_file():
        return {}
    try:
        previous_splits = json.loads(original.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if previous_splits != new_splits:
        return {}
    old_context = {key: value for key, value in old_plan.items() if key not in {"configurations", "active_config_ids"}}
    new_context = {key: value for key, value in new_plan.items() if key not in {"configurations", "active_config_ids"}}
    if old_context != new_context:
        return {}
    original_candidates = {item["config_id"]: item for item in old_plan["configurations"]}
    compatible = set()
    for item in new_plan["configurations"]:
        original = original_candidates.get(item["config_id"])
        if original == item:
            compatible.add(item["config_id"])
            continue
        if original is None or "resolution_levels" in original:
            continue
        legacy = {key: value for key, value in item.items() if key != "resolution_levels"}
        if original != legacy:
            continue
        name = item["resolution"]
        levels = item["resolution_levels"]
        if (name == "raw" and levels == ["all"]) or (name in ("kingdom", "phylum", "class", "order", "family", "genus", "species", "strain") and levels == [name]):
            compatible.add(item["config_id"])
    tables = path / "tables"
    names = ("inner_scores", "inner_predictions", "outer_metrics", "target_metrics", "outer_predictions")
    locations = {name: tables / f"joint_{name}.tsv" for name in names}
    if not all(location.is_file() for location in locations.values()):
        return {}
    try:
        frames = {name: pd.read_csv(location, sep="\t") for name, location in locations.items()}
    except (ValueError, OSError, pd.errors.ParserError):
        return {}
    return {"compatible": compatible, **frames}


def _cached_inner(cache, identifier, split, fold_no, fit_idx, valid_idx, y, targets, classification, metric):
    if identifier not in cache.get("compatible", ()):
        return None
    scores = cache["inner_scores"]
    mask = scores["config_id"].eq(identifier) & scores["split_key"].eq(split) & scores["inner_fold"].eq(fold_no)
    found = scores.loc[mask]
    if len(found) != 1 or str(found.iloc[0]["metric"]) != metric:
        return None
    predictions = cache["inner_predictions"]
    mask = predictions["config_id"].eq(identifier) & predictions["split_key"].eq(split) & predictions["inner_fold"].eq(fold_no)
    subset = predictions.loc[mask]
    if len(subset) != len(valid_idx) or subset["sample_index"].duplicated().any():
        return None
    if set(subset["sample_index"].astype(int)) != set(map(int, valid_idx)):
        return None
    ordered = subset.set_index("sample_index").loc[valid_idx]
    truth = ordered[[f"true__{target}" for target in targets]].to_numpy(dtype=float)
    if not np.allclose(truth, y[valid_idx], atol=1e-9, rtol=0):
        return None
    predicted = ordered[[f"pred__{target}" for target in targets]].to_numpy(dtype=float)
    try:
        measures, _ = _scores(y[valid_idx], predicted, targets, classification, y[fit_idx])
    except ValueError:
        return None
    if not np.isclose(measures[metric], float(found.iloc[0]["score"]), rtol=1e-7, atol=1e-8):
        return None
    return found.to_dict("records")[0], subset.to_dict("records")


def _cached_outer(cache, identifier, split, test_idx, y, targets, classification, selection_score, dataset):
    if identifier not in cache.get("compatible", ()):
        return None
    outer = cache["outer_metrics"]
    found = outer.loc[outer["split_key"].eq(split) & outer["config_id"].eq(identifier)]
    if len(found) != 1 or not np.isclose(float(found.iloc[0]["inner_selection_score"]), selection_score, rtol=1e-7, atol=1e-8):
        return None
    predictions = cache["outer_predictions"]
    selected = predictions.loc[predictions["split_key"].eq(split) & predictions["config_id"].eq(identifier)]
    expected_ids = [str(dataset.sample_ids[int(i)]) for i in test_idx]
    if len(selected) != len(test_idx) or selected["sample_id"].duplicated().any() or set(selected["sample_id"].astype(str)) != set(expected_ids):
        return None
    ordered = selected.set_index("sample_id").loc[expected_ids]
    truth = ordered[[f"true__{target}" for target in targets]].to_numpy(dtype=float)
    if not np.allclose(truth, y[test_idx], rtol=0, atol=1e-9):
        return None
    predicted = ordered[[f"pred__{target}" for target in targets]].to_numpy(dtype=float)
    if predicted.shape != y[test_idx].shape or not np.isfinite(predicted).all():
        return None
    details = cache["target_metrics"]
    subset = details.loc[details["split_key"].eq(split) & details["config_id"].eq(identifier)]
    if len(subset) != len(targets) or set(subset["target"].astype(str)) != set(targets):
        return None
    return found.to_dict("records"), subset.to_dict("records"), selected.to_dict("records")


def _invalidate_joint_downstream(root):
    import shutil

    for filename in ("joint_ensemble.json", "joint_ensemble_resources.json"):
        (root / filename).unlink(missing_ok=True)
    for path in (root / "tables").glob("joint_ensemble_*.tsv"):
        path.unlink()
    for folder in ("explainability", "report"):
        target = root / folder
        if target.is_dir():
            shutil.rmtree(target)
    for path in root.glob("*__report.html"):
        path.unlink()


def evaluate_joint(sweep):
    if not is_joint_sweep(sweep):
        raise ValueError("A joint evaluation requires Data.task='multilabel' or joint multioutput regression.")
    if sweep.evaluation.outer_only or sweep.gate.enabled:
        raise ValueError("Joint evaluation requires full nested CV without a qualification gate.")
    dataset, y, targets, classification = _load_joint_dataset(sweep)
    metric = str(sweep.evaluation.optimize_metric)
    if metric not in (_CLASSIFICATION_METRICS if classification else _REGRESSION_METRICS):
        raise ValueError(f"Unsupported joint optimization metric {metric!r}.")
    root = Path(sweep.root())
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    hardware = {"platform": platform.platform(), "python_version": platform.python_version(), "cpu_model": platform.processor() or None, "logical_cpus": os.cpu_count(), "evaluation_workers": sweep.evaluation.n_jobs}
    try:
        import psutil

        hardware["physical_cpus"] = psutil.cpu_count(logical=False)
        hardware["memory_total_bytes"] = psutil.virtual_memory().total
        hardware["memory_available_bytes"] = psutil.virtual_memory().available
    except ImportError:
        pass
    dump_json_standard(hardware, root / "joint_environment.json")
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outer, inner, strategy, group_col = _make_splits(sweep, dataset, y, classification)
    split_manifest = _split_manifest(dataset, y, outer, inner, strategy, group_col)
    candidates = _configurations(sweep, dataset)
    candidate_specs = []
    resolved_levels = {_parse_resolution(resolution)[0]: list(_parse_resolution(resolution)[1]) for resolution in sweep.resolutions}
    if len(resolved_levels) != len(sweep.resolutions):
        raise ValueError("Joint sweep resolutions must have unique names.")
    if len({config["config_id"] for config, _, _, _ in candidates}) != len(candidates):
        raise ValueError("Joint model configurations must have unique learner, resolution and transformation identities.")
    for config, _, _, make_estimator in candidates:
        estimator = make_estimator()
        params = estimator.get_params(deep=True)
        candidate_specs.append({**config, "resolution_levels": resolved_levels[config["resolution"]], "estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}", "parameters": repr(params)})
    plan = {
        "task": "multilabel" if classification else "multioutput_regression",
        "targets": list(targets),
        "selection_metric": metric,
        "threshold": 0.5 if classification else None,
        "selection_direction": "maximize" if metric in _MAXIMIZE else "minimize",
        "evaluation": {"protocol": sweep.evaluation.protocol, "outer_folds": sweep.evaluation.outer_folds, "inner_folds": sweep.evaluation.inner_folds, "repeats": sweep.evaluation.repeats, "random_state": sweep.evaluation.random_state},
        "split_strategy": strategy,
        "configurations": candidate_specs,
    }
    config_path = root / "joint_experiment.json"
    try:
        old_plan = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    except (ValueError, OSError):
        old_plan = None
    if old_plan is not None and not sweep.evaluation.redo:
        previous_context = {key: value for key, value in old_plan.items() if key not in {"configurations", "active_config_ids"}}
        current_context = {key: value for key, value in plan.items() if key not in {"configurations", "active_config_ids"}}
        if previous_context != current_context:
            raise ValueError("Existing joint evaluation uses a different task, target set, optimization metric, or cross-validation protocol. Use --redo or a new experiment directory.")
        previous_split_path = tables / "joint_cv_splits.json"
        if previous_split_path.is_file():
            try:
                previous_splits = json.loads(previous_split_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError("Existing joint cross-validation splits cannot be read.") from exc
            if previous_splits != split_manifest:
                raise ValueError("Existing joint evaluation uses different data or cross-validation splits. Use --redo or a new experiment directory.")
    if old_plan is not None:
        previous = {spec["config_id"]: spec for spec in old_plan.get("configurations", ())}
        previous.update({spec["config_id"]: spec for spec in candidate_specs})
        plan["configurations"] = list(previous.values())
    paths = {
        "configuration_scores": tables / "joint_configuration_scores.tsv",
        "inner_scores": tables / "joint_inner_scores.tsv",
        "inner_predictions": tables / "joint_inner_predictions.tsv",
        "outer_metrics": tables / "joint_outer_metrics.tsv",
        "target_metrics": tables / "joint_target_metrics.tsv",
        "outer_predictions": tables / "joint_outer_predictions.tsv",
        "summary": root / "joint_summary.json",
        "configuration": config_path,
        "splits": tables / "joint_cv_splits.json",
    }
    cache = _joint_cache(root, old_plan, plan, split_manifest, sweep.evaluation.redo)
    current_specs = {item["config_id"]: item for item in candidate_specs}
    old_active = set(old_plan.get("active_config_ids", ())) if old_plan is not None else set()
    plan["active_config_ids"] = list(current_specs)
    if cache and cache["compatible"] == set(current_specs) and old_plan == plan and old_active == set(current_specs) and all(path.is_file() for path in paths.values()):
        existing = json.loads(paths["summary"].read_text(encoding="utf-8"))
        prediction_count = len(cache["inner_predictions"])
        expected_count = len(plan["configurations"]) * sum(len(split["train_idx"]) for split in outer)
        if existing.get("n_candidates") == len(plan["configurations"]) and existing.get("n_outer_splits") == len(outer) and prediction_count == expected_count and not cache["inner_predictions"].duplicated(["config_id", "split_key", "sample_index"]).any():
            from .console import success

            success("Current joint configuration sweep is already complete; reusing evaluated predictions.")
            return paths
    archived = [item for item in plan["configurations"] if item["config_id"] not in current_specs]
    reused_inner = 0
    reused_outer = 0
    fitted_inner = 0
    fitted_outer = 0
    inner_rows, outer_rows, target_rows, pred_rows, inner_prediction_rows = [], [], [], [], []
    with JointEvaluationMonitor([item[0] for item in candidates], metric, len(outer), [len(inner[split["split_key"]]) for split in outer]) as monitor:
        for split in outer:
            train_idx = np.asarray(split["train_idx"], dtype=int)
            test_idx = np.asarray(split["test_idx"], dtype=int)
            key = split["split_key"]
            ranked = []
            for config, X, make_transform, make_estimator in candidates:
                scores = []
                for fold_no, (subtrain, valid) in enumerate(inner[key]):
                    fit_idx, valid_idx = train_idx[np.asarray(subtrain, dtype=int)], train_idx[np.asarray(valid, dtype=int)]
                    cached = _cached_inner(cache, config["config_id"], key, fold_no, fit_idx, valid_idx, y, targets, classification, metric) if cache else None
                    if cached is not None:
                        score_row, saved_predictions = cached
                        score = float(score_row["score"])
                        inner_rows.append(score_row)
                        inner_prediction_rows.extend(saved_predictions)
                        reused_inner += 1
                    else:
                        X_fit, X_valid = _transform(make_transform, X[fit_idx], X[valid_idx])
                        model = make_estimator()
                        model.fit(X_fit, y[fit_idx])
                        pred = _predict(model, X_valid, classification, len(targets))
                        measures, _ = _scores(y[valid_idx], pred, targets, classification, y[fit_idx])
                        score = measures[metric]
                        if not np.isfinite(score):
                            raise ValueError(f"Metric {metric!r} is undefined for {config['learner']} in {key}, inner fold {fold_no}.")
                        for row_index, sample_index in enumerate(valid_idx):
                            record = {
                                "config_id": config["config_id"],
                                "split_key": key,
                                "repeat": int(split.get("repeat", 0)),
                                "outer_fold": int(split.get("outer_fold", 0)),
                                "inner_fold": fold_no,
                                "sample_index": int(sample_index),
                                "sample_id": str(dataset.sample_ids[int(sample_index)]),
                            }
                            for target_index, target in enumerate(targets):
                                record[f"true__{target}"] = float(y[sample_index, target_index])
                                record[f"pred__{target}"] = float(pred[row_index, target_index])
                            inner_prediction_rows.append(record)
                        inner_rows.append({**config, "split_key": key, "repeat": int(split.get("repeat", 0)), "outer_fold": int(split.get("outer_fold", 0)), "inner_fold": fold_no, "metric": metric, "score": score, **measures})
                        fitted_inner += 1
                    scores.append(score)
                    monitor.record_inner(config["config_id"], score)
                ranked.append((float(np.mean(scores)), config["config_id"], config, X, make_transform, make_estimator))
            for archived_config in archived:
                identifier = archived_config["config_id"]
                scores = []
                saved_rows = []
                saved_predictions = []
                for fold_no, (subtrain, valid) in enumerate(inner[key]):
                    fit_idx = train_idx[np.asarray(subtrain, dtype=int)]
                    valid_idx = train_idx[np.asarray(valid, dtype=int)]
                    found = _cached_inner(cache, identifier, key, fold_no, fit_idx, valid_idx, y, targets, classification, metric) if cache else None
                    if found is None:
                        scores = []
                        break
                    row, values = found
                    saved_rows.append(row)
                    saved_predictions.extend(values)
                    scores.append(float(row["score"]))
                if len(scores) != len(inner[key]):
                    continue
                inner_rows.extend(saved_rows)
                inner_prediction_rows.extend(saved_predictions)
                reused_inner += len(scores)
                archived_metadata = {name: archived_config[name] for name in ("config_id", "learner", "resolution", "transformation")}
                ranked.append((float(np.mean(scores)), identifier, archived_metadata, None, None, None))
                for score in scores:
                    monitor.record_inner(identifier, score)
            ranked.sort(key=lambda row: ((-row[0] if metric in _MAXIMIZE else row[0]), row[1]))
            validation_score, _, selected, X, make_transform, make_estimator = ranked[0]
            cached = _cached_outer(cache, selected["config_id"], key, test_idx, y, targets, classification, validation_score, dataset) if cache else None
            if cached is not None:
                saved_outer, saved_targets, saved_predictions = cached
                outer_rows.extend(saved_outer)
                target_rows.extend(saved_targets)
                pred_rows.extend(saved_predictions)
                reused_outer += 1
            else:
                if make_estimator is None:
                    raise ValueError("The archived joint configuration selected in an outer fold has no reusable held-out prediction. Re-enable that learner to calculate the missing fold.")
                X_fit, X_test = _transform(make_transform, X[train_idx], X[test_idx])
                model = make_estimator()
                model.fit(X_fit, y[train_idx])
                pred = _predict(model, X_test, classification, len(targets))
                measures, details = _scores(y[test_idx], pred, targets, classification, y[train_idx])
                meta = {**selected, "split_key": key, "repeat": int(split.get("repeat", 0)), "outer_fold": int(split.get("outer_fold", 0)), "inner_selection_score": validation_score, "n_train": len(train_idx), "n_test": len(test_idx)}
                outer_rows.append({**meta, **measures})
                target_rows.extend({**meta, **item} for item in details)
                train_scales = np.std(y[train_idx], axis=0) if not classification else None
                for local, sample in enumerate(test_idx):
                    row = {**selected, "split_key": key, "repeat": int(split.get("repeat", 0)), "outer_fold": int(split.get("outer_fold", 0)), "sample_id": dataset.sample_ids[int(sample)], "subject_id": dataset.subject_ids[int(sample)]}
                    for j, target in enumerate(targets):
                        if train_scales is not None:
                            row[f"train_scale__{target}"] = float(max(train_scales[j], 1e-12))
                        row[f"true__{target}"] = float(y[sample, j])
                        row[f"pred__{target}"] = float(pred[local, j])
                    pred_rows.append(row)
                fitted_outer += 1
            monitor.record_outer(selected["config_id"], key)
    for name, rows in (("inner_scores", inner_rows), ("inner_predictions", inner_prediction_rows), ("outer_metrics", outer_rows), ("target_metrics", target_rows), ("outer_predictions", pred_rows)):
        destination = paths[name].with_suffix(".tsv.pending")
        pd.DataFrame(rows).to_csv(destination, sep="\t", index=False)
    inner_frame = pd.DataFrame(inner_rows)
    outer_frame = pd.DataFrame(outer_rows)
    by_split = inner_frame.groupby(["config_id", "split_key"], sort=False)[metric].mean()
    by_config = by_split.groupby(level="config_id")
    selection_counts = outer_frame["config_id"].value_counts().to_dict()
    configuration_rows = []
    for registered in plan["configurations"]:
        identifier = registered["config_id"]
        if identifier not in by_config.groups:
            continue
        config = {name: registered[name] for name in ("config_id", "learner", "resolution", "transformation")}
        values = by_config.get_group(identifier).to_numpy(dtype=float)
        configuration_rows.append({
            **config,
            "active": int(identifier in current_specs),
            "selection_metric": metric,
            "inner_mean": float(np.mean(values)),
            "inner_std_across_outer_splits": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "outer_splits_scored": len(values),
            "selected_outer_splits": int(selection_counts.get(identifier, 0)),
        })
    configuration_rows.sort(key=lambda row: ((-row["inner_mean"] if metric in _MAXIMIZE else row["inner_mean"]), row["config_id"]))
    pd.DataFrame(configuration_rows).to_csv(paths["configuration_scores"].with_suffix(".tsv.pending"), sep="\t", index=False)
    predictions = pd.DataFrame(pred_rows)
    repeat_rows = []
    for repeat, frame in predictions.groupby("repeat", sort=True):
        actual = frame[[f"true__{t}" for t in targets]].to_numpy()
        values = frame[[f"pred__{t}" for t in targets]].to_numpy()
        measures, per_target = _scores(actual, values, targets, classification, y)
        if not classification:
            normalized = []
            for index, target in enumerate(targets):
                scales = frame[f"train_scale__{target}"].to_numpy(dtype=float)
                value = float(np.sqrt(np.mean(((actual[:, index] - values[:, index]) / scales) ** 2)))
                per_target[index]["normalized_rmse"] = value
                normalized.append(value)
            measures["normalized_rmse"] = float(np.mean(normalized))
        repeat_rows.append({"repeat": int(repeat), "metrics": measures, "target_metrics": per_target})
    summary = {
        "task": plan["task"], "targets": targets, "n_samples": len(y), "n_features": dataset.X_by_level["all"].shape[1],
        "n_candidates": len(configuration_rows), "n_active_candidates": len(candidates), "n_outer_splits": len(outer), "n_repeats": len(repeat_rows),
        "model_selection": "Inner CV only; one jointly trained model is selected independently for each outer split.",
        "outer_evaluation": "Each sample is evaluated only when held out within each repeat. Repeats are not independent cohorts.",
        "split_strategy": strategy, "group_col": group_col, "optimize_metric": metric,
        "repeat_metrics": repeat_rows,
        "mean_repeat_metrics": {name: _nanmean([r["metrics"][name] for r in repeat_rows]) for name in repeat_rows[0]["metrics"] if name not in {"n_auroc_defined", "n_ap_defined"}},
        "selected_configurations": selection_counts,
        "configuration_scores": configuration_rows,
        "undefined_auc_policy": "Targets with a single class in an evaluation partition have undefined AUROC/AP and are excluded from macro AUROC/AP; defined counts are reported." if classification else None,
    }
    from .console import success

    changed = old_plan != plan or not cache or fitted_inner > 0 or fitted_outer > 0
    if changed:
        _invalidate_joint_downstream(root)
    for name in ("inner_scores", "inner_predictions", "outer_metrics", "target_metrics", "outer_predictions", "configuration_scores"):
        os.replace(paths[name].with_suffix(".tsv.pending"), paths[name])
    dump_json_standard(summary, paths["summary"])
    dump_json_standard(split_manifest, paths["splits"])
    dump_json_standard(plan, paths["configuration"])
    dump_json_standard({"wall_time_s": time.perf_counter() - started_wall, "parent_cpu_time_s": time.process_time() - started_cpu, "model_fits": fitted_inner + fitted_outer, "inner_fits": fitted_inner, "outer_fits": fitted_outer, "reused_inner_folds": reused_inner, "reused_outer_folds": reused_outer}, root / "joint_evaluation_resources.json")
    success(f"Joint evaluation complete: {fitted_inner} inner fits, {fitted_outer} outer refits, {reused_inner} reused inner folds, {reused_outer} reused outer folds.")
    return paths


def report_joint(sweep):
    root = Path(sweep.root())
    source = root / "joint_summary.json"
    if not source.exists():
        raise FileNotFoundError("Run the joint evaluation before generating its report.")
    summary = json.loads(source.read_text(encoding="utf-8"))
    from .joint_reporting import write_joint_html

    outputs = write_joint_html(root, sweep.title, summary, top_k=sweep.explainability.top_k, include_representation_impact=sweep.include_representation_impact)
    return {"report": outputs["html_report"], "summary": source, **outputs}
