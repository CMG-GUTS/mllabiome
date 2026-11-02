from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .console import stage, success
from .data import dataset_fingerprint
from .ensemble_progress import EnsembleSearchProgress
from .ensemble_sweep import _select_performance_diversity
from .joint_sweep import _CLASSIFICATION_METRICS, _MAXIMIZE, _REGRESSION_METRICS, _configurations, _load_joint_dataset, _predict, _scores, _transform
from .regression_ensemble import _caruana, _fit_convex_regression_weights, _fit_super_learner, _simple_members, aggregate_regression_predictions
from .utils import dump_json_standard

_SUPPORTED_SELECTION = frozenset({"top_k", "best_per_resolution", "best_per_learner_type", "performance_diversity", "caruana", "super_learner"})
_SUPPORTED_AGGREGATION = frozenset({"mean_proba", "weighted_mean_proba", "median_proba", "mean_prediction", "weighted_mean_prediction", "median_prediction"})


def _loss(actual, prediction, targets, classification, reference, metric):
    measures, _ = _scores(actual, np.clip(prediction, 0, 1) if classification else prediction, targets, classification, reference)
    value = float(measures[metric])
    if not np.isfinite(value):
        raise ValueError(f"Ensemble optimization metric {metric!r} is undefined.")
    return -value if metric in _MAXIMIZE else value


def _load_ensemble_splits(root, dataset, y):
    source = Path(root) / "tables" / "joint_cv_splits.json"
    if not source.exists():
        raise FileNotFoundError("Joint ensemble requires a completed evaluation stage.")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    signature = hashlib.sha256()
    signature.update(dataset_fingerprint(dataset).encode())
    signature.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    if signature.hexdigest() != manifest.get("data_signature"):
        raise ValueError("Current training data or target values differ from the evaluated split manifest.")
    return manifest["splits"]


def _inner_predictions(root, split, candidates, targets, y, sample_ids):
    source = Path(root) / "tables" / "joint_inner_predictions.tsv"
    if not source.exists():
        raise FileNotFoundError("Joint ensemble requires saved inner out-of-fold predictions. Rerun the evaluate stage with this version.")
    return pd.read_csv(source, sep="\t")


def _prepare_oof(frame, split, candidates, y, targets):
    train = np.asarray(split["train"], dtype=int)
    key = str(split["key"])
    subset = frame.loc[frame["split_key"].astype(str).eq(key)]
    if subset.empty:
        raise ValueError(f"No inner predictions exist for outer split {key!r}.")
    target_columns = [f"pred__{target}" for target in targets]
    true_columns = [f"true__{target}" for target in targets]
    oof = {}
    scores = {}
    for configuration, _, _, _ in candidates:
        identifier = configuration["config_id"]
        selected = subset.loc[subset["config_id"].astype(str).eq(identifier)]
        if selected.empty or selected["sample_index"].duplicated().any() or len(selected) != len(train):
            raise ValueError(f"Incomplete or repeated inner predictions for {identifier!r} in {key!r}.")
        sample_indexes = selected["sample_index"].to_numpy(dtype=int)
        if not np.array_equal(np.sort(sample_indexes), np.sort(train)):
            raise ValueError("Inner predictions are not aligned to the outer training partition.")
        ordered = selected.set_index("sample_index").loc[train]
        actual = ordered[true_columns].to_numpy(dtype=float)
        if not np.allclose(actual, y[train], atol=1e-9, rtol=0):
            raise ValueError("Saved inner targets disagree with the current dataset.")
        predicted = ordered[target_columns].to_numpy(dtype=float)
        if predicted.shape != y[train].shape or not np.isfinite(predicted).all():
            raise ValueError("Inner predictions have invalid shape or contain non-finite values.")
        oof[identifier] = predicted
    return oof


def _select(strategy, ranked, oof, y_train, configurations, scorer, metric, size):
    ranked_ids = [str(item) for item in ranked.index]
    if strategy in {"top_k", "best_per_resolution", "best_per_learner_type"}:
        members = _simple_members(strategy, ranked, configurations, size)
        return members, None
    stack = np.stack([oof[item] for item in ranked_ids])
    if strategy == "performance_diversity":
        members, _ = _select_performance_diversity(ranked_ids, stack, y_train, metric, size, score_function=scorer, minimize_score=metric not in _MAXIMIZE)
        return members, None
    if strategy == "caruana":
        return _caruana(ranked_ids, stack, y_train, metric, size, score_function=scorer, minimize_score=metric not in _MAXIMIZE)[:2]
    if strategy == "super_learner":
        return _fit_super_learner(ranked_ids, stack, y_train, metric, size, score_function=scorer, minimize_score=metric not in _MAXIMIZE)
    raise ValueError(f"Unknown ensemble selection strategy {strategy!r}.")


def ensemble_joint(sweep):
    if not sweep.ensemble.include_model_ensembles:
        return {}
    forbidden_selection = set(sweep.ensemble.selection_strategies) - _SUPPORTED_SELECTION
    forbidden_aggregation = set(sweep.ensemble.aggregation_strategies) - _SUPPORTED_AGGREGATION
    if forbidden_selection or forbidden_aggregation:
        raise ValueError(f"Unsupported multi-target ensemble strategies: selection={sorted(forbidden_selection)}, aggregation={sorted(forbidden_aggregation)}")
    if sweep.ensemble.late_integrations:
        raise ValueError("Late-modality fusion is not supported by the joint microbiome ensemble adapter.")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    dataset, y, targets, classification = _load_joint_dataset(sweep)
    candidates = _configurations(sweep, dataset)
    candidates_by_id = {item[0]["config_id"]: item for item in candidates}
    if len(candidates_by_id) < 2:
        return {}
    root = Path(sweep.root())
    splits = _load_ensemble_splits(root, dataset, y)
    experiment = json.loads((root / "joint_experiment.json").read_text(encoding="utf-8"))
    evaluated = {item["config_id"]: item for item in experiment["configurations"]}
    if not set(candidates_by_id).issubset(evaluated):
        raise ValueError("The current ensemble library contains configurations that have not been evaluated.")
    for identifier, (config, _, _, make_estimator) in candidates_by_id.items():
        estimator = make_estimator()
        original = evaluated[identifier]
        signature = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
        if original.get("estimator") != signature or original.get("parameters") != repr(estimator.get_params(deep=True)):
            raise ValueError(f"Estimator configuration {identifier!r} differs from the evaluated model.")
    oof_file = _inner_predictions(root, None, None, None, None, None)
    metric = str(sweep.ensemble.optimize_metric)
    if metric.casefold() == "auto":
        metric = str(sweep.evaluation.optimize_metric)
    if metric not in (_CLASSIFICATION_METRICS if classification else _REGRESSION_METRICS):
        raise ValueError(f"Invalid multi-target ensemble optimization metric {metric!r}.")
    options = [(size, strategy, aggregation) for size in sweep.ensemble.max_sizes for strategy in sweep.ensemble.selection_strategies for aggregation in sweep.ensemble.aggregation_strategies if (strategy not in {"caruana", "super_learner"} or aggregation in {"weighted_mean_proba", "weighted_mean_prediction"})]
    records, predictions, fold_metrics, members = [], [], [], []
    fit_count = 0
    stage("Ensemble", str(root))
    with EnsembleSearchProgress([str(split["key"]) for split in splits], len(options)) as monitor:
        for split in splits:
            key = str(split["key"])
            train = np.asarray(split["train"], dtype=int)
            test = np.asarray(split["test"], dtype=int)
            reference = y[train]
            oof = _prepare_oof(oof_file, split, candidates, y, targets)
            scores = {}
            def scorer(actual, prediction):
                truth = np.asarray(actual, dtype=float)
                estimate = np.asarray(prediction, dtype=float)
                if metric == "log_loss":
                    estimate = np.clip(estimate, 1e-15, 1 - 1e-15)
                    return float(-np.mean(truth * np.log(estimate) + (1 - truth) * np.log1p(-estimate)))
                if metric == "brier":
                    return float(np.mean((truth - estimate) ** 2))
                if metric == "normalized_rmse":
                    scale = np.maximum(np.std(reference, axis=0), 1e-12)
                    return float(np.mean(np.sqrt(np.mean((truth - estimate) ** 2, axis=0)) / scale))
                if metric == "MSE":
                    return float(np.mean((truth - estimate) ** 2))
                if metric == "RMSE":
                    return float(np.mean(np.sqrt(np.mean((truth - estimate) ** 2, axis=0))))
                if metric == "MAE":
                    return float(np.mean(np.abs(truth - estimate)))
                return float(_scores(actual, np.clip(prediction, 0, 1) if classification else prediction, targets, classification, reference)[0][metric])
            for identifier, candidate_prediction in oof.items():
                scores[identifier] = scorer(reference, candidate_prediction)
            ranked = pd.Series(scores, dtype=float).sort_values(ascending=metric in _MAXIMIZE, kind="mergesort")
            metadata = pd.DataFrame([item[0] for item in candidates])
            fitted = {}
            for index, (size, strategy, aggregation) in enumerate(options, 1):
                spec = {"selection_strategy": strategy, "aggregation_strategy": aggregation, "max_size": int(size)}
                monitor.update(key, "start", spec, index, len(options))
                selected, learned = _select(strategy, ranked, oof, reference, metadata, scorer, metric, int(size))
                if len(set(selected)) < 2:
                    monitor.update(key, "done", spec, index, len(options), "skipped")
                    continue
                stack = np.stack([oof[item] for item in selected])
                if aggregation in {"weighted_mean_proba", "weighted_mean_prediction"}:
                    weights = learned if learned is not None else _fit_convex_regression_weights(stack, reference, metric, score_function=scorer, minimize_score=metric not in _MAXIMIZE)
                else:
                    weights = np.full(len(selected), 1 / len(selected), dtype=float)
                inner_prediction = aggregate_regression_predictions(stack, aggregation, weights)
                inner_score = float(scorer(reference, inner_prediction))
                for identifier in selected:
                    if identifier not in fitted:
                        _, X, make_transform, make_estimator = candidates_by_id[identifier]
                        x_train, x_test = _transform(make_transform, X[train], X[test])
                        estimator = make_estimator()
                        estimator.fit(x_train, reference)
                        fitted[identifier] = _predict(estimator, x_test, classification, len(targets))
                        fit_count += 1
                prediction = aggregate_regression_predictions(np.stack([fitted[item] for item in selected]), aggregation, weights)
                if classification:
                    prediction = np.clip(prediction, 0.0, 1.0)
                measures, per_target = _scores(y[test], prediction, targets, classification, reference)
                name = f"{strategy}__{aggregation}__{size}"
                fold_metrics.append({"split_key": key, "repeat": int(split["repeat"]), "outer_fold": int(split["outer_fold"]), "strategy": name, "selection": strategy, "aggregation": aggregation, "max_size": int(size), "n_members": len(selected), "inner_score": inner_score, **measures})
                members.extend({"split_key": key, "strategy": name, "config_id": identifier, "weight": float(weight)} for identifier, weight in zip(selected, weights))
                scales = np.std(reference, axis=0) if not classification else None
                for j, sample in enumerate(test):
                    item = {"split_key": key, "repeat": int(split["repeat"]), "outer_fold": int(split["outer_fold"]), "strategy": name, "sample_id": str(dataset.sample_ids[int(sample)]), "subject_id": str(dataset.subject_ids[int(sample)])}
                    for k, target in enumerate(targets):
                        item[f"true__{target}"] = float(y[sample, k])
                        item[f"pred__{target}"] = float(prediction[j, k])
                        if scales is not None:
                            item[f"train_scale__{target}"] = float(max(scales[k], 1e-12))
                    predictions.append(item)
                records.append({"split_key": key, "strategy": name, "selection": strategy, "aggregation": aggregation, "inner_loss": float(inner_score if metric not in _MAXIMIZE else -inner_score), "members": selected, "weights": [float(value) for value in weights], "per_target_metrics": per_target})
                monitor.update(key, "done", spec, index, len(options), "valid")
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {"ensemble_metrics": tables / "joint_ensemble_outer_metrics.tsv", "ensemble_predictions": tables / "joint_ensemble_outer_predictions.tsv", "ensemble_members": tables / "joint_ensemble_members.tsv", "ensemble_manifest": root / "joint_ensemble.json"}
    for name, rows in (("ensemble_metrics", fold_metrics), ("ensemble_predictions", predictions), ("ensemble_members", members)):
        pd.DataFrame(rows).to_csv(outputs[name], sep="\t", index=False)
    dump_json_standard({"task": "multilabel" if classification else "multioutput_regression", "targets": targets, "metric": metric, "selection_protocol": "persisted_inner_oof_only", "strategies": sorted({row["strategy"] for row in records}), "records": records}, outputs["ensemble_manifest"])
    dump_json_standard({"wall_time_s": time.perf_counter() - started_wall, "parent_cpu_time_s": time.process_time() - started_cpu, "model_fits": fit_count}, root / "joint_ensemble_resources.json")
    success("Ensemble completed")
    return outputs
