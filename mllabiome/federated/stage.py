from __future__ import annotations

import hashlib
from collections import deque
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from ..configs_sweep import build_sweep_configs
from ..console import console, info, success
from ..data import dataset_fingerprint, load_dataset
from ..evaluation_splits import (
    _groups_from_metadata,
    _resolved_evaluation_splits,
    _strata_from_metadata,
)
from ..metrics import compute_metrics
from ..resolutions import _parse_resolution, materialize_mpdr_with_blocks
from ..transformations import (
    CountTransformationAdapter,
    _count_transformation_specs_for_blocks,
    _parse_transformation_identity,
)
from .config import Federated
from .model import predict_proba, validate_parameters


_LOCAL_SAFE_TRANSFORMS = frozenset(
    {
        "identity",
        "relative_abundance",
        "presence_absence",
        "hellinger",
        "arcsine_sqrt",
        "centered_log_ratio_multiplicative_replacement",
        "within_sample_fractional_rank",
    }
)


def _validate_sweep(sweep):
    spec = sweep.federated
    if not isinstance(spec, Federated):
        raise ValueError("Set FEDERATED = Federated(...) to use federated-evaluate.")
    if sweep.uses_modalities or sweep.data is None:
        raise ValueError(
            "Federated evaluation currently supports single-table DATA sweeps only."
        )
    if sweep.data.task != "classification" or sweep.evaluation.protocol not in {
        "lodo",
        "leave_one_dataset_out",
    }:
        raise ValueError(
            "Federated evaluation currently supports binary classification LODO only."
        )
    if len(sweep.learners) != 1:
        raise ValueError(
            "Exactly one SGDClassifier learner is supported in federated evaluation."
        )
    _, model = sweep.learners[0]
    if not isinstance(model, SGDClassifier):
        raise ValueError("MODELS must contain one sklearn.linear_model.SGDClassifier.")
    params = model.get_params()
    expected = {
        "loss": "log_loss",
        "penalty": "l2",
        "learning_rate": "constant",
        "fit_intercept": True,
        "average": False,
        "early_stopping": False,
        "class_weight": None,
        "warm_start": False,
    }
    incompatible = [key for key, value in expected.items() if params[key] != value]
    if params["shuffle"] is not True:
        incompatible.append("shuffle")
    if params["random_state"] not in (None, spec.random_state):
        incompatible.append("random_state")
    if incompatible or params["eta0"] <= 0 or params["alpha"] < 0:
        raise ValueError(
            f"SGDClassifier is not supported with these settings: {incompatible!r}."
        )
    return spec, params


def _transformed(X, identity, transform_spec, feature_names, blocks, seed):
    adapter = CountTransformationAdapter(
        identity,
        transform_spec,
        random_state=seed,
        feature_names=feature_names,
        feature_blocks=blocks,
    )
    adapter.fit(X)
    result = np.asarray(adapter.apply(X), dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Transformed feature matrix is invalid.")
    return result, adapter.get_feature_names_out(feature_names)


def _stage_clients(
    folder: Path,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
):
    cohort_ids = sorted(set(groups[train_idx].tolist()))
    sizes = []
    for client_index, cohort in enumerate(cohort_ids):
        selection = train_idx[groups[train_idx] == cohort]
        if len(selection) == 0:
            raise ValueError("Training cohort cannot be empty.")
        path = folder / f"client_{client_index}.npz"
        with path.open("wb") as handle:
            np.savez(handle, X=X[selection], y=y[selection])
        os.chmod(path, 0o600)
        sizes.append(int(len(selection)))
    return cohort_ids, sizes


def _flower_run(
    sweep,
    spec: Federated,
    params: dict,
    folder: Path,
    client_sizes: list[int],
    n_features: int,
    log_path: Path | None = None,
):
    if importlib.util.find_spec("flwr") is None:
        raise ImportError("Install Flower using uv pip install -e '.[federated]'.")
    flower_binary = Path(sys.executable).parent / "flwr"
    if not flower_binary.is_file():
        raise RuntimeError("Flower CLI is missing from the active Python environment.")
    default_project = (
        Path(__file__).resolve().parents[2] / "examples" / "federated" / "flower-secagg"
    )
    project = Path(spec.flower_project or default_project).resolve()
    app_config = project / "pyproject.toml"
    if not app_config.is_file():
        raise FileNotFoundError(
            f"Flower App project not found: {project}. Install from a repository checkout or set FEDERATED.flower_project."
        )
    with app_config.open("rb") as handle:
        metadata = tomllib.load(handle)
    app_version = metadata.get("project", {}).get("version")
    components = (
        metadata.get("tool", {}).get("flwr", {}).get("app", {}).get("components", {})
    )
    if (
        not isinstance(app_version, str)
        or not app_version
        or not all(components.get(key) for key in ("serverapp", "clientapp"))
    ):
        raise ValueError(
            f"Invalid Flower App metadata in {app_config}: a static project version and both app components are required."
        )
    result_file = folder / "aggregated.npz"
    n_clients = len(client_sizes)
    options = {
        "data-dir": str(folder.resolve()),
        "output-path": str(result_file.resolve()),
        "num-clients": n_clients,
        "num-features": n_features,
        "num-server-rounds": spec.num_rounds,
        "local-epochs": spec.local_epochs,
        "num-shares": spec.num_shares,
        "reconstruction-threshold": spec.reconstruction_threshold,
        "clipping-range": spec.clipping_range,
        "max-weight": max(client_sizes),
        "random-state": spec.random_state,
        "alpha": params["alpha"],
        "eta0": params["eta0"],
    }
    run_config_path = folder / "run_config.toml"
    run_config_path.write_text(
        "\n".join(f"{key} = {json.dumps(value)}" for key, value in options.items())
        + "\n",
        encoding="utf-8",
    )
    command = [
        str(flower_binary),
        "run",
        str(project),
        "local",
        "--run-config",
        str(run_config_path),
        "--federation-config",
        f"num-supernodes={n_clients} client-resources-num-cpus={int(spec.client_cpus)}",
        "--stream",
    ]
    preserved_log = (
        Path(log_path)
        if log_path is not None
        else folder.parent / "flower-logs" / f"{folder.name}.log"
    )
    preserved_log.parent.mkdir(parents=True, exist_ok=True)
    info(f"Launching Flower SecAgg+: {n_clients} clients, {spec.num_rounds} rounds")
    info(f"Flower output will be streamed and saved to {preserved_log}")
    tail = deque(maxlen=120)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with preserved_log.open("w", encoding="utf-8", buffering=1) as log:
        with console.status("Flower SecAgg+ running", spinner="dots"):
            with subprocess.Popen(
                command,
                cwd=project,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            ) as process:
                try:
                    if process.stdout is None:
                        raise RuntimeError("Flower output stream could not be opened.")
                    for line in process.stdout:
                        log.write(line)
                        message = line.rstrip("\r\n")
                        if message:
                            tail.append(message)
                            console.print(
                                message, markup=False, highlight=False, soft_wrap=True
                            )
                    exit_code = process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
    recent_output = "\n".join(tail)[-6000:]
    if exit_code != 0:
        raise RuntimeError(
            f"Flower returned exit code {exit_code}. Log: {preserved_log}\n{recent_output}"
        )
    if not result_file.is_file():
        raise RuntimeError(
            f"Flower returned exit code 0 without creating aggregated.npz. Log: {preserved_log}\n{recent_output}"
        )
    with np.load(result_file, allow_pickle=False) as source:
        arrays = [source["coef"].copy(), source["intercept"].copy()]
        final_round = int(source["round"])
    if final_round != spec.num_rounds:
        raise RuntimeError(
            f"Flower completed {final_round}/{spec.num_rounds} secure rounds."
        )
    validate_parameters(arrays, n_features)
    return arrays


def _validate_lodo_partitions(splits: list[dict], groups: np.ndarray) -> None:
    seen = set()
    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        held_out = str(split["outer_group"])
        if held_out in seen:
            raise ValueError(f"Repeated LODO held-out study {held_out!r}.")
        seen.add(held_out)
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"Empty LODO partition for study {held_out!r}.")
        if len(np.unique(np.r_[train_idx, test_idx])) != len(groups):
            raise ValueError(
                f"Incomplete or overlapping LODO partition for study {held_out!r}."
            )
        if not np.all(groups[test_idx] == held_out):
            raise ValueError(f"Held-out samples disagree with study {held_out!r}.")
        if not np.all(groups[train_idx] != held_out):
            raise ValueError(f"Training samples contain held-out study {held_out!r}.")
    if seen != set(groups.tolist()):
        raise ValueError(
            "LODO manifest does not cover each benchmark study exactly once."
        )


def federated_evaluate(sweep) -> dict[str, Path]:
    spec, params = _validate_sweep(sweep)
    info("Federated evaluation: loading data through mllabiome")
    resolutions = list(
        dict.fromkeys(_parse_resolution(item) for item in sweep.resolutions)
    )
    levels = tuple(
        dict.fromkeys(
            level
            for _, values in resolutions
            for level in values
            if level not in {"all", "features", "asis", "raw"}
        )
    )
    dataset = load_dataset(sweep.data, levels or ("all",))
    if len(dataset.class_labels) != 2 or set(np.unique(dataset.y)) != {0, 1}:
        raise ValueError(
            "Federated logistic classification requires binary labels 0 and 1."
        )
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    info(f"Loaded {len(dataset.y)} samples across {len(np.unique(groups))} studies")
    root = sweep.root() / "federated"
    root.mkdir(parents=True, exist_ok=True)
    strata = _strata_from_metadata(dataset.metadata, dataset.y, sweep.data.stratify_col)
    info("Resolving benchmark LODO split manifest")
    splits, _ = _resolved_evaluation_splits(
        root,
        sweep.evaluation,
        dataset,
        groups,
        strata,
        sweep.data.stratify_col,
        sweep.data.group_col,
    )
    _validate_lodo_partitions(splits, groups)
    if spec.held_out_study is not None:
        splits = [item for item in splits if item["outer_group"] == spec.held_out_study]
    if not splits:
        raise ValueError("No requested held-out LODO study exists.")
    info(
        f"Resolved {len(splits)} held-out study fold(s) and {len(resolutions)} MPDR resolution(s)"
    )
    if len(dataset.sample_ids) != len(set(dataset.sample_ids)):
        raise ValueError("Sample IDs must be unique.")
    subject_ids = np.asarray(dataset.subject_ids, dtype=object)
    labels_by_subject = pd.DataFrame({"subject": subject_ids, "label": dataset.y})
    if (labels_by_subject.groupby("subject")["label"].nunique() > 1).any():
        raise ValueError("Repeated subjects have inconsistent class labels.")
    fold_rows = []
    predictions = []
    configuration_rows = []
    for resolution, selected_levels in resolutions:
        info(f"Materializing MPDR resolution {resolution}")
        X_raw, feature_names, blocks = materialize_mpdr_with_blocks(
            dataset, selected_levels
        )
        transformations = _count_transformation_specs_for_blocks(
            sweep.count_transformations,
            blocks,
            resolution=resolution,
        )
        eligible = []
        for name, transform_spec in transformations:
            base, _ = _parse_transformation_identity(name)
            if (
                base not in _LOCAL_SAFE_TRANSFORMS
                or getattr(transform_spec, "feature_filter", None) is not None
                or getattr(transform_spec, "_fn", None) is not None
                or (
                    transform_spec is not None
                    and type(transform_spec).__name__
                    not in {"Transform", "CountTransformation"}
                )
            ):
                raise ValueError(
                    f"Transformation {name!r} is not federated-safe in the current implementation."
                )
            eligible.append((name, transform_spec))
        table = build_sweep_configs(
            [(resolution, selected_levels)],
            [
                spec_item if spec_item is not None else name
                for name, spec_item in eligible
            ],
            sweep.learners,
            resolution_feature_blocks={resolution: blocks},
            learner_resolutions=sweep.learner_resolutions,
        )
        configuration_rows.extend(table.to_dict("records"))
        info(
            f"Resolution {resolution}: {len(eligible)} compatible count transformation(s)"
        )
        for transformation_name, transform_spec in eligible:
            info(
                f"Preparing transformation {transformation_name} for {len(splits)} LODO fold(s)"
            )
            matches = table.loc[table["count_transformation"] == transformation_name]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one MPMA configuration for {resolution!r} and {transformation_name!r}."
                )
            config_id = str(matches.iloc[0]["config_id"])
            for fold_number, split in enumerate(splits, start=1):
                info(
                    f"LODO fold {fold_number}/{len(splits)}: preparing {split['outer_group']}"
                )
                train_idx = np.asarray(split["train_idx"], dtype=int)
                test_idx = np.asarray(split["test_idx"], dtype=int)
                heldout = str(split["outer_group"])
                if len(train_idx) == 0 or len(test_idx) == 0:
                    raise ValueError(
                        "LODO fold contains an empty train or test partition."
                    )
                if len(np.unique(dataset.y[train_idx])) != 2:
                    raise ValueError(
                        f"Training partition for {heldout!r} has only one class."
                    )
                if set(subject_ids[train_idx]) & set(subject_ids[test_idx]):
                    raise ValueError(f"LODO subject leakage detected for {heldout!r}.")
                cohort_groups = sorted(set(groups[train_idx].tolist()))
                if len(cohort_groups) < spec.num_shares:
                    raise ValueError(
                        f"LODO {heldout!r} has {len(cohort_groups)} training clients; at least {spec.num_shares} are needed for SecAgg+."
                    )
                X = np.empty((len(X_raw), 0), dtype=np.float64)
                transformed_schema = None
                for cohort in cohort_groups:
                    idx = train_idx[groups[train_idx] == cohort]
                    transformed, schema = _transformed(
                        X_raw[idx],
                        transformation_name,
                        transform_spec,
                        feature_names,
                        blocks,
                        spec.random_state,
                    )
                    if transformed_schema is None:
                        transformed_schema = schema
                        X = np.empty(
                            (len(X_raw), transformed.shape[1]), dtype=np.float64
                        )
                    elif schema != transformed_schema:
                        raise ValueError(
                            "Client transformations yielded different feature coordinates."
                        )
                    X[idx] = transformed
                X_test, test_schema = _transformed(
                    X_raw[test_idx],
                    transformation_name,
                    transform_spec,
                    feature_names,
                    blocks,
                    spec.random_state,
                )
                if test_schema != transformed_schema:
                    raise ValueError(
                        "Held-out transformation has incompatible coordinates."
                    )
                with tempfile.TemporaryDirectory(prefix="mllabiome-secagg-") as tmp:
                    folder = Path(tmp)
                    cohort_ids, client_sizes = _stage_clients(
                        folder, X, dataset.y, groups, train_idx
                    )
                    if cohort_ids != cohort_groups:
                        raise RuntimeError(
                            "Flower client cohort staging is inconsistent."
                        )
                    identifier = hashlib.sha256(
                        f"{config_id}|{split['split_key']}|{heldout}".encode("utf-8")
                    ).hexdigest()[:20]
                    info(
                        f"LODO {heldout}: staged {len(client_sizes)} study clients, {len(train_idx)} training samples, {X.shape[1]} features"
                    )
                    log_path = root / "flower-logs" / f"{identifier}.log"
                    parameters = _flower_run(
                        sweep,
                        spec,
                        params,
                        folder,
                        client_sizes,
                        X.shape[1],
                        log_path=log_path,
                    )
                probabilities = predict_proba(parameters, X_test)
                predicted = np.argmax(probabilities, axis=1)
                scores = compute_metrics(
                    dataset.y[test_idx],
                    predicted,
                    probabilities,
                    dataset.classes,
                    dataset.positive_class,
                )
                metric_summary = ", ".join(
                    f"{metric}={float(scores[metric]):.4f}"
                    for metric in ("AUROC", "log_loss")
                    if metric in scores and pd.notna(scores[metric])
                )
                success(
                    f"LODO {heldout} complete: {metric_summary or 'metrics recorded'}"
                )
                fold_rows.append(
                    {
                        "config_id": config_id,
                        "split_key": str(split["split_key"]),
                        "outer_group": heldout,
                        "resolution": resolution,
                        "count_transformation": transformation_name,
                        "learner": str(sweep.learners[0][0]),
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_clients": len(cohort_ids),
                        "num_rounds": spec.num_rounds,
                        **scores,
                    }
                )
                for row_index, source_index in enumerate(test_idx):
                    predictions.append(
                        {
                            "config_id": config_id,
                            "split_key": str(split["split_key"]),
                            "outer_group": heldout,
                            "sample_id": dataset.sample_ids[source_index],
                            "subject_id": dataset.subject_ids[source_index],
                            "y_true": int(dataset.y[source_index]),
                            "y_pred": int(predicted[row_index]),
                            "probability_0": float(probabilities[row_index, 0]),
                            "probability_1": float(probabilities[row_index, 1]),
                        }
                    )
    paths = {
        "metrics": root / "fold_metrics.csv",
        "predictions": root / "predictions.csv",
        "configs": root / "mpma_configs.csv",
        "settings": root / "run.json",
    }
    pd.DataFrame(fold_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(predictions).to_csv(paths["predictions"], index=False)
    pd.DataFrame(configuration_rows).drop_duplicates("config_id").to_csv(
        paths["configs"], index=False
    )
    payload = {
        "stage": "federated-evaluate",
        "backend": "flower-secaggplus",
        "model": "SGDClassifier",
        "federated": asdict(spec),
        "classes": dataset.class_labels,
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "benchmark_id": sweep.evaluation.benchmark_id,
        "split_manifest": str(sweep.evaluation.split_manifest or ""),
        "split_keys": [str(split["split_key"]) for split in splits],
        "mpma_config_ids": sorted(
            set(pd.DataFrame(configuration_rows)["config_id"].tolist())
        ),
    }
    payload["federated"]["flower_project"] = str(
        payload["federated"]["flower_project"] or ""
    )
    paths["settings"].write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    success(
        f"Federated evaluation complete: {len(fold_rows)} fold results written to {root}"
    )
    return paths
