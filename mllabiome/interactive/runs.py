from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..final_models import aggregate_member_predictions
from ..inference import (
    _abundance_scale,
    _align_resolution,
    _prediction_frame,
    _profile_resolution,
    _scale_validation_mode,
)
from ..metrics import _estimator_call, _predict_proba_aligned


@dataclass(frozen=True)
class Run:
    id: str
    path: Path
    target: str


class RunStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        paths = [(self.root.name, self.root)]
        manifest = self.root / "inference_manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            paths = [
                (
                    str(row["target"]),
                    Path(row["experiment_dir"]).resolve()
                    if Path(row["experiment_dir"]).is_absolute()
                    else (self.root / "targets" / str(row["target"])).resolve(),
                )
                for row in data.get("targets", [])
            ]
        self.runs = {}
        for index, (target, path) in enumerate(paths):
            if path != self.root and self.root not in path.parents:
                raise ValueError(
                    "An inference target lies outside the experiment directory."
                )
            if (path / "inference" / "deployment" / "manifest.json").is_file():
                self.runs[str(index)] = Run(str(index), path, target)
        if not self.runs:
            raise FileNotFoundError(
                "No completed inference run found. Run --stage inference first."
            )

    def get(self, run_id: str) -> Run:
        if run_id not in self.runs:
            raise KeyError("Unknown experiment target.")
        return self.runs[run_id]


def read_manifest(run: Run) -> dict[str, Any]:
    return json.loads(
        (run.path / "inference" / "deployment" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )


def read_summary(run: Run) -> dict[str, Any]:
    manifest = read_manifest(run)
    return {
        "id": run.id,
        "target": run.target,
        "task": manifest["task"],
        "strategies": manifest["targets"],
        "classes": manifest.get("class_labels", []),
        "explainable": (
            run.path / "inference" / "deployment" / "background.npz"
        ).is_file()
        and manifest["task"] == "classification",
    }


def read_predictions(run: Run) -> list[dict[str, Any]]:
    path = run.path / "inference" / "predictions.tsv"
    if not path.is_file():
        return []
    return json.loads(
        pd.read_csv(path, sep="\t", dtype={"sample_id": str}).to_json(orient="records")
    )


def read_profile(
    payload: bytes, format: str, sample_id_col: str = "sample_id"
) -> tuple[np.ndarray, list[str], list[str]]:
    from io import BytesIO

    if format == "matrix":
        try:
            frame = pd.read_csv(BytesIO(payload), sep="\t", index_col=0)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
            raise ValueError("Cannot parse the inference matrix.") from exc
        names = frame.index.astype(str).str.strip().tolist()
        samples = frame.columns.astype(str).str.strip().tolist()
        frame = frame.T
    elif format == "wide":
        try:
            frame = pd.read_csv(
                BytesIO(payload), sep=None, engine="python", dtype={sample_id_col: str}
            )
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
            raise ValueError("Cannot parse the inference table.") from exc
        if sample_id_col not in frame:
            raise ValueError(f"Missing {sample_id_col!r} column.")
        samples = frame.pop(sample_id_col).astype(str).str.strip().tolist()
        names = frame.columns.astype(str).str.strip().tolist()
    else:
        raise ValueError("Format must be wide or matrix.")
    if (
        not samples
        or not names
        or len(samples) != len(set(samples))
        or len(names) != len(set(names))
    ):
        raise ValueError("Input requires unique, nonempty sample IDs and features.")
    if any(not name for name in samples + names):
        raise ValueError("Sample IDs and feature names must not be empty.")
    X = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(X).all() or (X < 0).any():
        raise ValueError("Abundances must be finite and nonnegative.")
    return X, names, samples


def read_profile_for_run(
    run: Run, payload: bytes, format: str
) -> tuple[np.ndarray, list[str], list[str]]:
    X, names, samples = read_profile(payload, format)
    if format == "wide":
        try:
            _, expected = read_background(run)
        except FileNotFoundError:
            return X, names, samples
        lookup = {name: index for index, name in enumerate(names)}
        missing = [name for name in expected if name not in lookup]
        if missing:
            raise ValueError(
                f"Wide inference input is missing {len(missing)} training feature(s): {missing[:5]!r}."
            )
        return X[:, [lookup[name] for name in expected]], expected, samples
    return X, names, samples


@lru_cache(maxsize=64)
def read_bundle(path: Path, modified: int) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported fitted model bundle.")
    return bundle


def predict_values(
    run: Run,
    X: np.ndarray,
    names: list[str],
    strategy: str,
    *,
    validate_scale: bool = True,
) -> np.ndarray:
    manifest = read_manifest(run)
    if strategy not in manifest["targets"]:
        raise ValueError("Strategy is not available in this run.")
    models = manifest["final_models"]
    model = models[strategy]
    members = [model] if strategy == "MPMA-B" else model["members"]
    audit_by_id = {str(item["config_id"]): item for item in manifest["members"]}
    cache = {}
    values = []
    for member in members:
        config_id = str(member["config_id"])
        if config_id not in cache:
            path = (
                run.path
                / "inference"
                / "deployment"
                / "members"
                / f"{config_id}.joblib"
            )
            if path.parent != path.resolve().parent or not path.is_file():
                raise FileNotFoundError("A fitted model is unavailable.")
            bundle = read_bundle(path, path.stat().st_mtime_ns)
            if bundle.get("signature") != audit_by_id[config_id].get(
                "deployment_signature"
            ):
                raise ValueError(
                    "The fitted model signature does not match the deployment manifest."
                )
            raw_names = list(bundle["raw_feature_names"])
            resolved, resolved_names = _profile_resolution(
                X, names, tuple(bundle["levels"])
            )
            aligned, _ = _align_resolution(
                np.empty((0, len(raw_names))),
                raw_names,
                resolved,
                resolved_names,
                policy=read_inference_policy(run),
            )
            training_scale = audit_by_id[config_id]["training_abundance_scale"]
            if (
                validate_scale
                and _scale_validation_mode(
                    member, training_scale, _abundance_scale(aligned)
                )
                == "mismatch"
            ):
                raise ValueError(
                    f"Abundance scale differs from training for {config_id}."
                )
            transformed = bundle["transformer"].apply(aligned)
            if manifest["task"] == "classification":
                prediction = _predict_proba_aligned(
                    bundle["estimator"],
                    transformed,
                    np.arange(len(manifest["class_labels"])),
                )
            else:
                prediction = np.asarray(
                    _estimator_call(bundle["estimator"], "predict", transformed),
                    dtype=float,
                ).reshape(-1, 1)
            cache[config_id] = prediction
        values.append(cache[config_id])
    if strategy == "MPMA-B":
        result = values[0]
        alpha = float(model.get("probability_softening_alpha", 0))
        if alpha:
            from ..mpma_vnext import soften_probabilities

            result = soften_probabilities(result, alpha)
    else:
        aggregation = str(model["aggregation_strategy"])
        weights = None
        if aggregation in {"weighted_mean_proba", "weighted_mean_prediction"}:
            weights = [
                float(item.get("aggregation_weight", item.get("weight")))
                for item in members
            ]
        result = aggregate_member_predictions(
            np.stack(values),
            aggregation,
            weights,
            model.get("aggregation_parameters", {}),
        )
    result = np.asarray(result, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("The fitted model returned nonfinite predictions.")
    return result


def read_inference_policy(run: Run) -> str:
    path = run.path / "inference" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")).get("feature_policy", "strict")


def predict_table(
    run: Run, X: np.ndarray, names: list[str], samples: list[str], strategy: str
) -> list[dict[str, Any]]:
    from types import SimpleNamespace

    manifest = read_manifest(run)
    dataset = SimpleNamespace(
        task=manifest["task"], class_labels=manifest.get("class_labels", [])
    )
    frame = _prediction_frame(
        samples, dataset, {strategy: predict_values(run, X, names, strategy)}
    )
    return json.loads(frame.to_json(orient="records"))


def read_background(run: Run) -> tuple[np.ndarray, list[str]]:
    path = run.path / "inference" / "deployment" / "background.npz"
    if not path.is_file():
        raise FileNotFoundError(
            "Run inference again to save a training reference for explanations."
        )
    with np.load(path, allow_pickle=False) as data:
        fingerprint = str(data["fingerprint"].item())
        if fingerprint != read_manifest(run)["training_dataset_fingerprint"]:
            raise ValueError(
                "The explanation reference does not match the fitted model."
            )
        return data["X"].astype(float), data["names"].astype(str).tolist()


def read_saved_samples(run: Run) -> tuple[np.ndarray, list[str], list[str]]:
    path = run.path / "inference" / "samples.npz"
    if not path.is_file():
        raise FileNotFoundError(
            "Inference inputs are not saved in this run. Run --stage inference again or upload the original samples."
        )
    inference_manifest = json.loads(
        (run.path / "inference" / "manifest.json").read_text(encoding="utf-8")
    )
    with np.load(path, allow_pickle=False) as data:
        if (
            str(data["fingerprint"].item())
            != inference_manifest["input"]["abundance_sha256"]
        ):
            raise ValueError("Saved inference inputs do not match this experiment run.")
        return (
            data["X"].astype(float),
            data["names"].astype(str).tolist(),
            data["samples"].astype(str).tolist(),
        )
