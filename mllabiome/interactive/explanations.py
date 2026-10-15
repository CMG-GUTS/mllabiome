from __future__ import annotations

import numpy as np

from ..explainability_methods import LIME, SHAP
from ..explainability_values import _lime_values_for_data, _shap_values_for_data
from ..data import _taxonomic_rank
from ..inference import _abundance_scale
from .runs import Run, predict_values, read_background, read_bundle, read_manifest


def explain_sample(
    run: Run,
    X: np.ndarray,
    names: list[str],
    samples: list[str],
    sample_id: str,
    strategy: str,
    method: str,
) -> dict:
    if sample_id not in samples:
        raise ValueError("Unknown sample ID.")
    manifest = read_manifest(run)
    if manifest["task"] != "classification":
        raise ValueError(
            "Interactive SHAP/LIME is available for classification runs only."
        )
    if method not in {"shap", "lime"}:
        raise ValueError("Method must be shap or lime.")
    training, training_names = read_background(run)
    prediction = predict_values(run, X, names, strategy)[samples.index(sample_id)]
    positions = {name: index for index, name in enumerate(names)}
    row = np.asarray(
        [
            X[samples.index(sample_id), positions[name]] if name in positions else 0.0
            for name in training_names
        ]
    )
    model = manifest["final_models"][strategy]
    members = [model] if strategy == "MPMA-B" else model["members"]
    allowed = set()
    for member in members:
        path = (
            run.path
            / "inference"
            / "deployment"
            / "members"
            / f"{member['config_id']}.joblib"
        )
        allowed.update(read_bundle(path, path.stat().st_mtime_ns)["raw_feature_names"])
    candidates = np.asarray(
        [
            index
            for index, name in enumerate(training_names)
            if name in positions and name in allowed
        ],
        dtype=int,
    )
    if not len(candidates):
        raise ValueError("No training features match the inference profile.")
    baseline = np.mean(training, axis=0)
    deviation = np.abs(row[candidates] - baseline[candidates]) / (
        np.std(training[:, candidates], axis=0) + 1e-6
    )
    selected = candidates[np.argsort(-deviation, kind="stable")[:10]]
    groups = {}
    for index, name in enumerate(training_names):
        groups.setdefault(_taxonomic_rank(name), []).append(index)
    normalize = [
        indices
        for indices in groups.values()
        if _abundance_scale(training[:, indices]) in {"proportion", "percentage"}
    ]
    labels = [str(value) for value in manifest["class_labels"]]

    class Predictor:
        classes_ = np.arange(len(labels))

        def predict_proba(self, changes: np.ndarray) -> np.ndarray:
            changes = np.atleast_2d(np.asarray(changes, dtype=float))
            inputs = np.repeat(row[np.newaxis, :], len(changes), axis=0)
            inputs[:, selected] = np.maximum(changes, 0)
            for indices in normalize:
                totals = inputs[:, indices].sum(axis=1, keepdims=True)
                target = float(row[indices].sum())
                inputs[:, indices] = np.divide(
                    inputs[:, indices] * target,
                    totals,
                    out=np.zeros_like(inputs[:, indices]),
                    where=totals > 0,
                )
            return predict_values(
                run, inputs, training_names, strategy, validate_scale=False
            )

    classifier = Predictor()
    probability = prediction
    class_index = int(np.argmax(probability))
    background = training[:, selected]
    evaluation = row[selected][None, :]
    feature_names = [training_names[int(index)] for index in selected]
    if method == "shap":
        values, _, backend = _shap_values_for_data(
            classifier,
            background,
            evaluation,
            feature_names,
            labels,
            random_state=42,
            spec=SHAP(
                algorithm="permutation",
                background_size=min(24, len(training)),
                max_explain=1,
                permutation_rounds=2,
            ),
            show_progress=False,
        )
        indices = values.shape[-1]
        attribution = (
            values[0, :, class_index]
            if indices > 1
            else values[0, :, 0] * (1 if class_index == 1 else -1)
        )
    else:
        values, _ = _lime_values_for_data(
            classifier,
            background,
            evaluation,
            feature_names,
            labels,
            [class_index],
            random_state=42,
            spec=LIME(num_samples=400, max_explain=1),
        )
        attribution = values[0, :, 0]
        backend = "lime"
    order = np.argsort(-np.abs(attribution))
    return {
        "sample_id": sample_id,
        "strategy": strategy,
        "method": method,
        "backend": backend,
        "class_label": labels[class_index],
        "probability": float(probability[class_index]),
        "scope": "screened local attribution",
        "features": [
            {
                "name": feature_names[int(i)],
                "value": float(attribution[int(i)]),
                "abundance": float(row[selected[int(i)]]),
            }
            for i in order
        ],
    }
