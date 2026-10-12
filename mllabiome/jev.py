from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


SCZ_QUESTION = (
    "Based ONLY on the listed gut microbiome genus relative abundances, "
    "is this sample more consistent with schizophrenia (label 1) than "
    "with a healthy control (label 0)? Treat this as an exploratory "
    "microbiome phenotype comparison, not a clinical diagnosis."
)


def _genus_name(name: str) -> str:

    match = re.search(r"(?:^|[|;])g__([^|;]+)", str(name))
    if match:
        return match.group(1).replace("_", " ")
    return str(name)


def _state(row: np.ndarray, names: Sequence[str], top_k: int) -> dict[str, Any]:
    total = float(np.sum(row))
    if total <= 0:
        raise ValueError("Cannot serialize an all-zero microbiome sample.")
    relative = row / total * 100.0

    abundant = np.flatnonzero(relative > 0)
    selected = abundant[np.argsort(-relative[abundant], kind="stable")[:top_k]]
    return {
        "assay": "stool shotgun metagenomics",
        "taxonomic_rank": "genus",
        "abundance_unit": "percent of measured genera (within-sample normalized)",
        "profile_truncated_to_top_k": int(top_k),
        "genera": [
            {
                "name": _genus_name(names[j]),
                "relative_abundance_percent": round(float(relative[j]), 4),
            }
            for j in selected
        ],
    }


def _jev_noul(state: dict[str, Any], *, model: str, instructions: str) -> float:

    if not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError(
            "Set TYPESAFE_API_KEY for live Jev calls (or use cached predictions)."
        )
    try:
        from typesafe_sdk import Noul, TypeSafeClient
    except ImportError as exc:
        raise ImportError(
            "Install the optional Jev SDK: pip install typesafe-sdk"
        ) from exc
    with TypeSafeClient() as client:
        response = client.system_one(
            model=model,
            state=state,
            questions={"scz": Noul(instructions=instructions)},
        )
        return float(response.nouls["scz"].noul)


class JevClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        model: str = "jev-latest",
        top_k: int = 20,
        cache_dir: str = "runs/jev_cache",
        question: str = SCZ_QUESTION,
    ) -> None:

        self.feature_names = feature_names
        self.model = model
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.question = question

    def _validated_X(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("JevClassifier expects an (n_samples, n_genera) matrix.")
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"JevClassifier feature-name mismatch: {array.shape[1]} columns "
                f"versus {len(self.feature_names)} names. Use genus + identity "
                "transformation without a feature filter."
            )
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("JevClassifier expects finite nonnegative raw abundances.")
        if (array.sum(axis=1) <= 0).any():
            raise ValueError("JevClassifier cannot process an all-zero profile.")
        return array

    def fit(self, X: Any, y: Any) -> "JevClassifier":
        array = self._validated_X(X)
        labels = np.unique(np.asarray(y))
        if labels.tolist() != [0, 1]:
            raise ValueError("JevClassifier currently requires training labels 0/1.")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = int(array.shape[1])
        return self

    def _probability(self, state: dict[str, Any]) -> float:
        payload = {"model": self.model, "state": state, "question": self.question}
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache = Path(self.cache_dir).expanduser()
        path = cache / f"{key}.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                score = float(json.load(handle)["probability"])
        else:
            score = _jev_noul(state, model=self.model, instructions=self.question)
            if not np.isfinite(score) or not (0.0 <= score <= 1.0):
                raise ValueError("Jev Noul returned an invalid probability.")
            cache.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=cache, suffix=".tmp", delete=False
            ) as handle:
                json.dump({"probability": score}, handle)
                temporary = Path(handle.name)
            os.replace(temporary, path)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Invalid cached Jev score.")
        return score

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["classes_", "n_features_in_"])
        array = self._validated_X(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimensions changed after fit.")
        scores = np.array(
            [
                self._probability(_state(row, self.feature_names, self.top_k))
                for row in array
            ],
            dtype=float,
        )
        return np.column_stack((1.0 - scores, scores))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
