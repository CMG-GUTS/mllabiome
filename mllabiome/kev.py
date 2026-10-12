from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

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
    return match.group(1).replace("_", " ") if match else str(name)


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


def _loopback_endpoint(base_url: str) -> str:

    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.port is None
    ):
        raise ValueError(
            "LocalKevClassifier only permits http://127.0.0.1:<port>. "
            "Remote model endpoints are intentionally disabled."
        )
    return f"http://127.0.0.1:{parsed.port}/v1/systemone"


def _kev_noul(
    state: dict[str, Any], *, model: str, instructions: str, base_url: str
) -> float:

    url = _loopback_endpoint(base_url)
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {
                "model": model,
                "state": state,
                "questions": {"scz": {"type": "noul", "instructions": instructions}},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=90) as response:
            payload = json.load(response)
    except OSError as exc:
        raise RuntimeError(
            "Could not reach the LOCAL Kev server at 127.0.0.1. "
            "Start kev.serve before running the benchmark."
        ) from exc
    try:
        value = float(payload["answers"]["scz"]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Unexpected local Kev /v1/systemone response format.") from exc
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Local Kev returned an invalid probability.")
    return value


class LocalKevClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        model: str = "kev-latest",
        top_k: int = 20,
        cache_dir: str = "runs/local_kev_cache",
        question: str = SCZ_QUESTION,
        base_url: str = "http://127.0.0.1:8009",
    ) -> None:
        self.feature_names = feature_names
        self.model = model
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.question = question
        self.base_url = base_url

    def _validated_X(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("LocalKevClassifier expects (n_samples, n_genera).")
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature-name mismatch: {array.shape[1]} columns vs "
                f"{len(self.feature_names)} names. Use genus + identity, no feature filter."
            )
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Expected finite nonnegative raw abundances.")
        if (array.sum(axis=1) <= 0).any():
            raise ValueError("Cannot process an all-zero microbiome profile.")
        return array

    def fit(self, X: Any, y: Any) -> "LocalKevClassifier":
        _loopback_endpoint(self.base_url)
        array = self._validated_X(X)
        labels = np.unique(np.asarray(y))
        if labels.tolist() != [0, 1]:
            raise ValueError("Training labels must contain both classes 0 and 1.")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = int(array.shape[1])
        return self

    def _probability(self, state: dict[str, Any]) -> float:
        payload = {
            "model": self.model,
            "state": state,
            "question": self.question,
            "base_url": self.base_url,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_dir = Path(self.cache_dir).expanduser()
        cache_file = cache_dir / f"{digest}.json"
        if cache_file.is_file():
            with cache_file.open("r", encoding="utf-8") as stream:
                score = float(json.load(stream)["probability"])
        else:
            score = _kev_noul(
                state,
                model=self.model,
                instructions=self.question,
                base_url=self.base_url,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=cache_dir, suffix=".tmp", delete=False
            ) as handle:
                json.dump({"probability": score}, handle)
                temporary = Path(handle.name)
            os.replace(temporary, cache_file)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid cached local Kev score.")
        return score

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["classes_", "n_features_in_"])
        array = self._validated_X(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimensions changed after fit.")
        scores = np.asarray(
            [
                self._probability(_state(row, self.feature_names, self.top_k))
                for row in array
            ],
            dtype=float,
        )
        return np.column_stack((1.0 - scores, scores))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
