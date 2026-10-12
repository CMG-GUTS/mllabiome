from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

SCZ_QUESTION = (
    "Based ONLY on the listed gut microbiome taxon relative abundances, "
    "is this sample more consistent with schizophrenia (label 1) than "
    "with a healthy control (label 0)? Treat this as an exploratory "
    "microbiome phenotype comparison, not a clinical diagnosis."
)

PRESENCE_QUESTION = (
    "Based ONLY on the detected (present) versus undetected (absent) gut "
    "microbiome taxa, is this sample more consistent with schizophrenia "
    "(label 1) than with a control (label 0)? The listed taxa form a fixed "
    "reference panel, and absence is meaningful. Do not infer abundance "
    "from the binary statuses. This is an exploratory comparison, not a diagnosis."
)


def _taxon_name(name: str) -> str:
    lineage = re.findall(r"(?:^|[|;])([a-z]__[^|;]+)", str(name))
    if not lineage:
        return str(name)
    selected = [token for token in lineage if token.startswith(("g__", "s__", "t__"))]
    if not selected:
        selected = [lineage[-1]]
    if len(selected) == 1 and selected[0].startswith("g__"):
        return selected[0][3:].replace("_", " ")
    return " | ".join(selected)


def _state(
    row: np.ndarray, names: Sequence[str], top_k: int, *, state_format: str = "json"
) -> dict[str, Any] | str:
    total = float(np.sum(row))
    if total <= 0:
        raise ValueError("Cannot serialize an all-zero microbiome sample.")
    relative = row / total * 100.0
    abundant = np.flatnonzero(relative > 0)
    selected = abundant[np.argsort(-relative[abundant], kind="stable")[:top_k]]
    taxa = [
        {
            "name": _taxon_name(names[j]),
            "relative_abundance_percent": round(float(relative[j]), 4),
        }
        for j in selected
    ]
    if state_format == "json":
        return {
            "assay": "stool shotgun metagenomics",
            "taxonomic_rank": "as_provided",
            "abundance_unit": "percent of all measured taxonomic features",
            "profile_truncated_to_top_k": int(top_k),
            "taxa": taxa,
        }
    if state_format == "text":
        lines = [
            "Gut microbiome sample from stool shotgun metagenomics.",
            "The values below are percent of ALL measured taxonomic abundances,",
            f"not just the displayed top {top_k}. Other taxa are omitted.",
            "Most abundant taxa:",
        ]
        lines += [
            f"{entry['name']}: {entry['relative_abundance_percent']:.4f}%"
            for entry in taxa
        ]
        return "\n".join(lines)
    raise ValueError("state_format must be 'json' or 'text'.")


def _presence_state(
    row: np.ndarray,
    names: Sequence[str],
    panel_indices: Sequence[int],
    *,
    threshold_pct: float,
    state_format: str = "json",
) -> dict[str, Any] | str:
    total = float(np.sum(row))
    relative_pct = row / total * 100.0 if total > 0 else np.zeros_like(row)
    entries = [
        {
            "name": _taxon_name(names[j]),
            "present": bool(relative_pct[j] > threshold_pct),
        }
        for j in panel_indices
    ]
    if state_format == "json":
        return {
            "assay": "stool shotgun metagenomics",
            "taxonomic_rank": "as_provided",
            "representation": "presence_absence",
            "presence_threshold_percent": float(threshold_pct),
            "panel_selection": "top taxon prevalence in training fold, no label use",
            "taxa": entries,
        }
    if state_format == "text":
        lines = [
            "Stool shotgun metagenomics taxon detection/non-detection.",
            "This is a fixed taxon panel selected using training-sample prevalence.",
            f"Present means > {threshold_pct:g}% relative abundance of all measured genera.",
            "Absent taxa are explicitly included. No abundance values are given.",
        ]
        lines.extend(
            f"{v['name']}: {'present' if v['present'] else 'absent'}" for v in entries
        )
        return "\n".join(lines)
    raise ValueError("state_format must be 'json' or 'text'.")


def _local_endpoint(base_url: str) -> str:
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
            "OllamaDecisionClassifier only permits http://127.0.0.1:<port>. "
            "Remote inference URLs are intentionally disabled."
        )
    return f"http://127.0.0.1:{parsed.port}/v1/systemone"


class _BlockRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _post_systemone(
    state: dict[str, Any] | str,
    *,
    model: str,
    question: dict[str, Any],
    base_url: str,
) -> dict[str, Any]:
    url = _local_endpoint(base_url)
    if not model or model.endswith(":cloud") or model.endswith("-cloud"):
        raise ValueError(
            "Select a locally downloaded Ollama decision model, not a cloud model."
        )
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {
                "model": model,
                "state": state,
                "questions": {"scz": question},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _BlockRedirects()
    )
    try:
        with opener.open(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        try:
            detail = str(json.loads(detail).get("error", detail))
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(
            f"Local Ollama {model!r} returned HTTP {exc.code}: "
            f"{detail[:1500] or exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Cannot reach local Ollama {model!r} at {url}: {type(exc).__name__}: {exc}"
        ) from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Local Ollama {model!r} returned an invalid JSON response: {exc}"
        ) from exc
    try:
        answer = payload["answers"]["scz"]
        if not isinstance(answer, dict):
            raise TypeError("Malformed answer object")
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Unexpected response from local Ollama /v1/systemone."
        ) from exc
    return answer


def _checked_probability(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Decision probability must be numeric, not Boolean.")
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid decision probability in Ollama response.") from exc
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Local decision model returned an invalid score.")
    return probability


def _decision_noul(
    state: dict[str, Any] | str, *, model: str, instructions: str, base_url: str
) -> float:
    answer = _post_systemone(
        state,
        model=model,
        question={"type": "noul", "instructions": instructions},
        base_url=base_url,
    )
    try:
        return _checked_probability(answer["noul"])
    except KeyError as exc:
        raise ValueError("Expected a noul answer from local Ollama.") from exc


def _decision_choice(
    state: dict[str, Any] | str, *, model: str, instructions: str, base_url: str
) -> float:
    answer = _post_systemone(
        state,
        model=model,
        question={
            "type": "choice",
            "instructions": instructions,
            "criteria": {
                "control": "Profile from a participant without schizophrenia (control).",
                "scz": "Profile from a participant with schizophrenia (case).",
            },
        },
        base_url=base_url,
    )
    try:
        return _checked_probability(answer["probabilities"]["scz"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Expected choice probabilities for 'scz' from Ollama."
        ) from exc


class OllamaDecisionClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        feature_names: tuple[str, ...] | None = None,
        model: str = "tev1:4b",
        top_k: int = 20,
        cache_dir: str = "runs/ollama_decision_cache",
        question: str | None = None,
        base_url: str = "http://127.0.0.1:11434",
        state_format: str = "json",
        question_type: str = "noul",
        representation: str = "abundance",
        presence_threshold_pct: float = 0.0,
    ) -> None:
        self.feature_names = feature_names
        self.model = model
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.question = question
        self.base_url = base_url
        self.state_format = state_format
        self.question_type = question_type
        self.representation = representation
        self.presence_threshold_pct = presence_threshold_pct

    def bind_schema(self, names: Sequence[str]) -> "OllamaDecisionClassifier":
        self._bound_feature_names = tuple(str(name) for name in names)
        return self

    def _validated_X(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("Expected (n_samples, n_genera) numeric matrix.")
        names = getattr(self, "feature_names_", None)
        if names is None:
            names = self.feature_names
        if names is None:
            names = getattr(self, "_bound_feature_names", None)
        if names is None:
            raise ValueError("Feature names are required for Ollama decision models.")
        if array.shape[1] != len(names):
            raise ValueError(
                f"Feature-name mismatch: {array.shape[1]} columns vs {len(names)} names."
            )
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("Expected finite nonnegative raw genus abundances.")
        if (array.sum(axis=1) <= 0).any():
            raise ValueError("Cannot process an all-zero microbiome profile.")
        return array

    def fit(self, X: Any, y: Any) -> "OllamaDecisionClassifier":
        _local_endpoint(self.base_url)
        if (
            not self.model
            or self.model.endswith(":cloud")
            or self.model.endswith("-cloud")
        ):
            raise ValueError("Select a locally downloaded Ollama decision model.")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if self.state_format not in ("json", "text"):
            raise ValueError("state_format must be 'json' or 'text'.")
        if self.question_type not in ("noul", "choice"):
            raise ValueError("question_type must be 'noul' or 'choice'.")
        if self.representation not in ("abundance", "presence_absence"):
            raise ValueError(
                "representation must be 'abundance' or 'presence_absence'."
            )
        if (
            not np.isfinite(self.presence_threshold_pct)
            or not 0 <= self.presence_threshold_pct < 100
        ):
            raise ValueError("presence_threshold_pct must be >= 0 and < 100.")
        if self.feature_names is not None:
            names = tuple(str(name) for name in self.feature_names)
        elif hasattr(self, "_bound_feature_names"):
            names = self._bound_feature_names
        elif hasattr(X, "columns"):
            names = tuple(str(name) for name in X.columns)
        else:
            raise ValueError(
                "Feature names must be bound by mllabiome or provided explicitly."
            )
        self.feature_names_ = names
        array = self._validated_X(X)
        labels = np.unique(np.asarray(y))
        if labels.tolist() != [0, 1]:
            raise ValueError("Training labels must contain both classes 0 and 1.")
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = int(array.shape[1])
        if self.representation == "presence_absence":
            percentages = array / array.sum(axis=1, keepdims=True) * 100.0
            prevalence = (percentages > self.presence_threshold_pct).mean(axis=0)
            ranked = sorted(
                range(array.shape[1]),
                key=lambda j: (-prevalence[j], _taxon_name(self.feature_names_[j]), j),
            )
            self.panel_indices_ = np.asarray(ranked[: self.top_k], dtype=int)
        return self

    def _effective_question(self) -> str:
        if self.question is not None:
            return self.question
        return (
            PRESENCE_QUESTION
            if self.representation == "presence_absence"
            else SCZ_QUESTION
        )

    def _score(self, state: dict[str, Any] | str) -> float:

        payload = {
            "model": self.model,
            "state": state,
            "question": self._effective_question(),
            "question_type": self.question_type,
            "state_format": self.state_format,
            "representation": self.representation,
            "presence_threshold_pct": self.presence_threshold_pct,
            "base_url": self.base_url,
            "api_version": "systemone-v2",
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        cache_dir = Path(self.cache_dir).expanduser()
        cache_file = cache_dir / f"{digest}.json"
        if cache_file.is_file():
            with cache_file.open("r", encoding="utf-8") as stream:
                score = float(json.load(stream)["score"])
        else:
            inference = (
                _decision_noul if self.question_type == "noul" else _decision_choice
            )
            score = inference(
                state,
                model=self.model,
                instructions=self._effective_question(),
                base_url=self.base_url,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=cache_dir, suffix=".tmp", delete=False
            ) as handle:
                json.dump({"score": score}, handle)
                temporary = Path(handle.name)
            os.replace(temporary, cache_file)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid cached local Ollama decision score.")
        return score

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["classes_", "n_features_in_"])
        array = self._validated_X(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimensions changed after fit.")
        if self.representation == "abundance":
            states = (
                _state(
                    row, self.feature_names_, self.top_k, state_format=self.state_format
                )
                for row in array
            )
        else:
            check_is_fitted(self, ["panel_indices_"])
            states = (
                _presence_state(
                    row,
                    self.feature_names_,
                    self.panel_indices_,
                    threshold_pct=self.presence_threshold_pct,
                    state_format=self.state_format,
                )
                for row in array
            )
        scores = np.asarray([self._score(state) for state in states], dtype=float)
        return np.column_stack((1.0 - scores, scores))

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
