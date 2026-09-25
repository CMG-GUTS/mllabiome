from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator

from .metrics import _predict_proba_aligned
from .utils import _as_float_matrix


class _FittedMpmaEnsemble(BaseEstimator):
    def __init__(self, members: list[dict[str, Any]], aggregation: str = "mean_proba"):
        self.members = members
        self.aggregation = aggregation
        self.classes_ = (
            np.asarray(members[0]["classes"], dtype=int)
            if members
            else np.array([0, 1], dtype=int)
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_float_matrix(X)
        probs = []
        for member in self.members:
            sl = member["slice"]
            clf = member["estimator"]
            probs.append(_predict_proba_aligned(clf, X[:, sl], self.classes_))
        stack = np.stack(probs, axis=0)
        return _aggregate_member_proba(stack, self.aggregation)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _aggregate_member_proba(stack: np.ndarray, aggregation: str) -> np.ndarray:
    aggregation = str(aggregation or "mean_proba")
    if aggregation == "median_proba":
        proba = np.median(stack, axis=0)
    elif aggregation == "rank_mean":
        ranks = np.empty_like(stack, dtype=float)
        for m in range(stack.shape[0]):
            for i in range(stack.shape[1]):
                ranks[m, i, :] = rankdata(stack[m, i, :], method="average")
        proba = ranks.mean(axis=0)
    elif aggregation == "majority_vote":
        votes = np.argmax(stack, axis=2)
        proba = np.zeros((stack.shape[1], stack.shape[2]), dtype=float)
        for i in range(votes.shape[1]):
            counts = np.bincount(votes[:, i], minlength=stack.shape[2]).astype(float)
            proba[i] = counts / max(float(counts.sum()), 1.0)
    else:
        proba = stack.mean(axis=0)
    proba = np.asarray(proba, dtype=float)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return proba / row_sums


def _explain_predict_proba(
    clf: BaseEstimator, classes: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X_: _predict_proba_aligned(clf, _as_float_matrix(X_), classes)


def _explain_predict_class_probability(
    clf: BaseEstimator, classes: np.ndarray, class_index: int
) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X_: _predict_proba_aligned(clf, _as_float_matrix(X_), classes)[
        :, int(class_index)
    ]


def _sample_rows(X: np.ndarray, *, max_rows: int, random_state: int) -> np.ndarray:
    X = _as_float_matrix(X)
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return np.arange(X.shape[0])
    rng = np.random.default_rng(int(random_state))
    return np.sort(rng.choice(np.arange(X.shape[0]), size=int(max_rows), replace=False))


def _projection_required(geometry: Any) -> bool:
    text = str(geometry).casefold()
    return any(
        token in text for token in ("simplex", "sphere", "clr", "rank", "binary")
    )


def _project_input(X: np.ndarray, input_projector: Any | None) -> np.ndarray:
    arr = _as_float_matrix(X)
    if input_projector is None:
        return arr
    projected = input_projector(arr)
    return _as_float_matrix(projected)
