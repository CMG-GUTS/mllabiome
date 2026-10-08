from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesRegressor

from .transformations import TransformationCoordinate
from .utils import _as_float_matrix


def _simplex_project(X: np.ndarray) -> np.ndarray:
    arr = _as_float_matrix(X).astype(float, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0.0] = 0.0
    totals = arr.sum(axis=1, keepdims=True)
    positive = totals[:, 0] > 0.0
    if np.any(positive):
        arr[positive] = arr[positive] / totals[positive]
    return arr


class TaxonSpaceProbabilityProxy(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 750,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = float(max_features)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, teacher_proba: np.ndarray):
        X = _simplex_project(X)
        y = np.asarray(teacher_proba, dtype=float)
        if y.ndim != 2 or y.shape[0] != X.shape[0] or y.shape[1] < 2:
            raise ValueError(
                "teacher_proba must be n_samples × n_classes with at least two classes."
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("teacher_proba contains non-finite values.")
        self.model_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model_.fit(X, y)
        self.classes_ = np.arange(y.shape[1], dtype=int)
        self.n_features_in_ = int(X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TaxonSpaceProbabilityProxy has not been fitted.")
        X = _simplex_project(X)
        pred = np.asarray(self.model_.predict(X), dtype=float)
        if pred.ndim == 1:
            pred = pred[:, None]
        pred = np.clip(pred, 0.0, 1.0)
        totals = pred.sum(axis=1, keepdims=True)
        invalid = totals[:, 0] <= 0.0
        if np.any(invalid):
            pred[invalid] = 1.0 / pred.shape[1]
            totals = pred.sum(axis=1, keepdims=True)
        return pred / totals

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _safe_corr(fn: Any, a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    try:
        value = fn(x, y)
        return float(value.statistic if hasattr(value, "statistic") else value[0])
    except Exception:
        return float("nan")


def proxy_fidelity_metrics(
    teacher_proba: np.ndarray,
    proxy_proba: np.ndarray,
) -> dict[str, float]:
    teacher = np.asarray(teacher_proba, dtype=float)
    proxy = np.asarray(proxy_proba, dtype=float)
    if teacher.shape != proxy.shape or teacher.ndim != 2:
        raise ValueError(
            "Teacher and proxy probability matrices must have identical 2D shapes."
        )
    diff = proxy - teacher
    flat_teacher = teacher.ravel()
    flat_proxy = proxy.ravel()
    denom = float(np.sum((flat_teacher - float(np.mean(flat_teacher))) ** 2))
    r2 = float("nan")
    if denom > 0.0:
        r2 = 1.0 - float(np.sum((flat_proxy - flat_teacher) ** 2)) / denom
    return {
        "probability_mae": float(np.mean(np.abs(diff))),
        "probability_rmse": float(np.sqrt(np.mean(diff**2))),
        "probability_max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "probability_r2": float(r2),
        "probability_pearson": _safe_corr(pearsonr, flat_teacher, flat_proxy),
        "probability_spearman": _safe_corr(spearmanr, flat_teacher, flat_proxy),
        "predicted_class_agreement": float(
            np.mean(np.argmax(teacher, axis=1) == np.argmax(proxy, axis=1))
        ),
    }


def taxon_coordinate_metadata(
    feature_names: Sequence[str],
) -> list[TransformationCoordinate]:
    return [
        TransformationCoordinate(
            name=str(name),
            coordinate_type="taxon_proxy_input",
            anchor_feature=str(name),
            components=(str(name),),
            coefficients=(1.0,),
            exact_feature_identity=True,
        )
        for name in feature_names
    ]


def fit_taxon_probability_proxy(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    teacher_train_proba: np.ndarray,
    teacher_test_proba: np.ndarray,
    feature_names: Sequence[str],
    *,
    random_state: int,
) -> dict[str, Any]:
    X_train = _simplex_project(X_train_raw)
    X_test = _simplex_project(X_test_raw)
    proxy = TaxonSpaceProbabilityProxy(random_state=int(random_state)).fit(
        X_train, teacher_train_proba
    )
    proxy_test_proba = proxy.predict_proba(X_test)
    fidelity = proxy_fidelity_metrics(teacher_test_proba, proxy_test_proba)
    return {
        "estimator": proxy,
        "X_train": X_train,
        "X_test": X_test,
        "feature_names": [str(x) for x in feature_names],
        "coordinate_metadata": taxon_coordinate_metadata(feature_names),
        "input_projector": _simplex_project,
        "perturbation_geometry": "simplex_relative_abundance_proxy",
        "proxy_proba": proxy_test_proba,
        "proxy_fidelity": fidelity,
    }


def regression_proxy_fidelity_metrics(
    teacher_prediction: np.ndarray,
    proxy_prediction: np.ndarray,
) -> dict[str, float]:
    teacher = np.asarray(teacher_prediction, dtype=float).reshape(-1)
    proxy = np.asarray(proxy_prediction, dtype=float).reshape(-1)
    if teacher.shape != proxy.shape:
        raise ValueError(
            "Teacher and proxy regression predictions must have identical shapes."
        )
    diff = proxy - teacher
    denom = float(np.sum((teacher - float(np.mean(teacher))) ** 2))
    r2 = float("nan")
    if denom > 0.0:
        r2 = 1.0 - float(np.sum(diff**2)) / denom
    return {
        "prediction_mae": float(np.mean(np.abs(diff))),
        "prediction_rmse": float(np.sqrt(np.mean(diff**2))),
        "prediction_max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "prediction_r2": float(r2),
        "prediction_pearson": _safe_corr(pearsonr, teacher, proxy),
        "prediction_spearman": _safe_corr(spearmanr, teacher, proxy),
    }


def fit_taxon_regression_proxy(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    teacher_train_prediction: np.ndarray,
    teacher_test_prediction: np.ndarray,
    feature_names: Sequence[str],
    *,
    random_state: int,
) -> dict[str, Any]:
    X_train = _simplex_project(X_train_raw)
    X_test = _simplex_project(X_test_raw)
    model = ExtraTreesRegressor(
        n_estimators=750,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=int(random_state),
        n_jobs=1,
    )
    model.fit(X_train, np.asarray(teacher_train_prediction, dtype=float).reshape(-1))
    proxy_test_prediction = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    fidelity = regression_proxy_fidelity_metrics(
        teacher_test_prediction, proxy_test_prediction
    )
    return {
        "estimator": model,
        "X_train": X_train,
        "X_test": X_test,
        "feature_names": [str(x) for x in feature_names],
        "coordinate_metadata": taxon_coordinate_metadata(feature_names),
        "input_projector": _simplex_project,
        "perturbation_geometry": "simplex_relative_abundance_proxy",
        "proxy_prediction": proxy_test_prediction,
        "proxy_fidelity": fidelity,
    }
