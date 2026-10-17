from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class TabMDependencyError(ImportError):
    pass


class TabMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        device: str | None = None,
        random_state: int | None = None,
        val_metric_name: str | None = None,
        n_cv: int | None = None,
        n_refit: int | None = None,
        n_epochs: int | None = None,
        batch_size: int | None = None,
        n_threads: int | None = None,
        val_fraction: float | None = None,
        verbosity: int | None = None,
    ) -> None:
        self.device = device
        self.random_state = random_state
        self.val_metric_name = val_metric_name
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.val_fraction = val_fraction
        self.verbosity = verbosity

    @staticmethod
    def _backend_class():
        try:
            module = importlib.import_module("pytabkit")
        except ImportError as exc:
            raise TabMDependencyError(
                "TabMClassifier requires pytabkit. Install it with `uv sync --extra tabm`."
            ) from exc
        backend = getattr(module, "TabM_D_Classifier", None)
        if backend is None:
            raise TabMDependencyError(
                "The installed pytabkit package does not expose TabM_D_Classifier."
            )
        return backend

    def _backend_params(self) -> dict[str, Any]:
        names = (
            "device",
            "random_state",
            "val_metric_name",
            "n_cv",
            "n_refit",
            "n_epochs",
            "batch_size",
            "n_threads",
            "val_fraction",
            "verbosity",
        )
        return {
            name: getattr(self, name)
            for name in names
            if getattr(self, name) is not None
        }

    @staticmethod
    def _prepare_X(X: Any) -> Any:
        if sparse.issparse(X):
            bytes_required = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if bytes_required > 2 * 1024**3:
                raise MemoryError("Densifying the TabM feature matrix would exceed 2 GiB.")
            X = X.astype(np.float32).toarray()
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) != 2 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional feature matrix.")
        if isinstance(X, np.ndarray) and X.dtype.kind in "iuf" and np.isinf(X).any():
            raise ValueError("X contains infinite values.")
        if hasattr(X, "columns") and not X.columns.is_unique:
            raise ValueError("TabM feature names must be unique.")
        return X

    def fit(self, X: Any, y: Any, groups: Any = None):
        X = self._prepare_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("y must contain one label per training sample.")
        if groups is not None and np.asarray(groups).shape != (len(y),):
            raise ValueError("groups must contain one value per training sample.")
        encoder = LabelEncoder().fit(y)
        if len(encoder.classes_) < 2:
            raise ValueError("TabM requires at least two classes.")
        model = self._backend_class()(**self._backend_params())
        model.fit(X, encoder.transform(y))
        self.model_ = model
        self.classes_ = encoder.classes_
        self.n_features_in_ = int(X.shape[1])
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("model_", "classes_", "n_features_in_"))
        X = self._prepare_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError("Prediction feature names must match training features.")
        proba = np.asarray(self.model_.predict_proba(X), dtype=np.float64)
        if proba.shape != (X.shape[0], len(self.classes_)):
            raise ValueError(f"TabM returned an invalid probability shape: {proba.shape}.")
        if not np.isfinite(proba).all() or (proba < 0).any():
            raise ValueError("TabM returned invalid class probabilities.")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("TabM class probabilities do not sum to one.")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
