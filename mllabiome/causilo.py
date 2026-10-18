from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class CausiloDependencyError(ImportError):
    pass


class CausiloClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        n_estimators: int = 8,
        device: str = "auto",
        use_kv_cache: bool = False,
        retain_preprocessing: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.device = device
        self.use_kv_cache = use_kv_cache
        self.retain_preprocessing = retain_preprocessing
        self.random_state = random_state

    @staticmethod
    def _backend_class():
        try:
            module = importlib.import_module("causilo")
        except ImportError as exc:
            raise CausiloDependencyError(
                "CausiloClassifier requires causilo>=1.0.3,<2. "
                "Install it with `pip install 'causilo>=1.0.3,<2'`."
            ) from exc
        backend = getattr(module, "CausiloClassifier", None)
        if backend is None:
            raise CausiloDependencyError(
                "The installed causilo package does not expose CausiloClassifier."
            )
        return backend

    @staticmethod
    def _prepare_X(X: Any):
        if sparse.issparse(X):
            required_bytes = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if required_bytes > 2 * 1024**3:
                raise MemoryError("Densifying the Causilo matrix would exceed 2 GiB.")
            X = X.astype(np.float32).toarray()
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if isinstance(X, np.ndarray) and X.dtype.kind in "iuf":
            if np.isinf(X).any():
                raise ValueError("X contains infinite feature values.")
        return X

    def fit(self, X: Any, y: Any, groups: Any = None):
        if (
            isinstance(self.n_estimators, (bool, np.bool_))
            or not isinstance(self.n_estimators, (int, np.integer))
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")
        if (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, (int, np.integer))
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer.")
        X = self._prepare_X(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional target array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if groups is not None and np.asarray(groups).shape != (y.shape[0],):
            raise ValueError("groups must contain one value per training sample.")
        encoder = LabelEncoder().fit(y)
        if len(encoder.classes_) < 2:
            raise ValueError("CausiloClassifier requires at least two classes.")
        backend = self._backend_class()
        model = backend(
            n_estimators=int(self.n_estimators),
            device=self.device,
            use_kv_cache=self.use_kv_cache,
            retain_preprocessing=self.retain_preprocessing,
            random_state=int(self.random_state),
        )
        model.fit(X, encoder.transform(y))
        self.model_ = model
        self.classes_ = encoder.classes_
        self.n_features_in_ = X.shape[1]
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _validate_predict_X(self, X: Any):
        check_is_fitted(self, ("model_", "classes_", "n_features_in_"))
        X = self._prepare_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError(
                    "Prediction feature names must match training features."
                )
        return X

    def predict_proba(self, X: Any) -> np.ndarray:
        X = self._validate_predict_X(X)
        probabilities = np.asarray(self.model_.predict_proba(X), dtype=np.float64)
        expected_shape = (X.shape[0], len(self.classes_))
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"Expected probability shape {expected_shape}, got {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Causilo produced invalid class probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Causilo class probabilities do not sum to one.")
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
