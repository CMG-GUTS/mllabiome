from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class TabDPTDependencyError(ImportError):
    pass


class TabDPTClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        n_ensembles: int = 8,
        context_size: int | None = None,
        context_reduction: str = "subsample",
        feature_reduction: str = "pca",
        normalizer: str | None = "standard",
        missing_indicators: bool = False,
        clip_sigma: float = 8.0,
        faiss_metric: str = "l2",
        temperature: float = 1.0,
        batch_size: int | None = None,
        permute_classes: bool = True,
        model_weight_path: str | None = None,
        device: str | None = None,
        use_flash: bool = True,
        compile: bool = True,
        random_state: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.n_ensembles = n_ensembles
        self.context_size = context_size
        self.context_reduction = context_reduction
        self.feature_reduction = feature_reduction
        self.normalizer = normalizer
        self.missing_indicators = missing_indicators
        self.clip_sigma = clip_sigma
        self.faiss_metric = faiss_metric
        self.temperature = temperature
        self.batch_size = batch_size
        self.permute_classes = permute_classes
        self.model_weight_path = model_weight_path
        self.device = device
        self.use_flash = use_flash
        self.compile = compile
        self.random_state = random_state
        self.verbose = verbose

    def _backend_class(self):
        try:
            module = importlib.import_module("tabdpt")
        except ImportError as exc:
            raise TabDPTDependencyError(
                "TabDPTClassifier requires tabdpt>=1.3,<1.4. Install it with "
                "`pip install 'tabdpt>=1.3,<1.4'`."
            ) from exc
        backend = getattr(module, "TabDPTClassifier", None)
        if backend is None:
            raise TabDPTDependencyError(
                "The installed tabdpt package does not expose TabDPTClassifier. "
                "Install tabdpt>=1.3,<1.4."
            )
        return backend

    def _backend_params(self) -> dict[str, Any]:
        return {
            "normalizer": self.normalizer,
            "missing_indicators": self.missing_indicators,
            "clip_sigma": self.clip_sigma,
            "feature_reduction": self.feature_reduction,
            "context_reduction": self.context_reduction,
            "faiss_metric": self.faiss_metric,
            "device": self.device,
            "use_flash": self.use_flash,
            "compile": self.compile,
            "model_weight_path": self.model_weight_path,
            "verbose": self.verbose,
        }

    @staticmethod
    def _as_array(X: Any) -> np.ndarray:
        if sparse.issparse(X):
            bytes_required = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if bytes_required > 2 * 1024**3:
                raise MemoryError("Densifying the TabDPT matrix would exceed 2 GiB.")
            X = X.astype(np.float32).toarray()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if np.isinf(X).any():
            raise ValueError("X contains infinite feature values.")
        return X

    def fit(self, X: Any, y: Any, groups: Any = None):
        if not isinstance(self.n_ensembles, int) or self.n_ensembles < 1:
            raise ValueError("n_ensembles must be a positive integer.")
        if self.context_size is not None and self.context_size < 1:
            raise ValueError("context_size must be positive or None.")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive or None.")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be a finite positive number.")
        X_array = self._as_array(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            y_array = y_array.reshape(-1)
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if groups is not None and len(np.asarray(groups)) != len(y_array):
            raise ValueError("groups must contain one value per training sample.")
        classes, encoded_y = np.unique(y_array, return_inverse=True)
        if classes.size < 2:
            raise ValueError("TabDPTClassifier requires at least two classes.")
        backend = self._backend_class()
        self.model_ = backend(**self._backend_params())
        if (
            self.random_state is not None
            and self.feature_reduction == "pca"
            and X_array.shape[1] > self.model_.max_features
        ):
            import torch

            with torch.random.fork_rng():
                torch.manual_seed(int(self.random_state))
                self.model_.fit(X_array, encoded_y.astype(np.int64))
        else:
            self.model_.fit(X_array, encoded_y.astype(np.int64))
        self.classes_ = classes
        self.n_features_in_ = X_array.shape[1]
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("model_", "classes_", "n_features_in_"))
        X_array = self._as_array(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_array.shape[1]}."
            )
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError(
                    "Prediction feature names must match the training features."
                )
        proba = self.model_.ensemble_predict_proba(
            X_array,
            n_ensembles=self.n_ensembles,
            temperature=self.temperature,
            context_size=self.context_size,
            batch_size=self.batch_size,
            permute_classes=self.permute_classes,
            seed=self.random_state,
        )
        proba = np.asarray(proba, dtype=np.float64)
        if proba.shape != (len(X_array), len(self.classes_)):
            raise ValueError(f"Unexpected TabDPT probability shape: {proba.shape}.")
        if not np.isfinite(proba).all() or np.any(proba < 0):
            raise ValueError("TabDPT produced invalid class probabilities.")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("TabDPT class probabilities do not sum to one.")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
