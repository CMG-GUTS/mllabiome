from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class TabICLDependencyError(ImportError):
    pass


class TabICLClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        n_estimators: int | None = None,
        norm_methods: Any = None,
        feat_shuffle_method: str | None = None,
        class_shuffle_method: str | None = None,
        outlier_threshold: float | None = None,
        softmax_temperature: float | None = None,
        average_logits: bool | None = None,
        support_many_classes: bool | None = None,
        batch_size: int | None = None,
        kv_cache: bool | None = None,
        model_path: str | None = None,
        allow_auto_download: bool | None = None,
        checkpoint_version: str | None = None,
        device: str | None = None,
        use_amp: str | bool | None = None,
        use_fa3: str | bool | None = None,
        offload_mode: str | None = None,
        disk_offload_dir: str | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def _backend_class():
        try:
            module = importlib.import_module("tabicl")
        except ImportError as exc:
            raise TabICLDependencyError(
                "TabICLClassifier requires tabicl. Install it with `uv sync --extra tabicl`."
            ) from exc
        backend = getattr(module, "TabICLClassifier", None)
        if backend is None:
            raise TabICLDependencyError(
                "The installed tabicl package does not expose TabICLClassifier."
            )
        return backend

    def _backend_params(self) -> dict[str, Any]:
        names = (
            "n_estimators",
            "norm_methods",
            "feat_shuffle_method",
            "class_shuffle_method",
            "outlier_threshold",
            "softmax_temperature",
            "average_logits",
            "support_many_classes",
            "batch_size",
            "kv_cache",
            "model_path",
            "allow_auto_download",
            "checkpoint_version",
            "device",
            "use_amp",
            "use_fa3",
            "offload_mode",
            "disk_offload_dir",
            "random_state",
            "n_jobs",
            "verbose",
        )
        return {
            name: getattr(self, name)
            for name in names
            if getattr(self, name) is not None
        }

    @staticmethod
    def _prepare_X(X: Any):
        if sparse.issparse(X):
            bytes_required = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if bytes_required > 2 * 1024**3:
                raise MemoryError(
                    "Densifying the TabICL feature matrix would exceed 2 GiB."
                )
            X = X.astype(np.float32).toarray()
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) != 2 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional feature matrix.")
        if isinstance(X, np.ndarray) and X.dtype.kind in "iuf" and np.isinf(X).any():
            raise ValueError("X contains infinite values.")
        return X

    def fit(self, X: Any, y: Any, groups: Any = None):
        X = self._prepare_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("y must contain one label for each training row.")
        if groups is not None and np.asarray(groups).shape != (y.shape[0],):
            raise ValueError("groups must contain one value for each training row.")
        encoder = LabelEncoder().fit(y)
        if len(encoder.classes_) < 2:
            raise ValueError("TabICL requires at least two classes.")
        model = self._backend_class()(**self._backend_params())
        encoded = encoder.transform(y)
        model.fit(X, encoded)
        self.model_ = model
        self.X_train_ = X.copy()
        self.y_train_ = encoded
        self.classes_ = encoder.classes_
        self.n_features_in_ = int(X.shape[1])
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _get_model(self):
        model = getattr(self, "model_", None)
        if model is None:
            model = self._backend_class()(**self._backend_params())
            model.fit(self.X_train_, self.y_train_)
            self.model_ = model
        return model

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("classes_", "n_features_in_", "X_train_", "y_train_"))
        X = self._prepare_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError(
                    "Prediction feature names must match training feature names."
                )
        proba = np.asarray(self._get_model().predict_proba(X), dtype=np.float64)
        shape = (X.shape[0], len(self.classes_))
        if proba.shape != shape or not np.isfinite(proba).all() or (proba < 0).any():
            raise ValueError("TabICL returned invalid class probabilities.")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("TabICL class probabilities do not sum to one.")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state
