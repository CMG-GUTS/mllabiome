from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class EXAONETabularDependencyError(ImportError):
    pass


class EXAONETabularClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        device: str | None = None,
        random_state: int | None = None,
        ensemble_count: int | None = None,
        compute_dtype: str | None = None,
        weights: str | None = None,
        revision: str | None = None,
        max_vram_bytes: int | None = None,
    ) -> None:
        self.device = device
        self.random_state = random_state
        self.ensemble_count = ensemble_count
        self.compute_dtype = compute_dtype
        self.weights = weights
        self.revision = revision
        self.max_vram_bytes = max_vram_bytes

    @staticmethod
    def _prepare_X(X: Any) -> np.ndarray:
        if hasattr(X, "columns") and not X.columns.is_unique:
            raise ValueError("EXAONE feature names must be unique.")
        if sparse.issparse(X):
            bytes_required = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if bytes_required > 2 * 1024**3:
                raise MemoryError(
                    "Densifying the EXAONE feature matrix would exceed 2 GiB."
                )
            X = X.astype(np.float32).toarray()
        try:
            array = np.asarray(X, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise ValueError("EXAONE requires numeric feature values.") from exc
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional feature matrix.")
        if np.isinf(array).any():
            raise ValueError("X contains infinite feature values.")
        return np.ascontiguousarray(array)

    def _pretrained_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        device = self.device
        if device == "auto":
            try:
                torch = importlib.import_module("torch")
            except ImportError as exc:
                raise EXAONETabularDependencyError(
                    "EXAONE requires PyTorch for automatic device selection."
                ) from exc
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device is not None:
            params["device"] = device
        if self.random_state is not None:
            params["seed"] = self.random_state
        if self.ensemble_count is not None:
            if not isinstance(self.ensemble_count, int) or self.ensemble_count < 1:
                raise ValueError("ensemble_count must be a positive integer.")
            params["ensemble_count"] = self.ensemble_count
        if self.compute_dtype is not None:
            if self.compute_dtype not in ("float16", "bfloat16", "float32"):
                raise ValueError("compute_dtype must be float16, bfloat16, or float32.")
            params["compute_dtype"] = self.compute_dtype
        if self.weights is not None:
            params["weights"] = self.weights
        if self.revision is not None:
            params["revision"] = self.revision
        if self.max_vram_bytes is not None:
            if not isinstance(self.max_vram_bytes, int) or self.max_vram_bytes < 1:
                raise ValueError("max_vram_bytes must be a positive integer.")
            params["max_vram_bytes"] = self.max_vram_bytes
        return params

    def _load_backend(self):
        params = self._pretrained_params()
        try:
            module = importlib.import_module("exaonetabular")
        except ImportError as exc:
            raise EXAONETabularDependencyError(
                "EXAONETabularClassifier requires exaonetabular. "
                "Install it with `uv sync --extra exaone`."
            ) from exc
        backend = getattr(module, "EXAONETabularClassifier", None)
        factory = getattr(backend, "from_pretrained", None)
        if factory is None:
            raise EXAONETabularDependencyError(
                "Installed exaonetabular does not expose "
                "EXAONETabularClassifier.from_pretrained."
            )
        return factory(**params)

    def fit(self, X: Any, y: Any, groups: Any = None):
        names = getattr(X, "columns", None)
        X_array = self._prepare_X(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1 or X_array.shape[0] != y_array.shape[0]:
            raise ValueError("y must contain one label per training sample.")
        if groups is not None and np.asarray(groups).shape != (len(y_array),):
            raise ValueError("groups must contain one value per training sample.")
        encoder = LabelEncoder().fit(y_array)
        if len(encoder.classes_) < 2:
            raise ValueError("EXAONE requires at least two classes.")
        encoded = encoder.transform(y_array)
        model = self._load_backend()
        model.fit(X_array, encoded)
        self.model_ = model
        self.X_train_ = X_array.copy()
        self.y_train_ = encoded.copy()
        self.classes_ = encoder.classes_
        self.n_features_in_ = X_array.shape[1]
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if names is not None and all(isinstance(name, str) for name in names):
            self.feature_names_in_ = np.asarray(names, dtype=object)
        return self

    def _get_model(self):
        model = getattr(self, "model_", None)
        if model is None:
            model = self._load_backend()
            model.fit(self.X_train_, self.y_train_)
            self.model_ = model
        return model

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("classes_", "n_features_in_", "X_train_", "y_train_"))
        names = getattr(X, "columns", None)
        X_array = self._prepare_X(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_array.shape[1]}."
            )
        if hasattr(self, "feature_names_in_") and names is not None:
            if not np.array_equal(np.asarray(names), self.feature_names_in_):
                raise ValueError(
                    "Prediction feature names must match training feature names."
                )
        model = self._get_model()
        probabilities = np.asarray(model.predict_proba(X_array), dtype=np.float64)
        if probabilities.shape != (len(X_array), len(self.classes_)):
            raise ValueError(
                f"EXAONE returned an invalid probability shape: {probabilities.shape}."
            )
        backend_classes = getattr(model, "classes_", None)
        if backend_classes is not None:
            backend_classes = np.asarray(backend_classes)
            if backend_classes.shape != (len(self.classes_),):
                raise ValueError("EXAONE returned an invalid class ordering.")
            order = []
            for class_id in range(len(self.classes_)):
                positions = np.flatnonzero(backend_classes == class_id)
                if positions.size != 1:
                    raise ValueError("EXAONE returned an invalid class ordering.")
                order.append(int(positions[0]))
            probabilities = probabilities[:, order]
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("EXAONE returned invalid class probabilities.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("EXAONE class probabilities do not sum to one.")
        return probabilities

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state
