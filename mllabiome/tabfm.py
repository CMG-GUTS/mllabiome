from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class TabFMDependencyError(ImportError):
    pass


class TabFMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        backend: str = "pytorch",
        device: str | None = None,
        checkpoint_path: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.backend = backend
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.random_state = random_state

    @staticmethod
    def _prepare_X(X: Any) -> Any:
        if sparse.issparse(X):
            bytes_required = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if bytes_required > 2 * 1024**3:
                raise MemoryError(
                    "Densifying the TabFM feature matrix would exceed 2 GiB."
                )
            X = X.astype(np.float32).toarray()
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) != 2 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty two-dimensional feature matrix.")
        if isinstance(X, np.ndarray) and X.dtype.kind in "iuf" and np.isinf(X).any():
            raise ValueError("X contains infinite values.")
        if hasattr(X, "columns") and not X.columns.is_unique:
            raise ValueError("TabFM feature names must be unique.")
        return X

    def _load_backend(self) -> Any:
        if self.backend not in ("pytorch", "jax"):
            raise ValueError("backend must be 'pytorch' or 'jax'.")
        try:
            module = importlib.import_module("tabfm")
        except ImportError as exc:
            raise TabFMDependencyError(
                "TabFMClassifier requires TabFM. Install it with `uv sync --extra tabfm`."
            ) from exc
        name = f"tabfm_v1_0_0_{self.backend}"
        loader = getattr(module, name, None)
        classifier = getattr(module, "TabFMClassifier", None)
        if loader is None or classifier is None:
            raise TabFMDependencyError(
                f"TabFM {self.backend} backend is unavailable. Install the corresponding TabFM extra."
            )
        kwargs: dict[str, Any] = {"model_type": "classification"}
        if self.checkpoint_path is not None:
            kwargs["checkpoint_path"] = self.checkpoint_path
        if self.backend == "pytorch":
            device = self.device
            if device == "auto":
                try:
                    torch = importlib.import_module("torch")
                except ImportError as exc:
                    raise TabFMDependencyError(
                        "The PyTorch backend requires torch."
                    ) from exc
                if torch.cuda.is_available():
                    device = "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    device = "mps"
                else:
                    device = "cpu"
            if device is not None:
                kwargs["device"] = device
        elif self.device not in (None, "auto"):
            raise ValueError("For JAX, device selection is managed by the JAX runtime.")
        model = loader.load(**kwargs)
        estimator_kwargs = {}
        if self.random_state is not None:
            estimator_kwargs["random_state"] = self.random_state
        return classifier(model=model, **estimator_kwargs)

    def fit(self, X: Any, y: Any, groups: Any = None):
        X = self._prepare_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("y must contain one label per training sample.")
        if groups is not None and np.asarray(groups).shape != (len(y),):
            raise ValueError("groups must contain one value per training sample.")
        encoder = LabelEncoder().fit(y)
        if len(encoder.classes_) < 2:
            raise ValueError("TabFM requires at least two classes.")
        model = self._load_backend()
        encoded = encoder.transform(y)
        model.fit(X, encoded)
        self.model_ = model
        self.X_train_ = X.copy()
        self.y_train_ = encoded.copy()
        self.classes_ = encoder.classes_
        self.n_features_in_ = int(X.shape[1])
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(name, str) for name in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _get_model(self) -> Any:
        model = getattr(self, "model_", None)
        if model is None:
            model = self._load_backend()
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
        if proba.shape != (len(X), len(self.classes_)):
            raise ValueError(
                f"TabFM returned an invalid probability shape: {proba.shape}."
            )
        if not np.isfinite(proba).all() or (proba < 0).any():
            raise ValueError("TabFM returned invalid class probabilities.")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("TabFM class probabilities do not sum to one.")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state
