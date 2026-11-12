from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class SeldonDependencyError(ImportError):
    pass


class _SeldonBase(BaseEstimator):
    _model_name = "seldon"
    _backend_name = ""

    def __init__(
        self,
        *,
        device: str | None = None,
        checkpoint: str | None = None,
        revision: str | None = None,
        n_estimators: int | None = None,
        random_state: int | None = None,
        max_context_size: int | None = None,
        amp: bool | None = None,
    ) -> None:
        self.device = device
        self.checkpoint = checkpoint
        self.revision = revision
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_context_size = max_context_size
        self.amp = amp

    @classmethod
    def _backend_class(cls) -> type:
        try:
            module = importlib.import_module("neuralk_oss")
        except ImportError as exc:
            raise SeldonDependencyError(
                "Seldon requires neuralk-oss>=0.3 and Python >=3.11. "
                "Install it with `pip install 'neuralk-oss>=0.3'` and accept "
                "the model license on Hugging Face."
            ) from exc
        backend = getattr(module, cls._backend_name, None)
        if backend is None or not callable(getattr(backend, "from_pretrained", None)):
            raise SeldonDependencyError(
                f"The installed neuralk_oss package does not provide {cls._backend_name}. "
                "Install neuralk-oss>=0.3."
            )
        return backend

    def _load_backend(self) -> Any:
        params = {
            name: getattr(self, name)
            for name in (
                "device",
                "revision",
                "n_estimators",
                "random_state",
                "max_context_size",
                "amp",
            )
            if getattr(self, name) is not None
        }
        return self._backend_class().from_pretrained(
            self.checkpoint or self._model_name, **params
        )

    @staticmethod
    def _prepare_X(X: Any) -> Any:
        if sparse.issparse(X):
            required = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if required > 2 * 1024**3:
                raise MemoryError(
                    "Densifying the Seldon feature matrix would exceed 2 GiB."
                )
            X = X.toarray()
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) != 2 or X.shape[1] < 1:
            raise ValueError("X must be a nonempty two-dimensional feature matrix.")
        if hasattr(X, "columns") and not X.columns.is_unique:
            raise ValueError("Seldon feature names must be unique.")
        if hasattr(X, "select_dtypes"):
            numeric = X.select_dtypes(include=[np.number])
            if numeric.size and np.isinf(numeric.to_numpy(dtype=np.float64)).any():
                raise ValueError("X contains infinite feature values.")
        else:
            array = np.asarray(X)
            if array.dtype.kind in "iuf" and np.isinf(array).any():
                raise ValueError("X contains infinite feature values.")
        return X

    @staticmethod
    def _check_groups(groups: Any, n_samples: int) -> None:
        if groups is not None and np.asarray(groups).shape != (n_samples,):
            raise ValueError("groups must contain one value per training sample.")

    def _save_context(self, X: Any, y: np.ndarray) -> None:
        self.X_train_ = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)
        self.y_train_ = np.array(y, copy=True)
        self.n_features_in_ = int(X.shape[1])
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

    def _prepare_predict_X(self, X: Any) -> Any:
        check_is_fitted(self, ("X_train_", "y_train_", "n_features_in_"))
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
        return X

    def _get_model(self) -> Any:
        model = getattr(self, "model_", None)
        if model is None:
            model = self._load_backend()
            model.fit(self.X_train_, self.y_train_)
            self.model_ = model
        return model

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state


class SeldonClassifier(ClassifierMixin, _SeldonBase):
    _backend_name = "SeldonClassifier"

    def fit(self, X: Any, y: Any, groups: Any = None):
        X = self._prepare_X(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
            raise ValueError("y must contain one label per training sample.")
        if X.shape[0] < 2:
            raise ValueError(
                "Seldon classification requires at least two training samples."
            )
        self._check_groups(groups, len(labels))
        encoder = LabelEncoder().fit(labels)
        if len(encoder.classes_) < 2:
            raise ValueError("Seldon classification requires at least two classes.")
        encoded = encoder.transform(labels)
        model = self._load_backend()
        model.fit(X, encoded)
        self._save_context(X, encoded)
        self.classes_ = encoder.classes_
        self.model_ = model
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = self._prepare_predict_X(X)
        check_is_fitted(self, "classes_")
        model = self._get_model()
        proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        shape = (X.shape[0], len(self.classes_))
        if proba.shape != shape or not np.isfinite(proba).all() or (proba < 0).any():
            raise ValueError("Seldon returned invalid class probabilities.")
        backend_classes = getattr(model, "classes_", None)
        if backend_classes is not None:
            backend_classes = np.asarray(backend_classes)
            if backend_classes.shape != (len(self.classes_),) or not np.array_equal(
                np.sort(backend_classes), np.arange(len(self.classes_))
            ):
                raise ValueError("Seldon returned an unexpected class ordering.")
            proba = proba[:, np.argsort(backend_classes)]
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Seldon class probabilities do not sum to one.")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class SeldonRegressor(RegressorMixin, _SeldonBase):
    _backend_name = "SeldonRegressor"

    def fit(self, X: Any, y: Any, groups: Any = None):
        X = self._prepare_X(X)
        try:
            target = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Regression targets must be numeric.") from exc
        if target.ndim != 1 or target.shape[0] != X.shape[0]:
            raise ValueError("y must contain one target per training sample.")
        if X.shape[0] < 2 or not np.isfinite(target).all():
            raise ValueError(
                "Seldon regression requires at least two finite training targets."
            )
        self._check_groups(groups, len(target))
        model = self._load_backend()
        model.fit(X, target)
        self._save_context(X, target)
        self.model_ = model
        return self

    def predict(self, X: Any) -> np.ndarray:
        X = self._prepare_predict_X(X)
        values = np.asarray(self._get_model().predict(X), dtype=np.float64)
        if values.shape != (X.shape[0],) or not np.isfinite(values).all():
            raise ValueError("Seldon returned invalid regression predictions.")
        return values

    def predict_quantiles(self, X: Any) -> np.ndarray:
        X = self._prepare_predict_X(X)
        quantiles = np.asarray(self._get_model().predict_quantiles(X), dtype=np.float64)
        if quantiles.shape != (X.shape[0], 999) or not np.isfinite(quantiles).all():
            raise ValueError("Seldon returned invalid regression quantiles.")
        return quantiles


class SeldonMiniClassifier(SeldonClassifier):
    _model_name = "seldon-mini"


class SeldonMiniRegressor(SeldonRegressor):
    _model_name = "seldon-mini"
