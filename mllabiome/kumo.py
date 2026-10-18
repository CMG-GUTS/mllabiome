from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class KumoTabularDependencyError(ImportError):
    pass


class KumoTabularClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        size: str = "large",
        pretrained: bool = True,
        n_estimators: int = 8,
        device: str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.size = size
        self.pretrained = pretrained
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state

    @staticmethod
    def _dependencies():
        try:
            sdm = importlib.import_module("sdm")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise KumoTabularDependencyError(
                "KumoTabularClassifier requires Python >=3.11 and "
                "structured-data-models>=0.2.1. Install the kumo optional dependency "
                "with `uv sync --extra kumo` on a supported Python version."
            ) from exc
        if not hasattr(sdm, "TableTensor") or not hasattr(sdm.models, "KumoTabular"):
            raise KumoTabularDependencyError(
                "The installed sdm package does not expose KumoTabular and TableTensor."
            )
        return sdm, torch

    @staticmethod
    def _array(X: Any) -> np.ndarray:
        if sparse.issparse(X):
            required_bytes = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if required_bytes > 2 * 1024**3:
                raise MemoryError(
                    "Densifying the Kumo Tabular matrix would exceed 2 GiB."
                )
            X = X.toarray()
        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional numerical feature matrix.")
        if array.shape[1] < 1:
            raise ValueError("X must contain at least one feature.")
        if np.isinf(array).any():
            raise ValueError("X contains infinite feature values.")
        return array

    @staticmethod
    def _resolve_device(torch: Any, device: str | None):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if device is None:
            return torch.get_default_device()
        return torch.device(device)

    def _as_table(self, X: np.ndarray, *, columns: tuple[str, ...]):
        sdm, torch = self._dependencies()
        tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
        return sdm.TableTensor.from_tensor(tensor, columns=columns)

    def fit(self, X: Any, y: Any, groups: Any = None):
        if self.size not in {"small", "medium", "large"}:
            raise ValueError("size must be 'small', 'medium', or 'large'.")
        if (
            isinstance(self.n_estimators, (bool, np.bool_))
            or not isinstance(self.n_estimators, (int, np.integer))
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")
        if self.random_state is not None and (
            isinstance(self.random_state, (bool, np.bool_))
            or not isinstance(self.random_state, (int, np.integer))
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None.")
        array = self._array(X)
        target = np.asarray(y)
        if target.ndim != 1 or target.shape[0] != array.shape[0]:
            raise ValueError(
                "y must be a one-dimensional target with one value per sample."
            )
        if groups is not None and np.asarray(groups).shape != (array.shape[0],):
            raise ValueError("groups must contain one value per training sample.")
        encoder = LabelEncoder().fit(target)
        if encoder.classes_.size < 2:
            raise ValueError("KumoTabularClassifier requires at least two classes.")
        if encoder.classes_.size > 10:
            raise ValueError("Kumo Tabular supports at most ten classes.")
        sdm, torch = self._dependencies()
        self.device_ = self._resolve_device(torch, self.device)
        self.n_features_in_ = array.shape[1]
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_columns_ = tuple(
            f"feature_{i}" for i in range(self.n_features_in_)
        )
        self.classes_ = encoder.classes_
        context = self._as_table(array, columns=self.feature_columns_)
        labels = torch.as_tensor(
            encoder.transform(target).reshape(-1, 1),
            dtype=torch.int64,
            device=self.device_,
        )
        context_y = sdm.TableTensor.from_tensor(labels, columns=("target",))
        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=self.device_)
            generator.manual_seed(int(self.random_state))
        self.model_ = sdm.models.KumoTabular(
            task="classification",
            size=self.size,
            pretrained=self.pretrained,
            device=self.device_,
        )
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                self.device_.type,
                dtype=torch.float16,
                enabled=self.device_.type == "cuda",
            ),
        ):
            self.model_.fit(
                x=context,
                y=context_y,
                num_estimators=int(self.n_estimators),
                generator=generator,
            )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("model_", "classes_", "n_features_in_"))
        array = self._array(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {array.shape[1]}."
            )
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError(
                    "Prediction feature names must match the training features."
                )
        query = self._as_table(array, columns=self.feature_columns_)
        sdm, torch = self._dependencies()
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                self.device_.type,
                dtype=torch.float16,
                enabled=self.device_.type == "cuda",
            ),
        ):
            output = self.model_.predict(query)
        proba = output.numerical.detach().cpu().numpy().astype(np.float64, copy=False)
        names = tuple(output.columns[sdm.Stype.numerical])
        expected = tuple(str(i) for i in range(len(self.classes_)))
        if set(names) != set(expected) or len(names) != len(expected):
            raise ValueError(f"Unexpected Kumo Tabular probability columns: {names}.")
        proba = proba[:, [names.index(name) for name in expected]]
        if proba.shape != (array.shape[0], len(self.classes_)):
            raise ValueError(
                f"Unexpected Kumo Tabular probability shape: {proba.shape}."
            )
        if not np.isfinite(proba).all() or (proba < 0).any():
            raise ValueError("Kumo Tabular produced invalid probabilities.")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Kumo Tabular probabilities do not sum to one.")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
