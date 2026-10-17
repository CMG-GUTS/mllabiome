from __future__ import annotations

import importlib
import os
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ._limix2_compat import ensure_limix_torch_norm_interface, optional_limix_triton


class LimiX2DependencyError(ImportError):
    pass


class LimiX2Classifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        model_path: str | None = None,
        inference_config: str | None = None,
        device: str = "auto",
        mix_precision: bool = True,
        outlier_remove_std: float = 12.0,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = True,
        use_data_cache: bool = False,
        cache_dir: str | None = None,
        random_state: int = 0,
    ) -> None:
        self.model_path = model_path
        self.inference_config = inference_config
        self.device = device
        self.mix_precision = mix_precision
        self.outlier_remove_std = outlier_remove_std
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.use_data_cache = use_data_cache
        self.cache_dir = cache_dir
        self.random_state = random_state

    @staticmethod
    def _as_array(X: Any) -> np.ndarray:
        if sparse.issparse(X):
            required_bytes = X.shape[0] * X.shape[1] * np.dtype(np.float32).itemsize
            if required_bytes > 2 * 1024**3:
                raise MemoryError("Densifying the LimiX-2 feature matrix would exceed 2 GiB.")
            X = X.toarray()
        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("X must be a two-dimensional numerical feature matrix.")
        if np.isinf(array).any():
            raise ValueError("X contains infinite feature values.")
        return array

    @staticmethod
    def _resolve_device(torch: Any, device: str):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda", "mps"} and not device.startswith("cuda:"):
            raise ValueError("device must be 'auto', 'cpu', 'cuda', 'mps', or a CUDA device index.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device(device)

    @staticmethod
    def _dependencies():
        try:
            limix = importlib.import_module("limix")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise LimiX2DependencyError(
                "LimiX2Classifier requires LimiX 2.0 and Python >=3.12. "
                "Install the optional dependency with `uv sync --extra limix2`."
            ) from exc
        predictor = getattr(limix, "LimiXPredictor", None)
        if predictor is None:
            raise LimiX2DependencyError("The installed limix package has no LimiXPredictor. "
                "Install the official limix-ldm-ai/LimiX repository, not the unrelated PyPI limix project.")
        return predictor, torch

    def _checkpoint_path(self) -> str:
        if self.model_path is not None:
            path = Path(self.model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"LimiX-2 checkpoint does not exist: {path}")
            return str(path)
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise LimiX2DependencyError(
                "Automatic checkpoint downloading requires huggingface-hub. "
                "Install the limix2 optional dependency or pass model_path."
            ) from exc
        return str(hub.hf_hub_download(repo_id="stable-ai/LimiX-2", filename="LimiX-2.ckpt"))

    def _config_path(self) -> str:
        if self.inference_config is not None:
            path = Path(self.inference_config).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"LimiX-2 inference config does not exist: {path}")
            return str(path)
        try:
            return str(resources.files("config").joinpath("cls_default_noretrieval_v2.json"))
        except ModuleNotFoundError as exc:
            raise LimiX2DependencyError(
                "The installed LimiX package is missing its packaged v2 config. "
                "Install the official limix-ldm-ai/LimiX repository."
            ) from exc

    def fit(self, X: Any, y: Any, groups: Any = None):
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, (int, np.integer)
        ) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer.")
        if not np.isfinite(self.softmax_temperature) or self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive and finite.")
        if not np.isfinite(self.outlier_remove_std) or self.outlier_remove_std <= 0:
            raise ValueError("outlier_remove_std must be positive and finite.")
        if self.device not in {"auto", "cpu", "cuda", "mps"} and not self.device.startswith("cuda:"):
            raise ValueError("Invalid LimiX-2 device.")
        array = self._as_array(X)
        target = np.asarray(y)
        if target.ndim != 1 or target.shape[0] != array.shape[0]:
            raise ValueError("y must have one label per training sample.")
        if array.shape[0] < 2:
            raise ValueError("LimiX-2 requires at least two training rows.")
        if groups is not None and np.asarray(groups).shape != (array.shape[0],):
            raise ValueError("groups must have one entry per training sample.")
        encoder = LabelEncoder().fit(target)
        if encoder.classes_.size < 2:
            raise ValueError("LimiX-2 requires at least two classes.")
        self.X_train_ = array
        self.y_train_ = encoder.transform(target).astype(np.int64, copy=False)
        self.classes_ = encoder.classes_
        self.n_features_in_ = array.shape[1]
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.__dict__.pop("_predictor", None)
        return self

    def _portable_factory(self, factory):
        if getattr(factory, "__module__", None) != "inference.predictor":
            return factory
        upstream = importlib.import_module("inference.v2_0.predictor").LimiXPredictor
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        cache_path = Path(self.cache_dir).expanduser() if self.cache_dir is not None else Path(base) / "mllabiome" / "limix2"
        use_data_cache = self.use_data_cache

        class PortableCacheManager(upstream.CacheManager):
            def __init__(self, cache_dir=None):
                if use_data_cache:
                    super().__init__(cache_dir=str(cache_path))
                else:
                    self.cache_dir = str(cache_path)

        class PortablePredictor(upstream):
            CacheManager = PortableCacheManager

        return PortablePredictor

    def _get_predictor(self):
        predictor = getattr(self, "_predictor", None)
        if predictor is None:
            with optional_limix_triton():
                factory, torch = self._dependencies()
                ensure_limix_torch_norm_interface()
                factory = self._portable_factory(factory)
                predictor = factory(
                    device=self._resolve_device(torch, self.device),
                    model_path=self._checkpoint_path(),
                    inference_config=self._config_path(),
                    mix_precision=self.mix_precision,
                    outlier_remove_std=self.outlier_remove_std,
                    softmax_temperature=self.softmax_temperature,
                    average_before_softmax=self.average_before_softmax,
                    use_data_cache=self.use_data_cache,
                    seed=int(self.random_state),
                )
                ensure_limix_torch_norm_interface()
            self._predictor = predictor
        return predictor

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("X_train_", "y_train_", "classes_", "n_features_in_"))
        array = self._as_array(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {array.shape[1]}.")
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError("Prediction feature names must match training feature names.")
        with optional_limix_triton():
            predictor = self._get_predictor()
            ensure_limix_torch_norm_interface()
            result = predictor.predict(
                self.X_train_, self.y_train_, array, task_type="Classification"
            )
        proba = np.asarray(result, dtype=np.float64)
        shape = (array.shape[0], self.classes_.size)
        if proba.shape != shape:
            raise ValueError(f"Expected LimiX-2 probability shape {shape}, got {proba.shape}.")
        if not np.isfinite(proba).all() or np.any(proba < 0):
            raise ValueError("LimiX-2 returned invalid probabilities.")
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("LimiX-2 probabilities do not sum to one.")
        return proba

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_predictor", None)
        return state
