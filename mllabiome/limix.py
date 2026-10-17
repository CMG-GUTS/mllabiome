from __future__ import annotations

import importlib
import os
from importlib import resources
from pathlib import Path
from typing import Any

from .limix2 import LimiX2Classifier, LimiX2DependencyError


class LimiXClassifier(LimiX2Classifier):
    def __init__(
        self,
        *,
        size: str = "2m",
        retrieval: bool = False,
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
        self.size = size
        self.retrieval = retrieval
        super().__init__(
            model_path=model_path,
            inference_config=inference_config,
            device=device,
            mix_precision=mix_precision,
            outlier_remove_std=outlier_remove_std,
            softmax_temperature=softmax_temperature,
            average_before_softmax=average_before_softmax,
            use_data_cache=use_data_cache,
            cache_dir=cache_dir,
            random_state=random_state,
        )

    def fit(self, X: Any, y: Any, groups: Any = None):
        if self.size not in {"2m", "16m"}:
            raise ValueError("size must be '2m' or '16m'.")
        if not isinstance(self.retrieval, bool):
            raise ValueError("retrieval must be a boolean.")
        return super().fit(X, y, groups=groups)

    def _checkpoint_path(self) -> str:
        if self.model_path is not None:
            return super()._checkpoint_path()
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise LimiX2DependencyError(
                "Automatic LimiX checkpoint downloading requires huggingface-hub."
            ) from exc
        suffix = "2M" if self.size == "2m" else "16M"
        return str(
            hub.hf_hub_download(
                repo_id=f"stable-ai/LimiX-1_{suffix}",
                filename=f"LimiX-1_{suffix}.ckpt",
            )
        )

    def _config_path(self) -> str:
        if self.inference_config is not None:
            return super()._config_path()
        name = "cls_default_retrieval.json" if self.retrieval else "cls_default_noretrieval.json"
        try:
            return str(resources.files("config").joinpath(name))
        except ModuleNotFoundError as exc:
            raise LimiX2DependencyError(
                "The installed LimiX package is missing its V1 classification configs."
            ) from exc

    @staticmethod
    def _dependencies():
        try:
            torch = importlib.import_module("torch")
            predictor = importlib.import_module("inference.v1_0.predictor")
        except ImportError as exc:
            raise LimiX2DependencyError(
                "LimiXClassifier requires the official LimiX V1 inference backend. "
                "Install the optional dependency with `uv sync --extra limix2`."
            ) from exc
        backend = getattr(predictor, "LimiXPredictor", None)
        if backend is None:
            raise LimiX2DependencyError(
                "The official LimiX V1 backend does not expose LimiXPredictor."
            )
        return backend, torch

    def _portable_factory(self, factory):
        if getattr(factory, "__module__", None) not in {
            "inference.predictor",
            "inference.v1_0.predictor",
        }:
            return factory
        upstream = factory
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        cache_path = (
            Path(self.cache_dir).expanduser()
            if self.cache_dir is not None
            else Path(base) / "mllabiome" / "limix"
        )
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
