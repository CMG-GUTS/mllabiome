from __future__ import annotations

import copy
import importlib
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class TabPFNDependencyError(ImportError):
    pass


class TabPFNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        *,
        n_estimators: int | str = "auto",
        auto_scale_n_estimators: bool = True,
        categorical_features_indices: Any = None,
        softmax_temperature: float | str = "auto",
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        model_path: Any = "auto",
        device: Any = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: Any = "auto",
        fit_mode: str = "fit_preprocessors",
        memory_saving_mode: Any = "auto",
        keep_cache_on_device: bool = True,
        kv_cache_precision: str | None = None,
        random_state: Any = 42,
        n_preprocessing_jobs: int = 1,
        inference_config: Any = None,
        eval_metric: Any = None,
        tuning_config: Any = None,
        show_progress_bar: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.auto_scale_n_estimators = auto_scale_n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.fit_mode = fit_mode
        self.memory_saving_mode = memory_saving_mode
        self.keep_cache_on_device = keep_cache_on_device
        self.kv_cache_precision = kv_cache_precision
        self.random_state = random_state
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.inference_config = inference_config
        self.eval_metric = eval_metric
        self.tuning_config = tuning_config
        self.show_progress_bar = show_progress_bar

    def _backend_class(self):
        try:
            module = importlib.import_module("tabpfn")
        except ImportError as exc:
            raise TabPFNDependencyError(
                "TabPFNClassifier requires the mllabiome tabpfn extra. Run `uv sync --extra tabpfn` or `pip install 'mllabiome[tabpfn]'`."
            ) from exc
        backend = getattr(module, "TabPFNClassifier", None)
        if backend is None:
            raise TabPFNDependencyError(
                "The installed tabpfn package does not expose TabPFNClassifier. Install tabpfn>=9.1,<10."
            )
        return backend

    def _backend_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "auto_scale_n_estimators": self.auto_scale_n_estimators,
            "categorical_features_indices": copy.deepcopy(
                self.categorical_features_indices
            ),
            "softmax_temperature": self.softmax_temperature,
            "balance_probabilities": self.balance_probabilities,
            "average_before_softmax": self.average_before_softmax,
            "model_path": self.model_path,
            "device": self.device,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "inference_precision": self.inference_precision,
            "fit_mode": self.fit_mode,
            "memory_saving_mode": self.memory_saving_mode,
            "keep_cache_on_device": self.keep_cache_on_device,
            "kv_cache_precision": self.kv_cache_precision,
            "random_state": self.random_state,
            "n_preprocessing_jobs": self.n_preprocessing_jobs,
            "inference_config": copy.deepcopy(self.inference_config),
            "eval_metric": self.eval_metric,
            "tuning_config": copy.deepcopy(self.tuning_config),
            "show_progress_bar": self.show_progress_bar,
        }

    def fit(self, X: Any, y: Any, groups: Any = None):
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            y_array = y_array.reshape(-1)
        if len(np.unique(y_array)) < 2:
            raise ValueError("TabPFNClassifier requires at least two outcome classes.")
        if groups is not None:
            groups_array = np.asarray(groups)
            if groups_array.shape[0] != y_array.shape[0]:
                raise ValueError(
                    "groups must contain exactly one value per training sample."
                )
            if self.tuning_config is not None:
                raise ValueError(
                    "TabPFN internal tuning is not group-aware. Disable tuning_config for grouped evaluation."
                )
        backend_class = self._backend_class()
        self.model_ = backend_class(**self._backend_params())
        self.model_.fit(X, y_array)
        self.classes_ = np.asarray(self.model_.classes_)
        self.n_features_in_ = int(
            getattr(self.model_, "n_features_in_", np.asarray(X).shape[1])
        )
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        names = getattr(self.model_, "feature_names_in_", None)
        if names is not None:
            names_array = np.asarray(names, dtype=object)
            if names_array.ndim == 1 and names_array.shape[0] == self.n_features_in_:
                self.feature_names_in_ = names_array
        return self

    def predict(self, X: Any):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X: Any):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)
