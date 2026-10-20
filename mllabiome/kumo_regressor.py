from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .kumo import KumoTabularClassifier


class KumoTabularRegressor(RegressorMixin, BaseEstimator):
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
        return KumoTabularClassifier._dependencies()

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
        array = KumoTabularClassifier._array(X)
        try:
            target = np.asarray(y, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Regression targets must be numerical.") from exc
        if target.ndim != 1 or target.shape[0] != array.shape[0]:
            raise ValueError("y must have one numerical value per training sample.")
        if array.shape[0] < 2 or not np.isfinite(target).all():
            raise ValueError("At least two samples and finite targets are required.")
        if groups is not None and np.asarray(groups).shape != (array.shape[0],):
            raise ValueError("groups must contain one value per training sample.")
        sdm, torch = KumoTabularClassifier._dependencies()
        if self.device == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device is None:
            resolved = torch.get_default_device()
        else:
            resolved = torch.device(self.device)
        if resolved.type == "mps":
            raise ValueError("Kumo Tabular regression does not support MPS preprocessing; use device='cpu'.")
        self.device_ = resolved
        self.n_features_in_ = array.shape[1]
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if hasattr(X, "columns") and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_columns_ = tuple(
            f"feature_{i}" for i in range(self.n_features_in_)
        )
        context = KumoTabularClassifier._as_table(
            self, array, columns=self.feature_columns_
        )
        labels = torch.as_tensor(
            target.reshape(-1, 1), dtype=torch.float32, device=self.device_
        )
        context_y = sdm.TableTensor.from_tensor(labels, columns=("target",))
        generator = None
        if self.random_state is not None:
            generator = torch.Generator(device=self.device_)
            generator.manual_seed(int(self.random_state))
        self.model_ = sdm.models.KumoTabular(
            task="regression",
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

    def predict_quantiles(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("model_", "n_features_in_", "feature_columns_"))
        array = KumoTabularClassifier._array(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {array.shape[1]}."
            )
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            if not np.array_equal(np.asarray(X.columns), self.feature_names_in_):
                raise ValueError("Prediction feature names must match training features.")
        query = KumoTabularClassifier._as_table(
            self, array, columns=self.feature_columns_
        )
        sdm, torch = KumoTabularClassifier._dependencies()
        with (
            torch.inference_mode(),
            torch.amp.autocast(
                self.device_.type,
                dtype=torch.float16,
                enabled=self.device_.type == "cuda",
            ),
        ):
            output = self.model_.predict(query)
        quantiles = output.numerical.detach().cpu().numpy().astype(
            np.float64, copy=False
        )
        names = tuple(output.columns[sdm.Stype.numerical])
        expected = tuple(f"q{i:03d}" for i in range(1, 1000))
        if names != expected:
            if len(names) != len(expected) or set(names) != set(expected):
                raise ValueError("Unexpected Kumo Tabular regression quantile columns.")
            quantiles = quantiles[:, [names.index(name) for name in expected]]
        if quantiles.shape != (array.shape[0], 999):
            raise ValueError(f"Unexpected regression quantile shape: {quantiles.shape}.")
        if not np.isfinite(quantiles).all():
            raise ValueError("Kumo Tabular produced non-finite regression quantiles.")
        return quantiles

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_quantiles(X).mean(axis=1)
