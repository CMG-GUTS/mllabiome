from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.validation import check_is_fitted


class RidgeSoftmaxClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        alpha: float = 0.1,
        temperature: float = 1.0,
        solver: str = "auto",
        tol: float = 1e-4,
        max_iter: int | None = None,
        class_weight=None,
        random_state: int | None = None,
    ):
        self.alpha = alpha
        self.temperature = temperature
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be nonnegative and finite.")
        model = RidgeClassifier(
            alpha=self.alpha,
            solver=self.solver,
            tol=self.tol,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        model.fit(X, y, sample_weight=sample_weight)
        self.model_ = model
        self.classes_ = np.asarray(model.classes_)
        if hasattr(model, "n_features_in_"):
            self.n_features_in_ = int(model.n_features_in_)
        if hasattr(model, "feature_names_in_"):
            self.feature_names_in_ = np.asarray(model.feature_names_in_)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        return self.model_.decision_function(X)

    def predict_proba(self, X):
        scores = np.asarray(self.decision_function(X), dtype=float)
        if scores.ndim == 1:
            logits = np.column_stack((-scores, scores))
        elif scores.ndim == 2 and scores.shape[1] == len(self.classes_):
            logits = scores
        else:
            raise ValueError("Unexpected Ridge decision-score shape.")
        if not np.isfinite(logits).all():
            raise ValueError("Ridge decision scores must be finite.")
        return np.asarray(softmax(logits / self.temperature, axis=1), dtype=float)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
