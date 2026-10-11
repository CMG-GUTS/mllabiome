from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from .estimator_protocol import clone_estimator
from .ensemble_aggregation import ecdf_stacking_features
from .learners import GroupAwareCalibratedClassifier


class GroupedRankCalibratedClassifier(BaseEstimator):
    def __init__(
        self, estimator, n_splits=3, C=0.1, random_state=42, class_weight=None
    ):
        self.estimator = estimator
        self.n_splits = n_splits
        self.C = C
        self.random_state = random_state
        self.class_weight = class_weight

    def fit(self, X, y, groups=None):
        labels = np.asarray(y)
        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError("Grouped rank calibration requires binary outcomes.")
        splitter = GroupAwareCalibratedClassifier(
            self.estimator,
            n_splits=self.n_splits,
            random_state=self.random_state,
        )
        folds = splitter._splits(X, labels, groups)
        probabilities = np.full(len(labels), np.nan, dtype=float)
        for training, validation in folds:
            model = clone_estimator(self.estimator)
            model.fit(X[training], labels[training])
            index = int(np.flatnonzero(np.asarray(model.classes_) == classes[1])[0])
            probabilities[validation] = np.asarray(
                model.predict_proba(X[validation]), dtype=float
            )[:, index]
        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Grouped rank calibration failed to cover every training sample."
            )
        self.reference_ = np.sort(probabilities).tolist()
        matrix = np.stack([np.column_stack([1.0 - probabilities, probabilities])])
        features = ecdf_stacking_features(matrix, [self.reference_])
        self.calibrator_ = LogisticRegression(
            C=float(self.C),
            solver="lbfgs",
            class_weight=self.class_weight,
            max_iter=5000,
            random_state=int(self.random_state),
        ).fit(features, (labels == classes[1]).astype(int))
        self.model_ = clone_estimator(self.estimator)
        self.model_.fit(X, labels)
        self.classes_ = classes
        return self

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("The grouped rank calibrator is not fitted.")
        index = int(
            np.flatnonzero(np.asarray(self.model_.classes_) == self.classes_[1])[0]
        )
        probabilities = np.asarray(self.model_.predict_proba(X), dtype=float)[:, index]
        matrix = np.stack([np.column_stack([1.0 - probabilities, probabilities])])
        features = ecdf_stacking_features(matrix, [self.reference_])
        positive = self.calibrator_.predict_proba(features)[:, 1]
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
