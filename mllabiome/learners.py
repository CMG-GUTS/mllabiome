from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


def _feature_names_from_X(X: Any) -> list[str]:
    """Return stable string feature names for estimator backends that track them."""
    if hasattr(X, "columns"):
        return [str(c) for c in list(X.columns)]
    arr = np.asarray(X)
    n_features = 1 if arr.ndim == 1 else int(arr.shape[1])
    return [f"feature_{i:06d}" for i in range(n_features)]


def _as_named_frame(X: Any, feature_names: list[str]):
    """Coerce prediction input to a DataFrame with the names used at fit time."""
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        if list(map(str, X.columns)) == list(feature_names):
            return X
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"FLAMLClassifier expected {len(feature_names)} features but received {arr.shape[1]}."
        )
    return pd.DataFrame(arr, columns=list(feature_names))


def _quiet_known_flaml_sklearn_warnings():
    """Suppress noisy warnings emitted by FLAML's internal sklearn candidates."""
    warnings.filterwarnings(
        "ignore",
        message=r"'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Inconsistent values: penalty=.*",
        category=UserWarning,
        module=r"sklearn\.linear_model\._logistic",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"'n_jobs' has no effect since 1\.8.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )


def _logistic_regression(*, kind: str = "l2", **kwargs) -> LogisticRegression:
    """Construct LogisticRegression without scikit-learn 1.8 penalty deprecation noise."""
    params = dict(kwargs)
    default = inspect.signature(LogisticRegression).parameters["penalty"].default
    if default == "deprecated":
        # scikit-learn >=1.8: penalty is replaced by l1_ratio/C semantics.
        if kind == "l1":
            params.setdefault("solver", "saga")
            params.setdefault("l1_ratio", 1.0)
        elif kind == "l2":
            params.setdefault("l1_ratio", 0.0)
        elif kind == "elasticnet":
            params.setdefault("solver", "saga")
            params.setdefault("l1_ratio", 0.5)
        elif kind == "none":
            params.setdefault("C", np.inf)
            params.setdefault("l1_ratio", 0.0)
        return LogisticRegression(**params)
    # scikit-learn <1.8 compatibility.
    if kind == "l1":
        params.setdefault("penalty", "l1")
        params.setdefault("solver", "saga")
    elif kind == "l2":
        params.setdefault("penalty", "l2")
    elif kind == "elasticnet":
        params.setdefault("penalty", "elasticnet")
        params.setdefault("solver", "saga")
        params.setdefault("l1_ratio", 0.5)
    elif kind == "none":
        params.setdefault("penalty", None)
    return LogisticRegression(**params)


def _learner_name(item: Any) -> str:
    if hasattr(item, "name") and not isinstance(item, tuple):
        return str(getattr(item, "name"))
    return item if isinstance(item, str) else str(item[0])


def _learner_factory(item: Any) -> tuple[str, Callable[[], BaseEstimator]]:
    if isinstance(item, tuple):
        name, spec = item
        if isinstance(spec, BaseEstimator):
            return str(name), lambda spec=spec: clone(spec)
        if callable(spec):
            return str(name), spec
        raise TypeError(
            f"Learner tuple for {name!r} must contain an estimator or factory."
        )
    return str(item), lambda item=item: build_learner(str(item))


class FLAMLClassifier(BaseEstimator):
    """Small scikit-learn-compatible FLAML AutoML wrapper."""

    def __init__(
        self,
        time_budget=600,
        metric="roc_auc",
        estimator_list=None,
        n_jobs=1,
        random_state=42,
        verbose=0,
        **kwargs,
    ):
        self.time_budget = time_budget
        self.metric = metric
        self.estimator_list = estimator_list
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(self, X, y):
        from flaml import AutoML

        self.feature_names_in_ = np.asarray(_feature_names_from_X(X), dtype=object)
        X_frame = _as_named_frame(X, list(self.feature_names_in_))
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        metric = self.metric
        if len(self.classes_) > 2 and metric == "roc_auc":
            metric = "roc_auc_ovr"
        self.model_ = AutoML()
        fit_kwargs = dict(
            X_train=X_frame,
            y_train=y,
            task="classification",
            time_budget=self.time_budget,
            metric=metric,
            n_jobs=self.n_jobs,
            seed=self.random_state,
            verbose=self.verbose,
            **self.kwargs,
        )
        if self.estimator_list is not None:
            fit_kwargs["estimator_list"] = self.estimator_list
        with warnings.catch_warnings():
            _quiet_known_flaml_sklearn_warnings()
            self.model_.fit(**fit_kwargs)
        return self

    def predict(self, X):
        X_frame = _as_named_frame(X, list(self.feature_names_in_))
        return self.model_.predict(X_frame)

    def predict_proba(self, X):
        X_frame = _as_named_frame(X, list(self.feature_names_in_))
        return self.model_.predict_proba(X_frame)


def build_learner(name: str) -> BaseEstimator:
    key = str(name)
    base = key.lower()
    if base in {"lr", "lr_l2", "lr_l2_bal", "logistic"}:
        return _logistic_regression(
            kind="l2", max_iter=2000, class_weight="balanced", random_state=42
        )
    if base in {"lr_l1", "lr_l1_bal"}:
        return _logistic_regression(
            kind="l1", max_iter=3000, class_weight="balanced", random_state=42
        )
    if base in {"ridge", "ridge_a1", "ridgeclassifier"}:
        return RidgeClassifier(alpha=1.0, random_state=42)
    if base == "ridge_a01":
        return RidgeClassifier(alpha=0.1, random_state=42)
    if base in {"rf", "rf_500_msl5"}:
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, n_jobs=1, random_state=42
        )
    if base in {"rf_1000_msl5", "baseline_rf"}:
        return RandomForestClassifier(
            n_estimators=1000, min_samples_leaf=5, n_jobs=1, random_state=42
        )
    if base in {"rf_200", "rf_fast"}:
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=42
        )
    if base in {"et", "extratrees"}:
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, n_jobs=1, random_state=42
        )
    if base in {"histgb", "histgradientboosting"}:
        return HistGradientBoostingClassifier(random_state=42)
    if base in {"svc_rbf", "svc_rbf_bal"}:
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    if base in {"calib_lsvc", "caliblsvc_c1_sig"}:
        return CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
            method="sigmoid",
            cv=3,
        )
    if base in {"knn", "knn_default"}:
        return KNeighborsClassifier()
    if base in {"nearestcentroid", "nearestcentroid_raw"}:
        return NearestCentroid()
    if base in {"gnb", "gaussiannb"}:
        return GaussianNB()
    if base in {"bnb", "bernoullinb"}:
        return BernoulliNB()
    if base in {"mnb", "multinomialnb"}:
        return MultinomialNB()
    if base in {"lda"}:
        return LinearDiscriminantAnalysis()
    if base in {"qda"}:
        return QuadraticDiscriminantAnalysis(reg_param=0.1)
    if base in {"sgd_log", "sgd"}:
        return SGDClassifier(loss="log_loss", penalty="l2", random_state=42)
    if base in {"pa", "pa_default"}:
        return CalibratedClassifierCV(
            PassiveAggressiveClassifier(random_state=42), method="sigmoid", cv=3
        )
    if base in {"dt", "decisiontree"}:
        return DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    if base in {"flaml", "flaml_600s", "automl"}:
        return FLAMLClassifier(
            time_budget=600, metric="roc_auc", n_jobs=1, random_state=42
        )
    if base in {"siamcat"}:
        from .siamcat import SIAMCATClassifier

        return SIAMCATClassifier()
    if base.startswith("xgb"):
        mod = importlib.import_module("xgboost")
        return mod.XGBClassifier(
            eval_metric="logloss", random_state=42, verbosity=0, nthread=1
        )
    if base.startswith("lgb"):
        mod = importlib.import_module("lightgbm")
        return mod.LGBMClassifier(n_jobs=1, random_state=42, verbose=-1)
    if base.startswith("cat") or base.startswith("cb"):
        mod = importlib.import_module("catboost")
        return mod.CatBoostClassifier(
            verbose=False, random_seed=42, allow_writing_files=False
        )
    raise ValueError(
        f"Unknown learner {name!r}. Provide (name, estimator) in the sweep config to add it."
    )
