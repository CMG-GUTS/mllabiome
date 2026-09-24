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
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    PassiveAggressiveClassifier,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

_COMPUTE_ONLY_DISPLAY_PARAMS = {
    "n_jobs",
    "random_state",
    "verbose",
    "verbosity",
    "nthread",
    "thread_count",
    "allow_writing_files",
    "cache_size",
    "max_iter",
    "tol",
    "time_budget",
}

_DISPLAY_PARAM_PRIORITY = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "criterion",
    "bootstrap",
    "C",
    "penalty",
    "solver",
    "l1_ratio",
    "kernel",
    "gamma",
    "degree",
    "alpha",
    "hidden_layer_sizes",
    "activation",
    "learning_rate",
    "learning_rate_init",
    "early_stopping",
    "subsample",
    "loss",
    "reg_param",
    "n_neighbors",
    "weights",
    "p",
    "var_smoothing",
)

_LEARNER_DISPLAY_NAMES = {
    "RandomForestClassifier": "Random forest",
    "RandomForestRegressor": "Random forest",
    "ExtraTreesClassifier": "Extremely randomized trees",
    "ExtraTreesRegressor": "Extremely randomized trees",
    "MLPClassifier": "Multilayer perceptron",
    "MLPRegressor": "Multilayer perceptron",
    "LogisticRegression": "Logistic regression",
    "SVC": "Support vector machine",
    "LinearSVC": "Linear support vector machine",
    "KNeighborsClassifier": "k-nearest neighbours",
    "KNeighborsRegressor": "k-nearest neighbours",
    "DecisionTreeClassifier": "Decision tree",
    "DecisionTreeRegressor": "Decision tree",
    "HistGradientBoostingClassifier": "Histogram gradient boosting",
    "HistGradientBoostingRegressor": "Histogram gradient boosting",
    "RidgeClassifier": "Ridge classifier",
    "Ridge": "Ridge regression",
    "ElasticNet": "Elastic net",
    "GaussianNB": "Gaussian naive Bayes",
    "BernoulliNB": "Bernoulli naive Bayes",
    "MultinomialNB": "Multinomial naive Bayes",
    "LinearDiscriminantAnalysis": "Linear discriminant analysis",
    "QuadraticDiscriminantAnalysis": "Quadratic discriminant analysis",
    "SGDClassifier": "Stochastic gradient descent",
    "FLAMLClassifier": "FLAML",
    "XGBClassifier": "XGBoost",
    "XGBRegressor": "XGBoost",
    "LGBMClassifier": "LightGBM",
    "LGBMRegressor": "LightGBM",
    "CatBoostClassifier": "CatBoost",
    "CatBoostRegressor": "CatBoost",
}


def _display_param_equal(left: Any, right: Any) -> bool:
    try:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        value = left == right
        if isinstance(value, np.ndarray):
            return bool(np.all(value))
        return bool(value)
    except Exception:
        return False


def _display_param_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        return (
            "("
            + ", ".join(_display_param_value(x) for x in value)
            + ("," if len(value) == 1 else "")
            + ")"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_display_param_value(x) for x in value) + "]"
    return str(value)


def _changed_display_params(estimator: Any) -> list[tuple[str, Any]]:
    if not callable(getattr(estimator, "get_params", None)):
        return []
    try:
        current = estimator.get_params(deep=False)
        signature = inspect.signature(type(estimator).__init__)
    except Exception:
        return []
    defaults = {
        name: param.default
        for name, param in signature.parameters.items()
        if name != "self" and param.default is not inspect.Parameter.empty
    }
    try:
        baseline = type(estimator)()
        baseline_params = baseline.get_params(deep=False)
    except Exception:
        baseline_params = {}
    for name, value in baseline_params.items():
        defaults.setdefault(name, value)
    changed = {
        name: value
        for name, value in current.items()
        if name in defaults
        and name not in _COMPUTE_ONLY_DISPLAY_PARAMS
        and not callable(value)
        and not callable(getattr(value, "get_params", None))
        and not _display_param_equal(value, defaults[name])
    }
    ordered = [name for name in _DISPLAY_PARAM_PRIORITY if name in changed]
    ordered.extend(sorted(name for name in changed if name not in ordered))
    return [(name, changed[name]) for name in ordered[:2]]


def learner_display_label(item: Any) -> str:
    if not isinstance(item, tuple) or len(item) != 2:
        return str(item).replace("_", " ")
    name, specification = item
    estimator = specification
    if not isinstance(estimator, BaseEstimator) and callable(specification):
        try:
            estimator = specification()
        except Exception:
            estimator = None
    if estimator is None:
        return str(name).replace("_", " ")
    label = _LEARNER_DISPLAY_NAMES.get(
        type(estimator).__name__, str(name).replace("_", " ")
    )
    params = _changed_display_params(estimator)
    if params:
        body = ", ".join(
            f"{key}={_display_param_value(value)}" for key, value in params
        )
        label = f"{label} ({body})"
    return label


def _feature_names_from_X(X: Any) -> list[str]:
    if hasattr(X, "columns"):
        return [str(c) for c in list(X.columns)]
    arr = np.asarray(X)
    n_features = 1 if arr.ndim == 1 else int(arr.shape[1])
    return [f"feature_{i:06d}" for i in range(n_features)]


def _as_named_frame(X: Any, feature_names: list[str]):
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


def _logistic_regression(*, kind: str = "l2", **kwargs) -> LogisticRegression:
    params = dict(kwargs)
    default = inspect.signature(LogisticRegression).parameters["penalty"].default
    if default == "deprecated":
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


class GroupAwareCalibratedClassifier(BaseEstimator):
    def __init__(self, estimator, method="sigmoid", n_splits=3, random_state=42):
        self.estimator = estimator
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state

    def _splits(self, X, y, groups):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Calibration requires at least two outcome classes.")
        if groups is None:
            n_splits = min(int(self.n_splits), int(counts.min()))
            if n_splits < 2:
                raise ValueError(
                    "Calibration requires at least two samples in every class."
                )
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=int(self.random_state)
            )
            return list(splitter.split(np.zeros(len(y), dtype=np.uint8), y))
        groups = np.asarray(groups)
        if groups.shape[0] != y.shape[0]:
            raise ValueError(
                "groups must contain exactly one value per training sample."
            )
        unique_groups = np.unique(groups)
        class_group_counts = [len(np.unique(groups[y == klass])) for klass in classes]
        max_splits = min(
            int(self.n_splits), len(unique_groups), min(class_group_counts)
        )
        if max_splits < 2:
            raise ValueError(
                "Group-aware calibration requires every class to occur in at least two distinct training groups."
            )
        required = set(classes.tolist())
        for n_splits in range(max_splits, 1, -1):
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=int(self.random_state)
            )
            splits = list(splitter.split(np.zeros(len(y), dtype=np.uint8), y, groups))
            if all(
                set(np.unique(y[tr]).tolist()) == required
                and set(np.unique(y[va]).tolist()) == required
                for tr, va in splits
            ):
                return splits
        raise ValueError(
            "Group-aware calibration could not construct folds containing every class in both training and calibration partitions."
        )

    def fit(self, X, y, groups=None):
        splits = self._splits(X, y, groups)
        self.model_ = CalibratedClassifierCV(
            clone(self.estimator), method=self.method, cv=splits
        )
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_)
        if hasattr(self.model_, "feature_names_in_"):
            self.feature_names_in_ = np.asarray(
                self.model_.feature_names_in_, dtype=object
            )
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def fit_classifier(estimator: BaseEstimator, X, y, groups=None) -> BaseEstimator:
    groups_array = None if groups is None else np.asarray(groups)
    if groups_array is not None and groups_array.shape[0] != len(y):
        raise ValueError("groups must contain exactly one value per training sample.")
    if (
        groups_array is not None
        and isinstance(estimator, SVC)
        and bool(estimator.probability)
    ):
        raise ValueError(
            "SVC(probability=True) uses internal sample-level cross-validation. Use probability=False with GroupAwareCalibratedClassifier for grouped evaluation."
        )
    if groups_array is not None and isinstance(estimator, CalibratedClassifierCV):
        cv = estimator.cv
        if cv is None or isinstance(cv, (int, np.integer)):
            n_splits = 5 if cv is None else int(cv)
            base_estimator = getattr(
                estimator, "estimator", getattr(estimator, "base_estimator", None)
            )
            if base_estimator is None:
                raise TypeError(
                    "CalibratedClassifierCV does not expose its base estimator."
                )
            helper = GroupAwareCalibratedClassifier(
                base_estimator,
                method=estimator.method,
                n_splits=n_splits,
                random_state=42,
            )
            estimator.set_params(cv=helper._splits(X, y, groups_array))
    fit = estimator.fit
    try:
        parameters = inspect.signature(fit).parameters
    except (TypeError, ValueError):
        parameters = {}
    if groups_array is not None and "groups" in parameters:
        fit(X, y, groups=groups_array)
    else:
        fit(X, y)
    return estimator


def _calibrated(estimator: BaseEstimator) -> BaseEstimator:
    return GroupAwareCalibratedClassifier(
        estimator, method="sigmoid", n_splits=3, random_state=42
    )


def _require_probability_estimator(estimator: BaseEstimator) -> BaseEstimator:
    if callable(getattr(estimator, "predict_proba", None)):
        return estimator
    if callable(getattr(estimator, "decision_function", None)):
        return _calibrated(estimator)
    raise TypeError(
        f"{type(estimator).__name__} provides neither predict_proba nor decision_function and cannot be used as a probability-producing learner."
    )


def validate_model_specs(models: Any, *, context: str = "MODELS") -> None:

    if isinstance(models, dict):
        if not models:
            raise TypeError(f"{context} must not be empty.")
        for key, value in models.items():
            validate_model_specs(value, context=f"{context}[{key!r}]")
        return
    if isinstance(models, (str, bytes)):
        raise TypeError(
            f"{context} must contain explicit (name, estimator) pairs; "
            f"string model aliases such as {models!r} are not allowed."
        )
    try:
        items = tuple(models)
    except TypeError as exc:
        raise TypeError(
            f"{context} must be a sequence of (name, estimator) pairs."
        ) from exc
    if not items:
        raise TypeError(f"{context} must contain at least one model specification.")
    for index, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                f"{context}[{index}] must be exactly (name, estimator); got {item!r}."
            )
        name, spec = item
        if not str(name).strip():
            raise TypeError(f"{context}[{index}] has an empty model name.")
        if not isinstance(spec, BaseEstimator):
            raise TypeError(
                f"{context}[{index}] ({name!r}) must contain an instantiated "
                "scikit-learn BaseEstimator; factories and string aliases are not allowed."
            )


def _learner_name(item: Any) -> str:
    if isinstance(item, (str, bytes)):
        raise TypeError(
            "String model aliases are not allowed in sweep configs. Define models "
            "as explicit (name, estimator) pairs, for example "
            "('RF_1000_msl5', RandomForestClassifier(...))."
        )
    if not isinstance(item, tuple) or len(item) != 2:
        raise TypeError(
            f"Model specification must be (name, estimator_or_factory); got {item!r}."
        )
    return str(item[0])


def _learner_factory(
    item: Any, task: str = "classification"
) -> tuple[str, Callable[[], BaseEstimator]]:
    task = str(task).strip().casefold()

    def validate(estimator: BaseEstimator) -> BaseEstimator:
        if task == "regression":
            if not callable(getattr(estimator, "predict", None)):
                raise TypeError(
                    f"{type(estimator).__name__} does not provide predict()."
                )
            return estimator
        return _require_probability_estimator(estimator)

    if isinstance(item, (str, bytes)):
        raise TypeError(
            "String model aliases are not allowed in sweep configs. Define models "
            "as explicit (name, estimator) pairs."
        )
    if isinstance(item, tuple) and len(item) == 2:
        name, spec = item
        if isinstance(spec, BaseEstimator):
            return str(name), lambda spec=spec: validate(clone(spec))
        if callable(spec):

            def factory(spec=spec):
                estimator = spec()
                if not isinstance(estimator, BaseEstimator):
                    raise TypeError(
                        f"Learner factory for {name!r} must return a scikit-learn BaseEstimator."
                    )
                return validate(estimator)

            return str(name), factory
        raise TypeError(
            f"Learner tuple for {name!r} must contain an estimator or factory."
        )
    raise TypeError(
        f"Model specification must be (name, estimator_or_factory); got {item!r}."
    )


class FLAMLClassifier(BaseEstimator):
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

    def fit(self, X, y, groups=None):
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
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape[0] != y.shape[0]:
                raise ValueError(
                    "groups must contain exactly one value per training sample."
                )
            if len(np.unique(groups)) < 2:
                raise ValueError(
                    "Group-aware FLAML fitting requires at least two distinct training groups."
                )
            fit_kwargs["groups"] = groups
            fit_kwargs["split_type"] = "group"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.model_.fit(**fit_kwargs)
        if not callable(getattr(self.model_, "predict_proba", None)):
            raise TypeError(
                "FLAML selected a classifier that does not provide predict_proba."
            )
        return self

    def predict(self, X):
        X_frame = _as_named_frame(X, list(self.feature_names_in_))
        return self.model_.predict(X_frame)

    def predict_proba(self, X):
        X_frame = _as_named_frame(X, list(self.feature_names_in_))
        return self.model_.predict_proba(X_frame)


def build_learner(name: str, task: str = "classification") -> BaseEstimator:
    base = str(name).lower()
    if str(task).strip().casefold() == "regression":
        if base in {"rf", "rf_500_msl5"}:
            return RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, n_jobs=1, random_state=42
            )
        if base in {"rf_1000_msl5", "baseline_rf"}:
            return RandomForestRegressor(
                n_estimators=1000, min_samples_leaf=5, n_jobs=1, random_state=42
            )
        if base in {"rf_200", "rf_fast"}:
            return RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=42
            )
        if base in {"et", "extratrees"}:
            return ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=3, n_jobs=1, random_state=42
            )
        if base in {"histgb", "histgradientboosting"}:
            return HistGradientBoostingRegressor(random_state=42)
        if base in {"ridge", "ridge_a1"}:
            return Ridge(alpha=1.0)
        if base == "ridge_a01":
            return Ridge(alpha=0.1)
        if base in {"elasticnet", "enet"}:
            return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
        raise ValueError(
            f"Unknown regression learner {name!r}. Provide (name, estimator) in the sweep config to add it."
        )
    if base in {"lr", "lr_l2", "lr_l2_bal", "logistic"}:
        return _logistic_regression(
            kind="l2", max_iter=2000, class_weight="balanced", random_state=42
        )
    if base in {"lr_l1", "lr_l1_bal"}:
        return _logistic_regression(
            kind="l1", max_iter=3000, class_weight="balanced", random_state=42
        )
    if base in {"ridge", "ridge_a1", "ridgeclassifier"}:
        return _calibrated(RidgeClassifier(alpha=1.0, random_state=42))
    if base == "ridge_a01":
        return _calibrated(RidgeClassifier(alpha=0.1, random_state=42))
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
        return _calibrated(
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=False,
                class_weight="balanced",
                random_state=42,
            )
        )
    if base in {"calib_lsvc", "caliblsvc_c1_sig"}:
        return _calibrated(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
        )
    if base in {"knn", "knn_default"}:
        return KNeighborsClassifier()
    if base in {"nearestcentroid", "nearestcentroid_raw"}:
        return _require_probability_estimator(NearestCentroid())
    if base in {"gnb", "gaussiannb"}:
        return GaussianNB()
    if base in {"bnb", "bernoullinb"}:
        return BernoulliNB()
    if base in {"mnb", "multinomialnb"}:
        return MultinomialNB()
    if base == "lda":
        return LinearDiscriminantAnalysis()
    if base == "qda":
        return QuadraticDiscriminantAnalysis(reg_param=0.1)
    if base in {"sgd_log", "sgd"}:
        return SGDClassifier(loss="log_loss", penalty="l2", random_state=42)
    if base in {"pa", "pa_default"}:
        return _calibrated(PassiveAggressiveClassifier(random_state=42))
    if base in {"dt", "decisiontree"}:
        return DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    if base in {"flaml", "flaml_600s", "automl"}:
        return FLAMLClassifier(
            time_budget=600, metric="roc_auc", n_jobs=1, random_state=42
        )
    if base == "siamcat":
        from .siamcat import SIAMCATClassifier

        return _calibrated(SIAMCATClassifier())
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
