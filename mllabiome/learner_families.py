from __future__ import annotations

import importlib
import json
from functools import lru_cache
from typing import Any

_DEEP_PATHS = (
    "mllabiome.tabpfn.TabPFNClassifier",
    "mllabiome.tabdpt.TabDPTClassifier",
    "mllabiome.causilo.CausiloClassifier",
    "mllabiome.kumo.KumoTabularClassifier",
    "mllabiome.limix.LimiXClassifier",
    "mllabiome.limix2.LimiX2Classifier",
    "mllabiome.tabicl.TabICLClassifier",
    "mllabiome.realmlp.RealMLPClassifier",
    "mllabiome.tabfm.TabFMClassifier",
    "mllabiome.tabm.TabMClassifier",
    "mllabiome.exaone.EXAONETabularClassifier",
    "sklearn.neural_network._multilayer_perceptron.MLPClassifier",
)

_ML_PATHS = (
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.ensemble._forest.ExtraTreesClassifier",
    "sklearn.ensemble._gb.GradientBoostingClassifier",
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
    "sklearn.ensemble._bagging.BaggingClassifier",
    "sklearn.ensemble._weight_boosting.AdaBoostClassifier",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.tree._classes.ExtraTreeClassifier",
    "sklearn.linear_model._logistic.LogisticRegression",
    "sklearn.linear_model._stochastic_gradient.SGDClassifier",
    "sklearn.linear_model._ridge.RidgeClassifier",
    "sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier",
    "sklearn.svm._classes.SVC",
    "sklearn.svm._classes.LinearSVC",
    "sklearn.neighbors._classification.KNeighborsClassifier",
    "sklearn.neighbors._nearest_centroid.NearestCentroid",
    "sklearn.naive_bayes.GaussianNB",
    "sklearn.naive_bayes.BernoulliNB",
    "sklearn.naive_bayes.MultinomialNB",
    "sklearn.naive_bayes.ComplementNB",
    "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis",
    "xgboost.sklearn.XGBClassifier",
    "lightgbm.sklearn.LGBMClassifier",
    "catboost.core.CatBoostClassifier",
    "mllabiome.baseline_rf.BaselineRFClassifier",
    "mllabiome.ridge_softmax.RidgeSoftmaxClassifier",
)

_SINGLE_WRAPPERS = frozenset(
    {
        "sklearn.pipeline.Pipeline",
        "sklearn.model_selection._search.GridSearchCV",
        "sklearn.model_selection._search.RandomizedSearchCV",
        "sklearn.calibration.CalibratedClassifierCV",
        "sklearn.multiclass.OneVsRestClassifier",
        "sklearn.multiclass.OneVsOneClassifier",
        "sklearn.multioutput.MultiOutputClassifier",
        "sklearn.feature_selection._rfe.RFE",
        "sklearn.feature_selection._rfe.RFECV",
        "mllabiome.rank_calibration.GroupedRankCalibratedClassifier",
    }
)

_MULTI_WRAPPERS = frozenset(
    {
        "sklearn.ensemble._voting.VotingClassifier",
        "sklearn.ensemble._stacking.StackingClassifier",
    }
)

_REGISTERED: dict[type, str] = {}


def register_learner_family(estimator_class: type, family: str) -> None:
    if not isinstance(estimator_class, type):
        raise TypeError("estimator_class must be a class.")
    if family not in {"deep_learning", "machine_learning"}:
        raise ValueError("family must be 'deep_learning' or 'machine_learning'.")
    _REGISTERED[estimator_class] = family


@lru_cache(maxsize=512)
def _resolve_class(path: str) -> type | None:
    components = path.split(".")
    for index in range(len(components) - 1, 0, -1):
        try:
            value: Any = importlib.import_module(".".join(components[:index]))
            for component in components[index:]:
                value = getattr(value, component)
            return value if isinstance(value, type) else None
        except (ImportError, AttributeError, RuntimeError):
            continue
    return None


def _type_family(path: str) -> str | None:
    for registered, family in reversed(tuple(_REGISTERED.items())):
        identity = f"{registered.__module__}.{registered.__qualname__}"
        if identity == path:
            return family
    if path in _DEEP_PATHS:
        return "deep_learning"
    if path in _ML_PATHS:
        return "machine_learning"
    cls = _resolve_class(path)
    if cls is None:
        return None
    for registered, family in reversed(tuple(_REGISTERED.items())):
        if issubclass(cls, registered):
            return family
    for paths, family in (
        (_DEEP_PATHS, "deep_learning"),
        (_ML_PATHS, "machine_learning"),
    ):
        for base_path in paths:
            base = _resolve_class(base_path)
            if base is not None and issubclass(cls, base):
                return family
    return None


def _nested_estimator(value: Any) -> dict[str, Any] | None:
    if (
        isinstance(value, dict)
        and isinstance(value.get("class"), str)
        and isinstance(value.get("params"), dict)
    ):
        return value
    return None


def _family_for_spec(
    path: str, params: dict[str, Any], visited: set[str]
) -> str | None:
    if not path or path in visited:
        return None
    if path in _SINGLE_WRAPPERS:
        child = None
        if path == "sklearn.pipeline.Pipeline":
            steps = params.get("steps")
            if isinstance(steps, list) and steps:
                last = steps[-1]
                if isinstance(last, (list, tuple)) and len(last) == 2:
                    child = _nested_estimator(last[1])
        else:
            for field in ("estimator", "base_estimator", "classifier"):
                child = _nested_estimator(params.get(field))
                if child is not None:
                    break
        if child is None:
            return None
        return _family_for_spec(str(child["class"]), child["params"], visited | {path})
    if path in _MULTI_WRAPPERS:
        estimators = params.get("estimators")
        if not isinstance(estimators, list) or not estimators:
            return None
        families = set()
        for item in estimators:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return None
            child = _nested_estimator(item[1])
            if child is None:
                return None
            families.add(
                _family_for_spec(str(child["class"]), child["params"], visited | {path})
            )
        return (
            next(iter(families))
            if len(families) == 1 and None not in families
            else None
        )
    return _type_family(path)


def learner_family_from_spec(class_path: str, learner_params: Any = None) -> str | None:
    if not isinstance(class_path, str) or not class_path.strip():
        return None
    if isinstance(learner_params, str):
        try:
            learner_params = json.loads(learner_params)
        except (TypeError, ValueError):
            return None
    if not isinstance(learner_params, dict):
        learner_params = {}
    return _family_for_spec(class_path.strip(), learner_params, set())


def learner_family(estimator: Any) -> str | None:
    if not hasattr(estimator, "get_params"):
        return None
    from .configs_sweep import _scientific_value

    return learner_family_from_spec(
        f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        _scientific_value(estimator.get_params(deep=True)),
    )
