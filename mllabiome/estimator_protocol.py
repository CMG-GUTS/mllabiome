from __future__ import annotations

import copy
from typing import Any, Protocol, cast

from sklearn.base import clone


class EstimatorLike(Protocol):
    def fit(self, X: Any, y: Any = ..., **kwargs: Any) -> Any: ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> Any: ...


def is_estimator_instance(value: Any) -> bool:
    return (
        not isinstance(value, type)
        and callable(getattr(value, "fit", None))
        and callable(getattr(value, "get_params", None))
        and callable(getattr(value, "set_params", None))
    )


def clone_estimator(estimator: EstimatorLike) -> EstimatorLike:
    if not is_estimator_instance(estimator):
        raise TypeError(
            f"{type(estimator).__name__} is not a scikit-learn-compatible estimator."
        )
    try:
        return cast(EstimatorLike, clone(estimator))
    except (TypeError, RuntimeError):
        try:
            params = copy.deepcopy(estimator.get_params(deep=False))
            cloned = type(estimator)(**params)
        except Exception as exc:
            raise TypeError(
                f"Could not construct a fresh {type(estimator).__name__} "
                "from its parameters."
            ) from exc
        if not is_estimator_instance(cloned):
            raise TypeError(
                f"Cloned {type(estimator).__name__} is not a "
                "scikit-learn-compatible estimator."
            )
        return cast(EstimatorLike, cloned)
