from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from skbio.stats.composition import alr as skbio_alr
from skbio.stats.composition import clr as skbio_clr
from skbio.stats.composition import closure as skbio_closure
from skbio.stats.composition import ilr as skbio_ilr
from skbio.stats.composition import multi_replace as skbio_multi_replace
from skbio.stats.composition import rclr as skbio_rclr
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .utils import _as_float_matrix


@dataclass(frozen=True)
class TransformationLabel:
    key: str
    aliases: tuple[str, ...] = ()
    category: str = "core"

    @property
    def abbreviation(self) -> str:
        return self.key


TRANSFORMATION_LABELS: tuple[TransformationLabel, ...] = (
    TransformationLabel("identity", ("raw", "unchanged", "identity")),
    TransformationLabel(
        "relative_abundance", ("none", "ra", "relative", "tss", "relative abundance")
    ),
    TransformationLabel(
        "presence_absence",
        ("binary", "pa", "p/a", "presence-absence", "presence absence"),
    ),
    TransformationLabel("hellinger", ("sqrt", "sqrt_abundance", "hellinger")),
    TransformationLabel(
        "arcsine_sqrt", ("arcsin_sqrt", "asin_sqrt", "asin", "arcsine sqrt")
    ),
    TransformationLabel(
        "log10_relative_abundance_half_min_pseudocount",
        ("log_tss", "logtss", "log_tss_floor", "log10", "log10-ra"),
    ),
    TransformationLabel(
        "centered_log_ratio_multiplicative_replacement",
        (
            "clr",
            "scikit-bio_clr",
            "skbio_clr",
            "scikit_bio_clr",
            "scikitbio_clr",
            "clr_mult",
            "clr-mult",
        ),
    ),
    TransformationLabel(
        "robust_centered_log_ratio_zero_preserving",
        ("rclr", "robust_clr", "rclr"),
    ),
    TransformationLabel(
        "additive_log_ratio_first_reference",
        (
            "alr",
            "scikit-bio_alr",
            "skbio_alr",
            "scikit_bio_alr",
            "scikitbio_alr",
            "alr_first",
            "alr-first",
            "alr_last",
        ),
    ),
    TransformationLabel(
        "isometric_log_ratio_egozcue",
        (
            "ilr",
            "scikit-bio_ilr",
            "skbio_ilr",
            "scikit_bio_ilr",
            "scikitbio_ilr",
            "ilr_egoz",
            "ilr_egozcue",
            "ilr_seq",
            "ilr-egoz.",
        ),
    ),
    TransformationLabel(
        "standardized_centered_log_ratio_multiplicative_replacement",
        ("clr_std", "clr_epsilon_z", "clr_eps_z", "clr-mult+z"),
        "extended",
    ),
    TransformationLabel(
        "standardized_isometric_log_ratio_egozcue",
        ("ilr_std", "ilr_seq_z", "ilr-egoz.+z"),
        "extended",
    ),
    TransformationLabel(
        "yeo_johnson_relative_abundance",
        ("power", "yeo_johnson", "power_yj", "yeo-j-ra"),
        "extended",
    ),
    TransformationLabel(
        "quantile_normal_relative_abundance",
        ("quantile", "rank_gauss", "quantile_normal", "qnorm", "qnorm-ra"),
        "extended",
    ),
    TransformationLabel(
        "robust_scaled_relative_abundance",
        ("robust", "robust_z", "robust-ra"),
        "extended",
    ),
    TransformationLabel(
        "within_sample_fractional_rank",
        ("rank_frac", "row_rank", "rank", "rank_within_sample", "row-rank"),
        "extended",
    ),
    TransformationLabel(
        "training_ecdf_rank",
        ("rank_col", "ecdf_rank", "ecdf-rank"),
        "extended",
    ),
    TransformationLabel(
        "prevalence_weighted_relative_abundance",
        ("prev_weighted", "prevalence_weighted", "prev_wt", "prev-wt-ra"),
        "extended",
    ),
    TransformationLabel(
        "pairwise_log_ratio_multiplicative_replacement_500",
        ("pairwise_logratio", "pair_logr_500", "pair-logr-500"),
        "extended",
    ),
)


TRANSFORMATION_SPACE = TRANSFORMATION_LABELS

_TRANSFORMATION_LABEL_BY_KEY: dict[str, TransformationLabel] = {}
for _label in TRANSFORMATION_LABELS:
    _TRANSFORMATION_LABEL_BY_KEY[_label.key.lower()] = _label
    for _alias in _label.aliases:
        _TRANSFORMATION_LABEL_BY_KEY[str(_alias).lower()] = _label

_REMOVED_TRANSFORMATIONS = {
    "log": "log10_relative_abundance_half_min_pseudocount",
    "log1p": "log10_relative_abundance_half_min_pseudocount",
    "ln1p": "log10_relative_abundance_half_min_pseudocount",
    "log2": "log10_relative_abundance_half_min_pseudocount",
    "log2_1p": "log10_relative_abundance_half_min_pseudocount",
    "log_std": "standardized_centered_log_ratio_multiplicative_replacement",
    "row_log_z": "standardized_centered_log_ratio_multiplicative_replacement",
    "log_unit": "log10_relative_abundance_half_min_pseudocount",
    "row_log_l2": "log10_relative_abundance_half_min_pseudocount",
    "symlog": "log10_relative_abundance_half_min_pseudocount",
    "zi_log": "log10_relative_abundance_half_min_pseudocount",
    "zero_inflated_log": "log10_relative_abundance_half_min_pseudocount",
    "zscore": "robust_scaled_relative_abundance",
    "row_z": "robust_scaled_relative_abundance",
    "bclr": "centered_log_ratio_multiplicative_replacement",
    "clr_epsilon": "centered_log_ratio_multiplicative_replacement",
    "clr_eps": "centered_log_ratio_multiplicative_replacement",
    "rank_std": "within_sample_fractional_rank",
    "row_rank_z": "within_sample_fractional_rank",
    "rank_unit": "within_sample_fractional_rank",
    "row_rank_unit": "within_sample_fractional_rank",
}


def transformation_label(name: str) -> TransformationLabel:
    key = str(name).strip().lower()
    label = _TRANSFORMATION_LABEL_BY_KEY.get(key)
    if label is not None:
        return label
    if key in _REMOVED_TRANSFORMATIONS:
        replacement = _REMOVED_TRANSFORMATIONS[key]
        raise KeyError(f"Transformation {name!r} was removed. Use {replacement!r}.")
    return TransformationLabel(str(name), (), "custom")


def transformation_space_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "count_transformation": label.key,
                "category": label.category,
            }
            for label in TRANSFORMATION_LABELS
        ]
    )


def _matrix(
    X: np.ndarray, *, nonnegative: bool = False, nonzero_rows: bool = False
) -> np.ndarray:
    out = np.asarray(_as_float_matrix(X), dtype=np.float64)
    if out.ndim != 2:
        raise ValueError("Abundance transformations require a two-dimensional matrix.")
    if not np.isfinite(out).all():
        raise ValueError("Abundance matrix contains non-finite values.")
    if nonnegative and np.any(out < 0):
        raise ValueError("Abundance transformations require non-negative values.")
    if nonzero_rows and np.any(out.sum(axis=1) <= 0):
        raise ValueError(
            "Abundance transformations require every sample to contain at least one positive feature."
        )
    return out


def _relative_abundance(X: np.ndarray) -> np.ndarray:
    out = _matrix(X, nonnegative=True)
    sums = out.sum(axis=1, keepdims=True)
    return np.divide(out, sums, out=np.zeros_like(out), where=sums > 0)


def _positive_composition(X: np.ndarray) -> np.ndarray:
    raw = _matrix(X, nonnegative=True, nonzero_rows=True)
    rel = raw / raw.sum(axis=1, keepdims=True)
    return np.asarray(skbio_multi_replace(skbio_closure(rel)), dtype=np.float64)


def _finite_output(X: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError("A transformation returned a non-matrix result.")
    if not np.isfinite(out).all():
        raise ValueError("A transformation returned non-finite values.")
    return out.astype(np.float32, copy=False)


class _BuiltinTransformer:
    def __init__(self, name: str, random_state: int = 42):
        self.name = transformation_label(name).key
        self.random_state = int(random_state)
        self.pseudocount_: float | None = None
        self.scaler_: Any | None = None
        self.variable_mask_: np.ndarray | None = None
        self.sorted_columns_: list[np.ndarray] | None = None
        self.prevalence_: np.ndarray | None = None
        self.pair_i_: np.ndarray | None = None
        self.pair_j_: np.ndarray | None = None

    def _base_transform(self, X: np.ndarray) -> np.ndarray:
        name = self.name
        if name == "identity":
            return _matrix(X)
        if name == "relative_abundance":
            return _relative_abundance(X)
        if name == "presence_absence":
            return (_matrix(X, nonnegative=True) > 0).astype(np.float64)
        if name == "hellinger":
            return np.sqrt(_relative_abundance(X))
        if name == "arcsine_sqrt":
            return np.arcsin(np.sqrt(_relative_abundance(X)))
        if name == "centered_log_ratio_multiplicative_replacement":
            return np.asarray(skbio_clr(_positive_composition(X)), dtype=np.float64)
        if name == "robust_centered_log_ratio_zero_preserving":
            raw = _matrix(X, nonnegative=True, nonzero_rows=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.asarray(skbio_rclr(raw), dtype=np.float64)
            return np.where(np.isnan(out), 0.0, out)
        if name == "additive_log_ratio_first_reference":
            return np.asarray(
                skbio_alr(_positive_composition(X), ref_idx=0), dtype=np.float64
            )
        if name == "isometric_log_ratio_egozcue":
            return np.asarray(skbio_ilr(_positive_composition(X)), dtype=np.float64)
        if name == "within_sample_fractional_rank":
            raw = _matrix(X, nonnegative=True)
            if raw.shape[1] == 0:
                return raw
            ranks = np.apply_along_axis(rankdata, 1, raw)
            return ranks / (raw.shape[1] + 1.0)
        raise KeyError(name)

    def fit(self, X: np.ndarray) -> "_BuiltinTransformer":
        name = self.name
        if name == "log10_relative_abundance_half_min_pseudocount":
            _matrix(X, nonnegative=True, nonzero_rows=True)
            rel = _relative_abundance(X)
            positive = rel[rel > 0]
            if positive.size == 0:
                raise ValueError(
                    "Cannot estimate a log-relative-abundance pseudocount without positive abundances."
                )
            self.pseudocount_ = float(positive.min()) / 2.0
        elif name == "standardized_centered_log_ratio_multiplicative_replacement":
            base = np.asarray(skbio_clr(_positive_composition(X)), dtype=np.float64)
            self.scaler_ = StandardScaler().fit(base)
        elif name == "standardized_isometric_log_ratio_egozcue":
            base = np.asarray(skbio_ilr(_positive_composition(X)), dtype=np.float64)
            self.scaler_ = StandardScaler().fit(base)
        elif name == "yeo_johnson_relative_abundance":
            base = _relative_abundance(X)
            mask = np.ptp(base, axis=0) > 0
            self.variable_mask_ = mask
            if np.any(mask):
                self.scaler_ = PowerTransformer(
                    method="yeo-johnson", standardize=True
                ).fit(base[:, mask])
        elif name == "quantile_normal_relative_abundance":
            base = _relative_abundance(X)
            self.scaler_ = QuantileTransformer(
                n_quantiles=max(2, min(1000, base.shape[0])),
                output_distribution="normal",
                random_state=self.random_state,
                subsample=None,
            ).fit(base)
        elif name == "robust_scaled_relative_abundance":
            self.scaler_ = RobustScaler().fit(_relative_abundance(X))
        elif name == "training_ecdf_rank":
            raw = _matrix(X, nonnegative=True)
            self.sorted_columns_ = [np.sort(raw[:, j]) for j in range(raw.shape[1])]
        elif name == "prevalence_weighted_relative_abundance":
            raw = _matrix(X, nonnegative=True)
            self.prevalence_ = (raw > 0).mean(axis=0).astype(np.float64)
        elif name == "pairwise_log_ratio_multiplicative_replacement_500":
            raw = _matrix(X, nonnegative=True, nonzero_rows=True)
            n_features = raw.shape[1]
            if n_features < 2:
                raise ValueError(
                    "Pairwise log-ratio transformation requires at least two features."
                )
            pairs = np.array(np.triu_indices(n_features, k=1)).T
            if len(pairs) > 500:
                rng = np.random.RandomState(self.random_state)
                pairs = pairs[rng.choice(len(pairs), 500, replace=False)]
            self.pair_i_ = pairs[:, 0]
            self.pair_j_ = pairs[:, 1]
        else:
            self._base_transform(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        name = self.name
        if name in {
            "identity",
            "relative_abundance",
            "presence_absence",
            "hellinger",
            "arcsine_sqrt",
            "centered_log_ratio_multiplicative_replacement",
            "robust_centered_log_ratio_zero_preserving",
            "additive_log_ratio_first_reference",
            "isometric_log_ratio_egozcue",
            "within_sample_fractional_rank",
        }:
            return _finite_output(self._base_transform(X))
        if name == "log10_relative_abundance_half_min_pseudocount":
            if self.pseudocount_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            _matrix(X, nonnegative=True, nonzero_rows=True)
            return _finite_output(np.log10(_relative_abundance(X) + self.pseudocount_))
        if name == "standardized_centered_log_ratio_multiplicative_replacement":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            base = np.asarray(skbio_clr(_positive_composition(X)), dtype=np.float64)
            return _finite_output(self.scaler_.transform(base))
        if name == "standardized_isometric_log_ratio_egozcue":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            base = np.asarray(skbio_ilr(_positive_composition(X)), dtype=np.float64)
            return _finite_output(self.scaler_.transform(base))
        if name == "yeo_johnson_relative_abundance":
            base = _relative_abundance(X)
            out = np.zeros_like(base, dtype=np.float64)
            mask = self.variable_mask_
            if mask is None:
                raise RuntimeError("Transformation has not been fitted.")
            if np.any(mask):
                if self.scaler_ is None:
                    raise RuntimeError("Transformation has not been fitted.")
                out[:, mask] = self.scaler_.transform(base[:, mask])
            return _finite_output(out)
        if name == "quantile_normal_relative_abundance":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _finite_output(self.scaler_.transform(_relative_abundance(X)))
        if name == "robust_scaled_relative_abundance":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _finite_output(self.scaler_.transform(_relative_abundance(X)))
        if name == "training_ecdf_rank":
            raw = _matrix(X, nonnegative=True)
            if self.sorted_columns_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            if raw.shape[1] != len(self.sorted_columns_):
                raise ValueError(
                    "Feature count differs from the fitted ECDF transformation."
                )
            out = np.zeros_like(raw, dtype=np.float64)
            for j, sorted_col in enumerate(self.sorted_columns_):
                n = max(len(sorted_col), 1)
                out[:, j] = np.searchsorted(
                    sorted_col, raw[:, j], side="right"
                ) / float(n)
            return _finite_output(out)
        if name == "prevalence_weighted_relative_abundance":
            if self.prevalence_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            rel = _relative_abundance(X)
            weighted = rel * self.prevalence_[None, :]
            sums = weighted.sum(axis=1, keepdims=True)
            out = np.divide(weighted, sums, out=np.zeros_like(weighted), where=sums > 0)
            return _finite_output(out)
        if name == "pairwise_log_ratio_multiplicative_replacement_500":
            if self.pair_i_ is None or self.pair_j_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            comp = _positive_composition(X)
            return _finite_output(np.log(comp[:, self.pair_i_] / comp[:, self.pair_j_]))
        raise KeyError(f"Unknown abundance transformation {name!r}.")

    def apply_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X_tr)
        return self.transform(X_tr), self.transform(X_te)


class Transform:
    __slots__ = ("name", "_fn", "_is_bw")

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        is_bw: bool = False,
        abbreviation: str | None = None,
    ):
        requested = str(name).strip()
        info = transformation_label(requested)
        if fn is None and info.category == "custom":
            raise KeyError(
                f"Unknown abundance transformation {requested!r}. Provide a callable for a custom transformation."
            )
        self.name = info.key
        self._fn = fn
        self._is_bw = bool(is_bw)

    def apply(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fn is None:
            return _BuiltinTransformer(self.name, random_state=42).apply_pair(
                X_tr, X_te
            )
        if self._is_bw:
            a, b = self._fn(X_tr, X_te)
        else:
            a, b = self._fn(X_tr), self._fn(X_te)
        return _finite_output(a), _finite_output(b)


Transformation = Transform
_Transform = Transform
_Transformation = Transform


def build_count_transformations(
    *, include_inactive: bool = True
) -> list[Transformation]:
    items = [Transformation(label.key) for label in TRANSFORMATION_LABELS]
    if include_inactive:
        return items
    return [x for x in items if x.name == "arcsine_sqrt"]


class CountTransformation:
    def __init__(
        self, name: str, pseudo_count: float | None = None, random_state: int = 42
    ):
        self.name = transformation_label(str(name)).key
        self.pseudo_count = None if pseudo_count is None else float(pseudo_count)
        self.random_state = int(random_state)
        self._impl: _BuiltinTransformer | None = None

    def fit(self, X: np.ndarray) -> "CountTransformation":
        self._impl = _BuiltinTransformer(self.name, random_state=self.random_state).fit(
            X
        )
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        return self._impl.transform(X)

    def apply_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X_tr)
        return self.apply(X_tr), self.apply(X_te)

    def fit_apply(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.apply(X)


def _count_transformation_name(item: Any) -> str:
    if isinstance(item, Transform):
        name = item.name
    elif hasattr(item, "name") and hasattr(item, "apply"):
        name = str(getattr(item, "name"))
    else:
        name = item if isinstance(item, str) else str(item[0])
    return transformation_label(str(name)).key


def _count_transformation_spec(item: Any) -> tuple[str, Any | None]:
    if isinstance(item, Transform):
        return transformation_label(item.name).key, item
    if hasattr(item, "name") and hasattr(item, "apply"):
        return transformation_label(str(getattr(item, "name"))).key, item
    if isinstance(item, tuple):
        return transformation_label(str(item[0])).key, item[1]
    return transformation_label(str(item)).key, None


def _count_transformation_reporting_fields(item: Any) -> dict[str, str]:
    return {}


class CountTransformationAdapter:
    def __init__(self, name: str, spec: Any = None, *, random_state: int = 42):
        self.name = transformation_label(str(name)).key
        self.spec = spec
        self.random_state = int(random_state)
        self.obj: Any | None = None

    def _make(self) -> Any:
        if self.spec is None:
            return CountTransformation(self.name, random_state=self.random_state)
        if isinstance(self.spec, Transform):
            if self.spec._fn is None:
                return CountTransformation(
                    self.spec.name, random_state=self.random_state
                )
            return self.spec
        if isinstance(self.spec, BaseEstimator):
            return clone(self.spec)
        if hasattr(self.spec, "fit") or hasattr(self.spec, "apply"):
            return self.spec
        if callable(self.spec) and _callable_looks_like_factory(self.spec):
            return self.spec()
        return self.spec

    def fit(self, X: np.ndarray) -> "CountTransformationAdapter":
        obj = self._make()
        if hasattr(obj, "fit"):
            obj.fit(_as_float_matrix(X))
        self.obj = obj
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self.obj is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        obj = self.obj
        if isinstance(obj, Transform):
            raise RuntimeError(
                "Callable Transformation objects must be applied to a train/test pair."
            )
        if hasattr(obj, "apply"):
            return _finite_output(obj.apply(_as_float_matrix(X)))
        if hasattr(obj, "transform"):
            return _finite_output(obj.transform(_as_float_matrix(X)))
        if callable(obj):
            return _finite_output(obj(_as_float_matrix(X)))
        raise TypeError(
            f"Custom count transformation {self.name!r} must be callable or provide fit/apply or fit/transform."
        )

    def apply_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.spec, Transform) and self.spec._fn is not None:
            return self.spec.apply(_as_float_matrix(X_tr), _as_float_matrix(X_te))
        if callable(self.spec) and _callable_accepts_two_required(self.spec):
            a, b = self.spec(_as_float_matrix(X_tr), _as_float_matrix(X_te))
            return _finite_output(a), _finite_output(b)
        self.fit(X_tr)
        return self.apply(X_tr), self.apply(X_te)

    def fit_apply(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.apply(X)


def _callable_looks_like_factory(fn: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect._empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    return len(required) == 0


def _callable_accepts_two_required(fn: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is inspect._empty
    ]
    return len(positional) >= 2


def _count_transformation_factory(
    item: Any, *, random_state: int
) -> tuple[str, Callable[[], CountTransformationAdapter]]:
    if isinstance(item, Transform) or (
        hasattr(item, "name") and hasattr(item, "apply")
    ):
        name = transformation_label(str(getattr(item, "name"))).key
        return (
            name,
            lambda item=item, name=name, random_state=random_state: (
                CountTransformationAdapter(name, item, random_state=random_state)
            ),
        )
    if isinstance(item, tuple):
        raw_name, spec = item
        name = transformation_label(str(raw_name)).key
        return (
            name,
            lambda name=name, spec=spec, random_state=random_state: (
                CountTransformationAdapter(name, spec, random_state=random_state)
            ),
        )
    name = transformation_label(str(item)).key
    return (
        name,
        lambda name=name, random_state=random_state: CountTransformationAdapter(
            name, random_state=random_state
        ),
    )
