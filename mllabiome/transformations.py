from __future__ import annotations
import inspect
import hashlib
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from skbio.stats.composition import alr as skbio_alr
from skbio.stats.composition import closure as skbio_closure
from skbio.stats.composition import clr as skbio_clr
from skbio.stats.composition import ilr as skbio_ilr
from skbio.stats.composition import multi_replace as skbio_multi_replace
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


@dataclass(frozen=True)
class TransformationCoordinate:
    name: str
    coordinate_type: str
    anchor_feature: str | None
    components: tuple[str, ...]
    coefficients: tuple[float, ...]
    exact_feature_identity: bool


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
        "additive_log_ratio_first_reference_multiplicative_replacement",
        (
            "alr",
            "scikit-bio_alr",
            "skbio_alr",
            "scikit_bio_alr",
            "scikitbio_alr",
            "alr_mult",
            "alr-mult",
        ),
        "log_ratio_coordinate",
    ),
    TransformationLabel(
        "isometric_log_ratio_egozcue_multiplicative_replacement",
        (
            "ilr",
            "scikit-bio_ilr",
            "skbio_ilr",
            "scikit_bio_ilr",
            "scikitbio_ilr",
            "ilr_egozcue",
            "ilr_mult",
            "ilr-mult",
        ),
        "log_ratio_coordinate",
    ),
    TransformationLabel(
        "standardized_centered_log_ratio_multiplicative_replacement",
        ("clr_std", "clr_epsilon_z", "clr_eps_z", "clr-mult+z"),
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
        "training_ecdf_rank", ("rank_col", "ecdf_rank", "ecdf-rank"), "extended"
    ),
    TransformationLabel(
        "prevalence_weighted_relative_abundance",
        ("prev_weighted", "prevalence_weighted", "prev_wt", "prev-wt-ra"),
        "extended",
    ),
)
TRANSFORMATION_SPACE = TRANSFORMATION_LABELS
_TRANSFORMATION_LABEL_BY_KEY: dict[str, TransformationLabel] = {}
for _label in TRANSFORMATION_LABELS:
    _TRANSFORMATION_LABEL_BY_KEY[_label.key.lower()] = _label
    for _alias in _label.aliases:
        _TRANSFORMATION_LABEL_BY_KEY[str(_alias).lower()] = _label


def transformation_label(name: str) -> TransformationLabel:
    key = str(name).strip().lower()
    label = _TRANSFORMATION_LABEL_BY_KEY.get(key)
    if label is not None:
        return label
    return TransformationLabel(str(name), (), "custom")


def transformation_space_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"count_transformation": label.key, "category": label.category}
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
    out = np.asarray(skbio_multi_replace(skbio_closure(rel)), dtype=np.float64)
    if out.ndim == 1 and raw.shape[0] == 1 and (out.shape[0] == raw.shape[1]):
        out = out.reshape(1, -1)
    if out.shape != raw.shape:
        raise ValueError(
            f"Multiplicative zero replacement changed the abundance matrix shape: expected {raw.shape}, got {out.shape}."
        )
    return out


def _clr_matrix(X: np.ndarray) -> np.ndarray:
    positive = _positive_composition(X)
    out = np.asarray(skbio_clr(positive), dtype=np.float64)
    if out.ndim == 1 and positive.shape[0] == 1 and (out.shape[0] == positive.shape[1]):
        out = out.reshape(1, -1)
    if out.shape != positive.shape:
        raise ValueError(
            f"CLR transformation changed the abundance matrix shape: expected {positive.shape}, got {out.shape}."
        )
    return out


def _egozcue_basis(n_components: int) -> np.ndarray:
    n = int(n_components)
    if n < 2:
        raise ValueError(
            "Log-ratio coordinate transformations require at least two features."
        )
    basis = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n - 1):
        scale = np.sqrt((i + 1) * (i + 2))
        basis[i, : i + 1] = 1.0 / scale
        basis[i, i + 1] = -(i + 1) / scale
    return basis


def _alr_matrix(X: np.ndarray) -> np.ndarray:
    positive = _positive_composition(X)
    if positive.shape[1] < 2:
        raise ValueError("ALR requires at least two features.")
    out = np.asarray(skbio_alr(positive, ref_idx=0), dtype=np.float64)
    if out.ndim == 1 and positive.shape[0] == 1:
        out = out.reshape(1, -1)
    expected = (positive.shape[0], positive.shape[1] - 1)
    if out.shape != expected:
        raise ValueError(
            f"ALR transformation returned shape {out.shape}; expected {expected}."
        )
    return out


def _ilr_matrix(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    positive = _positive_composition(X)
    if positive.shape[1] < 2:
        raise ValueError("ILR requires at least two features.")
    out = np.asarray(skbio_ilr(positive, basis=basis), dtype=np.float64)
    if out.ndim == 1 and positive.shape[0] == 1:
        out = out.reshape(1, -1)
    expected = (positive.shape[0], positive.shape[1] - 1)
    if out.shape != expected:
        raise ValueError(
            f"ILR transformation returned shape {out.shape}; expected {expected}."
        )
    return out


def _default_feature_names(n_features: int) -> list[str]:
    return [f"feature_{i + 1:04d}" for i in range(int(n_features))]


def _coordinate_digest(features: list[str], index: int) -> str:
    payload = "\x1f".join(features) + f"\x1e{int(index)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _finite_output(
    X: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None = None,
    context: str = "transformation",
) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError("A transformation returned a non-matrix result.")
    if not np.isfinite(out).all():
        raise ValueError("A transformation returned non-finite values.")
    if expected_shape is not None and out.shape != expected_shape:
        raise ValueError(
            f"{context} returned an unexpected matrix shape: expected {expected_shape}, got {out.shape}."
        )
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
        self.basis_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None

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
            return _clr_matrix(X)
        if name == "additive_log_ratio_first_reference_multiplicative_replacement":
            return _alr_matrix(X)
        if name == "isometric_log_ratio_egozcue_multiplicative_replacement":
            if self.basis_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _ilr_matrix(X, self.basis_)
        if name == "within_sample_fractional_rank":
            raw = _matrix(X, nonnegative=True)
            if raw.shape[1] == 0:
                return raw
            ranks = np.apply_along_axis(rankdata, 1, raw)
            return ranks / (raw.shape[1] + 1.0)
        raise KeyError(name)

    def fit(self, X: np.ndarray) -> "_BuiltinTransformer":
        raw = _matrix(X)
        self.n_features_in_ = int(raw.shape[1])
        name = self.name
        if name in {
            "additive_log_ratio_first_reference_multiplicative_replacement",
            "isometric_log_ratio_egozcue_multiplicative_replacement",
        }:
            if raw.shape[1] < 2:
                raise ValueError(
                    "Log-ratio coordinate transformations require at least two features."
                )
            _positive_composition(X)
            self.n_features_out_ = int(raw.shape[1] - 1)
            if name == "isometric_log_ratio_egozcue_multiplicative_replacement":
                self.basis_ = _egozcue_basis(raw.shape[1])
        else:
            self.n_features_out_ = int(raw.shape[1])
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
            base = _clr_matrix(X)
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
            abundance = _matrix(X, nonnegative=True)
            self.sorted_columns_ = [
                np.sort(abundance[:, j]) for j in range(abundance.shape[1])
            ]
        elif name == "prevalence_weighted_relative_abundance":
            abundance = _matrix(X, nonnegative=True)
            self.prevalence_ = (abundance > 0).mean(axis=0).astype(np.float64)
        elif name not in {
            "additive_log_ratio_first_reference_multiplicative_replacement",
            "isometric_log_ratio_egozcue_multiplicative_replacement",
        }:
            self._base_transform(X)
        return self

    def _expected_shape(self, X: np.ndarray) -> tuple[int, int]:
        raw = _matrix(X)
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError("Transformation has not been fitted.")
        if raw.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count differs from the fitted abundance transformation: expected {self.n_features_in_}, got {raw.shape[1]}."
            )
        return (raw.shape[0], self.n_features_out_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        name = self.name
        expected_shape = self._expected_shape(X)
        if name in {
            "identity",
            "relative_abundance",
            "presence_absence",
            "hellinger",
            "arcsine_sqrt",
            "centered_log_ratio_multiplicative_replacement",
            "additive_log_ratio_first_reference_multiplicative_replacement",
            "isometric_log_ratio_egozcue_multiplicative_replacement",
            "within_sample_fractional_rank",
        }:
            return _finite_output(
                self._base_transform(X),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "log10_relative_abundance_half_min_pseudocount":
            if self.pseudocount_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            _matrix(X, nonnegative=True, nonzero_rows=True)
            return _finite_output(
                np.log10(_relative_abundance(X) + self.pseudocount_),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "standardized_centered_log_ratio_multiplicative_replacement":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            base = _clr_matrix(X)
            return _finite_output(
                self.scaler_.transform(base),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
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
            return _finite_output(
                out,
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "quantile_normal_relative_abundance":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _finite_output(
                self.scaler_.transform(_relative_abundance(X)),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "robust_scaled_relative_abundance":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _finite_output(
                self.scaler_.transform(_relative_abundance(X)),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "training_ecdf_rank":
            abundance = _matrix(X, nonnegative=True)
            if self.sorted_columns_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            if abundance.shape[1] != len(self.sorted_columns_):
                raise ValueError(
                    "Feature count differs from the fitted ECDF transformation."
                )
            out = np.zeros_like(abundance, dtype=np.float64)
            for j, sorted_col in enumerate(self.sorted_columns_):
                n = max(len(sorted_col), 1)
                out[:, j] = np.searchsorted(
                    sorted_col, abundance[:, j], side="right"
                ) / float(n)
            return _finite_output(
                out,
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "prevalence_weighted_relative_abundance":
            if self.prevalence_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            rel = _relative_abundance(X)
            weighted = rel * self.prevalence_[None, :]
            sums = weighted.sum(axis=1, keepdims=True)
            out = np.divide(weighted, sums, out=np.zeros_like(weighted), where=sums > 0)
            return _finite_output(
                out,
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        raise KeyError(f"Unknown abundance transformation {name!r}.")

    def get_feature_names_out(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[str]:
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError("Transformation has not been fitted.")
        features = (
            _default_feature_names(self.n_features_in_)
            if input_features is None
            else [str(x) for x in list(input_features)]
        )
        if len(features) != self.n_features_in_:
            raise ValueError(
                f"Input feature-name count differs from the fitted abundance transformation: expected {self.n_features_in_}, got {len(features)}."
            )
        if self.name == "additive_log_ratio_first_reference_multiplicative_replacement":
            reference = features[0]
            return [f"ALR[{feature}/{reference}]" for feature in features[1:]]
        if self.name == "isometric_log_ratio_egozcue_multiplicative_replacement":
            return [
                f"ILR_{i + 1:04d}_{_coordinate_digest(features, i)}"
                for i in range(self.n_features_out_)
            ]
        return features

    def coordinate_metadata(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[TransformationCoordinate]:
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError("Transformation has not been fitted.")
        features = (
            _default_feature_names(self.n_features_in_)
            if input_features is None
            else [str(x) for x in list(input_features)]
        )
        names = self.get_feature_names_out(features)
        if self.name == "additive_log_ratio_first_reference_multiplicative_replacement":
            reference = features[0]
            return [
                TransformationCoordinate(
                    name=name,
                    coordinate_type="alr_logcontrast",
                    anchor_feature=None,
                    components=(feature, reference),
                    coefficients=(1.0, -1.0),
                    exact_feature_identity=False,
                )
                for name, feature in zip(names, features[1:])
            ]
        if self.name == "isometric_log_ratio_egozcue_multiplicative_replacement":
            if self.basis_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            coordinates: list[TransformationCoordinate] = []
            for i, name in enumerate(names):
                row = np.asarray(self.basis_[i], dtype=float)
                mask = np.abs(row) > 1e-15
                coordinates.append(
                    TransformationCoordinate(
                        name=name,
                        coordinate_type="ilr_balance",
                        anchor_feature=None,
                        components=tuple(
                            feature
                            for feature, keep in zip(features, mask)
                            if bool(keep)
                        ),
                        coefficients=tuple(float(value) for value in row[mask]),
                        exact_feature_identity=False,
                    )
                )
            return coordinates
        return [
            TransformationCoordinate(
                name=name,
                coordinate_type="feature_coordinate",
                anchor_feature=feature,
                components=(feature,),
                coefficients=(1.0,),
                exact_feature_identity=True,
            )
            for name, feature in zip(names, features)
        ]

    def apply_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X_tr)
        return (self.transform(X_tr), self.transform(X_te))


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
        if fn is not None and bool(is_bw):
            raise ValueError(
                "Two-array custom transformations are disabled because passing both training and held-out matrices to the same callable cannot enforce train-only fitting. Use a stateless fn(X), or provide an estimator-like object with fit(X_train) and transform/apply(X)."
            )
        self.name = info.key
        self._fn = fn
        self._is_bw = False

    def apply(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._fn is None:
            return _BuiltinTransformer(self.name, random_state=42).apply_pair(
                X_tr, X_te
            )
        a = self._fn(X_tr)
        b = self._fn(X_te)
        return (
            _finite_output(
                a,
                expected_shape=_matrix(X_tr).shape,
                context=f"Custom transformation {self.name!r}",
            ),
            _finite_output(
                b,
                expected_shape=_matrix(X_te).shape,
                context=f"Custom transformation {self.name!r}",
            ),
        )


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
        if pseudo_count is not None:
            raise ValueError(
                "pseudo_count is not configurable. 'log10_relative_abundance_half_min_pseudocount' estimates exactly half the minimum positive relative abundance from the training fold."
            )
        self.pseudo_count = None
        self.random_state = int(random_state)
        self._impl: _BuiltinTransformer | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None

    def fit(self, X: np.ndarray) -> "CountTransformation":
        self._impl = _BuiltinTransformer(self.name, random_state=self.random_state).fit(
            X
        )
        self.n_features_in_ = self._impl.n_features_in_
        self.n_features_out_ = self._impl.n_features_out_
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self._impl is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        return self._impl.transform(X)

    def get_feature_names_out(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[str]:
        if self._impl is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        return self._impl.get_feature_names_out(input_features)

    def coordinate_metadata(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[TransformationCoordinate]:
        if self._impl is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        return self._impl.coordinate_metadata(input_features)

    def apply_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X_tr)
        return (self.apply(X_tr), self.apply(X_te))

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
        return (transformation_label(item.name).key, item)
    if hasattr(item, "name") and hasattr(item, "apply"):
        return (transformation_label(str(getattr(item, "name"))).key, item)
    if isinstance(item, tuple):
        return (transformation_label(str(item[0])).key, item[1])
    return (transformation_label(str(item)).key, None)


def _count_transformation_reporting_fields(item: Any) -> dict[str, str]:
    return {}


class CountTransformationAdapter:
    def __init__(self, name: str, spec: Any = None, *, random_state: int = 42):
        self.name = transformation_label(str(name)).key
        self.spec = spec
        self.random_state = int(random_state)
        self.obj: Any | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None

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
        if callable(self.spec) and _callable_accepts_two_required(self.spec):
            raise TypeError(
                "Two-array custom transformation callables are not supported. Use an estimator-like object with fit(X_train) and transform/apply(X), or a stateless one-array callable."
            )
        return self.spec

    def fit(self, X: np.ndarray) -> "CountTransformationAdapter":
        X_float = _as_float_matrix(X)
        self.n_features_in_ = int(np.asarray(X_float).shape[1])
        obj = self._make()
        if hasattr(obj, "fit"):
            obj.fit(X_float)
        self.obj = obj
        n_features_out = getattr(obj, "n_features_out_", None)
        self.n_features_out_ = (
            int(n_features_out) if n_features_out is not None else self.n_features_in_
        )
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        if self.obj is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        X_float = _as_float_matrix(X)
        input_shape = np.asarray(X_float).shape
        if self.n_features_in_ is not None and input_shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count differs from the fitted abundance transformation: expected {self.n_features_in_}, got {input_shape[1]}."
            )
        obj = self.obj
        if isinstance(obj, Transform):
            raise RuntimeError(
                "Callable Transformation objects must be applied to a train/test pair."
            )
        if hasattr(obj, "apply"):
            result = obj.apply(X_float)
        elif hasattr(obj, "transform"):
            result = obj.transform(X_float)
        elif callable(obj):
            result = obj(X_float)
        else:
            raise TypeError(
                f"Custom count transformation {self.name!r} must be callable or provide fit/apply or fit/transform."
            )
        expected_features = (
            int(self.n_features_out_)
            if self.n_features_out_ is not None
            else int(input_shape[1])
        )
        return _finite_output(
            result,
            expected_shape=(int(input_shape[0]), expected_features),
            context=f"Count transformation {self.name!r}",
        )

    def get_feature_names_out(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[str]:
        if self.obj is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        if hasattr(self.obj, "get_feature_names_out"):
            try:
                names = self.obj.get_feature_names_out(input_features)
            except TypeError:
                names = self.obj.get_feature_names_out()
            return [str(x) for x in list(names)]
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        features = (
            _default_feature_names(self.n_features_in_)
            if input_features is None
            else [str(x) for x in list(input_features)]
        )
        if self.n_features_out_ != len(features):
            raise ValueError(
                f"Transformation {self.name!r} changes feature dimension but does not expose get_feature_names_out()."
            )
        return features

    def coordinate_metadata(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[TransformationCoordinate]:
        if self.obj is None:
            raise RuntimeError(
                f"Count transformation {self.name!r} has not been fitted."
            )
        if hasattr(self.obj, "coordinate_metadata"):
            return list(self.obj.coordinate_metadata(input_features))
        names = self.get_feature_names_out(input_features)
        return [
            TransformationCoordinate(
                name=name,
                coordinate_type="feature_coordinate",
                anchor_feature=name,
                components=(name,),
                coefficients=(1.0,),
                exact_feature_identity=True,
            )
            for name in names
        ]

    def apply_pair(
        self, X_tr: np.ndarray, X_te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.spec, Transform) and self.spec._fn is not None:
            return self.spec.apply(_as_float_matrix(X_tr), _as_float_matrix(X_te))
        if callable(self.spec) and _callable_accepts_two_required(self.spec):
            raise TypeError(
                "Two-array custom transformation callables are not supported because they can inspect held-out data while fitting."
            )
        self.fit(X_tr)
        return (self.apply(X_tr), self.apply(X_te))

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
