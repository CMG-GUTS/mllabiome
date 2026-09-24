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
        "additive_log_ratio_training_reference_multiplicative_replacement",
        (
            "additive_log_ratio_first_reference_multiplicative_replacement",
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
        "standardize", ("zscore", "z_score", "standard_scale"), "generic"
    ),
    TransformationLabel("robust_scale", ("robust_numeric", "median_iqr"), "generic"),
    TransformationLabel(
        "power_yeo_johnson", ("yeo_johnson_numeric", "power_numeric"), "generic"
    ),
    TransformationLabel(
        "quantile_normal_numeric", ("rank_gauss_numeric", "qnorm_numeric"), "generic"
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


def _select_alr_reference_index(X: np.ndarray) -> int:
    raw = _matrix(X, nonnegative=True, nonzero_rows=True)
    if raw.shape[1] < 2:
        raise ValueError("ALR requires at least two features.")
    prevalence = np.mean(raw > 0.0, axis=0)
    positive = _positive_composition(raw)
    log_geometric_mean = np.mean(np.log(positive), axis=0)
    order = np.lexsort((np.arange(raw.shape[1]), -log_geometric_mean, -prevalence))
    return int(order[0])


def _alr_matrix(X: np.ndarray, ref_idx: int) -> np.ndarray:
    positive = _positive_composition(X)
    if positive.shape[1] < 2:
        raise ValueError("ALR requires at least two features.")
    ref_idx = int(ref_idx)
    if ref_idx < 0 or ref_idx >= positive.shape[1]:
        raise ValueError("ALR reference index is outside the fitted feature range.")
    out = np.asarray(skbio_alr(positive, ref_idx=ref_idx), dtype=np.float64)
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
        self.alr_reference_index_: int | None = None
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
        if name == "additive_log_ratio_training_reference_multiplicative_replacement":
            if self.alr_reference_index_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _alr_matrix(X, self.alr_reference_index_)
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
            "additive_log_ratio_training_reference_multiplicative_replacement",
            "isometric_log_ratio_egozcue_multiplicative_replacement",
        }:
            if raw.shape[1] < 2:
                raise ValueError(
                    "Log-ratio coordinate transformations require at least two features."
                )
            _positive_composition(X)
            self.n_features_out_ = int(raw.shape[1] - 1)
            if (
                name
                == "additive_log_ratio_training_reference_multiplicative_replacement"
            ):
                self.alr_reference_index_ = _select_alr_reference_index(X)
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
        elif name == "standardize":
            self.scaler_ = StandardScaler().fit(raw)
        elif name == "robust_scale":
            self.scaler_ = RobustScaler().fit(raw)
        elif name == "power_yeo_johnson":
            mask = np.ptp(raw, axis=0) > 0
            self.variable_mask_ = mask
            if np.any(mask):
                self.scaler_ = PowerTransformer(
                    method="yeo-johnson", standardize=True
                ).fit(raw[:, mask])
        elif name == "quantile_normal_numeric":
            self.scaler_ = QuantileTransformer(
                n_quantiles=max(2, min(1000, raw.shape[0])),
                output_distribution="normal",
                random_state=self.random_state,
                subsample=None,
            ).fit(raw)
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
            "additive_log_ratio_training_reference_multiplicative_replacement",
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
            "additive_log_ratio_training_reference_multiplicative_replacement",
            "isometric_log_ratio_egozcue_multiplicative_replacement",
            "within_sample_fractional_rank",
        }:
            return _finite_output(
                self._base_transform(X),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "standardize":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _finite_output(
                self.scaler_.transform(_matrix(X)),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "robust_scale":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _finite_output(
                self.scaler_.transform(_matrix(X)),
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "power_yeo_johnson":
            raw = _matrix(X)
            out = np.zeros_like(raw, dtype=np.float64)
            mask = self.variable_mask_
            if mask is None:
                raise RuntimeError("Transformation has not been fitted.")
            if np.any(mask):
                if self.scaler_ is None:
                    raise RuntimeError("Transformation has not been fitted.")
                out[:, mask] = self.scaler_.transform(raw[:, mask])
            return _finite_output(
                out,
                expected_shape=expected_shape,
                context=f"Built-in transformation {name!r}",
            )
        if name == "quantile_normal_numeric":
            if self.scaler_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            return _finite_output(
                self.scaler_.transform(_matrix(X)),
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
        if (
            self.name
            == "additive_log_ratio_training_reference_multiplicative_replacement"
        ):
            if self.alr_reference_index_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            reference = features[self.alr_reference_index_]
            return [
                f"ALR[{feature}/{reference}]"
                for index, feature in enumerate(features)
                if index != self.alr_reference_index_
            ]
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
        if (
            self.name
            == "additive_log_ratio_training_reference_multiplicative_replacement"
        ):
            if self.alr_reference_index_ is None:
                raise RuntimeError("Transformation has not been fitted.")
            reference = features[self.alr_reference_index_]
            numerators = [
                feature
                for index, feature in enumerate(features)
                if index != self.alr_reference_index_
            ]
            return [
                TransformationCoordinate(
                    name=name,
                    coordinate_type="alr_logcontrast",
                    anchor_feature=None,
                    components=(feature, reference),
                    coefficients=(1.0, -1.0),
                    exact_feature_identity=False,
                )
                for name, feature in zip(names, numerators)
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


_COMPOSITION_SENSITIVE = frozenset(
    {
        "relative_abundance",
        "hellinger",
        "arcsine_sqrt",
        "log10_relative_abundance_half_min_pseudocount",
        "centered_log_ratio_multiplicative_replacement",
        "additive_log_ratio_training_reference_multiplicative_replacement",
        "isometric_log_ratio_egozcue_multiplicative_replacement",
        "standardized_centered_log_ratio_multiplicative_replacement",
        "yeo_johnson_relative_abundance",
        "quantile_normal_relative_abundance",
        "robust_scaled_relative_abundance",
        "within_sample_fractional_rank",
        "prevalence_weighted_relative_abundance",
    }
)
_LOG_RATIO_COORDINATES = frozenset(
    {
        "additive_log_ratio_training_reference_multiplicative_replacement",
        "isometric_log_ratio_egozcue_multiplicative_replacement",
    }
)
_LOG_RATIO_BLOCK_TRANSFORMS = frozenset(
    {
        "centered_log_ratio_multiplicative_replacement",
        "standardized_centered_log_ratio_multiplicative_replacement",
        "additive_log_ratio_training_reference_multiplicative_replacement",
        "isometric_log_ratio_egozcue_multiplicative_replacement",
    }
)
_COMPOSITION_SCOPES = frozenset({"rank-wise", "joint"})


def _normalise_composition_scope(name: str, scope: str | None) -> str:
    base = transformation_label(str(name)).key
    default = "rank-wise" if base in _COMPOSITION_SENSITIVE else "joint"
    if scope is None:
        return default
    value = str(scope).strip().lower()
    if value not in _COMPOSITION_SCOPES:
        raise ValueError(
            f"composition_scope must be one of {sorted(_COMPOSITION_SCOPES)!r}; got {scope!r}."
        )
    if base not in _COMPOSITION_SENSITIVE and value != "joint":
        raise ValueError(
            f"Transformation {base!r} is feature-wise and does not accept composition_scope={value!r}."
        )
    return value


def _parse_transformation_identity(value: Any) -> tuple[str, str]:
    text = str(value).strip()
    if "@" in text:
        raw_name, raw_scope = text.rsplit("@", 1)
        if raw_scope.strip().lower() in _COMPOSITION_SCOPES:
            base = transformation_label(raw_name).key
            return base, _normalise_composition_scope(base, raw_scope)
    base = transformation_label(text).key
    return base, _normalise_composition_scope(base, None)


def _transformation_identity(name: str, scope: str | None = None) -> str:
    base = transformation_label(str(name)).key
    resolved = _normalise_composition_scope(base, scope)
    if base in _COMPOSITION_SENSITIVE:
        return f"{base}@{resolved}"
    return base


class Transform:
    __slots__ = ("name", "composition_scope", "identity", "_fn", "_is_bw")

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        is_bw: bool = False,
        abbreviation: str | None = None,
        composition_scope: str | None = None,
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
        self.composition_scope = _normalise_composition_scope(
            self.name, composition_scope
        )
        self.identity = _transformation_identity(self.name, self.composition_scope)
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


def build_count_transformations(
    *, include_inactive: bool = True
) -> list[Transformation]:
    items = [Transformation(label.key) for label in TRANSFORMATION_LABELS]
    if include_inactive:
        return items
    return [x for x in items if x.name == "arcsine_sqrt"]


class CountTransformation:
    def __init__(
        self,
        name: str,
        pseudo_count: float | None = None,
        random_state: int = 42,
        composition_scope: str | None = None,
    ):
        base, parsed_scope = _parse_transformation_identity(name)
        self.name = base
        self.composition_scope = _normalise_composition_scope(
            base, parsed_scope if composition_scope is None else composition_scope
        )
        self.identity = _transformation_identity(base, self.composition_scope)
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
        return item.identity
    if hasattr(item, "identity"):
        return str(getattr(item, "identity"))
    if hasattr(item, "name") and hasattr(item, "apply"):
        name = str(getattr(item, "name"))
        scope = getattr(item, "composition_scope", None)
        return _transformation_identity(name, scope)
    if isinstance(item, tuple):
        return _count_transformation_name(item[0])
    base, scope = _parse_transformation_identity(item)
    return _transformation_identity(base, scope)


def _count_transformation_spec(item: Any) -> tuple[str, Any | None]:
    if isinstance(item, Transform):
        return (item.identity, item)
    if hasattr(item, "name") and hasattr(item, "apply"):
        return (_count_transformation_name(item), item)
    if isinstance(item, tuple):
        raw_name, spec = item
        if isinstance(spec, Transform):
            return (spec.identity, spec)
        return (_count_transformation_name(raw_name), spec)
    return (_count_transformation_name(item), None)


def _effective_count_transformation_spec(
    item: Any, feature_blocks: Any = None
) -> tuple[str, Any | None]:
    name, spec = _count_transformation_spec(item)
    base, _ = _parse_transformation_identity(name)
    if base in _COMPOSITION_SENSITIVE and feature_blocks is not None:
        block_count = sum(1 for _, indices in feature_blocks if tuple(indices))
        if block_count <= 1:
            name = _transformation_identity(base, "rank-wise")
            if isinstance(spec, Transform) and spec.composition_scope != "rank-wise":
                spec = Transform(
                    spec.name,
                    fn=spec._fn,
                    composition_scope="rank-wise",
                )
    return name, spec


def _count_transformation_specs_for_blocks(
    items: Any, feature_blocks: Any = None
) -> tuple[tuple[str, Any | None], ...]:
    out: list[tuple[str, Any | None]] = []
    seen: set[str] = set()
    for item in items:
        name, spec = _effective_count_transformation_spec(item, feature_blocks)
        if name in seen:
            continue
        seen.add(name)
        out.append((name, spec))
    return tuple(out)


def _normalise_feature_blocks(
    blocks: Any, n_features: int
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if blocks is None:
        return (("all", tuple(range(int(n_features)))),)
    out: list[tuple[str, tuple[int, ...]]] = []
    seen: set[int] = set()
    for raw_name, raw_indices in blocks:
        indices = tuple(int(index) for index in raw_indices)
        if not indices:
            continue
        for index in indices:
            if index < 0 or index >= int(n_features):
                raise ValueError(
                    f"Feature block {raw_name!r} references column {index}, outside matrix width {n_features}."
                )
            if index in seen:
                raise ValueError(f"Feature column {index} occurs in multiple blocks.")
            seen.add(index)
        out.append((str(raw_name), indices))
    if seen != set(range(int(n_features))):
        missing = sorted(set(range(int(n_features))) - seen)
        raise ValueError(
            f"Feature blocks do not cover the transformed matrix exactly; missing columns: {missing[:8]!r}."
        )
    return tuple(out)


class CountTransformationAdapter:
    def __init__(
        self,
        name: str,
        spec: Any = None,
        *,
        random_state: int = 42,
        feature_blocks: Any = None,
        composition_scope: str | None = None,
    ):
        base, parsed_scope = _parse_transformation_identity(name)
        spec_scope = getattr(spec, "composition_scope", None)
        requested_scope = (
            composition_scope
            if composition_scope is not None
            else spec_scope
            if spec_scope is not None
            else parsed_scope
        )
        self.name = base
        self.composition_scope = _normalise_composition_scope(base, requested_scope)
        self.identity = _transformation_identity(base, self.composition_scope)
        self.spec = spec
        self.random_state = int(random_state)
        self.feature_blocks = feature_blocks
        self.obj: Any | None = None
        self.block_objects_: list[tuple[str, tuple[int, ...], Any]] | None = None
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None

    def _make(self) -> Any:
        if self.spec is None:
            return CountTransformation(
                self.name,
                random_state=self.random_state,
                composition_scope="joint",
            )
        if isinstance(self.spec, Transform):
            if self.spec._fn is None:
                return CountTransformation(
                    self.spec.name,
                    random_state=self.random_state,
                    composition_scope="joint",
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

    def _fit_single(self, X: np.ndarray) -> Any:
        obj = self._make()
        if hasattr(obj, "fit"):
            obj.fit(X)
        return obj

    def fit(self, X: np.ndarray) -> "CountTransformationAdapter":
        X_float = _as_float_matrix(X)
        n_features = int(np.asarray(X_float).shape[1])
        self.n_features_in_ = n_features
        use_blocks = (
            self.name in _COMPOSITION_SENSITIVE
            and self.composition_scope == "rank-wise"
        )
        if use_blocks:
            blocks = _normalise_feature_blocks(self.feature_blocks, n_features)
            unresolved = [name for name, _ in blocks if name == "unresolved"]
            if unresolved:
                raise ValueError(
                    "Rank-wise compositional transformation requires every feature to have a resolved taxonomic rank. Use explicit taxonomic ranks or composition_scope='joint'."
                )
            fitted_blocks: list[tuple[str, tuple[int, ...], Any]] = []
            total_out = 0
            for block_name, indices in blocks:
                if self.name in _LOG_RATIO_BLOCK_TRANSFORMS and len(indices) < 2:
                    continue
                block = np.asarray(X_float)[:, np.asarray(indices, dtype=int)]
                obj = self._fit_single(block)
                n_out = getattr(obj, "n_features_out_", None)
                total_out += int(n_out) if n_out is not None else len(indices)
                fitted_blocks.append((block_name, indices, obj))
            if not fitted_blocks:
                raise ValueError(
                    f"Transformation {self.identity!r} produced no coordinates because every taxonomic block contains fewer than two features."
                )
            self.block_objects_ = fitted_blocks
            self.obj = None
            self.n_features_out_ = int(total_out)
            return self
        obj = self._fit_single(np.asarray(X_float))
        self.obj = obj
        self.block_objects_ = None
        n_features_out = getattr(obj, "n_features_out_", None)
        self.n_features_out_ = (
            int(n_features_out) if n_features_out is not None else self.n_features_in_
        )
        return self

    def _apply_object(self, obj: Any, X: np.ndarray) -> np.ndarray:
        if isinstance(obj, Transform):
            raise RuntimeError(
                "Callable Transformation objects must be applied to a train/test pair."
            )
        if hasattr(obj, "apply"):
            result = obj.apply(X)
        elif hasattr(obj, "transform"):
            result = obj.transform(X)
        elif callable(obj):
            result = obj(X)
        else:
            raise TypeError(
                f"Custom count transformation {self.name!r} must be callable or provide fit/apply or fit/transform."
            )
        return np.asarray(result, dtype=np.float64)

    def apply(self, X: np.ndarray) -> np.ndarray:
        X_float = _as_float_matrix(X)
        input_shape = np.asarray(X_float).shape
        if self.n_features_in_ is None or self.n_features_out_ is None:
            raise RuntimeError(
                f"Count transformation {self.identity!r} has not been fitted."
            )
        if input_shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count differs from the fitted abundance transformation: expected {self.n_features_in_}, got {input_shape[1]}."
            )
        if self.block_objects_ is not None:
            pieces = []
            for _, indices, obj in self.block_objects_:
                block = np.asarray(X_float)[:, np.asarray(indices, dtype=int)]
                pieces.append(self._apply_object(obj, block))
            result = np.concatenate(pieces, axis=1)
        else:
            if self.obj is None:
                raise RuntimeError(
                    f"Count transformation {self.identity!r} has not been fitted."
                )
            result = self._apply_object(self.obj, np.asarray(X_float))
        return _finite_output(
            result,
            expected_shape=(int(input_shape[0]), int(self.n_features_out_)),
            context=f"Count transformation {self.identity!r}",
        )

    def _input_features(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None
    ) -> list[str]:
        if self.n_features_in_ is None:
            raise RuntimeError(
                f"Count transformation {self.identity!r} has not been fitted."
            )
        features = (
            _default_feature_names(self.n_features_in_)
            if input_features is None
            else [str(x) for x in list(input_features)]
        )
        if len(features) != self.n_features_in_:
            raise ValueError(
                f"Input feature-name count differs from the fitted abundance transformation: expected {self.n_features_in_}, got {len(features)}."
            )
        return features

    def get_feature_names_out(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[str]:
        features = self._input_features(input_features)
        if self.block_objects_ is not None:
            names: list[str] = []
            for _, indices, obj in self.block_objects_:
                block_names = [features[index] for index in indices]
                if hasattr(obj, "get_feature_names_out"):
                    try:
                        raw = obj.get_feature_names_out(block_names)
                    except TypeError:
                        raw = obj.get_feature_names_out()
                    names.extend(str(x) for x in list(raw))
                else:
                    names.extend(block_names)
            if self.n_features_out_ != len(names):
                raise ValueError(
                    f"Transformation {self.identity!r} produced inconsistent feature names."
                )
            return names
        if self.obj is None:
            raise RuntimeError(
                f"Count transformation {self.identity!r} has not been fitted."
            )
        if hasattr(self.obj, "get_feature_names_out"):
            try:
                names = self.obj.get_feature_names_out(features)
            except TypeError:
                names = self.obj.get_feature_names_out()
            return [str(x) for x in list(names)]
        if self.n_features_out_ != len(features):
            raise ValueError(
                f"Transformation {self.identity!r} changes feature dimension but does not expose get_feature_names_out()."
            )
        return features

    def coordinate_metadata(
        self, input_features: list[str] | tuple[str, ...] | np.ndarray | None = None
    ) -> list[TransformationCoordinate]:
        features = self._input_features(input_features)
        if self.block_objects_ is not None:
            coordinates: list[TransformationCoordinate] = []
            for _, indices, obj in self.block_objects_:
                block_names = [features[index] for index in indices]
                if hasattr(obj, "coordinate_metadata"):
                    coordinates.extend(list(obj.coordinate_metadata(block_names)))
                else:
                    coordinates.extend(
                        TransformationCoordinate(
                            name=name,
                            coordinate_type="feature_coordinate",
                            anchor_feature=name,
                            components=(name,),
                            coefficients=(1.0,),
                            exact_feature_identity=True,
                        )
                        for name in block_names
                    )
            return coordinates
        if self.obj is None:
            raise RuntimeError(
                f"Count transformation {self.identity!r} has not been fitted."
            )
        if hasattr(self.obj, "coordinate_metadata"):
            return list(self.obj.coordinate_metadata(features))
        names = self.get_feature_names_out(features)
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
            blocks = _normalise_feature_blocks(
                self.feature_blocks, np.asarray(_as_float_matrix(X_tr)).shape[1]
            )
            if (
                self.composition_scope == "rank-wise"
                and self.name in _COMPOSITION_SENSITIVE
            ):
                tr_parts = []
                te_parts = []
                for _, indices in blocks:
                    if self.name in _LOG_RATIO_BLOCK_TRANSFORMS and len(indices) < 2:
                        continue
                    cols = np.asarray(indices, dtype=int)
                    tr_part, te_part = self.spec.apply(
                        np.asarray(_as_float_matrix(X_tr))[:, cols],
                        np.asarray(_as_float_matrix(X_te))[:, cols],
                    )
                    tr_parts.append(tr_part)
                    te_parts.append(te_part)
                if not tr_parts:
                    raise ValueError(
                        f"Transformation {self.identity!r} produced no coordinates."
                    )
                return np.concatenate(tr_parts, axis=1), np.concatenate(
                    te_parts, axis=1
                )
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
    item: Any,
    *,
    random_state: int,
    feature_blocks: Any = None,
) -> tuple[str, Callable[[], CountTransformationAdapter]]:
    if isinstance(item, Transform) or (
        hasattr(item, "name") and hasattr(item, "apply")
    ):
        name = _count_transformation_name(item)
        return (
            name,
            lambda item=item, name=name, random_state=random_state, feature_blocks=feature_blocks: (
                CountTransformationAdapter(
                    name,
                    item,
                    random_state=random_state,
                    feature_blocks=feature_blocks,
                )
            ),
        )
    if isinstance(item, tuple):
        raw_name, spec = item
        name = _count_transformation_name(
            spec if isinstance(spec, Transform) else raw_name
        )
        return (
            name,
            lambda name=name, spec=spec, random_state=random_state, feature_blocks=feature_blocks: (
                CountTransformationAdapter(
                    name,
                    spec,
                    random_state=random_state,
                    feature_blocks=feature_blocks,
                )
            ),
        )
    name = _count_transformation_name(item)
    return (
        name,
        lambda name=name, random_state=random_state, feature_blocks=feature_blocks: (
            CountTransformationAdapter(
                name,
                random_state=random_state,
                feature_blocks=feature_blocks,
            )
        ),
    )
