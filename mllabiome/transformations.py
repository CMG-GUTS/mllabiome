from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

from .utils import _as_float_matrix, _clr, _finite, _relative

@dataclass(frozen=True)
class TransformationLabel:
    key: str
    abbreviation: str
    aliases: tuple[str, ...] = ()


TRANSFORMATION_LABELS: tuple[TransformationLabel, ...] = (
    TransformationLabel("identity", "Identity", ("raw", "unchanged")),
    TransformationLabel("relative_abundance", "RA", ("none", "ra", "relative")),
    TransformationLabel("binary", "P/A", ("presence_absence", "pa")),
    TransformationLabel("sqrt", r"$\sqrt{x}$", ("sqrt_abundance",)),
    TransformationLabel("hellinger", "Hellinger"),
    TransformationLabel("arcsin_sqrt", r"$\arcsin\sqrt{x}$", ("asin_sqrt",)),
    TransformationLabel("log", r"$\ln(1+x)$", ("log1p", "ln1p")),
    TransformationLabel("log2", r"$\log_2(1+x)$", ("log2_1p",)),
    TransformationLabel("log10", r"$\log_{10}(1+x)$", ("log10_1p",)),
    TransformationLabel("log_std", "row-log-z", ("row_log_z",)),
    TransformationLabel("log_tss_floor", "log-TSS"),
    TransformationLabel("log_unit", "row-log-L2", ("row_log_l2",)),
    TransformationLabel("symlog", "symlog"),
    TransformationLabel("zi_log", "ZI-log", ("zero_inflated_log",)),
    TransformationLabel("zscore", "row-z", ("row_z",)),
    TransformationLabel("scikit-bio_alr", "ALR-first", ("skbio_alr", "scikit_bio_alr", "scikitbio_alr", "alr_first")),
    TransformationLabel("scikit-bio_clr", "CLR-mult", ("skbio_clr", "scikit_bio_clr", "scikitbio_clr", "clr_mult")),
    TransformationLabel("scikit-bio_ilr", "ILR-Egoz.", ("skbio_ilr", "scikit_bio_ilr", "scikitbio_ilr", "ilr_egoz", "ilr_egozcue")),
    TransformationLabel("alr", "ALR-last", ("alr_last",)),
    TransformationLabel("bclr", "BCLR"),
    TransformationLabel("clr", r"CLR-$\epsilon$", ("clr_epsilon", "clr_eps")),
    TransformationLabel("clr_std", r"CLR-$\epsilon$+z", ("clr_epsilon_z", "clr_eps_z")),
    TransformationLabel("ilr", "ILR-seq", ("ilr_seq",)),
    TransformationLabel("ilr_std", "ILR-seq+z", ("ilr_seq_z",)),
    TransformationLabel("pairwise_logratio", "Pair-logR-500", ("pair_logr_500",)),
    TransformationLabel("rclr", "RCLR"),
    TransformationLabel("prev_weighted", "Prev-wt", ("prevalence_weighted", "prev_wt")),
    TransformationLabel("rank_col", "ECDF-rank", ("ecdf_rank",)),
    TransformationLabel("rank_frac", "row-rank", ("row_rank", "rank")),
    TransformationLabel("rank_std", "row-rank-z", ("row_rank_z",)),
    TransformationLabel("rank_unit", "row-rank-unit", ("row_rank_unit",)),
    TransformationLabel("power", "Yeo-J", ("yeo_johnson", "power_yj")),
    TransformationLabel("quantile", "QNorm", ("rank_gauss", "quantile_normal", "qnorm")),
    TransformationLabel("robust", "Robust", ("robust_z",)),
)


TRANSFORMATION_SPACE = TRANSFORMATION_LABELS

_TRANSFORMATION_LABEL_BY_KEY: dict[str, TransformationLabel] = {}
for _label in TRANSFORMATION_LABELS:
    _TRANSFORMATION_LABEL_BY_KEY[_label.key.lower()] = _label
    _TRANSFORMATION_LABEL_BY_KEY[_label.abbreviation.lower()] = _label
    for _alias in _label.aliases:
        _TRANSFORMATION_LABEL_BY_KEY[str(_alias).lower()] = _label


def transformation_label(name: str) -> TransformationLabel:

    key = str(name).strip().lower()
    label = _TRANSFORMATION_LABEL_BY_KEY.get(key)
    if label is not None:
        return label
    return TransformationLabel(str(name), str(name))


def transformation_space_table() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "count_transformation": label.key,
                "transformation_abbreviation": label.abbreviation,
                "aliases": ",".join(label.aliases),
            }
            for label in TRANSFORMATION_LABELS
        ]
    )



class Transform:









    __slots__ = ("name", "_fn", "_is_bw", "abbreviation")

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        is_bw: bool = False,
        abbreviation: str | None = None,
    ):
        requested_name = str(name)
        self.name = transformation_label(requested_name).key
        if fn is None:
            provided = {t.name: t for t in build_count_transformations(include_inactive=True)}
            if self.name not in provided:
                raise KeyError(f"Unknown abundance transformation {requested_name!r}. Provide a callable for a custom transformation.")
            base = provided[self.name]
            fn = base._fn
            is_bw = base._is_bw
            abbreviation = abbreviation or base.abbreviation
        self._fn = fn
        self._is_bw = bool(is_bw)
        info = transformation_label(self.name)
        self.abbreviation = abbreviation or info.abbreviation

    def apply(self, X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._is_bw:
            a, b = self._fn(X_tr, X_te)
            return _finite(a).astype(np.float32, copy=False), _finite(b).astype(np.float32, copy=False)
        return (
            _finite(self._fn(X_tr)).astype(np.float32, copy=False),
            _finite(self._fn(X_te)).astype(np.float32, copy=False),
        )


Transformation = Transform
_Transform = Transform
_Transformation = Transform


def build_count_transformations(*, include_inactive: bool = True) -> list[Transformation]:






    T = Transformation

    def _mr_local(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        out = np.asarray(X, dtype=np.float64).copy()
        out[out <= 0] = eps
        return out

    def identity_(x):


        return _as_float_matrix(x).astype(np.float32, copy=False)

    def _ra(x):
        return _relative(x)

    def binary_(x):
        return (_as_float_matrix(x) > 0).astype(np.float32)

    def sqrt_(x):
        return np.sqrt(np.clip(_relative(x), 0, None)).astype(np.float32)

    def hellinger_(x):
        return np.sqrt(_relative(x)).astype(np.float32)

    def arcsin_sqrt_(x):
        return np.arcsin(np.sqrt(np.clip(_relative(x), 0, 1))).astype(np.float32)

    def log_(x):
        return np.log1p(np.clip(_relative(x), 0, None)).astype(np.float32)

    def log10_(x):
        return np.log10(np.clip(_relative(x), 0, None) + 1).astype(np.float32)

    def log2_(x):
        return np.log2(np.clip(_relative(x), 0, None) + 1).astype(np.float32)

    def rank_std_(x):
        r = np.apply_along_axis(rankdata, 1, _as_float_matrix(x)).astype(np.float32)
        mu = r.mean(1, keepdims=True)
        sd = r.std(1, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return (r - mu) / sd

    def rank_unit_(x):
        r = np.apply_along_axis(rankdata, 1, _as_float_matrix(x)).astype(np.float32)
        n = np.sqrt((r ** 2).sum(1, keepdims=True))
        n[n < 1e-10] = 1.0
        return r / n

    def log_std_(x):
        lx = np.log1p(np.clip(_relative(x), 0, None))
        mu = lx.mean(1, keepdims=True)
        sd = lx.std(1, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return ((lx - mu) / sd).astype(np.float32)

    def log_unit_(x):
        lx = np.log1p(np.clip(_relative(x), 0, None))
        n = np.sqrt((lx ** 2).sum(1, keepdims=True))
        n[n < 1e-10] = 1.0
        return (lx / n).astype(np.float32)

    def zscore_(x):
        X = _as_float_matrix(x)
        mu = X.mean(1, keepdims=True)
        sd = X.std(1, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return ((X - mu) / sd).astype(np.float32)

    def rclr_(x):
        X = _as_float_matrix(x)
        lx = np.where(X > 0, np.log(np.clip(X, 1e-300, None)), 0.0)
        nz = (X > 0).astype(np.float64)
        nz_s = nz.sum(1, keepdims=True)
        nz_s[nz_s == 0] = 1.0
        gm = (lx * nz).sum(1, keepdims=True) / nz_s
        return np.where(X > 0, lx - gm, 0.0).astype(np.float32)

    def bclr_(x):
        X = _as_float_matrix(x).astype(np.float64)
        D = X.shape[1]
        xp = X + 0.5 / max(D, 1)
        xp /= np.maximum(xp.sum(1, keepdims=True), 1e-300)
        lx = np.log(xp)
        return (lx - lx.mean(1, keepdims=True)).astype(np.float32)

    def log_tss_floor_(x):
        X = np.clip(_as_float_matrix(x), 0, None).astype(np.float64)
        s = X.sum(1, keepdims=True)
        s[s == 0] = 1.0
        rel = np.maximum(X / s, 1e-10)
        return np.log(rel).astype(np.float32)

    def rank_frac_(x):
        X = _as_float_matrix(x)
        r = np.apply_along_axis(rankdata, 1, X).astype(np.float64)
        return (r / (X.shape[1] + 1.0)).astype(np.float32)

    def zi_log_(x):
        X = _as_float_matrix(x)
        out = np.zeros_like(X, dtype=np.float32)
        mask = X > 0
        out[mask] = np.log(X[mask].astype(np.float64)).astype(np.float32)
        return out

    def symlog_(x):
        X = _as_float_matrix(x)
        return (np.sign(X) * np.log1p(np.abs(X))).astype(np.float32)

    def clr_eps_(tr, te):
        def _c(X):
            X = _mr_local(X)
            X = X / np.maximum(X.sum(1, keepdims=True), 1e-300)
            lx = np.log(X)
            return (lx - lx.mean(1, keepdims=True)).astype(np.float32)
        return _c(tr), _c(te)

    def alr_last_(tr, te):
        def _a(X):
            X = _mr_local(X)
            X = X / np.maximum(X.sum(1, keepdims=True), 1e-300)
            if X.shape[1] < 2:
                return X.astype(np.float32)
            return np.log(X[:, :-1] / X[:, -1:]).astype(np.float32)
        return _a(tr), _a(te)

    def alr_first_(tr, te):
        def _a(X):
            X = _mr_local(X)
            X = X / np.maximum(X.sum(1, keepdims=True), 1e-300)
            if X.shape[1] < 2:
                return X.astype(np.float32)
            return np.log(X[:, 1:] / X[:, :1]).astype(np.float32)
        return _a(tr), _a(te)

    def ilr_seq_(tr, te):
        def _i(X):
            X = _mr_local(X)
            X = X / np.maximum(X.sum(1, keepdims=True), 1e-300)
            _, D = X.shape
            if D < 2:
                return X.astype(np.float32)
            lX = np.log(X).astype(np.float64)
            cum = np.cumsum(lX[:, :-1], axis=1)
            k = np.arange(1, D, dtype=np.float64)
            return (np.sqrt(k / (k + 1.0)) * (cum / k - lX[:, 1:])).astype(np.float32)
        return _i(tr), _i(te)

    def clr_std_(tr, te):
        tr_c, te_c = clr_eps_(tr, te)
        mu = tr_c.mean(0, keepdims=True)
        sd = tr_c.std(0, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return (tr_c - mu) / sd, (te_c - mu) / sd

    def ilr_std_(tr, te):
        tr_i, te_i = ilr_seq_(tr, te)
        mu = tr_i.mean(0, keepdims=True)
        sd = tr_i.std(0, keepdims=True)
        sd[sd < 1e-10] = 1.0
        return (tr_i - mu) / sd, (te_i - mu) / sd

    def power_(tr, te):
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        return pt.fit_transform(_as_float_matrix(tr)).astype(np.float32), pt.transform(_as_float_matrix(te)).astype(np.float32)

    def robust_(tr, te):
        sc = RobustScaler()
        return sc.fit_transform(_as_float_matrix(tr)).astype(np.float32), sc.transform(_as_float_matrix(te)).astype(np.float32)

    def quantile_(tr, te):
        qt = QuantileTransformer(output_distribution="normal", random_state=42, n_quantiles=max(2, min(100, tr.shape[0])))
        return qt.fit_transform(_as_float_matrix(tr)).astype(np.float32), qt.transform(_as_float_matrix(te)).astype(np.float32)

    def rank_col_(tr, te):
        tr = _as_float_matrix(tr)
        te = _as_float_matrix(te)
        n_tr = tr.shape[0]
        otr = np.zeros_like(tr, dtype=np.float32)
        ote = np.zeros_like(te, dtype=np.float32)
        for j in range(tr.shape[1]):
            col = np.sort(tr[:, j])
            otr[:, j] = rankdata(tr[:, j]) / (n_tr + 1.0)
            ote[:, j] = np.searchsorted(col, te[:, j], side="right") / max(float(n_tr), 1.0)
        return otr, ote

    def prev_weighted_(tr, te):
        tr = _as_float_matrix(tr)
        te = _as_float_matrix(te)
        prev = (tr > 0).mean(0)
        tw = tr * prev
        ew = te * prev
        st = tw.sum(1, keepdims=True)
        se = ew.sum(1, keepdims=True)
        st[st == 0] = 1.0
        se[se == 0] = 1.0
        return (tw / st).astype(np.float32), (ew / se).astype(np.float32)

    def pairwise_logratio_(tr, te):
        tr = _mr_local(tr)
        te = _mr_local(te)
        D = tr.shape[1]
        if D < 2:
            return tr.astype(np.float32), te.astype(np.float32)
        rng = np.random.RandomState(42)
        max_pairs = 500
        if D * (D - 1) // 2 <= max_pairs:
            i_idx, j_idx = np.triu_indices(D, k=1)
        else:
            all_p = np.array(np.triu_indices(D, k=1)).T
            ch = rng.choice(len(all_p), max_pairs, replace=False)
            i_idx, j_idx = all_p[ch, 0], all_p[ch, 1]
        return np.log(tr[:, i_idx] / tr[:, j_idx]).astype(np.float32), np.log(te[:, i_idx] / te[:, j_idx]).astype(np.float32)

    def _closure(X):
        X = np.clip(_as_float_matrix(X), 0, None).astype(np.float64)
        s = X.sum(1, keepdims=True)
        s[s <= 0] = 1.0
        return X / s

    def _multiplicative_replacement(X, eps: float = 1e-10):
        X = _closure(X)
        out = X.copy()
        for i in range(out.shape[0]):
            row = out[i]
            zero = row <= 0
            n_zero = int(zero.sum())
            if n_zero == 0:
                continue
            pos = row[~zero]
            if len(pos):
                delta = min(float(pos.min()) * 0.5, 1.0 / (row.size * row.size))
                delta = max(delta, eps)
            else:
                delta = 1.0 / max(row.size, 1)
            row[zero] = delta
            scale = max(1.0 - n_zero * delta, eps)
            if len(pos):
                row[~zero] = row[~zero] / max(row[~zero].sum(), eps) * scale
            row /= max(row.sum(), eps)
            out[i] = row
        return out.astype(np.float64)

    def scikit_clr_(tr, te):
        def _c(X):
            X = _multiplicative_replacement(X)
            lx = np.log(X)
            return (lx - lx.mean(1, keepdims=True)).astype(np.float32)
        return _c(tr), _c(te)

    def scikit_alr_(tr, te):
        def _a(X):
            X = _multiplicative_replacement(X)
            if X.shape[1] < 2:
                return X.astype(np.float32)
            return np.log(X[:, 1:] / X[:, :1]).astype(np.float32)
        return _a(tr), _a(te)

    def scikit_ilr_(tr, te):
        def _i(X):
            X = _multiplicative_replacement(X)
            _, D = X.shape
            if D < 2:
                return X.astype(np.float32)
            lX = np.log(X).astype(np.float64)
            cum = np.cumsum(lX[:, :-1], axis=1)
            k = np.arange(1, D, dtype=np.float64)
            return (np.sqrt(k / (k + 1.0)) * (cum / k - lX[:, 1:])).astype(np.float32)
        return _i(tr), _i(te)

    all_items = [
        T("identity", identity_),
        T("relative_abundance", _ra),
        T("binary", binary_),
        T("sqrt", sqrt_),
        T("hellinger", hellinger_),
        T("arcsin_sqrt", arcsin_sqrt_),
        T("log", log_),
        T("log10", log10_),
        T("log2", log2_),
        T("log_std", log_std_),
        T("log_unit", log_unit_),
        T("log_tss_floor", log_tss_floor_),
        T("zscore", zscore_),
        T("rank_frac", rank_frac_),
        T("rank_std", rank_std_),
        T("rank_unit", rank_unit_),
        T("zi_log", zi_log_),
        T("symlog", symlog_),
        T("scikit-bio_clr", scikit_clr_, True),
        T("scikit-bio_alr", scikit_alr_, True),
        T("scikit-bio_ilr", scikit_ilr_, True),
        T("clr", clr_eps_, True),
        T("alr", alr_last_, True),
        T("ilr", ilr_seq_, True),
        T("rclr", rclr_),
        T("bclr", bclr_),
        T("clr_std", clr_std_, True),
        T("ilr_std", ilr_std_, True),
        T("power", power_, True),
        T("quantile", quantile_, True),
        T("robust", robust_, True),
        T("rank_col", rank_col_, True),
        T("prev_weighted", prev_weighted_, True),
        T("pairwise_logratio", pairwise_logratio_, True),
    ]
    if include_inactive:
        return all_items
    return [x for x in all_items if x.name == "arcsin_sqrt"]


class CountTransformation:


    def __init__(self, name: str, pseudo_count: float = 1e-6, random_state: int = 42):
        self.name = transformation_label(str(name)).key
        self.pseudo_count = float(pseudo_count)
        self.random_state = int(random_state)
        self._estimator: Any | None = None

    def fit(self, X: np.ndarray) -> "CountTransformation":
        X = _as_float_matrix(X)
        kind = self.name.lower()
        if kind in {"rank_gauss", "quantile", "quantile_normal"}:
            n_quantiles = max(2, min(1000, X.shape[0]))
            self._estimator = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=self.random_state,
                subsample=None,
            )
            self._estimator.fit(_finite(_relative(X)))
        elif kind in {"robust_z", "robust"}:
            self._estimator = RobustScaler(quantile_range=(10, 90)).fit(_finite(X))
        elif kind in {"yeo_johnson", "power", "power_yj"}:
            self._estimator = PowerTransformer(method="yeo-johnson", standardize=True).fit(_finite(X))
        else:
            self._estimator = None
        return self

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = _as_float_matrix(X)
        kind = self.name.lower()
        if kind in {"identity", "raw", "unchanged"}:
            out = X
        elif kind in {"relative_abundance", "none", "relative"}:
            out = _relative(X)
        elif kind in {"log", "log1p"}:
            out = np.log1p(np.clip(_relative(X), 0.0, None))
        elif kind in {"arcsin_sqrt", "asin_sqrt"}:
            out = np.arcsin(np.sqrt(np.clip(_relative(X), 0.0, 1.0)))
        elif kind == "sqrt":
            out = np.sqrt(np.clip(_relative(X), 0.0, None))
        elif kind == "clr":
            out = _clr(X, self.pseudo_count)
        elif kind == "log_clr":
            out = _clr(np.log1p(np.clip(X, 0.0, None)), self.pseudo_count)
        elif kind in {"presence_absence", "binary", "prevalence"}:
            out = (X > 0).astype(np.float32)
        elif kind in {"rank", "rank_within_sample"}:
            out = np.apply_along_axis(rankdata, 1, X).astype(np.float32)
            out = out / max(1, X.shape[1])
        elif kind in {"rank_gauss", "quantile", "quantile_normal", "robust_z", "robust", "yeo_johnson", "power", "power_yj"}:
            if self._estimator is None:
                raise RuntimeError(f"Count transformation {self.name!r} has not been fitted.")
            base = _relative(X) if kind in {"rank_gauss", "quantile", "quantile_normal"} else X
            out = self._estimator.transform(_finite(base))
        else:
            raise ValueError(f"Unknown count transformation {self.name!r}.")
        return _finite(out).astype(np.float32, copy=False)

    def apply_pair(self, X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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
    if hasattr(item, "abbreviation"):
        name = _count_transformation_name(item)
        info = transformation_label(name)
        return {"transformation_abbreviation": str(getattr(item, "abbreviation", info.abbreviation))}
    if isinstance(item, tuple):
        name = str(item[0])
    else:
        name = str(item)
    return {"transformation_abbreviation": transformation_label(name).abbreviation}


class CountTransformationAdapter:


    def __init__(self, name: str, spec: Any = None, *, random_state: int = 42):
        self.name = transformation_label(str(name)).key
        self.spec = spec
        self.random_state = int(random_state)
        self.obj: Any | None = None

    def _make(self) -> Any:
        if self.spec is None:
            return CountTransformation(self.name, random_state=self.random_state)
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
            raise RuntimeError(f"Count transformation {self.name!r} has not been fitted.")
        X = _as_float_matrix(X)
        obj = self.obj
        if hasattr(obj, "apply"):
            return _finite(obj.apply(X)).astype(np.float32, copy=False)
        if hasattr(obj, "transform"):
            return _finite(obj.transform(X)).astype(np.float32, copy=False)
        if callable(obj):
            return _finite(obj(X)).astype(np.float32, copy=False)
        raise TypeError(
            f"Custom count transformation {self.name!r} must be callable or provide fit/apply or fit/transform."
        )

    def apply_pair(self, X_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        if isinstance(self.spec, Transform) or (hasattr(self.spec, "name") and hasattr(self.spec, "apply")):
            a, b = self.spec.apply(_as_float_matrix(X_tr), _as_float_matrix(X_te))
            return _finite(a).astype(np.float32, copy=False), _finite(b).astype(np.float32, copy=False)
        if callable(self.spec) and _callable_accepts_two_required(self.spec):
            a, b = self.spec(_as_float_matrix(X_tr), _as_float_matrix(X_te))
            return _finite(a).astype(np.float32, copy=False), _finite(b).astype(np.float32, copy=False)
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
        p for p in sig.parameters.values()
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
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is inspect._empty
    ]
    return len(positional) >= 2


def _count_transformation_factory(
    item: Any, *, random_state: int
) -> tuple[str, Callable[[], CountTransformationAdapter]]:
    if isinstance(item, Transform) or (hasattr(item, "name") and hasattr(item, "apply")):
        name = transformation_label(str(getattr(item, "name"))).key
        return name, lambda item=item, name=name, random_state=random_state: CountTransformationAdapter(
            name, item, random_state=random_state
        )
    if isinstance(item, tuple):
        raw_name, spec = item
        name = transformation_label(str(raw_name)).key
        return name, lambda name=name, spec=spec, random_state=random_state: CountTransformationAdapter(
            name, spec, random_state=random_state
        )
    name = transformation_label(str(item)).key
    return name, lambda name=name, random_state=random_state: CountTransformationAdapter(
        name, random_state=random_state
    )

