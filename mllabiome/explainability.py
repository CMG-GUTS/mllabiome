from __future__ import annotations

import json
import logging
import math
import io
import shutil
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .data import load_dataset
from .console import console, info, path_table, stage, success, summary_table
from .learners import _learner_factory
from .metrics import _estimator_call, _predict_proba_aligned
from .configs_sweep import Sweep, _groups_from_metadata, _outer_splits, _strata_from_metadata
from .resolutions import materialize_mpdr
from .transformations import CountTransformationAdapter, _count_transformation_factory
from .utils import _as_float_matrix, dump_json_standard
from .style import (
    ACC_D, ACC_L, BG, COL_W_2, DIM, INK, MID, TRACK, UC_CASE, UC_CTRL, apply as apply_style, save_all
)
from .explainability_visuals import plot_feature_support as _plot_feature_support_visual, plot_interaction_network as _plot_interaction_network_visual



# Keep third-party diagnostics from breaking the rich progress display.
for _logger_name in ("PyALE", "PyALE._ALE_generic"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
    logging.getLogger(_logger_name).propagate = False

@contextmanager
def _quiet_pyale_info():
    saved_disable = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(saved_disable)


@contextmanager
def _quiet_external_progress():
    """Suppress third-party tqdm/progress chatter while preserving exceptions.

    SHAP can instantiate tqdm progress bars from several import locations and,
    may ignore ``silent=True`` for model-agnostic
    explainers.  Redirecting stdout/stderr alone is not always enough because
    tqdm may keep a cached file handle.  This context therefore also patches the
    common tqdm constructors so third-party progress bars are disabled while the
    rich mllabiome status line remains the only terminal progress indicator.
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    patches: list[tuple[Any, str, Any]] = []

    def _disabled_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return _ORIG_TQDM(*args, **kwargs)

    try:
        import tqdm as _tqdm_mod  # type: ignore
        _ORIG_TQDM = _tqdm_mod.tqdm
        patches.append((_tqdm_mod, "tqdm", _ORIG_TQDM))
        _tqdm_mod.tqdm = _disabled_tqdm
        try:
            import tqdm.auto as _tqdm_auto  # type: ignore
            patches.append((_tqdm_auto, "tqdm", _tqdm_auto.tqdm))
            _tqdm_auto.tqdm = _disabled_tqdm
        except Exception:
            pass
        try:
            import tqdm.std as _tqdm_std  # type: ignore
            patches.append((_tqdm_std, "tqdm", _tqdm_std.tqdm))
            _tqdm_std.tqdm = _disabled_tqdm
        except Exception:
            pass
    except Exception:
        _ORIG_TQDM = None  # type: ignore[assignment]

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            yield
    finally:
        for obj, name, value in reversed(patches):
            try:
                setattr(obj, name, value)
            except Exception:
                pass

class ExplainabilityConfigurationError(RuntimeError):
    """Raised when explainability cannot run exactly as configured."""


class ExplainabilityDependencyError(RuntimeError):
    """Raised when a configured explainability method lacks a dependency."""


_EXPLAINABILITY_METHODS = {"shap", "lime", "ale", "permutation", "interactions"}


def _normalise_explainability_methods(methods: Sequence[str]) -> tuple[str, ...]:
    aliases = {"ale_interactions": "interactions", "interaction": "interactions", "lime_tabular": "lime"}
    out = tuple(aliases.get(str(m).strip().lower().replace(" ", "_"), str(m).strip().lower().replace(" ", "_")) for m in methods if str(m).strip())
    if not out:
        raise ExplainabilityConfigurationError("Explainability.methods must contain at least one explicit method.")
    unsupported = sorted(set(out) - _EXPLAINABILITY_METHODS)
    if unsupported:
        raise ExplainabilityConfigurationError(
            "Unsupported explainability method(s): " + ", ".join(unsupported) +
            ". Supported methods are: " + ", ".join(sorted(_EXPLAINABILITY_METHODS)) +
            ". No fallback method will be used."
        )
    return out


def _preflight_explainability_dependencies(methods: Sequence[str]) -> None:
    """Fail before fitting/explaining if any requested strict method lacks its package."""
    missing: list[str] = []
    if "shap" in methods:
        try:
            import shap  # noqa: F401
        except Exception as exc:  # pragma: no cover
            missing.append(f"shap ({exc})")
    if "lime" in methods:
        try:
            from lime.lime_tabular import LimeTabularExplainer  # noqa: F401
        except Exception as exc:  # pragma: no cover
            missing.append(f"lime ({exc})")
    if "ale" in methods or "interactions" in methods:
        try:
            from PyALE import ale as _pyale_preflight  # noqa: F401
        except Exception as exc:  # pragma: no cover
            missing.append(f"PyALE ({exc})")
    if "interactions" in methods:
        try:
            import networkx as nx  # noqa: F401
        except Exception as exc:  # pragma: no cover
            missing.append(f"networkx ({exc})")
    if missing:
        raise ExplainabilityDependencyError(
            "Strict explainability cannot start because requested method dependencies are missing: "
            + "; ".join(missing)
            + ". Install the required package dependencies or explicitly remove the corresponding method(s). No fallback method will be used."
        )


def _configured_count_transformation_factory(sweep: Sweep, key: str) -> Callable[[], CountTransformationAdapter]:
    factories = dict(
        _count_transformation_factory(x, random_state=sweep.explainability.random_state)
        for x in sweep.count_transformations
    )
    if key not in factories:
        raise ExplainabilityConfigurationError(
            f"The selected MPMA uses count transformation {key!r}, but that exact transformation "
            "is not present in sweep.count_transformations. Explainability is strict and will not "
            "reconstruct or substitute transformations by name."
        )
    return factories[key]


def _configured_learner_factory(sweep: Sweep, key: str) -> Callable[[], BaseEstimator]:
    factories = dict(_learner_factory(x) for x in sweep.learners)
    if key not in factories:
        raise ExplainabilityConfigurationError(
            f"The selected MPMA uses learner {key!r}, but that exact learner is not present in "
            "sweep.learners. Explainability is strict and will not reconstruct or substitute learners by name."
        )
    return factories[key]

class _FittedMpmaEnsemble(BaseEstimator):
    def __init__(self, members: list[dict[str, Any]], aggregation: str = "mean_proba"):
        self.members = members
        self.aggregation = aggregation
        self.classes_ = np.asarray(members[0]["classes"], dtype=int) if members else np.array([0, 1], dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_float_matrix(X)
        probs = []
        for member in self.members:
            sl = member["slice"]
            clf = member["estimator"]
            probs.append(_predict_proba_aligned(clf, X[:, sl], self.classes_))
        stack = np.stack(probs, axis=0)
        return _aggregate_member_proba(stack, self.aggregation)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _aggregate_member_proba(stack: np.ndarray, aggregation: str) -> np.ndarray:
    aggregation = str(aggregation or "mean_proba")
    if aggregation == "median_proba":
        proba = np.median(stack, axis=0)
    elif aggregation == "rank_mean":
        ranks = np.empty_like(stack, dtype=float)
        for m in range(stack.shape[0]):
            for i in range(stack.shape[1]):
                ranks[m, i, :] = rankdata(stack[m, i, :], method="average")
        proba = ranks.mean(axis=0)
    elif aggregation == "majority_vote":
        votes = np.argmax(stack, axis=2)
        proba = np.zeros((stack.shape[1], stack.shape[2]), dtype=float)
        for i in range(votes.shape[1]):
            counts = np.bincount(votes[:, i], minlength=stack.shape[2]).astype(float)
            proba[i] = counts / max(float(counts.sum()), 1.0)
    else:
        proba = stack.mean(axis=0)
    proba = np.asarray(proba, dtype=float)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return proba / row_sums


def _positive_response(clf: BaseEstimator, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    proba = _predict_proba_aligned(clf, X, classes)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba.max(axis=1)


def _explain_predict_proba(clf: BaseEstimator, classes: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X_: _predict_proba_aligned(clf, _as_float_matrix(X_), classes)


def _explain_predict_response(clf: BaseEstimator, classes: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X_: _positive_response(clf, _as_float_matrix(X_), classes)


def _explain_decision_function(clf: BaseEstimator) -> Callable[[np.ndarray], np.ndarray]:
    return lambda X_: _estimator_call(clf, "decision_function", _as_float_matrix(X_))


def _sample_rows(X: np.ndarray, *, max_rows: int, random_state: int) -> np.ndarray:
    X = _as_float_matrix(X)
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return np.arange(X.shape[0])
    rng = np.random.default_rng(int(random_state))
    return np.sort(rng.choice(np.arange(X.shape[0]), size=int(max_rows), replace=False))


def _permutation_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    classes = np.arange(len(class_labels), dtype=int)
    X = _as_float_matrix(X)
    y = np.asarray(y, dtype=int)

    if len(class_labels) == 2:
        scoring_name = "roc_auc"
        pos = int(classes[-1])
        pos_col = int(np.where(classes == pos)[0][0])

        def scoring(estimator: BaseEstimator, X_eval: np.ndarray, y_eval: np.ndarray) -> float:
            proba = _predict_proba_aligned(estimator, _as_float_matrix(X_eval), classes)
            y_bin = (np.asarray(y_eval, dtype=int) == pos).astype(int)
            if len(np.unique(y_bin)) < 2:
                pred = classes[np.argmax(proba, axis=1)]
                return float(balanced_accuracy_score(np.asarray(y_eval, dtype=int), pred))
            return float(roc_auc_score(y_bin, proba[:, pos_col]))
    else:
        scoring_name = "balanced_accuracy"

        def scoring(estimator: BaseEstimator, X_eval: np.ndarray, y_eval: np.ndarray) -> float:
            proba = _predict_proba_aligned(estimator, _as_float_matrix(X_eval), classes)
            pred = classes[np.argmax(proba, axis=1)]
            return float(balanced_accuracy_score(np.asarray(y_eval, dtype=int), pred))

    try:
        perm = permutation_importance(
            clf,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
        )
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityConfigurationError(
            "Permutation explainability failed for the selected MPMA. No fallback method will be used."
        ) from exc
    return pd.DataFrame(
        {
            "method": "permutation",
            "feature": list(feature_names),
            "importance_mean": perm.importances_mean,
            "importance_sd": perm.importances_std,
            "scoring": scoring_name,
        }
    ).sort_values("importance_mean", ascending=False)


def _shap_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    random_state: int,
    max_background: int,
    max_explain: int,
) -> pd.DataFrame:
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external package
        raise ExplainabilityDependencyError(
            "Explainability.methods includes 'shap', but the 'shap' package is not importable. "
            "Install shap or remove 'shap' from Explainability.methods. No fallback method will be used."
        ) from exc

    X = _as_float_matrix(X)
    rows_bg = _sample_rows(X, max_rows=max_background, random_state=random_state)
    rows_ex = _sample_rows(X, max_rows=max_explain, random_state=random_state + 13)
    background = X[rows_bg]
    X_explain = X[rows_ex]

    classes = np.arange(len(class_labels), dtype=int)
    if hasattr(clf, "predict_proba"):
        model_fn = _explain_predict_proba(clf, classes)
    elif hasattr(clf, "decision_function"):
        model_fn = _explain_decision_function(clf)
    else:
        raise ExplainabilityConfigurationError(
            f"Learner {type(clf).__name__} exposes neither predict_proba nor decision_function; "
            "SHAP explainability cannot run exactly as configured. No fallback method will be used."
        )

    min_max_evals = max(500, 2 * len(feature_names) + 1)
    try:
        with _quiet_external_progress():
            explainer = shap.Explainer(model_fn, background, feature_names=list(feature_names))
            try:
                values = explainer(X_explain, silent=True, max_evals=min_max_evals)
            except TypeError:
                try:
                    values = explainer(X_explain, max_evals=min_max_evals)
                except TypeError:
                    try:
                        values = explainer(X_explain, silent=True)
                    except TypeError:
                        values = explainer(X_explain)
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityConfigurationError(
            "SHAP failed for the selected MPMA. Explainability is strict, so execution stops "
            "instead of replacing SHAP with another method."
        ) from exc

    arr = np.asarray(values.values, dtype=float)
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            arr = arr[:, :, 1]
        else:
            arr = np.mean(np.abs(arr), axis=2)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ExplainabilityConfigurationError(
            f"Unexpected SHAP value shape {arr.shape}; expected n_samples × n_features. "
            "No fallback method will be used."
        )
    score = np.mean(np.abs(arr), axis=0)
    return pd.DataFrame(
        {
            "method": "shap",
            "feature": list(feature_names),
            "importance_mean": score,
            "importance_sd": np.std(np.abs(arr), axis=0, ddof=1) if arr.shape[0] > 1 else 0.0,
            "scoring": "mean_abs_shap",
        }
    ).sort_values("importance_mean", ascending=False)


def _lime_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    random_state: int,
    n_samples: int,
    max_explain: int,
) -> pd.DataFrame:
    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external package
        raise ExplainabilityDependencyError(
            "Explainability.methods includes 'lime', but the 'lime' package is not importable. "
            "Install lime or remove 'lime' from Explainability.methods. No fallback method will be used."
        ) from exc

    X = _as_float_matrix(X)
    rows = _sample_rows(X, max_rows=max_explain, random_state=random_state + 29)
    X_explain = X[rows]
    predict_fn = _explain_predict_proba(clf, np.arange(len(class_labels), dtype=int))
    class_names = list(class_labels)
    label = 1 if len(class_names) == 2 else 0
    try:
        explainer = LimeTabularExplainer(
            training_data=X,
            feature_names=list(feature_names),
            class_names=class_names,
            mode="classification",
            discretize_continuous=False,
            random_state=int(random_state),
        )
        coeffs = np.zeros((X_explain.shape[0], len(feature_names)), dtype=float)
        for i, row in enumerate(X_explain):
            exp = explainer.explain_instance(
                row,
                predict_fn=predict_fn,
                num_features=len(feature_names),
                num_samples=int(n_samples),
                labels=(label,),
            )
            for fi, coef in exp.local_exp[label]:
                if 0 <= int(fi) < len(feature_names):
                    coeffs[i, int(fi)] = float(coef)
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityConfigurationError(
            "LIME failed for the selected MPMA. Explainability is strict, so execution stops "
            "instead of replacing LIME with another method."
        ) from exc

    score = np.mean(np.abs(coeffs), axis=0)
    return pd.DataFrame(
        {
            "method": "lime",
            "feature": list(feature_names),
            "importance_mean": score,
            "importance_sd": np.std(np.abs(coeffs), axis=0, ddof=1) if coeffs.shape[0] > 1 else 0.0,
            "scoring": f"mean_abs_lime_coefficients_label_{label}",
        }
    ).sort_values("importance_mean", ascending=False)


class _AleModelWrapper:
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self._fn = fn

    def predict(self, X: Any) -> np.ndarray:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.asarray(self._fn(arr), dtype=float)


def _require_pyale():
    try:
        from PyALE import ale as pyale  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external package
        raise ExplainabilityDependencyError(
            "Explainability.methods includes 'ale' or 'interactions', but the 'PyALE' package is not importable. "
            "Install PyALE or remove ALE-based methods from Explainability.methods. No fallback method will be used."
        ) from exc
    return pyale


def _ale_result_values(result: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Extract finite ALE effect values and optional aligned coordinates.

    For 2D PyALE surfaces, coordinates are returned only when PyALE exposes a
    MultiIndex aligned one-to-one with the flattened effect vector.  Matrix-like
    2D outputs are still valid for the uncorrected RMS score, but their marginal
    correction is marked unavailable instead of trying to broadcast row/column
    axes against the flattened surface.
    """
    x1 = x2 = None
    if isinstance(result, pd.DataFrame):
        df = result.copy()
        if "eff" in df.columns:
            vals = df["eff"].to_numpy(dtype=float)
        else:
            num = df.select_dtypes(include=[np.number])
            drop_cols = [
                c for c in num.columns
                if str(c).lower() in {"size", "count", "counts", "n"} or "ci" in str(c).lower()
            ]
            if drop_cols:
                num = num.drop(columns=drop_cols, errors="ignore")
            vals = num.to_numpy(dtype=float).ravel()
        if isinstance(df.index, pd.MultiIndex) and len(df.index) == len(vals):
            x1 = np.asarray([idx[0] for idx in df.index], dtype=float)
            x2 = np.asarray([idx[1] for idx in df.index], dtype=float)
        elif not isinstance(df.index, pd.MultiIndex) and len(df.index) == len(vals):
            # 1D ALE result.
            try:
                x1 = df.index.to_numpy(dtype=float)
            except Exception:
                x1 = None
    elif isinstance(result, pd.Series):
        vals = result.to_numpy(dtype=float)
        if isinstance(result.index, pd.MultiIndex) and len(result.index) == len(vals):
            x1 = np.asarray([idx[0] for idx in result.index], dtype=float)
            x2 = np.asarray([idx[1] for idx in result.index], dtype=float)
        elif len(result.index) == len(vals):
            try:
                x1 = result.index.to_numpy(dtype=float)
            except Exception:
                x1 = None
    else:
        vals = np.asarray(result, dtype=float).ravel()
    vals = np.asarray(vals, dtype=float).ravel()
    finite = np.isfinite(vals)
    if x1 is not None and x2 is not None:
        finite = finite & np.isfinite(x1) & np.isfinite(x2)
        return vals[finite], x1[finite], x2[finite]
    if x1 is not None and x2 is None and len(x1) == len(vals):
        finite = finite & np.isfinite(x1)
        return vals[finite], x1[finite], None
    return vals[finite], None, None


def _ale_feature_importance(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    n_bins: int,
    top_features: Sequence[str] | None,
) -> pd.DataFrame:
    pyale = _require_pyale()
    X = _as_float_matrix(X)
    feature_names = list(feature_names)
    name_to_index = {str(name): i for i, name in enumerate(feature_names)}
    features = list(top_features) if top_features else feature_names
    features = [str(f) for f in features if str(f) in name_to_index]
    if not features:
        raise ExplainabilityConfigurationError("ALE was requested, but no candidate features were available. No fallback method will be used.")
    df_X = pd.DataFrame(X, columns=feature_names)
    wrapper = _AleModelWrapper(_explain_predict_response(clf, np.arange(len(class_labels), dtype=int)))
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for fname in features:
        col = X[:, name_to_index[fname]]
        finite_col = col[np.isfinite(col)]
        n_distinct = int(np.unique(finite_col).size)
        if n_distinct < 2:
            skipped.append({
                "feature": fname,
                "reason": "fewer_than_two_distinct_finite_values",
                "n_finite": int(finite_col.size),
                "n_distinct": n_distinct,
            })
            continue
        try:
            with _quiet_pyale_info():
                result = pyale(df_X, wrapper, [fname], grid_size=int(n_bins), include_CI=False, plot=False)
            vals, grid, _ = _ale_result_values(result)
        except Exception as exc:  # noqa: BLE001
            skipped.append({
                "feature": fname,
                "reason": "pyale_failed",
                "error": str(exc)[:500],
                "n_finite": int(finite_col.size),
                "n_distinct": n_distinct,
            })
            continue
        if vals.size == 0:
            skipped.append({
                "feature": fname,
                "reason": "no_finite_ale_effect_values",
                "n_finite": int(finite_col.size),
                "n_distinct": n_distinct,
            })
            continue
        centred = vals - float(np.mean(vals))
        strength = float(np.sqrt(np.mean(centred ** 2)))
        rows.append({
            "method": "ale",
            "feature": fname,
            "importance_mean": strength,
            "importance_sd": float(np.std(centred, ddof=1)) if len(centred) > 1 else 0.0,
            "scoring": "outer_fold_rms_centered_ale",
            "n_ale_points": int(vals.size),
        })
        if grid is not None and len(grid) == len(vals):
            for gx, gy in zip(grid, vals):
                curves.append({"feature": fname, "grid": float(gx), "ale_effect": float(gy)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("importance_mean", ascending=False)
    out.attrs["curves"] = pd.DataFrame(curves)
    out.attrs["skipped_features"] = pd.DataFrame(skipped)
    return out


def _same_lineage_pair(f1: str, f2: str) -> bool:
    return str(f1).startswith(str(f2)) or str(f2).startswith(str(f1))


def _candidate_pairs_from_scores(X: np.ndarray, feature_names: Sequence[str], scores: pd.Series, max_pairs: int) -> list[tuple[int, int]]:
    X = _as_float_matrix(X)
    if X.shape[1] < 2 or max_pairs <= 0:
        return []
    name_to_score = {str(k): float(v) for k, v in scores.items()}
    def usable(i: int) -> bool:
        x = X[:, i]
        return np.isfinite(x).sum() >= 3 and np.unique(x[np.isfinite(x)]).size >= 2 and not str(feature_names[i]).startswith("ILR")
    idx = [i for i in range(X.shape[1]) if usable(i)]
    if len(idx) < 2:
        return []
    order = sorted(idx, key=lambda i: name_to_score.get(str(feature_names[i]), float(np.nanvar(X[:, i]))), reverse=True)
    # take a modest frontier and then form pair candidates
    frontier = order[: min(len(order), max(4, int(math.ceil(math.sqrt(max_pairs * 2))) + 6))]
    cand: list[tuple[float, int, int]] = []
    for a, i in enumerate(frontier):
        for j in frontier[a + 1:]:
            if _same_lineage_pair(str(feature_names[i]), str(feature_names[j])):
                continue
            si = name_to_score.get(str(feature_names[i]), float(np.nanvar(X[:, i])))
            sj = name_to_score.get(str(feature_names[j]), float(np.nanvar(X[:, j])))
            cand.append((float(si + sj), i, j))
    cand.sort(key=lambda z: z[0], reverse=True)
    return [(i, j) for _, i, j in cand[:max_pairs]]


def _interp_unique(x: np.ndarray, y: np.ndarray, xp: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 2:
        raise ValueError("not enough finite support")
    ser = pd.Series(y, index=x).groupby(level=0).mean().sort_index()
    if len(ser) < 2:
        raise ValueError("not enough unique support")
    return np.interp(xp, ser.index.to_numpy(dtype=float), ser.to_numpy(dtype=float))


def _ale_interactions(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    n_bins: int,
    top_k_interactions: int,
    scores: pd.Series,
) -> dict[str, pd.DataFrame]:
    pyale = _require_pyale()
    X = _as_float_matrix(X)
    df_X = pd.DataFrame(X, columns=list(feature_names))
    wrapper = _AleModelWrapper(_explain_predict_response(clf, np.arange(len(class_labels), dtype=int)))
    pairs = _candidate_pairs_from_scores(X, feature_names, scores, max_pairs=int(top_k_interactions))
    if not pairs:
        raise ExplainabilityConfigurationError("Interactions were requested, but no usable feature pairs were available. No fallback method will be used.")
    raw_rows = []
    for i, j in pairs:
        f1, f2 = str(feature_names[i]), str(feature_names[j])
        try:
            with _quiet_pyale_info():
                result2d = pyale(df_X, wrapper, [f1, f2], grid_size=int(n_bins), include_CI=False, plot=False)
            vals2d, x1, x2 = _ale_result_values(result2d)
        except Exception:
            continue
        if vals2d.size == 0:
            continue
        centred = vals2d - float(np.mean(vals2d))
        current = float(np.sqrt(np.mean(centred ** 2)))
        corrected = float("nan")
        correction_applied = False
        correction_error = ""
        if x1 is None or x2 is None:
            correction_error = "2D PyALE result does not expose coordinates for marginal correction"
        else:
            try:
                with _quiet_pyale_info():
                    res1 = pyale(df_X, wrapper, [f1], grid_size=int(n_bins), include_CI=False, plot=False)
                    res2 = pyale(df_X, wrapper, [f2], grid_size=int(n_bins), include_CI=False, plot=False)
                vals1, g1, _ = _ale_result_values(res1)
                vals2, g2, _ = _ale_result_values(res2)
                if g1 is None or g2 is None:
                    raise ValueError("1D PyALE result does not expose grids")
                comp = vals2d - _interp_unique(g1, vals1, x1) - _interp_unique(g2, vals2, x2)
                comp = comp - float(np.mean(comp))
                corrected = float(np.sqrt(np.mean(comp ** 2)))
                correction_applied = True
            except Exception as exc:  # noqa: BLE001
                # This is not a fallback: corrected methods are reported as NaN when the correction cannot be defined,
                # no substitute score is reported for corrected ALE interactions.
                correction_error = str(exc)[:240]
        raw_rows.append({
            "feature_1": f1,
            "feature_2": f2,
            "current": current,
            "corrected": corrected,
            "fixed_pairs": current,
            "corrected_fixed": corrected,
            "n_ale_cells": int(vals2d.size),
            "correction_applied": bool(correction_applied),
            "correction_error": correction_error,
        })
    if not raw_rows:
        raise ExplainabilityConfigurationError("ALE interactions were requested, but no candidate pair produced a finite 2D ALE surface. No fallback method will be used.")
    raw = pd.DataFrame(raw_rows)
    out: dict[str, pd.DataFrame] = {}
    for method in ("current", "corrected", "fixed_pairs", "corrected_fixed"):
        d = raw[["feature_1", "feature_2", method, "n_ale_cells", "correction_applied", "correction_error"]].copy()
        d = d.rename(columns={method: "interaction_strength"})
        d["method"] = method
        d = d.sort_values("interaction_strength", ascending=False, na_position="last")
        out[method] = d
    out["all_methods"] = raw
    return out


def _collapse_duplicate_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate feature rows before ranking/support lookups."""
    if frame.empty or "feature" not in frame.columns or not frame["feature"].duplicated().any():
        return frame.copy()

    d = frame.copy()
    d["feature"] = d["feature"].astype(str)
    d["importance_mean"] = pd.to_numeric(d.get("importance_mean", 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "importance_sd" in d.columns:
        d["importance_sd"] = pd.to_numeric(d["importance_sd"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        d["importance_sd"] = 0.0

    def _combine_text(values: pd.Series) -> str:
        seen: list[str] = []
        for value in values:
            s = str(value)
            if s not in seen:
                seen.append(s)
        return "+".join(seen)

    agg: dict[str, Any] = {
        "importance_mean": "sum",
        "importance_sd": lambda s: float(np.sqrt(np.sum(np.square(pd.to_numeric(s, errors="coerce").fillna(0.0))))),
    }
    if "method" in d.columns:
        agg["method"] = "first"
    if "scoring" in d.columns:
        agg["scoring"] = _combine_text
    if "mean_rank" in d.columns:
        agg["mean_rank"] = "mean"
    if "n_methods" in d.columns:
        agg["n_methods"] = "max"

    collapsed = d.groupby("feature", as_index=False, sort=False).agg(agg)
    ordered = [c for c in frame.columns if c in collapsed.columns]
    ordered.extend(c for c in collapsed.columns if c not in ordered)
    return collapsed[ordered]

def _combine_feature_importance(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        return _collapse_duplicate_feature_importance(frames[0]).sort_values("importance_mean", ascending=False)
    long = []
    for frame in frames:
        d = _collapse_duplicate_feature_importance(frame)[["method", "feature", "importance_mean"]].copy()
        d["rank"] = d.groupby("method")["importance_mean"].rank(ascending=False, method="average")
        long.append(d)
    pooled = pd.concat(long, ignore_index=True)
    out = (
        pooled.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            mean_rank=("rank", "mean"),
            n_methods=("method", "nunique"),
        )
        .sort_values(["mean_rank", "importance_mean"], ascending=[True, False])
    )
    out["method"] = "consensus"
    out["importance_sd"] = np.nan
    out["scoring"] = "mean_rank_then_mean_importance"
    return out[["method", "feature", "importance_mean", "importance_sd", "scoring", "mean_rank", "n_methods"]]




def _single_method_support_table(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Top-feature table for one completed explainability method.

    This is intentionally method-local: the method figure is written immediately
    after that method finishes, without waiting for the full explainability suite.
    The combined support table is still built at the end from all completed
    methods.
    """
    frame = _collapse_duplicate_feature_importance(frame)
    method = _method_display(str(frame["method"].iloc[0]))
    top = frame.sort_values("importance_mean", ascending=False).head(int(top_k)).copy()
    top["rank"] = np.arange(1, len(top) + 1, dtype=int)
    top[method] = _support_from_importance(frame).reindex(top["feature"]).fillna(0.0).to_numpy(float)
    top["consensus"] = top[method].astype(float)
    if "mean_rank" not in top.columns:
        top["mean_rank"] = top["rank"].astype(float)
    keep = ["rank", "feature", method, "consensus", "importance_mean", "mean_rank"]
    if "n_methods" in top.columns:
        keep.append("n_methods")
    return top[keep]


def _write_method_outputs(
    frame: pd.DataFrame,
    *,
    target_dir: Path,
    figures_dir: Path,
    X_base: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_labels: list[str],
    top_k: int,
) -> dict[str, Path]:
    """Persist method-specific outputs immediately after a method completes."""
    method = str(frame["method"].iloc[0]).strip().lower()
    table_path = target_dir / f"feature_importance_{method}.tsv"
    frame.to_csv(table_path, sep="\t", index=False)

    top = _single_method_support_table(frame, top_k=top_k)
    top_path = target_dir / f"top_features_{method}.tsv"
    top.to_csv(top_path, sep="\t", index=False)

    dist = _feature_distribution_stats(
        top["feature"].astype(str).tolist(),
        feature_names,
        X_base,
        y,
        class_labels,
    )
    dist_path = target_dir / f"feature_distribution_stats_{method}.tsv"
    dist.to_csv(dist_path, sep="\t", index=False)

    importance_stem = figures_dir / f"feature_importance_{method}"
    support_stem = figures_dir / f"feature_support_{method}"
    _plot_feature_importance(top, dist, importance_stem, top_k, class_labels)
    _plot_feature_importance(top, dist, support_stem, top_k, class_labels)
    return {
        f"importance_{method}": table_path,
        f"top_features_{method}": top_path,
        f"feature_distribution_{method}": dist_path,
        f"figure_{method}": importance_stem.with_suffix(".png"),
        f"feature_support_{method}": support_stem.with_suffix(".png"),
    }


def _select_instance_indices(dataset: Any, clf: BaseEstimator, X: np.ndarray, *, sample_ids: Sequence[str], representative: bool) -> list[tuple[int, str]]:
    """Return (row_index, role) pairs for instance-level explanation."""
    selected: list[tuple[int, str]] = []
    sid_to_idx = {str(sid): i for i, sid in enumerate(dataset.sample_ids)}
    for sid in sample_ids:
        if str(sid) not in sid_to_idx:
            raise ExplainabilityConfigurationError(f"Requested instance sample_id {sid!r} was not found in the loaded dataset.")
        idx = sid_to_idx[str(sid)]
        selected.append((idx, f"requested:{sid}"))
    if representative:
        proba = _predict_proba_aligned(clf, X, np.arange(len(dataset.class_labels), dtype=int))
        pos = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        for cls in sorted(set(int(x) for x in dataset.y.tolist())):
            idxs = np.where(dataset.y == cls)[0]
            if len(idxs) == 0:
                continue
            cls_probs = pos[idxs]
            centre = float(np.nanmedian(cls_probs)) if np.isfinite(cls_probs).any() else 0.5
            local = int(idxs[int(np.nanargmin(np.abs(cls_probs - centre)))])
            role = "representative_case" if cls == 1 else "representative_control"
            if local not in [i for i, _ in selected]:
                selected.append((local, role))
    # preserve order, unique indices
    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for idx, role in selected:
        if idx not in seen:
            seen.add(idx); out.append((idx, role))
    return out


def _shap_instance_explanations(
    clf: BaseEstimator,
    X: np.ndarray,
    feature_names: Sequence[str],
    dataset: Any,
    *,
    selected: list[tuple[int, str]],
    max_background: int,
    top_features_per_direction: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute local SHAP rows for requested/representative samples."""
    if not selected:
        return pd.DataFrame(), pd.DataFrame()
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ExplainabilityDependencyError("SHAP instance explanations require shap. No fallback method will be used.") from exc
    rows_bg = _sample_rows(X, max_rows=max_background, random_state=random_state)
    background = X[rows_bg]
    sample_idx = np.array([i for i, _ in selected], dtype=int)
    X_sel = X[sample_idx]
    classes = np.arange(len(dataset.class_labels), dtype=int)
    if hasattr(clf, "predict_proba"):
        model_fn = _explain_predict_proba(clf, classes)
    elif hasattr(clf, "decision_function"):
        model_fn = _explain_decision_function(clf)
    else:
        raise ExplainabilityConfigurationError("Selected learner exposes neither predict_proba nor decision_function for SHAP instance explanations.")
    min_max_evals = max(500, 2 * len(feature_names) + 1)
    try:
        with _quiet_external_progress():
            explainer = shap.Explainer(model_fn, background, feature_names=list(feature_names))
            try:
                values = explainer(X_sel, silent=True, max_evals=min_max_evals)
            except TypeError:
                try:
                    values = explainer(X_sel, max_evals=min_max_evals)
                except TypeError:
                    try:
                        values = explainer(X_sel, silent=True)
                    except TypeError:
                        values = explainer(X_sel)
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityConfigurationError("SHAP instance explanations failed. No fallback method will be used.") from exc
    arr = np.asarray(values.values, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, 1] if arr.shape[2] == 2 else np.mean(arr, axis=2)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ExplainabilityConfigurationError(f"Unexpected SHAP instance value shape {arr.shape}. No fallback method will be used.")
    proba = _predict_proba_aligned(clf, X_sel, np.arange(len(dataset.class_labels), dtype=int))
    pos = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    pred = np.argmax(proba, axis=1)
    selected_rows = []
    top_rows = []
    k = max(1, int(top_features_per_direction))
    for local_row, (idx, role) in enumerate(selected):
        sid = str(dataset.sample_ids[int(idx)])
        selected_rows.append({
            "sample_id": sid,
            "sample_index": int(idx),
            "selection_role": role,
            "true_class": int(dataset.y[int(idx)]),
            "predicted_class": int(pred[local_row]),
            "p_positive": float(pos[local_row]),
            "p_positive_mean": float(pos[local_row]),
            "n_oof_explanations": 1,
        })
        vals = arr[local_row]
        order_neg = np.argsort(vals)[:k]
        order_pos = np.argsort(-vals)[:k]
        chosen = list(dict.fromkeys([int(i) for i in list(order_neg) + list(order_pos)]))
        chosen = sorted(chosen, key=lambda j: abs(float(vals[j])), reverse=True)
        for rank, j in enumerate(chosen, start=1):
            top_rows.append({
                "sample_id": sid,
                "sample_index": int(idx),
                "selection_role": role,
                "true_class": int(dataset.y[int(idx)]),
                "predicted_class": int(pred[local_row]),
                "p_positive": float(pos[local_row]),
                "p_positive_mean": float(pos[local_row]),
                "feature": str(feature_names[j]),
                "value": float(vals[j]),
                "abs_value": float(abs(vals[j])),
                "rank": int(rank),
            })
    return pd.DataFrame(top_rows), pd.DataFrame(selected_rows)


def _plot_ale_curves(curves: pd.DataFrame, top_features: Sequence[str], out_stem: Path, *, max_panels: int = 12) -> None:
    """Create small-multiple ALE curves for top features."""
    apply_style()
    import math as _math
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as mpe
    from .style import MM, BG, INK, MID, TRACK, ACC_D, ACC_L, save_all
    if curves is None or curves.empty or not {"feature", "grid", "ale_effect"}.issubset(curves.columns):
        fig = plt.figure(figsize=(120 * MM, 40 * MM)); ax = fig.add_axes([0.06, 0.15, 0.88, 0.70]); ax.axis("off")
        ax.text(0.5, 0.5, "ALE curves unavailable", ha="center", va="center", fontsize=7, color=MID)
        save_all(fig, out_stem); plt.close(fig); return
    order = [str(f) for f in top_features if str(f) in set(curves["feature"].astype(str))]
    if not order:
        order = list(dict.fromkeys(curves["feature"].astype(str).tolist()))
    order = order[: max(1, int(max_panels))]
    n = len(order)
    n_cols = 3 if n > 4 else min(2, n)
    n_rows = int(_math.ceil(n / max(n_cols, 1)))
    fig_w = 150 if n_cols == 3 else 112
    fig_h = max(44, 31 * n_rows + 10)
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM)); fig.patch.set_facecolor(BG)
    left, right, bottom, top = 0.075, 0.035, 0.095, 0.055
    gap_x, gap_y = 0.055, 0.115
    cell_w = (1 - left - right - gap_x * (n_cols - 1)) / n_cols
    cell_h = (1 - bottom - top - gap_y * (n_rows - 1)) / n_rows
    for idx, feat in enumerate(order):
        row, col = divmod(idx, n_cols)
        x0 = left + col * (cell_w + gap_x)
        y0 = 1 - top - (row + 1) * cell_h - row * gap_y
        ax = fig.add_axes([x0, y0, cell_w, cell_h])
        d = curves[curves["feature"].astype(str).eq(feat)].copy()
        d["grid"] = pd.to_numeric(d["grid"], errors="coerce")
        d["ale_effect"] = pd.to_numeric(d["ale_effect"], errors="coerce")
        d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["grid", "ale_effect"]).sort_values("grid")
        ax.set_facecolor(BG)
        if d.empty:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center", fontsize=5.5, color=MID, transform=ax.transAxes)
        else:
            x = d["grid"].to_numpy(float); y = d["ale_effect"].to_numpy(float)
            y = y - float(np.nanmean(y))
            ax.axhline(0, color=TRACK, lw=0.55, zorder=1)
            ax.plot(x, y, color=ACC_D, lw=0.95, zorder=3, solid_capstyle="round")
            ax.fill_between(x, 0, y, color=ACC_L, alpha=0.72, zorder=2, linewidth=0)
            if len(x):
                ax.scatter([x[0], x[-1]], [y[0], y[-1]], s=5, color=ACC_D, linewidths=0, zorder=4)
        ax.text(0.0, 1.055, _plain_taxon_label(feat, 34), ha="left", va="bottom", fontsize=5.2, color=INK,
                transform=ax.transAxes, path_effects=[mpe.withStroke(linewidth=1.4, foreground="white")])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color("#000000"); ax.spines["bottom"].set_linewidth(0.45)
        ax.tick_params(axis="x", length=2.0, width=0.4, labelsize=4.5, pad=1, colors="#000000")
        ax.tick_params(axis="y", length=0, labelsize=4.5, pad=1, colors=MID)
        ax.yaxis.set_ticks_position("none")
        try:
            ax.locator_params(axis="x", nbins=3); ax.locator_params(axis="y", nbins=3)
        except Exception:
            pass
    save_all(fig, out_stem); plt.close(fig)

def _plot_instance_explanations(inst_top: pd.DataFrame, out_stem: Path, *, top_features_per_direction: int) -> None:
    """Create a compact local SHAP explanation figure.

    The canvas is deliberately narrower than the global feature-support figure:
    instance-level panels only need feature names and signed local contribution
    bars, so a two-column layout avoids the stretched/smashed appearance that
    occurs on a full 180 mm canvas.
    """
    apply_style()
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as mpe
    from .style import MM, BG, INK, MID, TRACK, ACC_L, save_all
    if inst_top is None or inst_top.empty:
        fig = plt.figure(figsize=(112 * MM, 58 * MM)); ax = fig.add_axes([0.06,0.10,0.88,0.80]); ax.axis('off')
        ax.text(0.5,0.5,"Instance explanations unavailable",ha="center",va="center",fontsize=7,color=MID)
        save_all(fig, out_stem); plt.close(fig); return

    samples = list(dict.fromkeys(inst_top["sample_id"].astype(str).tolist()))
    n_panels = min(len(samples), 4)
    fig_h = max(48, 28 * n_panels + 8)
    fig = plt.figure(figsize=(118 * MM, fig_h * MM)); fig.patch.set_facecolor(BG)
    panel_h = 0.90 / max(n_panels, 1)
    for pi, sid in enumerate(samples[:n_panels]):
        sub = inst_top[inst_top["sample_id"].astype(str).eq(sid)].copy()
        sub = sub.sort_values("value", ascending=True)
        y0 = 0.055 + (n_panels - 1 - pi) * panel_h
        # Tight two-column geometry: labels end close to signed bars.
        ax_lab = fig.add_axes([0.055, y0 + 0.085*panel_h, 0.405, panel_h * 0.67])
        ax_bar = fig.add_axes([0.475, y0 + 0.085*panel_h, 0.455, panel_h * 0.67])
        role = str(sub["selection_role"].iloc[0]) if len(sub) else "sample"
        fig.text(0.055, y0 + panel_h * 0.86, f"{role.replace('_',' ')} · {sid}", ha="left", va="center", fontsize=6.0, color=INK, weight="bold")
        vals = pd.to_numeric(sub["value"], errors="coerce").fillna(0).to_numpy(float)
        feats = sub["feature"].astype(str).tolist()
        y = np.arange(len(sub))
        for ax in (ax_lab, ax_bar):
            ax.set_ylim(len(sub)-0.5, -0.5); ax.set_yticks([]); ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_visible(False)
            for yi in np.arange(len(sub)+1)-0.5: ax.axhline(yi, color=TRACK, lw=0.22, zorder=0)
        ax_lab.set_xlim(0,1); ax_lab.set_xticks([])
        for i, feat in enumerate(feats):
            ax_lab.text(0.02, i, _plain_taxon_label(feat, 36), ha="left", va="center", fontsize=4.95, color=INK,
                        path_effects=[mpe.withStroke(linewidth=1.1, foreground="white")])
        vmax = max(float(np.nanmax(np.abs(vals))) if len(vals) else 1.0, 1e-9)
        ax_bar.barh(y, vals, height=0.52, color=ACC_L, edgecolor="none", zorder=2)
        ax_bar.axvline(0, color="#000000", lw=0.45, zorder=3)
        ax_bar.set_xlim(-1.12*vmax, 1.12*vmax)
        ax_bar.set_xticks([-vmax, 0, vmax]); ax_bar.set_xticklabels(["−", "0", "+"], fontsize=4.7, color="#000000")
        ax_bar.tick_params(axis="x", length=2.0, width=0.4, pad=1)
        ax_bar.spines["bottom"].set_visible(True); ax_bar.spines["bottom"].set_color("#000000"); ax_bar.spines["bottom"].set_linewidth(0.45)
    save_all(fig, out_stem); plt.close(fig)



def _safe_cache_name(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _explainability_cache_complete(target_dir: Path, methods: Sequence[str]) -> bool:
    if not (target_dir / "explained_unit.json").exists():
        return False
    if not (target_dir / "feature_importance.tsv").exists():
        return False
    if not (target_dir / "top_features.tsv").exists():
        return False
    figs = target_dir / "figures"
    if not (figs / "feature_support.svg").exists() and not (figs / "feature_support.png").exists():
        return False
    normalised = set(_normalise_explainability_methods(methods))
    for method in ("shap", "lime", "ale", "permutation"):
        if method in normalised and not (target_dir / f"feature_importance_{method}.tsv").exists():
            return False
    if "interactions" in normalised and not any((target_dir / name).exists() for name in ("feature_interactions_current.csv", "feature_interactions_corrected.csv", "feature_interactions_fixed_pairs.csv", "feature_interactions_corrected_fixed.csv")):
        return False
    return True


def _copy_explainability_cache(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _ensure_mpma_member_explanations(sweep: Sweep, rankings: pd.DataFrame, members: Sequence[str]) -> None:
    if not members:
        return
    methods = _normalise_explainability_methods(sweep.explainability.methods)
    total = len(members)
    info(f"Preparing cached member explainability for MPMA-E · members={total}")
    for i, member in enumerate(members, start=1):
        matches = rankings[rankings["config_id"].astype(str).eq(str(member))]
        if matches.empty:
            # Tolerate shortened ids stored in older ensemble files.
            matches = rankings[rankings["config_id"].astype(str).map(lambda x: x.startswith(str(member)) or str(member).startswith(x))]
        cid = str(matches.iloc[0]["config_id"]) if not matches.empty else str(member)
        cache_dir = sweep.root() / "explainability" / "cache" / _safe_cache_name(cid)
        if _explainability_cache_complete(cache_dir, methods):
            info(f"MPMA-E member {i}/{total} · config={cid} · cached")
            continue
        if matches.empty:
            raise ExplainabilityConfigurationError(f"MPMA-E member {member!r} cannot be matched to mpma_rankings.tsv for cached explanation.")
        row = matches.iloc[0]
        desc = " · ".join([
            str(row.get("resolution", row.get("levels", "MPDR"))),
            str(row.get("count_transformation", "transformation")),
            str(row.get("learner", "learner")),
        ])
        info(f"MPMA-E member {i}/{total} · config={cid} · {desc}")
        _explain_one(
            sweep,
            target_override=cid,
            output_slug_override=f"cache/{_safe_cache_name(cid)}",
            display_label_override=f"MPMA-E member {i}/{total}",
            allow_member_cache=False,
        )

def _configured_explainability_targets(explainability: Any) -> str | tuple[str, ...]:
    return getattr(explainability, "targets", "auto")


def _single_configured_explainability_target(explainability: Any) -> str:
    configured = _configured_explainability_targets(explainability)
    if isinstance(configured, str):
        return configured
    values = tuple(str(x) for x in configured)
    if len(values) != 1:
        raise ExplainabilityConfigurationError(
            "Internal error: _explain_one requires one explainability target. "
            "Use explain(sweep) to process multiple targets."
        )
    return values[0]




def _shap_values_for_data(
    clf: BaseEstimator,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    random_state: int,
    max_background: int,
    max_explain: int,
    force_explain_rows: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Return SHAP values for held-out rows and their local row indices."""
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external package
        raise ExplainabilityDependencyError(
            "Explainability.methods includes 'shap', but the 'shap' package is not importable. "
            "Install shap or remove 'shap' from Explainability.methods. No fallback method will be used."
        ) from exc

    X_background = _as_float_matrix(X_background)
    X_explain = _as_float_matrix(X_explain)
    rows_bg = _sample_rows(X_background, max_rows=max_background, random_state=random_state)
    rows_ex = _sample_rows(X_explain, max_rows=max_explain, random_state=random_state + 13)
    if force_explain_rows:
        forced = np.asarray([int(i) for i in force_explain_rows if 0 <= int(i) < X_explain.shape[0]], dtype=int)
        if forced.size:
            rows_ex = np.array(sorted(set(rows_ex.tolist()) | set(forced.tolist())), dtype=int)
    background = X_background[rows_bg]
    X_selected = X_explain[rows_ex]

    classes = np.arange(len(class_labels), dtype=int)
    if hasattr(clf, "predict_proba"):
        model_fn = _explain_predict_proba(clf, classes)
    elif hasattr(clf, "decision_function"):
        model_fn = _explain_decision_function(clf)
    else:
        raise ExplainabilityConfigurationError(
            f"Learner {type(clf).__name__} exposes neither predict_proba nor decision_function; "
            "SHAP explainability cannot run exactly as configured. No fallback method will be used."
        )

    min_max_evals = max(500, 2 * len(feature_names) + 1)
    try:
        with _quiet_external_progress():
            explainer = shap.Explainer(model_fn, background, feature_names=list(feature_names))
            try:
                values = explainer(X_selected, silent=True, max_evals=min_max_evals)
            except TypeError:
                try:
                    values = explainer(X_selected, max_evals=min_max_evals)
                except TypeError:
                    try:
                        values = explainer(X_selected, silent=True)
                    except TypeError:
                        values = explainer(X_selected)
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityConfigurationError(
            "SHAP failed for an outer-test fold. Explainability is strict, so execution stops "
            "instead of replacing SHAP with another method."
        ) from exc

    arr = np.asarray(values.values, dtype=float)
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            arr = arr[:, :, 1]
        else:
            arr = np.mean(np.abs(arr), axis=2)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ExplainabilityConfigurationError(
            f"Unexpected SHAP value shape {arr.shape}; expected n_samples × n_features. "
            "No fallback method will be used."
        )
    return arr, rows_ex


def _importance_frame_from_value_blocks(
    method: str,
    blocks: Sequence[np.ndarray],
    feature_names: Sequence[str],
    scoring: str,
) -> pd.DataFrame:
    if not blocks:
        raise ExplainabilityConfigurationError(f"{method.upper()} produced no outer-test explanations. No fallback method will be used.")
    arr = np.concatenate([np.asarray(b, dtype=float) for b in blocks if np.asarray(b).size], axis=0)
    if arr.ndim != 2 or arr.shape[1] != len(feature_names):
        raise ExplainabilityConfigurationError(
            f"Unexpected {method.upper()} OOF value shape {arr.shape}; expected n_oof_samples × n_features."
        )
    abs_arr = np.abs(arr)
    return pd.DataFrame(
        {
            "method": method,
            "feature": list(feature_names),
            "importance_mean": np.mean(abs_arr, axis=0),
            "importance_sd": np.std(abs_arr, axis=0, ddof=1) if abs_arr.shape[0] > 1 else 0.0,
            "scoring": scoring,
        }
    ).sort_values("importance_mean", ascending=False)


def _lime_values_for_data(
    clf: BaseEstimator,
    X_training: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    *,
    random_state: int,
    n_samples: int,
    max_explain: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on external package
        raise ExplainabilityDependencyError(
            "Explainability.methods includes 'lime', but the 'lime' package is not importable. "
            "Install lime or remove 'lime' from Explainability.methods. No fallback method will be used."
        ) from exc

    X_training = _as_float_matrix(X_training)
    X_explain = _as_float_matrix(X_explain)
    rows = _sample_rows(X_explain, max_rows=max_explain, random_state=random_state + 29)
    X_selected = X_explain[rows]
    predict_fn = _explain_predict_proba(clf, np.arange(len(class_labels), dtype=int))
    class_names = list(class_labels)
    label = 1 if len(class_names) == 2 else 0
    try:
        explainer = LimeTabularExplainer(
            training_data=X_training,
            feature_names=list(feature_names),
            class_names=class_names,
            mode="classification",
            discretize_continuous=False,
            random_state=int(random_state),
        )
        coeffs = np.zeros((X_selected.shape[0], len(feature_names)), dtype=float)
        for i, row in enumerate(X_selected):
            exp = explainer.explain_instance(
                row,
                predict_fn=predict_fn,
                num_features=len(feature_names),
                num_samples=int(n_samples),
                labels=(label,),
            )
            for fi, coef in exp.local_exp[label]:
                if 0 <= int(fi) < len(feature_names):
                    coeffs[i, int(fi)] = float(coef)
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityConfigurationError(
            "LIME failed for an outer-test fold. Explainability is strict, so execution stops "
            "instead of replacing LIME with another method."
        ) from exc
    return coeffs, rows


def _mean_feature_importance_frames(method: str, frames: Sequence[pd.DataFrame], feature_names: Sequence[str], scoring: str) -> pd.DataFrame:
    if not frames:
        raise ExplainabilityConfigurationError(f"{method.upper()} produced no outer-test fold results. No fallback method will be used.")
    feature_names = list(feature_names)
    sparse_ale = str(method).lower() == "ale"
    means = []
    sds = []
    for frame in frames:
        d = frame.set_index("feature").reindex(feature_names)
        vals = pd.to_numeric(d["importance_mean"], errors="coerce").to_numpy(float)
        means.append(vals if sparse_ale else np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0))
        if "importance_sd" in d.columns:
            sd_vals = pd.to_numeric(d["importance_sd"], errors="coerce").to_numpy(float)
            sds.append(sd_vals if sparse_ale else np.nan_to_num(sd_vals, nan=0.0, posinf=0.0, neginf=0.0))
    mean_arr = np.vstack(means)
    valid = np.isfinite(mean_arr)
    n_valid = valid.sum(axis=0)
    if sparse_ale:
        sum_vals = np.where(valid, mean_arr, 0.0).sum(axis=0)
        importance_mean = np.divide(sum_vals, n_valid, out=np.zeros(mean_arr.shape[1], dtype=float), where=n_valid > 0)
    else:
        importance_mean = np.nanmean(mean_arr, axis=0)
    if sds:
        sd_arr = np.vstack(sds)
        sd_valid = np.isfinite(sd_arr)
        sd_sq = np.where(sd_valid, sd_arr ** 2, 0.0).sum(axis=0)
        sd_n = sd_valid.sum(axis=0)
        importance_sd = np.sqrt(np.divide(sd_sq, sd_n, out=np.zeros(sd_arr.shape[1], dtype=float), where=sd_n > 0))
    else:
        if sparse_ale:
            centred = np.where(valid, mean_arr - importance_mean[None, :], np.nan)
            importance_sd = np.nanstd(centred, axis=0, ddof=1) if mean_arr.shape[0] > 1 else np.zeros(mean_arr.shape[1])
            importance_sd = np.nan_to_num(importance_sd, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            importance_sd = np.nanstd(mean_arr, axis=0, ddof=1) if mean_arr.shape[0] > 1 else np.zeros(mean_arr.shape[1])
    out = pd.DataFrame(
        {
            "method": method,
            "feature": feature_names,
            "importance_mean": importance_mean,
            "importance_sd": importance_sd,
            "scoring": scoring,
        }
    )
    if sparse_ale:
        out["n_estimable_folds"] = n_valid.astype(int)
    return out.sort_values("importance_mean", ascending=False)


def _explainability_outer_splits(sweep: Sweep, dataset: Any) -> list[dict[str, Any]]:
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    y = np.asarray(dataset.y, dtype=int)
    strata = _strata_from_metadata(dataset.metadata, y, sweep.data.stratify_col)
    splits = _outer_splits(sweep.evaluation, y, groups, strata, sweep.data.stratify_col)
    if not splits:
        raise ExplainabilityConfigurationError("No outer splits are available for out-of-fold explainability.")
    return splits


def _ordered_member_rows(root: Path, rankings: pd.DataFrame, members: Sequence[str]) -> pd.DataFrame:
    config_path = root / "configs.tsv"
    configs = pd.read_csv(config_path, sep="\t") if config_path.exists() else rankings
    configs = configs.copy()
    configs["config_id"] = configs["config_id"].astype(str)
    rows: list[pd.Series] = []
    for member in members:
        m = str(member)
        match = configs[configs["config_id"].eq(m)]
        if match.empty:
            match = configs[configs["config_id"].map(lambda x: str(x).startswith(m) or m.startswith(str(x)))]
        if match.empty:
            raise ExplainabilityConfigurationError(f"MPMA-E member {member!r} cannot be matched to configs.tsv.")
        rows.append(match.iloc[0])
    return pd.DataFrame(rows).drop_duplicates("config_id")


def _fit_oof_single_for_explainability(
    sweep: Sweep,
    row: pd.Series,
) -> dict[str, Any]:
    levels = tuple(str(row["levels"]).split(","))
    dataset = load_dataset(sweep.data, levels)
    X_base, feature_names = materialize_mpdr(dataset, levels)
    splits = _explainability_outer_splits(sweep, dataset)
    transformation_key = str(row["count_transformation"])
    learner_key = str(row["learner"])
    ct_factory = _configured_count_transformation_factory(sweep, transformation_key)
    learner_factory = _configured_learner_factory(sweep, learner_key)
    folds: list[dict[str, Any]] = []
    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
            continue
        ct = ct_factory()
        X_train, X_test = ct.apply_pair(X_base[train_idx], X_base[test_idx])
        clf = learner_factory()
        clf.fit(X_train, dataset.y[train_idx])
        proba = _predict_proba_aligned(clf, X_test, np.arange(len(dataset.class_labels), dtype=int))
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": X_train,
                "X_test": X_test,
                "y_test": dataset.y[test_idx],
                "estimator": clf,
                "proba": proba,
            }
        )
    if not folds:
        raise ExplainabilityConfigurationError("No outer fold could be fitted for out-of-fold explainability.")
    return {"dataset": dataset, "X_base": X_base, "feature_names": list(feature_names), "folds": folds}


def _mpma_e_reference_and_folds(
    sweep: Sweep,
    rankings: pd.DataFrame,
) -> tuple[Any, np.ndarray, list[str], list[dict[str, Any]], pd.Series]:
    root = sweep.root()
    members, aggregation = _selected_ensemble_members(root)
    member_rows = _ordered_member_rows(root, rankings, members)
    all_levels: list[str] = []
    for _, r in member_rows.iterrows():
        for lv in _row_levels(r):
            if lv not in all_levels:
                all_levels.append(lv)
    if not all_levels:
        all_levels = ["all"]
    dataset = load_dataset(sweep.data, tuple(all_levels))
    splits = _explainability_outer_splits(sweep, dataset)

    feature_names: list[str] = []
    reference_blocks: list[np.ndarray] = []
    member_materialized: list[dict[str, Any]] = []
    for member_i, (_, r) in enumerate(member_rows.iterrows(), start=1):
        levels = _row_levels(r) or ("all",)
        X_base_member, names_member = materialize_mpdr(dataset, levels)
        transformation_key = str(r["count_transformation"])
        learner_key = str(r["learner"])
        label_prefix = (
            f"{str(r.get('resolution', '+'.join(levels)))}"
            f"|{transformation_key}"
            f"|{learner_key}"
            f"|{str(r.get('config_id', member_i))}"
        )
        feature_names.extend([f"{label_prefix}|{name}" for name in names_member])
        ct_ref = _configured_count_transformation_factory(sweep, transformation_key)()
        X_ref_member, _ = ct_ref.apply_pair(X_base_member, X_base_member)
        reference_blocks.append(X_ref_member)
        member_materialized.append(
            {
                "row": r,
                "levels": levels,
                "X_base": X_base_member,
                "n_features": X_base_member.shape[1],
                "transformation_key": transformation_key,
                "learner_key": learner_key,
            }
        )

    X_reference = np.concatenate(reference_blocks, axis=1) if len(reference_blocks) > 1 else reference_blocks[0]
    folds: list[dict[str, Any]] = []
    total_members = len(member_materialized)
    for split in splits:
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        if len(test_idx) == 0 or len(np.unique(dataset.y[train_idx])) < 2:
            continue
        X_train_blocks: list[np.ndarray] = []
        X_test_blocks: list[np.ndarray] = []
        fitted_members: list[dict[str, Any]] = []
        start = 0
        for member_i, spec in enumerate(member_materialized, start=1):
            r = spec["row"]
            info(
                "Fitting MPMA-E OOF member "
                f"{member_i}/{total_members} · split={str(split['split_key'])} · "
                f"config={str(r.get('config_id', ''))} · "
                f"{str(r.get('resolution', r.get('levels', 'MPDR')))} · "
                f"{spec['transformation_key']} · {spec['learner_key']}"
            )
            ct = _configured_count_transformation_factory(sweep, spec["transformation_key"])()
            X_train_member, X_test_member = ct.apply_pair(spec["X_base"][train_idx], spec["X_base"][test_idx])
            clf_member = _configured_learner_factory(sweep, spec["learner_key"])()
            clf_member.fit(X_train_member, dataset.y[train_idx])
            stop = start + X_train_member.shape[1]
            X_train_blocks.append(X_train_member)
            X_test_blocks.append(X_test_member)
            fitted_members.append(
                {
                    "config_id": str(r["config_id"]),
                    "slice": slice(start, stop),
                    "estimator": clf_member,
                    "classes": np.arange(len(dataset.class_labels), dtype=int),
                }
            )
            start = stop
        X_train = np.concatenate(X_train_blocks, axis=1)
        X_test = np.concatenate(X_test_blocks, axis=1)
        ensemble = _FittedMpmaEnsemble(fitted_members, aggregation=aggregation)
        proba = _predict_proba_aligned(ensemble, X_test, np.arange(len(dataset.class_labels), dtype=int))
        folds.append(
            {
                "split_key": str(split["split_key"]),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": X_train,
                "X_test": X_test,
                "y_test": dataset.y[test_idx],
                "estimator": ensemble,
                "proba": proba,
            }
        )
    if not folds:
        raise ExplainabilityConfigurationError("No outer fold could be fitted for MPMA-E out-of-fold explainability.")
    row = pd.Series(
        {
            "unit": "MPMA-E",
            "members": ",".join(member_rows["config_id"].astype(str).tolist()),
            "aggregation_strategy": aggregation,
            "levels": ",".join(all_levels),
            "count_transformation": "ensemble",
            "learner": "MPMA-E",
        }
    )
    return dataset, X_reference, feature_names, folds, row


def _aggregate_oof_predictions(dataset: Any, folds: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        test_idx = np.asarray(fold["test_idx"], dtype=int)
        proba = np.asarray(fold.get("proba"), dtype=float)
        pred = np.argmax(proba, axis=1) if proba.ndim == 2 and proba.shape[0] == len(test_idx) else np.full(len(test_idx), -1)
        for local_i, sample_index in enumerate(test_idx):
            rec = {
                "split_key": str(fold.get("split_key", "")),
                "sample_id": str(dataset.sample_ids[int(sample_index)]),
                "sample_index": int(sample_index),
                "y_true": int(dataset.y[int(sample_index)]),
                "y_pred": int(pred[local_i]),
            }
            if proba.ndim == 2 and local_i < proba.shape[0]:
                for j, label in enumerate(dataset.class_labels):
                    if j < proba.shape[1]:
                        rec[f"proba_{label}"] = float(proba[local_i, j])
                if proba.shape[1] > 1:
                    rec["y_proba_pos"] = float(proba[local_i, 1])
            rows.append(rec)
    return pd.DataFrame(rows)


def _write_oof_prediction_summary(dataset: Any, folds: Sequence[dict[str, Any]], target_dir: Path) -> Path:
    pred_df = _aggregate_oof_predictions(dataset, folds)
    path = target_dir / "oof_predictions.tsv"
    pred_df.to_csv(path, sep="\t", index=False)
    return path


def _select_oof_instance_explanations(
    dataset: Any,
    feature_names: Sequence[str],
    shap_oof_rows: Sequence[dict[str, Any]],
    *,
    sample_ids: Sequence[str],
    representative: bool,
    top_features_per_direction: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not shap_oof_rows:
        return pd.DataFrame(), pd.DataFrame()
    meta = pd.DataFrame(shap_oof_rows)
    if meta.empty or "values" not in meta.columns:
        return pd.DataFrame(), pd.DataFrame()
    values = np.vstack(meta.pop("values").to_list())
    sid_requested = {str(s) for s in sample_ids}
    candidates: list[tuple[int, str]] = []
    if sid_requested:
        for sample_index, sid in enumerate(dataset.sample_ids):
            if str(sid) in sid_requested:
                candidates.append((sample_index, f"requested:{sid}"))
    if representative:
        pred_summary = meta.groupby("sample_index", as_index=False).agg(
            sample_id=("sample_id", "first"),
            true_class=("true_class", "first"),
            p_positive=("p_positive", "mean"),
            n_oof_explanations=("sample_index", "size"),
        )
        for cls in sorted(set(int(x) for x in dataset.y.tolist())):
            sub = pred_summary[pred_summary["true_class"].astype(int).eq(cls)].copy()
            if sub.empty:
                continue
            centre = float(sub["p_positive"].median()) if np.isfinite(sub["p_positive"]).any() else 0.5
            sub["_dist"] = (sub["p_positive"].astype(float) - centre).abs()
            r = sub.sort_values(["_dist", "sample_index"]).iloc[0]
            candidates.append((int(r["sample_index"]), f"representative_class_{cls}"))
    # Preserve order and uniqueness.
    seen = set()
    selected_pairs: list[tuple[int, str]] = []
    for idx, role in candidates:
        if idx not in seen:
            selected_pairs.append((idx, role))
            seen.add(idx)
    if not selected_pairs:
        return pd.DataFrame(), pd.DataFrame()

    top_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    k = max(1, int(top_features_per_direction))
    for sample_index, role in selected_pairs:
        mask = meta["sample_index"].astype(int).eq(int(sample_index)).to_numpy()
        if not np.any(mask):
            continue
        vals = values[mask].mean(axis=0)
        vals_sd = values[mask].std(axis=0, ddof=1) if mask.sum() > 1 else np.zeros(values.shape[1])
        sub_meta = meta.loc[mask].copy()
        p_pos = pd.to_numeric(sub_meta.get("p_positive", np.nan), errors="coerce")
        pred_mode = pd.to_numeric(sub_meta.get("predicted_class", -1), errors="coerce").dropna()
        predicted = int(pred_mode.mode().iloc[0]) if not pred_mode.empty else -1
        sid = str(dataset.sample_ids[int(sample_index)])
        selected_rows.append(
            {
                "sample_id": sid,
                "sample_index": int(sample_index),
                "selection_role": role,
                "true_class": int(dataset.y[int(sample_index)]),
                "predicted_class": predicted,
                "p_positive": float(p_pos.mean()) if not p_pos.empty else float("nan"),
                "p_positive_mean": float(p_pos.mean()) if not p_pos.empty else float("nan"),
                "p_positive_sd": float(p_pos.std(ddof=1)) if len(p_pos) > 1 else 0.0,
                "n_oof_explanations": int(mask.sum()),
            }
        )
        order_neg = np.argsort(vals)[:k]
        order_pos = np.argsort(-vals)[:k]
        chosen = list(dict.fromkeys([int(i) for i in list(order_neg) + list(order_pos)]))
        chosen = sorted(chosen, key=lambda j: abs(float(vals[j])), reverse=True)
        for rank, j in enumerate(chosen, start=1):
            top_rows.append(
                {
                    "sample_id": sid,
                    "sample_index": int(sample_index),
                    "selection_role": role,
                    "true_class": int(dataset.y[int(sample_index)]),
                    "predicted_class": predicted,
                    "p_positive": float(p_pos.mean()) if not p_pos.empty else float("nan"),
                    "p_positive_mean": float(p_pos.mean()) if not p_pos.empty else float("nan"),
                    "feature": str(feature_names[j]),
                    "value": float(vals[j]),
                    "value_sd": float(vals_sd[j]),
                    "abs_value": float(abs(vals[j])),
                    "rank": int(rank),
                    "n_oof_explanations": int(mask.sum()),
                }
            )
    return pd.DataFrame(top_rows), pd.DataFrame(selected_rows)



def _aggregate_oof_interactions(tables_by_fold: Sequence[pd.DataFrame], method: str) -> pd.DataFrame:
    if not tables_by_fold:
        raise ExplainabilityConfigurationError(f"OOF ALE interactions produced no {method!r} fold results. No fallback method will be used.")
    raw = pd.concat(tables_by_fold, ignore_index=True)
    raw["interaction_strength"] = pd.to_numeric(raw["interaction_strength"], errors="coerce")
    out = (
        raw.groupby(["feature_1", "feature_2"], as_index=False)
        .agg(
            interaction_strength=("interaction_strength", "mean"),
            interaction_strength_sd=("interaction_strength", "std"),
            n_outer_folds=("fold_key", "nunique"),
            n_evaluations=("interaction_strength", "count"),
            n_ale_cells=("n_ale_cells", "mean"),
            correction_applied=("correction_applied", "max"),
            correction_error=("correction_error", lambda s: "; ".join([str(x) for x in s if str(x) and str(x) != "nan"])[:240]),
        )
    )
    out["method"] = method
    return out.sort_values("interaction_strength", ascending=False, na_position="last")


def _explain_one(
    sweep: Sweep,
    target_override: str | None = None,
    *,
    output_slug_override: str | None = None,
    display_label_override: str | None = None,
    allow_member_cache: bool = True,
) -> dict[str, Path]:
    methods = _normalise_explainability_methods(sweep.explainability.methods)
    _preflight_explainability_dependencies(methods)
    root = sweep.root()
    stage("Explainability", str(root))
    summary_table(
        "Explainability suite",
        {
            "target": target_override or _configured_explainability_targets(sweep.explainability),
            "methods": methods,
            "top features": sweep.explainability.top_k,
            "interaction pairs": sweep.explainability.top_k_interactions if "interactions" in methods else "not requested",
            "fallbacks": "off",
        },
    )
    rankings_path = root / "tables" / "mpma_rankings.tsv"
    if not rankings_path.exists():
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = pd.read_csv(rankings_path, sep="\t")
    if rankings.empty:
        raise RuntimeError("No ranked MPMA is available for explainability.")
    target = target_override or _single_configured_explainability_target(sweep.explainability)
    ensemble_explain = False
    if target in {"best", "best_individual", "best_mpma", "mpma_b", "MPMA-B"}:
        row = rankings.iloc[0]
        target_label = "MPMA-B"
        target_slug = "mpma_b"
    elif target in {"ensemble", "mpma_e", "MPMA-E"}:
        ensemble_explain = True
        row = pd.Series({"unit": "MPMA-E"})
        target_label = "MPMA-E"
        target_slug = "mpma_e"
    elif target in {"baseline_rf", "Baseline RF", "baseline-rf"}:
        row = _resolve_baseline_rf_row(rankings)
        if row is None:
            raise ExplainabilityConfigurationError(
                "Baseline RF explainability was requested, but no completed MPMA matches "
                "RF_1000_msl5 + arcsin_sqrt + deepest available single rank."
            )
        target_label = "Baseline RF"
        target_slug = "baseline_rf"
    else:
        matches = rankings[rankings["config_id"].astype(str).eq(str(target))]
        if matches.empty:
            raise ValueError(f"No MPMA with config_id={target!r} in rankings.")
        row = matches.iloc[0]
        target_label = "MPMA"
        target_slug = str(target)

    if output_slug_override:
        target_slug = output_slug_override.strip("/")
    if display_label_override:
        target_label = display_label_override

    if ensemble_explain and allow_member_cache:
        members_for_cache, _ = _selected_ensemble_members(root)
        _ensure_mpma_member_explanations(sweep, rankings, members_for_cache)

    config_id_for_cache = None
    if not ensemble_explain and "config_id" in row.index and not pd.isna(row.get("config_id")):
        config_id_for_cache = str(row.get("config_id"))
    target_dir = root / "explainability" / target_slug
    cache_dir = root / "explainability" / "cache" / _safe_cache_name(config_id_for_cache) if config_id_for_cache else None
    if allow_member_cache and target_dir.exists() and _explainability_cache_complete(target_dir, methods):
        info(f"Reusing existing explainability for {target_label}")
        return {
            "explainability_dir": target_dir,
            "importance": target_dir / "feature_importance.tsv",
            "figure": target_dir / "figures" / "feature_importance.png",
            "feature_support": target_dir / "figures" / "feature_support.png",
        }
    if not ensemble_explain and allow_member_cache and cache_dir is not None and cache_dir.exists() and _explainability_cache_complete(cache_dir, methods):
        if target_dir != cache_dir:
            _copy_explainability_cache(cache_dir, target_dir)
        info(f"Reusing cached explainability for {target_label} · config={config_id_for_cache}")
        return {
            "explainability_dir": target_dir,
            "importance": target_dir / "feature_importance.tsv",
            "figure": target_dir / "figures" / "feature_importance.png",
            "feature_support": target_dir / "figures" / "feature_support.png",
        }

    if ensemble_explain:
        with console.status("Fitting selected MPMA-E outer-fold units for OOF explanation...", spinner="dots"):
            dataset, X_base, feature_names, oof_folds, row = _mpma_e_reference_and_folds(sweep, rankings)
    else:
        with console.status("Fitting selected MPMA outer-fold units for OOF explanation...", spinner="dots"):
            oof_bundle = _fit_oof_single_for_explainability(sweep, row)
            dataset = oof_bundle["dataset"]
            X_base = oof_bundle["X_base"]
            feature_names = oof_bundle["feature_names"]
            oof_folds = oof_bundle["folds"]

    figures_dir = target_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    method_frames: list[pd.DataFrame] = []
    method_outputs: dict[str, Path] = {}
    interaction_outputs: dict[str, Path] = {}
    oof_pred_path = _write_oof_prediction_summary(dataset, oof_folds, target_dir)
    method_outputs["oof_predictions"] = oof_pred_path

    # All global explanations below are computed from outer-test rows predicted by
    # models fitted only on the corresponding outer-training fold. This preserves
    # the out-of-fold explanation contract for nested cross-validation.
    shap_oof_rows: list[dict[str, Any]] = []
    if "shap" in methods:
        info("Running OOF SHAP feature attribution")
        shap_blocks: list[np.ndarray] = []
        for fold_no, fold in enumerate(oof_folds, start=1):
            test_idx = np.asarray(fold["test_idx"], dtype=int)
            requested = {str(x) for x in sweep.explainability.instance_sample_ids}
            force_rows = [i for i, gi in enumerate(test_idx) if str(dataset.sample_ids[int(gi)]) in requested]
            with console.status(f"Computing SHAP values for outer fold {fold_no}/{len(oof_folds)}...", spinner="dots"):
                vals, rows_ex = _shap_values_for_data(
                    fold["estimator"],
                    fold["X_train"],
                    fold["X_test"],
                    feature_names,
                    dataset.class_labels,
                    random_state=sweep.explainability.random_state + fold_no * 997,
                    max_background=sweep.explainability.shap_background,
                    max_explain=sweep.explainability.shap_max_samples,
                    force_explain_rows=force_rows,
                )
            shap_blocks.append(vals)
            proba = np.asarray(fold.get("proba"), dtype=float)
            pred = np.argmax(proba, axis=1) if proba.ndim == 2 and proba.shape[0] == len(test_idx) else np.full(len(test_idx), -1)
            for local_value_row, local_test_row in enumerate(rows_ex):
                global_i = int(test_idx[int(local_test_row)])
                p_pos = float(proba[int(local_test_row), 1]) if proba.ndim == 2 and proba.shape[1] > 1 else float("nan")
                shap_oof_rows.append(
                    {
                        "split_key": str(fold.get("split_key", "")),
                        "sample_id": str(dataset.sample_ids[global_i]),
                        "sample_index": global_i,
                        "true_class": int(dataset.y[global_i]),
                        "predicted_class": int(pred[int(local_test_row)]) if int(local_test_row) < len(pred) else -1,
                        "p_positive": p_pos,
                        "values": vals[local_value_row],
                    }
                )
        shap_frame = _importance_frame_from_value_blocks("shap", shap_blocks, feature_names, "mean_abs_oof_shap")
        method_frames.append(shap_frame)
        with console.status("Writing OOF SHAP tables and figures...", spinner="dots"):
            method_outputs.update(_write_method_outputs(
                shap_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            ))
        if sweep.explainability.representative_instances or sweep.explainability.instance_sample_ids:
            info("Running OOF SHAP instance-level explanations")
            inst_top, inst_sel = _select_oof_instance_explanations(
                dataset,
                feature_names,
                shap_oof_rows,
                sample_ids=sweep.explainability.instance_sample_ids,
                representative=sweep.explainability.representative_instances,
                top_features_per_direction=sweep.explainability.top_instance_features,
            )
            inst_top.to_csv(target_dir / "instance_explanations_shap_top_features.tsv", sep="\t", index=False)
            inst_sel.to_csv(target_dir / "instance_explanations_shap_selected.tsv", sep="\t", index=False)
            _plot_instance_explanations(inst_top, figures_dir / "instance_explanations_shap", top_features_per_direction=sweep.explainability.top_instance_features)
            method_outputs["instance_explanations_shap"] = target_dir / "instance_explanations_shap_top_features.tsv"
            method_outputs["instance_explanations_shap_figure"] = figures_dir / "instance_explanations_shap.png"
        success("OOF SHAP completed")

    if "lime" in methods:
        info("Running OOF LIME feature attribution")
        lime_blocks: list[np.ndarray] = []
        for fold_no, fold in enumerate(oof_folds, start=1):
            with console.status(f"Computing LIME explanations for outer fold {fold_no}/{len(oof_folds)}...", spinner="dots"):
                coeffs, _rows = _lime_values_for_data(
                    fold["estimator"],
                    fold["X_train"],
                    fold["X_test"],
                    feature_names,
                    dataset.class_labels,
                    random_state=sweep.explainability.random_state + fold_no * 997,
                    n_samples=sweep.explainability.lime_samples,
                    max_explain=sweep.explainability.lime_max_samples,
                )
            lime_blocks.append(coeffs)
        label = 1 if len(dataset.class_labels) == 2 else 0
        lime_frame = _importance_frame_from_value_blocks("lime", lime_blocks, feature_names, f"mean_abs_oof_lime_coefficients_label_{label}")
        method_frames.append(lime_frame)
        with console.status("Writing OOF LIME tables and figures...", spinner="dots"):
            method_outputs.update(_write_method_outputs(
                lime_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            ))
        success("OOF LIME completed")

    if "permutation" in methods:
        info("Running OOF permutation importance")
        perm_frames: list[pd.DataFrame] = []
        for fold_no, fold in enumerate(oof_folds, start=1):
            with console.status(f"Computing permutation importance for outer fold {fold_no}/{len(oof_folds)}...", spinner="dots"):
                perm_frames.append(_permutation_feature_importance(
                    fold["estimator"],
                    fold["X_test"],
                    fold["y_test"],
                    feature_names,
                    dataset.class_labels,
                    n_repeats=sweep.explainability.n_repeats,
                    random_state=sweep.explainability.random_state + fold_no * 997,
                ))
        perm_scoring = "mean_outer_test_permutation_importance"
        permutation_frame = _mean_feature_importance_frames("permutation", perm_frames, feature_names, perm_scoring)
        method_frames.append(permutation_frame)
        with console.status("Writing OOF permutation tables and figures...", spinner="dots"):
            method_outputs.update(_write_method_outputs(
                permutation_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            ))
        success("OOF permutation importance completed")

    preliminary = _combine_feature_importance(method_frames) if method_frames else pd.DataFrame()
    top_for_ale = preliminary.head(sweep.explainability.top_k)["feature"].tolist() if not preliminary.empty else list(feature_names[: sweep.explainability.top_k])

    if "ale" in methods:
        info("Running OOF ALE feature effects")
        ale_frames: list[pd.DataFrame] = []
        curve_frames: list[pd.DataFrame] = []
        skipped_frames: list[pd.DataFrame] = []
        for fold_no, fold in enumerate(oof_folds, start=1):
            with console.status(f"Computing ALE curves for outer fold {fold_no}/{len(oof_folds)}...", spinner="dots"):
                frame = _ale_feature_importance(
                    fold["estimator"],
                    fold["X_test"],
                    feature_names,
                    dataset.class_labels,
                    n_bins=sweep.explainability.ale_bins,
                    top_features=top_for_ale,
                )
            skipped = frame.attrs.get("skipped_features")
            if isinstance(skipped, pd.DataFrame) and not skipped.empty:
                skipped = skipped.copy()
                skipped["fold_key"] = str(fold.get("split_key", ""))
                skipped["fold_no"] = int(fold_no)
                skipped_frames.append(skipped)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["fold_key"] = str(fold.get("split_key", ""))
            frame["fold_no"] = int(fold_no)
            ale_frames.append(frame)
            curves = frame.attrs.get("curves")
            if isinstance(curves, pd.DataFrame) and not curves.empty:
                curves = curves.copy()
                curves["fold_key"] = str(fold.get("split_key", ""))
                curves["fold_no"] = int(fold_no)
                curve_frames.append(curves)
        if skipped_frames:
            pd.concat(skipped_frames, ignore_index=True).to_csv(target_dir / "ale_skipped_features.tsv", sep="	", index=False)
        if not ale_frames:
            raise ExplainabilityConfigurationError(
                "ALE was requested, but no candidate feature was estimable in any outer-test fold. No fallback method will be used."
            )
        ale_frame = _mean_feature_importance_frames("ale", ale_frames, feature_names, "mean_estimable_outer_fold_rms_centered_ale")
        if curve_frames:
            curves_all = pd.concat(curve_frames, ignore_index=True)
            curves_all.to_csv(target_dir / "ale_curves.tsv", sep="	", index=False)
            _plot_ale_curves(curves_all, top_for_ale, figures_dir / "ale_curves", max_panels=min(12, sweep.explainability.top_k))
        method_frames.append(ale_frame)
        with console.status("Writing OOF ALE tables and figures...", spinner="dots"):
            method_outputs.update(_write_method_outputs(
                ale_frame,
                target_dir=target_dir,
                figures_dir=figures_dir,
                X_base=X_base,
                y=dataset.y,
                feature_names=feature_names,
                class_labels=dataset.class_labels,
                top_k=sweep.explainability.top_k,
            ))
        success("OOF ALE completed")

    if not method_frames:
        raise ExplainabilityConfigurationError("No feature-importance method was executed. No fallback method will be used.")

    imp = _combine_feature_importance(method_frames)
    imp.to_csv(target_dir / "feature_importance.tsv", sep="\t", index=False)

    if "interactions" in methods:
        info("Running OOF ALE interaction analysis")
        score_series = imp.set_index("feature")["importance_mean"]
        by_method: dict[str, list[pd.DataFrame]] = {m: [] for m in ("current", "corrected", "fixed_pairs", "corrected_fixed")}
        all_raw: list[pd.DataFrame] = []
        for fold_no, fold in enumerate(oof_folds, start=1):
            with console.status(f"Computing pairwise interactions for outer fold {fold_no}/{len(oof_folds)}...", spinner="dots"):
                tables = _ale_interactions(
                    fold["estimator"],
                    fold["X_test"],
                    feature_names,
                    dataset.class_labels,
                    n_bins=sweep.explainability.ale_bins,
                    top_k_interactions=sweep.explainability.top_k_interactions,
                    scores=score_series,
                )
            for name, tab in tables.items():
                tab = tab.copy()
                tab["fold_key"] = str(fold.get("split_key", ""))
                if name == "all_methods":
                    all_raw.append(tab)
                elif name in by_method:
                    by_method[name].append(tab)
        interaction_tables: dict[str, pd.DataFrame] = {}
        for name, frames in by_method.items():
            interaction_tables[name] = _aggregate_oof_interactions(frames, name)
        if all_raw:
            all_path = target_dir / "feature_interactions_by_target_all_methods.csv"
            pd.concat(all_raw, ignore_index=True).to_csv(all_path, index=False)
            interaction_outputs["interactions_all_methods"] = all_path
        for name, tab in interaction_tables.items():
            path = target_dir / f"feature_interactions_{name}.csv"
            tab.to_csv(path, index=False)
            interaction_outputs[f"interactions_{name}"] = path
        dist_for_network = _feature_distribution_stats(
            imp.head(sweep.explainability.top_k)["feature"].astype(str).tolist(),
            feature_names,
            X_base,
            dataset.y,
            dataset.class_labels,
        )
        with console.status("Writing OOF ALE interaction network figures...", spinner="dots"):
            for net_name in ("current", "corrected", "fixed_pairs", "corrected_fixed"):
                if net_name in interaction_tables:
                    ok = _plot_interaction_network(
                        interaction_tables[net_name],
                        dist_for_network,
                        figures_dir / f"interaction_network_{net_name}",
                        top_k=sweep.explainability.top_k_interactions,
                        class_labels=dataset.class_labels,
                        layout="default",
                    )
                    if ok:
                        interaction_outputs[f"interaction_network_{net_name}"] = figures_dir / f"interaction_network_{net_name}.png"
        success("OOF interaction analysis completed")

    with console.status("Writing feature-distribution summaries and figures...", spinner="dots"):
        dist = _feature_distribution_stats(
            imp.head(sweep.explainability.top_k)["feature"].tolist(),
            feature_names,
            X_base,
            dataset.y,
            dataset.class_labels,
        )
    dist.to_csv(target_dir / "feature_distribution_stats.tsv", sep="\t", index=False)
    top_features = _method_support_table(method_frames, imp, top_k=sweep.explainability.top_k)
    top_features.to_csv(target_dir / "top_features.tsv", sep="\t", index=False)
    _plot_feature_importance(top_features, dist, figures_dir / "feature_importance", sweep.explainability.top_k, dataset.class_labels)
    _plot_feature_importance(top_features, dist, figures_dir / "feature_support", sweep.explainability.top_k, dataset.class_labels)

    dump_json_standard(
        {
            "target": target_slug,
            "target_label": target_label,
            "config": row.to_dict(),
            "methods": list(methods),
            "default_suite": ["shap", "lime", "ale", "permutation", "interactions"],
            "strict": True,
            "fallbacks": False,
            "cross_validation_explanations": "outer_test_folds",
            "out_of_fold": True,
            "n_outer_folds_explained": len(oof_folds),
        },
        target_dir / "explained_unit.json",
    )
    if not ensemble_explain and cache_dir is not None and target_dir != cache_dir:
        _copy_explainability_cache(target_dir, cache_dir)
    success(f"Explainability completed · target={target_label} · methods={','.join(methods)}")
    outputs = {
        "explainability_dir": target_dir,
        "importance": target_dir / "feature_importance.tsv",
        "figure": figures_dir / "feature_importance.png",
        "feature_support": figures_dir / "feature_support.png",
    }
    outputs.update(method_outputs)
    outputs.update(interaction_outputs)
    path_table("Explainability outputs", outputs)
    return outputs



def _score_sort_column(df: pd.DataFrame) -> str | None:
    for col in ("inner_validation_score", "nMCC_inner_mean", "nMCC_mean", "score", "ROC_AUC_mean", "MCC_mean"):
        if col in df.columns:
            return col
    numeric = [c for c in df.columns if c.endswith("_mean") and pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def _row_levels(row: pd.Series) -> tuple[str, ...]:
    val = row.get("levels", row.get("resolution", ""))
    if pd.isna(val):
        return ()
    return tuple(x.strip() for x in str(val).replace("+", ",").split(",") if x.strip())


def _resolve_baseline_rf_row(rankings: pd.DataFrame) -> pd.Series | None:
    required = rankings.copy()
    if "learner" not in required.columns or "count_transformation" not in required.columns:
        return None
    required = required[
        required["learner"].astype(str).eq("RF_1000_msl5")
        & required["count_transformation"].astype(str).eq("arcsin_sqrt")
    ].copy()
    if required.empty:
        return None
    required["_single_rank"] = required.apply(lambda r: _row_levels(r)[0] if len(_row_levels(r)) == 1 else "", axis=1)
    for rank in ("strain", "species", "genus"):
        sub = required[required["_single_rank"].eq(rank)].copy()
        if sub.empty:
            continue
        sort_col = _score_sort_column(sub)
        if sort_col and sort_col in sub.columns:
            sub = sub.sort_values(sort_col, ascending=False)
        return sub.iloc[0].drop(labels=["_single_rank"], errors="ignore")
    return None


def _parse_members(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [x.strip() for x in text.split(",") if x.strip()]
    else:
        parsed = value
    out: list[str] = []
    if isinstance(parsed, dict):
        parsed = parsed.get("members", [])
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, dict):
                cid = item.get("config_id") or item.get("id")
            else:
                cid = item
            if cid is not None:
                out.append(str(cid))
    return out


def _selected_ensemble_members(root: Path) -> tuple[list[str], str]:
    selected_path = root / "ensembling" / "selected_unit.json"
    if not selected_path.exists():
        raise ExplainabilityConfigurationError("MPMA-E explainability requires ensembling/selected_unit.json.")
    with open(selected_path, "r", encoding="utf-8") as fh:
        selected = json.load(fh)
    unit = selected.get("inner_val_best_mpmas_ensemble", selected.get("MPMA-E", selected))
    members = _parse_members(unit.get("members") if isinstance(unit, dict) else None)
    aggregation = str(unit.get("aggregation_strategy", "mean_proba")) if isinstance(unit, dict) else "mean_proba"
    if not members:
        raise ExplainabilityConfigurationError("MPMA-E selected unit does not contain members.")
    return members, aggregation


def _fit_selected_mpma_e_for_explainability(
    sweep: Sweep,
    rankings: pd.DataFrame,
) -> tuple[Any, np.ndarray, np.ndarray, list[str], BaseEstimator, pd.Series]:
    root = sweep.root()
    members, aggregation = _selected_ensemble_members(root)
    config_path = root / "configs.tsv"
    configs = pd.read_csv(config_path, sep="\t") if config_path.exists() else rankings
    configs = configs.copy()
    configs["config_id"] = configs["config_id"].astype(str)
    member_rows = configs[configs["config_id"].isin([str(m) for m in members])].copy()
    if member_rows.empty:
        # tolerate shortened IDs stored in old ensemble JSON files
        keep = []
        for _, r in configs.iterrows():
            cid = str(r.get("config_id", ""))
            if any(cid.startswith(str(m)) or str(m).startswith(cid) for m in members):
                keep.append(r)
        member_rows = pd.DataFrame(keep)
    if member_rows.empty:
        raise ExplainabilityConfigurationError("MPMA-E members cannot be matched to configs.tsv.")
    member_rows = member_rows.drop_duplicates("config_id")
    all_levels: list[str] = []
    for _, r in member_rows.iterrows():
        for lv in _row_levels(r):
            if lv not in all_levels:
                all_levels.append(lv)
    if not all_levels:
        all_levels = ["all"]
    dataset = load_dataset(sweep.data, tuple(all_levels))
    X_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    fitted_members: list[dict[str, Any]] = []
    start = 0
    total_members = len(member_rows)
    for member_i, (_, r) in enumerate(member_rows.iterrows(), start=1):
        info(
            "Fitting MPMA-E member "
            f"{member_i}/{total_members} · config={str(r.get('config_id', ''))} · "
            f"{str(r.get('resolution', r.get('levels', 'MPDR')))} · "
            f"{str(r.get('count_transformation', 'transformation'))} · "
            f"{str(r.get('learner', 'learner'))}"
        )
        levels = _row_levels(r)
        if not levels:
            levels = ("all",)
        X_base_member, names_member = materialize_mpdr(dataset, levels)
        transformation_key = str(r["count_transformation"])
        ct = _configured_count_transformation_factory(sweep, transformation_key)()
        X_member, _ = ct.apply_pair(X_base_member, X_base_member)
        learner_key = str(r["learner"])
        clf_member = _configured_learner_factory(sweep, learner_key)()
        clf_member.fit(X_member, dataset.y)
        stop = start + X_member.shape[1]
        # In MPMA-E explainability each selected MPMA contributes a separate
        # transformed feature block to the concatenated ensemble input matrix.
        # Two members may share the same MPDR, for example the same resolution
        # and transformation with different learners.  The member identity must
        # therefore be part of the feature label; otherwise downstream support
        # tables see duplicate feature labels and pandas refuses to reindex them.
        label_prefix = (
            f"{str(r.get('resolution', '+'.join(levels)))}"
            f"|{transformation_key}"
            f"|{learner_key}"
            f"|{str(r.get('config_id', member_i))}"
        )
        feature_names.extend([f"{label_prefix}|{name}" for name in names_member])
        X_blocks.append(X_member)
        fitted_members.append({
            "config_id": str(r["config_id"]),
            "slice": slice(start, stop),
            "estimator": clf_member,
            "classes": np.arange(len(dataset.class_labels), dtype=int),
        })
        start = stop
    X = np.concatenate(X_blocks, axis=1)
    ensemble = _FittedMpmaEnsemble(fitted_members, aggregation=aggregation)
    row = pd.Series({
        "unit": "MPMA-E",
        "members": ",".join([m["config_id"] for m in fitted_members]),
        "aggregation_strategy": aggregation,
        "levels": ",".join(all_levels),
        "count_transformation": "ensemble",
        "learner": "MPMA-E",
    })
    return dataset, X, X, feature_names, ensemble, row


def _standard_explainability_targets(sweep: Sweep, rankings: pd.DataFrame) -> list[str]:
    targets: list[str] = []
    if (sweep.root() / "ensembling" / "selected_unit.json").exists():
        targets.append("mpma_e")
    targets.append("mpma_b")
    if _resolve_baseline_rf_row(rankings) is not None:
        targets.append("baseline_rf")
    return targets


def _automatic_explainability_targets(sweep: Sweep, rankings: pd.DataFrame) -> list[str]:
    configured = getattr(sweep.explainability, "targets", "auto")
    requested = [configured] if isinstance(configured, str) else list(configured)
    standard_aliases = {"auto", "all", "comparison", "standard", "standard_suite"}
    targets: list[str] = []
    for item in requested:
        target = str(item).strip()
        if not target:
            continue
        if target in standard_aliases:
            targets.extend(_standard_explainability_targets(sweep, rankings))
        else:
            targets.append(target)
    if not targets:
        targets.extend(_standard_explainability_targets(sweep, rankings))
    # Preserve user order while avoiding duplicate work when aliases expand.
    return list(dict.fromkeys(targets))


def explain(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    rankings_path = root / "tables" / "mpma_rankings.tsv"
    if not rankings_path.exists():
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = pd.read_csv(rankings_path, sep="\t")
    targets = _automatic_explainability_targets(sweep, rankings)
    outputs: dict[str, Path] = {}
    for target in targets:
        out = _explain_one(sweep, target_override=target)
        key = {
            "mpma_e": "mpma_e",
            "ensemble": "mpma_e",
            "MPMA-E": "mpma_e",
            "baseline_rf": "baseline_rf",
            "Baseline RF": "baseline_rf",
            "baseline-rf": "baseline_rf",
            "mpma_b": "mpma_b",
            "best_individual": "mpma_b",
            "best": "mpma_b",
            "MPMA-B": "mpma_b",
        }.get(str(target), str(target))
        for name, path in out.items():
            outputs[f"{key}_{name}"] = path
    return outputs

def _method_display(method: str) -> str:
    m = str(method).strip().lower()
    return {
        "shap": "SHAP",
        "lime": "LIME",
        "ale": "ALE",
        "permutation": "Permutation",
        "consensus": "consensus",
    }.get(m, str(method))


def _support_from_importance(frame: pd.DataFrame) -> pd.Series:
    vals = pd.to_numeric(frame.set_index("feature")["importance_mean"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    vals = vals.fillna(0.0)
    if len(vals) == 0:
        return vals
    if not vals.index.is_unique:
        # Support is looked up by feature label with pandas.reindex/map.  Those
        # operations require unique index labels.  Duplicate labels can occur in
        # older MPMA-E outputs where distinct ensemble-member inputs were given
        # the same display name.  Sum the duplicated importances so one support
        # value represents the total evidence attached to that label.
        vals = vals.groupby(level=0, sort=False).sum()
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin + 1e-12:
        # A flat importance profile carries no ranking information.  Rendering
        # this as 1.00 for every feature is visually and statistically
        # misleading, especially for permutation importance on weak/flat
        # models.  Keep the rows, but show zero normalised support.
        return pd.Series(np.zeros(len(vals), dtype=float), index=vals.index)
    return (vals - vmin) / (vmax - vmin)


def _method_support_table(frames: Sequence[pd.DataFrame], imp: pd.DataFrame, top_k: int) -> pd.DataFrame:
    top = imp.head(int(top_k)).copy()
    if "mean_rank" not in top.columns:
        top["mean_rank"] = np.arange(1, len(top) + 1, dtype=float)
    top["rank"] = np.arange(1, len(top) + 1, dtype=int)
    top["consensus"] = _support_from_importance(top.rename(columns={"importance_mean": "importance_mean"})).reindex(top["feature"]).fillna(0).to_numpy()
    for frame in frames:
        method = _method_display(frame["method"].iloc[0])
        support = _support_from_importance(frame)
        top[method] = top["feature"].map(support).fillna(0.0).astype(float)
    for col in ("SHAP", "LIME", "ALE", "Permutation"):
        if col not in top.columns:
            top[col] = 0.0
    top["consensus"] = top[["SHAP", "LIME", "ALE", "Permutation"]].mean(axis=1)
    return top[["rank", "feature", "SHAP", "LIME", "ALE", "Permutation", "consensus", "importance_mean", "mean_rank", "n_methods"] if "n_methods" in top.columns else ["rank", "feature", "SHAP", "LIME", "ALE", "Permutation", "consensus", "importance_mean", "mean_rank"]]


def _plain_taxon_label(feature_name: str, max_len: int = 34) -> str:
    s = str(feature_name)
    last = s.split("___")[-1].split("|")[-1]
    rank = ""
    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if last.startswith(pfx):
            rank = pfx[0] + ". "
            last = last[len(pfx):]
            break
    last = last.replace("_", " ").strip() or s.replace("_", " ")
    out = f"{rank}{last}"
    return out if len(out) <= max_len else out[: max_len - 1].rstrip() + "…"


def _class_colors(class_labels: Sequence[str]) -> tuple[str, str]:
    labels = " ".join(str(x).lower() for x in class_labels)
    if "mindset" in labels or "depression" in labels or "cde" in labels:
        return "#B8B8B8", "#2FA7D6"
    return UC_CTRL, UC_CASE


def _plot_feature_importance(top_features: pd.DataFrame, stats: pd.DataFrame, out_stem: Path, top_k: int, class_labels: Sequence[str]) -> None:
    """Create a feature-support figure.

    Method-specific and combined figures share the same spacing, brackets,
    support heatmap, and class-shift encodings.
    """
    _plot_feature_support_visual(top_features, stats, out_stem, top_k, class_labels)

def _hash_float(text: str) -> float:
    import hashlib
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _layout_graph(G):
    import networkx as nx
    nodes = list(G.nodes())
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: np.array([0.0, 0.0])}
    try:
        if nx.is_connected(G.to_undirected()):
            pos = nx.kamada_kawai_layout(G, weight="distance", scale=1.0)
        else:
            pos = nx.spring_layout(G, weight="weight", seed=7, iterations=600)
    except Exception:
        pos = nx.spring_layout(G, weight="weight", seed=7, iterations=600)
    arr = np.asarray([pos[n] for n in nodes], dtype=float)
    arr = arr - arr.mean(axis=0)
    maxabs = np.max(np.abs(arr)) if arr.size else 1.0
    if maxabs <= 0: maxabs = 1.0
    return {n: arr[i] / maxabs for i, n in enumerate(nodes)}


def _net_label(feature_name: str) -> str:
    label = _plain_taxon_label(feature_name, max_len=28)
    parts = label.split(". ", 1)
    if len(parts) == 2:
        return rf"$\mathit{{{parts[0]}.}}$ {parts[1]}"
    return label


def _plot_interaction_network(tab: pd.DataFrame, stats: pd.DataFrame, out_stem: Path, top_k: int, class_labels: Sequence[str], layout: str = "default") -> bool:
    """Write the 2D-ALE interaction network figure.

    An empty or non-finite interaction table means that no interaction network
    can be drawn from the configured 2D-ALE analysis.  The computed interaction
    artefact is still written, and a clearly labelled unavailable panel is saved
    for figure completeness.  No alternative interaction estimator or fabricated
    edge weights are used.
    """
    try:
        return bool(_plot_interaction_network_visual(tab, stats, out_stem, top_k, class_labels, layout=layout))
    except Exception as exc:  # noqa: BLE001
        # This is deliberately non-fatal: the method has completed and produced
        # its artefact table, but the table may contain no finite edges for a
        # network.  Keep strictness about methods while avoiding loss of all
        # completed explainability outputs.
        _write_unavailable_interaction_network(out_stem, str(exc))
        return False


def _write_unavailable_interaction_network(out_stem: Path, message: str) -> None:
    apply_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(COL_W_2, 82 * (1.0 / 25.4)))
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        "Interaction network unavailable\nno finite 2D-ALE interaction strengths",
        ha="center", va="center", fontsize=7, color=DIM, linespacing=1.15,
    )
    ax.text(
        0.5, 0.38,
        str(message)[:180],
        ha="center", va="center", fontsize=5.2, color=MID, linespacing=1.12,
    )
    save_all(fig, out_stem)
    plt.close(fig)

def _short_taxon(name: str, max_len: int = 70) -> str:
    s = str(name).split("|")[-1].split("___")[-1]
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _feature_distribution_stats(features: list[str], feature_names: list[str], X: np.ndarray, y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    idx = {f: i for i, f in enumerate(feature_names)}
    rows = []
    eps = 1e-10
    for feat in features:
        if feat not in idx:
            continue
        vals = X[:, idx[feat]].astype(float)
        row = {
            "feature": feat,
            "overall_mean": float(np.mean(vals)),
            "mean_relative_abundance": float(np.mean(vals)),
            "overall_median": float(np.median(vals)),
            "prevalence": float(np.mean(vals > 0)),
        }
        for c, label in enumerate(labels):
            sub = vals[y == c]
            row[f"mean_{label}"] = float(np.mean(sub)) if len(sub) else np.nan
            row[f"prevalence_{label}"] = float(np.mean(sub > 0)) if len(sub) else np.nan
        if len(labels) >= 2:
            ctrl = vals[y == 0]
            case = vals[y == 1]
            ctrl_mean = float(np.mean(ctrl)) if len(ctrl) else np.nan
            case_mean = float(np.mean(case)) if len(case) else np.nan
            row["control_mean"] = ctrl_mean
            row["case_mean"] = case_mean
            row["control_mean_pct"] = ctrl_mean * 100.0 if np.isfinite(ctrl_mean) else np.nan
            row["case_mean_pct"] = case_mean * 100.0 if np.isfinite(case_mean) else np.nan
            row["log2_case_vs_control_mean"] = float(np.log2((case_mean + eps) / (ctrl_mean + eps))) if np.isfinite(case_mean) and np.isfinite(ctrl_mean) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
