from __future__ import annotations

import base64
import html
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_rf import resolve_baseline_rf_configs
from .configs_sweep import Sweep
from .console import path_table, phase_progress, stage, success
from .metrics import canonical_metric_name, compute_metrics, metric_is_loss
from .report_compute import run_compute_accounting
from .report_statistics import run_report_statistics
from .storage import read_table, table_exists, write_table
from .utils import (
    TAXONOMIC_LEVELS,
    dump_json_standard,
    prepare_report_html_path,
    report_html_path,
)

_METRICS = [
    ("AUROC", "AUROC"),
    ("AUCPR", "AUCPR"),
    ("AP", "Average precision"),
    ("MCC", "MCC"),
    ("F1w", "F1$_{w}$"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
]


_SINGLE_RANK_PRIORITY = (
    "strain",
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "domain",
)


def _canonical_metric_name(metric: Any) -> str:
    return canonical_metric_name(str(metric))


def _metric_display_label(metric: Any) -> str:
    key = _canonical_metric_name(metric)
    labels = {
        "AUROC": "AUROC",
        "AUROC_macro": "AUROC macro (OvR)",
        "AUROC_weighted": "AUROC weighted (OvR)",
        "AUCPR": "AUCPR",
        "AUCPR_macro": "AUCPR macro (OvR)",
        "AUCPR_weighted": "AUCPR weighted (OvR)",
        "AP": "Average precision",
        "AP_macro": "Average precision macro",
        "MCC": "MCC",
        "F1w": "F1w",
        "Precision": "Precision",
        "Recall": "Recall",
        "log_loss": "Log loss",
        "subject_macro_log_loss": "Subject-macro log loss",
        "cohort_macro_log_loss": "Cohort-macro log loss",
        "brier": "Brier score",
        "BalAcc": "Balanced accuracy",
        "Accuracy": "Accuracy",
        "CalibrationIntercept": "Calibration intercept",
        "CalibrationSlope": "Calibration slope",
        "CalibrationIntercept_macro_OvR": "Calibration intercept macro (OvR)",
        "CalibrationSlope_macro_OvR": "Calibration slope macro (OvR)",
        "R2": "R²",
        "MAE": "Mean absolute error",
        "MSE": "Mean squared error",
        "RMSE": "Root mean squared error",
        "MedAE": "Median absolute error",
        "ExplainedVariance": "Explained variance",
        "PearsonR": "Pearson r",
        "SpearmanR": "Spearman ρ",
    }
    return labels.get(key, str(metric))


def _humanize_identifier(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(text.replace("_", " ").split()).capitalize()


def _humanize_level(value: Any) -> str:
    text = str(value).strip()
    if text.islower() and text.replace(" ", "").isalpha():
        return text.capitalize()
    return text


def _classification_display_metrics(selection_metric: Any) -> list[tuple[str, str]]:

    primary = _canonical_metric_name(selection_metric)

    ordered = [primary] + [metric for metric, _ in _METRICS if metric != primary]

    seen: set[str] = set()

    out: list[tuple[str, str]] = []

    for metric in ordered:
        if metric in seen:
            continue

        seen.add(metric)

        out.append((metric, _metric_display_label(metric)))

    return out


def _display_resolution(value: Any) -> str:

    text = str(value).strip()

    if not text or text.casefold() == "nan":
        return ""

    normalized = text.casefold().replace("→", "->")

    ranks = set(TAXONOMIC_LEVELS)

    for sep in ("->", "-"):
        if sep not in normalized:
            continue

        parts = [part.strip() for part in normalized.split(sep) if part.strip()]

        if len(parts) == 2 and all(part in ranks for part in parts):
            return f"{parts[0]}→{parts[1]}"

    return text


def _display_token(value: Any) -> str:

    text = str(value).strip()

    if not text or text.casefold() == "nan":
        return ""

    resolution = _display_resolution(text)

    if resolution != text:
        return resolution

    base, sep, scope = text.rpartition("@")

    if not sep or scope not in {"rank-wise", "joint"}:
        base, scope = text, ""

    replacements = {
        "identity": "identity",
        "relative_abundance": "RA",
        "presence_absence": "P/A",
        "hellinger": "√RA",
        "arcsine_sqrt": "arcsin√RA",
        r"$\arcsin\sqrt{x}$": "arcsin√RA",
        "log10_relative_abundance_half_min_pseudocount": "log10(RA + ε)",
        "centered_log_ratio_multiplicative_replacement": "CLR",
        "additive_log_ratio_first_reference_multiplicative_replacement": "ALR",
        "isometric_log_ratio_egozcue_multiplicative_replacement": "ILR",
        "standardize": "z-score",
        "robust_scale": "robust z",
        "power_yeo_johnson": "YJ",
        "quantile_normal_numeric": "QN",
        "standardized_centered_log_ratio_multiplicative_replacement": "CLR+z",
        "yeo_johnson_relative_abundance": "YJ(RA)",
        "quantile_normal_relative_abundance": "QN(RA)",
        "robust_scaled_relative_abundance": "robust z(RA)",
        "within_sample_fractional_rank": "within-sample rank",
        "training_ecdf_rank": "ECDF rank",
        "prevalence_weighted_relative_abundance": "prevalence-weighted RA",
        "log1p": "ln(1+x)",
        "log10p": "log10(1+x)",
        "log2p": "log2(1+x)",
        r"$\ln(1+x)$": "ln(1+x)",
        r"$\log_{10}(1+x)$": "log10(1+x)",
        r"$\log_2(1+x)$": "log2(1+x)",
    }

    label = replacements.get(base, base.replace("_", " "))

    return f"{label} ({scope})" if scope else label


_COMPUTE_ONLY_PARAMS = {
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


_HYPERPARAM_PRIORITY = (
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


_LEARNER_CLASS_LABELS = {
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


def _param_equal(left: Any, right: Any) -> bool:

    try:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))

        value = left == right

        if isinstance(value, np.ndarray):
            return bool(np.all(value))

        return bool(value)

    except Exception:
        return False


def _format_param_value(value: Any) -> str:

    if isinstance(value, str):
        return repr(value)

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        return f"{value:g}"

    if isinstance(value, tuple):
        return (
            "("
            + ", ".join(_format_param_value(x) for x in value)
            + ("," if len(value) == 1 else "")
            + ")"
        )

    if isinstance(value, list):
        return "[" + ", ".join(_format_param_value(x) for x in value) + "]"

    return str(value)


def _changed_estimator_params(estimator: Any) -> list[tuple[str, Any]]:

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

    changed = {
        name: value
        for name, value in current.items()
        if name in defaults
        and name not in _COMPUTE_ONLY_PARAMS
        and not callable(value)
        and not callable(getattr(value, "get_params", None))
        and not _param_equal(value, defaults[name])
    }

    ordered = [name for name in _HYPERPARAM_PRIORITY if name in changed]

    ordered.extend(sorted(name for name in changed if name not in ordered))

    return [(name, changed[name]) for name in ordered[:2]]


def _human_learner_name(value: Any) -> str:

    text = str(value).strip()

    if not text or text.casefold() == "nan":
        return ""

    aliases = {
        "rf": "Random forest",
        "random_forest": "Random forest",
        "random forest": "Random forest",
        "mlp": "Multilayer perceptron",
        "multilayer_perceptron": "Multilayer perceptron",
        "multilayer perceptron": "Multilayer perceptron",
    }

    return aliases.get(text.casefold(), text.replace("_", " "))


def _learner_display_map(sweep: Sweep) -> dict[str, str]:

    configured = getattr(sweep, "learners", ())

    if isinstance(configured, dict):
        groups = list(configured.values())

        items = []

        for group in groups:
            try:
                items.extend(list(group))

            except TypeError:
                continue

    else:
        try:
            items = list(configured)

        except TypeError:
            items = []

    labels: dict[str, str] = {}

    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            continue

        name, specification = item

        estimator = specification

        if not callable(getattr(estimator, "get_params", None)) and callable(
            specification
        ):
            try:
                estimator = specification()

            except Exception:
                estimator = None

        if estimator is None:
            labels[str(name)] = _human_learner_name(name)

            continue

        class_name = type(estimator).__name__

        label = _LEARNER_CLASS_LABELS.get(class_name, _human_learner_name(name))

        params = _changed_estimator_params(estimator)

        if params:
            body = ", ".join(
                f"{key}={_format_param_value(value)}" for key, value in params
            )

            label = f"{label} ({body})"

        labels[str(name)] = label

    return labels


def _display_learner(value: Any, learner_labels: dict[str, str] | None = None) -> str:

    text = str(value).strip()

    if learner_labels and text in learner_labels:
        return learner_labels[text]

    return _human_learner_name(text)


def _abbreviations_html(heading_level: int = 2) -> str:

    level = min(6, max(1, int(heading_level)))

    return (
        f'<h{level} id="abbreviations">Abbreviations</h{level}>'
        "<p>AP = average precision. AUCPR = trapezoidal area under the precision-recall curve. RA = relative abundance. "
        "P/A = presence/absence. CLR = centered log-ratio. ALR = additive log-ratio. "
        "ILR = isometric log-ratio. YJ = Yeo–Johnson. QN = quantile normalization. "
        "ECDF = empirical cumulative distribution function. rank-wise = transformation applied independently within each taxonomic-rank block. "
        "joint = transformation applied once to the concatenated multi-rank feature matrix. "
        "ε = training-derived half-minimum positive RA pseudocount.</p>"
    )


def _display_data_format(value: Any) -> str:

    text = str(value).strip()

    key = text.casefold().replace("-", "_")

    if key in {"mllab", "metaphlan", "metaphlan_tsv", "profile_tsv"}:
        return "mllab"

    return text


def _sanitize_report_html(value: str) -> str:

    parts = re.split(r"(<[^>]+>)", str(value))

    raw_tag = ""

    out: list[str] = []

    entity_pattern = re.compile(r"(&#(?:x[0-9a-fA-F]+|[0-9]+);|&[A-Za-z][A-Za-z0-9]+;)")

    for part in parts:
        if not part:
            continue

        if part.startswith("<"):
            tag = part.strip().lower()

            if tag.startswith("<style"):
                raw_tag = "style"

            elif tag.startswith("<script"):
                raw_tag = "script"

            elif tag.startswith("</style") or tag.startswith("</script"):
                raw_tag = ""

            out.append(part)

            continue

        if raw_tag:
            out.append(part)

            continue

        cleaned = (
            part.replace("&mdash;", "-")
            .replace("&#8212;", "-")
            .replace("&#x2014;", "-")
        )

        cleaned = cleaned.replace("—", "-")

        cleaned = re.sub(
            r"\brat(?:her)(?:\s|&nbsp;|&#160;|&#x0*a0;)+than\b",
            "instead of",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(r"\brat(?:her)\b", "instead", cleaned, flags=re.IGNORECASE)

        fragments = entity_pattern.split(cleaned)

        for index in range(0, len(fragments), 2):
            fragments[index] = fragments[index].replace(";", ".")

        out.append("".join(fragments))

    return "".join(out)


def _report_footer_html() -> str:

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    return f'<p class="report-footer">mllabiome · {html.escape(stamp)}</p>'


def _report_nav_html(
    has_robustness: bool = False, has_exploration: bool = False
) -> str:

    exploration_link = (
        '<a href="#data-exploration">Data exploration</a>' if has_exploration else ""
    )
    robustness_link = '<a href="#robustness">Robustness</a>' if has_robustness else ""
    return (
        '<header class="report-nav"><div class="report-nav-inner">'
        '<a class="brand" href="#top" aria-label="mllabiome report"><span class="brand-mark">mll</span><span>mllabiome</span></a>'
        '<nav class="report-links" aria-label="Report navigation">'
        '<a href="#performance-evaluation">Task and evaluation</a>'
        + exploration_link
        + '<a href="#performance-summary">Machine learning performance evaluation</a>'
        + '<a href="#explainability">Explainability</a>'
        + robustness_link
        + '<a href="#compute">Computational resources</a>'
        '<a href="#abbreviations">Abbreviations</a>'
        "</nav></div></header>"
        '<script>document.addEventListener("click",function(e){const a=e.target.closest(".report-links a[href^=\'#\'],.brand[href^=\'#\']");if(!a)return;const id=decodeURIComponent(a.getAttribute("href").slice(1));const t=document.getElementById(id);if(!t)return;e.preventDefault();t.scrollIntoView({behavior:"smooth",block:"start"})})</script>'
    )


def _performance_methodology_html(protocol: Any, n_bootstrap: int = 2000) -> str:

    key = str(protocol).strip().lower()

    if key in {"lodo", "leave_one_dataset_out"}:
        uncertainty = (
            f"Held-out strategy performance is summarized across held-out datasets as mean ± SD. "
            f"The 95% confidence intervals are percentile intervals from {int(n_bootstrap):,} bootstrap replicates that resample held-out datasets with replacement."
        )

    else:
        uncertainty = (
            f"Held-out strategy performance is summarized across outer test folds as mean ± SD. "
            f"The 95% confidence intervals are percentile intervals from {int(n_bootstrap):,} hierarchical bootstrap replicates that resample repeats and then outer folds within sampled repeats."
        )

    metrics = (
        " AUROC measures ranking discrimination across thresholds. AUCPR is the trapezoidal area under the empirical precision-recall curve. Average precision summarizes precision weighted by recall increments and is reported separately. "
        "MCC is reported on its conventional [-1, 1] scale. F1w is support-weighted F1. "
        "Precision and recall are macro-averaged across classes."
    )

    return f"<p>{html.escape(uncertainty)}{html.escape(metrics)}</p>"


def _inferential_layer_html(protocol: Any, metric: Any) -> str:

    key = str(protocol).strip().lower()

    metric_text = html.escape(_metric_display_label(metric))

    if key in {"lodo", "leave_one_dataset_out"}:
        body = (
            f"For each strategy pair, {metric_text} values are matched by held-out dataset. "
            "The reported difference is Strategy A minus Strategy B, and its 95% confidence interval is obtained by paired bootstrap resampling of the matched held-out datasets. "
            "The p-value is from a paired sign-flip randomization test at the held-out-dataset level. "
            "Holm adjustment is applied across all displayed strategy-pair and metric hypotheses. "
            "For higher-is-better metrics, a positive difference favors Strategy A. For loss metrics, a negative difference favors Strategy A."
        )

    else:
        body = (
            f"For each strategy pair, {metric_text} values are matched by outer test fold. "
            "The reported difference is Strategy A minus Strategy B, and its 95% confidence interval is obtained from a paired hierarchical bootstrap that resamples repeats and then matched outer folds within sampled repeats. "
            "The p-value is from the Nadeau–Bengio corrected resampled paired t-test, which adjusts the variance for dependence induced by overlapping training sets. "
            "Holm adjustment is applied across all displayed strategy-pair and metric hypotheses. "
            "For higher-is-better metrics, a positive difference favors Strategy A. For loss metrics, a negative difference favors Strategy A."
        )

    return f"<p><strong>Inferential layer.</strong> {body}</p>"


def _top_mpma_heading(table: pd.DataFrame) -> str:

    n = len(table)

    return f"Top {n} MPMA-B configurations" if n > 0 else "Top MPMA-B configurations"


REPORT_TEMPLATE_VERSION = "standard-v3"

REPORT_LAYERS = (
    "evaluation_procedure",
    "data_exploration",
    "descriptive_performance_uncertainty",
    "inferential_comparisons",
    "selected_specifications",
    "important_features",
    "explainability",
    "robustness",
    "compute_hardware",
)


_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect x="1" y="1" width="62" height="62" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/><text x="32" y="39" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="22" font-weight="700" fill="#0f172a">mll</text></svg>"""


def _favicon_href() -> str:

    payload = base64.b64encode(_FAVICON_SVG.encode("utf-8")).decode("ascii")

    return f"data:image/svg+xml;base64,{payload}"


def _single_rank_label(row: pd.Series | dict[str, Any]) -> str | None:

    res = str(row.get("resolution", "")).strip().lower()

    levels = str(row.get("levels", "")).strip().lower().replace(" ", "")

    for rank in _SINGLE_RANK_PRIORITY:
        if res == rank or levels == rank:
            return rank

    return None


def _pick_deepest_single_rank(
    sub: pd.DataFrame, *, sort_col: str, ascending: bool = False
) -> dict[str, Any]:

    if sub.empty:
        return {}

    candidates = []

    for rank in _SINGLE_RANK_PRIORITY:
        rr = sub[
            sub.apply(lambda r, rank=rank: _single_rank_label(r) == rank, axis=1)
        ].copy()

        if rr.empty:
            continue

        if sort_col in rr.columns:
            rr = rr.sort_values(sort_col, ascending=ascending)

        candidates.append(rr.iloc[0].to_dict())

        break

    if candidates:
        return candidates[0]

    return {}


from .report_explainability import (
    _explainability_report_blocks,
    _fig,
    _read_json,
    _read_table,
)


def _representation_impact_note(metric: str, modality: bool = False) -> str:

    metric_text = html.escape(str(metric).replace("_", "-"))

    heatmap_text = "panel c provides" if modality else "panels c–d provide"

    return (
        f"<p>Panel a shows the distribution of outer-unit mean {metric_text} values across transformation–learner configurations for each representation; boxes show the median and interquartile range, whiskers extend to 1.5×IQR, and points are individual outer evaluation units. "
        "Panel b reports split-adjusted hierarchical partial η² from a least-squares model containing representation, transformation, learner, representation×learner, and transformation×learner terms. Main-factor effects remove the factor together with interactions containing it; interaction effects remove only that interaction. Effects are conditional, need not sum to one, and non-estimable terms are shown as n/a. "
        f"{heatmap_text} mean held-out {metric_text} across learners and outer units.</p>"
    )


def _safe_float(x: Any) -> float:

    try:
        v = float(x)

        return v if np.isfinite(v) else float("nan")

    except Exception:
        return float("nan")


def _format_p_value(value: Any) -> str:

    v = _safe_float(value)

    if not np.isfinite(v):
        return ""

    if 0 <= v < 0.001:
        return "<0.001"

    return f"{v:.3f}"


def _primary_pairwise_display(pairwise: pd.DataFrame, metric: str) -> pd.DataFrame:

    if pairwise.empty or "metric" not in pairwise.columns:
        return pd.DataFrame()

    key = str(metric).strip().casefold().replace("-", "_").replace(" ", "_")

    key = {"logloss": "log_loss", "brier_loss": "brier"}.get(key, key)

    out = pairwise[pairwise["metric"].astype(str).str.casefold().eq(key)].copy()

    if out.empty:
        return out

    columns = [
        c
        for c in [
            "strategy_a",
            "strategy_b",
            "difference_a_minus_b",
            "difference_ci_low",
            "difference_ci_high",
            "n_matched_outer_units",
            "n_paired_outer_units",
            "test",
            "p_value",
            "p_holm",
            "significant_holm_0_05",
        ]
        if c in out.columns
    ]

    out = out[columns].copy()

    out = out.rename(
        columns={
            "strategy_a": "Strategy A",
            "strategy_b": "Strategy B",
            "difference_a_minus_b": "Difference A-B",
            "difference_ci_low": "95% CI low",
            "difference_ci_high": "95% CI high",
            "n_matched_outer_units": "Matched outer units",
            "n_paired_outer_units": "Matched outer units",
            "test": "Test",
            "p_value": "p",
            "p_holm": "Holm p",
            "significant_holm_0_05": "Holm p<0.05",
        }
    )

    for column in ("Difference A-B", "95% CI low", "95% CI high"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(
                lambda x: f"{x:.3f}" if np.isfinite(x) else ""
            )

    for column in ("p", "Holm p"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").map(
                _format_p_value
            )

    if "Test" in out.columns:
        labels = {
            "nadeau_bengio_corrected_t": "Nadeau–Bengio corrected t",
            "exact_paired_sign_flip": "Exact paired sign-flip",
            "monte_carlo_paired_sign_flip": "Monte Carlo paired sign-flip",
            "paired_sign_flip": "Paired sign-flip",
        }

        out["Test"] = (
            out["Test"].astype(str).map(lambda value: labels.get(value, value))
        )

    return out


def _html_number(value: Any) -> str | None:

    if isinstance(value, (bool, np.bool_)):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""

        return f"{float(value):.3f}"

    return None


def _pct_cell(
    mean: Any, std: Any, *, bold: bool = False, html_mode: bool = False
) -> str:

    m = _safe_float(mean)

    s = _safe_float(std)

    if not np.isfinite(m):
        body = "NA" if html_mode else r"NA"

        return body

    body = f"{100 * m:.3f}"

    if np.isfinite(s):
        sd = f"{100 * s:.3f}" if html_mode else f"{100 * s:06.3f}"

        body += (" ± " if html_mode else r"$\pm$") + sd

    if bold:
        return f"<strong>{body}</strong>" if html_mode else rf"\textbf{ {body} } "

    return body


def _pct_ci_cell(
    estimate: Any,
    std: Any,
    low: Any,
    high: Any,
    *,
    bold: bool = False,
    html_mode: bool = False,
) -> str:

    e = _safe_float(estimate)

    sd = _safe_float(std)

    lo = _safe_float(low)

    hi = _safe_float(high)

    if not np.isfinite(e):
        return "NA" if html_mode else r"NA"

    body = f"{100 * e:.3f}"

    if np.isfinite(sd):
        sd_text = f"{100 * sd:.3f}" if html_mode else f"{100 * sd:06.3f}"

        body += (" ± " if html_mode else r"$\pm$") + sd_text

    if np.isfinite(lo) and np.isfinite(hi):
        body += f" [{100 * lo:.3f}, {100 * hi:.3f}]"

    if bold:
        return f"<strong>{body}</strong>" if html_mode else rf"\textbf{ {body} } "

    return body


def _strategy_statistics_summary(root: Path) -> pd.DataFrame:

    return _read_table(
        root / "report" / "tables" / "strategy_metrics_bootstrap.parquet"
    )


def _mean_std_from_cols(
    row: pd.Series | dict[str, Any], prefix: str, metric: str
) -> tuple[float, float]:

    if isinstance(row, dict):
        get = row.get

    else:
        get = row.get

    mean = get(f"{prefix}_{metric}_mean", np.nan)

    std = get(f"{prefix}_{metric}_std", np.nan)

    if not np.isfinite(_safe_float(mean)):
        mean = get(f"{metric}_mean", np.nan)

        std = get(f"{metric}_std", std)

    return _safe_float(mean), _safe_float(std)


def _agg_metrics(
    path: Path, prefix: str, extra_metrics: tuple[str, ...] = ()
) -> pd.DataFrame:

    df = _read_table(path)

    if df.empty or "config_id" not in df.columns:
        return pd.DataFrame()

    if "ok" in df.columns:
        df = df[pd.to_numeric(df["ok"], errors="coerce").fillna(0).eq(1)]

    if df.empty:
        return pd.DataFrame()

    id_cols = [
        c
        for c in [
            "config_id",
            "count_transformation",
            "transformation_abbreviation",
            "resolution",
            "learner",
        ]
        if c in df.columns
    ]

    metric_names = [c for c, _ in _METRICS]

    metric_names.extend(_canonical_metric_name(c) for c in extra_metrics)

    metric_cols = [c for c in dict.fromkeys(metric_names) if c in df.columns]

    if not metric_cols:
        return df[id_cols].drop_duplicates("config_id") if id_cols else pd.DataFrame()

    g = df.groupby(id_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])

    g.columns = [f"{prefix}_{m}_{stat}" for m, stat in g.columns]

    return g.reset_index()


def _top_mpma_raw(
    root: Path, n: int = 5, selection_metric: str = "log_loss"
) -> pd.DataFrame:

    metric = _canonical_metric_name(selection_metric)

    inner = _agg_metrics(
        root / "inner_results" / "inner_results.parquet", "inner", (metric,)
    )

    outer = _agg_metrics(root / "results" / "outer_results.parquet", "outer", (metric,))

    if inner.empty and outer.empty:
        return pd.DataFrame()

    if not inner.empty:
        base = inner

        if not outer.empty:
            outer_metrics = outer[
                [c for c in outer.columns if c == "config_id" or c.startswith("outer_")]
            ].copy()

            base = base.merge(outer_metrics, on="config_id", how="left")

    else:
        base = outer

    sort_col = (
        f"inner_{metric}_mean"
        if f"inner_{metric}_mean" in base.columns
        else (
            f"outer_{metric}_mean"
            if f"outer_{metric}_mean" in base.columns
            else (
                "inner_log_loss_mean"
                if "inner_log_loss_mean" in base.columns
                else (
                    "outer_log_loss_mean"
                    if "outer_log_loss_mean" in base.columns
                    else None
                )
            )
        )
    )

    if sort_col:
        sort_metric = metric if metric in sort_col else "log_loss"

        base = base.sort_values(sort_col, ascending=metric_is_loss(sort_metric))

    base = base.head(int(n)).copy()

    base.insert(0, "rank", np.arange(1, len(base) + 1))

    return base


def _metric_mean_std_cell(
    mean: Any,
    std: Any,
    metric: str,
    *,
    bold: bool = False,
    html_mode: bool = False,
) -> str:

    if _canonical_metric_name(metric) in {
        "log_loss",
        "subject_macro_log_loss",
        "cohort_macro_log_loss",
        "brier",
    }:
        m = _safe_float(mean)

        sd = _safe_float(std)

        if not np.isfinite(m):
            return "NA" if html_mode else r"NA"

        body = f"{m:.3f}"

        if np.isfinite(sd):
            body += (" ± " if html_mode else r"$\pm$") + f"{sd:.3f}"

        if bold:
            return f"<strong>{body}</strong>" if html_mode else rf"\textbf{ {body} } "

        return body

    return _pct_cell(mean, std, bold=bold, html_mode=html_mode)


def _metric_ci_cell(
    estimate: Any,
    std: Any,
    low: Any,
    high: Any,
    metric: str,
    *,
    bold: bool = False,
    html_mode: bool = False,
) -> str:

    if _canonical_metric_name(metric) in {
        "log_loss",
        "subject_macro_log_loss",
        "cohort_macro_log_loss",
        "brier",
    }:
        e = _safe_float(estimate)

        sd = _safe_float(std)

        lo = _safe_float(low)

        hi = _safe_float(high)

        if not np.isfinite(e):
            return "NA" if html_mode else r"NA"

        body = f"{e:.3f}"

        if np.isfinite(sd):
            body += (" ± " if html_mode else r"$\pm$") + f"{sd:.3f}"

        if np.isfinite(lo) and np.isfinite(hi):
            body += f" [{lo:.3f}, {hi:.3f}]"

        if bold:
            return f"<strong>{body}</strong>" if html_mode else rf"\textbf{ {body} } "

        return body

    return _pct_ci_cell(estimate, std, low, high, bold=bold, html_mode=html_mode)


def _top_mpma_display(
    df: pd.DataFrame,
    *,
    selection_metric: str = "log_loss",
    html_mode: bool = False,
    learner_labels: dict[str, str] | None = None,
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    primary = _canonical_metric_name(selection_metric)

    metric_order = [primary] + [
        metric for metric in ("AUROC", "F1w") if metric != primary
    ]

    rows: list[dict[str, Any]] = []

    for _, r in df.iterrows():
        row = {
            "Rank": int(r.get("rank", len(rows) + 1)),
            "Taxonomic representation": _display_token(r.get("resolution", "")),
            "Count transformation": _display_token(
                r.get("transformation_abbreviation", r.get("count_transformation", ""))
            ),
            "Learner": _display_learner(r.get("learner", ""), learner_labels),
        }

        for metric in metric_order:
            label = _metric_display_label(metric)

            row[f"Inner {label}"] = _metric_mean_std_cell(
                *_mean_std_from_cols(r, "inner", metric), metric, html_mode=html_mode
            )

            row[f"Outer {label}"] = _metric_mean_std_cell(
                *_mean_std_from_cols(r, "outer", metric), metric, html_mode=html_mode
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _selected_json(root: Path) -> dict[str, Any]:

    return _read_json(root / "ensembling" / "selected_unit.json")


def _best_mpma_row(root: Path) -> dict[str, Any]:

    nested = _read_json(root / "tables" / "mpma_b_strategy_summary.json")

    if isinstance(nested, dict) and nested:
        return nested

    selected = _selected_json(root)

    best = (
        selected.get("inner_val_best_mpma")
        or selected.get("inner_val_best_individual")
        or {}
    )

    return best if isinstance(best, dict) else {}


def _ensemble_outer_metrics_from_predictions(
    root: Path, ensemble_config_id: str
) -> dict[str, Any]:

    path = root / "ensembling" / "ensemble_predictions.parquet"

    if not table_exists(path) or not str(ensemble_config_id):
        return {}

    df = _read_table(path)

    if (
        df.empty
        or "ensemble_config_id" not in df.columns
        or "outer_split_key" not in df.columns
    ):
        return {}

    df = df[df["ensemble_config_id"].astype(str).eq(str(ensemble_config_id))].copy()

    if df.empty:
        return {}

    pcols = [c for c in df.columns if c.startswith("proba_")]

    if not pcols or "y_true" not in df.columns or "y_pred" not in df.columns:
        return {}

    classes = np.arange(len(pcols), dtype=int)

    fold_rows: list[dict[str, float]] = []

    for _, sub in df.groupby("outer_split_key", sort=False):
        yy = pd.to_numeric(sub["y_true"], errors="coerce")

        mask = yy.notna()

        y_true = yy.loc[mask].astype(int).to_numpy()

        y_pred = (
            pd.to_numeric(sub.loc[mask, "y_pred"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )

        proba = (
            sub.loc[mask, pcols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        if len(y_true) == 0 or proba.shape[0] != len(y_true):
            continue

        fold_rows.append(compute_metrics(y_true, y_pred, proba, classes))

    if not fold_rows:
        return {}

    tab = pd.DataFrame(fold_rows)

    out: dict[str, Any] = {}

    for col in tab.columns:
        vals = (
            pd.to_numeric(tab[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if vals.empty:
            continue

        out[f"outer_{col}_mean"] = float(vals.mean())

        out[f"outer_{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        out[f"outer_{col}_count"] = len(vals)

        out[f"{col}_mean"] = out[f"outer_{col}_mean"]

        out[f"{col}_std"] = out[f"outer_{col}_std"]

        out[f"{col}_count"] = out[f"outer_{col}_count"]

    return out


def _ensemble_row(root: Path) -> dict[str, Any]:

    nested = _read_json(root / "ensembling" / "mpma_e_strategy_summary.json")

    if isinstance(nested, dict) and nested:
        return nested

    selected = _selected_json(root)

    ens = (
        selected.get("inner_val_best_mpmas_ensemble")
        or selected.get("inner_val_best_ensemble")
        or {}
    )

    if not isinstance(ens, dict):
        return {}

    row = dict(ens)

    eid = str(row.get("ensemble_config_id", ""))

    for k, v in _ensemble_outer_metrics_from_predictions(root, eid).items():
        if k not in row or not np.isfinite(_safe_float(row.get(k))):
            row[k] = v

    return row


def _ensemble_final_candidate(root: Path) -> dict[str, Any]:

    final = _read_json(root / "ensembling" / "mpma_e_final_candidate.json")

    if isinstance(final, dict) and final:
        return final

    selected = _selected_json(root)

    ens = (
        selected.get("inner_val_best_mpmas_ensemble")
        or selected.get("inner_val_best_ensemble")
        or {}
    )

    return ens if isinstance(ens, dict) else {}


def _strategy_rows(
    root: Path, selection_metric: str = "log_loss"
) -> list[dict[str, Any]]:

    by_strategy: dict[str, dict[str, Any]] = {}

    ens = _ensemble_row(root)

    if ens:
        by_strategy["MPMA-E"] = {"Strategy": "MPMA-E", "source": "ensemble", **ens}

    best = _best_mpma_row(root)

    if best:
        by_strategy["MPMA-B"] = {"Strategy": "MPMA-B", "source": "mpma", **best}

    metric = _canonical_metric_name(selection_metric)

    outer = _agg_metrics(root / "results" / "outer_results.parquet", "outer", (metric,))

    inner = _agg_metrics(
        root / "inner_results" / "inner_results.parquet", "inner", (metric,)
    )

    if not outer.empty:
        base = outer.copy()

        if not inner.empty:
            inner_cols = [
                c for c in inner.columns if c == "config_id" or c.startswith("inner_")
            ]

            base = base.merge(inner[inner_cols], on="config_id", how="left")

        def _pick(mask: pd.Series) -> dict[str, Any]:

            sub = base[mask.fillna(False)].copy()

            if sub.empty:
                return {}

            sort_col = (
                f"inner_{metric}_mean"
                if f"inner_{metric}_mean" in sub.columns
                else (
                    f"outer_{metric}_mean"
                    if f"outer_{metric}_mean" in sub.columns
                    else (
                        "inner_log_loss_mean"
                        if "inner_log_loss_mean" in sub.columns
                        else "outer_log_loss_mean"
                    )
                )
            )

            if sort_col in sub.columns:
                sort_metric = metric if metric in sort_col else "log_loss"

                sub = sub.sort_values(sort_col, ascending=metric_is_loss(sort_metric))

            return sub.iloc[0].to_dict()

        sort_col = (
            f"inner_{metric}_mean"
            if f"inner_{metric}_mean" in base.columns
            else (
                f"outer_{metric}_mean"
                if f"outer_{metric}_mean" in base.columns
                else (
                    "inner_log_loss_mean"
                    if "inner_log_loss_mean" in base.columns
                    else "outer_log_loss_mean"
                )
            )
        )

        sort_metric = metric if metric in sort_col else "log_loss"

        sort_ascending = metric_is_loss(sort_metric)

        learner = base.get(
            "learner", pd.Series("", index=base.index, dtype=str)
        ).astype(str)

        learner_class = base.get(
            "learner_class", pd.Series("", index=base.index, dtype=str)
        ).astype(str)

        transform = base.get(
            "count_transformation", pd.Series("", index=base.index, dtype=str)
        ).astype(str)

        resolution = base.get(
            "resolution", pd.Series("", index=base.index, dtype=str)
        ).astype(str)

        automl_pool = base[
            learner.str.contains("FLAML|AutoML", case=False, regex=True).fillna(False)
            & transform.str.fullmatch(
                r"relative_abundance(?:@(rank-wise|joint))?", case=False
            ).fillna(False)
        ].copy()

        automl = _pick_deepest_single_rank(
            automl_pool, sort_col=sort_col, ascending=sort_ascending
        )

        if automl:
            by_strategy["AutoML"] = {"Strategy": "AutoML", "source": "mpma", **automl}

        configs = _read_table(root / "configs.parquet")
        baseline_configs = resolve_baseline_rf_configs(configs)
        baseline = {}
        if not baseline_configs.empty and "config_id" in baseline_configs.columns:
            baseline_id = str(baseline_configs.iloc[0]["config_id"])
            baseline_rows = base[base["config_id"].astype(str).eq(baseline_id)].copy()
            if not baseline_rows.empty:
                if sort_col in baseline_rows.columns:
                    baseline_rows = baseline_rows.sort_values(
                        sort_col, ascending=sort_ascending
                    )
                baseline = baseline_rows.iloc[0].to_dict()

        if baseline:
            by_strategy["Baseline RF"] = {
                "Strategy": "Baseline RF",
                "source": "mpma",
                **baseline,
            }

        siamcat_mask = learner.str.fullmatch("SIAMCAT", case=False).fillna(False)
        siamcat_mask |= learner_class.str.endswith("SIAMCATClassifier", na=False)
        siamcat = _pick(
            siamcat_mask
            & transform.str.fullmatch("identity", case=False).fillna(False)
            & resolution.str.fullmatch("raw", case=False).fillna(False)
        )

        if siamcat:
            by_strategy["SIAMCAT"] = {
                "Strategy": "SIAMCAT",
                "source": "mpma",
                **siamcat,
            }

    return [
        by_strategy[k]
        for k in ("MPMA-E", "MPMA-B", "AutoML", "Baseline RF", "SIAMCAT")
        if k in by_strategy
    ]


def _baseline_rf_report_note(rows: list[dict[str, Any]]) -> str:
    if not any(str(row.get("Strategy", "")) == "Baseline RF" for row in rows):
        return ""
    return (
        "<p><strong>Baseline RF protocol.</strong> Deepest active eligible single taxonomic rank (strain, species, or genus); "
        "within-rank relative abundance followed by arcsine-square-root transformation; no prevalence filtering; "
        "RandomForestClassifier with 1,000 trees and minimum leaf size 5. All remaining random-forest "
        "hyperparameters use scikit-learn defaults; no dataset-specific random-forest hyperparameter tuning is performed.</p>"
    )


def _strategy_performance_display(
    root: Path, *, selection_metric: str = "log_loss", html_mode: bool = False
) -> pd.DataFrame:

    rows = _strategy_rows(root, selection_metric)

    if not rows:
        return pd.DataFrame()

    display_metrics = _classification_display_metrics(selection_metric)

    stats = _strategy_statistics_summary(root)

    if not stats.empty and {
        "Strategy",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
    }.issubset(stats.columns):
        best_by_metric: dict[str, str] = {}

        for metric, _ in display_metrics:
            sub = stats[stats["metric"].astype(str).eq(metric)].copy()

            sub["estimate"] = pd.to_numeric(sub["estimate"], errors="coerce")

            sub = sub[np.isfinite(sub["estimate"].to_numpy(dtype=float))]

            if not sub.empty:
                sub = sub.sort_values("estimate", ascending=metric_is_loss(metric))

                best_by_metric[metric] = str(sub.iloc[0]["Strategy"])

        out = []

        for row in rows:
            strategy = str(row.get("Strategy", ""))

            rr = {"Strategy": strategy}

            for metric, label in display_metrics:
                sub = stats[
                    stats["Strategy"].astype(str).eq(strategy)
                    & stats["metric"].astype(str).eq(metric)
                ]

                if sub.empty:
                    rr[label] = "NA" if html_mode else r"NA"

                    continue

                r = sub.iloc[0]

                rr[label] = _metric_ci_cell(
                    r.get("estimate"),
                    r.get("std"),
                    r.get("ci_low"),
                    r.get("ci_high"),
                    metric,
                    bold=(best_by_metric.get(metric) == strategy),
                    html_mode=html_mode,
                )

            out.append(rr)

        return pd.DataFrame(out)

    best_by_metric: dict[str, str] = {}

    for metric, _ in display_metrics:
        vals = []

        for row in rows:
            m, _ = _mean_std_from_cols(row, "outer", metric)

            if np.isfinite(m):
                vals.append((m, str(row.get("Strategy", ""))))

        if vals:
            chooser = min if metric_is_loss(metric) else max

            best_by_metric[metric] = chooser(vals, key=lambda x: x[0])[1]

    out = []

    for row in rows:
        strategy = str(row.get("Strategy", ""))

        rr = {"Strategy": strategy}

        for metric, label in display_metrics:
            rr[label] = _metric_mean_std_cell(
                *_mean_std_from_cols(row, "outer", metric),
                metric,
                bold=(best_by_metric.get(metric) == strategy),
                html_mode=html_mode,
            )

        out.append(rr)

    return pd.DataFrame(out)


def _ensemble_summary_table(root: Path) -> pd.DataFrame:

    ens = _ensemble_final_candidate(root)

    if not ens:
        ens = _ensemble_row(root)

    if not ens:
        return pd.DataFrame()

    members = ens.get("effective_member_count")

    if members is None:
        members = ens.get("member_count")

    if members is None:
        raw_members = ens.get("members")

        if isinstance(raw_members, str):
            try:
                raw_members = json.loads(raw_members)

            except Exception:
                raw_members = None

        if isinstance(raw_members, list):
            members = len(raw_members)

        else:
            members = ens.get("ensemble_size", "")

    max_size = ens.get("max_size")

    if max_size is None:
        max_size = ens.get("ensemble_size", members)

    row = {
        "Strategy": "MPMA-E",
        "Selection": ens.get("selection_strategy", ""),
        "Aggregation": ens.get("aggregation_strategy", ""),
        "Max size": max_size,
        "Members": members,
        "Selection metric": ens.get(
            "selection_metric",
            ens.get("optimize_metric", ""),
        ),
    }

    return pd.DataFrame([row])


def _ensemble_members_table(
    root: Path, learner_labels: dict[str, str] | None = None
) -> pd.DataFrame:

    ens = _ensemble_final_candidate(root)

    members = ens.get("members", []) if isinstance(ens, dict) else []

    if isinstance(members, str):
        try:
            members = json.loads(members)

        except Exception:
            members = []

    members = [str(x) for x in members] if isinstance(members, list) else []

    weights = ens.get("weights", []) if isinstance(ens, dict) else []

    if isinstance(weights, str):
        try:
            weights = json.loads(weights)

        except Exception:
            weights = []

    weights = list(weights) if isinstance(weights, list) else []

    p = root / "tables" / "mpma_e_members.parquet"

    if not table_exists(p):
        p = root / "figures" / "mpma_e_members.parquet"

    if table_exists(p):
        existing = read_table(p)

        if not existing.empty:
            rename = {
                "member_order": "Member",
                "config_id": "Config ID",
                "ranks": "Representation",
                "transformation": "Transformation",
                "classifier_family": "Learner",
                "n_features_in_demo": "Features",
                "modalities": "Modalities",
                "integration": "Integration",
                "weight": "Weight",
            }

            existing = existing.rename(
                columns={k: v for k, v in rename.items() if k in existing.columns}
            )

            columns = [
                "Member",
                "Config ID",
                "Family",
                "Modalities",
                "Integration",
                "Representation",
                "Transformation",
                "Learner",
                "Features",
                "Weight",
            ]

            selected = [column for column in columns if column in existing.columns]

            if selected:
                out = existing[selected].copy()

                for column in ("Representation", "Transformation", "Integration"):
                    if column in out.columns:
                        out[column] = out[column].map(_display_token)

                if "Learner" in out.columns:
                    learner_source: dict[str, str] = {}

                    learner_display_source: dict[str, str] = {}

                    configs_path = root / "configs.parquet"

                    if "Config ID" in out.columns and table_exists(configs_path):
                        configs = read_table(configs_path, dtype=str)

                        if not configs.empty and {"config_id", "learner"}.issubset(
                            configs.columns
                        ):
                            learner_source = dict(
                                zip(
                                    configs["config_id"].astype(str),
                                    configs["learner"].astype(str),
                                )
                            )

                        if not configs.empty and {
                            "config_id",
                            "learner_display",
                        }.issubset(configs.columns):
                            learner_display_source = dict(
                                zip(
                                    configs["config_id"].astype(str),
                                    configs["learner_display"].astype(str),
                                )
                            )

                    out["Learner"] = [
                        learner_display_source.get(str(config_id), "").strip()
                        or _display_learner(
                            learner_source.get(str(config_id), value),
                            learner_labels,
                        )
                        for config_id, value in zip(
                            out.get("Config ID", pd.Series([""] * len(out))),
                            out["Learner"],
                        )
                    ]

                return out

    configs_path = root / "configs.parquet"

    configs = (
        read_table(configs_path, dtype=str)
        if table_exists(configs_path)
        else pd.DataFrame()
    )

    if not members:
        return pd.DataFrame()

    if configs.empty or "config_id" not in configs.columns:
        return pd.DataFrame(
            {
                "Member": range(1, len(members) + 1),
                "Config ID": members,
                "Weight": [
                    float(weights[i]) if i < len(weights) else np.nan
                    for i in range(len(members))
                ],
            }
        )

    lookup = configs.drop_duplicates("config_id", keep="first").set_index("config_id")

    rows: list[dict[str, Any]] = []

    for index, config_id in enumerate(members, start=1):
        row: dict[str, Any] = {"Member": index, "Config ID": config_id}

        if config_id in lookup.index:
            source = lookup.loc[config_id]

            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]

            mapping = (
                ("candidate_family", "Family"),
                ("modalities", "Modalities"),
                ("integration", "Integration"),
                ("integration_n_components", "Components"),
                ("resolution", "Representation"),
                ("count_transformation", "Transformation"),
                ("learner", "Learner"),
            )

            for source_name, display_name in mapping:
                value = source.get(source_name, "")

                if pd.notna(value) and str(value).strip():
                    row[display_name] = str(value)

        if index - 1 < len(weights):
            try:
                value = float(weights[index - 1])

            except Exception:
                value = float("nan")

            if np.isfinite(value):
                row["Weight"] = value

        rows.append(row)

    out = pd.DataFrame(rows)

    columns = [
        "Member",
        "Config ID",
        "Family",
        "Modalities",
        "Integration",
        "Components",
        "Representation",
        "Transformation",
        "Learner",
        "Weight",
    ]

    out = out[[column for column in columns if column in out.columns]]

    for column in ("Representation", "Transformation", "Integration"):
        if column in out.columns:
            out[column] = out[column].map(_display_token)

    if "Learner" in out.columns:
        out["Learner"] = out["Learner"].map(
            lambda value: _display_learner(value, learner_labels)
        )

    return out


def _manifest(root: Path) -> dict[str, Any]:

    return _read_json(root / "manifest.json")


def _sweep_data_source(sweep: Sweep):

    return sweep.samples if getattr(sweep, "uses_modalities", False) else sweep.data


def _procedure_table(sweep: Sweep, root: Path) -> pd.DataFrame:

    manifest = _manifest(root)

    ev = sweep.evaluation

    gate = sweep.gate

    data = (
        manifest.get("sweep", {}).get("data", {})
        if isinstance(manifest.get("sweep"), dict)
        else {}
    )

    source = _sweep_data_source(sweep)

    classes = manifest.get("class_labels", getattr(source, "class_labels", ()))

    inclusion = (
        manifest.get("multimodal_inclusion", {})
        if getattr(sweep, "uses_modalities", False)
        else {}
    )

    sample_value = str(manifest.get("n_samples", ""))

    if isinstance(inclusion, dict) and inclusion:
        complete = inclusion.get("n_complete_samples", manifest.get("n_samples", ""))
        primary = inclusion.get("n_primary_samples", "")
        sample_value = (
            f"{complete} complete case(s) from {primary} primary-modality sample(s)"
        )

    rows = [
        ["Task", sweep.title],
        ["Experiment directory", str(root)],
        [
            "Data format",
            "multimodal"
            if getattr(sweep, "uses_modalities", False)
            else _display_data_format(
                getattr(source, "format", data.get("format", ""))
            ),
        ],
        ["Samples", sample_value],
        ["Classes", ", ".join(map(str, classes)) if classes else ""],
        ["Procedure", ev.protocol],
        [
            "Stratification",
            "target"
            if not getattr(source, "stratify_col", None)
            else f"target + {source.stratify_col}",
        ],
        [
            "Outer folds",
            ev.outer_folds
            if ev.protocol not in {"lodo", "leave_one_dataset_out"}
            else "held-out datasets",
        ],
        [
            "Inner folds",
            "leave-one-dataset-out across outer-training datasets"
            if ev.protocol in {"lodo", "leave_one_dataset_out"}
            else ev.inner_folds,
        ],
        ["Repeats", ev.repeats],
        ["Selection metric", ev.optimize_metric],
        ["Random seed", ev.random_state],
        [
            "Qualification gate",
            f"on ({gate.metric} ≥ {gate.threshold})" if gate.enabled else "off",
        ],
        [
            "Selection rule",
            "Reported performance uses held-out outer evaluation predictions.",
        ],
    ]

    if isinstance(inclusion, dict) and inclusion:
        rows.insert(
            4,
            [
                "Multimodal inclusion",
                "explicit complete-case intersection across all requested modalities",
            ],
        )
        rows.insert(5, ["Multimodal estimand", str(inclusion.get("estimand", ""))])

    return pd.DataFrame(rows, columns=["Field", "Value"])


def _format_count_fraction(count: Any, total: Any) -> str:

    try:
        count_i = int(count)
        total_i = int(total)
    except (TypeError, ValueError):
        return ""

    if total_i <= 0:
        return str(count_i)

    return f"{count_i} ({100.0 * count_i / total_i:.1f}%)"


def _distribution_frame(
    payload: dict[str, Any], totals: dict[str, int], label: str
) -> pd.DataFrame:

    categories: set[str] = set()

    for subset in ("primary", "complete", "excluded"):
        section = payload.get(subset, {}) if isinstance(payload, dict) else {}
        counts = section.get("counts", {}) if isinstance(section, dict) else {}
        if isinstance(counts, dict):
            categories.update(str(key) for key in counts)

    rows: list[dict[str, Any]] = []

    for category in sorted(categories):
        row: dict[str, Any] = {label: category}
        for subset, heading in (
            ("primary", "Primary"),
            ("complete", "Complete case"),
            ("excluded", "Excluded"),
        ):
            section = payload.get(subset, {}) if isinstance(payload, dict) else {}
            counts = section.get("counts", {}) if isinstance(section, dict) else {}
            count = counts.get(category, 0) if isinstance(counts, dict) else 0
            row[heading] = _format_count_fraction(count, totals.get(subset, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def _categorical_distribution_frame(
    payload: dict[str, Any], totals: dict[str, int], label: str
) -> pd.DataFrame:

    categories: set[str] = set()

    for subset in ("primary", "complete", "excluded"):
        values = payload.get(subset, {}) if isinstance(payload, dict) else {}
        if isinstance(values, dict):
            categories.update(str(key) for key in values)

    rows: list[dict[str, Any]] = []

    for category in sorted(categories):
        row: dict[str, Any] = {label: category}
        for subset, heading in (
            ("primary", "Primary"),
            ("complete", "Complete case"),
            ("excluded", "Excluded"),
        ):
            values = payload.get(subset, {}) if isinstance(payload, dict) else {}
            count = values.get(category, 0) if isinstance(values, dict) else 0
            row[heading] = _format_count_fraction(count, totals.get(subset, 0))
        rows.append(row)

    return pd.DataFrame(rows)


def _regression_target_frame(payload: dict[str, Any]) -> pd.DataFrame:

    labels = {
        "n": "N",
        "mean": "Mean",
        "std": "SD",
        "median": "Median",
        "q1": "Q1",
        "q3": "Q3",
        "min": "Minimum",
        "max": "Maximum",
    }

    rows: list[dict[str, Any]] = []

    for key, display in labels.items():
        row: dict[str, Any] = {"Statistic": display}
        for subset, heading in (
            ("primary", "Primary"),
            ("complete", "Complete case"),
            ("excluded", "Excluded"),
        ):
            section = payload.get(subset, {}) if isinstance(payload, dict) else {}
            value = section.get(key) if isinstance(section, dict) else None
            if value is None:
                row[heading] = ""
            elif key == "n":
                row[heading] = str(int(value))
            else:
                row[heading] = f"{float(value):.4g}"
        rows.append(row)

    return pd.DataFrame(rows)


def _multimodal_inclusion_html(manifest: dict[str, Any]) -> str:

    inclusion = manifest.get("multimodal_inclusion", {})

    if not isinstance(inclusion, dict) or not inclusion:
        return ""

    primary_n = int(inclusion.get("n_primary_samples", 0) or 0)
    complete_n = int(inclusion.get("n_complete_samples", 0) or 0)
    excluded_n = int(inclusion.get("n_excluded_samples", 0) or 0)
    primary_subjects = int(inclusion.get("n_primary_subjects", 0) or 0)
    complete_subjects = int(inclusion.get("n_complete_subjects", 0) or 0)
    fully_excluded_subjects = int(inclusion.get("n_fully_excluded_subjects", 0) or 0)

    incomplete_subjects = int(
        inclusion.get("n_subjects_with_incomplete_samples", 0) or 0
    )
    partially_retained_subjects = int(
        inclusion.get("n_partially_retained_subjects", 0) or 0
    )

    attrition = pd.DataFrame(
        [
            ["Primary-modality samples", str(primary_n)],
            ["Complete-case samples", _format_count_fraction(complete_n, primary_n)],
            ["Excluded samples", _format_count_fraction(excluded_n, primary_n)],
            ["Primary-modality subjects", str(primary_subjects)],
            [
                "Retained subjects",
                _format_count_fraction(complete_subjects, primary_subjects),
            ],
            [
                "Fully excluded subjects",
                _format_count_fraction(fully_excluded_subjects, primary_subjects),
            ],
            [
                "Subjects with ≥1 incomplete sample",
                _format_count_fraction(incomplete_subjects, primary_subjects),
            ],
            [
                "Partially retained subjects",
                _format_count_fraction(partially_retained_subjects, primary_subjects),
            ],
        ],
        columns=["Measure", "Value"],
    )

    availability_rows: list[dict[str, Any]] = []
    availability = inclusion.get("modality_availability", {})

    if isinstance(availability, dict):
        for modality, values in availability.items():
            if not isinstance(values, dict):
                continue
            available = int(values.get("available_primary_samples", 0) or 0)
            missing = int(values.get("missing_primary_samples", 0) or 0)
            availability_rows.append(
                {
                    "Modality": str(modality),
                    "Available": _format_count_fraction(available, primary_n),
                    "Missing": _format_count_fraction(missing, primary_n),
                }
            )

    availability_frame = pd.DataFrame(availability_rows)

    pattern_rows: list[dict[str, Any]] = []
    patterns = inclusion.get("missingness_patterns", [])

    if isinstance(patterns, list):
        for row in patterns:
            if not isinstance(row, dict):
                continue
            missing = row.get("missing_modalities", [])
            names = (
                [str(value) for value in missing] if isinstance(missing, list) else []
            )
            count = int(row.get("n_samples", 0) or 0)
            pattern_rows.append(
                {
                    "Missing modalities": ", ".join(names) if names else "none",
                    "Samples": _format_count_fraction(count, primary_n),
                }
            )

    pattern_frame = pd.DataFrame(pattern_rows)

    totals = {
        "primary": primary_n,
        "complete": complete_n,
        "excluded": excluded_n,
    }

    target = inclusion.get("target_distribution", {})
    target_frame = pd.DataFrame()

    if isinstance(target, dict):
        primary_target = target.get("primary", {})
        if (
            isinstance(primary_target, dict)
            and primary_target.get("kind") == "classification"
        ):
            target_frame = _distribution_frame(target, totals, "Target")
        elif (
            isinstance(primary_target, dict)
            and primary_target.get("kind") == "regression"
        ):
            target_frame = _regression_target_frame(target)

    group = inclusion.get("group_distribution")
    group_frame = pd.DataFrame()
    group_heading = "Grouping distribution"

    if isinstance(group, dict):
        group_frame = _categorical_distribution_frame(
            group, totals, str(group.get("column", "Group"))
        )
        group_heading = (
            f"Grouping distribution: {html.escape(str(group.get('column', 'group')))}"
        )

    characteristic_rows: list[dict[str, Any]] = []
    characteristics = inclusion.get("characteristic_balance", [])

    if isinstance(characteristics, list):
        for row in characteristics:
            if not isinstance(row, dict):
                continue
            value = row.get("balance_value")
            formatted = "" if value is None else f"{float(value):.3f}"
            metric = str(row.get("balance_metric", ""))
            metric_label = (
                "SMD"
                if metric.startswith("standardized_mean_difference")
                else "Max |Δ proportion|"
            )
            characteristic_rows.append(
                {
                    "Characteristic": str(row.get("variable", "")),
                    "Type": str(row.get("kind", "")),
                    "Primary": str(row.get("primary", "")),
                    "Complete case": str(row.get("complete", "")),
                    "Excluded": str(row.get("excluded", "")),
                    metric_label: formatted,
                }
            )

    characteristic_frame = pd.DataFrame(characteristic_rows)

    estimand = html.escape(
        str(
            inclusion.get(
                "estimand",
                "samples with complete observations across all requested modalities "
                "within the primary-modality cohort",
            )
        )
    )
    parts = [
        '<section id="multimodal-inclusion">',
        "<h2>Multimodal inclusion and missingness</h2>",
        (
            "<p>Multimodal evaluation uses an explicit complete-case intersection. "
            f"Performance estimates therefore apply to {estimand}. The primary-modality "
            "cohort remains the reporting denominator so exclusions caused by unavailable "
            "modalities are visible rather than silently discarded.</p>"
        ),
        "<h3>Inclusion and attrition</h3>",
        _html_table(attrition),
    ]

    if not availability_frame.empty:
        parts.extend(
            ["<h3>Modality availability</h3>", _html_table(availability_frame)]
        )

    if not pattern_frame.empty:
        parts.extend(["<h3>Missingness patterns</h3>", _html_table(pattern_frame)])

    if not target_frame.empty:
        parts.extend(
            [
                "<h3>Target distribution before and after complete-case restriction</h3>",
                _html_table(target_frame),
            ]
        )

    if not group_frame.empty:
        parts.extend([f"<h3>{group_heading}</h3>", _html_table(group_frame)])

    if not characteristic_frame.empty:
        parts.extend(
            [
                "<h3>Declared characteristic balance</h3>",
                (
                    "<p>Continuous variables are summarized as median [Q1, Q3] with "
                    "standardized mean difference between complete and excluded samples. "
                    "Categorical variables are summarized by counts and percentages with "
                    "the maximum absolute category-proportion difference.</p>"
                ),
                _html_table(characteristic_frame),
            ]
        )

    if excluded_n:
        parts.append(
            "<p>Observed differences between complete and excluded observations should be "
            "considered when interpreting multimodal generalizability because modality "
            "availability can change the analyzed population.</p>"
        )

    parts.append("</section>")

    return "".join(parts)


from .console import console as console
from .explainability_visuals import plot_feature_support as plot_feature_support
from .explainability_visuals import plot_local_attributions as plot_local_attributions
from .report_explainability import _asset_uri as _asset_uri
from .report_explainability import _inline_svg as _inline_svg
from .report_explainability import (
    _refresh_xai_support_figures as _refresh_xai_support_figures,
)
from .report_explainability import _target_dirs_by_label as _target_dirs_by_label
from .report_explainability import _xai_class_slug as _xai_class_slug
from .report_explainability import _xai_coordinate_column as _xai_coordinate_column
from .report_explainability import _xai_figure_for_class as _xai_figure_for_class
from .report_explainability import _xai_local_figure as _xai_local_figure
from .report_explainability import _xai_local_mode as _xai_local_mode
from .report_explainability import _xai_method_display as _xai_method_display
from .report_explainability import _xai_method_global_text as _xai_method_global_text
from .report_explainability import _xai_target_metadata as _xai_target_metadata
from .report_terminal import (
    _compact_procedure_for_terminal as _compact_procedure_for_terminal,
)
from .report_terminal import _feature_support_table
from .report_terminal import _feature_support_terminal as _feature_support_terminal
from .report_terminal import (
    _hardware_summary_table,
    _html_inline,
    _print_report_summary,
    _short_feature_label,
)
from .report_terminal import _strip_cell_markup as _strip_cell_markup
from .report_terminal import _target_dirs_for_terminal as _target_dirs_for_terminal
from .report_terminal import _terminal_feature_label as _terminal_feature_label
from .report_terminal import _terminal_table as _terminal_table
from .storage import glob_tables as glob_tables
from .utils import feature_tail_ellipsis as feature_tail_ellipsis


def _html_taxon_prefixes(value: Any) -> str:
    escaped = html.escape(str(value))
    return re.sub(
        r"(?<![A-Za-z])([dpcofgst]\.)\s+(?=[A-Za-z])",
        r"<i>\1</i> ",
        escaped,
    )


def _html_table(df: pd.DataFrame, *, raw_html_cols: set[str] | None = None) -> str:

    if df.empty:
        return "<p>No rows available.</p>"

    raw_html_cols = raw_html_cols or set()

    hidden_cols = {
        "Config ID",
        "config_id",
        "Model / ensemble",
        "model_or_ensemble_id",
        "Ensemble ID",
        "ensemble_config_id",
        "Model ID",
        "model_id",
        "MPDR ID",
        "mpdr_id",
        "Candidate ID",
        "candidate_id",
    }

    cols = [c for c in df.columns if str(c) not in hidden_cols]

    parts = ['<div class="table-wrap"><table><thead><tr>']

    parts.extend(f'<th scope="col">{html.escape(str(c))}</th>' for c in cols)

    parts.append("</tr></thead><tbody>")

    for _, r in df.iterrows():
        parts.append("<tr>")

        for c in cols:
            val = r[c]

            if pd.isna(val):
                txt = ""

            elif c in raw_html_cols:
                txt = str(val)

            else:
                number = _html_number(val)

                if number is not None:
                    txt = html.escape(number)

                else:
                    rendered = _html_inline(val)

                    column_name = str(c).casefold()

                    if "feature" in column_name or "taxon" in column_name:
                        rendered = _short_feature_label(val, 72)

                    txt = _html_taxon_prefixes(rendered)

            parts.append(f"<td>{txt}</td>")

        parts.append("</tr>")

    parts.append("</tbody></table></div>")

    return "".join(parts)


def _procedure_grid_html(procedure: pd.DataFrame) -> str:

    if procedure.empty or not {"Field", "Value"}.issubset(procedure.columns):
        return _html_table(procedure)

    rows = [(str(row["Field"]), str(row["Value"])) for _, row in procedure.iterrows()]

    wide = [row for row in rows if row[0] == "Selection rule"]

    compact = [row for row in rows if row[0] != "Selection rule"]

    midpoint = (len(compact) + 1) // 2

    columns = (compact[:midpoint], compact[midpoint:])

    parts = ['<div class="procedure-grid">']

    for column in columns:
        parts.append('<div class="procedure-column">')

        for field, value in column:
            parts.append('<div class="procedure-row">')

            parts.append(f'<div class="procedure-field">{html.escape(field)}</div>')

            parts.append(f'<div class="procedure-value">{html.escape(value)}</div>')

            parts.append("</div>")

        parts.append("</div>")

    for field, value in wide:
        parts.append('<div class="procedure-row procedure-wide">')

        parts.append(f'<div class="procedure-field">{html.escape(field)}</div>')

        parts.append(f'<div class="procedure-value">{html.escape(value)}</div>')

        parts.append("</div>")

    parts.append("</div>")

    return "".join(parts)


def _publication_layout_css() -> str:

    return """
.procedure-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); column-gap:34px; margin:10px 0 24px; }
.procedure-column { min-width:0; }
.procedure-row { display:grid; grid-template-columns:132px minmax(0,1fr); gap:12px; padding:6px 7px; border-bottom:1px solid var(--track); line-height:1.38; }
.procedure-field { color:var(--mid); font-size:var(--font-table); font-weight:700; }
.procedure-value { color:var(--ink); font-size:var(--font-table); min-width:0; overflow-wrap:anywhere; }
.procedure-wide { grid-column:1 / -1; margin-top:2px; }
.explore-rank { margin:18px 0 34px; }
.explore-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px 26px; align-items:start; margin:12px 0 24px; }
.explore-figure { min-width:0; }
.explore-figure-wide { grid-column:1 / -1; }
.explore-figure figure { width:max-content; max-width:100%; margin:0; overflow:visible; }
.explore-figure .embedded-svg { display:inline-block; width:max-content; max-width:100%; }
.explore-figure .embedded-svg svg { width:auto; max-width:100%; height:auto; display:block; }
.explore-detail-link { margin:6px 0 26px; }
@media (max-width: 760px) {
  .procedure-grid { grid-template-columns:1fr; column-gap:0; }
  .procedure-wide { grid-column:auto; }
  .explore-grid { grid-template-columns:1fr; }
  .explore-figure-wide { grid-column:auto; }
}
"""


def _report_css() -> str:

    return """
:root {
  --ink:#0f172a; --mid:#64748b; --dim:#94a3b8; --track:#e2e8f0; --soft:#f8fafc;
  --bg:#ffffff; --blue:#2563eb; --blue-hover:#0ea5e9; --blue-soft:#eff6ff; --blue-hover-soft:#f0f9ff;
  --nav-h:44px; --content-w:1400px;
  --font-body:14px; --font-small:13px; --font-table:13px;
}
*, *::before, *::after { box-sizing:border-box; }
html {
  background:var(--bg); color:var(--ink);
  font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, Helvetica, sans-serif;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
  letter-spacing:-0.01em; scroll-behavior:smooth;
}
body { margin:0; padding:0; background:var(--bg); }
a { color:var(--blue); text-decoration:none; }
a:hover { opacity:0.85; text-decoration:none; }
a:focus-visible { outline:2px solid #38bdf8; outline-offset:3px; border-radius:6px; }
.report-nav {
  position:sticky; top:0; z-index:30; height:var(--nav-h);
  background:rgba(255,255,255,0.93); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border-bottom:1px solid #f1f5f9; box-shadow:none;
}
.report-nav-inner {
  height:var(--nav-h); max-width:var(--content-w); margin:0 auto; padding:0 24px;
  display:flex; align-items:center; justify-content:space-between; gap:22px;
}
.brand { display:flex; align-items:center; gap:9px; color:var(--ink); font-size:0.82rem; font-weight:700; text-decoration:none; white-space:nowrap; }
.brand:hover { opacity:0.85; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border:1px solid var(--track); border-radius:7px; font-size:0.72rem; background:#fff; letter-spacing:-0.03em; }
.report-links { display:flex; align-items:center; gap:3px; overflow-x:auto; white-space:nowrap; min-width:0; }
.report-links a { color:var(--mid); font-size:0.78rem; font-weight:500; padding:7px 8px; border-radius:6px; }
.report-links a:hover { color:var(--blue-hover); background:var(--blue-hover-soft); opacity:1; }
.report-shell { max-width:var(--content-w); margin:0 auto; padding:28px 24px 70px; }
.report-content { min-width:0; }
h1 { font-size:26px; line-height:1.16; margin:0 0 8px; font-weight:800; letter-spacing:-0.04em; }
h2 { font-size:16px; margin:38px 0 14px; font-weight:750; border-top:1px solid var(--track); padding-top:20px; letter-spacing:-0.025em; }
h2.first-section { margin-top:0; border-top:0; padding-top:0; }
h1[id], h2[id], h3[id], h4[id], h5[id], h6[id], section[id], main[id] { scroll-margin-top:calc(var(--nav-h) + 14px); }
h3 { font-size:13px; font-weight:700; margin:16px 0 10px; }
h4 { font-size:13px; font-weight:750; margin:28px 0 10px; }
h5 { font-size:12px; font-weight:750; margin:22px 0 8px; color:var(--ink); }
h6 { font-size:12px; font-weight:650; margin:16px 0 7px; color:var(--mid); }
.xai-target { margin:0 0 34px; }
.xai-class { border-top:1px solid var(--track); margin-top:18px; padding-top:2px; }
.xai-class figure { margin:10px 0 16px; }
.xai-importance-grid { display:grid; grid-template-columns:repeat(var(--xai-importance-columns), minmax(0,1fr)); gap:12px; align-items:start; margin:12px 0 12px; }
.xai-importance-grid-single { width:min(100%, 360px); }
.xai-importance-cell { min-width:0; }
.xai-importance-cell h6 { margin:0 0 6px; color:var(--ink); }
.xai-importance-cell figure { width:100%; max-width:100%; margin:0; overflow:visible; }
.xai-importance-cell .embedded-svg { width:100%; max-width:100%; }
.xai-importance-cell .embedded-svg svg { width:100%; max-width:100%; height:auto; }
.xai-importance-cell img { width:100%; max-width:100%; height:auto; }
.xai-importance-cell figcaption { display:none; }
.xai-importance-definitions { margin:6px 0 18px; font-size:var(--font-small); }
p, li { font-size:var(--font-body); color:var(--mid); line-height:1.56; }
.report-path { margin-top:0; }
figure { margin:14px 0 22px; overflow-x:auto; width:max-content; max-width:100%; }
figure img { max-width:100%; width:auto; height:auto; display:block; }
.embedded-svg { display:inline-block; width:max-content; max-width:100%; }
.embedded-svg svg { width:auto; max-width:100%; height:auto; display:block; }
figcaption { font-size:var(--font-small); color:var(--mid); margin-top:7px; }
.compare-block { margin:17px 0 32px; overflow-x:auto; padding-bottom:2px; }
.compare-block > h3 { font-size:13px; margin:18px 0 11px; color:var(--ink); }
.compare-grid { display:grid; grid-template-columns:repeat(var(--compare-columns), minmax(430px, 1fr)); gap:16px; align-items:start; }
.compare-cell { min-width:0; }
.compare-cell h3 { font-size:12px; color:var(--mid); margin:0 0 7px; font-weight:700; }
.compare-cell figure { margin:0; }
.compare-cell figcaption { display:none; }
.compare-cell img, .compare-cell .embedded-svg svg { width:auto; max-width:100%; min-width:0; }
.missing-figure { border:1px solid var(--track); color:var(--mid); font-size:12px; padding:36px 12px; text-align:center; background:#fff; border-radius:10px; }
.table-wrap { overflow-x:auto; margin:12px 0 24px; }
table { border-collapse:collapse; width:100%; font-size:var(--font-table); }
th, td { border-bottom:1px solid var(--track); padding:8px 9px; text-align:left; vertical-align:top; line-height:1.38; }
th { color:var(--ink); font-weight:700; background:#f8fafc; border-top:1px solid var(--track); }
tbody tr:last-child td { border-bottom:1px solid #cbd5e1; }
td { font-variant-numeric:tabular-nums; }
tbody tr:hover td { background:#fafcff; }
code { font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:12px; }
.report-footer { border-top:1px solid var(--track); margin-top:42px; padding-top:16px; color:var(--dim); font-size:12px; }
@media (max-width: 860px) {
  .report-nav-inner { padding:0 16px; }
  .report-shell { padding:22px 18px 56px; }
  .brand span:last-child { display:none; }
  .report-links a { padding:7px 6px; }
  h1 { font-size:22px; }
  .compare-grid { grid-template-columns:repeat(var(--compare-columns), minmax(360px, 1fr)); }
  .compare-cell img, .compare-cell .embedded-svg svg { min-width:0; }
  .xai-importance-grid { grid-template-columns:repeat(2,minmax(0,1fr)) !important; }
}
@media (max-width: 620px) {
  .xai-importance-grid { grid-template-columns:1fr !important; }
}
""" + _publication_layout_css()


@dataclass(frozen=True)
class _ReportAnalysis:
    procedure: pd.DataFrame
    multimodal_inclusion_html: str
    learner_labels: dict[str, str]
    top_mpmas: pd.DataFrame
    strategy_rows: list[dict[str, Any]]
    compute_accounting: dict[str, Any]
    compute_display: pd.DataFrame
    compute_environment: dict[str, Any]
    statistics: dict[str, Any]
    strategy_html: pd.DataFrame
    primary_pairwise: pd.DataFrame
    ensemble_summary: pd.DataFrame
    ensemble_members: pd.DataFrame


@dataclass(frozen=True)
class _ReportVisuals:
    representation_figure: str
    mpma_e_figure: str
    explainability_html: str
    explainability_count: int


@dataclass(frozen=True)
class _ReportTables:
    feature_summary_path: Path
    hardware_summary_path: Path
    hardware_summary: pd.DataFrame


def _prepare_report_directories(root: Path) -> tuple[Path, Path]:
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    return report_dir, tables_dir


def _analyze_report(sweep: Sweep, root: Path) -> _ReportAnalysis:
    with phase_progress("Report analysis", 4) as phase:
        phase.phase("procedure and rankings")
        procedure = _procedure_table(sweep, root)
        multimodal_inclusion_html = (
            _multimodal_inclusion_html(_manifest(root))
            if getattr(sweep, "uses_modalities", False)
            else ""
        )
        learner_labels = _learner_display_map(sweep)
        top_mpmas = _top_mpma_display(
            _top_mpma_raw(root, 5, str(sweep.evaluation.optimize_metric)),
            selection_metric=str(sweep.evaluation.optimize_metric),
            html_mode=True,
            learner_labels=learner_labels,
        )
        if getattr(sweep, "uses_modalities", False) and not top_mpmas.empty:
            top_mpmas = top_mpmas.rename(
                columns={
                    "Taxonomic representation": "Representation",
                    "Resolution": "Representation",
                    "Count transformation": "Transformation",
                }
            )
        strategy_rows = _strategy_rows(root, str(sweep.evaluation.optimize_metric))
        phase.phase("compute accounting")
        compute_accounting = run_compute_accounting(root, sweep, strategy_rows)
        compute_display = compute_accounting.get("display", pd.DataFrame())
        compute_environment = compute_accounting.get("environment", {})
        phase.phase("statistics")
        statistics = run_report_statistics(
            root,
            sweep.evaluation.protocol,
            strategy_rows,
            n_bootstrap=2000,
            random_state=sweep.evaluation.random_state,
            selection_metric=str(sweep.evaluation.optimize_metric),
            diagnostic_thresholds=getattr(
                sweep.evaluation, "diagnostic_thresholds", ()
            ),
            decision_curve_min_threshold=getattr(
                sweep.evaluation, "decision_curve_min_threshold", 0.01
            ),
            decision_curve_max_threshold=getattr(
                sweep.evaluation, "decision_curve_max_threshold", 0.99
            ),
            decision_curve_points=getattr(
                sweep.evaluation, "decision_curve_points", 99
            ),
        )
        strategy_html = _strategy_performance_display(
            root,
            selection_metric=str(sweep.evaluation.optimize_metric),
            html_mode=True,
        )
        primary_pairwise = _primary_pairwise_display(
            statistics.get("pairwise", pd.DataFrame()),
            str(sweep.evaluation.optimize_metric),
        )
        phase.phase("ensemble summaries")
        ensemble_summary = _ensemble_summary_table(root)
        ensemble_members = _ensemble_members_table(root, learner_labels)
    return _ReportAnalysis(
        procedure=procedure,
        multimodal_inclusion_html=multimodal_inclusion_html,
        learner_labels=learner_labels,
        top_mpmas=top_mpmas,
        strategy_rows=strategy_rows,
        compute_accounting=compute_accounting,
        compute_display=compute_display,
        compute_environment=compute_environment,
        statistics=statistics,
        strategy_html=strategy_html,
        primary_pairwise=primary_pairwise,
        ensemble_summary=ensemble_summary,
        ensemble_members=ensemble_members,
    )


def _build_report_visuals(
    sweep: Sweep,
    root: Path,
    report_dir: Path,
) -> _ReportVisuals:
    with phase_progress("Report outputs", 2) as phase:
        phase.phase("representation and MPMA-E figure")
        representation_figure = _fig(
            root / "figures" / "representation_impact",
            report_dir,
            "Data representations and learner impact on held-out performance",
        )
        mpma_e_figure = _fig(
            root / "figures" / "mpma_e",
            report_dir,
            "Selected MPMA-E schematic",
        )
        phase.phase("explainability visuals")
        from .explainability import refresh_explainability_visuals

        refresh_explainability_visuals(root, sweep)
        explainability_html, explainability_count = _explainability_report_blocks(
            root,
            report_dir,
            top_n=int(getattr(sweep.explainability, "top_k", 15)),
        )
    return _ReportVisuals(
        representation_figure=representation_figure,
        mpma_e_figure=mpma_e_figure,
        explainability_html=explainability_html,
        explainability_count=explainability_count,
    )


def _write_report_tables(
    root: Path,
    tables_dir: Path,
    analysis: _ReportAnalysis,
) -> _ReportTables:
    feature_summary_path = tables_dir / "important_features.parquet"
    write_table(feature_summary_path, _feature_support_table(root, top_n=8))
    hardware_summary = _hardware_summary_table(analysis.compute_environment)
    hardware_summary_path = tables_dir / "hardware_environment.parquet"
    write_table(hardware_summary_path, hardware_summary)
    return _ReportTables(
        feature_summary_path=feature_summary_path,
        hardware_summary_path=hardware_summary_path,
        hardware_summary=hardware_summary,
    )


def _robustness_report_html(root: Path) -> str:
    directory = root / "robustness"
    if not directory.exists():
        return ""
    manifest_path = directory / "robustness_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                manifest = value
        except Exception:
            manifest = {}
    task = str(manifest.get("task", "classification"))
    confidence = float(manifest.get("confidence_level", 0.95))
    confidence_label = f"{100.0 * confidence:.0f}% CI"
    resampling_unit = str(manifest.get("robustness_resampling_unit", "subject_id"))
    subgroup_labels: dict[str, str] = {}
    parts = ['<h2 id="robustness">Robustness, confounding and generalization</h2>']
    inventory_path = directory / "metadata_inventory.parquet"
    if table_exists(inventory_path):
        table = read_table(inventory_path)
        if not table.empty:
            for _, row in table.iterrows():
                column = str(row.get("column", "")).strip()
                role = str(row.get("role", "")).strip()
                if column and role:
                    subgroup_labels[column] = _humanize_identifier(role)
            display = table.rename(
                columns={
                    "role": "Semantic role",
                    "category": "Role category",
                    "column": "Metadata column",
                    "kind": "Type",
                    "n_samples": "Samples",
                    "n_subjects": "Declared subjects",
                    "n_resampling_clusters": "Dependence clusters",
                    "observed": "Observed samples",
                    "observed_subjects": "Observed declared subjects",
                    "observed_clusters": "Observed clusters",
                    "missing": "Missing samples",
                    "missing_subjects": "Missing declared subjects",
                    "missing_fraction": "Missing fraction",
                    "unique": "Unique values",
                }
            )
            keep = [
                column
                for column in (
                    "Semantic role",
                    "Role category",
                    "Metadata column",
                    "Type",
                    "Samples",
                    "Declared subjects",
                    "Dependence clusters",
                    "Observed samples",
                    "Observed declared subjects",
                    "Observed clusters",
                    "Missing samples",
                    "Missing declared subjects",
                    "Missing fraction",
                    "Unique values",
                )
                if column in display.columns
            ]
            parts.extend(
                [
                    "<h3>Metadata specification</h3>",
                    "<p>Declared metadata semantics are carried from the data specification into robustness analyses. Semantic declarations identify variables available for descriptive balance and subgroup evaluation; they do not constitute causal adjustment.</p>",
                    _html_table(display[keep]),
                ]
            )
    balance_path = directory / "covariate_balance.parquet"
    if table_exists(balance_path):
        table = read_table(balance_path)
        if not table.empty:
            display = table.rename(
                columns={
                    "covariate": "Covariate",
                    "role": "Role",
                    "kind": "Type",
                    "analysis_unit": "Analysis unit",
                    "n_units": "Units",
                    "observed": "Observed",
                    "missing": "Missing",
                    "max_pairwise_smd": "Max pairwise SMD",
                    "max_pairwise_raw_difference": "Raw difference",
                    "comparison": "Outcome-group comparison",
                    "most_imbalanced_level": "Level",
                    "missingness_smd": "Missingness SMD",
                }
            )
            keep = [
                column
                for column in (
                    "Covariate",
                    "Role",
                    "Type",
                    "Analysis unit",
                    "Units",
                    "Observed",
                    "Missing",
                    "Max pairwise SMD",
                    "Raw difference",
                    "Outcome-group comparison",
                    "Level",
                    "Missingness SMD",
                )
                if column in display.columns
            ]
            parts.extend(
                [
                    "<h3>Covariate and technical balance</h3>",
                    "<p>Balance is descriptive and is not interpreted as causal adjustment. Numeric variables use the largest pairwise standardized mean difference with a pooled standard deviation. Categorical variables use the largest pairwise standardized difference in level prevalence. Missingness imbalance is evaluated separately. When both outcome and covariate are invariant within the declared dependence cluster, balance is evaluated once per cluster; otherwise it is sample-level.</p>",
                    _html_table(display[keep]),
                ]
            )
    support_path = directory / "subgroup_support.parquet"
    if table_exists(support_path):
        table = read_table(support_path)
        if not table.empty:
            display = table.copy()
            if "subgroup" in display.columns:
                display["subgroup"] = display["subgroup"].map(
                    lambda value: subgroup_labels.get(
                        str(value), _humanize_identifier(value)
                    )
                )
            if "level" in display.columns:
                display["level"] = display["level"].map(_humanize_level)
            if "support_status" in display.columns:
                display["support_status"] = display["support_status"].map(
                    _humanize_identifier
                )
            display = display.rename(
                columns={
                    "strategy": "Strategy",
                    "subgroup": "Subgroup variable",
                    "level": "Subgroup level",
                    "n_samples": "Unique samples, n",
                    "n_subjects": "Declared subjects, n",
                    "n_resampling_clusters": "Dependence clusters, n",
                    "n_repeats": "CV repeats, n",
                    "class_counts_samples": "Class counts by sample",
                    "class_counts_clusters": "Class counts by cluster",
                    "support_status": "Support",
                    "reason": "Support note",
                }
            )
            keep = [
                column
                for column in (
                    "Strategy",
                    "Subgroup variable",
                    "Subgroup level",
                    "Unique samples, n",
                    "Declared subjects, n",
                    "Dependence clusters, n",
                    "CV repeats, n",
                    "Class counts by sample",
                    "Class counts by cluster",
                    "Support",
                    "Support note",
                )
                if column in display.columns
            ]
            parts.extend(
                [
                    "<h3>Subgroup support and coverage</h3>",
                    f"<p>Exact sample, subject and dependence-cluster counts are reported before subgroup performance. The uncertainty resampling unit is <strong>{html.escape(resampling_unit)}</strong>. Limited class support is retained as a warning rather than hidden.</p>",
                    _html_table(display[keep]),
                ]
            )
    subgroup_path = directory / "subgroup_performance.parquet"
    if table_exists(subgroup_path):
        table = read_table(subgroup_path)
        if not table.empty:
            binary_metrics = (
                "AUROC",
                "AP",
                "log_loss",
                "brier",
                "CalibrationSlope",
                "MCC",
                "BalAcc",
            )
            multiclass_metrics = (
                "AUROC_macro",
                "AP_macro",
                "log_loss",
                "brier",
                "CalibrationSlope_macro_OvR",
                "MCC",
                "BalAcc",
            )
            regression_metrics = ("R2", "MAE", "RMSE", "SpearmanR")
            labels = manifest.get("class_labels", [])
            preferred = (
                regression_metrics
                if task == "regression"
                else (binary_metrics if len(labels) == 2 else multiclass_metrics)
            )
            selected = table[table["metric"].astype(str).isin(preferred)].copy()
            if not selected.empty:
                selected["subgroup"] = selected["subgroup"].map(
                    lambda value: subgroup_labels.get(
                        str(value), _humanize_identifier(value)
                    )
                )
                selected["level"] = selected["level"].map(_humanize_level)

                def interval_text(row: pd.Series) -> str:
                    estimate = pd.to_numeric(
                        pd.Series([row.get("estimate")]), errors="coerce"
                    ).iloc[0]
                    low = pd.to_numeric(
                        pd.Series([row.get("ci_low")]), errors="coerce"
                    ).iloc[0]
                    high = pd.to_numeric(
                        pd.Series([row.get("ci_high")]), errors="coerce"
                    ).iloc[0]
                    if not np.isfinite(estimate):
                        return ""
                    if np.isfinite(low) and np.isfinite(high):
                        return f"{float(estimate):.3f} ({float(low):.3f}–{float(high):.3f})"
                    return f"{float(estimate):.3f} (CI unavailable)"

                selected["value"] = selected.apply(interval_text, axis=1)
                index_columns = [
                    "strategy",
                    "subgroup",
                    "level",
                    "n_samples",
                    "n_resampling_clusters",
                ]
                wide = selected.pivot_table(
                    index=index_columns,
                    columns="metric",
                    values="value",
                    aggfunc="first",
                ).reset_index()
                wide.columns.name = None
                metric_renames = {
                    metric: f"{_metric_display_label(metric)} ({confidence_label})"
                    for metric in preferred
                }
                wide = wide.rename(
                    columns={
                        "strategy": "Strategy",
                        "subgroup": "Subgroup variable",
                        "level": "Subgroup level",
                        "n_samples": "Samples, n",
                        "n_resampling_clusters": "Dependence clusters, n",
                        **metric_renames,
                    }
                )
                metric_columns = [
                    metric_renames[metric]
                    for metric in preferred
                    if metric_renames[metric] in wide.columns
                ]
                ordered = [
                    column
                    for column in (
                        "Strategy",
                        "Subgroup variable",
                        "Subgroup level",
                        "Samples, n",
                        "Dependence clusters, n",
                        *metric_columns,
                    )
                    if column in wide.columns
                ]
                estimands = [
                    str(value)
                    for value in selected.get("estimand", pd.Series(dtype=str))
                    .dropna()
                    .unique()
                ]
                estimand_text = (
                    "mean of repeat-specific pooled OOF estimates"
                    if estimands == ["mean_repeat_pooled_oof"]
                    else ", ".join(estimands)
                )
                parts.extend(
                    [
                        "<h3>Subgroup performance</h3>",
                        f"<p>Subgroup performance is evaluated only from strategy-specific outer-held-out predictions, without subgroup-specific model selection or refitting. The estimand is the {html.escape(estimand_text)}. Values are point estimates with {confidence_label} from a nonparametric dependence-cluster bootstrap that preserves all repeated observations within each resampled cluster. Calibration and proper scoring metrics are shown only for strategies that produce valid probabilities. Intervals are conditional on the existing outer-held-out predictions and do not rerun model fitting or model selection.</p>",
                        _html_table(wide[ordered]),
                    ]
                )
    contrasts_path = directory / "subgroup_contrasts.parquet"
    if table_exists(contrasts_path):
        table = read_table(contrasts_path)
        if not table.empty:

            def contrast_interval(row: pd.Series) -> str:
                estimate = pd.to_numeric(
                    pd.Series([row.get("difference_a_minus_b")]), errors="coerce"
                ).iloc[0]
                low = pd.to_numeric(
                    pd.Series([row.get("ci_low")]), errors="coerce"
                ).iloc[0]
                high = pd.to_numeric(
                    pd.Series([row.get("ci_high")]), errors="coerce"
                ).iloc[0]
                if not np.isfinite(estimate):
                    return ""
                if np.isfinite(low) and np.isfinite(high):
                    return f"{float(estimate):.3f} ({float(low):.3f}–{float(high):.3f})"
                return f"{float(estimate):.3f} (CI unavailable)"

            display = table.copy()
            display["subgroup"] = display["subgroup"].map(
                lambda value: subgroup_labels.get(
                    str(value), _humanize_identifier(value)
                )
            )
            display["level_a"] = display["level_a"].map(_humanize_level)
            display["level_b"] = display["level_b"].map(_humanize_level)
            display["Difference"] = display.apply(contrast_interval, axis=1)
            display["metric"] = display["metric"].map(_metric_display_label)
            display = display.rename(
                columns={
                    "strategy": "Strategy",
                    "subgroup": "Subgroup variable",
                    "level_a": "Level A",
                    "level_b": "Level B",
                    "estimand": "Estimand",
                    "metric": "Metric",
                }
            )
            keep = [
                column
                for column in (
                    "Strategy",
                    "Subgroup variable",
                    "Level A",
                    "Level B",
                    "Estimand",
                    "Metric",
                    "Difference",
                )
                if column in display.columns
            ]
            parts.extend(
                [
                    "<h3>Subgroup performance contrasts</h3>",
                    f"<p>Pairwise contrasts are Level A minus Level B with paired {confidence_label} cluster-bootstrap intervals. They quantify performance heterogeneity on the held-out predictions. They are exploratory, carry no multiplicity-adjusted hypothesis test, and should not be interpreted by comparing subgroup confidence-interval overlap.</p>",
                    _html_table(display[keep]),
                ]
            )
    features_path = directory / "important_feature_robustness.parquet"
    if table_exists(features_path):
        table = read_table(features_path)
        if not table.empty:
            rename = {
                "strategy": "Strategy",
                "method": "Method",
                "class_label": "Class",
                "feature": "Feature",
                "importance_mean": "Mean importance",
                "median_rank": "Median rank",
                "rank_median": "Median rank",
                "rank_iqr": "Rank IQR",
                "top_k_frequency": "Top-k frequency",
                "fold_coverage": "Fold coverage",
                "sign_consistency": "Sign consistency",
            }
            display = table.rename(columns=rename)
            if "Method" in display.columns:
                display["Method"] = display["Method"].map(
                    lambda value: (
                        str(value).upper()
                        if str(value).strip().lower() in {"shap", "lime", "ale"}
                        else _humanize_identifier(value)
                    )
                )
            if "Class" in display.columns:
                display["Class"] = display["Class"].map(_humanize_level)
            for column in ("Top-k frequency", "Fold coverage", "Sign consistency"):
                if column in display.columns:
                    numeric = pd.to_numeric(display[column], errors="coerce")
                    display[column] = [
                        f"{100.0 * float(value):.0f}%" if pd.notna(value) else ""
                        for value in numeric
                    ]
            for column in ("Median rank", "Rank IQR"):
                if column in display.columns:
                    numeric = pd.to_numeric(display[column], errors="coerce")
                    display[column] = [
                        f"{float(value):.2f}" if pd.notna(value) else ""
                        for value in numeric
                    ]
            keep = [
                column
                for column in (
                    "Strategy",
                    "Method",
                    "Class",
                    "Feature",
                    "Mean importance",
                    "Median rank",
                    "Rank IQR",
                    "Top-k frequency",
                    "Fold coverage",
                    "Sign consistency",
                )
                if column in display.columns
            ]
            parts.extend(
                [
                    "<h3>Important-feature stability</h3>",
                    "<p>Feature robustness is summarized separately for MPMA-B and MPMA-E from their outer-test explanations. Method and class are shown explicitly when applicable. These are descriptive cross-fit stability measures and not independent-fold confidence intervals.</p>",
                    _html_table(display[keep]),
                ]
            )
    if len(parts) == 1:
        return ""
    return "\n".join(parts)


def _exploration_report_html(root: Path, report_dir: Path) -> str:
    manifest_path = root / "explore" / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    ranks = manifest.get("ranks", [])
    if not isinstance(ranks, list) or not ranks:
        return ""
    from .explore import (
        _alpha_display_table,
        _alpha_inference_context,
        _beta_display_table,
        _da_display_table,
    )

    confidence_level = float(manifest.get("confidence_level", 0.95))
    parts = [
        '<h2 id="data-exploration">Data exploration</h2>',
        "<p>Exploration follows the task definition and evaluation procedure and is organized around non-redundant scientific questions. Figures are used for distributions, multivariate geometry and feature-level patterns; exact inferential quantities are reported in adjacent tables. Community-level analyses use compositional geometry where appropriate, and all inferential summaries preserve the declared dependence structure.</p>",
    ]
    for rank_payload in ranks:
        rank = str(rank_payload.get("rank", "")).strip()
        if not rank:
            continue
        rank_dir = root / "explore" / rank
        table_dir = rank_dir / "tables"
        figure_dir = rank_dir / "figures"
        alpha = _read_table(table_dir / "alpha_statistics.parquet")
        beta = _read_table(table_dir / "beta_statistics.parquet")
        da = _read_table(table_dir / "differential_abundance.parquet")
        da_meta = (
            rank_payload.get("differential_abundance", {})
            if isinstance(rank_payload.get("differential_abundance", {}), dict)
            else {}
        )
        method = str(da_meta.get("method", "")).strip()
        model = str(da_meta.get("model", "")).strip()
        parts.append(
            f'<section class="explore-rank"><h3>{html.escape(rank.title())}</h3>'
        )

        alpha_context = (
            _alpha_inference_context(alpha, confidence_level) if not alpha.empty else ""
        )
        alpha_caption = "Alpha-diversity distributions. Exact dependence-aware effect estimates and multiplicity-adjusted P values are reported below."
        if alpha_context:
            alpha_caption = f"{alpha_caption} {alpha_context}"
        alpha_figure = _fig(figure_dir / "alpha_diversity", report_dir, alpha_caption)
        if alpha_figure:
            parts.append(
                f'<div class="explore-figure explore-figure-wide">{alpha_figure}</div>'
            )
        if not alpha.empty:
            parts.append("<h4>Alpha-diversity inference</h4>")
            parts.append(_html_table(_alpha_display_table(alpha, confidence_level)))

        pcoa_figure = _fig(
            figure_dir / "aitchison_pcoa",
            report_dir,
            "Aitchison principal coordinates analysis of centered-log-ratio (CLR) community profiles. For classification tasks, outlines are descriptive 80% covariance ellipses that summarize within-group concentration and are not confidence regions. PERMANOVA tests whether community composition differs with the target; PERMDISP checks whether apparent separation could instead reflect unequal within-group dispersion. Exact effect sizes and FDR-adjusted tests are reported below.",
        )
        if pcoa_figure:
            parts.append(f'<div class="explore-figure">{pcoa_figure}</div>')
        if not beta.empty:
            parts.append("<h4>Beta-diversity inference</h4>")
            parts.append(_html_table(_beta_display_table(beta)))

        heatmap_figure = _fig(
            figure_dir / "clr_heatmap",
            report_dir,
            "Standardized CLR abundance structure for prevalent, high-variance taxa. Cell colour runs from pale lower values to cyan higher values. This panel is descriptive and is intended to expose sample- and taxon-level structure rather than taxon-wise inference.",
        )
        if heatmap_figure:
            parts.append("<h4>Multivariate abundance structure</h4>")
            parts.append(
                f'<div class="explore-figure explore-figure-wide">{heatmap_figure}</div>'
            )

        volcano_figure = _fig(
            figure_dir / "differential_abundance",
            report_dir,
            "Differential-abundance or taxon-association volcano. Straight dashed leaders connect selected labels to their feature points; exact estimates, confidence intervals, prevalence and FDR-adjusted P values are reported below.",
        )
        if volcano_figure:
            parts.append(
                f'<div class="explore-figure explore-figure-wide">{volcano_figure}</div>'
            )
        if method or not da.empty:
            parts.append("<h4>Differential abundance and taxon associations</h4>")
            method_text = ". ".join(value for value in (method, model) if value)
            if method_text:
                readable_method = method_text.replace(
                    "CLR linear model", "Centered log-ratio (CLR) linear model"
                )
                readable_method = readable_method.replace(
                    "HC3 robust covariance",
                    "HC3 heteroskedasticity-robust standard errors",
                )
                readable_method = readable_method.replace(
                    "HC3 heteroskedasticity-robust standard errors",
                    "HC3 heteroskedasticity-robust standard errors, which reduce sensitivity to unequal residual variance and influential observations",
                )
                readable_method = readable_method.replace(
                    "cluster-robust covariance",
                    "cluster-robust standard errors that preserve repeated/dependent observations",
                )
                parts.append(f"<p>{html.escape(readable_method)}.</p>")
            if not da.empty:
                parts.append(
                    _html_table(_da_display_table(da, confidence_level).head(20))
                )
        parts.append("</section>")
    return "\n".join(parts)


def _render_report_html(
    sweep: Sweep,
    analysis: _ReportAnalysis,
    visuals: _ReportVisuals,
    tables: _ReportTables,
) -> str:
    top_metric_cols = {
        column
        for column in analysis.top_mpmas.columns
        if column.startswith("Inner") or column.startswith("Outer")
    }
    strategy_metric_cols = {
        column for column in analysis.strategy_html.columns if column != "Strategy"
    }
    representation_figure = (
        visuals.representation_figure
        if visuals.representation_figure
        else "<p>No representation-impact figure is available yet.</p>"
    )
    pairwise_html = (
        _html_table(analysis.primary_pairwise)
        if not analysis.primary_pairwise.empty
        else f"<p>No matched pairwise comparisons were estimable for {html.escape(str(sweep.evaluation.optimize_metric))}. At least two displayed strategies with finite values on the same held-out outer units are required.</p>"
    )
    explainability_html = (
        visuals.explainability_html
        if visuals.explainability_html
        else "<p>No explainability artefacts are available yet.</p>"
    )
    robustness_html = _robustness_report_html(sweep.root())
    exploration_html = _exploration_report_html(
        Path(sweep.root()), Path(sweep.root()) / "report"
    )
    hardware_html = (
        _html_table(tables.hardware_summary)
        if not tables.hardware_summary.empty
        else "<p>Hardware details are unavailable for this run.</p>"
    )
    html_text = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>mllabiome report</title><link rel=\"icon\" type=\"image/svg+xml\" href=\"{_favicon_href()}\"><style>{_report_css()}</style></head>
<body>
{_report_nav_html(bool(robustness_html), bool(exploration_html))}
<div class=\"report-shell\"><main id=\"top\" class=\"report-content\">
<h2 id=\"performance-evaluation\" class=\"first-section\">Task definition and evaluation procedure</h2>
{_procedure_grid_html(analysis.procedure)}
{analysis.multimodal_inclusion_html}
{exploration_html}
<h2 id=\"performance-summary\">Task performance summary</h2>
{_performance_methodology_html(sweep.evaluation.protocol, 2000)}
{_html_table(analysis.strategy_html, raw_html_cols=strategy_metric_cols)}
{_baseline_rf_report_note(analysis.strategy_rows)}
<h2 id=\"representation-impact\">Data representations and learner impact on performance</h2>
{_representation_impact_note(str(sweep.evaluation.optimize_metric), bool(getattr(sweep, "uses_modalities", False)))}
{representation_figure}
<section id=\"mpma-e-specification\">
<h3 id=\"mpma-e\">Final MPMA-E specification</h3>
{visuals.mpma_e_figure}
{_html_table(analysis.ensemble_summary)}
{_html_table(analysis.ensemble_members)}
</section>
<h2 id=\"top-mpmas\">{_top_mpma_heading(analysis.top_mpmas)}</h2>
{_html_table(analysis.top_mpmas, raw_html_cols=top_metric_cols)}
<h3 id=\"statistics\">Outer-unit strategy comparisons</h3>
{_inferential_layer_html(sweep.evaluation.protocol, sweep.evaluation.optimize_metric)}
{pairwise_html}

<h2 id=\"explainability\">Explainability</h2>
<h3>Feature attribution</h3>
{explainability_html}

{robustness_html}

<h2 id=\"compute\">Computational resources</h2>
<p>Compute is summarized by additive CPU core-hours, model-fit count, and peak resident memory for the worker process tree. CPU time includes child processes and external R processes when used. MPMA-B and MPMA-E share the MPMA search pool, so their compute totals overlap. Baseline RF, SIAMCAT and AutoML are listed when those comparators were actually evaluated; no cost is imputed for a comparator that was not run.</p>
{_html_table(analysis.compute_display)}
<h4>Hardware and runtime environment</h4>
{hardware_html}
{_abbreviations_html()}
{_report_footer_html()}
</main></div></body></html>
"""
    return _sanitize_report_html(html_text)


def _write_report_manifest(
    sweep: Sweep,
    report_dir: Path,
    visuals: _ReportVisuals,
) -> None:
    dump_json_standard(
        {
            "report_dir": report_dir,
            "task": "classification",
            "report_template": REPORT_TEMPLATE_VERSION,
            "report_layers": list(REPORT_LAYERS),
            "target": str(_sweep_data_source(sweep).target_col),
            "figures_embedded": len(
                [
                    figure
                    for figure in [
                        visuals.representation_figure,
                        visuals.mpma_e_figure,
                    ]
                    if figure
                ]
            ),
            "explainability_targets": visuals.explainability_count,
        },
        report_dir / "report_manifest.json",
    )


def _report_outputs(
    report_dir: Path,
    tables_dir: Path,
    analysis: _ReportAnalysis,
    tables: _ReportTables,
    emit_html: bool,
) -> dict[str, Path]:
    outputs = {
        "strategy_outer_unit_metrics": analysis.statistics.get(
            "unit_metrics_path", tables_dir / "strategy_outer_unit_metrics.parquet"
        ),
        "strategy_metrics_bootstrap": analysis.statistics.get(
            "summary_path", tables_dir / "strategy_metrics_bootstrap.parquet"
        ),
        "strategy_pairwise_tests": analysis.statistics.get(
            "pairwise_path", tables_dir / "strategy_pairwise_tests.parquet"
        ),
        "strategy_statistics_manifest": analysis.statistics.get(
            "manifest_path", tables_dir / "strategy_statistics_manifest.json"
        ),
        "strategy_compute": analysis.compute_accounting.get(
            "compute_path", tables_dir / "strategy_compute.parquet"
        ),
        "compute_accounting_manifest": analysis.compute_accounting.get(
            "manifest_path", tables_dir / "compute_accounting_manifest.json"
        ),
        "important_features": tables.feature_summary_path,
        "hardware_environment": tables.hardware_summary_path,
    }
    if emit_html:
        outputs = {"html_report": report_html_path(report_dir.parent), **outputs}
    return outputs


def write_report(sweep: Sweep, emit_html: bool = True) -> dict[str, Path]:
    root = sweep.root()
    report_dir, tables_dir = _prepare_report_directories(root)
    stage("Report", str(report_dir))
    analysis = _analyze_report(sweep, root)
    visuals = _build_report_visuals(sweep, root, report_dir)
    tables = _write_report_tables(root, tables_dir, analysis)
    html_path = prepare_report_html_path(root, emit_html=emit_html)
    if emit_html:
        html_path.write_text(
            _render_report_html(sweep, analysis, visuals, tables),
            encoding="utf-8",
        )
    _write_report_manifest(sweep, report_dir, visuals)
    _print_report_summary(
        sweep,
        root,
        analysis.procedure,
        analysis.strategy_html,
        analysis.top_mpmas,
        analysis.ensemble_summary,
        analysis.ensemble_members,
        primary_pairwise=analysis.primary_pairwise,
        compute_display=analysis.compute_display,
        compute_environment=analysis.compute_environment,
    )
    success("Report completed")
    outputs = _report_outputs(report_dir, tables_dir, analysis, tables, emit_html)
    path_table("Report outputs", outputs)
    return outputs
