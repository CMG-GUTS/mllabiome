from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .figures import _resolution_order_key
from .metrics import canonical_metric_name, metric_is_loss
from .storage import read_table, remove_table, table_exists, write_table


_LINEAR_NAMES = {
    "LR_default", "LR_tuned_scaled", "LASSO_tuned", "ElasticNet_tuned", "SIAMCAT"
}
_PRETRAINED_NAMES = {
    "TabPFN", "KumoTabular_small", "KumoTabular_medium", "KumoTabular_large",
    "LimiX_2M_noretrieval",
}
_LINEAR_CLASSES = {
    "LogisticRegression", "RidgeClassifier", "RidgeClassifierCV", "Ridge",
    "RidgeCV", "ElasticNet", "ElasticNetCV", "Lasso", "LassoCV",
    "LinearRegression", "LinearSVC", "LinearSVR", "SGDClassifier",
    "SGDRegressor", "PassiveAggressiveClassifier", "PassiveAggressiveRegressor",
    "LinearDiscriminantAnalysis", "RidgeSoftmaxClassifier", "SIAMCATClassifier",
}
_PRETRAINED_CLASSES = {
    "TabPFNClassifier", "TabPFNRegressor", "KumoTabularClassifier",
    "KumoTabularRegressor", "TabICLClassifier", "TabDPTClassifier",
    "TabFMClassifier", "LimiXClassifier", "LimiX2Classifier",
    "EXAONETabularClassifier",
}
_NONLINEAR_CLASSES = {
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier",
    "ExtraTreesRegressor", "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor", "GradientBoostingClassifier",
    "GradientBoostingRegressor", "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor", "CatBoostClassifier",
    "CatBoostRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor",
    "KNeighborsClassifier", "KNeighborsRegressor", "MLPClassifier",
    "MLPRegressor", "RealMLPClassifier", "TabMClassifier", "CausiloClassifier",
    "GaussianNB", "QuadraticDiscriminantAnalysis", "BernoulliNB",
    "MultinomialNB", "ComplementNB", "NearestCentroid", "FLAMLClassifier",
}
_TRANSFORM_LABELS = {
    "identity": "Identity",
    "relative_abundance": "Relative abundance",
    "presence_absence": "Presence/absence",
    "centered_log_ratio_multiplicative_replacement": "CLR",
    "hellinger": "Hellinger",
    "arcsine_sqrt": "Arcsin √RA",
    "log10_relative_abundance_half_min_pseudocount": "Log₁₀(RA + ε)",
}
_LEARNER_ORDER = (
    "Linear models", "Non-linear models", "Pretrained tabular FM", "Other learners"
)
_RESOLUTION_ORDER = (
    "phylum", "class", "order", "family", "genus", "species", "strain", "raw"
)
_TRANSFORM_ORDER = (
    "Identity", "Relative abundance", "Presence/absence", "CLR", "Hellinger",
    "Arcsin √RA", "Log₁₀(RA + ε)", "Other transformations"
)
_REPRESENTATION_ORDER = ("Waypoint", "MGM")


def _learner_family(row: pd.Series) -> str:
    name = str(row.get("learner", "")).strip()
    token = str(row.get("learner_class", "")).strip().split(".")[-1]
    token_lower = token.casefold()
    name_lower = name.casefold()
    if name in _PRETRAINED_NAMES or token in _PRETRAINED_CLASSES or any(
        word in f"{name_lower} {token_lower}"
        for word in ("tabpfn", "kumo", "tabicl", "tabdpt", "tabfm", "limix", "exaone")
    ):
        return "Pretrained tabular FM"
    if name in _LINEAR_NAMES or token in _LINEAR_CLASSES or any(
        word in name_lower for word in ("lasso", "elasticnet", "logistic", "ridge")
    ):
        return "Linear models"
    if token in {"SVC", "SVR", "NuSVC", "NuSVR"}:
        try:
            params = json.loads(str(row.get("learner_params", "{}")))
        except (ValueError, TypeError):
            params = {}
        if isinstance(params, dict) and str(params.get("kernel", "rbf")).casefold() == "linear":
            return "Linear models"
        return "Non-linear models"
    if token in _NONLINEAR_CLASSES or any(
        word in f"{name_lower} {token_lower}" for word in (
            "randomforest", "forest", "xgboost", "lightgbm", "catboost", "boost",
            "tree", "neighbor", "knn", "mlp", "causilo", "gaussian", "naive",
        )
    ):
        return "Non-linear models"
    return "Other learners"


def _learned_representation(value: object) -> str | None:
    text = str(value).strip().casefold()
    if re.search(r"(?:^|[^a-z])waypoint(?:$|[^a-z])", text):
        return "Waypoint"
    if re.search(r"(?:^|[^a-z])mgm(?:$|[^a-z])|microformer", text):
        return "MGM"
    return None


def _transformation(value: object) -> str:
    text = str(value).strip()
    base = re.split(r"[@|]", text, maxsplit=1)[0]
    return _TRANSFORM_LABELS.get(base, "Other transformations")


def _metric_scores(root: Path, metric: str) -> pd.DataFrame:
    stages = []
    outer_metadata = pd.DataFrame()
    for label, relative in (
        ("inner", "inner_results/inner_results.parquet"),
        ("outer", "results/outer_results.parquet"),
    ):
        path = root / relative
        if not table_exists(path):
            return pd.DataFrame()
        frame = read_table(path)
        if not {"config_id", "split_key", metric}.issubset(frame.columns):
            return pd.DataFrame()
        if "ok" in frame.columns:
            frame = frame.loc[pd.to_numeric(frame["ok"], errors="coerce").eq(1)]
        frame = frame.copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame.loc[np.isfinite(frame[metric])]
        if frame.empty:
            return pd.DataFrame()
        frame["config_id"] = frame["config_id"].astype(str)
        frame["split_key"] = frame["split_key"].astype(str)
        if label == "inner" and "inner_key" in frame.columns:
            frame = frame.drop_duplicates(
                ["config_id", "split_key", "inner_key"], keep="last"
            )
        stage = frame.groupby(["config_id", "split_key"], as_index=False)[metric].mean()
        if label == "outer":
            metadata_columns = [
                field for field in (
                    "config_id", "learner", "learner_class", "learner_params",
                    "resolution", "count_transformation"
                ) if field in frame.columns
            ]
            outer_metadata = frame[metadata_columns].drop_duplicates("config_id", keep="last")
        stages.append(stage.rename(columns={metric: label}))
    matched = stages[0].merge(
        stages[1], on=["config_id", "split_key"], validate="one_to_one"
    )
    if matched.empty:
        return pd.DataFrame()
    scores = matched.groupby("config_id", as_index=False).agg(
        inner=("inner", "mean"), outer=("outer", "mean"),
        matched_outer_splits=("split_key", "nunique")
    )
    configs = read_table(root / "configs.parquet")
    if not configs.empty and "config_id" in configs:
        fields = [
            key for key in (
                "config_id", "learner", "learner_class", "learner_params",
                "resolution", "count_transformation"
            ) if key in configs.columns
        ]
        metadata = configs[fields].copy()
        metadata["config_id"] = metadata["config_id"].astype(str)
        metadata = metadata.drop_duplicates("config_id", keep="last")
        if not outer_metadata.empty:
            metadata = metadata.merge(
                outer_metadata, on="config_id", how="outer", suffixes=("", "_outer")
            )
            for field in outer_metadata.columns.difference(["config_id"]):
                outer_field = f"{field}_outer"
                if outer_field in metadata:
                    metadata[field] = metadata[field].combine_first(metadata[outer_field])
                    metadata = metadata.drop(columns=outer_field)
    else:
        metadata = outer_metadata
    if metadata.empty:
        return pd.DataFrame()
    scores = scores.merge(metadata, on="config_id", how="inner", validate="one_to_one")
    for field in ("learner", "learner_class", "learner_params", "resolution", "count_transformation"):
        if field not in scores:
            scores[field] = ""
    scores["learner_family"] = scores.apply(_learner_family, axis=1)
    scores["learned_representation"] = scores["count_transformation"].map(
        _learned_representation
    )
    scores["transformation_family"] = scores["count_transformation"].map(
        _transformation
    )
    scores["generalisation_drop"] = (
        scores["outer"] - scores["inner"] if metric_is_loss(metric)
        else scores["inner"] - scores["outer"]
    )
    return scores


def _figure_categories(frame: pd.DataFrame) -> list[tuple[str, str, str, list[str], pd.DataFrame]]:
    learners = [
        value for value in _LEARNER_ORDER if value in frame["learner_family"].values
    ]
    observed_resolutions = list(frame["resolution"].dropna().astype(str).unique())
    taxonomic = [x for x in _RESOLUTION_ORDER if x in observed_resolutions]
    other_resolutions = [
        x for x in observed_resolutions if x not in _RESOLUTION_ORDER
    ]
    resolutions = taxonomic + sorted(other_resolutions, key=_resolution_order_key)
    transformations = frame.loc[frame["learned_representation"].isna()]
    counts = [
        x for x in _TRANSFORM_ORDER
        if x in transformations["transformation_family"].values
    ]
    learned = frame.loc[frame["learned_representation"].notna()]
    reprs = [
        x for x in _REPRESENTATION_ORDER if x in learned["learned_representation"].values
    ]
    return [
        ("a", "Learner families", "learner_family", learners, frame),
        ("b", "Taxonomic resolution", "resolution", resolutions, frame),
        ("c", "Count transformations", "transformation_family", counts, transformations),
        ("d", "Learned representations", "learned_representation", reprs, learned),
    ]


def render_generalisation_figure(frame: pd.DataFrame, path: Path, metric: str) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    sections = _figure_categories(frame)
    metric_values = np.concatenate((frame["inner"].to_numpy(), frame["outer"].to_numpy()))
    if metric.upper() == "AUROC":
        xlow, xhigh = 0.45, 1.0
    elif metric.upper() in {"AUCPR", "AP", "ACCURACY", "BALACC"}:
        xlow, xhigh = min(0.45, max(0.0, float(metric_values.min()) - 0.02)), 1.0
    else:
        vmin, vmax = float(metric_values.min()), float(metric_values.max())
        pad = max((vmax - vmin) * 0.10, 0.01)
        xlow, xhigh = vmin - pad, vmax + pad

    styling = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7.5,
        'axes.labelsize': 8.3,
        'xtick.labelsize': 7.2,
        'svg.fonttype': 'none',
        'axes.facecolor': '#ffffff',
        'figure.facecolor': '#ffffff',
        'savefig.facecolor': '#ffffff'
    }

    with plt.style.context('default'), mpl.rc_context(styling):
        fig = plt.figure(figsize=(10.4, 9.55))
        grid = fig.add_gridspec(2, 2, left=0.168, right=0.972, top=0.928, bottom=0.083, wspace=0.41, hspace=0.265)
        rng = np.random.default_rng(20261025)
        palettes = [
            {'fill': '#9bd4e7', 'line': '#519ec7', 'dot': '#9bd4e7'},
            {'fill': '#2d87b1', 'line': '#145886', 'dot': '#2d87b1'}
        ]
        for idx, (letter, title, column, categories, source) in enumerate(sections):
            ax = fig.add_subplot(grid[idx // 2, idx % 2])
            count = len(categories)
            ridge_height = 0.335 if count <= 3 else 0.31
            for j, label in enumerate(categories):
                block = source.loc[source[column] == label]
                baseline = j + 1
                for stage_index, col in enumerate(['inner', 'outer']):
                    values = block[col].dropna().to_numpy(dtype=float)
                    if len(values) == 0:
                        continue
                    offset = (-0.22 if count <= 3 else -0.215) if stage_index == 0 else (0.22 if count <= 3 else 0.215)
                    level = baseline + offset
                    scheme = palettes[stage_index]
                    x = np.linspace(xlow, xhigh, 360)
                    if len(values) >= 3 and np.std(values) > 1e-8:
                        try:
                            density = gaussian_kde(values, bw_method=0.30 if len(values) < 25 else 0.23)(x)
                            density = density / max(np.max(density), 1e-12) * ridge_height
                            ax.fill_between(x, level, level - density, color=scheme['fill'], alpha=0.4, linewidth=0, zorder=2)
                            ax.plot(x, level - density, color=scheme['line'], linewidth=0.85, zorder=3)
                        except np.linalg.LinAlgError:
                            pass
                    ax.hlines(level, xlow, xhigh, color=scheme['line'], linewidth=0.65, zorder=3)
                    jitter = rng.normal(0.022, 0.024 if count <= 3 else 0.017, size=len(values))
                    size = 10.0 if len(values) > 150 else 12.3
                    ax.scatter(values, np.full_like(values, level) + jitter, s=size, color=scheme['dot'], edgecolors=scheme['line'], alpha=0.60, linewidths=0.47, zorder=5)
                ax.text(xlow - (xhigh - xlow) * (0.012 / 0.55), baseline - 0.05, label, ha='right', va='center', fontsize=7.45, color='#000000', clip_on=False)
                if idx in (0, 3):
                    ax.text(xlow - (xhigh - xlow) * (0.012 / 0.55), baseline + 0.185, f'n = {len(block)}', ha='right', va='center', fontsize=6.55, color='#596a7b', clip_on=False)
            ax.set_xlim(xlow, xhigh)
            ax.set_ylim(count + 0.49, 0.30)
            if metric.upper() == 'AUROC':
                ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax.set_xticklabels([f'{v:.1f}' for v in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
            ax.tick_params(axis='x', colors='black', length=2.6, width=0.69, pad=3.6)
            ax.set_yticks([])
            for s in ['top', 'left', 'right']:
                ax.spines[s].set_visible(False)
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(0.72)
            ax.set_xlabel('AUROC' if metric.upper() == 'AUROC' else metric.replace('_', ' '), fontsize=8.35, labelpad=7)
            x0 = 0.014 if idx % 2 == 0 else 0.524
            y0 = 0.964 if idx < 2 else 0.486
            fig.text(x0, y0, letter, fontsize=10.4, fontweight='bold', color='black', va='center')
            fig.text(x0 + 0.024, y0, title, fontsize=9.3, fontweight='bold', color='black', va='center')

        fig.lines.append(mpl.lines.Line2D([0.755, 0.775], [0.984, 0.984], transform=fig.transFigure, lw=1.15, color='#519ec7'))
        fig.text(0.780, 0.984, 'Inner validation', color='#519ec7', fontsize=7.2, va='center')
        fig.lines.append(mpl.lines.Line2D([0.884, 0.904], [0.984, 0.984], transform=fig.transFigure, lw=1.15, color='#145886'))
        fig.text(0.910, 0.984, 'Outer held-out', color='#145886', fontsize=7.2, va='center')

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220)
        plt.close(fig)
    return path


def generalisation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for _, heading, column, names, subset in _figure_categories(frame):
        for category in names:
            block = subset.loc[subset[column].astype(str).eq(category)]
            rows.append({
                "Dimension": heading,
                "Category": category,
                "Configurations": len(block),
                "Matched outer splits": int(block["matched_outer_splits"].sum()),
                "Inner mean": float(block["inner"].mean()),
                "Outer mean": float(block["outer"].mean()),
                "Inner median": float(block["inner"].median()),
                "Outer median": float(block["outer"].median()),
                "Mean generalisation drop": float(block["generalisation_drop"].mean()),
                "Median generalisation drop": float(block["generalisation_drop"].median()),
            })
    return pd.DataFrame(rows)


def generalisation_paths(root: Path, metric: str) -> tuple[Path, Path]:
    root = Path(root)
    canonical = canonical_metric_name(metric)
    suffix = "" if canonical == "AUROC" else "_" + re.sub(
        r"[^a-z0-9]+", "_", canonical.casefold()
    ).strip("_")
    filename = "inner_outer_generalisation" + suffix
    return root / "figures" / f"{filename}.svg", root / "tables" / f"{filename}.parquet"


def write_generalisation_figure(root: Path, metric: str) -> tuple[Path | None, Path | None]:
    root = Path(root)
    metric = canonical_metric_name(metric)
    figure_path, summary_path = generalisation_paths(root, metric)
    scores = _metric_scores(root, metric)
    if scores.empty:
        if figure_path.exists():
            figure_path.unlink()
        if table_exists(summary_path):
            remove_table(summary_path)
        return None, None
    render_generalisation_figure(scores, figure_path, metric)
    write_table(summary_path, generalisation_summary(scores))
    return figure_path, summary_path


def write_generalisation_figures(
    root: Path, optimisation_metric: str
) -> dict[str, tuple[Path | None, Path | None]]:
    selected = canonical_metric_name(optimisation_metric)
    root = Path(root)
    results = {"AUROC": write_generalisation_figure(root, "AUROC")}
    if selected != "AUROC":
        results[selected] = write_generalisation_figure(root, selected)
    current = {generalisation_paths(root, name)[0].stem for name in results}
    for old in (root / "figures").glob("inner_outer_generalisation_*.svg"):
        if old.stem not in current:
            old.unlink()
    for old in (root / "tables").glob("inner_outer_generalisation_*.parquet"):
        if old.stem not in current:
            remove_table(old)
    return results
