from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Ellipse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import (
    friedmanchisquare,
    kruskal,
    mannwhitneyu,
    norm,
    spearmanr,
    wilcoxon,
)

from .ancombc2_runtime import ANCOMBC_VERSION, run_ancombc2
from .explainability_visuals import _plain_taxon_label
from .data import Dataset, load_dataset
from .storage import write_table
from .style import (
    ACC,
    BG,
    C_DARK,
    C_MID,
    C_NAVY,
    C_SKY,
    CORR_CMAP,
    DIM,
    HMAP_CMAP,
    INK,
    MID,
    MM,
    TRACK,
    apply,
    save_svg,
)
from .utils import TAXONOMIC_LEVELS, dump_json_standard

try:
    import statsmodels.api as sm
except Exception:
    sm = None


_GROUP_BASE = (
    MID,
    ACC,
    "#0f766e",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be123c",
    "#4338ca",
    "#8b5e34",
    "#475569",
)
_SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "explore_signed",
    [(241 / 255, 245 / 255, 249 / 255), (2 / 255, 155 / 255, 190 / 255)],
    N=256,
)


def _group_palette(n: int) -> list[Any]:
    count = max(0, int(n))
    if count <= len(_GROUP_BASE):
        return list(_GROUP_BASE[:count])
    return [HMAP_CMAP(value) for value in np.linspace(0.92, 0.18, count)]


def _taxon_palette(n: int) -> list[Any]:
    count = max(0, int(n))
    if count == 0:
        return []
    return [HMAP_CMAP(value) for value in np.linspace(0.92, 0.18, count)]


def _clean_axis(ax: Any, *, y_grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.45)
    ax.spines["bottom"].set_linewidth(0.45)
    ax.tick_params(length=2.0, width=0.4, pad=2)
    if y_grid:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=TRACK, linewidth=0.4, alpha=0.8)


def _feature_label(value: Any) -> str:
    return _plain_taxon_label(str(value).replace(";", "|"), 48)


def _relative_abundance(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Explore abundance matrix must be two-dimensional.")
    if not np.isfinite(matrix).all():
        raise ValueError("Explore abundance matrix contains NaN or infinite values.")
    if np.any(matrix < 0):
        raise ValueError("Explore abundance matrix contains negative values.")
    totals = matrix.sum(axis=1)
    if np.any(totals <= 0):
        bad = np.flatnonzero(totals <= 0).tolist()
        raise ValueError(f"Explore found zero-total samples at positions {bad[:10]!r}.")
    return matrix / totals[:, None], totals


def _infer_scale(X: np.ndarray, totals: np.ndarray) -> str:
    values = np.asarray(X, dtype=float)
    row_totals = np.asarray(totals, dtype=float)
    positive = values[values > 0]
    integer_fraction = (
        float(np.mean(np.isclose(positive, np.rint(positive), atol=1e-8)))
        if positive.size
        else 0.0
    )
    proportion_fraction = float(
        np.mean(np.isclose(row_totals, 1.0, rtol=0.03, atol=0.03))
    )
    percentage_fraction = float(
        np.mean(np.isclose(row_totals, 100.0, rtol=0.03, atol=2.0))
    )
    median_total = float(np.median(row_totals))
    if proportion_fraction >= 0.80:
        return "proportion"
    if percentage_fraction >= 0.80:
        return "percentage"
    if integer_fraction >= 0.98 and median_total > 1.0:
        return "counts"
    return "nonnegative_abundance"


def _cluster_ids(sweep: Any, dataset: Dataset) -> tuple[np.ndarray, str]:
    data = sweep.data
    if data is None:
        return np.asarray(dataset.sample_ids, dtype=str), "sample_id"
    if getattr(data, "subject_id_col", None):
        return np.asarray(
            dataset.subject_ids, dtype=str
        ), f"subject:{data.subject_id_col}"
    group_col = getattr(data, "group_col", None)
    if group_col and group_col in dataset.metadata.columns:
        return dataset.metadata[group_col].astype(str).to_numpy(), f"group:{group_col}"
    return np.asarray(dataset.sample_ids, dtype=str), "sample_id"


def _group_labels(dataset: Dataset) -> np.ndarray:
    if dataset.task == "classification":
        return np.asarray(
            [dataset.class_labels[int(value)] for value in dataset.y], dtype=object
        )
    return np.asarray(dataset.y, dtype=float)


def _classification_levels(
    dataset: Dataset, labels: np.ndarray | None = None
) -> list[str]:
    if dataset.task != "classification":
        return []
    observed = set(
        str(value) for value in (_group_labels(dataset) if labels is None else labels)
    )
    return [str(value) for value in dataset.class_labels if str(value) in observed]


def _alpha_table(
    relative: np.ndarray,
    dataset: Dataset,
    clusters: np.ndarray,
    cluster_source: str,
    detection_limit: float,
) -> pd.DataFrame:
    present = relative > float(detection_limit)
    richness = present.sum(axis=1).astype(float)
    logs = np.zeros_like(relative, dtype=float)
    mask = relative > 0
    logs[mask] = np.log(relative[mask])
    shannon = -(relative * logs).sum(axis=1)
    simpson = 1.0 - np.square(relative).sum(axis=1)
    pielou = np.divide(
        shannon,
        np.log(richness),
        out=np.full_like(shannon, np.nan, dtype=float),
        where=richness > 1,
    )
    dominance = relative.max(axis=1)
    frame = pd.DataFrame(
        {
            "sample_id": dataset.sample_ids,
            "cluster_id": clusters,
            "cluster_source": cluster_source,
            "target": _group_labels(dataset),
            "observed_richness": richness.astype(int),
            "shannon": shannon,
            "simpson": simpson,
            "pielou_evenness": pielou,
            "dominance": dominance,
        }
    )
    return frame


def _bh_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return out
    pv = p[indices]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[indices] = restored
    return out


def _bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    paired: bool,
    replicates: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(int(replicates), dtype=float)
    if paired:
        if len(first) != len(second) or not len(first):
            return np.nan, np.nan
        n = len(first)
        for i in range(len(values)):
            idx = rng.integers(0, n, n)
            values[i] = float(np.median(second[idx] - first[idx]))
    else:
        if not len(first) or not len(second):
            return np.nan, np.nan
        for i in range(len(values)):
            a = first[rng.integers(0, len(first), len(first))]
            b = second[rng.integers(0, len(second), len(second))]
            values[i] = float(np.median(b) - np.median(a))
    alpha = (1.0 - float(confidence)) / 2.0
    return tuple(np.quantile(values, [alpha, 1.0 - alpha]).astype(float))


def _alpha_statistics(
    alpha: pd.DataFrame, dataset: Dataset, explore: Any
) -> pd.DataFrame:
    metrics = (
        "observed_richness",
        "shannon",
        "simpson",
        "pielou_evenness",
        "dominance",
    )
    if dataset.task == "regression":
        rows = []
        collapsed = alpha.groupby("cluster_id", as_index=False).agg(
            target=("target", "mean"),
            **{metric: (metric, "median") for metric in metrics},
        )
        for metric in metrics:
            valid = collapsed[["target", metric]].dropna()
            if len(valid) < 3:
                rho, p = np.nan, np.nan
            else:
                result = spearmanr(
                    valid["target"].to_numpy(float), valid[metric].to_numpy(float)
                )
                rho, p = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "metric": metric,
                    "design": "cluster_aggregated_spearman",
                    "n_clusters": int(len(valid)),
                    "effect": rho,
                    "effect_name": "spearman_rho",
                    "p_value": p,
                }
            )
        frame = pd.DataFrame(rows)
        frame["q_value"] = _bh_adjust(frame["p_value"].to_numpy(float))
        return frame
    levels = _classification_levels(
        dataset, alpha["target"].astype(str).to_numpy(object)
    )
    cluster_targets = alpha[["cluster_id", "target"]].drop_duplicates()
    constant = cluster_targets.groupby("cluster_id")["target"].nunique().max() <= 1
    rows = []
    for metric_index, metric in enumerate(metrics):
        collapsed = (
            alpha.groupby(["cluster_id", "target"], as_index=False)[metric]
            .median()
            .dropna()
        )
        groups = {
            level: collapsed.loc[
                collapsed["target"].astype(str) == level, metric
            ].to_numpy(float)
            for level in levels
        }
        row: dict[str, Any] = {
            "metric": metric,
            "n_clusters": int(collapsed["cluster_id"].nunique()),
            "n_groups": int(len(levels)),
            "group_order": " | ".join(levels),
            "p_value": np.nan,
            "effect": np.nan,
            "effect_ci_low": np.nan,
            "effect_ci_high": np.nan,
            "median_difference": np.nan,
            "median_difference_ci_low": np.nan,
            "median_difference_ci_high": np.nan,
            "effect_name": "",
            "design": "",
        }
        if len(levels) == 2 and constant:
            a, b = groups[levels[0]], groups[levels[1]]
            if len(a) >= 2 and len(b) >= 2:
                test = mannwhitneyu(a, b, alternative="two-sided")
                denom = len(a) * len(b)
                effect = 1.0 - 2.0 * float(test.statistic) / denom if denom else np.nan
                low, high = _bootstrap_difference(
                    a,
                    b,
                    False,
                    explore.bootstrap_replicates,
                    explore.confidence_level,
                    explore.random_state + metric_index,
                )
                row.update(
                    {
                        "design": "cluster_aggregated_independent",
                        "p_value": float(test.pvalue),
                        "effect": effect,
                        "effect_name": f"rank_biserial_{levels[1]}_minus_{levels[0]}",
                        "median_difference": float(np.median(b) - np.median(a)),
                        "median_difference_ci_low": low,
                        "median_difference_ci_high": high,
                    }
                )
        elif len(levels) == 2:
            pivot = collapsed.pivot(index="cluster_id", columns="target", values=metric)
            if all(level in pivot.columns for level in levels):
                paired = pivot[levels].dropna()
                if len(paired) >= 3:
                    diff = paired[levels[1]].to_numpy(float) - paired[
                        levels[0]
                    ].to_numpy(float)
                    if np.allclose(diff, 0):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
                    low, high = _bootstrap_difference(
                        paired[levels[0]].to_numpy(float),
                        paired[levels[1]].to_numpy(float),
                        True,
                        explore.bootstrap_replicates,
                        explore.confidence_level,
                        explore.random_state + metric_index,
                    )
                    row.update(
                        {
                            "design": "cluster_paired",
                            "n_clusters": int(len(paired)),
                            "p_value": p_value,
                            "effect": float(np.median(diff)),
                            "effect_name": f"paired_median_difference_{levels[1]}_minus_{levels[0]}",
                            "effect_ci_low": low,
                            "effect_ci_high": high,
                        }
                    )
        elif len(levels) > 2 and constant:
            arrays = [groups[level] for level in levels if len(groups[level])]
            if len(arrays) == len(levels) and all(
                len(values) >= 2 for values in arrays
            ):
                test = kruskal(*arrays)
                n = sum(len(values) for values in arrays)
                k = len(arrays)
                epsilon = max(0.0, (float(test.statistic) - k + 1.0) / max(1.0, n - k))
                row.update(
                    {
                        "design": "cluster_aggregated_independent",
                        "p_value": float(test.pvalue),
                        "effect": epsilon,
                        "effect_name": "epsilon_squared",
                    }
                )
        elif len(levels) > 2:
            pivot = collapsed.pivot(index="cluster_id", columns="target", values=metric)
            if all(level in pivot.columns for level in levels):
                complete = pivot[levels].dropna()
                if len(complete) >= 3:
                    test = friedmanchisquare(
                        *(complete[level].to_numpy(float) for level in levels)
                    )
                    kendall_w = float(test.statistic) / (
                        len(complete) * max(1, len(levels) - 1)
                    )
                    row.update(
                        {
                            "design": "cluster_repeated",
                            "n_clusters": int(len(complete)),
                            "p_value": float(test.pvalue),
                            "effect": kendall_w,
                            "effect_name": "kendall_w",
                        }
                    )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["q_value"] = _bh_adjust(frame["p_value"].to_numpy(float))
    return frame


def _adaptive_zero_replacement(relative: np.ndarray, configured: float | None) -> float:
    matrix = np.asarray(relative, dtype=float)
    zeros_per_sample = np.sum(matrix == 0, axis=1)
    max_zeros = int(np.max(zeros_per_sample)) if len(zeros_per_sample) else 0
    if max_zeros == 0:
        return 0.0
    if configured is not None:
        value = float(configured)
    else:
        positive = matrix[matrix > 0]
        if not positive.size:
            raise ValueError(
                "Explore cannot determine zero replacement from an all-zero composition."
            )
        value = float(np.min(positive)) * 0.5
    upper = 0.5 / max_zeros
    return min(value, upper)


def _clr_matrix(relative: np.ndarray, zero_replacement: float) -> np.ndarray:
    composition = np.asarray(relative, dtype=float).copy()
    delta = float(zero_replacement)
    for row_index in range(composition.shape[0]):
        row = composition[row_index]
        zero = row == 0
        count = int(np.sum(zero))
        if count == 0:
            continue
        mass = count * delta
        if not 0.0 < mass < 1.0:
            raise ValueError(
                "Explore zero replacement is incompatible with the observed sparsity."
            )
        row[zero] = delta
        row[~zero] *= 1.0 - mass
    logged = np.log(composition)
    return logged - logged.mean(axis=1, keepdims=True)


def _distance_matrices(
    relative: np.ndarray, clr: np.ndarray, detection_limit: float
) -> dict[str, np.ndarray]:
    presence = relative > float(detection_limit)
    return {
        "aitchison": squareform(pdist(clr, metric="euclidean")),
        "bray_curtis": squareform(pdist(relative, metric="braycurtis")),
        "jaccard": squareform(pdist(presence.astype(bool), metric="jaccard")),
    }


def _pcoa_with_correction(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    D = np.asarray(distance, dtype=float)
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n), dtype=float) / n
    D2 = np.square(D)
    B = -0.5 * H @ D2 @ H
    eigenvalues, eigenvectors = np.linalg.eigh((B + B.T) / 2.0)
    scale = max(
        np.finfo(float).eps,
        float(np.max(np.abs(eigenvalues))) if len(eigenvalues) else 0.0,
    )
    correction = "none"
    if len(eigenvalues) and float(np.min(eigenvalues)) < -1e-10 * scale:
        constant = -float(np.min(eigenvalues))
        corrected = D2 + 2.0 * constant
        np.fill_diagonal(corrected, 0.0)
        B = -0.5 * H @ corrected @ H
        eigenvalues, eigenvectors = np.linalg.eigh((B + B.T) / 2.0)
        correction = "lingoes"
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    threshold = max(
        np.finfo(float).eps, abs(eigenvalues[0]) * 1e-12 if len(eigenvalues) else 0.0
    )
    positive = eigenvalues > threshold
    vals = eigenvalues[positive]
    vecs = eigenvectors[:, positive]
    coordinates = (
        vecs * np.sqrt(vals)[None, :] if len(vals) else np.zeros((n, 0), dtype=float)
    )
    explained = (
        vals / vals.sum()
        if len(vals) and vals.sum() > 0
        else np.zeros(len(vals), dtype=float)
    )
    return coordinates, explained, correction


def _pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates, explained, _ = _pcoa_with_correction(distance)
    return coordinates, explained


def _permanova_stat(distance: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    D2 = np.square(np.asarray(distance, dtype=float))
    n = D2.shape[0]
    levels = np.unique(labels)
    if n < 3 or len(levels) < 2 or len(levels) >= n:
        return np.nan, np.nan
    ss_total = float(np.triu(D2, 1).sum() / n)
    ss_within = 0.0
    for level in levels:
        idx = np.flatnonzero(labels == level)
        if not len(idx):
            continue
        block = D2[np.ix_(idx, idx)]
        ss_within += float(np.triu(block, 1).sum() / len(idx))
    ss_between = max(0.0, ss_total - ss_within)
    df_between = len(levels) - 1
    df_within = n - len(levels)
    if df_between <= 0 or df_within <= 0 or ss_within <= 0:
        return np.nan, ss_between / ss_total if ss_total > 0 else np.nan
    pseudo_f = (ss_between / df_between) / (ss_within / df_within)
    return float(pseudo_f), float(ss_between / ss_total) if ss_total > 0 else np.nan


def _anova_stat(values: np.ndarray, labels: np.ndarray) -> float:
    levels = np.unique(labels)
    n = len(values)
    if n < 3 or len(levels) < 2 or len(levels) >= n:
        return np.nan
    grand = float(np.mean(values))
    between = 0.0
    within = 0.0
    for level in levels:
        group = values[labels == level]
        if not len(group):
            continue
        mean = float(np.mean(group))
        between += len(group) * (mean - grand) ** 2
        within += float(np.sum(np.square(group - mean)))
    if within <= 0:
        return np.nan
    return float((between / (len(levels) - 1)) / (within / (n - len(levels))))


def _dispersion_stat_coordinates(coordinates: np.ndarray, labels: np.ndarray) -> float:
    if coordinates.shape[1] == 0:
        return np.nan
    dist = np.zeros(len(labels), dtype=float)
    for level in np.unique(labels):
        idx = np.flatnonzero(labels == level)
        centroid = coordinates[idx].mean(axis=0)
        dist[idx] = np.linalg.norm(coordinates[idx] - centroid, axis=1)
    return _anova_stat(dist, labels)


def _permuted_labels(
    labels: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    clusters = np.asarray(clusters, dtype=object)
    frame = pd.DataFrame({"cluster": clusters, "label": labels})
    constant = frame.groupby("cluster")["label"].nunique().max() <= 1
    if constant:
        unique = frame.drop_duplicates("cluster")
        cluster_values = unique["cluster"].to_numpy(object)
        cluster_labels = unique["label"].to_numpy(object)
        shuffled = rng.permutation(cluster_labels)
        mapping = dict(zip(cluster_values, shuffled))
        return np.asarray([mapping[value] for value in clusters], dtype=object)
    out = labels.copy()
    for cluster in pd.unique(clusters):
        idx = np.flatnonzero(clusters == cluster)
        out[idx] = rng.permutation(out[idx])
    return out


def _cluster_bootstrap_permanova_r2(
    distance: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    replicates: int,
    confidence_level: float,
    random_state: int,
) -> tuple[float, float]:
    unique = pd.unique(clusters)
    if len(unique) < 2:
        return np.nan, np.nan
    members = {value: np.flatnonzero(clusters == value) for value in unique}
    rng = np.random.default_rng(int(random_state))
    values: list[float] = []
    for _ in range(int(replicates)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[value] for value in sampled])
        _, r2 = _permanova_stat(distance[np.ix_(indices, indices)], labels[indices])
        if np.isfinite(r2):
            values.append(float(r2))
    if not values:
        return np.nan, np.nan
    alpha = (1.0 - float(confidence_level)) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _beta_statistics(
    distances: dict[str, np.ndarray],
    labels: np.ndarray,
    clusters: np.ndarray,
    explore: Any,
) -> pd.DataFrame:
    if labels.dtype.kind in {"f", "i", "u"}:
        return pd.DataFrame(
            columns=[
                "distance",
                "permanova_pseudo_f",
                "permanova_r2",
                "permanova_p_value",
                "permdisp_f",
                "permdisp_p_value",
                "permutations",
                "permutation_scheme",
            ]
        )
    frame = pd.DataFrame({"cluster": clusters, "label": labels})
    constant = frame.groupby("cluster")["label"].nunique().max() <= 1
    scheme = "between_cluster" if constant else "within_cluster"
    rng = np.random.default_rng(explore.random_state)
    permutations = [
        _permuted_labels(labels, clusters, rng)
        for _ in range(int(explore.permutations))
    ]
    rows = []
    for name, distance in distances.items():
        observed_f, observed_r2 = _permanova_stat(distance, labels)
        coordinates, _, correction = _pcoa_with_correction(distance)
        observed_disp = _dispersion_stat_coordinates(coordinates, labels)
        perm_f = np.asarray(
            [_permanova_stat(distance, perm)[0] for perm in permutations], dtype=float
        )
        perm_disp = np.asarray(
            [_dispersion_stat_coordinates(coordinates, perm) for perm in permutations],
            dtype=float,
        )
        valid_f = perm_f[np.isfinite(perm_f)]
        valid_disp = perm_disp[np.isfinite(perm_disp)]
        p_f = (
            (1.0 + float(np.sum(valid_f >= observed_f))) / (1.0 + len(valid_f))
            if np.isfinite(observed_f) and len(valid_f)
            else np.nan
        )
        p_disp = (
            (1.0 + float(np.sum(valid_disp >= observed_disp))) / (1.0 + len(valid_disp))
            if np.isfinite(observed_disp) and len(valid_disp)
            else np.nan
        )
        ci_low, ci_high = _cluster_bootstrap_permanova_r2(
            distance,
            labels,
            clusters,
            int(explore.bootstrap_replicates),
            float(explore.confidence_level),
            int(explore.random_state) + len(rows) * 997,
        )
        rows.append(
            {
                "distance": name,
                "permanova_pseudo_f": observed_f,
                "permanova_r2": observed_r2,
                "permanova_r2_ci_low": ci_low,
                "permanova_r2_ci_high": ci_high,
                "permanova_p_value": p_f,
                "permdisp_f": observed_disp,
                "permdisp_p_value": p_disp,
                "permutations": int(explore.permutations),
                "permutation_scheme": scheme,
                "permdisp_pcoa_correction": correction,
                "n_samples": int(len(labels)),
                "n_clusters": int(pd.Series(clusters).nunique()),
            }
        )
    result = pd.DataFrame(rows)
    result["permanova_q_value"] = _bh_adjust(
        result["permanova_p_value"].to_numpy(float)
    )
    result["permdisp_q_value"] = _bh_adjust(result["permdisp_p_value"].to_numpy(float))
    return result


def _taxon_summary(
    relative: np.ndarray, names: list[str], detection_limit: float
) -> pd.DataFrame:
    prevalence = np.mean(relative > float(detection_limit), axis=0)
    mean = np.mean(relative, axis=0)
    median = np.median(relative, axis=0)
    maximum = np.max(relative, axis=0)
    frame = pd.DataFrame(
        {
            "feature": names,
            "display_feature": [_feature_label(name) for name in names],
            "prevalence": prevalence,
            "mean_relative_abundance": mean,
            "median_relative_abundance": median,
            "max_relative_abundance": maximum,
        }
    )
    return frame.sort_values(
        ["mean_relative_abundance", "prevalence"], ascending=False
    ).reset_index(drop=True)


def _da_covariates(
    dataset: Dataset, columns: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    metadata = dataset.metadata.reset_index(drop=True)
    mask = np.ones(len(metadata), dtype=bool)
    parts: list[np.ndarray] = []
    names: list[str] = []
    for column in columns:
        if column not in metadata.columns:
            raise ValueError(
                f"Differential-abundance covariate {column!r} is not present in metadata."
            )
        series = metadata[column]
        missing = series.isna().to_numpy()
        if series.dtype == object:
            missing |= series.astype(str).str.strip().eq("").to_numpy()
        mask &= ~missing
    for column in columns:
        series = metadata.loc[mask, column]
        numeric = pd.to_numeric(series, errors="coerce")
        if int(numeric.notna().sum()) == len(series):
            values = numeric.to_numpy(float)
            sd = float(np.std(values, ddof=0))
            if np.isfinite(sd) and sd > 0:
                values = (values - float(np.mean(values))) / sd
                parts.append(values[:, None])
                names.append(str(column))
            continue
        dummies = pd.get_dummies(
            series.astype(str), prefix=str(column), drop_first=True, dtype=float
        )
        if dummies.shape[1]:
            parts.append(dummies.to_numpy(float))
            names.extend(str(value) for value in dummies.columns)
    matrix = (
        np.column_stack(parts) if parts else np.empty((int(mask.sum()), 0), dtype=float)
    )
    return mask, matrix, names


def _fit_clr_model(
    response: np.ndarray, design: np.ndarray, clusters: np.ndarray
) -> tuple[Any, str]:
    if sm is None:
        raise RuntimeError(
            "Differential association on relative-abundance data requires statsmodels."
        )
    model = sm.OLS(np.asarray(response, dtype=float), np.asarray(design, dtype=float))
    groups = np.asarray(clusters, dtype=str)
    n_clusters = int(pd.Series(groups).nunique())
    if n_clusters < len(groups) and n_clusters >= 3:
        return model.fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
            use_t=False,
        ), "cluster-robust"
    return model.fit(cov_type="HC3", use_t=False), "HC3-robust"


def _contrast_statistics(
    fit: Any, vector: np.ndarray
) -> tuple[float, float, float, float]:
    beta = np.asarray(fit.params, dtype=float)
    covariance = np.asarray(fit.cov_params(), dtype=float)
    contrast = np.asarray(vector, dtype=float)
    effect = float(contrast @ beta)
    variance = float(contrast @ covariance @ contrast)
    standard_error = math.sqrt(max(0.0, variance)) if np.isfinite(variance) else np.nan
    statistic = (
        effect / standard_error
        if np.isfinite(standard_error) and standard_error > 0
        else np.nan
    )
    p_value = float(2.0 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return effect, standard_error, statistic, p_value


def _adjust_da_results(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["q_value"] = np.nan
    for _, index in out.groupby("contrast", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        out.loc[positions, "q_value"] = _bh_adjust(
            out.loc[positions, "p_value"].to_numpy(float)
        )
    out["significant"] = pd.to_numeric(out["q_value"], errors="coerce") < float(alpha)
    out["sensitivity_passed"] = np.nan
    out["robust_significant"] = out["significant"]
    return out.sort_values(
        ["contrast", "q_value", "p_value", "prevalence"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def _clr_differential_association(
    clr: np.ndarray,
    relative: np.ndarray,
    names: list[str],
    dataset: Dataset,
    clusters: np.ndarray,
    explore: Any,
) -> dict[str, Any]:
    prevalence = np.mean(relative > float(explore.detection_limit), axis=0)
    keep = np.flatnonzero(prevalence >= float(explore.min_prevalence))
    cov_mask, covariates, covariate_names = _da_covariates(
        dataset, tuple(explore.da_covariates)
    )
    labels = _group_labels(dataset)
    mask = cov_mask & pd.Series(labels).notna().to_numpy()
    if dataset.task == "regression":
        mask &= np.isfinite(np.asarray(labels, dtype=float))
    kept_rows = np.flatnonzero(mask)
    if len(kept_rows) < 4 or not len(keep):
        return {
            "method": "CLR linear model",
            "results": pd.DataFrame(),
            "global": pd.DataFrame(),
            "pairwise": pd.DataFrame(),
            "model": "",
            "status": "insufficient_data",
        }
    covariates = (
        covariates[np.searchsorted(np.flatnonzero(cov_mask), kept_rows)]
        if covariates.shape[1]
        else np.empty((len(kept_rows), 0), dtype=float)
    )
    cluster_values = np.asarray(clusters, dtype=str)[mask]
    rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    if dataset.task == "classification":
        observed = np.asarray(labels, dtype=object)[mask].astype(str)
        levels = _classification_levels(dataset, observed)
        if len(levels) < 2:
            return {
                "method": "CLR linear model",
                "results": pd.DataFrame(),
                "global": pd.DataFrame(),
                "pairwise": pd.DataFrame(),
                "model": "",
                "status": "insufficient_classes",
            }
        target = np.column_stack(
            [(observed == level).astype(float) for level in levels[1:]]
        )
        design = np.column_stack([np.ones(len(observed)), target, covariates])
        target_offset = 1
        model_text = "CLR linear model; " + (
            "cluster-robust standard errors for repeated/dependent observations"
            if int(pd.Series(cluster_values).nunique()) < len(cluster_values)
            else "HC3 heteroskedasticity-robust standard errors"
        )
        if covariate_names:
            model_text += "; adjusted for " + ", ".join(
                str(value) for value in explore.da_covariates
            )
        for feature_index in keep:
            fit, inference = _fit_clr_model(
                clr[mask, feature_index], design, cluster_values
            )
            covariance = np.asarray(fit.cov_params(), dtype=float)
            if len(levels) > 2:
                restriction = np.zeros((len(levels) - 1, design.shape[1]), dtype=float)
                for i in range(len(levels) - 1):
                    restriction[i, target_offset + i] = 1.0
                test = fit.wald_test(restriction, scalar=True)
                global_rows.append(
                    {
                        "feature": names[feature_index],
                        "display_feature": _feature_label(names[feature_index]),
                        "prevalence": float(prevalence[feature_index]),
                        "method": "CLR linear model",
                        "statistic": float(np.asarray(test.statistic).reshape(-1)[0]),
                        "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
                        "n_samples": int(mask.sum()),
                        "n_clusters": int(pd.Series(cluster_values).nunique()),
                        "inference": inference,
                        "model": model_text,
                    }
                )
            for a_index in range(len(levels) - 1):
                for b_index in range(a_index + 1, len(levels)):
                    vector = np.zeros(design.shape[1], dtype=float)
                    if a_index > 0:
                        vector[target_offset + a_index - 1] -= 1.0
                    if b_index > 0:
                        vector[target_offset + b_index - 1] += 1.0
                    effect, standard_error, statistic, p_value = _contrast_statistics(
                        fit, vector
                    )
                    row = {
                        "feature": names[feature_index],
                        "display_feature": _feature_label(names[feature_index]),
                        "prevalence": float(prevalence[feature_index]),
                        "method": "CLR linear model",
                        "contrast": f"{levels[b_index]} − {levels[a_index]}",
                        "effect": effect,
                        "effect_scale": "CLR coefficient",
                        "standard_error": standard_error,
                        "statistic": statistic,
                        "p_value": p_value,
                        "n_samples": int(mask.sum()),
                        "n_clusters": int(pd.Series(cluster_values).nunique()),
                        "inference": inference,
                        "model": model_text,
                    }
                    rows.append(row)
                    pairwise_rows.append(dict(row))
    else:
        target_values = np.asarray(labels, dtype=float)[mask]
        design = np.column_stack(
            [np.ones(len(target_values)), target_values, covariates]
        )
        model_text = "CLR linear model; " + (
            "cluster-robust standard errors for repeated/dependent observations"
            if int(pd.Series(cluster_values).nunique()) < len(cluster_values)
            else "HC3 heteroskedasticity-robust standard errors"
        )
        if covariate_names:
            model_text += "; adjusted for " + ", ".join(
                str(value) for value in explore.da_covariates
            )
        for feature_index in keep:
            fit, inference = _fit_clr_model(
                clr[mask, feature_index], design, cluster_values
            )
            vector = np.zeros(design.shape[1], dtype=float)
            vector[1] = 1.0
            effect, standard_error, statistic, p_value = _contrast_statistics(
                fit, vector
            )
            rows.append(
                {
                    "feature": names[feature_index],
                    "display_feature": _feature_label(names[feature_index]),
                    "prevalence": float(prevalence[feature_index]),
                    "method": "CLR linear model",
                    "contrast": str(dataset.target_name),
                    "effect": effect,
                    "effect_scale": "CLR coefficient per target unit",
                    "standard_error": standard_error,
                    "statistic": statistic,
                    "p_value": p_value,
                    "n_samples": int(mask.sum()),
                    "n_clusters": int(pd.Series(cluster_values).nunique()),
                    "inference": inference,
                    "model": model_text,
                }
            )
    results = (
        _adjust_da_results(pd.DataFrame(rows), float(explore.da_alpha))
        if rows
        else pd.DataFrame()
    )
    pairwise = (
        _adjust_da_results(pd.DataFrame(pairwise_rows), float(explore.da_alpha))
        if pairwise_rows
        else pd.DataFrame()
    )
    global_frame = pd.DataFrame(global_rows)
    if not global_frame.empty:
        global_frame["q_value"] = _bh_adjust(global_frame["p_value"].to_numpy(float))
        global_frame["significant"] = global_frame["q_value"] < float(explore.da_alpha)
        global_frame = global_frame.sort_values(
            ["q_value", "p_value"], na_position="last"
        ).reset_index(drop=True)
    return {
        "method": "CLR linear model",
        "results": results,
        "global": global_frame,
        "pairwise": pairwise,
        "model": model_text,
        "status": "ok",
        "multiple_testing": "Benjamini–Hochberg within contrast",
        "runtime": None,
        "structural_zeros": pd.DataFrame(),
        "sensitivity": pd.DataFrame(),
    }


def _r_name(value: str) -> str:
    text = str(value)
    text = "".join(
        character if character.isalnum() or character in "._" else "."
        for character in text
    )
    if (
        not text
        or text[0].isdigit()
        or (text[0] == "." and len(text) > 1 and text[1].isdigit())
    ):
        text = "X" + text
    return text


def _ancombc_contrast(term: str, levels: list[str]) -> str:
    if len(levels) == 2:
        return f"{levels[1]} − {levels[0]}"
    suffix = term.removeprefix("mll_target")
    safe = {_r_name(level): level for level in levels}
    if suffix in safe:
        return f"{safe[suffix]} − {levels[0]}"
    for left_key, left in safe.items():
        for right_key, right in safe.items():
            if left == right:
                continue
            if suffix == f"{left_key}_mll_target{right_key}":
                return f"{left} − {right}"
    return suffix.replace("mll_target", "").replace("_", " − ")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", ""}:
        return False
    return bool(default)


def _ancombc_primary_table(
    raw: pd.DataFrame,
    relative: np.ndarray,
    names: list[str],
    dataset: Dataset,
    explore: Any,
    n_samples: int,
    n_clusters: int,
    model_text: str,
) -> pd.DataFrame:
    if raw.empty or "taxon" not in raw.columns:
        return pd.DataFrame()
    prevalence_map = dict(
        zip(names, np.mean(relative > float(explore.detection_limit), axis=0))
    )
    levels = _classification_levels(dataset) if dataset.task == "classification" else []
    lfc_columns = [column for column in raw.columns if str(column).startswith("lfc_")]
    terms = [str(column)[4:] for column in lfc_columns]
    if dataset.task == "regression":
        terms = [term for term in terms if term == "mll_target"]
    else:
        terms = [term for term in terms if term.startswith("mll_target")]
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        feature = str(row["taxon"])
        for term in terms:
            effect = pd.to_numeric(
                pd.Series([row.get(f"lfc_{term}")]), errors="coerce"
            ).iloc[0]
            if not np.isfinite(effect):
                continue
            se = pd.to_numeric(
                pd.Series([row.get(f"se_{term}")]), errors="coerce"
            ).iloc[0]
            statistic = pd.to_numeric(
                pd.Series([row.get(f"W_{term}")]), errors="coerce"
            ).iloc[0]
            p_value = pd.to_numeric(
                pd.Series([row.get(f"p_{term}")]), errors="coerce"
            ).iloc[0]
            q_value = pd.to_numeric(
                pd.Series([row.get(f"q_{term}")]), errors="coerce"
            ).iloc[0]
            passed = row.get(f"passed_ss_{term}", np.nan)
            significant = _as_bool(
                row.get(f"diff_{term}"),
                bool(np.isfinite(q_value) and q_value < float(explore.da_alpha)),
            )
            robust = row.get(f"diff_robust_{term}", np.nan)
            if pd.isna(robust):
                robust = bool(
                    significant
                    and (_as_bool(passed, True) if pd.notna(passed) else True)
                )
            rows.append(
                {
                    "feature": feature,
                    "display_feature": _feature_label(feature),
                    "prevalence": float(prevalence_map.get(feature, np.nan)),
                    "method": "ANCOM-BC2",
                    "contrast": _ancombc_contrast(term, levels)
                    if levels
                    else str(dataset.target_name),
                    "effect": float(effect),
                    "effect_scale": "Bias-corrected log fold change"
                    if levels
                    else "Bias-corrected log-abundance coefficient",
                    "standard_error": float(se) if np.isfinite(se) else np.nan,
                    "statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "q_value": float(q_value) if np.isfinite(q_value) else np.nan,
                    "significant": bool(significant),
                    "sensitivity_passed": _as_bool(passed)
                    if pd.notna(passed)
                    else np.nan,
                    "robust_significant": _as_bool(robust),
                    "n_samples": int(n_samples),
                    "n_clusters": int(n_clusters),
                    "inference": "ANCOM-BC2",
                    "model": model_text,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["contrast", "q_value", "p_value", "prevalence"],
            ascending=[True, True, True, False],
            na_position="last",
        )
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def _ancombc_pairwise_table(
    raw: pd.DataFrame,
    relative: np.ndarray,
    names: list[str],
    dataset: Dataset,
    explore: Any,
    n_samples: int,
    n_clusters: int,
    model_text: str,
) -> pd.DataFrame:
    if raw.empty or "taxon" not in raw.columns:
        return pd.DataFrame()
    prevalence_map = dict(
        zip(names, np.mean(relative > float(explore.detection_limit), axis=0))
    )
    levels = _classification_levels(dataset)
    rows: list[dict[str, Any]] = []
    for column in [
        value for value in raw.columns if str(value).startswith("lfc_mll_target")
    ]:
        term = str(column)[4:]
        contrast = _ancombc_contrast(term, levels)
        for _, row in raw.iterrows():
            effect = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[
                0
            ]
            if not np.isfinite(effect):
                continue
            feature = str(row["taxon"])
            q_value = pd.to_numeric(
                pd.Series([row.get(f"q_{term}")]), errors="coerce"
            ).iloc[0]
            passed = row.get(f"passed_ss_{term}", np.nan)
            significant = _as_bool(
                row.get(f"diff_{term}"),
                bool(np.isfinite(q_value) and q_value < float(explore.da_alpha)),
            )
            robust = row.get(f"diff_robust_{term}", np.nan)
            if pd.isna(robust):
                robust = bool(
                    significant
                    and (_as_bool(passed, True) if pd.notna(passed) else True)
                )
            rows.append(
                {
                    "feature": feature,
                    "display_feature": _feature_label(feature),
                    "prevalence": float(prevalence_map.get(feature, np.nan)),
                    "method": "ANCOM-BC2",
                    "contrast": contrast,
                    "effect": float(effect),
                    "effect_scale": "Bias-corrected log fold change",
                    "standard_error": pd.to_numeric(
                        pd.Series([row.get(f"se_{term}")]), errors="coerce"
                    ).iloc[0],
                    "statistic": pd.to_numeric(
                        pd.Series([row.get(f"W_{term}")]), errors="coerce"
                    ).iloc[0],
                    "p_value": pd.to_numeric(
                        pd.Series([row.get(f"p_{term}")]), errors="coerce"
                    ).iloc[0],
                    "q_value": float(q_value) if np.isfinite(q_value) else np.nan,
                    "significant": bool(significant),
                    "sensitivity_passed": _as_bool(passed)
                    if pd.notna(passed)
                    else np.nan,
                    "robust_significant": _as_bool(robust),
                    "n_samples": int(n_samples),
                    "n_clusters": int(n_clusters),
                    "inference": "ANCOM-BC2 mdFDR pairwise",
                    "model": model_text,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["contrast", "q_value", "p_value", "prevalence"],
            ascending=[True, True, True, False],
            na_position="last",
        )
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def _ancombc_global_table(
    raw: pd.DataFrame,
    names: list[str],
    relative: np.ndarray,
    explore: Any,
    n_samples: int,
    n_clusters: int,
    model_text: str,
) -> pd.DataFrame:
    if raw.empty or "taxon" not in raw.columns:
        return pd.DataFrame()
    prevalence_map = dict(
        zip(names, np.mean(relative > float(explore.detection_limit), axis=0))
    )
    frame = pd.DataFrame(
        {
            "feature": raw["taxon"].astype(str),
            "display_feature": raw["taxon"].astype(str).map(_feature_label),
            "prevalence": raw["taxon"].astype(str).map(prevalence_map),
            "method": "ANCOM-BC2 global test",
            "statistic": pd.to_numeric(raw.get("W"), errors="coerce"),
            "p_value": pd.to_numeric(raw.get("p_val"), errors="coerce"),
            "q_value": pd.to_numeric(raw.get("q_val"), errors="coerce"),
            "significant": raw.get("diff_abn", False),
            "sensitivity_passed": raw.get("passed_ss", np.nan),
            "robust_significant": raw.get(
                "diff_robust_abn", raw.get("diff_abn", False)
            ),
            "n_samples": int(n_samples),
            "n_clusters": int(n_clusters),
            "model": model_text,
        }
    )
    for column in ("significant", "sensitivity_passed", "robust_significant"):
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: _as_bool(value) if pd.notna(value) else np.nan
            )
    return frame.sort_values(["q_value", "p_value"], na_position="last").reset_index(
        drop=True
    )


def _ancombc2_differential_abundance(
    base_counts: np.ndarray,
    base_names: list[str],
    rank_counts: np.ndarray,
    relative: np.ndarray,
    names: list[str],
    dataset: Dataset,
    clusters: np.ndarray,
    explore: Any,
) -> dict[str, Any]:
    metadata = dataset.metadata.reset_index(drop=True)
    covariates = tuple(explore.da_covariates)
    for column in covariates:
        if column not in metadata.columns:
            raise ValueError(
                f"Differential-abundance covariate {column!r} is not present in metadata."
            )
    mask = np.ones(len(dataset.sample_ids), dtype=bool)
    for column in covariates:
        series = metadata[column]
        mask &= series.notna().to_numpy()
        if series.dtype == object:
            mask &= ~series.astype(str).str.strip().eq("").to_numpy()
    labels = _group_labels(dataset)
    if dataset.task == "regression":
        mask &= np.isfinite(np.asarray(labels, dtype=float))
    sample_ids = [str(value) for value, keep in zip(dataset.sample_ids, mask) if keep]
    meta = pd.DataFrame(index=sample_ids)
    if dataset.task == "classification":
        target_values = np.asarray(labels, dtype=object)[mask].astype(str)
        levels = _classification_levels(dataset, target_values)
        meta["mll_target"] = target_values
    else:
        levels = []
        meta["mll_target"] = np.asarray(labels, dtype=float)[mask]
    for index, column in enumerate(covariates, start=1):
        meta[f"mll_cov_{index}"] = metadata.loc[mask, column].to_numpy()
    cluster_values = np.asarray(clusters, dtype=str)[mask]
    repeated = int(pd.Series(cluster_values).nunique()) < len(cluster_values)
    if repeated:
        meta["mll_cluster"] = cluster_values
    fix_terms = [
        "mll_target",
        *[f"mll_cov_{index}" for index in range(1, len(covariates) + 1)],
    ]
    fix_formula = " + ".join(fix_terms)
    rand_formula = "(1 | mll_cluster)" if repeated else None
    model_text = f"ANCOM-BC2 {ANCOMBC_VERSION}; fixed effects: " + ", ".join(
        [str(dataset.target_name), *covariates]
    )
    if repeated:
        model_text += "; random intercept: dependence cluster"
    neg_lb = explore.ancombc2_neg_lb
    if neg_lb is None and dataset.task == "classification":
        counts = pd.Series(
            np.asarray(labels, dtype=object)[mask].astype(str)
        ).value_counts()
        neg_lb = bool(len(counts) and int(counts.min()) > 30)
    neg_lb = bool(neg_lb) if neg_lb is not None else False
    runtime_output = run_ancombc2(
        np.asarray(base_counts, dtype=float)[mask],
        sample_ids,
        base_names,
        meta,
        aggregate_counts=np.asarray(rank_counts, dtype=float)[mask],
        aggregate_feature_names=names,
        fix_formula=fix_formula,
        rand_formula=rand_formula,
        group="mll_target" if dataset.task == "classification" else None,
        classification_levels=levels,
        p_adjust_method=str(explore.ancombc2_p_adjust_method),
        pseudo_sens=bool(explore.ancombc2_pseudo_sens),
        prevalence_cutoff=float(explore.min_prevalence),
        alpha=float(explore.da_alpha),
        structural_zeros=bool(
            explore.ancombc2_structural_zeros and dataset.task == "classification"
        ),
        neg_lb=neg_lb,
        global_test=bool(dataset.task == "classification" and len(levels) > 2),
        pairwise=bool(dataset.task == "classification" and len(levels) > 2),
        workers=int(explore.ancombc2_workers),
        random_state=int(explore.random_state),
        runtime=str(explore.ancombc2_runtime),
    )
    n_samples = int(mask.sum())
    n_clusters = int(pd.Series(cluster_values).nunique())
    primary = _ancombc_primary_table(
        runtime_output["primary"],
        relative,
        names,
        dataset,
        explore,
        n_samples,
        n_clusters,
        model_text,
    )
    pairwise = _ancombc_pairwise_table(
        runtime_output["pairwise"],
        relative,
        names,
        dataset,
        explore,
        n_samples,
        n_clusters,
        model_text,
    )
    global_frame = _ancombc_global_table(
        runtime_output["global"],
        names,
        relative,
        explore,
        n_samples,
        n_clusters,
        model_text,
    )
    results = (
        pairwise
        if dataset.task == "classification" and len(levels) > 2 and not pairwise.empty
        else primary
    )
    runtime = runtime_output["runtime"]
    return {
        "method": "ANCOM-BC2",
        "results": results,
        "primary": primary,
        "global": global_frame,
        "pairwise": pairwise,
        "structural_zeros": runtime_output["structural_zeros"],
        "sensitivity": runtime_output["sensitivity"],
        "model": model_text,
        "status": "ok",
        "multiple_testing": str(explore.ancombc2_p_adjust_method),
        "runtime": {
            "R": runtime.r_version,
            "Bioconductor": runtime.bioconductor_version,
            "ANCOMBC": runtime.ancombc_version,
        },
    }


def _differential_abundance(
    scale: str,
    base_counts: np.ndarray,
    base_names: list[str],
    rank_counts: np.ndarray,
    relative: np.ndarray,
    clr: np.ndarray,
    names: list[str],
    dataset: Dataset,
    clusters: np.ndarray,
    explore: Any,
) -> dict[str, Any]:
    backend = str(explore.differential_abundance)
    if backend == "off":
        return {
            "method": "off",
            "results": pd.DataFrame(),
            "global": pd.DataFrame(),
            "pairwise": pd.DataFrame(),
            "structural_zeros": pd.DataFrame(),
            "sensitivity": pd.DataFrame(),
            "model": "",
            "status": "disabled",
            "runtime": None,
        }
    if backend == "auto":
        backend = "ancombc2" if scale == "counts" else "clr"
    if backend == "ancombc2":
        if scale != "counts":
            raise ValueError(
                "ANCOM-BC2 requires microbial count data. Relative-abundance profiles must use differential_abundance='clr' or 'auto'."
            )
        return _ancombc2_differential_abundance(
            base_counts,
            base_names,
            rank_counts,
            relative,
            names,
            dataset,
            clusters,
            explore,
        )
    return _clr_differential_association(
        clr, relative, names, dataset, clusters, explore
    )


def _sample_qc(
    X: np.ndarray,
    relative: np.ndarray,
    totals: np.ndarray,
    dataset: Dataset,
    detection_limit: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": dataset.sample_ids,
            "target": _group_labels(dataset),
            "total_abundance": totals,
            "observed_features": np.sum(relative > float(detection_limit), axis=1),
            "zero_fraction": np.mean(np.asarray(X) == 0, axis=1),
            "max_relative_abundance": np.max(relative, axis=1),
        }
    )


def _ellipse(ax: Any, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 3:
        return
    cov = np.cov(np.column_stack([x, y]), rowvar=False)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals <= 0):
        return
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    scale = math.sqrt(5.991)
    ellipse = Ellipse(
        (float(np.mean(x)), float(np.mean(y))),
        width=2.0 * scale * math.sqrt(vals[0]),
        height=2.0 * scale * math.sqrt(vals[1]),
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=0.8,
        alpha=0.8,
    )
    ax.add_patch(ellipse)


def _deterministic_jitter(n: int, width: float = 0.12, seed: int = 0) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=float)
    rng = np.random.default_rng(int(seed))
    return rng.uniform(-float(width), float(width), size=int(n))


def _target_axis_label(dataset: Dataset) -> str:
    value = str(getattr(dataset, "target_name", "")).strip().replace("_", " ")
    if value.casefold() in {"", "label", "target", "group", "class"}:
        return "Class" if dataset.task == "classification" else "Outcome"
    return (
        value[0].upper() + value[1:]
        if value
        else ("Class" if dataset.task == "classification" else "Outcome")
    )


def _plot_alpha(alpha: pd.DataFrame, dataset: Dataset, path: Path) -> Path:
    apply()
    metrics = (
        ("observed_richness", "Observed richness"),
        ("shannon", "Shannon diversity"),
        ("simpson", "Simpson diversity"),
        ("pielou_evenness", "Pielou evenness"),
    )
    target = alpha["target"]
    if dataset.task == "classification":
        levels = _classification_levels(dataset, target.astype(str).to_numpy(object))
        colors = _group_palette(len(levels))
        if len(levels) <= 4:
            fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 62 * MM))
            fig.subplots_adjust(
                left=0.055, right=0.995, bottom=0.24, top=0.94, wspace=0.42
            )
            for metric_index, (ax, (metric, label)) in enumerate(
                zip(axes.flat, metrics)
            ):
                for i, level in enumerate(levels):
                    values = np.sort(
                        alpha.loc[target.astype(str) == level, metric]
                        .dropna()
                        .to_numpy(float)
                    )
                    if not len(values):
                        continue
                    color = colors[i]
                    ax.scatter(
                        np.full(len(values), i, dtype=float)
                        + _deterministic_jitter(
                            len(values), seed=1009 * (metric_index + 1) + 97 * (i + 1)
                        ),
                        values,
                        s=7.5,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.58 if i else 0.72,
                        zorder=3,
                    )
                    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
                    low, high = np.quantile(values, [0.05, 0.95])
                    ax.plot([i, i], [low, high], color=INK, linewidth=0.55, zorder=1)
                    ax.add_patch(
                        plt.Rectangle(
                            (i - 0.16, q1),
                            0.32,
                            q3 - q1,
                            facecolor="none",
                            edgecolor=INK,
                            linewidth=0.55,
                            zorder=2,
                        )
                    )
                    ax.plot(
                        [i - 0.16, i + 0.16],
                        [median, median],
                        color=INK,
                        linewidth=0.8,
                        zorder=4,
                    )
                ax.set_xticks(np.arange(len(levels), dtype=float), levels)
                ax.set_ylabel(label)
                ax.set_xlabel("")
                _clean_axis(ax)
            fig.supxlabel(_target_axis_label(dataset), y=0.055, fontsize=7.0, color=INK)
        else:
            height = min(170.0, max(105.0, 58.0 + 7.0 * len(levels)))
            fig, axes = plt.subplots(2, 2, figsize=(183 * MM, height * MM))
            fig.subplots_adjust(
                left=0.14, right=0.99, bottom=0.10, top=0.93, wspace=0.34, hspace=0.34
            )
            for metric_index, (ax, (metric, label)) in enumerate(
                zip(axes.flat, metrics)
            ):
                positions = np.arange(len(levels), dtype=float)
                for i, level in enumerate(levels):
                    values = np.sort(
                        alpha.loc[target.astype(str) == level, metric]
                        .dropna()
                        .to_numpy(float)
                    )
                    if not len(values):
                        continue
                    color = colors[i]
                    jitter = _deterministic_jitter(
                        len(values),
                        width=0.10,
                        seed=1009 * (metric_index + 1) + 97 * (i + 1),
                    )
                    ax.scatter(
                        values,
                        np.full(len(values), i, dtype=float) + jitter,
                        s=7.5,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.58 if i else 0.72,
                        zorder=3,
                    )
                    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
                    low, high = np.quantile(values, [0.05, 0.95])
                    ax.plot([low, high], [i, i], color=INK, linewidth=0.55, zorder=1)
                    ax.add_patch(
                        plt.Rectangle(
                            (q1, i - 0.16),
                            q3 - q1,
                            0.32,
                            facecolor="none",
                            edgecolor=INK,
                            linewidth=0.55,
                            zorder=2,
                        )
                    )
                    ax.plot(
                        [median, median],
                        [i - 0.16, i + 0.16],
                        color=INK,
                        linewidth=0.8,
                        zorder=4,
                    )
                ax.set_yticks(positions, levels)
                ax.invert_yaxis()
                ax.set_xlabel(label)
                ax.set_ylabel(_target_axis_label(dataset))
                _clean_axis(ax)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(120 * MM, 108 * MM))
        fig.subplots_adjust(
            left=0.15, right=0.98, bottom=0.11, top=0.94, wspace=0.36, hspace=0.34
        )
        x = pd.to_numeric(target, errors="coerce").to_numpy(float)
        for ax, (metric, label) in zip(axes.flat, metrics):
            y = alpha[metric].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[mask],
                y[mask],
                s=10,
                facecolor=ACC,
                edgecolor=BG,
                linewidth=0.25,
                alpha=0.58,
            )
            ax.set_xlabel(_target_axis_label(dataset))
            ax.set_ylabel(label)
            _clean_axis(ax)
    save_svg(fig, path)
    plt.close(fig)
    return path


def _plot_pcoa(
    coordinates: np.ndarray,
    explained: np.ndarray,
    dataset: Dataset,
    path: Path,
) -> Path:
    apply()
    fig, ax = plt.subplots(figsize=(89 * MM, 78 * MM))
    if coordinates.shape[1] < 2:
        x = (
            coordinates[:, 0]
            if coordinates.shape[1]
            else np.zeros(len(dataset.sample_ids))
        )
        y = np.zeros_like(x)
    else:
        x, y = coordinates[:, 0], coordinates[:, 1]
    if dataset.task == "classification":
        labels = _group_labels(dataset).astype(str)
        levels = _classification_levels(dataset, labels)
        colors = _group_palette(len(levels))
        for i, level in enumerate(levels):
            idx = labels == level
            gx = np.asarray(x[idx], dtype=float)
            gy = np.asarray(y[idx], dtype=float)
            ax.scatter(
                gx,
                gy,
                s=15,
                facecolor=colors[i],
                edgecolor=BG,
                linewidth=0.35,
                alpha=0.70,
                label=level,
                zorder=3,
            )
            finite = np.isfinite(gx) & np.isfinite(gy)
            if int(np.sum(finite)) >= 4:
                points = np.column_stack([gx[finite], gy[finite]])
                covariance = np.cov(points, rowvar=False)
                if np.isfinite(covariance).all():
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    if np.all(eigenvalues > 0):
                        order = np.argsort(eigenvalues)[::-1]
                        eigenvalues = eigenvalues[order]
                        eigenvectors = eigenvectors[:, order]
                        angle = float(
                            np.degrees(
                                np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                            )
                        )
                        scale = math.sqrt(-2.0 * math.log(1.0 - 0.80))
                        width, height = 2.0 * scale * np.sqrt(eigenvalues)
                        ellipse = Ellipse(
                            xy=np.mean(points, axis=0),
                            width=float(width),
                            height=float(height),
                            angle=angle,
                            facecolor="none",
                            edgecolor=colors[i],
                            linewidth=0.65,
                            alpha=0.52,
                            zorder=2,
                        )
                        ax.add_patch(ellipse)
        columns = min(3, max(1, len(levels)))
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.55, 0.975),
            ncol=columns,
            handletextpad=0.35,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
        top = 0.84 if len(levels) <= 3 else 0.78
    else:
        scatter = ax.scatter(
            x,
            y,
            c=np.asarray(dataset.y, dtype=float),
            s=15,
            cmap=CORR_CMAP,
            edgecolor=BG,
            linewidth=0.25,
            alpha=0.76,
            zorder=3,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.035)
        cbar.set_label(dataset.target_name)
        cbar.outline.set_linewidth(0.4)
        cbar.outline.set_edgecolor(DIM)
        top = 0.96
    pc1 = 100.0 * float(explained[0]) if len(explained) > 0 else 0.0
    pc2 = 100.0 * float(explained[1]) if len(explained) > 1 else 0.0
    ax.set_xlabel(f"PCo1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PCo2 ({pc2:.1f}%)")
    ax.axhline(0, color=TRACK, linewidth=0.45, zorder=0)
    ax.axvline(0, color=TRACK, linewidth=0.45, zorder=0)
    _clean_axis(ax)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=top)
    save_svg(fig, path)
    plt.close(fig)
    return path


def _plot_composition(
    relative: np.ndarray, names: list[str], dataset: Dataset, top_n: int, path: Path
) -> Path:
    apply()
    overall = np.median(relative, axis=0)
    order = np.argsort(overall)[::-1]
    top = order[: min(max(1, int(top_n)), 12, len(order))]
    labels = [_feature_label(names[index]) for index in top]
    height = min(150.0, max(78.0, 32.0 + 6.2 * len(top)))
    if dataset.task == "classification":
        target = _group_labels(dataset).astype(str)
        groups = _classification_levels(dataset, target)
        colors = _group_palette(len(groups))
        summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for group in groups:
            values = relative[target == group][:, top] * 100.0
            summary[group] = (
                np.median(values, axis=0),
                np.quantile(values, 0.25, axis=0),
                np.quantile(values, 0.75, axis=0),
            )
        if len(groups) <= 4:
            fig, ax = plt.subplots(figsize=(120 * MM, height * MM))
            y = np.arange(len(top), dtype=float)[::-1]
            offsets = (
                np.linspace(-0.16, 0.16, len(groups))
                if len(groups) > 1
                else np.zeros(1)
            )
            if len(groups) == 2:
                first = summary[groups[0]][0]
                second = summary[groups[1]][0]
                for position, a, b in zip(y, first, second):
                    ax.plot(
                        [a, b],
                        [position, position],
                        color=TRACK,
                        linewidth=0.65,
                        zorder=1,
                        solid_capstyle="round",
                    )
            for index, group in enumerate(groups):
                median, q1, q3 = summary[group]
                position = y + offsets[index]
                for pos, low, high in zip(position, q1, q3):
                    ax.plot(
                        [low, high],
                        [pos, pos],
                        color=colors[index],
                        linewidth=0.65,
                        alpha=0.72,
                        zorder=2,
                        solid_capstyle="round",
                    )
                ax.scatter(
                    median,
                    position,
                    s=20,
                    facecolor=colors[index],
                    edgecolor=BG,
                    linewidth=0.45,
                    zorder=3,
                    label=group,
                )
            ax.set_yticks(y, labels)
            for tick in ax.get_yticklabels():
                tick.set_fontstyle("italic")
            ax.set_xlabel("Median relative abundance (%)")
            ax.set_ylabel("")
            ax.set_xlim(left=0)
            _clean_axis(ax)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, pad=5)
            fig.legend(
                loc="upper center",
                bbox_to_anchor=(0.57, 0.985),
                ncol=min(4, len(groups)),
                handletextpad=0.35,
                columnspacing=0.9,
                borderaxespad=0.0,
            )
            fig.subplots_adjust(left=0.30, right=0.98, bottom=0.12, top=0.88)
        else:
            matrix = np.vstack([summary[group][0] for group in groups]).T
            fig = plt.figure(figsize=(183 * MM, height * MM))
            ax = fig.add_axes([0.22, 0.18, 0.64, 0.70])
            cax = fig.add_axes([0.89, 0.29, 0.014, 0.42])
            image = ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap=HMAP_CMAP,
                vmin=0,
                vmax=max(0.1, float(np.nanmax(matrix))),
            )
            ax.set_yticks(np.arange(len(labels)), labels)
            for tick in ax.get_yticklabels():
                tick.set_fontstyle("italic")
            ax.set_xticks(np.arange(len(groups)), groups, rotation=35, ha="right")
            ax.tick_params(length=0, pad=4)
            for spine in ax.spines.values():
                spine.set_visible(False)
            cbar = fig.colorbar(image, cax=cax)
            cbar.outline.set_linewidth(0.4)
            cbar.outline.set_edgecolor(DIM)
            cbar.ax.tick_params(labelsize=6.0, width=0.35, length=1.8)
            cbar.set_label("Median relative abundance (%)")
    else:
        values = relative[:, top] * 100.0
        median = np.median(values, axis=0)
        q1 = np.quantile(values, 0.25, axis=0)
        q3 = np.quantile(values, 0.75, axis=0)
        fig, ax = plt.subplots(figsize=(89 * MM, height * MM))
        y = np.arange(len(top), dtype=float)[::-1]
        for position, low, high in zip(y, q1, q3):
            ax.plot(
                [low, high],
                [position, position],
                color=MID,
                linewidth=0.7,
                zorder=2,
                solid_capstyle="round",
            )
        ax.scatter(
            median, y, s=20, facecolor=ACC, edgecolor=BG, linewidth=0.45, zorder=3
        )
        ax.set_yticks(y, labels)
        for tick in ax.get_yticklabels():
            tick.set_fontstyle("italic")
        ax.set_xlabel("Relative abundance (%)")
        ax.set_ylabel("")
        ax.set_xlim(left=0)
        _clean_axis(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=5)
        fig.subplots_adjust(left=0.38, right=0.97, bottom=0.12, top=0.97)
    save_svg(fig, path)
    plt.close(fig)
    return path


def _plot_heatmap(
    clr: np.ndarray,
    relative: np.ndarray,
    names: list[str],
    dataset: Dataset,
    explore: Any,
    path: Path,
) -> Path:
    apply()
    prevalence = np.mean(relative > float(explore.detection_limit), axis=0)
    keep = np.flatnonzero(prevalence >= float(explore.min_prevalence))
    if not len(keep):
        keep = np.arange(clr.shape[1])
    variance = np.var(clr[:, keep], axis=0)
    selected = keep[
        np.argsort(variance)[::-1][: min(int(explore.heatmap_top), len(keep))]
    ]
    values = clr[:, selected]
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    z = np.divide(values - mean, std, out=np.zeros_like(values), where=std > 0)
    legend_rows = 0
    if dataset.task == "classification":
        labels = _group_labels(dataset).astype(str)
        levels = _classification_levels(dataset, labels)
        order_key = np.asarray(
            [
                levels.index(str(value)) if str(value) in levels else len(levels)
                for value in labels
            ]
        )
        sample_order = np.lexsort((np.asarray(dataset.sample_ids), order_key))
        legend_rows = int(math.ceil(len(levels) / 4.0))
    else:
        sample_order = np.argsort(np.asarray(dataset.y, dtype=float))
    z = z[sample_order].T
    height = min(170.0, max(82.0, 3.3 * len(selected) + 20.0 + 5.5 * legend_rows))
    fig = plt.figure(figsize=(183 * MM, height * MM))
    top_margin = 0.82 if dataset.task == "classification" else 0.88
    annotation = fig.add_axes([0.20, top_margin, 0.68, 0.025])
    ax = fig.add_axes([0.20, 0.12, 0.68, top_margin - 0.145])
    cax = fig.add_axes([0.91, 0.29, 0.014, 0.38])
    image = ax.imshow(
        z,
        aspect="auto",
        interpolation="nearest",
        cmap=_SIGNED_CMAP,
        vmin=-2.5,
        vmax=2.5,
    )
    feature_labels = [_feature_label(names[index]) for index in selected]
    ax.set_yticks(np.arange(len(selected)), feature_labels)
    for tick in ax.get_yticklabels():
        tick.set_fontstyle("italic")
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_linewidth(0.4)
    cbar.outline.set_edgecolor(DIM)
    cbar.ax.tick_params(labelsize=6.0, width=0.35, length=1.8)
    cbar.set_label("Standardized CLR abundance (lower → higher)")
    if dataset.task == "classification":
        ordered = _group_labels(dataset).astype(str)[sample_order]
        levels = _classification_levels(dataset, ordered)
        colors = _group_palette(len(levels))
        code = np.asarray([levels.index(str(value)) for value in ordered], dtype=float)[
            None, :
        ]
        cmap = LinearSegmentedColormap.from_list(
            "explore_groups", colors, N=max(2, len(colors))
        )
        annotation.imshow(
            code,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=max(1, len(levels) - 1),
        )
        boundaries = np.flatnonzero(ordered[1:] != ordered[:-1]) + 0.5
        for boundary in boundaries:
            annotation.axvline(boundary, color=BG, linewidth=1.0)
            ax.axvline(boundary, color=BG, linewidth=0.8)
        handles = [
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=4.0,
                markerfacecolor=color,
                markeredgecolor="none",
                label=str(level),
            )
            for level, color in zip(levels, colors)
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.975),
            ncol=min(4, len(handles)),
            handletextpad=0.35,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
    else:
        ordered_target = np.asarray(dataset.y, dtype=float)[sample_order]
        annotation.imshow(
            ordered_target[None, :],
            aspect="auto",
            interpolation="nearest",
            cmap=CORR_CMAP,
        )
        if len(ordered_target):
            annotation.text(
                0,
                -1.15,
                f"{float(np.nanmin(ordered_target)):.3g}",
                ha="left",
                va="bottom",
                fontsize=5.8,
                color=MID,
            )
            annotation.text(
                len(ordered_target) - 1,
                -1.15,
                f"{float(np.nanmax(ordered_target)):.3g}",
                ha="right",
                va="bottom",
                fontsize=5.8,
                color=MID,
            )
            annotation.text(
                (len(ordered_target) - 1) / 2.0,
                -1.15,
                dataset.target_name,
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=MID,
            )
    annotation.set_xticks([])
    annotation.set_yticks([])
    for spine in annotation.spines.values():
        spine.set_visible(False)
    save_svg(fig, path)
    plt.close(fig)
    return path


def _plot_beta_statistics(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    apply()
    table = frame.copy()
    labels = [str(value).replace("_", " ").title() for value in table["distance"]]
    y = np.arange(len(table))[::-1]
    estimates = pd.to_numeric(table["permanova_r2"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(
        table.get("permanova_r2_ci_low", np.nan), errors="coerce"
    ).to_numpy(float)
    high = pd.to_numeric(
        table.get("permanova_r2_ci_high", np.nan), errors="coerce"
    ).to_numpy(float)
    height = max(64.0, 30.0 + 10.5 * len(table))
    fig = plt.figure(figsize=(183 * MM, height * MM))
    ax = fig.add_axes([0.16, 0.24, 0.49, 0.54])
    stat = fig.add_axes([0.70, 0.24, 0.27, 0.54], sharey=ax)
    stat.set_xlim(0, 1)
    stat.set_ylim(-0.5, len(table) - 0.5)
    stat.axis("off")
    for position, estimate, lower, upper in zip(y, estimates, low, high):
        if np.isfinite(lower) and np.isfinite(upper):
            ax.plot(
                [lower, upper],
                [position, position],
                color=MID,
                linewidth=0.75,
                solid_capstyle="round",
                zorder=2,
            )
        if np.isfinite(estimate):
            ax.scatter(
                [estimate],
                [position],
                s=24,
                facecolor=ACC,
                edgecolor=BG,
                linewidth=0.45,
                zorder=3,
            )
    ax.set_yticks(y, labels)
    finite_high = high[np.isfinite(high)]
    finite_est = estimates[np.isfinite(estimates)]
    xmax = float(
        max(
            np.max(finite_high) if len(finite_high) else 0.0,
            np.max(finite_est) if len(finite_est) else 0.0,
            0.01,
        )
    )
    ax.set_xlim(0, xmax * 1.10)
    ax.set_xlabel("PERMANOVA R² (cluster-bootstrap 95% CI)")
    _clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=6)
    stat.text(
        0.04,
        len(table) - 0.16,
        "PERMANOVA P",
        fontsize=6.2,
        color=MID,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    stat.text(
        0.56,
        len(table) - 0.16,
        "PERMDISP P",
        fontsize=6.2,
        color=MID,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    for position, (_, row) in zip(y, table.iterrows()):
        p_perm = pd.to_numeric(
            pd.Series([row.get("permanova_p_value")]), errors="coerce"
        ).iloc[0]
        p_disp = pd.to_numeric(
            pd.Series([row.get("permdisp_p_value")]), errors="coerce"
        ).iloc[0]
        stat.text(
            0.04,
            position,
            "" if not np.isfinite(p_perm) else f"{float(p_perm):.3g}",
            fontsize=6.5,
            color=INK,
            ha="left",
            va="center",
        )
        stat.text(
            0.56,
            position,
            "" if not np.isfinite(p_disp) else f"{float(p_disp):.3g}",
            fontsize=6.5,
            color=INK,
            ha="left",
            va="center",
        )
    save_svg(fig, path)
    plt.close(fig)
    return path


def _pack_label_positions(
    values: np.ndarray, low: float, high: float, minimum_gap: float
) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return points
    order = np.argsort(points)
    placed = np.clip(points[order], low, high)
    for index in range(1, len(placed)):
        placed[index] = max(placed[index], placed[index - 1] + minimum_gap)
    overflow = placed[-1] - high
    if overflow > 0:
        placed -= overflow
    for index in range(len(placed) - 2, -1, -1):
        placed[index] = min(placed[index], placed[index + 1] - minimum_gap)
    underflow = low - placed[0]
    if underflow > 0:
        placed += underflow
    output = np.empty_like(placed)
    output[order] = placed
    return output


def _volcano_label_candidates(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    table = frame.copy()
    table["abs_effect"] = np.abs(pd.to_numeric(table["effect"], errors="coerce"))
    table["q_value"] = pd.to_numeric(table["q_value"], errors="coerce")
    significant = (
        pd.Series(table.get("significant", False), index=table.index)
        .fillna(False)
        .astype(bool)
    )
    robust = (
        pd.Series(table.get("robust_significant", significant), index=table.index)
        .fillna(False)
        .astype(bool)
    )
    table["significant"] = significant
    table["robust_significant"] = robust
    table = table[table["significant"]].copy()
    if table.empty:
        return table
    table["priority"] = np.where(table["robust_significant"], 0, 1)
    return (
        table.sort_values(
            ["priority", "q_value", "abs_effect"], ascending=[True, True, False]
        )
        .head(max(0, int(limit)))
        .copy()
    )


def _volcano_bbox_overlap(first: Any, second: Any, padding: float = 2.0) -> bool:
    a = first.expanded(
        (first.width + 2.0 * padding) / max(first.width, 1e-9),
        (first.height + 2.0 * padding) / max(first.height, 1e-9),
    )
    b = second.expanded(
        (second.width + 2.0 * padding) / max(second.width, 1e-9),
        (second.height + 2.0 * padding) / max(second.height, 1e-9),
    )
    return bool(a.overlaps(b))


def _volcano_segment_intersection(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> bool:
    a, b = first
    c, d = second

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    eps = 1e-9
    return bool((o1 * o2 < -eps) and (o3 * o4 < -eps))


def _volcano_segment_hits_bbox(
    start: np.ndarray, end: np.ndarray, bbox: Any, padding: float = 1.5
) -> bool:
    box = bbox.expanded(
        (bbox.width + 2.0 * padding) / max(bbox.width, 1e-9),
        (bbox.height + 2.0 * padding) / max(bbox.height, 1e-9),
    )
    for fraction in np.linspace(0.08, 0.92, 13):
        point = start + (end - start) * float(fraction)
        if box.contains(float(point[0]), float(point[1])):
            return True
    return False


def _draw_volcano_labels(
    fig: Any,
    ax: Any,
    candidates: pd.DataFrame,
    all_points: np.ndarray,
) -> None:
    if candidates.empty:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer=renderer)
    font_size = 5.8
    marker_radius = 4.2
    margin = 3.0
    base_gap = 12.0
    maximum_leader = 26.0
    vertical_offsets = (0.0, 6.0, -6.0, 12.0, -12.0, 18.0, -18.0)
    horizontal_offsets = (0.0, 4.0, 8.0, 12.0, 17.0)
    point_pixels = (
        ax.transData.transform(np.asarray(all_points, dtype=float))
        if len(all_points)
        else np.empty((0, 2), dtype=float)
    )
    placed_boxes: list[Any] = []
    placed_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for _, row in candidates.iterrows():
        point = np.asarray(
            ax.transData.transform((float(row["effect"]), float(row["minus_log10_q"]))),
            dtype=float,
        )
        preferred = 1.0 if float(row["effect"]) >= 0 else -1.0
        accepted = False
        for sign in (preferred, -preferred):
            ha = "left" if sign > 0 else "right"
            for horizontal in horizontal_offsets:
                for vertical in vertical_offsets:
                    anchor = point + np.asarray(
                        [sign * (base_gap + horizontal), vertical], dtype=float
                    )
                    data_position = ax.transData.inverted().transform(anchor)
                    artist = ax.text(
                        float(data_position[0]),
                        float(data_position[1]),
                        str(row["display_feature"]),
                        ha=ha,
                        va="center",
                        fontsize=font_size,
                        color=INK,
                        fontstyle="italic",
                        clip_on=True,
                        zorder=5,
                    )
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    box = artist.get_window_extent(renderer=renderer)
                    inside = bool(
                        box.x0 >= axes_box.x0 + margin
                        and box.x1 <= axes_box.x1 - margin
                        and box.y0 >= axes_box.y0 + margin
                        and box.y1 <= axes_box.y1 - margin
                    )
                    overlap = any(
                        _volcano_bbox_overlap(box, previous)
                        for previous in placed_boxes
                    )
                    point_overlap = False
                    if inside and not overlap and len(point_pixels):
                        for other in point_pixels:
                            if np.linalg.norm(other - point) <= 1e-6:
                                continue
                            if (
                                box.x0 - marker_radius
                                <= other[0]
                                <= box.x1 + marker_radius
                                and box.y0 - marker_radius
                                <= other[1]
                                <= box.y1 + marker_radius
                            ):
                                point_overlap = True
                                break
                    edge_x = box.x0 - 1.5 if sign > 0 else box.x1 + 1.5
                    edge_y = float(np.clip(point[1], box.y0 + 1.5, box.y1 - 1.5))
                    endpoint = np.asarray([edge_x, edge_y], dtype=float)
                    leader_length = float(np.linalg.norm(endpoint - point))
                    segment_blocked = any(
                        _volcano_segment_hits_bbox(point, endpoint, previous)
                        for previous in placed_boxes
                    )
                    segment_crossed = any(
                        _volcano_segment_intersection((point, endpoint), previous)
                        for previous in placed_segments
                    )
                    if (
                        inside
                        and not overlap
                        and not point_overlap
                        and leader_length <= maximum_leader
                        and not segment_blocked
                        and not segment_crossed
                    ):
                        start_data = ax.transData.inverted().transform(point)
                        end_data = ax.transData.inverted().transform(endpoint)
                        if leader_length >= 1.5:
                            ax.plot(
                                [float(start_data[0]), float(end_data[0])],
                                [float(start_data[1]), float(end_data[1])],
                                color=MID,
                                linewidth=0.48,
                                linestyle=(0, (1.8, 1.5)),
                                alpha=0.88,
                                solid_capstyle="butt",
                                zorder=2,
                            )
                            placed_segments.append((point.copy(), endpoint.copy()))
                        placed_boxes.append(box)
                        accepted = True
                        break
                    artist.remove()
                if accepted:
                    break
            if accepted:
                break


def _plot_da_volcano(frame: pd.DataFrame, path: Path, explore: Any) -> Path | None:
    if frame.empty or not {"effect", "q_value", "display_feature"}.issubset(
        frame.columns
    ):
        return None
    contrasts = (
        [str(value) for value in frame["contrast"].dropna().unique()]
        if "contrast" in frame.columns
        else []
    )
    if len(contrasts) > 1:
        return None
    valid = frame.copy()
    valid["effect"] = pd.to_numeric(valid["effect"], errors="coerce")
    valid["q_value"] = pd.to_numeric(valid["q_value"], errors="coerce")
    valid = valid[
        np.isfinite(valid["effect"])
        & np.isfinite(valid["q_value"])
        & (valid["q_value"] > 0)
    ].copy()
    if valid.empty:
        return None
    valid["minus_log10_q"] = -np.log10(
        np.clip(valid["q_value"].to_numpy(float), np.finfo(float).tiny, 1.0)
    )
    significant = valid.get("significant", valid["q_value"] < float(explore.da_alpha))
    robust = valid.get("robust_significant", significant)
    valid["significant"] = (
        pd.Series(significant, index=valid.index).fillna(False).astype(bool)
    )
    valid["robust_significant"] = (
        pd.Series(robust, index=valid.index).fillna(False).astype(bool)
    )
    apply()
    fig, ax = plt.subplots(figsize=(183 * MM, 94 * MM))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.965)
    effects = valid["effect"].to_numpy(float)
    y = valid["minus_log10_q"].to_numpy(float)
    data_extent = max(float(np.nanmax(np.abs(effects))) * 1.10, 0.25)
    outer_extent = data_extent * 1.45
    ymax = max(
        float(np.nanmax(y)) * 1.12, -math.log10(float(explore.da_alpha)) * 1.40, 1.0
    )
    ax.axvline(0, color=TRACK, linewidth=0.60, zorder=0)
    ax.axhline(
        -math.log10(float(explore.da_alpha)),
        color=MID,
        linewidth=0.50,
        linestyle=(0, (3.0, 2.2)),
        zorder=0,
    )
    nonsignificant = ~valid["significant"].to_numpy(bool)
    robust_mask = valid["robust_significant"].to_numpy(bool)
    sensitivity_only = valid["significant"].to_numpy(bool) & ~robust_mask
    ax.scatter(
        effects[nonsignificant],
        y[nonsignificant],
        s=15,
        facecolor="#cbd5e1",
        edgecolor=BG,
        linewidth=0.35,
        alpha=0.78,
        zorder=2,
    )
    negative = robust_mask & (effects < 0)
    positive = robust_mask & (effects >= 0)
    ax.scatter(
        effects[negative],
        y[negative],
        s=25,
        facecolor=MID,
        edgecolor=BG,
        linewidth=0.45,
        zorder=3,
    )
    ax.scatter(
        effects[positive],
        y[positive],
        s=25,
        facecolor=ACC,
        edgecolor=BG,
        linewidth=0.45,
        zorder=3,
    )
    if np.any(sensitivity_only):
        colors = [MID if value < 0 else ACC for value in effects[sensitivity_only]]
        ax.scatter(
            effects[sensitivity_only],
            y[sensitivity_only],
            s=26,
            facecolor=BG,
            edgecolor=colors,
            linewidth=0.75,
            zorder=3,
        )
    ax.set_xlim(-outer_extent, outer_extent)
    ax.set_ylim(0, ymax)
    ticks = np.linspace(-data_extent, data_extent, 5)
    if data_extent < 0.5:
        ticks = np.linspace(-data_extent, data_extent, 5)
    ax.set_xticks(ticks)
    ax.set_xlabel(
        str(valid.get("effect_scale", pd.Series(["Effect"])).dropna().iloc[0])
        if "effect_scale" in valid and valid["effect_scale"].notna().any()
        else "Effect"
    )
    ax.set_ylabel("−log10(FDR-adjusted P)", labelpad=5)
    _clean_axis(ax)
    candidates = _volcano_label_candidates(valid, int(explore.da_label_top))
    points = valid[["effect", "minus_log10_q"]].to_numpy(float)
    _draw_volcano_labels(fig, ax, candidates, points)
    save_svg(fig, path)
    plt.close(fig)
    return path


def _summary_cards(
    dataset: Dataset, scale: str, rank: str, names: list[str], clusters: np.ndarray
) -> list[tuple[str, str]]:
    return [
        ("Samples", f"{len(dataset.sample_ids):,}"),
        ("Dependence clusters", f"{pd.Series(clusters).nunique():,}"),
        ("Taxonomic rank", rank),
        ("Features", f"{len(names):,}"),
        ("Input scale", scale.replace("_", " ")),
    ]


def _html_table(frame: pd.DataFrame, limit: int = 25) -> str:
    if frame.empty:
        return '<p class="muted">No estimable rows.</p>'
    display = frame.head(int(limit)).copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.4g}"
        )
    return display.to_html(index=False, border=0, classes="dataframe")


def _metric_label(value: Any) -> str:
    labels = {
        "observed_richness": "Observed richness",
        "shannon": "Shannon diversity",
        "simpson": "Simpson diversity",
        "pielou_evenness": "Pielou evenness",
        "dominance": "Dominance",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def _format_value(value: Any, digits: int = 3) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(number):
        return ""
    return f"{float(number):.{int(digits)}g}"


def _format_effect_ci(effect: Any, low: Any, high: Any) -> str:
    estimate = pd.to_numeric(pd.Series([effect]), errors="coerce").iloc[0]
    lower = pd.to_numeric(pd.Series([low]), errors="coerce").iloc[0]
    upper = pd.to_numeric(pd.Series([high]), errors="coerce").iloc[0]
    if not np.isfinite(estimate):
        return ""
    if np.isfinite(lower) and np.isfinite(upper):
        return f"{float(estimate):.3f} ({float(lower):.3f}–{float(upper):.3f})"
    return f"{float(estimate):.3f}"


def _alpha_design_label(value: Any) -> str:
    return {
        "cluster_aggregated_independent": "Independent dependence clusters",
        "cluster_paired": "Paired dependence clusters",
        "cluster_repeated": "Repeated dependence clusters",
        "cluster_aggregated_spearman": "Cluster-aggregated Spearman association",
    }.get(str(value), str(value).replace("_", " "))


def _alpha_interpretation(row: pd.Series) -> str:
    q_value = pd.to_numeric(pd.Series([row.get("q_value")]), errors="coerce").iloc[0]
    if not np.isfinite(q_value):
        return "Not estimable"
    significant = float(q_value) < 0.05
    design = str(row.get("design", ""))
    effect = pd.to_numeric(pd.Series([row.get("effect")]), errors="coerce").iloc[0]
    median_difference = pd.to_numeric(
        pd.Series([row.get("median_difference")]), errors="coerce"
    ).iloc[0]
    if np.isfinite(median_difference):
        effect = median_difference
    if design == "cluster_aggregated_spearman":
        if not significant:
            return "No FDR-significant association"
        return (
            "Positive FDR-significant association"
            if effect > 0
            else "Negative FDR-significant association"
        )
    groups = [
        value.strip()
        for value in str(row.get("group_order", "")).split("|")
        if value.strip()
    ]
    n_groups = int(row.get("n_groups", len(groups)) or len(groups))
    if n_groups == 2 and len(groups) == 2:
        if not significant:
            return "No FDR-significant difference"
        if not np.isfinite(effect) or effect == 0:
            return "FDR-significant difference"
        return f"Higher in {groups[1]}" if effect > 0 else f"Higher in {groups[0]}"
    if n_groups > 2:
        return (
            "FDR-significant global difference"
            if significant
            else "No FDR-significant global difference"
        )
    return (
        "FDR-significant association"
        if significant
        else "No FDR-significant association"
    )


def _alpha_inference_context(
    frame: pd.DataFrame, confidence_level: float = 0.95
) -> str:
    if frame.empty:
        return ""
    parts = []
    designs = [
        str(value)
        for value in frame.get("design", pd.Series(dtype=object)).dropna().unique()
        if str(value)
    ]
    clusters = (
        pd.to_numeric(frame.get("n_clusters", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    groups = [
        str(value)
        for value in frame.get("group_order", pd.Series(dtype=object)).dropna().unique()
        if str(value)
    ]
    if len(designs) == 1:
        design_text = _alpha_design_label(designs[0])
        if design_text == "Independent dependence clusters":
            design_text = "Independent dependence-cluster design"
        elif design_text == "Paired dependence clusters":
            design_text = "Paired dependence-cluster design"
        elif design_text == "Repeated dependence clusters":
            design_text = "Repeated dependence-cluster design"
        parts.append(design_text)
    if len(clusters) == 1:
        parts.append(f"n={int(clusters[0])} dependence clusters")
    if len(groups) == 1 and " | " in groups[0]:
        ordered = [value.strip() for value in groups[0].split("|") if value.strip()]
        if len(ordered) == 2:
            parts.append(
                f"effect estimates are median differences {ordered[1]} − {ordered[0]}"
            )
    confidence = 100.0 * float(confidence_level)
    confidence_text = (
        f"{confidence:.0f}%" if confidence.is_integer() else f"{confidence:g}%"
    )
    if parts:
        return (
            "; ".join(parts)
            + f". Intervals are {confidence_text} bootstrap confidence intervals; adjusted P values use Benjamini–Hochberg FDR control."
        )
    return f"Intervals are {confidence_text} bootstrap confidence intervals; adjusted P values use Benjamini–Hochberg FDR control."


def _alpha_display_table(
    frame: pd.DataFrame, confidence_level: float = 0.95
) -> pd.DataFrame:
    if frame.empty:
        return frame
    designs = [
        str(value)
        for value in frame.get("design", pd.Series(dtype=object)).dropna().unique()
        if str(value)
    ]
    cluster_values = (
        pd.to_numeric(frame.get("n_clusters", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    include_design = len(designs) != 1
    include_clusters = len(cluster_values) != 1
    confidence = 100.0 * float(confidence_level)
    confidence_text = (
        f"{confidence:.0f}%" if confidence.is_integer() else f"{confidence:g}%"
    )
    rows = []
    for _, row in frame.iterrows():
        effect = row.get("effect", np.nan)
        low = row.get("effect_ci_low", np.nan)
        high = row.get("effect_ci_high", np.nan)
        if np.isfinite(
            pd.to_numeric(
                pd.Series([row.get("median_difference")]), errors="coerce"
            ).iloc[0]
        ):
            effect = row.get("median_difference")
            low = row.get("median_difference_ci_low")
            high = row.get("median_difference_ci_high")
        rendered = {
            "Measure": _metric_label(row.get("metric")),
            f"Effect estimate ({confidence_text} CI)": _format_effect_ci(
                effect, low, high
            ),
            "P value": _format_value(row.get("p_value")),
            "FDR-adjusted P value": _format_value(row.get("q_value")),
            "Interpretation": _alpha_interpretation(row),
        }
        if include_design:
            rendered = {
                "Measure": rendered.pop("Measure"),
                "Design": _alpha_design_label(row.get("design", "")),
                **rendered,
            }
        if include_clusters:
            position = {
                "Dependence clusters, n": int(row.get("n_clusters", 0))
                if pd.notna(row.get("n_clusters"))
                else ""
            }
            rendered = {"Measure": rendered.pop("Measure"), **position, **rendered}
        rows.append(rendered)
    return pd.DataFrame(rows)


def _beta_interpretation(row: pd.Series) -> str:
    permanova_q = pd.to_numeric(
        pd.Series([row.get("permanova_q_value")]), errors="coerce"
    ).iloc[0]
    permdisp_q = pd.to_numeric(
        pd.Series([row.get("permdisp_q_value")]), errors="coerce"
    ).iloc[0]
    composition = bool(np.isfinite(permanova_q) and float(permanova_q) < 0.05)
    dispersion = bool(np.isfinite(permdisp_q) and float(permdisp_q) < 0.05)
    if composition and dispersion:
        return "Community composition differs, but dispersion also differs; interpret PERMANOVA cautiously"
    if composition:
        return "Community composition differs; no FDR-significant dispersion difference"
    if dispersion:
        return "No FDR-significant composition difference; dispersion differs"
    return "No FDR-significant composition or dispersion difference"


def _beta_display_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "Distance": str(row.get("distance", "")).replace("_", " ").title(),
                "PERMANOVA R² (95% CI)": _format_effect_ci(
                    row.get("permanova_r2"),
                    row.get("permanova_r2_ci_low"),
                    row.get("permanova_r2_ci_high"),
                ),
                "PERMANOVA P value": _format_value(row.get("permanova_p_value")),
                "PERMANOVA FDR-adjusted P value": _format_value(
                    row.get("permanova_q_value")
                ),
                "PERMDISP P value": _format_value(row.get("permdisp_p_value")),
                "PERMDISP FDR-adjusted P value": _format_value(
                    row.get("permdisp_q_value")
                ),
                "Interpretation": _beta_interpretation(row),
            }
        )
    return pd.DataFrame(rows)


def _da_interpretation(row: pd.Series) -> str:
    q_value = pd.to_numeric(pd.Series([row.get("q_value")]), errors="coerce").iloc[0]
    effect = pd.to_numeric(pd.Series([row.get("effect")]), errors="coerce").iloc[0]
    if not np.isfinite(q_value) or float(q_value) >= 0.05:
        return "No FDR-significant taxon association"
    contrast = str(row.get("contrast", "")).strip()
    direction = (
        "positive"
        if np.isfinite(effect) and float(effect) > 0
        else "negative"
        if np.isfinite(effect) and float(effect) < 0
        else "non-zero"
    )
    if "−" in contrast:
        left, right = [part.strip() for part in contrast.split("−", 1)]
        if direction == "positive":
            message = f"Higher for {left} relative to {right}"
        elif direction == "negative":
            message = f"Lower for {left} relative to {right}"
        else:
            message = f"FDR-significant difference for {left} versus {right}"
    elif "-" in contrast and contrast.count("-") == 1:
        left, right = [part.strip() for part in contrast.split("-", 1)]
        if direction == "positive":
            message = f"Higher for {left} relative to {right}"
        elif direction == "negative":
            message = f"Lower for {left} relative to {right}"
        else:
            message = f"FDR-significant difference for {left} versus {right}"
    else:
        message = f"FDR-significant {direction} taxon association"
    robust = bool(row.get("robust_significant", row.get("significant", True)))
    if not robust:
        message += "; not sensitivity-robust"
    return message


def _da_display_table(frame: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    critical = float(norm.ppf(0.5 + float(confidence_level) / 2.0))
    scales = [
        str(value)
        for value in frame.get("effect_scale", pd.Series(dtype=object))
        .dropna()
        .unique()
    ]
    effect_label = (
        f"{scales[0]} (95% CI)"
        if len(scales) == 1 and scales[0]
        else "Effect estimate (95% CI)"
    )
    rows = []
    for _, row in frame.iterrows():
        effect = pd.to_numeric(pd.Series([row.get("effect")]), errors="coerce").iloc[0]
        standard_error = pd.to_numeric(
            pd.Series([row.get("standard_error")]), errors="coerce"
        ).iloc[0]
        low = (
            effect - critical * standard_error
            if np.isfinite(effect) and np.isfinite(standard_error)
            else np.nan
        )
        high = (
            effect + critical * standard_error
            if np.isfinite(effect) and np.isfinite(standard_error)
            else np.nan
        )
        prevalence = pd.to_numeric(
            pd.Series([row.get("prevalence")]), errors="coerce"
        ).iloc[0]
        robust = row.get("robust_significant", row.get("significant", False))
        n_samples = pd.to_numeric(
            pd.Series([row.get("n_samples")]), errors="coerce"
        ).iloc[0]
        n_clusters = pd.to_numeric(
            pd.Series([row.get("n_clusters")]), errors="coerce"
        ).iloc[0]
        rows.append(
            {
                "Taxon": _feature_label(
                    row.get("feature", row.get("display_feature", ""))
                ),
                "Contrast": str(row.get("contrast", "")),
                effect_label: _format_effect_ci(effect, low, high),
                "Samples, n": "" if not np.isfinite(n_samples) else int(n_samples),
                "Dependence clusters, n": ""
                if not np.isfinite(n_clusters)
                else int(n_clusters),
                "Prevalence": ""
                if not np.isfinite(prevalence)
                else f"{100.0 * float(prevalence):.1f}%",
                "P value": _format_value(row.get("p_value")),
                "FDR-adjusted P value": _format_value(row.get("q_value")),
                "Sensitivity-robust": "Yes" if bool(robust) else "No",
                "Interpretation": _da_interpretation(row),
            }
        )
    return pd.DataFrame(rows)


def _write_html(
    path: Path,
    title: str,
    dataset: Dataset,
    rank_outputs: list[dict[str, Any]],
    cards: list[tuple[str, str]],
    cluster_source: str,
    confidence_level: float,
) -> Path:
    card_html = "".join(
        f'<div class="card"><div class="card-value">{html.escape(value)}</div><div class="card-label">{html.escape(label)}</div></div>'
        for label, value in cards
    )
    sections = []
    captions = {
        "Alpha diversity": "Sample-level alpha-diversity distributions. Exact dependence-aware effect estimates and multiplicity-adjusted P values are reported immediately below.",
        "Aitchison PCoA": "Aitchison principal coordinates analysis of centered-log-ratio (CLR) community profiles. For classification tasks, outlines are descriptive 80% covariance ellipses showing within-group concentration; they are not confidence regions. PERMANOVA tests whether community composition differs with the target, while PERMDISP checks whether apparent separation could reflect unequal within-group dispersion. Exact results are reported immediately below.",
        "CLR abundance structure": "Standardized CLR abundance structure for prevalent, high-variance taxa. Slate denotes lower standardized CLR abundance, near-white denotes values near the taxon-specific mean and blue denotes higher standardized CLR abundance. This panel is descriptive and is intended to expose sample- and taxon-level structure rather than provide taxon-wise inference.",
        "Differential abundance and taxon associations": "Differential-abundance or taxon-association volcano. Straight dashed leaders connect selected feature labels to their points. Exact estimates, confidence intervals, prevalence and FDR-adjusted P values are reported immediately below.",
    }
    for output in rank_outputs:
        rank = str(output["rank"])
        figures = output["figures"]
        beta = _beta_display_table(output["beta_statistics"])
        alpha = _alpha_display_table(output["alpha_statistics"], confidence_level)
        alpha_context = _alpha_inference_context(
            output["alpha_statistics"], confidence_level
        )
        da = output["differential_abundance"]
        da_results = _da_display_table(
            da.get("results", pd.DataFrame()), confidence_level
        )
        da_method = str(da.get("method", ""))
        da_model = str(da.get("model", ""))
        da_note = f"Primary taxon-level analysis: {da_method}. {da_model}".strip()
        blocks = []
        for label in (
            "Alpha diversity",
            "Aitchison PCoA",
            "CLR abundance structure",
            "Differential abundance and taxon associations",
        ):
            figure = figures.get(label)
            if figure is None:
                continue
            rel = Path(figure).relative_to(path.parent).as_posix()
            wide = label in {
                "Alpha diversity",
                "CLR abundance structure",
                "Differential abundance and taxon associations",
            }
            caption = captions[label]
            if label == "Alpha diversity" and alpha_context:
                caption = f"{caption} {alpha_context}"
            blocks.append(
                f'<figure class="{"figure-wide" if wide else ""}"><img src="{html.escape(rel)}" alt="{html.escape(label)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
            )
            if label == "Alpha diversity" and not alpha.empty:
                blocks.append(
                    '<div class="table-block"><h3>Alpha-diversity inference</h3>'
                    + _html_table(alpha)
                    + "</div>"
                )
            elif label == "Aitchison PCoA" and not beta.empty:
                blocks.append(
                    '<div class="table-block"><h3>Beta-diversity inference</h3>'
                    + _html_table(beta)
                    + "</div>"
                )
            elif label == "Differential abundance and taxon associations":
                method_html = (
                    f'<p class="muted">{html.escape(da_note)}</p>' if da_note else ""
                )
                blocks.append(
                    '<div class="table-block"><h3>Differential abundance and taxon associations</h3>'
                    + method_html
                    + _html_table(da_results)
                    + "</div>"
                )
        sections.append(
            f'<section><div class="section-kicker">{html.escape(rank)}</div><h2>{html.escape(rank.title())} ecological structure and inference</h2>{"".join(blocks)}</section>'
        )
    css = """
:root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--panel:#f8fafc;--accent:#1565A8}*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font-family:Arial,Helvetica,sans-serif;line-height:1.45}.wrap{max-width:1180px;margin:0 auto;padding:48px 40px 80px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.13em;color:var(--accent);font-weight:700}h1{font-size:34px;line-height:1.08;margin:8px 0 10px;letter-spacing:-.025em}h2{font-size:24px;margin:4px 0 22px;letter-spacing:-.015em}h3{font-size:15px;margin:0 0 10px}.lede{color:var(--muted);max-width:850px;font-size:15px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin:28px 0 36px}.card{border:1px solid var(--line);border-radius:10px;padding:16px;background:#fff}.card-value{font-size:21px;font-weight:700}.card-label{font-size:11px;color:var(--muted);margin-top:3px}.note{border-left:2px solid var(--accent);padding:10px 14px;background:#f8fbff;color:#334155;font-size:13px;margin:22px 0 38px}.section-kicker{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:700}section{padding:34px 0;border-top:1px solid var(--line)}figure{margin:28px 0 8px;border-top:1px solid var(--line);padding:16px 0 0;background:#fff}figure img{width:auto;max-width:100%;height:auto;display:block;max-height:760px;object-fit:contain}figcaption{font-size:11px;color:var(--muted);padding:9px 3px 2px;max-width:960px}.table-block{margin:10px 0 34px}.dataframe{border-collapse:collapse;width:100%;font-size:11px;display:block;overflow:auto;font-variant-numeric:tabular-nums}.dataframe th,.dataframe td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.dataframe th{font-weight:700;background:var(--panel);position:sticky;top:0}.dataframe tbody tr:hover{background:#fbfdff}.muted{color:var(--muted);font-size:12px}@media(max-width:760px){.wrap{padding:28px 18px 56px}h1{font-size:28px}}
"""
    content = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · Explore</title><style>{css}</style></head><body><main class="wrap"><div class="eyebrow">mllabiome data exploration</div><h1>{html.escape(title)}</h1><p class="lede">Exploration is organized around non-redundant scientific questions: within-sample diversity, between-sample community geometry, multivariate abundance structure, and taxon-level association. Exact inferential values are shown in tables rather than duplicated as decorative plots.</p><div class="cards">{card_html}</div><div class="note">Dependence unit: <strong>{html.escape(cluster_source)}</strong>. Metadata declarations identify available variables but do not imply causal adjustment; differential-abundance covariates are included only when explicitly specified.</div>{"".join(sections)}</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _available_ranks(dataset: Dataset, requested: tuple[str, ...]) -> tuple[str, ...]:
    available = tuple(
        level
        for level in TAXONOMIC_LEVELS
        if level in dataset.X_by_level and dataset.X_by_level[level].shape[1] > 0
    )
    chosen = tuple(level for level in requested if level in available)
    if chosen:
        return chosen
    if available:
        return (available[-1],)
    return ("all",)


def run_explore(sweep: Any) -> dict[str, Any]:
    if getattr(sweep, "uses_modalities", False):
        raise ValueError("Explore currently requires a Data-based microbiome sweep.")
    if sweep.data is None:
        raise ValueError("Explore requires Sweep.data.")
    explore = sweep.explore
    requested = tuple(explore.ranks)
    levels = tuple(dict.fromkeys((*requested, *TAXONOMIC_LEVELS)))
    dataset = load_dataset(sweep.data, levels_needed=levels)
    ranks = _available_ranks(dataset, requested)
    available = tuple(
        level
        for level in TAXONOMIC_LEVELS
        if level in dataset.X_by_level and dataset.X_by_level[level].shape[1] > 0
    )
    base_rank = available[-1] if available else ranks[-1]
    base_X = np.asarray(dataset.X_by_level[base_rank], dtype=float)
    base_names = list(dataset.feature_names_by_level[base_rank])
    clusters, cluster_source = _cluster_ids(sweep, dataset)
    root = Path(sweep.root()) / "explore"
    root.mkdir(parents=True, exist_ok=True)
    rank_outputs = []
    manifest_ranks = []
    cards = []
    for rank_index, rank in enumerate(ranks):
        X = np.asarray(dataset.X_by_level[rank], dtype=float)
        names = list(dataset.feature_names_by_level[rank])
        relative, totals = _relative_abundance(X)
        scale = _infer_scale(X, totals)
        zero_replacement = _adaptive_zero_replacement(
            relative, explore.zero_replacement
        )
        clr = _clr_matrix(relative, zero_replacement)
        rank_dir = root / rank
        figure_dir = rank_dir / "figures"
        table_dir = rank_dir / "tables"
        alpha = _alpha_table(
            relative, dataset, clusters, cluster_source, explore.detection_limit
        )
        alpha_stats = _alpha_statistics(alpha, dataset, explore)
        distances = _distance_matrices(relative, clr, explore.detection_limit)
        beta_stats = _beta_statistics(
            distances, _group_labels(dataset), clusters, explore
        )
        summary = _taxon_summary(relative, names, explore.detection_limit)
        qc = _sample_qc(X, relative, totals, dataset, explore.detection_limit)
        coordinates, explained = _pcoa(distances["aitchison"])
        da = _differential_abundance(
            scale,
            base_X,
            base_names,
            X,
            relative,
            clr,
            names,
            dataset,
            clusters,
            explore,
        )
        table_frames: dict[str, pd.DataFrame] = {
            "sample_qc": qc,
            "alpha_diversity": alpha,
            "alpha_statistics": alpha_stats,
            "beta_statistics": beta_stats,
            "taxon_summary": summary,
            "differential_abundance": da.get("results", pd.DataFrame()),
            "differential_abundance_global": da.get("global", pd.DataFrame()),
            "differential_abundance_pairwise": da.get("pairwise", pd.DataFrame()),
            "differential_abundance_structural_zeros": da.get(
                "structural_zeros", pd.DataFrame()
            ),
            "differential_abundance_sensitivity": da.get("sensitivity", pd.DataFrame()),
        }
        if "primary" in da:
            table_frames["differential_abundance_primary"] = da.get(
                "primary", pd.DataFrame()
            )
        tables = {
            name: write_table(table_dir / f"{name}.parquet", frame)
            for name, frame in table_frames.items()
            if not frame.empty
        }
        for stale_name in ("composition.svg", "beta_diversity_statistics.svg"):
            (figure_dir / stale_name).unlink(missing_ok=True)
        figures: dict[str, Path | None] = {
            "Alpha diversity": _plot_alpha(
                alpha, dataset, figure_dir / "alpha_diversity.svg"
            ),
            "Aitchison PCoA": _plot_pcoa(
                coordinates, explained, dataset, figure_dir / "aitchison_pcoa.svg"
            ),
            "CLR abundance structure": _plot_heatmap(
                clr, relative, names, dataset, explore, figure_dir / "clr_heatmap.svg"
            ),
            "Differential abundance and taxon associations": _plot_da_volcano(
                da.get("results", pd.DataFrame()),
                figure_dir / "differential_abundance.svg",
                explore,
            ),
        }
        payload = {
            "rank": rank,
            "scale": scale,
            "zero_replacement_relative": zero_replacement,
            "n_features": len(names),
            "tables": {name: str(value) for name, value in tables.items()},
            "figures": {
                name: str(value) if value is not None else None
                for name, value in figures.items()
            },
            "differential_abundance": {
                "method": da.get("method"),
                "status": da.get("status"),
                "model": da.get("model"),
                "multiple_testing": da.get("multiple_testing"),
                "runtime": da.get("runtime"),
            },
        }
        dump_json_standard(payload, rank_dir / "manifest.json")
        rank_outputs.append(
            {
                "rank": rank,
                "alpha_statistics": alpha_stats,
                "beta_statistics": beta_stats,
                "differential_abundance": da,
                "figures": figures,
            }
        )
        manifest_ranks.append(payload)
        if rank_index == 0:
            cards = _summary_cards(dataset, scale, rank, names, clusters)
    manifest = {
        "stage": "explore",
        "target": dataset.target_name,
        "task": dataset.task,
        "n_samples": len(dataset.sample_ids),
        "n_dependence_clusters": int(pd.Series(clusters).nunique()),
        "dependence_cluster_source": cluster_source,
        "detection_limit_relative": float(explore.detection_limit),
        "min_prevalence": float(explore.min_prevalence),
        "permutations": int(explore.permutations),
        "bootstrap_replicates": int(explore.bootstrap_replicates),
        "confidence_level": float(explore.confidence_level),
        "zero_handling": "multiplicative_replacement_preserving_nonzero_ratios",
        "primary_beta_distance": "aitchison",
        "sensitivity_beta_distances": ["bray_curtis", "jaccard"],
        "differential_abundance": str(explore.differential_abundance),
        "da_covariates": list(explore.da_covariates),
        "multiple_testing": "method-specific; recorded for each rank",
        "ranks": manifest_ranks,
    }
    manifest_path = root / "manifest.json"
    dump_json_standard(manifest, manifest_path)
    report_path = _write_html(
        root / "index.html",
        sweep.title,
        dataset,
        rank_outputs,
        cards,
        cluster_source,
        float(explore.confidence_level),
    )
    return {
        "manifest": manifest_path,
        "report": report_path,
        "ranks": manifest_ranks,
    }
