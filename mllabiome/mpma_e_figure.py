"""Selected MPMA-E ensemble schematic rendering.

This module contains the package renderer used by the ensemble stage to
produce SVG, PDF, PNG, and member-metadata outputs for a single mllabiome
experiment.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import rankdata

try:
    from sklearn.preprocessing import (PowerTransformer, QuantileTransformer,
                                       RobustScaler)
except Exception:  # pragma: no cover - optional transformation support
    PowerTransformer = None
    QuantileTransformer = None
    RobustScaler = None

MM = 1 / 25.4
INK = "#0f172a"
MID = "#64748b"
DIM = "#94a3b8"
TRACK = "#cbd5e1"
BG = "#ffffff"
ACC = "#2563eb"
ACC_L = "#dbeafe"
ELLIPSIS_BG = "#f8fafc"

ABUND_CMAP = LinearSegmentedColormap.from_list(
    "abund", [(241 / 255, 245 / 255, 249 / 255), (2 / 255, 155 / 255, 190 / 255)], N=256
)
BIN_CMAP = ListedColormap(["#f1f5f9", "#1e3a8a"])
CLR_CMAP = LinearSegmentedColormap.from_list(
    "clr", ["#60a5fa", "#bfdbfe", "#1e3a8a"], N=256
)
TSS_CMAP = LinearSegmentedColormap.from_list("tss", ["#bde0fe", "#023e8a"], N=256)
LOG_CMAP = LinearSegmentedColormap.from_list("log", ["#ecfeff", "#0e7490"], N=256)
RANK_CMAP = LinearSegmentedColormap.from_list("rank", ["#f8fafc", "#475569"], N=256)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Inter",
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "DejaVu Sans",
        ],
        "font.size": 7.8,
        "axes.linewidth": 0.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species", "strain"]
LEVEL_MARKERS = {
    "domain": "d__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
    "strain": "t__",
}

# Compact publication-facing transformation labels.  These match the MPMA-B / MPMA-E
TR_ALIASES = {
    "skbio_clr": "scikit-bio_clr",
    "skbio_alr": "scikit-bio_alr",
    "skbio_ilr": "scikit-bio_ilr",
    "scikit_bio_clr": "scikit-bio_clr",
    "scikit_bio_alr": "scikit-bio_alr",
    "scikit_bio_ilr": "scikit-bio_ilr",
    "scikitbio_clr": "scikit-bio_clr",
    "scikitbio_alr": "scikit-bio_alr",
    "scikitbio_ilr": "scikit-bio_ilr",
}

TRANSFORM_DISPLAY = {
    "none": "RA",
    "binary": "P/A",
    "sqrt": r"$\sqrt{x}$",
    "hellinger": "Hellinger",
    "arcsin_sqrt": r"$\arcsin\sqrt{x}$",
    "arcsine_sqrt": r"$\arcsin\sqrt{x}$",
    "log": r"$\ln(1+x)$",
    "log2": r"$\log_2(1+x)$",
    "log10": r"$\log_{10}(1+x)$",
    "log_tss_floor": "log-TSS",
    "zi_log": "ZI-log",
    "symlog": "symlog",
    "clr": r"CLR-$\epsilon$",
    "scikit-bio_clr": "CLR-mult",
    "alr": "ALR-last",
    "scikit-bio_alr": "ALR-first",
    "ilr": "ILR-seq",
    "scikit-bio_ilr": "ILR-Egoz.",
    "rclr": "RCLR",
    "bclr": "BCLR",
    "clr_std": r"CLR-$\epsilon$+z",
    "ilr_std": "ILR-seq+z",
    "zscore": "row-z",
    "log_std": "row-log-z",
    "log_unit": "row-log-L2",
    "rank_frac": "row-rank",
    "rank_std": "row-rank-z",
    "rank_unit": "row-rank-unit",
    "rank_col": "ECDF-rank",
    "prev_weighted": "Prev-wt",
    "power": "Yeo-J",
    "robust": "Robust",
    "quantile": "QNorm",
    "pairwise_logratio": "Pair-logR-500",
    # Relative-namespace transformation labels.
    "relative_none": "RA",
    "relative_binary": "P/A",
    "relative_sqrt": r"$\sqrt{x}$",
    "relative_log10": r"$\log_{10}(1+x)$",
    "relative_log2": r"$\log_2(1+x)$",
    "relative_zscore": "row-z",
    "relative_log_unit": "row-log-L2",
    "relative_alr": "ALR-last",
    "rbinary": "P/A",
    "ralr": "ALR-last",
    "rilr": "ILR-seq",
}


# Task descriptor used by the package renderer.
@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    experiment_dirs: tuple[str, ...]


@dataclass
class MemberRecord:
    order: int | None
    config_id: str | None
    raw_resolution: str | None = None
    raw_levels: str | None = None
    raw_transform: str | None = None
    raw_model: str | None = None
    levels: tuple[str, ...] = ()
    ranks_display: str = "?"
    transform_display: str = "?"
    classifier_display: str = "?"
    n_features: int | None = None
    omitted_count: int = 0

    @property
    def is_ellipsis(self) -> bool:
        return self.config_id is None


@dataclass
class EnsembleTask:
    spec: TaskSpec
    experiment_dir: Path
    selected: dict[str, Any]
    members: list[MemberRecord]
    shown_members: list[MemberRecord]
    X: np.ndarray
    taxa: list[str]
    source: str
    diagnostics: dict[str, Any]


# ---------------------------------------------------------------------------
# Config/selected-unit readers
# ---------------------------------------------------------------------------


def read_selected_unit(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "ensembling" / "selected_unit.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_config_meta(
    experiment_dir: Path, include_inactive: bool = False
) -> pd.DataFrame:
    db_path = experiment_dir / "configs.db"
    tsv_path = experiment_dir / "configs.tsv"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        query = (
            "SELECT * FROM configs"
            if include_inactive
            else "SELECT * FROM configs WHERE active=1"
        )
        df = pd.read_sql(query, conn)
        conn.close()
    elif tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t", dtype=str)
        if not include_inactive and "active" in df.columns:
            active = pd.to_numeric(df["active"], errors="coerce").fillna(1).astype(int)
            df = df.loc[active.eq(1)].copy()
    else:
        raise FileNotFoundError(f"No configs.db/configs.tsv found in {experiment_dir}")
    # mllabiome stores the executable MPMA table with package-native column
    # names (count_transformation, learner). Some selected-unit files use
    # transform/model. Normalise here so the figure reads the selected MPMA
    # members instead of showing "Unknown classifier".
    if "transform" not in df.columns and "count_transformation" in df.columns:
        df["transform"] = df["count_transformation"]
    if "model" not in df.columns and "learner" in df.columns:
        df["model"] = df["learner"]
    for col in ["config_id", "resolution", "levels", "transform", "model"]:
        if col not in df.columns:
            df[col] = ""
    df["config_id"] = df["config_id"].astype(str)
    return df


def canonical_transform(x: Any) -> str:
    key = str(x or "").strip()
    return TR_ALIASES.get(key, key)


def transform_display(x: Any) -> str:
    key = canonical_transform(x)
    return TRANSFORM_DISPLAY.get(key, key.replace("_", " "))


def parse_levels(raw_levels: Any, resolution: Any = None) -> tuple[str, ...]:
    text = str(raw_levels or "").strip()
    if text and text.lower() != "nan":
        parts = [p.strip().lower() for p in re.split(r"[,;+\s]+", text) if p.strip()]
        levels = tuple(p for p in parts if p in LEVELS)
        if levels:
            return levels
    res = str(resolution or "").strip().lower()
    if "+" in res:
        levels = tuple(p for p in res.split("+") if p in LEVELS)
        if levels:
            return levels
    if "-" in res:
        a, b = [p.strip() for p in res.split("-", 1)]
        if a in LEVELS and b in LEVELS:
            ia, ib = LEVELS.index(a), LEVELS.index(b)
            if ia > ib:
                ia, ib = ib, ia
            return tuple(LEVELS[ia : ib + 1])
    if res in LEVELS:
        return (res,)
    return ()


def ranks_display(levels: Iterable[str], resolution: Any = None) -> str:
    vals = [x for x in levels if x in LEVELS]
    if not vals:
        return str(resolution or "?").replace("-", "→")
    # remove duplicates preserving order
    uniq: list[str] = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    if len(uniq) == 1:
        return uniq[0]
    idx = [LEVELS.index(x) for x in uniq]
    contiguous = sorted(idx) == list(range(min(idx), max(idx) + 1)) and len(
        set(idx)
    ) == len(idx)
    if contiguous:
        return f"{LEVELS[min(idx)]}→{LEVELS[max(idx)]}"
    return "+".join(uniq)


def model_family_display(model: Any, max_chars: int = 20) -> str:
    m = str(model or "").strip()
    ml = m.lower()
    pairs = [
        ("Random forest", ["rf_", "randomforest", "random_forest"]),
        ("Extremely randomized trees", ["et_", "extratree"]),
        ("Gradient boosting", ["gradientboost", "gbm", "gt_default"]),
        ("Histogram gradient boosting", ["histgb", "histgradient"]),
        ("Extreme Gradient Boosting", ["xgb"]),
        ("LightGBM", ["lgb", "lightgbm"]),
        ("CatBoost", ["catboost", "cb_"]),
        ("Logistic regression", ["lr_", "logistic", "logreg"]),
        ("Ridge classifier", ["ridge"]),
        ("Linear discriminant analysis", ["lda"]),
        ("Quadratic discriminant analysis", ["qda", "oasqda"]),
        ("Support vector machine", ["svc", "lsvc", "linearsvc"]),
        ("k-nearest neighbours", ["knn", "kneighbors"]),
        ("Nearest centroid", ["nearestcentroid", "nc_"]),
        ("Bernoulli naive Bayes", ["bnb", "bernoulli"]),
        ("Complement naive Bayes", ["cnb", "complement"]),
        ("Gaussian naive Bayes", ["gnb", "gaussian"]),
        ("Multinomial naive Bayes", ["mnb", "multinomial"]),
        ("Label spreading", ["labelspread"]),
        ("Label propagation", ["labelprop"]),
        ("Self-training", ["selftrain"]),
        ("Bagged classifier", ["bag_"]),
        ("Passive-aggressive", ["pa_"]),
        ("Stochastic gradient descent", ["sgd"]),
        ("Multilayer perceptron", ["mlp"]),
        ("Factor analysis + logistic regression", ["fa_", "fa_lr"]),
        ("Gaussian-mixture classifier", ["gmm"]),
        ("FLAML AutoML", ["flaml"]),
        ("Decision tree", ["dt_"]),
    ]
    for label, keys in pairs:
        if any(k in ml for k in keys):
            return "\n".join(textwrap.wrap(label, width=max_chars))
    if not m:
        return "Unknown\nclassifier"
    return "\n".join(textwrap.wrap(m.replace("_", " "), width=max_chars))


def selection_display(sel: Any, selected: dict[str, Any]) -> str:
    s = str(sel or "?")
    metric = str(selected.get("optimize_metric") or "nMCC")
    threshold = selected.get("threshold_score")
    # Some candidate rows include threshold_score even when selected_unit does not.
    if threshold is None and s == "threshold":
        threshold = 0.30
    max_members = selected.get("ensemble_size")
    if s == "threshold":
        return f"Selection: inner-validation threshold ({metric} ≥ {float(threshold):.2f}, max {int(max_members)})"
    mapping = {
        "top_k": "Selection: inner-validation top-k",
        "diverse_top_k": "Selection: inner-validation diverse top-k",
        "best_per_family": "Selection: inner-validation best per classifier family",
        "stratified": "Selection: inner-validation stratified",
        "best_per_resolution": "Selection: inner-validation best per rank set",
        "resolution_diverse": "Selection: inner-validation rank-diverse",
        "caruana": "Selection: inner-validation Caruana ensemble selection",
        "greedy_diverse": "Selection: inner-validation greedy diverse search",
        "hillclimb": "Selection: inner-validation hill-climbing search",
    }
    return mapping.get(s, f"Selection: inner-validation {s.replace('_', ' ')}")


def aggregation_display(agg: Any) -> str:
    a = str(agg or "?")
    mapping = {
        "mean_proba": "Aggregation: mean probability",
        "weighted_mean_proba": "Aggregation: inner-score weighted mean probability",
        "median_proba": "Aggregation: median probability",
        "trimmed_mean": "Aggregation: trimmed mean probability",
        "geometric_mean": "Aggregation: geometric mean probability",
        "log_odds_mean": "Aggregation: mean log-odds",
        "harmonic_mean": "Aggregation: harmonic mean probability",
        "minmax": "Aggregation: mean of minimum and maximum probabilities",
        "rank_mean": "Aggregation: mean probability rank",
        "borda_count": "Aggregation: Borda-count rank aggregation",
        "copeland": "Aggregation: Copeland-style rank aggregation",
        "majority_vote": "Aggregation: majority vote",
        "weighted_vote": "Aggregation: inner-score weighted vote",
        "confidence_weighted": "Aggregation: confidence-weighted probability",
        "softmax_mean": "Aggregation: softmax-weighted probability",
        "bayesian_avg": "Aggregation: Bayesian model-averaged probability",
        "power_mean_p3": "Aggregation: power mean (p=3)",
        "power_mean_p05": "Aggregation: power mean (p=0.5)",
        "dempster_shafer": "Aggregation: Dempster–Shafer combination",
        "max_proba": "Aggregation: maximum probability",
        "min_proba": "Aggregation: minimum probability",
        "superlearner__lr": "Aggregation: logistic-regression super learner",
        "superlearner__ridge": "Aggregation: ridge-regression super learner",
        "superlearner__rf": "Aggregation: random-forest super learner",
    }
    return mapping.get(a, f"Aggregation: {a.replace('_', ' ')}")


def cap_members(members: list[MemberRecord], max_members: int) -> list[MemberRecord]:
    if max_members <= 0 or len(members) <= max_members:
        return members
    n_first = max_members // 2
    n_last = max_members - n_first
    omitted = len(members) - n_first - n_last
    ell = MemberRecord(order=None, config_id=None, omitted_count=omitted)
    return members[:n_first] + [ell] + members[-n_last:]


def build_members(
    selected: dict[str, Any], config_meta: pd.DataFrame, max_members: int
) -> tuple[list[MemberRecord], list[MemberRecord]]:
    best = selected.get("inner_val_best_ensemble", selected)
    raw_members = best.get("members", [])
    if isinstance(raw_members, str):
        try:
            raw_members = json.loads(raw_members)
        except Exception:
            raw_members = [x.strip() for x in raw_members.split(",") if x.strip()]
    if not isinstance(raw_members, list):
        raw_members = []

    meta = config_meta.drop_duplicates("config_id").set_index("config_id", drop=False)

    def _member_id(x: Any) -> str:
        if isinstance(x, dict):
            for key in ("config_id", "member", "id"):
                if key in x and x[key] is not None:
                    return str(x[key])
        return str(x)

    def _lookup_member(cid: str):
        if cid in meta.index:
            return meta.loc[cid]
        # Be tolerant of JSON files that contain shortened or stringified IDs.
        matches = config_meta[
            config_meta["config_id"].astype(str).str.startswith(str(cid))
        ]
        if len(matches) == 1:
            return matches.iloc[0]
        matches = config_meta[
            config_meta["config_id"]
            .astype(str)
            .str.contains(str(cid), regex=False, na=False)
        ]
        if len(matches) == 1:
            return matches.iloc[0]
        return None

    records: list[MemberRecord] = []
    for i, cid in enumerate([_member_id(x) for x in raw_members], start=1):
        row = _lookup_member(cid)
        if row is not None:
            raw_res = str(row.get("resolution", ""))
            raw_levels = str(row.get("levels", ""))
            raw_transform = str(row.get("transform", ""))
            raw_model = str(row.get("model", ""))
        else:
            raw_res = raw_levels = raw_transform = raw_model = ""
        lv = parse_levels(raw_levels, raw_res)
        records.append(
            MemberRecord(
                order=i,
                config_id=cid,
                raw_resolution=raw_res,
                raw_levels=raw_levels,
                raw_transform=raw_transform,
                raw_model=raw_model,
                levels=lv,
                ranks_display=ranks_display(lv, raw_res),
                transform_display=transform_display(raw_transform),
                classifier_display=model_family_display(raw_model),
            )
        )
    return records, cap_members(records, max_members=max_members)


# ---------------------------------------------------------------------------
# Demo data loaders and taxonomic aggregation
# ---------------------------------------------------------------------------


def clean_taxon_name(name: Any) -> str:
    s = str(name).strip()
    if "|" in s:
        s = s.replace("k__", "d__", 1).replace("|", "___")
    return s


def _rank_prefix(level: str) -> str:
    return LEVEL_MARKERS[level]


def truncate_to_level(taxon: str, level: str) -> str | None:
    taxon = clean_taxon_name(taxon)
    parts = taxon.split("___")
    depth = LEVELS.index(level) + 1
    if len(parts) < depth:
        return None
    out = "___".join(parts[:depth])
    prefix = _rank_prefix(level)
    if level == "domain":
        return out if out.startswith("d__") else None
    return out if f"___{prefix}" in out or out.startswith(prefix) else None


def _exact_rank_of_taxon(taxon: str) -> str | None:
    taxon = clean_taxon_name(taxon)
    for level in reversed(LEVELS):
        marker = _rank_prefix(level)
        if level == "domain":
            if taxon.startswith("d__") and "___p__" not in taxon:
                return "domain"
        elif f"___{marker}" in taxon or taxon.startswith(marker):
            next_i = LEVELS.index(level) + 1
            if next_i >= len(LEVELS):
                return level
            next_marker = _rank_prefix(LEVELS[next_i])
            if f"___{next_marker}" not in taxon:
                return level
    return None


def aggregate_to_levels(
    X: np.ndarray, taxa: list[str], levels: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    if not levels:
        return X.astype(float, copy=True), taxa[:]
    blocks = []
    names_out: list[str] = []
    exact_rank = [_exact_rank_of_taxon(t) for t in taxa]
    for level in levels:
        # When the sweep matrix already contains exact rank-specific feature
        # blocks, use those exact columns.  Only aggregate descendants when an
        # exact block is not available.  This avoids double-counting ancestor
        # rows in the MPMA-E schematic.
        exact_idx = [j for j, r in enumerate(exact_rank) if r == level]
        if exact_idx:
            blocks.append(np.asarray(X[:, exact_idx], dtype=float))
            names_out.extend([taxa[j] for j in exact_idx])
            continue
        groups: dict[str, list[int]] = {}
        for j, tax in enumerate(taxa):
            tn = truncate_to_level(tax, level)
            if tn is not None:
                groups.setdefault(tn, []).append(j)
        if not groups:
            continue
        keys = sorted(groups)
        B = np.zeros((X.shape[0], len(keys)), dtype=float)
        for k, name in enumerate(keys):
            B[:, k] = X[:, groups[name]].sum(axis=1)
        blocks.append(B)
        names_out.extend(keys)
    if not blocks:
        return X.astype(float, copy=True), taxa[:]
    return np.concatenate(blocks, axis=1), names_out


def row_normalise(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = np.clip(X, 0, None)
    s = X.sum(axis=1, keepdims=True)
    s[s < eps] = 1.0
    return X / s


def synthetic_taxa_matrix(
    seed: int = 42, n_samples: int = 36
) -> tuple[np.ndarray, list[str], str]:
    rng = np.random.default_rng(seed)
    taxa: list[str] = []
    # Structured hierarchy: 2 phyla, 3 classes each, ... enough features for all ranks.
    for p in range(1, 4):
        for c in range(1, 4):
            for o in range(1, 3):
                for f in range(1, 3):
                    for g in range(1, 3):
                        for s in range(1, 3):
                            taxa.append(
                                f"d__Bacteria___p__P{p}___c__C{p}_{c}___o__O{p}_{c}_{o}"
                                f"___f__F{p}_{c}_{o}_{f}___g__G{p}_{c}_{o}_{f}_{g}"
                                f"___s__S{p}_{c}_{o}_{f}_{g}_{s}"
                            )
    alpha = rng.uniform(0.05, 1.5, size=len(taxa))
    X = rng.dirichlet(alpha, size=n_samples)
    # Add sparse zero inflation for microbiome-like appearance.
    mask = rng.random(X.shape) < 0.30
    X[mask] = 0.0
    X = row_normalise(X)
    return X, taxa, "synthetic demo matrix"


def _mr(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    out = np.asarray(X, dtype=float).copy()
    out[out <= 0] = eps
    return out


def _row_z(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-10] = 1.0
    return (X - mu) / sd


def _col_z(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    return (X - mu) / sd


def _clr(X: np.ndarray) -> np.ndarray:
    R = row_normalise(X)
    L = np.log(_mr(R))
    return L - L.mean(axis=1, keepdims=True)


def _alr(X: np.ndarray) -> np.ndarray:
    R = row_normalise(X)
    if R.shape[1] < 2:
        return R
    R = _mr(R)
    return np.log(R[:, :-1] / R[:, -1:])


def _ilr(X: np.ndarray) -> np.ndarray:
    R = row_normalise(X)
    if R.shape[1] < 2:
        return R
    L = np.log(_mr(R)).astype(float)
    D = L.shape[1]
    cum = np.cumsum(L[:, :-1], axis=1)
    k = np.arange(1, D, dtype=float)
    return np.sqrt(k / (k + 1.0)) * (cum / k - L[:, 1:])


def apply_transform(
    X: np.ndarray, raw_transform: Any, seed: int = 42
) -> tuple[np.ndarray, mpl.colors.Colormap, float, float]:
    key = canonical_transform(raw_transform)
    # old relative_* names map to same operation on row-normalised data
    key = {
        "relative_none": "none",
        "relative_binary": "binary",
        "rbinary": "binary",
        "relative_sqrt": "sqrt",
        "relative_log10": "log10",
        "relative_log2": "log2",
        "relative_zscore": "zscore",
        "relative_log_unit": "log_unit",
        "relative_alr": "alr",
        "ralr": "alr",
        "rilr": "ilr",
    }.get(key, key)

    R = row_normalise(X)
    if key == "none":
        Z, cmap = R, ABUND_CMAP
        return Z, cmap, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "binary":
        Z = (X > 0).astype(float)
        return Z, BIN_CMAP, 0.0, 1.0
    if key == "sqrt":
        Z = np.sqrt(np.clip(R, 0, None))
        return Z, TSS_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "hellinger":
        Z = np.sqrt(np.clip(R, 0, None))
        return Z, TSS_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key in {"arcsin_sqrt", "arcsine_sqrt", "arcsine", "arcsin"}:
        Z = np.arcsin(np.sqrt(np.clip(R, 0, 1)))
        return Z, TSS_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "log":
        Z = np.log1p(R)
        return Z, LOG_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "log10":
        Z = np.log10(R + 1.0)
        return Z, LOG_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "log2":
        Z = np.log2(R + 1.0)
        return Z, LOG_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "log_tss_floor":
        Z = np.log(np.clip(R, 1e-10, None))
        lim = max(float(np.nanmax(np.abs(Z))), 1.0)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "zi_log":
        Z = np.zeros_like(R)
        mask = R > 0
        Z[mask] = np.log(R[mask])
        lim = max(float(np.nanmax(np.abs(Z))), 1.0)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "symlog":
        Z = np.sign(R) * np.log1p(np.abs(R))
        return Z, LOG_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key in {"clr", "scikit-bio_clr", "centered_log_ratio"}:
        Z = _clr(R)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key in {"alr", "scikit-bio_alr"}:
        Z = _alr(R)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key in {"ilr", "scikit-bio_ilr"}:
        Z = _ilr(R)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "rclr":
        L = np.where(R > 0, np.log(np.clip(R, 1e-300, None)), 0.0)
        nz = (R > 0).astype(float)
        denom = nz.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        gm = (L * nz).sum(axis=1, keepdims=True) / denom
        Z = np.where(R > 0, L - gm, 0.0)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "bclr":
        D = max(R.shape[1], 1)
        Rp = R + 0.5 / D
        Rp = row_normalise(Rp)
        L = np.log(Rp)
        Z = L - L.mean(axis=1, keepdims=True)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "clr_std":
        # In the real pipeline this is column-standardised using the training fold.
        # The figure is an inference schematic showing a single abundance profile,
        # so column standardisation on one row would collapse to a constant strip.
        # For display only, preserve the log-ratio structure and use row-wise
        # contrast scaling when only one profile is drawn.
        Z0 = _clr(R)
        Z = _row_z(Z0) if Z0.shape[0] == 1 else _col_z(Z0)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "ilr_std":
        # Same display logic as CLR+z: avoid an all-zero one-row strip while
        # retaining the fact that this member uses standardised ILR coordinates.
        Z0 = _ilr(R)
        Z = _row_z(Z0) if Z0.shape[0] == 1 else _col_z(Z0)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "zscore":
        Z = _row_z(R)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "log_std":
        Z = _row_z(np.log1p(R))
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0
    if key == "log_unit":
        L = np.log1p(R)
        norm = np.sqrt((L**2).sum(axis=1, keepdims=True))
        norm[norm < 1e-10] = 1.0
        Z = L / norm
        return Z, LOG_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "rank_std":
        ranks = np.apply_along_axis(rankdata, 1, R).astype(float)
        Z = _row_z(ranks)
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, RANK_CMAP, -1.0, 1.0
    if key == "rank_unit":
        ranks = np.apply_along_axis(rankdata, 1, R).astype(float)
        norm = np.sqrt((ranks**2).sum(axis=1, keepdims=True))
        norm[norm < 1e-10] = 1.0
        Z = ranks / norm
        return Z, RANK_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "rank_frac":
        ranks = np.apply_along_axis(rankdata, 1, R).astype(float)
        Z = ranks / (R.shape[1] + 1.0)
        return Z, RANK_CMAP, 0.0, 1.0
    if key == "rank_col":
        Z = np.zeros_like(R, dtype=float)
        n = R.shape[0]
        for j in range(R.shape[1]):
            Z[:, j] = rankdata(R[:, j]) / (n + 1.0)
        return Z, RANK_CMAP, 0.0, 1.0
    if key == "prev_weighted":
        prev = (R > 0).mean(axis=0, keepdims=True)
        Z = row_normalise(R * prev)
        return Z, TSS_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "chi_square":
        W = np.sqrt(R + 1e-10)
        Z = R / W
        return Z, TSS_CMAP, 0.0, max(float(np.nanmax(Z)), 0.01)
    if key == "power":
        # In the real pipeline this is fit on the training fold.  The schematic can
        # be asked to draw a single transformed abundance profile, and fitting a
        # column-standardised transformer on one row collapses every displayed
        # feature to zero.  For display-only one-row strips, preserve the intended
        # signed/standardised contrast by scaling across features within the row.
        if R.shape[0] == 1:
            Z = _row_z(np.log1p(R))
            lim = max(float(np.nanmax(np.abs(Z))), 0.1)
            return Z / lim, CLR_CMAP, -1.0, 1.0
        if PowerTransformer is not None:
            try:
                Z = PowerTransformer(
                    method="yeo-johnson", standardize=True
                ).fit_transform(R)
                lim = max(float(np.nanmax(np.abs(Z))), 0.1)
                return Z / lim, CLR_CMAP, -1.0, 1.0
            except Exception:
                pass
        # Fallback when sklearn is unavailable or the transform cannot be fit.
        Z = _col_z(np.log1p(R)) if R.shape[0] > 1 else _row_z(np.log1p(R))
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0

    if key == "robust":
        # Same one-row display issue as Yeo-Johnson: RobustScaler centers each
        # feature column, so a single row becomes all zeros.  Use a within-profile
        # robust contrast only for one-row schematic strips.
        if R.shape[0] == 1:
            med = np.nanmedian(R, axis=1, keepdims=True)
            q75 = np.nanpercentile(R, 75, axis=1, keepdims=True)
            q25 = np.nanpercentile(R, 25, axis=1, keepdims=True)
            iqr = q75 - q25
            iqr[~np.isfinite(iqr) | (iqr < 1e-10)] = 1.0
            Z = (R - med) / iqr
            lim = max(float(np.nanmax(np.abs(Z))), 0.1)
            return Z / lim, CLR_CMAP, -1.0, 1.0
        if RobustScaler is not None:
            try:
                Z = RobustScaler(
                    with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
                ).fit_transform(R)
                lim = max(float(np.nanmax(np.abs(Z))), 0.1)
                return Z / lim, CLR_CMAP, -1.0, 1.0
            except Exception:
                pass
        med = np.nanmedian(R, axis=0, keepdims=True)
        q75 = np.nanpercentile(R, 75, axis=0, keepdims=True)
        q25 = np.nanpercentile(R, 25, axis=0, keepdims=True)
        iqr = q75 - q25
        iqr[~np.isfinite(iqr) | (iqr < 1e-10)] = 1.0
        Z = (R - med) / iqr
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0

    if key == "quantile":
        # QuantileTransformer with one sample has no distribution to learn and
        # visually collapses.  For a one-profile schematic, show feature ranks
        # within the profile so QNorm/quantile members still have a gradient.
        if R.shape[0] == 1:
            ranks = np.apply_along_axis(rankdata, 1, R).astype(float)
            Z = _row_z(ranks)
            lim = max(float(np.nanmax(np.abs(Z))), 0.1)
            return Z / lim, RANK_CMAP, -1.0, 1.0
        if QuantileTransformer is not None:
            try:
                q = QuantileTransformer(
                    output_distribution="normal",
                    random_state=seed,
                    n_quantiles=max(2, min(100, R.shape[0])),
                )
                Z = q.fit_transform(R)
                lim = max(float(np.nanmax(np.abs(Z))), 0.1)
                return Z / lim, CLR_CMAP, -1.0, 1.0
            except Exception:
                pass
        Z = np.zeros_like(R, dtype=float)
        n = R.shape[0]
        for j in range(R.shape[1]):
            Z[:, j] = rankdata(R[:, j]) / (n + 1.0)
        return Z, RANK_CMAP, 0.0, 1.0
    if key == "pairwise_logratio":
        Rp = _mr(R)
        D = Rp.shape[1]
        if D < 2:
            Z = Rp
        else:
            rng = np.random.default_rng(seed)
            pairs = np.array(np.triu_indices(D, k=1)).T
            if len(pairs) > 500:
                pairs = pairs[rng.choice(len(pairs), 500, replace=False)]
            Z = np.log(Rp[:, pairs[:, 0]] / Rp[:, pairs[:, 1]])
        lim = max(float(np.nanmax(np.abs(Z))), 0.1)
        return Z / lim, CLR_CMAP, -1.0, 1.0

    # Conservative fallback: row-normalised abundance.
    return R, ABUND_CMAP, 0.0, max(float(np.nanmax(R)), 0.01)


def select_demo_rows(X: np.ndarray, max_rows: int, seed: int = 42) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    # Deterministic evenly spaced sampling keeps case/control ordering if cases are first.
    idx = np.linspace(0, X.shape[0] - 1, max_rows, dtype=int)
    return X[idx]


def compress_features_for_display(Z: np.ndarray, max_features: int) -> np.ndarray:
    if Z.shape[1] <= max_features:
        return Z
    idx = np.linspace(0, Z.shape[1] - 1, max_features, dtype=int)
    return Z[:, idx]


# ---------------------------------------------------------------------------
# Figure drawing
# ---------------------------------------------------------------------------


def draw_heatmap(
    ax: plt.Axes,
    Z: np.ndarray,
    x0: float,
    y0: float,
    w: float,
    h: float,
    cmap,
    vmin: float,
    vmax: float,
) -> None:
    if Z.size == 0:
        Z = np.zeros((1, 1))
    ax.imshow(
        Z,
        extent=(x0, x0 + w, y0 + h, y0),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        zorder=10,
    )
    ax.add_patch(
        mpl.patches.Rectangle(
            (x0, y0), w, h, fill=False, ec="#e2e8f0", lw=0.25, zorder=11
        )
    )


def text_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = ACC_L,
    ec: str = ACC,
    color: str = ACC,
    fs: float = 6.3,
    weight: str | None = None,
    wrap_width: int = 20,
) -> None:
    label = str(text)
    if "\n" not in label:
        label = "\n".join(
            textwrap.wrap(
                label, width=wrap_width, break_long_words=False, break_on_hyphens=False
            )
        )
    ax.add_patch(
        mpl.patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.7",
            fc=fc,
            ec=ec,
            lw=0.35,
            zorder=8,
        )
    )
    ax.text(
        x + 1.2,
        y + h / 2,
        label,
        ha="left",
        va="center",
        fontsize=fs,
        color=color,
        weight=weight,
        linespacing=1.02,
        zorder=9,
    )


def wrap_plain(text: str, width: int) -> str:
    return "\n".join(
        textwrap.wrap(
            str(text), width=width, break_long_words=False, break_on_hyphens=False
        )
    )


def panel_height(n_units: int, unit_h: float = 8.4, gap: float = 1.4) -> float:
    """Panel height in pseudo-mm units.

    The height is driven by the number of shown ensemble rows, but all panels in
    the same row of the final figure are rendered at the same data scale.  That
    avoids the visual bug where shorter ensembles had larger classifier boxes.
    """
    title_h = 10.0
    raw_h = 11.0
    units_h = n_units * unit_h + max(0, n_units - 1) * gap
    output_h = 14.0
    return title_h + raw_h + 5.0 + units_h + output_h + 2.0


def aggregation_endpoint_label(agg: Any) -> str:
    """Label for the fitted inference-time aggregation rule.

    Super learners are not trained in this schematic; the figure depicts applying
    the fitted aggregation rule/meta-learner learned during model selection.
    """
    a = str(agg or "?")
    mapping = {
        "mean_proba": "mean\nprobability",
        "weighted_mean_proba": "weighted\nprobability\naverage",
        "median_proba": "median\nprobability",
        "trimmed_mean": "trimmed\nmean",
        "geometric_mean": "geometric\nmean",
        "log_odds_mean": "mean\nlog-odds",
        "harmonic_mean": "harmonic\nmean",
        "minmax": "min-max\naverage",
        "rank_mean": "mean\nrank",
        "borda_count": "Borda\ncount",
        "copeland": "Copeland\nrank",
        "majority_vote": "majority\nvote",
        "weighted_vote": "weighted\nvote",
        "confidence_weighted": "confidence-\nweighted",
        "softmax_mean": "softmax\nmean",
        "bayesian_avg": "Bayesian\naverage",
        "power_mean_p3": "power\nmean p=3",
        "power_mean_p05": "power\nmean p=0.5",
        "dempster_shafer": "Dempster-\nShafer",
        "max_proba": "maximum\nprobability",
        "min_proba": "minimum\nprobability",
        "superlearner__lr": "fitted logistic\nmeta-learner",
        "superlearner__ridge": "fitted ridge\nmeta-learner",
        "superlearner__rf": "fitted RF\nmeta-learner",
    }
    return mapping.get(
        a,
        "\n".join(textwrap.wrap(a.replace("_", " "), width=14, break_long_words=False)),
    )


def visual_abundance_profile(n_features: int, seed: int = 42) -> np.ndarray:
    """Create one stylised abundance profile for the ensemble schematic.

    One input profile is propagated through each selected MPMA-E member. The
    dimensionality p is real for the rank set, while the display profile is
    deterministic and high-contrast so transformed strips remain visible.
    """
    n = max(int(n_features), 1)
    rng = np.random.default_rng(seed % (2**31))
    row = np.zeros(n, dtype=float)
    mask = rng.random(n) >= 0.50
    vals = rng.random(n) ** 1.8
    row[mask] = vals[mask]
    # Ensure that very small p still has visible structure.
    if n <= 3 and row.sum() == 0:
        row[rng.integers(0, n)] = 1.0
    return row.reshape(1, -1)


def stable_member_seed(member: MemberRecord, base_seed: int) -> int:
    key = f"{member.config_id or ''}|{member.raw_resolution or ''}|{member.raw_transform or ''}|{member.raw_model or ''}|{member.order or 0}"
    # Python's built-in hash is intentionally randomized by process; use a stable checksum.
    import hashlib

    return (base_seed + int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)) % (2**31)


def draw_panel(
    ax: plt.Axes,
    task: EnsembleTask,
    max_demo_rows: int,
    max_features: int,
    seed: int,
    panel_h: float | None = None,
    panel_w: float = 90.0,
) -> None:
    # Inference schematic: show one abundance profile propagated through the selected ensemble.
    n_units = len(task.shown_members)
    unit_h = 8.4
    gap = 1.4
    # Keep the panel canvas close to the actually used schematic width.
    # v6 used 120 mm, which left large empty right margins inside each panel;
    W = float(panel_w)
    H = panel_h if panel_h is not None else panel_height(n_units, unit_h, gap)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Geometry uses a compact heatmap strip, labels below the strip, fixed-size
    # classifier cards, and a final aggregation endpoint inside the panel.
    trunk_x = 4.0
    strip_x = 8.0
    strip_w = 33.0
    learner_x = 51.0
    learner_w = 25.0
    merge_x = 82.5
    agg_w = learner_w
    yhat_x = strip_x + strip_w + 2.0  # align output with the input x symbol
    strip_h = 2.0

    title_y = 1.5
    ax.text(
        0.0,
        title_y,
        f"MPMA-E\n{task.spec.title} ({task.spec.key})",
        ha="left",
        va="top",
        fontsize=8.4,
        weight="bold",
        color=INK,
        linespacing=1.16,
    )
    best = task.selected.get("inner_val_best_ensemble", task.selected)

    # Raw abundance profile x.  Use the same high-contrast one-profile visual
    # visual language for the panel rather than a nearly blank row from a
    # particular biological sample.
    raw_y = 12.5
    X_raw_vis = visual_abundance_profile(task.X.shape[1], seed=7)
    X_raw_vis = compress_features_for_display(
        row_normalise(X_raw_vis), max_features=max_features
    )
    draw_heatmap(
        ax,
        X_raw_vis,
        strip_x,
        raw_y,
        strip_w,
        strip_h,
        ABUND_CMAP,
        0.0,
        max(float(np.nanmax(X_raw_vis)), 0.01),
    )
    ax.text(
        strip_x + strip_w + 2.0,
        raw_y + strip_h / 2,
        r"$\mathbf{x}$",
        ha="left",
        va="center",
        fontsize=12,
        color=INK,
    )
    ax.text(
        strip_x,
        raw_y + strip_h + 3.0,
        f"raw abundance  p={task.X.shape[1]:,}",
        ha="left",
        va="top",
        fontsize=6.6,
        color=MID,
    )

    # Dashed input trunk. The raw abundance profile is the input, so no arrowhead points into it.
    y0 = 27.5
    first_y = y0 + strip_h / 2
    last_member_y = (
        y0 + (n_units - 1) * (unit_h + gap) + strip_h / 2 if n_units else first_y
    )
    ax.plot(
        [trunk_x, trunk_x],
        [raw_y + strip_h / 2, last_member_y],
        color=TRACK,
        lw=0.65,
        ls=(0, (4, 3)),
        zorder=1,
    )
    # No arrowhead into the raw abundance vector: x is the input.  The dashed
    # trunk shows that this input profile is propagated to the transformed
    # member-specific profiles below.
    ax.plot(
        [trunk_x, strip_x - 0.5],
        [raw_y + strip_h / 2, raw_y + strip_h / 2],
        color=TRACK,
        lw=0.65,
        ls=(0, (4, 3)),
        zorder=1,
    )

    # Ensemble members.
    for row_i, member in enumerate(task.shown_members):
        y = y0 + row_i * (unit_h + gap)
        cy = y + strip_h / 2
        if member.is_ellipsis:
            ax.add_patch(
                mpl.patches.Rectangle(
                    (strip_x, y - 0.35),
                    merge_x - strip_x,
                    4.9,
                    fc=ELLIPSIS_BG,
                    ec="#e2e8f0",
                    lw=0.25,
                    zorder=2,
                )
            )
            ax.text(
                (strip_x + merge_x) / 2,
                y + 2.05,
                "...",
                ha="center",
                va="center",
                fontsize=8.5,
                color=MID,
                style="italic",
                zorder=3,
            )
            continue

        X_rank, features = aggregate_to_levels(task.X, task.taxa, member.levels)
        member.n_features = X_rank.shape[1]
        # One profile with true p for this rank set; then apply the real stored transform.
        X_demo = visual_abundance_profile(
            X_rank.shape[1], seed=stable_member_seed(member, seed)
        )
        Z, cmap, vmin, vmax = apply_transform(
            X_demo, member.raw_transform, seed=seed + int(member.order or 0)
        )
        Z = compress_features_for_display(Z, max_features=max_features)
        draw_heatmap(ax, Z, strip_x, y, strip_w, strip_h, cmap, vmin, vmax)
        ax.annotate(
            "",
            xy=(strip_x - 0.5, cy),
            xytext=(trunk_x, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=0.55,
                color=TRACK,
                linestyle=(0, (4, 3)),
                mutation_scale=5.5,
            ),
            zorder=1,
        )

        # Operation labels below each profile, with enough vertical space before the next strip.
        ax.text(
            strip_x,
            y + strip_h + 0.9,
            f"{member.ranks_display}  p={X_rank.shape[1]:,}",
            ha="left",
            va="top",
            fontsize=6.2,
            color=MID,
        )
        ax.text(
            strip_x,
            y + strip_h + 3.25,
            member.transform_display,
            ha="left",
            va="top",
            fontsize=6.2,
            color=MID,
        )

        # Learner card and member prediction line.  The card size is fixed for all tasks.
        card_h = 6.4
        text_box(
            ax,
            learner_x,
            y - 1.7,
            learner_w,
            card_h,
            member.classifier_display,
            fs=5.85,
            wrap_width=17,
        )
        ax.annotate(
            "",
            xy=(learner_x - 0.6, cy),
            xytext=(strip_x + strip_w + 0.8, cy),
            arrowprops=dict(arrowstyle="-|>", lw=0.55, color=DIM, mutation_scale=5.5),
            zorder=3,
        )
        ax.plot(
            [learner_x + learner_w, merge_x], [cy, cy], color=TRACK, lw=0.65, zorder=2
        )

    # Merge spine and fitted aggregation-rule endpoint.  The aggregation rule is
    # shown directly below the final displayed member, while the output label is
    # vertically aligned with the input vector label x above.
    if n_units:
        ax.plot(
            [merge_x, merge_x], [first_y, last_member_y], color=TRACK, lw=0.65, zorder=1
        )
        agg_y = y0 + n_units * (unit_h + gap) + 2.6
        agg_h = 7.2
        arrow_y = agg_y + agg_h / 2
        agg_x = learner_x
        # route the merged member predictions down to the fitted aggregation rule
        ax.plot(
            [merge_x, merge_x], [last_member_y, arrow_y], color=TRACK, lw=0.65, zorder=1
        )
        ax.annotate(
            "",
            xy=(agg_x + agg_w + 0.6, arrow_y),
            xytext=(merge_x, arrow_y),
            arrowprops=dict(arrowstyle="-|>", lw=0.65, color=TRACK, mutation_scale=6),
            zorder=2,
        )
        endpoint = aggregation_endpoint_label(best.get("aggregation_strategy"))
        text_box(
            ax,
            agg_x,
            agg_y,
            agg_w,
            agg_h,
            endpoint,
            fc="#ffffff",
            ec=TRACK,
            color=INK,
            fs=5.9,
            weight=None,
            wrap_width=14,
        )
        # compact return arm to the final prediction, aligned with the input x symbol
        ax.annotate(
            "",
            xy=(yhat_x + 4.0, arrow_y),
            xytext=(agg_x - 0.8, arrow_y),
            arrowprops=dict(arrowstyle="-|>", lw=0.65, color=TRACK, mutation_scale=6),
            zorder=2,
        )
        # Avoid SVG text-mode decomposition of mathtext ``\hat{y}`` into a
        # separate combining accent plus ``y``.  Browsers can place that accent
        # incorrectly even though PNG/PDF render correctly.  Use the precomposed
        # Unicode glyph so SVG, PDF, and PNG agree while keeping SVG text live.
        ax.text(yhat_x, arrow_y, "ŷ", ha="left", va="center", fontsize=12.0, color=INK)


def render_figure(
    tasks: list[EnsembleTask],
    out_prefix: Path,
    ncols: int,
    max_demo_rows: int,
    max_features: int,
    seed: int,
    panel_gap_mm: float = 2.0,
    panel_w_mm: float = 90.0,
) -> None:
    if not tasks:
        raise RuntimeError("No tasks to render")
    ncols = max(1, min(ncols, len(tasks)))
    nrows = math.ceil(len(tasks) / ncols)
    panel_w = float(panel_w_mm)
    panel_gap = float(panel_gap_mm)
    heights = [panel_height(len(t.shown_members), unit_h=8.4, gap=1.4) for t in tasks]
    row_heights: list[float] = []
    for r in range(nrows):
        row_heights.append(max(heights[r * ncols : (r + 1) * ncols]))
    fig_w = ncols * panel_w + (ncols - 1) * panel_gap
    fig_h = sum(row_heights) + (nrows - 1) * 8.0
    fig = plt.figure(figsize=(fig_w * MM, fig_h * MM))

    # Manual axes positions in figure fractions.  Each panel in a row receives
    # the same data-height and therefore the same visual scale.
    y_top = fig_h
    for r in range(nrows):
        row_h = row_heights[r]
        y_top -= row_h
        for c in range(ncols):
            idx = r * ncols + c
            if idx >= len(tasks):
                continue
            x = c * (panel_w + panel_gap)
            ax = fig.add_axes(
                [x / fig_w, y_top / fig_h, panel_w / fig_w, row_h / fig_h]
            )
            draw_panel(
                ax,
                tasks[idx],
                max_demo_rows=max_demo_rows,
                max_features=max_features,
                seed=seed + idx,
                panel_h=row_h,
                panel_w=panel_w,
            )
        y_top -= 8.0

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".svg"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)


def write_members_tsv(tasks: list[EnsembleTask], out_path: Path) -> None:
    rows = []
    for task in tasks:
        shown_ids = {m.config_id for m in task.shown_members if m.config_id is not None}
        for m in task.members:
            rows.append(
                {
                    "task": task.spec.key,
                    "task_title": task.spec.title,
                    "experiment_dir": str(task.experiment_dir),
                    "member_order": m.order,
                    "shown_in_figure": m.config_id in shown_ids,
                    "config_id": m.config_id,
                    "ranks": m.ranks_display,
                    "transformation": m.transform_display,
                    "classifier_family": m.classifier_display.replace("\n", " "),
                    "raw_resolution": m.raw_resolution,
                    "raw_levels": m.raw_levels,
                    "raw_transform": m.raw_transform,
                    "raw_model": m.raw_model,
                    "n_features_in_demo": m.n_features,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def _normalise_selected_unit_for_figure(selected: dict[str, Any]) -> dict[str, Any]:
    """Normalise selected-unit metadata for schematic rendering."""
    if "inner_val_best_ensemble" in selected:
        return selected
    out = dict(selected)
    if "inner_val_best_mpmas_ensemble" in selected:
        out["inner_val_best_ensemble"] = selected["inner_val_best_mpmas_ensemble"]
    return out


def write_single_task_mpma_e_figure(
    experiment_dir: Path,
    *,
    task_key: str = "task",
    task_title: str = "MPMA-E",
    X: np.ndarray | None = None,
    taxa: list[str] | None = None,
    source: str = "sweep abundance matrix",
    out_dir: Path | None = None,
    out_name: str = "mpma_e",
    include_inactive_configs: bool = True,
    max_members: int = 20,
    max_features: int = 80,
    seed: int = 42,
) -> dict[str, Path]:
    """Render the selected MPMA-E schematic for one sweep."""
    experiment_dir = Path(experiment_dir)
    out_dir = Path(out_dir) if out_dir is not None else experiment_dir / "figures"
    selected = _normalise_selected_unit_for_figure(read_selected_unit(experiment_dir))
    config_meta = read_config_meta(
        experiment_dir, include_inactive=include_inactive_configs
    )
    members, shown = build_members(selected, config_meta, max_members=max_members)
    if X is None or taxa is None:
        X, taxa, source = synthetic_taxa_matrix(seed=seed)
        source = f"synthetic schematic matrix; real sweep matrix unavailable for figure renderer"
    spec = TaskSpec(str(task_key), str(task_title), (str(experiment_dir),))
    task = EnsembleTask(
        spec=spec,
        experiment_dir=experiment_dir,
        selected=selected,
        members=members,
        shown_members=shown,
        X=np.asarray(X, dtype=float),
        taxa=[str(t) for t in taxa],
        source=source,
        diagnostics={
            "experiment_dir": str(experiment_dir),
            "n_members_total": len(members),
            "n_members_shown": sum(1 for m in shown if not m.is_ellipsis),
            "demo_source": source,
            "selected_inner_val_best_ensemble": selected.get(
                "inner_val_best_ensemble", {}
            ),
        },
    )
    out_prefix = out_dir / out_name
    render_figure(
        [task],
        out_prefix=out_prefix,
        ncols=1,
        max_demo_rows=1,
        max_features=max_features,
        seed=seed,
    )

    tables_dir = experiment_dir / "tables"
    metadata_dir = experiment_dir / "metadata"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    members_tsv = tables_dir / f"{out_name}_members.tsv"
    diagnostics_json = metadata_dir / f"{out_name}_diagnostics.json"

    for stale in (
        out_prefix.with_name(out_prefix.name + "_members.tsv"),
        out_prefix.with_name(out_prefix.name + "_diagnostics.json"),
    ):
        if stale.exists():
            stale.unlink()

    write_members_tsv([task], members_tsv)
    with diagnostics_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {"tasks": {task_key: task.diagnostics}},
            fh,
            indent=2,
            default=str,
            allow_nan=False,
        )
    return {
        "mpma_e_svg": out_prefix.with_suffix(".svg"),
        "mpma_e_pdf": out_prefix.with_suffix(".pdf"),
        "mpma_e_png": out_prefix.with_suffix(".png"),
        "mpma_e_members": members_tsv,
        "mpma_e_diagnostics": diagnostics_json,
    }
