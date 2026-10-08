from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_dm90
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP delivery mode <=90 days LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-DM90-LODO-CM-official-v2"

DATA = lampp_dm90.mllabiome()

EVALUATION = lampp_dm90.splits(
    benchmark="mllabiome-benchmark-v2",
).mllabiome(
    optimize_metric="log_loss",
    n_jobs="auto",
)

EXPLORE = mll.Explore(
    ranks=("genus",),
    top_taxa=12,
    heatmap_top=30,
    min_prevalence=0.10,
    differential_abundance="off",
    detection_limit=0.0,
    permutations=999,
    bootstrap_replicates=2000,
    confidence_level=0.95,
    random_state=42,
)


RESOLUTIONS = (
    ("strain", ("strain",)),
    ("phylum-order", ("phylum", "class", "order")),
    ("phylum-species", ("phylum", "class", "order", "family", "genus", "species")),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation(
        "identity",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.10,
        ),
    ),
    mll.Transformation(
        "clr",
        composition_scope="rank-wise",
        feature_filter=mll.PrevalenceFilter(threshold=0.20),
    ),
)

MODELS = (
    (
        "GB_e100",
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            loss="log_loss",
            max_features="sqrt",
            subsample=0.8,
            random_state=17,
        ),
    ),
    (
        "LR_C10_balanced",
        LogisticRegression(
            C=10.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
        ),
    ),
    (
        "LinearSVC_C1",
        LinearSVC(
            C=1.0,
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            tol=1e-4,
            fit_intercept=True,
            class_weight={0: 1.0, 1: 1.4},
            random_state=42,
            max_iter=10000,
        ),
    ),
)

GATE = mll.QualificationGate(
    enabled=False,
    metric="MCC",
    threshold=0.02,
)

ENSEMBLE = mll.Ensemble(
    max_sizes=(10,),
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
    ),
    optimize_metric="log_loss",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    profile="screening",
    methods=(
        mll.Permutation(),
        mll.SHAP(),
        mll.ALE(),
        mll.LIME(),
        mll.ALEInteractions(),
    ),
    classes="auto",
)

ROBUSTNESS = mll.Robustness(
    targets=("mpma_b",),
    top_k=30,
)

EXTERNAL_TEST = lampp_dm90.external_test

if EXTERNAL_TEST is None:
    raise RuntimeError("LAMPP external test set is unavailable")

INFERENCE = EXTERNAL_TEST.mllabiome(
    targets=("mpma_b", "mpma_e"),
    feature_policy="strict",
)


SWEEP = mll.Sweep(
    data=DATA,
    experiment_dir=EXPERIMENT_DIR,
    title=TITLE,
    resolutions=RESOLUTIONS,
    count_transformations=COUNT_TRANSFORMATIONS,
    learners=MODELS,
    evaluation=EVALUATION,
    gate=GATE,
    explore=EXPLORE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
    robustness=ROBUSTNESS,
    inference=INFERENCE,
)
