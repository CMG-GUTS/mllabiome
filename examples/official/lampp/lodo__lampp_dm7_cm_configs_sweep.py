from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_dm7
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP delivery mode <=7 days LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-DM7-LODO-CM-v2"

DATA = lampp_dm7.mllabiome()

EVALUATION = lampp_dm7.splits(
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
    ("raw", ("raw",)),
    ("strain", ("strain",)),
    ("phylum-order", ("phylum", "class", "order")),
    ("phylum-species", ("phylum", "class", "order", "family", "genus", "species")),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("clr", composition_scope="joint"),
    mll.Transformation("yeo_johnson", composition_scope="joint"),
    mll.Transformation("log10", composition_scope="rank-wise"),
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
    (
        "RF_1000_msl5",
        RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "CB_i300_d3",
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.04,
            depth=3,
            l2_leaf_reg=10,
            random_strength=1.0,
            rsm=0.60,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
    (
        "LGBM_e180",
        LGBMClassifier(
            n_estimators=180,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            colsample_bytree=0.50,
            reg_lambda=2.0,
            reg_alpha=0.1,
            class_weight="balanced",
            random_state=42,
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

EXTERNAL_TEST = lampp_dm7.external_test

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
