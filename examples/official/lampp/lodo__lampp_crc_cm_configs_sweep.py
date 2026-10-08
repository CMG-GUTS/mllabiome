from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_crc
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP colorectal cancer LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-CRC-LODO-CM-official-v3"

DATA = lampp_crc.mllabiome()

EVALUATION = lampp_crc.splits(
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
    ("species", ("species",)),
    ("strain", ("strain",)),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (mll.Transformation("presence_absence"),)

MODELS = (
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
        "RF_82_msl4",
        RandomForestClassifier(
            n_estimators=82,
            max_features=52,
            min_samples_leaf=4,
            min_samples_split=2,
            class_weight="balanced_subsample",
            criterion="gini",
            bootstrap=True,
            random_state=42,
        ),
    ),
    (
        "LightGBM_e500",
        LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=40,
            reg_lambda=30.0,
            reg_alpha=1.0,
            colsample_bytree=0.80,
            subsample=0.80,
            subsample_freq=1,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
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

EXTERNAL_TEST = lampp_crc.external_test

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
