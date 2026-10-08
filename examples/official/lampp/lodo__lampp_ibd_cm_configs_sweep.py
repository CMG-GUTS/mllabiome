from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_ibd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP inflammatory bowel disease LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-IBD-LODO-CM-official-v2-fm"

DATA = lampp_ibd.mllabiome()


EVALUATION = lampp_ibd.splits(
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
    ("genus", ("genus",)),
    ("species", ("species",)),
    ("strain", ("strain",)),
    ("order-genus", ("order", "family", "genus")),
    ("genus-species", ("genus", "species")),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("clr", composition_scope="joint"),
    mll.Transformation("yeo_johnson", composition_scope="joint"),
    mll.Transformation("log10", composition_scope="rank-wise"),
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
        "SIAMCAT",
        mll.SIAMCATClassifier(),
    ),
    (
        "LDA_eigen_shrinkage_auto",
        LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage="auto",
        ),
    ),
    (
        "LinearSVC_C1e-4",
        LinearSVC(
            C=1e-4,
            loss="squared_hinge",
            dual="auto",
            max_iter=30000,
            random_state=17,
        ),
    ),
    (
        "LR_4e-5_l2",
        LogisticRegression(
            C=4e-5,
            penalty="l2",
            solver="liblinear",
            fit_intercept=True,
            class_weight=None,
            tol=1e-7,
            max_iter=3000,
            random_state=0,
        ),
    ),
    (
        "LR_c1_balanced_l2",
        LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=42,
        ),
    ),
    (
        "TabPFN",
        mll.TabPFNClassifier(
            n_estimators="auto",
            auto_scale_n_estimators=True,
            model_path="auto",
            device="auto",
            inference_precision="auto",
            fit_mode="fit_preprocessors",
            memory_saving_mode="auto",
            keep_cache_on_device=True,
            random_state=42,
            n_preprocessing_jobs=1,
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
        "quality_diversity",
        "diversity",
        "performance_diversity",
        "caruana",
        "bagged_caruana",
        "caruana_multistart",
        "beam",
        "super_learner",
        "adaptive_super_learner",
        "safe_super_learner",
        "regularized_super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "trimmed_mean_proba",
        "trimmed_mean",
        "geometric_mean_proba",
        "logit_mean",
        "logistic_stacking",
        "elastic_net_stacking",
        "cohort_robust_stacking",
        "rf_stacking",
        "extra_trees_stacking",
        "boosted_stacking",
        "gated_stacking",
        "hierarchical_dirichlet_stacking",
        "temperature_scaled_mean_proba",
        "sigmoid_calibrated_mean_proba",
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

EXTERNAL_TEST = lampp_ibd.external_test

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
