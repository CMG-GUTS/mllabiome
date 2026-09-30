from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_ibd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP inflammatory bowel disease LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-IBD-LODO-CM"

DATA = lampp_ibd.mllabiome()


EVALUATION = lampp_ibd.splits(
    benchmark="mllabiome-benchmark-v1",
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
    # ("class", ("class",)),
    # ("genus", ("genus",)),
    # ("species", ("species",)),
    # ("class-order", ("class", "order")),
    ("strain", ("strain",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    # mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    # mll.Transformation("yeo_johnson", composition_scope="joint"),
    # mll.Transformation(
    #     "relative_abundance",
    #     composition_scope="joint",
    #     feature_filter=mll.PrevalenceFilter(
    #         threshold=0.20,
    #     ),
    # ),
    # mll.Transformation("log10", composition_scope="rank-wise"),
)

MODELS = (
    (
        "LinearSVC_sh",
        LinearSVC(
            C=5.5e-6,
            loss="squared_hinge",
            dual="auto",
            tol=1e-8,
            max_iter=30000,
            random_state=0,
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
    # (
    #     "RF_1000_msl5",
    #     RandomForestClassifier(
    #         n_estimators=1000,
    #         min_samples_leaf=5,
    #         n_jobs=1,
    #         random_state=42,
    #     ),
    # ),
    # (
    #     "MLP_h8",
    #     MLPClassifier(
    #         hidden_layer_sizes=(8,),
    #         activation="relu",
    #         solver="adam",
    #         alpha=5e-4,
    #         learning_rate_init=1e-2,
    #         max_iter=15,
    #         shuffle=False,
    #         random_state=42,
    #         early_stopping=False,
    #         tol=0.0,
    #         n_iter_no_change=100,
    #     ),
    # ),
    # (
    #     "XGB_600_mcw10",
    #     XGBClassifier(
    #         n_estimators=600,
    #         learning_rate=0.03,
    #         max_depth=2,
    #         min_child_weight=10,
    #         subsample=0.8,
    #         colsample_bytree=0.7,
    #         reg_alpha=0.5,
    #         reg_lambda=2,
    #         eval_metric="logloss",
    #         n_jobs=-1,
    #         random_state=17,
    #         tree_method="hist",
    #     ),
    # ),
    # (
    #     "CB_abundance_i300_d3",
    #     CatBoostClassifier(
    #         iterations=300,
    #         learning_rate=0.04,
    #         depth=3,
    #         l2_leaf_reg=10,
    #         random_strength=1.0,
    #         rsm=0.60,
    #         loss_function="Logloss",
    #         eval_metric="Logloss",
    #         random_seed=42,
    #         thread_count=1,
    #         verbose=False,
    #         allow_writing_files=False,
    #     ),
    # ),
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
