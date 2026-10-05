from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lightgbm import LGBMClassifier
import pandas as pd
from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_ibd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP inflammatory bowel disease LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-IBD-LODO-CM-official"

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

# RESOLUTIONS = (
#     # ("class", ("class",)),
#     # ("genus", ("genus",)),
#     # ("species", ("species",)),
#     # ("class-order", ("class", "order")),
#     ("strain", ("strain",)),
# )

# COUNT_TRANSFORMATIONS = (
#     mll.Transformation("presence_absence"),
#     mll.Transformation("identity"),
#     # mll.Transformation("arcsine_sqrt", composition_scope="joint"),
#     # mll.Transformation("yeo_johnson", composition_scope="joint"),
#     # mll.Transformation(
#     #     "relative_abundance",
#     #     composition_scope="joint",
#     #     feature_filter=mll.PrevalenceFilter(
#     #         threshold=0.20,
#     #     ),
#     # ),
#     # mll.Transformation("log10", composition_scope="rank-wise"),
# )

# MODELS = (
#     (
#         "LinearSVC_sh",
#         LinearSVC(
#             C=5.5e-6,
#             loss="squared_hinge",
#             dual="auto",
#             tol=1e-8,
#             max_iter=30000,
#             random_state=0,
#         ),
#     ),
#     (
#         "LR_4e-5_l2",
#         LogisticRegression(
#             C=4e-5,
#             penalty="l2",
#             solver="liblinear",
#             fit_intercept=True,
#             class_weight=None,
#             tol=1e-7,
#             max_iter=3000,
#             random_state=0,
#         ),
#     ),
#     # (
#     #     "RF_1000_msl5",
#     #     RandomForestClassifier(
#     #         n_estimators=1000,
#     #         min_samples_leaf=5,
#     #         n_jobs=1,
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "MLP_h8",
#     #     MLPClassifier(
#     #         hidden_layer_sizes=(8,),
#     #         activation="relu",
#     #         solver="adam",
#     #         alpha=5e-4,
#     #         learning_rate_init=1e-2,
#     #         max_iter=15,
#     #         shuffle=False,
#     #         random_state=42,
#     #         early_stopping=False,
#     #         tol=0.0,
#     #         n_iter_no_change=100,
#     #     ),
#     # ),
#     # (
#     #     "XGB_600_mcw10",
#     #     XGBClassifier(
#     #         n_estimators=600,
#     #         learning_rate=0.03,
#     #         max_depth=2,
#     #         min_child_weight=10,
#     #         subsample=0.8,
#     #         colsample_bytree=0.7,
#     #         reg_alpha=0.5,
#     #         reg_lambda=2,
#     #         eval_metric="logloss",
#     #         n_jobs=-1,
#     #         random_state=17,
#     #         tree_method="hist",
#     #     ),
#     # ),
#     # (
#     #     "CB_abundance_i300_d3",
#     #     CatBoostClassifier(
#     #         iterations=300,
#     #         learning_rate=0.04,
#     #         depth=3,
#     #         l2_leaf_reg=10,
#     #         random_strength=1.0,
#     #         rsm=0.60,
#     #         loss_function="Logloss",
#     #         eval_metric="Logloss",
#     #         random_seed=42,
#     #         thread_count=1,
#     #         verbose=False,
#     #         allow_writing_files=False,
#     #     ),
#     # ),
# )


RESOLUTIONS = (
    # ("phylum", ("phylum",)),
    # ("class", ("class",)),
    # ("order", ("order",)),
    # ("family", ("family",)),
    # ("genus", ("genus",)),
    ("species", ("species",)),
    ("strain", ("strain",)),
    # # # ("domain-phylum", ("domain", "phylum")),
    # # ("domain-class", ("domain", "phylum", "class")),
    # # # ("domain-order", ("domain", "phylum", "class", "order")),
    # # ("domain-family", ("domain", "phylum", "class", "order", "family")),
    # ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    # (
    #     "domain-species",
    #     ("domain", "phylum", "class", "order", "family", "genus", "species"),
    # ),
    # ("phylum-class", ("phylum", "class")),
    # ("phylum-order", ("phylum", "class", "order")),
    # # ("phylum-family", ("phylum", "class", "order", "family")),
    # # ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
    ("phylum-species", ("phylum", "class", "order", "family", "genus", "species")),
    # # ("class-order", ("class", "order")),
    # # ("class-family", ("class", "order", "family")),
    # ("class-genus", ("class", "order", "family", "genus")),
    # # ("class-species", ("class", "order", "family", "genus", "species")),
    # ("order-family", ("order", "family")),
    # ("order-genus", ("order", "family", "genus")),
    # ("order-species", ("order", "family", "genus", "species")),
    # ("family-genus", ("family", "genus")),
    ("family-species", ("family", "genus", "species")),
    # ("genus-species", ("genus", "species")),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("clr", composition_scope="rank-wise"),
    mll.Transformation("clr", composition_scope="joint"),
    mll.Transformation("arcsine_sqrt", composition_scope="rank-wise"),
    mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    mll.Transformation("yeo_johnson", composition_scope="joint"),
    # mll.Transformation(
    #     "clr",
    #     composition_scope="rank-wise",
    #     feature_filter=mll.PrevalenceFilter(threshold=0.20),
    # ),
    # mll.Transformation(
    #     "clr",
    #     composition_scope="rank-wise",
    #     feature_filter=mll.PrevalenceFilter(threshold=0.05),
    # ),
    mll.Transformation(
        "relative_abundance",
        composition_scope="joint",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.20,
        ),
    ),
    mll.Transformation(
        "identity",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.10,
        ),
    ),
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
        "XGB_e300_mcw0.25",
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=1,
            min_child_weight=0.25,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=0.725,
            tree_method="hist",
            random_state=17,
        ),
    ),
    (
        "LGBM_e300",
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.07,
            num_leaves=7,
            max_depth=3,
            min_child_samples=3,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0,
            reg_lambda=0,
            scale_pos_weight=1.0,
            random_state=17,
            verbosity=-1,
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
        "CB_i1600_d2",
        CatBoostClassifier(
            iterations=1600,
            learning_rate=0.03,
            depth=2,
            l2_leaf_reg=5.03,
            random_strength=2.8,
            border_count=128,
            rsm=0.6,
            bootstrap_type="Bernoulli",
            subsample=0.6,
            loss_function="Logloss",
            random_seed=17,
            verbose=False,
            allow_writing_files=False,
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
        "XGB_600_mcw10",
        XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=17,
            tree_method="hist",
        ),
    ),
    (
        "SVC_RBF_C1p5_g1p825_w1p25",
        SVC(
            C=1.5,
            kernel="rbf",
            gamma=1.825,
            class_weight={0: 1.0, 1: 1.25},
            cache_size=4000,
        ),
    ),
    (
        "XGB_lossguide_15l",
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=400,
            learning_rate=0.03,
            grow_policy="lossguide",
            max_depth=0,
            max_leaves=15,
            min_child_weight=15,
            subsample=0.85,
            colsample_bytree=0.50,
            reg_alpha=0.5,
            reg_lambda=5,
            random_state=17,
            n_jobs=-1,
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
