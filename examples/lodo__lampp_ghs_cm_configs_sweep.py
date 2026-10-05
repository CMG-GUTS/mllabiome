from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_ghs
from lightgbm import LGBMClassifier
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP general health status LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-GHS-LODO-CM"

DATA = lampp_ghs.mllabiome()

EVALUATION = lampp_ghs.splits(
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
#     # ("phylum", ("phylum",)),
#     # ("class", ("class",)),
#     # ("order", ("order",)),
#     # ("family", ("family",)),
#     # ("genus", ("genus",)),
#     ("strain", ("strain",)),
#     # ("domain-phylum", ("domain", "phylum")),
#     # ("domain-class", ("domain", "phylum", "class")),
#     # ("domain-order", ("domain", "phylum", "class", "order")),
#     # ("domain-family", ("domain", "phylum", "class", "order", "family")),
#     # ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
#     # ("phylum-class", ("phylum", "class")),
#     # ("phylum-order", ("phylum", "class", "order")),
#     # ("phylum-family", ("phylum", "class", "order", "family")),
#     # ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
#     # ("class-order", ("class", "order")),
#     # ("class-family", ("class", "order", "family")),
#     # ("class-genus", ("class", "order", "family", "genus")),
#     # ("order-family", ("order", "family")),
#     # ("order-genus", ("order", "family", "genus")),
#     # ("family-genus", ("family", "genus")),
#     # ("raw", ("raw",)),
# )

# COUNT_TRANSFORMATIONS = (
#     mll.Transformation("presence_absence"),
#     # mll.Transformation("identity"),
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
#     # mll.Transformation("clr", composition_scope="rank-wise"),
#     # mll.Transformation("clr", composition_scope="joint"),
#     # mll.Transformation("hellinger", composition_scope="rank-wise"),
#     # mll.Transformation("hellinger", composition_scope="joint"),
# )

# MODELS = (
#     (
#         "RF_1000_msl5",
#         RandomForestClassifier(
#             n_estimators=1000,
#             min_samples_leaf=5,
#             n_jobs=1,
#             random_state=42,
#         ),
#     ),
#     # (
#     #     "LR_4e-5_l2",
#     #     LogisticRegression(
#     #         C=4e-5,
#     #         penalty="l2",
#     #         solver="liblinear",
#     #         fit_intercept=True,
#     #         class_weight=None,
#     #         tol=1e-7,
#     #         max_iter=3000,
#     #         random_state=0,
#     #     ),
#     # ),
#     # (
#     #     "LinearSVC_sh",
#     #     LinearSVC(
#     #         C=5.5e-6,
#     #         loss="squared_hinge",
#     #         dual="auto",
#     #         tol=1e-8,
#     #         max_iter=30000,
#     #         random_state=0,
#     #     )
#     # ),
#     (
#         "CNB_1.6e7",
#         ComplementNB(
#             alpha=1.6e-7,
#             norm=True,
#             force_alpha=True,
#         ),
#     ),
#     #  (
#     #         "XGB_600_mcw10",
#     #         XGBClassifier(
#     #             n_estimators=600,
#     #             learning_rate=0.03,
#     #             max_depth=2,
#     #             min_child_weight=10,
#     #             subsample=0.8,
#     #             colsample_bytree=0.7,
#     #             reg_alpha=0.5,
#     #             reg_lambda=2,
#     #             eval_metric="logloss",
#     #             n_jobs=-1,
#     #             random_state=17,
#     #             tree_method="hist",
#     #         ),
#     #     ),
#     #     (
#     #         "XGB_500_mcw20",
#     #         XGBClassifier(
#     #             n_estimators=500,
#     #             learning_rate=0.03,
#     #             max_depth=2,
#     #             min_child_weight=20,
#     #             subsample=0.8,
#     #             colsample_bytree=0.7,
#     #             reg_lambda=2.0,
#     #             reg_alpha=0.5,
#     #             eval_metric="logloss",
#     #             n_jobs=-1,
#     #             random_state=17,
#     #         ),
#     #     ),
#     #     (
#     #         "CB_i300_d3",
#     #         CatBoostClassifier(
#     #             iterations=300,
#     #             learning_rate=0.04,
#     #             depth=3,
#     #             l2_leaf_reg=10,
#     #             random_strength=1.0,
#     #             rsm=0.60,
#     #             loss_function="Logloss",
#     #             eval_metric="Logloss",
#     #             random_seed=42,
#     #             thread_count=1,
#     #             verbose=False,
#     #             allow_writing_files=False,
#     #         ),
#     #     ),
#     #     (
#     #         "CB_i500_d6",
#     #         CatBoostClassifier(
#     #             iterations=500,
#     #             learning_rate=0.03,
#     #             depth=6,
#     #             l2_leaf_reg=5,
#     #             loss_function="Logloss",
#     #             random_strength=1,
#     #             random_seed=17,
#     #             thread_count=-1,
#     #             verbose=False,
#     #         ),
#     #     ),
#     #     (
#     #         "CB_i1600_d2",
#     #         CatBoostClassifier(
#     #             iterations=1600,
#     #             learning_rate=0.03,
#     #             depth=2,
#     #             l2_leaf_reg=5.03,
#     #             random_strength=2.8,
#     #             border_count=128,
#     #             rsm=0.6,
#     #             bootstrap_type="Bernoulli",
#     #             subsample=0.6,
#     #             loss_function="Logloss",
#     #             random_seed=17,
#     #             verbose=False,
#     #             allow_writing_files=False,
#     #         ),
#     #     ),
#     #     (
#     #         "HGB_i300_msl20",
#     #         HistGradientBoostingClassifier(
#     #             max_iter=300,
#     #             learning_rate=0.05,
#     #             max_leaf_nodes=15,
#     #             min_samples_leaf=20,
#     #             l2_regularization=0,
#     #             random_state=17,
#     #         ),
#     #     ),
#     #     (
#     #         "GB_e100",
#     #         GradientBoostingClassifier(
#     #             n_estimators=100,
#     #             learning_rate=0.1,
#     #             loss="log_loss",
#     #             max_features="sqrt",
#     #             subsample=0.8,
#     #             random_state=17,
#     #         ),
#     #     ),
# )


RESOLUTIONS = (
    # ("phylum", ("phylum",)),
    # ("class", ("class",)),
    # ("order", ("order",)),
    # ("family", ("family",)),
    ("genus", ("genus",)),
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
    ("genus-species", ("genus", "species")),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("clr", composition_scope="rank-wise"),
    mll.Transformation("clr", composition_scope="joint"),
    mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    mll.Transformation("yeo_johnson", composition_scope="joint"),
    mll.Transformation(
        "clr",
        composition_scope="rank-wise",
        feature_filter=mll.PrevalenceFilter(threshold=0.20),
    ),
    mll.Transformation(
        "clr",
        composition_scope="rank-wise",
        feature_filter=mll.PrevalenceFilter(threshold=0.05),
    ),
    mll.Transformation(
        "relative_abundance",
        composition_scope="joint",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.20,
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
    optimize_metric="MCC",
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

EXTERNAL_TEST = lampp_ghs.external_test

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
