from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_crc
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP colorectal cancer LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-CRC-LODO-CM-official"

DATA = lampp_crc.mllabiome()

EVALUATION = lampp_crc.splits(
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
    # # ("phylum-species", ("phylum", "class", "order", "family", "genus", "species")),
    # # ("class-order", ("class", "order")),
    # # ("class-family", ("class", "order", "family")),
    # ("class-genus", ("class", "order", "family", "genus")),
    # # ("class-species", ("class", "order", "family", "genus", "species")),
    # ("order-family", ("order", "family")),
    # ("order-genus", ("order", "family", "genus")),
    # ("order-species", ("order", "family", "genus", "species")),
    # ("family-genus", ("family", "genus")),
    # ("family-species", ("family", "genus", "species")),
    # ("genus-species", ("genus", "species")),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("clr", composition_scope="rank-wise"),
    # mll.Transformation("clr", composition_scope="joint"),
    # mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    # mll.Transformation("yeo_johnson", composition_scope="joint"),
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

#     (
#         "MLP_h4_a5_lr1e4",
#         Pipeline(
#             steps=(
#                 ("scaler", StandardScaler()),
#                 (
#                     "mlp",
#                     MLPClassifier(
#                         hidden_layer_sizes=(4,),
#                         activation="relu",
#                         solver="adam",
#                         alpha=5.0,
#                         batch_size="auto",
#                         learning_rate_init=1e-4,
#                         max_iter=1000,
#                         early_stopping=False,
#                         random_state=42,
#                     ),
#                 ),
#             ),
#         ),
#     ),
#  (
#         "MLP_h4_a5_lr2e4",
#         Pipeline(
#             steps=(
#                 ("scaler", StandardScaler()),
#                 (
#                     "mlp",
#                     MLPClassifier(
#                         hidden_layer_sizes=(4,),
#                         activation="relu",
#                         solver="adam",
#                         alpha=5.0,
#                         batch_size="auto",
#                         learning_rate_init=2e-4,
#                         max_iter=1000,
#                         early_stopping=False,
#                         random_state=42,
#                     ),
#                 ),
#             ),
#         ),
#     ),
#     (
#         "MLP_h4_relu_adam_a5",
#         Pipeline(
#             steps=(
#                 ("scaler", StandardScaler()),
#                 (
#                     "mlp",
#                     MLPClassifier(
#                         hidden_layer_sizes=(4,),
#                         activation="relu",
#                         solver="adam",
#                         alpha=5.0,
#                         batch_size=64,
#                         learning_rate_init=3e-4,
#                         max_iter=500,
#                         early_stopping=False,
#                         random_state=42,
#                     ),
#                 ),
#             ),
#         ),
#     ),
#     (
#         "Logistic_L2_C0002",
#         Pipeline(
#             steps=(
#                 ("scaler", StandardScaler()),
#                 (
#                     "logistic",
#                     LogisticRegression(
#                         penalty="l2",
#                         C=0.002,
#                         solver="liblinear",
#                         max_iter=5000,
#                         random_state=42,
#                     ),
#                 ),
#             ),
#         ),
#     ),
#     (
#         "RF_1000_msl5",
#         RandomForestClassifier(
#             n_estimators=1000,
#             min_samples_leaf=5,
#             n_jobs=1,
#             random_state=42,
#         ),
#     ),
#     (
#         "CB_abundance_i300_d3",
#         CatBoostClassifier(
#             iterations=300,
#             learning_rate=0.04,
#             depth=3,
#             l2_leaf_reg=10,
#             random_strength=1.0,
#             rsm=0.60,
#             loss_function="Logloss",
#             eval_metric="Logloss",
#             random_seed=42,
#             thread_count=1,
#             verbose=False,
#             allow_writing_files=False,
#         ),
#     ),
#     (
#         "LightGBM_e500",
#         LGBMClassifier(
#             objective="binary",
#             n_estimators=500,
#             learning_rate=0.03,
#             num_leaves=7,
#             min_child_samples=40,
#             reg_lambda=30.0,
#             reg_alpha=1.0,
#             colsample_bytree=0.80,
#             subsample=0.80,
#             subsample_freq=1,
#             random_state=42,
#             n_jobs=1,
#             verbosity=-1,
#         ),
#     ),
#     (
#         "GB_e100",
#         GradientBoostingClassifier(
#             n_estimators=100,
#             learning_rate=0.1,
#             loss="log_loss",
#             max_features="sqrt",
#             subsample=0.8,
#             random_state=17,
#         ),
#     ),
# )

# RESOLUTIONS = (
#     ("class", ("class",)),
#     ("genus", ("genus",)),
#     ("species", ("species",)),
#     ("strain", ("strain",)),
#     ("class-order", ("class", "order")),
#     ("family-genus", ("family", "genus")),
#     ("family-species", ("family", "genus", "species")),
#     ("genus-species", ("genus", "species")),
#     ("raw", ("all",)),
# )

# COUNT_TRANSFORMATIONS = (
#     mll.Transformation("presence_absence"),
#     mll.Transformation("identity"),
#     mll.Transformation("arcsine_sqrt", composition_scope="joint"),
#     mll.Transformation("yeo_johnson", composition_scope="joint"),
#     mll.Transformation(
#         "relative_abundance",
#         composition_scope="joint",
#         feature_filter=mll.PrevalenceFilter(
#             threshold=0.20,
#         ),
#     ),
#     mll.Transformation("log10", composition_scope="rank-wise"),
#     mll.Transformation("clr", composition_scope="rank-wise"),
#     mll.Transformation("clr", composition_scope="joint"),
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
#     (
#         "RF_82_msl4",
#         RandomForestClassifier(
#             n_estimators=82,
#             max_features=52,
#             min_samples_leaf=4,
#             min_samples_split=2,
#             class_weight="balanced_subsample",
#             criterion="gini",
#             bootstrap=True,
#             random_state=42,
#         ),
#     ),
#     (
#         "RF_82_msl4_1.3e-5",
#         RandomForestClassifier(
#             n_estimators=82,
#             max_features=52,
#             min_samples_leaf=4,
#             min_samples_split=2,
#             max_depth=26,
#             class_weight="balanced_subsample",
#             criterion="gini",
#             bootstrap=True,
#             min_impurity_decrease=1.3e-5,
#             ccp_alpha=0.000325,
#             random_state=42,
#         ),
#     ),
#     (
#         "RF_82_msl4_1.67e-5",
#         RandomForestClassifier(
#             n_estimators=82,
#             max_features=52,
#             min_samples_leaf=4,
#             min_samples_split=2,
#             max_depth=26,
#             class_weight="balanced_subsample",
#             criterion="gini",
#             bootstrap=True,
#             min_impurity_decrease=1.67e-5,
#             ccp_alpha=0.000245,
#             random_state=17,
#         ),
#     ),
#     (
#         "CB_abundance_i300_d3",
#         CatBoostClassifier(
#             iterations=300,
#             learning_rate=0.04,
#             depth=3,
#             l2_leaf_reg=10,
#             random_strength=1.0,
#             rsm=0.60,
#             loss_function="Logloss",
#             eval_metric="Logloss",
#             random_seed=42,
#             thread_count=1,
#             verbose=False,
#             allow_writing_files=False,
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
#     (
#         "RF_n1000_msl1",
#     RandomForestClassifier(
#         n_estimators=1000,
#         max_features="sqrt",
#         min_samples_leaf=1,
#         class_weight={0: 1.0, 1: 1.15},
#         bootstrap=True,
#         random_state=42,
#     ),
#     )
# )

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
