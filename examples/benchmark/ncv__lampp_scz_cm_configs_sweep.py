from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_scz
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP schizophrenia mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-SCZ-NCV-CM-official"

DATA = lampp_scz.mllabiome()

EVALUATION = lampp_scz.splits(
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
    # ("phylum", ("phylum",)),
    # ("class", ("class",)),
    # ("order", ("order",)),
    # ("family", ("family",)),
    # ("genus", ("genus",)),
    # ("species", ("species",)),
    # ("strain", ("strain",)),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    # mll.Transformation("presence_absence"),
    # mll.Transformation("identity"),
    # mll.Transformation("clr", composition_scope="rank-wise"),
    # mll.Transformation(
    #     "arcsine_sqrt",
    #     composition_scope="rank-wise",
    # ),
    # mll.Transformation(
    #     "hellinger",
    #     composition_scope="rank-wise",
    # ),
    # mll.Transformation(
    #     "relative_abundance",
    #     composition_scope="rank-wise",
    # ),
    # mll.Transformation(
    #     "log10",
    #     composition_scope="rank-wise",
    # ),
    # # # # additional
    # mll.Transformation("ilr", composition_scope="rank-wise"),
    # mll.Transformation("alr", composition_scope="rank-wise"),
    mll.Transformation("prev_weighted", composition_scope="rank-wise"),
    # mll.Transformation("rank", composition_scope="rank-wise"),
    # mll.Transformation("quantile", composition_scope="rank-wise"),
    # mll.Transformation(
    #     "yeo_johnson",
    #     composition_scope="rank-wise",
    # ),
    # mll.Transformation("ecdf_rank"),
    # mll.Transformation(
    #     "rclr",
    #     composition_scope="rank-wise",
    # ),
    # mll.Transformation(
    #     "prevalence_rank_fusion",
    #     composition_scope="rank-wise",
    # ),
)


# COUNT_TRANSFORMATIONS = (
#     mll.WaypointEmbedding(
#         model="outpost-bio/Waypoint-6m",
#         pooling="last_token",
#         taxonomy_format="full",
#         resolutions=("genus",),
#         batch_size=16,
#         max_length=512,
#         device=None,
#         cache=True,
#     ),
#     mll.MGMEmbedding(
#         taxonomy_format="full",
#         resolutions=("genus",),
#         batch_size=16,
#         max_length=512,
#         device=None,
#         cache=True,
#     ),
# )


MODELS = (
#     # # baselines
#     (
#         "Baseline_RF",
#         RandomForestClassifier(
#             n_estimators=1000,
#             min_samples_leaf=5,
#             random_state=42,
#             n_jobs=1,
#         ),
#     ),
#     # # (
#     # #     "SIAMCAT",
#     # #     mll.SIAMCATClassifier(
#     # #         random_state=42,
#     # #     ),
#     # # ),
#     # # # default standard models
#     (
#         "CB_default",
#         CatBoostClassifier(
#             random_seed=42,
#             thread_count=1,
#             verbose=False,
#             allow_writing_files=False,
#         ),
#     ),
#     (
#         "RF_default",
#         RandomForestClassifier(
#             random_state=42,
#         ),
#     ),
#     (
#         "LightGBM_default",
#         LGBMClassifier(
#             random_state=42,
#             n_jobs=1,
#             verbosity=-1,
#         ),
#     ),
#     (
#         "XGB_default",
#         XGBClassifier(
#             random_state=42,
#             n_jobs=1,
#             verbosity=0,
#             eval_metric="logloss",
#         ),
#     ),
#     (
#         "LR_default",
#         LogisticRegression(
#             random_state=42,
#         ),
#     ),
#     (
#         "TabPFN",
#         mll.TabPFNClassifier(
#             n_estimators="auto",
#             auto_scale_n_estimators=True,
#             model_path="auto",
#             device="auto",
#             inference_precision="auto",
#             fit_mode="fit_preprocessors",
#             memory_saving_mode="auto",
#             keep_cache_on_device=True,
#             random_state=42,
#             n_preprocessing_jobs=1,
#         ),
#     ),
#     (
#         "TabPFN_n8",
#         mll.TabPFNClassifier(
#             n_estimators=8,
#             model_path="auto",
#             device="auto",
#             inference_precision="auto",
#             fit_mode="fit_preprocessors",
#             memory_saving_mode="auto",
#             keep_cache_on_device=True,
#             ignore_pretraining_limits=True,
#             balance_probabilities=False,
#             random_state=17,
#             n_preprocessing_jobs=1,
#         ),
#     ),
#     # (
#     #     "TabDPT_v1_3",
#     #     mll.TabDPTClassifier(
#     #         n_ensembles=8,
#     #         context_size=512,
#     #         context_reduction="subsample",
#     #         feature_reduction="pca",
#     #         normalizer="standard",
#     #         temperature=1.0,
#     #         device=None,
#     #         use_flash=True,
#     #         compile=False,
#     #         random_state=42,
#     #         verbose=False,
#     #     ),
#     # ),
#     # (
#     #     "Causilo_default",
#     #     mll.CausiloClassifier(
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "TabDPT_default",
#     #     mll.TabDPTClassifier(
#     #         random_state=42,
#     #     ),
#     # ),
#     (
#         "KumoTabular_small",
#         mll.KumoTabularClassifier(
#             size="small",
#             device="mps",
#             random_state=42,
#         ),
#     ),
#     (
#         "KumoTabular_medium",
#         mll.KumoTabularClassifier(
#             size="medium",
#             device="mps",
#             random_state=42,
#         ),
#     ),
#     (
#         "KumoTabular_large",
#         mll.KumoTabularClassifier(
#             size="large",
#             device="mps",
#             random_state=42,
#         ),
#     ),
#     # (
#     #     "LimiX2_default",
#     #     mll.LimiX2Classifier(
#     #         device="auto",
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "LimiX_2M_noretrieval",
#     #     mll.LimiXClassifier(
#     #         size="2m",
#     #         retrieval=False,
#     #         device="auto",
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "LimiX_16M_noretrieval",
#     #     mll.LimiXClassifier(
#     #         size="16m",
#     #         retrieval=False,
#     #         device="auto",
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "TabICLv2_default",
#     #     mll.TabICLClassifier(
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "RealMLP_TD_default",
#     #     mll.RealMLPClassifier(
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "TabFM_default",
#     #     mll.TabFMClassifier(
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "TabM_default",
#     #     mll.TabMClassifier(
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "EXAONE_default",
#     #     mll.EXAONETabularClassifier(
#     #         random_state=42,
#     #     ),
#     # ),
#     # (
#     #     "ElasticNet_tuned",
#     #     GridSearchCV(
#     #         estimator=Pipeline([
#     #             ("scaler", StandardScaler(with_mean=False)),
#     #             ("lr", LogisticRegression(
#     #                 solver="saga",
#     #                 max_iter=10000,
#     #                 tol=1e-4,
#     #                 random_state=42,
#     #             )),
#     #         ]),
#     #         param_grid={
#     #             "lr__C": [0.01, 0.1, 1.0, 10.0],
#     #             "lr__l1_ratio": [0.25, 0.5, 0.75],
#     #         },
#     #         scoring="neg_log_loss",
#     #         cv=3,
#     #         refit=True,
#     #         n_jobs=1,
#     #         error_score="raise",
#     #     ),
#     # ),
#     # (
#     #     "LASSO_tuned",
#     #     GridSearchCV(
#     #         estimator=Pipeline(
#     #             [
#     #                 ("scaler", StandardScaler(with_mean=False)),
#     #                 (
#     #                     "lr",
#     #                     LogisticRegression(
#     #                         penalty="l1",
#     #                         solver="saga",
#     #                         max_iter=10000,
#     #                         tol=1e-4,
#     #                         random_state=42,
#     #                     ),
#     #                 ),
#     #             ]
#     #         ),
#     #         param_grid={
#     #             "lr__C": [0.01, 0.1, 1.0, 10.0],
#     #         },
#     #         scoring="neg_log_loss",
#     #         cv=3,
#     #         refit=True,
#     #         n_jobs=1,
#     #         error_score="raise",
#     #     ),
#     # ),
#     (
#         "ExtraTrees_default",
#         ExtraTreesClassifier(
#             random_state=42,
#         ),
#     ),
#     (
#         "XGB_default",
#         XGBClassifier(
#             random_state=42,
#             n_jobs=1,
#             verbosity=0,
#             eval_metric="logloss",
#         ),
#     ),
#     (
#         "LR_default",
#         LogisticRegression(
#             random_state=42,
#         ),
#     ),
#     (
#         "RF_default",
#         RandomForestClassifier(
#             random_state=42,
#         ),
#     ),
#     (
#         "LightGBM_default",
#         LGBMClassifier(
#             random_state=42,
#             n_jobs=1,
#             verbosity=-1,
#         ),
#     ),

#   (
#        "XGB_500",
#        XGBClassifier(
#            n_estimators=500,
#            random_state=42,
#            n_jobs=1,
#            verbosity=0,
#            eval_metric="logloss",
#        ),
#    ),
#    (
#        "RF_500_balanced",
#        RandomForestClassifier(
#            n_estimators=500,
#            class_weight="balanced",
#            random_state=42,
#            n_jobs=1,
#        ),
#    ),
#    (
#    "LR_elasticnet",
#    Pipeline([
#        ("scaler", StandardScaler(with_mean=False)),
#        (
#            "lr",
#            LogisticRegression(
#                solver="saga",
#                C=0.1,
#                l1_ratio=0.5,
#                max_iter=10000,
#                tol=1e-4,
#                random_state=42,
#            ),
#        ),
#    ]),
# ),
#    (
#        "RF_1000_entropy",
#        RandomForestClassifier(
#            n_estimators=1000,
#            criterion="entropy",
#            min_samples_leaf=5,
#            max_features=0.30,
#            random_state=42,
#            n_jobs=1,
#        ),
#    ),
#    (
#        "RF_200",
#        RandomForestClassifier(
#            n_estimators=200,
#            random_state=42,
#            n_jobs=1,
#        ),
#    ),
#    (
#        "LR_lasso",
#        Pipeline([
#            ("scaler", StandardScaler(with_mean=False)),
#            (
#                "lr",
#                LogisticRegression(
#                    solver="saga",
#                    l1_ratio=1.0,
#                    C=1.0,
#                    max_iter=10000,
#                    tol=1e-4,
#                    random_state=42,
#                ),
#            ),
#        ]),
#    ),
    # (
    #     "KumoTabular_small_e4",
    #     mll.KumoTabularClassifier(
    #         size="small",
    #         device="mps",
    #         pretrained=True,
    #         n_estimators=4,
    #         random_state=42,
    #     ),
    # ),
    # (
    #     "KumoTabular_medium_e4",
    #     mll.KumoTabularClassifier(
    #         size="medium",
    #         device="mps",
    #         pretrained=True,
    #         n_estimators=4,
    #         random_state=42,
    #     ),
    # ),
    # (
    #     "KumoTabular_large_e4",
    #     mll.KumoTabularClassifier(
    #         size="large",
    #         device="mps",
    #         pretrained=True,
    #         n_estimators=4,
    #         random_state=42,
    #     ),
    # ),
    (
        "KumoTabular_large_e10",
        mll.KumoTabularClassifier(
            size="large",
            device="mps",
            pretrained=True,
            n_estimators=10,
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
    include_inactive=True,
    max_sizes=(5,),

    # ==================================================
    # SELECTION STRATEGIES (18)
    # ==================================================
    selection_strategies=(
        # Performance-based
        "top_k",
        # "best_per_resolution",
        # "best_per_learner_type",

        # Deep ML + classical ML
        # "best_deep_ml_pair",
        "best_deep_ml_complementary_pair",

        # Diversity-based
        # "quality_diversity",
        # "diversity",
        # "performance_diversity",

        # Out-of-fold prediction-based
        # "oof_top_k",
        # "oof_similarity",

        # Greedy ensemble selection
        # "caruana",
        # "bagged_caruana",
        # "caruana_multistart",
        # "beam",

        # Super learner variants
        "super_learner",
        # "adaptive_super_learner",
        # "safe_super_learner",
        # "regularized_super_learner",
    ),

    # ==================================================
    # AGGREGATION STRATEGIES (29)
    # ==================================================
    aggregation_strategies=(
        # Probability averaging
        "mean_proba",
        "weighted_mean_proba",
        # "median_proba",
        # "trimmed_mean_proba",
        # "geometric_mean_proba",
        # "trimmed_mean",
        # "logit_mean",

        # Rank, vote, and extrema
        # "rank_mean",
        # "majority_vote",
        # "max_proba",
        # "min_proba",

        # Linear stacking
        # "logistic_stacking",
        # "elastic_net_stacking",
        # "cohort_robust_stacking",

        # ECDF-based aggregation
        # "ecdf_logistic_stacking",
        # "ecdf_grouped_stacking",
        # "ecdf_weighted_quantile",
        # "ecdf_learned_quantile",

        # Tree-based stacking
        # "rf_stacking",
        # "extra_trees_stacking",
        # "boosted_stacking",

        # Adaptive and hierarchical stacking
        # "gated_stacking",
        # "hierarchical_dirichlet_stacking",
        # "sparse_nonnegative_stacking",

        # Probabilistic ensemble methods
        "bayesian_model_averaging",
        # "mixture_of_experts",
        # "cross_study_dro",

        # Probability calibration
        # "temperature_scaled_mean_proba",
        # "sigmoid_calibrated_mean_proba",
    ),

    # ==================================================
    # OPTIMIZATION METRIC
    # ==================================================
    # Classification: 21 supported ensemble metrics
    #
    # ROC metrics (maximize)
    #   AUROC, AUROC_macro, AUROC_weighted
    #
    # Precision-recall metrics (maximize)
    #   AUCPR, AUCPR_macro, AUCPR_weighted
    #   AP, AP_macro
    #
    # Classification metrics (maximize)
    #   MCC, F1w, F1_macro
    #   Precision, Recall
    #   Sensitivity, Specificity
    #   PPV, NPV
    #   BalAcc, Accuracy
    #
    # Probabilistic losses (minimize)
    #   log_loss, brier
    #
    # Also accepted: auto (= log_loss for classification)
    #
    # Evaluation-only grouped metrics (not Ensemble)
    #   subject_macro_log_loss
    #   cohort_macro_log_loss
    #
    optimize_metric="AUROC",
)


EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    profile="screening",
    methods=(
        mll.Permutation(),
        # mll.SHAP(),
        mll.ALE(),
        mll.LIME(),
        mll.ALEInteractions(
            top_k=50,
        ),
    ),
    classes="auto",
)

ROBUSTNESS = mll.Robustness(
    targets=("mpma_b",),
    top_k=30,
)

EXTERNAL_TEST = lampp_scz.external_test

if EXTERNAL_TEST is None:
    raise RuntimeError("External test set is unavailable")

INFERENCE = EXTERNAL_TEST.mllabiome(
    targets=("mpma_b", "mpma_e"),
    feature_policy="strict",
    save_weights=False,
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
