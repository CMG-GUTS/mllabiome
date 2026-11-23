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
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import SVC, LinearSVC, NuSVC
from xgboost import XGBClassifier

from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


from mllabiome import mll
from mllabiome.causilo import CausiloClassifier


HERE = Path(__file__).resolve().parent
TITLE = "LAMPP schizophrenia mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-SCZ-NCV-CM-official"

DATA = lampp_scz.mllabiome()

EVALUATION = lampp_scz.splits(
    benchmark="mllabiome-benchmark-v2",
).mllabiome(
    optimize_metric="log_loss",
    n_jobs=1,
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
    ("phylum", ("phylum",)),
    # ("class", ("class",)),
    # ("order", ("order",)),
    # ("family", ("family",)),
    # ("genus", ("genus",)),
    # ("species", ("species",)),
    # ("strain", ("strain",)),
    # ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("identity"),
    mll.Transformation("presence_absence"),
    mll.Transformation(
        "relative_abundance",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "arcsine_sqrt",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "hellinger",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "log10_relative_abundance_half_min_pseudocount",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "centered_log_ratio_multiplicative_replacement",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "rclr",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "additive_log_ratio_training_reference_multiplicative_replacement",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "isometric_log_ratio_egozcue_multiplicative_replacement",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "prevalence_weighted_relative_abundance",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "prevalence_rank_fusion",
        composition_scope="rank-wise",
        alpha=0.25,
        c=100.0,
        gamma=1.0,
        pseudo=0.5,
    ),
    mll.Transformation(
        "within_sample_fractional_rank",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "training_ecdf_rank",
    ),
    mll.Transformation(
        "quantile_normal_relative_abundance",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "yeo_johnson_relative_abundance",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "standardize",
    ),
    mll.Transformation(
        "quantile_normal_numeric",
    ),
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
    # (
    #     'SIAMCAT',
    #     mll.SIAMCATClassifier(
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'Baseline_RF',
    #     RandomForestClassifier(
    #         n_estimators=1000,
    #         min_samples_leaf=5,
    #         random_state=42,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     'LR_default',
    #     LogisticRegression(
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'LR_lasso',
    #     Pipeline(
    #         [
    #             ('scaler', StandardScaler(with_mean=False)),
    #             (
    #                 'lr',
    #                 LogisticRegression(
    #                     solver='saga',
    #                     l1_ratio=1.0,
    #                     C=1.0,
    #                     max_iter=10000,
    #                     tol=0.0001,
    #                     random_state=42,
    #                 ),
    #             ),
    #         ],
    #     ),
    # ),
    # (
    #     'LR_elasticnet',
    #     Pipeline(
    #         [
    #             ('scaler', StandardScaler(with_mean=False)),
    #             (
    #                 'lr',
    #                 LogisticRegression(
    #                     solver='saga',
    #                     C=0.1,
    #                     l1_ratio=0.5,
    #                     max_iter=10000,
    #                     tol=0.0001,
    #                     random_state=42,
    #                 ),
    #             ),
    #         ],
    #     ),
    # ),
    # (
    #     'LR_ElasticNet_Balanced',
    #     Pipeline(
    #         [
    #             ('scaler', StandardScaler(with_mean=False)),
    #             (
    #                 'lr',
    #                 LogisticRegression(
    #                     solver='saga',
    #                     C=0.1,
    #                     l1_ratio=0.5,
    #                     class_weight='balanced',
    #                     max_iter=10000,
    #                     tol=0.0001,
    #                     random_state=42,
    #                 ),
    #             ),
    #         ],
    #     ),
    # ),
    # (
    #     'EN_default',
    #     make_pipeline(
    #         StandardScaler(),
    #         LogisticRegression(
    #             solver='saga',
    #             l1_ratio=0.5,
    #             max_iter=5000,
    #             random_state=42,
    #         ),
    #     ),
    # ),
    # (
    #     'LDA_shrinkage',
    #     make_pipeline(
    #         StandardScaler(),
    #         LinearDiscriminantAnalysis(
    #             solver='lsqr',
    #             shrinkage='auto',
    #         ),
    #     ),
    # ),
    # (
    #     'KNN_default',
    #     make_pipeline(
    #         StandardScaler(),
    #         KNeighborsClassifier(),
    #     ),
    # ),
    # (
    #     'MLP_default',
    #     make_pipeline(
    #         StandardScaler(),
    #         MLPClassifier(random_state=42),
    #     ),
    # ),
    # (
    #     'RF_default',
    #     RandomForestClassifier(
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'RF_200',
    #     RandomForestClassifier(
    #         n_estimators=200,
    #         random_state=42,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     'RF_100_balanced',
    #     RandomForestClassifier(
    #         n_estimators=100,
    #         class_weight='balanced',
    #         random_state=42,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     'RF_500_balanced',
    #     RandomForestClassifier(
    #         n_estimators=500,
    #         class_weight='balanced',
    #         random_state=42,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     'RF_1000_balanced',
    #     RandomForestClassifier(
    #         n_estimators=1000,
    #         class_weight='balanced',
    #         random_state=42,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     'RF_1000_entropy',
    #     RandomForestClassifier(
    #         n_estimators=1000,
    #         criterion='entropy',
    #         min_samples_leaf=5,
    #         max_features=0.3,
    #         random_state=42,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     'ExtraTrees_default',
    #     ExtraTreesClassifier(
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'GB_default',
    #     GradientBoostingClassifier(
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'CB_default',
    #     CatBoostClassifier(
    #         random_seed=42,
    #         thread_count=1,
    #         verbose=False,
    #         allow_writing_files=False,
    #     ),
    # ),
    # (
    #     'CatBoost_SqrtBalanced',
    #     CatBoostClassifier(
    #         iterations=600,
    #         depth=4,
    #         learning_rate=0.03,
    #         l2_leaf_reg=10,
    #         auto_class_weights='SqrtBalanced',
    #         random_seed=42,
    #         thread_count=1,
    #         verbose=False,
    #         allow_writing_files=False,
    #     ),
    # ),
    # (
    #     'LightGBM_default',
    #     LGBMClassifier(
    #         random_state=42,
    #         n_jobs=1,
    #         verbosity=-1,
    #     ),
    # ),
    # (
    #     'XGB_default',
    #     XGBClassifier(
    #         random_state=42,
    #         n_jobs=1,
    #         verbosity=0,
    #         eval_metric='logloss',
    #     ),
    # ),
    # (
    #     'XGB_500',
    #     XGBClassifier(
    #         n_estimators=500,
    #         random_state=42,
    #         n_jobs=1,
    #         verbosity=0,
    #         eval_metric='logloss',
    #     ),
    # ),
    # (
    #     'ComplementNB_default',
    #     ComplementNB(),
    # ),
    # (
    #     'Causilo_default',
    #     CausiloClassifier(
    #         n_estimators=8,
    #         random_state=42,
    #         device='mps',
    #     ),
    # ),
    # (
    #     'TabPFN',
    #     mll.TabPFNClassifier(
    #         n_estimators='auto',
    #         auto_scale_n_estimators=True,
    #         model_path='auto',
    #         device='mps',
    #         inference_precision='auto',
    #         fit_mode='fit_preprocessors',
    #         memory_saving_mode='auto',
    #         keep_cache_on_device=True,
    #         random_state=42,
    #         n_preprocessing_jobs=1,
    #     ),
    # ),
    # (
    #     'TabPFN_n8',
    #     mll.TabPFNClassifier(
    #         n_estimators=8,
    #         model_path='auto',
    #         device='mps',
    #         inference_precision='auto',
    #         fit_mode='fit_preprocessors',
    #         memory_saving_mode='auto',
    #         keep_cache_on_device=True,
    #         ignore_pretraining_limits=True,
    #         balance_probabilities=False,
    #         random_state=17,
    #         n_preprocessing_jobs=1,
    #     ),
    # ),
    # (
    #     'KumoTabular_small',
    #     mll.KumoTabularClassifier(
    #         size='small',
    #         n_estimators=8,
    #         pretrained=True,
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'KumoTabular_medium',
    #     mll.KumoTabularClassifier(
    #         size='medium',
    #         n_estimators=8,
    #         pretrained=True,
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'KumoTabular_large',
    #     mll.KumoTabularClassifier(
    #         size='large',
    #         n_estimators=8,
    #         pretrained=True,
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'KumoTabular_small_e4',
    #     mll.KumoTabularClassifier(
    #         size='small',
    #         n_estimators=4,
    #         pretrained=True,
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'KumoTabular_medium_e4',
    #     mll.KumoTabularClassifier(
    #         size='medium',
    #         n_estimators=4,
    #         pretrained=True,
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'KumoTabular_large_e4',
    #     mll.KumoTabularClassifier(
    #         size='large',
    #         n_estimators=4,
    #         pretrained=True,
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    (
        'KumoTabular_large_e10',
        mll.KumoTabularClassifier(
            size='large',
            n_estimators=10,
            pretrained=True,
            device='mps',
            random_state=42,
        ),
    ),
    # (
    #     'Seldon',
    #     mll.SeldonClassifier(
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'Seldon_mini',
    #     mll.SeldonMiniClassifier(
    #         device='mps',
    #         random_state=42,
    #     ),
    # ),
    # (
    #     'LimiX-2-1016',
    #     mll.LimiX2Classifier(
    #         device='mps',
    #         random_state=42,
    #         use_data_cache=False,
    #     ),
    # ),
)


# MODELS = (
#     (
#     "LimiX-2-1016",
#     mll.LimiX2Classifier(
#         device="mps",
#         random_state=42,
#         use_data_cache=False,
#     ),
# ),
#     (
#         "Seldon_mini",
#         mll.SeldonMiniClassifier(
#             device="mps",
#             random_state=42,
#         ),
#     ),
#     (
#         "Seldon",
#         mll.SeldonClassifier(
#             device="mps",
#             random_state=42,
#         ),
#     ),
# # # #     # baselines
#     # (
#     #     "Baseline_RF",
#     #     RandomForestClassifier(
#     #         n_estimators=1000,
#     #         min_samples_leaf=5,
#     #         random_state=42,
#     #         n_jobs=1,
#     #     ),
#     # ),
# # # # #     # (
# # # # #     #     "SIAMCAT",
# # # # #     #     mll.SIAMCATClassifier(
# # # # #     #         random_state=42,
# # # # #     #     ),
# # # # #     # ),
# # # # #     # # default standard models
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
#         "ComplementNB_default",
#         ComplementNB(),
#     ),

# # #     (
# # #         "HGB_default",
# # #         HistGradientBoostingClassifier(
# # #             random_state=42,
# # #         ),
# # #     ),
# # #     (
# # #         "GaussianProcess_default",
# # #         GaussianProcessClassifier(
# # #             random_state=42,
# # #         ),
# # #     ),

# # #     (
# # #         "BernoulliNB_default",
# # #         BernoulliNB(),
# # #     ),
# # #     (
# # #         "LDA_shrinkage",
# # #         make_pipeline(
# # #             StandardScaler(),
# # #             LinearDiscriminantAnalysis(
# # #                 solver="lsqr",
# # #                 shrinkage="auto",
# # #             ),
# # #         ),
# # #     ),
# # #     (
# # #         "AdaBoost_default",
# # #         AdaBoostClassifier(
# # #             random_state=42,
# # #         ),
# # #     ),
# # #     (
# # #         "KNN_default",
# # #         make_pipeline(
# # #             StandardScaler(),
# # #             KNeighborsClassifier(),
# # #         ),
# # #     ),
#     (
#         "MLP_default",
#         make_pipeline(
#             StandardScaler(),
#             MLPClassifier(
#                 random_state=42,
#             ),
#         ),
#     ),
# # #     (
# # #         "NuSVC_default",
# # #         make_pipeline(
# # #             StandardScaler(),
# # #             NuSVC(
# # #                 probability=True,
# # #                 random_state=42,
# # #             ),
# # #         ),
# # #     ),
#     (
#         "ExtraTrees_default",
#         ExtraTreesClassifier(
#             random_state=42,
#         ),
#     ),
# # #     (
# # #         "GB_default",
# # #         GradientBoostingClassifier(
# # #             random_state=42,
# # #         ),
# # #     ),
# # #     (
# # #         "RF_200",
# # #         RandomForestClassifier(
# # #             n_estimators=200,
# # #             random_state=42,
# # #             n_jobs=1,
# # #         ),
# # #     ),
# # #     (
# # #         "RF_500_balanced",
# # #         RandomForestClassifier(
# # #             n_estimators=500,
# # #             class_weight="balanced",
# # #             random_state=42,
# # #             n_jobs=1,
# # #         ),
# # #     ),
# # #     (
# # #         "RF_1000_entropy",
# # #         RandomForestClassifier(
# # #             n_estimators=1000,
# # #             criterion="entropy",
# # #             min_samples_leaf=5,
# # #             max_features=0.30,
# # #             random_state=42,
# # #             n_jobs=1,
# # #         ),
# # #     ),
# # #     (
# # #         "XGB_500",
# # #         XGBClassifier(
# # #             n_estimators=500,
# # #             random_state=42,
# # #             n_jobs=1,
# # #             verbosity=0,
# # #             eval_metric="logloss",
# # #         ),
# # #     ),
# # #     # (
# # #     #     "BalancedRF_500",
# # #     #     BalancedRandomForestClassifier(
# # #     #         n_estimators=500,
# # #     #         sampling_strategy="all",
# # #     #         replacement=True,
# # #     #         bootstrap=False,
# # #     #         min_samples_leaf=5,
# # #     #         max_features="sqrt",
# # #     #         random_state=42,
# # #     #         n_jobs=1,
# # #     #     ),
# # #     # ),
# # #     (
# # #         "CatBoost_SqrtBalanced",
# # #         CatBoostClassifier(
# # #             iterations=600,
# # #             depth=4,
# # #             learning_rate=0.03,
# # #             l2_leaf_reg=10,
# # #             auto_class_weights="SqrtBalanced",
# # #             random_seed=42,
# # #             thread_count=1,
# # #             verbose=False,
# # #             allow_writing_files=False,
# # #         ),
# # #     ),
# # #     (
# # #         "LR_lasso",
# # #         Pipeline([
# # #             ("scaler", StandardScaler(with_mean=False)),
# # #             ("lr", LogisticRegression(
# # #                 solver="saga",
# # #                 l1_ratio=1.0,
# # #                 C=1.0,
# # #                 max_iter=10000,
# # #                 tol=1e-4,
# # #                 random_state=42,
# # #             )),
# # #         ]),
# # #     ),
# # #     (
# # #         "LR_elasticnet",
# # #         Pipeline([
# # #             ("scaler", StandardScaler(with_mean=False)),
# # #             ("lr", LogisticRegression(
# # #                 solver="saga",
# # #                 C=0.1,
# # #                 l1_ratio=0.5,
# # #                 max_iter=10000,
# # #                 tol=1e-4,
# # #                 random_state=42,
# # #             )),
# # #         ]),
# # #     ),
# # #     (
# # #         "LR_ElasticNet_Balanced",
# # #         Pipeline([
# # #             ("scaler", StandardScaler(with_mean=False)),
# # #             ("lr", LogisticRegression(
# # #                 solver="saga",
# # #                 C=0.1,
# # #                 l1_ratio=0.5,
# # #                 class_weight="balanced",
# # #                 max_iter=10000,
# # #                 tol=1e-4,
# # #                 random_state=42,
# # #             )),
# # #         ]),
# # #     ),
# # #     (
# # #         "EN_default",
# # #         make_pipeline(
# # #             StandardScaler(),
# # #             LogisticRegression(
# # #                 solver="saga",
# # #                 l1_ratio=0.5,
# # #                 max_iter=5000,
# # #                 random_state=42,
# # #             ),
# # #         ),
# # #     ),
#     (
#         "KumoTabular_small",
#         mll.KumoTabularClassifier(
#             size="small",
#             n_estimators=8,
#             pretrained=True,
#             device="mps",
#             random_state=42,
#         ),
#     ),
# #     (
# #         "KumoTabular_medium",
# #         mll.KumoTabularClassifier(
# #             size="medium",
# #             n_estimators=8,
# #             pretrained=True,
# #             device="mps",
# #             random_state=42,
# #         ),
# #     ),
# #     (
# #         "KumoTabular_large",
# #         mll.KumoTabularClassifier(
# #             size="large",
# #             n_estimators=8,
# #             pretrained=True,
# #             device="mps",
# #             random_state=42,
# #         ),
# #     ),
# #     (
# #         "KumoTabular_small_e4",
# #         mll.KumoTabularClassifier(
# #             size="small",
# #             n_estimators=4,
# #             pretrained=True,
# #             device="mps",
# #             random_state=42,
# #         ),
# #     ),
# #     (
# #         "KumoTabular_medium_e4",
# #         mll.KumoTabularClassifier(
# #             size="medium",
# #             n_estimators=4,
# #             pretrained=True,
# #             device="mps",
# #             random_state=42,
# #         ),
# #     ),
# #     (
# #         "KumoTabular_large_e4",
# #         mll.KumoTabularClassifier(
# #             size="large",
# #             n_estimators=4,
# #             pretrained=True,
# #             device="mps",
# #             random_state=42,
# #         ),
# #     ),
# #     (
# #         "KumoTabular_large_e10",
# #         mll.KumoTabularClassifier(
# #             size="large",
# #             n_estimators=10,
# #             pretrained=True,
# #             device="mps",
# #             random_state=42,
# #         ),
# #     ),
# # #     (
# # #         "TabPFN_n8",
# # #         mll.TabPFNClassifier(
# # #             n_estimators=8,
# # #             model_path="auto",
# # #             device="auto",
# # #             inference_precision="auto",
# # #             fit_mode="fit_preprocessors",
# # #             memory_saving_mode="auto",
# # #             keep_cache_on_device=True,
# # #             ignore_pretraining_limits=True,
# # #             balance_probabilities=False,
# # #             random_state=17,
# # #             n_preprocessing_jobs=1,
# # #         ),
# # #     ),
# # #     (
# # #         "TabPFN",
# # #         mll.TabPFNClassifier(
# # #             n_estimators="auto",
# # #             auto_scale_n_estimators=True,
# # #             model_path="auto",
# # #             device="auto",
# # #             inference_precision="auto",
# # #             fit_mode="fit_preprocessors",
# # #             memory_saving_mode="auto",
# # #             keep_cache_on_device=True,
# # #             random_state=42,
# # #             n_preprocessing_jobs=1,
# # #         ),
# # #     ),
# #     (
# #         "Causilo_default",
# #         CausiloClassifier(
# #             n_estimators=8,
# #             random_state=42,
# #             device="mps",
# #         ),
# #     ),
# # #     (
# # #         "TabICLv2_default",
# # #         mll.TabICLClassifier(
# # #             checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
# # #             random_state=42,
# # #         ),
# # #     ),
# # #     (
# # #         "Bagging_default",
# # #         BaggingClassifier(
# # #             random_state=42,
# # #             n_jobs=1,
# # #         ),
# # #     ),
# )


GATE = mll.QualificationGate(
    enabled=False,
    metric="MCC",
    threshold=0.02,
)

# BASE_SELECTION = mll.BaseSelection(
#     metrics={
#         "AUROC": 0.5,
#         "MCC": 0.5,
#     },
#     normalization="chance_adjusted",
#     include_inactive=True,
# )
# BASE_SELECTION = mll.BaseSelection(
#     metrics={"AUROC": 1.0},
#     normalization="none",
#     include_inactive=True,
#     auroc_tolerance=0.01,
# )

BASE_SELECTION = mll.BaseSelection(
    strategy="pareto_auroc_logloss_ideal",
    metrics={"AUROC": 1.0},
    normalization="none",
    include_inactive=True,
)

ENSEMBLE = mll.Ensemble(
    selection_mode="pooled_grid",
    include_inactive=True,
    max_sizes=tuple(range(2, 5)),
    pooled_super_learner_pool_size=128,
    selection_strategies=(
        "top_k",
        "pareto_optimal_ml_dl_auroc_logloss_capped",
        "pareto_optimal_taxonomic_rank_auroc_logloss_capped",
        "caruana",
        "super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "bayesian_model_averaging",
        "weighted_mean_proba",
    ),
    optimize_metric="AUROC",
)



# ENSEMBLE = mll.Ensemble(
#     include_inactive=True,
#     max_sizes=(10,),
#     selection_strategies=(
#         "pareto_optimal_auroc_logloss",
#     ),
#     aggregation_strategies=(
#         "mean_proba",
#     ),
#     optimize_metric="AUROC",
# )



# ENSEMBLE = mll.Ensemble(
#     include_inactive=True,
#     max_sizes=(5,),

#     # ==================================================
#     # SELECTION STRATEGIES (18)
#     # ==================================================
#     selection_strategies=(
#         # Performance-based
#         "top_k",
#         # "best_per_resolution",
#         # "best_per_learner_type",

#         # Deep ML + classical ML
#         # "best_deep_ml_pair",
#         "best_deep_ml_complementary_pair",

#         # Diversity-based
#         "quality_diversity",
#         "diversity",
#         # "performance_diversity",

#         # Out-of-fold prediction-based
#         # "oof_top_k",
#         # "oof_similarity",

#         # Greedy ensemble selection
#         # "caruana",
#         # "bagged_caruana",
#         # "caruana_multistart",
#         # "beam",

#         # Super learner variants
#         "super_learner",
#         # "adaptive_super_learner",
#         # "safe_super_learner",
#         # "regularized_super_learner",
#     ),

#     # ==================================================
#     # AGGREGATION STRATEGIES (29)
#     # ==================================================
#     aggregation_strategies=(
#         # Probability averaging
#         # "mean_proba",
#         # "weighted_mean_proba",
#         # "median_proba",
#         # "trimmed_mean_proba",
#         # "geometric_mean_proba",
#         # "trimmed_mean",
#         # "logit_mean",

#         # Rank, vote, and extrema
#         # "rank_mean",
#         # "majority_vote",
#         # "max_proba",
#         # "min_proba",

#         # Linear stacking
#         # "logistic_stacking",
#         # "elastic_net_stacking",
#         # "cohort_robust_stacking",

#         # ECDF-based aggregation
#         # "ecdf_logistic_stacking",
#         # "ecdf_grouped_stacking",
#         # "ecdf_weighted_quantile",
#         # "ecdf_learned_quantile",

#         # Tree-based stacking
#         # "rf_stacking",
#         # "extra_trees_stacking",
#         # "boosted_stacking",

#         # Adaptive and hierarchical stacking
#         # "gated_stacking",
#         # "hierarchical_dirichlet_stacking",
#         # "sparse_nonnegative_stacking",

#         # Probabilistic ensemble methods
#         "bayesian_model_averaging",
#         # "mixture_of_experts",
#         # "cross_study_dro",

#         # Probability calibration
#         # "temperature_scaled_mean_proba",
#         # "sigmoid_calibrated_mean_proba",
#     ),

#     # ==================================================
#     # OPTIMIZATION METRIC
#     # ==================================================
#     # Classification: 21 supported ensemble metrics
#     #
#     # ROC metrics (maximize)
#     #   AUROC, AUROC_macro, AUROC_weighted
#     #
#     # Precision-recall metrics (maximize)
#     #   AUCPR, AUCPR_macro, AUCPR_weighted
#     #   AP, AP_macro
#     #
#     # Classification metrics (maximize)
#     #   MCC, F1w, F1_macro
#     #   Precision, Recall
#     #   Sensitivity, Specificity
#     #   PPV, NPV
#     #   BalAcc, Accuracy
#     #
#     # Probabilistic losses (minimize)
#     #   log_loss, brier
#     #
#     # Also accepted: auto (= log_loss for classification)
#     #
#     # Evaluation-only grouped metrics (not Ensemble)
#     #   subject_macro_log_loss
#     #   cohort_macro_log_loss
#     #
#     optimize_metric="AUROC",
# )


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
    base_selection=BASE_SELECTION,
    gate=GATE,
    explore=EXPLORE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
    robustness=ROBUSTNESS,
    inference=INFERENCE,
)
