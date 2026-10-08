from __future__ import annotations

from pathlib import Path

from catboost import CatBoostRegressor
from curated_microbiota.collections import brown_mdd
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "Brown MDD PROMIS depression severity regression mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "BROWN-MDD-PROMIS-REGRESSION-NCV-CM-v2"

DATA = brown_mdd.mllabiome(target="promis_depression")

EVALUATION = brown_mdd.splits(
    target="promis_depression",
    benchmark="mllabiome-benchmark-v2",
).mllabiome(
    optimize_metric="RMSE",
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
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("species", ("species",)),
    # # ("domain-phylum", ("domain", "phylum")),
    # # ("domain-class", ("domain", "phylum", "class")),
    # # ("domain-order", ("domain", "phylum", "class", "order")),
    # # ("domain-family", ("domain", "phylum", "class", "order", "family")),
    # # ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    # # ("phylum-class", ("phylum", "class")),
    # # ("phylum-order", ("phylum", "class", "order")),
    # ("phylum-family", ("phylum", "class", "order", "family")),
    ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
    # ("class-order", ("class", "order")),
    # # ("class-family", ("class", "order", "family")),
    # ("class-genus", ("class", "order", "family", "genus")),
    ("order-family", ("order", "family")),
    ("order-genus", ("order", "family", "genus")),
    ("family-genus", ("family", "genus")),
    # ("raw", ("raw",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation(
        "clr",
        composition_scope="rank-wise",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.20,
        ),
    ),
    mll.Transformation(
        "clr",
        composition_scope="rank-wise",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.10,
        ),
    ),
    mll.Transformation(
        "relative_abundance",
        composition_scope="joint",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.20,
        ),
    ),
    mll.Transformation(
        "arcsine_sqrt",
        composition_scope="joint",
    ),
    mll.Transformation(
        "presence_absence",
    ),
    mll.Transformation(
        "log10",
        composition_scope="rank-wise",
    ),
)

MODELS = (
    # ---------------------------------------------------------
    # Ridge regression
    # Primary model family for this small p >> n dataset
    # ---------------------------------------------------------
    (
        "Ridge_a0.1",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=0.1),
        ),
    ),
    (
        "Ridge_a1",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
    ),
    (
        "Ridge_a10",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
    ),
    (
        "Ridge_a100",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0),
        ),
    ),
    # ---------------------------------------------------------
    # Elastic Net
    # Sparse + correlated feature alternative to Ridge
    # ---------------------------------------------------------
    (
        "EN_a0.01_l1r0.2",
        make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.20,
                max_iter=20000,
                tol=1e-4,
                random_state=42,
            ),
        ),
    ),
    (
        "EN_a0.1_l1r0.2",
        make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.10,
                l1_ratio=0.20,
                max_iter=20000,
                tol=1e-4,
                random_state=42,
            ),
        ),
    ),
    (
        "EN_a0.1_l1r0.5",
        make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.10,
                l1_ratio=0.50,
                max_iter=20000,
                tol=1e-4,
                random_state=42,
            ),
        ),
    ),
    (
        "EN_a1_l1r0.2",
        make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=1.0,
                l1_ratio=0.20,
                max_iter=20000,
                tol=1e-4,
                random_state=42,
            ),
        ),
    ),
    # ---------------------------------------------------------
    # Support Vector Regression
    # Non-linear competitor
    # ---------------------------------------------------------
    (
        "SVR_RBF_C1",
        make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
                gamma="scale",
            ),
        ),
    ),
    (
        "SVR_RBF_C10",
        make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.1,
                gamma="scale",
            ),
        ),
    ),
    # ---------------------------------------------------------
    # Linear SVR
    # Useful intermediate between Ridge and nonlinear SVR
    # ---------------------------------------------------------
    (
        "SVR_linear_C1",
        make_pipeline(
            StandardScaler(),
            SVR(
                kernel="linear",
                C=1.0,
                epsilon=0.1,
            ),
        ),
    ),
    # ---------------------------------------------------------
    # Partial Least Squares
    # Designed for correlated high-dimensional predictors
    # ---------------------------------------------------------
    (
        "PLS_2",
        PLSRegression(
            n_components=2,
            scale=True,
            max_iter=1000,
        ),
    ),
    (
        "PLS_5",
        PLSRegression(
            n_components=5,
            scale=True,
            max_iter=1000,
        ),
    ),
    # ---------------------------------------------------------
    # Random Forest comparator
    # ---------------------------------------------------------
    (
        "RFReg_1000_msl5_mf0.5",
        RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features=0.50,
            n_jobs=1,
            random_state=42,
        ),
    ),
    # ---------------------------------------------------------
    # Conservative CatBoost comparator
    # ---------------------------------------------------------
    (
        "CBReg_i500_d3_l2_10",
        CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=10,
            random_strength=1.0,
            rsm=0.60,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
    (
        "XGB_d2_lr0.03",
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=10.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "XGB_d3_lr0.02",
        XGBRegressor(
            n_estimators=700,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=20.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "RFReg_1000_msl2_mf0.33",
        RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=2,
            max_features=0.33,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "RFReg_1000_msl5_mf0.33",
        RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features=0.33,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "RFReg_1000_msl10_mf0.5",
        RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=10,
            max_features=0.50,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "KRR_RBF_a1_gscale",
        make_pipeline(
            StandardScaler(),
            KernelRidge(
                alpha=1.0,
                kernel="rbf",
                gamma=None,
            ),
        ),
    ),
    (
        "BayesianRidge",
        make_pipeline(
            StandardScaler(),
            BayesianRidge(),
        ),
    ),
)

# MODELS = (
#     (
#         "RFReg_1000_msl5",
#         RandomForestRegressor(
#             n_estimators=1000,
#             min_samples_leaf=5,
#             n_jobs=1,
#             random_state=42,
#         ),
#     ),
#     (
#         "CBReg_abundance_i300_d3",
#         CatBoostRegressor(
#             iterations=300,
#             learning_rate=0.04,
#             depth=3,
#             l2_leaf_reg=10,
#             random_strength=1.0,
#             rsm=0.60,
#             loss_function="RMSE",
#             eval_metric="RMSE",
#             random_seed=42,
#             thread_count=1,
#             verbose=False,
#             allow_writing_files=False,
#         ),
#     ),
# )

GATE = mll.QualificationGate(
    enabled=False,
    metric="RMSE",
)

ENSEMBLE = mll.Ensemble(
    max_sizes=(10,),
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        # "super_learner",
    ),
    aggregation_strategies=(
        "mean_prediction",
        "weighted_mean_prediction",
        "median_prediction",
    ),
    optimize_metric="RMSE",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    profile="screening",
    methods=(
        mll.Permutation(scoring="RMSE"),
        mll.SHAP(),
        mll.ALE(),
        mll.LIME(),
        mll.ALEInteractions(),
    ),
)

ROBUSTNESS = mll.Robustness(
    targets=("mpma_b",),
    top_k=30,
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
)
