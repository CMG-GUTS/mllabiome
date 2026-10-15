from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_scz
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP schizophrenia mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-SCZ-NCV-CM-base"

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
    ("phylum", ("phylum",)),
    ("class", ("class",)),
    ("order", ("order",)),
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("species", ("species",)),
    ("strain", ("strain",)),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("clr", composition_scope="rank-wise"),
    mll.Transformation(
        "arcsine_sqrt",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "relative_abundance",
        composition_scope="rank-wise",
    ),
    mll.Transformation(
        "log10",
        composition_scope="rank-wise",
    ),
)

MODELS = (
    # baselines
    (
        "Baseline_RF",
        RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        ),
    ),
    (
        "SIAMCAT",
        mll.SIAMCATClassifier(
            random_state=42,
        ),
    ),
    # # default standard models
    (
        "CB_default",
        CatBoostClassifier(
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
    (
        "RF_default",
        RandomForestClassifier(
            random_state=42,
        ),
    ),
    (
        "LightGBM_default",
        LGBMClassifier(
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ),
    ),
    (
        "XGB_default",
        XGBClassifier(
            random_state=42,
            n_jobs=1,
            verbosity=0,
            eval_metric="logloss",
        ),
    ),
    (
        "LR_default",
        LogisticRegression(
            random_state=42,
        ),
    ),
    # (
    #     "LR_tuned_scaled",
    #     GridSearchCV(
    #         estimator=Pipeline(
    #             [
    #                 ("scaler", StandardScaler(with_mean=False)),
    #                 (
    #                     "lr",
    #                     LogisticRegression(
    #                         solver="saga",
    #                         max_iter=5000,
    #                         tol=1e-3,
    #                         random_state=42,
    #                     ),
    #                 ),
    #             ]
    #         ),
    #         param_grid={
    #             "lr__C": [0.03, 0.3, 3.0],
    #             "lr__l1_ratio": [0.0, 0.5, 1.0],
    #         },
    #         scoring="neg_log_loss",
    #         cv=3,
    #         refit=True,
    #         n_jobs=1,
    #         error_score="raise",
    #     ),
    # ),
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
    include_inactive=True,
    max_sizes=(10,),
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "performance_diversity",
        "caruana",
        "bagged_caruana",
        "super_learner",
        "safe_super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "logistic_stacking",
        "rf_stacking",
    ),
    optimize_metric="log_loss",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b", "baseline_rf"),
    profile="screening",
    methods=(
        mll.Permutation(),
        mll.SHAP(),
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
