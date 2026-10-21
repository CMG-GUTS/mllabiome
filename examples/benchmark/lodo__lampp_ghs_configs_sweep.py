from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from curated_microbiota.collections import lampp_ghs
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP general health status LODO MPMA sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-GHS-LODO-CM-official"

DATA = lampp_ghs.mllabiome()

EVALUATION = lampp_ghs.splits(
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
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("identity"),
    # mll.Transformation("presence_absence"),
    # mll.Transformation(
    #     "relative_abundance",
    #     composition_scope="rank-wise",
    # ),
    # mll.Transformation(
    #     "arcsine_sqrt",
    #     composition_scope="rank-wise",
    # ),
    # mll.Transformation("clr", composition_scope="rank-wise"),
    # mll.Transformation(
    #     "log10",
    #     composition_scope="rank-wise",
    # ),
)

MODELS = (
    # baselines
    # (
    #     "Baseline_RF",
    #     RandomForestClassifier(
    #         n_estimators=1000,
    #         min_samples_leaf=5,
    #         random_state=42,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     "SIAMCAT",
    #     mll.SIAMCATClassifier(
    #         random_state=42,
    #     ),
    # ),
    # # default standard models
    # (
    #     "CB_default",
    #     CatBoostClassifier(
    #         random_seed=42,
    #         thread_count=1,
    #         verbose=False,
    #         allow_writing_files=False,
    #     ),
    # ),
    # (
    #     "RF_default",
    #     RandomForestClassifier(
    #         random_state=42,
    #     ),
    # ),
    # (
    #     "LightGBM_default",
    #     LGBMClassifier(
    #         random_state=42,
    #         n_jobs=1,
    #         verbosity=-1,
    #     ),
    # ),
    # (
    #     "XGB_default",
    #     XGBClassifier(
    #         random_state=42,
    #         n_jobs=1,
    #         verbosity=0,
    #         eval_metric="logloss",
    #     ),
    # ),
    # (
    #     "LR_default",
    #     LogisticRegression(
    #         random_state=42,
    #     ),
    # ),
    # (
    #     "TabPFN",
    #     mll.TabPFNClassifier(
    #         n_estimators="auto",
    #         auto_scale_n_estimators=True,
    #         model_path="auto",
    #         device="auto",
    #         inference_precision="auto",
    #         fit_mode="fit_preprocessors",
    #         memory_saving_mode="auto",
    #         keep_cache_on_device=True,
    #         random_state=42,
    #         n_preprocessing_jobs=1,
    #     ),
    # ),
    # (
    #     "TabDPT_v1_3",
    #     mll.TabDPTClassifier(
    #         n_ensembles=8,
    #         context_size=512,
    #         context_reduction="subsample",
    #         feature_reduction="pca",
    #         normalizer="standard",
    #         temperature=1.0,
    #         device=None,
    #         use_flash=True,
    #         compile=False,
    #         random_state=42,
    #         verbose=False,
    #     ),
    # ),
    #     (
    #     "KumoTabular_small",
    #     mll.KumoTabularClassifier(
    #         size="small",
    #         device="cpu",
    #         random_state=42,
    #     ),
    # ),
    (
        "Causilo_default",
        mll.CausiloClassifier(
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
