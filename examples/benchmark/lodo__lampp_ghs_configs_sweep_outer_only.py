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
                                                                        
                                      
EVALUATION.outer_only = True

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
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("species", ("species",)),
    ("strain", ("strain",)),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("identity"), 
    mll.Transformation(
        "hellinger",
        composition_scope="rank-wise",
    ),
)

MODELS = (
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
    (
        "Causilo_default",
        mll.CausiloClassifier(
            random_state=42,
        ),
    ),
    (
        "KumoTabular_small",
        mll.KumoTabularClassifier(
            size="small",
            device="cpu",
            random_state=42,
        ),
    ),
    (
        "KumoTabular_medium",
        mll.KumoTabularClassifier(
            size="medium",
            device="cpu",
            random_state=42,
        ),
    ),
    (
        "KumoTabular_large",
        mll.KumoTabularClassifier(
            size="large",
            device="cpu",
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
