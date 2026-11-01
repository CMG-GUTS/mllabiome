from __future__ import annotations

from pathlib import Path

from curated_microbiota.collections import lampp_scz
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mllabiome import mll

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = (
    HERE if HERE.name == "benchmark" else HERE.parent / "examples" / "benchmark"
)
TITLE = "Schizophrenia inner-selected MPMA-B and MPMA-E"
EXPERIMENT_DIR = BENCHMARK_DIR / "runs" / "scz--inner-selected"

DATA = lampp_scz.mllabiome()

EVALUATION = lampp_scz.splits(
    benchmark="mllabiome-benchmark-v2",
).mllabiome(
    optimize_metric="log_loss",
    n_jobs="auto",
)

RESOLUTIONS = (
    ("species", ("species",)),
    ("order", ("order",)),
    ("strain", ("strain",)),
)

COUNT_TRANSFORMATIONS = (
    mll.ScopedTransformation("prev_weighted", resolutions=("species",)),
    mll.ScopedTransformation("presence_absence", resolutions=("order",)),
    mll.ScopedTransformation("arcsine_sqrt", resolutions=("strain",)),
)

MODELS = (
    (
        "KumoTabular_medium",
        mll.KumoTabularClassifier(
            size="medium",
            device="mps",
            pretrained=True,
            n_estimators=8,
            random_state=42,
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
)

LEARNER_RESOLUTIONS = {
    "KumoTabular_medium": ("species",),
    "XGB_default": ("order",),
    "Baseline_RF": ("strain",),
    "SIAMCAT": ("species",),
}

GATE = mll.QualificationGate(enabled=False)

ENSEMBLE = mll.Ensemble(
    include_inactive=True,
    mpma_b_candidate_learners=("KumoTabular_medium",),
    exclude_learners=("Baseline_RF", "SIAMCAT"),
    exclude_transformations=("relative_abundance@rank-wise",),
    max_sizes=(5,),
    selection_strategies=("diversity",),
    aggregation_strategies=("bayesian_model_averaging",),
    optimize_metric="AUROC",
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

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b", "baseline_rf"),  # "mpma_e"),
    profile="screening",
    methods=(
        mll.Permutation(),
        mll.ALE(),
        mll.LIME(),
        mll.ALEInteractions(top_k=50),
    ),
    classes="auto",
)

ROBUSTNESS = mll.Robustness(
    targets=("mpma_b", "mpma_e"),
    top_k=30,
)

EXTERNAL_TEST = lampp_scz.external_test

if EXTERNAL_TEST is None:
    raise RuntimeError("SCZ external test set is unavailable")

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
    learner_resolutions=LEARNER_RESOLUTIONS,
    evaluation=EVALUATION,
    gate=GATE,
    explore=EXPLORE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
    robustness=ROBUSTNESS,
    inference=INFERENCE,
)
