from __future__ import annotations

from pathlib import Path

from curated_microbiota.collections import lampp_ghs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "LAMPP general health status LODO mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "LAMPP-GHS-LODO-CM-official-v2"

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
    ("species", ("species",)),
    ("strain", ("strain",)),
)

COUNT_TRANSFORMATIONS = (mll.Transformation("presence_absence"),)

MODELS = (
    (
        "CNB_1.6e7",
        ComplementNB(
            alpha=1.6e-7,
            norm=True,
            force_alpha=True,
        ),
    ),
    # (
    #     "LinearSVC_c1.1e-6_cw5.4",
    #     LinearSVC(
    #         C=1.1e-6,
    #         loss="squared_hinge",
    #         dual="auto",
    #         class_weight={"0": 1.0, "1": 5.4},
    #         tol=1e-8,
    #         max_iter=30000,
    #         random_state=17,
    #     ),
    # ),
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
