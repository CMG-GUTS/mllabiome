from __future__ import annotations

from pathlib import Path

from curated_microbiota.collections import lampp_scz

from mllabiome import mll


HERE = Path(__file__).resolve().parent
TITLE = "CAMPP schizophrenia mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "CAMPP-SCZ-NCV-CM-official"

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

RESOLUTIONS = (("raw", ("all",)),)

COUNT_TRANSFORMATIONS = (
    mll.Transformation(
        "prevalence_rank_fusion",
        composition_scope="rank-wise",
        alpha=0.25,
        c=100.0,
        gamma=1.0,
        pseudo=0.5,
    ),
)


MODELS = (
    (
        "KumoTabular_small_e4",
        mll.KumoTabularClassifier(
            size="small",
            n_estimators=4,
            pretrained=True,
            device="mps",
            random_state=42,
        ),
    ),
)


GATE = mll.QualificationGate(
    enabled=False,
    metric="MCC",
    threshold=0.02,
)


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
    targets=("mpma_b",),
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
