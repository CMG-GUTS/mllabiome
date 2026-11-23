
from __future__ import annotations

from pathlib import Path

from curated_microbiota.collections import lampp_ghs

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "CAMPP general health status LODO MPMA sweep"
EXPERIMENT_DIR = HERE / "runs" / "CAMPP-GHS-NCV-53-repeated2"

DATA = lampp_ghs.mllabiome()

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    outer_only=False,
    repeats=2,
    optimize_metric="log_loss",
    random_state=42,
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
    ("raw", ("all",)),
)


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
        'KumoTabular_small_e4',
        mll.KumoTabularClassifier(
            size='small',
            n_estimators=4,
            pretrained=True,
            device='mps',
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
    max_sizes=(2, 3, 4, 5),
    pooled_super_learner_pool_size=128,
    selection_strategies=(
        # "top_k",
        # "super_learner",
        "qo_es",
        "qdo_es",
        "drep",
        "pseo_bqp",
    ),
    aggregation_strategies=(
        "mean_proba",
        "cawpe",
        "weighted_mean_proba",
        "logistic_stacking",
    ),
    optimize_metric="AUROC",
    final_selection_strategy="pareto_auroc_logloss_ideal",
    advanced_selection_pool_size=128,
    population_iterations=400,
    population_archive_size=32,
    population_random_state=42,
    qdo_archive_bins=6,
    drep_rho=0.5,
    pseo_bqp_omega=0.5,
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
    gate=GATE,
    explore=EXPLORE,
    base_selection=BASE_SELECTION,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
    robustness=ROBUSTNESS,
    inference=INFERENCE,
)
