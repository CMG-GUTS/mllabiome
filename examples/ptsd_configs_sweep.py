from __future__ import annotations

from pathlib import Path

from mllabiome import mll

TITLE = "PTSD within-dataset MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/PTSD-NCV")

DATA = mll.Data(
    abundance_path=Path("examples/data/PTSD/PTSD_profiles.tsv"),
    metadata_path=Path("examples/data/PTSD/PTSD_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sampleId",
    target_col="group",
    task="classification",
    class_labels=("Placebo", "Active"),
    positive_class="Active",
)

RESOLUTIONS = (
    ("genus", ("genus",)),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    # mll.Transformation("clr"),
    mll.Transformation("arcsine_sqrt"),
    mll.Transformation("alr"),
    # mll.Transformation("ilr"),
)

LEARNERS = (
    "RF_1000_msl5",
    "BNB",
)

EVALUATION = mll.Evaluation(
    protocol="nested_cv",
    outer_folds=5,
    inner_folds=3,
    optimize_metric="log_loss",
    random_state=42,
    n_jobs="auto",
)

GATE = mll.QualificationGate(
    enabled=False,
    metric="nMCC",
    threshold=0.51,
)

ENSEMBLE = mll.Ensemble(
    max_sizes=(3,),
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
        "rank_mean",
        "majority_vote",
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

SWEEP = mll.Sweep(
    data=DATA,
    experiment_dir=EXPERIMENT_DIR,
    title=TITLE,
    resolutions=RESOLUTIONS,
    count_transformations=COUNT_TRANSFORMATIONS,
    learners=LEARNERS,
    evaluation=EVALUATION,
    gate=GATE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
)
