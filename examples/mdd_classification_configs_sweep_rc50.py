from __future__ import annotations

from pathlib import Path

from mllabiome import mll

TITLE = "MDD case-control within-dataset MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/MDD-CLS-NCV")

DATA = mll.Data(
    abundance_path=Path("examples/data/MDD/BROWN_MDD_profiles_counts.tsv"),
    metadata_path=Path("examples/data/MDD/BROWN_MDD_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sampleId",
    target_col="group",
    task="classification",
    class_labels=("Control", "MDD"),
    positive_class="MDD",
)

RESOLUTIONS = (
    ("genus", ("genus",)),
    ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("identity"),
    mll.Transformation("arcsine_sqrt"),
)

LEARNERS = ("RF_1000_msl5",)

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
    top_k=15,
    representative_instances=False,
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
