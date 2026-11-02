from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from mllabiome import mll

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
TITLE = "Joint multilabel microbiome sweep"
EXPERIMENT_DIR = HERE / "runs" / "joint_multilabel"

TARGETS = (
    "MoodDisFinal",
    "AnxDisFinal",
    "ADHD_Final",
    "ASD_Final",
)

DATA = mll.Data(
    abundance_path=Path("examples/multi-target-joint-optimization/data"),
    metadata=mll.Metadata(
        metadata_path=Path("examples/multi-target-joint-optimization/data") / "simulated_metadata_rdoc_regression_with_rdoc_targets.tsv",
    ),
    format="mllab",
    sample_id_col="Deelnemerscode",
    target_col=TARGETS,
    task="multilabel",
)

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
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
    permutations=99,
    bootstrap_replicates=200,
    random_state=42,
)

RESOLUTIONS = (
    ("class", ("class",)),

    ("family", ("family",)),

    ("genus", ("genus",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("relative_abundance", composition_scope="rank-wise"),
    mll.Transformation("hellinger", composition_scope="rank-wise"),
    mll.Transformation("presence_absence"),
)

MODELS = (
    (
        "JointRF_100",
        RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
    ),
    (
        "JointExtraTrees_100",
        ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
    ),
    (
        "JointMLP_64_32",
        mll.JointMLPClassifier(
            hidden_layer_sizes=(64, 32),
            dropout=0.1,
            epochs=80,
            batch_size=32,
            random_state=42,
        ),
    ),
)

GATE = mll.QualificationGate(enabled=False)

ENSEMBLE = mll.Ensemble(
    max_sizes=(2,),
    selection_strategies=(
        "top_k",
        "best_per_learner_type",
        "performance_diversity",
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
    targets=("mpma_b", ),
    profile="screening",
    methods=(mll.Permutation(n_repeats=5, scoring="log_loss"),),
    top_k=20,
    random_state=42,
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
)

