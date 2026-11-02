from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mllabiome import mll

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
TITLE = "Joint RDoC multi-target microbiome sweep"
EXPERIMENT_DIR = HERE / "runs" / "joint_rdoc"

TARGETS = (
    "Negative_Valence",
    "Cognitive_Systems",
    "Arousal_Regulatory",
    "Social_Processes",
)

DATA = mll.Data(
    abundance_path=Path("examples/multi-target-joint-optimization/data") / "simulated_biome_rdoc_signal.tsv",
    metadata=mll.Metadata(
        metadata_path=Path("examples/multi-target-joint-optimization/data") / "simulated_metadata_rdoc_regression_with_rdoc_targets.tsv",
    ),
    format="mllab",
    sample_id_col="Deelnemerscode",
    target_col=TARGETS,
    task="multioutput",
    target_tasks={target: "regression" for target in TARGETS},
    joint=True,
)

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=2,
    optimize_metric="normalized_rmse",
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
    # ("class", ("class",)),
    # ("family", ("family",)),
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
        RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
    ),
    (
        "JointExtraTrees_100",
        ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
    ),
    (
        "JointMLP_64_32",
        mll.JointMLPRegressor(
            hidden_layer_sizes=(64, 32),
            dropout=0.1,
            epochs=80,
            batch_size=32,
            random_state=42,
        ),
    ),
    (
        "JointPLS_4",
        Pipeline((
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=4)),
        )),
    ),
)

GATE = mll.QualificationGate(enabled=False)

ENSEMBLE = mll.Ensemble(
    max_sizes=(2, 3),
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "performance_diversity",
        "caruana",
        "super_learner",
    ),
    aggregation_strategies=(
        "mean_prediction",
        "weighted_mean_prediction",
        "median_prediction",
    ),
    optimize_metric="normalized_rmse",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b", "ensemble:top_k__mean_prediction__2"),
    profile="screening",
    methods=(mll.Permutation(n_repeats=5, scoring="rmse"),),
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

