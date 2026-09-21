from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from mllabiome import mll

TITLE = "Regression MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/REGRESSION-NCV")

DATA = mll.Data(
    abundance_path=Path("examples/data/REGRESSION/profiles.tsv"),
    metadata_path=Path("examples/data/REGRESSION/metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sample_id",
    target_col="continuous_target",
    task="regression",
)

RESOLUTIONS = (
    ("genus", ("genus",)),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("arcsine_sqrt"),
    mll.Transformation("alr"),
)

MODELS = (
    (
        "RFREG_1000_msl5",
        RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
    ),
)

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=1,
    optimize_metric="RMSE",
    random_state=42,
    n_jobs="auto",
)

GATE = mll.QualificationGate(enabled=False)

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
        "mean_prediction",
        "weighted_mean_prediction",
        "median_prediction",
    ),
    optimize_metric="RMSE",
)

EXPLAINABILITY = mll.Explainability(
    targets="auto",
    top_k=20,
    methods=(mll.Permutation(), mll.SHAP(), mll.ALE(), mll.LIME()),
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
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
)
