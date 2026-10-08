from __future__ import annotations

from pathlib import Path

from catboost import CatBoostRegressor
from curated_microbiota.collections import healthy_colombia
from sklearn.ensemble import RandomForestRegressor

from mllabiome import mll

HERE = Path(__file__).resolve().parent
TITLE = "Healthy Colombia Zung depression severity regression mllabiome benchmark sweep"
EXPERIMENT_DIR = HERE / "runs" / "HEALTHY-COLOMBIA-ZUNG-REGRESSION-NCV-CM"

DATA = healthy_colombia.mllabiome(target="zung_depression")

EVALUATION = healthy_colombia.splits(
    target="zung_depression",
    benchmark="mllabiome-benchmark-v2",
).mllabiome(
    optimize_metric="RMSE",
    n_jobs="auto",
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
    ("class", ("class",)),
    ("genus", ("genus",)),
    ("class-order", ("class", "order")),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    mll.Transformation("yeo_johnson", composition_scope="joint"),
    mll.Transformation(
        "relative_abundance",
        composition_scope="joint",
        feature_filter=mll.PrevalenceFilter(threshold=0.20),
    ),
    mll.Transformation("log10", composition_scope="rank-wise"),
)

MODELS = (
    (
        "RFReg_1000_msl5",
        RandomForestRegressor(
            n_estimators=1000, min_samples_leaf=5, n_jobs=1, random_state=42
        ),
    ),
    (
        "CBReg_abundance_i300_d3",
        CatBoostRegressor(
            iterations=300,
            learning_rate=0.04,
            depth=3,
            l2_leaf_reg=10,
            random_strength=1.0,
            rsm=0.60,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
)

GATE = mll.QualificationGate(enabled=False, metric="RMSE")

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
    targets=("mpma_b",),
    profile="screening",
    methods=(
        mll.Permutation(scoring="RMSE"),
        mll.SHAP(),
        mll.ALE(),
        mll.LIME(),
        mll.ALEInteractions(),
    ),
)

ROBUSTNESS = mll.Robustness(targets=("mpma_b",), top_k=30)

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
)
