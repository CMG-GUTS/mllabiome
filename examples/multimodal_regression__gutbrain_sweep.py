from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mllabiome import mll

TITLE = "GUTBRAIN microbiota + BrainHarmonix regression sweep"
EXPERIMENT_DIR = Path("examples/runs/GUTBRAIN-MULTIMODAL-NCV")

SAMPLES = mll.Samples(
    path=Path("examples/data/GUTBRAIN/participants.tsv"),
    sample_id_col="participant_id",
    target_col="Zung_S",
    task="regression",
)

MODALITIES = (
    mll.Modality(
        "microbiota",
        path=Path("examples/data/GUTBRAIN/HEALTHY_COLOMBIA_microbiota_profiles.tsv"),
        format="matrix_tsv",
        primary=True,
    ),
    mll.Modality(
        "brainharmonix",
        path=Path("examples/data/GUTBRAIN/brainharmonix_f_mllabiome.tsv"),
        format="matrix_tsv",
    ),
)

REPRESENTATIONS = {
    "microbiota": (("species", ("species",)),),
}

TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    # mll.Transformation("arcsine_sqrt", composition_scope="rank-wise"),
    # mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    # mll.Transformation("yeo_johnson", composition_scope="rank-wise"),
    # mll.Transformation("yeo_johnson", composition_scope="joint"),
    # mll.Transformation("hellinger", composition_scope="rank-wise"),
    # mll.Transformation("hellinger", composition_scope="joint"),
    # mll.Transformation("relative_abundance", composition_scope="rank-wise"),
    # mll.Transformation("relative_abundance", composition_scope="joint"),
    # mll.Transformation("clr", composition_scope="rank-wise"),
    # mll.Transformation("clr", composition_scope="joint"),
    # mll.Transformation("log10", composition_scope="rank-wise"),
    # mll.Transformation("log10", composition_scope="joint"),
    mll.Transformation(
        "relative_abundance",
        composition_scope="joint",
        feature_filter=mll.PrevalenceFilter(
            threshold=0.05,
        ),
    ),
)


INTEGRATIONS = (
    # mll.Integration("unimodal"),
    # mll.Integration("early_concat"),
    mll.Integration("intermediate_joint_pca", n_components=(16, 32)),
    mll.Integration("late_super_learner"),
)

LEARNERS = (
    # (
    #     "ENet_a0p01_l10p5",
    #     make_pipeline(
    #         StandardScaler(),
    #         ElasticNet(
    #             alpha=0.01,
    #             l1_ratio=0.50,
    #             max_iter=20_000,
    #             random_state=42,
    #         ),
    #     ),
    # ),
    (
        "RF_500_msl5",
        RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "LGBM_SMALL",
        lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.025,
            num_leaves=7,
            max_depth=3,
            min_child_samples=15,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=0.10,
            reg_lambda=10.0,
            n_jobs=1,
            random_state=42,
            verbosity=-1,
        ),
    ),
)

EVALUATION = mll.Evaluation(
    protocol="nested_cv",
    outer_folds=5,
    inner_folds=3,
    optimize_metric="RMSE",
    random_state=42,
    n_jobs="auto",
)

GATE = mll.QualificationGate(
    enabled=False,
    metric="RMSE",
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
        mll.Permutation(),
        mll.SHAP(),
        mll.ALE(),
        mll.LIME(),
        mll.ALEInteractions(),
    ),
)

SWEEP = mll.Sweep(
    data=None,
    experiment_dir=EXPERIMENT_DIR,
    title=TITLE,
    learners=LEARNERS,
    evaluation=EVALUATION,
    gate=GATE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
    samples=SAMPLES,
    modalities=MODALITIES,
    representations=REPRESENTATIONS,
    transformations=TRANSFORMATIONS,
    integrations=INTEGRATIONS,
)
