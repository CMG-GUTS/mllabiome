from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
from curated_microbiota_gut_brain_axis.collections import gutbrain
from sklearn.ensemble import RandomForestRegressor

from mllabiome import mll

TARGET = "Zung_S"
TITLE = f"GUTBRAIN microbiota + BrainHarmonix {TARGET} regression sweep"
EXPERIMENT_DIR = Path("examples/runs/GUTBRAIN-MULTIMODAL-NCV")

DATA = gutbrain.mllabiome(target=TARGET)

TRANSFORMATIONS = (
    # mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    # mll.Transformation(
    #     "relative_abundance",
    #     composition_scope="joint",
    #     feature_filter=mll.PrevalenceFilter(threshold=0.05),
    # ),
)

INTEGRATIONS = (
    mll.Integration("intermediate_joint_pca", n_components=(16, 32)),
    mll.Integration("late_super_learner"),
    mll.Integration("early_concat"),
)

LEARNERS = (
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
    (
        "KumoTabular_small",
        mll.KumoTabularRegressor(
            size="small",
            pretrained=True,
            n_estimators=8,
            device="cpu",
            random_state=42,
        ),
    ),
    (
        "KumoTabular_medium",
        mll.KumoTabularClassifier(
            size="medium",
            device="cpu",
            random_state=42,
        ),
    ),
    (
        "KumoTabular_large",
        mll.KumoTabularClassifier(
            size="large",
            device="cpu",
            random_state=42,
        ),
    ),
)

EVALUATION = gutbrain.splits(target=TARGET).mllabiome(
    optimize_metric="RMSE",
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
    **DATA,
    experiment_dir=EXPERIMENT_DIR,
    title=TITLE,
    learners=LEARNERS,
    evaluation=EVALUATION,
    gate=GATE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
    transformations=TRANSFORMATIONS,
    integrations=INTEGRATIONS,
)
