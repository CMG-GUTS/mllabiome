from pathlib import Path

import lightgbm as lgb
from curated_microbiota_gut_brain_axis.collections import gutbrain
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mllabiome import mll


TARGET = "Zung_S"
EXPERIMENT_DIR = Path("examples/runs/GUTBRAIN-BRAIN-REPRESENTATIONS-v2")
OUTCOME = gutbrain.target(TARGET)

SAMPLES = mll.Samples(
    path=gutbrain.metadata,
    sample_id_col="participant_id",
    target_col=OUTCOME.column,
    task=OUTCOME.task,
)

MICROBIOTA = mll.Modality(
    "microbiota",
    path=gutbrain.microbiota,
    format="matrix_tsv",
    primary=True,
)

RESOLUTIONS = (("species", ("species",)),)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("identity"),
    mll.Transformation("clr", composition_scope="rank-wise"),
)

BRAIN_REPRESENTATIONS = (
    mll.BrainRepresentation("brainharmonix_f"),
    mll.BrainRepresentation("tissue_volumes"),
)

BRAIN = mll.Brain(
    derivatives_dir=Path("/Users/agata/dev/MAIGBA/derivatives/fmriprep"),
    layout="modalities",
    n_jobs="auto",
    representations=BRAIN_REPRESENTATIONS,
)

MODELS = (
    (
        "RF",
        RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        ),
    ),
    (
        "LGBM",
        lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.025,
            num_leaves=7,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ),
    ),
    (
        "Ridge_1",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
    ),
    (
        "Ridge_10",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
    ),
    (
        "Ridge_100",
        make_pipeline(
            StandardScaler(),
            Ridge(alpha=100.0),
        ),
    ),
    (
        "KumoTabular_small",
        mll.KumoTabularRegressor(
            size="small",
            device="cpu",
        ),
    ),
)

INTEGRATIONS = (
    # mll.Integration("unimodal"),
    # mll.Integration("early_concat"),
    mll.Integration("intermediate_joint_pca", n_components=(16, 32)),
    # mll.Integration("late_super_learner"),
)

EVALUATION = gutbrain.splits(target=TARGET).mllabiome(
    optimize_metric="RMSE",
    n_jobs="auto",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    profile="screening",
    methods=(mll.Permutation(),),
)

SWEEP = mll.Sweep(
    data=None,
    samples=SAMPLES,
    modalities=(MICROBIOTA,),
    brain=BRAIN,
    experiment_dir=EXPERIMENT_DIR,
    title=f"GUTBRAIN microbiota and brain representations · {TARGET}",
    resolutions=RESOLUTIONS,
    count_transformations=COUNT_TRANSFORMATIONS,
    representations={"microbiota": RESOLUTIONS},
    transformations={"microbiota": COUNT_TRANSFORMATIONS},
    learners=MODELS,
    integrations=INTEGRATIONS,
    evaluation=EVALUATION,
    gate=mll.QualificationGate(enabled=False, metric="RMSE"),
    explainability=EXPLAINABILITY,
)
