from pathlib import Path

from catboost import CatBoostRegressor
import lightgbm as lgb
from curated_microbiota_gut_brain_axis.collections import gutbrain
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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

DIETARY_FEATURES = (
    "Kcal",
    "Prot",
    "TF",
    "SatFat",
    "MUF",
    "Poly_InsFat",
    "Chol",
    "Carb",
    "Fiber",
)

DIET = mll.Modality(
    "diet",
    path=gutbrain.metadata,
    format="table",
    feature_cols=DIETARY_FEATURES,
)

RESOLUTIONS = (
    # ("family", ("family",)),
    # ("genus", ("genus",)),
    ("species", ("species",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("identity"),
    # mll.Transformation("clr", composition_scope="rank-wise"),
)

BRAIN_REPRESENTATIONS = (
    mll.BrainRepresentation("brainharmonix_f"),
    # mll.BrainRepresentation("brainharmonix_s"),
    # mll.BrainRepresentation("brainharmonix_sf"),
    # mll.BrainRepresentation("cortexmae_p"),
    # mll.BrainRepresentation("omni_fmri"),
    # mll.BrainRepresentation(
    #     "graph_centrality",
    #     parcels=100,
    #     graph_density=0.15,
    # ),
    mll.BrainRepresentation("labus_2019_centrality"),
    mll.BrainRepresentation("tissue_volumes"),
    # mll.BrainRepresentation("alff", parcels=100),
    # mll.BrainRepresentation("alff", parcels=200),
    # mll.BrainRepresentation("alff", parcels=400),
    # mll.BrainRepresentation("falff", parcels=100),
    # mll.BrainRepresentation("falff", parcels=200),
    # mll.BrainRepresentation("falff", parcels=400),
    # mll.BrainRepresentation("pearson_fc", parcels=100),
    # mll.BrainRepresentation("pearson_fc", parcels=200),
    # mll.BrainRepresentation("pearson_fc", parcels=400),
    # mll.BrainRepresentation("partial_fc", parcels=100),
    # mll.BrainRepresentation("partial_fc", parcels=200),
    # mll.BrainRepresentation("partial_fc", parcels=400),
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
    # (
    #     "CatBoost",
    #     CatBoostRegressor(
    #         random_seed=42,
    #         thread_count=1,
    #         verbose=False,
    #         allow_writing_files=False,
    #     ),
    # ),
    # (
    #     "LGBM",
    #     lgb.LGBMRegressor(
    #         n_estimators=500,
    #         learning_rate=0.025,
    #         num_leaves=7,
    #         random_state=42,
    #         n_jobs=1,
    #         verbosity=-1,
    #     ),
    # ),
    # (
    #     "XGB",
    #     XGBRegressor(),
    # ),
    # (
    #     "Ridge_1",
    #     make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    # ),
    # (
    #     "Ridge_10",
    #     make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    # ),
    # (
    #     "Ridge_100",
    #     make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
    # ),
    # (
    #     "KumoTabular_small",
    #     mll.KumoTabularRegressor(size="small", device="cpu"),
    # ),
)

# INTEGRATIONS = (
#     # mll.Integration("unimodal"),
#     mll.Integration("early_concat", max_modalities=3),
#     mll.Integration(
#         "intermediate_joint_pca",
#         n_components=(16, 32),
#         max_modalities=3,
#     ),
#     # mll.Integration("late_super_learner", max_modalities=3),
# )
INTEGRATIONS = (
    mll.Integration("unimodal"),
    mll.Integration("early_concat", max_modalities=2),
    mll.Integration(
        "intermediate_modality_pca", n_components=(8, 16), max_modalities=2
    ),
    mll.Integration(
        "intermediate_mkl",
        n_components=(8, 16),
        max_modalities=2,
    ),
    mll.Integration(
        "intermediate_block_pls",
        n_components=(2, 4, 8),
        max_modalities=2,
    ),
    mll.Integration(
        "intermediate_sparse_cross_covariance",
        n_components=(2, 4, 8),
        sparsity=0.25,
        max_modalities=2,
    ),
    mll.Integration(
        "intermediate_snf_inductive",
        n_components=(8, 16),
        n_neighbors=10,
        n_iterations=15,
        max_modalities=2,
    ),
    mll.Integration(
        "intermediate_mofa",
        n_components=(4, 8),
        mofa_max_iter=500,
        max_modalities=2,
    ),
    mll.Integration(
        "intermediate_joint_pca",
        n_components=(8, 16),
        max_modalities=2,
    ),
    mll.Integration("late_mean_proba", max_modalities=2),
    mll.Integration("late_weighted_mean_proba", max_modalities=2),
    mll.Integration("late_super_learner", max_modalities=2),
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
    modalities=(MICROBIOTA, DIET),
    brain=BRAIN,
    experiment_dir=EXPERIMENT_DIR,
    title=f"GUTBRAIN multimodal regression · {TARGET}",
    representations={"microbiota": RESOLUTIONS},
    transformations={
        "microbiota": COUNT_TRANSFORMATIONS,
        "diet": (mll.Transformation("standardize"),),
    },
    learners=MODELS,
    integrations=INTEGRATIONS,
    evaluation=EVALUATION,
    gate=mll.QualificationGate(enabled=False, metric="RMSE"),
    explainability=EXPLAINABILITY,
)
