from __future__ import annotations
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mllabiome import mll

TITLE = "MDD group depression severity and sex multi-output MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/MDD-MULTIOUTPUT-3-NCV")
DATA = mll.Data(
    abundance_path=Path("examples/data/MDD/BROWN_MDD_profiles_counts.tsv"),
    metadata_path=Path("examples/data/MDD/BROWN_MDD_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sampleId",
    target_col=("group", "depression_severity_PROMIS", "sex"),
    task="multioutput",
    target_tasks={
        "group": "classification",
        "depression_severity_PROMIS": "regression",
        "sex": "classification",
    },
    target_class_labels={"sex": ("Female", "Male")},
)
_RESOLUTION_SETS = [
    ("genus", ("genus",)),
    ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    ("raw", ("all",)),
]


def _build_count_transformations():
    T = mll.Transformation
    return [T("identity"), T("arcsine_sqrt")]


def _build_models():
    return {
        "classification": [
            (
                "RF_1000_msl5",
                RandomForestClassifier(
                    n_estimators=1000,
                    min_samples_leaf=5,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ],
        "regression": [
            (
                "RF_1000_msl5",
                RandomForestRegressor(
                    n_estimators=1000,
                    min_samples_leaf=5,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ],
    }


EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=1,
    optimize_metric={"classification": "log_loss", "regression": "RMSE"},
    random_state=42,
    n_jobs="auto",
)
GATE = mll.QualificationGate(enabled=False)
ENSEMBLE = mll.Ensemble(
    optimize_metric={"classification": "log_loss", "regression": "RMSE"}
)
EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    profile="screening",
    methods=(mll.Permutation(), mll.SHAP()),
    classes="auto",
    top_k=15,
    representative_instances=False,
)
