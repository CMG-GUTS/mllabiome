from __future__ import annotations
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from mllabiome import mll

TITLE = "MDD depression severity within-dataset MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/MDD-REG-NCV")
DATA = mll.Data(
    abundance_path=Path("examples/data/MDD/BROWN_MDD_profiles_counts.tsv"),
    metadata_path=Path("examples/data/MDD/BROWN_MDD_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sampleId",
    target_col="depression_severity_PROMIS",
    task="regression",
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
    return [
        (
            "RF_1000_msl5",
            RandomForestRegressor(
                n_estimators=1000,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=42,
            ),
        ),
    ]


EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=1,
    optimize_metric="RMSE",
    random_state=42,
    n_jobs="auto",
)
GATE = mll.QualificationGate(enabled=False, metric="RMSE", threshold=None)
ENSEMBLE = mll.Ensemble(optimize_metric="RMSE")
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
