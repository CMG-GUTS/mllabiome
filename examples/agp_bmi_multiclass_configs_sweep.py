from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from mllabiome import mll

TITLE = "AGP-2021 BMI-category multiclass MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/AGP-2021-BMI-NCV")

DATA = mll.Data(
    abundance_path=Path("examples/data/IBS/AGP-2021_profiles.tsv"),
    metadata_path=Path("examples/data/IBS/AGP-2021_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="Name",
    target_col="bmi_cat",
    class_labels=("Underweight", "Normal", "Overweight", "Obese"),
)

_RESOLUTION_SETS: list[tuple[str, tuple[str, ...]]] = [
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("raw", ("all",)),
]


def _build_count_transformations():
    T = mll.Transformation
    return [
        T("identity"),
        T("relative_abundance"),
        T("arcsine_sqrt"),
    ]


def _build_models():
    return [
        (
            "RF_1000_msl5",
            RandomForestClassifier(
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
    optimize_metric="nMCC",
    random_state=42,
    n_jobs="auto",
)

GATE = mll.QualificationGate(
    enabled=False,
    metric="nMCC",
    threshold=0.51,
)

ENSEMBLE = mll.Ensemble(
    sizes=(3,),
    optimize_metric="nMCC",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    methods=("permutation",),
    n_repeats=1,
    top_k=15,
    representative_instances=False,
)
