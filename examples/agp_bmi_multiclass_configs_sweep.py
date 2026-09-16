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
    # ("phylum", ("phylum",)),
    # ("class", ("class",)),
    # ("order", ("order",)),
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("raw", ("all",)),
]


def _build_count_transformations():
    T = mll.Transformation
    return [
        T("identity"),
        T("relative_abundance"),
        T("presence_absence"),
        T("hellinger"),
        T("arcsine_sqrt"),
        # T("log10_relative_abundance_half_min_pseudocount"),
        # T("centered_log_ratio_multiplicative_replacement"),
        # T("standardized_centered_log_ratio_multiplicative_replacement"),
        # T("yeo_johnson_relative_abundance"),
        # T("quantile_normal_relative_abundance"),
        # T("robust_scaled_relative_abundance"),
        # T("within_sample_fractional_rank"),
        # T("training_ecdf_rank"),
        # T("prevalence_weighted_relative_abundance"),
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
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    ),
    aggregation_strategies=("mean_proba",),
    optimize_metric="log_loss",
)

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b",),
    methods=("permutation",),
    n_repeats=1,
    top_k=15,
    representative_instances=False,
)
