from __future__ import annotations
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from mllabiome import mll

TITLE = "PTSD within-dataset MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/PTSD-NCV")
DATA = mll.Data(
    abundance_path=Path("examples/data/PTSD/PTSD_profiles.tsv"),
    metadata_path=Path("examples/data/PTSD/PTSD_metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="sampleId",
    target_col="group",
    class_labels=("Placebo", "Active"),
    positive_class=1,
)
_RESOLUTION_SETS = [
    ("genus", ("genus",)),
    ("domain-family", ("domain", "phylum", "class", "order", "family")),
    ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    ("raw", ("all",)),
]


def _build_count_transformations():
    T = mll.Transformation
    return [
        T("relative_abundance"),
        T("identity"),
        T("presence_absence"),
        T("hellinger"),
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
        ("BNB", BernoulliNB()),
        # ("SIAMCAT"),
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
GATE = mll.QualificationGate(enabled=False, metric="nMCC", threshold=0.51)
ENSEMBLE = mll.Ensemble(
    sizes=(3,),
    selection_strategies=(
        "top_k",
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
