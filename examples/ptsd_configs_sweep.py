from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import NearestCentroid

from xgboost import XGBClassifier


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

_RESOLUTION_SETS: list[tuple[str, tuple[str, ...]]] = [
    # ("family", ("family",)),
    # ("genus", ("genus",)),
    # ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
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
        T("log10_relative_abundance_half_min_pseudocount"),
        T("centered_log_ratio_multiplicative_replacement"),
        T("standardized_centered_log_ratio_multiplicative_replacement"),
        T("yeo_johnson_relative_abundance"),
        T("quantile_normal_relative_abundance"),
        T("robust_scaled_relative_abundance"),
        T("within_sample_fractional_rank"),
        T("training_ecdf_rank"),
        T("prevalence_weighted_relative_abundance"),
    ]


def _build_models():
    M = []

    M.append(
        (
            "RF_1000_msl5",
            RandomForestClassifier(
                n_estimators=1000,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=42,
            ),
        )
    )

    # M.append(("Ridge_a1", RidgeClassifier(alpha=1.0, random_state=42)))
    M.append(("BNB", BernoulliNB()))
    # M.append(("NearestCentroid_raw", NearestCentroid()))

    # M.append(
    #     (
    #         "XGB",
    #         XGBClassifier(
    #             objective="binary:logistic",
    #             n_estimators=500,
    #             learning_rate=0.03,
    #             max_depth=3,
    #             min_child_weight=5,
    #             subsample=0.8,
    #             colsample_bytree=0.8,
    #             reg_alpha=0.1,
    #             reg_lambda=1.0,
    #             eval_metric="logloss",
    #             tree_method="hist",
    #             n_jobs=1,
    #             random_state=42,
    #             verbosity=0,
    #         ),
    #     )
    # )

    return M


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
