from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

from mllabiome import mll

# PTSD within-dataset repeated nested-CV sweep.
# Data layout:
#   examples/data/PTSD/PTSD_profiles.tsv   rows=clades, columns=samples
#   examples/data/PTSD/PTSD_metadata.tsv   sampleId + group
# Label encoding:
#   label_map={"Placebo": 0, "Active": 1}

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

# Resolutions sets entries, transformations, and models can be uncommented/edited/added to customise the sweep.
# Resuming is supported - uncommenting/adding entries will add them to the sweep, while already completed entries will be skipped. Commenting out entries will exclude them from the sweep, but they will not be deleted from the report if already completed.

# Active taxonomic-resolution sets. Edit by commenting/uncommenting entries.
# Species and strain are intentionally absent for the provided PTSD example data.
_RESOLUTION_SETS: list[tuple[str, tuple[str, ...]]] = [
    # # Single-rank resolutions
    # # ("domain", ("domain",)),
    # ("phylum", ("phylum",)),
    # ("class", ("class",)),
    ("order", ("order",)),
    # ("family", ("family",)),
    # ("genus", ("genus",)),
    # # Continuous ranges
    # ("domain-phylum", ("domain", "phylum")),
    # ("domain-class", ("domain", "phylum", "class")),
    # ("domain-order", ("domain", "phylum", "class", "order")),
    # ("domain-family", ("domain", "phylum", "class", "order", "family")),
    # ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    # ("domain-species", ("domain", "phylum", "class", "order", "family", "genus", "species")),
    # ("phylum-class", ("phylum", "class")),
    # ("phylum-order", ("phylum", "class", "order")),
    # # ("phylum-family", ("phylum", "class", "order", "family")),
    # # ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
    # ("class-order", ("class", "order")),
    # ("class-family", ("class", "order", "family")),
    # ("class-genus", ("class", "order", "family", "genus")),
    # ("order-family", ("order", "family")),
    # ("order-genus", ("order", "family", "genus")),
    # ("family-genus", ("family", "genus")),
    # # Non-adjacent two-rank selections
    # ("domain+class", ("domain", "class")),
    # ("domain+order", ("domain", "order")),
    # # ("domain+family", ("domain", "family")),
    # # ("domain+genus", ("domain", "genus")),
    # ("phylum+order", ("phylum", "order")),
    # ("phylum+family", ("phylum", "family")),
    # ("phylum+genus", ("phylum", "genus")),
    # ("class+family", ("class", "family")),
    # ("class+genus", ("class", "genus")),
    # ("order+genus", ("order", "genus")),
    ("raw", ("all",)),
]


def _build_count_transformations():
    T = mll.Transformation
    return [
        T("identity"),
        T("relative_abundance"),  # RA
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

    # # Uncomment additional learners as needed.
    # # M.append(("RF_default", RandomForestClassifier(n_jobs=1, random_state=42)))
    # # M.append(("ET_default", ExtraTreesClassifier(n_jobs=1, random_state=42)))
    # # M.append(("HistGB", HistGradientBoostingClassifier(random_state=42)))
    # # M.append(("LR_l2_C1_bal", LogisticRegression(
    # #     penalty="l2", C=1.0, max_iter=3000,
    # #     solver="saga", class_weight="balanced", random_state=42,
    # # )))
    # M.append(("Ridge_a1", RidgeClassifier(alpha=1.0, random_state=42)))
    # M.append(("BNB", BernoulliNB()))
    # # M.append(("GNB", GaussianNB()))
    # # M.append(("kNN_default", KNeighborsClassifier()))
    # M.append(("NearestCentroid_raw", NearestCentroid()))
    # AutoML comparator. Uncomment to include the AutoML strategy in the report.
    # M.append(("FLAML_600s", mll.FLAMLClassifier(time_budget=6, metric="roc_auc", n_jobs=1, random_state=42)))

    # M.append(("SVC_rbf_C1_bal", SVC(
    #     kernel="rbf", C=1.0, gamma="scale",
    #     probability=True, class_weight="balanced", random_state=42,
    # )))

    # M.append((
    #     "SIAMCAT",
    #     mll.SIAMCATClassifier(
    #         method="lasso",
    #         filter_method="abundance",
    #         filter_cutoff=0.001,
    #         normalization="log.std",
    #         num_folds=2,
    #         num_resample=1,
    #         random_state=42,
    #         verbose=0,
    #     ),
    # ))

    # M.append(
    #     "SIAMCAT",
    # )

    return M


EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=2,
    optimize_metric="nMCC",
    random_state=42,
    n_jobs=1,
)

GATE = mll.QualificationGate(enabled=False, metric="nMCC", threshold=0.51)
ENSEMBLE = mll.Ensemble(sizes=(3,), optimize_metric="nMCC")
# EXPLAINABILITY = mll.Explainability(
#     targets="auto",
#     top_k=30,
#     representative_instances=True,
#     instance_sample_ids=(),  # e.g. ("sample_id_to_explain",)
#     top_instance_features=5,
# )

EXPLAINABILITY = mll.Explainability(
    targets=("mpma_b", "mpma_e"),
    methods=("permutation",),
    n_repeats=1,
    top_k=15,
    representative_instances=False,
)
