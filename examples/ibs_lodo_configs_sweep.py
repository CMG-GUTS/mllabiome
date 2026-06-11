from __future__ import annotations
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC

from mllabiome import mll


# IBS leave-one-dataset-out sweep.
# Input directory contains one pair per cohort:
#   examples/data/IBS/<study>_profiles.tsv
#   examples/data/IBS/<study>_metadata.tsv
# The profile TSVs are MetaPhlAn-style matrices: rows=clades, columns=samples.

TITLE = "IBS LODO MPMA sweep"
DATA_DIR = Path("examples/data/IBS")
PREPARED_CSV = DATA_DIR / "IBS_lodo_mllabiome.csv"
EXPERIMENT_DIR = Path("examples/runs/IBS-LODO")

STUDY_IDS = (
    "AGP-2021",
    "Fukui-2020",
    "Hugerth-2019",
    "Liu-2020",
    "LoPresti-2019",
    "Nagel-2016",
)

SAMPLE_ID_COL = "Name"
TARGET_COL = "host_disease"
POSITIVE_LABEL = "IBS"


def _prepare_ibs_lodo_csv(data_dir: Path = DATA_DIR, out_path: Path = PREPARED_CSV) -> Path:
    frames = []
    rank_prefixes = ("d__", "k__", "p__", "c__", "o__", "f__", "g__", "s__", "t__")

    for study_id in STUDY_IDS:
        profiles_path = data_dir / f"{study_id}_profiles.tsv"
        metadata_path = data_dir / f"{study_id}_metadata.tsv"
        if not profiles_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Missing IBS input files for {study_id}: {profiles_path} / {metadata_path}"
            )

        profiles = pd.read_csv(profiles_path, sep="\t", index_col=0)
        profiles.index = profiles.index.astype(str).str.strip()
        profiles = profiles[profiles.index.str.startswith(rank_prefixes)]
        profiles = profiles.T
        profiles.index = profiles.index.astype(str).str.strip()
        profiles.index.name = "sample_id"

        metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
        if SAMPLE_ID_COL not in metadata.columns:
            metadata = metadata.rename(columns={metadata.columns[0]: SAMPLE_ID_COL})
        if TARGET_COL not in metadata.columns:
            raise ValueError(
                f"{metadata_path} is missing {TARGET_COL!r}. Available columns: {list(metadata.columns)}"
            )
        metadata[SAMPLE_ID_COL] = metadata[SAMPLE_ID_COL].astype(str).str.strip()
        metadata = metadata.drop_duplicates(subset=[SAMPLE_ID_COL]).set_index(SAMPLE_ID_COL)

        common = profiles.index.intersection(metadata.index)
        if len(common) == 0:
            raise ValueError(f"No sample overlap for IBS study {study_id}.")

        X = profiles.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        raw_label = metadata.loc[common, TARGET_COL].astype(str).str.strip()
        y = raw_label.where(raw_label.eq(POSITIVE_LABEL), "non-IBS")

        frame = X.copy()
        frame.insert(0, "study_id", study_id)
        frame.insert(0, "label", y.to_numpy(dtype=object))
        frame.insert(0, "sample_id", [f"{study_id}::{sid}" for sid in common.astype(str)])
        frames.append(frame.reset_index(drop=True))

    out = pd.concat(frames, axis=0, join="outer", ignore_index=True).fillna(0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


DATA = mll.Data(
    abundance_path=_prepare_ibs_lodo_csv(),
    metadata_path=None,
    format="csv",
    sample_id_col="sample_id",
    target_col="label",
    group_col="study_id",
    label_map={"non-IBS": 0, "IBS": 1},
    class_labels=("non-IBS", "IBS"),
    positive_class=1,
)


# Active taxonomic-resolution sets. Edit by commenting/uncommenting entries.
_RESOLUTION_SETS: list[tuple[str, tuple[str, ...]]] = [
    # Single-rank resolutions
    # ("domain", ("domain",)),
    # ("phylum", ("phylum",)),
    ("class", ("class",)),
    ("order", ("order",)),
    # ("family", ("family",)),
    # ("genus", ("genus",)),

    # Continuous ranges up to genus
    # ("domain-phylum", ("domain", "phylum")),
    # ("domain-class", ("domain", "phylum", "class")),
    # ("domain-order", ("domain", "phylum", "class", "order")),
    # ("domain-family", ("domain", "phylum", "class", "order", "family")),
    # ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    # ("phylum-class", ("phylum", "class")),
    # ("phylum-order", ("phylum", "class", "order")),
    # ("phylum-family", ("phylum", "class", "order", "family")),
    # ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
    # ("class-order", ("class", "order")),
    # ("class-family", ("class", "order", "family")),
    # ("class-genus", ("class", "order", "family", "genus")),
    # ("order-family", ("order", "family")),
    # ("order-genus", ("order", "family", "genus")),
    # ("family-genus", ("family", "genus")),

    # Non-adjacent two-rank selections
    # ("domain+class", ("domain", "class")),
    # ("domain+order", ("domain", "order")),
    # ("domain+family", ("domain", "family")),
    # ("domain+genus", ("domain", "genus")),
    # ("phylum+order", ("phylum", "order")),
    # ("phylum+family", ("phylum", "family")),
    # ("phylum+genus", ("phylum", "genus")),
    # ("class+family", ("class", "family")),
    # ("class+genus", ("class", "genus")),
    # ("order+genus", ("order", "genus")),
]


def _build_count_transformations():
    T = mll.Transformation
    return [
        T("none"),          # RA
        # T("arcsin_sqrt"), # arcsin-sqrt
        # T("binary"),      # P/A
        # T("hellinger"),   # Hellinger
        # T("rank_col"),    # ECDF-rank
        # T("zi_log"),      # ZI-log
        # T("prev_weighted"),
        # T("pairwise_logratio"),
        # T("scikit-bio_clr"),
        # T("scikit-bio_alr"),
        # T("scikit-bio_ilr"),
    ]


def _build_models():
    M = []

    # M.append(("RF_1000_msl5", RandomForestClassifier(
    #     n_estimators=1000,
    #     min_samples_leaf=5,
    #     n_jobs=1,
    #     random_state=42,
    # )))

    # Uncomment additional learners as needed.
    # M.append(("RF_default", RandomForestClassifier(n_jobs=1, random_state=42)))
    # M.append(("ET_default", ExtraTreesClassifier(n_jobs=1, random_state=42)))
    # M.append(("LR_l2_C1_bal", LogisticRegression(
    #     penalty="l2", C=1.0, max_iter=3000,
    #     solver="saga", class_weight="balanced", random_state=42,
    # )))
    M.append(("Ridge_a1", RidgeClassifier(alpha=1.0, random_state=42)))
    M.append(("BNB", BernoulliNB()))
    # M.append(("GNB", GaussianNB()))
    # M.append(("kNN_default", KNeighborsClassifier()))
    # M.append(("NearestCentroid_raw", NearestCentroid()))
    # AutoML comparator. Uncomment to include the AutoML strategy in the report.
    # M.append(("FLAML_600s", mll.FLAMLClassifier(time_budget=600, metric="roc_auc", n_jobs=1, random_state=42)))

    # M.append(("SVC_rbf_C1_bal", SVC(
    #     kernel="rbf", C=1.0, gamma="scale",
    #     probability=True, class_weight="balanced", random_state=42,
    # )))

    return M


EVALUATION = mll.Evaluation(
    protocol="lodo",
    inner_folds=3,
    repeats=1,
    optimize_metric="nMCC",
    random_state=42,
    n_jobs=1,
)

GATE = mll.QualificationGate(enabled=False, metric="nMCC", threshold=0.51)
ENSEMBLE = mll.Ensemble(sizes=(3,), optimize_metric="nMCC")
EXPLAINABILITY = mll.Explainability(
    targets="auto",
    top_k=30,
    representative_instances=True,
    instance_sample_ids=(),
    top_instance_features=5,
)
