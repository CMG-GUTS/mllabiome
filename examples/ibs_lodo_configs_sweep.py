from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mllabiome import mll

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
CASE_LABEL = "IBS"
CONTROL_LABEL = "Healthy"


def _prepare_ibs_lodo_csv(
    data_dir: Path = DATA_DIR,
    out_path: Path = PREPARED_CSV,
) -> Path:
    frames = []
    rank_prefixes = ("d__", "k__", "p__", "c__", "o__", "f__", "g__", "s__", "t__")
    allowed_labels = {CASE_LABEL, CONTROL_LABEL}

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
                f"{metadata_path} is missing {TARGET_COL!r}. "
                f"Available columns: {list(metadata.columns)}"
            )
        metadata[SAMPLE_ID_COL] = metadata[SAMPLE_ID_COL].astype(str).str.strip()
        metadata = metadata.drop_duplicates(subset=[SAMPLE_ID_COL]).set_index(
            SAMPLE_ID_COL
        )

        common = profiles.index.intersection(metadata.index)
        if len(common) == 0:
            raise ValueError(f"No sample overlap for IBS study {study_id}.")
        X = profiles.loc[common].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        raw_label = metadata.loc[common, TARGET_COL].astype(str).str.strip()
        unexpected = sorted(set(raw_label.unique()) - allowed_labels)
        if unexpected:
            raise ValueError(
                f"Unexpected {TARGET_COL!r} labels in {study_id}: {unexpected}. "
                f"Expected only {sorted(allowed_labels)}."
            )

        y = raw_label.map({CONTROL_LABEL: "non-IBS", CASE_LABEL: "IBS"})
        frame = X.copy()
        frame.insert(0, "study_id", study_id)
        frame.insert(0, "label", y.to_numpy(dtype=object))
        frame.insert(
            0, "sample_id", [f"{study_id}::{sid}" for sid in common.astype(str)]
        )
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
    task="classification",
    group_col="study_id",
    label_map={"non-IBS": 0, "IBS": 1},
    class_labels=("non-IBS", "IBS"),
    positive_class="IBS",
)

RESOLUTIONS = (
    ("genus", ("genus",)),
    ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("identity"),
    mll.Transformation("arcsine_sqrt"),
)

MODELS = (
    (
        "RF_1000_msl5",
        RandomForestClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "MLP_32_relu_lbfgs",
        Pipeline(
            steps=(
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        solver="lbfgs",
                        alpha=1e-3,
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ),
        ),
    ),
)

EVALUATION = mll.Evaluation(
    protocol="lodo",
    inner_folds=3,
    optimize_metric="nMCC",
    random_state=42,
    n_jobs=1,
)

GATE = mll.QualificationGate(
    enabled=False,
    metric="nMCC",
    threshold=0.51,
)

ENSEMBLE = mll.Ensemble(
    max_sizes=(3,),
    selection_strategies=(
        "top_k",
        "best_per_resolution",
        "best_per_learner_type",
        "caruana",
        "super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "rank_mean",
        "majority_vote",
    ),
    optimize_metric="log_loss",
)

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
)
SWEEP = mll.Sweep(
    data=DATA,
    experiment_dir=EXPERIMENT_DIR,
    title=TITLE,
    resolutions=RESOLUTIONS,
    count_transformations=COUNT_TRANSFORMATIONS,
    learners=MODELS,
    evaluation=EVALUATION,
    gate=GATE,
    ensemble=ENSEMBLE,
    explainability=EXPLAINABILITY,
)
