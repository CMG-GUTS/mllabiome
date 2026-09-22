from __future__ import annotations

from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
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
    task="classification",
    class_labels=("Placebo", "Active"),
    positive_class="Active",
)

DATA = mll.Data(
    abundance_path=Path("examples/data/UC/microbiome.tsv"),
    metadata_path=Path("examples/data/UC/metadata.tsv"),
    format="metaphlan_tsv",
    sample_id_col="patientId",
    target_col="group",
    task="classification",
    class_labels=("control", "colitis"),
    positive_class="colitis",
)

RESOLUTIONS = (
    ("phylum", ("phylum",)),
    ("class", ("class",)),
    ("order", ("order",)),
    ("family", ("family",)),
    ("genus", ("genus",)),
    ("domain-phylum", ("domain", "phylum")),
    ("domain-class", ("domain", "phylum", "class")),
    ("domain-order", ("domain", "phylum", "class", "order")),
    ("domain-family", ("domain", "phylum", "class", "order", "family")),
    ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    ("phylum-class", ("phylum", "class")),
    ("phylum-order", ("phylum", "class", "order")),
    ("phylum-family", ("phylum", "class", "order", "family")),
    ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
    ("class-order", ("class", "order")),
    ("class-family", ("class", "order", "family")),
    ("class-genus", ("class", "order", "family", "genus")),
    ("order-family", ("order", "family")),
    ("order-genus", ("order", "family", "genus")),
    ("family-genus", ("family", "genus")),
    # ("raw", ("all",)),
)

COUNT_TRANSFORMATIONS = (
    mll.Transformation("presence_absence"),
    mll.Transformation("identity"),
    mll.Transformation("arcsine_sqrt"),
    mll.Transformation("yeo_johnson"),
    mll.Transformation("hellinger"),
    mll.Transformation("relative_abundance"),
    mll.Transformation("clr"),
    mll.Transformation("log10"),
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
        "LDA_eigen_shrinkage_auto",
        LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage="auto",
        ),
    ),
    # (
    #     "LabelSpreading_rbf_g0.25_a0.2",
    #     LabelSpreading(
    #         kernel="rbf",
    #         gamma=0.25,
    #         alpha=0.2,
    #         max_iter=1000,
    #         n_jobs=1,
    #     ),
    # ),
    # (
    #     "MLP_64_16_relu_lbfgs_a0.01",
    #     Pipeline(
    #         steps=(
    #             ("scaler", StandardScaler()),
    #             (
    #                 "mlp",
    #                 MLPClassifier(
    #                     hidden_layer_sizes=(64, 16),
    #                     activation="relu",
    #                     solver="lbfgs",
    #                     alpha=1e-2,
    #                     max_iter=3000,
    #                     max_fun=50000,
    #                     random_state=42,
    #                 ),
    #             ),
    #         ),
    #     ),
    # ),
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
    (
        "LightGBM_100_lr0.1_msl20_md20_sub0.8_col0.8",
        LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=20,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        ),
    ),
    (
        "XGB_250_lr0.1_msl3_g0.1_col0.5_sub0.5",
        XGBClassifier(
            learning_rate=0.1,
            n_estimators=250,
            colsample_bytree=0.5,
            subsample=0.5,
            max_depth=3,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=0.5,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        ),
    ),
    (
        "logreg_scaler_saga_c1.0",
        Pipeline(
            steps=(
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        C=1.0,
                        l1_ratio=0.5,
                        max_iter=10000,
                        random_state=42,
                    ),
                ),
            ),
        ),
    ),
    # (
    #     "LightGBM_400_lr0.03_nl7_md3_mcs10_sub0.8_col0.6_l10.1_l21",
    #     LGBMClassifier(
    #         objective="binary",
    #         boosting_type="gbdt",
    #         n_estimators=400,
    #         learning_rate=0.03,
    #         num_leaves=7,
    #         max_depth=3,
    #         min_child_samples=10,
    #         subsample=0.8,
    #         subsample_freq=1,
    #         colsample_bytree=0.6,
    #         reg_alpha=0.1,
    #         reg_lambda=1.0,
    #         random_state=42,
    #         n_jobs=1,
    #         deterministic=True,
    #         force_col_wise=True,
    #         verbose=-1,
    #     ),
    # ),
    (
        "ExtraTrees_1000_msl5",
        ExtraTreesClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=42,
        ),
    ),
    # (
    #     "MLP_32_tanh_lbfgs_a0.01",
    #     Pipeline(
    #         steps=(
    #             ("scaler", StandardScaler()),
    #             (
    #                 "mlp",
    #                 MLPClassifier(
    #                     hidden_layer_sizes=(32,),
    #                     activation="tanh",
    #                     solver="lbfgs",
    #                     alpha=1e-2,
    #                     max_iter=3000,
    #                     max_fun=50000,
    #                     random_state=42,
    #                 ),
    #             ),
    #         ),
    #     ),
    # ),
    # (
    #     "BernoulliNB_a0.1",
    #     BernoulliNB(
    #         alpha=0.1,
    #     ),
    # ),
)

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=2,
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
    max_sizes=(10,),
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
    optimize_metric="nMCC",
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
