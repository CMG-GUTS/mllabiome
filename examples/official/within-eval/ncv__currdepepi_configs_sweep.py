from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from mllabiome import mll

TITLE = "CDE CurrDepEpiFinal binary MPMA sweep"
EXPERIMENT_DIR = Path("examples/runs/CDE-CURRDEPI-NCV-fm-210g")

DATA = mll.Data(
    abundance_path=Path(
        "/Users/agata/released/mllabiome/sandbox/manuscript-experiments/data-internal/CDE/microbiota.tsv"
    ),
    metadata_path=Path(
        "/Users/agata/released/mllabiome/sandbox/manuscript-experiments/data-internal/CDE/metadata_currdepepi.tsv"
    ),
    format="metaphlan_tsv",
    sample_id_col="feces",
    target_col="CurrDepEpiFinal",
    task="classification",
    class_labels=(0, 1),
    positive_class=1,
)

RESOLUTIONS = (
    # ("phylum", ("phylum",)),
    # ("class", ("class",)),
    # ("order", ("order",)),
    # ("family", ("family",)),
    ("genus", ("genus",)),
    ("species", ("species",)),
    # # ("domain-phylum", ("domain", "phylum")),
    # # ("domain-class", ("domain", "phylum", "class")),
    # # ("domain-order", ("domain", "phylum", "class", "order")),
    # # ("domain-family", ("domain", "phylum", "class", "order", "family")),
    # # ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    # # ("phylum-class", ("phylum", "class")),
    # # ("phylum-order", ("phylum", "class", "order")),
    # ("phylum-family", ("phylum", "class", "order", "family")),
    # # ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
    # ("class-order", ("class", "order")),
    # # ("class-family", ("class", "order", "family")),
    # ("class-genus", ("class", "order", "family", "genus")),
    # # ("order-family", ("order", "family")),
    # # ("order-genus", ("order", "family", "genus")),
    # ("family-genus", ("family", "genus")),
    # ("raw", ("raw",)),
)

COUNT_TRANSFORMATIONS = (
    mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-6m",
        pooling="last_token",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    mll.MGM2Embedding(
        model="small",
        embedding_strategy="cls",
        batch_size=64,
        top_k_otus=768,
    ),
    mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-6m",
        pooling="mean",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-45m",
        pooling="last_token",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-45m",
        pooling="mean",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-170m",
        pooling="last_token",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-170m",
        pooling="mean",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    mll.MGMEmbedding(
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    # mll.Transformation("presence_absence"),
    # mll.Transformation("identity"),
    # mll.Transformation("arcsine_sqrt", composition_scope="rank-wise"),
    mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    # # # mll.Transformation("yeo_johnson", composition_scope="rank-wise"),
    # mll.Transformation("yeo_johnson", composition_scope="joint"),
    # # mll.Transformation("hellinger", composition_scope="rank-wise"),
    # # mll.Transformation("hellinger", composition_scope="joint"),
    # # # mll.Transformation("relative_abundance", composition_scope="rank-wise"),
    # # mll.Transformation("relative_abundance", composition_scope="joint"),
    # mll.Transformation(
    #     "relative_abundance",
    #     composition_scope="joint",
    #     feature_filter=mll.PrevalenceFilter(
    #         threshold=0.20,
    #     ),
    # ),
    # # mll.Transformation("clr", composition_scope="rank-wise"),
    # # mll.Transformation("clr", composition_scope="joint"),
    # mll.Transformation("log10", composition_scope="rank-wise"),
    # # mll.Transformation("log10", composition_scope="joint"),
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
    # (
    #     "CB_i300_d3",
    #     CatBoostClassifier(
    #         iterations=300,
    #         learning_rate=0.04,
    #         depth=3,
    #         l2_leaf_reg=10,
    #         random_strength=1.0,
    #         rsm=0.60,
    #         loss_function="Logloss",
    #         eval_metric="Logloss",
    #         random_seed=42,
    #         thread_count=1,
    #         verbose=False,
    #         allow_writing_files=False,
    #     ),
    # ),
    # (
    #     "SIAMCAT",
    #     mll.SIAMCATClassifier(
    #     ),
    # ),
    # (
    #     "LDA_eigen_shrinkage_auto",
    #     LinearDiscriminantAnalysis(
    #         solver="eigen",
    #         shrinkage="auto",
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
    # (
    #     "TabPFN",
    #     mll.TabPFNClassifier(
    #         n_estimators="auto",
    #         auto_scale_n_estimators=True,
    #         model_path="auto",
    #         device="auto",
    #         inference_precision="auto",
    #         fit_mode="fit_preprocessors",
    #         memory_saving_mode="auto",
    #         keep_cache_on_device=True,
    #         random_state=42,
    #         n_preprocessing_jobs=1,
    #     ),
    # ),
)

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=2,
    optimize_metric="log_loss",
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
        "quality_diversity",
        "diversity",
        "performance_diversity",
        "caruana",
        "bagged_caruana",
        "caruana_multistart",
        "beam",
        "super_learner",
        "adaptive_super_learner",
        "safe_super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "trimmed_mean_proba",
        "geometric_mean_proba",
        "rank_mean",
        "majority_vote",
        "trimmed_mean",
        "logit_mean",
        "logistic_stacking",
        "elastic_net_stacking",
        "cohort_robust_stacking",
    ),
    optimize_metric="MCC",
)

EVALUATION = mll.Evaluation(
    protocol="repeated_nested_cv",
    outer_folds=5,
    inner_folds=3,
    repeats=2,
    optimize_metric="log_loss",
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
        "quality_diversity",
        "caruana",
        "bagged_caruana",
        "super_learner",
        "adaptive_super_learner",
        "safe_super_learner",
    ),
    aggregation_strategies=(
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "trimmed_mean_proba",
        "geometric_mean_proba",
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
