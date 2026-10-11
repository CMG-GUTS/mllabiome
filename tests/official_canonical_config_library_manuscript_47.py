from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from mllabiome import mll
from mllabiome.learners import GroupAwareCalibratedClassifier

RESOLUTION_CATALOG = {
    "R01": ("class", ("class",)),
    "R02": ("class-order", ("class", "order")),
    "R03": ("family", ("family",)),
    "R04": ("genus", ("genus",)),
    "R05": ("genus-species", ("genus", "species")),
    "R06": ("order", ("order",)),
    "R07": ("order-family", ("order", "family")),
    "R08": ("order-genus", ("order", "family", "genus")),
    "R09": ("phylum", ("phylum",)),
    "R10": ("phylum-genus", ("phylum", "class", "order", "family", "genus")),
    "R11": ("phylum-order", ("phylum", "class", "order")),
    "R12": (
        "phylum-species",
        ("phylum", "class", "order", "family", "genus", "species"),
    ),
    "R13": ("raw", ("all",)),
    "R15": ("species", ("species",)),
    "R16": ("species-strain", ("species", "strain")),
    "R17": ("strain", ("strain",)),
    "R18": ("class-family", ("class", "order", "family")),
    "R19": ("class-genus", ("class", "order", "family", "genus")),
    "R20": ("domain-class", ("domain", "phylum", "class")),
    "R21": ("domain-family", ("domain", "phylum", "class", "order", "family")),
    "R22": ("domain-genus", ("domain", "phylum", "class", "order", "family", "genus")),
    "R23": ("domain-order", ("domain", "phylum", "class", "order")),
    "R24": ("domain-phylum", ("domain", "phylum")),
    "R25": ("family-genus", ("family", "genus")),
    "R26": ("phylum-class", ("phylum", "class")),
    "R27": ("phylum-family", ("phylum", "class", "order", "family")),
    "R28": ("family+species", ("family", "species")),
    "R29": ("family-strain", ("family", "genus", "species", "strain")),
}

REPRESENTATION_CATALOG = {
    "T01": mll.MGM2Embedding(
        model="small", embedding_strategy="cls", batch_size=64, top_k_otus=768
    ),
    "T02": mll.MGMEmbedding(batch_size=32, max_length=512, taxonomy_format="full"),
    "T03": mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-6m",
        pooling="last_token",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    "T04": mll.Transformation("arcsine_sqrt", composition_scope="joint"),
    "T05": mll.Transformation("arcsine_sqrt", composition_scope="rank-wise"),
    "T06": mll.Transformation("clr", composition_scope="joint"),
    "T07": mll.Transformation(
        "clr",
        composition_scope="rank-wise",
        feature_filter=mll.PrevalenceFilter(threshold=0.2),
    ),
    "T08": mll.Transformation("identity"),
    "T09": mll.Transformation(
        "identity", feature_filter=mll.PrevalenceFilter(threshold=0.1)
    ),
    "T10": mll.Transformation("log10", composition_scope="rank-wise"),
    "T11": mll.Transformation("presence_absence"),
    "T12": mll.Transformation(
        "relative_abundance",
        composition_scope="joint",
        feature_filter=mll.PrevalenceFilter(threshold=0.2),
    ),
    "T13": mll.Transformation(
        "relative_abundance",
        composition_scope="rank-wise",
        feature_filter=mll.PrevalenceFilter(threshold=0.2),
    ),
    "T14": mll.Transformation("yeo_johnson", composition_scope="joint"),
    "T15": mll.Transformation("yeo_johnson", composition_scope="rank-wise"),
    "T16": mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-170m",
        pooling="last_token",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    "T17": mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-170m",
        pooling="mean",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    "T18": mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-45m",
        pooling="last_token",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    "T19": mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-45m",
        pooling="mean",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    "T20": mll.WaypointEmbedding(
        model="outpost-bio/Waypoint-6m",
        pooling="mean",
        batch_size=32,
        max_length=512,
        taxonomy_format="full",
    ),
    "T21": mll.Transformation(
        "arcsine_sqrt",
        composition_scope="joint",
        feature_filter=mll.PrevalenceFilter(threshold=0.05),
    ),
    "T22": mll.Transformation("clr", composition_scope="rank-wise"),
    "T23": mll.Transformation("hellinger", composition_scope="joint"),
    "T24": mll.Transformation("hellinger", composition_scope="rank-wise"),
    "T25": mll.Transformation("log10", composition_scope="joint"),
    "T26": mll.Transformation("relative_abundance", composition_scope="joint"),
    "T27": mll.Transformation("relative_abundance", composition_scope="rank-wise"),
}

LEARNER_CATALOG = {
    "L01": (
        "CB_i300_d3",
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.04,
            depth=3,
            l2_leaf_reg=10,
            random_strength=1.0,
            rsm=0.6,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
    "L02": (
        "CB_multiclass_i300_d3",
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.04,
            depth=3,
            l2_leaf_reg=10,
            random_strength=1.0,
            rsm=0.6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    ),
    "L03": ("CNB_1.6e7", ComplementNB(alpha=1.6e-07, norm=True, force_alpha=True)),
    "L04": (
        "ExtraTrees_1000_msl5",
        ExtraTreesClassifier(
            n_estimators=1000,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=42,
        ),
    ),
    "L05": (
        "GB_e100",
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            loss="log_loss",
            max_features="sqrt",
            subsample=0.8,
            random_state=17,
        ),
    ),
    "L06": (
        "LDA_eigen_shrinkage_auto",
        LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
    ),
    "L07": (
        "LDA_shrinkage",
        Pipeline(
            steps=(
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            )
        ),
    ),
    "L08": (
        "LGBM_e180",
        LGBMClassifier(
            n_estimators=180,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            colsample_bytree=0.5,
            reg_lambda=2.0,
            reg_alpha=0.1,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        ),
    ),
    "L09": (
        "LGBM_e300",
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.07,
            num_leaves=7,
            max_depth=3,
            min_child_samples=3,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0,
            reg_lambda=0,
            scale_pos_weight=1.0,
            random_state=17,
            verbosity=-1,
        ),
    ),
    "L10": (
        "LR_4e-5_l2",
        LogisticRegression(
            C=4e-05,
            penalty="l2",
            solver="liblinear",
            fit_intercept=True,
            class_weight=None,
            tol=1e-07,
            max_iter=3000,
            random_state=0,
        ),
    ),
    "L11": (
        "LR_C10_balanced",
        LogisticRegression(
            C=10.0, solver="liblinear", class_weight="balanced", max_iter=5000
        ),
    ),
    "L12": (
        "LR_L2_C0002",
        LogisticRegression(
            penalty="l2", C=0.002, solver="liblinear", max_iter=5000, random_state=42
        ),
    ),
    "L13": (
        "LR_c1_balanced_l2",
        LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=42,
        ),
    ),
    "L14": (
        "LR_scaler_c1.0_i3k",
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=3000, random_state=42)),
            ]
        ),
    ),
    "L15": (
        "LR_scaler_saga_c1.0",
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
            )
        ),
    ),
    "L16": (
        "LightGBM_e500",
        LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=40,
            reg_lambda=30.0,
            reg_alpha=1.0,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ),
    ),
    "L17": (
        "LinearSVC_C1",
        LinearSVC(
            C=1.0,
            penalty="l2",
            loss="squared_hinge",
            dual=False,
            tol=0.0001,
            fit_intercept=True,
            class_weight={0: 1.0, 1: 1.4},
            random_state=42,
            max_iter=10000,
        ),
    ),
    "L18": (
        "LinearSVC_C1e-4",
        LinearSVC(
            C=0.0001, loss="squared_hinge", dual="auto", max_iter=30000, random_state=17
        ),
    ),
    "L19": (
        "MLP_32_relu_lbfgs",
        Pipeline(
            steps=(
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        solver="lbfgs",
                        alpha=0.001,
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            )
        ),
    ),
    "L20": (
        "MLP_64_relu_adam",
        Pipeline(
            steps=(
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        activation="relu",
                        solver="adam",
                        alpha=0.0001,
                        learning_rate_init=0.001,
                        max_iter=2000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        random_state=42,
                    ),
                ),
            )
        ),
    ),
    "L21": (
        "RF_1000_msl5",
        RandomForestClassifier(
            n_estimators=1000, min_samples_leaf=5, n_jobs=1, random_state=42
        ),
    ),
    "L22": (
        "RF_82_msl4",
        RandomForestClassifier(
            n_estimators=82,
            max_features=52,
            min_samples_leaf=4,
            min_samples_split=2,
            class_weight="balanced_subsample",
            criterion="gini",
            bootstrap=True,
            random_state=42,
        ),
    ),
    "L23": ("SIAMCAT", mll.SIAMCATClassifier()),
    "L24": (
        "SVM_RBF",
        Pipeline(
            steps=(
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        random_state=42,
                    ),
                ),
            )
        ),
    ),
    "L25": (
        "TabPFN",
        mll.TabPFNClassifier(
            n_estimators="auto",
            auto_scale_n_estimators=True,
            model_path="auto",
            device="auto",
            inference_precision="auto",
            fit_mode="fit_preprocessors",
            memory_saving_mode="auto",
            keep_cache_on_device=True,
            random_state=42,
            n_preprocessing_jobs=1,
        ),
    ),
    "L26": (
        "kNN5_cosine",
        Pipeline(
            steps=(
                ("normalizer", Normalizer(norm="l2")),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                        metric="cosine",
                        algorithm="brute",
                    ),
                ),
            )
        ),
    ),
    "L27": (
        "LGBM_500_l7",
        LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.08,
            reg_lambda=15.0,
            reg_alpha=0.5,
            n_jobs=1,
            random_state=42,
            verbosity=-1,
        ),
    ),
    "L28": (
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
    "L29": (
        "LinearSVC_c1.1e-6_cw5.4",
        LinearSVC(
            C=1.1e-06,
            loss="squared_hinge",
            dual="auto",
            class_weight={"0": 1.0, "1": 5.4},
            tol=1e-08,
            max_iter=30000,
            random_state=17,
        ),
    ),
    "L30": (
        "RF_1000_mf007_msl8",
        RandomForestClassifier(
            n_estimators=1000,
            max_features=0.07,
            min_samples_leaf=8,
            min_samples_split=2,
            max_depth=None,
            criterion="gini",
            bootstrap=True,
            class_weight=None,
            n_jobs=1,
            random_state=42,
        ),
    ),
    "L31": (
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
    "L32": (
        "RC_a0p02",
        RidgeClassifier(alpha=0.02, solver="lsqr", tol=0.001, max_iter=1000),
    ),
    "L33": (
        "ET_500_d17_mf0p02_CW0x8",
        ExtraTreesClassifier(
            n_estimators=500,
            criterion="gini",
            max_depth=17,
            max_features=0.02,
            min_samples_leaf=3,
            class_weight={0: 8.0, 1: 1.0},
            n_jobs=1,
            random_state=42,
        ),
    ),
    "L34": (
        "LS_RBF_g0p25_a0p2",
        LabelSpreading(kernel="rbf", gamma=0.25, alpha=0.2, max_iter=1000),
    ),
    "L35": (
        "XGB_500_d6_lr0p05",
        XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        ),
    ),
    "L36": (
        "ET_500_leaf3_Bal",
        ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        ),
    ),
    "L37": (
        "SVM_RBF_C10_Bal_GroupCal",
        GroupAwareCalibratedClassifier(
            estimator=SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=False,
                class_weight="balanced",
            ),
            method="sigmoid",
            n_splits=3,
            random_state=42,
        ),
    ),
    "L38": ("RC_a0p1_Softmax", mll.RidgeSoftmaxClassifier(alpha=0.1)),
    "L39": ("BNB_a0p1", BernoulliNB(alpha=0.1)),
    "L40": (
        "RF_750_leaf3",
        RandomForestClassifier(
            n_estimators=750, min_samples_leaf=3, n_jobs=1, random_state=42
        ),
    ),
}

ORIGINAL_LEARNER_NAMES = {key: entry[0] for key, entry in LEARNER_CATALOG.items()}

CANONICAL_LEARNER_NAMES = {
    "L01": "CB_300_d3",
    "L02": "CB_300_d3_MC",
    "L03": "CNB_a1p6e-7",
    "L04": "ET_1000_leaf5",
    "L05": "GB_100_lr0p1",
    "L06": "LDA_Eigen_Shrink",
    "L07": "LDA_LSQR_Shrink_Scaled",
    "L08": "LGBM_180_lr0p03",
    "L09": "LGBM_300_lr0p07",
    "L10": "LR_L2_C4e-5",
    "L11": "LR_L2_C10_Bal",
    "L12": "LR_L2_C0p002",
    "L13": "LR_L2_C1_Bal",
    "L14": "LR_L2_C1_Scaled",
    "L15": "LR_SAGA_C1_Scaled",
    "L16": "LGBM_500_l7_leaf40",
    "L17": "SVM_Linear_C1_Bal",
    "L18": "SVM_Linear_C1e-4",
    "L19": "MLP_32_LBFGS",
    "L20": "MLP_64_Adam",
    "L21": "RF_1000_leaf5",
    "L22": "RF_82_mf52_leaf4",
    "L23": "SIAMCAT_Default",
    "L24": "SVM_RBF_C1",
    "L25": "TabPFN_Auto",
    "L26": "KNN_5_Cosine",
    "L27": "LGBM_500_l7_leaf30",
    "L28": "LGBM_100_lr0p1",
    "L29": "SVM_Linear_C1p1e-6_Bal",
    "L30": "RF_1000_mf0p07_leaf8",
    "L31": "XGB_250_d3",
    "L32": "RC_a0p02",
    "L33": "ET_500_d17_mf0p02_CW0x8",
    "L34": "LS_RBF_g0p25_a0p2",
    "L35": "XGB_500_d6_lr0p05",
    "L36": "ET_500_leaf3_Bal",
    "L37": "SVM_RBF_C10_Bal_GroupCal",
    "L38": "RC_a0p1_Softmax",
    "L39": "BNB_a0p1",
    "L40": "RF_750_leaf3",
}

LEARNER_CATALOG = {
    key: (CANONICAL_LEARNER_NAMES[key], estimator)
    for key, (_, estimator) in LEARNER_CATALOG.items()
}

RESOLUTION_ID_ALIASES = {"R14": "R13"}
SOURCE_ACTIVE_LEARNER_IDS = tuple(f"L{value:02d}" for value in range(1, 27))
MANUSCRIPT_ADDITION_IDS = tuple(f"L{value:02d}" for value in range(32, 41))
MANUSCRIPT_TABLE_LEARNER_IDS = (
    "L19",
    "L21",
    "L34",
    "L35",
    "L36",
    "L37",
    "L38",
    "L06",
    "L39",
    "L40",
)
MANUSCRIPT_TABLE_RESOLUTION_IDS = (
    "R25",
    "R15",
    "R08",
    "R04",
    "R28",
    "R16",
    "R23",
    "R29",
)
ACTIVE_IDS = {
    "RESOLUTIONS": tuple(key for key in RESOLUTION_CATALOG if int(key[1:]) <= 17),
    "COUNT_TRANSFORMATIONS": tuple(
        key for key in REPRESENTATION_CATALOG if int(key[1:]) <= 15
    ),
    "MODELS": SOURCE_ACTIVE_LEARNER_IDS,
}

LEARNER_CATALOG.update(
    {
        "L41": ("MNB_a0p5", MultinomialNB(alpha=0.5)),
        "L42": ("MNB_a0p1", MultinomialNB(alpha=0.1)),
        "L43": (
            "BAG_LR_40_C0p03",
            BaggingClassifier(
                estimator=Pipeline(
                    steps=(
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(C=0.03, max_iter=3000, random_state=42),
                        ),
                    )
                ),
                n_estimators=40,
                max_samples=0.8,
                bootstrap=True,
                n_jobs=1,
                random_state=42,
            ),
        ),
        "L44": (
            "LR_L2_C0p01_Scaled",
            Pipeline(
                steps=(
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(C=0.01, max_iter=3000, random_state=42),
                    ),
                )
            ),
        ),
    }
)

ORIGINAL_LEARNER_NAMES.update(
    {
        key: item[0]
        for key, item in LEARNER_CATALOG.items()
        if key in ("L41", "L42", "L43", "L44")
    }
)
CANONICAL_LEARNER_NAMES.update(
    {
        key: item[0]
        for key, item in LEARNER_CATALOG.items()
        if key in ("L41", "L42", "L43", "L44")
    }
)
PTSD_SCREEN_ADDITION_IDS = ("L41", "L42", "L43", "L44")
PTSD_SCREEN_PRIMARY_IDS = ("L41", "L42", "L43", "L44")
PTSD_SCREEN_REPRESENTATION_POLICY = {
    "L41": ("relative_abundance",),
    "L42": ("relative_abundance",),
    "L43": ("arcsine_sqrt",),
    "L44": ("arcsine_sqrt",),
}

LEARNER_CATALOG.update(
    {
        "L45": (
            "SGD_MH_a1e-7",
            SGDClassifier(
                loss="modified_huber",
                alpha=1e-7,
                penalty="l2",
                max_iter=2000,
                tol=1e-4,
                random_state=42,
                class_weight=None,
                average=False,
            ),
        ),
        "L46": (
            "CNB_a1p8e-8",
            ComplementNB(
                alpha=1.8e-8,
                norm=True,
                fit_prior=True,
                force_alpha=True,
            ),
        ),
        "L47": (
            "LR_L2_C1e-4",
            LogisticRegression(
                C=0.0001,
                solver="liblinear",
                max_iter=2500,
                random_state=42,
                class_weight=None,
            ),
        ),
    }
)

ORIGINAL_LEARNER_NAMES.update(
    {key: LEARNER_CATALOG[key][0] for key in ("L45", "L46", "L47")}
)
CANONICAL_LEARNER_NAMES.update(
    {key: LEARNER_CATALOG[key][0] for key in ("L45", "L46", "L47")}
)
NEW_LEARNER_IDS = ("L45", "L46", "L47")
NEW_LEARNER_REPRESENTATION_POLICY = {
    "L45": ("arcsine_sqrt", "relative_abundance", "clr"),
    "L46": ("relative_abundance",),
    "L47": ("arcsine_sqrt", "relative_abundance", "clr"),
}
