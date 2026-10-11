from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MNB_MODELS = (
    ("MNB_a0p5", MultinomialNB(alpha=0.5)),
    ("MNB_a0p1", MultinomialNB(alpha=0.1)),
)

ARC_MODELS = (
    (
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
    (
        "LR_L2_C0p01_Scaled",
        Pipeline(
            steps=(
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.01, max_iter=3000, random_state=42)),
            )
        ),
    ),
)

MODEL_REPRESENTATION = {
    "MNB_a0p5": "relative_abundance",
    "MNB_a0p1": "relative_abundance",
    "BAG_LR_40_C0p03": "arcsine_sqrt",
    "LR_L2_C0p01_Scaled": "arcsine_sqrt",
}
