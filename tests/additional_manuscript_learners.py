from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB

ADDITIONAL_MODELS = (
    (
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
    (
        "CNB_a1p8e-8",
        ComplementNB(
            alpha=1.8e-8,
            norm=True,
            fit_prior=True,
            force_alpha=True,
        ),
    ),
    (
        "LR_L2_C1e-4",
        LogisticRegression(
            C=0.0001,
            solver="liblinear",
            max_iter=2500,
            random_state=42,
            class_weight=None,
        ),
    ),
)
