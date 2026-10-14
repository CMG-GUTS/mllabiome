from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.linear_model import SGDClassifier


def initial_parameters(n_features: int) -> list[np.ndarray]:
    if n_features < 1:
        raise ValueError("At least one model feature is required.")
    return [np.zeros((1, n_features), dtype=np.float64), np.zeros(1, dtype=np.float64)]


def validate_parameters(parameters: list[np.ndarray], n_features: int) -> None:
    if len(parameters) != 2:
        raise ValueError("Binary logistic model requires coefficient and intercept arrays.")
    if np.asarray(parameters[0]).shape != (1, n_features) or np.asarray(parameters[1]).shape != (1,):
        raise ValueError("Global parameters do not match the shared feature schema.")
    if not all(np.isfinite(np.asarray(value)).all() for value in parameters):
        raise ValueError("Global parameters contain non-finite values.")


def train_local(
    parameters: list[np.ndarray], X: np.ndarray, y: np.ndarray,
    *, alpha: float, eta0: float, local_epochs: int, random_state: int,
) -> list[np.ndarray]:
    features = np.asarray(X, dtype=np.float64)
    targets = np.asarray(y, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(targets) or not len(features):
        raise ValueError("Local training requires a nonempty aligned matrix and labels.")
    if not np.isfinite(features).all() or not np.isin(targets, (0, 1)).all():
        raise ValueError("Local matrix or labels are invalid.")
    validate_parameters(parameters, features.shape[1])
    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=float(alpha),
        learning_rate="constant", eta0=float(eta0), fit_intercept=True,
        random_state=int(random_state), shuffle=True, average=False,
    )
    clf.partial_fit(features[:1], targets[:1], classes=np.array([0, 1]))
    clf.coef_[:] = np.asarray(parameters[0], dtype=np.float64)
    clf.intercept_[:] = np.asarray(parameters[1], dtype=np.float64)
    for _ in range(int(local_epochs)):
        clf.partial_fit(features, targets)
    result = [np.asarray(clf.coef_, dtype=np.float64).copy(), np.asarray(clf.intercept_, dtype=np.float64).copy()]
    validate_parameters(result, features.shape[1])
    return result


def predict_proba(parameters: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Prediction requires a feature matrix.")
    validate_parameters(parameters, features.shape[1])
    positive = expit(features @ parameters[0].reshape(-1) + parameters[1][0])
    return np.column_stack((1.0 - positive, positive))
