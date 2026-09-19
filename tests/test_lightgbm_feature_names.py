import warnings

import numpy as np
import pytest

from mllabiome.metrics import _estimator_call


def test_lgbm_regression_prediction_is_feature_name_consistent():
    lightgbm = pytest.importorskip("lightgbm")
    X = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0],
            [5.0, 4.0, 3.0],
        ],
        dtype=float,
    )
    y = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    model = lightgbm.LGBMRegressor(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.1,
        verbosity=-1,
        random_state=42,
    ).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pred = _estimator_call(model, "predict", X)

    assert np.asarray(pred).shape == (len(X),)
