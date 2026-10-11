import numpy as np
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from ptsd_selected_models import ARC_MODELS, MNB_MODELS, MODEL_REPRESENTATION


def test_unique_names_and_representation_policy():
    learners = dict((*MNB_MODELS, *ARC_MODELS))
    assert len(learners) == 4
    assert set(learners) == set(MODEL_REPRESENTATION)
    assert MODEL_REPRESENTATION["MNB_a0p5"] == "relative_abundance"
    assert MODEL_REPRESENTATION["MNB_a0p1"] == "relative_abundance"


def test_grouped_fitting_produces_valid_probabilities():
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(24), 2)
    y = np.repeat(np.arange(24) % 2, 2)
    x = rng.gamma(1.0, 1.0, size=(len(y), 20))
    x[y == 1, 0] += 2.5
    x = x / x.sum(axis=1, keepdims=True)
    split = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train, valid in split.split(x, y, groups):
        assert set(groups[train]).isdisjoint(set(groups[valid]))
        for _, base in (*MNB_MODELS, *ARC_MODELS):
            model = clone(base)
            model.fit(x[train], y[train])
            proba = model.predict_proba(x[valid])
            assert proba.shape == (len(valid), 2)
            assert np.isfinite(proba).all()
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
            assert np.isfinite(log_loss(y[valid], proba, labels=[0, 1]))


def test_multinomial_nb_rejects_negative_features():
    x = np.asarray([[1.0, -0.2], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    y = np.asarray([0, 1, 0, 1])
    for _, model in MNB_MODELS:
        try:
            clone(model).fit(x, y)
        except ValueError:
            continue
        raise AssertionError("MultinomialNB must reject negative inputs")
