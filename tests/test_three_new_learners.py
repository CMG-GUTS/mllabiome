import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB

from additional_manuscript_learners import ADDITIONAL_MODELS


def test_names_unique_and_explicit_parameters():
    models = dict(ADDITIONAL_MODELS)
    assert tuple(models) == ("SGD_MH_a1e-7", "CNB_a1p8e-8", "LR_L2_C1e-4")
    assert isinstance(models["SGD_MH_a1e-7"], SGDClassifier)
    assert isinstance(models["CNB_a1p8e-8"], ComplementNB)
    assert isinstance(models["LR_L2_C1e-4"], LogisticRegression)
    assert models["SGD_MH_a1e-7"].get_params()["loss"] == "modified_huber"
    assert models["SGD_MH_a1e-7"].get_params()["alpha"] == 1e-7
    assert models["CNB_a1p8e-8"].get_params()["alpha"] == 1.8e-8
    assert models["CNB_a1p8e-8"].get_params()["force_alpha"] is True
    assert models["LR_L2_C1e-4"].get_params()["C"] == 0.0001
    assert models["LR_L2_C1e-4"].get_params()["solver"] == "liblinear"


def test_positive_feature_group_safe_probabilities():
    rng = np.random.default_rng(17)
    groups = np.repeat(np.arange(24), 2)
    y = np.repeat(np.arange(24) % 2, 2)
    X = rng.gamma(shape=2.0, scale=1.0, size=(len(y), 16))
    X[y == 1, :4] += 2.0
    X /= X.sum(axis=1, keepdims=True)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in cv.split(X, y, groups):
        assert not (set(groups[train]) & set(groups[test]))
        for name, estimator in ADDITIONAL_MODELS:
            fit = clone(estimator).fit(X[train], y[train])
            proba = fit.predict_proba(X[test])
            assert proba.shape == (len(test), 2), name
            assert np.isfinite(proba).all(), name
            assert np.all(proba >= 0.0) and np.all(proba <= 1.0), name
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
            assert np.isfinite(log_loss(y[test], proba, labels=[0, 1]))


def test_signed_features_only_valid_for_discriminative_models():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(60, 12))
    y = np.arange(60) % 2
    for name, estimator in ADDITIONAL_MODELS:
        if name.startswith("CNB_"):
            with pytest.raises(ValueError):
                clone(estimator).fit(X, y)
        else:
            proba = clone(estimator).fit(X, y).predict_proba(X[:10])
            assert proba.shape == (10, 2)
            assert np.isfinite(proba).all()


def test_catalog_contains_47_unique_entries():
    source = Path(__file__).with_name(
        "official_canonical_config_library_manuscript_47.py"
    )
    tree = ast.parse(source.read_text())
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "LEARNER_CATALOG" for t in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                for key, value in zip(node.value.keys, node.value.values):
                    k = ast.literal_eval(key)
                    names[k] = ast.literal_eval(value.elts[0])
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "update"
                and isinstance(func.value, ast.Name)
                and func.value.id == "LEARNER_CATALOG"
            ):
                data = node.value.args[0]
                assert isinstance(data, ast.Dict)
                for key, value in zip(data.keys, data.values):
                    k = ast.literal_eval(key)
                    assert k not in names
                    names[k] = ast.literal_eval(value.elts[0])
    assert len(names) == 47
    assert len(set(names.values())) == 47
    assert tuple(names[k] for k in ("L45", "L46", "L47")) == tuple(
        name for name, _ in ADDITIONAL_MODELS
    )
