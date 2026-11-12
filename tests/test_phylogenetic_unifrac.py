from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[1] / "mllabiome" / "phylogenetic_unifrac.py"
spec = importlib.util.spec_from_file_location("phylogenetic_unifrac", MODULE)
phylo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phylo)


@pytest.fixture
def simple_tree(tmp_path):
    path = tmp_path / "species.nwk"
    path.write_text("((A:1,B:1):1,C:1);", encoding="utf-8")
    return path


def test_gunifrac_formula_alpha_1_and_0_5(simple_tree):

    X = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    a1 = phylo.generalized_unifrac_distance_matrix(
        X,
        feature_names=["A", "B", "C"],
        tree_path=simple_tree,
        alpha=1,
    )
    a05 = phylo.generalized_unifrac_distance_matrix(
        X,
        feature_names=["A", "B", "C"],
        tree_path=simple_tree,
        alpha=0.5,
    )
    assert a1[0, 1] == pytest.approx(0.5)
    assert a05[0, 1] == pytest.approx(1 / (1 + 1 / np.sqrt(2)))
    assert np.array_equal(np.diag(a05), [0, 0])


def test_projection_is_training_fitted_not_joint(simple_tree):
    train = np.array([[8, 0, 1], [0, 9, 1], [2, 2, 6], [4, 4, 2]], dtype=float)
    heldout = np.array([[1, 0, 8], [0, 2, 3]], dtype=float)
    model = phylo.GeneralizedUniFracPCoA(simple_tree, n_components=6)
    model.bind_schema(["A", "B", "C"], feature_blocks=(("species", (0, 1, 2)),))
    model.fit(train)
    fit_axes = model.train_coordinates_.copy()
    fitted_evals = model._eigenvectors_.copy()
    xtrain, xtest = model.apply_fitted_pair(train, heldout)
    np.testing.assert_allclose(xtrain, fit_axes)
    np.testing.assert_allclose(model.transform(train), fit_axes, atol=1e-11)
    assert xtrain.shape == (4, 6) and xtest.shape == (2, 6)
    assert np.isfinite(xtest).all()
    assert np.array_equal(fitted_evals, model._eigenvectors_)
    assert np.all(xtrain[:, 3:] == 0)


def test_unmatched_strict_and_coverage(simple_tree):
    data = np.array([[5, 5], [1, 9]], dtype=float)
    with pytest.raises(ValueError, match="cannot be matched"):
        phylo.generalized_unifrac_distance_matrix(
            data,
            feature_names=["A", "Unknown"],
            tree_path=simple_tree,
        )
    with pytest.raises(ValueError, match="coverage below"):
        phylo.generalized_unifrac_distance_matrix(
            data,
            feature_names=["A", "Unknown"],
            tree_path=simple_tree,
            unmatched="drop",
            min_sample_coverage=0.9,
        )


def test_cross_dataset_name_mapping_and_subset(simple_tree, tmp_path):
    mapping = tmp_path / "map.tsv"
    mapping.write_text("feature\ttip\nSpecies X\tA\nSpecies Y\tB\n", encoding="utf-8")
    first = phylo.generalized_unifrac_distance_matrix(
        np.array([[1, 0], [0, 1.0]]),
        feature_names=["Species X", "Species Y"],
        tree_path=simple_tree,
        feature_to_tip=mapping,
    )
    second = phylo.generalized_unifrac_distance_matrix(
        np.array([[1, 0], [0, 1.0]]),
        feature_names=["A", "B"],
        tree_path=simple_tree,
    )
    np.testing.assert_allclose(first, second)
    assert np.isfinite(
        phylo.generalized_unifrac_distance_matrix(
            np.array([[1, 0], [0, 1.0]]),
            feature_names=["A", "C"],
            tree_path=simple_tree,
        )
    ).all()


def test_tree_and_mapping_immutable_after_setup(simple_tree, tmp_path):
    mapping = tmp_path / "map.csv"
    mapping.write_text("feature,tip\nX,A\nY,B\n", encoding="utf-8")
    model = phylo.GeneralizedUniFracPCoA(simple_tree, feature_to_tip=mapping)
    model.bind_schema(["X", "Y"])
    mapping.write_text("feature,tip\nX,B\nY,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping changed"):
        model.fit(np.array([[1, 0], [0, 1.0]]))


def test_validation_on_tree_and_mixed_ranks(tmp_path, simple_tree):
    bad = tmp_path / "invalid.nwk"
    bad.write_text("(A:1,B:-1);", encoding="utf-8")
    model = phylo.GeneralizedUniFracPCoA(bad)
    model.bind_schema(["A", "B"])
    with pytest.raises(ValueError, match="branch length"):
        model.fit(np.array([[1, 0], [0, 1.0]]))
    another = phylo.GeneralizedUniFracPCoA(simple_tree)
    with pytest.raises(ValueError, match="must not mix"):
        another.bind_schema(
            ["A", "B"], feature_blocks=(("species", (0,)), ("genus", (1,)))
        )
