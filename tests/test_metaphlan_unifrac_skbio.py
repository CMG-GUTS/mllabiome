from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


skbio = pytest.importorskip("skbio")
PACKAGE_NAME = "_mllabiome_unifrac_test_package"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "mllabiome")]
sys.modules[PACKAGE_NAME] = package
phylo = importlib.import_module(f"{PACKAGE_NAME}.metaphlan_unifrac")


@pytest.fixture
def reference_tree(tmp_path):
    tree = tmp_path / "reference.nwk"
    tree.write_text("((1:0.5,2:0.5):1,3:1);", encoding="utf-8")
    return tree


def test_unifrac_mathematics_reference_tree(reference_tree):
    train = np.array([[10, 1, 0], [1, 10, 0], [0, 1, 9], [2, 4, 4]], dtype=float)
    names = ["k__Bacteria|s__TaxonA|t__SGB1", "t__SGB2", "t__SGB3"]
    model = phylo.MetaPhlAnUniFracPCoA(
        metric="unweighted_unifrac", tree_path=reference_tree, n_components=4
    )
    model.bind_schema(names, feature_blocks=(("strain", (0, 1, 2)),))
    model.fit(train)
    distances = model._distance_matrix(model._train_counts_)
    assert distances[0, 1] == pytest.approx(0.0)
    assert distances[0, 2] > 0
    assert model.n_positive_axes_ <= 3


def test_training_only_pcoa_projection(reference_tree):
    train = np.array(
        [[8, 0, 1, 8], [0, 9, 1, 9], [2, 2, 6, 2], [4, 4, 2, 4]], dtype=float
    )
    heldout = np.array([[1, 0, 8, 1], [0, 2, 3, 0]], dtype=float)
    names = ["t__SGB1", "t__SGB2", "t__SGB3", "s__Species"]
    model = phylo.MetaPhlAnUniFracPCoA(tree_path=reference_tree, n_components=4)
    model.bind_schema(names, feature_blocks=(("strain", (0, 1, 2)), ("species", (3,))))
    model.fit(train)
    baseline = model.train_coordinates_.copy()
    first, second = model.apply_fitted_pair(train, heldout)
    np.testing.assert_allclose(first, baseline)
    np.testing.assert_allclose(model.apply(train), baseline)
    assert second.shape == (2, 4)
    assert np.isfinite(second).all()
    np.testing.assert_allclose(model.train_coordinates_, baseline)
    assert model.reference_sha256_ and model.reference_path_


def test_rank_selection_and_reference_mismatch(reference_tree):
    model = phylo.MetaPhlAnUniFracPCoA(tree_path=reference_tree)
    model.bind_schema(["s__TaxonA", "s__TaxonB"])
    assert model.selected_rank_ == "species"
    model.bind_schema(["t__SGB1", "t__SGB4"])
    assert model.selected_rank_ == "strain"
    with pytest.raises(
        ValueError, match="at least two distinct reference-matched phylogenetic tips"
    ):
        model.fit(np.array([[1, 19], [10, 10]], dtype=float))


def test_reference_does_not_depend_on_taxon_name_hierarchy(reference_tree):
    train = np.array([[8, 1, 1], [1, 8, 1], [1, 1, 8], [2, 2, 6]], dtype=float)
    first = phylo.MetaPhlAnUniFracPCoA(tree_path=reference_tree)
    first.bind_schema(["a|t__SGB1", "a|t__SGB2", "a|t__SGB3"])
    first.fit(train)
    other = phylo.MetaPhlAnUniFracPCoA(tree_path=reference_tree)
    other.bind_schema(["x|t__SGB1", "y|t__SGB2", "z|t__SGB3"])
    other.fit(train)
    np.testing.assert_allclose(
        first._distance_matrix(first._train_counts_),
        other._distance_matrix(other._train_counts_),
    )
