from __future__ import annotations

import bz2
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / "mllabiome"
PACKAGE = "_metaphlan_reference_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
for name in ("taxonomy_features", "metaphlan_unifrac"):
    fullname = f"{PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(fullname, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)

reference = sys.modules[f"{PACKAGE}.metaphlan_unifrac"]


@pytest.fixture
def resources(tmp_path):
    tree = tmp_path / "small.nwk"
    tree.write_text("((1:0.5,2:0.5):1,3:1);", encoding="utf-8")
    taxonomy = tmp_path / "small_species.txt.bz2"
    with bz2.open(taxonomy, "wt", encoding="utf-8") as output:
        output.write("SGB1\tk__Bacteria|g__A|s__A_x\n")
        output.write("SGB2\tk__Bacteria|g__A|s__A_y\n")
        output.write("SGB3\tk__Bacteria|g__B|s__B_x\n")
    return tree, taxonomy


def test_auto_selects_finest_available_rank(resources):
    tree, taxonomy = resources
    model = reference.MetaPhlAnUniFracPCoA(tree_path=tree, taxonomy_path=taxonomy)
    model.bind_schema(("k__Bacteria___g__A", "k__Bacteria___g__B"))
    assert model.selected_rank_ == "genus"
    model.bind_schema(("g__A", "s__A_x", "s__B_x", "g__B"))
    assert model.selected_rank_ == "species"
    model.bind_schema(("g__A", "s__A_x", "t__SGB1", "g__B"))
    assert model.selected_rank_ == "strain"


def test_official_mapping_matches_name_and_lineage(resources):
    _, taxonomy = resources
    mapping = reference._load_taxonomy(taxonomy)
    matched, missing = reference._resolve_named_taxa(
        ("k__Bacteria___g__A", "g__B"),
        "genus",
        mapping,
        frozenset(("SGB1", "SGB2", "SGB3")),
    )
    assert matched == {0: ("SGB1", "SGB2"), 1: ("SGB3",)}
    assert missing == ()
    matched, missing = reference._resolve_named_taxa(
        ("g__Unknown", "g__B"), "genus", mapping, frozenset(("SGB1", "SGB2", "SGB3"))
    )
    assert matched == {1: ("SGB3",)}
    assert missing == ("g__Unknown",)


def test_coverage_rejects_missing_reference_abundance(resources):
    tree, taxonomy = resources
    model = reference.MetaPhlAnUniFracPCoA(tree_path=tree, taxonomy_path=taxonomy)
    model.bind_schema(("g__A", "g__B", "g__Unknown"))
    model.n_features_in_ = 3
    model._matched_columns_ = np.asarray((0, 1), dtype=int)
    with pytest.raises(ValueError, match="coverage"):
        model._extract(np.array(((1, 1, 10), (3, 2, 0)), dtype=float))
