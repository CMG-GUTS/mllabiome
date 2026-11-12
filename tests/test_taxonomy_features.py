from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "mllabiome" / "taxonomy_features.py"
spec = importlib.util.spec_from_file_location("taxonomy_features", MODULE)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


@pytest.mark.parametrize(
    "label,rank",
    [
        ("k__Bacteria___p__Firmicutes___g__Blautia", "genus"),
        ("k__Bacteria|g__Blautia", "genus"),
        ("g__Blautia", "genus"),
        ("k__Bacteria___g__Blautia___s__Blautia_wexlerae", "species"),
        ("s__Blautia_wexlerae", "species"),
        ("k__Bacteria___g__Blautia___s__Blautia_wexlerae___t__SGB123", "strain"),
        ("t__SGB123_group", "strain"),
        ("p__Firmicutes;g__Blautia", "genus"),
    ],
)
def test_rank_across_delimiters(label, rank):
    assert features.taxonomic_rank(label) == rank


@pytest.mark.parametrize(
    "label,expected",
    [
        ("k__Bacteria___g__Blautia___t__SGB123", "123"),
        ("t__SGB123_group", "123"),
        ("g__Blautia|t__SGB10", "10"),
        ("s__GGB1_SGB123", None),
        ("g__Blautia", None),
    ],
)
def test_sgb_ids_only_at_tip_rank(label, expected):
    assert features.sgb_identifier(label) == expected


def test_lineage_specific_matching():
    target = features.taxonomic_tokens("k__Bacteria___g__Blautia")
    reference = features.taxonomic_tokens(
        "k__Bacteria|p__Firmicutes|g__Blautia|s__Blautia_wexlerae"
    )
    unrelated = features.taxonomic_tokens("k__Archaea|g__Blautia")
    assert features.lineage_matches(target, reference, "genus")
    assert not features.lineage_matches(target, unrelated, "genus")
