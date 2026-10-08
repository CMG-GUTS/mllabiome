from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mllabiome.data import (
    Data,
    Dataset,
    _dataset_from_feature_matrix,
    _taxonomic_rank,
    load_dataset,
)
from mllabiome.resolutions import (
    _manual_range,
    _parse_resolution,
    _resolution_from_name,
    materialize_mpdr,
)
from mllabiome.utils import TAXONOMIC_LEVELS

ROOT = Path(__file__).resolve().parents[1]
PTSD_PROFILE = ROOT / "examples" / "data" / "PTSD" / "PTSD_profiles.tsv"
PTSD_METADATA = ROOT / "examples" / "data" / "PTSD" / "PTSD_metadata.tsv"
AGP_PROFILE = ROOT / "examples" / "data" / "IBS" / "AGP-2021_profiles.tsv"
AGP_METADATA = ROOT / "examples" / "data" / "IBS" / "AGP-2021_metadata.tsv"

EXPECTED_LEVELS = (
    "domain",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "strain",
)


def _require_example_data():
    missing = [
        path
        for path in (PTSD_PROFILE, PTSD_METADATA, AGP_PROFILE, AGP_METADATA)
        if not path.exists()
    ]
    assert not missing, f"Missing repository example data: {missing}"


@pytest.fixture(scope="module")
def ptsd_dataset():
    _require_example_data()
    return load_dataset(
        Data(
            abundance_path=PTSD_PROFILE,
            metadata_path=PTSD_METADATA,
            format="metaphlan_tsv",
            sample_id_col="sampleId",
            target_col="group",
            class_labels=("Placebo", "Active"),
        ),
        EXPECTED_LEVELS,
    )


@pytest.fixture(scope="module")
def agp_dataset():
    _require_example_data()
    return load_dataset(
        Data(
            abundance_path=AGP_PROFILE,
            metadata_path=AGP_METADATA,
            format="metaphlan_tsv",
            sample_id_col="Name",
            target_col="host_disease",
        ),
        EXPECTED_LEVELS,
    )


def _direct_ptsd_matrix():
    bio = pd.read_csv(PTSD_PROFILE, sep="\t", index_col=0, low_memory=False)
    meta = pd.read_csv(PTSD_METADATA, sep="\t", dtype=str)
    meta["sampleId"] = meta["sampleId"].astype(str).str.strip()
    bio.columns = bio.columns.astype(str).str.strip()
    ids = [sid for sid in meta["sampleId"].tolist() if sid in set(bio.columns)]
    return (
        bio[ids].T.to_numpy(dtype=np.float32),
        bio.index.astype(str).str.strip().tolist(),
        ids,
    )


def _direct_agp_matrix():
    meta = pd.read_csv(AGP_METADATA, sep="\t", dtype=str)
    ids = meta["Name"].astype(str).str.strip().tolist()
    first = AGP_PROFILE.read_text(encoding="utf-8").splitlines()[0].split("\t")
    if first[0].lstrip("#").strip().lower() == "clade_name":
        bio = pd.read_csv(AGP_PROFILE, sep="\t", index_col=0, low_memory=False)
        bio.columns = bio.columns.astype(str).str.strip()
        selected = [sid for sid in ids if sid in set(bio.columns)]
        bio.index = bio.index.astype(str).str.strip()
        return bio[selected].T.to_numpy(dtype=np.float32), bio.index.tolist(), selected
    bio = pd.read_csv(AGP_PROFILE, sep="\t", header=None, index_col=0, low_memory=False)
    if bio.shape[1] != len(ids):
        raise ValueError(
            "Headerless AGP profile width does not match metadata row count."
        )
    bio.index = bio.index.astype(str).str.strip()
    return bio.to_numpy(dtype=np.float32).T, bio.index.tolist(), ids


def _assert_exact_rank_partition(dataset):
    X_all = dataset.X_by_level["all"]
    names_all = dataset.feature_names_by_level["all"]
    for level in EXPECTED_LEVELS:
        idx = [i for i, name in enumerate(names_all) if _taxonomic_rank(name) == level]
        if idx:
            assert level in dataset.X_by_level
            assert dataset.feature_names_by_level[level] == [names_all[i] for i in idx]
            np.testing.assert_array_equal(dataset.X_by_level[level], X_all[:, idx])
        else:
            assert level not in dataset.X_by_level
            assert level not in dataset.feature_names_by_level


def test_taxonomic_level_order_is_stable():
    assert tuple(TAXONOMIC_LEVELS) == EXPECTED_LEVELS


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("d__Bacteria", "domain"),
        ("k__Bacteria", "domain"),
        ("k__Bacteria|p__Firmicutes", "phylum"),
        ("k__Bacteria|p__Firmicutes|c__Clostridia", "class"),
        ("d__Bacteria___p__Firmicutes", "phylum"),
        ("d__Bacteria___p__Firmicutes___c__Clostridia", "class"),
        ("d__Bacteria___p__Firmicutes___c__Clostridia___o__Oscillospirales", "order"),
        (
            "d__Bacteria___p__Firmicutes___c__Clostridia___o__Oscillospirales___f__Ruminococcaceae",
            "family",
        ),
        (
            "d__Bacteria___p__Firmicutes___c__Clostridia___o__Oscillospirales___f__Ruminococcaceae___g__Faecalibacterium",
            "genus",
        ),
        (
            "d__Bacteria___p__Firmicutes___c__Clostridia___o__Oscillospirales___f__Ruminococcaceae___g__Faecalibacterium___s__Faecalibacterium_prausnitzii",
            "species",
        ),
        (
            "d__Bacteria___p__Firmicutes___c__Clostridia___o__Oscillospirales___f__Ruminococcaceae___g__Faecalibacterium___s__Faecalibacterium_prausnitzii___t__SGB15342",
            "strain",
        ),
        (
            "k__Bacteria|p__Firmicutes|c__Clostridia|o__Oscillospirales|f__Ruminococcaceae|g__Faecalibacterium",
            "genus",
        ),
        ("unclassified_feature", None),
    ],
)
def test_taxonomic_rank_detects_deepest_lineage(feature, expected):
    assert _taxonomic_rank(feature) == expected


@pytest.mark.parametrize(
    ("name", "expected_name", "expected_levels"),
    [
        ("genus", "genus", ("genus",)),
        ("domain", "domain", ("domain",)),
        ("kingdom", "domain", ("domain",)),
        (
            "kingdom-genus",
            "domain→genus",
            ("domain", "phylum", "class", "order", "family", "genus"),
        ),
        ("kingdom+genus", "domain+genus", ("domain", "genus")),
        ("kingdom,genus", "domain+genus", ("domain", "genus")),
        ("phylum-family", "phylum→family", ("phylum", "class", "order", "family")),
        ("family-phylum", "family→phylum", ("phylum", "class", "order", "family")),
        ("phylum+genus", "phylum+genus", ("phylum", "genus")),
        ("genus+phylum", "genus+phylum", ("genus", "phylum")),
        ("phylum,genus", "phylum+genus", ("phylum", "genus")),
        ("raw", "raw", ("all",)),
        ("all", "raw", ("all",)),
        ("features", "raw", ("all",)),
        ("asis", "raw", ("all",)),
    ],
)
def test_resolution_name_parsing(name, expected_name, expected_levels):
    assert _resolution_from_name(name) == (expected_name, expected_levels)


def test_ranges_use_canonical_taxonomic_order():
    expected = ("phylum", "class", "order", "family", "genus")
    assert _manual_range("phylum", "genus") == expected
    assert _manual_range("genus", "phylum") == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "superkingdom",
        "phylum-superkingdom",
        "phylum+superkingdom",
        "phylum,superkingdom",
    ],
)
def test_invalid_named_resolutions_are_rejected(value):
    with pytest.raises(ValueError):
        _resolution_from_name(value)


@pytest.mark.parametrize(
    "value",
    [
        ("bad", ("genus", "superkingdom")),
        SimpleNamespace(name="bad", levels=("genus", "superkingdom")),
        ("dup", ("genus", "genus")),
        SimpleNamespace(name="dup", levels=("genus", "genus")),
        "genus+genus",
        "genus,genus",
        ("mixed", ("all", "genus")),
        ("mixed", ("raw", "genus")),
    ],
)
def test_invalid_structured_resolutions_are_rejected(value):
    with pytest.raises(ValueError):
        _parse_resolution(value)


def test_domain_and_kingdom_resolution_inputs_materialize_the_same_features():
    dataset = Dataset(
        X_by_level={
            "domain": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "all": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        },
        feature_names_by_level={
            "domain": ["d__Bacteria", "k__Archaea"],
            "all": ["d__Bacteria", "k__Archaea"],
        },
        y=np.array([0, 1]),
        sample_ids=["s1", "s2"],
        subject_ids=["s1", "s2"],
        metadata=pd.DataFrame({"sample_id": ["s1", "s2"]}),
        class_labels=["control", "case"],
        positive_class=1,
    )
    X_domain, names_domain = materialize_mpdr(dataset, ("domain",))
    X_kingdom, names_kingdom = materialize_mpdr(dataset, ("kingdom",))
    np.testing.assert_array_equal(X_domain, X_kingdom)
    assert names_domain == names_kingdom


def test_structured_kingdom_aliases_canonicalize_to_domain():
    assert _parse_resolution(("kingdom", ("kingdom",))) == ("domain", ("domain",))
    assert _parse_resolution(
        ("kingdom-genus", ("kingdom", "phylum", "class", "order", "family", "genus"))
    ) == (
        "domain→genus",
        ("domain", "phylum", "class", "order", "family", "genus"),
    )
    assert _parse_resolution(
        SimpleNamespace(name="kingdom+genus", levels=("kingdom", "genus"))
    ) == (
        "domain+genus",
        ("domain", "genus"),
    )


def test_domain_and_kingdom_aliases_cannot_create_duplicate_top_rank():
    with pytest.raises(ValueError, match="Duplicate"):
        _parse_resolution(("domain+kingdom", ("domain", "kingdom")))


def test_ptsd_real_profile_is_headered():
    _require_example_data()
    first = PTSD_PROFILE.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert first[0] == "clade_name"
    meta = pd.read_csv(PTSD_METADATA, sep="\t", dtype=str)
    assert set(first[1:]) == set(meta["sampleId"].astype(str))


def test_agp_real_profile_header_contract_matches_checked_in_file():
    _require_example_data()
    with AGP_PROFILE.open("r", encoding="utf-8") as handle:
        first = handle.readline().rstrip("\r\n").split("\t")
    meta = pd.read_csv(AGP_METADATA, sep="\t", dtype=str)
    if first[0].lstrip("#").strip().lower() == "clade_name":
        assert first[1:]
        assert set(first[1:]).issubset(set(meta["Name"].astype(str)))
    else:
        assert _taxonomic_rank(first[0]) is not None
        assert len(first) - 1 == len(meta)


def test_ptsd_real_parser_preserves_raw_matrix_exactly(ptsd_dataset):
    expected_X, expected_names, expected_ids = _direct_ptsd_matrix()
    assert ptsd_dataset.sample_ids == expected_ids
    assert ptsd_dataset.feature_names_by_level["all"] == expected_names
    np.testing.assert_array_equal(ptsd_dataset.X_by_level["all"], expected_X)


def test_agp_real_parser_preserves_raw_matrix_exactly(agp_dataset):
    expected_X, expected_names, expected_ids = _direct_agp_matrix()
    assert agp_dataset.sample_ids == expected_ids
    assert agp_dataset.feature_names_by_level["all"] == expected_names
    assert agp_dataset.feature_names_by_level["all"][0] == expected_names[0]
    np.testing.assert_array_equal(agp_dataset.X_by_level["all"], expected_X)


def test_ptsd_real_rank_partition_is_selection_only_without_aggregation(ptsd_dataset):
    _assert_exact_rank_partition(ptsd_dataset)


def test_agp_real_rank_partition_is_selection_only_without_aggregation(agp_dataset):
    _assert_exact_rank_partition(agp_dataset)


@pytest.mark.parametrize("level", EXPECTED_LEVELS)
def test_real_parser_rank_values_are_exact_subsets_of_raw(
    level, ptsd_dataset, agp_dataset
):
    for dataset in (ptsd_dataset, agp_dataset):
        names = dataset.feature_names_by_level["all"]
        idx = [i for i, name in enumerate(names) if _taxonomic_rank(name) == level]
        if not idx:
            continue
        np.testing.assert_array_equal(
            dataset.X_by_level[level], dataset.X_by_level["all"][:, idx]
        )


def test_real_raw_resolution_returns_complete_original_ptsd_matrix(ptsd_dataset):
    X, names = materialize_mpdr(ptsd_dataset, ("raw",))
    np.testing.assert_array_equal(X, ptsd_dataset.X_by_level["all"])
    assert names == ptsd_dataset.feature_names_by_level["all"]


def test_real_raw_resolution_returns_complete_original_agp_matrix(agp_dataset):
    X, names = materialize_mpdr(agp_dataset, ("raw",))
    np.testing.assert_array_equal(X, agp_dataset.X_by_level["all"])
    assert names == agp_dataset.feature_names_by_level["all"]


def test_real_multirank_resolution_concatenates_existing_rows_without_aggregation(
    ptsd_dataset,
):
    available = [
        level
        for level in ("order", "family", "genus")
        if level in ptsd_dataset.X_by_level
    ]
    assert len(available) >= 2
    X, names = materialize_mpdr(ptsd_dataset, available)
    expected_X = np.concatenate(
        [ptsd_dataset.X_by_level[level] for level in available], axis=1
    )
    expected_names = [
        name
        for level in available
        for name in ptsd_dataset.feature_names_by_level[level]
    ]
    np.testing.assert_array_equal(X, expected_X)
    assert names == expected_names


def test_already_relative_headered_matrix_is_preserved_exactly(tmp_path):
    profile = tmp_path / "relative.tsv"
    metadata = tmp_path / "metadata.tsv"
    profile.write_text(
        "clade_name\ts1\ts2\n"
        "d__Bacteria___p__Firmicutes___g__A\t0.25\t0.50\n"
        "d__Bacteria___p__Bacteroidota___g__B\t0.75\t0.50\n",
        encoding="utf-8",
    )
    metadata.write_text("sample_id\tlabel\ns1\tcontrol\ns2\tcase\n", encoding="utf-8")
    dataset = load_dataset(
        Data(
            abundance_path=profile,
            metadata_path=metadata,
            format="metaphlan_tsv",
            sample_id_col="sample_id",
            target_col="label",
        ),
        ("genus",),
    )
    expected = np.array([[0.25, 0.75], [0.50, 0.50]], dtype=np.float32)
    np.testing.assert_array_equal(dataset.X_by_level["all"], expected)
    np.testing.assert_array_equal(dataset.X_by_level["genus"], expected)
    np.testing.assert_array_equal(
        dataset.X_by_level["all"].sum(axis=1), np.ones(2, dtype=np.float32)
    )


def test_headerless_matrix_uses_metadata_order_only_when_width_matches_exactly(
    tmp_path,
):
    profile = tmp_path / "headerless.tsv"
    metadata = tmp_path / "metadata.tsv"
    profile.write_text(
        "d__Bacteria___p__Firmicutes\t10\t20\nd__Bacteria___p__Bacteroidota\t30\t40\n",
        encoding="utf-8",
    )
    metadata.write_text("sample_id\tlabel\nb\tcase\na\tcontrol\n", encoding="utf-8")
    dataset = load_dataset(
        Data(
            abundance_path=profile,
            metadata_path=metadata,
            format="metaphlan_tsv",
            sample_id_col="sample_id",
            target_col="label",
        ),
        ("phylum",),
    )
    assert dataset.sample_ids == ["b", "a"]
    np.testing.assert_array_equal(
        dataset.X_by_level["all"],
        np.array([[10.0, 30.0], [20.0, 40.0]], dtype=np.float32),
    )


def test_headerless_matrix_with_metadata_width_mismatch_is_rejected(tmp_path):
    profile = tmp_path / "headerless.tsv"
    metadata = tmp_path / "metadata.tsv"
    profile.write_text(
        "d__Bacteria___p__Firmicutes\t10\t20\nd__Bacteria___p__Bacteroidota\t30\t40\n",
        encoding="utf-8",
    )
    metadata.write_text(
        "sample_id\tlabel\na\tcontrol\nb\tcase\nc\tcontrol\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="headered or headerless"):
        load_dataset(
            Data(
                abundance_path=profile,
                metadata_path=metadata,
                format="metaphlan_tsv",
                sample_id_col="sample_id",
                target_col="label",
            ),
            ("phylum",),
        )


def test_species_only_input_does_not_silently_create_genus_abundances():
    names = [
        "d__Bacteria___p__Firmicutes___g__A___s__A1",
        "d__Bacteria___p__Firmicutes___g__A___s__A2",
        "d__Bacteria___p__Firmicutes___g__B___s__B1",
    ]
    X = np.array([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]], dtype=np.float32)
    dataset = _dataset_from_feature_matrix(
        X,
        names,
        np.array([0, 1]),
        ["s1", "s2"],
        ["s1", "s2"],
        pd.DataFrame({"sample_id": ["s1", "s2"]}),
        ["control", "case"],
        positive_class=1,
        levels_needed=("species", "genus"),
    )
    assert "species" in dataset.X_by_level
    assert "genus" not in dataset.X_by_level
    np.testing.assert_array_equal(dataset.X_by_level["species"], X)
    with pytest.raises(ValueError, match="genus"):
        materialize_mpdr(dataset, ("genus",))


def test_unranked_features_are_available_only_as_raw():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dataset = _dataset_from_feature_matrix(
        X,
        ["feature_a", "feature_b"],
        np.array([0, 1]),
        ["s1", "s2"],
        ["s1", "s2"],
        pd.DataFrame({"sample_id": ["s1", "s2"]}),
        ["control", "case"],
        positive_class=1,
        levels_needed=("genus",),
    )
    assert set(dataset.X_by_level) == {"all"}
    np.testing.assert_array_equal(dataset.X_by_level["all"], X)
    with pytest.raises(ValueError, match="genus"):
        materialize_mpdr(dataset, ("genus",))


def test_partially_missing_multirank_resolution_is_rejected():
    dataset = Dataset(
        X_by_level={
            "family": np.array([[1.0], [2.0]], dtype=np.float32),
            "all": np.array([[1.0], [2.0]], dtype=np.float32),
        },
        feature_names_by_level={"family": ["f1"], "all": ["f1"]},
        y=np.array([0, 1]),
        sample_ids=["s1", "s2"],
        subject_ids=["s1", "s2"],
        metadata=pd.DataFrame({"sample_id": ["s1", "s2"]}),
        class_labels=["control", "case"],
        positive_class=1,
    )
    with pytest.raises(ValueError, match="genus"):
        materialize_mpdr(dataset, ("family", "genus"))
