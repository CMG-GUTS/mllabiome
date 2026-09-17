import numpy as np
import pytest
from scipy.stats import rankdata
from skbio.stats.composition import alr as skbio_alr
from skbio.stats.composition import closure as skbio_closure
from skbio.stats.composition import clr as skbio_clr
from skbio.stats.composition import ilr as skbio_ilr
from skbio.stats.composition import multi_replace as skbio_multi_replace
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from mllabiome.configs_sweep import build_sweep_configs
from mllabiome.data import Dataset
from mllabiome.resolutions import _parse_resolution, materialize_mpdr
from mllabiome.transformations import (
    TRANSFORMATION_LABELS,
    CountTransformation,
    CountTransformationAdapter,
    Transformation,
    transformation_space_table,
)

EXPECTED_TRANSFORMATIONS = {
    "identity",
    "relative_abundance",
    "presence_absence",
    "hellinger",
    "arcsine_sqrt",
    "log10_relative_abundance_half_min_pseudocount",
    "centered_log_ratio_multiplicative_replacement",
    "additive_log_ratio_first_reference_multiplicative_replacement",
    "isometric_log_ratio_egozcue_multiplicative_replacement",
    "standardized_centered_log_ratio_multiplicative_replacement",
    "yeo_johnson_relative_abundance",
    "quantile_normal_relative_abundance",
    "robust_scaled_relative_abundance",
    "within_sample_fractional_rank",
    "training_ecdf_rank",
    "prevalence_weighted_relative_abundance",
}

LOG_RATIO_COORDINATE_TRANSFORMATIONS = {
    "additive_log_ratio_first_reference_multiplicative_replacement",
    "isometric_log_ratio_egozcue_multiplicative_replacement",
}


def _relative(X):
    X = np.asarray(X, dtype=float)
    sums = X.sum(axis=1, keepdims=True)
    return np.divide(X, sums, out=np.zeros_like(X), where=sums > 0)


def _positive_composition(X):
    rel = _relative(X)
    return np.asarray(skbio_multi_replace(skbio_closure(rel)), dtype=float)


def _train_matrix():
    return np.array(
        [
            [10.0, 0.0, 30.0, 5.0],
            [5.0, 5.0, 0.0, 10.0],
            [1.0, 2.0, 3.0, 4.0],
            [8.0, 2.0, 1.0, 9.0],
            [4.0, 6.0, 2.0, 8.0],
            [9.0, 1.0, 5.0, 5.0],
        ],
        dtype=float,
    )


def _test_matrix():
    return np.array(
        [[2.0, 1.0, 7.0, 5.0], [1.0, 5.0, 4.0, 2.0], [3.0, 2.0, 1.0, 9.0]], dtype=float
    )


def _resolution_dataset():
    phylum = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    class_ = np.array([[3.0], [1.0], [2.0]], dtype=np.float32)
    order = np.array([[4.0, 5.0], [2.0, 3.0], [1.0, 4.0]], dtype=np.float32)
    family = np.array([[6.0], [4.0], [5.0]], dtype=np.float32)
    genus = np.array([[7.0, 8.0], [5.0, 6.0], [2.0, 7.0]], dtype=np.float32)
    raw_extra = np.array([[9.0], [7.0], [11.0]], dtype=np.float32)
    all_ = np.concatenate([phylum, class_, order, family, genus, raw_extra], axis=1)
    return Dataset(
        X_by_level={
            "phylum": phylum,
            "class": class_,
            "order": order,
            "family": family,
            "genus": genus,
            "all": all_,
        },
        feature_names_by_level={
            "phylum": ["p1", "p2"],
            "class": ["c1"],
            "order": ["o1", "o2"],
            "family": ["f1"],
            "genus": ["g1", "g2"],
            "all": ["p1", "p2", "c1", "o1", "o2", "f1", "g1", "g2", "raw_extra"],
        },
        y=np.array([0, 1, 0]),
        sample_ids=["s1", "s2", "s3"],
        metadata=None,
        class_labels=["control", "case"],
        positive_class=1,
    )


def test_every_builtin_transformation_has_an_explicit_correctness_contract():
    assert {label.key for label in TRANSFORMATION_LABELS} == EXPECTED_TRANSFORMATIONS


def test_canonical_aliases():
    assert Transformation("raw").name == "identity"
    assert Transformation("none").name == "relative_abundance"
    assert Transformation("binary").name == "presence_absence"
    assert Transformation("sqrt").name == "hellinger"
    assert Transformation("arcsin_sqrt").name == "arcsine_sqrt"
    assert Transformation("clr").name == "centered_log_ratio_multiplicative_replacement"
    assert (
        Transformation("alr").name
        == "additive_log_ratio_first_reference_multiplicative_replacement"
    )
    assert (
        Transformation("ilr").name
        == "isometric_log_ratio_egozcue_multiplicative_replacement"
    )


def test_unknown_transformation_requires_an_explicit_custom_callable():
    with pytest.raises(KeyError, match="Unknown abundance transformation"):
        Transformation("not_a_builtin_transformation")


def test_identity_is_exact_no_op():
    X = _train_matrix()
    out = CountTransformation("identity").fit_apply(X)
    np.testing.assert_allclose(out, X, rtol=0, atol=0)


def test_relative_abundance_is_exact_and_rows_sum_to_one():
    X = _train_matrix()
    out = CountTransformation("relative_abundance").fit_apply(X)
    expected = _relative(X)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-07)
    np.testing.assert_allclose(
        out.sum(axis=1), np.ones(X.shape[0]), rtol=1e-06, atol=1e-07
    )


def test_presence_absence_is_exact_binary_indicator():
    X = _train_matrix()
    out = CountTransformation("presence_absence").fit_apply(X)
    np.testing.assert_array_equal(out, (X > 0).astype(np.float32))


def test_hellinger_is_sqrt_relative_abundance_and_squared_rows_sum_to_one():
    X = _train_matrix()
    out = CountTransformation("hellinger").fit_apply(X)
    expected = np.sqrt(_relative(X))
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-07)
    np.testing.assert_allclose(
        np.square(out).sum(axis=1), np.ones(X.shape[0]), rtol=1e-06, atol=1e-07
    )


def test_arcsine_sqrt_is_exact_on_relative_abundance():
    X = _train_matrix()
    out = CountTransformation("arcsine_sqrt").fit_apply(X)
    expected = np.arcsin(np.sqrt(_relative(X)))
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-07)


def test_log10_relative_abundance_pseudocount_is_fitted_from_training_only():
    train = np.array([[1.0, 3.0, 0.0], [2.0, 2.0, 4.0], [4.0, 1.0, 5.0]], dtype=float)
    test = np.array([[1e-12, 1.0, 0.0], [5.0, 0.0, 5.0]], dtype=float)
    train_rel = _relative(train)
    expected_pseudocount = train_rel[train_rel > 0].min() / 2.0
    transform = CountTransformation("log10_relative_abundance_half_min_pseudocount")
    transform.fit(train)
    out = transform.apply(test)
    expected = np.log10(_relative(test) + expected_pseudocount)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-06)
    assert transform._impl is not None
    assert transform._impl.pseudocount_ == pytest.approx(expected_pseudocount)
    contaminated_batch = np.vstack(
        [test[0], [1e-30, 1.0, 0.0], [1000000000000.0, 1.0, 1.0]]
    )
    contaminated_out = transform.apply(contaminated_batch)
    np.testing.assert_allclose(contaminated_out[0], out[0], rtol=1e-06, atol=1e-06)


def test_pseudocount_argument_is_not_silently_ignored():
    with pytest.raises(ValueError, match="not configurable"):
        CountTransformation(
            "log10_relative_abundance_half_min_pseudocount", pseudo_count=1e-06
        )


def test_clr_matches_scikit_bio_and_has_zero_row_mean():
    X = _train_matrix()
    out = CountTransformation(
        "centered_log_ratio_multiplicative_replacement"
    ).fit_apply(X)
    expected = np.asarray(skbio_clr(_positive_composition(X)), dtype=float)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(
        out.mean(axis=1), np.zeros(X.shape[0]), rtol=0, atol=1e-06
    )


def test_clr_single_sample_preserves_2d_shape_and_matches_scikit_bio():
    X = _test_matrix()[0:1]
    out = CountTransformation(
        "centered_log_ratio_multiplicative_replacement"
    ).fit_apply(X)
    expected = np.asarray(skbio_clr(_positive_composition(X)), dtype=float)
    if expected.ndim == 1:
        expected = expected.reshape(1, -1)
    assert out.shape == X.shape
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-06)


def test_alr_matches_scikit_bio_first_reference_and_coordinate_metadata():
    X = _train_matrix()
    features = ["a", "b", "c", "d"]
    transform = CountTransformation("alr").fit(X)
    out = transform.apply(X)
    expected = np.asarray(skbio_alr(_positive_composition(X), ref_idx=0), dtype=float)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-06)
    assert out.shape == (X.shape[0], X.shape[1] - 1)
    assert transform.get_feature_names_out(features) == [
        "ALR[b/a]",
        "ALR[c/a]",
        "ALR[d/a]",
    ]
    metadata = transform.coordinate_metadata(features)
    assert [item.name for item in metadata] == ["ALR[b/a]", "ALR[c/a]", "ALR[d/a]"]
    assert all(item.coordinate_type == "alr_logcontrast" for item in metadata)
    assert all(item.anchor_feature is None for item in metadata)
    assert all(not item.exact_feature_identity for item in metadata)
    assert [item.components for item in metadata] == [("b", "a"), ("c", "a"), ("d", "a")]
    assert all(item.coefficients == (1.0, -1.0) for item in metadata)


def test_ilr_matches_scikit_bio_egozcue_basis_and_coordinate_metadata():
    X = _train_matrix()
    features = ["a", "b", "c", "d"]
    transform = CountTransformation("ilr").fit(X)
    assert transform._impl is not None
    basis = np.asarray(transform._impl.basis_, dtype=float)
    expected_basis = np.asarray(
        [
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0, 0.0],
            [1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 0.0],
            [
                1.0 / np.sqrt(12.0),
                1.0 / np.sqrt(12.0),
                1.0 / np.sqrt(12.0),
                -3.0 / np.sqrt(12.0),
            ],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(basis, expected_basis, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        basis @ basis.T,
        np.eye(X.shape[1] - 1),
        rtol=1e-06,
        atol=1e-06,
    )
    np.testing.assert_allclose(basis.sum(axis=1), 0.0, rtol=0, atol=1e-12)
    out = transform.apply(X)
    expected = np.asarray(skbio_ilr(_positive_composition(X), basis=basis), dtype=float)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-06)
    assert out.shape == (X.shape[0], X.shape[1] - 1)
    metadata = transform.coordinate_metadata(features)
    assert len(metadata) == X.shape[1] - 1
    assert [item.name for item in metadata] == transform.get_feature_names_out(features)
    assert all(item.coordinate_type == "ilr_balance" for item in metadata)
    assert all(item.anchor_feature is None for item in metadata)
    assert all(not item.exact_feature_identity for item in metadata)
    for index, item in enumerate(metadata):
        row = basis[index]
        mask = np.abs(row) > 1e-15
        assert item.components == tuple(np.asarray(features, dtype=object)[mask].tolist())
        np.testing.assert_allclose(
            np.asarray(item.coefficients, dtype=float),
            row[mask],
            rtol=0,
            atol=1e-12,
        )


def test_alr_and_ilr_single_sample_preserve_2d_coordinate_shape():
    X = _test_matrix()[0:1]
    for name in LOG_RATIO_COORDINATE_TRANSFORMATIONS:
        out = CountTransformation(name).fit_apply(X)
        assert out.shape == (1, X.shape[1] - 1)


def test_standardized_clr_single_test_sample_preserves_2d_shape():
    train = _train_matrix()
    X = _test_matrix()[0:1]
    transform = CountTransformation(
        "standardized_centered_log_ratio_multiplicative_replacement"
    ).fit(train)
    out = transform.apply(X)
    assert out.shape == X.shape


def test_log_ratio_transform_is_invariant_to_samplewise_positive_scaling():
    X = _train_matrix()
    scales = np.array([7.0, 0.25, 1000.0, 2.0, 13.0, 0.1])[:, None]
    scaled = X * scales
    original = CountTransformation(
        "centered_log_ratio_multiplicative_replacement"
    ).fit_apply(X)
    rescaled = CountTransformation(
        "centered_log_ratio_multiplicative_replacement"
    ).fit_apply(scaled)
    np.testing.assert_allclose(original, rescaled, rtol=1e-05, atol=1e-05)


def test_standardized_clr_matches_scikit_bio_then_training_standardization():
    train = _train_matrix()
    test = _test_matrix()
    train_base = np.asarray(skbio_clr(_positive_composition(train)), dtype=float)
    test_base = np.asarray(skbio_clr(_positive_composition(test)), dtype=float)
    scaler = StandardScaler().fit(train_base)
    expected_train = scaler.transform(train_base)
    expected_test = scaler.transform(test_base)
    out_train, out_test = CountTransformation(
        "standardized_centered_log_ratio_multiplicative_replacement"
    ).apply_pair(train, test)
    np.testing.assert_allclose(out_train, expected_train, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(out_test, expected_test, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(out_train.mean(axis=0), 0.0, rtol=0, atol=1e-06)


def test_yeo_johnson_relative_abundance_matches_training_fitted_sklearn_transform():
    train = _train_matrix()
    test = _test_matrix()
    train_rel = _relative(train)
    test_rel = _relative(test)
    mask = np.ptp(train_rel, axis=0) > 0
    expected_train = np.zeros_like(train_rel)
    expected_test = np.zeros_like(test_rel)
    scaler = PowerTransformer(method="yeo-johnson", standardize=True).fit(
        train_rel[:, mask]
    )
    expected_train[:, mask] = scaler.transform(train_rel[:, mask])
    expected_test[:, mask] = scaler.transform(test_rel[:, mask])
    out_train, out_test = CountTransformation(
        "yeo_johnson_relative_abundance"
    ).apply_pair(train, test)
    np.testing.assert_allclose(out_train, expected_train, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(out_test, expected_test, rtol=1e-05, atol=1e-05)


def test_quantile_normal_relative_abundance_matches_training_fitted_sklearn_transform():
    train = _train_matrix()
    test = _test_matrix()
    train_rel = _relative(train)
    test_rel = _relative(test)
    scaler = QuantileTransformer(
        n_quantiles=max(2, min(1000, train.shape[0])),
        output_distribution="normal",
        random_state=42,
        subsample=None,
    ).fit(train_rel)
    expected_train = scaler.transform(train_rel)
    expected_test = scaler.transform(test_rel)
    out_train, out_test = CountTransformation(
        "quantile_normal_relative_abundance", random_state=42
    ).apply_pair(train, test)
    np.testing.assert_allclose(out_train, expected_train, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(out_test, expected_test, rtol=1e-05, atol=1e-05)


def test_robust_scaled_relative_abundance_matches_training_fitted_sklearn_transform():
    train = _train_matrix()
    test = _test_matrix()
    train_rel = _relative(train)
    test_rel = _relative(test)
    scaler = RobustScaler().fit(train_rel)
    expected_train = scaler.transform(train_rel)
    expected_test = scaler.transform(test_rel)
    out_train, out_test = CountTransformation(
        "robust_scaled_relative_abundance"
    ).apply_pair(train, test)
    np.testing.assert_allclose(out_train, expected_train, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(out_test, expected_test, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(np.median(out_train, axis=0), 0.0, rtol=0, atol=1e-06)


def test_within_sample_fractional_rank_matches_rankdata_definition():
    X = np.array([[4.0, 1.0, 3.0, 2.0], [1.0, 1.0, 4.0, 2.0]], dtype=float)
    out = CountTransformation("within_sample_fractional_rank").fit_apply(X)
    expected = np.apply_along_axis(rankdata, 1, X) / (X.shape[1] + 1.0)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-07)


def test_training_ecdf_rank_uses_only_training_distribution():
    train = np.array([[1.0, 10.0], [2.0, 40.0], [3.0, 20.0], [4.0, 30.0]], dtype=float)
    test = np.array([[2.5, 25.0], [0.0, 100.0], [4.0, 10.0]], dtype=float)
    transform = CountTransformation("training_ecdf_rank")
    transform.fit(train)
    out = transform.apply(test)
    expected = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.25]], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-07)
    valid_contaminated_batch = np.vstack(
        [test[0], [0.0, 1000000000.0], [1000000000.0, 0.0]]
    )
    contaminated_out = transform.apply(valid_contaminated_batch)
    np.testing.assert_allclose(contaminated_out[0], out[0], rtol=0, atol=1e-07)


def test_prevalence_weighted_relative_abundance_uses_training_prevalence_only():
    train = np.array(
        [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    test = np.array([[1.0, 1.0, 1.0], [0.0, 4.0, 1.0]], dtype=float)
    prevalence = np.array([0.75, 0.5, 0.5])
    rel = _relative(test)
    weighted = rel * prevalence[None, :]
    expected = weighted / weighted.sum(axis=1, keepdims=True)
    transform = CountTransformation("prevalence_weighted_relative_abundance")
    transform.fit(train)
    out = transform.apply(test)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-07)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-06, atol=1e-07)


@pytest.mark.parametrize("name", sorted(EXPECTED_TRANSFORMATIONS))
def test_every_builtin_preserves_samples_and_has_declared_feature_dimension(name):
    train = _train_matrix()
    test = _test_matrix()
    out_train, out_test = CountTransformation(name).apply_pair(train, test)
    expected_features = (
        train.shape[1] - 1
        if name in LOG_RATIO_COORDINATE_TRANSFORMATIONS
        else train.shape[1]
    )
    assert out_train.shape == (train.shape[0], expected_features)
    assert out_test.shape == (test.shape[0], expected_features)


@pytest.mark.parametrize("name", sorted(EXPECTED_TRANSFORMATIONS))
def test_test_sample_transform_is_invariant_to_other_test_samples(name):
    train = _train_matrix()
    first = _test_matrix()[0:1]
    batch = np.vstack([first, [1000.0, 1.0, 1.0, 1.0], [1.0, 1000.0, 2.0, 3.0]])
    transform = CountTransformation(name).fit(train)
    alone = transform.apply(first)[0]
    together = transform.apply(batch)[0]
    np.testing.assert_allclose(alone, together, rtol=1e-06, atol=1e-06)


def test_fitted_transform_rejects_different_feature_count():
    transform = CountTransformation("relative_abundance").fit(_train_matrix())
    with pytest.raises(ValueError, match="Feature count differs"):
        transform.apply(np.ones((2, _train_matrix().shape[1] + 1)))


def test_custom_transform_must_preserve_dimensions():
    transform = Transformation("bad_custom", lambda X: np.asarray(X)[:, :-1])
    with pytest.raises(ValueError, match="unexpected matrix shape"):
        transform.apply(_train_matrix(), _test_matrix())


def test_two_array_custom_transform_is_rejected_to_protect_train_test_boundary():
    with pytest.raises(
        ValueError, match="Two-array custom transformations are disabled"
    ):
        Transformation("unsafe", lambda X_train, X_test: (X_train, X_test), True)


def test_adapter_rejects_two_array_callable():

    def unsafe(X_train, X_test):
        return (X_train, X_test)

    adapter = CountTransformationAdapter("unsafe_custom", unsafe)
    with pytest.raises(TypeError, match="Two-array custom transformation"):
        adapter.apply_pair(_train_matrix(), _test_matrix())


@pytest.mark.parametrize(
    "resolution,expected_levels",
    [
        ("genus", ("genus",)),
        ("family+genus", ("family", "genus")),
        ("phylum-order", ("phylum", "class", "order")),
        ("phylum+genus", ("phylum", "genus")),
        ("raw", ("all",)),
    ],
)
def test_mpdr_always_selects_and_concatenates_resolution_before_transforming(
    resolution, expected_levels
):
    dataset = _resolution_dataset()
    _, levels = _parse_resolution(resolution)
    assert levels == expected_levels
    X, names = materialize_mpdr(dataset, levels)
    if expected_levels == ("all",):
        expected_X = dataset.X_by_level["all"]
        expected_names = dataset.feature_names_by_level["all"]
    else:
        expected_X = np.concatenate(
            [dataset.X_by_level[level] for level in expected_levels], axis=1
        )
        expected_names = [
            name
            for level in expected_levels
            for name in dataset.feature_names_by_level[level]
        ]
    np.testing.assert_allclose(X, expected_X, rtol=0, atol=0)
    assert names == expected_names
    transformed = CountTransformation("relative_abundance").fit_apply(X)
    expected_transformed = _relative(expected_X)
    np.testing.assert_allclose(
        transformed, expected_transformed, rtol=1e-06, atol=1e-07
    )
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0, rtol=1e-06, atol=1e-07)


def test_multi_rank_relative_abundance_is_global_across_the_selected_matrix():
    dataset = _resolution_dataset()
    _, levels = _parse_resolution("family+genus")
    X, _ = materialize_mpdr(dataset, levels)
    out = CountTransformation("relative_abundance").fit_apply(X)
    expected_X = np.concatenate(
        [dataset.X_by_level["family"], dataset.X_by_level["genus"]], axis=1
    )
    expected = expected_X / expected_X.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-07)
    family_width = dataset.X_by_level["family"].shape[1]
    assert not np.allclose(out[:, :family_width].sum(axis=1), 1.0)
    assert not np.allclose(out[:, family_width:].sum(axis=1), 1.0)


def test_raw_relative_abundance_uses_exact_complete_original_feature_matrix():
    dataset = _resolution_dataset()
    _, levels = _parse_resolution("raw")
    X, names = materialize_mpdr(dataset, levels)
    np.testing.assert_allclose(X, dataset.X_by_level["all"], rtol=0, atol=0)
    assert names == dataset.feature_names_by_level["all"]
    assert "raw_extra" in names
    out = CountTransformation("relative_abundance").fit_apply(X)
    expected = _relative(dataset.X_by_level["all"])
    np.testing.assert_allclose(out, expected, rtol=1e-06, atol=1e-07)


def test_compositional_log_transforms_reject_all_zero_samples():
    X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=float)
    for name in (
        "centered_log_ratio_multiplicative_replacement",
        "additive_log_ratio_first_reference_multiplicative_replacement",
        "isometric_log_ratio_egozcue_multiplicative_replacement",
        "standardized_centered_log_ratio_multiplicative_replacement",
        "log10_relative_abundance_half_min_pseudocount",
    ):
        with pytest.raises(ValueError):
            CountTransformation(name).fit_apply(X)


def test_all_abundance_based_transforms_reject_negative_values():
    X = _train_matrix().copy()
    X[0, 0] = -1.0
    for name in EXPECTED_TRANSFORMATIONS - {"identity"}:
        with pytest.raises(ValueError):
            CountTransformation(name).fit_apply(X)


def test_single_canonical_name_in_outputs():
    table = transformation_space_table()
    assert list(table.columns) == ["count_transformation", "category"]
    configs = build_sweep_configs(
        [("genus", ("genus",))], [Transformation("none")], ["RF_1000_msl5"]
    )
    assert configs.loc[0, "count_transformation"] == "relative_abundance"
    assert "transformation_abbreviation" not in configs.columns
