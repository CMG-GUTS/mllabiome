from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

import mllabiome.configs_sweep as cs
import mllabiome.explainability as ex
from mllabiome.configs_sweep import Ensemble
from mllabiome.ensemble_sweep import select_mpma_e_by_outer_fold
from mllabiome.selection import select_mpma_b_by_outer_fold
from mllabiome.transformations import CountTransformation, _count_transformation_factory

TRAIN_IDX = np.arange(8, dtype=int)
TEST_IDX = np.arange(8, 12, dtype=int)
FEATURE_NAMES = ["g__alpha", "g__beta", "g__gamma"]


def _composition_matrices():
    base = np.asarray(
        [
            [8.0, 2.0, 1.0],
            [7.0, 3.0, 1.0],
            [6.0, 1.0, 3.0],
            [5.0, 4.0, 1.0],
            [9.0, 1.0, 2.0],
            [8.0, 2.0, 2.0],
            [7.0, 1.0, 4.0],
            [6.0, 3.0, 2.0],
            [4.0, 2.0, 1.0],
            [3.0, 4.0, 1.0],
            [5.0, 1.0, 2.0],
            [2.0, 5.0, 1.0],
        ],
        dtype=float,
    )
    heldout = np.zeros((len(base), 1), dtype=float)
    heldout[TEST_IDX, 0] = np.asarray([1.0e12, 2.0e12, 3.0e12, 4.0e12])
    return base, np.column_stack([base, heldout])


def _fit_lodo_transformation(X, names, transformation):
    X_train, X_test, mask = cs._lodo_feature_pair(X, TRAIN_IDX, TEST_IDX, "lodo")
    kept = [str(name) for name, use in zip(names, mask) if bool(use)]
    ct = CountTransformation(transformation, random_state=19)
    transformed_train, transformed_test = ct.apply_pair(X_train, X_test)
    metadata = ct.coordinate_metadata(kept)
    return (
        ct,
        transformed_train,
        transformed_test,
        np.asarray(mask, dtype=bool),
        metadata,
    )


def _coordinate_signature(metadata):
    return [
        (
            str(item.name),
            str(item.coordinate_type),
            None if item.anchor_feature is None else str(item.anchor_feature),
            tuple(str(x) for x in item.components),
            tuple(float(x) for x in item.coefficients),
            bool(item.exact_feature_identity),
        )
        for item in metadata
    ]


def test_heldout_only_feature_cannot_change_lodo_filter_or_prevalence_state():
    base, adversarial = _composition_matrices()
    base_ct, base_train, base_test, base_mask, _ = _fit_lodo_transformation(
        base,
        FEATURE_NAMES,
        "prevalence_weighted_relative_abundance",
    )
    adv_ct, adv_train, adv_test, adv_mask, _ = _fit_lodo_transformation(
        adversarial,
        [*FEATURE_NAMES, "g__heldout_only"],
        "prevalence_weighted_relative_abundance",
    )
    np.testing.assert_array_equal(base_mask, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(adv_mask, np.asarray([True, True, True, False]))
    np.testing.assert_allclose(base_ct._impl.prevalence_, adv_ct._impl.prevalence_)
    np.testing.assert_allclose(base_train, adv_train)
    np.testing.assert_allclose(base_test, adv_test)


def test_heldout_only_feature_cannot_change_relative_abundance_normalization():
    base, adversarial = _composition_matrices()
    _, base_train, base_test, _, _ = _fit_lodo_transformation(
        base,
        FEATURE_NAMES,
        "relative_abundance",
    )
    _, adv_train, adv_test, adv_mask, _ = _fit_lodo_transformation(
        adversarial,
        [*FEATURE_NAMES, "g__heldout_only"],
        "relative_abundance",
    )
    assert not bool(adv_mask[-1])
    np.testing.assert_allclose(base_train, adv_train)
    np.testing.assert_allclose(base_test, adv_test)
    np.testing.assert_allclose(base_train.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(base_test.sum(axis=1), 1.0, atol=1e-6)


def test_heldout_only_feature_cannot_change_alr_reference_taxon():
    base, adversarial = _composition_matrices()
    base_ct, base_train, base_test, _, base_metadata = _fit_lodo_transformation(
        base,
        FEATURE_NAMES,
        "additive_log_ratio_training_reference_multiplicative_replacement",
    )
    adv_ct, adv_train, adv_test, adv_mask, adv_metadata = _fit_lodo_transformation(
        adversarial,
        [*FEATURE_NAMES, "g__heldout_only"],
        "additive_log_ratio_training_reference_multiplicative_replacement",
    )
    assert not bool(adv_mask[-1])
    assert base_ct._impl.alr_reference_index_ == adv_ct._impl.alr_reference_index_
    np.testing.assert_allclose(base_train, adv_train)
    np.testing.assert_allclose(base_test, adv_test)
    assert _coordinate_signature(base_metadata) == _coordinate_signature(adv_metadata)
    assert all("g__heldout_only" not in item.components for item in adv_metadata)


@pytest.mark.parametrize(
    "transformation,state_attribute",
    [
        ("standardized_centered_log_ratio_multiplicative_replacement", "scaler_"),
        ("isometric_log_ratio_egozcue_multiplicative_replacement", "basis_"),
    ],
)
def test_heldout_only_feature_cannot_change_clr_or_ilr_training_state(
    transformation, state_attribute
):
    base, adversarial = _composition_matrices()
    base_ct, base_train, base_test, _, base_metadata = _fit_lodo_transformation(
        base,
        FEATURE_NAMES,
        transformation,
    )
    adv_ct, adv_train, adv_test, adv_mask, adv_metadata = _fit_lodo_transformation(
        adversarial,
        [*FEATURE_NAMES, "g__heldout_only"],
        transformation,
    )
    assert not bool(adv_mask[-1])
    base_state = getattr(base_ct._impl, state_attribute)
    adv_state = getattr(adv_ct._impl, state_attribute)
    if state_attribute == "scaler_":
        np.testing.assert_allclose(base_state.mean_, adv_state.mean_)
        np.testing.assert_allclose(base_state.scale_, adv_state.scale_)
    else:
        np.testing.assert_allclose(base_state, adv_state)
    np.testing.assert_allclose(base_train, adv_train)
    np.testing.assert_allclose(base_test, adv_test)
    assert _coordinate_signature(base_metadata) == _coordinate_signature(adv_metadata)
    assert all("g__heldout_only" not in item.components for item in adv_metadata)


class _SignalEstimator(BaseEstimator):
    def __init__(self, invert=False):
        self.invert = bool(invert)

    def fit(self, X, y):
        self.classes_ = np.asarray([0, 1], dtype=int)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        positive = X[:, 0] > 0.5
        if self.invert:
            positive = ~positive
        p1 = np.where(positive, 0.9, 0.1)
        return np.column_stack([1.0 - p1, p1])


def _selection_matrices():
    y = np.tile(np.asarray([0, 1, 0, 1], dtype=int), 3)
    signal = y.astype(float)
    shared = np.ones(len(y), dtype=float)
    nuisance = np.arange(1, len(y) + 1, dtype=float)
    clean = np.column_stack([signal, shared, nuisance])
    heldout = np.zeros(len(y), dtype=float)
    heldout[TEST_IDX] = y[TEST_IDX].astype(float) * 1.0e12
    adversarial = np.column_stack([clean, heldout])
    return y, clean, adversarial


def _run_candidate(X, y, config_id, invert):
    feature_blocks = (("all", tuple(range(X.shape[1]))),)
    inner_splits = (
        (np.asarray([4, 5, 6, 7]), np.asarray([0, 1, 2, 3])),
        (np.asarray([0, 1, 2, 3]), np.asarray([4, 5, 6, 7])),
    )
    return cs._evaluate_mpma_split_task(
        X_base=np.asarray(X, dtype=float),
        feature_blocks=feature_blocks,
        y=np.asarray(y, dtype=int),
        groups=None,
        classes=np.asarray([0, 1], dtype=int),
        class_labels=("control", "case"),
        sample_ids=tuple(f"s{i}" for i in range(len(y))),
        subject_ids=tuple(f"s{i}" for i in range(len(y))),
        train_idx=TRAIN_IDX,
        test_idx=TEST_IDX,
        inner_splits=inner_splits,
        split_key="lodo__heldout",
        protocol="lodo",
        res_name="genus",
        levels=("genus",),
        ct_name="identity",
        ct_item="identity",
        transformation_factory_builder=_count_transformation_factory,
        learner_name=config_id,
        learner_factory=lambda: _SignalEstimator(invert=invert),
        cid=config_id,
        gate_enabled=False,
        gate_metric="MCC",
        gate_threshold=None,
        selection_metric="MCC",
        existing_inner_keys=set(),
        existing_inner_scores=(),
        existing_qualification=None,
        needs_outer=True,
        random_state=23,
        threads_per_worker=1,
        resource_sample_interval_s=0.01,
    )


def _candidate_frames(X):
    y, _, _ = _selection_matrices()
    results = [
        _run_candidate(X, y, "hp_signal", False),
        _run_candidate(X, y, "hp_inverted", True),
    ]
    inner = pd.DataFrame([row for result in results for row in result["inner_metrics"]])
    inner_predictions = pd.DataFrame(
        [row for result in results for row in result["inner_predictions"]]
    )
    outer_predictions = pd.DataFrame(
        [row for result in results for row in result["outer_predictions"]]
    )
    configs = pd.DataFrame(
        [
            {
                "config_id": "hp_signal",
                "learner": "signal",
                "resolution": "genus",
                "count_transformation": "identity",
                "active": 1,
            },
            {
                "config_id": "hp_inverted",
                "learner": "inverted",
                "resolution": "genus",
                "count_transformation": "identity",
                "active": 1,
            },
        ]
    )
    return inner, inner_predictions, outer_predictions, configs


def test_heldout_only_feature_cannot_change_inner_hyperparameter_candidate_selection():
    _, clean, adversarial = _selection_matrices()
    clean_inner, clean_inner_predictions, clean_outer_predictions, configs = (
        _candidate_frames(clean)
    )
    adv_inner, adv_inner_predictions, adv_outer_predictions, _ = _candidate_frames(
        adversarial
    )
    pd.testing.assert_frame_equal(
        clean_inner.sort_values(["config_id", "inner_key"]).reset_index(drop=True),
        adv_inner.sort_values(["config_id", "inner_key"]).reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        clean_inner_predictions.sort_values(
            ["config_id", "split_key", "sample_id"]
        ).reset_index(drop=True),
        adv_inner_predictions.sort_values(
            ["config_id", "split_key", "sample_id"]
        ).reset_index(drop=True),
        check_dtype=False,
    )
    clean_selection = select_mpma_b_by_outer_fold(clean_inner, configs, "MCC")
    adv_selection = select_mpma_b_by_outer_fold(adv_inner, configs, "MCC")
    pd.testing.assert_frame_equal(clean_selection, adv_selection, check_dtype=False)
    assert clean_selection.iloc[0]["config_id"] == "hp_signal"
    pd.testing.assert_frame_equal(
        clean_outer_predictions.sort_values(["config_id", "sample_id"]).reset_index(
            drop=True
        ),
        adv_outer_predictions.sort_values(["config_id", "sample_id"]).reset_index(
            drop=True
        ),
        check_dtype=False,
    )


def test_heldout_only_feature_cannot_change_ensemble_selection_even_if_outer_predictions_are_adversarial():
    y, clean, _ = _selection_matrices()
    inner, inner_predictions, outer_predictions, configs = _candidate_frames(clean)
    plan = Ensemble(
        max_sizes=(2,),
        selection_strategies=("top_k",),
        aggregation_strategies=("mean_proba",),
        optimize_metric="MCC",
    )
    baseline, _, _ = select_mpma_e_by_outer_fold(
        inner,
        inner_predictions,
        outer_predictions,
        configs,
        plan,
        "MCC",
    )
    adversarial_outer = outer_predictions.copy()
    heldout_signal = y[TEST_IDX]
    for config_id in ("hp_signal", "hp_inverted"):
        mask = adversarial_outer["config_id"].astype(str).eq(config_id)
        order = adversarial_outer.loc[mask, "sample_index"].to_numpy(dtype=int)
        local = np.asarray(
            [heldout_signal[int(index - TEST_IDX[0])] for index in order]
        )
        if config_id == "hp_inverted":
            local = 1 - local
        p1 = np.where(local == 1, 0.999, 0.001)
        adversarial_outer.loc[mask, "proba_control"] = 1.0 - p1
        adversarial_outer.loc[mask, "proba_case"] = p1
        adversarial_outer.loc[mask, "y_proba_pos"] = p1
        adversarial_outer.loc[mask, "y_pred"] = (p1 >= 0.5).astype(int)
    changed, _, _ = select_mpma_e_by_outer_fold(
        inner,
        inner_predictions,
        adversarial_outer,
        configs,
        plan,
        "MCC",
    )
    pd.testing.assert_frame_equal(baseline, changed, check_dtype=False)


@pytest.mark.parametrize(
    "transformation",
    [
        "centered_log_ratio_multiplicative_replacement",
        "additive_log_ratio_training_reference_multiplicative_replacement",
        "isometric_log_ratio_egozcue_multiplicative_replacement",
    ],
)
def test_heldout_only_feature_cannot_enter_oof_xai_coordinates(transformation):
    base, adversarial = _composition_matrices()
    y = np.tile(np.asarray([0, 1], dtype=int), 6)
    split = {
        "split_key": "lodo__heldout",
        "train_idx": TRAIN_IDX,
        "test_idx": TEST_IDX,
    }
    _, _, base_mask = cs._lodo_feature_pair(base, TRAIN_IDX, TEST_IDX, "lodo")
    _, _, adv_mask = cs._lodo_feature_pair(adversarial, TRAIN_IDX, TEST_IDX, "lodo")
    _, base_fold = ex._fit_oof_single_fold_task(
        1,
        split,
        base,
        y,
        None,
        2,
        "lodo",
        FEATURE_NAMES,
        base_mask,
        lambda: CountTransformation(transformation, random_state=29),
        lambda: _SignalEstimator(),
        1,
    )
    _, adv_fold = ex._fit_oof_single_fold_task(
        1,
        split,
        adversarial,
        y,
        None,
        2,
        "lodo",
        [*FEATURE_NAMES, "g__heldout_only"],
        adv_mask,
        lambda: CountTransformation(transformation, random_state=29),
        lambda: _SignalEstimator(),
        1,
    )
    assert base_fold is not None
    assert adv_fold is not None
    assert not bool(adv_fold["feature_mask"][-1])
    assert adv_fold["input_feature_names"] == FEATURE_NAMES
    assert base_fold["feature_names"] == adv_fold["feature_names"]
    assert _coordinate_signature(
        base_fold["coordinate_metadata"]
    ) == _coordinate_signature(adv_fold["coordinate_metadata"])
    assert all(
        "g__heldout_only" not in item.components
        for item in adv_fold["coordinate_metadata"]
    )
    np.testing.assert_allclose(base_fold["X_train"], adv_fold["X_train"])
    np.testing.assert_allclose(base_fold["X_test"], adv_fold["X_test"])
    np.testing.assert_allclose(base_fold["proba"], adv_fold["proba"])
