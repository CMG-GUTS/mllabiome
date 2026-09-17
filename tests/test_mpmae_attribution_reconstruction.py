from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mllabiome.configs_sweep import Explainability
from mllabiome.ensemble_aggregation import aggregate_member_predictions
from mllabiome.explainability_methods import SHAP
from mllabiome.transformations import TransformationCoordinate
from mllabiome.mpma_e_explainability import (
    _run_hierarchical_shap,
    _write_prediction_tables,
)


def _member(config_id: str, proba: np.ndarray, base: np.ndarray, values: np.ndarray):
    return {
        "config_id": config_id,
        "row": pd.Series({"resolution": "genus"}),
        "levels": ("genus",),
        "transformation_key": "arcsine_sqrt",
        "learner_key": "RF",
        "feature_names_fold": ["taxon_a", "taxon_b"],
        "coordinate_metadata_fold": [
            TransformationCoordinate(
                name="taxon_a",
                coordinate_type="feature_coordinate",
                anchor_feature="taxon_a",
                components=("taxon_a",),
                coefficients=(1.0,),
                exact_feature_identity=True,
            ),
            TransformationCoordinate(
                name="taxon_b",
                coordinate_type="feature_coordinate",
                anchor_feature="taxon_b",
                components=("taxon_b",),
                coefficients=(1.0,),
                exact_feature_identity=True,
            ),
        ],
        "proba": np.asarray(proba, dtype=float),
        "_test_base": np.asarray(base, dtype=float),
        "_test_values": np.asarray(values, dtype=float),
    }


def _shap_decomposition(proba: np.ndarray, base: np.ndarray, first_fraction: float):
    diff = np.asarray(proba, dtype=float) - np.asarray(base, dtype=float)
    values = np.empty((proba.shape[0], 2, proba.shape[1]), dtype=float)
    values[:, 0, :] = diff * float(first_fraction)
    values[:, 1, :] = diff * float(1.0 - first_fraction)
    return values


def _bundle(weights: np.ndarray, aggregation: str):
    p1 = np.asarray([[0.80, 0.20], [0.30, 0.70]], dtype=float)
    p2 = np.asarray([[0.40, 0.60], [0.65, 0.35]], dtype=float)
    b1 = np.asarray([[0.55, 0.45], [0.55, 0.45]], dtype=float)
    b2 = np.asarray([[0.45, 0.55], [0.45, 0.55]], dtype=float)
    v1 = _shap_decomposition(p1, b1, 0.25)
    v2 = _shap_decomposition(p2, b2, 0.60)
    members = [
        _member("m1", p1, b1, v1),
        _member("m2", p2, b2, v2),
    ]
    stack = np.stack([p1, p2], axis=0)
    weighted_input = weights.tolist() if aggregation == "weighted_mean_proba" else None
    ensemble = aggregate_member_predictions(stack, aggregation, weighted_input)
    dataset = SimpleNamespace(
        sample_ids=np.asarray(["s0", "s1", "s2", "s3"], dtype=object),
        y=np.asarray([0, 1, 0, 1], dtype=int),
        class_labels=("Placebo", "Active"),
    )
    return {
        "dataset": dataset,
        "linear_weights": np.asarray(weights, dtype=float),
        "folds": [
            {
                "split_key": "outer_0",
                "train_idx": np.asarray([2, 3], dtype=int),
                "test_idx": np.asarray([0, 1], dtype=int),
                "members": members,
                "member_stack": stack,
                "proba": ensemble,
            }
        ],
    }


def _sweep(tmp_path):
    return SimpleNamespace(
        explainability=Explainability(
            methods=(SHAP(background_size=50, max_explain=50),),
            classes="auto",
            random_state=42,
            representative_instances=False,
        )
    )


@pytest.mark.parametrize(
    ("aggregation", "weights"),
    [
        ("mean_proba", np.asarray([0.5, 0.5], dtype=float)),
        ("weighted_mean_proba", np.asarray([0.25, 0.75], dtype=float)),
    ],
)
def test_member_probabilities_reconstruct_linear_ensemble(
    tmp_path, aggregation, weights
):
    bundle = _bundle(weights, aggregation)
    mpma_e = {
        "aggregation_strategy": aggregation,
        "members": [
            {"config_id": "m1", "aggregation_weight": float(weights[0])},
            {"config_id": "m2", "aggregation_weight": float(weights[1])},
        ],
    }
    outputs = _write_prediction_tables(bundle, mpma_e, tmp_path)
    member = pd.read_csv(outputs["member_probability_decomposition"], sep="\t")
    ensemble = pd.read_csv(outputs["oof_predictions"], sep="\t")
    reconstructed = (
        member.groupby(["sample_id", "class_label"], as_index=False)[
            "weighted_probability_contribution"
        ]
        .sum()
        .pivot(
            index="sample_id",
            columns="class_label",
            values="weighted_probability_contribution",
        )
    )
    expected = ensemble.set_index("sample_id")[
        ["proba_Placebo", "proba_Active"]
    ].rename(columns={"proba_Placebo": "Placebo", "proba_Active": "Active"})
    reconstructed = reconstructed[expected.columns].reindex(expected.index)
    np.testing.assert_allclose(
        reconstructed.to_numpy(), expected.to_numpy(), atol=1e-12
    )


@pytest.mark.parametrize(
    ("aggregation", "weights"),
    [
        ("mean_proba", np.asarray([0.5, 0.5], dtype=float)),
        ("weighted_mean_proba", np.asarray([0.25, 0.75], dtype=float)),
    ],
)
def test_weighted_signed_member_shap_reconstructs_ensemble_probability(
    monkeypatch, tmp_path, aggregation, weights
):
    bundle = _bundle(weights, aggregation)
    mpma_e = {
        "aggregation_strategy": aggregation,
        "members": [
            {"config_id": "m1", "aggregation_weight": float(weights[0])},
            {"config_id": "m2", "aggregation_weight": float(weights[1])},
        ],
    }

    def fake_member_shap(
        member,
        class_labels,
        *,
        rows_ex,
        rows_bg,
        spec,
        random_state,
        progress_callback=None,
    ):
        rows = np.asarray(rows_ex, dtype=int)
        return member["_test_values"][rows], member["_test_base"][rows]

    monkeypatch.setattr(
        "mllabiome.mpma_e_explainability._shap_member_values",
        fake_member_shap,
    )
    outputs = _run_hierarchical_shap(_sweep(tmp_path), bundle, mpma_e, tmp_path)
    diagnostics = pd.read_csv(outputs["shap_additivity_diagnostics"], sep="\t")
    ensemble_rows = diagnostics[diagnostics["config_id"].eq("__MPMA_E__")].copy()
    assert len(ensemble_rows) == 2
    assert set(ensemble_rows["class_index"].astype(int)) == {1}
    np.testing.assert_allclose(
        ensemble_rows["ensemble_shap_additivity_residual"].to_numpy(dtype=float),
        0.0,
        atol=1e-12,
    )
    reconstructed = (
        ensemble_rows["ensemble_base_value"].to_numpy(dtype=float)
        + ensemble_rows["ensemble_probability"].to_numpy(dtype=float) * 0.0
    )
    member_raw = pd.read_csv(outputs["shap_member_attributions"], sep="\t")
    propagated = (
        member_raw.groupby(["sample_id", "class_index"], as_index=False)[
            "propagated_shap"
        ]
        .sum()
        .rename(columns={"propagated_shap": "sum_phi"})
    )
    bases = ensemble_rows[
        ["sample_id", "class_index", "ensemble_base_value", "ensemble_probability"]
    ]
    merged = bases.merge(propagated, on=["sample_id", "class_index"], how="left")
    reconstructed = merged["ensemble_base_value"].to_numpy(dtype=float) + merged[
        "sum_phi"
    ].to_numpy(dtype=float)
    np.testing.assert_allclose(
        reconstructed,
        merged["ensemble_probability"].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_taxon_net_shap_is_signed_weighted_sum_and_reports_cancellation(
    monkeypatch, tmp_path
):
    weights = np.asarray([0.5, 0.5], dtype=float)
    bundle = _bundle(weights, "mean_proba")
    mpma_e = {
        "aggregation_strategy": "mean_proba",
        "members": [
            {"config_id": "m1", "aggregation_weight": 0.5},
            {"config_id": "m2", "aggregation_weight": 0.5},
        ],
    }

    def fake_member_shap(
        member,
        class_labels,
        *,
        rows_ex,
        rows_bg,
        spec,
        random_state,
        progress_callback=None,
    ):
        rows = np.asarray(rows_ex, dtype=int)
        return member["_test_values"][rows], member["_test_base"][rows]

    monkeypatch.setattr(
        "mllabiome.mpma_e_explainability._shap_member_values",
        fake_member_shap,
    )
    outputs = _run_hierarchical_shap(_sweep(tmp_path), bundle, mpma_e, tmp_path)
    raw = pd.read_csv(outputs["shap_member_attributions"], sep="\t")
    taxon = pd.read_csv(outputs["shap_taxon_net_oof"], sep="\t")
    expected = (
        raw.groupby(["sample_id", "class_index", "biological_feature"], as_index=False)
        .agg(
            net=("propagated_shap", "sum"),
            gross=(
                "propagated_shap",
                lambda x: np.abs(np.asarray(x, dtype=float)).sum(),
            ),
        )
        .rename(columns={"biological_feature": "feature"})
    )
    merged = taxon.merge(
        expected, on=["sample_id", "class_index", "feature"], how="inner"
    )
    assert len(merged) == len(taxon)
    np.testing.assert_allclose(merged["net_propagated_shap"], merged["net"], atol=1e-12)
    np.testing.assert_allclose(
        merged["gross_propagated_shap"], merged["gross"], atol=1e-12
    )
    expected_cancellation = np.where(
        merged["gross"].to_numpy(dtype=float) > 0.0,
        1.0
        - np.abs(merged["net"].to_numpy(dtype=float))
        / merged["gross"].to_numpy(dtype=float),
        0.0,
    )
    np.testing.assert_allclose(
        merged["cancellation_fraction"].to_numpy(dtype=float),
        expected_cancellation,
        atol=1e-12,
    )
