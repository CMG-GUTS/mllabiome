from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mllabiome import explainability as ex
from mllabiome.transformations import TransformationCoordinate


class IdentityCoordinateTransform:
    def apply_pair(self, X_train, X_test):
        return np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)

    def coordinate_metadata(self, names):
        return [
            TransformationCoordinate(
                name=str(name),
                coordinate_type="feature_coordinate",
                anchor_feature=str(name),
                components=(str(name),),
                coefficients=(1.0,),
                exact_feature_identity=True,
            )
            for name in names
        ]

    def perturbation_geometry(self):
        return "unconstrained"


@pytest.mark.parametrize("protocol", ["lodo", "leave_one_dataset_out"])
def test_single_fold_refit_uses_lodo_training_feature_mask(protocol):
    X = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 1.0, 9.0],
            [2.0, 1.0, 8.0],
        ]
    )
    y = np.asarray([0, 1, 0, 1])
    split = {
        "split_key": "outer_1",
        "train_idx": np.asarray([0, 1]),
        "test_idx": np.asarray([2, 3]),
    }
    mask = np.asarray([True, True, False])
    _, fold = ex._fit_oof_single_fold_task(
        1,
        split,
        X,
        y,
        None,
        2,
        protocol,
        ["a", "b", "test_only"],
        mask,
        IdentityCoordinateTransform,
        lambda: LogisticRegression(random_state=7),
        1,
    )
    assert fold is not None
    np.testing.assert_array_equal(fold["feature_mask"], mask)
    assert fold["input_feature_names"] == ["a", "b"]
    assert fold["feature_names"] == ["a", "b"]
    assert fold["X_train"].shape[1] == 2
    assert fold["X_test"].shape[1] == 2


def test_coordinate_alignment_disambiguates_same_name_with_different_semantics():
    first = TransformationCoordinate(
        name="CLR[a]",
        coordinate_type="clr_logcontrast",
        anchor_feature="a",
        components=("a", "b"),
        coefficients=(0.5, -0.5),
        exact_feature_identity=False,
    )
    second = TransformationCoordinate(
        name="CLR[a]",
        coordinate_type="clr_logcontrast",
        anchor_feature="a",
        components=("a", "c"),
        coefficients=(0.5, -0.5),
        exact_feature_identity=False,
    )
    folds = [
        {
            "split_key": "outer_1",
            "test_idx": np.asarray([0]),
            "X_train": np.asarray([[0.1]]),
            "X_test": np.asarray([[0.2]]),
            "coordinate_metadata": [first],
        },
        {
            "split_key": "outer_2",
            "test_idx": np.asarray([1]),
            "X_train": np.asarray([[0.3]]),
            "X_test": np.asarray([[0.4]]),
            "coordinate_metadata": [second],
        },
    ]
    reference, names, metadata, rows = ex._align_oof_single_coordinates(folds, 2)
    assert len(names) == 2
    assert names[0] != names[1]
    assert all(name.startswith("CLR[a]__") for name in names)
    assert [item.name for item in metadata] == names
    assert len(rows) == 2
    assert np.isfinite(reference[0, 0])
    assert np.isnan(reference[0, 1])
    assert np.isnan(reference[1, 0])
    assert np.isfinite(reference[1, 1])


def test_coordinate_semantics_ignore_positive_scale_but_not_reference_identity():
    a = TransformationCoordinate(
        name="x",
        coordinate_type="standardized_clr_logcontrast",
        anchor_feature="a",
        components=("a", "b"),
        coefficients=(1.0, -1.0),
        exact_feature_identity=False,
    )
    b = TransformationCoordinate(
        name="x",
        coordinate_type="standardized_clr_logcontrast",
        anchor_feature="a",
        components=("a", "b"),
        coefficients=(2.0, -2.0),
        exact_feature_identity=False,
    )
    c = TransformationCoordinate(
        name="x",
        coordinate_type="alr_logcontrast",
        anchor_feature=None,
        components=("a", "c"),
        coefficients=(1.0, -1.0),
        exact_feature_identity=False,
    )
    assert ex._coordinate_semantic_key(a) == ex._coordinate_semantic_key(b)
    assert ex._coordinate_semantic_key(a) != ex._coordinate_semantic_key(c)


def test_prediction_reproduction_invariant(monkeypatch, tmp_path):
    stored = pd.DataFrame(
        {
            "outer_split_key": ["outer_1", "outer_1"],
            "sample_id": ["s1", "s2"],
            "config_id": ["cfg", "cfg"],
            "proba_control": [0.8, 0.2],
            "proba_case": [0.2, 0.8],
        }
    )
    monkeypatch.setattr(ex, "table_exists", lambda path: True)
    monkeypatch.setattr(ex, "read_table", lambda path: stored.copy())
    dataset = SimpleNamespace(
        sample_ids=np.asarray(["s1", "s2"]),
        class_labels=["control", "case"],
    )
    row = pd.Series({"config_id": "cfg"})
    folds = [
        {
            "split_key": "outer_1",
            "test_idx": np.asarray([0, 1]),
            "proba": np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        }
    ]
    records = ex._assert_oof_prediction_reproduction(tmp_path, row, dataset, folds)
    assert records[0]["verified"] == 1
    folds[0]["proba"] = np.asarray([[0.7, 0.3], [0.2, 0.8]])
    with pytest.raises(ex.ExplainabilityConfigurationError):
        ex._assert_oof_prediction_reproduction(tmp_path, row, dataset, folds)
