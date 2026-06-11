import numpy as np
import pandas as pd

from mllabiome.configs_sweep import Evaluation, _outer_splits, _strata_from_metadata
from mllabiome.data import Data, load_dataset


def test_repeated_nested_cv_is_stratified_by_target_by_default():
    y = np.array([0] * 12 + [1] * 12)
    plan = Evaluation(protocol="repeated_nested_cv", outer_folds=4, repeats=1, random_state=7)
    splits = _outer_splits(plan, y, groups=None)
    assert len(splits) == 4
    for split in splits:
        y_test = y[split["test_idx"]]
        assert set(y_test) == {0, 1}
        assert (y_test == 0).sum() == 3
        assert (y_test == 1).sum() == 3


def test_data_stratify_col_uses_target_plus_metadata_strata():
    y = np.array([0, 0, 1, 1] * 6)
    meta = pd.DataFrame({"batch": ["A", "A", "A", "A", "B", "B", "B", "B"] * 3})
    strata = _strata_from_metadata(meta, y, "batch")
    plan = Evaluation(protocol="repeated_nested_cv", outer_folds=3, repeats=1, random_state=11)
    splits = _outer_splits(plan, y, groups=None, strata=strata, stratify_col="batch")
    assert len(splits) == 3
    full_counts = pd.Series(strata).value_counts().sort_index()
    for split in splits:
        test_counts = pd.Series(strata[split["test_idx"]]).value_counts().sort_index()
        # Every composite target+batch stratum is represented in every fold.
        assert set(test_counts.index) == set(full_counts.index)


def test_wide_csv_stratify_col_is_metadata_not_feature(tmp_path):
    path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "label": ["control", "case"] * 3,
            "batch_numeric": [1, 1, 2, 2, 3, 3],
            "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_b": [1, 0, 1, 0, 1, 0],
        }
    )
    df.to_csv(path, index=False)
    dataset = load_dataset(
        Data(
            abundance_path=path,
            format="wide_csv",
            sample_id_col="sample_id",
            target_col="label",
            stratify_col="batch_numeric",
            label_map={"control": 0, "case": 1},
            class_labels=("control", "case"),
        ),
        ("all",),
    )
    assert "batch_numeric" in dataset.metadata.columns
    assert "batch_numeric" not in dataset.feature_names_by_level["all"]
