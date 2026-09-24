from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mllabiome.data import Data, load_dataset


def _wide_csv(tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "participant_id": ["p1", "p2", "p3", "p4"],
            "label": ["control", "case", "control", "case"],
            "g__A": [0.1, 0.2, 0.3, 0.4],
            "g__B": [0.9, 0.8, 0.7, 0.6],
            "age": [21, 34, 48, 57],
            "bmi": [20.5, 24.0, 28.5, 31.0],
        }
    ).to_csv(path, index=False)
    return path


def test_wide_csv_rejects_implicit_numeric_feature_inference(tmp_path):
    path = _wide_csv(tmp_path)
    with pytest.raises(ValueError, match="must be selected explicitly"):
        load_dataset(
            Data(
                abundance_path=path,
                format="wide_csv",
                sample_id_col="sample_id",
                target_col="label",
                subject_id_col="participant_id",
            ),
            ("genus",),
        )


def test_wide_csv_feature_prefixes_exclude_numeric_covariates(tmp_path):
    path = _wide_csv(tmp_path)
    dataset = load_dataset(
        Data(
            abundance_path=path,
            format="wide_csv",
            sample_id_col="sample_id",
            target_col="label",
            subject_id_col="participant_id",
            feature_prefixes=("g__",),
        ),
        ("genus",),
    )
    assert dataset.feature_names_by_level["all"] == ["g__A", "g__B"]
    assert dataset.subject_ids == ["p1", "p2", "p3", "p4"]
    np.testing.assert_allclose(
        dataset.X_by_level["all"],
        np.asarray(
            [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
            dtype=np.float32,
        ),
    )


def test_wide_csv_explicit_feature_columns_are_exact(tmp_path):
    path = _wide_csv(tmp_path)
    dataset = load_dataset(
        Data(
            abundance_path=path,
            format="wide_csv",
            sample_id_col="sample_id",
            target_col="label",
            feature_cols=("g__B", "g__A"),
        ),
        ("genus",),
    )
    assert dataset.feature_names_by_level["all"] == ["g__B", "g__A"]
    assert dataset.subject_ids == dataset.sample_ids


def test_wide_csv_implicit_numeric_features_require_explicit_opt_in(tmp_path):
    path = _wide_csv(tmp_path)
    dataset = load_dataset(
        Data(
            abundance_path=path,
            format="wide_csv",
            sample_id_col="sample_id",
            target_col="label",
            subject_id_col="participant_id",
            allow_implicit_numeric_features=True,
        ),
        ("genus",),
    )
    assert dataset.feature_names_by_level["all"] == ["g__A", "g__B", "age", "bmi"]


def test_wide_csv_duplicate_sample_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicates.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "label": ["control", "case"],
            "g__A": [0.1, 0.2],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        load_dataset(
            Data(
                abundance_path=path,
                format="wide_csv",
                sample_id_col="sample_id",
                target_col="label",
                feature_cols=("g__A",),
            ),
            ("genus",),
        )
