from __future__ import annotations

import numpy as np
import pandas as pd

from mllabiome.data import Dataset, dataset_fingerprint


def _dataset(
    X: np.ndarray,
    y: np.ndarray | None = None,
    subject_ids: list[str] | None = None,
) -> Dataset:
    target = np.array([0, 1], dtype=int) if y is None else np.asarray(y)
    return Dataset(
        X_by_level={"all": np.asarray(X, dtype=np.float32)},
        feature_names_by_level={"all": ["f1", "f2"]},
        y=target,
        sample_ids=["s1", "s2"],
        subject_ids=["s1", "s2"] if subject_ids is None else subject_ids,
        metadata=pd.DataFrame({"sample_id": ["s1", "s2"]}),
        class_labels=["control", "case"],
        positive_class=1,
        task="classification",
        target_name="group",
    )


def test_dataset_fingerprint_is_stable_for_identical_model_input():
    first = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
    second = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert dataset_fingerprint(first) == dataset_fingerprint(second)


def test_dataset_fingerprint_changes_with_abundance_content():
    first = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
    second = _dataset(np.array([[1.0, 2.0], [3.0, 5.0]]))
    assert dataset_fingerprint(first) != dataset_fingerprint(second)


def test_dataset_fingerprint_changes_with_target_content():
    first = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]))
    second = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1, 0]))
    assert dataset_fingerprint(first) != dataset_fingerprint(second)


def test_dataset_fingerprint_changes_with_subject_identity():
    first = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), subject_ids=["p1", "p2"])
    second = _dataset(np.array([[1.0, 2.0], [3.0, 4.0]]), subject_ids=["p1", "p1"])
    assert dataset_fingerprint(first) != dataset_fingerprint(second)
