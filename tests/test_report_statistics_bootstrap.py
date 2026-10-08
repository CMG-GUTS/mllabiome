from __future__ import annotations

import numpy as np
import pandas as pd

from mllabiome.report_statistics import (
    _paired_resample_indices,
    _prepare_oof_frame,
    _resample_subjects,
)


def _repeated_frame():
    rows = []
    for repeat in range(3):
        for index, subject in enumerate(("p1", "p2", "p3")):
            y_true = index % 2
            p1 = 0.8 if y_true else 0.2
            rows.append(
                {
                    "outer_split_key": f"r{repeat}_o{index}",
                    "sample_id": f"s{index + 1}",
                    "subject_id": subject,
                    "y_true": y_true,
                    "y_pred": y_true,
                    "proba_control": 1.0 - p1,
                    "proba_case": p1,
                }
            )
    return _prepare_oof_frame(pd.DataFrame(rows), "repeated_nested_cv")


def _lodo_frame():
    rows = []
    sample_no = 0
    for cohort, subjects in (("A", ("p1", "p2")), ("B", ("p3", "p4"))):
        for subject_no, subject in enumerate(subjects):
            for visit in range(2):
                sample_no += 1
                y_true = subject_no % 2
                p1 = 0.75 if y_true else 0.25
                rows.append(
                    {
                        "outer_split_key": cohort,
                        "sample_id": f"s{sample_no}",
                        "subject_id": subject,
                        "y_true": y_true,
                        "y_pred": y_true,
                        "proba_control": 1.0 - p1,
                        "proba_case": p1,
                    }
                )
    return _prepare_oof_frame(pd.DataFrame(rows), "lodo")


def test_repeated_cv_subject_resampling_keeps_all_repeats_together():
    frame = _repeated_frame()
    sampled = _resample_subjects(frame, np.random.default_rng(12))
    for _, group in sampled.groupby("_subject_id", sort=False):
        counts = group["_repeat"].value_counts()
        assert set(counts.index) == {"r0", "r1", "r2"}
        assert counts.nunique() == 1


def test_repeated_cv_paired_resampling_uses_subject_clusters():
    frame = _repeated_frame()
    sampled_indices = _paired_resample_indices(
        frame,
        "repeated_nested_cv",
        np.random.default_rng(31),
    )
    assert len(sampled_indices) == 1
    sampled = frame.iloc[sampled_indices[0]]
    for _, group in sampled.groupby("_subject_id", sort=False):
        counts = group["_repeat"].value_counts()
        assert set(counts.index) == {"r0", "r1", "r2"}
        assert counts.nunique() == 1


def test_lodo_paired_resampling_is_cohort_then_subject_clustered():
    frame = _lodo_frame()
    sampled_indices = _paired_resample_indices(frame, "lodo", np.random.default_rng(7))
    assert len(sampled_indices) == frame["_cluster"].nunique()
    for indices in sampled_indices:
        sampled = frame.iloc[indices]
        assert sampled["_cluster"].nunique() == 1
        for _, group in sampled.groupby("_subject_id", sort=False):
            assert len(group) % 2 == 0
            assert set(group["sample_id"]).issubset(set(frame["sample_id"]))
