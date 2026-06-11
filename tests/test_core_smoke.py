from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mllabiome import mll
from mllabiome.resolutions import materialize_mpdr


def _write_small_wide_csv(path: Path, n: int = 12) -> None:
    labels = ["control", "case"] * (n // 2)
    rows = []
    for i, label in enumerate(labels):
        case = 1.0 if label == "case" else 0.0
        rows.append(
            {
                "sample_id": f"S{i:02d}",
                "label": label,
                "cohort": "A" if i < n // 2 else "B",
                "d__Bacteria___p__Firmicutes___c__Bacilli___o__Lactobacillales___f__Lactobacillaceae": 10.0 + case * 5.0 + (i % 3),
                "d__Bacteria___p__Bacteroidota___c__Bacteroidia___o__Bacteroidales___f__Bacteroidaceae": 20.0 - case * 4.0 + (i % 2),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_wide_csv_loading_materialization_and_transform(tmp_path: Path) -> None:
    csv_path = tmp_path / "abundance.csv"
    _write_small_wide_csv(csv_path)

    dataset = mll.load_dataset(
        mll.Data(
            abundance_path=csv_path,
            format="wide_csv",
            sample_id_col="sample_id",
            target_col="label",
            group_col="cohort",
            label_map={"control": 0, "case": 1},
            class_labels=("control", "case"),
        ),
        levels_needed=("family",),
    )

    X_family, names = materialize_mpdr(dataset, ("family",))
    assert X_family.shape == (12, 2)
    assert len(names) == 2
    assert dataset.y.tolist() == [0, 1] * 6

    tr = mll.Transformation("arcsin_sqrt")
    X_tr, X_te = tr.apply(X_family[:8], X_family[8:])
    assert X_tr.shape == (8, 2)
    assert X_te.shape == (4, 2)
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_te).all()


def test_nested_cv_evaluate_smoke(tmp_path: Path) -> None:
    csv_path = tmp_path / "abundance.csv"
    _write_small_wide_csv(csv_path, n=12)

    sweep = mll.Sweep(
        title="core smoke sweep",
        experiment_dir=tmp_path / "run",
        data=mll.Data(
            abundance_path=csv_path,
            format="wide_csv",
            sample_id_col="sample_id",
            target_col="label",
            label_map={"control": 0, "case": 1},
            class_labels=("control", "case"),
            positive_class=1,
        ),
        resolutions=(("family", ("family",)),),
        count_transformations=(mll.Transformation("none"),),
        learners=(
            (
                "LR_liblinear",
                LogisticRegression(solver="liblinear", random_state=42),
            ),
        ),
        evaluation=mll.Evaluation(
            protocol="nested_cv",
            outer_folds=2,
            inner_folds=2,
            repeats=1,
            random_state=42,
            n_jobs=1,
        ),
        gate=mll.QualificationGate(enabled=False),
        ensemble=mll.Ensemble(sizes=(2,)),
        explainability=mll.Explainability(methods=("shap",), top_k=3),
    )

    outputs = mll.evaluate(sweep)
    assert outputs["configs"].exists()
    assert outputs["rankings"].exists()

    rankings = pd.read_csv(outputs["rankings"], sep="\t")
    assert len(rankings) == 1
    assert rankings.loc[0, "resolution"] == "family"
    assert rankings.loc[0, "count_transformation"] == "none"


def test_explainability_targets_configuration() -> None:
    default = mll.Explainability()
    assert default.targets == "auto"

    explicit = mll.Explainability(targets=("mpma_e", "mpma_b"), methods=("shap",))
    assert explicit.targets == ("mpma_e", "mpma_b")
    assert explicit.methods == ("shap",)

    with pytest.raises(TypeError):
        mll.Explainability(unknown_option="mpma_b")
