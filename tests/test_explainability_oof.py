from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from mllabiome import mll


def _write_small_wide_csv(path: Path, n: int = 12) -> None:
    labels = ["control", "case"] * (n // 2)
    rows = []
    for i, label in enumerate(labels):
        case = 1.0 if label == "case" else 0.0
        rows.append(
            {
                "sample_id": f"S{i:02d}",
                "label": label,
                "d__Bacteria___p__Firmicutes___c__Bacilli___o__Lactobacillales___f__Lactobacillaceae": 10.0 + case * 5.0 + (i % 3),
                "d__Bacteria___p__Bacteroidota___c__Bacteroidia___o__Bacteroidales___f__Bacteroidaceae": 20.0 - case * 4.0 + (i % 2),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_explainability_uses_outer_test_oof_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "abundance.csv"
    _write_small_wide_csv(csv_path, n=12)

    sweep = mll.Sweep(
        title="oof explainability smoke",
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
        learners=(("LR_liblinear", LogisticRegression(solver="liblinear", random_state=42)),),
        evaluation=mll.Evaluation(
            protocol="nested_cv",
            outer_folds=2,
            inner_folds=2,
            repeats=1,
            random_state=42,
            n_jobs=1,
        ),
        gate=mll.QualificationGate(enabled=False),
        explainability=mll.Explainability(
            targets=("mpma_b",),
            methods=("permutation",),
            n_repeats=1,
            top_k=2,
            representative_instances=False,
        ),
    )

    mll.evaluate(sweep)
    out = mll.explain(sweep)

    explained = json.loads((out["mpma_b_explainability_dir"] / "explained_unit.json").read_text())
    assert explained["out_of_fold"] is True
    assert explained["cross_validation_explanations"] == "outer_test_folds"

    oof = pd.read_csv(out["mpma_b_oof_predictions"], sep="\t")
    assert len(oof) == 12
    assert oof["sample_id"].nunique() == 12
    assert set(oof.columns) >= {"split_key", "sample_id", "sample_index", "y_true", "y_pred"}
