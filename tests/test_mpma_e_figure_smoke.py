from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mllabiome.mpma_e_figure import write_single_task_mpma_e_figure


def test_single_task_mpma_e_figure_writes_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "ensembling").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "config_id": "cfg1",
                "resolution": "family",
                "levels": "family",
                "count_transformation": "none",
                "learner": "LR_liblinear",
                "active": 1,
            },
            {
                "config_id": "cfg2",
                "resolution": "family",
                "levels": "family",
                "count_transformation": "arcsin_sqrt",
                "learner": "RF_100",
                "active": 1,
            },
        ]
    ).to_csv(run / "configs.tsv", sep="\t", index=False)
    (run / "ensembling" / "selected_unit.json").write_text(
        json.dumps(
            {
                "inner_val_best_mpmas_ensemble": {
                    "members": ["cfg1", "cfg2"],
                    "selection_strategy": "top_k",
                    "aggregation_strategy": "mean_proba",
                    "ensemble_size": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    X = np.array([[1.0, 3.0], [2.0, 1.0], [3.0, 2.0]], dtype=float)
    taxa = [
        "d__Bacteria___p__Firmicutes___c__Bacilli___o__Lactobacillales___f__Lactobacillaceae",
        "d__Bacteria___p__Bacteroidota___c__Bacteroidia___o__Bacteroidales___f__Bacteroidaceae",
    ]

    outputs = write_single_task_mpma_e_figure(
        run,
        task_key="SMOKE",
        task_title="Smoke MPMA-E",
        X=X,
        taxa=taxa,
        out_dir=run / "figures",
        out_name="mpma_e",
        max_members=5,
    )

    for key in ["mpma_e_svg", "mpma_e_pdf", "mpma_e_png", "mpma_e_members", "mpma_e_diagnostics"]:
        assert outputs[key].exists()
    assert "ŷ" in outputs["mpma_e_svg"].read_text(encoding="utf-8")
