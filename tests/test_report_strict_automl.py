from __future__ import annotations

from pathlib import Path

import pandas as pd

from mllabiome import report


def _write_outer(root: Path, rows: list[dict[str, object]]) -> None:
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "outer_results.tsv", sep="\t", index=False)


def _automl_strategy(root: Path) -> dict[str, object]:
    rows = report._strategy_rows(root)
    return next(r for r in rows if r.get("Strategy") == "AutoML")


def test_automl_strategy_has_no_fallback_to_non_strict_representation(tmp_path: Path) -> None:
    _write_outer(
        tmp_path,
        [
            {
                "config_id": "auto_multirank_raw",
                "learner": "FLAML_600s",
                "count_transformation": "none",
                "resolution": "family-genus",
                "ok": 1,
                "AUC": 0.99,
                "nMCC": 0.90,
                "F1w": 0.91,
                "Precision": 0.92,
                "Recall": 0.93,
            },
            {
                "config_id": "auto_species_arcsin",
                "learner": "FLAML_600s",
                "count_transformation": "arcsin_sqrt",
                "resolution": "species",
                "ok": 1,
                "AUC": 0.98,
                "nMCC": 0.89,
                "F1w": 0.90,
                "Precision": 0.91,
                "Recall": 0.92,
            },
        ],
    )

    row = _automl_strategy(tmp_path)

    assert row == {"Strategy": "AutoML", "source": "mpma"}


def test_automl_strategy_uses_deepest_available_raw_single_rank(tmp_path: Path) -> None:
    _write_outer(
        tmp_path,
        [
            {
                "config_id": "auto_genus_high_score",
                "learner": "FLAML_600s",
                "count_transformation": "none",
                "resolution": "genus",
                "ok": 1,
                "AUC": 0.99,
                "nMCC": 0.95,
                "F1w": 0.94,
                "Precision": 0.93,
                "Recall": 0.92,
            },
            {
                "config_id": "auto_species_lower_score",
                "learner": "FLAML_600s",
                "count_transformation": "none",
                "resolution": "species",
                "ok": 1,
                "AUC": 0.80,
                "nMCC": 0.70,
                "F1w": 0.71,
                "Precision": 0.72,
                "Recall": 0.73,
            },
        ],
    )

    row = _automl_strategy(tmp_path)

    assert row["config_id"] == "auto_species_lower_score"
    assert row["resolution"] == "species"
    assert row["count_transformation"] == "none"
