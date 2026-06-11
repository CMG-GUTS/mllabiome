from __future__ import annotations

import pandas as pd

from mllabiome.explainability import _single_method_support_table, _support_from_importance


def test_support_from_importance_collapses_duplicate_feature_labels() -> None:
    frame = pd.DataFrame(
        {
            "method": ["shap", "shap", "shap"],
            "feature": ["f1", "f1", "f2"],
            "importance_mean": [0.2, 0.3, 0.1],
        }
    )

    support = _support_from_importance(frame)

    assert support.index.is_unique
    assert set(support.index) == {"f1", "f2"}
    assert float(support.loc["f1"]) >= float(support.loc["f2"])


def test_single_method_support_table_accepts_duplicate_feature_labels() -> None:
    frame = pd.DataFrame(
        {
            "method": ["shap", "shap", "shap"],
            "feature": ["same", "same", "other"],
            "importance_mean": [0.25, 0.35, 0.1],
            "importance_sd": [0.0, 0.0, 0.0],
            "scoring": ["mean_abs", "mean_abs", "mean_abs"],
        }
    )

    top = _single_method_support_table(frame, top_k=2)

    assert top["feature"].tolist()[0] == "same"
    assert top["feature"].is_unique
