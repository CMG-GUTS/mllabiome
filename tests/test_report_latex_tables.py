from __future__ import annotations

from pathlib import Path

import pandas as pd

from mllabiome import report


def test_strategy_latex_table_avoids_array_package_column_syntax(monkeypatch, tmp_path: Path) -> None:
    def fake_display(root: Path, html_mode: bool = False) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Strategy": "MPMA-E", "ROC-AUC": r"\textbf{93.0 ± 1.0}", "nMCC": "82.0 ± 2.0", "F1w": "88.0 ± 1.0", "Precision": "89.0 ± 1.0", "Recall": "87.0 ± 1.0"},
                {"Strategy": "MPMA-B", "ROC-AUC": "91.0 ± 1.0", "nMCC": r"\textbf{83.0 ± 2.0}", "F1w": "87.0 ± 1.0", "Precision": "88.0 ± 1.0", "Recall": "86.0 ± 1.0"},
                {"Strategy": "AutoML", "ROC-AUC": "--", "nMCC": "--", "F1w": "--", "Precision": "--", "Recall": "--"},
                {"Strategy": "Baseline RF", "ROC-AUC": "90.0 ± 1.0", "nMCC": "80.0 ± 2.0", "F1w": "85.0 ± 1.0", "Precision": "86.0 ± 1.0", "Recall": r"\textbf{89.0 ± 1.0}"},
            ]
        )

    monkeypatch.setattr(report, "_strategy_performance_display", fake_display)
    out = tmp_path / "strategy_table.tex"
    report._strategy_latex_table(tmp_path, out, "PTSD within-dataset MPMA sweep")
    tex = out.read_text(encoding="utf-8")

    assert "@{}" not in tex
    assert r"\begin{table}[H]" not in tex
    assert r"\begin{table}[htbp]" in tex
    assert r"\arraybackslash" not in tex
    assert ">{" not in tex
    assert r"m{\mllabiome" not in tex
    assert r"\begin{tabular}{p{\mllabiometaskcol}" in tex


def test_generic_latex_tabular_uses_plain_column_spec(tmp_path: Path) -> None:
    out = tmp_path / "generic_table.tex"
    report._latex_tabular(
        pd.DataFrame([{"A": "x", "B": "y"}]),
        out,
        caption="Caption",
        label="tab:test",
        align="ll",
    )
    tex = out.read_text(encoding="utf-8")

    assert r"\begin{tabular}{ll}" in tex
    assert "@{}" not in tex
    assert r"\begin{table}[H]" not in tex
    assert r"\begin{table}[htbp]" in tex
