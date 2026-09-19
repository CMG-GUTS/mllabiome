from mllabiome import console as console_module


def test_path_table_prints_only_common_main_location(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(
        console_module.console, "print", lambda value: printed.append(str(value))
    )
    rows = {
        "outer results": tmp_path / "results" / "outer_results.parquet",
        "predictions": tmp_path / "results" / "outer_predictions.parquet",
    }

    console_module.path_table("Evaluation outputs", rows)

    assert len(printed) == 1
    assert str(tmp_path / "results") in printed[0]
    assert "outer_results.parquet" not in printed[0]
    assert "outer_predictions.parquet" not in printed[0]
