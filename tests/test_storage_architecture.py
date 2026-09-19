from pathlib import Path

import pandas as pd

from mllabiome import storage


def test_write_table_preserves_existing_text_siblings(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "require_parquet_engine", lambda: None)

    def fake_to_parquet(self, path, index=False, compression=None):
        assert index is False
        assert compression == "zstd"
        Path(path).write_bytes(b"parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    target = tmp_path / "results.parquet"
    tsv = tmp_path / "results.tsv"
    csv = tmp_path / "results.csv"
    tsv.write_text("keep\n", encoding="utf-8")
    csv.write_text("keep\n", encoding="utf-8")

    out = storage.write_table(target, pd.DataFrame({"x": [1, 2]}))

    assert out == target
    assert target.exists()
    assert tsv.read_text(encoding="utf-8") == "keep\n"
    assert csv.read_text(encoding="utf-8") == "keep\n"
    assert not (tmp_path / "results.tmp.parquet").exists()


def test_write_table_retries_mixed_text_metadata_only_after_arrow_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage, "require_parquet_engine", lambda: None)
    calls = []

    class ArrowTypeError(Exception):
        pass

    def fake_to_parquet(self, path, index=False, compression=None):
        calls.append(self.copy())
        values = self["integration_n_components"].dropna().tolist()
        has_text = any(isinstance(value, str) for value in values)
        has_nontext = any(not isinstance(value, str) for value in values)
        if has_text and has_nontext:
            raise ArrowTypeError("mixed metadata")
        Path(path).write_bytes(b"parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    frame = pd.DataFrame(
        {
            "integration_n_components": ["", 32, None],
            "score": [1.0, 2.0, 3.0],
        }
    )

    target = storage.write_table(tmp_path / "mixed.parquet", frame)

    assert target.exists()
    assert len(calls) == 2
    assert calls[0]["integration_n_components"].dtype == object
    converted = calls[1]["integration_n_components"].dropna().tolist()
    assert converted == ["", "32"]
    assert calls[1]["score"].tolist() == [1.0, 2.0, 3.0]


def test_tsv_export_ignores_temporary_parquet_files(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "require_parquet_engine", lambda: None)
    canonical = tmp_path / "results" / "scores.parquet"
    temporary = tmp_path / "results" / "scores.tmp.parquet"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    temporary.write_bytes(b"temporary")

    def fake_read_parquet(path):
        assert Path(path) == canonical
        return pd.DataFrame({"score": [0.1, 0.2]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    exported = storage.export_tsv_tree(tmp_path)

    assert len(exported["files"]) == 1
    assert exported["files"][0]["source"] == "results/scores.parquet"
    assert (tmp_path / "exports" / "tsv" / "results" / "scores.tsv").exists()
    assert not (tmp_path / "exports" / "tsv" / "results" / "scores.tmp.tsv").exists()
