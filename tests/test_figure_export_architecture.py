from pathlib import Path
from types import SimpleNamespace

from mllabiome import figure_export, style


def test_canonical_figure_save_writes_svg_only(tmp_path):
    saved = []

    class Figure:
        def savefig(self, path, **kwargs):
            saved.append((Path(path), dict(kwargs)))

    style.save_all(Figure(), tmp_path / "figure")

    assert [path.suffix for path, _ in saved] == [".svg"]
    assert saved[0][1]["dpi"] == 300


def test_explicit_figure_export_mirrors_svg_tree(tmp_path, monkeypatch):
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a.svg").write_text("<svg></svg>", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.svg").write_text("<svg></svg>", encoding="utf-8")
    preexisting_export = tmp_path / "exports" / "png" / "old.svg"
    preexisting_export.parent.mkdir(parents=True)
    preexisting_export.write_text("<svg></svg>", encoding="utf-8")

    def convert(url, write_to, **kwargs):
        assert Path(url).suffix == ".svg"
        Path(write_to).write_bytes(b"converted")

    monkeypatch.setattr(
        figure_export,
        "_converter",
        lambda: SimpleNamespace(svg2png=convert, svg2pdf=convert),
    )

    out = figure_export.export_svg_tree(tmp_path, ["png", "pdf", "png"])

    assert len(out["files"]) == 4
    assert (tmp_path / "exports" / "png" / "figures" / "a.png").exists()
    assert (tmp_path / "exports" / "png" / "nested" / "b.png").exists()
    assert (tmp_path / "exports" / "pdf" / "figures" / "a.pdf").exists()
    assert (tmp_path / "exports" / "pdf" / "nested" / "b.pdf").exists()
    assert not (tmp_path / "exports" / "png" / "exports" / "png" / "old.png").exists()


def test_figure_export_rejects_non_derived_formats(tmp_path):
    try:
        figure_export.export_svg_tree(tmp_path, ["jpg"])
    except ValueError as exc:
        assert "Unsupported figure export format" in str(exc)
    else:
        raise AssertionError("Unsupported figure format was accepted")
