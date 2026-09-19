from pathlib import Path

import pytest

from mllabiome import cli


def test_report_opening_uses_local_file_uri_without_server(tmp_path, monkeypatch):
    report_path = tmp_path / "report" / "index.html"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("<html></html>", encoding="utf-8")
    opened = []
    monkeypatch.setattr(
        cli.webbrowser,
        "open",
        lambda target, new=0, autoraise=True: opened.append(target) or True,
    )

    cli._open_report(tmp_path)

    assert opened == [report_path.resolve().as_uri()]
    assert opened[0].startswith("file:")


@pytest.mark.parametrize("flag", ["--export-tsv", "--export-png", "--export-pdf"])
def test_export_flags_are_report_stage_only(tmp_path, flag):
    config = tmp_path / "config.py"
    with pytest.raises(SystemExit) as exc:
        cli.main([str(config), "--stage", "evaluate", flag])
    assert exc.value.code == 2
