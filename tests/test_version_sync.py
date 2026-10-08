import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sync_version.py"
SPEC = importlib.util.spec_from_file_location("sync_version", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SYNC_VERSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC_VERSION)


def test_update_citation_version():
    text = 'cff-version: 1.2.0\nversion: "0.1.0rc1"\n'
    assert (
        SYNC_VERSION.update_citation(text, "0.1.0rc2")
        == 'cff-version: 1.2.0\nversion: "0.1.0rc2"\n'
    )


def test_update_docs_version():
    text = "These pages describe the `0.1.0rc1` release candidate."
    assert (
        SYNC_VERSION.update_docs(text, "0.1.0rc2")
        == "These pages describe mllabiome version `0.1.0rc2`."
    )


def test_update_changelog_version():
    text = "# Changelog\n\n## 0.1.0rc1 - Unreleased\n\n## 0.1.0 - 2026-06-11\n"
    assert (
        SYNC_VERSION.update_changelog(text, "0.1.0rc2")
        == "# Changelog\n\n## 0.1.0rc2 - Unreleased\n\n## 0.1.0 - 2026-06-11\n"
    )


def test_update_changelog_accepts_released_version():
    text = "# Changelog\n\n## 0.1.0 - 2026-10-05\n"
    assert SYNC_VERSION.update_changelog(text, "0.1.0") == text
