from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "mllabiome" / "_version.py"
CITATION_FILE = ROOT / "CITATION.cff"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
DOCS_VERSION_FILE = (
    ROOT / "docs" / "content" / "docs" / "current" / "outputs-and-reproducibility.mdx"
)


def read_version() -> str:
    module = ast.parse(
        VERSION_FILE.read_text(encoding="utf-8"), filename=str(VERSION_FILE)
    )
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    raise RuntimeError(f"Unable to read __version__ from {VERSION_FILE}")


def update_citation(text: str, version: str) -> str:
    updated, count = re.subn(
        r'(?m)^version:[ \t]*["\']?[^"\'\n]+["\']?[ \t]*$',
        f'version: "{version}"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Unable to update {CITATION_FILE}")
    return updated


def update_docs(text: str, version: str) -> str:
    updated, count = re.subn(
        r"These pages describe (?:the |mllabiome version )`[^`]+`(?: release candidate| release)?\.",
        f"These pages describe mllabiome version `{version}`.",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Unable to update {DOCS_VERSION_FILE}")
    return updated


def update_changelog(text: str, version: str) -> str:
    heading = f"## {version} - Unreleased"
    updated, count = re.subn(r"(?m)^## .* - Unreleased$", heading, text, count=1)
    if count == 1:
        return updated
    match = re.search(r"(?m)^# Changelog\s*$", text)
    if match is None:
        raise RuntimeError(f"Unable to update {CHANGELOG_FILE}")
    insert_at = match.end()
    return text[:insert_at] + f"\n\n{heading}" + text[insert_at:]


def synchronize(check: bool) -> int:
    version = read_version()
    updates = {
        CITATION_FILE: update_citation(
            CITATION_FILE.read_text(encoding="utf-8"), version
        ),
        CHANGELOG_FILE: update_changelog(
            CHANGELOG_FILE.read_text(encoding="utf-8"), version
        ),
        DOCS_VERSION_FILE: update_docs(
            DOCS_VERSION_FILE.read_text(encoding="utf-8"), version
        ),
    }
    changed = [
        path
        for path, content in updates.items()
        if path.read_text(encoding="utf-8") != content
    ]
    if check:
        if changed:
            for path in changed:
                print(path.relative_to(ROOT))
            return 1
        return 0
    for path in changed:
        path.write_text(updates[path], encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return synchronize(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
