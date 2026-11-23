from __future__ import annotations
import re
from typing import Any

RANKS = ("domain", "phylum", "class", "order", "family", "genus", "species", "strain")
PREFIXES = {
    "d__": "domain",
    "k__": "domain",
    "p__": "phylum",
    "c__": "class",
    "o__": "order",
    "f__": "family",
    "g__": "genus",
    "s__": "species",
    "t__": "strain",
    "st__": "strain",
}
_SEPARATORS = re.compile(r"___|[|;,\t]+")
_SGB = re.compile(r"^t__SGB([0-9]+)(?:_group)?$", re.IGNORECASE)


def taxonomic_tokens(name: Any) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for part in _SEPARATORS.split(str(name).strip()):
        part = part.strip()
        lower = part.casefold()
        for prefix, level in PREFIXES.items():
            if lower.startswith(prefix) and len(part) > len(prefix):
                tokens[level] = (
                    "d__" + lower[len(prefix) :] if level == "domain" else lower
                )
                break
    return tokens


def taxonomic_rank(name: Any) -> str | None:
    tokens = taxonomic_tokens(name)
    return next((rank for rank in reversed(RANKS) if rank in tokens), None)


def sgb_identifier(name: Any) -> str | None:
    part = taxonomic_tokens(name).get("strain")
    if part is None:
        return None
    result = _SGB.fullmatch(part)
    return result.group(1) if result else None


def lineage_matches(
    query: dict[str, str], reference: dict[str, str], rank: str
) -> bool:
    if rank not in query or rank not in reference:
        return False
    requested_depth = RANKS.index(rank)
    return all(
        level not in query or query[level] == reference.get(level)
        for level in RANKS[: requested_depth + 1]
    )
