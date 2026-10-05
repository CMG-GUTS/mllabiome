from __future__ import annotations

SUPPORTED_EVALUATION_PROTOCOLS = frozenset(
    {
        "repeated_nested_cv",
        "nested_cv",
        "lodo",
        "leave_one_dataset_out",
        "hierarchical_lodo",
    }
)
LODO_PROTOCOLS = frozenset(
    {
        "lodo",
        "leave_one_dataset_out",
        "hierarchical_lodo",
    }
)


def normalize_evaluation_protocol(protocol: object) -> str:
    return str(protocol).strip().casefold().replace("-", "_")


def is_lodo_protocol(protocol: object) -> bool:
    return normalize_evaluation_protocol(protocol) in LODO_PROTOCOLS


def is_hierarchical_lodo_protocol(protocol: object) -> bool:
    return normalize_evaluation_protocol(protocol) == "hierarchical_lodo"
