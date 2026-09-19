from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .console import info
from .storage import read_table, table_exists
from . import explainability as _core
from .final_models import build_final_models
from .mpma_e_explainability import explain_mpma_e


def _read_explained(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _member_ids(value) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in text.split(",") if item.strip()]
        items = parsed if isinstance(parsed, list) else []
    else:
        items = []
    return [
        str(item.get("config_id") if isinstance(item, dict) else item).strip()
        for item in items
        if str(item.get("config_id") if isinstance(item, dict) else item).strip()
    ]


def _invalidate_stale(root: Path, models: dict, explainability) -> None:
    mpma_b_dir = root / "explainability" / "mpma_b"
    current_b = _read_explained(mpma_b_dir / "explained_unit.json")
    current_b_id = (
        str(current_b.get("config", {}).get("config_id", ""))
        if isinstance(current_b.get("config"), dict)
        else ""
    )
    final_b_id = str(models["MPMA-B"]["config_id"])
    current_b_signature = str(current_b.get("explainability_config_signature", ""))
    expected_signature = _core._explainability_config_signature(explainability)
    if mpma_b_dir.exists() and (
        current_b_id != final_b_id or current_b_signature != expected_signature
    ):
        shutil.rmtree(mpma_b_dir)

    mpma_e = models.get("MPMA-E")
    if not isinstance(mpma_e, dict):
        return
    mpma_e_dir = root / "explainability" / "mpma_e"
    current_e = _read_explained(mpma_e_dir / "explained_unit.json")
    current_id = str(current_e.get("ensemble_config_id", ""))
    final_id = str(mpma_e.get("ensemble_config_id", ""))
    current_members = _member_ids(current_e.get("members"))
    final_members = [str(member["config_id"]) for member in mpma_e["members"]]
    current_aggregation = str(current_e.get("aggregation_strategy", ""))
    final_aggregation = str(mpma_e["aggregation_strategy"])
    current_e_signature = str(current_e.get("explainability_config_signature", ""))
    if mpma_e_dir.exists() and (
        (current_id and final_id and current_id != final_id)
        or current_members != final_members
        or current_aggregation != final_aggregation
        or current_e_signature != expected_signature
    ):
        shutil.rmtree(mpma_e_dir)


def explain(sweep):
    source = sweep.samples if getattr(sweep, "uses_modalities", False) else sweep.data
    if (
        str(getattr(source, "task", "classification")).strip().casefold()
        == "regression"
    ):
        from .regression_explainability import explain_regression

        return explain_regression(sweep)
    root = Path(sweep.root())
    models = build_final_models(root)
    _invalidate_stale(root, models, sweep.explainability)
    rankings_path = root / "tables" / "mpma_rankings.parquet"
    if not table_exists(rankings_path):
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = read_table(rankings_path)
    targets = _core._automatic_explainability_targets(sweep, rankings)
    mpma_e = models.get("MPMA-E")

    outputs = {}
    for target_no, target in enumerate(targets, start=1):
        info(f"Explainability · target {target_no}/{len(targets)} · {target}")
        text = str(target)
        if text in {"best", "best_individual", "best_mpma", "mpma_b", "MPMA-B"}:
            out = _core._explain_one(
                sweep,
                target_override=str(models["MPMA-B"]["config_id"]),
                output_slug_override="mpma_b",
                display_label_override="MPMA-B",
            )
            key = "mpma_b"
        elif text in {"ensemble", "mpma_e", "MPMA-E"}:
            if getattr(sweep, "uses_modalities", False):
                raise _core.ExplainabilityConfigurationError(
                    "MPMA-E explainability for modality-based candidates is not enabled in rc24. Ensemble selection and prediction are supported, but hierarchical attribution across heterogeneous modality and integration pipelines requires a separate exact reconstruction contract and is intentionally not approximated."
                )
            if not isinstance(mpma_e, dict):
                raise _core.ExplainabilityConfigurationError(
                    "MPMA-E explainability was requested, but no final MPMA-E specification is available."
                )

            member_ids = [str(member["config_id"]) for member in mpma_e["members"]]
            _core._ensure_mpma_member_explanations(sweep, rankings, member_ids)
            out = explain_mpma_e(sweep, rankings)
            key = "mpma_e"
        else:
            out = _core._explain_one(sweep, target_override=text)
            key = {
                "baseline_rf": "baseline_rf",
                "Baseline RF": "baseline_rf",
                "baseline-rf": "baseline_rf",
            }.get(text, text)
        for name, path in out.items():
            outputs[f"{key}_{name}"] = path
    return outputs
