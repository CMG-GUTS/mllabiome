from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from . import explainability as _core
from .final_models import aggregate_member_predictions, build_final_models


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


def _invalidate_stale(root: Path, models: dict) -> None:
    mpma_b_dir = root / "explainability" / "mpma_b"
    current_b = _read_explained(mpma_b_dir / "explained_unit.json")
    current_b_id = (
        str(current_b.get("config", {}).get("config_id", ""))
        if isinstance(current_b.get("config"), dict)
        else ""
    )
    final_b_id = str(models["MPMA-B"]["config_id"])
    if mpma_b_dir.exists() and current_b_id != final_b_id:
        shutil.rmtree(mpma_b_dir)
    mpma_e = models.get("MPMA-E")
    if not isinstance(mpma_e, dict):
        return
    mpma_e_dir = root / "explainability" / "mpma_e"
    current_e = _read_explained(mpma_e_dir / "explained_unit.json")
    config = (
        current_e.get("config", {}) if isinstance(current_e.get("config"), dict) else {}
    )
    current_members = _member_ids(config.get("members"))
    final_members = [str(member["config_id"]) for member in mpma_e["members"]]
    current_aggregation = str(config.get("aggregation_strategy", ""))
    final_aggregation = str(mpma_e["aggregation_strategy"])
    if mpma_e_dir.exists() and (
        current_members != final_members or current_aggregation != final_aggregation
    ):
        shutil.rmtree(mpma_e_dir)


def explain(sweep):
    root = Path(sweep.root())
    models = build_final_models(root)
    _invalidate_stale(root, models)
    rankings_path = root / "tables" / "mpma_rankings.tsv"
    if not rankings_path.exists():
        raise FileNotFoundError("Run evaluate(sweep) before explain(sweep).")
    rankings = pd.read_csv(rankings_path, sep="\t")
    targets = _core._automatic_explainability_targets(sweep, rankings)
    mpma_e = models.get("MPMA-E")
    original_selected = _core._selected_ensemble_members
    original_aggregate = _core._aggregate_member_proba
    if isinstance(mpma_e, dict):
        member_ids = [str(member["config_id"]) for member in mpma_e["members"]]
        aggregation = str(mpma_e["aggregation_strategy"])
        weights = (
            [float(member["weight"]) for member in mpma_e["members"]]
            if aggregation == "weighted_mean_proba"
            else None
        )

        def selected_members(_root):
            return member_ids, aggregation

        def aggregate(stack, method):
            active_weights = weights if str(method) == "weighted_mean_proba" else None
            return aggregate_member_predictions(stack, str(method), active_weights)

        _core._selected_ensemble_members = selected_members
        _core._aggregate_member_proba = aggregate
    outputs = {}
    try:
        for target in targets:
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
                if not isinstance(mpma_e, dict):
                    raise _core.ExplainabilityConfigurationError(
                        "MPMA-E explainability was requested, but no final MPMA-E specification is available."
                    )
                out = _core._explain_one(
                    sweep,
                    target_override="mpma_e",
                    output_slug_override="mpma_e",
                    display_label_override="MPMA-E",
                )
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
    finally:
        _core._selected_ensemble_members = original_selected
        _core._aggregate_member_proba = original_aggregate
    return outputs
