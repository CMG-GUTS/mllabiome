from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .final_models import build_final_models, fixed_strategy_predictions
from .metrics import _renormalize_proba, compute_metrics
from .utils import dump_json_standard

DISPLAY_METRICS = ("AUC", "PR_AUC", "nMCC", "F1w", "Precision", "Recall")
METRIC_LABELS = {
    "AUC": "ROC-AUC",
    "PR_AUC": "PR-AUC (AP)",
    "nMCC": "nMCC",
    "F1w": "F1w",
    "Precision": "Precision",
    "Recall": "Recall",
}
_LODO_PROTOCOLS = {"lodo", "leave_one_dataset_out"}
_NESTED_PROTOCOLS = {"repeated_nested_cv", "nested_cv"}


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _parse_member_values(value: Any) -> list[Any]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("members", [])
    if not isinstance(parsed, (list, tuple)):
        return []
    return list(parsed)


def _positive_weight_count(weights: Any) -> int:
    values = _parse_member_values(weights)
    numeric = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numeric.append(number)
    if not numeric:
        return 0
    return int(sum(value > 1e-12 for value in numeric))


def _candidate_from_selected_unit(root: Path) -> dict[str, Any]:
    selected = _read_json(root / "ensembling" / "selected_unit.json")
    for key in ("inner_val_best_mpmas_ensemble", "inner_val_best_ensemble", "MPMA-E"):
        value = selected.get(key)
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _candidate_from_final_models(root: Path) -> dict[str, Any]:
    models = _read_json(root / "final_models.json")
    value = models.get("MPMA-E")
    if not isinstance(value, dict) or not value:
        return {}
    out = dict(value)
    members = value.get("members")
    if isinstance(members, list):
        ids = []
        weights = []
        for member in members:
            if isinstance(member, dict):
                config_id = member.get("config_id")
                if config_id is not None and str(config_id).strip():
                    ids.append(str(config_id).strip())
                if "weight" in member:
                    weights.append(member.get("weight"))
            elif member is not None and str(member).strip():
                ids.append(str(member).strip())
        if ids:
            out["members"] = ids
        if weights and len(weights) == len(ids):
            out["weights"] = weights
    return out


def _candidate_from_table(root: Path) -> dict[str, Any]:
    table = _read_tsv(root / "ensembling" / "ensemble_candidate_scores.tsv")
    if table.empty:
        return {}
    if "inner_score" in table.columns:
        score = pd.to_numeric(table["inner_score"], errors="coerce")
        valid = table[np.isfinite(score.to_numpy(dtype=float))].copy()
        if not valid.empty:
            metric = ""
            selected = _candidate_from_selected_unit(root)
            if selected:
                metric = str(selected.get("selection_metric", "")).strip().lower()
            ascending = metric in {"log_loss", "logloss", "brier", "brier_multiclass"}
            valid["_score"] = pd.to_numeric(valid["inner_score"], errors="coerce")
            table = valid.sort_values("_score", ascending=ascending, kind="mergesort")
    return table.iloc[0].to_dict() if not table.empty else {}


def _candidate_from_outer_selection(root: Path) -> dict[str, Any]:
    table = _read_tsv(root / "ensembling" / "mpma_e_outer_selection.tsv")
    if table.empty:
        return {}
    row = table.iloc[0].to_dict()
    if "selection_metric" not in row and "selection_metric" in table.columns:
        values = table["selection_metric"].dropna().astype(str)
        if not values.empty:
            row["selection_metric"] = values.iloc[0]
    return row


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip() and value.strip().lower() != "nan":
                return value
            continue
        return value
    return None


def _normalise_mpma_e_final_candidate(root: Path) -> dict[str, Any]:
    selected = _candidate_from_selected_unit(root)
    existing = _read_json(root / "ensembling" / "mpma_e_final_candidate.json")
    models = _candidate_from_final_models(root)
    table = _candidate_from_table(root)
    outer = _candidate_from_outer_selection(root)
    sources = (selected, existing, models, table, outer)
    if not any(source for source in sources):
        return {}
    result: dict[str, Any] = {}
    for source in reversed(sources):
        if isinstance(source, dict):
            result.update(source)
    selection_strategy = _first_nonempty(
        selected.get("selection_strategy"),
        existing.get("selection_strategy"),
        models.get("selection_strategy"),
        table.get("selection_strategy"),
        outer.get("selection_strategy"),
        selected.get("method"),
        existing.get("method"),
        table.get("method"),
    )
    aggregation_strategy = _first_nonempty(
        selected.get("aggregation_strategy"),
        existing.get("aggregation_strategy"),
        models.get("aggregation_strategy"),
        table.get("aggregation_strategy"),
        outer.get("aggregation_strategy"),
    )
    selection_metric = _first_nonempty(
        selected.get("selection_metric"),
        existing.get("selection_metric"),
        table.get("selection_metric"),
        outer.get("selection_metric"),
        selected.get("optimize_metric"),
        existing.get("optimize_metric"),
        models.get("selection_metric"),
    )
    members = _parse_member_values(
        _first_nonempty(
            selected.get("members"),
            existing.get("members"),
            models.get("members"),
            table.get("members"),
            outer.get("members"),
        )
    )
    weights = _parse_member_values(
        _first_nonempty(
            selected.get("weights"),
            existing.get("weights"),
            models.get("weights"),
            table.get("weights"),
            outer.get("weights"),
        )
    )
    effective_member_count = 0
    for source in (selected, existing, table, outer):
        value = (
            source.get("effective_member_count") if isinstance(source, dict) else None
        )
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            effective_member_count = count
            break
    if effective_member_count == 0 and weights:
        effective_member_count = _positive_weight_count(weights)
    if effective_member_count == 0:
        for source in (selected, existing, table, outer):
            value = source.get("member_count") if isinstance(source, dict) else None
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                effective_member_count = count
                break
    if effective_member_count == 0 and members:
        effective_member_count = len(members)
    if selection_strategy is not None:
        result["selection_strategy"] = str(selection_strategy)
    if aggregation_strategy is not None:
        result["aggregation_strategy"] = str(aggregation_strategy)
    if selection_metric is not None:
        result["selection_metric"] = str(selection_metric)
        result["optimize_metric"] = str(selection_metric)
    if members:
        result["members"] = members
        result["member_count"] = len(members)
    if weights:
        result["weights"] = weights
    if effective_member_count > 0:
        result["effective_member_count"] = int(effective_member_count)
        result["ensemble_size"] = int(effective_member_count)
    path = root / "ensembling" / "mpma_e_final_candidate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_json_standard(result, path)
    return result


def _ensure_outer_split_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outer_split_key" not in out.columns and "split_key" in out.columns:
        out["outer_split_key"] = out["split_key"]
    if "outer_split_key" in out.columns:
        out["outer_split_key"] = out["outer_split_key"].astype(str)
    return out


def _strategy_prediction_frames(
    root: Path, strategy_rows: list[dict[str, Any]]
) -> dict[str, pd.DataFrame]:
    rows = {
        str(row.get("Strategy", "")).strip(): row
        for row in strategy_rows
        if str(row.get("Strategy", "")).strip()
    }
    frames: dict[str, pd.DataFrame] = {}
    if "MPMA-B" in rows or "MPMA-E" in rows:
        build_final_models(root)
    outer = _ensure_outer_split_key(
        _read_tsv(root / "predictions" / "outer_predictions.tsv")
    )
    if "MPMA-B" in rows:
        frame = _ensure_outer_split_key(fixed_strategy_predictions(root, "MPMA-B"))
        if not frame.empty:
            frames["MPMA-B"] = frame
    if "MPMA-E" in rows:
        try:
            frame = _ensure_outer_split_key(fixed_strategy_predictions(root, "MPMA-E"))
            if not frame.empty:
                frames["MPMA-E"] = frame
        except ValueError as exc:
            if "No final MPMA-E specification" not in str(exc):
                raise
    if not outer.empty and "config_id" in outer.columns:
        outer["config_id"] = outer["config_id"].astype(str)
        for strategy in ("AutoML", "Baseline RF", "SIAMCAT"):
            row = rows.get(strategy)
            if row is None:
                continue
            config_id = str(row.get("config_id", "")).strip()
            if not config_id:
                continue
            frame = outer[outer["config_id"].eq(config_id)].copy()
            if not frame.empty:
                frames[strategy] = frame
    return frames


def _metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "outer_split_key" not in predictions.columns:
        return pd.DataFrame()
    pcols = [column for column in predictions.columns if column.startswith("proba_")]
    if not pcols or "y_true" not in predictions.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    classes = np.arange(len(pcols), dtype=int)
    for outer_split_key, group in predictions.groupby("outer_split_key", sort=True):
        y_true = pd.to_numeric(group["y_true"], errors="coerce")
        valid = y_true.notna()
        if not bool(valid.any()):
            continue
        y_true_array = y_true.loc[valid].astype(int).to_numpy()
        proba = (
            group.loc[valid, pcols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        if not np.isfinite(proba).all():
            continue
        proba = _renormalize_proba(proba, len(classes))
        if "y_pred" in group.columns:
            y_pred = pd.to_numeric(group.loc[valid, "y_pred"], errors="coerce")
            if y_pred.isna().any():
                y_pred_array = classes[np.argmax(proba, axis=1)]
            else:
                y_pred_array = y_pred.astype(int).to_numpy()
        else:
            y_pred_array = classes[np.argmax(proba, axis=1)]
        metrics = compute_metrics(y_true_array, y_pred_array, proba, classes)
        rows.append(
            {
                "outer_split_key": str(outer_split_key),
                "n_samples": int(len(y_true_array)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _repeat_id(split_key: str) -> str:
    match = re.match(r"^r(\d+)_o\d+$", str(split_key))
    return f"r{match.group(1)}" if match else "r0"


def _bootstrap_unit_mean(
    values: pd.DataFrame,
    metric: str,
    protocol: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    frame = values[["outer_split_key", metric]].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[np.isfinite(frame[metric].to_numpy(dtype=float))]
    if frame.empty:
        return np.asarray([], dtype=float)
    out = np.empty(int(n_bootstrap), dtype=float)
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        values_array = frame[metric].to_numpy(dtype=float)
        n = len(values_array)
        indices = rng.integers(0, n, size=(int(n_bootstrap), n))
        return values_array[indices].mean(axis=1)
    frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
    groups = [
        group[metric].to_numpy(dtype=float)
        for _, group in frame.groupby("repeat", sort=True)
    ]
    if not groups:
        return np.asarray([], dtype=float)
    n_repeats = len(groups)
    for index in range(int(n_bootstrap)):
        sampled_repeat_indices = rng.integers(0, n_repeats, size=n_repeats)
        sampled: list[float] = []
        for repeat_index in sampled_repeat_indices:
            values_array = groups[int(repeat_index)]
            sampled.extend(
                values_array[
                    rng.integers(0, len(values_array), size=len(values_array))
                ].tolist()
            )
        out[index] = float(np.mean(sampled))
    return out


def _summary_rows(
    unit_metrics: pd.DataFrame,
    protocol: str,
    n_bootstrap: int,
    random_state: int,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, group in unit_metrics.groupby("Strategy", sort=False):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if values.empty:
                continue
            seed = _stable_seed(random_state, "summary", strategy, metric)
            boot = _bootstrap_unit_mean(
                group,
                metric,
                protocol,
                n_bootstrap,
                np.random.default_rng(seed),
            )
            ci_low = ci_high = np.nan
            if len(boot):
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "Strategy": str(strategy),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "estimate": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                    "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    "n_outer_units": int(len(values)),
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_method": (
                        "cluster_outer_cohort"
                        if str(protocol).lower() in _LODO_PROTOCOLS
                        else "hierarchical_repeat_outer_fold"
                    ),
                    "valid_bootstrap": int(np.isfinite(boot).sum()),
                }
            )
    return pd.DataFrame(rows)


def _paired_bootstrap_difference(
    merged: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    protocol: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    frame = merged[["outer_split_key", metric_a, metric_b]].copy()
    frame[metric_a] = pd.to_numeric(frame[metric_a], errors="coerce")
    frame[metric_b] = pd.to_numeric(frame[metric_b], errors="coerce")
    frame = frame[
        np.isfinite(frame[metric_a].to_numpy(dtype=float))
        & np.isfinite(frame[metric_b].to_numpy(dtype=float))
    ]
    if frame.empty:
        return float("nan"), float("nan")
    frame["diff"] = frame[metric_a] - frame[metric_b]
    protocol_key = str(protocol).lower()
    if protocol_key in _LODO_PROTOCOLS:
        differences = frame["diff"].to_numpy(dtype=float)
        n = len(differences)
        indices = rng.integers(0, n, size=(int(n_bootstrap), n))
        boot = differences[indices].mean(axis=1)
    else:
        frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
        groups = [
            group["diff"].to_numpy(dtype=float)
            for _, group in frame.groupby("repeat", sort=True)
        ]
        boot = np.empty(int(n_bootstrap), dtype=float)
        n_repeats = len(groups)
        for index in range(int(n_bootstrap)):
            sampled_repeat_indices = rng.integers(0, n_repeats, size=n_repeats)
            sampled: list[float] = []
            for repeat_index in sampled_repeat_indices:
                values_array = groups[int(repeat_index)]
                sampled.extend(
                    values_array[
                        rng.integers(0, len(values_array), size=len(values_array))
                    ].tolist()
                )
            boot[index] = float(np.mean(sampled))
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _exact_sign_flip_test(
    differences: np.ndarray, random_state: int
) -> tuple[float, float, str]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), "paired_sign_flip"
    observed = float(np.mean(values))
    if n <= 16:
        total = 1 << n
        extreme = 0
        for bits in range(total):
            signs = np.fromiter(
                (1.0 if bits & (1 << index) else -1.0 for index in range(n)),
                dtype=float,
                count=n,
            )
            if abs(float(np.mean(signs * values))) >= abs(observed) - 1e-15:
                extreme += 1
        return observed, float(extreme / total), "exact_paired_sign_flip"
    rng = np.random.default_rng(int(random_state))
    n_permutations = 50000
    extreme = 0
    batch_size = 2000
    completed = 0
    while completed < n_permutations:
        current = min(batch_size, n_permutations - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(current, n))
        permuted = np.mean(signs * values[None, :], axis=1)
        extreme += int(np.sum(np.abs(permuted) >= abs(observed) - 1e-15))
        completed += current
    return (
        observed,
        float((extreme + 1) / (n_permutations + 1)),
        "monte_carlo_paired_sign_flip",
    )


def _corrected_resampled_t_test(
    merged: pd.DataFrame, metric_a: str, metric_b: str
) -> tuple[float, float, str]:
    frame = merged[["outer_split_key", metric_a, metric_b]].copy()
    frame[metric_a] = pd.to_numeric(frame[metric_a], errors="coerce")
    frame[metric_b] = pd.to_numeric(frame[metric_b], errors="coerce")
    frame = frame[
        np.isfinite(frame[metric_a].to_numpy(dtype=float))
        & np.isfinite(frame[metric_b].to_numpy(dtype=float))
    ]
    if len(frame) < 2:
        return float("nan"), float("nan"), "nadeau_bengio_corrected_t"
    differences = (frame[metric_a] - frame[metric_b]).to_numpy(dtype=float)
    mean_difference = float(np.mean(differences))
    variance = float(np.var(differences, ddof=1))
    frame["repeat"] = frame["outer_split_key"].map(_repeat_id)
    fold_counts = (
        frame.groupby("repeat")["outer_split_key"].nunique().to_numpy(dtype=float)
    )
    k = int(round(float(np.median(fold_counts)))) if len(fold_counts) else len(frame)
    if k < 2:
        return mean_difference, float("nan"), "nadeau_bengio_corrected_t"
    test_train_ratio = 1.0 / float(k - 1)
    corrected_variance = (1.0 / len(differences) + test_train_ratio) * variance
    if corrected_variance <= 0:
        p_value = 1.0 if abs(mean_difference) <= 1e-15 else 0.0
        return mean_difference, p_value, "nadeau_bengio_corrected_t"
    statistic = mean_difference / math.sqrt(corrected_variance)
    p_value = float(2.0 * student_t.sf(abs(statistic), df=len(differences) - 1))
    return float(statistic), p_value, "nadeau_bengio_corrected_t"


def _holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["p_holm"] = np.nan
    for metric, group in out.groupby("metric", sort=False):
        indices = group.index[
            np.isfinite(
                pd.to_numeric(group["p_value"], errors="coerce").to_numpy(dtype=float)
            )
        ]
        if len(indices) == 0:
            continue
        ordered = sorted(indices, key=lambda index: float(out.loc[index, "p_value"]))
        m = len(ordered)
        running = 0.0
        for rank, index in enumerate(ordered):
            adjusted = min(1.0, (m - rank) * float(out.loc[index, "p_value"]))
            running = max(running, adjusted)
            out.loc[index, "p_holm"] = running
    out["significant_holm_0_05"] = pd.to_numeric(out["p_holm"], errors="coerce").lt(
        0.05
    )
    return out


def _pairwise_rows(
    unit_metrics: pd.DataFrame,
    protocol: str,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    if unit_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    strategies = list(dict.fromkeys(unit_metrics["Strategy"].astype(str).tolist()))
    for strategy_a, strategy_b in itertools.combinations(strategies, 2):
        left = unit_metrics[unit_metrics["Strategy"].eq(strategy_a)].copy()
        right = unit_metrics[unit_metrics["Strategy"].eq(strategy_b)].copy()
        merged = left.merge(
            right,
            on="outer_split_key",
            suffixes=("_a", "_b"),
            how="inner",
        )
        if merged.empty:
            continue
        for metric in DISPLAY_METRICS:
            metric_a = f"{metric}_a"
            metric_b = f"{metric}_b"
            if metric_a not in merged.columns or metric_b not in merged.columns:
                continue
            valid = merged[
                np.isfinite(
                    pd.to_numeric(merged[metric_a], errors="coerce").to_numpy(
                        dtype=float
                    )
                )
                & np.isfinite(
                    pd.to_numeric(merged[metric_b], errors="coerce").to_numpy(
                        dtype=float
                    )
                )
            ].copy()
            if valid.empty:
                continue
            estimate_a = float(pd.to_numeric(valid[metric_a], errors="coerce").mean())
            estimate_b = float(pd.to_numeric(valid[metric_b], errors="coerce").mean())
            seed = _stable_seed(
                random_state,
                "pairwise",
                strategy_a,
                strategy_b,
                metric,
            )
            ci_low, ci_high = _paired_bootstrap_difference(
                valid,
                metric_a,
                metric_b,
                protocol,
                n_bootstrap,
                np.random.default_rng(seed),
            )
            if str(protocol).lower() in _LODO_PROTOCOLS:
                statistic, p_value, test = _exact_sign_flip_test(
                    (
                        pd.to_numeric(valid[metric_a], errors="coerce")
                        - pd.to_numeric(valid[metric_b], errors="coerce")
                    ).to_numpy(dtype=float),
                    seed,
                )
            else:
                statistic, p_value, test = _corrected_resampled_t_test(
                    valid,
                    metric_a,
                    metric_b,
                )
            rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "estimate_a": estimate_a,
                    "estimate_b": estimate_b,
                    "difference_a_minus_b": estimate_a - estimate_b,
                    "difference_ci_low": ci_low,
                    "difference_ci_high": ci_high,
                    "test": test,
                    "test_statistic": statistic,
                    "p_value": p_value,
                    "n_paired_outer_units": int(len(valid)),
                }
            )
    return _holm_adjust(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _stable_seed(random_state: int, *parts: Any) -> int:
    text = "|".join(str(value) for value in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little", signed=False)
    return int((int(random_state) + offset) % (2**32 - 1))


def _selection_metric_from_run(
    root: Path,
    strategy_rows: list[dict[str, Any]],
    explicit: str | None,
) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    manifest = _read_json(root / "manifest.json")
    sweep = manifest.get("sweep", {})
    if isinstance(sweep, dict):
        evaluation = sweep.get("evaluation", {})
        if isinstance(evaluation, dict):
            metric = str(evaluation.get("optimize_metric", "")).strip()
            if metric:
                return metric
    for row in strategy_rows:
        if str(row.get("Strategy", "")).strip() == "MPMA-B":
            metric = str(
                row.get("selection_metric", row.get("optimize_metric", ""))
            ).strip()
            if metric:
                return metric
    return "nMCC"


def _file_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _statistics_fingerprint(
    root: Path,
    protocol: str,
    strategy_rows: list[dict[str, Any]],
    n_bootstrap: int,
    random_state: int,
    selection_metric: str,
) -> str:
    relevant_rows = []
    for row in strategy_rows:
        relevant_rows.append(
            {
                key: row.get(key)
                for key in (
                    "Strategy",
                    "source",
                    "config_id",
                    "ensemble_config_id",
                    "selection_strategy",
                    "aggregation_strategy",
                    "members",
                    "weights",
                )
                if key in row
            }
        )
    paths = [
        root / "predictions" / "outer_predictions.tsv",
        root / "configs.tsv",
        root / "inner_results" / "inner_results.tsv",
        root / "tables" / "mpma_b_final_candidate.json",
        root / "ensembling" / "selected_unit.json",
        root / "ensembling" / "mpma_e_final_candidate.json",
    ]
    payload = {
        "protocol": str(protocol),
        "strategies": relevant_rows,
        "n_bootstrap": int(n_bootstrap),
        "random_state": int(random_state),
        "selection_metric": str(selection_metric),
        "display_metrics": list(DISPLAY_METRICS),
        "files": [_file_signature(path) for path in paths],
        "schema_version": 2,
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cached_result(
    root: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    tables = root / "report" / "tables"
    manifest_path = tables / "strategy_statistics_manifest.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("fingerprint", "")) != fingerprint:
        return None
    unit_path = tables / "strategy_outer_unit_metrics.tsv"
    summary_path = tables / "strategy_metrics_bootstrap.tsv"
    pairwise_path = tables / "strategy_pairwise_tests.tsv"
    if (
        not unit_path.exists()
        or not summary_path.exists()
        or not pairwise_path.exists()
    ):
        return None
    return {
        "unit_metrics": _read_tsv(unit_path),
        "summary": _read_tsv(summary_path),
        "pairwise": _read_tsv(pairwise_path),
        "unit_metrics_path": unit_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "manifest_path": manifest_path,
        "selection_metric": str(manifest.get("selection_metric", "nMCC")),
        "cache_hit": True,
    }


def run_report_statistics(
    root: Path | str,
    protocol: str,
    strategy_rows: list[dict[str, Any]],
    n_bootstrap: int = 2000,
    random_state: int = 42,
    *,
    selection_metric: str | None = None,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    root = Path(root)
    _normalise_mpma_e_final_candidate(root)
    resolved_selection = _selection_metric_from_run(
        root, strategy_rows, selection_metric
    )
    fingerprint = _statistics_fingerprint(
        root,
        protocol,
        strategy_rows,
        n_bootstrap,
        random_state,
        resolved_selection,
    )
    cached = _cached_result(root, fingerprint)
    if cached is not None:
        return cached
    frames = _strategy_prediction_frames(root, strategy_rows)
    unit_frames: list[pd.DataFrame] = []
    for strategy, predictions in frames.items():
        metrics = _metric_frame(predictions)
        if metrics.empty:
            continue
        metrics.insert(0, "Strategy", strategy)
        unit_frames.append(metrics)
    unit_metrics = (
        pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    )
    summary_metrics = tuple(
        metric
        for metric in DISPLAY_METRICS
        if unit_metrics.empty or metric in unit_metrics.columns
    )
    summary = (
        _summary_rows(
            unit_metrics,
            protocol,
            int(n_bootstrap),
            int(random_state),
            summary_metrics,
        )
        if not unit_metrics.empty
        else pd.DataFrame()
    )
    pairwise = (
        _pairwise_rows(
            unit_metrics,
            protocol,
            int(n_bootstrap),
            int(random_state),
        )
        if not unit_metrics.empty
        else pd.DataFrame()
    )
    tables = root / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    unit_path = tables / "strategy_outer_unit_metrics.tsv"
    summary_path = tables / "strategy_metrics_bootstrap.tsv"
    pairwise_path = tables / "strategy_pairwise_tests.tsv"
    manifest_path = tables / "strategy_statistics_manifest.json"
    unit_metrics.to_csv(unit_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    pairwise.to_csv(pairwise_path, sep="\t", index=False)
    stale = (
        "strategy_oof_performance.tsv",
        "strategy_oof_performance_table.tsv",
        "strategy_oof_calibration.tsv",
        "strategy_oof_pairwise_contrasts.tsv",
        "strategy_oof_statistics_manifest.json",
        "strategy_oof_coverage.tsv",
        "strategy_oof_pairwise_coverage.tsv",
    )
    for name in stale:
        path = tables / name
        if path.exists():
            path.unlink()
    manifest = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "protocol": str(protocol),
        "strategies": list(frames),
        "display_metrics_bootstrapped": list(summary_metrics),
        "pairwise_metrics": list(DISPLAY_METRICS),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": 0.95,
        "nested_cv_bootstrap": "hierarchical repeat/outer-fold bootstrap of held-out displayed-strategy metrics",
        "lodo_bootstrap": "cluster bootstrap of held-out cohorts for displayed-strategy metrics",
        "nested_cv_pairwise_test": "Nadeau-Bengio corrected resampled paired t-test",
        "lodo_pairwise_test": "paired sign-flip randomization test at held-out cohort level",
        "multiple_testing": "Holm adjustment across displayed strategy pairs within each displayed metric",
        "scope": "displayed report strategies only",
        "selection_metric": resolved_selection,
        "cache": "statistics are reused when source artefacts and statistical settings are unchanged",
    }
    dump_json_standard(manifest, manifest_path)
    return {
        "unit_metrics": unit_metrics,
        "summary": summary,
        "pairwise": pairwise,
        "unit_metrics_path": unit_path,
        "summary_path": summary_path,
        "pairwise_path": pairwise_path,
        "manifest_path": manifest_path,
        "selection_metric": resolved_selection,
        "cache_hit": False,
    }
