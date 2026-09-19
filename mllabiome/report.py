from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .configs_sweep import Sweep
from .console import console, path_table, phase_progress, stage, success
from .utils import dump_json_standard
from .storage import read_table, write_table, table_exists, glob_tables
from .metrics import compute_metrics
from .report_statistics import run_report_statistics
from .report_compute import run_compute_accounting
from .explainability_visuals import plot_feature_support

_METRICS = [
    ("AUC", "ROC-AUC"),
    ("PR_AUC", "PR-AUC (AP)"),
    ("nMCC", "nMCC"),
    ("F1w", "F1$_{w}$"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
]

_BASELINE_RANK_PRIORITY = ("strain", "species", "genus")

_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect x="1" y="1" width="62" height="62" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/><text x="32" y="39" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="22" font-weight="700" fill="#0f172a">mll</text></svg>"""


def _favicon_href() -> str:
    payload = base64.b64encode(_FAVICON_SVG.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{payload}"


def _single_rank_label(row: pd.Series | dict[str, Any]) -> str | None:
    res = str(row.get("resolution", "")).strip().lower()
    levels = str(row.get("levels", "")).strip().lower().replace(" ", "")
    for rank in _BASELINE_RANK_PRIORITY:
        if res == rank or levels == rank:
            return rank
    return None


def _pick_deepest_single_rank(sub: pd.DataFrame, *, sort_col: str) -> dict[str, Any]:
    if sub.empty:
        return {}
    candidates = []
    for rank in _BASELINE_RANK_PRIORITY:
        rr = sub[
            sub.apply(lambda r, rank=rank: _single_rank_label(r) == rank, axis=1)
        ].copy()
        if rr.empty:
            continue
        if sort_col in rr.columns:
            rr = rr.sort_values(sort_col, ascending=False)
        candidates.append(rr.iloc[0].to_dict())
        break
    if candidates:
        return candidates[0]
    return {}


def _asset_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".pdf": "application/pdf"}.get(
        suffix, "application/octet-stream"
    )
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _inline_svg(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find("<svg")
    end = text.rfind("</svg>")
    if start < 0 or end < 0:
        return ""
    return text[start : end + 6]


def _fig(stem_or_path: Path, report_dir: Path, caption: str = "") -> str:
    path = stem_or_path
    if path.suffix.lower() not in {".svg", ".png", ".pdf"}:
        path = stem_or_path.with_suffix(".svg")
    if not path.exists():
        png = path.with_suffix(".png")
        if png.exists():
            path = png
        else:
            return ""
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    alt = html.escape(caption or path.stem)
    if path.suffix.lower() == ".svg":
        svg = _inline_svg(path)
        if not svg:
            return ""
        return f'<figure><div class="embedded-svg" role="img" aria-label="{alt}">{svg}</div>{cap}</figure>'
    uri = _asset_uri(path)
    if path.suffix.lower() == ".png":
        return f'<figure><img src="{uri}" alt="{alt}" loading="lazy">{cap}</figure>'
    return f'<figure><p><a href="{uri}">{alt}</a></p>{cap}</figure>'


def _read_table(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _format_p_value(value: Any) -> str:
    v = _safe_float(value)
    if not np.isfinite(v):
        return ""
    if 0 <= v < 0.001:
        return "<0.001"
    return f"{v:.3f}"


def _html_number(value: Any) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.3f}"
    return None


def _pct_cell(
    mean: Any, std: Any, *, bold: bool = False, html_mode: bool = False
) -> str:
    m = _safe_float(mean)
    s = _safe_float(std)
    if not np.isfinite(m):
        body = "—" if html_mode else r"--"
        return body

    body = f"{100 * m:.3f}"
    if np.isfinite(s):
        sd = f"{100 * s:.3f}" if html_mode else f"{100 * s:06.3f}"
        body += (" ± " if html_mode else r"$\pm$") + sd
    if bold:
        return f"<strong>{body}</strong>" if html_mode else rf"\textbf{{{body}}}"
    return body


def _pct_ci_cell(
    estimate: Any,
    std: Any,
    low: Any,
    high: Any,
    *,
    bold: bool = False,
    html_mode: bool = False,
) -> str:
    e = _safe_float(estimate)
    sd = _safe_float(std)
    lo = _safe_float(low)
    hi = _safe_float(high)
    if not np.isfinite(e):
        return "—" if html_mode else r"--"
    body = f"{100 * e:.3f}"
    if np.isfinite(sd):
        sd_text = f"{100 * sd:.3f}" if html_mode else f"{100 * sd:06.3f}"
        body += (" ± " if html_mode else r"$\pm$") + sd_text
    if np.isfinite(lo) and np.isfinite(hi):
        body += f" [{100 * lo:.3f}, {100 * hi:.3f}]"
    if bold:
        return f"<strong>{body}</strong>" if html_mode else rf"\textbf{{{body}}}"
    return body


def _strategy_statistics_summary(root: Path) -> pd.DataFrame:
    return _read_table(
        root / "report" / "tables" / "strategy_metrics_bootstrap.parquet"
    )


def _mean_std_from_cols(
    row: pd.Series | dict[str, Any], prefix: str, metric: str
) -> tuple[float, float]:
    if isinstance(row, dict):
        get = row.get
    else:
        get = row.get
    mean = get(f"{prefix}_{metric}_mean", np.nan)
    std = get(f"{prefix}_{metric}_std", np.nan)
    if not np.isfinite(_safe_float(mean)):
        mean = get(f"{metric}_mean", np.nan)
        std = get(f"{metric}_std", std)
    return _safe_float(mean), _safe_float(std)


def _agg_metrics(path: Path, prefix: str) -> pd.DataFrame:
    df = _read_table(path)
    if df.empty or "config_id" not in df.columns:
        return pd.DataFrame()
    if "ok" in df.columns:
        df = df[pd.to_numeric(df["ok"], errors="coerce").fillna(0).eq(1)]
    if df.empty:
        return pd.DataFrame()
    id_cols = [
        c
        for c in [
            "config_id",
            "count_transformation",
            "transformation_abbreviation",
            "resolution",
            "learner",
        ]
        if c in df.columns
    ]
    metric_cols = [c for c, _ in _METRICS if c in df.columns]
    if not metric_cols:
        return df[id_cols].drop_duplicates("config_id") if id_cols else pd.DataFrame()
    g = df.groupby(id_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"])
    g.columns = [f"{prefix}_{m}_{stat}" for m, stat in g.columns]
    return g.reset_index()


def _top_mpma_raw(root: Path, n: int = 5) -> pd.DataFrame:
    inner = _agg_metrics(root / "inner_results" / "inner_results.parquet", "inner")
    outer = _agg_metrics(root / "results" / "outer_results.parquet", "outer")
    if inner.empty and outer.empty:
        return pd.DataFrame()
    if not inner.empty:
        base = inner
        if not outer.empty:
            outer_metrics = outer[
                [c for c in outer.columns if c == "config_id" or c.startswith("outer_")]
            ].copy()
            base = base.merge(outer_metrics, on="config_id", how="left")
    else:
        base = outer
    sort_col = (
        "inner_nMCC_mean"
        if "inner_nMCC_mean" in base.columns
        else ("outer_nMCC_mean" if "outer_nMCC_mean" in base.columns else None)
    )
    if sort_col:
        base = base.sort_values(sort_col, ascending=False)
    base = base.head(int(n)).copy()
    base.insert(0, "rank", np.arange(1, len(base) + 1))
    return base


def _top_mpma_display(df: pd.DataFrame, *, html_mode: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        row = {
            "Rank": int(r.get("rank", len(rows) + 1)),
            "Resolution": str(r.get("resolution", "")),
            "Count transformation": str(
                r.get("transformation_abbreviation", r.get("count_transformation", ""))
            ),
            "Learner": str(r.get("learner", "")),
        }
        for metric, label_latex, label_html in [
            ("nMCC", "nMCC", "nMCC"),
            ("AUC", "ROC-AUC", "ROC-AUC"),
            ("F1w", "F1$_w$", "F1w"),
        ]:
            label = label_html if html_mode else label_latex
            row[f"Inner {label}"] = _pct_cell(
                *_mean_std_from_cols(r, "inner", metric), html_mode=html_mode
            )
            row[f"Outer {label}"] = _pct_cell(
                *_mean_std_from_cols(r, "outer", metric), html_mode=html_mode
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _selected_json(root: Path) -> dict[str, Any]:
    return _read_json(root / "ensembling" / "selected_unit.json")


def _best_mpma_row(root: Path) -> dict[str, Any]:
    nested = _read_json(root / "tables" / "mpma_b_strategy_summary.json")
    if isinstance(nested, dict) and nested:
        return nested
    selected = _selected_json(root)
    best = (
        selected.get("inner_val_best_mpma")
        or selected.get("inner_val_best_individual")
        or {}
    )
    return best if isinstance(best, dict) else {}


def _ensemble_outer_metrics_from_predictions(
    root: Path, ensemble_config_id: str
) -> dict[str, Any]:
    path = root / "ensembling" / "ensemble_predictions.parquet"
    if not table_exists(path) or not str(ensemble_config_id):
        return {}
    df = _read_table(path)
    if (
        df.empty
        or "ensemble_config_id" not in df.columns
        or "outer_split_key" not in df.columns
    ):
        return {}
    df = df[df["ensemble_config_id"].astype(str).eq(str(ensemble_config_id))].copy()
    if df.empty:
        return {}
    pcols = [c for c in df.columns if c.startswith("proba_")]
    if not pcols or "y_true" not in df.columns or "y_pred" not in df.columns:
        return {}
    classes = np.arange(len(pcols), dtype=int)
    fold_rows: list[dict[str, float]] = []
    for _, sub in df.groupby("outer_split_key", sort=False):
        yy = pd.to_numeric(sub["y_true"], errors="coerce")
        mask = yy.notna()
        y_true = yy.loc[mask].astype(int).to_numpy()
        y_pred = (
            pd.to_numeric(sub.loc[mask, "y_pred"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        proba = (
            sub.loc[mask, pcols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        if len(y_true) == 0 or proba.shape[0] != len(y_true):
            continue
        fold_rows.append(compute_metrics(y_true, y_pred, proba, classes))
    if not fold_rows:
        return {}
    tab = pd.DataFrame(fold_rows)
    out: dict[str, Any] = {}
    for col in tab.columns:
        vals = (
            pd.to_numeric(tab[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if vals.empty:
            continue
        out[f"outer_{col}_mean"] = float(vals.mean())
        out[f"outer_{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"outer_{col}_count"] = int(len(vals))
        out[f"{col}_mean"] = out[f"outer_{col}_mean"]
        out[f"{col}_std"] = out[f"outer_{col}_std"]
        out[f"{col}_count"] = out[f"outer_{col}_count"]
    return out


def _ensemble_row(root: Path) -> dict[str, Any]:
    nested = _read_json(root / "ensembling" / "mpma_e_strategy_summary.json")
    if isinstance(nested, dict) and nested:
        return nested
    selected = _selected_json(root)
    ens = (
        selected.get("inner_val_best_mpmas_ensemble")
        or selected.get("inner_val_best_ensemble")
        or {}
    )
    if not isinstance(ens, dict):
        return {}
    row = dict(ens)
    eid = str(row.get("ensemble_config_id", ""))
    for k, v in _ensemble_outer_metrics_from_predictions(root, eid).items():
        if k not in row or not np.isfinite(_safe_float(row.get(k))):
            row[k] = v
    return row


def _ensemble_final_candidate(root: Path) -> dict[str, Any]:
    final = _read_json(root / "ensembling" / "mpma_e_final_candidate.json")
    if isinstance(final, dict) and final:
        return final
    selected = _selected_json(root)
    ens = (
        selected.get("inner_val_best_mpmas_ensemble")
        or selected.get("inner_val_best_ensemble")
        or {}
    )
    return ens if isinstance(ens, dict) else {}


def _strategy_rows(root: Path) -> list[dict[str, Any]]:
    by_strategy: dict[str, dict[str, Any]] = {}
    ens = _ensemble_row(root)
    by_strategy["MPMA-E"] = (
        {"Strategy": "MPMA-E", "source": "ensemble", **ens}
        if ens
        else {"Strategy": "MPMA-E", "source": "ensemble"}
    )
    best = _best_mpma_row(root)
    by_strategy["MPMA-B"] = (
        {"Strategy": "MPMA-B", "source": "mpma", **best}
        if best
        else {"Strategy": "MPMA-B", "source": "mpma"}
    )

    outer = _agg_metrics(root / "results" / "outer_results.parquet", "outer")
    inner = _agg_metrics(root / "inner_results" / "inner_results.parquet", "inner")
    if not outer.empty:
        base = outer.copy()
        if not inner.empty:
            inner_cols = [
                c for c in inner.columns if c == "config_id" or c.startswith("inner_")
            ]
            base = base.merge(inner[inner_cols], on="config_id", how="left")

        def _pick(mask: pd.Series) -> dict[str, Any]:
            sub = base[mask.fillna(False)].copy()
            if sub.empty:
                return {}
            sort_col = (
                "inner_nMCC_mean"
                if "inner_nMCC_mean" in sub.columns
                else "outer_nMCC_mean"
            )
            if sort_col in sub.columns:
                sub = sub.sort_values(sort_col, ascending=False)
            return sub.iloc[0].to_dict()

        sort_col = (
            "inner_nMCC_mean"
            if "inner_nMCC_mean" in base.columns
            else "outer_nMCC_mean"
        )
        learner = base.get(
            "learner", pd.Series("", index=base.index, dtype=str)
        ).astype(str)
        transform = base.get(
            "count_transformation", pd.Series("", index=base.index, dtype=str)
        ).astype(str)
        resolution = base.get(
            "resolution", pd.Series("", index=base.index, dtype=str)
        ).astype(str)

        automl_pool = base[
            learner.str.contains("FLAML|AutoML", case=False, regex=True).fillna(False)
            & transform.str.fullmatch("relative_abundance", case=False).fillna(False)
        ].copy()

        automl = _pick_deepest_single_rank(automl_pool, sort_col=sort_col)
        if automl:
            by_strategy["AutoML"] = {"Strategy": "AutoML", "source": "mpma", **automl}

        baseline_pool = base[
            learner.str.fullmatch("RF_1000_msl5", case=False).fillna(False)
            & transform.str.fullmatch("arcsine_sqrt", case=False).fillna(False)
        ].copy()
        baseline = _pick_deepest_single_rank(baseline_pool, sort_col=sort_col)
        if baseline:
            by_strategy["Baseline RF"] = {
                "Strategy": "Baseline RF",
                "source": "mpma",
                **baseline,
            }

        siamcat = _pick(
            learner.str.fullmatch("SIAMCAT", case=False).fillna(False)
            & transform.str.fullmatch("identity", case=False).fillna(False)
            & resolution.str.fullmatch("raw", case=False).fillna(False)
        )
        if siamcat:
            by_strategy["SIAMCAT"] = {
                "Strategy": "SIAMCAT",
                "source": "mpma",
                **siamcat,
            }

    by_strategy.setdefault("AutoML", {"Strategy": "AutoML", "source": "mpma"})
    by_strategy.setdefault("Baseline RF", {"Strategy": "Baseline RF", "source": "mpma"})
    by_strategy.setdefault("SIAMCAT", {"Strategy": "SIAMCAT", "source": "mpma"})
    return [
        by_strategy[k] for k in ("MPMA-E", "MPMA-B", "AutoML", "Baseline RF", "SIAMCAT")
    ]


def _strategy_performance_display(
    root: Path, *, html_mode: bool = False
) -> pd.DataFrame:
    rows = _strategy_rows(root)
    if not rows:
        return pd.DataFrame()
    stats = _strategy_statistics_summary(root)
    if not stats.empty and {
        "Strategy",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
    }.issubset(stats.columns):
        best_by_metric: dict[str, str] = {}
        for metric, _ in _METRICS:
            sub = stats[stats["metric"].astype(str).eq(metric)].copy()
            sub["estimate"] = pd.to_numeric(sub["estimate"], errors="coerce")
            sub = sub[np.isfinite(sub["estimate"].to_numpy(dtype=float))]
            if not sub.empty:
                best_by_metric[metric] = str(
                    sub.sort_values("estimate", ascending=False).iloc[0]["Strategy"]
                )
        out = []
        for row in rows:
            strategy = str(row.get("Strategy", ""))
            rr = {"Strategy": strategy}
            for metric, label in _METRICS:
                sub = stats[
                    stats["Strategy"].astype(str).eq(strategy)
                    & stats["metric"].astype(str).eq(metric)
                ]
                key = label.replace("$", "").replace("_{w}", "w")
                if sub.empty:
                    rr[key] = "—" if html_mode else r"--"
                    continue
                r = sub.iloc[0]
                rr[key] = _pct_ci_cell(
                    r.get("estimate"),
                    r.get("std"),
                    r.get("ci_low"),
                    r.get("ci_high"),
                    bold=(best_by_metric.get(metric) == strategy),
                    html_mode=html_mode,
                )
            out.append(rr)
        return pd.DataFrame(out)
    best_by_metric: dict[str, str] = {}
    for metric, _ in _METRICS:
        vals = []
        for row in rows:
            m, _ = _mean_std_from_cols(row, "outer", metric)
            if np.isfinite(m):
                vals.append((m, str(row.get("Strategy", ""))))
        if vals:
            best_by_metric[metric] = max(vals, key=lambda x: x[0])[1]
    out = []
    for row in rows:
        strategy = str(row.get("Strategy", ""))
        rr = {"Strategy": strategy}
        for metric, label in _METRICS:
            rr[label.replace("$", "").replace("_{w}", "w")] = _pct_cell(
                *_mean_std_from_cols(row, "outer", metric),
                bold=(best_by_metric.get(metric) == strategy),
                html_mode=html_mode,
            )
        out.append(rr)
    return pd.DataFrame(out)


def _ensemble_summary_table(root: Path) -> pd.DataFrame:
    ens = _ensemble_final_candidate(root)
    if not ens:
        ens = _ensemble_row(root)
    if not ens:
        return pd.DataFrame()

    members = ens.get("effective_member_count")
    if members is None:
        members = ens.get("member_count")
    if members is None:
        raw_members = ens.get("members")
        if isinstance(raw_members, str):
            try:
                raw_members = json.loads(raw_members)
            except Exception:
                raw_members = None
        if isinstance(raw_members, list):
            members = len(raw_members)
        else:
            members = ens.get("ensemble_size", "")

    max_size = ens.get("max_size")
    if max_size is None:
        max_size = ens.get("ensemble_size", members)

    row = {
        "Strategy": "MPMA-E",
        "Selection": ens.get("selection_strategy", ""),
        "Aggregation": ens.get("aggregation_strategy", ""),
        "Max size": max_size,
        "Members": members,
        "Selection metric": ens.get(
            "selection_metric",
            ens.get("optimize_metric", ""),
        ),
    }

    return pd.DataFrame([row])


def _ensemble_members_table(root: Path) -> pd.DataFrame:
    ens = _ensemble_final_candidate(root)
    members = ens.get("members", []) if isinstance(ens, dict) else []
    if isinstance(members, str):
        try:
            members = json.loads(members)
        except Exception:
            members = []
    members = [str(x) for x in members] if isinstance(members, list) else []
    weights = ens.get("weights", []) if isinstance(ens, dict) else []
    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
        except Exception:
            weights = []
    weights = list(weights) if isinstance(weights, list) else []

    p = root / "tables" / "mpma_e_members.parquet"
    if not table_exists(p):
        p = root / "figures" / "mpma_e_members.parquet"
    if table_exists(p):
        existing = read_table(p)
        informative = {
            "config_id",
            "Config ID",
            "resolution",
            "Representation",
            "learner",
            "Learner",
            "modalities",
            "Modalities",
            "integration",
            "Integration",
        }
        if any(c in existing.columns for c in informative):
            return existing.copy()

    configs_path = root / "configs.parquet"
    configs = (
        read_table(configs_path, dtype=str)
        if table_exists(configs_path)
        else pd.DataFrame()
    )
    if not members:
        return pd.DataFrame()
    if configs.empty or "config_id" not in configs.columns:
        return pd.DataFrame(
            {
                "Member": range(1, len(members) + 1),
                "Config ID": members,
                "Weight": [
                    float(weights[i]) if i < len(weights) else np.nan
                    for i in range(len(members))
                ],
            }
        )

    lookup = configs.drop_duplicates("config_id", keep="first").set_index("config_id")
    rows: list[dict[str, Any]] = []
    for index, config_id in enumerate(members, start=1):
        row: dict[str, Any] = {"Member": index, "Config ID": config_id}
        if config_id in lookup.index:
            source = lookup.loc[config_id]
            if isinstance(source, pd.DataFrame):
                source = source.iloc[0]
            mapping = (
                ("candidate_family", "Family"),
                ("modalities", "Modalities"),
                ("integration", "Integration"),
                ("integration_n_components", "Components"),
                ("resolution", "Representation"),
                ("count_transformation", "Transformation"),
                ("learner", "Learner"),
            )
            for source_name, display_name in mapping:
                value = source.get(source_name, "")
                if pd.notna(value) and str(value).strip():
                    row[display_name] = str(value)
        if index - 1 < len(weights):
            try:
                value = float(weights[index - 1])
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                row["Weight"] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    columns = [
        "Member",
        "Config ID",
        "Family",
        "Modalities",
        "Integration",
        "Components",
        "Representation",
        "Transformation",
        "Learner",
        "Weight",
    ]
    return out[[column for column in columns if column in out.columns]]


def _manifest(root: Path) -> dict[str, Any]:
    return _read_json(root / "manifest.json")


def _sweep_data_source(sweep: Sweep):
    return sweep.samples if getattr(sweep, "uses_modalities", False) else sweep.data


def _procedure_table(sweep: Sweep, root: Path) -> pd.DataFrame:
    manifest = _manifest(root)
    ev = sweep.evaluation
    gate = sweep.gate
    data = (
        manifest.get("sweep", {}).get("data", {})
        if isinstance(manifest.get("sweep"), dict)
        else {}
    )
    source = _sweep_data_source(sweep)
    classes = manifest.get("class_labels", getattr(source, "class_labels", ()))
    return pd.DataFrame(
        [
            ["Task", sweep.title],
            ["Experiment directory", str(root)],
            [
                "Data format",
                "multimodal"
                if getattr(sweep, "uses_modalities", False)
                else str(getattr(source, "format", data.get("format", ""))),
            ],
            ["Samples", str(manifest.get("n_samples", ""))],
            ["Classes", ", ".join(map(str, classes)) if classes else ""],
            ["Procedure", ev.protocol],
            [
                "Stratification",
                "target"
                if not getattr(source, "stratify_col", None)
                else f"target + {getattr(source, 'stratify_col')}",
            ],
            [
                "Outer folds",
                ev.outer_folds
                if ev.protocol not in {"lodo", "leave_one_dataset_out"}
                else "held-out datasets",
            ],
            [
                "Inner folds",
                "leave-one-dataset-out across outer-training datasets"
                if ev.protocol in {"lodo", "leave_one_dataset_out"}
                else ev.inner_folds,
            ],
            ["Repeats", ev.repeats],
            ["Selection metric", ev.optimize_metric],
            ["Random seed", ev.random_state],
            [
                "Qualification gate",
                f"on ({gate.metric} ≥ {gate.threshold})" if gate.enabled else "off",
            ],
            [
                "Selection rule",
                "Model and ensemble selection use inner-validation performance; reported performance uses held-out outer evaluation predictions.",
            ],
        ],
        columns=["Field", "Value"],
    )


def _html_inline(value: Any) -> str:
    text = str(value)
    replacements = {
        r"$\arcsin\sqrt{x}$": "arcsin√x",
        r"$\sqrt{x}$": "√x",
        r"$\ln(1+x)$": "ln(1+x)",
        r"$\log_{10}(1+x)$": "log10(1+x)",
        r"$\log_2(1+x)$": "log2(1+x)",
        r"CLR-$\epsilon$+z": "CLR-ε+z",
        r"CLR-$\epsilon$": "CLR-ε",
        r"$10^{-10}$": "10⁻¹⁰",
        r"F1$_{w}$": "F1w",
        r"F1$_w$": "F1w",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(r"\textsubscript{w}", "w")
    text = text.replace(r"\epsilon", "ε")
    text = text.replace(r"\arcsin", "arcsin")
    text = text.replace(r"\sqrt{x}", "√x")
    text = text.replace(r"\ln", "ln")
    text = text.replace(r"\log_{10}", "log10")
    text = text.replace(r"\log_2", "log2")
    text = text.replace("$", "")
    return text


def _html_table(df: pd.DataFrame, *, raw_html_cols: set[str] | None = None) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    raw_html_cols = raw_html_cols or set()
    hidden_cols = {"Config ID", "config_id"}
    cols = [c for c in df.columns if str(c) not in hidden_cols]
    parts = ['<div class="table-wrap"><table><thead><tr>']
    parts.extend(f"<th>{html.escape(str(c))}</th>" for c in cols)
    parts.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        parts.append("<tr>")
        for c in cols:
            val = r[c]
            if pd.isna(val):
                txt = ""
            elif c in raw_html_cols:
                txt = str(val)
            else:
                number = _html_number(val)
                txt = html.escape(number if number is not None else _html_inline(val))
            parts.append(f"<td>{txt}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _target_dirs_by_label(root: Path) -> dict[str, Path]:
    exp_root = root / "explainability"
    if not exp_root.exists():
        return {}
    aliases = (
        ("MPMA-E", ("mpma_e", "ensemble", "best_mpmas_ensemble", "mpma-e")),
        ("MPMA-B", ("mpma_b", "best_individual", "best_mpma", "mpma-b")),
        ("Baseline RF", ("baseline_rf", "baseline", "rf_baseline", "baseline-rf")),
    )
    out: dict[str, Path] = {}
    for label, slugs in aliases:
        for slug in slugs:
            d = exp_root / slug
            if d.exists() and d.is_dir():
                out[label] = d
                break
    return out


def _xai_class_slug(label: Any) -> str:
    text = str(label).strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "class"


def _xai_method_display(method: str) -> str:
    return {
        "shap": "SHAP",
        "permutation": "Permutation",
        "ale": "ALE",
        "lime": "LIME",
        "interactions": "ALE interactions",
    }.get(str(method).strip().lower(), str(method))


def _xai_coordinate_column(target_dir: Path) -> str:
    table = _read_table(target_dir / "coordinate_metadata.parquet")
    if table.empty or "exact_feature_identity" not in table.columns:
        return "Feature"
    values = table["exact_feature_identity"].astype(str).str.strip().str.lower()
    exact = values.isin({"true", "1", "yes"})
    return "Model coordinate" if bool((~exact).any()) else "Feature"


def _xai_target_metadata(target_dir: Path) -> tuple[list[str], list[tuple[int, str]]]:
    meta = _read_json(target_dir / "explained_unit.json")
    methods = [
        str(x).strip().lower() for x in meta.get("methods", []) if str(x).strip()
    ]
    for path in glob_tables(target_dir, "feature_stability_*"):
        name = path.stem.removeprefix("feature_stability_").strip().lower()
        if name and name not in methods:
            methods.append(name)
    if (
        table_exists(target_dir / "feature_interactions_current.parquet")
        and "interactions" not in methods
    ):
        methods.append("interactions")
    indices = list(meta.get("explained_class_indices", []))
    labels = list(meta.get("explained_class_labels", []))
    classes: list[tuple[int, str]] = []
    for pos, label in enumerate(labels):
        try:
            idx = int(indices[pos]) if pos < len(indices) else int(pos)
        except Exception:
            idx = int(pos)
        classes.append((idx, str(label)))
    if not classes:
        for path in [
            target_dir / "feature_stability.parquet",
            target_dir / "top_features.parquet",
        ]:
            tab = _read_table(path)
            if tab.empty or "class_index" not in tab.columns:
                continue
            lab_col = "class_label" if "class_label" in tab.columns else None
            seen: set[int] = set()
            for _, row in tab.sort_values("class_index").iterrows():
                try:
                    idx = int(row.get("class_index"))
                except Exception:
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                label = (
                    str(row.get(lab_col, f"class_{idx}")) if lab_col else f"class_{idx}"
                )
                classes.append((idx, label))
            if classes:
                break
    return methods, classes


def _refresh_xai_support_figures(target_dir: Path, fallback_top_k: int) -> None:
    top = _read_table(target_dir / "top_features.parquet")
    if top.empty or "feature" not in top.columns:
        return
    meta = _read_json(target_dir / "explained_unit.json")
    try:
        top_k = max(1, int(meta.get("top_k", fallback_top_k)))
    except Exception:
        top_k = max(1, int(fallback_top_k))
    stats = _read_table(target_dir / "feature_distribution_stats.parquet")
    stability = _read_table(target_dir / "feature_stability.parquet")
    figures = target_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if "class_index" in top.columns:
        groups = top.groupby("class_index", sort=True)
    else:
        groups = [(0, top)]
    for class_index, class_top in groups:
        label = (
            str(class_top["class_label"].dropna().iloc[0])
            if "class_label" in class_top.columns
            and not class_top["class_label"].dropna().empty
            else f"class_{int(class_index)}"
        )
        class_stats = stats
        if not stats.empty and "class_index" in stats.columns:
            idx = pd.to_numeric(stats["class_index"], errors="coerce")
            class_stats = stats[idx.eq(int(class_index))].copy()
        plot_feature_support(
            class_top,
            class_stats,
            figures / f"feature_support__{_xai_class_slug(label)}",
            top_k,
            [label],
            stability,
        )


def _xai_figure_for_class(
    target_dir: Path, stem: str, class_label: str, report_dir: Path, caption: str
) -> str:
    slug = _xai_class_slug(class_label)
    return _fig(target_dir / "figures" / f"{stem}__{slug}", report_dir, caption)


def _xai_method_global_text(method: str) -> str:
    return {
        "shap": "Aggregated absolute OOF SHAP attribution across held-out samples for the class probability. Signed local SHAP values are retained separately for local explanations and sign-stability summaries.",
        "permutation": "OOF predictive importance measured by degradation in class-specific loss after disrupting one feature in held-out data.",
        "ale": "Global accumulated local effects for the class probability. Importance summarizes the magnitude of the centered ALE effect; curves show effect shape across the observed feature distribution.",
        "lime": "Aggregated absolute OOF LIME local-surrogate coefficients across held-out samples. Signed local coefficients are retained separately when local explanations are requested.",
    }.get(str(method).strip().lower(), "Global OOF feature explanation.")


def _xai_local_table(target_dir: Path, method: str, top_n: int = 5) -> pd.DataFrame:
    selected = _read_table(
        target_dir / f"instance_explanations_{method}_selected.parquet"
    )
    top = _read_table(
        target_dir / f"instance_explanations_{method}_top_features.parquet"
    )
    if selected.empty or top.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    feature_column = _xai_coordinate_column(target_dir)
    group_cols = [c for c in ("sample_id", "class_index") if c in top.columns]
    grouped = top.groupby(group_cols, sort=False) if group_cols else [((), top)]
    for _, group in grouped:
        group = (
            group.sort_values("rank", ascending=True)
            if "rank" in group.columns
            else group
        )
        for _, row in group.head(int(top_n)).iterrows():
            item: dict[str, Any] = {
                "Sample": str(row.get("sample_id", "")),
                "Role": str(row.get("selection_role", "")),
                "Class": str(row.get("class_label", row.get("class_index", ""))),
                feature_column: _short_feature_label(row.get("feature", "")),
            }
            p = _safe_float(row.get("p_class_mean", np.nan))
            item["P(class)"] = f"{p:.3f}" if np.isfinite(p) else ""
            value = _safe_float(row.get("value", np.nan))
            label = "SHAP contribution" if method == "shap" else "LIME coefficient"
            item[label] = f"{value:+.3f}" if np.isfinite(value) else ""
            sd = _safe_float(row.get("value_sd", np.nan))
            if np.isfinite(sd):
                item["Across-fold SD"] = f"{sd:.3f}"
            rank = _safe_float(row.get("rank", np.nan))
            item["Local rank"] = int(rank) if np.isfinite(rank) else ""
            rows.append(item)
    return pd.DataFrame(rows)


def _xai_local_mode(target_dir: Path) -> str:
    meta = _read_json(target_dir / "explained_unit.json")
    cfg = meta.get("explainability_config", {}) if isinstance(meta, dict) else {}
    value = str(cfg.get("local_explanations", "")).strip()
    if value:
        return value
    representative = bool(cfg.get("representative_instances", False))
    requested = bool(cfg.get("instance_sample_ids", []))
    if representative and requested:
        return "representative_and_requested"
    if representative:
        return "representative"
    if requested:
        return "requested"
    return "none"


def _explainability_report_blocks(
    root: Path, report_dir: Path, top_n: int = 15
) -> tuple[str, int]:
    target_dirs = _target_dirs_by_label(root)
    if not target_dirs:
        return "", 0
    parts: list[str] = []
    blocks = 0
    for label in ("MPMA-B", "MPMA-E", "Baseline RF"):
        target_dir = target_dirs.get(label)
        if target_dir is None:
            continue
        methods, classes = _xai_target_metadata(target_dir)
        if not methods and not classes:
            continue
        blocks += 1
        method_text = (
            ", ".join(_xai_method_display(x) for x in methods)
            if methods
            else "available methods"
        )
        _refresh_xai_support_figures(target_dir, top_n)
        local_mode = _xai_local_mode(target_dir)
        coordinate_mode = _xai_coordinate_column(target_dir) == "Model coordinate"
        unit_singular = "model coordinate" if coordinate_mode else "feature"
        unit_plural = "model coordinates" if coordinate_mode else "features"
        parts.append(f'<section class="xai-target"><h3>{html.escape(label)}</h3>')
        parts.append(
            f"<p>Cross-fitted OOF explanations of the final selected specification. Methods: {html.escape(method_text)}. Global explanation with fold stability and local sample explanation are reported as distinct layers.</p>"
        )
        parts.append("<h4>Global explanations</h4>")
        parts.append(
            f"<p>Global results summarize held-out predictions across samples. SHAP and LIME global importance are aggregations of local OOF attributions; permutation importance and ALE are population-level quantities by construction. The reported units are {html.escape(unit_plural)}.</p>"
        )
        for class_index, class_label in classes:
            parts.append(
                f'<section class="xai-class"><h5>{html.escape(class_label)}</h5>'
            )
            consensus_fig = _xai_figure_for_class(
                target_dir,
                "feature_support",
                class_label,
                report_dir,
                f"{label} · {class_label}: cross-method top-k support and fold stability",
            )
            if consensus_fig:
                parts.append("<h6>Cross-method support and fold stability</h6>")
                parts.append(
                    f"<p>Top-k support is a within-method rank score for each {html.escape(unit_singular)}: rank 1 scores 1, rank k scores 1/k, and ranks below k score 0. Fold stability is the fraction of estimable outer folds in which that {html.escape(unit_singular)} ranks within the method-specific top k. Both are scale-free 0–1 summaries; method-specific effect magnitudes are not compared across methods.</p>"
                )
                parts.append(consensus_fig)
            for method in methods:
                if method == "interactions":
                    continue
                display = _xai_method_display(method)
                global_fig = _xai_figure_for_class(
                    target_dir,
                    f"feature_importance_{method}",
                    class_label,
                    report_dir,
                    f"{label} · {class_label}: global {display} explanation",
                )
                ale_curve = (
                    _xai_figure_for_class(
                        target_dir,
                        "ale_curves",
                        class_label,
                        report_dir,
                        f"{label} · {class_label}: ALE effect curves",
                    )
                    if method == "ale"
                    else ""
                )
                if not global_fig and not ale_curve:
                    continue
                parts.append(f"<h6>{html.escape(display)}</h6>")
                parts.append(f"<p>{html.escape(_xai_method_global_text(method))}</p>")
                if global_fig:
                    parts.append(global_fig)
                if ale_curve:
                    parts.append(ale_curve)
            if "interactions" in methods:
                interaction_figs_class = []
                for stem, caption in [
                    ("interaction_network_current", "2D ALE interaction network"),
                ]:
                    block = _xai_figure_for_class(
                        target_dir,
                        stem,
                        class_label,
                        report_dir,
                        f"{label} · {class_label}: {caption}",
                    )
                    if block:
                        interaction_figs_class.append(block)
                if interaction_figs_class:
                    parts.append("<h6>ALE interactions</h6>")
                    parts.append(
                        "<p>Exploratory class-specific OOF 2D ALE interaction strengths. Edge weight represents interaction magnitude; node abundance compares the target class with the remaining classes.</p>"
                    )
                    parts.extend(interaction_figs_class)
            parts.append("</section>")
        interaction_figs = []
        for stem, caption in [
            ("interaction_network_current", "2D ALE interaction network"),
        ]:
            block = _fig(
                target_dir / "figures" / stem, report_dir, f"{label}: {caption}"
            )
            if block:
                interaction_figs.append(block)
        if interaction_figs:
            parts.append("<h5>Global interactions</h5>")
            parts.append(
                "<p>Interaction outputs are exploratory population-level 2D ALE summaries and are not local sample explanations.</p>"
            )
            parts.extend(interaction_figs)
        parts.append("<h4>Local explanations</h4>")
        if local_mode == "none":
            parts.append(
                "<p>Local sample-level reporting was not requested for this run. SHAP/LIME may still compute local values internally to form global summaries, but individual samples are not presented.</p>"
            )
        else:
            parts.append(
                f"<p>Local OOF explanation mode: {html.escape(local_mode)}. Each displayed sample is explained only by outer-fold model(s) that did not train on that sample.</p>"
            )
            found_local = False
            for method in ("shap", "lime"):
                if method not in methods:
                    continue
                local_tab = _xai_local_table(target_dir, method, top_n=5)
                if local_tab.empty:
                    continue
                found_local = True
                parts.append(
                    f"<h5>{html.escape(_xai_method_display(method))} local explanations</h5>"
                )
                if method == "shap":
                    parts.append(
                        "<p>Signed SHAP contributions indicate whether each feature pushes the class probability upward or downward relative to the SHAP baseline.</p>"
                    )
                else:
                    parts.append(
                        "<p>Signed LIME coefficients are local surrogate effects around the selected held-out sample and should not be interpreted as global model coefficients.</p>"
                    )
                parts.append(_html_table(local_tab))
            if not found_local:
                parts.append(
                    "<p>No local SHAP/LIME table is available for the requested mode.</p>"
                )
        parts.append("</section>")
    return "".join(parts), blocks


def _strip_cell_markup(value: Any) -> str:
    text = _html_inline(value)
    text = text.replace("<strong>", "").replace("</strong>", "")
    text = text.replace(r"\textbf{", "").replace("}", "")
    text = text.replace(r"$\pm$", " ± ").replace(r"\pm", "±")
    text = text.replace("F1$_w$", "F1w").replace("F1$_{w}$", "F1w")
    return text


def _terminal_table(
    title: str, df: pd.DataFrame, *, max_rows: int | None = None
) -> None:
    if df.empty:
        return
    from rich.table import Table

    tab = Table(title=title, show_lines=False)
    for col in df.columns:
        tab.add_column(str(col), overflow="fold", no_wrap=False)
    table_preview = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for _, row in table_preview.iterrows():
        tab.add_row(
            *[_strip_cell_markup(row.get(col, "")) for col in table_preview.columns]
        )
    console.print(tab)


def _compact_procedure_for_terminal(procedure: pd.DataFrame) -> pd.DataFrame:
    if procedure.empty:
        return procedure
    keep = {
        "Task",
        "Procedure",
        "Samples",
        "Classes",
        "Outer folds",
        "Inner folds",
        "Repeats",
        "Selection metric",
        "Qualification gate",
    }
    out = procedure[procedure["Field"].isin(keep)].copy()
    order = {
        k: i
        for i, k in enumerate(
            [
                "Task",
                "Procedure",
                "Samples",
                "Classes",
                "Outer folds",
                "Inner folds",
                "Repeats",
                "Selection metric",
                "Qualification gate",
            ]
        )
    }
    out["_order"] = out["Field"].map(order).fillna(999)
    return out.sort_values("_order").drop(columns="_order")


def _terminal_feature_label(value: Any) -> str:
    text = str(value).split("___")[-1].split("|")[-1]
    rank = ""
    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if text.startswith(pfx):
            rank = f"{pfx[0]}. "
            text = text[len(pfx) :]
            break
    text = text.replace("_", " ").strip()
    return f"{rank}{text}" if text else str(value).replace("_", " ")


def _short_feature_label(feature: Any, max_len: int = 64) -> str:
    text = str(feature)
    if text.startswith("ALR[") and text.endswith("]"):
        body = text[4:-1]
        if "/" in body:
            numerator, reference = body.split("/", 1)
            label = f"ALR[{_terminal_feature_label(numerator)} / {_terminal_feature_label(reference)}]"
        else:
            label = text
    elif text.startswith("ILR_"):
        parts = text.split("_", 2)
        if len(parts) == 3 and parts[1].isdigit():
            label = f"ILR balance {int(parts[1])} · {parts[2][:10]}"
        else:
            label = text.replace("_", " ")
    else:
        label = _terminal_feature_label(text)
    return label if len(label) <= max_len else label[: max_len - 1].rstrip() + "…"


def _target_dirs_for_terminal(root: Path) -> list[tuple[str, Path]]:
    dirs = _target_dirs_by_label(root)
    return [
        (label, dirs[label])
        for label in ("MPMA-E", "MPMA-B", "Baseline RF")
        if label in dirs
    ]


def _feature_support_terminal(root: Path, *, top_n: int = 8) -> None:
    rows: list[dict[str, Any]] = []
    for label, target_dir in _target_dirs_for_terminal(root):
        path = target_dir / "top_features.parquet"
        if not table_exists(path):
            path = target_dir / "top_features_shap.parquet"
        tab = _read_table(path)
        if tab.empty or "feature" not in tab.columns:
            continue
        if "rank" in tab.columns:
            tab = tab.sort_values("rank", ascending=True)
        methods = [
            c
            for c in (
                "SHAP",
                "LIME",
                "Permutation",
                "ALE",
                "consensus",
                "importance_mean",
            )
            if c in tab.columns
        ]
        for i, r in tab.head(top_n).iterrows():
            support = ""
            if methods:
                vals = pd.to_numeric(
                    pd.Series([r.get(c) for c in methods]), errors="coerce"
                ).dropna()
                if not vals.empty:
                    support = f"{float(vals.mean()):.3f}"
            rows.append(
                {
                    "Strategy": label,
                    "Rank": int(r.get("rank", len(rows) + 1))
                    if str(r.get("rank", "")).strip()
                    else len(rows) + 1,
                    "Feature": _short_feature_label(r.get("feature", "")),
                    "Support": support,
                }
            )
    if rows:
        _terminal_table(
            "Top explainability features", pd.DataFrame(rows), max_rows=len(rows)
        )


def _print_report_summary(
    sweep: Sweep,
    root: Path,
    procedure: pd.DataFrame,
    strategy_html: pd.DataFrame,
    top_mpmas: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
    ensemble_members: pd.DataFrame,
) -> None:
    stage("Run summary", str(root))
    _terminal_table("Evaluation procedure", _compact_procedure_for_terminal(procedure))
    _terminal_table("Task performance", strategy_html)
    if not top_mpmas.empty:
        cols = [
            c
            for c in [
                "Rank",
                "Resolution",
                "Count transformation",
                "Learner",
                "Inner nMCC",
                "Outer nMCC",
                "Inner ROC-AUC",
                "Outer ROC-AUC",
            ]
            if c in top_mpmas.columns
        ]
        _terminal_table(
            "Top MPMA-B configurations",
            top_mpmas[cols] if cols else top_mpmas,
            max_rows=5,
        )
    _terminal_table("Final MPMA-E specification", ensemble_summary)
    if not ensemble_members.empty:
        cols = [
            c
            for c in [
                "Member",
                "Resolution",
                "Count transformation",
                "Learner type",
                "count_transformation",
                "learner",
            ]
            if c in ensemble_members.columns
        ]
        _terminal_table(
            "Final MPMA-E members", ensemble_members[cols] if cols else ensemble_members
        )
    _feature_support_terminal(root, top_n=8)


def _procedure_grid_html(procedure: pd.DataFrame) -> str:
    if procedure.empty or not {"Field", "Value"}.issubset(procedure.columns):
        return _html_table(procedure)
    rows = [(str(row["Field"]), str(row["Value"])) for _, row in procedure.iterrows()]
    wide = [row for row in rows if row[0] == "Selection rule"]
    compact = [row for row in rows if row[0] != "Selection rule"]
    midpoint = (len(compact) + 1) // 2
    columns = (compact[:midpoint], compact[midpoint:])
    parts = ['<div class="procedure-grid">']
    for column in columns:
        parts.append('<div class="procedure-column">')
        for field, value in column:
            parts.append('<div class="procedure-row">')
            parts.append(f'<div class="procedure-field">{html.escape(field)}</div>')
            parts.append(f'<div class="procedure-value">{html.escape(value)}</div>')
            parts.append("</div>")
        parts.append("</div>")
    for field, value in wide:
        parts.append('<div class="procedure-row procedure-wide">')
        parts.append(f'<div class="procedure-field">{html.escape(field)}</div>')
        parts.append(f'<div class="procedure-value">{html.escape(value)}</div>')
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _publication_layout_css() -> str:
    return """
.procedure-grid { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); column-gap:34px; margin:10px 0 24px; }
.procedure-column { min-width:0; }
.procedure-row { display:grid; grid-template-columns:132px minmax(0,1fr); gap:12px; padding:6px 7px; border-bottom:1px solid var(--track); line-height:1.38; }
.procedure-field { color:var(--mid); font-size:var(--font-table); font-weight:700; }
.procedure-value { color:var(--ink); font-size:var(--font-table); min-width:0; overflow-wrap:anywhere; }
.procedure-wide { grid-column:1 / -1; margin-top:2px; }
#mpma-b-composition, #mpma-e-specification { margin:4px 0 22px; }
#mpma-b-composition h3, #mpma-e-specification h3 { margin-top:8px; }
#mpma-b-composition .table-wrap, #mpma-e-specification .table-wrap { margin-top:8px; margin-bottom:14px; }
@media (max-width: 760px) {
  .procedure-grid { grid-template-columns:1fr; column-gap:0; }
  .procedure-wide { grid-column:auto; }
}
"""


def _report_css() -> str:
    return """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --ink:#0f172a; --mid:#64748b; --dim:#94a3b8; --track:#e2e8f0; --soft:#f8fafc;
  --bg:#ffffff; --blue:#2563eb; --blue-hover:#0ea5e9; --blue-soft:#eff6ff; --blue-hover-soft:#f0f9ff;
  --nav-h:44px; --content-w:1120px;
  --font-body:13px; --font-small:12px; --font-table:12px;
}
*, *::before, *::after { box-sizing:border-box; }
html {
  background:var(--bg); color:var(--ink);
  font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
  letter-spacing:-0.01em; scroll-behavior:smooth;
}
body { margin:0; padding:0; background:var(--bg); }
a { color:var(--blue); text-decoration:none; }
a:hover { opacity:0.85; text-decoration:none; }
a:focus-visible { outline:2px solid #38bdf8; outline-offset:3px; border-radius:6px; }
.report-nav {
  position:sticky; top:0; z-index:30; height:var(--nav-h);
  background:rgba(255,255,255,0.93); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border-bottom:1px solid #f1f5f9; box-shadow:none;
}
.report-nav-inner {
  height:var(--nav-h); max-width:var(--content-w); margin:0 auto; padding:0 24px;
  display:flex; align-items:center; justify-content:space-between; gap:22px;
}
.brand { display:flex; align-items:center; gap:9px; color:var(--ink); font-size:0.82rem; font-weight:700; text-decoration:none; white-space:nowrap; }
.brand:hover { opacity:0.85; }
.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border:1px solid var(--track); border-radius:7px; font-size:0.72rem; background:#fff; letter-spacing:-0.03em; }
.report-links { display:flex; align-items:center; gap:3px; overflow-x:auto; white-space:nowrap; min-width:0; }
.report-links a { color:var(--mid); font-size:0.78rem; font-weight:500; padding:7px 8px; border-radius:6px; }
.report-links a:hover { color:var(--blue-hover); background:var(--blue-hover-soft); opacity:1; }
.report-shell { max-width:var(--content-w); margin:0 auto; padding:28px 24px 70px; }
.report-content { min-width:0; }
h1 { font-size:26px; line-height:1.16; margin:0 0 8px; font-weight:800; letter-spacing:-0.04em; }
h2 { font-size:16px; margin:38px 0 14px; font-weight:750; border-top:1px solid var(--track); padding-top:20px; letter-spacing:-0.025em; }
h2.first-section { margin-top:0; border-top:0; padding-top:0; }
h3 { font-size:13px; font-weight:700; margin:16px 0 10px; }
h4 { font-size:13px; font-weight:750; margin:28px 0 10px; }
h5 { font-size:12px; font-weight:750; margin:22px 0 8px; color:var(--ink); }
h6 { font-size:12px; font-weight:650; margin:16px 0 7px; color:var(--mid); }
.xai-target { margin:0 0 34px; }
.xai-class { border-top:1px solid var(--track); margin-top:20px; padding-top:2px; }
p, li { font-size:var(--font-body); color:var(--mid); line-height:1.56; }
.report-path { margin-top:0; }
figure { margin:18px 0 28px; overflow-x:auto; }
figure img, .embedded-svg svg { max-width:100%; width:auto; height:auto; display:block; }
figcaption { font-size:var(--font-small); color:var(--mid); margin-top:7px; }
.compare-block { margin:17px 0 32px; overflow-x:auto; padding-bottom:2px; }
.compare-block > h3 { font-size:13px; margin:18px 0 11px; color:var(--ink); }
.compare-grid { display:grid; grid-template-columns:repeat(var(--compare-columns), minmax(430px, 1fr)); gap:16px; align-items:start; }
.compare-cell { min-width:0; }
.compare-cell h3 { font-size:12px; color:var(--mid); margin:0 0 7px; font-weight:700; }
.compare-cell figure { margin:0; }
.compare-cell figcaption { display:none; }
.compare-cell img, .compare-cell .embedded-svg svg { min-width:430px; }
.missing-figure { border:1px solid var(--track); color:var(--mid); font-size:12px; padding:36px 12px; text-align:center; background:#fff; border-radius:10px; }
.table-wrap { overflow-x:auto; margin:12px 0 24px; }
table { border-collapse:collapse; width:100%; font-size:var(--font-table); }
th, td { border-bottom:1px solid var(--track); padding:7px 7px; text-align:left; vertical-align:top; line-height:1.38; }
th { color:var(--mid); font-weight:700; background:#fff; }
code { font-family:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; }
.report-footer { border-top:1px solid var(--track); margin-top:42px; padding-top:16px; color:var(--dim); font-size:12px; }
@media (max-width: 860px) {
  .report-nav-inner { padding:0 16px; }
  .report-shell { padding:22px 18px 56px; }
  .brand span:last-child { display:none; }
  .report-links a { padding:7px 6px; }
  h1 { font-size:22px; }
  .compare-grid { grid-template-columns:repeat(var(--compare-columns), minmax(360px, 1fr)); }
  .compare-cell img, .compare-cell .embedded-svg svg { min-width:360px; }
}
""" + _publication_layout_css()


def write_report(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    stage("Report", str(report_dir))

    with phase_progress("Report analysis", 4) as phase:
        phase.phase("procedure and rankings")
        procedure = _procedure_table(sweep, root)
        top_mpmas = _top_mpma_display(_top_mpma_raw(root, 5), html_mode=True)
        if getattr(sweep, "uses_modalities", False) and not top_mpmas.empty:
            top_mpmas = top_mpmas.rename(
                columns={
                    "Resolution": "Representation",
                    "Count transformation": "Transformation",
                }
            )
        strategy_rows = _strategy_rows(root)
        phase.phase("compute accounting")
        compute_accounting = run_compute_accounting(root, sweep, strategy_rows)
        compute_display = compute_accounting.get("display", pd.DataFrame())
        compute_environment = compute_accounting.get("environment", {})
        phase.phase("statistics")
        statistics = run_report_statistics(
            root,
            sweep.evaluation.protocol,
            strategy_rows,
            n_bootstrap=2000,
            random_state=sweep.evaluation.random_state,
        )
        strategy_html = _strategy_performance_display(root, html_mode=True)
        pairwise = statistics.get("pairwise", pd.DataFrame())
        primary_pairwise = (
            pairwise[
                pairwise["metric"].astype(str).eq(str(sweep.evaluation.optimize_metric))
            ].copy()
            if not pairwise.empty and "metric" in pairwise.columns
            else pd.DataFrame()
        )
        if not primary_pairwise.empty:
            primary_pairwise = primary_pairwise[
                [
                    c
                    for c in [
                        "strategy_a",
                        "strategy_b",
                        "difference_a_minus_b",
                        "difference_ci_low",
                        "difference_ci_high",
                        "test",
                        "p_value",
                        "p_holm",
                        "significant_holm_0_05",
                    ]
                    if c in primary_pairwise.columns
                ]
            ].copy()
            primary_pairwise = primary_pairwise.rename(
                columns={
                    "strategy_a": "Strategy A",
                    "strategy_b": "Strategy B",
                    "difference_a_minus_b": "Difference A-B",
                    "difference_ci_low": "95% CI low",
                    "difference_ci_high": "95% CI high",
                    "test": "Test",
                    "p_value": "p",
                    "p_holm": "Holm p",
                    "significant_holm_0_05": "Holm p<0.05",
                }
            )
            for c in ("Difference A-B", "95% CI low", "95% CI high"):
                if c in primary_pairwise.columns:
                    primary_pairwise[c] = pd.to_numeric(
                        primary_pairwise[c], errors="coerce"
                    ).map(lambda x: f"{x:.3f}" if np.isfinite(x) else "")
            for c in ("p", "Holm p"):
                if c in primary_pairwise.columns:
                    primary_pairwise[c] = pd.to_numeric(
                        primary_pairwise[c], errors="coerce"
                    ).map(lambda x: _format_p_value(x))
        phase.phase("ensemble summaries")
        ensemble_summary = _ensemble_summary_table(root)
        ensemble_members = _ensemble_members_table(root)

    cpu_model = str(compute_environment.get("cpu_model", "")).strip()
    logical_cpus = compute_environment.get("logical_cpus", "")
    physical_cpus = compute_environment.get("physical_cpus", "")
    workers = compute_environment.get("evaluation_workers", "")
    threads_per_worker = compute_environment.get("threads_per_worker", "")
    total_memory = compute_environment.get("memory_total_bytes")
    memory_text = ""
    try:
        if total_memory is not None and np.isfinite(float(total_memory)):
            memory_text = f"{float(total_memory) / (1024**3):.3f} GiB RAM"
    except Exception:
        memory_text = ""
    hardware_parts = [
        x
        for x in [
            cpu_model,
            f"{physical_cpus} physical / {logical_cpus} logical CPUs"
            if physical_cpus and logical_cpus
            else "",
            memory_text,
            f"{workers} workers × {threads_per_worker} threads"
            if workers and threads_per_worker
            else "",
        ]
        if x
    ]
    hardware_text = " · ".join(hardware_parts)

    with phase_progress("Report outputs", 3) as phase:
        phase.phase("writing report tables")
        write_table(tables_dir / "evaluation_procedure.parquet", procedure)
        write_table(tables_dir / "top5_mpma_inner_outer_performance.parquet", top_mpmas)
        write_table(tables_dir / "task_strategy_performance.parquet", strategy_html)
        write_table(
            tables_dir / "strategy_pairwise_primary_metric.parquet", primary_pairwise
        )
        write_table(tables_dir / "strategy_compute_display.parquet", compute_display)
        write_table(tables_dir / "mpma_e_selection.parquet", ensemble_summary)
        write_table(tables_dir / "mpma_e_members.parquet", ensemble_members)

        phase.phase("global figures")
        figs = []
        figs.append(
            _fig(
                root / "figures" / "representation_impact",
                report_dir,
                "MPDR representation impact overview",
            )
        )
        figs.append(
            _fig(root / "figures" / "mpma_e", report_dir, "Selected MPMA-E schematic")
        )
        phase.phase("explainability visuals")
        from .explainability import refresh_explainability_visuals

        refresh_explainability_visuals(root, sweep)
        explainability_html, explainability_count = _explainability_report_blocks(
            root, report_dir, top_n=int(getattr(sweep.explainability, "top_k", 15))
        )

    css = _report_css()
    top_metric_cols = {
        c for c in top_mpmas.columns if c.startswith("Inner") or c.startswith("Outer")
    }
    strat_metric_cols = {c for c in strategy_html.columns if c != "Strategy"}
    html_text = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>mllabiome report</title><link rel="icon" type="image/svg+xml" href="{_favicon_href()}"><style>{css}</style></head>
<body>
<header class="report-nav"><div class="report-nav-inner"><a class="brand" href="#top" aria-label="mllabiome report"><span class="brand-mark">mll</span><span>mllabiome</span></a><nav class="report-links" aria-label="Report navigation">
<a href="#procedure">Evaluation</a>
<a href="#performance">Performance</a>
<a href="#mpma-b-composition">MPMA-B</a>
<a href="#mpma-e">MPMA-E</a>
<a href="#explainability-comparison">Explainability</a>
<a href="#figures">Figures</a>
<a href="#statistics">Statistics</a>
</nav></div></header>
<div class="report-shell"><main id="top" class="report-content">
<h2 id="procedure" class="first-section">Evaluation procedure</h2>
{_procedure_grid_html(procedure)}
<h2 id="performance">Task performance summary</h2>
<p>Held-out strategy performance is reported as mean ± SD across outer evaluation units with 95% bootstrap confidence intervals. For LODO, outer units are held-out datasets; for nested cross-validation, they are outer test folds. PR-AUC (AP) is average precision. Additional metric-level summaries and pairwise tests are available in the report tables.</p>
{_html_table(strategy_html, raw_html_cols=strat_metric_cols)}
<section id="mpma-e-specification">
<h3 id="mpma-e">Final MPMA-E specification</h3>
{_html_table(ensemble_summary)}
{_html_table(ensemble_members)}
</section>
<h3 id="statistics">Statistical comparisons</h3>
<p>Pairwise differences are Strategy A minus Strategy B. The table shows the configured primary metric; complete pairwise results are available in strategy_pairwise_tests.parquet. Holm adjustment is applied across strategy pairs within each metric.</p>
{_html_table(primary_pairwise)}
<h2 id="top-mpmas">Top 5 MPMA-B configurations</h2>
{_html_table(top_mpmas, raw_html_cols=top_metric_cols)}

<h2 id="explainability-comparison">Explainability</h2>
{explainability_html if explainability_html else "<p>No explainability artefacts are available yet.</p>"}

<h2 id="figures">Global figures</h2>
{"".join(figs) if figs else "<p>No figure artefacts found yet.</p>"}
<h3 id="compute">Computational resources</h3>
<p>Compute is summarized by additive CPU core-hours, model-fit count, and peak resident memory for the worker process tree. CPU time includes child processes and external R processes when used. MPMA-B and MPMA-E share the MPMA search pool, so their compute totals overlap.{(" Hardware: " + html.escape(hardware_text) + ".") if hardware_text else ""}</p>
{_html_table(compute_display)}
<p class="report-footer">mllabiome · generated report</p>
</main></div></body></html>
"""
    (report_dir / "index.html").write_text(html_text, encoding="utf-8")
    dump_json_standard(
        {
            "report_dir": report_dir,
            "task": "classification",
            "target": str(_sweep_data_source(sweep).target_col),
            "figures_embedded": len([f for f in figs if f]),
            "explainability_targets": explainability_count,
        },
        report_dir / "report_manifest.json",
    )
    _print_report_summary(
        sweep,
        root,
        procedure,
        strategy_html,
        top_mpmas,
        ensemble_summary,
        ensemble_members,
    )
    success("Report completed")
    outputs = {
        "html_report": report_dir / "index.html",
        "top5_mpmas": tables_dir / "top5_mpma_inner_outer_performance.parquet",
        "mpma_e_selection": tables_dir / "mpma_e_selection.parquet",
        "mpma_e_members": tables_dir / "mpma_e_members.parquet",
        "strategy_outer_unit_metrics": statistics.get(
            "unit_metrics_path", tables_dir / "strategy_outer_unit_metrics.parquet"
        ),
        "strategy_metrics_bootstrap": statistics.get(
            "summary_path", tables_dir / "strategy_metrics_bootstrap.parquet"
        ),
        "strategy_pairwise_tests": statistics.get(
            "pairwise_path", tables_dir / "strategy_pairwise_tests.parquet"
        ),
        "strategy_statistics_manifest": statistics.get(
            "manifest_path", tables_dir / "strategy_statistics_manifest.json"
        ),
        "strategy_compute": compute_accounting.get(
            "compute_path", tables_dir / "strategy_compute.parquet"
        ),
        "compute_accounting_manifest": compute_accounting.get(
            "manifest_path", tables_dir / "compute_accounting_manifest.json"
        ),
    }
    path_table("Report outputs", outputs)
    return outputs
