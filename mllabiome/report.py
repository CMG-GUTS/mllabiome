from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .configs_sweep import Sweep
from .console import console, path_table, stage, success
from .utils import dump_json_standard
from .metrics import compute_metrics
from .report_statistics import run_report_statistics

_METRICS = [
    ("AUC", "ROC-AUC"),
    ("PR_AUC", "PR-AUC (AP)"),
    ("nMCC", "nMCC"),
    ("F1w", "F1$_{w}$"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
]

_BASELINE_RANK_PRIORITY = ("strain", "species", "genus")

_FAVICON_DATA_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAACZElEQVR42u2bTWgTURSFz7w0jQvbRItm4uhCsSgUq6ilC0UUF7ULtYi6iyhFkVJBN0XciQhSFBX8hy4EbcWFLjShUlsFf4qgCHYh3RXcJKaNDIVYqHFcJcybmh8XQt68c3Z37mTI+d59Ny8w13Acx4HGqqt0w7e0rbTBVdFw2bxRqgJUN14tiL8C8JqvRLFWVY2PBQDcH1LVeDkQXk8SgMKNfjFeCoTbn/Drnq+2IsS/dk2/NUKh2+p7q0DosvqlPApoLgIgAAIgAAIgAAIgAAIgAAIgAAIgAA1VV6tfrMlqRT6fL8YjiUG0bdlYdZ4VQAAEQAD/rQlGzBYpHkkMYlEohP6rt/Hu/Uf8nJtD89rVOBY/jKPxQzAMA5npLK5cu4vE8BjS6QyWNi3B7p3bcK6vF9YKU+1fgefJUdy6dx/z87+K175MfMWZvvP49HkCPSfi2HewG9Mz2WI+lfqOh4+eYvTVW7x+8RimuVzdLXD95oBk3q0HQ0/Q2XVEMu9WKp3Bxf4bavcAIQQuXTiLqclxvEwOoaFhsZS37Vm0t23ChzfPMDU5jq69HVI+OTymNoAD+/fg5PE4IuFGbN3cih3b26V8IBDAwJ3LWNe8BpFwI06f6pbyM9kfsO1ZdQF0duySYjO6TIo3tKzHSitWjGNmdMEzcrmcugBingYWDAal2LLkLv/bdYQtyFF5C3gNexWqr68JszwIEQABEAABeGQ4juP4/fU4r9x+uQUIgAAIgAAIgAAIgAAIAHq8Nu/1KHT6D+BWwbOoRMjPqy8B0KkK3F45NsfBSY7Olp8e13Z4Whf9ATkE686w7OUiAAAAAElFTkSuQmCC"


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


def _rel(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), start.resolve()).replace(os.sep, "/")
    except Exception:
        return path.as_posix()


def _normalise_svg_for_report(svg: str) -> str:
    return svg


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
        try:
            svg = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            svg = path.read_text(encoding="utf-8", errors="ignore")
        svg = _normalise_svg_for_report(_strip_svg_preamble(svg))
        return f'<figure><div class="embedded-svg" role="img" aria-label="{alt}">{svg}</div>{cap}</figure>'
    if path.suffix.lower() == ".png":
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<figure><img src="data:image/png;base64,{data}" alt="{alt}">{cap}</figure>'
    return f'<figure><p><a href="{html.escape(_rel(path, report_dir))}">{alt}</a></p>{cap}</figure>'


def _strip_svg_preamble(svg: str) -> str:
    lines = svg.splitlines()
    out: list[str] = []
    in_doctype = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("<?xml"):
            continue
        if stripped.startswith("<!DOCTYPE"):
            in_doctype = not stripped.rstrip().endswith(">")
            continue
        if in_doctype:
            if stripped.rstrip().endswith(">"):
                in_doctype = False
            continue
        out.append(line)
    text = "\n".join(out).strip()
    start = text.find("<svg")
    if start > 0:
        text = text[start:]
    return text


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
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


def _pct_cell(
    mean: Any, std: Any, *, bold: bool = False, html_mode: bool = False
) -> str:
    m = _safe_float(mean)
    s = _safe_float(std)
    if not np.isfinite(m):
        body = "—" if html_mode else r"--"
        return body

    body = f"{100 * m:.2f}"
    if np.isfinite(s):
        sd = f"{100 * s:.2f}" if html_mode else f"{100 * s:05.2f}"
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
    body = f"{100 * e:.2f}"
    if np.isfinite(sd):
        sd_text = f"{100 * sd:.2f}" if html_mode else f"{100 * sd:05.2f}"
        body += (" ± " if html_mode else r"$\pm$") + sd_text
    if np.isfinite(lo) and np.isfinite(hi):
        body += f" [{100 * lo:.2f}, {100 * hi:.2f}]"
    if bold:
        return f"<strong>{body}</strong>" if html_mode else rf"\textbf{{{body}}}"
    return body


def _strategy_statistics_summary(root: Path) -> pd.DataFrame:
    return _read_tsv(root / "report" / "tables" / "strategy_metrics_bootstrap.tsv")


def _strategy_pairwise_tests(root: Path) -> pd.DataFrame:
    return _read_tsv(root / "report" / "tables" / "strategy_pairwise_tests.tsv")


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
    df = _read_tsv(path)
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


def _top_mpma_raw(root: Path, n: int = 10) -> pd.DataFrame:
    inner = _agg_metrics(root / "inner_results" / "inner_results.tsv", "inner")
    outer = _agg_metrics(root / "results" / "outer_results.tsv", "outer")
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
            "MPDR transformation": str(
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
    path = root / "ensembling" / "ensemble_predictions.tsv"
    if not path.exists() or not str(ensemble_config_id):
        return {}
    df = _read_tsv(path)
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

    outer = _agg_metrics(root / "results" / "outer_results.tsv", "outer")
    inner = _agg_metrics(root / "inner_results" / "inner_results.tsv", "inner")
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
        return pd.DataFrame()
    row = {
        "Strategy": "MPMA-E",
        "Selection": ens.get("selection_strategy", ""),
        "Aggregation": ens.get("aggregation_strategy", ""),
        "Members": ens.get("ensemble_size", ""),
        "Selection metric": ens.get("optimize_metric", "nMCC"),
    }
    return pd.DataFrame([row])


def _ensemble_members_table(root: Path) -> pd.DataFrame:
    p = root / "tables" / "mpma_e_members.tsv"
    if not p.exists():
        p = root / "figures" / "mpma_e_members.tsv"
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        rename = {
            "member_order": "Member",
            "ranks": "Resolution",
            "transformation": "MPDR transformation",
            "classifier_family": "Learner family",
            "raw_transform": "count_transformation",
            "raw_model": "learner",
        }
        cols = [
            c
            for c in [
                "member_order",
                "ranks",
                "transformation",
                "classifier_family",
                "raw_transform",
                "raw_model",
            ]
            if c in df.columns
        ]
        out = df[cols].rename(columns=rename).copy() if cols else df.copy()
        return out
    ens = _ensemble_final_candidate(root)
    members = ens.get("members", [])
    if isinstance(members, str):
        try:
            members = json.loads(members)
        except Exception:
            members = []
    return pd.DataFrame({"Member": range(1, len(members) + 1)})


def _manifest(root: Path) -> dict[str, Any]:
    return _read_json(root / "manifest.json")


def _procedure_table(sweep: Sweep, root: Path) -> pd.DataFrame:
    manifest = _manifest(root)
    ev = sweep.evaluation
    gate = sweep.gate
    data = (
        manifest.get("sweep", {}).get("data", {})
        if isinstance(manifest.get("sweep"), dict)
        else {}
    )
    classes = manifest.get("class_labels", getattr(sweep.data, "class_labels", ()))
    return pd.DataFrame(
        [
            ["Task", sweep.title],
            ["Experiment directory", str(root)],
            ["Data format", str(getattr(sweep.data, "format", data.get("format", "")))],
            ["Samples", str(manifest.get("n_samples", ""))],
            ["Classes", ", ".join(map(str, classes)) if classes else ""],
            ["Procedure", ev.protocol],
            [
                "Stratification",
                "target"
                if not getattr(sweep.data, "stratify_col", None)
                else f"target + {getattr(sweep.data, 'stratify_col')}",
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
                "MPMA-B and MPMA-E selected by inner-validation score; outer folds are reserved for final held-out performance estimation.",
            ],
        ],
        columns=["Field", "Value"],
    )


def _latex_escape(s: Any) -> str:
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _latex_tabular(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    align: str | None = None,
    already_latex_cols: set[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    already_latex_cols = already_latex_cols or set()
    if df.empty:
        path.write_text("% No rows available.\n", encoding="utf-8")
        return
    align = align or ("l" * len(df.columns))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\ifdefined\captionsetup\captionsetup{justification=justified,singlelinecheck=false}\fi",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\setlength{\tabcolsep}{2.5pt}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{_latex_escape(c)}}}" for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = row[c]
            if pd.isna(val):
                cells.append("")
            elif c in already_latex_cols:
                cells.append(str(val))
            else:
                cells.append(_latex_escape(val))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_task_name(title: str) -> str:
    title = str(title).replace("MPMA sweep", "").strip()
    parts = [x.strip() for x in title.split(" · ") if x.strip()]
    return (
        r"\\".join(_latex_escape(x) for x in parts) if parts else _latex_escape(title)
    )


def _strategy_latex_table(root: Path, path: Path, task_title: str) -> pd.DataFrame:
    disp = _strategy_performance_display(root, html_mode=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if disp.empty:
        path.write_text("% No performance rows available.\n", encoding="utf-8")
        return disp
    rows_by_strategy = {str(r["Strategy"]): r for _, r in disp.iterrows()}
    strategies = ["MPMA-E", "MPMA-B", "AutoML", "Baseline RF", "SIAMCAT"]
    metric_cols = ["ROC-AUC", "PR-AUC (AP)", "nMCC", "F1w", "Precision", "Recall"]

    def _cell(strategy: str, metric: str) -> str:
        row = rows_by_strategy.get(strategy)
        if row is None:
            return r"--"
        return str(row.get(metric, r"--")) or r"--"

    def _metric_block(metric: str) -> str:
        vals = [_cell(s, metric) for s in strategies]
        return r"\mllabiomemetricblock{" + "}{".join(vals) + "}"

    task = _latex_task_name(task_title)
    caption = (
        r"Held-out performance for the configured task. "
        r"MPMA-E = MPMAs Ensemble selected automatically by the framework based on inner validation score; "
        r"MPMA-B = single highest-scoring, based on inner validation, MPMA pipeline; "
        r"AutoML = FLAML automated model search on raw abundances at the deepest available single rank; "
        r"Baseline RF = 1000-tree random forest on arcsin-sqrt abundances at the deepest available single rank; "
        r"SIAMCAT = SIAMCAT workflow on the complete original taxonomic lineage without a mllabiome abundance transformation. "
        r"PR-AUC (AP) = average precision, the framework's precision-recall summary. "
        r"F1\textsubscript{w} = weighted F1. "
        r"Values are mean $\pm$ SD across outer evaluation units, followed by 95\% bootstrap confidence intervals for the mean. Bold = best mean per metric. All values are percentages."
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\ifdefined\captionsetup\captionsetup{justification=justified,singlelinecheck=false}\fi",
        rf"\caption{{{caption}}}",
        r"\label{tab:mllabiome-task-performance}",
        "",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\setlength{\tabcolsep}{1.2pt}",
        "",
        r"\makeatletter",
        r"\@ifundefined{mllabiometaskcol}{\newlength{\mllabiometaskcol}}{}",
        r"\@ifundefined{mllabiomestrategycol}{\newlength{\mllabiomestrategycol}}{}",
        r"\@ifundefined{mllabiomemetriccol}{\newlength{\mllabiomemetriccol}}{}",
        r"\makeatother",
        "",
        r"\setlength{\mllabiometaskcol}{3.00cm}",
        r"\setlength{\mllabiomestrategycol}{1.75cm}",
        r"\setlength{\mllabiomemetriccol}{%",
        r"  \dimexpr(\linewidth-\mllabiometaskcol-\mllabiomestrategycol-14\tabcolsep)/6\relax",
        r"}",
        "",
        r"\newcommand{\mllabiometaskblock}[1]{%",
        r"  \parbox[c]{\mllabiometaskcol}{\raggedright #1}%",
        r"}",
        "",
        r"\newcommand{\mllabiomestrategyblock}[5]{%",
        r"  \begin{tabular}[t]{l}",
        r"    #1\\",
        r"    #2\\",
        r"    #3\\",
        r"    #4\\",
        r"    #5",
        r"  \end{tabular}%",
        r"}",
        "",
        r"\newcommand{\mllabiomemetricblock}[5]{%",
        r"  \makebox[\mllabiomemetriccol][c]{%",
        r"    \begin{tabular}[t]{c}",
        r"      #1\\",
        r"      #2\\",
        r"      #3\\",
        r"      #4\\",
        r"      #5",
        r"    \end{tabular}%",
        r"  }%",
        r"}",
        "",
        r"\begin{tabular}{p{\mllabiometaskcol}p{\mllabiomestrategycol}p{\mllabiomemetriccol}p{\mllabiomemetriccol}p{\mllabiomemetriccol}p{\mllabiomemetriccol}p{\mllabiomemetriccol}p{\mllabiomemetriccol}}",
        "",
        r"\toprule",
        r"\multicolumn{1}{l}{\textbf{Task}} &",
        r"\multicolumn{1}{l}{\textbf{Strategy}} &",
        r"\multicolumn{1}{c}{\textbf{ROC-AUC}} &",
        r"\multicolumn{1}{c}{\textbf{PR-AUC}} &",
        r"\multicolumn{1}{c}{\textbf{nMCC}} &",
        r"\multicolumn{1}{c}{\textbf{F1\textsubscript{w}}} &",
        r"\multicolumn{1}{c}{\textbf{Precision}} &",
        r"\multicolumn{1}{c}{\textbf{Recall}} \\",
        r"\midrule",
        "",
        rf"\mllabiometaskblock{{{task}}}",
        r"&",
        r"\mllabiomestrategyblock{\mbox{MPMA-E}}{\mbox{MPMA-B}}{AutoML}{\mbox{Baseline RF}}{SIAMCAT}",
        r"&",
        _metric_block("ROC-AUC"),
        r"&",
        _metric_block("PR-AUC (AP)"),
        r"&",
        _metric_block("nMCC"),
        r"&",
        _metric_block("F1w"),
        r"&",
        _metric_block("Precision"),
        r"&",
        _metric_block("Recall") + r" \\",
        "",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return disp


def _top10_latex_table(root: Path, path: Path, task_title: str) -> pd.DataFrame:
    raw = _top_mpma_raw(root, 10)
    disp = _top_mpma_display(raw, html_mode=False)
    latex_cols = {
        c for c in disp.columns if c.startswith("Inner") or c.startswith("Outer")
    }
    caption = (
        f"Top 10 {task_title} MPMA configurations ranked by inner-validation nMCC. "
        "Inner-validation scores are used for model selection; outer-fold scores are reported only for final held-out performance estimation. "
        "All metric values are percentages and reported as mean $\\pm$ standard deviation."
    )
    align = "r" + "l" * max(0, len(disp.columns) - 1)
    _latex_tabular(
        disp,
        path,
        caption=caption,
        label="tab:mllabiome-top10-mpma",
        align=align[: len(disp.columns)],
        already_latex_cols=latex_cols,
    )
    return disp


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
    cols = list(df.columns)
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
                txt = html.escape(_html_inline(val))
            parts.append(f"<td>{txt}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _available_strategy_labels(root: Path) -> list[str]:
    labels: list[str] = []
    for row in _strategy_rows(root):
        label = str(row.get("Strategy", "")).strip()
        if not label:
            continue
        has_metric = False
        for metric, _ in _METRICS:
            m, _ = _mean_std_from_cols(row, "outer", metric)
            if np.isfinite(m):
                has_metric = True
                break
        if has_metric and label not in labels:
            labels.append(label)
    return labels


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


def _has_target_figure(target_dir: Path, stem: str) -> bool:
    fdir = target_dir / "figures"
    return any(
        (fdir / stem).with_suffix(ext).exists() for ext in (".svg", ".png", ".pdf")
    )


def _side_by_side_explainability_blocks(
    root: Path, report_dir: Path
) -> tuple[str, int]:
    target_dirs = _target_dirs_by_label(root)
    available_labels = _available_strategy_labels(root)
    preferred_order = [
        x for x in ("MPMA-E", "MPMA-B", "Baseline RF") if x in available_labels
    ]
    if len(preferred_order) < 2:
        preferred_order = [
            x for x in ("MPMA-E", "MPMA-B", "Baseline RF") if x in target_dirs
        ]
    if len(preferred_order) < 2:
        return "", 0
    figure_sets = (
        ("feature_support", "Feature support"),
        ("feature_support_shap", "SHAP feature support"),
        ("feature_support_lime", "LIME feature support"),
        ("feature_support_ale", "ALE feature support"),
        ("feature_support_permutation", "Permutation feature support"),
        ("ale_curves", "ALE curves"),
        ("interaction_network_current", "2D ALE interaction network"),
        ("instance_explanations_shap", "Instance-level SHAP explanations"),
    )
    blocks: list[str] = []
    n_blocks = 0
    for stem, title in figure_sets:
        if not any(
            label in target_dirs and _has_target_figure(target_dirs[label], stem)
            for label in preferred_order
        ):
            continue
        cells: list[str] = []
        for label in preferred_order:
            target_dir = target_dirs.get(label)
            fig_html = (
                _fig((target_dir / "figures" / stem), report_dir, "")
                if target_dir is not None
                else ""
            )
            if fig_html:
                body = fig_html
            else:
                body = '<div class="missing-figure">Not computed</div>'
            cells.append(
                f'<div class="compare-cell"><h3>{html.escape(label)}</h3>{body}</div>'
            )
        n_blocks += 1
        blocks.append(
            f'<section class="compare-block"><h3>{html.escape(title)}</h3><div class="compare-grid" style="--compare-columns:{len(preferred_order)}">{"".join(cells)}</div></section>'
        )
    if not blocks:
        return "", 0
    return "".join(blocks), n_blocks


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
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for _, row in view.iterrows():
        tab.add_row(*[_strip_cell_markup(row.get(col, "")) for col in view.columns])
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


def _short_feature_label(feature: Any, max_len: int = 46) -> str:
    text = str(feature)
    last = text.split("___")[-1]
    for pfx in ("s__", "g__", "f__", "o__", "c__", "p__", "d__", "t__"):
        if last.startswith(pfx):
            last = f"{pfx[0]}. " + last[len(pfx) :]
            break
    last = last.replace("_", " ").strip() or text
    return last if len(last) <= max_len else last[: max_len - 1].rstrip() + "…"


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
        path = target_dir / "top_features.tsv"
        if not path.exists():
            path = target_dir / "top_features_shap.tsv"
        tab = _read_tsv(path)
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
                    support = f"{float(vals.mean()):.2f}"
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
    top10_html: pd.DataFrame,
    ensemble_summary: pd.DataFrame,
    ensemble_members: pd.DataFrame,
) -> None:
    stage("Run summary", str(root))
    _terminal_table("Evaluation procedure", _compact_procedure_for_terminal(procedure))
    _terminal_table("Task performance", strategy_html)
    if not top10_html.empty:
        cols = [
            c
            for c in [
                "Rank",
                "Resolution",
                "MPDR transformation",
                "Learner",
                "Inner nMCC",
                "Outer nMCC",
                "Inner ROC-AUC",
                "Outer ROC-AUC",
            ]
            if c in top10_html.columns
        ]
        _terminal_table(
            "Top MPMA-B configurations",
            top10_html[cols] if cols else top10_html,
            max_rows=10,
        )
    _terminal_table("Selected MPMA-E", ensemble_summary)
    if not ensemble_members.empty:
        cols = [
            c
            for c in [
                "Member",
                "Resolution",
                "MPDR transformation",
                "Learner family",
                "count_transformation",
                "learner",
            ]
            if c in ensemble_members.columns
        ]
        _terminal_table(
            "MPMA-E members", ensemble_members[cols] if cols else ensemble_members
        )
    _feature_support_terminal(root, top_n=8)


def _tex_link(path: Path, report_dir: Path, label: str) -> str:
    if not path.exists():
        return ""
    return f'<p><a href="{html.escape(_rel(path, report_dir))}">{html.escape(label)}</a></p>'


def write_report(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    stage("Report", str(report_dir))

    procedure = _procedure_table(sweep, root)
    top10_html = _top_mpma_display(_top_mpma_raw(root, 10), html_mode=True)
    strategy_rows = _strategy_rows(root)
    statistics = run_report_statistics(
        root,
        sweep.evaluation.protocol,
        strategy_rows,
        n_bootstrap=2000,
        random_state=sweep.evaluation.random_state,
    )
    _strategy_latex_table(
        root, tables_dir / "task_strategy_performance.tex", sweep.title
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
                ).map(lambda x: f"{x:.4f}" if np.isfinite(x) else "")
        for c in ("p", "Holm p"):
            if c in primary_pairwise.columns:
                primary_pairwise[c] = pd.to_numeric(
                    primary_pairwise[c], errors="coerce"
                ).map(lambda x: f"{x:.4g}" if np.isfinite(x) else "")
    ensemble_summary = _ensemble_summary_table(root)
    ensemble_members = _ensemble_members_table(root)

    procedure.to_csv(tables_dir / "evaluation_procedure.tsv", sep="\t", index=False)
    top10_html.to_csv(
        tables_dir / "top10_mpma_inner_outer_performance.tsv", sep="\t", index=False
    )
    strategy_html.to_csv(
        tables_dir / "task_strategy_performance.tsv", sep="\t", index=False
    )
    primary_pairwise.to_csv(
        tables_dir / "strategy_pairwise_primary_metric.tsv", sep="\t", index=False
    )
    ensemble_summary.to_csv(tables_dir / "mpma_e_selection.tsv", sep="\t", index=False)
    ensemble_members.to_csv(tables_dir / "mpma_e_members.tsv", sep="\t", index=False)

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
    side_by_side_figs, side_by_side_count = _side_by_side_explainability_blocks(
        root, report_dir
    )
    exp_root = root / "explainability"
    if side_by_side_count == 0:
        for target_dir in sorted(exp_root.glob("*")):
            if not target_dir.is_dir():
                continue
            label = (
                "MPMA-B"
                if target_dir.name in {"mpma_b", "best_individual"}
                else target_dir.name
            )
            fdir = target_dir / "figures"
            for stem, cap in [
                ("feature_support", f"{label}: feature support"),
                ("feature_support_shap", f"{label}: SHAP feature support"),
                ("feature_support_lime", f"{label}: LIME feature support"),
                ("feature_support_ale", f"{label}: ALE feature support"),
                (
                    "feature_support_permutation",
                    f"{label}: permutation feature support",
                ),
                ("ale_curves", f"{label}: ALE curves"),
                ("interaction_network_current", f"{label}: 2D ALE interaction network"),
                (
                    "instance_explanations_shap",
                    f"{label}: instance-level SHAP explanations",
                ),
            ] + (
                [
                    (
                        "interaction_network_current_kamada_kawai",
                        f"{label}: 2D ALE interaction network (Kamada-Kawai)",
                    ),
                ]
                if bool(
                    getattr(sweep.explainability, "interaction_kamada_kawai", False)
                )
                else []
            ):
                block = _fig(fdir / stem, report_dir, cap)
                if block:
                    figs.append(block)

    css = """
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
.report-kicker { font-size:12px; color:var(--blue); font-weight:700; margin:0 0 7px; }
h1 { font-size:26px; line-height:1.16; margin:0 0 8px; font-weight:800; letter-spacing:-0.04em; }
h2 { font-size:16px; margin:38px 0 14px; font-weight:750; border-top:1px solid var(--track); padding-top:20px; letter-spacing:-0.025em; }
h2.first-section { margin-top:0; border-top:0; padding-top:0; }
h3 { font-size:13px; font-weight:700; margin:16px 0 10px; }
p, li { font-size:var(--font-body); color:var(--mid); line-height:1.56; }
.report-path { margin-top:0; }
figure { margin:18px 0 28px; overflow-x:auto; }
figure img { max-width:100%; width:auto; height:auto; display:block; }
.embedded-svg { overflow-x:auto; }
.embedded-svg svg { max-width:100%; height:auto; display:block; font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }
figcaption { font-size:var(--font-small); color:var(--mid); margin-top:7px; }
.compare-block { margin:17px 0 32px; overflow-x:auto; padding-bottom:2px; }
.compare-block > h3 { font-size:13px; margin:18px 0 11px; color:var(--ink); }
.compare-grid { display:grid; grid-template-columns:repeat(var(--compare-columns), minmax(430px, 1fr)); gap:16px; align-items:start; }
.compare-cell { min-width:0; }
.compare-cell h3 { font-size:12px; color:var(--mid); margin:0 0 7px; font-weight:700; }
.compare-cell figure { margin:0; }
.compare-cell figcaption { display:none; }
.compare-cell .embedded-svg svg { min-width:430px; }
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
  .compare-cell .embedded-svg svg { min-width:360px; }
}
"""
    top_metric_cols = {
        c for c in top10_html.columns if c.startswith("Inner") or c.startswith("Outer")
    }
    strat_metric_cols = {c for c in strategy_html.columns if c != "Strategy"}
    html_text = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>mllabiome report</title><link rel="icon" type="image/png" sizes="64x64" href="{_FAVICON_DATA_URI}"><link rel="shortcut icon" type="image/png" href="{_FAVICON_DATA_URI}"><style>{css}</style></head>
<body>
<header class="report-nav"><div class="report-nav-inner"><a class="brand" href="#top" aria-label="mllabiome report"><span class="brand-mark">mll</span><span>mllabiome</span></a><nav class="report-links" aria-label="Report navigation">
<a href="#procedure">Evaluation</a>
<a href="#performance">Performance</a>
<a href="#top-mpmas">MPMA-B</a>
<a href="#mpma-e">MPMA-E</a>
<a href="#mpma-e-members">Members</a>
<a href="#explainability-comparison">Explainability</a>
<a href="#figures">Figures</a>
<a href="#statistics">Statistics</a>
</nav></div></header>
<div class="report-shell"><main id="top" class="report-content">
<h2 id="procedure" class="first-section">Evaluation procedure</h2>
{_html_table(procedure)}
<h2 id="performance">Task performance summary</h2>
<p>Held-out strategy performance is reported as mean ± SD across outer evaluation units, followed by a 95% bootstrap confidence interval for the mean. For LODO, outer units are held-out datasets; for nested cross-validation, they are outer test folds. PR-AUC (AP) is average precision. All available framework metrics, bootstrap summaries, and pairwise tests are saved in the report tables.</p>
{_html_table(strategy_html, raw_html_cols=strat_metric_cols)}
{_tex_link(tables_dir / "task_strategy_performance.tex", report_dir, "task_strategy_performance.tex")}
<h3 id="statistics">Statistical comparisons</h3>
<p>Differences are Strategy A minus Strategy B. The table below shows the configured primary selection metric. Full pairwise results for ROC-AUC, PR-AUC (AP), nMCC, weighted F1, precision, and recall are saved in strategy_pairwise_tests.tsv. Holm adjustment is applied across strategy pairs within each metric.</p>
{_html_table(primary_pairwise)}
<h2 id="top-mpmas">Top 10 MPMA-B configurations</h2>
{_html_table(top10_html, raw_html_cols=top_metric_cols)}

<h2 id="mpma-e">Selected MPMA-E</h2>
{_html_table(ensemble_summary)}

<h2 id="mpma-e-members">Selected MPMA-E members</h2>
{_html_table(ensemble_members)}

<h2 id="explainability-comparison">Explainability comparison</h2>
{side_by_side_figs if side_by_side_figs else "<p>No side-by-side explainability comparison is available yet.</p>"}

<h2 id="figures">Global figures</h2>
{"".join(figs) if figs else "<p>No figure artefacts found yet.</p>"}
<p class="report-footer">mllabiome · generated report</p>
</main></div></body></html>
"""
    (report_dir / "index.html").write_text(html_text, encoding="utf-8")
    dump_json_standard(
        {
            "report_dir": report_dir,
            "figures_embedded": len([f for f in figs if f]),
            "side_by_side_explainability_blocks": side_by_side_count,
        },
        report_dir / "report_manifest.json",
    )
    _print_report_summary(
        sweep,
        root,
        procedure,
        strategy_html,
        top10_html,
        ensemble_summary,
        ensemble_members,
    )
    success("Report completed")
    outputs = {
        "html_report": report_dir / "index.html",
        "task_performance_latex": tables_dir / "task_strategy_performance.tex",
        "top10_mpmas_tsv": tables_dir / "top10_mpma_inner_outer_performance.tsv",
        "mpma_e_selection_tsv": tables_dir / "mpma_e_selection.tsv",
        "mpma_e_members_tsv": tables_dir / "mpma_e_members.tsv",
        "strategy_outer_unit_metrics": statistics.get(
            "unit_metrics_path", tables_dir / "strategy_outer_unit_metrics.tsv"
        ),
        "strategy_metrics_bootstrap": statistics.get(
            "summary_path", tables_dir / "strategy_metrics_bootstrap.tsv"
        ),
        "strategy_pairwise_tests": statistics.get(
            "pairwise_path", tables_dir / "strategy_pairwise_tests.tsv"
        ),
        "strategy_statistics_manifest": statistics.get(
            "manifest_path", tables_dir / "strategy_statistics_manifest.json"
        ),
    }
    path_table("Report outputs", outputs)
    return outputs
