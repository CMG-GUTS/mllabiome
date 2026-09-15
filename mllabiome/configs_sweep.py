from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .data import Data, Dataset, load_dataset
from .console import info, path_table, progress, stage, success, summary_table
from .figures import _write_representation_impact_figure
from .learners import _learner_factory, _learner_name
from .metrics import (
    _predict_proba_aligned as _metrics_predict_proba_aligned,
    compute_metrics,
)
from .resolutions import _parse_resolution, materialize_mpdr
from .utils import METRIC_COLUMNS, dump_json_standard
from .transformations import (
    TRANSFORMATION_LABELS,
    _count_transformation_factory,
    _count_transformation_name,
    _count_transformation_spec,
)


def _default_transformations():
    from .transformations import Transformation

    return (Transformation("arcsine_sqrt"),)


@dataclass(frozen=True)
class MPDR:
    resolution: str
    levels: tuple[str, ...]
    count_transformation: str


@dataclass(frozen=True)
class MPMA:
    config_id: str
    mpdr: MPDR
    learner: str


@dataclass
class QualificationGate:
    enabled: bool = False
    metric: str = "nMCC"
    threshold: float | None = None

    def qualifies(self, score: float) -> bool:
        if not self.enabled:
            return True
        if self.threshold is None:
            raise ValueError(
                "QualificationGate.threshold is required when the gate is enabled."
            )
        return np.isfinite(score) and float(score) >= float(self.threshold)


@dataclass
class Evaluation:
    protocol: Literal[
        "repeated_nested_cv", "nested_cv", "lodo", "leave_one_dataset_out"
    ] = "repeated_nested_cv"
    outer_folds: int = 5
    inner_folds: int = 3
    repeats: int = 2
    random_state: int = 42
    optimize_metric: str = "nMCC"
    n_jobs: int = 1
    redo: bool = False


@dataclass
class Ensemble:
    sizes: tuple[int, ...] = (3,)
    selection_strategies: tuple[str, ...] = (
        "top_k",
        "diverse_top_k",
        "best_per_family",
        "best_per_resolution",
        "threshold",
    )
    aggregation_strategies: tuple[str, ...] = (
        "mean_proba",
        "weighted_mean_proba",
        "median_proba",
        "rank_mean",
        "majority_vote",
    )
    optimize_metric: str = "nMCC"
    threshold_score: float = 0.30
    threshold_max_members: int = 50
    include_inactive: bool = False

    exclude_config_ids: tuple[str, ...] = ()
    exclude_learners: tuple[str, ...] = ()
    exclude_resolutions: tuple[str, ...] = ()
    exclude_transformations: tuple[str, ...] = ()


@dataclass(init=False)
class Explainability:
    targets: str | tuple[str, ...] = "auto"
    top_k: int = 30
    methods: tuple[str, ...] = ("shap", "lime", "ale", "permutation", "interactions")
    n_repeats: int = 10
    random_state: int = 42
    shap_background: int = 50
    shap_max_samples: int = 200
    lime_samples: int = 500
    lime_max_samples: int = 200
    ale_bins: int = 8
    top_k_interactions: int = 50
    interaction_kamada_kawai: bool = False
    representative_instances: bool = True
    instance_sample_ids: tuple[str, ...] = ()
    top_instance_features: int = 5

    def __init__(
        self,
        targets: str | Sequence[str] = "auto",
        top_k: int = 30,
        methods: Sequence[str] = ("shap", "lime", "ale", "permutation", "interactions"),
        n_repeats: int = 10,
        random_state: int = 42,
        shap_background: int = 50,
        shap_max_samples: int = 200,
        lime_samples: int = 500,
        lime_max_samples: int = 200,
        ale_bins: int = 8,
        top_k_interactions: int = 50,
        interaction_kamada_kawai: bool = False,
        representative_instances: bool = True,
        instance_sample_ids: Sequence[str] = (),
        top_instance_features: int = 5,
        **unknown_options: Any,
    ) -> None:
        if unknown_options:
            unknown = ", ".join(sorted(unknown_options))
            raise TypeError(
                f"Unknown Explainability option(s): {unknown}. Use Explainability(targets=...) to select explainability targets."
            )
        self.targets = _normalise_explainability_targets_config(targets)
        self.top_k = int(top_k)
        self.methods = tuple(str(m) for m in methods)
        self.n_repeats = int(n_repeats)
        self.random_state = int(random_state)
        self.shap_background = int(shap_background)
        self.shap_max_samples = int(shap_max_samples)
        self.lime_samples = int(lime_samples)
        self.lime_max_samples = int(lime_max_samples)
        self.ale_bins = int(ale_bins)
        self.top_k_interactions = int(top_k_interactions)
        self.interaction_kamada_kawai = bool(interaction_kamada_kawai)
        self.representative_instances = bool(representative_instances)
        self.instance_sample_ids = tuple(str(x) for x in instance_sample_ids)
        self.top_instance_features = int(top_instance_features)


def _normalise_explainability_targets_config(
    targets: str | Sequence[str],
) -> str | tuple[str, ...]:
    if isinstance(targets, str):
        text = targets.strip()
        return text or "auto"
    out = tuple(str(x).strip() for x in targets if str(x).strip())
    return out or "auto"


@dataclass
class Sweep:
    data: Data
    experiment_dir: Path | str
    resolutions: Sequence[tuple[str, Sequence[str]] | str] = field(
        default_factory=lambda: (("species", ("species",)),)
    )
    count_transformations: Sequence[Any] = field(
        default_factory=_default_transformations
    )
    learners: Sequence[Any] = field(default_factory=lambda: ("RF_1000_msl5",))
    evaluation: Evaluation = field(default_factory=Evaluation)
    gate: QualificationGate = field(default_factory=QualificationGate)
    ensemble: Ensemble = field(default_factory=Ensemble)
    explainability: Explainability = field(default_factory=Explainability)
    title: str = "mllabiome sweep"

    def root(self) -> Path:
        return Path(self.experiment_dir)


def build_sweep_from_module(mod: Any) -> Sweep:

    if hasattr(mod, "build_sweep"):
        obj = mod.build_sweep()
        if not isinstance(obj, Sweep):
            raise TypeError("build_sweep() must return mllabiome.Sweep.")
        return obj
    if hasattr(mod, "SWEEP"):
        obj = mod.SWEEP
        if not isinstance(obj, Sweep):
            raise TypeError("SWEEP must be an instance of mllabiome.Sweep.")
        return obj

    required = [
        "EXPERIMENT_DIR",
        "DATA",
        "_RESOLUTION_SETS",
        "_build_count_transformations",
        "_build_models",
    ]
    missing = [name for name in required if not hasattr(mod, name)]
    if missing:
        raise TypeError(
            "Config must define "
            + ", ".join(required)
            + f". Missing: {', '.join(missing)}."
        )
    data = getattr(mod, "DATA")
    if not isinstance(data, Data):
        raise TypeError("DATA must be an instance of mllabiome.Data(...).")
    return Sweep(
        title=getattr(mod, "TITLE", Path(getattr(mod, "EXPERIMENT_DIR")).name),
        experiment_dir=getattr(mod, "EXPERIMENT_DIR"),
        data=data,
        resolutions=getattr(mod, "_RESOLUTION_SETS"),
        count_transformations=mod._build_count_transformations(),
        learners=mod._build_models(),
        evaluation=getattr(mod, "EVALUATION", Evaluation()),
        gate=getattr(mod, "GATE", QualificationGate()),
        ensemble=getattr(mod, "ENSEMBLE", Ensemble()),
        explainability=getattr(mod, "EXPLAINABILITY", Explainability()),
    )


_MPDR_SEMANTICS = "select_then_transform_v1"


def _mpdr_id(count_transformation: str, resolution: str) -> str:
    return hashlib.sha1(
        f"{_MPDR_SEMANTICS}__{count_transformation}__{resolution}".encode()
    ).hexdigest()[:12]


def _config_id(count_transformation: str, resolution: str, learner_name: str) -> str:
    return hashlib.sha1(
        f"{_MPDR_SEMANTICS}__{count_transformation}__{resolution}__{learner_name}".encode()
    ).hexdigest()[:12]


def build_sweep_configs(
    resolutions: Sequence[Any],
    count_transformations: Sequence[Any],
    learners: Sequence[Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    learner_names = [_learner_name(x) for x in learners]
    for res in resolutions:
        res_name, levels = _parse_resolution(res)
        for ct in count_transformations:
            ct_name = _count_transformation_name(ct)
            for lname in learner_names:
                rows.append(
                    {
                        "config_id": _config_id(ct_name, res_name, lname),
                        "mpdr_id": _mpdr_id(ct_name, res_name),
                        "count_transformation": ct_name,
                        "resolution": res_name,
                        "levels": ",".join(levels),
                        "learner": lname,
                        "active": 1,
                    }
                )
    return pd.DataFrame(rows)


def evaluate(sweep: Sweep) -> dict[str, Path]:
    root = sweep.root()
    _prepare_dirs(root)

    resolutions = [_parse_resolution(r) for r in sweep.resolutions]
    all_levels = tuple(
        dict.fromkeys(
            lv
            for _, levels in resolutions
            for lv in levels
            if lv not in {"all", "features", "asis", "raw"}
        )
    )
    dataset = load_dataset(sweep.data, all_levels or ("all",))
    learner_factories = [_learner_factory(x) for x in sweep.learners]
    count_transformation_specs = [
        _count_transformation_spec(x) for x in sweep.count_transformations
    ]
    configs = build_sweep_configs(
        sweep.resolutions, sweep.count_transformations, sweep.learners
    )
    _write_config_table(root, configs)
    _write_manifest(root, sweep, dataset)

    y = dataset.y
    groups = _groups_from_metadata(dataset.metadata, sweep.data.group_col)
    strata = _strata_from_metadata(dataset.metadata, y, sweep.data.stratify_col)
    outer_splits = _outer_splits(
        sweep.evaluation, y, groups, strata, sweep.data.stratify_col
    )
    current_config_ids = set(configs["config_id"].astype(str))
    current_outer_keys = {str(s["split_key"]) for s in outer_splits}

    existing = _load_existing_evaluation(
        root, current_config_ids, current_outer_keys, redo=sweep.evaluation.redo
    )
    if not sweep.gate.enabled:
        existing["qualification"] = pd.DataFrame()
        qpath = root / "tables" / "qualification_gate.tsv"
        if qpath.exists():
            qpath.unlink()
    inner_done = _done_pairs(existing["inner_metrics"], "inner_key")
    outer_done = _done_pairs(existing["outer_metrics"], "split_key")
    qualification_map = (
        _qualification_map(existing["qualification"]) if sweep.gate.enabled else {}
    )

    outer_metric_rows: list[dict[str, Any]] = []
    inner_metric_rows: list[dict[str, Any]] = []
    outer_pred_rows: list[dict[str, Any]] = []
    inner_pred_rows: list[dict[str, Any]] = []
    qualification_rows: list[dict[str, Any]] = []

    expected_pairs = len(outer_splits) * len(configs)
    completed_pairs = sum(
        1
        for split in outer_splits
        for cid in current_config_ids
        if _outer_pair_complete(
            str(split["split_key"]), cid, outer_done, qualification_map
        )
    )

    stage("Configuration sweep", sweep.title)
    summary_table(
        "Sweep overview",
        {
            "samples": f"{len(y):,}",
            "classes": dataset.class_labels,
            "MPDRs": f"{len(sweep.resolutions) * len(count_transformation_specs):,}",
            "MPMAs": f"{len(configs):,}",
            "protocol": sweep.evaluation.protocol,
            "stratification": "target"
            if not sweep.data.stratify_col
            else f"target + {sweep.data.stratify_col}",
            "outer splits": f"{len(outer_splits):,}",
            "inner folds": sweep.evaluation.inner_folds,
            "gate": "on" if sweep.gate.enabled else "off",
            "completed MPMA/split pairs": f"{completed_pairs:,}/{expected_pairs:,}",
            "experiment dir": root,
        },
    )
    if completed_pairs and completed_pairs < expected_pairs:
        info(
            "Resuming sweep: completed MPMA/split pairs are kept; newly enabled or missing pairs will be evaluated."
        )
    elif completed_pairs == expected_pairs and expected_pairs > 0:
        success(
            "Current sweep configuration is already complete; no evaluation jobs to run."
        )
        _write_tables(
            root,
            outer_metric_rows,
            inner_metric_rows,
            outer_pred_rows,
            inner_pred_rows,
            qualification_rows,
            existing=existing,
            gate_enabled=sweep.gate.enabled,
        )
        _write_rankings_and_figures(
            root, dataset.class_labels, sweep.evaluation.optimize_metric
        )
        _write_representation_impact_figure(
            root, metric_col=sweep.evaluation.optimize_metric
        )
        path_table("Configuration sweep outputs", _existing_outputs(root))
        return _existing_outputs(root)

    t0 = time.perf_counter()
    total_mpma_units = max(1, expected_pairs)
    with progress() as prog:
        outer_task = prog.add_task("Outer splits", total=len(outer_splits))
        mpma_task = prog.add_task(
            "MPMA evaluations", total=total_mpma_units, completed=completed_pairs
        )
        for split_no, split in enumerate(outer_splits, start=1):
            split_key = str(split["split_key"])
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
                prog.advance(outer_task)
                continue
            inner_splits = _inner_splits(
                sweep.evaluation,
                y,
                train_idx,
                groups,
                split,
                strata,
                sweep.data.stratify_col,
            )
            inner_keys = [
                f"{split_key}__i{inner_no}" for inner_no, _ in enumerate(inner_splits)
            ]
            prog.update(
                outer_task,
                description=f"Outer split {split_no}/{len(outer_splits)} · {split_key}",
            )

            for res_name, levels in resolutions:
                X_base, _ = materialize_mpdr(dataset, levels)
                for ct_name, ct_spec in count_transformation_specs:
                    mpdr_cids = [
                        _config_id(str(ct_name), res_name, learner_name)
                        for learner_name, _ in learner_factories
                    ]
                    if (
                        inner_keys
                        and all(
                            (ik, cid) in inner_done
                            for ik in inner_keys
                            for cid in mpdr_cids
                        )
                        and all(
                            _outer_pair_complete(
                                split_key, cid, outer_done, qualification_map
                            )
                            for cid in mpdr_cids
                        )
                    ):
                        continue

                    _, ct_factory = (
                        _count_transformation_factory(
                            (ct_name, ct_spec),
                            random_state=sweep.evaluation.random_state,
                        )
                        if ct_spec is not None
                        else _count_transformation_factory(
                            ct_name, random_state=sweep.evaluation.random_state
                        )
                    )
                    mpdr = MPDR(
                        resolution=res_name,
                        levels=tuple(levels),
                        count_transformation=str(ct_name),
                    )
                    inner_scores: dict[str, list[float]] = {}
                    for learner_name, _ in learner_factories:
                        cid = _config_id(str(ct_name), res_name, learner_name)
                        inner_scores[learner_name] = _existing_inner_scores(
                            existing["inner_metrics"], split_key, cid, sweep.gate.metric
                        )

                    for inner_no, (inner_train_local, inner_val_local) in enumerate(
                        inner_splits
                    ):
                        inner_key = f"{split_key}__i{inner_no}"
                        tr_idx = train_idx[inner_train_local]
                        va_idx = train_idx[inner_val_local]
                        if len(np.unique(y[tr_idx])) < 2 or len(va_idx) == 0:
                            continue
                        missing_learners = [
                            (learner_name, factory)
                            for learner_name, factory in learner_factories
                            if (
                                inner_key,
                                _config_id(str(ct_name), res_name, learner_name),
                            )
                            not in inner_done
                        ]
                        if not missing_learners:
                            continue
                        _, inner_ct_factory = (
                            _count_transformation_factory(
                                (ct_name, ct_spec),
                                random_state=sweep.evaluation.random_state + inner_no,
                            )
                            if ct_spec is not None
                            else _count_transformation_factory(
                                ct_name,
                                random_state=sweep.evaluation.random_state + inner_no,
                            )
                        )
                        fitted = inner_ct_factory()
                        X_tr, X_va = fitted.apply_pair(X_base[tr_idx], X_base[va_idx])
                        for learner_name, factory in missing_learners:
                            cid = _config_id(str(ct_name), res_name, learner_name)
                            try:
                                clf = factory()
                                clf.fit(X_tr, y[tr_idx])
                                proba = _predict_proba_aligned(
                                    clf, X_va, dataset.classes
                                )
                                pred = dataset.classes[proba.argmax(axis=1)]
                                metrics = compute_metrics(
                                    y[va_idx], pred, proba, dataset.classes
                                )
                                row = _metric_row(
                                    metrics,
                                    split_key,
                                    inner_key,
                                    cid,
                                    mpdr,
                                    learner_name,
                                    "inner",
                                )
                                inner_metric_rows.append(row)
                                inner_scores[learner_name].append(
                                    float(metrics.get(sweep.gate.metric, np.nan))
                                )
                                inner_pred_rows.extend(
                                    _prediction_rows(
                                        inner_key,
                                        cid,
                                        va_idx,
                                        dataset,
                                        pred,
                                        proba,
                                        "inner",
                                        split_key,
                                    )
                                )
                            except Exception as exc:
                                inner_metric_rows.append(
                                    _failed_metric_row(
                                        split_key,
                                        inner_key,
                                        cid,
                                        mpdr,
                                        learner_name,
                                        "inner",
                                        exc,
                                    )
                                )

                    learners_needing_outer = [
                        (learner_name, factory)
                        for learner_name, factory in learner_factories
                        if not _outer_pair_complete(
                            split_key,
                            _config_id(str(ct_name), res_name, learner_name),
                            outer_done,
                            qualification_map,
                        )
                    ]
                    if not learners_needing_outer:
                        continue

                    fitted_outer = ct_factory()
                    X_train, X_test = fitted_outer.apply_pair(
                        X_base[train_idx], X_base[test_idx]
                    )

                    for learner_name, factory in learners_needing_outer:
                        cid = _config_id(str(ct_name), res_name, learner_name)
                        pair = (split_key, cid)
                        qrow = qualification_map.get(pair)
                        if not sweep.gate.enabled:
                            qualified = True
                        elif qrow is not None:
                            qualified = bool(int(qrow.get("qualified", 0)))
                            score = float(qrow.get("inner_score", np.nan))
                        else:
                            scores = [
                                x
                                for x in inner_scores.get(learner_name, [])
                                if np.isfinite(x)
                            ]
                            score = float(np.mean(scores)) if scores else float("nan")
                            qualified = sweep.gate.qualifies(score)
                            qrow_new = {
                                "split_key": split_key,
                                "config_id": cid,
                                "mpdr_id": _mpdr_id(ct_name, res_name),
                                "count_transformation": str(ct_name),
                                "resolution": res_name,
                                "levels": ",".join(levels),
                                "learner": learner_name,
                                "gate_enabled": 1,
                                "gate_metric": sweep.gate.metric,
                                "gate_threshold": sweep.gate.threshold,
                                "inner_score": score,
                                "qualified": int(qualified),
                            }
                            qualification_rows.append(qrow_new)
                            qualification_map[pair] = qrow_new
                        try:
                            if qualified and pair not in outer_done:
                                clf = factory()
                                clf.fit(X_train, y[train_idx])
                                proba = _predict_proba_aligned(
                                    clf, X_test, dataset.classes
                                )
                                pred = dataset.classes[proba.argmax(axis=1)]
                                metrics = compute_metrics(
                                    y[test_idx], pred, proba, dataset.classes
                                )
                                outer_metric_rows.append(
                                    _metric_row(
                                        metrics,
                                        split_key,
                                        None,
                                        cid,
                                        mpdr,
                                        learner_name,
                                        "outer",
                                    )
                                )
                                outer_pred_rows.extend(
                                    _prediction_rows(
                                        split_key,
                                        cid,
                                        test_idx,
                                        dataset,
                                        pred,
                                        proba,
                                        "outer",
                                        split_key,
                                    )
                                )
                        except Exception as exc:
                            outer_metric_rows.append(
                                _failed_metric_row(
                                    split_key,
                                    None,
                                    cid,
                                    mpdr,
                                    learner_name,
                                    "outer",
                                    exc,
                                )
                            )
                        finally:
                            prog.advance(mpma_task)
            prog.advance(outer_task)

    _write_tables(
        root,
        outer_metric_rows,
        inner_metric_rows,
        outer_pred_rows,
        inner_pred_rows,
        qualification_rows,
        existing=existing,
        gate_enabled=sweep.gate.enabled,
    )
    _write_rankings_and_figures(
        root, dataset.class_labels, sweep.evaluation.optimize_metric
    )
    _write_representation_impact_figure(
        root, metric_col=sweep.evaluation.optimize_metric
    )
    elapsed = time.perf_counter() - t0
    dump_json_standard(
        {
            "elapsed_s": elapsed,
            "n_outer_rows_added": len(outer_metric_rows),
            "n_inner_rows_added": len(inner_metric_rows),
            "n_qualification_rows_added": len(qualification_rows),
        },
        root / "run_summary.json",
    )
    success(
        f"Configuration sweep completed in {elapsed:.1f}s · added {len(outer_metric_rows):,} outer rows and {len(inner_metric_rows):,} inner rows"
    )
    outputs = _existing_outputs(root)
    path_table("Configuration sweep outputs", outputs)
    return outputs


def _prepare_dirs(root: Path) -> None:
    for d in [
        "results",
        "predictions",
        "inner_results",
        "inner_predictions",
        "tables",
        "figures",
        "ensembling",
        "explainability",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)


def _existing_outputs(root: Path) -> dict[str, Path]:
    return {
        "experiment_dir": root,
        "configs": root / "configs.tsv",
        "rankings": root / "tables" / "mpma_rankings.tsv",
        "outer_results": root / "results" / "outer_results.tsv",
        "inner_results": root / "inner_results" / "inner_results.tsv",
        "outer_predictions": root / "predictions" / "outer_predictions.tsv",
    }


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _filter_current(
    df: pd.DataFrame, current_config_ids: set[str], outer_keys: set[str]
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "config_id" in out.columns:
        out = out[out["config_id"].astype(str).isin(current_config_ids)]
    if "outer_split_key" in out.columns:
        out = out[out["outer_split_key"].astype(str).isin(outer_keys)]
    elif "split_key" in out.columns:
        split_s = out["split_key"].astype(str)
        out = out[
            split_s.isin(outer_keys)
            | split_s.str.rsplit("__i", n=1).str[0].isin(outer_keys)
        ]
    return out.reset_index(drop=True)


def _load_existing_evaluation(
    root: Path, current_config_ids: set[str], outer_keys: set[str], redo: bool = False
) -> dict[str, pd.DataFrame]:
    empty = {
        "outer_metrics": pd.DataFrame(),
        "inner_metrics": pd.DataFrame(),
        "outer_predictions": pd.DataFrame(),
        "inner_predictions": pd.DataFrame(),
        "qualification": pd.DataFrame(),
    }
    if redo:
        return empty
    tables = {
        "outer_metrics": root / "results" / "outer_results.tsv",
        "inner_metrics": root / "inner_results" / "inner_results.tsv",
        "outer_predictions": root / "predictions" / "outer_predictions.tsv",
        "inner_predictions": root / "inner_predictions" / "inner_predictions.tsv",
        "qualification": root / "tables" / "qualification_gate.tsv",
    }
    return {
        name: _filter_current(_read_tsv(path), current_config_ids, outer_keys)
        for name, path in tables.items()
    }


def _done_pairs(df: pd.DataFrame, split_col: str) -> set[tuple[str, str]]:
    if df.empty or split_col not in df.columns or "config_id" not in df.columns:
        return set()
    return set(zip(df[split_col].astype(str), df["config_id"].astype(str)))


def _qualification_map(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if df.empty or "split_key" not in df.columns or "config_id" not in df.columns:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        out[(str(row.get("split_key")), str(row.get("config_id")))] = row
    return out


def _outer_pair_complete(
    split_key: str,
    config_id: str,
    outer_done: set[tuple[str, str]],
    qualification_map: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    pair = (str(split_key), str(config_id))
    qrow = qualification_map.get(pair)
    if qrow is not None:
        try:
            if int(qrow.get("qualified", 0)) == 0:
                return True
        except Exception:
            pass
        return pair in outer_done
    return pair in outer_done


def _existing_inner_scores(
    df: pd.DataFrame, outer_split_key: str, config_id: str, metric: str
) -> list[float]:
    if (
        df.empty
        or metric not in df.columns
        or "config_id" not in df.columns
        or "split_key" not in df.columns
    ):
        return []
    sub = df[
        df["config_id"].astype(str).eq(str(config_id))
        & df["split_key"].astype(str).eq(str(outer_split_key))
    ]
    if "ok" in sub.columns:
        sub = sub[sub["ok"].eq(1)]
    vals = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
    return [float(x) for x in vals if np.isfinite(x)]


def _concat_existing_new(
    existing_df: pd.DataFrame, new_rows: list[dict[str, Any]], subset: list[str]
) -> pd.DataFrame:
    frames = []
    if existing_df is not None and not existing_df.empty:
        frames.append(existing_df)
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    keep_subset = [c for c in subset if c in out.columns]
    if keep_subset:
        out = out.drop_duplicates(subset=keep_subset, keep="last")
    return out


def _write_manifest(root: Path, sweep: Sweep, dataset: Dataset) -> None:
    safe_sweep = asdict(sweep)
    for section in ["data"]:
        for k, v in list(safe_sweep[section].items()):
            if isinstance(v, Path):
                safe_sweep[section][k] = str(v)
    safe_sweep["experiment_dir"] = str(safe_sweep["experiment_dir"])
    manifest = {
        "package": "mllabiome",
        "terminology": {
            "MPDR": "Microbiome Profile Data Representation = taxonomic resolution + count transformation",
            "MPMA": "Microbiome Profile Modelling Algorithm = MPDR + learner",
        },
        "sweep": safe_sweep,
        "class_labels": dataset.class_labels,
        "n_samples": len(dataset.y),
        "transformations": [label.key for label in TRANSFORMATION_LABELS],
        "mpdr_semantics": _MPDR_SEMANTICS,
    }
    dump_json_standard(manifest, root / "manifest.json")


def _write_config_table(root: Path, configs: pd.DataFrame) -> None:
    db_path = root / "configs.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS configs (
                config_id TEXT PRIMARY KEY,
                mpdr_id TEXT,
                count_transformation TEXT NOT NULL,
                resolution TEXT NOT NULL,
                levels TEXT NOT NULL,
                learner TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )"""
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(configs)")}
        if "transformation_abbreviation" in columns:
            conn.execute(
                """CREATE TABLE configs_canonical (
                    config_id TEXT PRIMARY KEY,
                    mpdr_id TEXT,
                    count_transformation TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    levels TEXT NOT NULL,
                    learner TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )"""
            )
            conn.execute(
                """INSERT OR REPLACE INTO configs_canonical
                   (config_id, mpdr_id, count_transformation, resolution, levels, learner, active)
                   SELECT config_id, mpdr_id, count_transformation, resolution, levels, learner, active FROM configs"""
            )
            conn.execute("DROP TABLE configs")
            conn.execute("ALTER TABLE configs_canonical RENAME TO configs")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS completions (
                config_id TEXT NOT NULL,
                split_key TEXT NOT NULL,
                stage TEXT NOT NULL,
                ok INTEGER NOT NULL,
                elapsed_s REAL,
                PRIMARY KEY (config_id, split_key, stage)
            )"""
        )
        conn.execute("UPDATE configs SET active=0")
        for r in configs.to_dict(orient="records"):
            conn.execute(
                """INSERT INTO configs
                   (config_id, mpdr_id, count_transformation, resolution, levels, learner, active)
                   VALUES (?, ?, ?, ?, ?, ?, 1)
                   ON CONFLICT(config_id) DO UPDATE SET
                     mpdr_id=excluded.mpdr_id,
                     count_transformation=excluded.count_transformation,
                     resolution=excluded.resolution,
                     levels=excluded.levels,
                     learner=excluded.learner,
                     active=1""",
                (
                    str(r.get("config_id")),
                    str(r.get("mpdr_id", "")),
                    str(r.get("count_transformation")),
                    str(r.get("resolution")),
                    str(r.get("levels")),
                    str(r.get("learner")),
                ),
            )
        conn.commit()
        export = pd.read_sql_query(
            "SELECT * FROM configs ORDER BY active DESC, count_transformation, resolution, learner",
            conn,
        )
    finally:
        conn.close()
    export.to_csv(root / "configs.tsv", sep="\t", index=False)


def _groups_from_metadata(
    meta: pd.DataFrame, group_col: str | None
) -> np.ndarray | None:
    if not group_col:
        return None
    if group_col not in meta.columns:
        raise ValueError(f"Group column {group_col!r} not found in metadata.")
    return meta[group_col].astype(str).to_numpy()


def _normalise_column_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None or value == "":
        return tuple()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value if str(v))


def _strata_from_metadata(
    meta: pd.DataFrame, y: np.ndarray, stratify_col: str | Sequence[str] | None
) -> np.ndarray:

    cols = _normalise_column_names(stratify_col)
    if not cols:
        return np.asarray(y, dtype=str)
    missing = [c for c in cols if c not in meta.columns]
    if missing:
        raise ValueError(f"Stratification column(s) not found in metadata: {missing}")
    base = pd.Series(np.asarray(y, dtype=str), index=meta.index).astype(str)
    parts = [base]
    for c in cols:
        vals = meta[c].astype(str).fillna("NA").reset_index(drop=True)
        parts.append(vals)
    joined = parts[0].reset_index(drop=True)
    for vals in parts[1:]:
        joined = joined.str.cat(vals.astype(str), sep="__strata__")
    return joined.to_numpy(dtype=str)


def _safe_n_splits(strata: np.ndarray, requested: int) -> int:
    counts = pd.Series(strata).value_counts()
    if counts.empty:
        return 0
    return int(max(0, min(requested, counts.min(), len(strata))))


def _stratification_error_context(
    plan: Evaluation, stratify_col: str | Sequence[str] | None
) -> str:
    cols = _normalise_column_names(stratify_col)
    if not cols:
        return "target labels"
    return "target labels plus " + ", ".join(cols)


def _outer_splits(
    plan: Evaluation,
    y: np.ndarray,
    groups: np.ndarray | None,
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    protocol = plan.protocol.lower()
    out: list[dict[str, Any]] = []
    if protocol in {"lodo", "leave_one_dataset_out"}:
        if groups is None:
            raise ValueError("LODO requires DATA.group_col.")
        for i, g in enumerate(pd.unique(groups)):
            test_idx = np.where(groups == g)[0]
            train_idx = np.where(groups != g)[0]
            out.append(
                {
                    "split_key": f"lodo_{i}__{g}",
                    "repeat": 0,
                    "outer_fold": i,
                    "outer_group": str(g),
                    "train_idx": train_idx,
                    "test_idx": test_idx,
                }
            )
        return out
    n_repeats = plan.repeats if protocol == "repeated_nested_cv" else 1
    split_strata = np.asarray(strata if strata is not None else y, dtype=str)
    for r in range(n_repeats):
        n_splits = _safe_n_splits(split_strata, plan.outer_folds)
        if n_splits < 2:
            context = _stratification_error_context(plan, stratify_col)
            raise ValueError(
                f"Not enough samples in each stratum for outer cross-validation using {context}. "
                "Reduce outer_folds, remove/merge sparse strata, or omit DATA.stratify_col."
            )
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=plan.random_state + r
        )
        for o, (tr, te) in enumerate(skf.split(np.zeros(len(y)), split_strata)):
            out.append(
                {
                    "split_key": f"r{r}_o{o}",
                    "repeat": r,
                    "outer_fold": o,
                    "train_idx": tr,
                    "test_idx": te,
                }
            )
    return out


def _inner_splits(
    plan: Evaluation,
    y: np.ndarray,
    outer_train_idx: np.ndarray,
    groups: np.ndarray | None,
    outer_split: dict[str, Any],
    strata: np.ndarray | None = None,
    stratify_col: str | Sequence[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if (
        plan.protocol.lower() in {"lodo", "leave_one_dataset_out"}
        and groups is not None
    ):
        g_train = groups[outer_train_idx]
        unique = list(pd.unique(g_train))
        if len(unique) >= 2:
            out = []
            for g in unique:
                va = np.where(g_train == g)[0]
                tr = np.where(g_train != g)[0]
                if len(np.unique(y[outer_train_idx[tr]])) >= 2:
                    out.append((tr, va))
            if out:
                return out
    y_train = y[outer_train_idx]
    split_strata = np.asarray(
        strata[outer_train_idx] if strata is not None else y_train, dtype=str
    )
    n_splits = _safe_n_splits(split_strata, plan.inner_folds)
    if n_splits < 2:
        return []
    seed = (
        plan.random_state
        + int(outer_split.get("repeat", 0)) * 1009
        + int(outer_split.get("outer_fold", 0))
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y_train)), split_strata)]


def _predict_proba_aligned(
    clf: BaseEstimator, X: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    return _metrics_predict_proba_aligned(clf, X, classes)


def _renormalize_proba(p: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    if p.shape[1] != n_classes:
        q = np.zeros((p.shape[0], n_classes), dtype=float)
        width = min(n_classes, p.shape[1])
        q[:, :width] = p[:, :width]
        p = q
    p = np.nan_to_num(
        p, nan=1.0 / n_classes, posinf=1.0 / n_classes, neginf=1.0 / n_classes
    )
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=1, keepdims=True)
    empty = s.squeeze() <= 1e-12
    s = np.where(s > 1e-12, s, 1.0)
    p = p / s
    if np.any(empty):
        p[empty, :] = 1.0 / n_classes
    return p


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = _renormalize_proba(y_proba, len(classes))
    out: dict[str, float] = {k: float("nan") for k in METRIC_COLUMNS}
    if len(y_true) == 0:
        return out
    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["BalAcc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["F1w"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["F1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["Precision"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    out["Recall"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    out["nMCC"] = float((matthews_corrcoef(y_true, y_pred) + 1.0) / 2.0)
    present = np.array([c for c in classes if c in set(y_true.tolist())], dtype=int)
    try:
        if len(classes) == 2:
            pos = classes[-1]
            pos_col = int(np.where(classes == pos)[0][0])
            yt = (y_true == pos).astype(int)
            if len(np.unique(yt)) == 2:
                out["AUC"] = float(roc_auc_score(yt, y_proba[:, pos_col]))
                out["PR_AUC"] = float(average_precision_score(yt, y_proba[:, pos_col]))
        elif len(present) >= 2:
            cols = [int(np.where(classes == c)[0][0]) for c in present]
            pp = _renormalize_proba(y_proba[:, cols], len(cols))
            out["AUC_macro"] = float(
                roc_auc_score(
                    y_true,
                    pp,
                    labels=present.tolist(),
                    multi_class="ovr",
                    average="macro",
                )
            )
            out["AUC_weighted"] = float(
                roc_auc_score(
                    y_true,
                    pp,
                    labels=present.tolist(),
                    multi_class="ovr",
                    average="weighted",
                )
            )
            out["AUC"] = out["AUC_macro"]
            out["PR_AUC_macro"] = float(
                average_precision_score(
                    pd.get_dummies(y_true).reindex(columns=present, fill_value=0),
                    pp,
                    average="macro",
                )
            )
    except Exception:
        pass
    return {
        k: (round(v, 6) if np.isfinite(v) else float("nan")) for k, v in out.items()
    }


def _metric_row(
    metrics: dict[str, float],
    split_key: str,
    inner_key: str | None,
    cid: str,
    mpdr: MPDR,
    learner: str,
    stage: str,
) -> dict[str, Any]:
    row = {
        "stage": stage,
        "split_key": split_key,
        "inner_key": inner_key or "",
        "config_id": cid,
        "mpdr_id": _mpdr_id(mpdr.count_transformation, mpdr.resolution),
        "count_transformation": mpdr.count_transformation,
        "resolution": mpdr.resolution,
        "levels": ",".join(mpdr.levels),
        "learner": learner,
        "ok": 1,
        "error": "",
    }
    row.update(metrics)
    return row


def _failed_metric_row(
    split_key: str,
    inner_key: str | None,
    cid: str,
    mpdr: MPDR,
    learner: str,
    stage: str,
    exc: Exception,
) -> dict[str, Any]:
    row = _metric_row(
        {k: float("nan") for k in METRIC_COLUMNS},
        split_key,
        inner_key,
        cid,
        mpdr,
        learner,
        stage,
    )
    row["ok"] = 0
    row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _prediction_rows(
    key: str,
    cid: str,
    idx: np.ndarray,
    dataset: Dataset,
    pred: np.ndarray,
    proba: np.ndarray,
    stage: str,
    outer_split_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_no, sample_idx in enumerate(idx):
        r = {
            "stage": stage,
            "split_key": key,
            "outer_split_key": outer_split_key,
            "sample_id": dataset.sample_ids[int(sample_idx)],
            "sample_index": int(sample_idx),
            "config_id": cid,
            "y_true": int(dataset.y[int(sample_idx)]),
            "y_pred": int(pred[row_no]),
        }
        for j, label in enumerate(dataset.class_labels):
            r[f"proba_{label}"] = float(proba[row_no, j])
        if len(dataset.class_labels) == 2:
            r["y_proba_pos"] = float(proba[row_no, 1])
        rows.append(r)
    return rows


def _write_tables(
    root: Path,
    outer_metrics: list[dict[str, Any]],
    inner_metrics: list[dict[str, Any]],
    outer_preds: list[dict[str, Any]],
    inner_preds: list[dict[str, Any]],
    qualification: list[dict[str, Any]],
    existing: dict[str, pd.DataFrame] | None = None,
    gate_enabled: bool = False,
) -> None:
    existing = existing or {}
    outer_df = _concat_existing_new(
        existing.get("outer_metrics", pd.DataFrame()),
        outer_metrics,
        ["split_key", "config_id"],
    )
    inner_df = _concat_existing_new(
        existing.get("inner_metrics", pd.DataFrame()),
        inner_metrics,
        ["inner_key", "config_id"],
    )
    outer_pred_df = _concat_existing_new(
        existing.get("outer_predictions", pd.DataFrame()),
        outer_preds,
        ["split_key", "config_id", "sample_id"],
    )
    inner_pred_df = _concat_existing_new(
        existing.get("inner_predictions", pd.DataFrame()),
        inner_preds,
        ["split_key", "config_id", "sample_id"],
    )
    qual_df = (
        _concat_existing_new(
            existing.get("qualification", pd.DataFrame()),
            qualification,
            ["split_key", "config_id"],
        )
        if gate_enabled
        else pd.DataFrame()
    )

    for frame in (outer_df, inner_df):
        if "transformation_abbreviation" in frame.columns:
            frame.drop(columns=["transformation_abbreviation"], inplace=True)
        if "count_transformation" in frame.columns:
            frame["count_transformation"] = frame["count_transformation"].map(
                _count_transformation_name
            )
    outer_df.to_csv(root / "results" / "outer_results.tsv", sep="\t", index=False)
    inner_df.to_csv(root / "inner_results" / "inner_results.tsv", sep="\t", index=False)
    outer_pred_df.to_csv(
        root / "predictions" / "outer_predictions.tsv", sep="\t", index=False
    )
    inner_pred_df.to_csv(
        root / "inner_predictions" / "inner_predictions.tsv", sep="\t", index=False
    )
    qpath = root / "tables" / "qualification_gate.tsv"
    if gate_enabled:
        qual_df.to_csv(qpath, sep="\t", index=False)
    elif qpath.exists():
        qpath.unlink()
    _update_completion_db(root, outer_df, inner_df)


def _update_completion_db(
    root: Path, outer_df: pd.DataFrame, inner_df: pd.DataFrame
) -> None:
    db_path = root / "configs.db"
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS completions (
                config_id TEXT NOT NULL,
                split_key TEXT NOT NULL,
                stage TEXT NOT NULL,
                ok INTEGER NOT NULL,
                elapsed_s REAL,
                PRIMARY KEY (config_id, split_key, stage)
            )"""
        )
        rows = []
        if not outer_df.empty and {"config_id", "split_key", "ok"}.issubset(
            outer_df.columns
        ):
            for r in (
                outer_df[["config_id", "split_key", "ok"]]
                .dropna(subset=["config_id", "split_key"])
                .to_dict(orient="records")
            ):
                rows.append(
                    (
                        str(r["config_id"]),
                        str(r["split_key"]),
                        "outer",
                        int(r.get("ok", 0)),
                        None,
                    )
                )
        if not inner_df.empty and {"config_id", "inner_key", "ok"}.issubset(
            inner_df.columns
        ):
            for r in (
                inner_df[["config_id", "inner_key", "ok"]]
                .dropna(subset=["config_id", "inner_key"])
                .to_dict(orient="records")
            ):
                rows.append(
                    (
                        str(r["config_id"]),
                        str(r["inner_key"]),
                        "inner",
                        int(r.get("ok", 0)),
                        None,
                    )
                )
        if rows:
            conn.executemany(
                """INSERT OR REPLACE INTO completions(config_id, split_key, stage, ok, elapsed_s)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
        conn.commit()
    finally:
        conn.close()


def _write_rankings_and_figures(
    root: Path, class_labels: list[str], optimize_metric: str
) -> None:
    path = root / "results" / "outer_results.tsv"
    if not path.exists() or path.stat().st_size == 0:
        return
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return
    metrics = [c for c in METRIC_COLUMNS if c in df.columns]
    group_cols = [
        "config_id",
        "mpdr_id",
        "count_transformation",
        "resolution",
        "levels",
        "learner",
    ]
    agg = (
        df[df["ok"].eq(1)]
        .groupby(group_cols, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
    )
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    rank = agg.reset_index()
    sort_col = (
        f"{optimize_metric}_mean"
        if f"{optimize_metric}_mean" in rank.columns
        else "nMCC_mean"
    )
    rank = rank.sort_values(sort_col, ascending=False)
    rank.insert(0, "rank", np.arange(1, len(rank) + 1))
    rank.to_csv(root / "tables" / "mpma_rankings.tsv", sep="\t", index=False)
    stale = root / "figures" / "mpma_top_metric.png"
    if stale.exists():
        stale.unlink()
