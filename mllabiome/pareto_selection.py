from __future__ import annotations

import numpy as np
import pandas as pd


def _pareto_ranks_auroc_logloss(
    ranked: pd.DataFrame, auc_col: str, loss_col: str
) -> np.ndarray:
    loss_values = np.sort(ranked[loss_col].unique())
    tree = np.zeros(len(loss_values) + 1, dtype=np.int64)
    ranks = np.zeros(len(ranked), dtype=np.int64)
    previous = None
    previous_rank = 0
    for index, (auc, loss) in enumerate(
        zip(ranked[auc_col].to_numpy(), ranked[loss_col].to_numpy())
    ):
        pair = (float(auc), float(loss))
        if pair == previous:
            ranks[index] = previous_rank
            continue
        position = int(np.searchsorted(loss_values, loss)) + 1
        prefix = position
        depth = 1
        while prefix:
            depth = max(depth, int(tree[prefix]) + 1)
            prefix -= prefix & -prefix
        ranks[index] = depth
        previous = pair
        previous_rank = depth
        while position < len(tree):
            tree[position] = max(tree[position], depth)
            position += position & -position
    return ranks


def rank_pareto_auroc_logloss(
    frame: pd.DataFrame,
    auc_col: str = "AUROC",
    loss_col: str = "log_loss",
    id_col: str = "config_id",
) -> pd.DataFrame:
    missing = {auc_col, loss_col, id_col} - set(frame)
    if missing:
        raise ValueError(f"Pareto selection requires columns: {sorted(missing)}")
    ranked = frame.copy()
    for key in (auc_col, loss_col):
        ranked[key] = pd.to_numeric(ranked[key], errors="coerce")
    ranked = ranked.loc[
        np.isfinite(ranked[[auc_col, loss_col]].to_numpy(dtype=float)).all(axis=1)
    ].copy()
    if ranked.empty:
        ranked["pareto_optimal"] = pd.Series(dtype=bool)
        ranked["pareto_ideal_distance"] = pd.Series(dtype=float)
        ranked["pareto_rank"] = pd.Series(dtype=int)
        return ranked
    ranked[id_col] = ranked[id_col].astype(str)
    ranked = ranked.sort_values(
        [auc_col, loss_col, id_col],
        ascending=[False, True, True],
        kind="mergesort",
    )
    ranked["pareto_rank"] = _pareto_ranks_auroc_logloss(ranked, auc_col, loss_col)
    ranked["pareto_optimal"] = ranked["pareto_rank"].eq(1)
    frontier = ranked.loc[ranked["pareto_optimal"]].copy()
    auc_span = float(frontier[auc_col].max() - frontier[auc_col].min())
    loss_span = float(frontier[loss_col].max() - frontier[loss_col].min())
    auc_gap = (
        (float(frontier[auc_col].max()) - frontier[auc_col]) / auc_span
        if auc_span > 0
        else np.zeros(len(frontier))
    )
    loss_gap = (
        (frontier[loss_col] - float(frontier[loss_col].min())) / loss_span
        if loss_span > 0
        else np.zeros(len(frontier))
    )
    frontier["pareto_ideal_distance"] = np.hypot(auc_gap, loss_gap)
    frontier = frontier.sort_values(
        ["pareto_ideal_distance", loss_col, auc_col, id_col],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    dominated = ranked.loc[~ranked["pareto_optimal"]].copy()
    dominated["pareto_ideal_distance"] = np.nan
    dominated = dominated.sort_values(
        ["pareto_rank", auc_col, loss_col, id_col],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return pd.concat([frontier, dominated], ignore_index=True)


def rank_pareto_auroc_mcc(
    frame: pd.DataFrame,
    auc_col: str = "AUROC",
    mcc_col: str = "MCC",
    id_col: str = "config_id",
) -> pd.DataFrame:
    negative_col = "__pareto_negative_mcc__"
    if negative_col in frame.columns:
        raise ValueError("Reserved Pareto column already exists.")
    if mcc_col not in frame.columns:
        raise ValueError(f"Pareto selection requires column: {mcc_col}")
    transformed = frame.copy()
    transformed[negative_col] = -pd.to_numeric(transformed[mcc_col], errors="coerce")
    ranked = rank_pareto_auroc_logloss(transformed, auc_col, negative_col, id_col)
    return ranked.drop(columns=[negative_col])
