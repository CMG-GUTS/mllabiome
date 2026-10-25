from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .timeseries import BOLDPreprocessing


LABUS_ROIS = (
    "Tha",
    "Pu",
    "CaN",
    "Nacc",
    "Pal",
    "PosCG",
    "PosCS",
    "CS",
    "SbCG_S",
    "PRCG",
    "InfPrCS",
    "SupPrCs",
    "SupFG",
    "SupFS",
    "SupCirInS",
    "LoInG_CInS",
    "InfCirInS",
    "PosLS",
)
LABUS_PREPROCESSING = BOLDPreprocessing(
    high_pass=0.008,
    low_pass=0.08,
    standardize=True,
    confounds=(
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "white_matter",
        "csf",
        "a_comp_cor_00",
        "a_comp_cor_01",
        "a_comp_cor_02",
        "a_comp_cor_03",
        "a_comp_cor_04",
    ),
)


def labus_2019_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{metric}_{side}_{roi}"
        for metric in ("EigenvecCent", "NodeBWCent", "Strength")
        for side in ("L", "R")
        for roi in LABUS_ROIS
    )


def load_labus_2019_mapping(path: str | Path) -> dict[tuple[str, str], int]:
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = {"value", "side", "roi"}
    if not required.issubset(frame.columns) or len(frame) != 165:
        raise ValueError(
            "Labus labels TSV must contain 165 rows with value, side and roi columns"
        )
    values = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=int)
    if sorted(values.tolist()) != list(range(1, 166)):
        raise ValueError("Atlas labels must be consecutive integer values 1 to 165")
    mapping: dict[tuple[str, str], int] = {}
    for value, side, roi in zip(values, frame["side"], frame["roi"]):
        side, roi = str(side).strip(), str(roi).strip()
        if not side and not roi:
            continue
        if side not in {"L", "R"} or roi not in LABUS_ROIS:
            raise ValueError(f"Unknown Labus ROI specification {side!r}, {roi!r}")
        key = (side, roi)
        if key in mapping:
            raise ValueError(f"Repeated Labus ROI mapping: {key}")
        mapping[key] = int(value)
    expected = {(side, roi) for side in ("L", "R") for roi in LABUS_ROIS}
    if set(mapping) != expected:
        raise ValueError(
            f"Missing Labus ROI assignments: {sorted(expected - set(mapping))}"
        )
    return mapping


def labus_2019_centrality(
    time_series: np.ndarray,
    mapping: dict[tuple[str, str], int],
) -> np.ndarray:
    x = np.asarray(time_series, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 165 or x.shape[0] < 10 or not np.isfinite(x).all():
        raise ValueError(
            "Labus 2019 centrality requires finite time series of shape (timepoints, 165)"
        )
    if np.any(np.std(x, axis=0) < 1e-8):
        raise ValueError("All 165 anatomical parcels must have nonconstant signals")
    expected = {(side, roi) for side in ("L", "R") for roi in LABUS_ROIS}
    if (
        set(mapping) != expected
        or len(set(mapping.values())) != 36
        or any(not 1 <= v <= 165 for v in mapping.values())
    ):
        raise ValueError("Mapping must contain exactly 36 distinct anatomical ROIs")
    corr = np.corrcoef(x, rowvar=False)
    if not np.isfinite(corr).all():
        raise ValueError("Invalid correlation matrix")
    adjacency = np.clip((corr + corr.T) / 2, 0.0, 1.0)
    np.fill_diagonal(adjacency, 0.0)
    graph = nx.from_numpy_array(adjacency)
    for _, _, attr in graph.edges(data=True):
        attr["distance"] = 1.0 / attr["weight"]
    strength = adjacency.sum(axis=1)
    betweenness_map = nx.betweenness_centrality(
        graph, weight="distance", normalized=True
    )
    betweenness = np.array([betweenness_map[i] for i in range(165)], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(adjacency)
    if eigvals[-1] <= 1e-10 or eigvals[-1] - eigvals[-2] < 1e-10:
        raise ValueError("The leading adjacency eigenvector is not uniquely determined")
    eigenvector = eigvecs[:, -1]
    if eigenvector.sum() < 0:
        eigenvector = -eigenvector
    if eigenvector.min() < -1e-8:
        raise ValueError("Unexpected signed leading eigenvector of nonnegative graph")
    selected = np.array(
        [mapping[(side, roi)] - 1 for side in ("L", "R") for roi in LABUS_ROIS],
        dtype=int,
    )
    result = np.concatenate(
        (eigenvector[selected], betweenness[selected], strength[selected])
    )
    if result.shape != (108,) or not np.isfinite(result).all():
        raise ValueError("Invalid Labus 2019 output")
    return result
