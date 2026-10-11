from __future__ import annotations

import numpy as np
import pandas as pd

ALPHAS = (0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)


def soften_probabilities(probabilities: np.ndarray, alpha: float) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    value = float(alpha)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2 or not 0.0 <= value < 1.0:
        raise ValueError("Invalid probability matrix or softening coefficient")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    return (1.0 - value) * probabilities + value / probabilities.shape[1]


def aligned_oof(frame: pd.DataFrame, candidate_ids: list[str]):
    columns = [
        str(column) for column in frame.columns if str(column).startswith("proba_")
    ]
    required = {"outer_split_key", "split_key", "sample_id", "config_id", "y_true"}
    if len(columns) < 2 or not required.issubset(frame.columns):
        raise ValueError("Incomplete OOF prediction schema")
    values = frame.copy()
    for name in ("outer_split_key", "split_key", "sample_id", "config_id"):
        values[name] = values[name].astype(str)
    keys = ["outer_split_key", "split_key", "sample_id"]
    reference = values[keys + ["y_true"]].drop_duplicates()
    if reference.duplicated(keys).any():
        raise ValueError("Conflicting inner-OOF labels")
    reference = reference.sort_values(keys).reset_index(drop=True)
    matrices = {}
    for config_id in candidate_ids:
        sub = values.loc[values["config_id"].eq(str(config_id)), keys + columns]
        if sub.duplicated(keys).any():
            raise ValueError(f"Repeated OOF prediction for {config_id}")
        aligned = reference[keys].merge(sub, on=keys, how="left", validate="one_to_one")
        if aligned[columns].isna().any().any():
            continue
        prediction = aligned[columns].to_numpy(dtype=float)
        if (
            not np.isfinite(prediction).all()
            or np.any(prediction < -1e-8)
            or np.any(prediction > 1 + 1e-8)
        ):
            continue
        if not np.allclose(prediction.sum(axis=1), 1.0, atol=1e-5):
            continue
        matrices[str(config_id)] = prediction
    if not matrices:
        raise ValueError("No complete OOF predictions available")
    folds = (reference["outer_split_key"] + "::" + reference["split_key"]).to_numpy(
        dtype=str
    )
    labels = reference["y_true"].to_numpy(dtype=int)
    if (labels < 0).any() or (labels >= len(columns)).any():
        raise ValueError("Inner OOF class alignment is invalid")
    return matrices, labels, folds


def fold_losses(
    probabilities: np.ndarray, labels: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    truth = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    losses = -np.log(truth)
    return np.asarray(
        [losses[folds == key].mean() for key in sorted(set(folds))], dtype=float
    )


def consensus_medoid(
    frame: pd.DataFrame, candidate_ids: list[str], size: int = 7
) -> str:
    matrices, labels, folds = aligned_oof(frame, candidate_ids)
    ranking = []
    for config_id, prediction in matrices.items():
        scores = fold_losses(prediction, labels, folds)
        ranking.append((float(scores.mean() + 0.5 * scores.std(ddof=0)), config_id))
    ranking.sort(key=lambda value: (value[0], value[1]))
    ids = [config_id for _, config_id in ranking[:size]]
    centroid = np.mean([matrices[config_id] for config_id in ids], axis=0)
    return min(
        ids,
        key=lambda config_id: (
            float(np.mean(0.5 * np.abs(matrices[config_id] - centroid).sum(axis=1))),
            ids.index(config_id),
            config_id,
        ),
    )


def choose_softening(frame: pd.DataFrame, config_id: str) -> float:
    matrices, labels, folds = aligned_oof(frame, [str(config_id)])
    probabilities = matrices[str(config_id)]
    return float(
        min(
            ALPHAS,
            key=lambda alpha: (
                float(
                    fold_losses(
                        soften_probabilities(probabilities, alpha), labels, folds
                    ).mean()
                ),
                float(alpha),
            ),
        )
    )


def rank_oof_loss(
    frame: pd.DataFrame, candidate_ids: list[str], robust: bool = False
) -> list[str]:
    matrices, labels, folds = aligned_oof(frame, candidate_ids)
    values = []
    for config_id, probability in matrices.items():
        if robust:
            losses = fold_losses(probability, labels, folds)
            score = float(losses.mean() + 0.5 * losses.std(ddof=0))
        else:
            score = float(
                -np.log(
                    np.clip(probability[np.arange(len(labels)), labels], 1e-12, 1)
                ).mean()
            )
        values.append((score, config_id))
    return [config_id for _, config_id in sorted(values)]


def similarity_members(
    stack: np.ndarray,
    candidate_ids: list[str],
    size: int,
    threshold: float = 0.025,
    pool: int = 30,
) -> list[str]:
    ids = candidate_ids[:pool]
    chosen = []
    indices = []
    for index, config_id in enumerate(ids):
        if (
            not indices
            or min(
                float(np.mean(0.5 * np.abs(stack[index] - stack[other]).sum(axis=1)))
                for other in indices
            )
            >= threshold
        ):
            indices.append(index)
            chosen.append(config_id)
        if len(chosen) >= size:
            return chosen
    for config_id in ids:
        if config_id not in chosen:
            chosen.append(config_id)
        if len(chosen) >= size:
            break
    return chosen
