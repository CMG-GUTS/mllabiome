from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class GaussianMOFAFusion:
    def __init__(
        self, n_components: int = 8, max_iter: int = 500, random_state: int = 42
    ):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    def fit(self, blocks: Mapping[str, np.ndarray]):
        try:
            from mofapy2.run.entry_point import entry_point
        except ImportError as exc:
            raise ImportError(
                "MOFA+ fusion requires mofapy2. Install it with 'uv sync --extra mofa'."
            ) from exc
        self.modalities_ = tuple(blocks)
        if len(self.modalities_) < 2:
            raise ValueError("MOFA+ integration requires at least two modalities.")
        if self.n_components < 1 or self.max_iter < 1:
            raise ValueError(
                "MOFA+ factors and training iterations must be positive integers."
            )
        arrays = [
            np.asarray(blocks[name], dtype=np.float64) for name in self.modalities_
        ]
        n_samples = arrays[0].shape[0]
        if n_samples < 4 or self.n_components >= n_samples:
            raise ValueError(
                "MOFA+ requires more training samples than requested factors."
            )
        for matrix in arrays:
            if matrix.ndim != 2 or matrix.shape[0] != n_samples or matrix.shape[1] < 2:
                raise ValueError(
                    "MOFA+ needs aligned, nonempty two-dimensional modality matrices."
                )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    "MOFA+ currently requires finite input after fold-specific preprocessing."
                )
        self.means_ = {
            name: matrix.mean(axis=0) for name, matrix in zip(self.modalities_, arrays)
        }
        centered = [
            matrix - self.means_[name] for name, matrix in zip(self.modalities_, arrays)
        ]
        ep = entry_point()
        ep.set_data_options(
            center_groups=False,
            scale_groups=False,
            scale_views=False,
            use_float32=False,
        )
        ep.set_data_matrix(
            data=[[matrix.copy()] for matrix in centered],
            likelihoods=["gaussian"] * len(centered),
            views_names=list(self.modalities_),
            groups_names=["training"],
        )
        ep.set_model_options(
            factors=self.n_components,
            spikeslab_factors=False,
            spikeslab_weights=True,
            ard_factors=False,
            ard_weights=True,
        )
        ep.set_train_options(
            iter=self.max_iter,
            convergence_mode="medium",
            seed=self.random_state,
            dropR2=None,
            quiet=True,
            verbose=False,
            gpu_mode=False,
        )
        ep.build()
        ep.run()
        weights = ep.model.nodes["W"].getExpectation()
        noise = ep.model.nodes["Tau"].getExpectation()
        if not isinstance(weights, (list, tuple)) or len(weights) != len(centered):
            raise RuntimeError("Unexpected mofapy2 W posterior format.")
        if not isinstance(noise, (list, tuple)) or len(noise) != len(centered):
            raise RuntimeError("Unexpected mofapy2 Tau posterior format.")
        self.loadings_ = {}
        self.precisions_ = {}
        k = None
        for name, matrix, raw_w, raw_tau in zip(
            self.modalities_, centered, weights, noise
        ):
            w = np.asarray(raw_w, dtype=float)
            if w.ndim != 2 or w.shape[0] != matrix.shape[1]:
                raise RuntimeError("Unexpected mofapy2 loading matrix shape.")
            if k is None:
                k = w.shape[1]
            if w.shape[1] != k:
                raise RuntimeError(
                    "Inconsistent MOFA+ latent dimensions across modalities."
                )
            tau = np.asarray(raw_tau, dtype=float)
            if tau.ndim == 2 and tau.shape[1] == matrix.shape[1]:
                tau = np.nanmean(tau, axis=0)
            elif tau.ndim == 1 and tau.shape[0] == matrix.shape[1]:
                pass
            else:
                raise RuntimeError("Unexpected mofapy2 Gaussian noise precision shape.")
            if (
                not np.all(np.isfinite(w))
                or not np.all(np.isfinite(tau))
                or np.any(tau <= 0)
            ):
                raise RuntimeError(
                    "MOFA+ did not return finite weights and positive noise precisions."
                )
            self.loadings_[name] = w
            self.precisions_[name] = tau
        precision = np.eye(k)
        rhs = []
        for name in self.modalities_:
            w = self.loadings_[name]
            tau = self.precisions_[name]
            precision += w.T @ (tau[:, None] * w)
            rhs.append(w.T * tau[None, :])
        self.projection_ = np.linalg.solve(precision, np.concatenate(rhs, axis=1)).T
        self.n_factors_ = k
        self.n_features_ = {
            name: block.shape[1] for name, block in zip(self.modalities_, centered)
        }
        return self

    def transform(self, blocks: Mapping[str, np.ndarray]) -> np.ndarray:
        if not hasattr(self, "projection_"):
            raise ValueError("MOFA+ fusion must be fitted before transforming samples.")
        if tuple(blocks) != self.modalities_:
            raise ValueError("Modality order must match fitted MOFA+ model.")
        matrices = []
        n_samples = None
        for name in self.modalities_:
            matrix = np.asarray(blocks[name], dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != self.n_features_[name]:
                raise ValueError(f"MOFA+ feature dimension mismatch for {name!r}.")
            if n_samples is not None and n_samples != len(matrix):
                raise ValueError("All modalities must have the same number of samples.")
            if not np.isfinite(matrix).all():
                raise ValueError("MOFA+ inference requires finite modality values.")
            n_samples = len(matrix)
            matrices.append(matrix - self.means_[name])
        return np.concatenate(matrices, axis=1) @ self.projection_
