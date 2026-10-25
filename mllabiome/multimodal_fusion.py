from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import euclidean_distances


def _array(value):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Fusion inputs must be finite two-dimensional arrays.")
    return result


def _centered(kernel):
    return (
        kernel
        - kernel.mean(axis=0, keepdims=True)
        - kernel.mean(axis=1, keepdims=True)
        + kernel.mean()
    )


def _rbf_cross(X, Y, bandwidth):
    return np.exp(-np.maximum(euclidean_distances(X, Y, squared=True), 0.0) / bandwidth)


def _bandwidth(X):
    distances = euclidean_distances(X, X, squared=True)
    upper = distances[np.triu_indices(len(X), k=1)]
    positive = upper[upper > 1e-14]
    return float(np.median(positive)) if len(positive) else 1.0


def _standardized_blocks(blocks):
    return {name: _array(matrix) for name, matrix in blocks.items()}


class AlignmentKernelFusion:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, blocks: Mapping[str, np.ndarray], y, task):
        matrices = _standardized_blocks(blocks)
        if len(matrices) < 2:
            raise ValueError("Kernel fusion requires at least two modalities.")
        self.modalities_ = tuple(matrices)
        n = next(iter(matrices.values())).shape[0]
        if n < 3 or any(X.shape[0] != n or X.shape[1] < 1 for X in matrices.values()):
            raise ValueError(
                "Kernel fusion requires aligned nonempty training modalities with at least three samples."
            )
        if n > 2000:
            raise ValueError(
                "Exact kernel alignment is limited to 2000 training samples due to quadratic memory and cubic eigendecomposition cost."
            )
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != n:
            raise ValueError(
                "Kernel alignment requires training outcomes aligned with the input rows."
            )
        if task == "classification":
            _, inverse = np.unique(target, return_inverse=True)
            if len(np.unique(inverse)) < 2:
                raise ValueError(
                    "Kernel alignment requires at least two training classes."
                )
            response = np.eye(int(inverse.max()) + 1)[inverse]
        elif task == "regression":
            response = np.asarray(target, dtype=float)[:, None]
            if not np.isfinite(response).all() or np.var(response) <= 1e-15:
                raise ValueError(
                    "Kernel alignment requires a nonconstant finite training outcome."
                )
        else:
            raise ValueError(
                "Kernel alignment requires an explicit classification or regression task."
            )
        Y = _centered(response @ response.T)
        norm_y = np.linalg.norm(Y)
        self.train_ = matrices
        self.bandwidths_ = {name: _bandwidth(X) for name, X in matrices.items()}
        raw_kernels = [
            _rbf_cross(X, X, self.bandwidths_[name]) for name, X in matrices.items()
        ]
        kernels = [_centered(K) for K in raw_kernels]
        norms = np.asarray([np.linalg.norm(K) for K in kernels])
        if not (norms > 1e-12).any():
            raise ValueError("All modality kernels are constant after centering.")
        self.norms_ = dict(zip(self.modalities_, np.maximum(norms, 1e-12)))
        stack = np.stack(
            [K / self.norms_[name] for K, name in zip(kernels, self.modalities_)]
        )
        similarities = np.einsum("aij,ij->a", stack, Y) / max(norm_y, 1e-12)
        inner = np.einsum("aij,bij->ab", stack, stack)

        def objective(weights):
            denominator = float(np.sqrt(max(weights @ inner @ weights, 1e-24)))
            return -float(weights @ similarities) / denominator

        result = minimize(
            objective,
            np.full(len(kernels), 1 / len(kernels)),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(kernels),
            constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
            options={"ftol": 1e-11, "maxiter": 500},
        )
        if result.success and np.isfinite(result.x).all() and result.x.sum() > 0:
            weights = np.maximum(result.x, 0)
            weights /= weights.sum()
        else:
            weights = np.zeros(len(kernels))
            weights[int(np.argmax(similarities))] = 1.0
        self.weights_ = dict(zip(self.modalities_, weights))
        fused = np.einsum("a,aij->ij", weights, stack)
        values, vectors = np.linalg.eigh((fused + fused.T) / 2)
        order = np.argsort(values)[::-1]
        keep = [j for j in order if values[j] > 1e-10]
        if not keep:
            raise ValueError("The learned fused kernel has no positive eigenvalues.")
        maximum = n - 1 if self.n_components is None else int(self.n_components)
        keep = keep[: min(maximum, n - 1)]
        self.eigenvalues_ = values[keep]
        self.eigenvectors_ = vectors[:, keep]
        self.training_means_ = {
            name: K.mean(axis=0) for name, K in zip(self.modalities_, raw_kernels)
        }
        self.training_grand_means_ = {
            name: float(K.mean()) for name, K in zip(self.modalities_, raw_kernels)
        }
        return self

    def transform(self, blocks):
        matrices = _standardized_blocks(blocks)
        if tuple(matrices) != self.modalities_:
            raise ValueError("Kernel fusion modalities must match fitted order.")
        cross = None
        for name, X in matrices.items():
            if X.shape[1] != self.train_[name].shape[1]:
                raise ValueError("Kernel fusion feature dimensions changed.")
            K = _rbf_cross(X, self.train_[name], self.bandwidths_[name])
            K -= K.mean(axis=1, keepdims=True)
            K -= self.training_means_[name][None, :]
            K += self.training_grand_means_[name]
            K *= self.weights_[name] / self.norms_[name]
            cross = K if cross is None else cross + K
        return cross @ (self.eigenvectors_ / np.sqrt(self.eigenvalues_)[None, :])


class BlockPLSFusion:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, blocks, y, task):
        matrices = _standardized_blocks(blocks)
        self.modalities_ = tuple(matrices)
        n = next(iter(matrices.values())).shape[0]
        if (
            len(matrices) < 2
            or n < 3
            or any(X.shape[0] != n or X.shape[1] == 0 for X in matrices.values())
        ):
            raise ValueError(
                "Supervised block PLS requires at least two aligned nonempty modalities and three samples."
            )
        self.means_ = {name: X.mean(axis=0) for name, X in matrices.items()}
        self.scales_ = {}
        for name, X in matrices.items():
            std = X.std(axis=0)
            self.scales_[name] = np.where(std > 1e-12, std, 1.0) * np.sqrt(X.shape[1])
        X = self._block_matrix(matrices)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != n:
            raise ValueError("Supervised block PLS requires aligned training outcomes.")
        if task == "classification":
            classes, indices = np.unique(target, return_inverse=True)
            if len(classes) < 2:
                raise ValueError("Supervised block PLS requires at least two classes.")
            Y = np.eye(len(classes))[indices]
        elif task == "regression":
            Y = np.asarray(target, dtype=float).reshape(-1, 1)
            if not np.isfinite(Y).all() or float(np.var(Y)) <= 1e-15:
                raise ValueError(
                    "Supervised block PLS requires a varying finite outcome."
                )
        else:
            raise ValueError("Supervised block PLS requires an explicit task.")
        limit = min(n - 1, X.shape[1])
        components = (
            limit if self.n_components is None else min(int(self.n_components), limit)
        )
        self.model_ = PLSRegression(
            n_components=components, scale=False, max_iter=1000, tol=1e-8
        )
        self.model_.fit(X, Y)
        return self

    def _block_matrix(self, blocks):
        if tuple(blocks) != self.modalities_:
            raise ValueError("Supervised block PLS modalities must match fitted order.")
        return np.concatenate(
            [
                (X - self.means_[name]) / self.scales_[name]
                for name, X in blocks.items()
            ],
            axis=1,
        )

    def transform(self, blocks):
        return np.asarray(
            self.model_.transform(self._block_matrix(_standardized_blocks(blocks))),
            dtype=float,
        )


class SparseCovarianceFusion:
    def __init__(self, n_components=None, sparsity=0.25):
        self.n_components = n_components
        self.sparsity = float(sparsity)

    def fit(self, blocks):
        matrices = _standardized_blocks(blocks)
        if len(matrices) != 2:
            raise ValueError(
                "Sparse cross-covariance fusion requires exactly two modalities."
            )
        if not 0 <= self.sparsity < 1:
            raise ValueError("Sparse cross-covariance threshold must lie in [0, 1).")
        self.modalities_ = tuple(matrices)
        n = next(iter(matrices.values())).shape[0]
        if n < 3 or any(X.shape[0] != n or X.shape[1] == 0 for X in matrices.values()):
            raise ValueError(
                "Sparse cross-covariance fusion requires aligned nonempty training blocks."
            )
        self.means_ = {name: X.mean(axis=0) for name, X in matrices.items()}
        self.scales_ = {
            name: np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
            for name, X in matrices.items()
        }
        A, B = [
            (X - self.means_[name]) / self.scales_[name] for name, X in matrices.items()
        ]
        cross = A.T @ B / (n - 1)
        limit = min(n - 1, min(cross.shape))
        count = (
            limit if self.n_components is None else min(limit, int(self.n_components))
        )
        left, right = [], []
        for _ in range(count):
            u, singular, vh = np.linalg.svd(cross, full_matrices=False)
            if singular[0] <= 1e-12:
                break
            a, b = u[:, 0], vh[0]
            for _ in range(200):
                av = cross @ b
                av = np.sign(av) * np.maximum(
                    np.abs(av) - self.sparsity * np.max(np.abs(av)), 0
                )
                if np.linalg.norm(av) <= 1e-12:
                    break
                av /= np.linalg.norm(av)
                bv = cross.T @ av
                bv = np.sign(bv) * np.maximum(
                    np.abs(bv) - self.sparsity * np.max(np.abs(bv)), 0
                )
                if np.linalg.norm(bv) <= 1e-12:
                    break
                bv /= np.linalg.norm(bv)
                if np.linalg.norm(av - a) + np.linalg.norm(bv - b) < 1e-8:
                    a, b = av, bv
                    break
                a, b = av, bv
            left.append(a)
            right.append(b)
            cross = cross - float(a @ cross @ b) * np.outer(a, b)
        if not left:
            raise ValueError(
                "Sparse cross-covariance fusion found no nonconstant components."
            )
        self.loadings_ = {
            self.modalities_[0]: np.column_stack(left),
            self.modalities_[1]: np.column_stack(right),
        }
        return self

    def transform(self, blocks):
        matrices = _standardized_blocks(blocks)
        if tuple(matrices) != self.modalities_:
            raise ValueError(
                "Sparse cross-covariance modalities must match fitted order."
            )
        return np.concatenate(
            [
                ((X - self.means_[name]) / self.scales_[name]) @ self.loadings_[name]
                for name, X in matrices.items()
            ],
            axis=1,
        )


class InductiveSNFFusion:
    def __init__(self, n_components=None, neighbors=10, iterations=15):
        self.n_components = n_components
        self.neighbors = int(neighbors)
        self.iterations = int(iterations)

    def fit(self, blocks):
        matrices = _standardized_blocks(blocks)
        self.modalities_ = tuple(matrices)
        n = next(iter(matrices.values())).shape[0]
        if (
            len(matrices) < 2
            or n < 4
            or any(X.shape[0] != n or X.shape[1] == 0 for X in matrices.values())
        ):
            raise ValueError(
                "Inductive SNF requires two or more aligned training modalities with at least four samples."
            )
        if self.neighbors < 2 or self.iterations < 1:
            raise ValueError("SNF neighbors must be >=2 and iterations must be >=1.")
        if n > 1200:
            raise ValueError(
                "Inductive SNF is limited to 1200 training samples due to dense graph diffusion cost."
            )
        k = min(self.neighbors, n - 1)
        self.train_ = matrices
        self.bandwidths_ = {name: _bandwidth(X) for name, X in matrices.items()}
        transitions = []
        neighborhoods = []
        for name, X in matrices.items():
            W = _rbf_cross(X, X, self.bandwidths_[name])
            Woff = W.copy()
            np.fill_diagonal(Woff, 0)
            transition = 0.5 * Woff / np.maximum(Woff.sum(axis=1, keepdims=True), 1e-12)
            np.fill_diagonal(transition, 0.5)
            selected = np.argpartition(W, -(k + 1), axis=1)[:, -(k + 1) :]
            S = np.zeros_like(W)
            np.put_along_axis(
                S, selected, np.take_along_axis(W, selected, axis=1), axis=1
            )
            S /= np.maximum(S.sum(axis=1, keepdims=True), 1e-12)
            transitions.append(transition)
            neighborhoods.append(S)
        m = len(transitions)
        for _ in range(self.iterations):
            combined = sum(transitions)
            updated = []
            for P, S in zip(transitions, neighborhoods):
                next_p = S @ ((combined - P) / (m - 1)) @ S.T
                next_p = np.maximum((next_p + next_p.T) / 2, 0)
                np.fill_diagonal(next_p, 0)
                next_p = (
                    0.5 * next_p / np.maximum(next_p.sum(axis=1, keepdims=True), 1e-12)
                )
                np.fill_diagonal(next_p, 0.5)
                updated.append(next_p)
            transitions = updated
        fused = sum(transitions) / m
        values, vectors = np.linalg.eigh((fused + fused.T) / 2)
        order = np.argsort(np.abs(values))[::-1]
        keep = [j for j in order if abs(values[j]) > 1e-12]
        if not keep:
            raise ValueError("Inductive SNF has no nonzero spectral components.")
        requested = n - 1 if self.n_components is None else int(self.n_components)
        self.reference_embedding_ = vectors[:, keep[: min(requested, n - 1)]]
        return self

    def transform(self, blocks):
        matrices = _standardized_blocks(blocks)
        if tuple(matrices) != self.modalities_:
            raise ValueError("Inductive SNF modalities must match fitted order.")
        projection = None
        for name, X in matrices.items():
            if X.shape[1] != self.train_[name].shape[1]:
                raise ValueError("Inductive SNF feature dimensions changed.")
            W = _rbf_cross(X, self.train_[name], self.bandwidths_[name])
            W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
            projection = W if projection is None else projection + W
        return (projection / len(self.modalities_)) @ self.reference_embedding_
