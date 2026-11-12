from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(source: Mapping[str, str] | str | Path | None) -> dict[str, str]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        result = {str(k).strip(): str(v).strip() for k, v in source.items()}
    else:
        path = Path(source).expanduser()
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            dialect = (
                "excel-tab" if path.suffix.lower() in {".tsv", ".txt"} else "excel"
            )
            reader = csv.DictReader(stream, dialect=dialect)
            if reader.fieldnames is None or not {"feature", "tip"}.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    "The taxon mapping must have 'feature' and 'tip' columns."
                )
            result = {}
            for row in reader:
                feature, tip = str(row["feature"]).strip(), str(row["tip"]).strip()
                if feature in result:
                    raise ValueError(
                        f"Duplicate feature in taxon mapping: {feature!r}."
                    )
                result[feature] = tip
    if any(not k or not v for k, v in result.items()):
        raise ValueError("Feature-to-tip mapping cannot contain blank labels.")
    return result


@lru_cache(maxsize=3)
def _cached_reference_tree(path: str, digest: str) -> Any:
    try:
        from Bio import Phylo
    except ImportError as exc:
        raise ImportError("GeneralizedUniFracPCoA requires biopython.") from exc
    return Phylo.read(path, "newick")


def _abundance_matrix(X: Any, n_features: int | None = None) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] < 1 or out.shape[1] < 1:
        raise ValueError(
            "Generalized UniFrac needs a nonempty 2D samples-by-features matrix."
        )
    if n_features is not None and out.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features; received {out.shape[1]}.")
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError("Generalized UniFrac requires finite, nonnegative abundances.")
    return out


class GeneralizedUniFracPCoA:
    name = "generalized_unifrac_pcoa"

    def __init__(
        self,
        tree_path: str | Path,
        *,
        feature_to_tip: Mapping[str, str] | str | Path | None = None,
        alpha: float = 0.5,
        n_components: int = 16,
        unmatched: str = "error",
        min_sample_coverage: float = 0.95,
        resolutions: Sequence[str] = ("species",),
        pair_batch_size: int = 128,
        feature_filter: Any = None,
    ) -> None:
        self.tree_path = str(Path(tree_path).expanduser().resolve())
        if not Path(self.tree_path).is_file():
            raise FileNotFoundError(
                f"Reference phylogenetic tree not found: {self.tree_path}"
            )
        self.tree_sha256 = _sha256_file(Path(self.tree_path))
        self.feature_to_tip = (
            dict(feature_to_tip)
            if isinstance(feature_to_tip, Mapping)
            else str(Path(feature_to_tip).expanduser().resolve())
            if feature_to_tip is not None
            else None
        )
        self.mapping_sha256 = (
            _sha256_file(Path(self.feature_to_tip))
            if isinstance(self.feature_to_tip, str)
            else None
        )
        self.alpha = float(alpha)
        self.n_components = int(n_components)
        self.unmatched = str(unmatched)
        self.min_sample_coverage = float(min_sample_coverage)
        self.resolutions = tuple(str(x) for x in resolutions)
        self.pair_batch_size = int(pair_batch_size)
        self.feature_filter = feature_filter
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0, 1].")
        if self.n_components < 1 or self.pair_batch_size < 1:
            raise ValueError(
                "n_components and pair_batch_size must be positive integers."
            )
        if self.unmatched not in {"error", "drop"}:
            raise ValueError("unmatched must be 'error' or 'drop'.")
        if (
            not np.isfinite(self.min_sample_coverage)
            or not 0 < self.min_sample_coverage <= 1
        ):
            raise ValueError("min_sample_coverage must lie in (0, 1].")
        if not self.resolutions:
            raise ValueError("resolutions must contain at least one supported rank.")
        mapping = _load_mapping(self.feature_to_tip)

        self.identity = (
            f"generalized_unifrac_pcoa_a{self.alpha:g}_k{self.n_components}"
            f"_{self.tree_sha256[:12]}"
        )
        if mapping:
            fingerprint = hashlib.sha256(
                repr(sorted(mapping.items())).encode("utf-8")
            ).hexdigest()
            self.identity += f"_map{fingerprint[:12]}"
        if self.unmatched != "error":
            self.identity += f"_drop{self.min_sample_coverage:g}"
        if feature_filter is not None:
            self.identity += f"|filter={feature_filter.identity}"
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None
        self.n_positive_axes_: int | None = None
        self.explained_positive_variance_: np.ndarray | None = None
        self.negative_inertia_fraction_: float | None = None
        self.matched_features_: tuple[str, ...] | None = None
        self.unmatched_features_: tuple[str, ...] | None = None
        self.train_coordinates_: np.ndarray | None = None

    def fresh(self) -> "GeneralizedUniFracPCoA":
        return type(self)(
            self.tree_path,
            feature_to_tip=self.feature_to_tip,
            alpha=self.alpha,
            n_components=self.n_components,
            unmatched=self.unmatched,
            min_sample_coverage=self.min_sample_coverage,
            resolutions=self.resolutions,
            pair_batch_size=self.pair_batch_size,
            feature_filter=self.feature_filter,
        )

    def bind_schema(
        self, feature_names: Sequence[str], *, feature_blocks: Any = None
    ) -> None:
        self.feature_names_ = tuple(str(x) for x in feature_names)
        if len(set(self.feature_names_)) != len(self.feature_names_):
            raise ValueError(
                "Dataset feature labels must be unique for phylogenetic mapping."
            )
        if not self.feature_names_:
            raise ValueError(
                "No feature names supplied for reference phylogeny mapping."
            )

        if feature_blocks is not None:
            ranks = {str(name) for name, indices in feature_blocks if indices}
            if len(ranks) > 1:
                raise ValueError(
                    "A single UniFrac tree must not mix taxonomic levels. "
                    "Configure a single resolution, e.g. 'species'."
                )

    def _build_branch_matrix(self) -> None:
        path = Path(self.tree_path)
        if _sha256_file(path) != self.tree_sha256:
            raise ValueError(
                "Reference tree content changed since the transform was configured."
            )
        if (
            isinstance(self.feature_to_tip, str)
            and _sha256_file(Path(self.feature_to_tip)) != self.mapping_sha256
        ):
            raise ValueError(
                "Feature-to-tip mapping changed since the transform was configured."
            )
        tree = _cached_reference_tree(str(path), self.tree_sha256)
        if len(tree.root.clades) < 2:
            raise ValueError(
                "The Newick reference must have a multifurcating/bifurcating root."
            )
        terminal_names = [str(clade.name) for clade in tree.get_terminals()]
        if not terminal_names or any(
            not name or name == "None" for name in terminal_names
        ):
            raise ValueError("Every reference-tree tip needs a nonblank, unique label.")
        if len(set(terminal_names)) != len(terminal_names):
            raise ValueError("Reference tree has duplicate tip labels.")
        tip_set = set(terminal_names)
        mapping = _load_mapping(self.feature_to_tip)
        labels = [mapping.get(name, name) for name in self.feature_names_]
        matched = np.array([name in tip_set for name in labels], dtype=bool)
        missing = tuple(
            name for name, ok in zip(self.feature_names_, matched) if not ok
        )
        if missing and self.unmatched == "error":
            raise ValueError(
                f"{len(missing)}/{len(labels)} dataset features cannot be matched to tree tips: "
                f"{missing[:8]!r}. Provide feature_to_tip or set unmatched='drop' "
                "and a defensible min_sample_coverage."
            )
        if not np.any(matched):
            raise ValueError("No dataset features match reference-tree tips.")
        self.matched_features_ = tuple(
            name for name, ok in zip(self.feature_names_, matched) if ok
        )
        self.unmatched_features_ = missing
        self._matched_mask_ = matched
        tip_columns: dict[str, list[int]] = {}
        for index, (tip, keep) in enumerate(zip(labels, matched)):
            if keep:
                tip_columns.setdefault(tip, []).append(index)
        rows: list[int] = []
        cols: list[int] = []
        branch_lengths: list[float] = []

        def traverse(clade: Any, *, root: bool = False) -> list[int]:
            if clade.is_terminal():
                indices = list(tip_columns.get(str(clade.name), ()))
            else:
                indices = []
                for child in clade.clades:
                    indices.extend(traverse(child))
            if not root:
                length = clade.branch_length
                if (
                    length is None
                    or not np.isfinite(float(length))
                    or float(length) < 0
                ):
                    raise ValueError(
                        "Every non-root reference-tree edge needs a finite, nonnegative branch length."
                    )
                if length > 0 and indices:
                    edge = len(branch_lengths)
                    branch_lengths.append(float(length))
                    rows.extend(indices)
                    cols.extend([edge] * len(indices))
            return indices

        traverse(tree.root, root=True)
        if not branch_lengths:
            raise ValueError(
                "No positive-length reference-tree edges cover the matched features."
            )
        self._incidence_ = csr_matrix(
            (np.ones(len(rows), dtype=float), (rows, cols)),
            shape=(len(labels), len(branch_lengths)),
            dtype=np.float64,
        )
        self._lengths_ = np.asarray(branch_lengths, dtype=np.float64)

    def _branch_abundances(self, X: np.ndarray) -> np.ndarray:
        values = _abundance_matrix(X, self.n_features_in_)
        original_totals = values.sum(axis=1)
        if np.any(original_totals <= 0):
            raise ValueError(
                "Generalized UniFrac cannot normalize zero-abundance samples."
            )
        kept_totals = values[:, self._matched_mask_].sum(axis=1)
        coverage = kept_totals / original_totals
        if np.any(coverage < self.min_sample_coverage - 1e-12):
            raise ValueError(
                f"Reference-tree mapping coverage below {self.min_sample_coverage:.1%} "
                f"for {int(np.sum(coverage < self.min_sample_coverage - 1e-12))} sample(s); "
                f"minimum observed coverage {coverage.min():.1%}."
            )
        if np.any(kept_totals <= 0):
            raise ValueError(
                "At least one sample has no abundance matching reference tree."
            )
        relative = values / kept_totals[:, None]
        return np.asarray(relative @ self._incidence_, dtype=np.float64)

    def _distances(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:

        out = np.empty((left.shape[0], right.shape[0]), dtype=float)
        lengths = self._lengths_[None, :]
        for i, single in enumerate(left):
            for start in range(0, right.shape[0], self.pair_batch_size):
                rhs = right[start : start + self.pair_batch_size]
                summed = single[None, :] + rhs
                ratios = np.divide(
                    np.abs(single[None, :] - rhs),
                    summed,
                    out=np.zeros_like(summed),
                    where=summed > 0,
                )
                weights = (
                    np.power(
                        summed,
                        self.alpha,
                        out=np.zeros_like(summed),
                        where=summed > 0,
                    )
                    * lengths
                )
                denominator = weights.sum(axis=1)
                numerator = (weights * ratios).sum(axis=1)
                out[i, start : start + len(rhs)] = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0,
                )
        return np.clip(out, 0.0, 1.0)

    def fit(self, X: Any, y: Any = None) -> "GeneralizedUniFracPCoA":
        values = _abundance_matrix(X)
        if values.shape[0] < 2:
            raise ValueError("PCoA needs at least two training samples.")
        if not hasattr(self, "feature_names_"):
            raise ValueError(
                "Call bind_schema(feature_names) before fitting; positional features are not phylogenetic identities."
            )
        if len(self.feature_names_) != values.shape[1]:
            raise ValueError("Feature-name count and abundance width disagree.")
        self.n_features_in_ = int(values.shape[1])
        self.n_features_out_ = self.n_components
        self._build_branch_matrix()
        self._train_branches_ = self._branch_abundances(values)
        distances = self._distances(self._train_branches_, self._train_branches_)

        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0.0)
        squared = distances**2
        self._train_row_means_ = squared.mean(axis=1)
        self._train_grand_mean_ = float(squared.mean())
        gram = -0.5 * (
            squared
            - self._train_row_means_[:, None]
            - self._train_row_means_[None, :]
            + self._train_grand_mean_
        )
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        total_abs = float(np.abs(eigenvalues).sum())
        self.negative_inertia_fraction_ = (
            float(-eigenvalues[eigenvalues < 0].sum()) / total_abs
            if total_abs > 0
            else 0.0
        )
        threshold = max(1e-12, float(eigenvalues[0]) * 1e-10)
        positive = np.flatnonzero(eigenvalues > threshold)[: self.n_components]
        self._eigenvalues_ = eigenvalues[positive]
        self._eigenvectors_ = eigenvectors[:, positive].copy()

        for column in range(len(positive)):
            pivot = np.argmax(np.abs(self._eigenvectors_[:, column]))
            if self._eigenvectors_[pivot, column] < 0:
                self._eigenvectors_[:, column] *= -1
        self.n_positive_axes_ = len(positive)
        self.explained_positive_variance_ = (
            eigenvalues[positive] / float(eigenvalues[eigenvalues > threshold].sum())
            if len(positive)
            else np.zeros(0, dtype=float)
        )
        self.train_coordinates_ = np.zeros(
            (len(values), self.n_components), dtype=float
        )
        if len(positive):
            self.train_coordinates_[:, : len(positive)] = (
                self._eigenvectors_ * np.sqrt(self._eigenvalues_)[None, :]
            )
        self._train_fingerprint_ = hashlib.sha256(
            np.ascontiguousarray(values).tobytes()
        ).digest()
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.train_coordinates_ is None:
            raise RuntimeError(
                "GeneralizedUniFracPCoA must be fitted on training data first."
            )
        query = self._branch_abundances(X)
        distances_sq = self._distances(query, self._train_branches_) ** 2
        centered_cross_gram = -0.5 * (
            distances_sq
            - distances_sq.mean(axis=1, keepdims=True)
            - self._train_row_means_[None, :]
            + self._train_grand_mean_
        )
        result = np.zeros((query.shape[0], self.n_components), dtype=float)
        if self.n_positive_axes_:
            result[:, : self.n_positive_axes_] = (
                centered_cross_gram @ self._eigenvectors_
            ) / np.sqrt(self._eigenvalues_)[None, :]
        return result

    def apply(self, X: Any) -> np.ndarray:
        return self.transform(X)

    def apply_fitted_pair(
        self, X_train: Any, X_test: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.train_coordinates_ is None:
            raise RuntimeError("GeneralizedUniFracPCoA must be fitted first.")
        train = _abundance_matrix(X_train, self.n_features_in_)
        if (
            train.shape[0] != self.train_coordinates_.shape[0]
            or hashlib.sha256(np.ascontiguousarray(train).tobytes()).digest()
            != self._train_fingerprint_
        ):
            raise ValueError("Training matrix differs from the one used to fit PCoA.")
        return self.train_coordinates_.copy(), self.transform(X_test)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        if self.train_coordinates_ is None:
            raise RuntimeError("GeneralizedUniFracPCoA must be fitted first.")
        return [
            f"generalized_unifrac_pcoa_{i + 1:03d}" for i in range(self.n_components)
        ]

    def coordinate_metadata(self, input_features: Any = None) -> list[Any]:
        from .transformations import TransformationCoordinate

        return [
            TransformationCoordinate(
                name=name,
                coordinate_type="nonlinear_phylogenetic_pcoa_axis",
                anchor_feature=None,
                components=(),
                coefficients=(),
                exact_feature_identity=False,
            )
            for name in self.get_feature_names_out()
        ]

    def perturbation_geometry(self) -> str:
        return "nonlinear_phylogenetic_ordination;no_exact_taxon_attribution"


def generalized_unifrac_distance_matrix(
    X: Any,
    *,
    feature_names: Sequence[str],
    tree_path: str | Path,
    feature_to_tip: Mapping[str, str] | str | Path | None = None,
    alpha: float = 0.5,
    unmatched: str = "error",
    min_sample_coverage: float = 0.95,
) -> np.ndarray:

    X = _abundance_matrix(X)
    geometry = GeneralizedUniFracPCoA(
        tree_path,
        feature_to_tip=feature_to_tip,
        alpha=alpha,
        unmatched=unmatched,
        min_sample_coverage=min_sample_coverage,
        resolutions=("species",),
    )
    geometry.bind_schema(feature_names)
    if len(feature_names) != X.shape[1]:
        raise ValueError("feature_names and abundance width disagree.")
    geometry.n_features_in_ = X.shape[1]
    geometry._build_branch_matrix()
    branches = geometry._branch_abundances(X)
    distances = geometry._distances(branches, branches)
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    return distances
