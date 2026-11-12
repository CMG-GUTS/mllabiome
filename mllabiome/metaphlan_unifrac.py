from __future__ import annotations

import bz2
import csv
import hashlib
import os
import re
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .taxonomy_features import (
    RANKS,
    lineage_matches,
    sgb_identifier,
    taxonomic_rank,
    taxonomic_tokens,
)


_REFERENCE_ROOT = "https://cmprod1.cibio.unitn.it/biobakery4/metaphlan_databases"
_SUPPORTED_REFERENCES = (
    "mpa_vJan26_CHOCOPhlAnSGB_202605",
    "mpa_vJan25_CHOCOPhlAnSGB_202503",
    "mpa_vJun23_CHOCOPhlAnSGB_202403",
    "mpa_vJun23_CHOCOPhlAnSGB_202307",
    "mpa_vOct22_CHOCOPhlAnSGB_202212",
    "mpa_vJan21_CHOCOPhlAnSGB_202103",
)
_REFERENCE_SGB = re.compile(r"^(?:SGB)?([0-9]+)(?:_group)?$", re.IGNORECASE)
_UNIFRAC_COUNT_SCALE = 1_000_000_000_000


def _canonical_sgb(value: Any) -> str | None:
    matched = _REFERENCE_SGB.fullmatch(str(value).strip())
    return f"SGB{int(matched.group(1))}" if matched is not None else None


def _valid_matrix(X: Any, n_features: int | None = None) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("UniFrac requires a nonempty samples-by-features matrix.")
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError(
            "UniFrac input columns do not match the training feature schema."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("UniFrac requires finite nonnegative abundances.")
    return values


def _encode_unifrac_counts(
    X: Any, *, metric: str, scale: int = _UNIFRAC_COUNT_SCALE
) -> np.ndarray:
    values = _valid_matrix(X)
    if metric == "unweighted_unifrac":
        return np.ascontiguousarray(values > 0, dtype=np.int64)
    if metric != "weighted_unifrac":
        raise ValueError(f"Unsupported UniFrac metric {metric!r}.")
    totals = values.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError(
            "Weighted UniFrac requires a positive finite abundance total per sample."
        )
    scaled = np.rint(values / totals * scale).astype(np.int64)
    np.maximum(scaled, (values > 0).astype(np.int64), out=scaled)
    return np.ascontiguousarray(scaled)


def _reference_file(
    version: str, cache_dir: str | Path | None = None, *, kind: str = "tree"
) -> Path:
    if version not in _SUPPORTED_REFERENCES:
        raise ValueError(f"Unknown MetaPhlAn reference version {version!r}.")
    suffix = (
        ".nwk" if kind == "tree" else "_species.txt.bz2" if kind == "taxonomy" else None
    )
    if suffix is None:
        raise ValueError(f"Unsupported reference resource {kind!r}.")
    root = Path(
        cache_dir
        if cache_dir is not None
        else os.environ.get("MLLABIOME_PHYLOGENY_CACHE", "~/.cache/mllabiome/phylogeny")
    ).expanduser()
    target = root / f"{version}{suffix}"
    if target.is_file() and target.stat().st_size > 1000:
        return target
    root.mkdir(parents=True, exist_ok=True)
    url = f"{_REFERENCE_ROOT}/{version}{suffix}"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{version}-", suffix=suffix, dir=root
    )
    try:
        with os.fdopen(descriptor, "wb") as destination:
            with urllib.request.urlopen(url, timeout=60) as source:
                while True:
                    data = source.read(1024 * 1024)
                    if not data:
                        break
                    destination.write(data)
        temporary = Path(temporary_name)
        if temporary.stat().st_size <= 1000:
            raise ValueError("Downloaded reference is empty or incomplete.")
        if kind == "tree" and not temporary.read_bytes().rstrip().endswith(b";"):
            raise ValueError("Downloaded reference is not a complete Newick tree.")
        if kind == "taxonomy":
            with bz2.open(temporary, "rt", encoding="utf-8") as source:
                if not source.readline().strip():
                    raise ValueError("Downloaded taxonomy mapping is empty.")
        os.replace(temporary, target)
    except Exception as exc:
        Path(temporary_name).unlink(missing_ok=True)
        raise RuntimeError(
            f"The official MetaPhlAn {kind} reference {version!r} could not be retrieved from {url}. "
            "Genuine phylogenetic UniFrac cannot be reconstructed from taxonomy labels alone."
        ) from exc
    return target


def _load_taxonomy(path: Path) -> dict[str, tuple[dict[str, str], ...]]:
    opener = bz2.open if path.suffix == ".bz2" else open
    mapping: dict[str, tuple[dict[str, str], ...]] = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n\r").split("\t", 1)
            if len(fields) < 2:
                continue
            match = _REFERENCE_SGB.fullmatch(fields[0].strip())
            if match is None:
                continue
            taxonomies = tuple(
                tokens
                for name in fields[1].split(",")
                if (tokens := taxonomic_tokens(name))
            )
            if taxonomies:
                key = _canonical_sgb(fields[0])
                if key in mapping and mapping[key] != taxonomies:
                    raise ValueError(
                        f"Conflicting reference taxonomy records for {key}."
                    )
                mapping[key] = taxonomies
    if not mapping:
        raise ValueError("Reference taxonomy mapping contains no SGB lineage records.")
    return mapping


def _resolve_named_taxa(
    names: tuple[str, ...],
    rank: str,
    taxonomy: dict[str, tuple[dict[str, str], ...]],
    tip_names: frozenset[str],
) -> tuple[dict[int, tuple[str, ...]], tuple[str, ...]]:
    by_feature: dict[int, tuple[str, ...]] = {}
    unusable: list[str] = []
    reference_by_name: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for sgb, lineage_candidates in taxonomy.items():
        if sgb not in tip_names:
            continue
        for lineage in lineage_candidates:
            label = lineage.get(rank)
            if label:
                reference_by_name.setdefault(label, []).append((sgb, lineage))
    for index, name in enumerate(names):
        if taxonomic_rank(name) != rank:
            continue
        tokens = taxonomic_tokens(name)
        found = tuple(
            sorted(
                {
                    sgb
                    for sgb, reference in reference_by_name.get(
                        tokens.get(rank, ""), ()
                    )
                    if lineage_matches(tokens, reference, rank)
                },
                key=lambda item: int(item[3:]),
            )
        )
        if found:
            by_feature[index] = found
        else:
            unusable.append(name)

    visited: dict[str, str] = {}
    for index, members in by_feature.items():
        for sgb in members:
            if sgb in visited:
                raise ValueError(
                    f"Taxonomic features {visited[sgb]!r} and {names[index]!r} overlap at {sgb}. "
                    "Phylogenetic abundance would be counted twice."
                )
            visited[sgb] = names[index]
    return by_feature, tuple(unusable)


def _filter_monophyletic_groups(
    tree: Any,
    groups: dict[int, tuple[str, ...]],
    *,
    policy: str,
) -> tuple[dict[int, tuple[str, ...]], tuple[int, ...]]:
    retained: dict[int, tuple[str, ...]] = {}
    excluded: list[int] = []
    for column, members in groups.items():
        if len(members) == 1:
            retained[column] = members
            continue
        ancestor = tree.lca(members)
        descendant_names = {str(node.name) for node in ancestor.tips()}
        if descendant_names == set(members):
            retained[column] = members
        elif policy == "error":
            raise ValueError(
                f"Taxon at feature column {column} is not monophyletic in the matching reference tree. "
                "Its abundance cannot be uniquely placed on a single phylogenetic branch."
            )
        else:
            excluded.append(column)
    return retained, tuple(excluded)


def _collapse_to_rank(
    tree: Any, groups: dict[int, tuple[str, ...]]
) -> tuple[Any, tuple[int, ...], tuple[str, ...]]:
    if len(groups) < 2:
        raise ValueError(
            f"Rank-level UniFrac requires at least two mapped taxa, but found {len(groups)}. "
            "Check that the reference version matches the MetaPhlAn profiling database "
            "and that species or genus labels match the SGB taxonomy reference."
        )
    wanted = set().union(*(set(members) for members in groups.values()))
    reduced = tree.shear(wanted)
    selected_columns = tuple(groups)
    identifiers: list[str] = []
    for index in selected_columns:
        group = groups[index]
        ancestor = reduced.lca(group) if len(group) > 1 else reduced.find(group[0])
        if {str(tip.name) for tip in ancestor.tips()} != set(group) and not (
            ancestor.is_tip() and {str(ancestor.name)} == set(group)
        ):
            raise ValueError(
                f"Taxon at feature column {index} is not monophyletic in the matching reference tree. "
                "Collapsing it would misrepresent evolutionary branch lengths."
            )
        for child in list(ancestor.children):
            ancestor.remove(child)
        label = f"phylo_rank_{index}"
        ancestor.name = label
        identifiers.append(label)
    if len(reduced.children) != 2:
        raise ValueError("Rank-collapsed phylogeny no longer has a bifurcating root.")
    return reduced, selected_columns, tuple(identifiers)


def _load_tree(path: Path, *, normalise_sgb: bool = True) -> Any:
    from skbio import TreeNode

    tree = TreeNode.read(str(path))
    if normalise_sgb:
        for tip in tree.tips():
            canonical = _canonical_sgb(tip.name)
            if canonical is not None:
                tip.name = canonical
    tips = [node.name for node in tree.tips()]
    if not tips or len(tips) != len(set(tips)) or any(not tip for tip in tips):
        raise ValueError("The reference tree must contain unique, nonempty tip names.")
    if len(tree.children) != 2:
        raise ValueError("The reference tree must be rooted with a bifurcating root.")
    for node in tree.postorder():
        if node is tree:
            continue
        if node.length is None or not np.isfinite(node.length) or node.length < 0:
            raise ValueError(
                "Reference tree branch lengths must be finite and nonnegative."
            )
    return tree, frozenset(str(tip) for tip in tips)


class MetaPhlAnUniFracPCoA:
    name = "unifrac"

    def __init__(
        self,
        metric: str = "weighted_unifrac",
        *,
        n_components: int = 16,
        normalized: bool = True,
        reference_version: str = "mpa_vJan26_CHOCOPhlAnSGB_202605",
        tree_path: str | Path | None = None,
        min_sample_coverage: float = 0.95,
        minimum_usable_coverage: float = 0.80,
        coverage_policy: str = "report",
        nonmonophyletic: str = "exclude",
        cache_dir: str | Path | None = None,
        feature_filter: Any = None,
        rank: str = "auto",
        taxonomy_path: str | Path | None = None,
        reference: str = "metaphlan",
        feature_to_tip: Any = None,
    ) -> None:
        self.metric = "weighted_unifrac" if metric == "unifrac" else str(metric)
        self.reference = str(reference).strip().casefold()
        if self.reference not in {"metaphlan", "silva", "custom"}:
            raise ValueError("reference must be metaphlan, silva, or custom.")
        self.feature_to_tip = feature_to_tip
        if self.metric not in {"weighted_unifrac", "unweighted_unifrac"}:
            raise ValueError(
                "UniFrac metric must be weighted_unifrac or unweighted_unifrac."
            )
        self.n_components = int(n_components)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")
        self.normalized = bool(normalized)
        self.reference_version = (
            "unverified"
            if self.reference != "metaphlan"
            and reference_version == "mpa_vJan26_CHOCOPhlAnSGB_202605"
            else str(reference_version)
        )
        if (
            self.reference == "metaphlan"
            and self.reference_version not in _SUPPORTED_REFERENCES
            and tree_path is None
        ):
            raise ValueError(
                f"Unsupported official MetaPhlAn tree version {reference_version!r}."
            )
        self.tree_path = (
            str(Path(tree_path).expanduser().resolve())
            if tree_path is not None
            else None
        )
        self.cache_dir = (
            str(Path(cache_dir).expanduser()) if cache_dir is not None else None
        )
        self.min_sample_coverage = float(min_sample_coverage)
        self.minimum_usable_coverage = float(minimum_usable_coverage)
        if (
            not np.isfinite(self.minimum_usable_coverage)
            or not 0 < self.minimum_usable_coverage <= 1
        ):
            raise ValueError("minimum_usable_coverage must be in (0, 1].")
        if self.minimum_usable_coverage > self.min_sample_coverage:
            raise ValueError(
                "minimum_usable_coverage cannot exceed min_sample_coverage."
            )
        self.coverage_policy = str(coverage_policy).strip().casefold()
        if self.coverage_policy not in {"report", "error"}:
            raise ValueError("coverage_policy must be 'report' or 'error'.")
        self.nonmonophyletic = str(nonmonophyletic).strip().casefold()
        if self.nonmonophyletic not in {"exclude", "error"}:
            raise ValueError("nonmonophyletic must be 'exclude' or 'error'.")
        if (
            not np.isfinite(self.min_sample_coverage)
            or not 0 < self.min_sample_coverage <= 1
        ):
            raise ValueError("min_sample_coverage must be in (0, 1].")
        self.rank = str(rank).strip().casefold()
        if self.rank not in {"auto", "feature", *RANKS}:
            raise ValueError(f"Unknown UniFrac taxonomic rank {rank!r}.")
        self.taxonomy_path = (
            str(Path(taxonomy_path).expanduser().resolve())
            if taxonomy_path is not None
            else None
        )
        self.resolutions = None
        self.feature_filter = feature_filter
        suffix = (
            f"{self.reference}-{self.reference_version}-rank{self.rank}"
            f"-k{self.n_components}-coverage{self.min_sample_coverage:g}"
            f"-usable{self.minimum_usable_coverage:g}"
            f"-coveragepolicy{self.coverage_policy}-nonmono{self.nonmonophyletic}"
        )
        if self.tree_path is not None:
            suffix += f"-local{hashlib.sha256(Path(self.tree_path).read_bytes()).hexdigest()[:16]}"
        if self.cache_dir is not None:
            suffix += f"-cache{hashlib.sha256(self.cache_dir.encode()).hexdigest()[:8]}"
        if self.taxonomy_path is not None:
            suffix += f"-taxonomy{hashlib.sha256(Path(self.taxonomy_path).read_bytes()).hexdigest()[:16]}"
        if self.feature_to_tip is not None:
            if isinstance(self.feature_to_tip, (str, Path)):
                identity_bytes = Path(self.feature_to_tip).read_bytes()
            else:
                identity_bytes = repr(
                    sorted((str(k), str(v)) for k, v in self.feature_to_tip.items())
                ).encode()
            suffix += f"-mapping{hashlib.sha256(identity_bytes).hexdigest()[:16]}"
        if self.metric == "weighted_unifrac":
            suffix += f"-normalized{int(self.normalized)}"
        self.identity = f"{self.metric}_pcoa_{suffix}_skbio_v6"
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None
        self.n_positive_axes_: int | None = None
        self.negative_inertia_fraction_: float | None = None

    def fresh(self) -> MetaPhlAnUniFracPCoA:
        return type(self)(
            metric=self.metric,
            n_components=self.n_components,
            normalized=self.normalized,
            reference_version=self.reference_version,
            tree_path=self.tree_path,
            min_sample_coverage=self.min_sample_coverage,
            minimum_usable_coverage=self.minimum_usable_coverage,
            coverage_policy=self.coverage_policy,
            nonmonophyletic=self.nonmonophyletic,
            cache_dir=self.cache_dir,
            feature_filter=self.feature_filter,
            rank=self.rank,
            taxonomy_path=self.taxonomy_path,
            reference=self.reference,
            feature_to_tip=self.feature_to_tip,
        )

    def bind_schema(self, feature_names: Any, *, feature_blocks: Any = None) -> None:
        self.feature_names_ = tuple(str(name) for name in feature_names)
        if not self.feature_names_ or len(set(self.feature_names_)) != len(
            self.feature_names_
        ):
            raise ValueError("UniFrac requires unique, nonempty feature names.")
        levels = {
            level
            for name in self.feature_names_
            if (level := taxonomic_rank(name)) is not None
        }
        sgb_columns = tuple(
            index
            for index, name in enumerate(self.feature_names_)
            if sgb_identifier(name) is not None
        )
        if self.rank == "auto":
            self.selected_rank_ = (
                "feature"
                if self.reference != "metaphlan"
                else (
                    "strain"
                    if sgb_columns
                    else next(
                        (level for level in reversed(RANKS[:-1]) if level in levels),
                        None,
                    )
                )
            )
        else:
            self.selected_rank_ = self.rank
        if self.selected_rank_ is None:
            raise ValueError(
                "UniFrac requires recognizable taxonomic lineage identifiers."
            )
        if self.reference != "metaphlan" and self.selected_rank_ != "feature":
            raise ValueError(
                "SILVA and custom trees require sequence or reference tip identifiers; a named taxonomic rank alone does not establish phylogenetic placement. Use rank='feature'."
            )
        if self.reference == "metaphlan" and self.selected_rank_ == "feature":
            raise ValueError(
                "MetaPhlAn reference requires SGB or taxonomic-rank features."
            )
        if (
            self.reference != "metaphlan"
            and all(taxonomic_rank(name) is not None for name in self.feature_names_)
            and self.feature_to_tip is None
        ):
            raise ValueError(
                "16S genus/species taxonomy labels alone cannot determine phylogenetic sequence tips. Provide ASV/OTU identifiers and their tree, or a validated feature_to_tip mapping."
            )
        if self.selected_rank_ == "strain" and not sgb_columns:
            raise ValueError(
                "The SGB-level reference requires t__SGB identifiers; other strain names cannot be matched."
            )
        self._selected_columns_ = np.asarray(
            tuple(range(len(self.feature_names_)))
            if self.selected_rank_ == "feature"
            else sgb_columns
            if self.selected_rank_ == "strain"
            else tuple(
                index
                for index, name in enumerate(self.feature_names_)
                if taxonomic_rank(name) == self.selected_rank_
            ),
            dtype=int,
        )
        if not len(self._selected_columns_):
            raise ValueError(
                f"No abundance columns were found at rank {self.selected_rank_!r}."
            )
        self._domain_columns_ = np.asarray(
            [
                i
                for i, name in enumerate(self.feature_names_)
                if taxonomic_rank(name) == "domain"
            ],
            dtype=int,
        )
        self._sgb_ids_ = tuple(
            sgb_identifier(self.feature_names_[index]) for index in sgb_columns
        )
        if len(self._sgb_ids_) != len(set(self._sgb_ids_)):
            raise ValueError(
                "Duplicate SGB identifiers would double-count phylogenetic abundances."
            )

    def _load_feature_tip_mapping(self) -> dict[str, str]:
        if self.feature_to_tip is None:
            return {}
        if isinstance(self.feature_to_tip, dict):
            return {str(key): str(value) for key, value in self.feature_to_tip.items()}
        path = Path(self.feature_to_tip)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature-to-tip mapping: {path}")
        result: dict[str, str] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(
                stream,
                delimiter="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",",
            )
            for row in reader:
                if len(row) != 2:
                    raise ValueError(
                        "Feature-to-tip mapping needs exactly two columns."
                    )
                feature, tip = (value.strip() for value in row)
                if feature.lower() in {
                    "feature",
                    "feature_id",
                    "asv",
                } and tip.lower() in {"tip", "tip_id", "reference_id"}:
                    continue
                if not feature or not tip or feature in result:
                    raise ValueError(
                        "Feature-to-tip mappings must have unique, nonempty feature IDs and tip IDs."
                    )
                result[feature] = tip
        return result

    def _extract(self, X: Any) -> np.ndarray:
        raw = _valid_matrix(X, self.n_features_in_)
        selected = raw[:, self._matched_columns_]
        denominator = raw[:, self._selected_columns_].sum(axis=1)
        numerator = selected.sum(axis=1)
        if (denominator <= 0).any():
            raise ValueError(
                f"At least one sample has zero {self.selected_rank_} abundance."
            )
        if (numerator <= 0).any():
            raise ValueError(
                "At least one sample has zero reference-mapped abundance. "
                "UniFrac on the reference-covered subcommunity is undefined for that sample."
            )
        coverage = numerator / denominator
        observed_total = np.count_nonzero(raw[:, self._selected_columns_] > 0, axis=1)
        observed_matched = np.count_nonzero(selected > 0, axis=1)
        presence_coverage = observed_matched / np.maximum(observed_total, 1)
        summary = {
            "n_samples": int(len(coverage)),
            "minimum": float(coverage.min()),
            "minimum_presence": float(presence_coverage.min()),
            "median_presence": float(np.median(presence_coverage)),
            "median": float(np.median(coverage)),
            "mean": float(coverage.mean()),
            "maximum": float(coverage.max()),
            "below_threshold": int(
                np.count_nonzero(coverage < self.min_sample_coverage - 1e-12)
            ),
            "below_presence_threshold": int(
                np.count_nonzero(presence_coverage < self.min_sample_coverage - 1e-12)
            ),
            "threshold": self.min_sample_coverage,
            "minimum_usable_coverage": self.minimum_usable_coverage,
            "rank": self.selected_rank_,
            "reference_version": self.reference_version,
            "nonmonophyletic_taxa_excluded": len(
                getattr(self, "nonmonophyletic_taxa_", ())
            ),
            "unmatched_taxa": len(getattr(self, "unmatched_taxa_", ())),
            "representation": "unifrac_on_reference_covered_subcommunity",
        }
        self.last_coverage_summary_ = summary
        if (
            self.reference == "metaphlan"
            and self.selected_rank_ != "domain"
            and len(self._domain_columns_)
        ):
            domain_total = raw[:, self._domain_columns_].sum(axis=1)
            usable = domain_total > 0
            insufficient = usable & (
                denominator / np.maximum(domain_total, 1e-300)
                < self.min_sample_coverage - 1e-12
            )
            if np.any(insufficient):
                raise ValueError(
                    f"Selected {self.selected_rank_} rank covers less than {self.min_sample_coverage:g} "
                    f"of domain-level abundance for {int(insufficient.sum())} sample(s). "
                    "This represents missing input rank coverage, not missing reference tips."
                )
        if (coverage < self.minimum_usable_coverage - 1e-12).any():
            raise ValueError(
                f"Reference coverage below minimum_usable_coverage={self.minimum_usable_coverage:g} "
                f"for {int((coverage < self.minimum_usable_coverage - 1e-12).sum())} sample(s); "
                f"minimum observed coverage is {summary['minimum']:.3%}. "
                "A reference-covered UniFrac representation cannot be considered sufficiently representative."
            )
        if summary["below_threshold"]:
            message = (
                f"Reference-covered {self.selected_rank_} UniFrac has only "
                f"{summary['minimum']:.3%} minimum and {summary['median']:.3%} median "
                f"matched abundance; {summary['below_threshold']} of {len(coverage)} "
                f"samples fall below the configured {self.min_sample_coverage:.1%} "
                "reporting threshold. Distances describe the retained subcommunity, "
                "not the entire measured community."
            )
            if self.coverage_policy == "error":
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        if self.metric == "unweighted_unifrac" and summary["below_presence_threshold"]:
            message = (
                f"Reference-covered unweighted UniFrac retains only "
                f"{summary['minimum_presence']:.3%} minimum and "
                f"{summary['median_presence']:.3%} median detected-taxon coverage; "
                f"{summary['below_presence_threshold']} of {len(coverage)} samples fall below "
                f"the configured {self.min_sample_coverage:.1%} reporting threshold. "
                "Unweighted distances may be sensitive to omitted rare taxa."
            )
            if self.coverage_policy == "error":
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        return np.ascontiguousarray(selected, dtype=np.float64)

    def _distance_matrix(self, X: np.ndarray) -> np.ndarray:
        from skbio.diversity import beta_diversity

        values = _valid_matrix(X, len(self._matched_ids_))
        counts = _encode_unifrac_counts(values, metric=self.metric)
        kwargs: dict[str, Any] = {"taxa": self._matched_ids_, "tree": self._tree_}
        if self.metric == "weighted_unifrac":
            kwargs["normalized"] = self.normalized
        distances = np.asarray(
            beta_diversity(self.metric, counts, **kwargs).data, dtype=np.float64
        )
        if not np.isfinite(distances).all():
            raise ValueError("scikit-bio returned non-finite UniFrac distances.")
        if np.any(distances < -1e-12) or not np.allclose(
            distances, distances.T, atol=1e-10
        ):
            raise ValueError("scikit-bio returned an invalid UniFrac distance matrix.")
        return distances

    def _cross_distances(self, query: np.ndarray) -> np.ndarray:
        training_count = len(self._train_counts_)
        combined = np.concatenate((self._train_counts_, query), axis=0)
        distances = self._distance_matrix(combined)[training_count:, :training_count]
        if not np.isfinite(distances).all():
            raise ValueError(
                "scikit-bio returned non-finite held-out UniFrac distances."
            )
        return distances

    def fit(self, X: Any, y: Any = None) -> MetaPhlAnUniFracPCoA:
        if not hasattr(self, "_selected_columns_"):
            raise ValueError("Call bind_schema(feature_names) before fitting UniFrac.")
        values = _valid_matrix(X, len(self.feature_names_))
        if len(values) < 2:
            raise ValueError("PCoA requires at least two training samples.")
        self.n_features_in_ = values.shape[1]
        self.n_features_out_ = self.n_components
        if self.reference != "metaphlan" and self.tree_path is None:
            raise ValueError(
                "SILVA/16S UniFrac needs a rooted sequence phylogeny with tips corresponding to measured ASVs/OTUs or validated placements. SILVA taxonomy labels alone are not sufficient; provide tree_path or an ASV phylogeny from the 16S pipeline."
            )
        reference = (
            Path(self.tree_path)
            if self.tree_path is not None
            else _reference_file(self.reference_version, self.cache_dir)
        )
        if not reference.is_file():
            raise FileNotFoundError(f"Phylogenetic reference not found: {reference}")
        self.reference_sha256_ = hashlib.sha256(reference.read_bytes()).hexdigest()
        self.reference_path_ = str(reference)
        self._tree_, tips = _load_tree(
            reference, normalise_sgb=self.reference == "metaphlan"
        )
        self.reference_taxonomy_sha256_ = None
        if self.selected_rank_ == "feature":
            mapped = self._load_feature_tip_mapping()
            identifiers = tuple(mapped.get(name, name) for name in self.feature_names_)
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(
                    "Multiple abundance features map to the same tree tip; this would double-count phylogenetic abundance."
                )
            keep = np.array(
                [identifier in tips for identifier in identifiers], dtype=bool
            )
            if not keep.any():
                raise ValueError(
                    "No ASV/OTU feature corresponds to a phylogenetic tree tip; verify reference sequence accession IDs or feature_to_tip."
                )
            self._matched_columns_ = self._selected_columns_[keep]
            self._matched_ids_ = tuple(
                identifier
                for identifier, retained in zip(identifiers, keep)
                if retained
            )
            self.unmatched_taxa_ = tuple(
                name
                for name, retained in zip(self.feature_names_, keep)
                if not retained
            )
            self.matched_taxa_count_ = len(self._matched_ids_)
            self.unmatched_taxa_count_ = len(self.unmatched_taxa_)
            self.phylogeny_mode_ = f"sequence_phylogeny_{self.reference}"
        elif self.selected_rank_ == "strain":
            sgb_identifiers = tuple(
                _canonical_sgb(identifier) for identifier in self._sgb_ids_
            )
            keep = np.array(
                [identifier in tips for identifier in sgb_identifiers], dtype=bool
            )
            if not keep.any():
                raise ValueError(
                    "No SGB feature matches the phylogenetic reference tips."
                )
            self.unmatched_sgbs_ = tuple(
                identifier
                for identifier, matched in zip(sgb_identifiers, keep)
                if not matched
            )
            self._matched_columns_ = self._selected_columns_[keep]
            self._matched_ids_ = tuple(
                identifier
                for identifier, matched in zip(sgb_identifiers, keep)
                if matched
            )
            self.phylogeny_mode_ = "genome_phylogeny_sgb"
        else:
            if self.tree_path is not None and self.taxonomy_path is None:
                raise ValueError(
                    "A custom MetaPhlAn tree requires its matching taxonomy_path for rank-level UniFrac."
                )
            taxonomy_path = (
                Path(self.taxonomy_path)
                if self.taxonomy_path is not None
                else _reference_file(
                    self.reference_version, self.cache_dir, kind="taxonomy"
                )
            )
            if not taxonomy_path.is_file():
                raise FileNotFoundError(
                    f"MetaPhlAn taxonomy reference not found: {taxonomy_path}"
                )
            self.reference_taxonomy_sha256_ = hashlib.sha256(
                taxonomy_path.read_bytes()
            ).hexdigest()
            mapping = _load_taxonomy(taxonomy_path)
            matched, unmatched_names = _resolve_named_taxa(
                self.feature_names_, self.selected_rank_, mapping, tips
            )
            observed_reference_tips = {
                tip for members in matched.values() for tip in members
            }
            observed_tree = (
                self._tree_.shear(observed_reference_tips)
                if observed_reference_tips
                else self._tree_
            )
            retained, excluded = _filter_monophyletic_groups(
                observed_tree, matched, policy=self.nonmonophyletic
            )
            self.nonmonophyletic_taxa_ = tuple(
                self.feature_names_[index] for index in excluded
            )
            self.unmatched_taxa_ = unmatched_names + self.nonmonophyletic_taxa_
            self.matched_taxa_count_ = len(retained)
            self.unmatched_taxa_count_ = len(self.unmatched_taxa_)
            if len(retained) < 2:
                examples = ", ".join(repr(name) for name in self.unmatched_taxa_[:3])
                raise ValueError(
                    f"UniFrac reference matching failed at rank {self.selected_rank_!r}: "
                    f"{len(retained)} of {len(self._selected_columns_)} taxonomic features retained "
                    f"from reference {self.reference_version!r}; "
                    f"{len(self.nonmonophyletic_taxa_)} nonmonophyletic features excluded. "
                    f"First unmatched or excluded features: {examples}. "
                    "The reference must be compatible with the original profiling database."
                )
            self._tree_, selected, self._matched_ids_ = _collapse_to_rank(
                self._tree_, retained
            )
            self._matched_columns_ = np.asarray(selected, dtype=int)
            self.phylogeny_mode_ = (
                f"reference_phylogeny_collapsed_to_{self.selected_rank_}"
            )
        if len(self._matched_ids_) < 2:
            raise ValueError(
                "UniFrac requires at least two distinct reference-matched phylogenetic tips."
            )
        self._train_counts_ = self._extract(values)
        self.training_coverage_summary_ = dict(self.last_coverage_summary_)
        self._train_sha256_ = hashlib.sha256(values.tobytes()).hexdigest()
        distances = self._distance_matrix(self._train_counts_)
        self.max_training_unifrac_distance_ = float(np.max(distances))
        self.count_encoding_ = (
            "presence_absence"
            if self.metric == "unweighted_unifrac"
            else "relative_abundance_fixed_point_1e12"
        )
        self._train_squared_ = distances**2
        self._column_mean_squared_ = self._train_squared_.mean(axis=0)
        self._global_mean_squared_ = float(self._train_squared_.mean())
        from skbio.stats.ordination import pcoa
        from skbio.stats.distance import DistanceMatrix

        ordination = pcoa(
            DistanceMatrix(distances),
            method="eigh",
            dimensions=min(self.n_components, len(values)),
        )
        eigenvalues = np.asarray(ordination.eigvals, dtype=np.float64)
        coordinates = np.asarray(ordination.samples, dtype=np.float64)
        tolerance = max(1e-12, float(eigenvalues.max(initial=0.0)) * 1e-10)
        retained = np.flatnonzero(eigenvalues > tolerance)[: self.n_components]
        if retained.size == 0:
            raise ValueError(
                f"The training {self.metric} distances have no positive PCoA axes; "
                f"largest pairwise distance was {self.max_training_unifrac_distance_:.6g}. "
                "Check whether the samples have any phylogenetic variation at the chosen rank."
            )
        self._eigenvalues_ = eigenvalues[retained]
        self._eigenvectors_ = (
            coordinates[:, retained] / np.sqrt(self._eigenvalues_)[None, :]
        )
        self.n_positive_axes_ = len(retained)
        self.train_coordinates_ = np.zeros(
            (len(values), self.n_components), dtype=np.float64
        )
        self.train_coordinates_[:, : self.n_positive_axes_] = coordinates[:, retained]
        self.negative_inertia_fraction_ = None
        return self

    def apply(self, X: Any) -> np.ndarray:
        if not hasattr(self, "train_coordinates_"):
            raise RuntimeError("UniFrac must be fitted before transformation.")
        values = _valid_matrix(X, self.n_features_in_)
        fingerprint = hashlib.sha256(values.tobytes()).hexdigest()
        if fingerprint == self._train_sha256_ and values.shape[0] == len(
            self._train_counts_
        ):
            return self.train_coordinates_.copy()
        query = self._extract(values)
        squared = self._cross_distances(query) ** 2
        gram = -0.5 * (
            squared
            - squared.mean(axis=1, keepdims=True)
            - self._column_mean_squared_[None, :]
            + self._global_mean_squared_
        )
        result = np.zeros((len(values), self.n_components), dtype=np.float64)
        result[:, : self.n_positive_axes_] = (gram @ self._eigenvectors_) / np.sqrt(
            self._eigenvalues_
        )[None, :]
        return result

    transform = apply

    def apply_fitted_pair(
        self, X_train: Any, X_test: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        train = _valid_matrix(X_train, self.n_features_in_)
        if hashlib.sha256(train.tobytes()).hexdigest() != self._train_sha256_:
            raise ValueError("The training matrix has changed since fitting PCoA.")
        return self.train_coordinates_.copy(), self.apply(X_test)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        if self.n_positive_axes_ is None:
            raise RuntimeError(
                "UniFrac must be fitted before requesting coordinate names."
            )
        return [
            f"{self.metric}_pcoa_{index + 1:03d}" for index in range(self.n_components)
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
