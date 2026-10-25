from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .atlas import load_schaefer_atlas, schaefer_atlas
from .cache import (
    BrainCache,
    atomic_path,
    cache_root,
    file_identity,
    input_identity,
    settings_identity,
    worker_count,
)
from .progress import BrainProgress, active_progress
from .storage import RepresentationStore
from ..console import info, success
from .fmriprep import discover_fmriprep_runs
from .handcrafted import (
    HandcraftedSettings,
    partial_fc,
    pearson_fc,
    spectral_amplitudes,
)
from .graph_centrality import graph_centrality
from .labus_2019 import (
    labus_2019_centrality,
    labus_2019_feature_names,
    load_labus_2019_mapping,
    LABUS_PREPROCESSING,
)
from .source import BrainSource
from .structural import discover_tissue_maps, tissue_volumes
from .timeseries import BOLDPreprocessing, extract_schaefer_timeseries


FUNCTIONAL_METHODS = frozenset(
    {
        "alff",
        "falff",
        "pearson_fc",
        "partial_fc",
        "graph_centrality",
        "labus_2019_centrality",
    }
)
METHODS = FUNCTIONAL_METHODS | {
    "brainharmonix_f",
    "brainharmonix_s",
    "brainharmonix_sf",
    "cortexmae_p",
    "omni_fmri",
    "tissue_volumes",
}


@dataclass(frozen=True, slots=True)
class BrainRepresentation:
    method: str
    parcels: int | None = None
    checkpoint: Path | str | None = None
    structural_checkpoint: Path | str | None = None
    graph_density: float = 0.15

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unsupported brain representation: {self.method}")
        if self.method in FUNCTIONAL_METHODS - {
            "labus_2019_centrality"
        } and self.parcels not in {100, 200, 400}:
            raise ValueError(
                "Functional handcrafted features require 100, 200 or 400 Schaefer parcels"
            )
        if not np.isfinite(self.graph_density) or not 0 < self.graph_density <= 1:
            raise ValueError("Graph density must be finite and in (0, 1].")
        if self.method != "graph_centrality" and self.graph_density != 0.15:
            raise ValueError("graph_density is only configurable for graph_centrality")
        if (
            self.method == "graph_centrality"
            and int(np.ceil(self.graph_density * self.parcels * (self.parcels - 1) / 2))
            < self.parcels - 1
        ):
            raise ValueError("graph_density too low for a connected graph")
        if (
            self.method not in FUNCTIONAL_METHODS
            or self.method == "labus_2019_centrality"
        ) and self.parcels is not None:
            raise ValueError(
                "This brain representation does not accept a parcels argument"
            )
        if self.checkpoint is not None and self.method not in {
            "brainharmonix_s",
            "brainharmonix_sf",
            "cortexmae_p",
            "omni_fmri",
        }:
            raise ValueError(
                "Checkpoint overrides are supported for BrainHarmonix-S/SF, CortexMAE-P and Omni-fMRI"
            )
        if self.structural_checkpoint is not None and self.method != "brainharmonix_sf":
            raise ValueError(
                "structural_checkpoint is only supported for BrainHarmonix-SF"
            )
        for value in (self.checkpoint, self.structural_checkpoint):
            if value is not None and (
                not isinstance(value, (str, Path)) or not str(value).strip()
            ):
                raise ValueError("A checkpoint must be a filesystem path or 'auto'")

    @property
    def key(self) -> str:
        if self.method == "labus_2019_centrality":
            return self.method
        if self.method == "graph_centrality":
            tag = str(self.graph_density).replace(".", "p").replace("-", "m")
            return f"schaefer{self.parcels}_{self.method}_d{tag}"
        return f"schaefer{self.parcels}_{self.method}" if self.parcels else self.method


@dataclass(frozen=True, slots=True)
class Brain:
    derivatives_dir: Path | str
    representations: tuple[BrainRepresentation, ...]
    bids_root: Path | str | None = None
    atlas_paths: Mapping[int, Path | str] = field(default_factory=dict)
    labus_2019_labels_path: Path | str | None = None
    labus_2019_public_approximate: bool = False
    labus_2019_preprocessing: BOLDPreprocessing = field(
        default_factory=lambda: LABUS_PREPROCESSING
    )
    atlas_order_verified: bool = False
    atlas_space_verified: bool = False
    normalization_stats: Path | str | None = None
    normalization: str = "auto"
    reference_derivatives_dir: Path | str | None = None
    reference_bids_root: Path | str | None = None
    sample_id_map: Mapping[str, str] | None = None
    preprocessing: BOLDPreprocessing = field(default_factory=BOLDPreprocessing)
    handcrafted: HandcraftedSettings = field(default_factory=HandcraftedSettings)
    name: str = "brain"
    replace_modality: str | None = None
    overwrite: bool = False
    trace_subject_ids: tuple[str, ...] = ()
    trace_max_features_per_target: int = 2
    layout: str = "representations"
    cache_dir: Path | str | None = None
    n_jobs: int | str = "auto"
    migrate_legacy_artifacts: bool = False
    retain_timeseries: bool = True

    def __post_init__(self) -> None:
        worker_count(self.n_jobs, 1)
        if self.layout not in {"representations", "modalities"}:
            raise ValueError("Brain.layout must be representations or modalities")
        keys = [rep.key for rep in self.representations]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Brain representations must be nonempty and unique")
        if not self.name or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError("Brain modality name must be a nonempty identifier")
        if any(rep.method == "labus_2019_centrality" for rep in self.representations):
            if (165 in self.atlas_paths) != (self.labus_2019_labels_path is not None):
                raise ValueError(
                    "Custom Labus-2019 atlas and label mapping must be supplied together"
                )
            if self.labus_2019_labels_path is not None:
                load_labus_2019_mapping(self.labus_2019_labels_path)
        if (
            self.atlas_paths
            and not (self.atlas_order_verified and self.atlas_space_verified)
            and not self.labus_2019_public_approximate
        ):
            raise ValueError(
                "Custom atlas files require explicit spatial and ROI-order verification"
            )
        if self.normalization not in {"auto", "per_run", "reference"}:
            raise ValueError(
                "BrainHarmonix normalization must be auto, per_run or reference"
            )
        if (
            self.normalization_stats is not None
            and self.reference_derivatives_dir is not None
        ):
            raise ValueError(
                "Provide either reference data or saved statistics, not both"
            )
        if (
            self.normalization == "reference"
            and not (
                self.normalization_stats is not None
                or self.reference_derivatives_dir is not None
            )
            and any(
                rep.method in {"brainharmonix_f", "brainharmonix_sf"}
                for rep in self.representations
            )
        ):
            raise ValueError(
                "Reference normalization requires independent reference data or statistics"
            )
        if self.normalization == "per_run" and (
            self.normalization_stats is not None
            or self.reference_derivatives_dir is not None
        ):
            raise ValueError(
                "Reference data or statistics require normalization='reference'"
            )
        if self.trace_subject_ids:
            if not any(rep.method == "brainharmonix_f" for rep in self.representations):
                raise ValueError(
                    "Encoder sensitivity tracing requires a BrainHarmonix representation"
                )
            if self.normalization == "per_run" or not (
                self.normalization_stats is not None
                or self.reference_derivatives_dir is not None
            ):
                raise ValueError(
                    "Automated encoder tracing currently requires reference normalization"
                )
            if (
                len(self.trace_subject_ids) != len(set(self.trace_subject_ids))
                or self.trace_max_features_per_target < 1
            ):
                raise ValueError(
                    "Trace subjects must be unique and the feature limit positive"
                )
        if self.sample_id_map is not None:
            target = [str(v) for v in self.sample_id_map.values()]
            if len(target) != len(set(target)) or not all(target):
                raise ValueError("Subject mapping must be one-to-one and nonempty")

    def prepare(self, output_dir: Path | str):
        return prepare_brain_representations(self, output_dir)


@dataclass(frozen=True, slots=True)
class BrainRepresentationResult:
    modality: Any
    selectors: tuple[tuple[str, tuple[str, ...]], ...]
    provenance: Path
    manifest: Path
    modalities: tuple[Any, ...] = ()


def _feature_names(rep: BrainRepresentation) -> tuple[str, ...]:
    n = rep.parcels
    if rep.method in {"alff", "falff"}:
        return tuple(f"{rep.method}_{i:03d}" for i in range(n))
    if rep.method in {"pearson_fc", "partial_fc"}:
        prefix = "fc" if rep.method == "pearson_fc" else "pfc"
        return tuple(
            f"{prefix}_{a:03d}_{b:03d}" for a in range(n) for b in range(a + 1, n)
        )
    if rep.method == "labus_2019_centrality":
        return labus_2019_feature_names()
    if rep.method == "graph_centrality":
        return tuple(
            f"{measure}_{i:03d}"
            for measure in ("strength", "betweenness", "eigenvector")
            for i in range(n)
        )
    if rep.method == "brainharmonix_f":
        return tuple(f"bhf_{i:03d}" for i in range(768))
    if rep.method in {"brainharmonix_s", "brainharmonix_sf"}:
        prefix = "bhs" if rep.method == "brainharmonix_s" else "bhsf"
        return tuple(f"{prefix}_{i:03d}" for i in range(768))
    if rep.method == "cortexmae_p":
        return tuple(f"cmp_{i:03d}" for i in range(768))
    if rep.method == "omni_fmri":
        return tuple(f"omni_{i:03d}" for i in range(768))
    return ("tissue_gm_mm3", "tissue_wm_mm3", "tissue_csf_mm3")


def _resolved_brain(brain: Brain) -> Brain:
    from dataclasses import replace

    required = {
        rep.parcels
        for rep in brain.representations
        if rep.method in FUNCTIONAL_METHODS and rep.method != "labus_2019_centrality"
    }
    if any(
        rep.method in {"brainharmonix_f", "brainharmonix_sf"}
        for rep in brain.representations
    ):
        required.add(400)
    atlases = dict(brain.atlas_paths)
    for parcels in sorted(required - atlases.keys()):
        info(f"Brain · resolving Schaefer-{parcels} atlas through TemplateFlow")
        atlases[parcels] = schaefer_atlas(parcels)
    labels = brain.labus_2019_labels_path
    auto_labus = False
    if (
        any(rep.method == "labus_2019_centrality" for rep in brain.representations)
        and 165 not in atlases
    ):
        from .labus_atlas_auto import resolve_labus_public_atlas

        info(
            "Brain · resolving approximate Labus-2019 anatomical atlas from public sources"
        )
        atlases[165], labels = resolve_labus_public_atlas()
        auto_labus = True
    if atlases == dict(brain.atlas_paths) and labels == brain.labus_2019_labels_path:
        return brain
    return replace(
        brain,
        atlas_paths=atlases,
        labus_2019_labels_path=labels,
        labus_2019_public_approximate=auto_labus or brain.labus_2019_public_approximate,
        atlas_order_verified=True,
        atlas_space_verified=True,
    )


def _handcrafted_key(
    cache: BrainCache, brain: Brain, rep: BrainRepresentation, item: Any
) -> str:
    if rep.method == "tissue_volumes":
        identity = {
            "subject": item.subject_id,
            "maps": [file_identity(path) for path in item.maps],
        }
    else:
        identity = input_identity(item)
        identity["atlas"] = file_identity(
            brain.atlas_paths[
                165 if rep.method == "labus_2019_centrality" else rep.parcels
            ]
        )
    settings = (
        settings_identity(brain.preprocessing),
        settings_identity(brain.handcrafted),
    )
    if rep.method == "labus_2019_centrality":
        identity["atlas"] = file_identity(brain.atlas_paths[165])
        identity["labels"] = file_identity(brain.labus_2019_labels_path)
        settings = (
            settings_identity(brain.labus_2019_preprocessing),
            "labus_2019_full_positive_weighted_v1",
        )
    if rep.method == "graph_centrality":
        import networkx as nx

        settings += (
            "positive_pearson_mst_density_inverse_weighted_v1",
            rep.graph_density,
            nx.__version__,
        )
    return cache.key("representation", rep.key, identity, settings)


def _handcrafted_worker(
    brain: Brain,
    rep: BrainRepresentation,
    item: Any,
    key: str,
    output_dir: Path | None = None,
) -> tuple[str, np.ndarray]:
    from dataclasses import replace
    from threadpoolctl import threadpool_limits

    cache = BrainCache(
        cache_root(brain.cache_dir), migrate_legacy=brain.migrate_legacy_artifacts
    )
    columns = tuple(f"brain_{rep.key}__{name}" for name in _feature_names(rep))
    store = RepresentationStore(output_dir or cache.root.parent, rep.key, columns)
    with cache.locked("representation", key):
        staged = None if brain.overwrite else store.staged(item.subject_id, key)
        if staged is not None:
            return item.subject_id, staged
        saved = None if brain.overwrite else cache.read("representation", key)
        if saved is not None and "vector" in saved:
            vector = np.asarray(saved["vector"], dtype=np.float64)
            if vector.shape == (len(columns),) and np.isfinite(vector).all():
                store.stage(item.subject_id, key, vector)
                return item.subject_id, vector
        with threadpool_limits(limits=1):
            if rep.method == "tissue_volumes":
                volumes = tissue_volumes(item)
                vector = np.array(
                    [volumes[name] for name in _feature_names(rep)], dtype=np.float64
                )
            else:
                spectral = rep.method in {"alff", "falff"}
                settings = (
                    brain.labus_2019_preprocessing
                    if rep.method == "labus_2019_centrality"
                    else (
                        replace(
                            brain.preprocessing,
                            high_pass=None,
                            low_pass=None,
                            standardize=False,
                        )
                        if spectral
                        else brain.preprocessing
                    )
                )
                ts_key = cache.key(
                    "timeseries",
                    input_identity(item),
                    file_identity(
                        brain.atlas_paths[
                            165
                            if rep.method == "labus_2019_centrality"
                            else rep.parcels
                        ]
                    ),
                    165 if rep.method == "labus_2019_centrality" else rep.parcels,
                    settings_identity(settings),
                )
                with cache.locked("timeseries", ts_key):
                    signal = (
                        None if brain.overwrite else cache.read("timeseries", ts_key)
                    )
                    if signal is None:
                        if rep.method == "labus_2019_centrality":
                            from nilearn import image

                            atlas = image.load_img(str(brain.atlas_paths[165]))
                            if len(atlas.shape) != 3:
                                raise ValueError(
                                    "The Labus-2019 atlas must be a 3D labeled image"
                                )
                            atlas_values = np.unique(np.asarray(atlas.dataobj))
                            if not np.array_equal(atlas_values, np.arange(166)):
                                raise ValueError(
                                    "The Labus-2019 atlas must contain background 0 and all labels 1..165"
                                )
                        else:
                            atlas = load_schaefer_atlas(
                                brain.atlas_paths[rep.parcels], rep.parcels
                            )
                        series, _ = extract_schaefer_timeseries(
                            item,
                            n_parcels=165
                            if rep.method == "labus_2019_centrality"
                            else rep.parcels,
                            preprocessing=settings,
                            atlas_img=atlas,
                        )
                        if brain.retain_timeseries:
                            cache.write(
                                "timeseries",
                                ts_key,
                                series=np.asarray(series, dtype=np.float32),
                            )
                    else:
                        series = signal["series"]
                if rep.method == "labus_2019_centrality":
                    mapping = load_labus_2019_mapping(brain.labus_2019_labels_path)
                    vector = labus_2019_centrality(series, mapping)
                elif rep.method == "graph_centrality":
                    vector = graph_centrality(series, density=rep.graph_density)
                elif rep.method == "pearson_fc":
                    vector = pearson_fc(series, fisher_z=brain.handcrafted.fisher_z)
                elif rep.method == "partial_fc":
                    vector = partial_fc(series, fisher_z=brain.handcrafted.fisher_z)
                else:
                    alff, falff = spectral_amplitudes(
                        series, item.repetition_time, settings=brain.handcrafted
                    )
                    vector = alff if rep.method == "alff" else falff
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (len(columns),) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid feature vector for {item.subject_id}: {rep.key}")
        store.stage(item.subject_id, key, vector)
        return item.subject_id, vector


def _handcrafted_table(brain: Brain, rep: BrainRepresentation, folder: Path) -> Path:
    columns = tuple(f"brain_{rep.key}__{name}" for name in _feature_names(rep))
    store = RepresentationStore(folder.parent, rep.key, columns)
    with store.locked():
        return _handcrafted_table_unlocked(brain, rep, folder)


def _handcrafted_table_unlocked(
    brain: Brain, rep: BrainRepresentation, folder: Path
) -> Path:
    from dataclasses import replace

    folder.mkdir(parents=True, exist_ok=True)
    monitor = active_progress()
    if monitor is not None:
        monitor.phase(f"Discovering inputs for {rep.key}")
    items = (
        discover_tissue_maps(brain.derivatives_dir)
        if rep.method == "tissue_volumes"
        else discover_fmriprep_runs(brain.derivatives_dir, bids_root=brain.bids_root)
    )
    if brain.cache_dir is None:
        brain = replace(
            brain,
            cache_dir=cache_root(None, folder.parent),
            migrate_legacy_artifacts=True,
        )
    cache = BrainCache(
        cache_root(brain.cache_dir), migrate_legacy=brain.migrate_legacy_artifacts
    )
    columns = tuple(f"brain_{rep.key}__{name}" for name in _feature_names(rep))
    store = RepresentationStore(folder.parent, rep.key, columns)
    records = [(item, _handcrafted_key(cache, brain, rep, item)) for item in items]
    existing, previous_keys = store.existing() if not brain.overwrite else ({}, {})
    rows = {
        item.subject_id: existing[item.subject_id]
        for item, key in records
        if previous_keys.get(item.subject_id) == key and item.subject_id in existing
    }
    pending = []
    for item, key in records:
        if item.subject_id in rows:
            continue
        staged = None if brain.overwrite else store.staged(item.subject_id, key)
        if staged is not None:
            rows[item.subject_id] = staged
        else:
            pending.append((item, key))
    jobs = worker_count(brain.n_jobs, len(pending))
    info(
        f"Brain · {rep.key} · {len(rows)}/{len(items)} reusable · {len(pending)} to compute · {jobs} workers"
    )
    if monitor is not None:
        monitor.begin_units(f"{rep.key} · completed subjects", len(items))
        for sid in rows:
            monitor.working(f"{sid} · reused")
            monitor.advance()
    if jobs == 1:
        for item, key in pending:
            if monitor is not None:
                monitor.working(item.subject_id)
            sid, vector = _handcrafted_worker(brain, rep, item, key, folder.parent)
            rows[sid] = vector
            if monitor is not None:
                monitor.advance()
    elif pending:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _handcrafted_worker, brain, rep, item, key, folder.parent
                ): item.subject_id
                for item, key in pending
            }
            for future in as_completed(futures):
                sid, vector = future.result()
                rows[sid] = vector
                if monitor is not None:
                    monitor.working(sid)
                    monitor.advance()
    if (
        not pending
        and len(rows) == len(items)
        and store.path.is_file()
        and not brain.overwrite
    ):
        if all(previous_keys.get(item.subject_id) == key for item, key in records):
            success(f"Brain · {rep.key} · reused · {store.path}")
            return store.path
    if monitor is not None:
        monitor.phase(f"Consolidating {rep.key} · {len(items)} subjects")
    destination = store.commit([(item.subject_id, key) for item, key in records], rows)
    for item, key in records:
        cache.path("representation", key).unlink(missing_ok=True)
        for root in cache.legacy_roots:
            if root != Path.home() / ".cache" / "mllabiome" / "brain" / "v2":
                (root / "representation" / key[:2] / f"{key}.npz").unlink(
                    missing_ok=True
                )
    for redundant in (folder / f"{rep.key}.tsv", folder / f"{rep.key}_modality.tsv"):
        redundant.unlink(missing_ok=True)
    success(f"Brain · {rep.key} · ready · {destination}")
    return destination


def _embedding_table(brain: Brain, folder: Path) -> Path:
    columns = tuple(f"brain_brainharmonix_f__bhf_{i:03d}" for i in range(768))
    store = RepresentationStore(folder.parent, "brainharmonix_f", columns)
    with store.locked():
        return _embedding_table_unlocked(brain, folder)


def _embedding_table_unlocked(brain: Brain, folder: Path) -> Path:
    from .brainharmonix import embed_fmriprep

    folder.mkdir(parents=True, exist_ok=True)
    columns = tuple(f"brain_brainharmonix_f__bhf_{i:03d}" for i in range(768))
    store = RepresentationStore(folder.parent, "brainharmonix_f", columns)
    cache = BrainCache(
        cache_root(brain.cache_dir, folder.parent),
        migrate_legacy=brain.migrate_legacy_artifacts,
    )
    runs = discover_fmriprep_runs(brain.derivatives_dir, bids_root=brain.bids_root)
    reference_runs = (
        discover_fmriprep_runs(
            brain.reference_derivatives_dir, bids_root=brain.reference_bids_root
        )
        if brain.reference_derivatives_dir is not None
        else []
    )
    mode = (
        "reference"
        if brain.normalization == "auto"
        and (
            brain.reference_derivatives_dir is not None
            or brain.normalization_stats is not None
        )
        else "per_run"
        if brain.normalization == "auto"
        else brain.normalization
    )
    normalization_identity = (
        file_identity(brain.normalization_stats)
        if brain.normalization_stats is not None
        else tuple(input_identity(run) for run in reference_runs)
    )
    settings = (
        "brainharmonix_f",
        settings_identity(brain.preprocessing),
        file_identity(brain.atlas_paths[400]),
        mode,
        normalization_identity,
    )
    records = [
        (run, cache.key("embedding", input_identity(run), settings, foundation=True))
        for run in runs
    ]
    previous, keys = store.existing() if not brain.overwrite else ({}, {})
    reusable = {
        run.subject_id: (key, previous[run.subject_id])
        for run, key in records
        if keys.get(run.subject_id) == key and run.subject_id in previous
    }
    if len(reusable) == len(runs) and not brain.overwrite:
        success(f"Brain · brainharmonix_f · reused · {store.path}")
        return store.path
    result = embed_fmriprep(
        brain.derivatives_dir,
        folder,
        reference_derivatives_dir=brain.reference_derivatives_dir,
        reference_bids_root=brain.reference_bids_root,
        normalization_stats=brain.normalization_stats,
        normalization=brain.normalization,
        bids_root=brain.bids_root,
        atlas_path=brain.atlas_paths[400],
        roi_order_verified=True,
        preprocessing=brain.preprocessing,
        overwrite=brain.overwrite,
        cache_dir=brain.cache_dir,
        n_jobs=brain.n_jobs,
        migrate_legacy_artifacts=brain.migrate_legacy_artifacts,
        output_format="parquet",
        reused_embeddings=reusable,
    )
    if result.feature_table != store.path:
        raise RuntimeError("BrainHarmonix canonical storage path mismatch")
    store.adopt([(run.subject_id, key) for run, key in records])
    import shutil

    shutil.rmtree(cache.root / "embedding", ignore_errors=True)
    for redundant in (
        folder / "brainharmonix_f_features.tsv",
        folder / "brainharmonix_f_modality.tsv",
    ):
        redundant.unlink(missing_ok=True)
    if not brain.retain_timeseries:
        shutil.rmtree(cache.root / "timeseries", ignore_errors=True)
    return store.path


def _atlas_labels(n: int) -> tuple[str, ...]:
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n, yeo_networks=7, resolution_mm=2
    )
    labels = tuple(
        x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in atlas.labels
    )
    if len(labels) == n + 1 and labels[0].strip().lower() == "background":
        labels = labels[1:]
    if (
        len(labels) != n
        or any(not label.strip() for label in labels)
        or len(set(labels)) != n
    ):
        raise ValueError(
            f"Unexpected Schaefer-{n} atlas label count or ordering: {len(labels)}"
        )
    return labels


def _records(
    brain: Brain, selectors: tuple[tuple[str, tuple[str, ...]], ...]
) -> list[dict[str, Any]]:
    atlas_labels = {
        n: _atlas_labels(n)
        for n in {rep.parcels for rep in brain.representations if rep.parcels}
    }
    labus_mapping = (
        load_labus_2019_mapping(brain.labus_2019_labels_path)
        if any(rep.method == "labus_2019_centrality" for rep in brain.representations)
        else None
    )
    records = []
    for rep in brain.representations:
        names = _feature_names(rep)
        for original in names:
            feature = f"brain_{rep.key}__{original}"
            indices = (
                tuple(int(x) for x in re.findall(r"\d{3}", original))
                if rep.parcels
                else ()
            )
            if rep.method == "labus_2019_centrality":
                _, side, roi = original.split("_", 2)
                indices = (labus_mapping[(side, roi)] - 1,)
            if rep.method in {"graph_centrality", "labus_2019_centrality"}:
                relation = "graph_node_centrality"
            elif rep.method in {"alff", "falff"}:
                relation = "exact_parcel"
            elif rep.method in {"pearson_fc", "partial_fc"}:
                relation = "exact_edge_endpoints"
            else:
                relation = {
                    "brainharmonix_f": "requires_encoder_attribution",
                    "brainharmonix_s": "requires_structural_encoder_attribution",
                    "brainharmonix_sf": "requires_joint_encoder_attribution",
                    "cortexmae_p": "requires_cortexmae_encoder_attribution",
                    "omni_fmri": "requires_voxel_encoder_attribution",
                }.get(rep.method, "global_tissue")
            records.append(
                {
                    "feature": feature,
                    "representation": rep.key,
                    "method": rep.method,
                    "parcels": rep.parcels,
                    "region_indices": indices,
                    "region_names": tuple(atlas_labels[rep.parcels][i] for i in indices)
                    if rep.parcels and indices
                    else (
                        tuple(original.split("_", 1)[1:])
                        if rep.method == "labus_2019_centrality"
                        else ()
                    ),
                    "relationship": relation,
                    "graph_density": rep.graph_density
                    if rep.method == "graph_centrality"
                    else None,
                    "graph_definition": "positive_pearson_mst_density_inverse_weighted_v1"
                    if rep.method == "graph_centrality"
                    else (
                        "labus_2019_full_positive_weighted_v1"
                        if rep.method == "labus_2019_centrality"
                        else None
                    ),
                }
            )
    return records


def prepare_brain_representations(
    brain: Brain, output_dir: Path | str
) -> BrainRepresentationResult:
    from dataclasses import replace
    from ..modalities import Modality

    with BrainProgress(len(brain.representations)) as monitor:
        monitor.phase("Resolving anatomical atlases")
        info(
            f"Brain · preparing {len(brain.representations)} representations from {Path(brain.derivatives_dir).expanduser()}"
        )
        brain = _resolved_brain(brain)
        folder = Path(output_dir).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        if brain.cache_dir is None:
            brain = replace(
                brain, cache_dir=cache_root(None, folder), migrate_legacy_artifacts=True
            )
        manifest = folder / "brain_representation_manifest.json"
        provenance = folder / "brain_feature_provenance.json"
        selectors = tuple(
            (rep.key, (f"prefix:brain_{rep.key}__",)) for rep in brain.representations
        )
        paths: list[Path] = []
        subjects_by_rep: dict[str, set[str]] = {}
        for index, rep in enumerate(brain.representations, 1):
            monitor.phase(
                f"Representation {index}/{len(brain.representations)} · {rep.key}"
            )
            individual = folder / rep.key
            if rep.method == "brainharmonix_f":
                path = _embedding_table(brain, individual)
            elif rep.method in {"brainharmonix_s", "brainharmonix_sf"}:
                from .brainharmonix_multimodal import prepare_harmonix_representation

                path = prepare_harmonix_representation(brain, rep, folder)
            elif rep.method == "cortexmae_p":
                from .cortexmae import prepare_cortexmae_representation

                path = prepare_cortexmae_representation(brain, rep, folder)
            elif rep.method == "omni_fmri":
                from .omni import prepare_omni_representation

                path = prepare_omni_representation(brain, rep, folder)
            else:
                path = _handcrafted_table(brain, rep, individual)
            monitor.phase(f"Validating {rep.key} and sample identifiers")
            frame = pd.read_parquet(path)
            if (
                frame.empty
                or "sample_id" not in frame.columns
                or frame["sample_id"].duplicated().any()
            ):
                raise ValueError(
                    f"Invalid or duplicated subject identifiers for {rep.key}"
                )
            columns = tuple(f"brain_{rep.key}__{name}" for name in _feature_names(rep))
            if tuple(frame.columns[1:]) != columns:
                raise ValueError(f"Unexpected feature order for {rep.key}")
            if not np.isfinite(frame.iloc[:, 1:].to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite representation values: {rep.key}")
            selected = set(frame["sample_id"].astype(str))
            if brain.sample_id_map is not None:
                missing = selected - set(brain.sample_id_map)
                if missing:
                    raise ValueError(f"Unmapped BIDS subjects: {sorted(missing)}")
                selected = {brain.sample_id_map[sid] for sid in selected}
            subjects_by_rep[rep.key] = selected
            paths.append(path)
            monitor.complete_representation(rep.key)
        if (
            brain.layout == "representations"
            and len({frozenset(ids) for ids in subjects_by_rep.values()}) != 1
        ):
            raise ValueError("Subject sets differ between brain representations")
        monitor.phase("Saving anatomical provenance")
        atlas_records = _records(brain, selectors)
        with atomic_path(provenance) as temporary:
            temporary.write_text(
                json.dumps(
                    {
                        "schema": "mllabiome.brain.representation_provenance.v1",
                        "records": atlas_records,
                        "atlas_paths": {
                            str(k): str(Path(v).resolve())
                            for k, v in brain.atlas_paths.items()
                        },
                        "atlas_order_verified": brain.atlas_order_verified,
                        "atlas_space_verified": brain.atlas_space_verified,
                        "normalization": brain.normalization,
                        "preprocessing": settings_identity(brain.preprocessing),
                        "labus_2019": {
                            "study_doi": "10.1186/s40168-019-0656-z",
                            "atlas": "automatic_public_165_region_approximation"
                            if brain.labus_2019_public_approximate
                            else "user_verified_165_label_atlas",
                            "exact_original_atlas": False
                            if brain.labus_2019_public_approximate
                            else None,
                            "source_registration_verified": not brain.labus_2019_public_approximate,
                            "labels_path": str(brain.labus_2019_labels_path)
                            if brain.labus_2019_labels_path is not None
                            else None,
                            "metric_graph": "full_positive_pearson_weighted",
                            "betweenness_distance": "reciprocal_positive_weight",
                            "betweenness_normalization": "networkx_normalized",
                            "eigenvector_normalization": "L2",
                            "preprocessing": settings_identity(
                                brain.labus_2019_preprocessing
                            ),
                            "status": "public_atlas_resampled_without_nonlinear_registration_not_exact_study_reproduction"
                            if brain.labus_2019_public_approximate
                            else "study_matched_graph_principles_not_exact_SPM8_CONN13_GTG_reproduction",
                        }
                        if any(
                            rep.method == "labus_2019_centrality"
                            for rep in brain.representations
                        )
                        else None,
                        "graph_protocol": {
                            rep.key: {
                                "density": rep.graph_density,
                                "connectivity": "positive Pearson correlation without Fisher Z",
                                "threshold": "maximum spanning tree backbone plus strongest positive edges",
                                "distance": "reciprocal of Pearson correlation",
                                "betweenness": "normalized weighted shortest-path betweenness",
                                "eigenvector": "L2-normalized weighted eigenvector centrality",
                                "strength": "unnormalized weighted degree",
                            }
                            for rep in brain.representations
                            if rep.method == "graph_centrality"
                        },
                        "layout": brain.layout,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        (folder / "brain_representations.tsv").unlink(missing_ok=True)
        with atomic_path(manifest) as temporary:
            temporary.write_text(
                json.dumps(
                    {
                        "schema": "mllabiome.brain.representations.parquet.v1",
                        "layout": brain.layout,
                        "representations": {
                            rep.key: str(path)
                            for rep, path in zip(brain.representations, paths)
                        },
                        "subjects": {
                            key: len(ids) for key, ids in subjects_by_rep.items()
                        },
                        "provenance": str(provenance),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if brain.layout == "representations":
            modalities = (
                Modality(
                    brain.name,
                    tuple(paths),
                    format="parquet",
                    sample_id_col="sample_id",
                    allow_implicit_numeric_features=True,
                    sample_id_map=brain.sample_id_map,
                ),
            )
        else:
            modalities = tuple(
                Modality(
                    rep.key,
                    path,
                    format="parquet",
                    sample_id_col="sample_id",
                    allow_implicit_numeric_features=True,
                    sample_id_map=brain.sample_id_map,
                )
                for rep, path in zip(brain.representations, paths)
            )
        success(f"Brain · {len(modalities)} modality source(s) ready")
        return BrainRepresentationResult(
            modalities[0], selectors, provenance, manifest, modalities
        )


def attach_brain_representations(sweep: Any) -> None:
    brain = sweep.brain
    if brain is None:
        return
    if not isinstance(brain, Brain) or sweep.samples is None:
        raise TypeError("Sweep.brain requires Brain and Sweep.samples")
    current = tuple(sweep.modalities)
    existing = {modality.name for modality in current}
    names = (
        {brain.name}
        if brain.layout == "representations"
        else {rep.key for rep in brain.representations}
    )
    if len(names) != len(brain.representations) and brain.layout == "modalities":
        raise ValueError("Brain modality names must be unique")
    if existing & names:
        raise ValueError(
            f"Brain modality names conflict with existing modalities: {sorted(existing & names)}"
        )
    if brain.replace_modality is not None:
        if brain.replace_modality not in existing:
            raise ValueError(f"Missing existing modality: {brain.replace_modality}")
        current = tuple(mod for mod in current if mod.name != brain.replace_modality)
    result = brain.prepare(Path(sweep.root()) / "brain_representations")
    from dataclasses import replace

    primary = not any(mod.primary for mod in current)
    sweep.modalities = (
        *current,
        *(
            replace(mod, primary=primary and index == 0)
            for index, mod in enumerate(result.modalities)
        ),
    )
    mapping = dict(sweep.representations)
    if brain.layout == "representations":
        mapping[brain.name] = result.selectors
    else:
        for rep in brain.representations:
            if rep.key in mapping:
                raise ValueError(
                    f"Representation selector already configured for {rep.key}"
                )
            mapping[rep.key] = ((rep.key, ("all",)),)
    sweep.representations = mapping
    if brain.trace_subject_ids and sweep.brain_traceability is None:
        from .explanation import BrainTraceabilityConfig

        folder = Path(sweep.root()) / "brain_representations" / "brainharmonix_f"
        normalization = folder / "brainharmonix_f_normalization.npz"
        resolved = _resolved_brain(brain)
        source = BrainSource(
            derivatives_dir=brain.derivatives_dir,
            bids_root=brain.bids_root,
            atlas_path=resolved.atlas_paths[400],
            roi_order_verified=True,
            normalization="reference",
            preprocessing=brain.preprocessing,
            normalization_stats=normalization,
        )
        sweep.brain_traceability = BrainTraceabilityConfig(
            source=source,
            normalization_stats=normalization,
            subject_ids=brain.trace_subject_ids,
            max_features_per_target=brain.trace_max_features_per_target,
        )
