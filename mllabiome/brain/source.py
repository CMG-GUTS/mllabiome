from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Mapping

import pandas as pd

from .timeseries import BOLDPreprocessing
from .progress import active_progress


def _source_fingerprint(source: "BrainSource") -> str:
    configuration = json.dumps(asdict(source), default=str, sort_keys=True)
    digest = hashlib.sha256(configuration.encode("utf-8"))
    for package in ("mllabiome", "brainmarks", "nilearn"):
        try:
            release = version(package)
        except PackageNotFoundError:
            release = "not-installed"
        digest.update(f"{package}:{release}".encode("utf-8"))
    for location in (
        source.derivatives_dir,
        source.reference_derivatives_dir,
        source.bids_root,
        source.reference_bids_root,
        source.normalization_stats,
        source.atlas_path,
    ):
        if location is None:
            continue
        path = Path(location).expanduser().resolve()
        digest.update(str(path).encode("utf-8"))
        if path.is_file():
            files = (path,)
        elif path.is_dir():
            files = (
                item
                for item in sorted(path.rglob("*"))
                if item.is_file()
                and item.name.endswith((".nii.gz", ".json", ".tsv", ".npz"))
            )
        else:
            files = ()
        for file in files:
            metadata = file.stat()
            digest.update(
                f"{file}:{metadata.st_size}:{metadata.st_mtime_ns}".encode("utf-8")
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BrainSource:
    derivatives_dir: Path | str
    representation: str = "brainharmonix_f"
    bids_root: Path | str | None = None
    atlas_path: Path | str | None = None
    reference_derivatives_dir: Path | str | None = None
    reference_bids_root: Path | str | None = None
    normalization_stats: Path | str | None = None
    normalization: str = "auto"
    roi_order_verified: bool = False
    sample_id_map: Mapping[str, str] | None = None
    preprocessing: BOLDPreprocessing = field(default_factory=BOLDPreprocessing)

    def prepare(
        self,
        output_dir: Path | str,
        *,
        overwrite: bool = False,
        name: str = "brain",
        primary: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ):
        folder = Path(output_dir)
        signature = folder / "brain_source_cache.json"
        tracker = active_progress()
        if tracker is not None:
            tracker.phase(f"Checking {self.representation} input signatures and cache")
        fingerprint = _source_fingerprint(self)
        prefixes = {
            "brainharmonix_f": "bhf_",
            "pearson_fc": "fc_",
            "partial_fc": "pfc_",
            "alff": "alff_",
            "falff": "falff_",
            "tissue_volumes": "tissue_",
        }
        prefix = prefixes.get(self.representation)
        if prefix is None:
            raise ValueError(f"Unsupported brain representation: {self.representation}")
        base = (
            "brainharmonix_f_features.tsv"
            if self.representation == "brainharmonix_f"
            else f"brain_{self.representation}_features.tsv"
        )
        original = folder / base
        aligned = (
            folder / f"{original.stem}_aligned.tsv"
            if self.sample_id_map is not None
            else original
        )
        if not overwrite and signature.exists():
            manifest = json.loads(signature.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Brain source data/configuration changed. Explicitly recompute with overwrite=True"
                )
            if not aligned.is_file() or (
                self.representation != "tissue_volumes"
                and not (folder / "brain_feature_atlas_labels.json").is_file()
            ):
                raise FileNotFoundError(
                    "Brain source cache is incomplete; recompute with overwrite=True"
                )
            if tracker is not None:
                tracker.phase(f"Reusing verified {self.representation} cache")
            from ..modalities import Modality

            return Modality(
                name=name,
                path=aligned,
                format="table",
                primary=primary,
                sample_id_col="sample_id",
                feature_prefixes=prefix,
            )
        if not overwrite and original.exists():
            raise FileExistsError(
                "Brain feature output exists without a verified cache signature; pass overwrite=True"
            )
        if self.representation == "brainharmonix_f":
            from .brainharmonix import embed_fmriprep

            result = embed_fmriprep(
                self.derivatives_dir,
                output_dir,
                bids_root=self.bids_root,
                atlas_path=self.atlas_path,
                reference_derivatives_dir=self.reference_derivatives_dir,
                reference_bids_root=self.reference_bids_root,
                normalization_stats=self.normalization_stats,
                normalization=self.normalization,
                roi_order_verified=self.roi_order_verified,
                preprocessing=self.preprocessing,
                overwrite=overwrite,
                progress_callback=progress_callback,
            )
            modality = result.as_modality(name=name, primary=primary)
        else:
            from .handcrafted import ALL_REPRESENTATIONS, export_handcrafted_fmriprep

            if self.representation not in ALL_REPRESENTATIONS:
                raise ValueError(
                    f"Unsupported brain representation: {self.representation}"
                )
            result = export_handcrafted_fmriprep(
                self.derivatives_dir,
                output_dir,
                bids_root=self.bids_root,
                atlas_path=self.atlas_path,
                preprocessing=self.preprocessing,
                representations=(self.representation,),
                overwrite=overwrite,
            )
            modality = result.as_modality(
                self.representation, name=name, primary=primary
            )
        if self.sample_id_map is not None:
            mapping = {str(k): str(v) for k, v in self.sample_id_map.items()}
            original = pd.read_csv(modality.path, sep="\t", dtype={"sample_id": str})
            ids = original["sample_id"].astype(str)
            if not set(ids).issubset(mapping) or any(
                not mapping.get(sid) for sid in ids
            ):
                raise ValueError(
                    f"Missing explicit subject-to-sample mappings: {sorted(set(ids) - set(mapping))}"
                )
            translated = ids.map(mapping)
            if translated.duplicated().any():
                raise ValueError(
                    "Subject-to-sample mapping creates duplicate identifiers"
                )
            original["sample_id"] = translated
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            aligned = Path(output_dir) / f"{Path(modality.path).stem}_aligned.tsv"
            if aligned.exists() and not overwrite:
                raise FileExistsError(f"Aligned brain feature table exists: {aligned}")
            original.to_csv(aligned, sep="\t", index=False)
            pd.DataFrame({"bids_subject": ids, "sample_id": translated}).to_csv(
                folder / "brain_subject_alignment.tsv",
                sep="\t",
                index=False,
            )
            modality = replace(modality, path=aligned)
        if self.representation != "tissue_volumes":
            from .traceability import Schaefer400Provenance

            Schaefer400Provenance.from_nilearn(self.atlas_path).save(
                Path(output_dir) / "brain_feature_atlas_labels.json"
            )
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "mllabiome.brain.source_cache.v1",
            "fingerprint": _source_fingerprint(self),
            "feature_table": str(modality.path),
        }
        signature.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return modality


def with_brain_modality(
    data: Mapping[str, Any],
    source: BrainSource,
    output_dir: Path | str,
    *,
    name: str = "brain",
    primary: bool = False,
    overwrite: bool = False,
    replace_modality: str | None = None,
) -> dict[str, Any]:
    if "samples" not in data or "modalities" not in data:
        raise ValueError(
            "Expected modality-based sweep data with samples and modalities"
        )
    original = tuple(data["modalities"])
    known = {modality.name for modality in original}
    if replace_modality is not None and replace_modality not in known:
        raise ValueError(
            f"Requested modality replacement {replace_modality!r} not found; available: {sorted(known)}"
        )
    other_brain = any(
        "brain" in str(key).casefold() or "fmri" in str(key).casefold() for key in known
    )
    if replace_modality is None and name not in known and other_brain:
        raise ValueError(
            "An existing brain-related modality has a different name; "
            "specify replace_modality to prevent accidental double inclusion"
        )
    existing = tuple(
        modality
        for modality in original
        if modality.name not in {name, replace_modality}
    )
    if not any(bool(item.primary) for item in existing):
        primary = True
    created = source.prepare(
        output_dir, name=name, primary=primary, overwrite=overwrite
    )
    modalities = (*existing, created)
    sample_settings = data["samples"]
    if hasattr(sample_settings, "path") and hasattr(sample_settings, "sample_id_col"):
        identifiers = pd.read_csv(
            sample_settings.path,
            sep=None,
            engine="python",
            usecols=[sample_settings.sample_id_col],
            dtype=str,
        )[sample_settings.sample_id_col]
        found = pd.read_csv(created.path, sep="\t", usecols=["sample_id"], dtype=str)[
            "sample_id"
        ]
        overlap = set(identifiers) & set(found)
        if len(overlap) < 2:
            raise ValueError(
                "Fewer than two brain subjects match the sweep samples. "
                "Supply an explicit BrainSource.sample_id_map when BIDS IDs differ"
            )
    if sum(bool(item.primary) for item in modalities) != 1:
        raise ValueError(
            "Exactly one primary modality is required; adjust primary and existing modality declarations"
        )
    return {**data, "modalities": modalities}
