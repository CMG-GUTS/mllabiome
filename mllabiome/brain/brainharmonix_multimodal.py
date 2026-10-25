from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .cache import (
    BrainCache,
    atomic_path,
    cache_root,
    file_identity,
    input_identity,
    settings_identity,
)
from .fmriprep import discover_fmriprep_runs
from .storage import RepresentationStore


@dataclass(frozen=True, slots=True)
class StructuralScan:
    subject_id: str
    image: Path


def discover_structural_scans(derivatives_dir: str | Path) -> list[StructuralScan]:
    root = Path(derivatives_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*_desc-preproc_T1w.nii.gz")):
        if path.parent.name != "anat" or "_space-MNI152NLin2009cAsym" not in path.name:
            continue
        match = re.search(r"(?:^|_)sub-([A-Za-z0-9]+)(?:_|$)", path.name)
        if match is None:
            continue
        subject = f"sub-{match.group(1)}"
        if subject not in path.parts:
            raise ValueError(f"Structural subject and path disagree: {path}")
        candidates.setdefault(subject, []).append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No MNI152NLin2009cAsym preprocessed T1w images found in {root}. Request anatomical template-space T1w outputs from fMRIPrep."
        )
    scans = []
    for subject, paths in sorted(candidates.items()):
        priorities = (
            [p for p in paths if "_res-1_" in p.name],
            [p for p in paths if re.search(r"_res-[A-Za-z0-9]+_", p.name) is None],
            [p for p in paths if "_res-2_" in p.name],
        )
        matching = next((group for group in priorities if group), [])
        if len(matching) != 1:
            raise ValueError(
                f"Ambiguous or unsupported MNI T1w images for {subject}: {paths}"
            )
        scans.append(StructuralScan(subject, matching[0]))
    return scans


@lru_cache(maxsize=1)
def _structural_reference_file() -> Path:
    from templateflow.api import get

    result = get(
        "MNI152NLin2009cAsym",
        resolution=1,
        suffix="T1w",
        desc=None,
        extension=".nii.gz",
    )
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise ValueError(
                f"Expected one 1 mm MNI152NLin2009cAsym T1w reference, found {len(result)}"
            )
        result = result[0]
    if result is None:
        raise FileNotFoundError(
            "TemplateFlow MNI152NLin2009cAsym 1 mm T1w reference not found"
        )
    path = Path(result).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_t1_volume(scan: StructuralScan) -> np.ndarray:
    import nibabel as nib
    from nibabel.processing import resample_from_to

    image = nib.as_closest_canonical(nib.load(str(scan.image)))
    reference = nib.as_closest_canonical(nib.load(str(_structural_reference_file())))
    if len(image.shape) != 3 or len(reference.shape) != 3:
        raise ValueError(f"Expected 3D T1w image and reference: {scan.image}")
    if (
        image.header.get_xyzt_units()[0] != "mm"
        or reference.header.get_xyzt_units()[0] != "mm"
    ):
        raise ValueError(
            f"T1w image and reference must use mm spatial units: {scan.image}"
        )
    spacing = np.linalg.norm(np.asarray(image.affine)[:3, :3], axis=0)
    reference_spacing = np.linalg.norm(np.asarray(reference.affine)[:3, :3], axis=0)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"Invalid spatial sampling of T1w image: {scan.image}")
    if not np.allclose(reference_spacing, 1.0, atol=0.05, rtol=0):
        raise ValueError(f"Unexpected reference voxel spacing: {reference_spacing}")
    target = (160, 192, 160)
    if any(length < required for length, required in zip(reference.shape, target)):
        raise ValueError(
            f"MNI T1w reference is smaller than BrainHarmonix-S input {target}: {reference.shape}"
        )
    if image.shape == reference.shape and np.allclose(
        image.affine, reference.affine, atol=1e-5, rtol=0
    ):
        aligned = image
    else:
        aligned = resample_from_to(
            image, (reference.shape, reference.affine), order=1, cval=0
        )
    data = np.asarray(aligned.get_fdata(dtype=np.float32), dtype=np.float32)
    if not np.isfinite(data).all() or not np.any(data != 0):
        raise ValueError(f"Invalid or nonoverlapping MNI T1w intensities: {scan.image}")
    start = tuple((a - b) // 2 for a, b in zip(data.shape, target))
    volume = data[tuple(slice(a, a + b) for a, b in zip(start, target))]
    center = float(np.mean(data, dtype=np.float64))
    spread = float(np.std(data, dtype=np.float64))
    if spread <= 1e-8 or not np.any(volume != 0):
        raise ValueError(f"Empty or constant T1w crop: {scan.image}")
    return np.ascontiguousarray((volume - center) / spread, dtype=np.float32)


def _device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if requested == "mps" and not getattr(torch.backends, "mps", None).is_available():
        raise RuntimeError("MPS unavailable")
    return requested


CHECKPOINT_REPOSITORY = "eugenehp/BrainHarmony"
CHECKPOINT_REVISION = "510bb9ade07bad6d3d408ecc745e310f381c7495"
CHECKPOINT_FILES = {
    "s": (
        "harmonix-s.safetensors",
        "295652bd3f744baffb2b6305cb3b0639788ce9f0496421d32d6a954ccaaaa608",
    ),
    "sf": (
        "harmonizer.safetensors",
        "80c319256f8758b433a6507664e9fcc4b7994ccbdcaf57cbbc6bb3b54e0bc969",
    ),
}


def _checkpoint(name: str, supplied: str | Path | None) -> Path:
    if name not in CHECKPOINT_FILES:
        raise ValueError(f"Unsupported BrainHarmonix checkpoint type: {name}")
    if supplied is not None and str(supplied) != "auto":
        result = Path(supplied).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    env = os.environ.get(f"MLLABIOME_BRAINHARMONIX_{name.upper()}_CHECKPOINT")
    if env and env != "auto":
        return _checkpoint(name, env)
    filename, expected_digest = CHECKPOINT_FILES[name]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install mllabiome[brainharmonix] for automatic pretrained checkpoint downloads"
        ) from exc
    try:
        result = Path(
            hf_hub_download(
                repo_id=CHECKPOINT_REPOSITORY,
                filename=filename,
                revision=CHECKPOINT_REVISION,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download BrainHarmonix-{name.upper()} weights from "
            f"{CHECKPOINT_REPOSITORY}@{CHECKPOINT_REVISION}/{filename}. "
            f"The automatically hosted weights are third-party conversions, not officially validated. "
            f"Check network access or set BrainRepresentation(checkpoint=Path(...))."
        ) from exc
    digest = hashlib.sha256()
    with result.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise ValueError(
            f"Checkpoint digest mismatch for {result}. Remove the corrupted Hugging Face cache entry before retrying."
        )
    return result


def _checkpoint_format(path: Path) -> str:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) < 8:
            raise ValueError(f"BrainHarmonix checkpoint is empty or truncated: {path}")
        header_length = int.from_bytes(prefix, "little")
        if 2 <= header_length <= min(path.stat().st_size - 8, 32 * 1024 * 1024):
            header_start = stream.read(min(header_length, 128)).lstrip()
            if header_start.startswith(b"{"):
                return "safetensors"
        if prefix.startswith(b"PK\x03\x04") or prefix[:2] in (
            b"\x80\x02",
            b"\x80\x03",
            b"\x80\x04",
            b"\x80\x05",
        ):
            return "pytorch"
    raise ValueError(
        f"Unrecognized BrainHarmonix checkpoint format: {path}. "
        f"Expected SafeTensors or a PyTorch state dictionary. "
        f"Check the file contents and checkpoint override; do not disable weights_only."
    )


def _weights(torch: Any, path: Path) -> dict[str, Any]:
    file_format = _checkpoint_format(path)
    if file_format == "safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Install safetensors to load BrainHarmonix checkpoints"
            ) from exc
        try:
            state = load_file(str(path), device="cpu")
        except Exception as exc:
            raise ValueError(
                f"Could not read SafeTensors checkpoint {path}: {exc}"
            ) from exc
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                f"Could not safely read PyTorch BrainHarmonix checkpoint {path}: {exc}. "
                f"Do not set weights_only=False for automatically obtained checkpoints."
            ) from exc
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no model state dictionary: {path}")
    return state


def _attention_blocks(torch: Any, *, dim: int, layers: int, heads: int):
    nn = torch.nn
    functional = torch.nn.functional

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x, allowed=None):
            batch, length, channels = x.shape
            qkv = (
                self.qkv(x)
                .reshape(batch, length, 3, heads, channels // heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            mask = allowed[:, None, None, :] if allowed is not None else None
            result = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return self.proj(result.transpose(1, 2).reshape(batch, length, channels))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 4)
            self.fc2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.fc2(functional.gelu(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attention()
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = MLP()

        def forward(self, x, allowed=None):
            x = x + self.attn(self.norm1(x), allowed)
            return x + self.mlp(self.norm2(x))

    return nn.ModuleList(Block() for _ in range(layers))


def _structural_model(torch: Any):
    nn = torch.nn

    class Patch(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv3d(1, 768, kernel_size=16, stride=16)

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = Patch()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1201, 768), requires_grad=False
            )
            self.blocks = _attention_blocks(torch, dim=768, layers=12, heads=12)
            self.norm = nn.LayerNorm(768, eps=1e-6)

        def forward(self, x):
            x = self.patch_embed(x) + self.pos_embed[:, 1:, :]
            x = torch.cat((self.cls_token + self.pos_embed[:, :1, :], x), dim=1)
            for block in self.blocks:
                x = block(x)
            return self.norm(x)[:, 1:, :]

    return Model()


def _fusion_model(torch: Any, token_count: int):
    nn = torch.nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 8401, 768), requires_grad=False
            )
            self.pos_drop = nn.Dropout(0.0)
            self.latent_tokens = nn.Parameter(torch.zeros(token_count, 768))
            self.enc_latent_token_positional_embedding = nn.Parameter(
                torch.zeros(token_count, 768)
            )
            self.blocks = _attention_blocks(torch, dim=768, layers=12, heads=12)

        def forward(self, combined, mask):
            x = torch.cat(
                (self.cls_token.expand(len(combined), -1, -1), combined), dim=1
            )
            x = self.pos_drop(x + self.pos_embed)
            hubs = (
                (self.latent_tokens + self.enc_latent_token_positional_embedding)
                .unsqueeze(0)
                .expand(len(x), -1, -1)
            )
            x = torch.cat((x, hubs), dim=1)
            permitted = torch.cat(
                (
                    torch.ones((len(mask), 1), device=mask.device, dtype=torch.bool),
                    mask,
                    torch.ones(
                        (len(mask), 1200 + token_count),
                        device=mask.device,
                        dtype=torch.bool,
                    ),
                ),
                dim=1,
            )
            for block in self.blocks:
                x = block(x, permitted)
            return torch.nn.functional.layer_norm(
                x[:, -token_count:, :].mean(dim=1), (768,)
            )

    return Model()


def _load_pretrained(torch: Any, model: Any, name: str, path: Path) -> None:
    state = _weights(torch, path)
    expected = model.state_dict()
    required = {
        key
        for key in expected
        if key.startswith("blocks.")
        or key.startswith("patch_embed.")
        or key
        in {
            "pos_embed",
            "cls_token",
            "latent_tokens",
            "enc_latent_token_positional_embedding",
            "norm.weight",
            "norm.bias",
        }
    }
    candidates = []
    for prefix in (
        "",
        "module.",
        "model.",
        "encoder.",
        "module.encoder.",
        "model.encoder.",
    ):
        compatible = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
            and key[len(prefix) :] in expected
            and getattr(value, "shape", None) == expected[key[len(prefix) :]].shape
        }
        candidates.append(compatible)
    usable = max(candidates, key=lambda candidate: len(candidate.keys() & required))
    missing = sorted(required - usable.keys())
    if missing:
        raise ValueError(
            f"Incompatible BrainHarmonix-{name.upper()} checkpoint {path}: missing {len(missing)} architecture tensors. Examples: {missing[:5]}. No random-weight embeddings will be produced."
        )
    model.load_state_dict(usable, strict=False)


class BrainHarmonixSEncoder:
    def __init__(
        self, checkpoint: str | Path | None = None, device: str = "auto"
    ) -> None:
        import torch

        self.torch = torch
        self.device = _device(torch, device)
        self.checkpoint = _checkpoint("s", checkpoint)
        self.model = _structural_model(torch)
        _load_pretrained(torch, self.model, "s", self.checkpoint)
        self.model = self.model.to(self.device).eval()

    def encode_tokens(self, scan: StructuralScan):
        volume = load_t1_volume(scan)
        tensor = self.torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            result = self.model(tensor)
        if result.shape != (1, 1200, 768) or not self.torch.isfinite(result).all():
            raise ValueError(f"Invalid structural encoder output for {scan.subject_id}")
        return result

    def encode(self, scan: StructuralScan) -> np.ndarray:
        result = self.encode_tokens(scan).mean(dim=1).squeeze(0)
        return result.detach().float().cpu().numpy().astype(np.float32)


class BrainHarmonixSFEncoder:
    def __init__(
        self,
        *,
        structural_checkpoint: str | Path | None = None,
        fusion_checkpoint: str | Path | None = None,
        device: str = "auto",
        n_hubs: int = 128,
        normalization: str = "per_run",
        normalization_stats: str | Path | None = None,
        reference_observations: Any = None,
    ) -> None:
        import torch
        from .brainharmonix import BrainHarmonixFEncoder

        self.torch = torch
        self.device = _device(torch, device)
        self.structural = BrainHarmonixSEncoder(structural_checkpoint, self.device)
        self.functional = BrainHarmonixFEncoder(
            device=self.device, normalization=normalization
        )
        if normalization == "reference":
            if normalization_stats is not None:
                self.functional.load_normalization(normalization_stats)
            elif reference_observations is not None:
                self.functional.fit_reference(reference_observations)
            else:
                raise ValueError(
                    "Reference normalization requires saved statistics or independent reference time series"
                )
        self.checkpoint = _checkpoint("sf", fusion_checkpoint)
        self.model = _fusion_model(torch, n_hubs)
        _load_pretrained(torch, self.model, "sf", self.checkpoint)
        self.model = self.model.to(self.device).eval()

    def encode(
        self, structural: StructuralScan, series: np.ndarray, tr: float
    ) -> np.ndarray:
        _, fmri_tokens, mask = self.functional.encode(series, tr)
        t1_tokens = self.structural.encode_tokens(structural)
        fmri = self.torch.from_numpy(fmri_tokens).unsqueeze(0).to(self.device)
        valid = self.torch.from_numpy(mask).unsqueeze(0).to(self.device)
        if tuple(fmri.shape) != (1, 7200, 768) or tuple(t1_tokens.shape) != (
            1,
            1200,
            768,
        ):
            raise ValueError("Invalid BrainHarmonix input-token ordering")
        combined = self.torch.cat((fmri, t1_tokens), dim=1)
        with self.torch.inference_mode():
            pooled = self.model(combined, valid)
        result = pooled.squeeze(0).detach().float().cpu().numpy()
        if result.shape != (768,) or not np.isfinite(result).all():
            raise ValueError("Non-finite BrainHarmonix fused embedding")
        return result.astype(np.float32)


def prepare_harmonix_representation(brain: Any, rep: Any, output_dir: Path) -> Path:
    from ..console import info, success
    from .brainharmonix import _extract_one
    from .progress import active_progress

    method = rep.method
    prefix = "bhs" if method == "brainharmonix_s" else "bhsf"
    cols = tuple(f"brain_{method}__{prefix}_{i:03d}" for i in range(768))
    store = RepresentationStore(output_dir, method, cols)
    with store.locked():
        structural = {
            item.subject_id: item
            for item in discover_structural_scans(brain.derivatives_dir)
        }
        if method == "brainharmonix_sf":
            runs = discover_fmriprep_runs(
                brain.derivatives_dir, bids_root=brain.bids_root
            )
            absent = {r.subject_id for r in runs} - structural.keys()
            if absent:
                raise ValueError(
                    f"Cannot fuse fMRI and T1 for subjects missing structural T1w: {sorted(absent)}"
                )
            items = [(r.subject_id, structural[r.subject_id], r) for r in runs]
        else:
            items = [
                (subject, scan, None) for subject, scan in sorted(structural.items())
            ]
        cache = BrainCache(
            cache_root(brain.cache_dir, output_dir), migrate_legacy=False
        )
        mode = (
            "reference"
            if brain.normalization == "auto"
            and (
                brain.normalization_stats is not None
                or brain.reference_derivatives_dir is not None
            )
            else "per_run"
            if brain.normalization == "auto"
            else brain.normalization
        )
        reference_inputs = []
        if method == "brainharmonix_sf" and mode == "reference":
            if brain.normalization_stats is not None:
                reference_inputs.append(file_identity(brain.normalization_stats))
            elif brain.reference_derivatives_dir is not None:
                reference_runs = discover_fmriprep_runs(
                    brain.reference_derivatives_dir, bids_root=brain.reference_bids_root
                )
                overlap = set(structural) & {r.subject_id for r in reference_runs}
                if overlap:
                    raise ValueError(
                        f"Independent functional normalization subjects overlap target cohort: {sorted(overlap)}"
                    )
                reference_inputs = [input_identity(r) for r in reference_runs]
            else:
                raise ValueError(
                    "Reference normalization requires independent statistics or reference subjects"
                )
        checkpoint = _checkpoint(
            "s" if method == "brainharmonix_s" else "sf",
            getattr(rep, "checkpoint", None),
        )
        structural_checkpoint = (
            _checkpoint("s", getattr(rep, "structural_checkpoint", None))
            if method == "brainharmonix_sf"
            else checkpoint
        )
        structural_reference = file_identity(_structural_reference_file())
        atlas = (
            file_identity(brain.atlas_paths[400])
            if method == "brainharmonix_sf"
            else None
        )
        records = []
        for subject, scan, run in items:
            inputs = [
                file_identity(scan.image),
                file_identity(checkpoint),
                file_identity(structural_checkpoint),
            ]
            if run is not None:
                inputs.extend(
                    (
                        input_identity(run),
                        atlas,
                        settings_identity(brain.preprocessing),
                        mode,
                        reference_inputs,
                    )
                )
            key = cache.key(
                "embedding",
                method,
                subject,
                inputs,
                structural_reference,
                "templateflow_res1_linear_full_zscore_center_crop_160_192_160_v2",
                "mean_pool_frozen_tokens_v1",
                foundation=True,
            )
            records.append((subject, key))
        previous, keys = store.existing() if not brain.overwrite else ({}, {})
        if len(previous) == len(items) and all(
            subject in previous and keys.get(subject) == key for subject, key in records
        ):
            _save_provenance(
                output_dir,
                method,
                brain,
                items,
                checkpoint,
                structural_checkpoint,
                mode,
                reference_inputs,
            )
            success(f"Brain · {method} · reused · {store.path}")
            return store.path
        values = {
            subject: previous[subject].astype(np.float32)
            for subject, key in records
            if keys.get(subject) == key and subject in previous
        }
        pending = [
            (subject, scan, run, key)
            for (subject, scan, run), (_, key) in zip(items, records)
            if subject not in values
        ]
        if pending:
            info(
                f"Brain · {method} · {len(values)}/{len(items)} reused · {len(pending)} to encode"
            )
            if method == "brainharmonix_s":
                encoder = BrainHarmonixSEncoder(checkpoint)
            else:
                reference_observations = None
                if mode == "reference" and brain.reference_derivatives_dir is not None:
                    reference_observations = [
                        (
                            _extract_one(
                                run,
                                brain.preprocessing,
                                Path(brain.atlas_paths[400]),
                                cache.root,
                                brain.overwrite,
                            )[1],
                            run.repetition_time,
                        )
                        for run in reference_runs
                    ]
                encoder = BrainHarmonixSFEncoder(
                    structural_checkpoint=structural_checkpoint,
                    fusion_checkpoint=checkpoint,
                    normalization=mode,
                    normalization_stats=brain.normalization_stats,
                    reference_observations=reference_observations,
                )
            tracker = active_progress()
            if tracker is not None:
                tracker.begin_units(f"{method} · encoding subjects", len(pending))
            for index, (subject, scan, run, key) in enumerate(pending, 1):
                staged = None if brain.overwrite else store.staged(subject, key)
                if staged is not None:
                    values[subject] = staged.astype(np.float32)
                    if tracker is not None:
                        tracker.working(subject)
                        tracker.advance()
                    continue
                if method == "brainharmonix_s":
                    vector = encoder.encode(scan)
                else:
                    result = _extract_one(
                        run,
                        brain.preprocessing,
                        Path(brain.atlas_paths[400]),
                        cache.root,
                        brain.overwrite,
                    )
                    vector = encoder.encode(scan, result[1], run.repetition_time)
                values[subject] = vector
                store.stage(subject, key, vector)
                if tracker is not None:
                    tracker.working(subject)
                    tracker.advance()
                elif index == 1 or index == len(pending) or index % 10 == 0:
                    info(f"Brain · {method} · {index}/{len(pending)} · {subject}")
            del encoder
        store.commit(records, values)
        _save_provenance(
            output_dir,
            method,
            brain,
            items,
            checkpoint,
            structural_checkpoint,
            mode,
            reference_inputs,
        )
        if not brain.retain_timeseries and method == "brainharmonix_sf":
            import shutil

            shutil.rmtree(cache.root / "timeseries", ignore_errors=True)
        success(f"Brain · {method} · ready · {store.path}")
        return store.path


def _save_provenance(
    output_dir: Path,
    method: str,
    brain: Any,
    items: list,
    checkpoint: Path,
    structural_checkpoint: Path,
    normalization: str,
    reference: list,
) -> None:
    payload = {
        "schema": "mllabiome.brain.brainharmonix_structural.v2",
        "representation": method,
        "encoder_source": "Dong et al. Brain Harmony, NeurIPS 2025",
        "structural_preprocessing": "RAS_reorient_resample_to_TemplateFlow_MNI152NLin2009cAsym_res1_linear_full_zscore_center_crop_160_192_160",
        "structural_reference": file_identity(_structural_reference_file()),
        "structural_token_count": 1200,
        "functional_token_count": 7200 if method == "brainharmonix_sf" else None,
        "harmonizer_hub_count": 128 if method == "brainharmonix_sf" else None,
        "pooling": "mean_across_unmasked_structural_tokens"
        if method == "brainharmonix_s"
        else "mean_across_harmonizer_hub_tokens_then_untrained_layer_normalization",
        "embedding_features": 768,
        "structural_checkpoint": file_identity(structural_checkpoint),
        "joint_checkpoint": file_identity(checkpoint)
        if method == "brainharmonix_sf"
        else None,
        "default_checkpoint_repository": CHECKPOINT_REPOSITORY,
        "default_checkpoint_revision": CHECKPOINT_REVISION,
        "default_checkpoint_source": "third_party_converted_unvalidated",
        "functional_normalization": normalization
        if method == "brainharmonix_sf"
        else None,
        "functional_normalization_reference": reference
        if method == "brainharmonix_sf"
        else None,
        "subject_ids": [subject for subject, _, _ in items],
        "input_images": {subject: str(scan.image) for subject, scan, _ in items},
        "limitations": "FMRIPrep-derived T1w preprocessing has not been benchmark-validated against Brain Harmony published structural inputs; joint embedding may differ from original training pipeline. No direct anatomical localization for latent coordinates.",
    }
    path = output_dir / f"{method}.provenance.json"
    with atomic_path(path) as pending:
        pending.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
