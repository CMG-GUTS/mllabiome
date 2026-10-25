from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from .cache import atomic_path, file_identity, input_identity, signature
from .fmriprep import discover_fmriprep_runs
from .progress import active_progress
from .storage import RepresentationStore
from ..console import info, success


REPOSITORY = "https://github.com/OneMore1/Omni-fMRI.git"
CHECKPOINT_REPOSITORY = "OneMore1/Omni-fMRI"
CHECKPOINT_FILE = "checkpoint.pth"
CHECKPOINT_REVISION = "0bd3c34ba191e6ed9ac2a453b314f79178b7dfb2"
CHECKPOINT_SHA256 = "cb1f3631ba1b481f024121fae96f342c7d5c1ad290f4c62ff05fc674c8d44a40"
SPATIAL_SHAPE = (96, 96, 96)
SEGMENT_FRAMES = 40
EMBEDDING_SIZE = 768
EXTRACTOR_PROTOCOL = "omnifmri_official_voxel_cls_segment_mean_v1"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _repository() -> tuple[Path, str]:
    root = (
        Path(
            os.environ.get(
                "MLLABIOME_OMNI_CODE_DIR",
                Path.home() / ".cache" / "mllabiome" / "models" / "omni-fmri",
            )
        )
        .expanduser()
        .resolve()
    )
    if not (root / "src" / "models" / "mae_model.py").is_file():
        if root.exists():
            raise FileNotFoundError(f"Incomplete Omni-fMRI source checkout: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", REPOSITORY, str(root)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Unable to retrieve the official Omni-fMRI source from {REPOSITORY}"
            ) from exc
    if not (root / "extract_feat.py").is_file() or not (root / ".git").is_dir():
        raise ValueError(f"Not an official Omni-fMRI source checkout: {root}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot identify Omni-fMRI code revision: {root}") from exc
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"Invalid Omni-fMRI source revision: {revision}")
    return root, revision


def _checkpoint(override: str | Path | None) -> tuple[Path, str]:
    if override is not None and str(override) != "auto":
        path = Path(override).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, f"local:{_digest(path)}"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Omni-fMRI requires the optional mllabiome[omni] dependencies"
        ) from exc
    checkpoint = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPOSITORY,
            filename=CHECKPOINT_FILE,
            revision=CHECKPOINT_REVISION,
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checksum = _digest(checkpoint)
    if checksum != CHECKPOINT_SHA256:
        raise ValueError(f"Omni-fMRI checkpoint has unexpected SHA-256: {checkpoint}")
    return checkpoint, f"{CHECKPOINT_REPOSITORY}@{CHECKPOINT_REVISION}:{checksum}"


def _safe_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        dtype_names = (
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "bool",
        )
        globals_allowed = [np.dtype, *(type(np.dtype(name)) for name in dtype_names)]
        scalar = (
            np._core.multiarray.scalar
            if hasattr(np, "_core")
            else np.core.multiarray.scalar
        )
        globals_allowed.extend(
            (
                (scalar, "numpy._core.multiarray.scalar"),
                (scalar, "numpy.core.multiarray.scalar"),
            )
        )
        with torch.serialization.safe_globals(globals_allowed):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Omni-fMRI checkpoint cannot be safely decoded: {path}. "
            f"Restricted PyTorch loading reported: {exc}. "
            "No unsafe pickle loading was attempted."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected Omni-fMRI checkpoint structure: {path}")
    return checkpoint


def _validate_encoder_state(loaded: Any) -> None:
    unused = {"encoder.classifier.weight", "encoder.classifier.bias"}
    missing = [
        name
        for name in loaded.missing_keys
        if name.startswith("encoder.") and name not in unused
    ]
    if missing:
        raise ValueError(
            f"Omni-fMRI checkpoint does not fully initialize its encoder: {missing[:6]}"
        )


def _official_encoder(code: Path, checkpoint_path: Path):
    import torch
    from torch import nn

    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    try:
        from src.models.mae_model import AdaptiveMAE
        from src.models.patch_embed_3d import TokenizedZeroConvPatchAttn3D
    except ImportError as exc:
        raise ImportError(
            "Omni-fMRI needs the optional mllabiome[omni] dependencies and official source code"
        ) from exc
    checkpoint = _safe_checkpoint(checkpoint_path)
    config = checkpoint.get("config")
    if config is None:
        import yaml

        config = yaml.safe_load(
            (code / "configs" / "pretrain.yaml").read_text(encoding="utf-8")
        )
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Omni-fMRI checkpoint lacks a valid model configuration")
    m = config["model"]
    if (
        tuple(m["img_size"]) != SPATIAL_SHAPE
        or int(m["in_chans"]) != SEGMENT_FRAMES
        or int(m["embed_dim"]) != EMBEDDING_SIZE
    ):
        raise ValueError(
            "Omni-fMRI checkpoint has incompatible spatial, temporal or embedding dimensions"
        )
    model = AdaptiveMAE(
        img_size=tuple(m["img_size"]),
        patch_size=tuple(m["patch_size"]),
        in_chans=int(m["in_chans"]),
        embed_dim=int(m["embed_dim"]),
        depth=int(m["depth"]),
        qkv_bias=bool(m["qkv_bias"]),
        qk_norm=bool(m["qk_norm"]),
        num_heads=int(m["num_heads"]),
        decoder_embed_dim=int(m["decoder_embed_dim"]),
        drop_path_rate=float(m["drop_path_rate"]),
        decoder_depth=int(m["decoder_depth"]),
        decoder_num_heads=int(m["decoder_num_heads"]),
        mlp_ratio=float(m["mlp_ratio"]),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=float(m["mask_ratio"]),
        mixed_patch_embed=TokenizedZeroConvPatchAttn3D,
        patch_norm=bool(m["enable_patch_norm"]),
        gate_attention=m["gate_attention"],
    )
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Omni-fMRI weights must be a state dictionary")
    loaded = model.load_state_dict(state, strict=False)
    _validate_encoder_state(loaded)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model.encoder.to(device).eval()
    return encoder, device


def _spatial_pad_crop(
    data: np.ndarray, shape: tuple[int, int, int] = SPATIAL_SHAPE
) -> np.ndarray:
    if data.ndim != 4 or not np.isfinite(data).all():
        raise ValueError("Omni-fMRI requires finite 4D voxel BOLD data")
    slices = []
    pads = []
    for source, target in zip(data.shape[:3], shape):
        if source >= target:
            left = (source - target) // 2
            slices.append(slice(left, left + target))
            pads.append((0, 0))
        else:
            extra = target - source
            slices.append(slice(None))
            pads.append((extra // 2, extra - extra // 2))
    cropped = data[(*slices, slice(None))]
    if any(pair != (0, 0) for pair in pads):
        cropped = np.pad(cropped, (*pads, (0, 0)), mode="constant", constant_values=0.0)
    if cropped.shape[:3] != shape:
        raise ValueError(f"Unexpected Omni-fMRI voxel shape: {cropped.shape}")
    return cropped


def _normalized_volume(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Omni-fMRI requires the optional mllabiome[omni] dependencies"
        ) from exc
    image = nib.load(str(path))
    if len(image.shape) != 4 or image.shape[3] < SEGMENT_FRAMES:
        raise ValueError(f"Omni-fMRI requires at least 40 BOLD frames: {path}")
    data = _spatial_pad_crop(np.asarray(image.dataobj, dtype=np.float32))
    mask = data != 0.0
    if not np.any(mask):
        raise ValueError(f"Empty Omni-fMRI BOLD volume: {path}")
    values = data[mask]
    mean, std = float(values.mean()), float(values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
        raise ValueError(f"Invalid Omni-fMRI BOLD intensity distribution: {path}")
    data[mask] = (values - mean) / (std + 1e-8)
    return data


def _extract_cls(backbone: Any, sample: Any) -> np.ndarray:
    import torch

    with torch.inference_mode():
        input_dict = backbone.patch_tokenizer(sample)
        tokens, _, _, _, _ = backbone.mixed_patch(
            sample, backbone.pos_embed, input_dict, current_img_size=sample.shape[2:]
        )
        seqlens = torch.as_tensor(
            input_dict["seqlens"], device=sample.device, dtype=torch.long
        )
        if seqlens.numel() != 1 or int(seqlens[0]) < 1:
            raise ValueError("Omni-fMRI tokenization yielded no usable patches")
        valid_mask = torch.arange(tokens.shape[1], device=sample.device).unsqueeze(
            0
        ) < seqlens.unsqueeze(1)
        packed = tokens[valid_mask]
        cumulative = torch.cat(
            [
                torch.zeros(1, device=sample.device, dtype=torch.int32),
                seqlens.cumsum(0, dtype=torch.int32),
            ]
        )
        encoded = backbone.forward_features(
            packed, cu_seqlens=cumulative.contiguous(), max_seqlen=int(seqlens.max())
        )[-1]
        vector = encoded[0].detach().float().cpu().numpy()
    if vector.shape != (EMBEDDING_SIZE,) or not np.isfinite(vector).all():
        raise ValueError("Invalid pretrained Omni-fMRI CLS token")
    return np.asarray(vector, dtype=np.float32)


def _encode_run(run: Any, backbone: Any, device: Any) -> tuple[np.ndarray, int]:
    import torch

    volume = _normalized_volume(run.bold)
    length = volume.shape[-1]
    segments = length // SEGMENT_FRAMES
    vectors = []
    monitor = active_progress()
    for index in range(segments):
        if monitor is not None:
            monitor.working(
                f"Omni-fMRI · {run.subject_id} · clip {index + 1}/{segments}"
            )
        window = volume[..., index * SEGMENT_FRAMES : (index + 1) * SEGMENT_FRAMES]
        input_array = np.ascontiguousarray(np.moveaxis(window, -1, 0))
        sample = torch.from_numpy(input_array).unsqueeze(0).to(device)
        vectors.append(_extract_cls(backbone, sample))
    pooled = np.mean(np.stack(vectors), axis=0, dtype=np.float64).astype(np.float32)
    if pooled.shape != (EMBEDDING_SIZE,) or not np.isfinite(pooled).all():
        raise ValueError(f"Invalid Omni-fMRI pooled embedding for {run.subject_id}")
    return pooled, segments


def prepare_omni_representation(brain: Any, rep: Any, output_dir: Path) -> Path:
    columns = tuple(f"brain_omni_fmri__omni_{i:03d}" for i in range(EMBEDDING_SIZE))
    store = RepresentationStore(output_dir, rep.key, columns)
    with store.locked():
        runs = discover_fmriprep_runs(brain.derivatives_dir, bids_root=brain.bids_root)
        code, revision = _repository()
        checkpoint, checkpoint_identity = _checkpoint(rep.checkpoint)
        settings = (
            EXTRACTOR_PROTOCOL,
            revision,
            checkpoint_identity,
            SPATIAL_SHAPE,
            SEGMENT_FRAMES,
            "mean_cls_all_full_segments",
        )
        records = [(run, signature((input_identity(run), settings))) for run in runs]
        previous, previous_keys = store.existing() if not brain.overwrite else ({}, {})
        vectors = {
            run.subject_id: np.asarray(previous[run.subject_id], dtype=np.float32)
            for run, key in records
            if previous_keys.get(run.subject_id) == key and run.subject_id in previous
        }
        for run, key in records:
            if run.subject_id not in vectors and not brain.overwrite:
                staged = store.staged(run.subject_id, key)
                if staged is not None:
                    vectors[run.subject_id] = np.asarray(staged, dtype=np.float32)
        pending = [(run, key) for run, key in records if run.subject_id not in vectors]
        monitor = active_progress()
        if monitor is not None:
            monitor.begin_units("Omni-fMRI · subjects", len(records))
            for _ in vectors:
                monitor.advance()
        info(
            f"Brain · omni_fmri · {len(vectors)}/{len(runs)} reused · {len(pending)} to encode"
        )
        if not pending and store.path.is_file() and not brain.overwrite:
            success(f"Brain · omni_fmri · reused · {store.path}")
            return store.path
        backbone, device = (
            _official_encoder(code, checkpoint) if pending else (None, None)
        )
        counts = {}
        for run, key in pending:
            if monitor is not None:
                monitor.working(f"Omni-fMRI · {run.subject_id}")
            vector, count = _encode_run(run, backbone, device)
            vectors[run.subject_id] = vector
            counts[run.subject_id] = count
            store.stage(run.subject_id, key, vector)
            if monitor is not None:
                monitor.advance()
        if monitor is not None:
            monitor.phase(f"Consolidating omni_fmri · {len(records)} subjects")
        destination = store.commit(
            [(run.subject_id, key) for run, key in records], vectors
        )
        folder = output_dir / rep.key
        folder.mkdir(parents=True, exist_ok=True)
        with atomic_path(folder / "omni_fmri_provenance.json") as temporary:
            temporary.write_text(
                json.dumps(
                    {
                        "model": "Omni-fMRI",
                        "method": rep.key,
                        "source": REPOSITORY,
                        "source_revision": revision,
                        "checkpoint": checkpoint_identity,
                        "preprocessing": "official symmetric spatial crop/pad and global nonzero z-score",
                        "voxel_shape": SPATIAL_SHAPE,
                        "segment_frames": SEGMENT_FRAMES,
                        "pooling": "mean of final-layer CLS tokens across nonoverlapping full 40-frame clips",
                        "segment_counts_new": counts,
                        "uses_atlas": False,
                        "inputs": "fMRIPrep MNI152NLin2009cAsym res-2 task-rest preproc BOLD",
                        "limitations": "Unverified against the official released checkpoint on this platform",
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        success(f"Brain · omni_fmri · ready · {destination}")
        return destination
