from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .atlas import load_schaefer_atlas
from .cache import (
    BrainCache,
    cache_root,
    file_identity,
    input_identity,
    settings_identity,
    signature,
)
from .fmriprep import discover_fmriprep_runs
from .progress import active_progress
from .storage import RepresentationStore
from .timeseries import extract_schaefer_timeseries
from ..console import info, success


CHECKPOINT_REPOSITORY = "medarc/CortexMAE"
CHECKPOINT_FILE = "input_space_v3/schaefer400_lr3e-4_1/pretrain/checkpoint-last.pth"


def cortexmae_atlas() -> Path:
    from templateflow.api import get

    result = get(
        "MNI152NLin2009cAsym",
        atlas="Schaefer2018",
        desc="400Parcels17Networks",
        resolution=2,
        suffix="dseg",
        extension=".nii.gz",
    )
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise ValueError(
                f"Expected one Schaefer-400/17 atlas, received {len(result)}"
            )
        result = result[0]
    if result is None:
        raise FileNotFoundError(
            "Schaefer-400/17-networks atlas unavailable from TemplateFlow"
        )
    path = Path(result).expanduser().resolve()
    if (
        not path.is_file()
        or "desc-400Parcels17Networks" not in path.name
        or "MNI152NLin2009cAsym" not in path.name
    ):
        raise ValueError(f"Unexpected CortexMAE atlas source: {path}")
    load_schaefer_atlas(path, 400)
    return path


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(override: str | Path | None) -> tuple[Path, str]:
    if override is not None and str(override) != "auto":
        path = Path(override).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, f"local:{_checksum(path)}"
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "CortexMAE requires the optional mllabiome[cortexmae] dependencies"
        ) from exc
    try:
        revision = HfApi().model_info(CHECKPOINT_REPOSITORY).sha
        if not revision or len(revision) < 40:
            raise RuntimeError(
                "Could not pin the CortexMAE checkpoint to a Hugging Face revision"
            )
        path = Path(
            hf_hub_download(
                repo_id=CHECKPOINT_REPOSITORY,
                filename=CHECKPOINT_FILE,
                revision=revision,
            )
        )
        identity = f"{CHECKPOINT_REPOSITORY}@{revision}"
    except (OSError, ConnectionError, TimeoutError):
        path = Path(
            hf_hub_download(
                repo_id=CHECKPOINT_REPOSITORY,
                filename=CHECKPOINT_FILE,
                local_files_only=True,
            )
        )
        identity = f"{CHECKPOINT_REPOSITORY}@offline-cache"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, f"{identity}:{_checksum(path)}"


def _input_sample(series: np.ndarray, tr: float) -> dict[str, Any]:
    array = np.asarray(series, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 400 or array.shape[0] < 16:
        raise ValueError(
            f"CortexMAE-P requires at least 16 frames of 400 parcel signals; got {array.shape}"
        )
    if not np.isfinite(array).all() or not np.isfinite(tr) or tr <= 0:
        raise ValueError(
            "CortexMAE-P requires finite time series and positive repetition time"
        )
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    valid = std > 1e-6
    scaled = np.where(valid, (array - mean) / np.maximum(std, 1e-6), 0.0)
    if not np.isfinite(scaled).all() or not valid.all():
        raise ValueError("CortexMAE-P input has constant or invalid parcels")
    return {
        "bold": np.asarray(scaled, dtype=np.float16),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "tr": float(tr),
    }


def _pooled_embedding(result: Any) -> np.ndarray:
    tokens = getattr(result, "cls_embeds", None)
    if tokens is None:
        raise ValueError(
            "CortexMAE-P checkpoint has no CLS embeddings; refusing unvalidated pooling"
        )
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().float().cpu().numpy()
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (1, 768) or len(values) == 0:
        raise ValueError(f"Unexpected CortexMAE-P CLS token shape: {values.shape}")
    pooled = values[:, 0, :].mean(axis=0, dtype=np.float64).astype(np.float32)
    if pooled.shape != (768,) or not np.isfinite(pooled).all():
        raise ValueError("Invalid CortexMAE-P pooled embedding")
    return pooled


def _load_model(checkpoint: Path):
    try:
        from cortex_mae import CortexMAE
    except ImportError as exc:
        raise ImportError(
            "Install mllabiome[cortexmae] for the CortexMAE-P encoder"
        ) from exc
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CortexMAE.from_checkpoint(str(checkpoint), device=device)
    if model.args.input_space != "schaefer400":
        raise ValueError(
            f"CortexMAE checkpoint input space is {model.args.input_space}, not schaefer400"
        )
    if model.num_frames < 1:
        raise ValueError("CortexMAE checkpoint has invalid temporal window length")
    return model


def _series(brain: Any, cache: BrainCache, run: Any, atlas: Path) -> np.ndarray:
    key = cache.key(
        "timeseries",
        input_identity(run),
        file_identity(atlas),
        400,
        settings_identity(brain.preprocessing),
        "cortexmae_p_17networks",
    )
    with cache.locked("timeseries", key):
        data = None if brain.overwrite else cache.read("timeseries", key)
        if data is not None and "series" in data:
            return np.asarray(data["series"], dtype=np.float32)
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            series, _ = extract_schaefer_timeseries(
                run,
                n_parcels=400,
                preprocessing=brain.preprocessing,
                atlas_img=load_schaefer_atlas(atlas, 400),
            )
        if brain.retain_timeseries:
            cache.write("timeseries", key, series=np.asarray(series, dtype=np.float32))
        return np.asarray(series, dtype=np.float32)


def prepare_cortexmae_representation(brain: Any, rep: Any, output_dir: Path) -> Path:
    store = RepresentationStore(
        output_dir,
        rep.key,
        tuple(f"brain_{rep.key}__cmp_{i:03d}" for i in range(768)),
    )
    with store.locked():
        info("Brain · CortexMAE-P · resolving Schaefer-400/17-networks atlas")
        atlas = cortexmae_atlas()
        checkpoint, checkpoint_identity = _checkpoint(rep.checkpoint)
        runs = discover_fmriprep_runs(brain.derivatives_dir, bids_root=brain.bids_root)
        cache = BrainCache(
            cache_root(brain.cache_dir, output_dir), migrate_legacy=False
        )
        settings = (
            "cortexmae_p.v1",
            settings_identity(brain.preprocessing),
            file_identity(atlas),
            checkpoint_identity,
            "cls_window_mean",
            "official_input_transform",
        )
        records = [(run, signature((input_identity(run), settings))) for run in runs]
        previous, keys = store.existing() if not brain.overwrite else ({}, {})
        vectors = {
            run.subject_id: previous[run.subject_id]
            for run, key in records
            if keys.get(run.subject_id) == key and run.subject_id in previous
        }
        for run, key in records:
            if run.subject_id not in vectors and not brain.overwrite:
                staged = store.staged(run.subject_id, key)
                if staged is not None:
                    vectors[run.subject_id] = staged
        monitor = active_progress()
        if monitor is not None:
            monitor.begin_units("CortexMAE-P · subjects", len(records))
            for _ in vectors:
                monitor.advance()
        pending = [(run, key) for run, key in records if run.subject_id not in vectors]
        info(
            f"Brain · cortexmae_p · {len(vectors)}/{len(runs)} reused · {len(pending)} to encode"
        )
        if not pending and store.path.is_file() and not brain.overwrite:
            success(f"Brain · cortexmae_p · reused · {store.path}")
            return store.path
        model = _load_model(checkpoint) if pending else None
        for run, key in pending:
            if monitor is not None:
                monitor.working(f"CortexMAE-P · {run.subject_id}")
            sample = _input_sample(
                _series(brain, cache, run, atlas), run.repetition_time
            )
            vector = _pooled_embedding(
                model.run_embedding(sample, batch_size=4, amp=False)
            )
            vectors[run.subject_id] = vector
            store.stage(run.subject_id, key, vector)
            if monitor is not None:
                monitor.advance()
        if monitor is not None:
            monitor.phase(f"Consolidating cortexmae_p · {len(runs)} subjects")
        destination = store.commit(
            [(run.subject_id, key) for run, key in records], vectors
        )
        folder = output_dir / rep.key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cortexmae_p_provenance.json").write_text(
            json.dumps(
                {
                    "model": "CortexMAE-P",
                    "checkpoint": checkpoint_identity,
                    "atlas": str(atlas),
                    "atlas_identity": file_identity(atlas),
                    "atlas_networks": 17,
                    "spatial_space": "MNI152NLin2009cAsym",
                    "pooling": "mean CLS over complete temporal clips",
                    "input_transform": "upstream CortexMAE transform",
                    "preprocessing": settings_identity(brain.preprocessing),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        success(f"Brain · cortexmae_p · ready · {destination}")
        return destination
