from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

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
from .progress import active_progress
from .fmriprep import FMRIPrepRun, discover_fmriprep_runs
from .timeseries import BOLDPreprocessing, extract_schaefer400


@dataclass(frozen=True)
class BrainHarmonixFResult:
    feature_table: Path
    manifest: Path
    normalization_stats: Path | None
    sample_ids: tuple[str, ...]
    n_features: int = 768

    def as_modality(self, *, name: str = "brain", primary: bool = False):

        from ..modalities import Modality

        parquet = self.feature_table.suffix.lower() == ".parquet"
        return Modality(
            name=name,
            path=self.feature_table,
            format="parquet" if parquet else "table",
            primary=primary,
            sample_id_col="sample_id",
            feature_prefixes="brain_brainharmonix_f__bhf_" if parquet else "bhf_",
        )


def _get_torch_and_factory():
    try:
        import torch
        from brainmarks.models.brainharmonix_f import brain_harmonix_f
    except ImportError as exc:
        raise ImportError(
            'BrainHarmonix-F is optional. Install: uv pip install -e ".[brainharmonix]"'
        ) from exc
    return torch, brain_harmonix_f


def _sample(torch: Any, array: np.ndarray, tr: float) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 400 or arr.shape[0] < 2:
        raise ValueError(f"Expect (T, 400) Schaefer time series, got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite BrainHarmonix-F input.")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError(f"Invalid TR {tr!r}.")
    return {
        "bold": torch.from_numpy(np.ascontiguousarray(arr)),
        "mean": torch.zeros((1, 400), dtype=torch.float32),
        "std": torch.ones((1, 400), dtype=torch.float32),
        "tr": float(tr),
    }


class BrainHarmonixFEncoder:
    def __init__(
        self, *, device: str = "auto", normalization: str = "reference"
    ) -> None:
        if normalization not in {"reference", "per_run"}:
            raise ValueError("Unknown normalization mode")
        self.normalization = normalization
        torch, factory = _get_torch_and_factory()
        self.torch = torch
        self.transform, self.model = factory(coord_normalize=True)
        self.device = self._device(device)
        self.model = self.model.to(self.device).eval()
        self._fitted = False

    def _device(self, requested: str) -> str:
        torch = self.torch
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            return "mps" if mps is not None and mps.is_available() else "cpu"
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if (
            requested == "mps"
            and not getattr(torch.backends, "mps", None).is_available()
        ):
            raise RuntimeError("Apple MPS requested but unavailable")
        return requested

    def fit_reference(
        self, observations: Iterable[tuple[np.ndarray, float]]
    ) -> "BrainHarmonixFEncoder":
        samples = [_sample(self.torch, arr, tr) for arr, tr in observations]
        if not samples:
            raise ValueError(
                "At least one independent reference time series is required."
            )
        self.transform.fit(samples)
        self._sanitize_stats()
        self._fitted = True
        return self

    def _sanitize_stats(self) -> None:
        median, iqr = self.transform.global_stats_
        torch = self.torch
        if tuple(median.shape) != (400,) or tuple(iqr.shape) != (400,):
            raise RuntimeError("Unexpected Brainmarks normalization statistics shape.")
        if not torch.isfinite(median).all() or not torch.isfinite(iqr).all():
            raise RuntimeError("Non-finite Brainmarks normalization statistics.")
        self.transform.global_stats_ = (median, torch.clamp(iqr, min=1e-6))

    def save_normalization(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Normalization not fitted.")
        median, iqr = self.transform.global_stats_
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            median=median.cpu().numpy().astype(np.float32),
            iqr=iqr.cpu().numpy().astype(np.float32),
            schema_version=np.asarray(1),
            normalization=np.asarray("brainmarks_coord_normalize_true"),
        )
        return target

    def load_normalization(self, path: str | Path) -> "BrainHarmonixFEncoder":

        with np.load(path, allow_pickle=False) as stats:
            if (
                int(stats["schema_version"]) != 1
                or str(stats["normalization"]) != "brainmarks_coord_normalize_true"
            ):
                raise ValueError(
                    "Incompatible BrainHarmonix-F normalization statistics."
                )
            median, iqr = (
                np.asarray(stats["median"], dtype=np.float32),
                np.asarray(stats["iqr"], dtype=np.float32),
            )
        self.transform.global_stats_ = (
            self.torch.from_numpy(median),
            self.torch.from_numpy(iqr),
        )
        self._sanitize_stats()
        self._fitted = True
        return self

    def encode(
        self, array: np.ndarray, tr: float, *, freeze_normalization: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.normalization == "per_run" and not freeze_normalization:
            samples = _sample(self.torch, array, tr)["bold"]
            q1, median, q3 = self.torch.quantile(
                samples,
                self.torch.tensor([0.25, 0.5, 0.75], dtype=samples.dtype),
                dim=0,
            )
            self.transform.global_stats_ = median, q3 - q1
            self._sanitize_stats()
            self._fitted = True
        if not self._fitted:
            raise RuntimeError(
                "Call fit_reference or load_normalization before encode."
            )
        transformed = self.transform(_sample(self.torch, array, tr))
        batch = {
            "bold": transformed["bold"].unsqueeze(0).to(self.device),
            "attn_mask": transformed["attn_mask"].unsqueeze(0).to(self.device),
            "patch_size": self.torch.tensor(
                [transformed["patch_size"]], dtype=self.torch.long, device=self.device
            ),
        }
        with self.torch.inference_mode():
            _, _, tensor = self.model(batch)
        tokens = tensor.squeeze(0).detach().float().cpu().numpy()
        mask = transformed["attn_mask"].detach().cpu().numpy().astype(bool)
        if tokens.ndim != 2 or tokens.shape != (mask.size, 768):
            raise RuntimeError(
                f"Incompatible BrainHarmonix-F token/mask shape: {tokens.shape}, {mask.shape}."
            )
        if not mask.any() or not np.isfinite(tokens[mask]).all():
            raise RuntimeError("No valid finite BrainHarmonix-F tokens.")
        pooled = np.asarray(tokens[mask].mean(axis=0), dtype=np.float32)
        return pooled, tokens.astype(np.float32), mask


def _extract_one(
    run: FMRIPrepRun,
    settings: BOLDPreprocessing,
    atlas_path: Path,
    cache_dir: Path,
    overwrite: bool,
    migrate_legacy: bool = False,
) -> tuple[FMRIPrepRun, np.ndarray, dict[str, Any]]:
    from threadpoolctl import threadpool_limits

    cache = BrainCache(cache_dir, migrate_legacy=migrate_legacy)
    key = cache.key(
        "timeseries",
        input_identity(run),
        file_identity(atlas_path),
        400,
        settings_identity(settings),
    )
    with cache.locked("timeseries", key):
        cached = None if overwrite else cache.read("timeseries", key)
        if cached is None:
            with threadpool_limits(limits=1):
                atlas = load_schaefer_atlas(atlas_path, 400)
                array, qc = extract_schaefer400(
                    run, preprocessing=settings, atlas_img=atlas
                )
            cache.write(
                "timeseries",
                key,
                series=np.asarray(array, dtype=np.float32),
                qc=np.asarray(json.dumps(qc, default=str)),
            )
        else:
            array = cached["series"]
            qc = json.loads(str(cached["qc"].item())) if "qc" in cached else {}
    return run, np.asarray(array, dtype=np.float32), qc


def _extract_runs(
    runs: list[FMRIPrepRun],
    settings: BOLDPreprocessing,
    *,
    atlas_img=None,
    progress_callback: Callable[[str], None] | None = None,
    label: str = "scan",
    atlas_path: Path | None = None,
    cache_dir: Path | None = None,
    overwrite: bool = False,
    n_jobs: int | str = "auto",
    migrate_legacy: bool = False,
) -> list[tuple[FMRIPrepRun, np.ndarray, dict[str, Any]]]:
    tracker = active_progress()
    if tracker is not None and runs:
        tracker.begin_units(f"BrainHarmonix-F · extracting {label} runs", len(runs))
    if atlas_path is None or cache_dir is None:
        extracted = []
        for index, run in enumerate(runs, 1):
            if progress_callback is not None:
                progress_callback(
                    f"BrainHarmonix-F · extracting {label} {index}/{len(runs)} · {run.subject_id}"
                )
            extracted.append(
                (
                    run,
                    *extract_schaefer400(
                        run, preprocessing=settings, atlas_img=atlas_img
                    ),
                )
            )
            if tracker is not None:
                tracker.working(run.subject_id)
                tracker.advance()
        return extracted
    jobs = worker_count(n_jobs, len(runs))
    outputs = {}
    if jobs == 1:
        for run in runs:
            result = _extract_one(
                run, settings, atlas_path, cache_dir, overwrite, migrate_legacy
            )
            outputs[run.subject_id] = result
            if tracker is not None:
                tracker.working(run.subject_id)
                tracker.advance()
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _extract_one,
                    run,
                    settings,
                    atlas_path,
                    cache_dir,
                    overwrite,
                    migrate_legacy,
                ): run.subject_id
                for run in runs
            }
            for future in as_completed(futures):
                result = future.result()
                outputs[result[0].subject_id] = result
                if tracker is not None:
                    tracker.working(result[0].subject_id)
                    tracker.advance()
    return [outputs[run.subject_id] for run in runs]


def embed_fmriprep(
    derivatives_dir: str | Path,
    output_dir: str | Path,
    *,
    reference_derivatives_dir: str | Path | None = None,
    normalization_stats: str | Path | None = None,
    normalization: str = "auto",
    bids_root: str | Path | None = None,
    reference_bids_root: str | Path | None = None,
    atlas_path: str | Path | None = None,
    allow_template_mismatch: bool = False,
    roi_order_verified: bool = False,
    task: str = "rest",
    preprocessing: BOLDPreprocessing = BOLDPreprocessing(),
    device: str = "auto",
    save_tokens: bool = False,
    overwrite: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    cache_dir: str | Path | None = None,
    n_jobs: int | str = "auto",
    migrate_legacy_artifacts: bool = False,
    output_format: str = "tsv",
    reused_embeddings: Mapping[str, tuple[str, np.ndarray]] | None = None,
) -> BrainHarmonixFResult:

    tracker = active_progress()
    if normalization == "auto":
        normalization = (
            "reference"
            if reference_derivatives_dir is not None or normalization_stats is not None
            else "per_run"
        )
    if normalization not in {"reference", "per_run"}:
        raise ValueError("Normalization must be auto, per_run or reference")
    if normalization == "per_run" and (
        reference_derivatives_dir is not None or normalization_stats is not None
    ):
        raise ValueError("Reference inputs require normalization='reference'")
    if normalization == "reference" and (reference_derivatives_dir is None) == (
        normalization_stats is None
    ):
        raise ValueError(
            "Reference normalization requires exactly one reference source"
        )
    if atlas_path is None:
        if tracker is not None:
            tracker.phase("Resolving BrainHarmonix-F Schaefer-400 atlas")
        atlas_path = schaefer_atlas(400)
    elif not roi_order_verified:
        raise ValueError(
            "Custom BrainHarmonix atlas ordering must be independently verified"
        )
    atlas_path = Path(atlas_path).expanduser().resolve()
    if progress_callback is not None:
        progress_callback("BrainHarmonix-F · discovering preprocessed runs")
    if tracker is not None:
        tracker.phase("Discovering BrainHarmonix-F target runs")
    runs = discover_fmriprep_runs(derivatives_dir, bids_root=bids_root, task=task)
    if progress_callback is not None:
        progress_callback(f"BrainHarmonix-F · discovered {len(runs)} runs")
    reference_runs: list[FMRIPrepRun] = []
    if reference_derivatives_dir is not None:
        if tracker is not None:
            tracker.phase("Discovering independent normalization runs")
        reference_runs = discover_fmriprep_runs(
            reference_derivatives_dir,
            bids_root=reference_bids_root,
            task=task,
        )
        overlap = sorted(
            {r.subject_id for r in runs} & {r.subject_id for r in reference_runs}
        )
        if overlap:
            raise ValueError(
                f"Reference participants overlap evaluated participants: {overlap}."
            )
    output = Path(output_dir).expanduser().resolve()
    if output_format not in {"tsv", "parquet"}:
        raise ValueError("Output format must be tsv or parquet")
    table = (
        output / "brainharmonix_f_features.tsv"
        if output_format == "tsv"
        else output.parent / "brainharmonix_f.parquet"
    )
    manifest = output / "brainharmonix_f_manifest.tsv"
    stats_path = output / "brainharmonix_f_normalization.npz"
    provenance = output / "brainharmonix_f_provenance.json"
    cache = BrainCache(
        cache_root(cache_dir, output),
        migrate_legacy=cache_dir is None or migrate_legacy_artifacts,
    )
    atlas_identity = file_identity(atlas_path)
    reference_identity = (
        file_identity(normalization_stats)
        if normalization_stats is not None
        else tuple(input_identity(run) for run in reference_runs)
    )
    settings = (
        "brainharmonix_f",
        settings_identity(preprocessing),
        atlas_identity,
        normalization,
        reference_identity,
    )
    keyed = [
        (run, cache.key("embedding", input_identity(run), settings, foundation=True))
        for run in runs
    ]
    pooled: dict[str, np.ndarray] = {}
    quality_rows: dict[str, dict[str, Any]] = {}
    pending = []
    for run, key in keyed:
        injected = (reused_embeddings or {}).get(run.subject_id)
        if not overwrite and injected is not None and injected[0] == key:
            prior = np.asarray(injected[1], dtype=np.float32)
            if prior.shape == (768,) and np.isfinite(prior).all():
                pooled[run.subject_id] = prior
                quality_rows[run.subject_id] = {
                    "subject": run.subject_id,
                    "source": "canonical_parquet",
                }
                continue
        stored = None if overwrite else cache.read("embedding", key)
        files_ready = not save_tokens or (
            (output / "tokens" / f"{run.subject_id}_tokens.npy").exists()
            and (output / "masks" / f"{run.subject_id}_mask.npy").exists()
        )
        usable = (
            stored is not None
            and stored.get("pooled", np.empty(0)).shape == (768,)
            and np.isfinite(stored["pooled"]).all()
            and "quality" in stored
            and files_ready
        )
        if usable:
            try:
                quality = json.loads(str(stored["quality"].item()))
                usable = isinstance(quality, dict)
            except (ValueError, TypeError, KeyError):
                usable = False
        if usable:
            pooled[run.subject_id] = stored["pooled"]
            quality_rows[run.subject_id] = quality
        else:
            pending.append((run, key))
    if progress_callback is not None:
        progress_callback(
            f"BrainHarmonix-F · {len(pooled)}/{len(runs)} cached · {len(pending)} to encode"
        )
    if tracker is not None:
        tracker.phase(f"BrainHarmonix-F · {len(pooled)}/{len(runs)} embeddings cached")
    output.mkdir(parents=True, exist_ok=True)
    if save_tokens:
        (output / "tokens").mkdir(exist_ok=True)
        (output / "masks").mkdir(exist_ok=True)
    reference_key = cache.key("reference_normalization", settings, foundation=True)
    normalized = (
        None if overwrite else cache.read("reference_normalization", reference_key)
    )
    if normalized is not None:
        valid_stats = (
            all(
                name in normalized and normalized[name].shape == (400,)
                for name in ("median", "iqr")
            )
            and all(np.isfinite(normalized[name]).all() for name in ("median", "iqr"))
            and np.all(normalized["iqr"] > 0)
        )
        if not valid_stats:
            normalized = None
    if normalization == "reference" and normalized is not None:
        with atomic_path(stats_path) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    median=normalized["median"],
                    iqr=normalized["iqr"],
                    schema_version=np.asarray(1),
                    normalization=np.asarray("brainmarks_coord_normalize_true"),
                )
    if pending:
        target_runs = [run for run, _ in pending]
        extracted = _extract_runs(
            target_runs,
            preprocessing,
            atlas_path=atlas_path,
            cache_dir=cache.root.parent,
            overwrite=overwrite,
            n_jobs=n_jobs,
            migrate_legacy=cache.legacy_root is not None,
        )
        reference = (
            _extract_runs(
                reference_runs,
                preprocessing,
                atlas_path=atlas_path,
                cache_dir=cache.root.parent,
                overwrite=overwrite,
                n_jobs=n_jobs,
                label="reference",
                migrate_legacy=cache.legacy_root is not None,
            )
            if reference_runs and normalized is None
            else []
        )
        if progress_callback is not None:
            progress_callback(
                "BrainHarmonix-F · initializing pretrained encoder and checkpoint"
            )
        if tracker is not None:
            tracker.phase("Loading BrainHarmonix-F model and pretrained weights")
        encoder = BrainHarmonixFEncoder(device=device, normalization=normalization)
        if tracker is not None:
            tracker.phase("Preparing BrainHarmonix-F normalization")
        if normalization == "reference":
            if normalized is not None:
                encoder.load_normalization(stats_path)
            elif reference:
                encoder.fit_reference(
                    (arr, run.repetition_time) for run, arr, _ in reference
                )
            else:
                encoder.load_normalization(normalization_stats)
            encoder.save_normalization(stats_path)
            with np.load(stats_path, allow_pickle=False) as stored_stats:
                cache.write(
                    "reference_normalization",
                    reference_key,
                    median=stored_stats["median"],
                    iqr=stored_stats["iqr"],
                )
        if tracker is not None:
            tracker.begin_units(
                "BrainHarmonix-F · completed embeddings", len(extracted)
            )
        lookup = dict((run.subject_id, key) for run, key in pending)
        for run, arr, qc in extracted:
            if tracker is not None:
                tracker.working(run.subject_id)
            vector, tokens, mask = encoder.encode(arr, run.repetition_time)
            row = {
                **qc,
                "bold_file": str(run.bold),
                "confounds_file": str(run.confounds),
                "tr_source_file": str(run.bold_json),
                "n_tokens": int(mask.size),
                "n_valid_tokens": int(mask.sum()),
            }
            cache.write(
                "embedding",
                lookup[run.subject_id],
                pooled=vector,
                quality=np.asarray(json.dumps(row, default=str)),
            )
            pooled[run.subject_id] = vector
            quality_rows[run.subject_id] = row
            if save_tokens:
                with atomic_path(
                    output / "tokens" / f"{run.subject_id}_tokens.npy"
                ) as temporary:
                    with temporary.open("wb") as stream:
                        np.save(stream, tokens)
                with atomic_path(
                    output / "masks" / f"{run.subject_id}_mask.npy"
                ) as temporary:
                    with temporary.open("wb") as stream:
                        np.save(stream, mask)
            if tracker is not None:
                tracker.advance()
    if normalization == "reference" and not pending and not stats_path.exists():
        if normalization_stats is not None:
            with atomic_path(stats_path) as temporary:
                temporary.write_bytes(Path(normalization_stats).read_bytes())
        else:
            raise RuntimeError(
                "Reference statistics missing from cache; embeddings cannot be traced reproducibly"
            )
    if tracker is not None:
        tracker.phase("Saving BrainHarmonix-F tables and provenance")
    if output_format == "parquet":
        import pandas as pd

        frame = pd.DataFrame(
            [pooled[run.subject_id] for run in runs],
            columns=[f"brain_brainharmonix_f__bhf_{i:03d}" for i in range(768)],
            dtype=np.float32,
        )
        frame.insert(0, "sample_id", [run.subject_id for run in runs])
        with atomic_path(table) as temporary:
            frame.to_parquet(
                temporary, engine="pyarrow", compression="zstd", index=False
            )
    else:
        with atomic_path(table) as temporary:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["sample_id", *(f"bhf_{i:03d}" for i in range(768))])
                for run in runs:
                    writer.writerow(
                        [
                            run.subject_id,
                            *(f"{float(v):.9g}" for v in pooled[run.subject_id]),
                        ]
                    )
    manifest_rows = [quality_rows[run.subject_id] for run in runs]
    columns = list(dict.fromkeys(key for row in manifest_rows for key in row))
    with atomic_path(manifest) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(manifest_rows)
    try:
        brainmarks_version = version("brainmarks")
    except PackageNotFoundError:
        brainmarks_version = "unknown"
    with atomic_path(provenance) as temporary:
        temporary.write_text(
            json.dumps(
                {
                    "schema": "mllabiome.brain.brainharmonix_f.v2",
                    "derivatives_dir": str(Path(derivatives_dir).resolve()),
                    "reference_derivatives_dir": str(
                        Path(reference_derivatives_dir).resolve()
                    )
                    if reference_derivatives_dir
                    else None,
                    "external_normalization_file": str(
                        Path(normalization_stats).resolve()
                    )
                    if normalization_stats
                    else None,
                    "normalization_mode": normalization,
                    "reference_subject_ids": [run.subject_id for run in reference_runs],
                    "target_subject_ids": [run.subject_id for run in runs],
                    "brainmarks_version": brainmarks_version,
                    "input_space": "MNI152NLin2009cAsym:res-2",
                    "atlas_path": str(atlas_path),
                    "atlas_order": "Schaefer2018_400Parcels_7Networks_order",
                    "approximate_cross_template_resampling": False,
                    "atlas_roi_order_verified_by_user": bool(roi_order_verified),
                    "preprocessing": asdict(preprocessing),
                    "masking": "valid_token_mean",
                    "cache": str(cache.root),
                    "noncanonical_normalization": normalization == "per_run",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if progress_callback is not None:
        progress_callback(f"BrainHarmonix-F · exported {len(runs)} subjects to {table}")
    return BrainHarmonixFResult(
        table,
        manifest,
        stats_path if normalization == "reference" else None,
        tuple(r.subject_id for r in runs),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="fMRIPrep -> BrainHarmonix-F -> mllabiome feature modality"
    )
    p.add_argument("derivatives_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    norm = p.add_mutually_exclusive_group()
    norm.add_argument("--reference-derivatives-dir", type=Path)
    norm.add_argument("--normalization-stats", type=Path)
    p.add_argument("--bids-root", type=Path)
    p.add_argument("--reference-bids-root", type=Path)
    p.add_argument(
        "--normalization", choices=("auto", "per_run", "reference"), default="auto"
    )
    p.add_argument("--atlas-path", type=Path)
    p.add_argument("--allow-template-mismatch", action="store_true")
    p.add_argument(
        "--roi-order-verified",
        action="store_true",
        help="Assert validated BrainHarmonix pretrained positional-embedding ROI ordering",
    )
    p.add_argument("--task", default="rest")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--save-tokens", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    result = embed_fmriprep(
        args.derivatives_dir,
        args.output_dir,
        reference_derivatives_dir=args.reference_derivatives_dir,
        normalization_stats=args.normalization_stats,
        normalization=args.normalization,
        bids_root=args.bids_root,
        reference_bids_root=args.reference_bids_root,
        atlas_path=args.atlas_path,
        allow_template_mismatch=args.allow_template_mismatch,
        roi_order_verified=args.roi_order_verified,
        task=args.task,
        device=args.device,
        save_tokens=args.save_tokens,
        overwrite=args.overwrite,
    )
    print(f"Exported {len(result.sample_ids)} subjects to {result.feature_table}")


if __name__ == "__main__":
    main()
