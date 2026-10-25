from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .fmriprep import discover_fmriprep_runs
from .structural import discover_tissue_maps, tissue_volumes
from .timeseries import BOLDPreprocessing, extract_schaefer400


FUNCTIONAL = frozenset({"pearson_fc", "partial_fc", "alff", "falff"})
ALL_REPRESENTATIONS = FUNCTIONAL | {"tissue_volumes"}
PREFIXES = {
    "pearson_fc": "fc_",
    "partial_fc": "pfc_",
    "alff": "alff_",
    "falff": "falff_",
    "tissue_volumes": "tissue_",
}


@dataclass(frozen=True)
class HandcraftedSettings:
    low_frequency: float = 0.01
    high_frequency: float = 0.08
    fisher_z: bool = True

    def __post_init__(self):
        if not (
            np.isfinite(self.low_frequency)
            and np.isfinite(self.high_frequency)
            and 0 < self.low_frequency < self.high_frequency
        ):
            raise ValueError(
                "Require 0 < low_frequency < high_frequency, both finite (Hz)."
            )
        if not isinstance(self.fisher_z, bool):
            raise ValueError("fisher_z must be boolean.")


@dataclass(frozen=True)
class HandcraftedResult:
    feature_tables: dict[str, Path]
    manifest: Path
    provenance: Path
    sample_ids: tuple[str, ...]

    def as_modality(
        self, representation: str, *, name: str = "brain", primary: bool = False
    ):

        if representation not in self.feature_tables:
            raise KeyError(f"Representation {representation!r} was not exported.")
        from ..modalities import Modality

        return Modality(
            name=name,
            path=self.feature_tables[representation],
            format="table",
            sample_id_col="sample_id",
            feature_prefixes=PREFIXES[representation],
            primary=primary,
        )


def _check_ts(array: np.ndarray, *, tr: float) -> np.ndarray:
    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError(f"Expected ROI array (T>=3, N>=2), got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite ROI values.")
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError(f"Invalid repetition time: {tr!r}.")
    if np.any(np.std(x, axis=0) < 1e-8):
        raise ValueError("One or more time series has zero/near-zero variance.")
    return x


def _edge_vector(matrix: np.ndarray, *, fisher_z: bool) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Connectivity matrix must be square.")
    edges = matrix[np.triu_indices(matrix.shape[0], 1)]
    if not np.isfinite(edges).all() or np.any(np.abs(edges) > 1 + 1e-7):
        raise ValueError("Invalid/non-finite connectivity coefficients.")
    if fisher_z:
        edges = np.arctanh(np.clip(edges, -1 + 1e-7, 1 - 1e-7))
    return edges


def pearson_fc(time_series: np.ndarray, *, fisher_z: bool = True) -> np.ndarray:

    x = _check_ts(time_series, tr=1.0)
    return _edge_vector(np.corrcoef(x, rowvar=False), fisher_z=fisher_z)


def partial_fc(time_series: np.ndarray, *, fisher_z: bool = True) -> np.ndarray:

    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as exc:
        raise ImportError("Partial FC requires scikit-learn.") from exc
    x = _check_ts(time_series, tr=1.0)
    covariance = LedoitWolf().fit(x).covariance_
    precision = np.linalg.inv(covariance)
    denom = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    if np.any(denom <= 0) or not np.isfinite(denom).all():
        raise ValueError("Degenerate regularized precision matrix.")
    partial = -precision / denom
    np.fill_diagonal(partial, 1.0)
    return _edge_vector(partial, fisher_z=fisher_z)


def spectral_amplitudes(
    broadband_time_series: np.ndarray,
    tr: float,
    *,
    settings: HandcraftedSettings = HandcraftedSettings(),
) -> tuple[np.ndarray, np.ndarray]:

    x = _check_ts(broadband_time_series, tr=tr)
    n = x.shape[0]
    nyquist = 0.5 / tr
    if settings.high_frequency >= nyquist:
        raise ValueError("ALFF band upper bound must be strictly below Nyquist.")
    if n * tr < 1.0 / settings.low_frequency:
        raise ValueError(
            "Scan duration is too short to resolve the lowest ALFF frequency."
        )
    frequencies = np.fft.rfftfreq(n, d=tr)
    low = (frequencies >= settings.low_frequency) & (
        frequencies <= settings.high_frequency
    )
    broad = frequencies > 0
    if low.sum() < 2:
        raise ValueError("Too few nonzero frequency bins for ALFF band.")

    amplitudes = np.abs(np.fft.rfft(x - x.mean(axis=0, keepdims=True), axis=0))
    band_amp = amplitudes[low]
    alff = band_amp.mean(axis=0) * (2.0 / n)
    denom = amplitudes[broad].sum(axis=0)
    if np.any(denom <= 1e-12) or not np.isfinite(denom).all():
        raise ValueError("Zero/non-finite broadband amplitude; fALFF undefined.")
    falff = band_amp.sum(axis=0) / denom
    if not np.isfinite(alff).all() or not np.isfinite(falff).all():
        raise ValueError("Non-finite spectral features.")
    return alff, falff


def _names(representation: str, n: int = 400) -> list[str]:
    if representation in ("pearson_fc", "partial_fc"):
        prefix = PREFIXES[representation]
        i, j = np.triu_indices(n, 1)
        return [f"{prefix}{a:03d}_{b:03d}" for a, b in zip(i, j)]
    if representation in ("alff", "falff"):
        return [f"{PREFIXES[representation]}{i:03d}" for i in range(n)]
    if representation == "tissue_volumes":
        return [f"tissue_{t}_mm3" for t in ("gm", "wm", "csf")]
    raise ValueError(f"Unsupported representation: {representation}.")


def _load_atlas(path: str | Path | None):
    if path is None:
        return None
    try:
        from nilearn import image
    except ImportError as exc:
        raise ImportError("Atlas reading requires nilearn.") from exc
    img = image.load_img(str(path))
    if len(img.shape) != 3:
        raise ValueError("Custom Schaefer-400 atlas must be 3D.")
    values = np.unique(np.asanyarray(img.dataobj))
    if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
        raise ValueError("Custom atlas contains non-integer or non-finite labels.")
    labels = set(values.astype(int))
    if not set(range(1, 401)).issubset(labels) or labels.difference(range(401)):
        raise ValueError(
            "Custom Schaefer-400 atlas must contain each label 1..400 (plus optional background 0)."
        )
    return img


def export_handcrafted_fmriprep(
    derivatives_dir: str | Path,
    output_dir: str | Path,
    *,
    representations: tuple[str, ...] = ("pearson_fc",),
    bids_root: str | Path | None = None,
    atlas_path: str | Path | None = None,
    allow_template_mismatch: bool = False,
    task: str = "rest",
    preprocessing: BOLDPreprocessing = BOLDPreprocessing(),
    settings: HandcraftedSettings = HandcraftedSettings(),
    overwrite: bool = False,
) -> HandcraftedResult:

    requested = tuple(representations)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Provide at least one unique brain representation.")
    invalid = set(requested) - ALL_REPRESENTATIONS
    if invalid:
        raise ValueError(f"Unsupported brain representations: {sorted(invalid)}")
    func = bool(FUNCTIONAL.intersection(requested))
    struct = "tissue_volumes" in requested
    if func and atlas_path is None and not allow_template_mismatch:
        raise ValueError(
            "fMRIPrep's MNI152NLin2009cAsym and Nilearn's Schaefer FSL MNI152 "
            "atlas are different template spaces. Supply an atlas aligned to "
            "MNI152NLin2009cAsym or set allow_template_mismatch=True explicitly."
        )
    runs = (
        discover_fmriprep_runs(derivatives_dir, bids_root=bids_root, task=task)
        if func
        else []
    )
    tissues = discover_tissue_maps(derivatives_dir) if struct else []
    run_by_id = {r.subject_id: r for r in runs}
    tissue_by_id = {r.subject_id: r for r in tissues}
    if func and struct and run_by_id.keys() != tissue_by_id.keys():
        raise ValueError(
            "Functional and structural subject sets differ. No silent complete-case "
            f"selection: functional-only={sorted(run_by_id.keys() - tissue_by_id.keys())}, "
            f"structural-only={sorted(tissue_by_id.keys() - run_by_id.keys())}."
        )
    sample_ids = tuple(sorted(run_by_id if func else tissue_by_id))
    if not sample_ids:
        raise ValueError("No participants found.")
    atlas = _load_atlas(atlas_path) if func else None
    output = Path(output_dir).expanduser().resolve()
    tables = {r: output / f"brain_{r}_features.tsv" for r in requested}
    manifest = output / "brain_handcrafted_manifest.tsv"
    provenance = output / "brain_handcrafted_provenance.json"
    existing = [p for p in (*tables.values(), manifest, provenance) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Handcrafted brain output files already exist: {existing}"
        )
    output.mkdir(parents=True, exist_ok=True)
    feature_names = {rep: _names(rep) for rep in requested}

    with TemporaryDirectory(prefix=".handcrafted_stage_", dir=output) as stage_dir:
        stage = Path(stage_dir)
        staged_tables = {r: stage / p.name for r, p in tables.items()}
        staged_manifest = stage / manifest.name
        staged_provenance = stage / provenance.name
        manifest_rows: list[dict[str, Any]] = []

        broadband_settings = replace(
            preprocessing, low_pass=None, high_pass=None, standardize=False
        )
        need_conn = bool({"pearson_fc", "partial_fc"} & set(requested))
        need_spectral = bool({"alff", "falff"} & set(requested))

        with ExitStack() as stack:
            writers = {}
            for rep, path in tables.items():
                f = stack.enter_context(
                    staged_tables[rep].open("w", encoding="utf-8", newline="")
                )
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["sample_id", *feature_names[rep]])
                writers[rep] = writer
            for sid in sample_ids:
                values: dict[str, np.ndarray] = {}
                quality: dict[str, Any] = {"sample_id": sid}
                if func:
                    run = run_by_id[sid]
                    quality.update(
                        {
                            "bold_path": str(run.bold),
                            "confounds_path": str(run.confounds),
                            "repetition_time": run.repetition_time,
                            "tr_metadata_path": str(run.bold_json),
                        }
                    )
                    if need_conn:
                        signals, qc = extract_schaefer400(
                            run, preprocessing=preprocessing, atlas_img=atlas
                        )
                        quality.update(qc)
                        if "pearson_fc" in requested:
                            values["pearson_fc"] = pearson_fc(
                                signals, fisher_z=settings.fisher_z
                            )
                        if "partial_fc" in requested:
                            values["partial_fc"] = partial_fc(
                                signals, fisher_z=settings.fisher_z
                            )
                    if need_spectral:
                        broadband, qc = extract_schaefer400(
                            run, preprocessing=broadband_settings, atlas_img=atlas
                        )
                        quality.update(qc)
                        alff, falff = spectral_amplitudes(
                            broadband, run.repetition_time, settings=settings
                        )
                        if "alff" in requested:
                            values["alff"] = alff
                        if "falff" in requested:
                            values["falff"] = falff
                if struct:
                    maps = tissue_by_id[sid]
                    tissue = tissue_volumes(maps)
                    values["tissue_volumes"] = np.asarray(
                        [tissue[k] for k in feature_names["tissue_volumes"]]
                    )
                    quality.update(
                        {
                            f"tissue_{t}_map": str(p)
                            for t, p in zip(("GM", "WM", "CSF"), maps.maps)
                        }
                    )
                for rep in requested:
                    x = np.asarray(values[rep], dtype=np.float64)
                    if (
                        x.shape != (len(feature_names[rep]),)
                        or not np.isfinite(x).all()
                    ):
                        raise RuntimeError(
                            f"{sid}: {rep} produced invalid feature array shape/values: {x.shape}."
                        )
                    writers[rep].writerow([sid, *(f"{float(v):.10g}" for v in x)])
                manifest_rows.append(quality)

        all_fields = sorted(
            {field for row in manifest_rows for field in row} - {"sample_id"}
        )
        with staged_manifest.open("w", encoding="utf-8", newline="") as mf:
            writer = csv.DictWriter(
                mf, fieldnames=["sample_id", *all_fields], delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        staged_provenance.write_text(
            json.dumps(
                {
                    "schema": "mllabiome.brain.handcrafted.v1",
                    "source": "fMRIPrep derivatives",
                    "derivatives_dir": str(
                        Path(derivatives_dir).expanduser().resolve()
                    ),
                    "bids_root": str(Path(bids_root).expanduser().resolve())
                    if bids_root
                    else None,
                    "input_space_functional": "MNI152NLin2009cAsym:res-2"
                    if func
                    else None,
                    "input_space_structural": "native T1w (no space-* entity)"
                    if struct
                    else None,
                    "atlas": str(Path(atlas_path).expanduser().resolve())
                    if atlas_path
                    else (
                        "Nilearn Schaefer-400/7-networks FSL-MNI152" if func else None
                    ),
                    "approximate_cross_template_resampling": bool(
                        func and atlas_path is None
                    ),
                    "atlas_description": "Schaefer2018 400/7-network, 2-mm atlas (cortical only)"
                    if func
                    else None,
                    "representations": list(requested),
                    "feature_counts": {
                        rep: len(names) for rep, names in feature_names.items()
                    },
                    "settings": asdict(settings),
                    "filtered_preprocessing": asdict(preprocessing)
                    if need_conn
                    else None,
                    "broadband_preprocessing": asdict(broadband_settings)
                    if need_spectral
                    else None,
                    "structural_definition": "sum(native T1w tissue probabilities) * abs(det(affine_3x3)) mm3"
                    if struct
                    else None,
                    "cross_subject_normalization": "none; fit subsequent standardization within supervised training folds",
                    "participants": list(sample_ids),
                    "caution": "Regional FC/ALFF indices are Schaefer parcels; tissue summary volumes are not regional morphometry or cortical thickness. Review motion/QC, atlas template correspondence, test-retest reliability, and harmonization before inferential interpretation.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        for rep, dest in tables.items():
            staged_tables[rep].replace(dest)
        staged_manifest.replace(manifest)
        staged_provenance.replace(provenance)
    return HandcraftedResult(tables, manifest, provenance, sample_ids)


def main(argv: list[str] | None = None) -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description="fMRIPrep -> handcrafted functional/structural features -> mllabiome tables"
    )
    parser.add_argument("derivatives_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["pearson_fc"],
        choices=sorted(ALL_REPRESENTATIONS),
    )
    parser.add_argument("--bids-root", type=Path)
    parser.add_argument("--atlas-path", type=Path)
    parser.add_argument("--allow-template-mismatch", action="store_true")
    parser.add_argument("--task", default="rest")
    parser.add_argument("--low-frequency", type=float, default=0.01)
    parser.add_argument("--high-frequency", type=float, default=0.08)
    parser.add_argument("--no-fisher-z", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = export_handcrafted_fmriprep(
        args.derivatives_dir,
        args.output_dir,
        representations=tuple(args.representations),
        bids_root=args.bids_root,
        atlas_path=args.atlas_path,
        allow_template_mismatch=args.allow_template_mismatch,
        task=args.task,
        settings=HandcraftedSettings(
            low_frequency=args.low_frequency,
            high_frequency=args.high_frequency,
            fisher_z=not args.no_fisher_z,
        ),
        overwrite=args.overwrite,
    )
    for name, path in result.feature_tables.items():
        print(f"{name}: {len(result.sample_ids)} participants -> {path}")
    print(f"QC manifest: {result.manifest}")
    print(f"Provenance: {result.provenance}")


if __name__ == "__main__":
    main()
