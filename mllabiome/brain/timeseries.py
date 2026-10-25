from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .atlas import schaefer_atlas
from .fmriprep import FMRIPrepRun


_MOTION = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
CONFOUNDS_26 = (
    *_MOTION,
    *(f"{x}_derivative1" for x in _MOTION),
    *(f"{x}_power2" for x in _MOTION),
    *(f"{x}_derivative1_power2" for x in _MOTION),
    "white_matter",
    "csf",
)


@dataclass(frozen=True)
class BOLDPreprocessing:
    high_pass: float | None = 0.01
    low_pass: float | None = 0.10
    standardize: bool = True
    yeo_networks: int = 7
    atlas_resolution_mm: int = 2
    confounds: tuple[str, ...] = CONFOUNDS_26

    maximum_mean_fd: float | None = None
    maximum_high_motion_fraction: float | None = None
    high_motion_fd: float = 0.5

    def __post_init__(self) -> None:
        for name, cutoff in (
            ("high_pass", self.high_pass),
            ("low_pass", self.low_pass),
        ):
            if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0):
                raise ValueError(f"{name} must be finite and positive when specified.")
        if (
            self.high_pass is not None
            and self.low_pass is not None
            and self.high_pass >= self.low_pass
        ):
            raise ValueError("Require high_pass < low_pass (Hz).")
        if not isinstance(self.standardize, bool):
            raise ValueError("standardize must be boolean.")
        if self.yeo_networks != 7 or self.atlas_resolution_mm != 2:
            raise ValueError(
                "BrainHarmonix-F support currently requires the specified "
                "Schaefer-400/7-networks/2-mm ordering; other orderings "
                "have not been validated."
            )
        if len(set(self.confounds)) != len(self.confounds):
            raise ValueError("Duplicate confound regressors.")
        for name, val in (
            ("maximum_mean_fd", self.maximum_mean_fd),
            ("maximum_high_motion_fraction", self.maximum_high_motion_fraction),
        ):
            if val is not None and (not np.isfinite(val) or val < 0):
                raise ValueError(f"{name} must be finite and >= 0.")
        if (
            self.maximum_high_motion_fraction is not None
            and self.maximum_high_motion_fraction > 1
        ):
            raise ValueError("maximum_high_motion_fraction must be <= 1.")
        if not np.isfinite(self.high_motion_fd) or self.high_motion_fd <= 0:
            raise ValueError("high_motion_fd must be finite and > 0.")


def extract_schaefer400(
    run: FMRIPrepRun,
    *,
    preprocessing: BOLDPreprocessing = BOLDPreprocessing(),
    atlas_img=None,
    n_parcels: int = 400,
) -> tuple[np.ndarray, dict[str, float | int | str]]:

    try:
        from nilearn.maskers import NiftiLabelsMasker
        from nilearn import image
    except ImportError as exc:
        raise ImportError(
            "Functional fMRI extraction requires nilearn and nibabel. "
            "Install the optional mllabiome brain dependencies."
        ) from exc
    if n_parcels not in {100, 165, 200, 400}:
        raise ValueError(
            "Supported parcels are 100, 165 (verified Labus atlas), 200 and 400"
        )
    if n_parcels == 165 and atlas_img is None:
        raise ValueError("A verified 165-region anatomical atlas is required")
    if preprocessing.low_pass is not None and run.repetition_time >= 1.0 / (
        2.0 * preprocessing.low_pass
    ):
        raise ValueError(
            f"{run.subject_id}: TR {run.repetition_time} s is incompatible "
            f"with low-pass {preprocessing.low_pass} Hz (Nyquist constraint)."
        )
    metadata = pd.read_csv(run.confounds, sep="\t")
    missing = sorted(set(preprocessing.confounds) - set(metadata.columns))
    if missing:
        raise ValueError(f"{run.subject_id}: missing required confounds: {missing}")
    reg = metadata.loc[:, list(preprocessing.confounds)].apply(
        pd.to_numeric, errors="coerce"
    )

    for col in reg:
        if col.endswith("_derivative1") or col.endswith("_derivative1_power2"):
            if pd.isna(reg[col].iloc[0]):
                reg.loc[reg.index[0], col] = 0.0
    if not np.isfinite(reg.to_numpy(dtype=float)).all():
        raise ValueError(
            f"{run.subject_id}: remaining invalid/missing confound values."
        )
    motion = {}
    if "framewise_displacement" in metadata:
        fd = pd.to_numeric(metadata["framewise_displacement"], errors="coerce")
        if pd.isna(fd.iloc[0]):
            fd.iloc[0] = 0.0
        if not np.isfinite(fd.to_numpy(dtype=float)).all() or (fd < 0).any():
            raise ValueError(
                f"{run.subject_id}: invalid framewise displacement values."
            )
        motion = {
            "mean_fd": float(fd.mean()),
            "fraction_fd_above_threshold": float(
                (fd > preprocessing.high_motion_fd).mean()
            ),
        }
    if (
        preprocessing.maximum_mean_fd is not None
        or preprocessing.maximum_high_motion_fraction is not None
    ):
        if not motion:
            raise ValueError(f"{run.subject_id}: FD required for requested motion QC.")
        if (
            preprocessing.maximum_mean_fd is not None
            and motion["mean_fd"] > preprocessing.maximum_mean_fd
        ):
            raise ValueError(
                f"{run.subject_id}: mean FD exceeds pre-specified QC threshold."
            )
        if (
            preprocessing.maximum_high_motion_fraction is not None
            and motion["fraction_fd_above_threshold"]
            > preprocessing.maximum_high_motion_fraction
        ):
            raise ValueError(
                f"{run.subject_id}: fraction of high-motion frames exceeds QC threshold."
            )

    if atlas_img is None:
        atlas_img = schaefer_atlas(n_parcels)

    bold_img = image.load_img(str(run.bold))
    if len(bold_img.shape) != 4:
        raise ValueError(f"{run.subject_id}: expected a 4D BOLD NIfTI image.")
    if bold_img.shape[-1] != len(reg):
        raise ValueError(
            f"{run.subject_id}: BOLD frames ({bold_img.shape[-1]}) != "
            f"confound rows ({len(reg)})."
        )
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize="zscore_sample" if preprocessing.standardize else None,
        detrend=True,
        low_pass=preprocessing.low_pass,
        high_pass=preprocessing.high_pass,
        t_r=run.repetition_time,
        standardize_confounds=True,
        resampling_target="data",
    )
    arr = np.asarray(
        masker.fit_transform(bold_img, confounds=reg.to_numpy()), dtype=np.float32
    )
    if arr.ndim != 2 or arr.shape != (bold_img.shape[-1], n_parcels):
        raise ValueError(
            f"{run.subject_id}: wrong Schaefer-{n_parcels} shape {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{run.subject_id}: non-finite Schaefer signals.")
    if (np.std(arr, axis=0) < 1e-5).any():
        raise ValueError(
            f"{run.subject_id}: one or more Schaefer parcels are constant; "
            "inspect atlas coverage, signal masking and QC."
        )
    quality: dict[str, float | int | str] = {
        "sample_id": run.subject_id,
        "timepoints": int(arr.shape[0]),
        "tr_seconds": float(run.repetition_time),
        "atlas": "user_verified_labus_165"
        if n_parcels == 165
        else f"Schaefer2018_{n_parcels}Parcels_7Networks_MNI152NLin2009cAsym_res2",
        **motion,
    }
    return arr, quality


def extract_schaefer_timeseries(
    run: FMRIPrepRun,
    *,
    n_parcels: int,
    preprocessing: BOLDPreprocessing = BOLDPreprocessing(),
    atlas_img=None,
):
    return extract_schaefer400(
        run, preprocessing=preprocessing, atlas_img=atlas_img, n_parcels=n_parcels
    )
