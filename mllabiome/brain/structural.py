from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SUBJECT_RE = re.compile(r"(?:^|_)sub-([a-zA-Z0-9]+)(?:_|$)")
_TISSUES = ("GM", "WM", "CSF")


@dataclass(frozen=True)
class TissueMaps:
    subject_id: str
    maps: tuple[Path, Path, Path]


def discover_tissue_maps(derivatives_dir: str | Path) -> list[TissueMaps]:

    root = Path(derivatives_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"fMRIPrep derivatives not found: {root}")
    discovered: dict[str, TissueMaps] = {}
    for gm in sorted(root.rglob("*_label-GM_probseg.nii.gz")):
        if gm.parent.name != "anat":
            continue
        prefix = gm.name.removesuffix("_label-GM_probseg.nii.gz")
        if "_space-" in prefix or "_res-" in prefix:
            continue
        match = _SUBJECT_RE.search(prefix)
        if match is None:
            raise ValueError(f"Cannot identify subject from native tissue map: {gm}")
        sid = f"sub-{match.group(1)}"
        if sid not in gm.parts:
            raise ValueError(f"Subject path does not match filename: {gm}")
        maps = tuple(gm.parent / f"{prefix}_label-{t}_probseg.nii.gz" for t in _TISSUES)
        missing = [str(p) for p in maps if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete native T1w tissue maps for {sid}: {missing}"
            )
        if sid in discovered:
            raise ValueError(
                f"Multiple native T1w tissue segmentations for {sid}; "
                "session/run selection must be explicitly specified."
            )
        discovered[sid] = TissueMaps(sid, maps)
    if not discovered:
        raise FileNotFoundError(f"No native-space GM/WM/CSF probseg trios under {root}")
    return list(discovered.values())


def tissue_volumes(tissue: TissueMaps) -> dict[str, float]:

    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "T1w volumetry requires nibabel (optional brain dependency)."
        ) from exc

    result: dict[str, float] = {}
    geometry = None
    for name, path in zip(_TISSUES, tissue.maps):
        img = nib.load(str(path))
        if len(img.shape) != 3:
            raise ValueError(f"{path}: expected a 3D probability map.")

        units = img.header.get_xyzt_units()[0]
        if units != "mm":
            raise ValueError(
                f"{path}: spatial units must explicitly be mm, got {units!r}."
            )
        affine = np.asarray(img.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"{path}: invalid affine.")
        voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
        if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
            raise ValueError(f"{path}: invalid native voxel volume.")
        if geometry is None:
            geometry = (img.shape, affine)
        elif img.shape != geometry[0] or not np.allclose(
            affine, geometry[1], atol=1e-5, rtol=0
        ):
            raise ValueError(
                f"{path}: tissue maps are not on the same native-space grid."
            )
        data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float64)
        if not np.isfinite(data).all() or data.min() < -1e-4 or data.max() > 1.0 + 1e-4:
            raise ValueError(
                f"{path}: non-finite values or values outside probability range [0,1]."
            )
        volume = float(np.clip(data, 0, 1).sum(dtype=np.float64) * voxel_mm3)
        if volume <= 0:
            raise ValueError(f"{path}: empty tissue map.")
        result[f"tissue_{name.lower()}_mm3"] = volume
    return result
