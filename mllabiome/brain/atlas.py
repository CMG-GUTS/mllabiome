from __future__ import annotations

from pathlib import Path

import numpy as np


SPACE = "MNI152NLin2009cAsym"


def schaefer_atlas(parcels: int) -> Path:
    if parcels not in {100, 200, 400}:
        raise ValueError("Supported Schaefer parcellations are 100, 200, and 400")
    try:
        from templateflow.api import get
    except ImportError as exc:
        raise ImportError("Automatic atlas loading requires mllabiome[brain]") from exc
    result = get(
        SPACE,
        atlas="Schaefer2018",
        desc=f"{parcels}Parcels7Networks",
        resolution=2,
        suffix="dseg",
        extension=".nii.gz",
    )
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise RuntimeError(
                f"Expected one TemplateFlow Schaefer-{parcels} atlas; found {len(result)}"
            )
        result = result[0]
    if result is None:
        raise FileNotFoundError(
            f"TemplateFlow does not provide Schaefer-{parcels} in {SPACE}"
        )
    path = Path(result).expanduser().resolve()
    if not path.is_file() or SPACE not in path.name:
        raise FileNotFoundError(
            f"TemplateFlow atlas is missing or has the wrong spatial reference: {path}"
        )
    if f"desc-{parcels}Parcels7Networks" not in path.name:
        raise ValueError(f"Unexpected TemplateFlow Schaefer atlas identity: {path}")
    return path


def load_schaefer_atlas(path: str | Path, parcels: int):
    try:
        from nilearn import image
    except ImportError as exc:
        raise ImportError(
            "Schaefer atlas validation requires mllabiome[brain]"
        ) from exc
    volume = image.load_img(str(path))
    if len(volume.shape) != 3:
        raise ValueError("Schaefer atlas must be a 3D label image")
    values = np.unique(np.asanyarray(volume.dataobj))
    if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
        raise ValueError("Schaefer atlas must contain integer-valued labels")
    labels = set(values.astype(int))
    if labels not in (set(range(parcels + 1)), set(range(1, parcels + 1))):
        raise ValueError(f"Atlas must contain every Schaefer label 1..{parcels}")
    return volume
