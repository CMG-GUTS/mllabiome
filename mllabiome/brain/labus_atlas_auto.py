from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .labus_2019 import LABUS_ROIS, load_labus_2019_mapping


CORTICAL_ROIS = {
    "PosCG": "G_postcentral",
    "PosCS": "S_postcentral",
    "CS": "S_central",
    "SbCG_S": "G_and_S_subcentral",
    "PRCG": "G_precentral",
    "InfPrCS": "S_precentral-inf-part",
    "SupPrCs": "S_precentral-sup-part",
    "SupFG": "G_front_sup",
    "SupFS": "S_front_sup",
    "SupCirInS": "S_circular_insula_sup",
    "LoInG_CInS": "G_Ins_lg_and_S_cent_ins",
    "InfCirInS": "S_circular_insula_inf",
    "PosLS": "Lat_Fis-post",
}
SUBCORTICAL_ROIS = {
    "Tha": "thalamus",
    "Pu": "putamen",
    "CaN": "caudate",
    "Nacc": "accumbens",
    "Pal": "pallidum",
}
EXTRA_SUBCORTICAL = ("hippocampus", "amygdala")
VERSION = "labus_2019_public_atlas_approximation_v2"


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _lateral_name(value: str) -> tuple[str, str] | None:
    text = (value.decode("utf-8") if isinstance(value, bytes) else str(value)).strip()
    found = re.match(r"^(left|right|l|r)[\s_\-]+(.+)$", text, re.IGNORECASE)
    if found is None:
        return None
    return ("L" if found.group(1).casefold() in {"left", "l"} else "R", found.group(2))


def _index_labels(
    names: list[str], values: list[int], region_count: int, source: str
) -> dict[int, str]:
    if len(names) != len(values):
        raise ValueError(
            f"{source}: atlas labels and numeric values have different lengths"
        )
    result = {int(value): str(name) for value, name in zip(values, names)}
    if len(result) != len(values) or (0 not in result):
        raise ValueError(
            f"{source}: missing background or duplicate atlas label values"
        )
    if len(result) < region_count + 1:
        raise ValueError(f"{source}: expected at least {region_count} labeled regions")
    return result


def _fetch_public_atlases():
    try:
        from nilearn import datasets
        from nilearn import image
        from templateflow.api import get
    except ImportError as exc:
        raise ImportError(
            "The automatic Labus atlas requires mllabiome[brain]"
        ) from exc
    reference = get(
        "MNI152NLin2009cAsym",
        desc=None,
        resolution=2,
        suffix="T1w",
        extension=".nii.gz",
    )
    if isinstance(reference, (tuple, list)):
        if len(reference) != 1:
            raise RuntimeError(
                "Expected exactly one MNI152NLin2009cAsym 2-mm T1w reference"
            )
        reference = reference[0]
    if reference is None:
        raise FileNotFoundError(
            "The MNI152NLin2009cAsym 2-mm reference could not be fetched from TemplateFlow"
        )
    destrieux = datasets.fetch_atlas_destrieux_2009(lateralized=True)
    harvard = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    aseg = get(
        "MNI152NLin2009cAsym",
        resolution=2,
        suffix="dseg",
        atlas=None,
        desc=None,
        extension=".nii.gz",
    )
    if not isinstance(aseg, (tuple, list)):
        aseg = [] if aseg is None else [aseg]
    expected = "tpl-MNI152NLin2009cAsym_res-02_seg-aseg_dseg.nii.gz"
    if len(aseg) != 1 or Path(aseg[0]).name != expected:
        names = [Path(path).name for path in aseg]
        raise FileNotFoundError(
            f"Expected the exact TemplateFlow FreeSurfer segmentation {expected}; received {names}"
        )
    aseg = aseg[0]
    return image, reference, destrieux, harvard, aseg


def _atlas_entries(atlas, *, kind: str) -> list[tuple[int, str]]:
    if hasattr(atlas, "lut") and atlas.lut is not None and "index" in atlas.lut:
        values = [int(value) for value in atlas.lut["index"]]
    else:
        values = list(range(len(atlas.labels)))
    return list(_index_labels(list(atlas.labels), values, 14, kind).items())


def _collect_regions(
    atlas, kind: str, image, reference
) -> tuple[np.ndarray, list[tuple[int, str, str]]]:
    source = atlas if kind == "aseg" else atlas.maps
    resampled = image.resample_to_img(
        source,
        reference,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    voxels = np.asarray(resampled.dataobj)
    if voxels.ndim != 3 or not np.isfinite(voxels).all():
        raise ValueError(f"{kind} atlas must have finite, three-dimensional labels")
    if not np.allclose(voxels, np.rint(voxels)):
        raise ValueError(f"{kind} atlas labels must be integers")
    result = np.rint(voxels).astype(np.int32)
    if kind == "aseg":
        regions = ((8, "L", "cerebellum"), (47, "R", "cerebellum"))
        if any(not np.any(result == value) for value, _, _ in regions):
            raise ValueError(
                "TemplateFlow FreeSurfer aseg lacks left or right cerebellar cortex labels"
            )
        return result, list(regions)
    rows = []
    for value, name in _atlas_entries(atlas, kind=kind):
        if value == 0:
            continue
        lateral = _lateral_name(name)
        if (
            kind == "destrieux"
            and lateral is not None
            and _normalize(lateral[1]) != "medialwall"
        ):
            if np.any(result == value):
                rows.append((value, lateral[0], lateral[1]))
        elif kind == "harvard" and lateral is not None:
            if _normalize(lateral[1]) in {
                "thalamus",
                "caudate",
                "putamen",
                "pallidum",
                "hippocampus",
                "amygdala",
                "accumbens",
            }:
                rows.append((value, lateral[0], lateral[1]))
        elif kind == "harvard" and _normalize(name) in {
            "brainstem",
            "brainstemmidbrain",
        }:
            rows.append((value, "", "brainstem"))
    return result, rows


def _build_from_atlases(image, reference, destrieux, harvard, aseg):
    cortex, cortical_regions = _collect_regions(
        destrieux, "destrieux", image, reference
    )
    subcortex, subcortical_regions = _collect_regions(
        harvard, "harvard", image, reference
    )
    cerebellum, cerebellar_regions = _collect_regions(aseg, "aseg", image, reference)
    if (
        len(cortical_regions) != 148
        or len({(side, _normalize(name)) for _, side, name in cortical_regions}) != 148
    ):
        raise ValueError(
            f"Expected 148 unique lateralized Destrieux regions, found {len(cortical_regions)}"
        )
    subs = [(code, side, name) for code, side, name in subcortical_regions if side]
    stem = [(code, side, name) for code, side, name in subcortical_regions if not side]
    if len(subs) != 14 or len(stem) != 1:
        raise ValueError(
            "Harvard-Oxford atlas must provide 14 bilateral subcortical regions and one brainstem"
        )
    if {(_normalize(name), side) for _, side, name in subs} != {
        (name, side)
        for name in (
            "thalamus",
            "caudate",
            "putamen",
            "pallidum",
            "hippocampus",
            "amygdala",
            "accumbens",
        )
        for side in ("L", "R")
    }:
        raise ValueError(
            "Harvard-Oxford atlas has incomplete or ambiguous lateralized subcortical labels"
        )
    if {side for _, side, _ in cerebellar_regions} != {"L", "R"}:
        raise ValueError(
            "TemplateFlow FreeSurfer aseg must provide bilateral cerebellar cortex"
        )
    volume = np.zeros(cortex.shape, dtype=np.int16)
    mapping: dict[tuple[str, str], int] = {}
    entries: list[tuple[int, str, str, str, str]] = []
    ordered = [
        ("destrieux", cortex, code, side, name)
        for code, side, name in sorted(
            cortical_regions, key=lambda row: (row[1], row[2])
        )
    ] + [
        ("harvard", subcortex, code, side, name)
        for code, side, name in sorted(
            subcortical_regions, key=lambda row: (row[1], row[2])
        )
    ]
    for side in ("L", "R"):
        codes = {
            int(code)
            for code, selected_side, _ in cerebellar_regions
            if selected_side == side
        }
        if not codes:
            raise ValueError(f"No {side} cerebellar voxels")
        ordered.append(("aseg", cerebellum, codes, side, "cerebellum"))
    for index, (source, data, code, side, name) in enumerate(ordered, start=1):
        selected = np.isin(data, list(code)) if isinstance(code, set) else data == code
        if source != "destrieux":
            selected &= volume == 0
        if not selected.any():
            raise ValueError(
                f"Empty region after atlas resampling/priority combination: {side} {name}"
            )
        volume[selected] = index
        normalized = _normalize(name)
        roi = None
        for candidate, original in CORTICAL_ROIS.items():
            if source == "destrieux" and normalized == _normalize(original):
                roi = candidate
                break
        if roi is None:
            for candidate, original in SUBCORTICAL_ROIS.items():
                if source == "harvard" and normalized == _normalize(original):
                    roi = candidate
                    break
        if roi is not None:
            pair = (side, roi)
            if pair in mapping:
                raise ValueError(f"Ambiguous ROI in public atlas: {pair}")
            mapping[pair] = index
        entries.append(
            (index, side if roi else "", roi or "", source, f"{side} {name}".strip())
        )
    if len(ordered) != 165 or set(np.unique(volume)) != set(range(166)):
        raise ValueError(
            "Combined public atlas must contain background and exactly 165 nonempty regions"
        )
    if set(mapping) != {(side, roi) for side in ("L", "R") for roi in LABUS_ROIS}:
        raise ValueError(
            f"Public atlases do not unambiguously cover the study's 36 sensorimotor ROIs: {sorted(set((s, r) for s in ('L', 'R') for r in LABUS_ROIS) - set(mapping))}"
        )
    return volume, entries


def resolve_labus_public_atlas(root: str | Path | None = None) -> tuple[Path, Path]:
    folder = (
        Path(root).expanduser()
        if root is not None
        else Path.home() / ".cache" / "mllabiome" / "atlases" / VERSION
    )
    folder.mkdir(parents=True, exist_ok=True)
    atlas_file = folder / "labus_2019_public_approx_165.nii.gz"
    labels_file = folder / "labus_2019_public_approx_165.tsv"
    manifest_file = folder / "labus_2019_public_approx_manifest.json"
    with FileLock(str(folder / "labus_2019_public_approx.lock")):
        if atlas_file.is_file() and labels_file.is_file() and manifest_file.is_file():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if manifest.get("version") == VERSION:
                load_labus_2019_mapping(labels_file)
                actual = hashlib.sha256(atlas_file.read_bytes()).hexdigest()
                if actual != manifest.get("sha256"):
                    raise ValueError("Automatic atlas cache integrity failure")
                return atlas_file, labels_file
        image, reference, destrieux, harvard, aseg = _fetch_public_atlases()
        volume, entries = _build_from_atlases(
            image, reference, destrieux, harvard, aseg
        )
        import nibabel as nib

        ref = image.load_img(reference)
        dest = nib.Nifti1Image(volume, ref.affine, ref.header.copy())
        dest.set_data_dtype(np.int16)
        temporary = folder / "labus_2019_public_approx_165.partial.nii.gz"
        nib.save(dest, str(temporary))
        pd.DataFrame(
            entries, columns=("value", "side", "roi", "source", "anatomical_name")
        ).to_csv(labels_file, sep="\t", index=False)
        load_labus_2019_mapping(labels_file)
        temporary.replace(atlas_file)
        manifest_file.write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "sha256": hashlib.sha256(atlas_file.read_bytes()).hexdigest(),
                    "n_regions": 165,
                    "roi_count": 36,
                    "space": "MNI152NLin2009cAsym_res-2_grid",
                    "sources": (
                        "nilearn:destrieux_2009_lateralized",
                        "nilearn:harvard_oxford_sub_maxprob_thr25_2mm",
                        "templateflow:MNI152NLin2009cAsym_res02_seg-aseg_dseg",
                        "templateflow:MNI152NLin2009cAsym_res02_T1w",
                    ),
                    "exact_study_atlas": False,
                    "warning": "Study-inspired approximation, not a verified registered atlas; grid resampling does not correct nonlinear source-template differences.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return atlas_file, labels_file
