from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


_SPACE = "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
_SUBJECT = re.compile(r"(?:^|_)sub-([A-Za-z0-9]+)(?=_|$)")


@dataclass(frozen=True)
class FMRIPrepRun:
    subject_id: str
    bold: Path
    confounds: Path
    repetition_time: float
    prefix: str
    bold_json: Path


def _json_tr(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"Invalid BOLD metadata JSON: {path}") from exc
    value = values.get("RepetitionTime")
    if value is None:
        return None
    try:
        tr = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RepetitionTime in {path}: {value!r}") from exc
    if not math.isfinite(tr) or tr <= 0:
        raise ValueError(f"RepetitionTime must be finite and positive in {path}.")
    return tr


def _match_raw_sidecar(root: Path, bold: Path, prefix: str) -> Path:

    dirs: list[str] = []
    for p in bold.parent.parts:
        if p.startswith("sub-"):
            dirs = [p]
        elif p.startswith("ses-") and dirs:
            dirs.append(p)
    if not dirs:
        raise ValueError(f"Cannot locate subject directory for {bold}.")
    path = root.joinpath(*dirs, "func", prefix + "_bold.json")
    if not path.is_file():
        raise FileNotFoundError(
            f"No run-specific raw BIDS JSON {path}. Supply --bids-root, or an "
            "fMRIPrep derivative JSON with RepetitionTime. BIDS inheritance is "
            "not automatically resolved by this minimal loader."
        )
    return path


def discover_fmriprep_runs(
    derivatives_dir: str | Path,
    *,
    bids_root: str | Path | None = None,
    task: str = "rest",
) -> list[FMRIPrepRun]:

    root = Path(derivatives_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"fMRIPrep derivatives directory missing: {root}")
    original = Path(bids_root).expanduser().resolve() if bids_root else None
    if original is not None and not original.is_dir():
        raise FileNotFoundError(f"Raw BIDS root missing: {original}")
    if not task or not re.fullmatch(r"[A-Za-z0-9]+", task):
        raise ValueError("task must be a nonempty BIDS task identifier")

    files = sorted(root.rglob("*" + _SPACE))
    files = [p for p in files if f"_task-{task}_" in p.name]
    if not files:
        raise FileNotFoundError(
            f"No task-{task} MNI152NLin2009cAsym res-2 preproc BOLD "
            f"NIfTI files found under {root}."
        )
    runs: list[FMRIPrepRun] = []
    seen: dict[str, Path] = {}
    for bold in files:
        prefix = bold.name.removesuffix(_SPACE)
        match = _SUBJECT.search(prefix)
        if not match:
            raise ValueError(f"No BIDS subject entity in {bold.name}")
        subject = "sub-" + match.group(1)
        if subject in seen:
            raise ValueError(
                f"Multiple task-{task} BOLD runs for {subject}: {seen[subject]} and "
                f"{bold}. Aggregate runs using an explicitly designed and "
                "validated run-level strategy; silent overwrite is prohibited."
            )
        if not any(part == subject for part in bold.parts):
            raise ValueError(f"Subject path/name mismatch for {bold}.")
        confounds = bold.parent / (prefix + "_desc-confounds_timeseries.tsv")
        if not confounds.is_file():
            raise FileNotFoundError(f"Missing matching fMRIPrep confounds: {confounds}")
        deriv_json = bold.parent / (prefix + _SPACE.removesuffix(".nii.gz") + ".json")
        tr = _json_tr(deriv_json)
        source = deriv_json
        if tr is None:
            if original is None:
                raise ValueError(
                    f"RepetitionTime unavailable for {bold}. Pass bids_root to "
                    "read the matching raw BIDS run JSON."
                )
            source = _match_raw_sidecar(original, bold, prefix)
            tr = _json_tr(source)
            if tr is None:
                raise ValueError(f"RepetitionTime missing in {source}.")
        runs.append(FMRIPrepRun(subject, bold, confounds, tr, prefix, source))
        seen[subject] = bold
    return runs
