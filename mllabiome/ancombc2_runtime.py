from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .siamcat_runtime import (
    BIOCONDUCTOR_VERSION,
    R_VERSION,
    _cache_root,
    _choose_rscript,
    _installation_lock,
    _r_env,
    _r_version,
    _run,
)

ANCOMBC_VERSION = "2.14.0"


@dataclass(frozen=True)
class ANCOMBC2Runtime:
    rscript: Path
    library: Path
    r_version: str
    bioconductor_version: str
    ancombc_version: str


def _installed_version(rscript: Path, library: Path) -> str | None:
    expr = (
        ".libPaths(c(Sys.getenv('R_LIBS_USER'), .libPaths())); "
        "if (requireNamespace('ANCOMBC', quietly=TRUE)) cat(as.character(packageVersion('ANCOMBC')))"
    )
    try:
        result = _run(
            [str(rscript), "--vanilla", "-e", expr],
            env=_r_env(library),
            timeout=60,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _install(rscript: Path, library: Path) -> None:
    library.mkdir(parents=True, exist_ok=True)
    expr = f"""
lib <- Sys.getenv('R_LIBS_USER')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
options(repos=c(CRAN='https://cloud.r-project.org'))
if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager', lib=lib, quiet=TRUE)
BiocManager::install(version='{BIOCONDUCTOR_VERSION}', ask=FALSE, update=FALSE)
if (!requireNamespace('ANCOMBC', quietly=TRUE) || as.character(packageVersion('ANCOMBC')) != '{ANCOMBC_VERSION}') BiocManager::install('ANCOMBC', ask=FALSE, update=FALSE, lib=lib)
if (!requireNamespace('ANCOMBC', quietly=TRUE)) stop('ANCOMBC installation failed')
cat(as.character(packageVersion('ANCOMBC')))
"""
    result = _run(
        [str(rscript), "--vanilla", "-e", expr],
        env=_r_env(library),
        timeout=5400,
    )
    version = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if version != ANCOMBC_VERSION:
        raise RuntimeError(
            f"Expected ANCOMBC {ANCOMBC_VERSION} from Bioconductor {BIOCONDUCTOR_VERSION}, but found {version or 'unknown'}."
        )


def ensure_ancombc2_runtime(
    *,
    runtime: str = "auto",
    rscript: str | os.PathLike[str] | None = None,
) -> ANCOMBC2Runtime:
    chosen = _choose_rscript(runtime, rscript)
    r_version = _r_version(chosen)
    library = _cache_root() / "r-library" / f"R-{R_VERSION}_Bioc-{BIOCONDUCTOR_VERSION}"
    marker = library / ".mllabiome-ancombc2.json"
    installed = _installed_version(chosen, library)
    if installed != ANCOMBC_VERSION:
        lock = _cache_root() / "locks" / f"ANCOMBC-{ANCOMBC_VERSION}.lock"
        with _installation_lock(lock):
            installed = _installed_version(chosen, library)
            if installed != ANCOMBC_VERSION:
                _install(chosen, library)
                installed = _installed_version(chosen, library)
    if installed != ANCOMBC_VERSION:
        raise RuntimeError(
            f"ANCOMBC {ANCOMBC_VERSION} could not be prepared; found {installed!r}."
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "rscript": str(chosen),
                "r_version": r_version,
                "bioconductor_version": BIOCONDUCTOR_VERSION,
                "ancombc_version": installed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ANCOMBC2Runtime(
        rscript=chosen,
        library=library,
        r_version=r_version,
        bioconductor_version=BIOCONDUCTOR_VERSION,
        ancombc_version=installed,
    )


def run_ancombc2(
    counts: np.ndarray,
    sample_ids: list[str],
    feature_names: list[str],
    metadata: pd.DataFrame,
    *,
    aggregate_counts: np.ndarray | None = None,
    aggregate_feature_names: list[str] | None = None,
    fix_formula: str,
    rand_formula: str | None,
    group: str | None,
    classification_levels: list[str] | None,
    p_adjust_method: str,
    pseudo_sens: bool,
    prevalence_cutoff: float,
    alpha: float,
    structural_zeros: bool,
    neg_lb: bool,
    global_test: bool,
    pairwise: bool,
    workers: int,
    random_state: int,
    runtime: str = "auto",
    rscript: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    matrix = np.asarray(counts, dtype=float)
    aggregate = (
        matrix
        if aggregate_counts is None
        else np.asarray(aggregate_counts, dtype=float)
    )
    aggregate_names = (
        feature_names
        if aggregate_feature_names is None
        else list(aggregate_feature_names)
    )
    if matrix.ndim != 2 or matrix.shape != (len(sample_ids), len(feature_names)):
        raise ValueError(
            "ANCOM-BC2 count matrix dimensions do not match sample and feature identifiers."
        )
    if aggregate.ndim != 2 or aggregate.shape != (
        len(sample_ids),
        len(aggregate_names),
    ):
        raise ValueError(
            "ANCOM-BC2 aggregate count matrix dimensions do not match sample and feature identifiers."
        )
    if (
        not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.isfinite(aggregate).all()
        or np.any(aggregate < 0)
    ):
        raise ValueError("ANCOM-BC2 requires finite nonnegative counts.")
    if not np.allclose(matrix, np.rint(matrix), atol=1e-8) or not np.allclose(
        aggregate, np.rint(aggregate), atol=1e-8
    ):
        raise ValueError(
            "ANCOM-BC2 requires integer count data; relative-abundance profiles are not accepted."
        )
    if list(metadata.index.astype(str)) != [str(value) for value in sample_ids]:
        raise ValueError("ANCOM-BC2 metadata row order must match sample identifiers.")
    prepared = ensure_ancombc2_runtime(runtime=runtime, rscript=rscript)
    script = Path(__file__).resolve().parent / "_r" / "ancombc2.R"
    with tempfile.TemporaryDirectory(prefix="mllabiome-ancombc2-") as tmp:
        work = Path(tmp)
        count_path = work / "counts.tsv"
        aggregate_path = work / "aggregate.tsv"
        metadata_path = work / "metadata.tsv"
        levels_path = work / "levels.txt"
        output_dir = work / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            matrix.T,
            index=[str(value) for value in feature_names],
            columns=[str(value) for value in sample_ids],
        ).to_csv(count_path, sep="\t", index=True, index_label="taxon")
        pd.DataFrame(
            aggregate.T,
            index=[str(value) for value in aggregate_names],
            columns=[str(value) for value in sample_ids],
        ).to_csv(aggregate_path, sep="\t", index=True, index_label="taxon")
        metadata.to_csv(metadata_path, sep="\t", index=True, index_label="sample_id")
        levels_path.write_text(
            "\n".join(str(value) for value in (classification_levels or [])),
            encoding="utf-8",
        )
        args = [
            str(prepared.rscript),
            "--vanilla",
            str(script),
            str(count_path),
            str(aggregate_path),
            str(metadata_path),
            str(levels_path),
            str(output_dir),
            str(fix_formula),
            str(rand_formula) if rand_formula else "__NONE__",
            str(group) if group else "__NONE__",
            str(p_adjust_method),
            "1" if pseudo_sens else "0",
            f"{float(prevalence_cutoff):.17g}",
            f"{float(alpha):.17g}",
            "1" if structural_zeros else "0",
            "1" if neg_lb else "0",
            "1" if global_test else "0",
            "1" if pairwise else "0",
            str(max(1, int(workers))),
            str(int(random_state)),
        ]
        _run(args, env=_r_env(prepared.library), timeout=7200)
        result: dict[str, Any] = {
            "runtime": prepared,
            "primary": pd.DataFrame(),
            "global": pd.DataFrame(),
            "pairwise": pd.DataFrame(),
            "structural_zeros": pd.DataFrame(),
            "sensitivity": pd.DataFrame(),
        }
        for key, name in (
            ("primary", "primary.tsv"),
            ("global", "global.tsv"),
            ("pairwise", "pairwise.tsv"),
            ("structural_zeros", "structural_zeros.tsv"),
            ("sensitivity", "sensitivity.tsv"),
        ):
            path = output_dir / name
            if path.exists() and path.stat().st_size:
                result[key] = pd.read_csv(path, sep="\t")
        return result


__all__ = [
    "ANCOMBC_VERSION",
    "ANCOMBC2Runtime",
    "ensure_ancombc2_runtime",
    "run_ancombc2",
]
