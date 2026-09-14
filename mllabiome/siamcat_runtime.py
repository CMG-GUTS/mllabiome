from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

R_VERSION = "4.6.1"
BIOCONDUCTOR_VERSION = "3.23"
SIAMCAT_VERSION = "2.16.0"


@dataclass(frozen=True)
class SIAMCATRuntime:
    rscript: Path
    library: Path
    r_version: str
    bioconductor_version: str
    siamcat_version: str


def _cache_root() -> Path:
    override = os.environ.get("MLLABIOME_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "mllabiome" / "Cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "mllabiome"
    return Path.home() / ".cache" / "mllabiome"


def _run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{details}") from exc


def _r_version(rscript: Path) -> str:
    result = _run([str(rscript), "--vanilla", "-e", "cat(as.character(getRversion()))"], timeout=30)
    return result.stdout.strip()


def _compatible_r(version: str) -> bool:
    return version == "4.6" or version.startswith("4.6.")


def _system_rscript() -> Path | None:
    explicit = os.environ.get("MLLABIOME_RSCRIPT")
    candidates = [explicit] if explicit else []
    found = shutil.which("Rscript")
    if found:
        candidates.append(found)
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser().resolve()
        if not path.exists():
            continue
        try:
            if _compatible_r(_r_version(path)):
                return path
        except Exception:
            continue
    return None


def _glibc_version() -> tuple[int, int] | None:
    name, version = platform.libc_ver()
    if name.lower() != "glibc" or not version:
        return None
    try:
        major, minor, *_ = (int(x) for x in version.split("."))
        return major, minor
    except Exception:
        return None


def _managed_archive() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        if machine not in {"amd64", "x86_64"}:
            raise RuntimeError(
                "Automatic SIAMCAT R installation currently supports Windows x86_64 only. "
                "Set MLLABIOME_RSCRIPT to a compatible R 4.6.x Rscript executable."
            )
        name = f"portable-r-{R_VERSION}-win-x64.zip"
        return (
            f"https://github.com/portable-r/portable-r-windows/releases/download/v{R_VERSION}/{name}",
            name,
        )

    if system == "darwin":
        if machine in {"arm64", "aarch64"}:
            arch = "arm64"
        elif machine in {"x86_64", "amd64"}:
            arch = "x86_64"
        else:
            raise RuntimeError(f"Unsupported macOS architecture for managed R: {machine}")
        name = f"portable-r-{R_VERSION}-macos-{arch}.tar.gz"
        return (
            f"https://github.com/portable-r/portable-r-macos/releases/download/v{R_VERSION}/{name}",
            name,
        )

    if system == "linux":
        if machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
            raise RuntimeError(f"Unsupported Linux architecture for managed R: {machine}")
        glibc = _glibc_version()
        if glibc is None or glibc < (2, 34):
            raise RuntimeError(
                "Automatic SIAMCAT R installation on Linux requires glibc >= 2.34 "
                "(for example Ubuntu 22.04+, Debian 12+, RHEL 9+). "
                "Set MLLABIOME_RSCRIPT to a compatible R 4.6.x installation on older systems."
            )
        suffix = "-arm64" if machine in {"aarch64", "arm64"} else ""
        name = f"R-{R_VERSION}-manylinux_2_34{suffix}.tar.gz"
        return (f"https://cdn.posit.co/r/manylinux_2_34/{name}", name)

    raise RuntimeError(
        f"Automatic SIAMCAT R installation is not available on {platform.system()}. "
        "Set MLLABIOME_RSCRIPT to a compatible R 4.6.x Rscript executable."
    )


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in R archive: {member.name}")
        tf.extractall(destination)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in R archive: {member.filename}")
        zf.extractall(destination)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mllabiome-siamcat-runtime"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not download the managed R runtime from {url}. "
            "Check network access or set MLLABIOME_RSCRIPT to an existing R 4.6.x installation."
        ) from exc


def _find_rscript(root: Path) -> Path:
    name = "Rscript.exe" if os.name == "nt" else "Rscript"
    candidates = sorted(root.rglob(name), key=lambda p: (len(p.parts), str(p)))
    for candidate in candidates:
        if candidate.parent.name == "bin" and candidate.is_file():
            if os.name != "nt":
                candidate.chmod(candidate.stat().st_mode | 0o111)
            return candidate
    raise RuntimeError(f"R runtime was extracted, but {name} could not be found under {root}.")


@contextlib.contextmanager
def _installation_lock(path: Path, timeout: float = 1800.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}\n".encode())
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Timed out waiting for SIAMCAT runtime installation lock: {path}")
            time.sleep(1.0)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _managed_rscript() -> Path:
    root = _cache_root() / "runtimes" / f"R-{R_VERSION}"
    marker = root / "runtime.json"
    if marker.exists():
        try:
            info = json.loads(marker.read_text(encoding="utf-8"))
            rscript = root / info["rscript"]
            if rscript.exists() and _compatible_r(_r_version(rscript)):
                return rscript
        except Exception:
            pass

    lock = _cache_root() / "locks" / f"R-{R_VERSION}.lock"
    with _installation_lock(lock):
        if marker.exists():
            try:
                info = json.loads(marker.read_text(encoding="utf-8"))
                rscript = root / info["rscript"]
                if rscript.exists() and _compatible_r(_r_version(rscript)):
                    return rscript
            except Exception:
                pass

        url, filename = _managed_archive()
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mllabiome-r-install-") as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / filename
            extracted = tmp_path / "extracted"
            extracted.mkdir()
            _download(url, archive)
            if filename.endswith(".zip"):
                _safe_extract_zip(archive, extracted)
            else:
                _safe_extract_tar(archive, extracted)
            candidate = _find_rscript(extracted)
            relative = candidate.relative_to(extracted)
            if root.exists():
                shutil.rmtree(root)
            shutil.copytree(extracted, root)
            rscript = root / relative
            version = _r_version(rscript)
            if not _compatible_r(version):
                shutil.rmtree(root, ignore_errors=True)
                raise RuntimeError(f"Downloaded R {version}, but mllabiome requires R 4.6.x for Bioconductor {BIOCONDUCTOR_VERSION}.")
            marker.write_text(
                json.dumps({"rscript": str(relative), "r_version": version, "source_url": url}, indent=2),
                encoding="utf-8",
            )
            return rscript


def _choose_rscript(mode: str = "auto", explicit: str | os.PathLike[str] | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Rscript does not exist: {path}")
        version = _r_version(path)
        if not _compatible_r(version):
            raise RuntimeError(f"SIAMCAT backend requires R 4.6.x; found R {version} at {path}.")
        return path

    mode = str(mode).strip().lower()
    if mode not in {"auto", "system", "managed"}:
        raise ValueError("runtime must be one of: 'auto', 'system', 'managed'.")
    if mode in {"auto", "system"}:
        system = _system_rscript()
        if system is not None:
            return system
        if mode == "system":
            raise RuntimeError(
                "No compatible R 4.6.x Rscript was found. Set MLLABIOME_RSCRIPT or use runtime='managed'."
            )
    return _managed_rscript()


def _r_env(library: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(library)
    env["MLLABIOME_R_LIBRARY"] = str(library)
    return env


def _installed_siamcat_version(rscript: Path, library: Path) -> str | None:
    expr = (
        ".libPaths(c(Sys.getenv('R_LIBS_USER'), .libPaths())); "
        "if (requireNamespace('SIAMCAT', quietly=TRUE)) cat(as.character(packageVersion('SIAMCAT')))"
    )
    try:
        result = _run([str(rscript), "--vanilla", "-e", expr], env=_r_env(library), timeout=60)
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _install_siamcat(rscript: Path, library: Path) -> None:
    library.mkdir(parents=True, exist_ok=True)
    expr = f"""
lib <- Sys.getenv('R_LIBS_USER')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
options(repos=c(CRAN='https://cloud.r-project.org'))
if (!requireNamespace('BiocManager', quietly=TRUE)) {{
    install.packages('BiocManager', lib=lib, quiet=TRUE)
}}
BiocManager::install(version='{BIOCONDUCTOR_VERSION}', ask=FALSE, update=FALSE)
if (!requireNamespace('SIAMCAT', quietly=TRUE) || as.character(packageVersion('SIAMCAT')) != '{SIAMCAT_VERSION}') {{
    BiocManager::install('SIAMCAT', ask=FALSE, update=FALSE, lib=lib)
}}
if (!requireNamespace('SIAMCAT', quietly=TRUE)) stop('SIAMCAT installation failed')
cat(as.character(packageVersion('SIAMCAT')))
"""
    result = _run([str(rscript), "--vanilla", "-e", expr], env=_r_env(library), timeout=3600)
    version = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if version != SIAMCAT_VERSION:
        raise RuntimeError(
            f"Expected SIAMCAT {SIAMCAT_VERSION} from Bioconductor {BIOCONDUCTOR_VERSION}, but found {version or 'unknown'}."
        )


def ensure_siamcat_runtime(
    *,
    runtime: str = "auto",
    rscript: str | os.PathLike[str] | None = None,
) -> SIAMCATRuntime:
    """Return a ready-to-use R/SIAMCAT runtime, provisioning it lazily if needed."""
    chosen = _choose_rscript(runtime, rscript)
    r_version = _r_version(chosen)
    library = _cache_root() / "r-library" / f"R-{R_VERSION}_Bioc-{BIOCONDUCTOR_VERSION}"
    marker = library / ".mllabiome-siamcat.json"

    installed = _installed_siamcat_version(chosen, library)
    if installed != SIAMCAT_VERSION:
        lock = _cache_root() / "locks" / f"SIAMCAT-{SIAMCAT_VERSION}.lock"
        with _installation_lock(lock):
            installed = _installed_siamcat_version(chosen, library)
            if installed != SIAMCAT_VERSION:
                _install_siamcat(chosen, library)
                installed = _installed_siamcat_version(chosen, library)

    if installed != SIAMCAT_VERSION:
        raise RuntimeError(f"SIAMCAT {SIAMCAT_VERSION} could not be prepared; found {installed!r}.")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "rscript": str(chosen),
                "r_version": r_version,
                "bioconductor_version": BIOCONDUCTOR_VERSION,
                "siamcat_version": installed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return SIAMCATRuntime(
        rscript=chosen,
        library=library,
        r_version=r_version,
        bioconductor_version=BIOCONDUCTOR_VERSION,
        siamcat_version=installed,
    )


__all__ = [
    "BIOCONDUCTOR_VERSION",
    "R_VERSION",
    "SIAMCAT_VERSION",
    "SIAMCATRuntime",
    "ensure_siamcat_runtime",
]
