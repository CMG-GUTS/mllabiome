from __future__ import annotations
import contextlib
import hashlib
import json
import os
import platform
import re
import signal
import socket
import threading
import shutil
import subprocess
import sys
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
R_STARTUP_ARGS = ("--no-init-file",)


def _progress_interval() -> float:
    try:
        return max(
            0.2, float(os.environ.get("MLLABIOME_SIAMCAT_PROGRESS_INTERVAL", "10"))
        )
    except (ValueError, OverflowError):
        return 10.0


def _announce(message: str) -> None:
    print(f"[SIAMCAT] {message}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _heartbeat(message: str):
    stopped = threading.Event()
    started = time.monotonic()
    interval = _progress_interval()

    def report() -> None:
        while not stopped.wait(interval):
            _announce(f"{message} ({int(time.monotonic() - started)}s elapsed)")

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def _run_with_progress(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    phase: str,
    timeout: float | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _announce(f"{phase} started; output: {log_path}")
    with log_path.open("wb") as writer:
        process = subprocess.Popen(
            command,
            stdout=writer,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        started = time.monotonic()
        last_report = started
        last_output = started
        last_line = "no output yet"
        try:
            with log_path.open("rb") as reader:
                while True:
                    chunk = reader.read(65536)
                    if chunk:
                        content = chunk.decode("utf-8", errors="replace")
                        print(content, end="", file=sys.stderr, flush=True)
                        visible_lines = content.strip().splitlines()
                        if visible_lines:
                            last_line = visible_lines[-1][-160:]
                        last_output = time.monotonic()
                    code = process.poll()
                    if code is not None:
                        remainder = reader.read()
                        if remainder:
                            print(
                                remainder.decode("utf-8", errors="replace"),
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )
                        break
                    now = time.monotonic()
                    if timeout is not None and now - started >= timeout:
                        raise TimeoutError(
                            f"{phase} exceeded {int(timeout)}s; log: {log_path}"
                        )
                    if now - last_report >= _progress_interval():
                        _announce(
                            f"{phase} running; elapsed={int(now - started)}s; "
                            f"no R output for {int(now - last_output)}s; "
                            f"R pid={process.pid}; last: {last_line}; log: {log_path}"
                        )
                        last_report = now
                    time.sleep(0.2)
            code = process.wait()
        except BaseException:
            _terminate_tree(process)
            raise
    if code != 0:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        raise RuntimeError(
            f"{phase} failed with exit code {code}; log: {log_path}\n"
            + "\n".join(lines[-35:])
        )
    _announce(f"{phase} completed in {int(time.monotonic() - started)}s")


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


def _run(
    cmd: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess:
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
    result = _run(
        [str(rscript), *R_STARTUP_ARGS, "-e", "cat(as.character(getRversion()))"],
        timeout=30,
    )
    return result.stdout.strip()


def _compatible_r(version: str) -> bool:
    return version == "4.6" or version.startswith("4.6.")


def _system_rscript() -> Path | None:
    explicit = os.environ.get("MLLABIOME_RSCRIPT")
    candidates = [explicit] if explicit else []
    if platform.system() == "Darwin":
        candidates.extend(
            [
                "/Library/Frameworks/R.framework/Resources/bin/Rscript",
                "/usr/local/bin/Rscript",
                "/opt/homebrew/bin/Rscript",
            ]
        )
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
            raise RuntimeError(
                f"Unsupported macOS architecture for managed R: {machine}"
            )
        name = f"portable-r-{R_VERSION}-macos-{arch}.tar.gz"
        return (
            f"https://github.com/portable-r/portable-r-macos/releases/download/v{R_VERSION}/{name}",
            name,
        )
    if system == "linux":
        if machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
            raise RuntimeError(
                f"Unsupported Linux architecture for managed R: {machine}"
            )
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
    request = urllib.request.Request(
        url, headers={"User-Agent": "mllabiome-siamcat-runtime"}
    )
    count = 0
    try:
        with _heartbeat(f"Downloading managed R runtime to {destination}"):
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                destination.open("wb") as out,
            ):
                total = response.headers.get("Content-Length")
                _announce(
                    f"R runtime download connected; total={total or 'unknown'} bytes"
                )
                last_report = time.monotonic()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last_report >= _progress_interval():
                        _announce(f"Managed R download: {count:,} bytes received")
                        last_report = time.monotonic()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not download the managed R runtime from {url}. "
            "Check network access or set MLLABIOME_RSCRIPT to an existing R 4.6.x installation."
        ) from exc
    _announce(f"Managed R download completed: {count:,} bytes")


def _find_rscript(root: Path) -> Path:
    name = "Rscript.exe" if os.name == "nt" else "Rscript"
    candidates = sorted(root.rglob(name), key=lambda p: (len(p.parts), str(p)))
    for candidate in candidates:
        if candidate.parent.name == "bin" and candidate.is_file():
            if os.name != "nt":
                candidate.chmod(candidate.stat().st_mode | 0o111)
            return candidate
    raise RuntimeError(
        f"R runtime was extracted, but {name} could not be found under {root}."
    )


def _lock_state(path: Path) -> tuple[bool, str, os.stat_result | None]:
    try:
        status = path.stat()
        content = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False, "lock disappeared", None
    except OSError as exc:
        return False, f"lock unreadable: {exc}", None
    match = re.search(r"(?:^|\n)pid=(\d+)(?:\n|$)", content)
    host_match = re.search(r"(?:^|\n)host=([^\n]+)", content)
    if host_match and host_match.group(1).strip() != socket.gethostname():
        return False, f"holder on host {host_match.group(1).strip()}", status
    if match:
        pid = int(match.group(1))
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True, f"abandoned by exited process {pid}", status
        except PermissionError:
            pass
        except OSError:
            pass
        return False, f"held by active process {pid}", status
    age = time.time() - status.st_mtime
    if age >= 30:
        return True, f"incomplete lock without PID, age {int(age)}s", status
    return False, f"lock is being initialized, age {int(age)}s", status


@contextlib.contextmanager
def _installation_lock(path: Path, timeout: float = 300.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_report = 0.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            data = (
                f"pid={os.getpid()}\nhost={socket.gethostname()}\n"
                f"created={time.time()}\n"
            )
            os.write(descriptor, data.encode("utf-8"))
            os.fsync(descriptor)
            _announce(f"Acquired installation lock: {path}")
        except FileExistsError:
            stale, owner, previous = _lock_state(path)
            if stale and previous is not None:
                try:
                    current = path.stat()
                    if (
                        current.st_ino == previous.st_ino
                        and current.st_mtime_ns == previous.st_mtime_ns
                    ):
                        path.unlink()
                        _announce(
                            f"Recovered stale installation lock ({owner}): {path}"
                        )
                        continue
                except FileNotFoundError:
                    continue
            now = time.monotonic()
            if now - last_report >= _progress_interval() or last_report == 0.0:
                _announce(
                    f"Waiting for installation lock ({owner}); elapsed={int(now - started)}s; {path}"
                )
                last_report = now
            if now - started >= timeout:
                raise TimeoutError(
                    f"SIAMCAT installation lock remained busy for {int(timeout)}s "
                    f"({owner}): {path}. Check whether that process is still installing R packages."
                )
            time.sleep(min(0.5, _progress_interval()))
    try:
        yield
    finally:
        original_inode = os.fstat(descriptor).st_ino if descriptor is not None else None
        if descriptor is not None:
            os.close(descriptor)
        try:
            if path.stat().st_ino == original_inode:
                path.unlink()
                _announce(f"Released installation lock: {path}")
            else:
                _announce(f"Installation lock was replaced by another process: {path}")
        except FileNotFoundError:
            pass


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
        print(
            f"[SIAMCAT] Downloading managed R {R_VERSION} from {url}",
            file=sys.stderr,
            flush=True,
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mllabiome-r-install-") as tmp:
            archive = Path(tmp) / filename
            _download(url, archive)
            with tempfile.TemporaryDirectory(
                prefix=f".R-{R_VERSION}-staging-", dir=root.parent
            ) as staging_dir:
                extracted = Path(staging_dir) / "runtime"
                extracted.mkdir()
                _announce(f"Extracting managed R runtime: {archive}")
                with _heartbeat("Extracting managed R runtime"):
                    if filename.endswith(".zip"):
                        _safe_extract_zip(archive, extracted)
                    else:
                        _safe_extract_tar(archive, extracted)
                _announce("Managed R runtime extracted")
                candidate = _find_rscript(extracted)
                relative = candidate.relative_to(extracted)
                if root.exists():
                    shutil.rmtree(root)
                extracted.replace(root)
                rscript = root / relative
                version = _r_version(rscript)
                if not _compatible_r(version):
                    shutil.rmtree(root, ignore_errors=True)
                    raise RuntimeError(
                        f"Downloaded R {version}, but mllabiome requires R 4.6.x for Bioconductor {BIOCONDUCTOR_VERSION}."
                    )
                marker.write_text(
                    json.dumps(
                        {
                            "rscript": str(relative),
                            "r_version": version,
                            "source_url": url,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return rscript


def _choose_rscript(
    mode: str = "auto", explicit: str | os.PathLike[str] | None = None
) -> Path:
    if explicit is None and os.environ.get("MLLABIOME_RSCRIPT"):
        explicit = os.environ["MLLABIOME_RSCRIPT"]
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Rscript does not exist: {path}")
        version = _r_version(path)
        if not _compatible_r(version):
            raise RuntimeError(
                f"SIAMCAT backend requires R 4.6.x; found R {version} at {path}."
            )
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


def _library_for_rscript(rscript: Path, version: str) -> Path:
    base = _cache_root() / "r-library"
    if platform.system() == "Darwin" and _portable_macos_library(rscript) is None:
        digest = hashlib.sha256(str(rscript.resolve()).encode("utf-8")).hexdigest()[:12]
        return base / f"R-{version}_Bioc-{BIOCONDUCTOR_VERSION}_system-{digest}"
    return base / f"R-{R_VERSION}_Bioc-{BIOCONDUCTOR_VERSION}"


def _r_env(library: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(library)
    env["MLLABIOME_R_LIBRARY"] = str(library)
    return env


def _portable_macos_library(rscript: Path) -> Path | None:
    if platform.system() != "Darwin":
        return None
    root = rscript.parent.parent
    target = root / "lib" / "libR.dylib"
    if root.name.startswith("portable-r-") and target.is_file():
        return target
    return None


def _repair_portable_macos_packages(rscript: Path, library: Path) -> None:
    lib_r = _portable_macos_library(rscript)
    if lib_r is None or not library.is_dir():
        return
    installer = shutil.which("install_name_tool")
    inspector = shutil.which("otool")
    signer = shutil.which("codesign")
    if not all((installer, inspector, signer)):
        raise RuntimeError(
            "Portable R on macOS requires otool, install_name_tool and codesign to "
            "repair external R package binaries. Install Apple Command Line Tools "
            "or use an official R 4.6.x installation via MLLABIOME_RSCRIPT."
        )
    _announce(f"Checking compiled R package links for portable runtime: {library}")
    modified = 0
    scanned = 0
    with _heartbeat("Auditing portable R package dynamic libraries"):
        for path in sorted(library.rglob("*")):
            if not path.is_file() or path.suffix not in {".so", ".dylib"}:
                continue
            scanned += 1
            result = subprocess.run(
                [inspector, "-L", str(path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not inspect compiled R package library {path}: {result.stderr.strip()}"
                )
            updates = []
            linked = False
            for line in result.stdout.splitlines()[1:]:
                name = line.strip().split(" (", 1)[0]
                target = None
                if (
                    name.startswith("/Library/Frameworks/R.framework/")
                    and "/lib/" in name
                ):
                    target = lib_r.parent / name.rsplit("/", 1)[-1]
                elif name == "@rpath/libR.dylib":
                    target = lib_r
                if target is None:
                    continue
                if not target.is_file():
                    raise RuntimeError(
                        f"Cannot repair {path}: dependency {name} has no portable replacement at {target}"
                    )
                link = path.parent / ".mllabiome-r-lib"
                if link.is_symlink() and link.resolve() != lib_r.parent:
                    link.unlink()
                if not link.exists():
                    link.symlink_to(lib_r.parent, target_is_directory=True)
                if not linked:
                    linked = True
                updated_name = f"@loader_path/{link.name}/{target.name}"
                if name != updated_name:
                    updates.extend(("-change", name, updated_name))
            if not updates:
                continue
            update = subprocess.run(
                [installer, *updates, str(path)],
                capture_output=True,
                text=True,
            )
            if update.returncode != 0:
                raise RuntimeError(
                    f"Failed to repair compiled R package library {path}: {update.stderr.strip()}"
                )
            signature = subprocess.run(
                [signer, "--force", "--sign", "-", str(path)],
                capture_output=True,
                text=True,
            )
            if signature.returncode != 0:
                raise RuntimeError(
                    f"Failed to sign repaired R package library {path}: {signature.stderr.strip()}"
                )
            modified += 1
            _announce(
                f"Repaired compiled package linkage: {path.name} ({modified} files)"
            )
    _announce(
        f"Portable R linkage audit complete: scanned={scanned}; repaired={modified}"
    )


def _compiled_packages(library: Path) -> list[str]:
    if not library.exists():
        return []
    candidates = []
    for package in sorted(library.iterdir()):
        if package.is_dir() and (package / "DESCRIPTION").exists():
            if any((package / "libs").rglob("*.so")) or any(
                (package / "libs").rglob("*.dylib")
            ):
                candidates.append(package.name)
    return candidates


def diagnose_siamcat_compiled_packages(
    *,
    runtime: str = "auto",
    rscript: str | os.PathLike[str] | None = None,
) -> bool:
    with _heartbeat("Selecting R runtime for diagnostics"):
        chosen = _choose_rscript(runtime, rscript)
        version = _r_version(chosen)
    library = _library_for_rscript(chosen, version)
    _announce(f"Diagnosing native R package loading with R {version}: {chosen}")
    _announce(f"Testing installed compiled packages in isolated R processes: {library}")
    packages = _compiled_packages(library)
    if not packages:
        _announce("No installed compiled packages found in the managed R library")
        return False
    failure = False
    started = time.monotonic()
    for index, package in enumerate(packages, start=1):
        _announce(f"Native dependency check {index}/{len(packages)}: {package}")
        expr = (
            ".libPaths(c(Sys.getenv('R_LIBS_USER'), .libPaths())); "
            f"loadNamespace({json.dumps(package)}); cat('OK')"
        )
        try:
            with _heartbeat(f"Loading native R package {package}"):
                result = subprocess.run(
                    [str(chosen), *R_STARTUP_ARGS, "-e", expr],
                    env=_r_env(library),
                    capture_output=True,
                    text=True,
                    timeout=40,
                )
        except subprocess.TimeoutExpired:
            _announce(f"TIMEOUT loading {package} (40s)")
            failure = True
            continue
        if result.returncode != 0:
            failure = True
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            excerpt = " | ".join(detail[-5:])[:700]
            _announce(
                f"FAILED loading {package}; exit={result.returncode}; "
                f"native_crash={result.returncode < 0 or result.returncode in {134, 139}}; {excerpt}"
            )
    _announce(
        f"Native dependency checks finished: {len(packages)} packages in "
        f"{int(time.monotonic() - started)}s; failures={'yes' if failure else 'no'}"
    )
    return not failure


def _installed_siamcat_version(rscript: Path, library: Path) -> str | None:
    expr = (
        ".libPaths(c(Sys.getenv('R_LIBS_USER'), .libPaths())); "
        "if (requireNamespace('SIAMCAT', quietly=TRUE)) cat(as.character(packageVersion('SIAMCAT')))"
    )
    with _heartbeat("Checking SIAMCAT namespace and dependencies"):
        try:
            result = subprocess.run(
                [str(rscript), *R_STARTUP_ARGS, "-e", expr],
                env=_r_env(library),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "SIAMCAT namespace loading exceeded 60 seconds. "
                "Check the macOS R runtime and installed compiled packages."
            ) from exc
    if result.returncode != 0:
        if result.returncode < 0 or result.returncode in {134, 139}:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"R crashed while loading SIAMCAT (exit code {result.returncode}). "
                "A compiled dependency may be incompatible with the selected R runtime. "
                "On macOS, install official CRAN R 4.6.x, set MLLABIOME_RSCRIPT "
                "to /Library/Frameworks/R.framework/Resources/bin/Rscript, and use "
                "a fresh MLLABIOME_CACHE_DIR. Run siamcat_preflight --diagnose "
                "to test installed compiled packages individually. "
                f"R output: {detail[-1600:]}"
            )
        return None
    return result.stdout.strip() or None


def _install_siamcat(rscript: Path, library: Path) -> None:
    library.mkdir(parents=True, exist_ok=True)
    expr = f"""
lib <- Sys.getenv('R_LIBS_USER')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib, .libPaths()))
options(repos=c(CRAN='https://cloud.r-project.org'))
if (!requireNamespace('BiocManager', quietly=TRUE)) {{
    message('[SIAMCAT:R] Installing BiocManager')
    install.packages('BiocManager', lib=lib)
}}
if (!requireNamespace('BiocManager', quietly=TRUE)) stop('BiocManager installation failed')
message('[SIAMCAT:R] Configuring Bioconductor {BIOCONDUCTOR_VERSION}')
BiocManager::install(version='{BIOCONDUCTOR_VERSION}', ask=FALSE, update=FALSE)
options(repos=BiocManager::repositories(version='{BIOCONDUCTOR_VERSION}'))
installed <- utils::installed.packages()
if (!('Rcpp' %in% rownames(installed))) {{
    message('[SIAMCAT:R] Installing missing Rcpp in ', lib)
    BiocManager::install('Rcpp', lib=lib, ask=FALSE, update=FALSE)
}}
installed_version <- tryCatch(as.character(utils::packageVersion('SIAMCAT')),
                              error=function(e) NA_character_)
if (is.na(installed_version) || installed_version != '{SIAMCAT_VERSION}') {{
    message('[SIAMCAT:R] Installing SIAMCAT {SIAMCAT_VERSION} in ', lib)
    BiocManager::install('SIAMCAT', ask=FALSE, update=FALSE, lib=lib,
                         dependencies=NA)
}} else {{
    message('[SIAMCAT:R] SIAMCAT ', installed_version,
            ' is installed; Python will repair compiled libraries before loading it')
}}
message('[SIAMCAT:R] R package installation phase completed; validating in a fresh R process')
"""
    log_path = _cache_root() / "logs" / "siamcat-install.log"
    _run_with_progress(
        [str(rscript), *R_STARTUP_ARGS, "-e", expr],
        env=_r_env(library),
        log_path=log_path,
        phase="R package installation and repair",
        timeout=3600,
    )
    _repair_portable_macos_packages(rscript, library)
    version = _installed_siamcat_version(rscript, library) or ""
    if version != SIAMCAT_VERSION:
        raise RuntimeError(
            f"Expected loadable SIAMCAT {SIAMCAT_VERSION} from Bioconductor {BIOCONDUCTOR_VERSION}, "
            f"but found {version or 'unavailable'}. Installation log: {log_path}. "
            "If the installation is damaged, use an official CRAN R 4.6.x "
            "installation via MLLABIOME_RSCRIPT."
        )


def ensure_siamcat_runtime(
    *,
    runtime: str = "auto",
    rscript: str | os.PathLike[str] | None = None,
) -> SIAMCATRuntime:
    _announce("Checking compatible R runtime")
    with _heartbeat("Locating compatible R runtime"):
        chosen = _choose_rscript(runtime, rscript)
        r_version = _r_version(chosen)
    _announce(f"Using R {r_version} at {chosen}")
    library = _library_for_rscript(chosen, r_version)
    marker = library / ".mllabiome-siamcat.json"
    _repair_portable_macos_packages(chosen, library)
    installed = _installed_siamcat_version(chosen, library)
    if installed != SIAMCAT_VERSION:
        lock = _cache_root() / "locks" / f"SIAMCAT-{SIAMCAT_VERSION}.lock"
        _announce(f"Preparing R library for SIAMCAT {SIAMCAT_VERSION}")
        with _installation_lock(lock):
            installed = _installed_siamcat_version(chosen, library)
            if installed != SIAMCAT_VERSION:
                _install_siamcat(chosen, library)
                installed = _installed_siamcat_version(chosen, library)
    if installed != SIAMCAT_VERSION:
        raise RuntimeError(
            f"SIAMCAT {SIAMCAT_VERSION} could not be prepared; found {installed!r}."
        )
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
    _announce(f"SIAMCAT {installed} ready in {library}")
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
    "diagnose_siamcat_compiled_packages",
]
