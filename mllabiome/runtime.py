from __future__ import annotations

import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from typing import Any

from joblib import Parallel, cpu_count, parallel_backend


@dataclass(frozen=True)
class ExecutionPlan:
    logical_cpus: int
    physical_cpus: int
    memory_bytes: int | None
    workers: int
    threads_per_worker: int
    backend: str


def _linux_available_memory() -> int | None:
    host = None
    path = "/proc/meminfo"
    if os.path.exists(path):
        values = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    values[parts[0].rstrip(":")] = int(parts[1]) * 1024
        host = values.get("MemAvailable") or values.get("MemFree")
    cgroup = None
    try:
        max_path = "/sys/fs/cgroup/memory.max"
        current_path = "/sys/fs/cgroup/memory.current"
        if os.path.exists(max_path) and os.path.exists(current_path):
            with open(max_path, "r", encoding="utf-8") as fh:
                limit_text = fh.read().strip()
            with open(current_path, "r", encoding="utf-8") as fh:
                current_text = fh.read().strip()
            if limit_text != "max":
                cgroup = max(0, int(limit_text) - int(current_text))
        else:
            limit_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            usage_path = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
            if os.path.exists(limit_path) and os.path.exists(usage_path):
                with open(limit_path, "r", encoding="utf-8") as fh:
                    limit = int(fh.read().strip())
                with open(usage_path, "r", encoding="utf-8") as fh:
                    usage = int(fh.read().strip())
                if limit < 1 << 60:
                    cgroup = max(0, limit - usage)
    except Exception:
        cgroup = None
    values = [x for x in (host, cgroup) if x is not None and x > 0]
    return min(values) if values else None


def _darwin_available_memory() -> int | None:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        first = out.splitlines()[0]
        page_size = 4096
        for token in first.replace(".", "").split():
            if token.isdigit():
                page_size = int(token)
                break
        pages = {}
        for line in out.splitlines()[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            number = value.strip().rstrip(".").replace(".", "")
            if number.isdigit():
                pages[key.strip()] = int(number)
        reclaimable = sum(
            pages.get(key, 0)
            for key in (
                "Pages free",
                "Pages inactive",
                "Pages speculative",
                "Pages purgeable",
            )
        )
        if reclaimable > 0:
            return int(reclaimable * page_size)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(int(out) * 0.75)
    except Exception:
        return None


def _windows_available_memory() -> int | None:
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    except Exception:
        return None
    return None


def available_memory_bytes() -> int | None:
    system = platform.system().lower()
    if system == "linux":
        return _linux_available_memory()
    if system == "darwin":
        return _darwin_available_memory()
    if system == "windows":
        return _windows_available_memory()
    return None


def _physical_cpu_count() -> int:
    try:
        value = int(cpu_count(only_physical_cores=True))
    except TypeError:
        value = int(cpu_count())
    return max(1, value)


def _logical_cpu_count() -> int:
    return max(1, int(cpu_count()))


def resolve_execution_plan(
    n_jobs: int | str,
    task_count: int,
    *,
    backend: str = "loky",
    memory_fraction: float = 0.80,
    min_worker_memory_gib: float = 1.0,
) -> ExecutionPlan:
    logical = _logical_cpu_count()
    physical = _physical_cpu_count()
    memory = available_memory_bytes()
    count = max(1, int(task_count))
    if isinstance(n_jobs, str):
        text = n_jobs.strip().lower()
        if text not in {"auto", "all"}:
            raise ValueError("Evaluation.n_jobs must be an integer, 'auto', or 'all'.")
        desired = min(physical, 8) if text == "auto" else logical
    else:
        requested = int(n_jobs)
        if requested == 0:
            desired = physical
        elif requested == -1:
            desired = logical
        elif requested < -1:
            desired = max(1, logical + 1 + requested)
        else:
            desired = min(logical, max(1, requested))
    memory_cap = desired
    if memory is not None and min_worker_memory_gib > 0:
        usable = max(1, int(float(memory) * float(memory_fraction)))
        per_worker = max(1, int(float(min_worker_memory_gib) * (1024**3)))
        memory_cap = max(1, usable // per_worker)
    workers = max(1, min(desired, count, memory_cap))
    threads = max(1, physical // workers)
    return ExecutionPlan(
        logical_cpus=logical,
        physical_cpus=physical,
        memory_bytes=memory,
        workers=workers,
        threads_per_worker=threads,
        backend=str(backend),
    )


_WAVE_COUNTER = count()


def _loky_wave_initializer(token: int) -> None:
    os.environ["MLLABIOME_WORKER_WAVE"] = str(int(token))


def iter_parallel_tasks(tasks: list[Any], execution: ExecutionPlan):
    items = list(tasks)
    if not items:
        return
    workers = max(1, min(int(execution.workers), len(items)))
    if workers == 1:
        for fn, args, kwargs in items:
            yield fn(*args, **kwargs)
        return
    if str(execution.backend) == "loky":
        for start in range(0, len(items), workers):
            wave = items[start : start + workers]
            wave_workers = max(1, min(workers, len(wave)))
            token = next(_WAVE_COUNTER)
            yield from Parallel(
                n_jobs=wave_workers,
                backend="loky",
                return_as="generator_unordered",
                pre_dispatch=wave_workers,
                batch_size=1,
                max_nbytes="1M",
                mmap_mode="r",
                inner_max_num_threads=max(1, int(execution.threads_per_worker)),
                initializer=_loky_wave_initializer,
                initargs=(token,),
            )(wave)
        return
    with parallel_backend(str(execution.backend), n_jobs=workers):
        yield from Parallel(
            n_jobs=workers,
            backend=str(execution.backend),
            return_as="generator_unordered",
            pre_dispatch=workers,
            batch_size=1,
        )(items)


@contextmanager
def thread_environment(threads: int):
    value = str(max(1, int(threads)))
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "RCPP_PARALLEL_NUM_THREADS",
    )
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def configure_estimator_threads(estimator: Any, threads: int) -> Any:
    if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
        return estimator
    try:
        params = estimator.get_params(deep=True)
    except Exception:
        return estimator
    names = {"n_jobs", "nthread", "thread_count", "num_threads", "n_threads"}
    updates = {}
    for key in params:
        leaf = key.rsplit("__", 1)[-1]
        if leaf not in names:
            continue
        owner_key = key.rsplit("__", 1)[0] if "__" in key else ""
        owner = params.get(owner_key, estimator) if owner_key else estimator
        if leaf == "n_jobs" and owner.__class__.__name__ == "LogisticRegression":
            continue
        updates[key] = int(max(1, threads))
    if updates:
        try:
            estimator.set_params(**updates)
        except Exception:
            pass
    return estimator
