from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except Exception:
    psutil = None

try:
    import resource
except Exception:
    resource = None


@dataclass
class ResourceTracker:
    sample_interval_s: float = 0.10

    def __post_init__(self) -> None:
        self._started = False
        self._start_wall = 0.0
        self._cpu_before = (0.0, 0.0)
        self._peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.result: dict[str, Any] = {}

    def _cpu_snapshot(self) -> tuple[float, float]:
        if resource is not None:
            try:
                self_usage = resource.getrusage(resource.RUSAGE_SELF)
                child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                return (
                    float(self_usage.ru_utime + child_usage.ru_utime),
                    float(self_usage.ru_stime + child_usage.ru_stime),
                )
            except Exception:
                pass
        if psutil is not None:
            try:
                proc = psutil.Process(os.getpid())
                user = 0.0
                system = 0.0
                for item in [proc, *proc.children(recursive=True)]:
                    try:
                        times = item.cpu_times()
                        user += float(times.user)
                        system += float(times.system)
                    except Exception:
                        pass
                return user, system
            except Exception:
                pass
        return 0.0, 0.0

    def _rss_snapshot(self) -> int:
        if psutil is not None:
            try:
                proc = psutil.Process(os.getpid())
                total = 0
                for item in [proc, *proc.children(recursive=True)]:
                    try:
                        total += int(item.memory_info().rss)
                    except Exception:
                        pass
                return total
            except Exception:
                pass
        if resource is not None:
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                value = int(usage.ru_maxrss)
                if platform.system().lower() != "darwin":
                    value *= 1024
                return value
            except Exception:
                pass
        return 0

    def _sample_loop(self) -> None:
        interval = max(0.02, float(self.sample_interval_s))
        while not self._stop_event.wait(interval):
            self._peak_rss_bytes = max(self._peak_rss_bytes, self._rss_snapshot())

    def start(self) -> ResourceTracker:
        if self._started:
            return self
        self._started = True
        self._start_wall = time.perf_counter()
        self._cpu_before = self._cpu_snapshot()
        self._peak_rss_bytes = self._rss_snapshot()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if not self._started:
            return dict(self.result)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, float(self.sample_interval_s) * 3.0))
        self._peak_rss_bytes = max(self._peak_rss_bytes, self._rss_snapshot())
        user_after, system_after = self._cpu_snapshot()
        user_before, system_before = self._cpu_before
        wall = max(0.0, float(time.perf_counter() - self._start_wall))
        user = max(0.0, float(user_after - user_before))
        system = max(0.0, float(system_after - system_before))
        cpu = user + system
        self.result = {
            "wall_time_s": wall,
            "cpu_user_s": user,
            "cpu_system_s": system,
            "cpu_time_s": cpu,
            "cpu_core_hours": cpu / 3600.0,
            "peak_rss_bytes": int(self._peak_rss_bytes),
            "peak_rss_gib": float(self._peak_rss_bytes) / float(1024**3),
            "mean_cpu_cores": cpu / wall if wall > 0 else 0.0,
        }
        self._started = False
        return dict(self.result)

    def __enter__(self) -> ResourceTracker:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _cpu_model() -> str:
    system = platform.system().lower()
    if system == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
                ).strip()
                if value:
                    return value
            except Exception:
                pass
    if system == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name") and ":" in line:
                        value = line.split(":", 1)[1].strip()
                        if value:
                            return value
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def machine_profile(
    logical_cpus: int | None = None, physical_cpus: int | None = None
) -> dict[str, Any]:
    total_memory = None
    available_memory = None
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            total_memory = int(vm.total)
            available_memory = int(vm.available)
        except Exception:
            pass
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": int(logical_cpus)
        if logical_cpus is not None
        else (int(psutil.cpu_count(logical=True)) if psutil is not None else None),
        "physical_cpus": int(physical_cpus)
        if physical_cpus is not None
        else (
            int(psutil.cpu_count(logical=False) or 0) if psutil is not None else None
        ),
        "memory_total_bytes": total_memory,
        "memory_available_bytes": available_memory,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
    }
