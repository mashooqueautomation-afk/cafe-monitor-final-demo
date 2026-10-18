from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import psutil

_process = psutil.Process(os.getpid())


def _read_int(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip()
        if value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def get_system_resources() -> dict:
    """Best-effort process + container resource metrics (not GPU metrics)."""
    try:
        process_cpu = _process.cpu_percent(interval=None)
        process_rss = _process.memory_info().rss
        virtual = psutil.virtual_memory()
        # cgroup v2 first, then v1. Host data is a fallback outside containers.
        cgroup_used = _read_int("/sys/fs/cgroup/memory.current")
        cgroup_limit = _read_int("/sys/fs/cgroup/memory.max")
        if cgroup_used is None:
            cgroup_used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            cgroup_limit = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if cgroup_limit is not None and cgroup_limit > (1 << 60):
            cgroup_limit = None
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process_cpu_percent": round(float(process_cpu), 1),
            "process_ram_bytes": int(process_rss),
            "host_cpu_percent": round(float(psutil.cpu_percent(interval=None)), 1),
            "host_ram_used_bytes": int(virtual.used),
            "host_ram_total_bytes": int(virtual.total),
            "container_ram_used_bytes": int(cgroup_used) if cgroup_used is not None else int(process_rss),
            "container_ram_limit_bytes": int(cgroup_limit) if cgroup_limit is not None else None,
            "cpu_count": psutil.cpu_count(logical=True) or 1,
            "note": "Backend/container resource usage; GPU usage is not included.",
        }
    except Exception as exc:
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "error": f"{type(exc).__name__}: {exc}"}
