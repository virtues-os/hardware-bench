"""Memory snapshot from /proc/meminfo."""

from __future__ import annotations

from pathlib import Path


def read_meminfo() -> dict[str, int]:
    """Return /proc/meminfo as a dict of {field: kB}. Empty if not Linux."""
    p = Path("/proc/meminfo")
    if not p.exists():
        return {}
    out: dict[str, int] = {}
    for line in p.read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            out[key.strip()] = int(parts[0])
        except ValueError:
            continue
    return out


def mem_available_mb() -> int | None:
    info = read_meminfo()
    val = info.get("MemAvailable")
    return val // 1024 if val is not None else None
