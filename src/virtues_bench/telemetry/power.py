"""Power reading from /sys/class/hwmon. Returns watts."""

from __future__ import annotations

from pathlib import Path


def read_hwmon_watts(paths: list[str]) -> float | None:
    """Sum watts across the declared hwmon paths. Returns None if none available.

    Each path may be either a single `power*_input` file (microwatts) or a
    hwmon device dir that we'll scan for `power*_input` entries.
    """
    if not paths:
        return None
    total = 0.0
    any_read = False
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else list(path.glob("power*_input"))
        for f in candidates:
            try:
                total += int(f.read_text().strip()) / 1_000_000.0
                any_read = True
            except (OSError, ValueError):
                continue
    return total if any_read else None
