"""Thermal zone + cpufreq throttle detection via time_in_state.

Throttle counting via `throttle_count` is non-standard and absent on most
ARM kernels. Instead we read each cpufreq policy's `time_in_state` (a stable
sysfs interface), sample at start and end of a workload, and count ticks
spent below the policy's max frequency. Any non-trivial below-max time
during a sustained 30-min loop is evidence of throttling.
"""

from __future__ import annotations

from pathlib import Path

CPUFREQ_BASE = Path("/sys/devices/system/cpu/cpufreq")


def read_zone(zone_path: str) -> float | None:
    """Read /sys/class/thermal/thermal_zone*/temp. Returns °C or None if missing."""
    p = Path(zone_path)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_zones(zones: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for z in zones:
        v = read_zone(z)
        if v is not None:
            out[z] = v
    return out


def _read_time_in_state(policy: Path) -> dict[int, int]:
    """Return {freq_khz: ticks} for one cpufreq policy, empty if unavailable."""
    f = policy / "stats" / "time_in_state"
    if not f.exists():
        return {}
    out: dict[int, int] = {}
    try:
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                out[int(parts[0])] = int(parts[1])
            except ValueError:
                continue
    except OSError:
        return {}
    return out


def _policy_max_khz(policy: Path) -> int | None:
    f = policy / "cpuinfo_max_freq"
    if not f.exists():
        return None
    try:
        return int(f.read_text().strip())
    except (OSError, ValueError):
        return None


def snapshot_time_in_state() -> dict[str, dict[int, int]]:
    """Snapshot every cpufreq policy's time_in_state. Returns {policy_name: {khz: ticks}}."""
    if not CPUFREQ_BASE.exists():
        return {}
    return {p.name: _read_time_in_state(p) for p in sorted(CPUFREQ_BASE.glob("policy*"))}


def throttle_summary(
    start: dict[str, dict[int, int]],
    end: dict[str, dict[int, int]],
) -> dict:
    """Compare two time_in_state snapshots; report below-max ticks per policy.

    "Below-max" means any time spent at a frequency strictly less than that
    policy's `cpuinfo_max_freq`. On a board running flat-out at peak, this is
    near zero. Non-trivial below-max time during a sustained load is a
    throttle signal. We report ticks (10ms units on most kernels), the
    derived seconds (assuming USER_HZ=100), and the dominant slowdown freq.
    """
    if not start or not end:
        return {"available": False}
    per_policy: dict[str, dict] = {}
    total_below_max_ticks = 0
    for name, end_ticks in end.items():
        start_ticks = start.get(name, {})
        deltas = {khz: end_ticks.get(khz, 0) - start_ticks.get(khz, 0) for khz in end_ticks}
        max_khz = max(deltas) if deltas else 0
        below = {khz: d for khz, d in deltas.items() if khz < max_khz and d > 0}
        below_ticks = sum(below.values())
        total_below_max_ticks += below_ticks
        per_policy[name] = {
            "max_khz": max_khz,
            "total_ticks": sum(deltas.values()),
            "below_max_ticks": below_ticks,
            "dominant_below_khz": (max(below, key=below.get) if below else None),
        }
    return {
        "available": True,
        "total_below_max_ticks": total_below_max_ticks,
        "approx_below_max_s": total_below_max_ticks / 100.0,  # USER_HZ=100 assumption
        "per_policy": per_policy,
    }
