"""Pre-flight checks — validate the environment *before* a 90-min run.

Goal: catch dumb operator/config errors in the first 10 seconds, not the last
5 minutes of a long bench. Each check returns (ok, message); the orchestrator
prints them all and aborts on any failure unless --force is passed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import psycopg

from virtues_bench.models import fetch_all
from virtues_bench.telemetry import power, thermal


CheckResult = tuple[bool, str]


def check_models() -> CheckResult:
    try:
        fetched = fetch_all()
    except Exception as exc:
        return False, f"model fetch/verify failed: {exc!r}"
    labels = sorted({f"{name}/{variant}" for name, variant in fetched.keys()})
    return True, f"models verified: {', '.join(labels)}"


def check_postgres(dsn: str) -> CheckResult:
    try:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=5) as conn:
            row = conn.execute(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if row is None:
                # Try to create it; succeeds if user has perms
                try:
                    conn.execute("CREATE EXTENSION vector")
                except Exception as exc:
                    return False, f"pgvector extension not installed in DB: {exc!r}"
            return True, f"postgres reachable, pgvector available ({dsn})"
    except Exception as exc:
        return False, f"postgres unreachable at {dsn}: {exc!r}"


def check_governor_writable() -> CheckResult:
    """Can we write scaling_governor? Needed for pass 1b."""
    base = Path("/sys/devices/system/cpu/cpufreq")
    if not base.exists():
        return True, "no cpufreq (likely macOS); pass 1b will be skipped"
    policies = sorted(base.glob("policy*"))
    if not policies:
        return True, "cpufreq present but no policies (skipping check)"
    g = policies[0] / "scaling_governor"
    if not g.exists():
        return True, "no scaling_governor file (skipping check)"
    try:
        current = g.read_text().strip()
        g.write_text(current)  # write current value back — non-destructive test
        return True, f"governor is writable (current: {current})"
    except OSError as exc:
        return False, (
            f"cannot write scaling_governor ({exc!r}); "
            f"pass 1b needs root — re-run with sudo or pass --skip-perf-pass"
        )


def check_thermal_zones(zones: list[str]) -> CheckResult:
    if not zones:
        return True, "no thermal zones declared (K5 max temp will be null)"
    missing = [z for z in zones if not Path(z).exists()]
    if missing:
        return False, f"thermal zones declared but missing: {missing}"
    sample = thermal.read_zones(zones)
    return True, f"thermal zones readable: {sample}"


def check_power_paths(paths: list[str]) -> CheckResult:
    if not paths:
        return True, "no hwmon paths declared (K6/K7 need --idle/peak-power-w override)"
    reading = power.read_hwmon_watts(paths)
    if reading is None:
        return False, f"hwmon paths declared but unreadable: {paths}"
    return True, f"hwmon power read: {reading:.2f} W"


def run_preflight(
    board: dict,
    dsn: str,
    skip_pgvector: bool,
    skip_perf_pass: bool,
    skip_thermal: bool,
) -> list[CheckResult]:
    checks: list[tuple[str, Callable[[], CheckResult]]] = [
        ("models", check_models),
    ]
    if not skip_pgvector:
        checks.append(("postgres", lambda: check_postgres(dsn)))
    if not skip_perf_pass:
        checks.append(("governor", check_governor_writable))
    if not skip_thermal:
        zones = board.get("thermal", {}).get("zones", []) or []
        hwmon = board.get("power", {}).get("hwmon_paths", []) or []
        checks.append(("thermal_zones", lambda: check_thermal_zones(zones)))
        checks.append(("power_paths", lambda: check_power_paths(hwmon)))

    results: list[CheckResult] = []
    for label, fn in checks:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"check raised: {exc!r}"
        marker = "✓" if ok else "✗"
        print(f"[preflight] {marker} {label}: {msg}", file=sys.stderr)
        results.append((ok, f"{label}: {msg}"))
    return results
