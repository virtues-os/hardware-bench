"""System fingerprint capture — what was the actual environment for this run."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import onnxruntime as ort

from virtues_bench.telemetry.memory import read_meminfo


@dataclass
class SystemFingerprint:
    kernel: str
    machine: str
    python: str
    ort_version: str
    available_eps: list[str]
    os_release: dict
    governor: str | None
    governor_per_policy: dict[str, str]
    ram_total_mb: int | None


def _read_os_release() -> dict[str, str]:
    p = Path("/etc/os-release")
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"')
    return out


def read_governors() -> dict[str, str]:
    base = Path("/sys/devices/system/cpu/cpufreq")
    if not base.exists():
        return {}
    out: dict[str, str] = {}
    for policy in sorted(base.glob("policy*")):
        g = policy / "scaling_governor"
        if g.exists():
            try:
                out[policy.name] = g.read_text().strip()
            except OSError:
                continue
    return out


def set_governor(name: str) -> dict[str, str | None]:
    """Best-effort set scaling_governor across all cpufreq policies.

    Returns the *previous* governor per policy (so the caller can restore).
    Failures (typically EPERM without root) are logged in the returned dict
    as None values and the policy is left untouched.
    """
    previous: dict[str, str | None] = {}
    base = Path("/sys/devices/system/cpu/cpufreq")
    for policy in sorted(base.glob("policy*")):
        g = policy / "scaling_governor"
        if not g.exists():
            continue
        try:
            previous[policy.name] = g.read_text().strip()
        except OSError:
            previous[policy.name] = None
            continue
        try:
            g.write_text(name)
        except OSError:
            previous[policy.name] = None
    return previous


def restore_governors(previous: dict[str, str | None]) -> None:
    base = Path("/sys/devices/system/cpu/cpufreq")
    for name, gov in previous.items():
        if gov is None:
            continue
        f = base / name / "scaling_governor"
        if f.exists():
            try:
                f.write_text(gov)
            except OSError:
                pass


def capture() -> SystemFingerprint:
    governors = read_governors()
    governor = next(iter(set(governors.values())), None) if governors else None
    info = read_meminfo()
    ram_total_mb = info.get("MemTotal", 0) // 1024 if info else None
    return SystemFingerprint(
        kernel=platform.release(),
        machine=platform.machine(),
        python=platform.python_version(),
        ort_version=ort.__version__,
        available_eps=list(ort.get_available_providers()),
        os_release=_read_os_release(),
        governor=governor,
        governor_per_policy=governors,
        ram_total_mb=ram_total_mb,
    )


def as_dict(fp: SystemFingerprint) -> dict:
    return asdict(fp)
