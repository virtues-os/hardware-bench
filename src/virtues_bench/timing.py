"""Latency timing utilities — perf_counter_ns based, with warmup + percentile reporting."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]


def measure(fn: Callable[[], None], n: int, warmup: int = 50) -> LatencyStats:
    """Run `fn` n+warmup times, discard warmup, return percentile stats.

    Caller is responsible for keeping per-iteration inputs identical so that
    the only variable being measured is hardware/EP behavior.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return _stats(samples)


def _stats(samples_ms: list[float]) -> LatencyStats:
    s = sorted(samples_ms)
    n = len(s)

    def pct(p: float) -> float:
        if n == 0:
            return 0.0
        k = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return s[k]

    return LatencyStats(
        n=n,
        mean_ms=statistics.fmean(s) if n else 0.0,
        std_ms=statistics.pstdev(s) if n > 1 else 0.0,
        p50_ms=pct(50),
        p95_ms=pct(95),
        p99_ms=pct(99),
        min_ms=min(s) if s else 0.0,
        max_ms=max(s) if s else 0.0,
        samples_ms=samples_ms,
    )
