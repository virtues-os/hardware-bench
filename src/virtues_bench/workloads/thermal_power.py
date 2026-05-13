"""K5 — sustained max temp + throttle count. K6 — idle power. K7 — peak power.

60 sec idle measurement, then 30-min alternating embed+rerank loop. A side
thread polls thermal zones and hwmon power at 1 Hz. We track peak temp, throttle
events delta, idle wattage mean, and peak wattage during the loop.

External meter fallback: pass --idle-power-w and --peak-power-w to bypass sysfs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from virtues_bench.datasets import embed_inputs, rerank_inputs
from virtues_bench.models import fetch_and_verify
from virtues_bench.runners.onnx_session import (
    load_tokenizer,
    make_session,
    session_input_names,
)
from virtues_bench.telemetry import power, thermal

REPO_ROOT = Path(__file__).resolve().parents[3]
IDLE_DURATION_S = 60
LOAD_DURATION_S = 30 * 60
POLL_HZ = 1.0


@dataclass
class ThermalPowerResult:
    k5_max_temp_c: float | None
    k5_throttle: dict | None  # see thermal.throttle_summary
    k6_idle_power_w: float | None
    k7_peak_power_w: float | None
    thermal_samples: list[dict] = field(default_factory=list)
    power_samples: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _load_board(slug: str) -> dict:
    with (REPO_ROOT / "boards" / f"{slug}.toml").open("rb") as f:
        return tomllib.load(f)


class _Poller(threading.Thread):
    def __init__(self, thermal_zones: list[str], hwmon_paths: list[str]):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.thermal_zones = thermal_zones
        self.hwmon_paths = hwmon_paths
        self.thermal_samples: list[dict] = []
        self.power_samples: list[float] = []

    def run(self) -> None:
        interval = 1.0 / POLL_HZ
        while not self._stop.is_set():
            t = time.time()
            zones = thermal.read_zones(self.thermal_zones)
            if zones:
                self.thermal_samples.append({"t": t, "zones": zones})
            w = power.read_hwmon_watts(self.hwmon_paths)
            if w is not None:
                self.power_samples.append(w)
            self._stop.wait(interval)

    def stop(self) -> None:
        self._stop.set()


def _make_workloads(provider: str):
    # Thermal loop measures sustained load with the configuration prod will ship.
    embed = fetch_and_verify("embed", "int8")
    rerank = fetch_and_verify("rerank", "int8")
    e_tok = load_tokenizer(embed.tokenizer_dir)
    r_tok = load_tokenizer(rerank.tokenizer_dir)
    e_sess = make_session(embed.onnx_path, provider)
    r_sess = make_session(rerank.onnx_path, provider)
    embed_strings = embed_inputs(n=32, target_words=15)
    query, candidates = rerank_inputs()

    def do_embed() -> None:
        enc = e_tok(embed_strings, padding=True, truncation=True, max_length=64, return_tensors="np")
        feeds = {k: v for k, v in enc.items() if k in session_input_names(e_sess)}
        e_sess.run(None, feeds)

    def do_rerank() -> None:
        enc = r_tok([query] * len(candidates), candidates, padding=True,
                    truncation=True, max_length=256, return_tensors="np")
        feeds = {k: v for k, v in enc.items() if k in session_input_names(r_sess)}
        r_sess.run(None, feeds)

    return do_embed, do_rerank


def run_thermal_power(
    board_slug: str,
    provider: str = "CPUExecutionProvider",
    override_idle_w: float | None = None,
    override_peak_w: float | None = None,
    short_run_s: int | None = None,
) -> ThermalPowerResult:
    board = _load_board(board_slug)
    zones = board.get("thermal", {}).get("zones", []) or []
    hwmon = board.get("power", {}).get("hwmon_paths", []) or []
    notes: list[str] = []
    load_s = short_run_s if short_run_s is not None else LOAD_DURATION_S

    if not zones:
        notes.append(f"no thermal zones declared for board '{board_slug}'")
    if not hwmon:
        notes.append(f"no hwmon paths declared; K6/K7 fall back to CLI overrides")

    throttle_start = thermal.snapshot_time_in_state()

    # K6 — idle power measurement.
    idle_poller = _Poller(zones, hwmon)
    idle_poller.start()
    time.sleep(IDLE_DURATION_S)
    idle_poller.stop()
    idle_poller.join()
    idle_w = (
        statistics.fmean(idle_poller.power_samples)
        if idle_poller.power_samples
        else override_idle_w
    )

    # K5 + K7 — sustained load.
    do_embed, do_rerank = _make_workloads(provider)
    load_poller = _Poller(zones, hwmon)
    load_poller.start()
    t_end = time.time() + load_s
    toggle = 0
    while time.time() < t_end:
        (do_embed if toggle % 2 == 0 else do_rerank)()
        toggle += 1
    load_poller.stop()
    load_poller.join()

    throttle_end = thermal.snapshot_time_in_state()
    throttle = thermal.throttle_summary(throttle_start, throttle_end)

    max_temp: float | None = None
    for s in load_poller.thermal_samples:
        for v in s["zones"].values():
            max_temp = v if max_temp is None else max(max_temp, v)

    peak_w = (
        max(load_poller.power_samples)
        if load_poller.power_samples
        else override_peak_w
    )

    return ThermalPowerResult(
        k5_max_temp_c=max_temp,
        k5_throttle=throttle if throttle.get("available") else None,
        k6_idle_power_w=idle_w,
        k7_peak_power_w=peak_w,
        thermal_samples=load_poller.thermal_samples,
        power_samples=load_poller.power_samples,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--idle-power-w", type=float, default=None)
    parser.add_argument("--peak-power-w", type=float, default=None)
    parser.add_argument(
        "--short", type=int, default=None,
        help="Override load duration in seconds (smoke-testing only)",
    )
    args = parser.parse_args()
    result = run_thermal_power(
        args.board,
        args.provider,
        override_idle_w=args.idle_power_w,
        override_peak_w=args.peak_power_w,
        short_run_s=args.short,
    )
    json.dump(asdict(result), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
