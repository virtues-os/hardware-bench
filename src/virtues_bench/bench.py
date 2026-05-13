"""Orchestrator — composes all workloads into a single per-board result JSON.

Runs two passes per board: 1a (as-shipped governor) and 1b (performance forced,
restored on exit). Each pass runs the full KPI suite. Writes one JSON file:

    results/<board>-<timestamp>.json

Workload failures are caught and recorded as nulls so a single broken probe
doesn't take down the whole run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from virtues_bench import __version__, preflight, system
from virtues_bench.workloads import embed as embed_wl
from virtues_bench.workloads import memory as memory_wl
from virtues_bench.workloads import pgvector as pgvector_wl
from virtues_bench.workloads import rerank as rerank_wl
from virtues_bench.workloads import thermal_power as tp_wl
from virtues_bench.workloads import validate_npu as npu_wl

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def _load_board(slug: str) -> dict:
    with (REPO_ROOT / "boards" / f"{slug}.toml").open("rb") as f:
        return tomllib.load(f)


def _safe(fn, *args, **kwargs):
    """Run a workload, return (result_dict, error_str). Never raise."""
    try:
        result = fn(*args, **kwargs)
        return asdict(result), None
    except Exception:
        return None, traceback.format_exc()


def _run_pass(
    board: dict,
    pass_name: str,
    short_thermal_s: int | None,
    dsn: str,
    skip_thermal: bool,
    skip_pgvector: bool = False,
) -> dict:
    eps = board.get("execution_providers", ["CPUExecutionProvider"])
    has_qnn = "QNNExecutionProvider" in eps
    slug = board["slug"]
    out: dict = {"governors": system.read_governors()}

    if has_qnn:
        out["k0_npu_gate"], err = _safe(npu_wl.run_gate, slug)
        if err:
            out["k0_npu_gate_error"] = err
    else:
        out["k0_npu_gate"] = None

    out["embed"] = {}
    out["rerank"] = {}
    # V1 benches int8 only (matches the proposed prod stack). To also measure
    # fp32, extend BENCH_VARIANTS to ("fp32", "int8") — the workloads already
    # accept a variant arg.
    BENCH_VARIANTS = ("int8",)
    for ep in eps:
        out["embed"][ep] = {}
        out["rerank"][ep] = {}
        for variant in BENCH_VARIANTS:
            r, err = _safe(embed_wl.run_embed, ep, variant)
            out["embed"][ep][variant] = r or {"error": err}
            r, err = _safe(rerank_wl.run_rerank, ep, variant)
            out["rerank"][ep][variant] = r or {"error": err}

    if skip_pgvector:
        out["pgvector"] = None
    else:
        out["pgvector"], err = _safe(pgvector_wl.run_pgvector, dsn)
        if err:
            out["pgvector_error"] = err

    if skip_thermal:
        out["thermal_power"] = None
    else:
        out["thermal_power"], err = _safe(
            tp_wl.run_thermal_power, slug, "CPUExecutionProvider", None, None, short_thermal_s
        )
        if err:
            out["thermal_power_error"] = err

    if skip_pgvector:
        out["memory"] = None
    else:
        out["memory"], err = _safe(memory_wl.run_memory, dsn)
        if err:
            out["memory_error"] = err

    return out


def run(
    board_slug: str,
    dsn: str,
    skip_perf_pass: bool = False,
    short_thermal_s: int | None = None,
    skip_thermal: bool = False,
    skip_pgvector: bool = False,
    force: bool = False,
) -> dict:
    board = _load_board(board_slug)

    print("[bench] preflight checks...", file=sys.stderr)
    checks = preflight.run_preflight(
        board, dsn, skip_pgvector, skip_perf_pass, skip_thermal
    )
    failed = [m for ok, m in checks if not ok]
    if failed and not force:
        print(
            f"[bench] {len(failed)} preflight check(s) failed; aborting "
            f"(use --force to override):\n  " + "\n  ".join(failed),
            file=sys.stderr,
        )
        sys.exit(2)

    fingerprint = system.capture()
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    print(f"[bench] pass 1a (as-shipped governor: {fingerprint.governor})...", file=sys.stderr)
    pass_1a = _run_pass(board, "1a", short_thermal_s, dsn, skip_thermal, skip_pgvector)

    pass_1b: dict | None = None
    if not skip_perf_pass:
        print(f"[bench] pass 1b (forcing performance governor)...", file=sys.stderr)
        previous = system.set_governor("performance")
        try:
            pass_1b = _run_pass(board, "1b", short_thermal_s, dsn, skip_thermal, skip_pgvector)
        finally:
            system.restore_governors(previous)
            print(f"[bench] restored governors", file=sys.stderr)

    elapsed_s = time.perf_counter() - t0
    return {
        "harness_version": __version__,
        "board": board_slug,
        "board_config": board,
        "started_at": started,
        "elapsed_s": elapsed_s,
        "system": system.as_dict(fingerprint),
        "pass_1a_stock": pass_1a,
        "pass_1b_performance": pass_1b,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, help="board slug, e.g. dragon-q6a")
    parser.add_argument(
        "--dsn", default="postgresql:///bench", help="Postgres DSN for K8/K9"
    )
    parser.add_argument(
        "--skip-perf-pass",
        action="store_true",
        help="Run only pass 1a (skip the performance-governor pass)",
    )
    parser.add_argument(
        "--short-thermal",
        type=int,
        default=None,
        help="Override thermal loop duration in seconds (smoke testing)",
    )
    parser.add_argument(
        "--skip-thermal",
        action="store_true",
        help="Skip the thermal/power loop entirely (smoke testing)",
    )
    parser.add_argument(
        "--skip-pgvector",
        action="store_true",
        help="Skip pgvector and memory workloads (no Postgres required)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if preflight checks fail",
    )
    args = parser.parse_args()

    result = run(
        args.board,
        args.dsn,
        skip_perf_pass=args.skip_perf_pass,
        short_thermal_s=args.short_thermal,
        skip_thermal=args.skip_thermal,
        skip_pgvector=args.skip_pgvector,
        force=args.force,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{args.board}-{ts}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[bench] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
