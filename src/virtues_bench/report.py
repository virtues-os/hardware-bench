"""Read results/*.json, emit RESULTS.md with side-by-side KPI comparisons.

Markdown only — no plotting. One table per KPI, columns are (board, pass, EP)
tuples sorted by recency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = REPO_ROOT / "RESULTS.md"


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _column_label(board: str, pass_name: str, ep: str | None = None) -> str:
    ep_short = (ep or "").replace("ExecutionProvider", "") or "—"
    return f"{board} / {pass_name}" + (f" / {ep_short}" if ep else "")


def _collect(runs: list[dict]) -> list[tuple[str, dict]]:
    """Return [(column_label, pass_dict), ...] for both passes of every run."""
    columns: list[tuple[str, dict]] = []
    for run in runs:
        board = run["board"]
        for pass_key, pass_label in (
            ("pass_1a_stock", "1a"),
            ("pass_1b_performance", "1b"),
        ):
            p = run.get(pass_key)
            if p is None:
                continue
            columns.append((_column_label(board, pass_label), p))
    return columns


def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return f"### {title}\n\n_no data_\n"
    head = "| KPI | " + " | ".join(headers) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"### {title}\n\n{head}\n{sep}\n{body}\n"


def build_markdown(runs: list[dict]) -> str:
    columns = _collect(runs)
    if not columns:
        return "# Hardware Bench Results\n\n_no completed runs yet_\n"
    headers = [label for label, _ in columns]
    rows: list[list[str]] = []

    def add(kpi: str, extractor) -> None:
        rows.append([kpi] + [_fmt(extractor(p)) for _, p in columns])

    add("K0 — NPU gate passed", lambda p: (p.get("k0_npu_gate") or {}).get("passed"))

    # V1 bench reports int8 only. If fp32 is re-enabled in bench.py
    # (BENCH_VARIANTS), add it to REPORT_VARIANTS here.
    REPORT_VARIANTS = ("int8",)
    for ep_short, ep_full in (("CPU", "CPUExecutionProvider"), ("QNN", "QNNExecutionProvider")):
        for var in REPORT_VARIANTS:
            add(
                f"K1 — Rerank p95 ms ({ep_short}/{var})",
                lambda p, e=ep_full, v=var: ((p.get("rerank", {}).get(e) or {}).get(v) or {}).get("k1_p95_ms"),
            )
            add(
                f"K2 — Rerank p99 ms ({ep_short}/{var})",
                lambda p, e=ep_full, v=var: ((p.get("rerank", {}).get(e) or {}).get(v) or {}).get("k2_p99_ms"),
            )
            add(
                f"K3 — Embed throughput emb/sec ({ep_short}/{var})",
                lambda p, e=ep_full, v=var: ((p.get("embed", {}).get(e) or {}).get(v) or {}).get("k3_throughput_emb_per_sec"),
            )
            add(
                f"K4 — Embed p95 ms batch=1 ({ep_short}/{var})",
                lambda p, e=ep_full, v=var: ((p.get("embed", {}).get(e) or {}).get(v) or {}).get("k4_p95_ms"),
            )

    add("K5 — Max temp °C", lambda p: (p.get("thermal_power") or {}).get("k5_max_temp_c"))
    add(
        "K5 — Below-max time (s)",
        lambda p: ((p.get("thermal_power") or {}).get("k5_throttle") or {}).get("approx_below_max_s"),
    )
    add("K6 — Idle power W", lambda p: (p.get("thermal_power") or {}).get("k6_idle_power_w"))
    add("K7 — Peak power W", lambda p: (p.get("thermal_power") or {}).get("k7_peak_power_w"))
    add("K8 — pgvector ANN p95 ms", lambda p: (p.get("pgvector") or {}).get("k8_ann_p95_ms"))
    add("K9 — RAM free MB", lambda p: (p.get("memory") or {}).get("k9_ram_free_mb"))

    md = ["# Hardware Bench Results", "", "_Generated from `results/*.json`. Run `just report` to regenerate._", ""]
    md.append("## Run summary\n")
    md.append("| Run | Board | Started | OS | Governor (1a) | Elapsed |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(runs):
        os_name = (r.get("system", {}).get("os_release") or {}).get("PRETTY_NAME", "—")
        md.append(
            f"| {i + 1} | {r['board']} | {r['started_at']} | {os_name} | "
            f"{r.get('system', {}).get('governor', '—')} | "
            f"{r.get('elapsed_s', 0):.0f}s |"
        )
    md.append("")
    md.append("## KPIs\n")
    md.append(_table("All KPIs (lower is better for latency, higher for throughput)", headers, rows))
    return "\n".join(md) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    if not RESULTS_DIR.exists():
        print(f"no results dir at {RESULTS_DIR}", file=sys.stderr)
        return 1
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with path.open() as f:
            runs.append(json.load(f))
    md = build_markdown(runs)
    Path(args.out).write_text(md)
    print(f"wrote {args.out} ({len(runs)} runs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
