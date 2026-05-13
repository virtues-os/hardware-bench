"""K3 — embedding throughput (batch 32). K4 — embedding latency p95 (batch 1).

Runs once per (execution_provider × variant) combination.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

from virtues_bench.datasets import embed_inputs
from virtues_bench.models import fetch_and_verify
from virtues_bench.runners.onnx_session import (
    load_tokenizer,
    make_session,
    session_input_names,
)
from virtues_bench.timing import LatencyStats, measure

EMBED_N_STRINGS = 1000
EMBED_BATCH = 32
WARMUP_BATCHES = 5
LATENCY_N = 200
LATENCY_WARMUP = 20


@dataclass
class EmbedResult:
    provider: str
    variant: str
    k3_throughput_emb_per_sec: float
    k3_throughput_std_pct: float
    k4_p95_ms: float
    k4_latency: dict


def _run_batch(sess, tokenizer, strings: list[str]) -> None:
    enc = tokenizer(
        strings, padding=True, truncation=True, max_length=64, return_tensors="np"
    )
    feeds = {k: v for k, v in enc.items() if k in session_input_names(sess)}
    sess.run(None, feeds)


def run_embed(provider: str, variant: str = "fp32") -> EmbedResult:
    model = fetch_and_verify("embed", variant)
    tokenizer = load_tokenizer(model.tokenizer_dir)
    sess = make_session(model.onnx_path, provider)

    strings = embed_inputs(n=EMBED_N_STRINGS, target_words=15)
    batches = [strings[i : i + EMBED_BATCH] for i in range(0, len(strings), EMBED_BATCH)]

    for b in batches[:WARMUP_BATCHES]:
        _run_batch(sess, tokenizer, b)

    per_batch_ms: list[float] = []
    total_strings = 0
    t_start = time.perf_counter_ns()
    for b in batches:
        tb = time.perf_counter_ns()
        _run_batch(sess, tokenizer, b)
        per_batch_ms.append((time.perf_counter_ns() - tb) / 1e6)
        total_strings += len(b)
    total_s = (time.perf_counter_ns() - t_start) / 1e9
    throughput = total_strings / total_s if total_s > 0 else 0.0

    mean_batch_ms = sum(per_batch_ms) / len(per_batch_ms)
    var = sum((x - mean_batch_ms) ** 2 for x in per_batch_ms) / len(per_batch_ms)
    std_batch_ms = var**0.5
    throughput_std_pct = 100.0 * std_batch_ms / mean_batch_ms if mean_batch_ms > 0 else 0.0

    one = strings[0]
    latency = measure(
        lambda: _run_batch(sess, tokenizer, [one]),
        n=LATENCY_N,
        warmup=LATENCY_WARMUP,
    )

    return EmbedResult(
        provider=provider,
        variant=variant,
        k3_throughput_emb_per_sec=throughput,
        k3_throughput_std_pct=throughput_std_pct,
        k4_p95_ms=latency.p95_ms,
        k4_latency=_stats_summary(latency),
    )


def _stats_summary(s: LatencyStats) -> dict:
    d = asdict(s)
    d.pop("samples_ms")
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--variant", default="fp32", choices=("fp32", "int8"))
    args = parser.parse_args()
    result = run_embed(args.provider, args.variant)
    json.dump(asdict(result), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
