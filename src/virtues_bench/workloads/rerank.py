"""K1 — reranker p95 latency. K2 — reranker p99 latency.

Each timed iteration reranks 50 candidates × ~200 tokens against a single
query. Tokenization is included in the timed section.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

from virtues_bench.datasets import rerank_inputs
from virtues_bench.models import fetch_and_verify
from virtues_bench.runners.onnx_session import (
    load_tokenizer,
    make_session,
    session_input_names,
)
from virtues_bench.timing import measure

RERANK_N = 200
RERANK_WARMUP = 20


@dataclass
class RerankResult:
    provider: str
    variant: str
    k1_p95_ms: float
    k2_p99_ms: float
    latency: dict


def _run_rerank(sess, tokenizer, query: str, candidates: list[str]) -> None:
    queries = [query] * len(candidates)
    enc = tokenizer(
        queries, candidates,
        padding=True, truncation=True, max_length=256, return_tensors="np",
    )
    feeds = {k: v for k, v in enc.items() if k in session_input_names(sess)}
    sess.run(None, feeds)


def run_rerank(provider: str, variant: str = "fp32") -> RerankResult:
    model = fetch_and_verify("rerank", variant)
    tokenizer = load_tokenizer(model.tokenizer_dir)
    sess = make_session(model.onnx_path, provider)

    query, candidates = rerank_inputs()
    latency = measure(
        lambda: _run_rerank(sess, tokenizer, query, candidates),
        n=RERANK_N,
        warmup=RERANK_WARMUP,
    )
    summary = asdict(latency)
    summary.pop("samples_ms")

    return RerankResult(
        provider=provider,
        variant=variant,
        k1_p95_ms=latency.p95_ms,
        k2_p99_ms=latency.p99_ms,
        latency=summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--variant", default="fp32", choices=("fp32", "int8"))
    args = parser.parse_args()
    result = run_rerank(args.provider, args.variant)
    json.dump(asdict(result), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
