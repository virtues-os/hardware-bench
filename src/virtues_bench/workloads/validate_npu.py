"""K0 — NPU toolchain gate.

Binary pass/fail check for the Dragon Q6A's QNN execution provider. Loads
bge-reranker-base via ONNX Runtime on both CPU (reference) and QNN (target),
runs identical inputs through both, and compares output logits.

Pass criterion: cosine similarity > 0.999 across all sample (query, candidate)
pairs. Anything less means the QNN path is producing meaningfully different
numbers, and any latency wins are moot.

If this gate fails, the rest of the NPU KPIs (K1.npu, K2.npu, K3.npu, K4.npu)
are skipped on the Q6A and the board's NPU pitch is marked unrealized.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from virtues_bench.models import fetch_and_verify
from virtues_bench.runners.onnx_session import load_tokenizer, make_session, session_input_names

REPO_ROOT = Path(__file__).resolve().parents[3]
COSINE_THRESHOLD = 0.999

# Deterministic sample inputs — same on every board.
SAMPLE_PAIRS: list[tuple[str, str]] = [
    ("what is the capital of France", "Paris is the capital and most populous city of France."),
    ("what is the capital of France", "Bananas are a popular tropical fruit grown in many regions."),
    ("how do I reset my password", "Click 'forgot password' on the login screen and follow the email link."),
    ("how do I reset my password", "The history of the Roman Empire spans over a thousand years."),
    ("symptoms of vitamin D deficiency", "Common symptoms include fatigue, bone pain, and muscle weakness."),
    ("symptoms of vitamin D deficiency", "Vitamin C is found in citrus fruits and supports immune function."),
    ("python list comprehension example", "A list comprehension: [x*2 for x in range(10) if x % 2 == 0]"),
    ("python list comprehension example", "Java requires explicit type declarations for all variables."),
]


@dataclass
class GateResult:
    passed: bool
    cosine_min: float
    cosine_mean: float
    cpu_logits: list[float]
    npu_logits: list[float]
    error: str | None = None


def _load_board(slug: str) -> dict:
    path = REPO_ROOT / "boards" / f"{slug}.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def _run_session(model_path: Path, tokenizer, provider: str) -> np.ndarray:
    sess = make_session(model_path, provider)
    queries = [q for q, _ in SAMPLE_PAIRS]
    candidates = [c for _, c in SAMPLE_PAIRS]
    enc = tokenizer(
        queries,
        candidates,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="np",
    )
    feeds = {k: v for k, v in enc.items() if k in session_input_names(sess)}
    outputs = sess.run(None, feeds)
    return np.asarray(outputs[0]).reshape(-1)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_gate(board_slug: str) -> GateResult:
    board = _load_board(board_slug)
    providers = board.get("execution_providers", [])
    if "QNNExecutionProvider" not in providers:
        return GateResult(
            passed=False,
            cosine_min=0.0,
            cosine_mean=0.0,
            cpu_logits=[],
            npu_logits=[],
            error=f"board '{board_slug}' does not declare QNNExecutionProvider",
        )

    rerank = fetch_and_verify("rerank", "int8")  # NPU pitch is int8 throughput
    tokenizer = load_tokenizer(rerank.tokenizer_dir)

    cpu_logits = _run_session(rerank.onnx_path, tokenizer, "CPUExecutionProvider")
    try:
        npu_logits = _run_session(rerank.onnx_path, tokenizer, "QNNExecutionProvider")
    except Exception as exc:  # broad on purpose — QNN failures take many forms
        return GateResult(
            passed=False,
            cosine_min=0.0,
            cosine_mean=0.0,
            cpu_logits=cpu_logits.tolist(),
            npu_logits=[],
            error=f"QNN session failed: {exc!r}",
        )

    # Pair-wise cosine across each sample's scalar logit is undefined; instead
    # compare the full vector of logits as a single distribution.
    cos_full = _cosine(cpu_logits, npu_logits)
    diffs = np.abs(cpu_logits - npu_logits) / (np.abs(cpu_logits) + 1e-9)
    return GateResult(
        passed=cos_full > COSINE_THRESHOLD,
        cosine_min=cos_full,  # single vector pair, min == mean == cos_full
        cosine_mean=cos_full,
        cpu_logits=cpu_logits.tolist(),
        npu_logits=npu_logits.tolist(),
        error=None if cos_full > COSINE_THRESHOLD else f"max rel diff {float(diffs.max()):.4f}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True, help="Board slug, e.g. dragon-q6a")
    args = parser.parse_args()
    result = run_gate(args.board)
    json.dump(asdict(result), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
