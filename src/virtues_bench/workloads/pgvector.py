"""K8 — pgvector HNSW ANN p95 latency over 10k 768-dim vectors.

Populates the index once per board (cached to .cache/embeddings-10k-<sha>.npy),
builds HNSW (m=16, ef_construction=200), runs 1000 k=10 queries from a held-out
query set, reports p95.

Postgres prereq: `postgres-15` (or newer) + `pgvector` extension installed and
reachable. Default DSN is `postgresql:///bench` (peer auth, no password); set
PGUSER / PGPASSWORD / PGHOST or pass --dsn to override.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from virtues_bench.datasets import k8_corpus
from virtues_bench.models import fetch_and_verify
from virtues_bench.runners.onnx_session import (
    load_tokenizer,
    make_session,
    session_input_names,
)
from virtues_bench.timing import LatencyStats, measure

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / ".cache"
N_VECTORS = 10_000
N_QUERIES = 1000
QUERY_WARMUP = 50
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
EMBED_DIM = 768
EMBED_BATCH = 32


@dataclass
class PgvectorResult:
    k8_ann_p95_ms: float
    latency: dict
    n_vectors: int
    index_build_ms: float
    cache_hit: bool


def _embed_corpus(provider: str) -> tuple[np.ndarray, bool, str]:
    """Return (vectors[N, D], cache_hit, embed_revision)."""
    # Use int8 for K8 corpus generation — population is incidental setup work,
    # int8 is faster, vector quality is sufficient for ANN benchmarking.
    embed = fetch_and_verify("embed", "int8")
    cache_key = f"embeddings-{N_VECTORS}-{embed.revision[:12]}.npy"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return np.load(cache_path), True, embed.revision

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(embed.tokenizer_dir)
    sess = make_session(embed.onnx_path, provider)
    input_names = session_input_names(sess)

    strings = k8_corpus(n=N_VECTORS, target_words=200)
    out = np.empty((N_VECTORS, EMBED_DIM), dtype=np.float32)
    for i in range(0, N_VECTORS, EMBED_BATCH):
        batch = strings[i : i + EMBED_BATCH]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="np"
        )
        feeds = {k: v for k, v in enc.items() if k in input_names}
        outputs = sess.run(None, feeds)
        first = outputs[0]
        # bge-base ONNX exports vary: some emit last_hidden_state (B, T, D),
        # others emit the CLS-pooled sentence embedding directly (B, D).
        # Handle both.
        if first.ndim == 3:
            mask = enc["attention_mask"][..., None].astype(np.float32)
            pooled = (first * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1, None)
        elif first.ndim == 2:
            pooled = first
        else:
            raise RuntimeError(
                f"unexpected embed model output shape {first.shape}; expected 2D or 3D"
            )
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        out[i : i + len(batch)] = pooled / np.clip(norm, 1e-9, None)

    np.save(cache_path, out)
    return out, False, embed.revision


def run_pgvector(dsn: str, provider: str = "CPUExecutionProvider") -> PgvectorResult:
    vectors, cache_hit, _rev = _embed_corpus(provider)

    rng = np.random.default_rng(0xD15C)
    query_idx = rng.choice(N_VECTORS, size=N_QUERIES + QUERY_WARMUP, replace=True)
    queries = vectors[query_idx]

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS bench_vectors")
        conn.execute(
            f"CREATE TABLE bench_vectors (id integer PRIMARY KEY, embedding vector({EMBED_DIM}))"
        )
        with conn.cursor() as cur, cur.copy(
            "COPY bench_vectors (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as cp:
            cp.set_types(["int4", "vector"])
            for i, v in enumerate(vectors):
                cp.write_row([i, v])

        t0 = time.perf_counter_ns()
        conn.execute(
            f"CREATE INDEX ON bench_vectors USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )
        index_build_ms = (time.perf_counter_ns() - t0) / 1e6

        cur = conn.cursor()
        i = {"n": 0}

        def query_once() -> None:
            q = queries[i["n"]]
            i["n"] += 1
            cur.execute(
                "SELECT id FROM bench_vectors ORDER BY embedding <=> %s LIMIT 10", (q,)
            )
            cur.fetchall()

        latency = measure(query_once, n=N_QUERIES, warmup=QUERY_WARMUP)

    summary = asdict(latency)
    summary.pop("samples_ms")
    return PgvectorResult(
        k8_ann_p95_ms=latency.p95_ms,
        latency=summary,
        n_vectors=N_VECTORS,
        index_build_ms=index_build_ms,
        cache_hit=cache_hit,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn",
        default=os.environ.get("PGURL", "postgresql:///bench"),
        help="Postgres connection string",
    )
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()
    result = run_pgvector(args.dsn, args.provider)
    json.dump(asdict(result), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
