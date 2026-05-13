"""K9 — RAM free under full load (MB).

Loads both models into ONNX Runtime sessions, holds a Postgres connection
open, runs a single ANN query against the bench_vectors table (populated by
the pgvector workload), then snapshots MemAvailable from /proc/meminfo.

Assumes the pgvector workload has already run on this board so the table
exists. If not, populates a tiny placeholder so the query path still
exercises the connection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from virtues_bench.models import fetch_and_verify
from virtues_bench.runners.onnx_session import load_tokenizer, make_session
from virtues_bench.telemetry.memory import mem_available_mb, read_meminfo

EMBED_DIM = 768


@dataclass
class MemoryResult:
    k9_ram_free_mb: int | None
    mem_total_mb: int | None
    swap_used_mb: int | None


def run_memory(dsn: str, provider: str = "CPUExecutionProvider") -> MemoryResult:
    # K9 measures RAM headroom in the configuration prod is most likely to ship.
    embed = fetch_and_verify("embed", "int8")
    rerank = fetch_and_verify("rerank", "int8")
    _e = make_session(embed.onnx_path, provider)
    _r = make_session(rerank.onnx_path, provider)
    _et = load_tokenizer(embed.tokenizer_dir)
    _rt = load_tokenizer(rerank.tokenizer_dir)

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        # If the pgvector workload hasn't populated, fall back to a single-row
        # ad-hoc table so the query still runs and the connection is real.
        row = conn.execute(
            "SELECT to_regclass('public.bench_vectors')"
        ).fetchone()
        if row is None or row[0] is None:
            conn.execute(
                f"CREATE TEMP TABLE bench_vectors (id int, embedding vector({EMBED_DIM}))"
            )
            conn.execute(
                "INSERT INTO bench_vectors VALUES (0, %s)",
                (np.zeros(EMBED_DIM, dtype=np.float32),),
            )
        q = np.random.default_rng(0xCAFE).standard_normal(EMBED_DIM).astype(np.float32)
        conn.execute(
            "SELECT id FROM bench_vectors ORDER BY embedding <=> %s LIMIT 10", (q,)
        ).fetchall()

        info = read_meminfo()
        return MemoryResult(
            k9_ram_free_mb=mem_available_mb(),
            mem_total_mb=info.get("MemTotal", 0) // 1024 if info else None,
            swap_used_mb=(
                (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) // 1024
                if info
                else None
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn", default=os.environ.get("PGURL", "postgresql:///bench")
    )
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()
    result = run_memory(args.dsn, args.provider)
    json.dump(asdict(result), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
