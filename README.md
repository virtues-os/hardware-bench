# hardware-bench

Benchmark suite for evaluating single-board computers as the V1 platform for [Virtues](https://virtues.com). Results at [bench.virtues.com](https://bench.virtues.com) (TBD).

## What it measures

10 KPIs, same definition on every board.

| | KPI | Notes |
|---|---|---|
| K0 | NPU toolchain works | Binary gate. QNN EP loads the reranker and matches CPU output to cosine sim > 0.999. |
| K1 | Reranker p95 latency (ms) | 50 candidates × ~200 tokens, per EP × variant |
| K2 | Reranker p99 latency (ms) | tail |
| K3 | Embedding throughput (emb/sec) | batch 32, 1000 strings |
| K4 | Embedding p95 latency (ms) | batch 1 |
| K5 | Sustained max temp (°C) + below-max time (s) | 30-min mixed embed+rerank loop |
| K6 | Idle power (W) | 60-sec idle window |
| K7 | Peak power (W) | during sustained load |
| K8 | pgvector ANN p95 (ms) | 10k 768-dim, HNSW (m=16, ef_construction=200), k=10 |
| K9 | RAM free under load (MB) | both models + Postgres + one query |

200 samples per percentile KPI, first 20 discarded as warmup. Std dev reported alongside.

Each board runs **two passes**: 1a uses the as-shipped governor (`schedutil` on modern Debian), 1b forces `performance`. Both land in one JSON.

## Models

Both int8, downloaded from HF, SHA-pinned in [models.toml](models.toml):

- Embedder: `nomic-ai/nomic-embed-text-v1.5` → `onnx/model_quantized.onnx`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual` → `onnx/model_int8.onnx`

To bench fp32 alongside int8, flip `BENCH_VARIANTS` in [src/virtues_bench/bench.py](src/virtues_bench/bench.py).

## Run

Prereqs: Python 3.11+, [uv](https://github.com/astral-sh/uv), [just](https://github.com/casey/just), Postgres 15+ with pgvector.

```bash
# Postgres (macOS)
brew install postgresql@15 pgvector
brew services start postgresql@15
createdb bench
psql bench -c "CREATE EXTENSION vector;"

# Postgres (Debian/Ubuntu)
sudo apt install postgresql postgresql-16-pgvector
sudo -u postgres createdb bench
sudo -u postgres psql bench -c "CREATE EXTENSION vector;"
sudo -u postgres createuser --superuser "$USER"

# Bench
uv sync
just bench <board-slug>          # full ~90 min two-pass
just smoke mac-m5pro             # quick dev-machine smoke (no Postgres, no thermal)
just validate-npu dragon-q6a     # K0 gate only
just report                      # results/*.json → RESULTS.md
```

Each run writes `results/<board>-<timestamp>.json`.

## Boards

- [orange-pi-5-plus](boards/orange-pi-5-plus.toml) — RK3588, dev target
- [dragon-q6a](boards/dragon-q6a.toml) — Qualcomm QCS6490 + Hexagon NPU, V1 candidate. Needs QAIRT SDK and the `onnxruntime-qnn` wheel; see [docs/q6a-setup.md](docs/q6a-setup.md) (TBD).
- [mac-m5pro](boards/mac-m5pro.toml) — local dev smoke target

### Add a board

1. Create `boards/<slug>.toml` (copy an existing one).
2. Discover sysfs paths on the board:
   ```bash
   for z in /sys/class/thermal/thermal_zone*/type; do echo "$z -> $(cat $z)"; done
   ls /sys/class/hwmon/
   ```
3. Fill in `[thermal].zones`, `[power].hwmon_paths`, and the top-level fields.
4. `just bench <slug>`.

## Layout

```
hardware-bench/
├── pyproject.toml           # uv-managed
├── Justfile                 # entrypoints
├── models.toml              # HF refs + SHA pins
├── boards/                  # one .toml per target
├── src/virtues_bench/
│   ├── bench.py             # orchestrator
│   ├── report.py            # results/ → RESULTS.md
│   ├── models.py            # fetch + SHA verify
│   ├── datasets.py          # deterministic synthetic inputs
│   ├── preflight.py         # validate env before workloads run
│   ├── telemetry/           # thermal, power, memory readers
│   ├── runners/             # ORT session helpers
│   └── workloads/           # one file per KPI group
├── results/                 # JSON per (board, timestamp), committed
├── web/                     # SvelteKit static site (deploys to Vercel)
└── docs/
```

## Why Python

The harness loads models, times inference, reads sysfs. The inference ecosystem is Python-native; a Rust port wouldn't buy anything. Virtues itself stays Rust — see [docs/virtues-migration-spec.md](docs/virtues-migration-spec.md).

## License

MIT.
