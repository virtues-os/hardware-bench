set shell := ["bash", "-cu"]

default:
    @just --list

bench board:
    uv run python -m virtues_bench.bench --board {{board}}

# Mac/local smoke test — validates Python paths without Postgres, thermal,
# or governor manipulation. Use before SSH'ing into real boards.
smoke board="mac-m5pro":
    uv run python -m virtues_bench.bench --board {{board}} \
        --skip-perf-pass --skip-thermal --skip-pgvector

validate-npu board:
    uv run python -m virtues_bench.workloads.validate_npu --board {{board}}

report:
    uv run python -m virtues_bench.report

fetch-models:
    uv run python -m virtues_bench.models --fetch
