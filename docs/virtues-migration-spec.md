# Virtues Core — Search Stack Migration Spec

**Handoff doc for Claude Code running in `virtues/`.** Replaces the existing fastembed-rs embedder + reranker pipeline with direct `ort` + `tokenizers` usage, switches the reranker model, and unlocks int8 + future NPU execution. Strategic decisions are locked in below; execution details are open.

This spec is the output of decisions made while building the public `hardware-bench` repo (which now measures the proposed stack end-to-end). Bench rationale lives in [hardware-bench/README.md](https://github.com/virtues/hardware-bench/blob/main/README.md). This doc covers the prod-side change.

---

## 1. Strategic context

Virtues is shipping V1 on a Qualcomm-based SBC (Radxa Dragon Q6A) with a 12 TOPS Hexagon NPU. Two constraints fall out of that:

- **The Hexagon NPU is the V1 acceleration story.** It only ships through ORT's QNN execution provider. fastembed-rs is a thin wrapper over `ort` but does not expose QNN configuration and does not support arbitrary EP selection. To use the NPU, virtues has to drop fastembed and call `ort` directly.
- **fp32 models are too slow on ARM SBC CPUs.** The reranker today (`rozgo/bge-reranker-v2-m3`, 568M params, fp32) takes hours per 1k-iter bench pass on these boards. Prod needs int8.

These constraints force two changes that compose well:

1. **Inference runtime:** fastembed-rs → direct `ort` + `tokenizers` crates.
2. **Model selection:** keep nomic embedder, switch reranker to `jinaai/jina-reranker-v2-base-multilingual` (half the params, comparable quality, ships a published int8 variant — the bge reranker has no published int8 and self-quantizing in Rust at startup is undesirable).

The combined change is roughly 100-200 lines of Rust touching `core/src/search/`. fastembed comes out of `Cargo.toml`. `ort` and `tokenizers` go in.

---

## 2. Decisions locked in

Do not relitigate these:

- **Runtime:** `ort` crate, 2.x major version. Same ORT engine fastembed wraps today. Used in prod by HF Text Embeddings Inference, Bloop, Google Magika, SurrealDB, Supabase Edge Functions.
- **Tokenization:** HF `tokenizers` crate (the Rust crate, same one fastembed uses internally).
- **Embedder model:** `nomic-ai/nomic-embed-text-v1.5` — file `onnx/model_quantized.onnx` (int8, 137 MB). Tokenizer files in the repo root.
- **Reranker model:** `jinaai/jina-reranker-v2-base-multilingual` — file `onnx/model_int8.onnx` (int8, 280 MB). Tokenizer files in the repo root.
- **Model delivery:** download from HuggingFace on first boot, cached to disk under `~/.cache/virtues/models/` (or platform-equivalent). No baking models into the binary. No custom HF org hosting. Use `hf-hub` crate (HF's official Rust client) for the download.
- **Quantization:** the bench will validate int8 quality against fp32 on actual V1 hardware before this lands. Default to int8 in prod once the bench confirms acceptable quality.
- **Execution providers:** CPU initially. QNN added as a feature flag once K0 gate passes on Q6A hardware. CoreML for Mac dev. No CUDA in V1.
- **Backwards compatibility:** no. Drop fastembed cleanly, change the model file paths, force a re-download on first boot of the new version. Personal corpora will need to re-index against the new embedder — note in the changelog.

---

## 3. What changes, what stays

### Stays the same

- The **search API surface** in `virtues/core/src/search/` — function signatures for embedding and reranking that callers depend on. Internal implementations change; public functions stay compatible.
- The **two-stage retrieval pipeline** — bi-encoder embedding → cross-encoder reranking. Same architecture, faster runtime.
- The **embedding dimension** — nomic-embed-text-v1.5 is 768-dim in both fp32 and int8 variants. Existing pgvector schema works unchanged once vectors are re-embedded.
- **Async behavior** — inference still runs on tokio's blocking thread pool. Don't await ORT calls directly on the async runtime.

### Changes

- **`Cargo.toml`:** remove `fastembed = "5"`. Add `ort = "2"` (with features for the EPs you target — start with `["download-binaries"]` for portable builds, add `"coreml"` for Mac dev), `tokenizers = "0.20"` (or current), `hf-hub = "0.4"` (or current — for model download).
- **`core/src/search/embedder.rs`:** rewrite around `ort::Session` + `tokenizers::Tokenizer`. Replace fastembed's `TextEmbedding::try_new(...)` with manual model loading and a `Session::builder()` chain. Mean-pool the model output mask-aware (nomic returns `(B, T, D)` `last_hidden_state`), L2-normalize.
- **`core/src/search/reranker.rs`:** rewrite around `ort::Session` + `tokenizers::Tokenizer`. Tokenize `(query, candidate)` pairs as paired sequences; pass through the ONNX session; extract the relevance logits from output `0`.
- **New module `core/src/search/model_cache.rs`:** owns the first-boot HF download + cache directory layout + SHA verification. Pattern after `hardware-bench/src/virtues_bench/models.py` (which does the same job in Python).

---

## 4. Implementation sequence

Order matters. Each step is independently shippable and reviewable.

1. **Add `ort`, `tokenizers`, `hf-hub` to `Cargo.toml`.** Do NOT remove `fastembed` yet — keep the old code working while you build the replacement.
2. **Add `core/src/search/model_cache.rs`** with: cache directory resolution, HF download via `hf-hub`, SHA-256 verification against pinned values, returns `(onnx_path, tokenizer_dir)` for each model. Pin hashes by copying from `hardware-bench/models.toml` after that repo's first end-to-end run. Mirror the constants from there so prod and bench load byte-identical files.
3. **Add `core/src/search/ort_session.rs`** with: helpers to build a `Session` from an ONNX path + a list of EPs to try (in order; first one that initializes wins), tokenizer loading from the cache dir. Keep EP selection in one place so adding QNN later is a one-line change.
4. **Replace `embedder.rs`** to use the new helpers. Wire `NomicEmbedTextV15Q` model file. Implement mean-pooling + L2-normalize identically to how fastembed did it (verify by comparing a few sample embeddings against fastembed output before deleting the old path). Keep the public function signature the caller already uses.
5. **Replace `reranker.rs`** to use the new helpers. Wire jina-reranker-v2 int8. Verify a few sample (query, candidate) score outputs against the bench's CPU reference (the bench's `validate_npu.py` produces these).
6. **Remove `fastembed` from `Cargo.toml`** and delete the old code paths.
7. **Add a feature flag `qnn`** that adds `"qnn"` to the `ort` crate features and includes `QNNExecutionProvider` in the EP list. Default off; flip on for Q6A builds once `hardware-bench`'s K0 gate passes.
8. **Document the user-visible change** in the changelog: re-indexing required for existing corpora, new model files downloaded on first boot, behavior should otherwise be identical.

---

## 5. Code patterns to follow

### Session construction (ort 2.x)

```rust
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::SessionBuilder, Session},
    value::Value,
};

fn build_session(onnx_path: &Path) -> ort::Result<Session> {
    Session::builder()?
        .with_execution_providers([
            // QNN first when feature enabled, falls back to CPU.
            #[cfg(feature = "qnn")]
            ort::execution_providers::QNNExecutionProvider::default().build(),
            #[cfg(target_os = "macos")]
            ort::execution_providers::CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])?
        .with_intra_threads(num_cpus::get_physical())?
        .commit_from_file(onnx_path)
}
```

### Tokenization (`tokenizers` 0.20+)

```rust
let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_dir.join("tokenizer.json"))?;
let encoding = tokenizer.encode_batch(strings, true)?;
let input_ids: ndarray::Array2<i64> = ...;  // shape (B, T)
let attention_mask: ndarray::Array2<i64> = ...;
```

### Reranker forward pass

Cross-encoders tokenize `(query, candidate)` as a paired sequence. Use `Tokenizer::encode_batch` with `(String, String)` pairs. The model's first output is a `(B, 1)` logit tensor; flatten it and sort the candidates by score.

### Inference call

```rust
let outputs = session.run(ort::inputs![
    "input_ids" => input_ids.view(),
    "attention_mask" => attention_mask.view(),
]?)?;
let last_hidden: ndarray::ArrayView<f32, _> = outputs["last_hidden_state"].try_extract_tensor()?;
```

### Mean pooling (embedder)

Mask-aware mean over the token dimension, then L2-normalize. Match the reference implementation in [hardware-bench/src/virtues_bench/workloads/pgvector.py](https://github.com/virtues/hardware-bench/blob/main/src/virtues_bench/workloads/pgvector.py) (`_embed_corpus` function) to ensure byte-compatible embeddings — pgvector indices built with the bench's vectors should match prod's.

### Thread safety

`ort::Session` is `Send + Sync`. Wrap one instance per model in an `Arc` and share across worker tasks. Don't construct sessions per-request — initialization cost is ~100ms.

---

## 6. Open questions for Adam

- **Cache directory.** `~/.cache/virtues/models/` on Linux/Mac, what on Windows? Or just always `directories::ProjectDirs::from("com", "virtues", "virtues").cache_dir()`? Recommend the latter for portability.
- **HF auth.** Both models are open. No `HF_TOKEN` needed today. Should we document the env-var pattern for future gated models? Recommend yes, in the README, no code logic.
- **EP fallback policy.** If QNN init fails on Q6A at runtime (driver issue, etc.), should we fall back silently to CPU or hard-fail with a clear error? Recommend fall back + log a warning — a working-but-slow search is better than a broken server. Make this visible in `/health` so ops can spot it.
- **Re-index migration.** Existing user corpora are indexed against fastembed's nomic-embed fp32 vectors. After this migration they'd need re-embedding against int8 vectors (the dimensions match, but the values won't be byte-identical because int8 ≠ fp32). Options: (a) auto-re-embed on first boot of the new version (slow but seamless), (b) require a manual `virtues reindex` command, (c) keep fp32 in prod until corpus migration tooling exists. Recommend (a) for V1 since corpora are still small.
- **Backwards-compat shim.** Should we keep a "legacy fastembed" feature flag for one release? Recommend no — clean break, fewer code paths, the search semantics shouldn't change observably.

---

## 7. Validation criteria

The migration is done when:

- `cargo build` works without `fastembed` in `Cargo.toml`.
- `cargo test` passes — including search integration tests.
- A handful of embeddings and rerank scores match the bench's CPU reference within numerical tolerance (cosine sim > 0.999 on embeddings, abs diff < 0.05 on rerank logits). The bench's `validate_npu.py` produces these references.
- First-boot model download works on a clean cache, completes in under 60 seconds on consumer internet.
- Verify on the Q6A: with `qnn` feature off, search functions on CPU. With `qnn` feature on (and `hardware-bench`'s K0 gate passing), search functions on NPU.

---

## 8. Anti-goals (do not build)

- **No fastembed compatibility shim.** Clean break.
- **No custom model hosting.** Both models have published int8 ONNX directly on the original HF repos.
- **No runtime quantization.** Both int8 variants are pre-quantized; don't add `ort` quantization tooling to the Rust build.
- **No GGUF / llama.cpp.** ONNX is the only format with a real NPU story across Qualcomm / Intel / Apple / NVIDIA. Don't be tempted.
- **No new model registry abstraction.** Two hardcoded models in a `model_cache.rs` module is enough. Don't rebuild fastembed.
- **No batching middleware (yet).** Single-request inference is fine for V1's request rate. Revisit when throughput becomes a constraint.

---

## 9. Where the receipts live

- **Bench measuring this stack:** [github.com/virtues/hardware-bench](https://github.com/virtues/hardware-bench) — public, MIT, models.toml pins exact files, results/ committed per board.
- **Bench rationale and model decisions:** [hardware-bench/README.md](https://github.com/virtues/hardware-bench/blob/main/README.md) (sections "Models", "Two-pass methodology").
- **Why ort over alternatives:** the research log in this repo's PR description should cite the comparison (candle / burn / TEI / mistral.rs all lack Hexagon NPU support; fastembed-rs blocks QNN configuration).

---

## 10. First commits to aim for

1. `Cargo.toml` — add `ort`, `tokenizers`, `hf-hub`; keep `fastembed` for now.
2. `core/src/search/model_cache.rs` with HF download + SHA pin verification.
3. `core/src/search/ort_session.rs` with EP-aware session builder.
4. `core/src/search/embedder.rs` rewrite (parallel to existing fastembed path; feature-flagged switch).
5. `core/src/search/reranker.rs` rewrite (same pattern).
6. Validation test cases comparing new path to old on a handful of inputs.
7. Remove `fastembed` from `Cargo.toml`; delete old code.
8. Add `qnn` feature flag and EP wiring for Q6A.

Commit early, commit often. Each step is independently reviewable.
