"""Thin wrapper around onnxruntime InferenceSession with EP selection."""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort
from transformers import AutoTokenizer


def make_session(model_path: Path, provider: str) -> ort.InferenceSession:
    """Build an InferenceSession with EP-appropriate options.

    For QNNExecutionProvider we set `session.disable_cpu_ep_fallback=1` so
    the session hard-fails if any node can't run on the NPU — without this,
    ORT silently falls back to CPU and we'd be measuring CPU latency while
    claiming NPU numbers. Also passes `backend_path=libQnnHtp.so` which the
    QAIRT runtime resolves via LD_LIBRARY_PATH (set by `bin/envsetup.sh`).
    """
    sess_options = ort.SessionOptions()
    provider_options: list[dict] | None = None
    if provider == "QNNExecutionProvider":
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        provider_options = [{"backend_path": "libQnnHtp.so"}]
    if provider_options is not None:
        return ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=[provider],
            provider_options=provider_options,
        )
    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=[provider])


def load_tokenizer(tokenizer_dir: Path):
    """Load tokenizer from a directory containing tokenizer.json + config.

    `trust_remote_code=True` is required for models like Jina's reranker which
    ship a custom tokenizer class. The remote code is already on local disk
    (HF cache) and the model is SHA-pinned in models.toml; we're not
    executing arbitrary internet code at runtime.

    Future cleanup: switch to the `tokenizers` library directly (loads only
    tokenizer.json, no Python eval) — matches what virtues/ Rust uses via
    the `tokenizers` crate.
    """
    return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)


def session_input_names(sess: ort.InferenceSession) -> set[str]:
    return {i.name for i in sess.get_inputs()}
