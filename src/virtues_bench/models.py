"""Model fetcher and SHA256 integrity gate.

Each top-level model section in models.toml declares:
  - `repo`, `revision` — HF repo and pinned commit (revision filled on first
    fetch, then verified on every subsequent run)
  - `shared_files` — tokenizer / config files used by all variants
  - `files_fp32`, `files_int8` — variant-specific ONNX files (both published
    on HF; the harness never quantizes anything itself)

`fetch_and_verify(name, variant)` returns the path to the ONNX file for the
requested variant plus a directory containing tokenizer/config files.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from huggingface_hub import hf_hub_download, model_info

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_TOML = REPO_ROOT / "models.toml"
TODO = "TODO"
VARIANTS = ("fp32", "int8")


@dataclass(frozen=True)
class LoadedModel:
    name: str
    variant: str
    repo: str
    revision: str
    onnx_path: Path
    tokenizer_dir: Path  # directory containing tokenizer.json + config files


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_doc() -> tomlkit.TOMLDocument:
    with MODELS_TOML.open("r", encoding="utf-8") as f:
        return tomlkit.parse(f.read())


def _save_doc(doc: tomlkit.TOMLDocument) -> None:
    with MODELS_TOML.open("w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))


def _ensure_revision(section, repo: str, name: str) -> str:
    revision = section["revision"]
    if revision == TODO:
        info = model_info(repo, revision="main")
        revision = info.sha
        section["revision"] = revision
        print(f"[models] pinned {name} revision -> {revision}", file=sys.stderr)
    return revision


def _download_files(
    repo: str, revision: str, filenames: list[str]
) -> dict[str, Path]:
    return {
        fn: Path(hf_hub_download(repo_id=repo, filename=fn, revision=revision))
        for fn in filenames
    }


def _verify_or_seed_hashes(
    section,
    paths: dict[str, Path],
    model_name: str,
) -> list[str]:
    """Hash each file, seed TODOs or verify pins. Returns mismatch messages."""
    sha_table = section["sha256"]
    mismatches: list[str] = []
    seeded: list[str] = []
    for fn, p in paths.items():
        actual = _sha256(p)
        pinned = sha_table.get(fn, TODO) if fn in sha_table else TODO
        if pinned == TODO:
            sha_table[fn] = actual
            seeded.append(fn)
        elif pinned != actual:
            mismatches.append(f"{fn}: pinned {pinned}, got {actual}")
    if seeded:
        print(
            f"[models] seeded {len(seeded)} hash(es) for {model_name}; commit models.toml",
            file=sys.stderr,
        )
    return mismatches


def fetch_and_verify(name: str, variant: str = "fp32") -> LoadedModel:
    """Fetch a model variant, verify (or seed) hashes, return paths."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    doc = _load_doc()
    if name not in doc:
        raise KeyError(f"model '{name}' not declared in models.toml")
    section = doc[name]
    repo = section["repo"]
    revision = _ensure_revision(section, repo, name)

    shared = list(section.get("shared_files", []))
    variant_files = list(section.get(f"files_{variant}", []))
    if not variant_files:
        raise RuntimeError(f"no files declared for {name}.{variant} in models.toml")

    to_download = list(dict.fromkeys(shared + variant_files))
    paths = _download_files(repo, revision, to_download)

    mismatches = _verify_or_seed_hashes(section, paths, name)
    if mismatches:
        details = "\n  ".join(mismatches)
        raise RuntimeError(
            f"SHA256 mismatch for {name} ({repo} @ {revision}):\n  {details}"
        )
    _save_doc(doc)

    # hf_hub_download colocates files from the same repo+revision, so the
    # tokenizer dir is just the parent of any downloaded file.
    tokenizer_dir = next(iter(paths.values())).parent
    onnx_path = paths[variant_files[0]]

    return LoadedModel(
        name=name,
        variant=variant,
        repo=repo,
        revision=revision,
        onnx_path=onnx_path,
        tokenizer_dir=tokenizer_dir,
    )


def fetch_all() -> dict[tuple[str, str], LoadedModel]:
    """Fetch every (model, variant) pair declared in models.toml."""
    doc = _load_doc()
    out: dict[tuple[str, str], LoadedModel] = {}
    for name in doc.keys():
        for v in VARIANTS:
            out[(name, v)] = fetch_and_verify(name, v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        for (name, variant), mf in fetch_all().items():
            print(f"{name}.{variant}: {mf.repo}@{mf.revision[:12]} -> {mf.onnx_path.name}")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
