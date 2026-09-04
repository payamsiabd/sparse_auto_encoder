"""Download models to a project-local directory so the rest of the
pipeline never has to name a Hugging Face hub id or rely on the shared
`~/.cache/huggingface` cache: every config default points at a path
under `models/` in this repo, and everything after the download step
runs fully offline.

Currently only Qwen3-VL-4B-Thinking is required. `MODEL_REGISTRY` is a
plain dict so adding a second local model (e.g. a local judge model for
`rise.annotate.LLMJudgeAnnotator`, should you want one instead of an
API-based judge) is a one-line addition, not a new code path.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

# name -> HF hub repo id. `configs/default.yaml`'s `models:` list picks
# which of these (or any other repo id) to fetch and where to put it.
MODEL_REGISTRY: dict[str, str] = {
    "qwen3-vl-4b-thinking": "Qwen/Qwen3-VL-4B-Thinking",
}


@dataclasses.dataclass
class DownloadedModel:
    name: str
    repo_id: str
    local_dir: Path


def resolve_repo_id(name_or_repo_id: str) -> str:
    """Accept either a short registry key (`"qwen3-vl-4b-thinking"`) or
    an arbitrary Hugging Face repo id (`"Qwen/Qwen3-VL-4B-Thinking"`,
    or someone else's fine-tune) -- so the registry is a convenience,
    not a restriction."""
    return MODEL_REGISTRY.get(name_or_repo_id, name_or_repo_id)


def download_model(
    repo_id: str,
    local_dir: str | Path,
    revision: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> DownloadedModel:
    """Snapshot a full model repo (weights, tokenizer, processor config,
    chat template, etc.) to `local_dir`. Idempotent / resumable: an
    interrupted or re-run download only fetches missing/changed files.

    Requires network access to huggingface.co and `pip install
    huggingface_hub` (a transitive dependency of `transformers`, so
    normally already present).
    """
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Skip the (large) original-format checkpoint shards some repos ship
    # alongside the safetensors ones; we only need one weight format.
    default_ignore = ["*.bin", "*.pt", "*.pth", "original/*", "*.gguf"]

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=None if allow_patterns else default_ignore,
    )
    return DownloadedModel(name=repo_id, repo_id=repo_id, local_dir=local_dir)


def download_models(model_specs: list[dict], models_root: str | Path) -> list[DownloadedModel]:
    """Batch entry point used by `scripts/00_download_models.py`.
    `model_specs` is `configs/default.yaml`'s `models:` list: each entry
    is `{"name": <local dir name>, "repo_id": <registry key or hub id>,
    "revision": <optional>}`."""
    models_root = Path(models_root)
    results = []
    for spec in model_specs:
        repo_id = resolve_repo_id(spec["repo_id"])
        local_dir = models_root / spec["name"]
        print(f"Downloading {repo_id} -> {local_dir} ...")
        result = download_model(repo_id, local_dir, revision=spec.get("revision"))
        result.name = spec["name"]
        results.append(result)
        print(f"Done: {local_dir}")
    return results


def model_is_downloaded(local_dir: str | Path) -> bool:
    """Cheap readiness check used by the loader to give a clear error
    instead of a confusing `from_pretrained` failure or an implicit hub
    download when the download step was simply never run."""
    local_dir = Path(local_dir)
    return local_dir.is_dir() and any(local_dir.glob("*.safetensors")) and (local_dir / "config.json").exists()
