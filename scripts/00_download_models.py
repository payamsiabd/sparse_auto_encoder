#!/usr/bin/env python
"""Download every model this pipeline needs into a project-local
`models/` directory (path from `configs/default.yaml`'s `models:`
list), so nothing later has to name a Hugging Face hub id or depend on
the shared `~/.cache/huggingface`.

Requires network access to huggingface.co and `pip install huggingface_hub`
(a transitive dependency of `transformers`). Qwen3-VL-4B-Thinking is
~8-9GB in bf16 -- this can take a while depending on bandwidth, and is
safe to re-run if interrupted (only missing/changed files are re-fetched).

Usage:
    python scripts/00_download_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.config import load_config
from rise.models import download_models


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])
    results = download_models(cfg["models"], models_root=cfg["models_root"])

    print("\nDownloaded models:")
    for r in results:
        print(f"  {r.name}: {r.repo_id} -> {r.local_dir}")
    print(
        "\nconfigs/default.yaml's model.model_id already points at the "
        "first entry's local_dir by default, so scripts/02+ need no "
        "further configuration."
    )


if __name__ == "__main__":
    main()
