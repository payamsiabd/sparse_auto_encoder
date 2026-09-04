#!/usr/bin/env python
"""Download MathVista and prepare it as this pipeline's prompt set.

Requires network access to huggingface.co and `pip install datasets`.

Usage:
    python scripts/00_download_mathvista.py --config configs/default.yaml
    python scripts/00_download_mathvista.py --mathvista.num_samples 500
    python scripts/00_download_mathvista.py \\
        --mathvista.task_filter "['chart question answering','geometry problem solving']"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.config import load_config
from rise.mathvista import download_mathvista


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])
    mv_cfg = cfg["mathvista"]

    download_mathvista(
        out_dir=mv_cfg["out_dir"],
        split=mv_cfg["split"],
        num_samples=mv_cfg["num_samples"],
        seed=mv_cfg["seed"],
        task_filter=mv_cfg["task_filter"],
        dataset_id=mv_cfg["dataset_id"],
    )


if __name__ == "__main__":
    main()
