#!/usr/bin/env python
"""Stage (i)+(ii) of Sec. 3.2: run Qwen3-VL-4B-Thinking over every
(image, question) prompt, generate a CoT response, split it into
sentence-level steps, and cache everything to disk so extraction
(script 02) doesn't need to re-run generation.

Usage:
    python scripts/01_generate_responses.py --config configs/default.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.activations import generate_response
from rise.config import load_config
from rise.dataset import build_messages, load_images, load_prompts
from rise.utils import load_qwen3_vl, set_seed


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])
    set_seed(cfg["sae"]["seed"])

    handle = load_qwen3_vl(
        model_id=cfg["model"]["model_id"], dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"], attn_implementation=cfg["model"]["attn_implementation"],
    )

    prompts = load_prompts(cfg["data"]["prompts_jsonl"], cfg["data"]["image_root"])
    out_dir = Path(cfg["activations"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"

    gen_cfg = cfg["generation"]
    with out_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            images = load_images(prompt)
            messages = build_messages(prompt, images)
            response = generate_response(
                handle, images, messages, prompt.id,
                max_new_tokens=gen_cfg["max_new_tokens"], do_sample=gen_cfg["do_sample"],
                temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
            )
            f.write(json.dumps({
                "prompt_id": response.prompt_id,
                "question": prompt.question,
                "images": [str(p) for p in prompt.images],
                "full_text": response.full_text,
                "thinking_text": response.thinking_text,
                "answer_text": response.answer_text,
                "num_steps": len(response.steps),
            }) + "\n")
            f.flush()
            print(f"[{prompt.id}] {len(response.steps)} steps")

    print(f"Wrote responses to {out_path}")


if __name__ == "__main__":
    main()
