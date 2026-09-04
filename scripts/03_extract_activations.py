#!/usr/bin/env python
"""Stage (iii) of Sec. 3.2: re-run inference feeding (question, full
response) and extract the residual-stream activation of the "\\n\\n"
token at every step boundary, for every layer requested in the config.
Reads the cache written by script 01.

Usage:
    python scripts/03_extract_activations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.activations import GeneratedResponse, extract_step_activations
from rise.config import load_config
from rise.dataset import build_messages, load_images, load_prompts
from rise.store import append_layer_activations, consolidate_shards
from rise.utils import load_qwen3_vl, split_into_steps


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])

    handle = load_qwen3_vl(
        model_id=cfg["model"]["model_id"], dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"], attn_implementation=cfg["model"]["attn_implementation"],
    )

    prompts_by_id = {p.id: p for p in load_prompts(cfg["data"]["prompts_jsonl"], cfg["data"]["image_root"])}
    out_dir = Path(cfg["activations"]["out_dir"])
    responses_path = out_dir / "responses.jsonl"
    layers = cfg["activations"]["layers"]

    n_steps_total = 0
    with responses_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt = prompts_by_id[rec["prompt_id"]]
            images = load_images(prompt)
            messages = build_messages(prompt, images)

            steps = split_into_steps(rec["thinking_text"]) if rec["thinking_text"] else split_into_steps(rec["full_text"])
            response = GeneratedResponse(
                prompt_id=rec["prompt_id"], full_text=rec["full_text"],
                thinking_text=rec["thinking_text"], answer_text=rec["answer_text"], steps=steps,
            )
            if not response.steps:
                continue

            step_activations = extract_step_activations(handle, images, messages, response, layers)
            append_layer_activations(step_activations, layers, out_dir)
            n_steps_total += len(step_activations)
            print(f"[{rec['prompt_id']}] extracted {len(step_activations)} step activations")

    for layer in layers:
        tensor = consolidate_shards(out_dir, layer)
        print(f"layer {layer}: consolidated {tensor.shape[0]} activations -> "
              f"{out_dir / f'activations_layer{layer:02d}.pt'}")

    print(f"Done. {n_steps_total} total step activations across {len(layers)} layers.")


if __name__ == "__main__":
    main()
