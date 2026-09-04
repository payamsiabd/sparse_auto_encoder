#!/usr/bin/env python
"""Stage (i)+(ii) of Sec. 3.2: run Qwen3-VL-4B-Thinking over every
(image, question) prompt, generate a CoT response, split it into
sentence-level steps, and cache everything to disk so extraction
(script 03) doesn't need to re-run generation.

Two generation backends, picked by `generation.backend`:
  - "transformers" (default): `model.generate()`, one prompt at a time.
    Always available, works on CPU or GPU.
  - "vllm": batches every prompt into one vLLM engine call for much
    higher throughput on a GPU. See `rise/vllm_backend.py`'s module
    docstring for exactly what vLLM can and can't accelerate in this
    pipeline (short version: generation only -- extraction, steering,
    and the Eq. 7 entropy search all need model internals vLLM doesn't
    expose, so they always run on `transformers` regardless of this
    setting). Requires `pip install vllm` with Qwen3-VL support.

Usage:
    python scripts/02_generate_responses.py
    python scripts/02_generate_responses.py --generation.backend vllm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.activations import GeneratedResponse, generate_response
from rise.config import load_config
from rise.dataset import VisualPrompt, build_messages, load_images, load_prompts
from rise.utils import load_qwen3_vl, set_seed


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])

    prompts = load_prompts(cfg["data"]["prompts_jsonl"], cfg["data"]["image_root"])
    out_dir = Path(cfg["activations"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "responses.jsonl"

    gen_cfg = cfg["generation"]
    backend = gen_cfg["backend"]

    if backend == "vllm":
        # Deliberately do NOT call set_seed() (or touch CUDA in any other
        # way) before this: vLLM's engine core runs in a forked worker
        # subprocess that must inherit a *completely uninitialized* CUDA
        # context. torch.cuda.manual_seed_all() inside set_seed() would
        # initialize one in this (parent) process first, which then makes
        # the fork fail with "Cannot re-initialize CUDA in forked
        # subprocess." vLLM seeds itself internally (passed through
        # below), so skipping our own seeding here is safe.
        responses = _generate_with_vllm(cfg, prompts)
    elif backend == "transformers":
        set_seed(cfg["sae"]["seed"])
        responses = _generate_with_transformers(cfg, prompts)
    else:
        raise ValueError(f"Unknown generation.backend: {backend!r} (expected 'transformers' or 'vllm')")

    with out_path.open("w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, responses):
            f.write(json.dumps({
                "prompt_id": response.prompt_id,
                "question": prompt.question,
                "images": [str(p) for p in prompt.images],
                "full_text": response.full_text,
                "thinking_text": response.thinking_text,
                "answer_text": response.answer_text,
                "num_steps": len(response.steps),
            }) + "\n")
            print(f"[{prompt.id}] {len(response.steps)} steps")

    print(f"Wrote responses to {out_path}")


def _generate_with_transformers(cfg: dict, prompts: list[VisualPrompt]) -> list[GeneratedResponse]:
    handle = load_qwen3_vl(
        model_id=cfg["model"]["model_id"], dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"], attn_implementation=cfg["model"]["attn_implementation"],
    )
    gen_cfg = cfg["generation"]

    responses = []
    for prompt in prompts:
        images = load_images(prompt)
        messages = build_messages(prompt, images)
        responses.append(generate_response(
            handle, images, messages, prompt.id,
            max_new_tokens=gen_cfg["max_new_tokens"], do_sample=gen_cfg["do_sample"],
            temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
        ))
    return responses


def _generate_with_vllm(cfg: dict, prompts: list[VisualPrompt]) -> list[GeneratedResponse]:
    from rise.vllm_backend import generate_responses_batch, load_vllm

    gen_cfg = cfg["generation"]
    handle = load_vllm(
        model_id=cfg["model"]["model_id"], dtype=cfg["model"]["dtype"],
        max_model_len=gen_cfg["vllm_max_model_len"], gpu_memory_utilization=gen_cfg["vllm_gpu_memory_utilization"],
        tensor_parallel_size=gen_cfg["vllm_tensor_parallel_size"], limit_mm_per_prompt=gen_cfg["vllm_limit_mm_per_prompt"],
        seed=cfg["sae"]["seed"],
    )
    return generate_responses_batch(
        handle, prompts, max_new_tokens=gen_cfg["max_new_tokens"], do_sample=gen_cfg["do_sample"],
        temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
    )


if __name__ == "__main__":
    main()
