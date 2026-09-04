#!/usr/bin/env python
"""Reproduce Sec. 4.4 / Fig. 4-6's causal-intervention demo, targeting
the `visual_reflection` behavior vector: build the vector from a trained
SAE + annotations, then generate a response to a held-out visual prompt
under negative / vanilla / positive intervention and report how the
count of visual-reflection steps shifts (the same style of evidence as
Fig. 5/6, and the accuracy/step-count table in Sec. 4.4.1).

Usage:
    python scripts/06_intervene_demo.py --config configs/default.yaml \\
        --intervene.target_label visual_reflection
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.annotate import KeywordAnnotator, annotate_steps
from rise.config import load_config
from rise.dataset import build_messages, load_images, load_prompts
from rise.geometry import associate_columns_with_behaviors
from rise.intervene import build_behavior_vector, generate_with_intervention
from rise.store import load_layer_activations, load_steps_metadata
from rise.train_sae import load_sae
from rise.utils import load_qwen3_vl, split_into_steps, split_thinking_and_answer


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])
    sae_cfg, iv_cfg = cfg["sae"], cfg["intervene"]
    layer = iv_cfg["layer"]

    sae_path = Path(sae_cfg["out_dir"]) / f"layer{layer:02d}" / "sae.pt"
    sae, meta = load_sae(sae_path)

    steps_meta = load_steps_metadata(cfg["activations"]["out_dir"])
    annotations = annotate_steps(steps_meta, KeywordAnnotator())
    activations = load_layer_activations(cfg["activations"]["out_dir"], layer)

    assoc = associate_columns_with_behaviors(sae, activations, annotations, input_scale=meta["input_scale"])
    vector = build_behavior_vector(
        sae.reasoning_vectors(), assoc.behavior_column_counts,
        target_label=iv_cfg["target_label"], top_k=iv_cfg["top_k_columns"],
    )
    print(f"Built '{iv_cfg['target_label']}' vector from top-{iv_cfg['top_k_columns']} "
          f"disentangled decoder columns at layer {layer}.")

    handle = load_qwen3_vl(
        model_id=cfg["model"]["model_id"], dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"], attn_implementation=cfg["model"]["attn_implementation"],
    )
    vector = vector.to(handle.device, handle.model.dtype)

    prompts = load_prompts(cfg["data"]["prompts_jsonl"], cfg["data"]["image_root"])
    demo_prompt = prompts[0]
    images = load_images(demo_prompt)
    messages = build_messages(demo_prompt, images)
    prompt_text = handle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = handle.processor(text=[prompt_text], images=images, return_tensors="pt")
    inputs = {k: v.to(handle.device) for k, v in inputs.items()}

    annotator = KeywordAnnotator()
    results = {}
    for name, alpha in [("negative", 1.0), ("vanilla", 0.0), ("positive", -1.0)]:
        def edit_fn(h_last, _alpha=alpha):
            coeff = (h_last * vector).sum(dim=-1, keepdim=True)
            return h_last - _alpha * coeff * vector

        text = generate_with_intervention(
            handle, inputs, layer, edit_fn,
            max_new_tokens=cfg["generation"]["max_new_tokens"], do_sample=False,
        )
        thinking, answer = split_thinking_and_answer(text)
        steps = split_into_steps(thinking) if thinking else split_into_steps(text)
        n_target = sum(1 for s in steps if annotator.annotate(s) == iv_cfg["target_label"])

        results[name] = {"num_steps": len(steps), f"num_{iv_cfg['target_label']}_steps": n_target, "answer": answer}
        print(f"[{name}, alpha={alpha}] {len(steps)} steps, "
              f"{n_target} labeled '{iv_cfg['target_label']}' -> answer: {answer[:120]!r}")

    out_path = Path(cfg["geometry"]["out_dir"]) / f"intervention_demo_{iv_cfg['target_label']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
