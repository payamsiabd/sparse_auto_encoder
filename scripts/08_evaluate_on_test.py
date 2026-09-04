#!/usr/bin/env python
"""Evaluate the trained SAE + discovered behavior vectors on the
MathVista *test* split -- data neither the SAE nor the behavior-column
association (`scripts/05_annotate_and_visualize.py`'s saved
`association_layer*.json`) has seen.

Two results, both about whether `visual_reflection` (or whichever
`intervene.target_label` you set) is a real, generalizing direction and
not an artifact of the train split:

1. **Classification agreement.** For every held-out reasoning step,
   predict its behavior label from its SAE code alone
   (`rise.geometry.predict_label`, using only the train-derived
   column->behavior association) and compare against the keyword
   annotator applied directly to that step's text. High agreement,
   especially on `visual_reflection`, is evidence the SAE learned a
   real, reusable direction rather than overfitting train-specific
   activations.
2. **Causal intervention effect.** Build the target behavior vector
   from the same saved association and steer generation (Eq. 6) on
   held-out prompts, exactly as `scripts/07_intervene_demo.py` does on
   a single train prompt -- but here averaged over several *test*
   prompts, the way Fig. 5 reports effects across many tasks rather
   than one example.

Usage:
    python scripts/08_evaluate_on_test.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.activations import extract_step_activations, generate_response
from rise.annotate import KeywordAnnotator
from rise.config import load_config
from rise.dataset import build_messages, load_images, load_prompts
from rise.geometry import load_association, predict_label
from rise.intervene import build_behavior_vector, generate_with_intervention
from rise.train_sae import load_sae
from rise.utils import load_qwen3_vl, split_into_steps, split_thinking_and_answer

LABELS = ["reflection", "backtracking", "visual_reflection", "others"]


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])
    sae_cfg, geo_cfg, iv_cfg, eval_cfg = cfg["sae"], cfg["geometry"], cfg["intervene"], cfg["evaluate"]
    layer = sae_cfg["train_layer"]

    sae_path = Path(sae_cfg["out_dir"]) / f"layer{layer:02d}" / "sae.pt"
    assoc_path = Path(geo_cfg["out_dir"]) / f"association_layer{layer:02d}.json"
    if not sae_path.exists() or not assoc_path.exists():
        raise FileNotFoundError(
            f"Missing {sae_path if not sae_path.exists() else assoc_path}. "
            f"Run scripts/04_train_sae.py and scripts/05_annotate_and_visualize.py first."
        )
    sae, meta = load_sae(sae_path)
    assoc = load_association(assoc_path)

    handle = load_qwen3_vl(
        model_id=cfg["model"]["model_id"], dtype=cfg["model"]["dtype"],
        device_map=cfg["model"]["device_map"], attn_implementation=cfg["model"]["attn_implementation"],
    )

    test_prompts = load_prompts(cfg["data"]["test_prompts_jsonl"], cfg["data"]["image_root"])
    if eval_cfg["num_samples"] is not None:
        test_prompts = test_prompts[: eval_cfg["num_samples"]]
    print(f"Evaluating on {len(test_prompts)} held-out MathVista test examples.")

    annotator = KeywordAnnotator()
    out_dir = Path(eval_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. classification agreement -------------------------------------
    y_true, y_pred, records = [], [], []
    for prompt in test_prompts:
        images = load_images(prompt)
        messages = build_messages(prompt, images)
        response = generate_response(
            handle, images, messages, prompt.id,
            max_new_tokens=cfg["generation"]["max_new_tokens"], do_sample=cfg["generation"]["do_sample"],
        )
        if not response.steps:
            continue
        step_acts = extract_step_activations(handle, images, messages, response, [layer])
        sae_device = next(sae.parameters()).device
        for sa in step_acts:
            h = (sa.hidden_states[layer].float() * meta["input_scale"]).to(sae_device)
            code = sae.encode(h).cpu()
            pred = predict_label(code, assoc, top_k=geo_cfg["top_k_per_step"])
            true = annotator.annotate(sa.step_text)
            y_true.append(true)
            y_pred.append(pred)
            records.append({
                "prompt_id": prompt.id, "step_index": sa.step_index, "step_text": sa.step_text,
                "annotated_label": true, "predicted_label": pred,
            })
        print(f"[{prompt.id}] {len(step_acts)} steps")

    with (out_dir / "step_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    report, confusion = _classification_report(y_true, y_pred, LABELS)
    with (out_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "report": report, "confusion_matrix": confusion}, f, indent=2)

    print("\nSAE-predicted vs. keyword-annotated label, held-out test split:")
    print(json.dumps(report, indent=2))

    # -- 2. causal intervention effect on held-out prompts ----------------
    target = iv_cfg["target_label"]
    vector = build_behavior_vector(
        sae.reasoning_vectors(), assoc.behavior_column_counts, target_label=target, top_k=iv_cfg["top_k_columns"],
    ).to(handle.device, handle.model.dtype)

    n_iv = min(eval_cfg["num_intervene_samples"], len(test_prompts))
    iv_rows = {"negative": [], "vanilla": [], "positive": []}
    for prompt in test_prompts[:n_iv]:
        images = load_images(prompt)
        messages = build_messages(prompt, images)
        prompt_text = handle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = handle.processor(text=[prompt_text], images=images, return_tensors="pt")
        inputs = {k: v.to(handle.device) for k, v in inputs.items()}

        for name, alpha in [("negative", 1.0), ("vanilla", 0.0), ("positive", -1.0)]:
            def edit_fn(h_last, _alpha=alpha, _vector=vector):
                coeff = (h_last * _vector).sum(dim=-1, keepdim=True)
                return h_last - _alpha * coeff * _vector

            text = generate_with_intervention(
                handle, inputs, layer, edit_fn, max_new_tokens=cfg["generation"]["max_new_tokens"],
            )
            thinking, _ = split_thinking_and_answer(text)
            steps = split_into_steps(thinking) if thinking else split_into_steps(text)
            n_target = sum(1 for s in steps if annotator.annotate(s) == target)
            iv_rows[name].append({"prompt_id": prompt.id, "num_steps": len(steps), "num_target_steps": n_target})
        print(f"[{prompt.id}] intervention done")

    iv_summary = {
        name: {
            "mean_num_steps": statistics.fmean(r["num_steps"] for r in rows) if rows else 0.0,
            "mean_target_steps": statistics.fmean(r["num_target_steps"] for r in rows) if rows else 0.0,
        }
        for name, rows in iv_rows.items()
    }
    with (out_dir / "intervention_effect.json").open("w", encoding="utf-8") as f:
        json.dump({"target_label": target, "per_sample": iv_rows, "summary": iv_summary}, f, indent=2)

    print(f"\nCausal intervention effect on {n_iv} held-out test prompts (target={target!r}):")
    print(json.dumps(iv_summary, indent=2))
    print(f"\nWrote full test evaluation report to {out_dir}")


def _classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[dict, list[list[int]]]:
    """Per-label precision/recall/F1/support + a label x label confusion
    matrix. Implemented directly (no sklearn dependency) since the
    computation is a handful of counts."""
    confusion = [[0] * len(labels) for _ in labels]
    idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        confusion[idx.get(t, idx["others"])][idx.get(p, idx["others"])] += 1

    report = {}
    for i, label in enumerate(labels):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted = sum(confusion[r][i] for r in range(len(labels)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    total = sum(sum(row) for row in confusion)
    accuracy = sum(confusion[i][i] for i in range(len(labels))) / total if total else 0.0
    report["accuracy"] = accuracy
    report["total_steps"] = total
    return report, confusion


if __name__ == "__main__":
    main()
