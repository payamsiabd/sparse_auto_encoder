#!/usr/bin/env python
"""Annotate every cached reasoning step with a behavior label (Sec. 4.3
/ Appendix D, extended with `visual_reflection`), then reproduce the
paper's decoder-geometry analysis (Fig. 2/3): UMAP projection of decoder
columns highlighted by behavior, plus normalized silhouette scores.

Produces two geometry views:
  - `geometry_layer{L}.png` (primary): binary reflection (reflection /
    backtracking / visual_reflection, collapsed) vs others -- "does
    reasoning-interruption cluster separately from ordinary forward
    reasoning at all", the coarser, usually cleaner-looking question.
  - `geometry_layer{L}_finegrained.png`: the original 4-way split,
    which you still need for this project's actual goal (isolating
    *visual* reflection from plain reflection/backtracking) once the
    binary view confirms reflection clusters at all.

Which decoder columns get associated with which behavior is controlled
by `geometry.method` -- "stats" (default, a per-column statistical test;
see rise/feature_stats.py) or "argmax" (the paper's literal Sec. 4.3
methodology). "stats" tends to find far more, cleaner columns per
behavior when there are only a modest number of labeled steps.

Usage:
    python scripts/05_annotate_and_visualize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.annotate import KeywordAnnotator, annotate_steps, save_annotations, to_binary_labels
from rise.config import load_config
from rise.feature_stats import build_association
from rise.geometry import normalized_silhouette_scores, plot_decoder_geometry, save_association, umap_projection
from rise.store import load_layer_activations, load_steps_metadata
from rise.train_sae import load_sae


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])

    steps_meta = load_steps_metadata(cfg["activations"]["out_dir"])

    ann_cfg = cfg["annotate"]
    if ann_cfg["method"] == "keyword":
        annotator = KeywordAnnotator()
    else:
        raise NotImplementedError(
            "For method='llm', build an LLMJudgeAnnotator with your own API "
            "client in a short driver script -- see rise/annotate.py's "
            "LLMJudgeAnnotator docstring for the one-liner."
        )

    annotations = annotate_steps(steps_meta, annotator)
    out_path = Path(ann_cfg["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_annotations(annotations, str(out_path))

    counts = {}
    for a in annotations:
        counts[a.label] = counts.get(a.label, 0) + 1
    print(f"Annotated {len(annotations)} steps: {counts}")
    print(f"Saved annotations to {out_path}")

    layer = cfg["sae"]["train_layer"]
    sae_path = Path(cfg["sae"]["out_dir"]) / f"layer{layer:02d}" / "sae.pt"
    if not sae_path.exists():
        print(f"No trained SAE found at {sae_path}; skipping geometry analysis.")
        return

    sae, meta = load_sae(sae_path)
    activations = load_layer_activations(cfg["activations"]["out_dir"], layer)

    geo_cfg = cfg["geometry"]
    decoder_columns = sae.reasoning_vectors().numpy()
    embedding = umap_projection(decoder_columns)
    out_dir = Path(geo_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fine-grained (4-way) association: this is the one downstream
    # scripts (07, 08) load to build the visual_reflection vector
    # specifically, so it's the one persisted to the stable filename.
    assoc = build_association(geo_cfg, sae, activations, annotations, input_scale=meta["input_scale"])
    plot_decoder_geometry(embedding, assoc, str(out_dir / f"geometry_layer{layer:02d}_finegrained.png"))
    silhouette = normalized_silhouette_scores(decoder_columns, assoc.behavior_top_columns)
    with (out_dir / f"silhouette_layer{layer:02d}.json").open("w") as f:
        json.dump(silhouette, f, indent=2)
    assoc_path = out_dir / f"association_layer{layer:02d}.json"
    save_association(assoc, assoc_path)

    # Binary (reflection vs. others) association: the primary, coarser
    # view -- collapses reflection/backtracking/visual_reflection into
    # one group before clustering, per label_groups above.
    binary_annotations = to_binary_labels(annotations)
    assoc_binary = build_association(geo_cfg, sae, activations, binary_annotations, input_scale=meta["input_scale"])
    plot_decoder_geometry(
        embedding, assoc_binary, str(out_dir / f"geometry_layer{layer:02d}.png"),
        labels_to_plot=["reflection", "others"],
    )
    silhouette_binary = normalized_silhouette_scores(decoder_columns, assoc_binary.behavior_top_columns)
    with (out_dir / f"silhouette_layer{layer:02d}_binary.json").open("w") as f:
        json.dump(silhouette_binary, f, indent=2)

    print(f"Binary (reflection vs. others) silhouette scores (layer {layer}): {silhouette_binary}")
    print(f"Fine-grained silhouette scores (layer {layer}): {silhouette}")
    print(f"Wrote geometry_layer{layer:02d}.png (binary, primary) and "
          f"geometry_layer{layer:02d}_finegrained.png to {out_dir}")
    print(f"Saved fine-grained behavior-column association to {assoc_path} "
          f"(reused by scripts/07_intervene_demo.py and scripts/08_evaluate_on_test.py)")


if __name__ == "__main__":
    main()
