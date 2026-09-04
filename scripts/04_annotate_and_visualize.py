#!/usr/bin/env python
"""Annotate every cached reasoning step with a behavior label (Sec. 4.3
/ Appendix D, extended with `visual_reflection`), then reproduce the
paper's decoder-geometry analysis (Fig. 2/3): UMAP projection of decoder
columns highlighted by behavior, plus normalized silhouette scores.

Usage:
    python scripts/04_annotate_and_visualize.py --config configs/default.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.annotate import KeywordAnnotator, annotate_steps, save_annotations
from rise.config import load_config
from rise.geometry import associate_columns_with_behaviors, normalized_silhouette_scores, plot_decoder_geometry, umap_projection
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
    assoc = associate_columns_with_behaviors(
        sae, activations, annotations, input_scale=meta["input_scale"], top_k_per_step=geo_cfg["top_k_per_step"],
    )

    decoder_columns = sae.reasoning_vectors().numpy()
    embedding = umap_projection(decoder_columns)

    out_dir = Path(geo_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_decoder_geometry(embedding, assoc, str(out_dir / f"geometry_layer{layer:02d}.png"))

    silhouette = normalized_silhouette_scores(decoder_columns, assoc.behavior_top_columns)
    with (out_dir / f"silhouette_layer{layer:02d}.json").open("w") as f:
        json.dump(silhouette, f, indent=2)

    print(f"Silhouette scores (layer {layer}): {silhouette}")
    print(f"Wrote geometry plot + silhouette scores to {out_dir}")


if __name__ == "__main__":
    main()
