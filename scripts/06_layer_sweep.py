#!/usr/bin/env python
"""Reproduce Fig. 3: train one SAE per cached layer and plot normalized
silhouette scores (overall + reflection/backtracking/visual_reflection
pairwise) across layers, to pick the layer with the most disentangled
behavior geometry for the intervention stage (script 06).

Usage:
    python scripts/06_layer_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.annotate import KeywordAnnotator, annotate_steps
from rise.config import load_config
from rise.feature_stats import build_association
from rise.geometry import normalized_silhouette_scores
from rise.store import load_layer_activations, load_steps_metadata
from rise.train_sae import TrainConfig, train_sae


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])
    sae_cfg, geo_cfg = cfg["sae"], cfg["geometry"]

    steps_meta = load_steps_metadata(cfg["activations"]["out_dir"])
    annotations = annotate_steps(steps_meta, KeywordAnnotator())

    curve: dict[int, dict] = {}
    for layer in cfg["activations"]["layers"]:
        activations = load_layer_activations(cfg["activations"]["out_dir"], layer)
        train_cfg = TrainConfig(
            d_hidden=sae_cfg["d_hidden"], activation=sae_cfg["activation"], k=sae_cfg["k"],
            sparsity_coef=sae_cfg["sparsity_coef"], sparsity_penalty=sae_cfg["sparsity_penalty"],
            batch_size=min(sae_cfg["batch_size"], activations.shape[0]),
            lr=sae_cfg["lr"], warmup_frac=sae_cfg["warmup_frac"], num_epochs=sae_cfg["num_epochs"],
            normalize_inputs=sae_cfg["normalize_inputs"], seed=sae_cfg["seed"],
        )
        out_dir = Path(sae_cfg["out_dir"]) / f"layer{layer:02d}"
        sae, history = train_sae(activations, train_cfg, out_dir=out_dir)

        # sae was trained on activations rescaled by history["input_scale"]
        # (Sec. normalize_inputs); encode() must see inputs in the same
        # units, not the raw (unscaled) `activations` tensor as-is.
        assoc = build_association(geo_cfg, sae, activations, annotations, input_scale=history["input_scale"])
        decoder_columns = sae.reasoning_vectors().numpy()
        scores = normalized_silhouette_scores(decoder_columns, assoc.behavior_top_columns)
        curve[layer] = scores
        print(f"layer {layer}: {scores}")

    out_dir = Path(geo_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "silhouette_by_layer.json").open("w") as f:
        json.dump(curve, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        layers = sorted(curve.keys())
        keys = sorted({k for v in curve.values() for k in v})
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in keys:
            ys = [curve[l].get(k, float("nan")) for l in layers]
            ax.plot(layers, ys, marker="o", label=k)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Normalized Silhouette Score")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "silhouette_by_layer.png", dpi=160)
        print(f"Wrote plot to {out_dir / 'silhouette_by_layer.png'}")
    except ImportError:
        pass

    print(f"Wrote {out_dir / 'silhouette_by_layer.json'}")


if __name__ == "__main__":
    main()
