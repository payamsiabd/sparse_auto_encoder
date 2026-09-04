#!/usr/bin/env python
"""Train the SAE (Sec. 3.1, 4.2) on the cached activations for one layer.

Usage:
    python scripts/04_train_sae.py \\
        --sae.train_layer 16 --sae.d_hidden 2048
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.config import load_config
from rise.store import load_layer_activations
from rise.train_sae import TrainConfig, train_sae


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", sys.argv[1:])
    sae_cfg = cfg["sae"]

    layer = sae_cfg["train_layer"]
    activations = load_layer_activations(cfg["activations"]["out_dir"], layer)
    print(f"Loaded {activations.shape[0]} activations of dim {activations.shape[1]} for layer {layer}")

    train_cfg = TrainConfig(
        d_hidden=sae_cfg["d_hidden"], activation=sae_cfg["activation"], k=sae_cfg["k"],
        sparsity_coef=sae_cfg["sparsity_coef"], sparsity_penalty=sae_cfg["sparsity_penalty"],
        batch_size=min(sae_cfg["batch_size"], activations.shape[0]),
        lr=sae_cfg["lr"], warmup_frac=sae_cfg["warmup_frac"], num_epochs=sae_cfg["num_epochs"],
        normalize_inputs=sae_cfg["normalize_inputs"], seed=sae_cfg["seed"],
    )

    out_dir = Path(sae_cfg["out_dir"]) / f"layer{layer:02d}"
    sae, history = train_sae(activations, train_cfg, out_dir=out_dir)

    print(f"Final loss={history['loss'][-1]:.4f} recon={history['recon_loss'][-1]:.4f} "
          f"L0={history['l0'][-1]:.1f}/{train_cfg.d_hidden}")
    print(f"Saved SAE checkpoint to {out_dir / 'sae.pt'}")


if __name__ == "__main__":
    main()
