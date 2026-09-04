"""Training loop for the SAE, matching Section 4.2 of the RISE paper.

Paper hyperparameters (Sec. 4.2, used as defaults here):
    D (dict size)      = 2048
    batch size         = 1024
    optimizer          = Adam
    learning rate      = 1e-4, 10% linear warmup, then cosine decay to 0
    sparsity coef λ    = 2e-3 (only used with activation="relu", see rise/sae.py)

Default ``activation`` here is ``"topk"``, not the paper's ``"relu"``:
an L1-penalty SAE's sparsity depends on how ``sparsity_coef`` interacts
with the target model's activation scale, which varies enough across
models/layers that the paper's own reported lambda can land anywhere
from mildly sparse to nearly dense elsewhere (their Fig. 9 shows ~15%
density on DeepSeek-R1-1.5B with lambda=2e-3; a different model's
residual stream can end up far denser with the same value, which is
exactly the failure mode ``activation="topk"`` avoids by construction
-- see ``rise/sae.py``'s module docstring for the full reasoning).

Input: a single tensor of step-boundary activations {h^l_i} of shape
(N, d) produced by ``rise.activations`` for one chosen layer l (Sec.
3.2: "We ... train the SAE on a single chosen layer").
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from .sae import SparseAutoencoder


@dataclasses.dataclass
class TrainConfig:
    d_hidden: int = 2048           # D, SAE dictionary size
    activation: str = "topk"       # "topk" (recommended, exact sparsity) or "relu" (paper's literal Eq. 1)
    k: Optional[int] = 32          # active latents per sample when activation="topk"
    sparsity_coef: float = 2e-3    # lambda; only used when activation="relu"
    sparsity_penalty: str = "l1"   # only used when activation="relu"
    batch_size: int = 1024
    lr: float = 1e-4
    warmup_frac: float = 0.10
    num_epochs: int = 50
    grad_clip: Optional[float] = 1.0
    normalize_inputs: bool = True   # standardize h to unit avg norm (stabilizes SAE training; common practice)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    log_every: int = 50


def _make_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_sae(
    activations: torch.Tensor,
    config: TrainConfig = TrainConfig(),
    out_dir: Optional[Path] = None,
) -> tuple[SparseAutoencoder, dict]:
    """Train an SAE on step-level activations.

    Args:
        activations: float tensor of shape (N, d) -- one row per reasoning
            step, extracted from a single layer (see rise/activations.py).
        config: training hyperparameters.
        out_dir: if given, checkpoints + a training log are written there.

    Returns:
        (trained SAE, history dict of per-log-step metrics)
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    activations = activations.float()
    input_scale = 1.0
    if config.normalize_inputs:
        # Standard SAE-training trick: rescale activations so their mean
        # L2 norm is sqrt(d). This keeps the reconstruction loss and the
        # fixed sparsity_coef in comparable ranges across layers/models
        # with different activation magnitudes, without changing the
        # geometry (direction) of any activation.
        d = activations.shape[-1]
        mean_norm = activations.norm(dim=-1).mean().clamp_min(1e-8)
        input_scale = (d ** 0.5) / mean_norm.item()
        activations = activations * input_scale

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    sae = SparseAutoencoder(
        d_in=activations.shape[-1],
        d_hidden=config.d_hidden,
        sparsity_coef=config.sparsity_coef,
        sparsity_penalty=config.sparsity_penalty,
        activation=config.activation,
        k=config.k,
    ).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    total_steps = max(1, len(loader) * config.num_epochs)
    scheduler = _make_lr_scheduler(optimizer, total_steps, config.warmup_frac)

    history: dict = {"step": [], "loss": [], "recon_loss": [], "l0": [], "lr": [], "input_scale": input_scale}
    step = 0
    for epoch in range(config.num_epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            out = sae(batch)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()

            # Keep decoder columns unit-norm exactly (Sec. 4.4.1): remove the
            # gradient component that would change a column's norm, then
            # renormalize after stepping to correct residual drift.
            sae.remove_decoder_grad_parallel_component_()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), config.grad_clip)
            optimizer.step()
            sae.normalize_decoder_()
            scheduler.step()

            if step % config.log_every == 0:
                history["step"].append(step)
                history["loss"].append(out.loss.item())
                history["recon_loss"].append(out.recon_loss.item())
                history["l0"].append(out.l0.item())
                history["lr"].append(scheduler.get_last_lr()[0])
            step += 1

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": sae.state_dict(),
                "config": dataclasses.asdict(config),
                "input_scale": input_scale,
                "d_in": activations.shape[-1],
            },
            out_dir / "sae.pt",
        )

    return sae, history


def load_sae(path: str | Path, device: str = "cpu") -> tuple[SparseAutoencoder, dict]:
    """Load a checkpoint saved by ``train_sae``. Returns (sae, meta) where
    meta contains ``input_scale`` -- the factor activations were multiplied
    by before training, which must be re-applied at inference/analysis
    time for encode/decode to operate in the same units as training."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = SparseAutoencoder(
        d_in=ckpt["d_in"],
        d_hidden=cfg["d_hidden"],
        sparsity_coef=cfg["sparsity_coef"],
        sparsity_penalty=cfg["sparsity_penalty"],
        activation=cfg.get("activation", "relu"),
        k=cfg.get("k"),
    )
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    meta = {"input_scale": ckpt["input_scale"], "config": cfg}
    return sae, meta
