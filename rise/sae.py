"""Sparse Auto-Encoder (SAE) for reasoning-vector discovery.

Implements the SAE defined in Section 3.1 of
"Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised
Discovery of the Reasoning Process" (RISE, Google DeepMind, 2025).

Paper definitions (Eq. 1-2):
    h_hat = W_decoder^T * sigma(z) + b_decoder
    z     = sigma(W_encoder^T * h + b_encoder)
    L     = || h_hat - h ||_2^2 + lambda * || sigma(z) ||_0

where h in R^d is the residual-stream activation at a reasoning-step
boundary, sigma is ReLU, W_decoder in R^{D x d} (each *row* w_i is a
"reasoning vector" / atomic behavior direction), D is the SAE's
dictionary size (hidden dim) and d is the target model's hidden size.

Two implementation notes, made explicit because they are not spelled
out at the equation level in the paper but are required to reproduce
its stated setup:

1. ``||z||_0`` is not differentiable, so -- as is standard for this
   family of SAEs (the paper cites Cunningham et al., 2023, i.e. the
   "standard" ReLU SAE recipe) -- we optimize the L1 relaxation
   ``lambda * ||z||_1`` and report the *true* L0 (average number of
   active latents) purely as a diagnostic. This is controlled by
   ``sparsity_penalty="l1"`` (default) vs. ``"l0"`` (a literal, biased
   straight-through L0 estimator, provided for completeness but not
   used by default since it does not reliably train).
2. Section 4.4.1 states decoder columns are "independently normalized
   to unit length ||w_i||_2 = 1" before being used for interventions.
   We enforce this as a hard constraint *during training* (renormalize
   after every optimizer step), which is the standard way SAEs keep a
   well-conditioned, non-collapsing dictionary and guarantees the
   property the paper relies on for Eq. 6-7 holds by construction,
   not just post-hoc.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class SAEOutput:
    h_hat: torch.Tensor          # reconstruction, shape (..., d)
    z: torch.Tensor              # sparse code (post-ReLU), shape (..., D)
    recon_loss: torch.Tensor     # scalar, ||h_hat - h||_2^2 (mean over batch)
    sparsity_loss: torch.Tensor  # scalar, the (surrogate) sparsity term
    loss: torch.Tensor           # scalar, recon_loss + lambda * sparsity_loss
    l0: torch.Tensor             # scalar, mean number of active latents (diagnostic)


class SparseAutoencoder(nn.Module):
    """Standard ReLU SAE with unit-norm decoder columns.

    Args:
        d_in: dimensionality d of the target model's residual stream.
        d_hidden: dictionary size D (number of reasoning-vector candidates).
            The paper uses D = 2048 for a 1.5B-parameter model (Sec. 4.2).
        sparsity_coef: lambda in Eq. 2. Paper default: 2e-3.
        sparsity_penalty: "l1" (default, differentiable surrogate for L0)
            or "l0" (straight-through estimator of the true L0 term).
        tied_init: if True, initialize W_encoder = W_decoder^T (common,
            improves early-training stability); both are still optimized
            independently afterwards (untied SAE, matching Eq. 1 which
            gives the encoder and decoder separate parameters).
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        sparsity_coef: float = 2e-3,
        sparsity_penalty: Literal["l1", "l0"] = "l1",
        tied_init: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.sparsity_coef = sparsity_coef
        self.sparsity_penalty = sparsity_penalty

        # W_encoder in R^{d x D}; W_decoder in R^{D x d} (rows = w_i).
        W_dec = torch.randn(d_hidden, d_in)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_decoder = nn.Parameter(W_dec)

        if tied_init:
            W_enc = W_dec.t().clone()
        else:
            W_enc = torch.empty(d_in, d_hidden)
            nn.init.kaiming_uniform_(W_enc, a=5 ** 0.5)
        self.W_encoder = nn.Parameter(W_enc)

        self.b_encoder = nn.Parameter(torch.zeros(d_hidden))
        self.b_decoder = nn.Parameter(torch.zeros(d_in))

    # -- core math, mirrors Eq. 1 exactly -------------------------------
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """z = sigma(W_encoder^T h + b_encoder)."""
        return F.relu(h @ self.W_encoder + self.b_encoder)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """h_hat = W_decoder^T z + b_decoder."""
        return z @ self.W_decoder + self.b_decoder

    def forward(self, h: torch.Tensor) -> SAEOutput:
        z = self.encode(h)
        h_hat = self.decode(z)

        recon_loss = F.mse_loss(h_hat, h, reduction="none").sum(dim=-1).mean()

        if self.sparsity_penalty == "l1":
            sparsity_loss = z.abs().sum(dim=-1).mean()
        elif self.sparsity_penalty == "l0":
            # Straight-through estimator: forward pass uses the true count,
            # backward pass borrows the L1 gradient. Biased but literal.
            l0_forward = (z > 0).float().sum(dim=-1)
            l1_backward = z.abs().sum(dim=-1)
            sparsity_loss = (l0_forward - l1_backward).detach().mean() + l1_backward.mean()
        else:
            raise ValueError(f"Unknown sparsity_penalty: {self.sparsity_penalty}")

        loss = recon_loss + self.sparsity_coef * sparsity_loss
        l0 = (z > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            h_hat=h_hat, z=z, recon_loss=recon_loss.detach(),
            sparsity_loss=sparsity_loss.detach(), loss=loss, l0=l0.detach(),
        )

    # -- decoder-column unit-norm constraint (Sec. 4.4.1) ----------------
    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Rescale every decoder row (reasoning vector) to unit L2 norm."""
        norms = self.W_decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_decoder.div_(norms)

    @torch.no_grad()
    def remove_decoder_grad_parallel_component_(self) -> None:
        """Project out the gradient component parallel to each decoder
        column before the optimizer step, so that a plain gradient step
        does not change a column's norm (only its direction). This keeps
        ``normalize_decoder_`` a no-op rescale rather than fighting the
        optimizer every step. Call after ``loss.backward()`` and before
        ``optimizer.step()``; combine with ``normalize_decoder_`` (called
        after the step) to guarantee ||w_i||_2 == 1 at all times.
        """
        if self.W_decoder.grad is None:
            return
        W = self.W_decoder
        g = self.W_decoder.grad
        unit = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        parallel = (g * unit).sum(dim=1, keepdim=True) * unit
        g.sub_(parallel)

    def reasoning_vectors(self) -> torch.Tensor:
        """Return the decoder column matrix W_decoder in R^{D x d}, i.e.
        the discovered reasoning vectors {w_i}, already unit-normalized
        as long as training kept the unit-norm invariant."""
        return self.W_decoder.detach()
