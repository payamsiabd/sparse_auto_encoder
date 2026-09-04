"""Synthetic dictionary-recovery test for `rise.sae.SparseAutoencoder`,
built directly from the generative model assumed by Theorem 1 (paper
Sec. 4, Appendix B):

    h = W a + eps,  a is k-sparse, |a_i| >= alpha on the support,
    W has low pairwise coherence (max_i!=j |<w_i,w_j>| = mu < 1).

Theorem 1 claims that, in the large-sample limit, any local optimum of
the SAE training objective recovers W up to a permutation and (in our
unit-norm-constrained variant) a sign flip. This test trains a small
SAE on synthetic data generated exactly this way and checks the learned
decoder columns match the true dictionary columns (via best-match cosine
similarity) closely -- i.e. it validates the *implementation* against
the paper's own theoretical justification for what an SAE trained this
way should recover, independent of any real language model.

Runs with only `torch` + `numpy` installed (no GPU, no transformers, no
scipy). Usable directly with pytest or as `python tests/test_sae.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.sae import SparseAutoencoder
from rise.train_sae import TrainConfig, train_sae


def _make_incoherent_dictionary(d: int, m: int, seed: int) -> torch.Tensor:
    """Random unit-norm columns in R^d; with d >> m this is incoherent
    with overwhelming probability (random vectors on a high-dim sphere
    are nearly orthogonal), matching Theorem 1's Assumption (i)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d, m, generator=g)
    W = W / W.norm(dim=0, keepdim=True)
    return W


def _sample_sparse_codes(m: int, k: int, n: int, alpha: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros(n, m)
    for i in range(n):
        support = torch.randperm(m, generator=g)[:k]
        magnitudes = alpha + torch.rand(k, generator=g) * alpha  # in [alpha, 2*alpha]
        signs = torch.where(torch.rand(k, generator=g) > 0.5, 1.0, -1.0)
        A[i, support] = magnitudes * signs
    return A


def best_match_cosine_similarities(W_true: torch.Tensor, W_learned: torch.Tensor) -> np.ndarray:
    """For each true column, the max |cosine similarity| to any learned
    column (permutation- and sign-invariant, as Theorem 1's recovery
    guarantee is stated up to a permutation matrix and scaling
    diagonal). Greedy one-to-one matching (no scipy dependency)."""
    Wt = W_true / W_true.norm(dim=0, keepdim=True)  # (d, m)
    Wl = (W_learned / W_learned.norm(dim=1, keepdim=True)).t()  # (d, D)
    sims = (Wt.t() @ Wl).abs().numpy()  # (m, D)

    m = sims.shape[0]
    best = np.zeros(m)
    used = set()
    order = np.argsort(-sims.max(axis=1))
    for i in order:
        row = sims[i].copy()
        for j in used:
            row[j] = -1
        j_best = int(np.argmax(row))
        best[i] = row[j_best]
        used.add(j_best)
    return best


def test_sae_recovers_dictionary() -> None:
    torch.manual_seed(0)
    d, m, k, n = 96, 24, 3, 30_000
    alpha, noise_std = 1.0, 0.02

    W_true = _make_incoherent_dictionary(d, m, seed=1)
    A = _sample_sparse_codes(m, k, n, alpha=alpha, seed=2)
    H = A @ W_true.t() + noise_std * torch.randn(n, d)

    # SAE dictionary size D is 2x overcomplete relative to the true m,
    # as in the paper's own D=2048 >> (a handful of true behaviors)
    # setup; sparsity_coef is tuned (higher than the paper's 2e-3 default,
    # which was calibrated for real, larger-scale LLM activations) so
    # that, on this small synthetic problem, the sparsity term is strong
    # enough relative to reconstruction to actually enforce a sparse code
    # rather than a dense one that also reconstructs well.
    cfg = TrainConfig(
        d_hidden=2 * m, sparsity_coef=3e-2, batch_size=1024, lr=1e-2,
        warmup_frac=0.05, num_epochs=100, normalize_inputs=False, seed=0, log_every=50,
    )
    sae, history = train_sae(H, cfg)

    sims = best_match_cosine_similarities(W_true, sae.reasoning_vectors())
    mean_sim = float(sims.mean())
    frac_recovered = float((sims > 0.9).mean())

    print(f"mean best-match cosine similarity: {mean_sim:.3f}; "
          f"fraction of true columns recovered (>0.9 cos-sim): {frac_recovered:.2f}; "
          f"final recon_loss={history['recon_loss'][-1]:.4f} L0={history['l0'][-1]:.2f}")

    assert mean_sim > 0.85, f"SAE failed to recover the synthetic dictionary (mean cos-sim={mean_sim:.3f})"
    assert frac_recovered > 0.7, f"Too few dictionary columns recovered ({frac_recovered:.2f})"

    # Decoder columns must stay unit-norm throughout training (Sec. 4.4.1's invariant).
    norms = sae.reasoning_vectors().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "decoder columns are not unit-norm"


def test_sae_forward_shapes() -> None:
    sae = SparseAutoencoder(d_in=16, d_hidden=8)
    h = torch.randn(5, 16)
    out = sae(h)
    assert out.h_hat.shape == (5, 16)
    assert out.z.shape == (5, 8)
    assert (out.z >= 0).all(), "ReLU code must be non-negative"
    assert out.loss.item() >= 0


if __name__ == "__main__":
    test_sae_forward_shapes()
    print("test_sae_forward_shapes: OK")
    test_sae_recovers_dictionary()
    print("test_sae_recovers_dictionary: OK")
