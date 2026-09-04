"""Causal interventions on SAE decoder columns (Sec. 4.4, Eq. 6; Sec. 5,
Eq. 7): steering the target model's generation by editing its residual
stream along discovered reasoning-vector directions, and searching for
new (e.g. confidence / visual-reflection) directions defined purely by
their causal effect on model outputs.

All interventions operate on the residual stream *after* transformer
layer ``layer`` (the same layer the SAE was trained on), matching
Sec. 3.2 / 4.4 exactly.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import torch
import torch.nn.functional as F

from .sae import SparseAutoencoder
from .utils import ModelHandle


# ---------------------------------------------------------------------------
# Eq. 6: h' = h - alpha * w_i (w_i^T h)
# alpha = 1  -> paper's "negative intervention" (fully project out w_i)
# alpha = -1 -> paper's "positive intervention" (double the w_i component)
# alpha = 0  -> vanilla (no-op)
# Section 4.4.1 also sweeps alpha in {-1.5, -1, 0, 1, 1.5} for fine control.
# ---------------------------------------------------------------------------
def project_intervene(h: torch.Tensor, w: torch.Tensor, alpha: float) -> torch.Tensor:
    """h, w: (..., d) and (d,) respectively (w assumed unit-norm)."""
    coeff = torch.einsum("...d,d->...", h, w)
    return h - alpha * coeff.unsqueeze(-1) * w


def combined_vector_intervene(h: torch.Tensor, vectors: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """h' = h + sum_i coeffs_i * vectors_i (Sec. 5.1 "Reasoning Enhancement
    via Confidence Vectors": h' = h + sum_{i in 1,2,3} alpha_i c_i, a
    sample-dependent additive combination of top confidence vectors)."""
    delta = torch.einsum("i,id->d", coeffs, vectors)
    return h + delta


# ---------------------------------------------------------------------------
# Behavior-vector construction (Sec. 4.4.1): "filter out decoder columns
# that exhibit strong activations across multiple behaviors ... From the
# remaining ... columns, compute their average to obtain a single
# <behavior> vector."
# ---------------------------------------------------------------------------
def build_behavior_vector(
    reasoning_vectors: torch.Tensor,          # (D, d), unit-norm decoder columns
    behavior_column_counts: dict[str, "torch.Tensor | list"],
    target_label: str,
    exclusive: bool = True,
    top_k: int = 20,
) -> torch.Tensor:
    """Average the top-firing decoder columns for `target_label`, after
    excluding any column that also fires strongly for a different label
    (the paper's disentanglement filter)."""
    import numpy as np

    counts = {k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in behavior_column_counts.items()}
    target_counts = counts[target_label]
    candidate_idx = np.argsort(-target_counts)[: top_k * 4]  # oversample before filtering
    candidate_idx = [i for i in candidate_idx if target_counts[i] > 0]

    if exclusive:
        other_labels = [l for l in counts if l != target_label]
        kept = []
        for i in candidate_idx:
            others_fire = [counts[l][i] for l in other_labels]
            if not others_fire or target_counts[i] > max(others_fire):
                kept.append(i)
        candidate_idx = kept

    candidate_idx = candidate_idx[:top_k]
    if not candidate_idx:
        raise ValueError(f"No columns survived filtering for label {target_label!r}; lower top_k/exclusive.")

    vecs = reasoning_vectors[candidate_idx]  # (k, d)
    vec = vecs.mean(dim=0)
    return vec / vec.norm().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Locating the target model's decoder-layer modules, robustly across the
# handful of attribute paths different transformers versions/architectures
# have used for VLM language backbones.
# ---------------------------------------------------------------------------
def _get_decoder_layers(model) -> torch.nn.ModuleList:
    candidates = [
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj
    raise AttributeError(
        "Could not locate the decoder-layer ModuleList on this model; "
        "inspect `model.named_modules()` and add the correct attribute "
        "path to `_get_decoder_layers`."
    )


class SteeringHook:
    """Registers a forward hook on decoder layer `layer` that applies
    `edit_fn` to that layer's output hidden state at every generation
    step. Usable as a context manager so the hook is always removed."""

    def __init__(self, handle: ModelHandle, layer: int, edit_fn: Callable[[torch.Tensor], torch.Tensor], positions: str = "last"):
        self.handle = handle
        self.layer = layer
        self.edit_fn = edit_fn
        self.positions = positions  # "last" (every generation step's newest token) or "all"
        self._handle = None

    def _hook(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()

        if self.positions == "last":
            hidden[:, -1, :] = self.edit_fn(hidden[:, -1, :])
        else:
            hidden = self.edit_fn(hidden)

        return (hidden,) + output[1:] if is_tuple else hidden

    def __enter__(self):
        layers = _get_decoder_layers(self.handle.model)
        self._handle = layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def generate_with_intervention(
    handle: ModelHandle,
    inputs: dict,
    layer: int,
    edit_fn: Callable[[torch.Tensor], torch.Tensor],
    max_new_tokens: int = 4096,
    do_sample: bool = False,
) -> str:
    """Reproduces the paper's intervention protocol (Sec. 4.4.1, Fig. 4):
    ``edit_fn`` is applied to the hidden representation of the last token
    at every reasoning step during generation -- i.e. at every decoding
    step, which subsumes "every step boundary" since the last token is
    always the newest generated one."""
    with SteeringHook(handle, layer, edit_fn, positions="last"):
        output_ids = handle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return handle.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Eq. 7: unsupervised discovery of a target-objective vector (Sec. 5).
# argmin_S E[-sum_k p_k log p_k],  p = softmax(f_{l->L}(h + S W_decoder))
# Implemented via a hook that adds S @ W_decoder into the residual stream
# at layer `layer`, position `position`, then reads next-token logits at
# that same position from a normal forward pass through the rest of the
# (frozen) model -- exactly f_{l->L} composed with the fixed f_{1->l}
# that produced `h` in the first place.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class EntropyVectorResult:
    S: torch.Tensor            # (D,) score vector
    vector: torch.Tensor       # (d,) = S @ W_decoder, the resulting steering direction
    history: list[float]


def search_entropy_vector(
    handle: ModelHandle,
    sae: SparseAutoencoder,
    layer: int,
    batches: list[dict],
    positions: list[list[int]],
    num_iterations: int = 1000,
    lr: float = 0.01,
    batch_size: int = 256,
) -> EntropyVectorResult:
    """`batches`/`positions`: parallel lists where `batches[i]` is a
    tokenized (image-expanded) model input (as returned by
    `locate_step_tokens`) and `positions[i]` are the step-boundary token
    positions within it to supervise entropy at (typically all of them).
    For simplicity/memory this processes one sample per forward pass and
    accumulates gradients over `batch_size` samples per optimizer step,
    matching the paper's batch size of 256 without requiring padding a
    heterogeneous-length, multi-image batch into one tensor.
    """
    device = handle.device
    handle.model.requires_grad_(False)  # only S is optimized; freeze the target model
    W_decoder = sae.reasoning_vectors().to(device)  # (D, d), frozen
    D = W_decoder.shape[0]

    S = torch.zeros(D, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([S], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

    flat_samples = [(b, p) for b, ps in zip(batches, positions) for p in ps]
    history = []

    with torch.enable_grad():
        for it in range(num_iterations):
            idx = torch.randint(0, len(flat_samples), (min(batch_size, len(flat_samples)),))
            optimizer.zero_grad(set_to_none=True)

            total_entropy = 0.0
            for j in idx.tolist():
                inputs, pos = flat_samples[j]
                delta = S @ W_decoder  # (d,)

                logits = _forward_with_position_edit(handle, inputs, layer, pos, delta)
                logp = F.log_softmax(logits.float(), dim=-1)
                p = logp.exp()
                entropy = -(p * logp).sum()
                (entropy / len(idx)).backward()
                total_entropy += entropy.item()

            optimizer.step()
            scheduler.step()
            history.append(total_entropy / max(1, len(idx)))

    return EntropyVectorResult(S=S.detach().cpu(), vector=(S.detach() @ W_decoder).cpu(), history=history)


def _forward_with_position_edit(handle: ModelHandle, inputs: dict, layer: int, position: int, delta: torch.Tensor) -> torch.Tensor:
    """Run the model with `delta` added to the residual stream at
    (layer, position), returning next-token logits at that position."""
    layers = _get_decoder_layers(handle.model)

    def hook(module, hook_inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hidden = hidden.clone()
        hidden[:, position, :] = hidden[:, position, :] + delta.to(hidden.dtype)
        return (hidden,) + output[1:] if is_tuple else hidden

    h = layers[layer].register_forward_hook(hook)
    try:
        out = handle.model(**inputs, use_cache=False)
    finally:
        h.remove()
    return out.logits[0, position, :]
