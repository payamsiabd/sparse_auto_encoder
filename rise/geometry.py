"""Visualize and quantify the geometry of the SAE decoder column space
(Sec. 4.3, Fig. 2-3): UMAP projection of reasoning vectors {w_i},
highlighted by human/LLM-annotated behavior, plus per-behavior-pair
silhouette scores.

"Top-active" column selection follows the paper exactly: "the activity
of a channel is measured by the largest magnitude of its latent
feature" -- i.e. for behavior class c, we take, for every step labeled
c, its top-|k| firing latent index (argmax_i z_i over the SAE code),
then aggregate across steps (most-frequent argmax indices) to get the
decoder columns "associated with" that behavior.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch

from .annotate import Annotation, Label
from .sae import SparseAutoencoder


@dataclasses.dataclass
class ColumnAssociation:
    behavior_top_columns: dict[Label, list[int]]        # label -> ranked decoder-column indices
    behavior_column_counts: dict[Label, np.ndarray]      # label -> (D,) count of times each column was top-active


@torch.no_grad()
def associate_columns_with_behaviors(
    sae: SparseAutoencoder,
    activations: torch.Tensor,           # (N, d), same order as `annotations`
    annotations: list[Annotation],
    input_scale: float = 1.0,
    top_k_per_step: int = 1,
) -> ColumnAssociation:
    assert activations.shape[0] == len(annotations)
    z = sae.encode(activations.float() * input_scale)  # (N, D)
    D = z.shape[1]

    counts: dict[Label, np.ndarray] = {}
    for ann, code in zip(annotations, z):
        counts.setdefault(ann.label, np.zeros(D, dtype=np.int64))
        top_idx = torch.topk(code, k=min(top_k_per_step, D)).indices.tolist()
        for i in top_idx:
            if code[i] > 0:
                counts[ann.label][i] += 1

    top_columns = {
        label: [int(i) for i in np.argsort(-c)[: min(50, int((c > 0).sum()))]]
        for label, c in counts.items()
    }
    return ColumnAssociation(behavior_top_columns=top_columns, behavior_column_counts=counts)


def umap_projection(decoder_columns: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0) -> np.ndarray:
    """2-D UMAP embedding of decoder columns using cosine distance, as
    specified in the paper ("we adopt UMAP ... which leverages cosine
    similarity ... particularly appropriate since activation vectors are
    primarily meaningful in terms of their direction")."""
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed)
    return reducer.fit_transform(decoder_columns)


def plot_decoder_geometry(
    embedding: np.ndarray,
    column_association: ColumnAssociation,
    out_path: str,
    labels_to_plot: Optional[list[Label]] = None,
) -> None:
    """Reproduce Fig. 2's panel layout: raw geometry, then one panel per
    behavior with that behavior's top columns highlighted."""
    import matplotlib.pyplot as plt

    labels_to_plot = labels_to_plot or ["reflection", "backtracking", "visual_reflection", "others"]
    present = [l for l in labels_to_plot if l in column_association.behavior_top_columns]
    n_panels = 1 + len(present)

    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    axes[0].scatter(embedding[:, 0], embedding[:, 1], s=6, alpha=0.3, color="lightblue")
    axes[0].set_title("Geometry of SAE")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")

    colors = {"reflection": "firebrick", "backtracking": "darkolivegreen", "visual_reflection": "darkorange", "others": "slateblue"}
    for ax, label in zip(axes[1:], present):
        ax.scatter(embedding[:, 0], embedding[:, 1], s=6, alpha=0.15, color="lightblue")
        idx = column_association.behavior_top_columns[label]
        if idx:
            ax.scatter(embedding[idx, 0], embedding[idx, 1], s=30, color=colors.get(label, "black"))
        ax.set_title(f"w. {label.replace('_', ' ').title()} Highlighted")
        ax.set_xlabel("UMAP-1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def normalized_silhouette_scores(
    decoder_columns: np.ndarray,
    column_labels: dict[Label, list[int]],
    pairs: Optional[list[tuple[Label, Label]]] = None,
) -> dict[str, float]:
    """Reproduce Fig. 3: silhouette score between pairs of behavior-column
    groups (and "overall" across all labeled groups), using cosine
    distance on the raw decoder columns, min-max normalized to [0, 1]
    the same way the paper normalizes "for better visualization" (here
    normalized per call against the observed score range across the
    requested pairs, so multiple `normalized_silhouette_scores` calls --
    one per layer -- can be assembled into the Fig. 3-style curve by the
    caller)."""
    from sklearn.metrics import silhouette_score

    pairs = pairs or [
        ("reflection", "backtracking"),
        ("others", "reflection"),
        ("others", "backtracking"),
        ("reflection", "visual_reflection"),
        ("backtracking", "visual_reflection"),
        ("others", "visual_reflection"),
    ]

    raw_scores: dict[str, float] = {}
    all_idx, all_y = [], []
    for label, idx in column_labels.items():
        all_idx.extend(idx)
        all_y.extend([label] * len(idx))
    if len(set(all_y)) >= 2 and len(all_idx) > len(set(all_y)):
        raw_scores["overall"] = silhouette_score(decoder_columns[all_idx], all_y, metric="cosine")

    for a, b in pairs:
        idx_a, idx_b = column_labels.get(a, []), column_labels.get(b, [])
        if len(idx_a) < 2 or len(idx_b) < 2:
            continue
        idx = idx_a + idx_b
        y = [a] * len(idx_a) + [b] * len(idx_b)
        raw_scores[f"{a} vs {b}"] = silhouette_score(decoder_columns[idx], y, metric="cosine")

    if not raw_scores:
        return raw_scores
    values = np.array(list(raw_scores.values()))
    lo, hi = values.min(), values.max()
    span = max(hi - lo, 1e-8)
    return {k: (v - lo) / span for k, v in raw_scores.items()}
