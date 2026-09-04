"""Statistical (rather than argmax-counting) association between SAE
decoder columns and behavior labels.

``rise.geometry.associate_columns_with_behaviors`` follows the paper's
own methodology (Sec. 4.3: "the activity of a channel is measured by
the largest magnitude of its latent feature") -- for each labeled step,
credit only that step's single top-firing latent. With a modest number
of labeled steps per behavior (typical for this pipeline's train
splits), that throws away most of the signal: a latent that fires
reliably for a behavior but is rarely *the single strongest* one for
any given step gets zero credit, and the resulting top-column lists can
end up small and noisy -- exactly the "clusters still look bad even
though the SAE is now properly sparse" symptom.

This module instead runs a per-latent statistical test -- Mann-Whitney
U, Cohen's d effect size, and ROC-AUC -- comparing a behavior's full
activation *distribution* against every other step's, for every
dictionary column independently, and keeps only columns that pass
significance thresholds. This uses the entire per-step activation
value rather than a single top-1 bit, and is modeled directly on the
statistical machinery a mature SAE-based reasoning study uses to rank
"reasoning features" against a pretrained SAE
(github.com/GeorgeMLP/reasoning-probing, using Gemma Scope +
per-feature Mann-Whitney/Cohen's d/ROC-AUC testing) -- adapted here to
run on the SAE this project trains itself, for any of its behavior
labels rather than a single reasoning/non-reasoning split.

Produces the same ``rise.geometry.ColumnAssociation`` shape
``associate_columns_with_behaviors`` does, so it's a drop-in swap
everywhere that consumes one (geometry plots, ``build_behavior_vector``,
``predict_label``): ``configs/default.yaml``'s ``geometry.method``
picks which builder each script uses.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import torch

from .annotate import Annotation, Label
from .geometry import ColumnAssociation, associate_columns_with_behaviors
from .sae import SparseAutoencoder


@dataclasses.dataclass
class FeatureStats:
    feature_index: int
    mean_target: float
    mean_other: float
    cohens_d: float           # pooled-std standardized mean difference
    roc_auc: float             # separability of target vs. other by this latent's activation alone
    mannwhitney_p: float       # two-sided Mann-Whitney U test p-value
    freq_active_target: float  # fraction of target steps with activation > 1% of this latent's max
    freq_active_other: float

    def is_significant(self, min_auc: float, max_p: float, min_effect: float) -> bool:
        return (
            self.roc_auc >= min_auc
            and self.mannwhitney_p <= max_p
            and abs(self.cohens_d) >= min_effect
            and self.mean_target > self.mean_other
        )


def compute_feature_stats(code: np.ndarray, is_target: np.ndarray) -> list[FeatureStats]:
    """One ``FeatureStats`` per dictionary column (code.shape[1] of
    them), comparing rows where ``is_target`` is True against the rest.

    Args:
        code: (N, D) SAE activations (``sae.encode(...)`` output), one
            row per reasoning step.
        is_target: (N,) boolean mask, True for steps belonging to the
            behavior of interest.
    """
    n_target = int(is_target.sum())
    n_other = int((~is_target).sum())
    if n_target < 2 or n_other < 2:
        raise ValueError(
            f"Need >= 2 samples in both the target and 'other' groups for "
            f"a meaningful statistical test (got {n_target} target / {n_other} other)."
        )

    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    D = code.shape[1]
    labels_int = is_target.astype(int)
    stats = []
    for j in range(D):
        acts = code[:, j]
        target_acts = acts[is_target]
        other_acts = acts[~is_target]

        mean_t, mean_o = float(target_acts.mean()), float(other_acts.mean())
        std_t, std_o = float(target_acts.std()), float(other_acts.std())
        pooled_std = np.sqrt(
            ((n_target - 1) * std_t ** 2 + (n_other - 1) * std_o ** 2) / (n_target + n_other - 2)
        )
        cohens_d = (mean_t - mean_o) / (pooled_std + 1e-10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, p_value = mannwhitneyu(target_acts, other_acts, alternative="two-sided")
            except ValueError:
                p_value = 1.0  # e.g. all values identical (both groups all-zero)

        try:
            auc = roc_auc_score(labels_int, acts)
        except ValueError:
            auc = 0.5  # e.g. this latent never fires for anyone

        max_act = max(float(acts.max()), 1e-10)
        threshold = 0.01 * max_act
        freq_t = float((target_acts > threshold).mean())
        freq_o = float((other_acts > threshold).mean())

        stats.append(FeatureStats(
            feature_index=j, mean_target=mean_t, mean_other=mean_o,
            cohens_d=float(cohens_d), roc_auc=float(auc), mannwhitney_p=float(p_value),
            freq_active_target=freq_t, freq_active_other=freq_o,
        ))
    return stats


@torch.no_grad()
def associate_columns_by_stats(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    annotations: list[Annotation],
    input_scale: float = 1.0,
    min_auc: float = 0.6,
    min_effect: float = 0.3,
    max_p: float = 0.01,
    bonferroni: bool = True,
    top_k: int = 50,
) -> ColumnAssociation:
    """Statistical alternative to
    ``rise.geometry.associate_columns_with_behaviors`` -- see this
    module's docstring for why. Runs one label-vs-rest test per label
    present in ``annotations``, over every SAE dictionary column.

    ``behavior_column_counts[label]`` holds each column's ROC-AUC where
    it passed significance (``FeatureStats.is_significant``) and 0.0
    otherwise, so downstream code that does ``np.argsort(-counts)`` or
    ``(counts > 0).sum()`` (exactly what ``rise.geometry``'s plotting,
    ``build_behavior_vector``, and ``predict_label`` all do) ranks and
    filters columns the same way it would with the argmax-based counts,
    without needing to know which method produced them.

    Thresholds default to the same values the reference implementation
    uses (AUC >= 0.6, effect size >= 0.3); ``max_p`` is Bonferroni-
    corrected across all D columns by default (``bonferroni=True``) --
    testing thousands of columns at an uncorrected p <= 0.01 would
    otherwise pass ~1% of them by chance alone.
    """
    assert activations.shape[0] == len(annotations)
    z = sae.encode(activations.float() * input_scale).numpy()
    D = z.shape[1]
    labels: list[Label] = sorted({a.label for a in annotations})
    p_threshold = max_p / D if bonferroni else max_p

    top_columns: dict[Label, list[int]] = {}
    scores: dict[Label, np.ndarray] = {}
    for label in labels:
        is_target = np.array([a.label == label for a in annotations])
        score = np.zeros(D, dtype=np.float64)
        if is_target.sum() >= 2 and (~is_target).sum() >= 2:
            for stat in compute_feature_stats(z, is_target):
                if stat.is_significant(min_auc, p_threshold, min_effect):
                    score[stat.feature_index] = stat.roc_auc
        scores[label] = score
        n_significant = int((score > 0).sum())
        top_columns[label] = [int(i) for i in np.argsort(-score)[: min(top_k, n_significant)]]

    return ColumnAssociation(behavior_top_columns=top_columns, behavior_column_counts=scores)


def build_association(
    geo_cfg: dict,
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    annotations: list[Annotation],
    input_scale: float = 1.0,
) -> ColumnAssociation:
    """Single dispatch point used by scripts/05-08: builds a
    ``ColumnAssociation`` via whichever method ``geo_cfg["method"]``
    (``configs/default.yaml``'s ``geometry.method``) names, so every
    script that needs one picks up config changes uniformly instead of
    each re-implementing the if/else."""
    method = geo_cfg.get("method", "stats")
    if method == "stats":
        return associate_columns_by_stats(
            sae, activations, annotations, input_scale=input_scale,
            min_auc=geo_cfg["stats_min_auc"], min_effect=geo_cfg["stats_min_effect"],
            max_p=geo_cfg["stats_max_p"], bonferroni=geo_cfg["stats_bonferroni"],
            top_k=geo_cfg["stats_top_k"],
        )
    elif method == "argmax":
        return associate_columns_with_behaviors(
            sae, activations, annotations, input_scale=input_scale,
            top_k_per_step=geo_cfg["top_k_per_step"],
        )
    else:
        raise ValueError(f"Unknown geometry.method: {method!r} (expected 'stats' or 'argmax')")
