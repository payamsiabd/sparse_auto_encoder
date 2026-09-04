"""Validates `rise.feature_stats`'s statistical column-behavior
association against synthetic data with a known ground truth: a few
dictionary columns genuinely discriminate between two groups, the rest
are pure noise, and the module should identify exactly the former.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.annotate import Annotation
from rise.feature_stats import associate_columns_by_stats, compute_feature_stats
from rise.sae import SparseAutoencoder


def _make_synthetic_code(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """10 latent columns, 200 samples (100/100 split). Columns 0 and 1
    are strongly higher for the "target" group; columns 2-9 are pure
    noise with no true group difference."""
    rng = np.random.default_rng(seed)
    n_target, n_other, D = 100, 100, 10

    code = np.zeros((n_target + n_other, D))
    code[:n_target, 0] = rng.normal(5.0, 1.0, n_target).clip(min=0)
    code[n_target:, 0] = rng.normal(0.0, 1.0, n_other).clip(min=0)
    code[:n_target, 1] = rng.normal(2.0, 1.0, n_target).clip(min=0)
    code[n_target:, 1] = rng.normal(0.0, 1.0, n_other).clip(min=0)
    for j in range(2, D):
        code[:, j] = rng.normal(1.0, 1.0, n_target + n_other).clip(min=0)

    is_target = np.array([True] * n_target + [False] * n_other)
    return code, is_target


def test_compute_feature_stats_separates_signal_from_noise() -> None:
    code, is_target = _make_synthetic_code()
    stats = compute_feature_stats(code, is_target)

    assert len(stats) == code.shape[1]
    assert stats[0].roc_auc > 0.9, "column 0 (strong signal) should be near-perfectly separable"
    assert stats[0].mannwhitney_p < 1e-6
    assert stats[0].cohens_d > 1.0
    assert stats[0].is_significant(min_auc=0.6, max_p=0.01, min_effect=0.3)

    assert stats[1].roc_auc > 0.7, "column 1 (moderate signal) should still separate"
    assert stats[1].is_significant(min_auc=0.6, max_p=0.01, min_effect=0.3)

    for j in range(2, len(stats)):
        assert not stats[j].is_significant(min_auc=0.6, max_p=0.001, min_effect=0.3), (
            f"noise column {j} should not pass a strict significance threshold "
            f"(auc={stats[j].roc_auc:.3f}, p={stats[j].mannwhitney_p:.4f})"
        )


def test_compute_feature_stats_requires_both_groups() -> None:
    code = np.random.default_rng(0).normal(size=(3, 3)).clip(min=0)
    is_target = np.array([True, False, False])  # only 1 target sample
    try:
        compute_feature_stats(code, is_target)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_associate_columns_by_stats_identifies_signal_columns() -> None:
    code, is_target = _make_synthetic_code()
    D = code.shape[1]

    # Identity encoder (k=D, i.e. no top-k filtering): sae.encode(h) ==
    # relu(h) == h for our nonnegative synthetic code, so the SAE layer
    # is transparent and associate_columns_by_stats operates on exactly
    # the `code` array constructed above.
    sae = SparseAutoencoder(d_in=D, d_hidden=D, activation="topk", k=D, tied_init=False)
    with torch.no_grad():
        sae.W_encoder.copy_(torch.eye(D))
        sae.b_encoder.zero_()

    annotations = [
        Annotation("p1", i, "x", "reflection" if t else "others")
        for i, t in enumerate(is_target)
    ]
    activations = torch.tensor(code, dtype=torch.float32)

    assoc = associate_columns_by_stats(
        sae, activations, annotations, min_auc=0.6, min_effect=0.3, max_p=0.01, bonferroni=True,
    )

    assert 0 in assoc.behavior_top_columns["reflection"]
    assert 1 in assoc.behavior_top_columns["reflection"]
    noise_columns_flagged = [c for c in assoc.behavior_top_columns["reflection"] if c >= 2]
    assert not noise_columns_flagged, f"noise columns incorrectly flagged as significant: {noise_columns_flagged}"

    # ColumnAssociation shape must match rise.geometry's -- same dict
    # structure, values usable the same way (argsort descending, >0 filter).
    assert set(assoc.behavior_column_counts.keys()) == {"reflection", "others"}
    assert assoc.behavior_column_counts["reflection"].shape == (D,)


if __name__ == "__main__":
    test_compute_feature_stats_separates_signal_from_noise()
    print("test_compute_feature_stats_separates_signal_from_noise: OK")
    test_compute_feature_stats_requires_both_groups()
    print("test_compute_feature_stats_requires_both_groups: OK")
    test_associate_columns_by_stats_identifies_signal_columns()
    print("test_associate_columns_by_stats_identifies_signal_columns: OK")
