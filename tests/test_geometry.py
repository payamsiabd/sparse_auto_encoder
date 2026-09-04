"""Validates the parts of `rise.geometry` that don't need UMAP/sklearn
or a real SAE: persisting a `ColumnAssociation` to disk and back, and
`predict_label`'s post-hoc classification logic (used by
`scripts/08_evaluate_on_test.py` to score the discovered
`visual_reflection` columns on held-out data)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.geometry import ColumnAssociation, load_association, predict_label, save_association


def _fake_association() -> ColumnAssociation:
    # 8 latent dictionary; columns 0,1 -> reflection; 2,3 -> backtracking;
    # 4,5 -> visual_reflection; the rest belong to no behavior.
    counts = {
        "reflection": np.array([5, 3, 0, 0, 0, 0, 0, 0]),
        "backtracking": np.array([0, 0, 4, 2, 0, 0, 0, 0]),
        "visual_reflection": np.array([0, 0, 0, 0, 6, 1, 0, 0]),
        "others": np.array([0, 0, 0, 0, 0, 0, 1, 1]),
    }
    top_columns = {label: [int(i) for i in np.argsort(-c) if c[i] > 0] for label, c in counts.items()}
    return ColumnAssociation(behavior_top_columns=top_columns, behavior_column_counts=counts)


def test_save_and_load_association_round_trips() -> None:
    assoc = _fake_association()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "assoc.json"
        save_association(assoc, path)
        loaded = load_association(path)

    assert loaded.behavior_top_columns == assoc.behavior_top_columns
    for label in assoc.behavior_column_counts:
        assert np.array_equal(loaded.behavior_column_counts[label], assoc.behavior_column_counts[label])


def test_predict_label_matches_dominant_column() -> None:
    assoc = _fake_association()

    code = torch.zeros(8)
    code[4] = 0.9  # a visual_reflection column
    assert predict_label(code, assoc, top_k=1) == "visual_reflection"

    code = torch.zeros(8)
    code[0] = 0.5  # a reflection column
    assert predict_label(code, assoc, top_k=1) == "reflection"


def test_predict_label_falls_back_to_others_when_unclaimed() -> None:
    assoc = _fake_association()
    code = torch.zeros(8)
    code[6] = 0.7  # not claimed by any behavior in _fake_association's top_columns
    assert predict_label(code, assoc, top_k=1) == "others"

    # No active latent at all.
    assert predict_label(torch.zeros(8), assoc, top_k=1) == "others"


def test_predict_label_top_k_majority_vote() -> None:
    assoc = _fake_association()
    code = torch.zeros(8)
    code[4] = 0.9   # visual_reflection
    code[5] = 0.8   # visual_reflection
    code[0] = 0.1   # reflection, but weaker and out of top-2
    assert predict_label(code, assoc, top_k=2) == "visual_reflection"


if __name__ == "__main__":
    test_save_and_load_association_round_trips()
    print("test_save_and_load_association_round_trips: OK")
    test_predict_label_matches_dominant_column()
    print("test_predict_label_matches_dominant_column: OK")
    test_predict_label_falls_back_to_others_when_unclaimed()
    print("test_predict_label_falls_back_to_others_when_unclaimed: OK")
    test_predict_label_top_k_majority_vote()
    print("test_predict_label_top_k_majority_vote: OK")
