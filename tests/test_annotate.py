"""Validates `rise.annotate`'s pure-Python logic: the keyword
annotator's precedence rules, and `to_binary_labels`'s collapse of the
4-class taxonomy into a reflection/others view (used as the primary
geometry visualization in `scripts/05_annotate_and_visualize.py`)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.annotate import Annotation, KeywordAnnotator, agreement_ratio, to_binary_labels


def test_keyword_annotator_labels() -> None:
    ann = KeywordAnnotator()
    assert ann.annotate("Wait, let me double-check this calculation.") == "reflection"
    assert ann.annotate("Alternatively, let's try another approach entirely.") == "backtracking"
    assert ann.annotate("Looking at the image again, the bar is actually taller.") == "visual_reflection"
    assert ann.annotate("So the final answer is 42.") == "others"


def test_to_binary_labels_collapses_reflective_classes() -> None:
    annotations = [
        Annotation("p1", 0, "reflection step", "reflection"),
        Annotation("p1", 1, "backtracking step", "backtracking"),
        Annotation("p1", 2, "visual step", "visual_reflection"),
        Annotation("p1", 3, "plain step", "others"),
    ]
    binary = to_binary_labels(annotations)

    assert [a.label for a in binary] == ["reflection", "reflection", "reflection", "others"]
    # Original annotations must be untouched (dataclasses.replace, not mutation).
    assert annotations[1].label == "backtracking"
    # Identity/order-preserving fields carried through unchanged.
    assert [a.step_index for a in binary] == [0, 1, 2, 3]
    assert [a.step_text for a in binary] == [a.step_text for a in annotations]


def test_to_binary_labels_custom_grouping() -> None:
    annotations = [
        Annotation("p1", 0, "x", "reflection"),
        Annotation("p1", 1, "x", "visual_reflection"),
        Annotation("p1", 2, "x", "backtracking"),
    ]
    # Only visual_reflection counts as "reflection" in this grouping.
    binary = to_binary_labels(annotations, reflective_labels=("visual_reflection",))
    assert [a.label for a in binary] == ["others", "reflection", "others"]


def test_agreement_ratio_on_binary_vs_finegrained() -> None:
    fine = [
        Annotation("p1", 0, "x", "reflection"),
        Annotation("p1", 1, "x", "backtracking"),
        Annotation("p1", 2, "x", "others"),
    ]
    same_binary_family = [
        Annotation("p1", 0, "x", "visual_reflection"),  # still "reflection" once binarized
        Annotation("p1", 1, "x", "reflection"),           # still "reflection" once binarized
        Annotation("p1", 2, "x", "others"),
    ]
    # Disagree at fine granularity (reflection != visual_reflection, backtracking != reflection)...
    assert agreement_ratio(fine, same_binary_family) == 1 / 3
    # ...but fully agree once both are collapsed to the binary view.
    assert agreement_ratio(to_binary_labels(fine), to_binary_labels(same_binary_family)) == 1.0


if __name__ == "__main__":
    test_keyword_annotator_labels()
    print("test_keyword_annotator_labels: OK")
    test_to_binary_labels_collapses_reflective_classes()
    print("test_to_binary_labels_collapses_reflective_classes: OK")
    test_to_binary_labels_custom_grouping()
    print("test_to_binary_labels_custom_grouping: OK")
    test_agreement_ratio_on_binary_vs_finegrained()
    print("test_agreement_ratio_on_binary_vs_finegrained: OK")
