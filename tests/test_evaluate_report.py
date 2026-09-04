"""Validates `_classification_report` in scripts/08_evaluate_on_test.py
-- the precision/recall/F1/confusion-matrix computation the held-out
evaluation script uses to score SAE-predicted vs. annotated labels.
Loaded via importlib since `scripts/` isn't a package (filenames start
with a digit, by design, so the pipeline reads as an ordered list)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "evaluate_on_test", REPO_ROOT / "scripts" / "08_evaluate_on_test.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_classification_report = _module._classification_report

LABELS = ["reflection", "backtracking", "visual_reflection", "others"]


def test_classification_report_perfect_agreement() -> None:
    y_true = ["visual_reflection", "reflection", "others", "backtracking"]
    y_pred = list(y_true)
    report, confusion = _classification_report(y_true, y_pred, LABELS)

    assert report["accuracy"] == 1.0
    assert report["total_steps"] == 4
    for label in LABELS:
        if report[label]["support"] > 0:
            assert report[label]["precision"] == 1.0
            assert report[label]["recall"] == 1.0
            assert report[label]["f1"] == 1.0
    assert sum(sum(row) for row in confusion) == 4


def test_classification_report_known_confusion() -> None:
    # 3 true visual_reflection: 2 correctly predicted, 1 predicted as
    # reflection. 1 true reflection, correctly predicted.
    y_true = ["visual_reflection", "visual_reflection", "visual_reflection", "reflection"]
    y_pred = ["visual_reflection", "visual_reflection", "reflection", "reflection"]
    report, confusion = _classification_report(y_true, y_pred, LABELS)

    vr = report["visual_reflection"]
    assert vr["support"] == 3
    assert vr["recall"] == 2 / 3
    assert vr["precision"] == 1.0  # both visual_reflection predictions were correct

    refl = report["reflection"]
    assert refl["support"] == 1
    assert refl["recall"] == 1.0
    assert refl["precision"] == 1 / 2  # 2 predicted reflection, only 1 truly was

    assert report["accuracy"] == 3 / 4


def test_classification_report_handles_empty_input() -> None:
    report, confusion = _classification_report([], [], LABELS)
    assert report["total_steps"] == 0
    assert report["accuracy"] == 0.0
    for label in LABELS:
        assert report[label]["support"] == 0
        assert report[label]["precision"] == 0.0


if __name__ == "__main__":
    test_classification_report_perfect_agreement()
    print("test_classification_report_perfect_agreement: OK")
    test_classification_report_known_confusion()
    print("test_classification_report_known_confusion: OK")
    test_classification_report_handles_empty_input()
    print("test_classification_report_handles_empty_input: OK")
