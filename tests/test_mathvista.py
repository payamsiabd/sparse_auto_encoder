"""Validates `rise.mathvista`'s dataset-conversion logic against an
in-memory fake dataset shaped like MathVista's documented schema (a
list of dicts with `pid`, `question`, `query`, `choices`, `answer`,
`question_type`, `metadata.task`, and a PIL `decoded_image`), so the
export path -- image saving, prompts.jsonl writing, query/choices
fallback, task filtering, subsampling -- is checked without needing
network access to Hugging Face. Doesn't touch `download_mathvista`
itself (the `datasets.load_dataset` call), which needs the real network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.dataset import load_prompts
from rise.mathvista import build_query, export_rows, export_train_test_split


def _fake_row(pid: str, task: str, question_type: str, with_query: bool, with_choices: bool) -> dict:
    img = Image.new("RGB", (8, 8), color=(int(pid) % 256, 0, 0))
    row = {
        "pid": pid,
        "question": f"What is shown in figure {pid}?",
        "decoded_image": img,
        "answer": "42",
        "question_type": question_type,
        "metadata": {"task": task},
    }
    if with_choices:
        row["choices"] = ["10", "42", "7"]
    if with_query:
        row["query"] = f"Full formatted MathVista prompt for {pid}."
    return row


def test_build_query_prefers_query_field() -> None:
    row = _fake_row("1", "geometry problem solving", "free_form", with_query=True, with_choices=False)
    assert build_query(row) == row["query"]


def test_build_query_falls_back_to_choices() -> None:
    row = _fake_row("2", "geometry problem solving", "multi_choice", with_query=False, with_choices=True)
    q = build_query(row)
    assert "(A) 10" in q and "(B) 42" in q and "(C) 7" in q
    assert "Answer with the letter" in q


def test_build_query_falls_back_to_free_form() -> None:
    row = _fake_row("3", "geometry problem solving", "free_form", with_query=False, with_choices=False)
    q = build_query(row)
    assert row["question"] in q


def test_export_rows_writes_prompts_and_images(tmp_path: Path = None) -> None:
    import tempfile

    rows = [
        _fake_row("100", "chart question answering", "free_form", with_query=True, with_choices=False),
        _fake_row("101", "geometry problem solving", "multi_choice", with_query=False, with_choices=True),
        _fake_row("102", "algebraic reasoning", "free_form", with_query=True, with_choices=False),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        stats = export_rows(rows, out_dir)

        assert stats.num_written == 3
        assert stats.task_counts == {
            "chart question answering": 1, "geometry problem solving": 1, "algebraic reasoning": 1,
        }
        assert (out_dir / "images" / "100.png").exists()
        assert (out_dir / "images" / "101.png").exists()
        assert (out_dir / "images" / "102.png").exists()

        lines = (out_dir / "prompts.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3
        rec = json.loads(lines[0])
        assert rec["id"] == "100"
        assert rec["image"] == "images/100.png"
        assert rec["answer"] == "42"
        assert "look back at the image" in rec["system_prompt"]

        # Round-trips through the actual loader the rest of the pipeline uses.
        prompts = load_prompts(out_dir / "prompts.jsonl")
        assert len(prompts) == 3
        assert prompts[0].id == "100"
        assert prompts[0].images[0].exists()
        assert prompts[0].question == rows[0]["query"]
        assert prompts[0].reference_answer == "42"


def test_export_rows_task_filter_and_subsample() -> None:
    import tempfile

    rows = [
        _fake_row(str(i), "chart question answering" if i % 2 == 0 else "algebraic reasoning",
                  "free_form", with_query=True, with_choices=False)
        for i in range(10)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        stats = export_rows(rows, out_dir, task_filter=["chart question answering"])
        assert stats.num_written == 5
        assert set(stats.task_counts) == {"chart question answering"}

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        stats = export_rows(rows, out_dir, num_samples=3, seed=0)
        assert stats.num_written == 3


def test_export_train_test_split_is_disjoint_and_self_contained() -> None:
    import tempfile

    rows = [
        _fake_row(str(i), "chart question answering", "free_form", with_query=True, with_choices=False)
        for i in range(20)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        split = export_train_test_split(rows, out_dir, train_frac=0.75, seed=0)

        assert split.train.num_written == 15
        assert split.test.num_written == 5

        train_prompts = load_prompts(out_dir / "train" / "prompts.jsonl")
        test_prompts = load_prompts(out_dir / "test" / "prompts.jsonl")
        assert len(train_prompts) == 15
        assert len(test_prompts) == 5

        # Disjoint ids, and every image path actually resolves (each
        # split has its own self-contained images/ directory).
        train_ids = {p.id for p in train_prompts}
        test_ids = {p.id for p in test_prompts}
        assert train_ids.isdisjoint(test_ids)
        assert train_ids | test_ids == {str(i) for i in range(20)}
        assert all(p.images[0].exists() for p in train_prompts)
        assert all(p.images[0].exists() for p in test_prompts)


def test_export_train_test_split_rejects_degenerate_fractions() -> None:
    import tempfile

    rows = [_fake_row(str(i), "chart question answering", "free_form", True, False) for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            export_train_test_split(rows, Path(tmp), train_frac=0.99)
            raised = False
        except ValueError:
            raised = True
        assert raised, "expected a ValueError when a split would be empty"


if __name__ == "__main__":
    test_build_query_prefers_query_field()
    print("test_build_query_prefers_query_field: OK")
    test_build_query_falls_back_to_choices()
    print("test_build_query_falls_back_to_choices: OK")
    test_build_query_falls_back_to_free_form()
    print("test_build_query_falls_back_to_free_form: OK")
    test_export_rows_writes_prompts_and_images()
    print("test_export_rows_writes_prompts_and_images: OK")
    test_export_rows_task_filter_and_subsample()
    print("test_export_rows_task_filter_and_subsample: OK")
    test_export_train_test_split_is_disjoint_and_self_contained()
    print("test_export_train_test_split_is_disjoint_and_self_contained: OK")
    test_export_train_test_split_rejects_degenerate_fractions()
    print("test_export_train_test_split_rejects_degenerate_fractions: OK")
