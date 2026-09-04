"""Download and convert MathVista (AI4Math/MathVista on Hugging Face)
into the `prompts.jsonl` format `rise.dataset.load_prompts` expects, as
the visual-reasoning source dataset for the RISE pipeline.

MathVista (Lu et al., ICLR 2024) is a good fit for finding *visual*
reflection points specifically: many of its problems require reading a
precise value off a chart, diagram, or geometric figure, so a careful
solver has real reason to look back at the image mid-reasoning --
unlike generic VQA, where the answer is usually settled after a single
glance. See `VISUAL_HEAVY_TASKS` below to bias sampling toward those
problem types.

Note on the exact column names: this module was written against the
MathVista schema as documented/commonly used (`pid`, `question`,
`query`, `choices`, `answer`, `question_type`, `metadata.task`, and
either a decoded `decoded_image` column or an `image` column), but it
was not possible to hit huggingface.co from the sandbox this was
authored in to confirm the schema against a live pull (network policy
in that session blocked the host outright). `_get_image` and
`build_query` are written defensively with fallbacks, and the first
thing `download_mathvista` does is print `dataset.column_names` --
if anything below throws a KeyError, check that printout against the
attribute names used here and adjust.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_DATASET_ID = "AI4Math/MathVista"

# Encourages exactly the behavior this project wants to find: treating
# the image as evidence to be re-checked, not just glanced at once. Used
# as the per-sample system_prompt (VisualPrompt.system_prompt), which
# overrides rise.dataset.DEFAULT_SYSTEM_PROMPT for every MathVista example.
MATHVISTA_SYSTEM_PROMPT = (
    "You are a careful visual mathematics assistant solving a MathVista "
    "problem. Think step by step. Whenever a reasoning step depends on a "
    "detail in the image (a number, label, axis value, count, position, "
    "or shape), explicitly look back at the image and re-check that "
    "detail before using it -- say so in your reasoning (e.g. 'Looking "
    "at the image again, ...' or 'Let me re-check the chart: ...'). If "
    "you are unsure about something you read from the image, re-examine "
    "the relevant part rather than guessing. End with a clear final "
    "answer in the format requested by the question."
)

# MathVista's `metadata.task` values that most reward re-checking the
# image (reading precise values, comparing regions, counting) -- a
# reasonable starting point for `task_filter` if you want to boost the
# yield of visual_reflection steps rather than pure symbolic/algebraic
# reasoning that happens to have a picture attached. Pass `None` (the
# default) to keep every task.
VISUAL_HEAVY_TASKS = [
    "figure question answering",
    "chart question answering",
    "table question answering",
    "geometry problem solving",
    "textbook question answering",
    "visual question answering",
]


@dataclasses.dataclass
class MathVistaExportStats:
    prompts_path: Path
    num_written: int
    task_counts: dict[str, int]
    question_type_counts: dict[str, int]


def build_query(row: dict) -> str:
    """MathVista's `query` column (when present) is already the fully
    formatted prompt -- question + choices + answer-format instructions
    -- used by the benchmark's own evaluation harness, so prefer it
    verbatim for a fair, standard prompt. Fall back to assembling one
    from `question`/`choices` if `query` isn't present."""
    if row.get("query"):
        return row["query"]

    question = row["question"]
    choices = row.get("choices")
    if choices:
        lettered = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        return f"{question}\nChoices:\n{lettered}\nAnswer with the letter of the correct choice."
    return f"{question}\nAnswer with a single number or short phrase."


def _get_image(row: dict):
    """Return a PIL.Image for this row. MathVista's HF dataset exposes a
    decoded `decoded_image` (PIL.Image) column and/or a raw `image`
    column (path or bytes, depending on dataset revision/features)."""
    img = row.get("decoded_image")
    if img is None:
        img = row.get("image")
    if img is None:
        raise KeyError(
            f"MathVista row has neither 'decoded_image' nor 'image' "
            f"(available keys: {sorted(row.keys())}); inspect the dataset "
            f"schema and adjust `_get_image`."
        )
    if hasattr(img, "convert"):  # already a PIL.Image
        return img.convert("RGB")

    from PIL import Image
    if isinstance(img, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(img)).convert("RGB")
    return Image.open(img).convert("RGB")  # path-like


def download_mathvista(
    out_dir: str | Path,
    split: str = "testmini",
    num_samples: Optional[int] = 200,
    seed: int = 0,
    task_filter: Optional[list[str]] = None,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> MathVistaExportStats:
    """Fetch MathVista from Hugging Face and write `<out_dir>/prompts.jsonl`
    plus `<out_dir>/images/<pid>.png`, ready for `rise.dataset.load_prompts`.

    Requires network access to huggingface.co and the `datasets` package
    (`pip install datasets`). MathVista is publicly downloadable, no auth
    token needed as of this writing. `split="testmini"` (1,000 examples
    with public ground-truth answers) is the right default for research
    use; `split="test"` (~5,100 examples) has answers withheld for
    leaderboard submission, so `answer` will be null there.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    print(f"Loaded {dataset_id}[{split}]: {len(ds)} rows, columns={ds.column_names}")
    return export_rows(ds, out_dir, num_samples=num_samples, seed=seed, task_filter=task_filter)


def export_rows(
    rows: Iterable[dict],
    out_dir: str | Path,
    num_samples: Optional[int] = None,
    seed: int = 0,
    task_filter: Optional[list[str]] = None,
) -> MathVistaExportStats:
    """Convert an iterable of MathVista-schema rows (a live HF `Dataset`,
    or a plain list of dicts for testing) into `prompts.jsonl` + saved
    images. Split out from `download_mathvista` so the conversion logic
    is testable without any network access -- see `tests/test_mathvista.py`.
    """
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if task_filter:
        rows = [r for r in rows if (r.get("metadata") or {}).get("task") in task_filter]
    if num_samples is not None and num_samples < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:num_samples]

    prompts_path = out_dir / "prompts.jsonl"
    task_counts: dict[str, int] = {}
    qtype_counts: dict[str, int] = {}

    with prompts_path.open("w", encoding="utf-8") as f:
        for row in rows:
            pid = str(row["pid"])
            image = _get_image(row)
            image.save(image_dir / f"{pid}.png")

            meta = row.get("metadata") or {}
            task = meta.get("task", "unknown")
            qtype = row.get("question_type", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
            qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1

            record = {
                "id": pid,
                "image": f"images/{pid}.png",
                "question": build_query(row),
                "answer": row.get("answer"),
                "system_prompt": MATHVISTA_SYSTEM_PROMPT,
                "mathvista_task": task,
                "mathvista_question_type": qtype,
            }
            f.write(json.dumps(record) + "\n")

    stats = MathVistaExportStats(
        prompts_path=prompts_path, num_written=len(rows),
        task_counts=task_counts, question_type_counts=qtype_counts,
    )
    print(f"Wrote {stats.num_written} MathVista prompts to {prompts_path}")
    print(f"Task distribution: {stats.task_counts}")
    print(f"Question-type distribution: {stats.question_type_counts}")
    return stats
