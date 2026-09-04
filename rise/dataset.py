"""Loading (image, question) prompts and building Qwen3-VL chat messages.

The paper trains on step-level activations gathered from many (question,
CoT response) pairs (Sec. 3.2, Sec. 4.2: 500 MATH examples). Here the
prompts are visual-reasoning questions, since the goal is to later find
*visual* reflection points -- steps where the model revisits the image.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class VisualPrompt:
    """One (image(s), question) input.

    ``images`` supports multi-image samples since Qwen3-VL is natively
    multi-image capable and several visual-reflection behaviors
    (comparison, cross-referencing) only show up with >1 image.
    """
    id: str
    images: list[Path]
    question: str
    system_prompt: Optional[str] = None
    reference_answer: Optional[str] = None


def load_prompts(jsonl_path: str | Path, image_root: Optional[str | Path] = None) -> list[VisualPrompt]:
    """Load prompts from a JSONL file with one record per line, e.g.::

        {"id": "chartqa_0001", "image": "images/0001.png",
         "question": "What is the peak value shown in the chart, and in
         which year did it occur?", "answer": "42, 2019"}

    ``image`` may also be ``images`` (a list) for multi-image samples.
    Relative image paths are resolved against ``image_root`` (defaults to
    the JSONL file's own directory).
    """
    jsonl_path = Path(jsonl_path)
    root = Path(image_root) if image_root is not None else jsonl_path.parent

    prompts: list[VisualPrompt] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw_images = rec.get("images", rec.get("image"))
            if raw_images is None:
                raise ValueError(f"{jsonl_path}:{line_num} missing 'image'/'images' field")
            if isinstance(raw_images, str):
                raw_images = [raw_images]
            images = [_resolve_image_path(root, p) for p in raw_images]

            prompts.append(
                VisualPrompt(
                    id=str(rec.get("id", line_num)),
                    images=images,
                    question=rec["question"],
                    system_prompt=rec.get("system_prompt"),
                    reference_answer=rec.get("answer"),
                )
            )
    return prompts


def _resolve_image_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def load_images(prompt: VisualPrompt) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in prompt.images]


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful visual reasoning assistant. Look at the image(s) "
    "closely, think step by step, and refer back to specific visual "
    "details whenever they matter for your answer."
)


def build_messages(prompt: VisualPrompt, images: list[Image.Image]) -> list[dict]:
    """Build a Qwen3-VL chat-template message list: system + one user turn
    containing all images followed by the question text."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt.system_prompt or DEFAULT_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": prompt.question}],
        },
    ]
    return messages
