"""Behavior annotation for reasoning steps.

Reproduces the paper's LLM-as-judge + keyword-matching annotation
methodology (Sec. 4.3, Appendix D), extended with a fourth class,
``visual_reflection``, for the downstream goal of this project: finding
reasoning steps where the model revisits / re-attends to the image
rather than (or in addition to) its own prior text reasoning.

Paper taxonomy (3-class):
    reflection   -- step checks its previous reasoning and states
                    uncertainty about it.
    backtracking -- step explicitly retracts/pivots, proposing an
                    alternative strategy to replace the current one.
    others       -- anything else.

Added class:
    visual_reflection -- step explicitly re-examines, re-describes, or
                    re-checks specific visual evidence in the image(s)
                    (as opposed to re-checking a purely symbolic/logical
                    step). E.g. "Let me look at the image again", "Wait,
                    I should re-check the chart", "Looking more closely
                    at the picture, the object is actually on the left".
                    A step can be both textually reflective *and* visual
                    (e.g. "Wait, looking at the image again, ..."); the
                    judge is instructed to prefer ``visual_reflection``
                    in that case since it is the more specific label and
                    the one this project cares about isolating.

Two interchangeable annotators are provided, exactly mirroring the
paper's consistency check (Appendix D, Fig. 10):
  - ``KeywordAnnotator``: fast, offline, deterministic (Table 3 + visual
    cues added).
  - ``LLMJudgeAnnotator``: any OpenAI-chat-completions-compatible client
    (works with OpenAI, or a locally hosted judge model), using a
    prompt that is a direct extension of the paper's Appendix D prompt.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Callable, Iterable, Literal

Label = Literal["reflection", "backtracking", "visual_reflection", "others"]
LABELS: tuple[Label, ...] = ("reflection", "backtracking", "visual_reflection", "others")


@dataclasses.dataclass
class Annotation:
    prompt_id: str
    step_index: int
    step_text: str
    label: Label


# -- Table 3 (paper) + visual-reflection cues added --------------------------
REFLECTION_KEYWORDS = [
    "wait", "verify", "make sure", "hold on", "think again",
    "'s correct", "'s incorrect", "let me check", "seems right",
]
BACKTRACKING_KEYWORDS = [
    "alternatively", "think differently", "another way", "another approach",
    "another method", "another solution", "another strategy", "another technique",
]
VISUAL_REFLECTION_KEYWORDS = [
    "the image", "the picture", "the photo", "the figure", "the chart",
    "the graph", "the diagram", "the screenshot", "in the image",
    "looking at the image", "look at the image again", "looking again at",
    "re-examine the image", "zoom in", "looking more closely",
    "as shown in the image", "the visual", "in the picture",
    "look back at the image", "checking the image", "the region",
    "the bounding box", "upon closer inspection",
]


class KeywordAnnotator:
    """Deterministic offline annotator. Order of precedence when a step
    matches multiple categories: visual_reflection > reflection >
    backtracking > others (visual_reflection is the most specific / most
    informative for this project's downstream analysis, and a step that
    is both textually and visually reflective should be counted as the
    latter, mirroring how the LLM judge is instructed below)."""

    def annotate(self, step_text: str) -> Label:
        text = step_text.lower()
        has_visual = any(k in text for k in VISUAL_REFLECTION_KEYWORDS)
        has_reflection = any(k in text for k in REFLECTION_KEYWORDS)
        has_backtracking = any(k in text for k in BACKTRACKING_KEYWORDS)

        if has_visual and (has_reflection or "look" in text or "image" in text or "picture" in text):
            return "visual_reflection"
        if has_reflection:
            return "reflection"
        if has_backtracking:
            return "backtracking"
        return "others"


JUDGE_SYSTEM_PROMPT = """\
You are a helpful expert that is good at classifying reasoning steps.
You will be given a single reasoning step from a visual
question-answering / multimodal math or logic solution (the model can
see one or more images in addition to the text shown to you).
Your task is to classify the reasoning step according to the provided
taxonomy and decision rules.

The available labels are:
(1) reflection: step checking its previous reasoning process and
    stating its own uncertainty, WITHOUT referring back to specific
    visual content of the image(s).
(2) backtracking: steps that explicitly retract/pivot, proposing an
    alternative strategy to replace the current one.
(3) visual_reflection: step that explicitly revisits, re-examines,
    re-describes, or double-checks specific visual evidence in the
    image(s) (e.g. re-reading a label, re-counting objects, re-checking
    a color/position/value shown in the image, noticing a previous
    visual misreading). If a step is both a general reflection AND
    specifically about re-checking the image, label it
    visual_reflection, not reflection.
(4) others: steps that do not fall into the above three categories.

You must select the label based on the above criteria and decision
rules and assign a single class for each step.

Your output should be a strict label from the four options:
"reflection", "backtracking", "visual_reflection", or "others".
If you cannot determine the label, please assign "others".

Now, please classify the following reasoning step delimited by triple
backticks, according to the taxonomy and decision rules provided in the
system prompt.
Reasoning Step: '''{text}'''
"""


class LLMJudgeAnnotator:
    """Wraps any OpenAI-chat-completions-compatible client.

    Example::

        from openai import OpenAI
        client = OpenAI()
        judge = LLMJudgeAnnotator(lambda prompt: client.chat.completions.create(
            model="gpt-5", messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content)
    """

    def __init__(self, call_fn: Callable[[str], str]):
        self._call_fn = call_fn

    def annotate(self, step_text: str) -> Label:
        prompt = JUDGE_SYSTEM_PROMPT.format(text=step_text)
        raw = self._call_fn(prompt).strip().lower()
        for label in LABELS:
            if label in raw:
                return label
        return "others"


def annotate_steps(
    steps_metadata: Iterable[dict],
    annotator,
    show_progress: bool = True,
) -> list[Annotation]:
    """Run ``annotator`` (a ``KeywordAnnotator`` or ``LLMJudgeAnnotator``,
    or anything exposing ``.annotate(step_text) -> Label``) over every
    step in ``steps_metadata`` (as produced by
    ``rise.store.load_steps_metadata``)."""
    steps_metadata = list(steps_metadata)
    iterator = steps_metadata
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(steps_metadata, desc="annotating steps")
        except ImportError:
            pass

    results = []
    for rec in iterator:
        label = annotator.annotate(rec["step_text"])
        results.append(Annotation(
            prompt_id=rec["prompt_id"], step_index=rec["step_index"],
            step_text=rec["step_text"], label=label,
        ))
    return results


def save_annotations(annotations: list[Annotation], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(dataclasses.asdict(a)) + "\n")


def load_annotations(path: str) -> list[Annotation]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Annotation(**json.loads(line)))
    return out


def to_binary_labels(
    annotations: list[Annotation],
    reflective_labels: tuple[Label, ...] = ("reflection", "backtracking", "visual_reflection"),
) -> list[Annotation]:
    """Collapse the fine-grained 4-class taxonomy into two groups:
    "reflection" (any reasoning-interruption behavior -- reflection,
    backtracking, and visual_reflection are all, at this coarser
    resolution, "the model pausing to re-check or reconsider something")
    vs "others" (ordinary forward-moving reasoning).

    Useful as the primary decoder-geometry view: a dense/under-trained
    SAE, or simply too little data, can make the 4-way split look noisy
    even when reflection-vs-not is cleanly separable, since 3 of the 4
    classes are lumped together and diluted across sub-types. Checking
    the binary split first answers "does this behavior cluster at all"
    before "which sub-type does it cluster into" -- see
    `scripts/05_annotate_and_visualize.py`, which produces both.
    """
    return [
        dataclasses.replace(a, label=("reflection" if a.label in reflective_labels else "others"))
        for a in annotations
    ]


def agreement_ratio(a: list[Annotation], b: list[Annotation]) -> float:
    """Fraction of steps (matched by (prompt_id, step_index)) where two
    annotation sets agree -- reproduces the metric used in Fig. 10."""
    key = lambda x: (x.prompt_id, x.step_index)
    b_map = {key(x): x.label for x in b}
    matched = [x for x in a if key(x) in b_map]
    if not matched:
        return float("nan")
    agree = sum(1 for x in matched if b_map[key(x)] == x.label)
    return agree / len(matched)
