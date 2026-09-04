"""Small helpers shared across the RISE-for-Qwen3-VL pipeline."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Thinking"

# Qwen3-VL-Thinking, like other "thinking" chat models, wraps its
# reasoning trace in <think>...</think> before the final answer.
THINK_START = "<think>"
THINK_END = "</think>"

# Step delimiter used throughout the paper (Sec. 3.2): reasoning traces
# are segmented into sentence-level "steps" at every "\n\n".
STEP_DELIMITER = "\n\n"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


@dataclass
class ModelHandle:
    model: Any
    processor: Any
    num_layers: int
    hidden_size: int
    device: torch.device


def load_qwen3_vl(
    model_id: str = DEFAULT_MODEL_ID,
    dtype: str = "bf16",
    device_map: str | dict = "auto",
    attn_implementation: str | None = "sdpa",
) -> ModelHandle:
    """Load Qwen3-VL-4B-Thinking (or any Qwen3-VL checkpoint) plus its
    processor, and auto-detect the language-model hidden size / layer
    count from the (nested, vision+text) model config so the rest of the
    pipeline never hardcodes architecture-specific numbers.

    Requires a `transformers` build with Qwen3-VL support
    (``pip install -U transformers`` or build from source -- Qwen3-VL is
    a very recent architecture at the time of writing).
    """
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls  # generic fallback

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    kwargs = dict(dtype=resolve_dtype(dtype), device_map=device_map, trust_remote_code=True)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = ModelCls.from_pretrained(model_id, **kwargs)
    except (TypeError, ImportError, ValueError):
        # Older transformers: `dtype` kwarg was `torch_dtype`; flash-attn may
        # not be installed. Fall back to the broadly-compatible call.
        kwargs.pop("attn_implementation", None)
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = ModelCls.from_pretrained(model_id, **kwargs)
    model.eval()

    hidden_size, num_layers = _detect_text_backbone_shape(model)
    device = next(model.parameters()).device

    return ModelHandle(
        model=model, processor=processor,
        num_layers=num_layers, hidden_size=hidden_size, device=device,
    )


def _detect_text_backbone_shape(model: Any) -> tuple[int, int]:
    """Qwen3-VL's config is a composite of `vision_config` and
    `text_config`. Different transformers versions have exposed this
    slightly differently over time, so probe defensively rather than
    hardcoding a single attribute path."""
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg

    hidden_size = getattr(text_cfg, "hidden_size", None) or getattr(cfg, "hidden_size", None)
    num_layers = (
        getattr(text_cfg, "num_hidden_layers", None)
        or getattr(cfg, "num_hidden_layers", None)
    )
    if hidden_size is None or num_layers is None:
        raise ValueError(
            "Could not auto-detect hidden_size / num_hidden_layers from "
            "model.config. Inspect `model.config` and pass them explicitly "
            "to downstream calls."
        )
    return int(hidden_size), int(num_layers)


def split_thinking_and_answer(generated_text: str) -> tuple[str, str]:
    """Qwen3-VL-Thinking emits `<think>...</think>` before the final
    answer. Returns (thinking_text, answer_text). If no <think> tags are
    present (e.g. a non-thinking checkpoint, or the model skipped
    reasoning), the whole text is treated as the answer with empty
    thinking, mirroring how the paper falls back to using the full
    response when there is no explicit reasoning/answer boundary."""
    if THINK_START in generated_text and THINK_END in generated_text:
        start = generated_text.index(THINK_START) + len(THINK_START)
        end = generated_text.index(THINK_END, start)
        thinking = generated_text[start:end].strip()
        answer = generated_text[end + len(THINK_END):].strip()
        return thinking, answer
    return "", generated_text.strip()


def split_into_steps(thinking_text: str) -> list[str]:
    """Segment a reasoning trace into sentence-level steps at every
    occurrence of the step delimiter (Sec. 3.2).

    Only delimiter-*terminated* chunks are returned: Sec. 3.2 assigns
    each step the activation of its trailing "\\n\\n" token, so a
    trailing chunk with no following delimiter (e.g. the last bit of
    text before the reasoning trace simply ends) has no such token to
    extract and is dropped here -- this keeps `split_into_steps`'s
    output exactly the set of steps `activations.locate_step_tokens`
    can find a "<\\n\\n>" position for. Empty steps produced by
    leading/duplicate delimiters are dropped too.
    """
    pieces = thinking_text.split(STEP_DELIMITER)
    if not thinking_text.endswith(STEP_DELIMITER):
        pieces = pieces[:-1]
    return [s for s in pieces if s.strip()]
