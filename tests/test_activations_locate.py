"""Validates the step-boundary token-localization logic in
`rise.activations.locate_step_tokens` -- specifically, that it correctly
accounts for image-token expansion when computing the position of each
"\\n\\n" delimiter in the *real* (image-expanded) input sequence.

This is the one piece of the pipeline with no direct equivalent in the
paper (DeepSeek-R1-1.5B is text-only, so there is no image expansion to
account for), and the part most likely to silently produce wrong
activations if it has a bug -- worth testing in isolation without
needing the actual multi-GB Qwen3-VL weights. We fake just enough of
the HF `processor`/`tokenizer` API surface (a whitespace-ish regex
tokenizer with offset mapping, and an `<image>` placeholder that
expands into several tokens, exactly like a real vision-language
processor) to exercise the real `locate_step_tokens` code path
end-to-end.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.activations import locate_step_tokens
from rise.utils import split_into_steps

TOKEN_RE = re.compile(r"\n\n|\S+|\s")


class FakeTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def _id_for(self, s: str) -> int:
        return self._vocab.setdefault(s, len(self._vocab))

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids, offsets = [], []
        for m in TOKEN_RE.finditer(text):
            ids.append(self._id_for(m.group()))
            offsets.append((m.start(), m.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class FakeProcessor:
    """Mimics just enough of AutoProcessor's behavior: a chat template
    that deterministically renders (system + user [images, text]) then
    'ASSISTANT:' then an optional assistant response, and a __call__
    that expands every '<image>' placeholder token into several tokens
    -- the behavior that shifts every downstream text token's position,
    which `locate_step_tokens` must compensate for."""

    def __init__(self, tokens_per_image: int = 7) -> None:
        self.tokenizer = FakeTokenizer()
        self.tokens_per_image = tokens_per_image

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False) -> str:
        parts = []
        for m in messages:
            if m["role"] == "assistant":
                continue
            for item in m["content"]:
                parts.append("<image>" if item["type"] == "image" else item["text"])
        text = " ".join(parts) + " ASSISTANT:"
        for m in messages:
            if m["role"] == "assistant":
                resp = " ".join(item["text"] for item in m["content"])
                text += " " + resp
        return text

    def __call__(self, text, images=None, return_tensors="pt"):
        raw = self.tokenizer(text[0])
        image_tok_id = self.tokenizer._vocab.get("<image>")
        expanded = []
        for tid in raw["input_ids"]:
            expanded.extend([tid] * self.tokens_per_image if tid == image_tok_id else [tid])
        return {"input_ids": torch.tensor([expanded])}


@dataclass
class FakeHandle:
    processor: FakeProcessor
    device: torch.device = torch.device("cpu")
    model: Optional[object] = None


def test_locate_step_tokens_accounts_for_image_expansion() -> None:
    handle = FakeHandle(processor=FakeProcessor(tokens_per_image=7))

    steps = [
        "Let me look at the picture to see what color the shirt is",
        "Wait, looking again at the image, the shirt is actually blue not red",
        "So the final answer is blue",
    ]
    response_text = "\n\n".join(steps)
    parsed_steps = split_into_steps(response_text)
    # The trailing step has no *following* "\n\n" (the joined text doesn't
    # end with the delimiter), so it has no delimiter token to extract an
    # activation from and split_into_steps correctly drops it -- see its
    # docstring. Only the first two, delimiter-terminated steps remain.
    assert parsed_steps == steps[:-1]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys prompt"}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "img1"},
                {"type": "image", "image": "img2"},
                {"type": "text", "text": "What color is the shirt in the image?"},
            ],
        },
    ]
    messages_with_response = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
    ]

    full_inputs, token_positions = locate_step_tokens(
        handle, images=["img1", "img2"], messages_with_response=messages_with_response,
        response_text=response_text, steps=parsed_steps,
    )

    input_ids = full_inputs["input_ids"][0]
    delimiter_id = handle.processor.tokenizer._vocab["\n\n"]

    assert len(token_positions) == len(parsed_steps)
    for pos in token_positions:
        assert 0 <= pos < input_ids.shape[0], f"position {pos} out of range (seq len {input_ids.shape[0]})"
        assert input_ids[pos].item() == delimiter_id, (
            f"position {pos} does not point at the '\\n\\n' token "
            f"(got token id {input_ids[pos].item()}, expected {delimiter_id})"
        )

    # Sanity: expansion actually happened (2 images * (7-1) extra tokens each).
    raw_len = len(handle.processor.tokenizer(handle.processor.apply_chat_template(
        messages_with_response, tokenize=False))["input_ids"])
    assert input_ids.shape[0] == raw_len + 2 * (7 - 1)


if __name__ == "__main__":
    test_locate_step_tokens_accounts_for_image_expansion()
    print("test_locate_step_tokens_accounts_for_image_expansion: OK")
