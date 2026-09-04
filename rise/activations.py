"""Generate CoT responses and extract step-boundary residual-stream
activations, implementing Section 3.2 ("Thought Representation
Construction") of the RISE paper for Qwen3-VL:

  (i)   Collect model responses: feed each (image, question) into the
        target model to generate a response.
  (ii)  Split responses into sentence-level steps using the delimiter
        "\\n\\n", producing k steps per response.
  (iii) Embed each step: re-run inference feeding the question and the
        response, and extract the hidden representation of the "\\n\\n"
        token at each step boundary from the residual stream after a
        chosen transformer layer.

The one Qwen3-VL-specific wrinkle relative to the paper's text-only
DeepSeek-R1 setup is that the processor *expands* each `<image>`
placeholder into many vision tokens, which shifts every text token's
position in the final ``input_ids``. ``locate_step_tokens`` below
computes that shift exactly (rather than assuming a fixed number of
image tokens) so the returned positions index the *actual* sequence fed
to the model.
"""
from __future__ import annotations

import dataclasses

import torch
from PIL import Image

from .utils import ModelHandle, STEP_DELIMITER, split_into_steps, split_thinking_and_answer


@dataclasses.dataclass
class StepActivation:
    prompt_id: str
    step_index: int
    step_text: str
    token_position: int          # index into the full (image-expanded) input_ids
    hidden_states: dict[int, torch.Tensor]  # layer index -> activation vector, shape (d,)


@dataclasses.dataclass
class GeneratedResponse:
    prompt_id: str
    full_text: str
    thinking_text: str
    answer_text: str
    steps: list[str]


@torch.no_grad()
def generate_response(
    handle: ModelHandle,
    images: list[Image.Image],
    messages: list[dict],
    prompt_id: str,
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> GeneratedResponse:
    """Stage (i): generate a full response (reasoning + answer) for one
    (image(s), question) sample."""
    processor = handle.processor
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=images if images else None, return_tensors="pt")
    inputs = {k: v.to(handle.device) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    output_ids = handle.model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    full_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    thinking_text, answer_text = split_thinking_and_answer(full_text)
    steps = split_into_steps(thinking_text) if thinking_text else split_into_steps(full_text)

    return GeneratedResponse(
        prompt_id=prompt_id, full_text=full_text,
        thinking_text=thinking_text, answer_text=answer_text, steps=steps,
    )


def _render(handle: ModelHandle, messages: list[dict]) -> str:
    return handle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def locate_step_tokens(
    handle: ModelHandle,
    images: list[Image.Image],
    messages_with_response: list[dict],
    response_text: str,
    steps: list[str],
) -> tuple[dict, list[int]]:
    """Find the exact index (into the *image-expanded* ``input_ids`` the
    model actually consumes) of the token spanning each step-delimiter
    occurrence in ``response_text``.

    Method (see module docstring): tokenize the fully-rendered chat text
    twice -- once as plain text (no image expansion, but with reliable
    character offsets via a fast tokenizer) and once through the real
    multimodal ``processor`` (which performs image expansion but not
    offset mapping). Because image expansion only splices extra tokens
    into the region *before* the response (images are placed earlier in
    the conversation) and leaves the text tokenization after that region
    untouched, the two token streams differ by one constant integer
    shift from the start of the response onward. We compute that shift
    from the two tokenizations' lengths up to (but excluding) the
    response, then apply it to the plain-text offsets.

    Returns:
        (full_inputs, token_positions) where ``full_inputs`` is the dict
        of tensors to feed the model (already image-expanded, ready for
        ``model(**full_inputs, output_hidden_states=True)``), and
        ``token_positions[i]`` is the sequence index of the delimiter
        ending step ``i`` (0-indexed, aligned with ``steps``).
    """
    processor = handle.processor
    tokenizer = processor.tokenizer

    prompt_messages = messages_with_response[:-1]
    prompt_only_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = _render(handle, messages_with_response)

    if not full_text.startswith(prompt_only_text):
        raise AssertionError(
            "Chat template did not render the response as a pure suffix of "
            "the prompt; step-position localization assumptions are "
            "violated for this template. Inspect `full_text` / "
            "`prompt_only_text` and adjust `locate_step_tokens`."
        )

    # Real (image-expanded) input the model will process.
    full_inputs = processor(text=[full_text], images=images if images else None, return_tensors="pt")
    full_inputs = {k: v.to(handle.device) for k, v in full_inputs.items()}

    # Plain-text tokenizations (no image processing) purely to get exact
    # character offsets for locating delimiters, and the prompt/full token
    # count *before* image expansion, i.e. how many image-placeholder
    # tokens (typically 1 per image reference) get expanded.
    raw_prompt = tokenizer(prompt_only_text, add_special_tokens=False)
    raw_full = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    raw_prompt_len = len(raw_prompt["input_ids"])

    expanded_prompt_len = _find_response_start(handle, prompt_only_text, images)
    shift = expanded_prompt_len - raw_prompt_len

    response_start_char = len(prompt_only_text)
    offsets = raw_full["offset_mapping"]

    token_positions: list[int] = []
    search_from = response_start_char
    for step_text in steps:
        idx = full_text.find(step_text, search_from)
        if idx == -1:
            raise ValueError(f"Could not locate step text in rendered response: {step_text[:80]!r}")
        delim_char = idx + len(step_text)  # char right after the step, i.e. inside/at the "\n\n"
        # target char = last character of the delimiter itself, so the
        # returned token is "the <\n\n> token" per the paper, not the
        # first token of the *next* step.
        target_char = min(delim_char + len(STEP_DELIMITER) - 1, len(full_text) - 1)

        raw_token_idx = _char_to_token(offsets, target_char, fallback_min=raw_prompt_len)
        token_positions.append(raw_token_idx + shift)
        search_from = idx + len(step_text)

    return full_inputs, token_positions


def _char_to_token(offsets: list[tuple[int, int]], char_idx: int, fallback_min: int) -> int:
    for tok_idx, (s, e) in enumerate(offsets):
        if s <= char_idx < e:
            return tok_idx
    # Delimiter fell on a token boundary / whitespace collapsed by the
    # tokenizer (e.g. "\n\n" merged into an adjacent token); fall back to
    # the nearest preceding token with non-empty span.
    for tok_idx in range(len(offsets) - 1, fallback_min - 1, -1):
        s, e = offsets[tok_idx]
        if e <= char_idx and e > s:
            return tok_idx
    raise ValueError(f"Could not map char index {char_idx} to a token.")


def _find_response_start(handle: ModelHandle, prompt_only_text: str, images: list[Image.Image]) -> int:
    """Number of tokens (in the *image-expanded* sequence) occupied by
    the prompt alone, i.e. the index at which the response begins."""
    processor = handle.processor
    prompt_inputs = processor(text=[prompt_only_text], images=images if images else None, return_tensors="pt")
    return prompt_inputs["input_ids"].shape[1]


@torch.no_grad()
def extract_step_activations(
    handle: ModelHandle,
    images: list[Image.Image],
    messages: list[dict],
    response: GeneratedResponse,
    layers: list[int],
) -> list[StepActivation]:
    """Stage (iii): a single forced forward pass over (question, full
    response) that returns, for every requested layer and every step
    boundary, the residual-stream vector at the "\\n\\n" token -- i.e.
    ``hidden_states[l]`` (output of transformer layer l) at the located
    position, matching Sec. 3.2 exactly ("the representations of the
    token <\\n\\n>", "residual stream representations after each
    transformer layer")."""
    if not response.steps:
        return []

    full_messages = messages + [{"role": "assistant", "content": [{"type": "text", "text": response.full_text}]}]
    full_inputs, token_positions = locate_step_tokens(
        handle, images, full_messages, response.full_text, response.steps
    )

    outputs = handle.model(**full_inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple length num_layers+1; index 0 = embeddings

    results: list[StepActivation] = []
    for step_idx, (step_text, pos) in enumerate(zip(response.steps, token_positions)):
        layer_vecs = {l: hidden_states[l][0, pos, :].detach().to("cpu", torch.float32) for l in layers}
        results.append(
            StepActivation(
                prompt_id=response.prompt_id, step_index=step_idx, step_text=step_text,
                token_position=pos, hidden_states=layer_vecs,
            )
        )
    return results
