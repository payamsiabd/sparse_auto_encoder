"""Optional vLLM-accelerated generation backend for
`scripts/02_generate_responses.py`.

vLLM's continuous batching + paged attention makes it dramatically
faster than `transformers`' `model.generate()` for exactly this
workload: many independent (image, question) prompts, each generating
up to several thousand tokens, with nothing needed from the model but
the final text. That's *only* true of the generation stage, though --
nothing else in this pipeline can move to vLLM:

- `scripts/03_extract_activations.py` needs `output_hidden_states` at
  an arbitrary transformer layer, for arbitrary token positions --
  vLLM's inference engine doesn't expose intermediate per-layer hidden
  states through its stable API at all.
- `rise/intervene.py`'s steering hooks need to edit the residual stream
  *during* generation at a specific layer -- not something vLLM's
  public API supports; doing it anyway would mean patching vLLM
  internals, which breaks across versions.
- `rise.intervene.search_entropy_vector` (Eq. 7) needs gradients
  through the model -- vLLM is inference-only, no autograd, so this is
  categorically incompatible.

So this module is deliberately narrow: it produces exactly the
`GeneratedResponse` objects `rise.activations.generate_response` does,
so `scripts/02` is a drop-in swap (`generation.backend: "vllm"` in
`configs/default.yaml`) and every downstream script is completely
unaffected -- they read `responses.jsonl` either way and never know
which backend produced it.

Requires `pip install vllm` with a version that supports Qwen3-VL --
this is a very recent architecture at the time of writing, so check
your installed vLLM's supported-models list before relying on this
path; if it isn't supported yet, use `generation.backend: "transformers"`
(the default) instead.
"""
from __future__ import annotations

import base64
import dataclasses
import io
from typing import Any

from PIL import Image

from .activations import GeneratedResponse
from .dataset import VisualPrompt, build_messages, load_images
from .utils import split_into_steps, split_thinking_and_answer

# vLLM's `dtype` argument wants "bfloat16"/"float16"/"float32", not this
# project's "bf16"/"fp16"/"fp32" shorthand used elsewhere.
_VLLM_DTYPE_NAMES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


@dataclasses.dataclass
class VLLMHandle:
    llm: Any


def load_vllm(
    model_id: str,
    dtype: str = "bf16",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    limit_mm_per_prompt: int = 4,
    trust_remote_code: bool = True,
) -> VLLMHandle:
    """Load Qwen3-VL-4B-Thinking (or any Qwen3-VL checkpoint, local
    directory or hub id -- same as `rise.utils.load_qwen3_vl`) as a
    vLLM engine. `limit_mm_per_prompt` bounds how many images a single
    prompt may contain (matches this project's multi-image support in
    `rise.dataset.VisualPrompt.images`)."""
    from vllm import LLM

    llm = LLM(
        model=model_id,
        dtype=_VLLM_DTYPE_NAMES.get(dtype, dtype),
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        limit_mm_per_prompt={"image": limit_mm_per_prompt},
    )
    return VLLMHandle(llm=llm)


def _pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def to_vllm_messages(messages: list[dict]) -> list[dict]:
    """Convert this project's chat message format (`rise.dataset.build_messages`,
    which mirrors the HF processor convention: content items are
    `{"type": "image", "image": PIL.Image}` / `{"type": "text", "text": str}`)
    into vLLM/OpenAI-style messages (`{"type": "image_url", "image_url":
    {"url": "data:..."}}`), the most broadly version-stable multimodal
    input format across vLLM releases. Split out from `generate_responses_batch`
    so the conversion is unit-testable without a GPU or vLLM installed."""
    converted = []
    for m in messages:
        content = []
        for item in m["content"]:
            if item["type"] == "image":
                content.append({"type": "image_url", "image_url": {"url": _pil_to_data_uri(item["image"])}})
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
            else:
                raise ValueError(f"Unsupported content type for vLLM conversion: {item['type']!r}")
        converted.append({"role": m["role"], "content": content})
    return converted


def parse_vllm_output(prompt_id: str, full_text: str) -> GeneratedResponse:
    """Turn one vLLM completion's raw text into the same `GeneratedResponse`
    shape `rise.activations.generate_response` returns, so downstream
    code (JSONL writing in `scripts/02`, everything in `scripts/03+`)
    can't tell which backend produced it. Split out for unit testing."""
    thinking_text, answer_text = split_thinking_and_answer(full_text)
    steps = split_into_steps(thinking_text) if thinking_text else split_into_steps(full_text)
    return GeneratedResponse(
        prompt_id=prompt_id, full_text=full_text,
        thinking_text=thinking_text, answer_text=answer_text, steps=steps,
    )


def generate_responses_batch(
    handle: VLLMHandle,
    prompts: list[VisualPrompt],
    max_new_tokens: int = 4096,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[GeneratedResponse]:
    """Generate a CoT response for every prompt in one batched vLLM call
    -- vLLM's scheduler handles the continuous batching internally, so
    unlike `rise.activations.generate_response`'s one-prompt-at-a-time
    loop, this hands the whole prompt set to the engine at once."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
    )

    conversations = []
    for prompt in prompts:
        images = load_images(prompt)
        messages = build_messages(prompt, images)
        conversations.append(to_vllm_messages(messages))

    outputs = handle.llm.chat(conversations, sampling_params=sampling_params, use_tqdm=True)

    return [
        parse_vllm_output(prompt.id, output.outputs[0].text)
        for prompt, output in zip(prompts, outputs)
    ]
