"""Validates `rise.vllm_backend`'s pure-Python logic -- message-format
conversion, output parsing, and the batched-generation call shape --
against a fake `vllm` module and a fake engine, so none of this needs
`pip install vllm` or a GPU. What's NOT tested here: whether a real
vLLM engine actually accepts these messages and successfully runs
Qwen3-VL (that needs the real package + weights + GPU).
"""
from __future__ import annotations

import base64
import io
import sys
import tempfile
import types
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.dataset import VisualPrompt, build_messages
from rise.vllm_backend import parse_vllm_output, to_vllm_messages


def _install_fake_vllm():
    """Injects a minimal fake `vllm` module into sys.modules so
    `rise.vllm_backend.generate_responses_batch`'s lazy `from vllm
    import SamplingParams` resolves to our fake instead of requiring
    the real package."""

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCompletion:
        def __init__(self, text: str):
            self.text = text

    class FakeRequestOutput:
        def __init__(self, text: str):
            self.outputs = [FakeCompletion(text)]

    class FakeLLM:
        def __init__(self, canned_texts: list[str]):
            self.canned_texts = canned_texts
            self.received_conversations = None
            self.received_sampling_params = None

        def chat(self, conversations, sampling_params=None, use_tqdm=True):
            self.received_conversations = conversations
            self.received_sampling_params = sampling_params
            return [FakeRequestOutput(t) for t in self.canned_texts]

    fake_module = types.ModuleType("vllm")
    fake_module.SamplingParams = FakeSamplingParams
    sys.modules["vllm"] = fake_module
    return FakeLLM


def _make_prompt(tmp_dir: Path, pid: str, color: tuple[int, int, int]) -> VisualPrompt:
    img_path = tmp_dir / f"{pid}.png"
    Image.new("RGB", (4, 4), color=color).save(img_path)
    return VisualPrompt(id=pid, images=[img_path], question=f"What color is image {pid}?")


def test_to_vllm_messages_converts_images_and_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prompt = _make_prompt(Path(tmp), "p1", (255, 0, 0))
        images = [Image.open(prompt.images[0])]
        messages = build_messages(prompt, images)

        converted = to_vllm_messages(messages)

        assert converted[0]["role"] == "system"
        user_msg = next(m for m in converted if m["role"] == "user")
        image_items = [c for c in user_msg["content"] if c["type"] == "image_url"]
        text_items = [c for c in user_msg["content"] if c["type"] == "text"]

        assert len(image_items) == 1
        assert text_items[-1]["text"] == prompt.question

        url = image_items[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        round_tripped = Image.open(io.BytesIO(decoded))
        assert round_tripped.size == (4, 4)


def test_to_vllm_messages_rejects_unknown_content_type() -> None:
    messages = [{"role": "user", "content": [{"type": "video", "video": None}]}]
    try:
        to_vllm_messages(messages)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_parse_vllm_output_splits_thinking_and_steps() -> None:
    text = "<think>Step one.\n\nWait, let me check the image again.</think>The answer is 4."
    response = parse_vllm_output("p1", text)

    assert response.prompt_id == "p1"
    assert response.full_text == text
    assert response.thinking_text == "Step one.\n\nWait, let me check the image again."
    assert response.answer_text == "The answer is 4."
    assert response.steps == ["Step one."]  # trailing step has no following "\n\n", dropped


def test_generate_responses_batch_calls_engine_with_expected_shape() -> None:
    FakeLLM = _install_fake_vllm()
    from rise.vllm_backend import VLLMHandle, generate_responses_batch  # import after fake vllm installed

    with tempfile.TemporaryDirectory() as tmp:
        prompts = [
            _make_prompt(Path(tmp), "p1", (255, 0, 0)),
            _make_prompt(Path(tmp), "p2", (0, 255, 0)),
        ]
        canned = [
            "<think>Looking at the image, it is red.\n\n</think>Red.",
            "<think>Looking at the image, it is green.\n\n</think>Green.",
        ]
        fake_llm = FakeLLM(canned)
        handle = VLLMHandle(llm=fake_llm)

        responses = generate_responses_batch(handle, prompts, max_new_tokens=128, do_sample=False)

        assert len(responses) == 2
        assert responses[0].answer_text == "Red."
        assert responses[1].answer_text == "Green."
        assert responses[0].prompt_id == "p1"
        assert responses[1].prompt_id == "p2"

        # One batched call covering every prompt, greedy sampling params.
        assert len(fake_llm.received_conversations) == 2
        assert fake_llm.received_sampling_params.kwargs["max_tokens"] == 128
        assert fake_llm.received_sampling_params.kwargs["temperature"] == 0.0


if __name__ == "__main__":
    test_to_vllm_messages_converts_images_and_text()
    print("test_to_vllm_messages_converts_images_and_text: OK")
    test_to_vllm_messages_rejects_unknown_content_type()
    print("test_to_vllm_messages_rejects_unknown_content_type: OK")
    test_parse_vllm_output_splits_thinking_and_steps()
    print("test_parse_vllm_output_splits_thinking_and_steps: OK")
    test_generate_responses_batch_calls_engine_with_expected_shape()
    print("test_generate_responses_batch_calls_engine_with_expected_shape: OK")
