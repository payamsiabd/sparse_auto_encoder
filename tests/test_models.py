"""Validates `rise.models`'s non-networked logic: registry key
resolution and the local-snapshot readiness check used by
`rise.utils.load_qwen3_vl` to fail with a clear message instead of an
implicit hub download when `scripts/00_download_models.py` hasn't been
run yet."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rise.models import MODEL_REGISTRY, model_is_downloaded, resolve_repo_id


def test_resolve_repo_id_uses_registry_for_known_keys() -> None:
    assert resolve_repo_id("qwen3-vl-4b-thinking") == MODEL_REGISTRY["qwen3-vl-4b-thinking"]
    assert resolve_repo_id("qwen3-vl-4b-thinking") == "Qwen/Qwen3-VL-4B-Thinking"


def test_resolve_repo_id_passes_through_unknown_ids() -> None:
    assert resolve_repo_id("someone-else/a-fine-tune") == "someone-else/a-fine-tune"


def test_model_is_downloaded_false_for_missing_or_incomplete_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does-not-exist"
        assert not model_is_downloaded(missing)

        incomplete = Path(tmp) / "incomplete"
        incomplete.mkdir()
        (incomplete / "config.json").write_text("{}")
        # no *.safetensors yet -> not ready
        assert not model_is_downloaded(incomplete)


def test_model_is_downloaded_true_for_complete_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        complete = Path(tmp) / "complete"
        complete.mkdir()
        (complete / "config.json").write_text("{}")
        (complete / "model-00001-of-00001.safetensors").write_bytes(b"\x00")
        assert model_is_downloaded(complete)


if __name__ == "__main__":
    test_resolve_repo_id_uses_registry_for_known_keys()
    print("test_resolve_repo_id_uses_registry_for_known_keys: OK")
    test_resolve_repo_id_passes_through_unknown_ids()
    print("test_resolve_repo_id_passes_through_unknown_ids: OK")
    test_model_is_downloaded_false_for_missing_or_incomplete_dir()
    print("test_model_is_downloaded_false_for_missing_or_incomplete_dir: OK")
    test_model_is_downloaded_true_for_complete_snapshot()
    print("test_model_is_downloaded_true_for_complete_snapshot: OK")
