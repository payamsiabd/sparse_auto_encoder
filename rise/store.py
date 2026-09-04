"""Persist / load step-level activations and their metadata.

Two files per extraction run:
  - ``activations_layerLL.pt``: float32 tensor (N, d), one row per step,
    for a single layer LL (Sec. 3.2: "train the SAE on a single chosen
    layer").
  - ``steps_metadata.jsonl``: one JSON record per row, in the same order,
    with ``prompt_id``, ``step_index``, ``step_text`` -- lets later stages
    (annotation, geometry, intervention) map SAE rows back to the exact
    reasoning step and source sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .activations import StepActivation


def save_layer_activations(records: list[StepActivation], layer: int, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensor = torch.stack([r.hidden_states[layer] for r in records], dim=0)
    torch.save(tensor, out_dir / f"activations_layer{layer:02d}.pt")

    meta_path = out_dir / "steps_metadata.jsonl"
    if not meta_path.exists():
        with meta_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({
                    "prompt_id": r.prompt_id,
                    "step_index": r.step_index,
                    "step_text": r.step_text,
                    "token_position": r.token_position,
                }) + "\n")


def append_layer_activations(records: list[StepActivation], layers: list[int], out_dir: str | Path) -> None:
    """Incrementally append activations for many layers + the shared
    metadata file, one generated response's worth of steps at a time.
    Uses per-layer ``.pt`` shard files (one row-batch each) so a long
    extraction run can be resumed / parallelized; call
    ``consolidate_shards`` afterward to merge into single tensors."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_id = _next_shard_id(out_dir)

    for layer in layers:
        tensor = torch.stack([r.hidden_states[layer] for r in records], dim=0)
        torch.save(tensor, out_dir / f"layer{layer:02d}_shard{shard_id:06d}.pt")

    with (out_dir / "steps_metadata.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "prompt_id": r.prompt_id,
                "step_index": r.step_index,
                "step_text": r.step_text,
                "token_position": r.token_position,
            }) + "\n")


def _next_shard_id(out_dir: Path) -> int:
    existing = list(out_dir.glob("layer*_shard*.pt"))
    if not existing:
        return 0
    ids = [int(p.stem.split("shard")[-1]) for p in existing]
    return max(ids) + 1


def consolidate_shards(out_dir: str | Path, layer: int) -> torch.Tensor:
    out_dir = Path(out_dir)
    shard_paths = sorted(out_dir.glob(f"layer{layer:02d}_shard*.pt"))
    tensors = [torch.load(p, weights_only=True) for p in shard_paths]
    full = torch.cat(tensors, dim=0)
    torch.save(full, out_dir / f"activations_layer{layer:02d}.pt")
    return full


def load_layer_activations(out_dir: str | Path, layer: int) -> torch.Tensor:
    out_dir = Path(out_dir)
    path = out_dir / f"activations_layer{layer:02d}.pt"
    if not path.exists():
        return consolidate_shards(out_dir, layer)
    return torch.load(path, weights_only=True)


def load_steps_metadata(out_dir: str | Path) -> list[dict]:
    out_dir = Path(out_dir)
    records = []
    with (out_dir / "steps_metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
