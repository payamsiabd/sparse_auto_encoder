"""Tiny YAML config loader with dotted-key CLI overrides, used by all
scripts/*.py entry points (e.g. ``--sae.d_hidden 4096``)."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(default_path: str | Path, argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(default_path))
    known, rest = parser.parse_known_args(argv)

    with open(known.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    i = 0
    while i < len(rest):
        token = rest[i]
        if not token.startswith("--"):
            raise ValueError(f"Unrecognized argument: {token}")
        key = token[2:]
        if i + 1 >= len(rest):
            raise ValueError(f"Missing value for --{key}")
        value = rest[i + 1]
        _set_dotted(cfg, key, _coerce(value))
        i += 2
    return cfg


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(f"Config has no section '{p}' (from --{dotted_key})")
        node = node[p]
    node[parts[-1]] = value


def _coerce(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        return yaml.safe_load(value)
    return value
