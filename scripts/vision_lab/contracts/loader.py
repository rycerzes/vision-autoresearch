"""Load a validated ``RunContract`` from JSON or YAML on disk."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from vision_lab.contracts.run_contract import RunContract, run_contract_from_mapping


def load_run_contract(path: Path) -> RunContract:
    """Read ``path`` (.json / .yaml / .yml) and return a validated ``RunContract``."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported contract file extension: {path.suffix!r}")
    if not isinstance(raw, dict):
        raise ValueError("Contract file must contain a JSON object / YAML mapping")
    return run_contract_from_mapping(raw)
