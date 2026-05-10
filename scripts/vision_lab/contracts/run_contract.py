"""Canonical run contract datamodel.

Defines structure, validation, serialization, and fingerprinting.
Resolvers populate these fields; training consumes only validated ``RunContract`` instances.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Literal

from vision_lab.metrics import MetricDirection

CONTRACT_VERSION: Literal[1] = 1

BackendId = Literal["hf_trainer", "ultralytics"]
DatasetSourceKind = Literal["hf_hub", "local"]
MixedPrecisionKind = Literal["fp16", "bf16", "none"]

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]

MappingLike = Mapping[str, Any]


@dataclass(frozen=True)
class ContractDataset:
    """Resolved dataset wiring for one run."""

    source: DatasetSourceKind
    identifier: str
    """Hub repo id (``org/name``) or local root path string."""
    revision: str | None
    config_name: str | None
    split: str
    profile_id: str
    column_mapping: tuple[tuple[str, str], ...]
    """Canonical role name -> dataset column name, sorted by role for stable fingerprints."""


@dataclass(frozen=True)
class ContractModel:
    model_id: str
    loader_strategy: str
    architecture_hints: dict[str, JSONValue]


@dataclass(frozen=True)
class ContractGate:
    metric: str
    gate_min: float | None = None
    gate_max: float | None = None


@dataclass(frozen=True)
class ContractPromotion:
    primary: str
    direction: MetricDirection
    min_delta: float
    secondary: str | None
    gates: tuple[ContractGate, ...]
    tie_breakers: tuple[str, ...]


@dataclass(frozen=True)
class ContractPipeline:
    transform_recipe_id: str
    transform_recipe_params: dict[str, JSONValue]
    collator_id: str
    loss_id: str
    metric_set_id: str
    promotion: ContractPromotion


@dataclass(frozen=True)
class ContractTraining:
    """Hyperparameters and trainer-specific knobs (JSON-serializable values only)."""

    hyperparameters: dict[str, JSONValue]


@dataclass(frozen=True)
class ContractRuntime:
    seed: int
    mixed_precision: MixedPrecisionKind
    device: str
    dataloader_num_workers: int


@dataclass(frozen=True)
class RunContract:
    """Single source of truth for a benchmark run (contract version 1)."""

    contract_version: Literal[1]
    task: str
    backend: BackendId
    dataset: ContractDataset
    model: ContractModel
    pipeline: ContractPipeline
    training: ContractTraining
    runtime: ContractRuntime


def _json_sort_keys(obj: JSONValue) -> JSONValue:
    """Return a deep copy with all dict keys sorted recursively (for canonical JSON)."""
    if isinstance(obj, dict):
        return {k: _json_sort_keys(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_json_sort_keys(x) for x in obj]
    return obj


def canonical_json_bytes(payload: JSONValue) -> bytes:
    """Deterministic UTF-8 JSON: sorted keys, no insignificant whitespace."""
    sorted_payload = _json_sort_keys(payload)
    return json.dumps(
        sorted_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def contract_fingerprint_from_payload(payload: JSONValue) -> str:
    """SHA-256 hex digest of canonical JSON for an arbitrary JSON-serializable payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _dataclass_to_primitive(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_primitive(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_primitive(v) for k, v in sorted(obj.items())}
    if is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in fields(obj):
            out[f.name] = _dataclass_to_primitive(getattr(obj, f.name))
        return out
    raise TypeError(f"Unsupported type for contract serialization: {type(obj)!r}")


def run_contract_to_primitive_dict(contract: RunContract) -> dict[str, Any]:
    """Serialize ``RunContract`` to nested dicts/lists suitable for JSON or YAML."""
    return _dataclass_to_primitive(contract)  # type: ignore[return-value]


def contract_fingerprint(contract: RunContract) -> str:
    """Reproducible fingerprint for a resolved run contract."""
    return contract_fingerprint_from_payload(run_contract_to_primitive_dict(contract))  # type: ignore[arg-type]


def run_contract_from_mapping(raw: MappingLike) -> RunContract:
    """Parse and validate a mapping (e.g. from JSON/YAML) into ``RunContract``."""
    from vision_lab.contracts.schema import parse_run_contract

    return parse_run_contract(raw)
