"""Run contract schema and strict validation."""

from __future__ import annotations

from vision_lab.contracts.run_contract import (
    CONTRACT_VERSION,
    ContractDataset,
    ContractGate,
    ContractModel,
    ContractPipeline,
    ContractPromotion,
    ContractRuntime,
    ContractTraining,
    RunContract,
    canonical_json_bytes,
    contract_fingerprint,
    contract_fingerprint_from_payload,
    run_contract_from_mapping,
    run_contract_to_primitive_dict,
)
from vision_lab.contracts.schema import RunContractValidationError, parse_run_contract

__all__ = [
    "CONTRACT_VERSION",
    "ContractDataset",
    "ContractGate",
    "ContractModel",
    "ContractPipeline",
    "ContractPromotion",
    "ContractRuntime",
    "ContractTraining",
    "RunContract",
    "RunContractValidationError",
    "canonical_json_bytes",
    "contract_fingerprint",
    "contract_fingerprint_from_payload",
    "parse_run_contract",
    "run_contract_from_mapping",
    "run_contract_to_primitive_dict",
]
