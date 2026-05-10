"""Dataset inspection and strict profile-to-contract resolution."""

from __future__ import annotations

from vision_lab.resolution.inspect_dataset import (
    HubDatasetProfile,
    inspect_hf_hub_dataset,
    profile_fingerprint,
    profile_to_primitive_dict,
)
from vision_lab.resolution.pipeline_resolver import (
    ModelCapabilities,
    PipelineResolution,
    PipelineResolutionError,
    list_pipeline_resolver_ids,
    register_pipeline_resolver,
    resolve_contract_pipeline,
)
from vision_lab.resolution.profile_resolver import (
    ProfileResolution,
    ProfileResolutionError,
    list_profile_resolver_ids,
    register_profile_resolver,
    resolve_contract_dataset,
)

__all__ = [
    "HubDatasetProfile",
    "ModelCapabilities",
    "PipelineResolution",
    "PipelineResolutionError",
    "ProfileResolution",
    "ProfileResolutionError",
    "inspect_hf_hub_dataset",
    "list_pipeline_resolver_ids",
    "list_profile_resolver_ids",
    "profile_fingerprint",
    "profile_to_primitive_dict",
    "register_pipeline_resolver",
    "register_profile_resolver",
    "resolve_contract_dataset",
    "resolve_contract_pipeline",
]
