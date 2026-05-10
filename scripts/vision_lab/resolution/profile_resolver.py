"""Strict profile resolver registry: explicit resolver id only, no implicit selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vision_lab.contracts.run_contract import ContractDataset
from vision_lab.resolution.inspect_dataset import HubDatasetProfile
from vision_lab.task_registry import get_task


class ProfileResolutionError(ValueError):
    """Profile cannot be mapped to ``ContractDataset`` under the chosen resolver."""


@dataclass(frozen=True)
class ProfileResolution:
    """Outcome of applying one registered resolver to a hub profile."""

    resolver_id: str
    contract_dataset: ContractDataset
    confidence: float
    rationale: str


ResolverFn = Callable[[str, HubDatasetProfile], ContractDataset]

_PROFILE_RESOLVERS: dict[str, ResolverFn] = {}


def register_profile_resolver(resolver_id: str, fn: ResolverFn) -> None:
    """Register or replace a resolver (call from library init or tests)."""
    if not resolver_id or not resolver_id.strip():
        raise ValueError("resolver_id must be non-empty")
    _PROFILE_RESOLVERS[resolver_id] = fn


def list_profile_resolver_ids() -> tuple[str, ...]:
    return tuple(sorted(_PROFILE_RESOLVERS))


def resolve_contract_dataset(
    *,
    task_id: str,
    profile: HubDatasetProfile,
    resolver_id: str,
) -> ProfileResolution:
    """
    Map a hub inspection profile into ``ContractDataset`` using exactly ``resolver_id``.

    Raises ``ProfileResolutionError`` if the resolver is unknown, the profile has
    inspection hard errors, or semantic checks fail.
    """
    fn = _PROFILE_RESOLVERS.get(resolver_id)
    if fn is None:
        raise ProfileResolutionError(
            f"unknown resolver_id {resolver_id!r} (known: {', '.join(list_profile_resolver_ids())})"
        )
    if profile.hard_errors:
        raise ProfileResolutionError(
            "profile has inspection errors: " + "; ".join(profile.hard_errors)
        )
    try:
        ds = fn(task_id, profile)
    except ProfileResolutionError:
        raise
    except Exception as e:
        raise ProfileResolutionError(str(e)) from e
    if ds.profile_id != resolver_id:
        raise ProfileResolutionError(
            f"internal error: resolver produced profile_id {ds.profile_id!r}, expected {resolver_id!r}"
        )
    rationale = (
        f"resolver={resolver_id!r}; task={task_id!r}; "
        f"columns={dict(ds.column_mapping)!r}; inspection_confidence={profile.inspection_confidence}"
    )
    conf = 1.0 if profile.inspection_confidence >= 1.0 else float(profile.inspection_confidence)
    return ProfileResolution(
        resolver_id=resolver_id,
        contract_dataset=ds,
        confidence=conf,
        rationale=rationale,
    )


def _schema_columns(profile: HubDatasetProfile) -> frozenset[str]:
    return frozenset(name for name, _ in profile.feature_schema)


def _require_pinned_revision(profile: HubDatasetProfile) -> None:
    if not str(profile.revision_resolved or "").strip():
        raise ProfileResolutionError(
            "profile.revision_resolved is empty; pin a hub revision before resolving"
        )


def _resolve_classify_image_label_v1(task_id: str, profile: HubDatasetProfile) -> ContractDataset:
    spec = get_task(task_id)
    if spec.dataset_schema_kind != "classification":
        raise ProfileResolutionError(
            f"task {task_id!r} is not a classification schema task for this resolver"
        )
    _require_pinned_revision(profile)
    cols = _schema_columns(profile)
    if "image" not in cols:
        raise ProfileResolutionError(f"expected an 'image' column; have {sorted(cols)}")
    label_candidates = ("label", "labels", "fine_label", "label_ids")
    found = [c for c in label_candidates if c in cols]
    if not found:
        raise ProfileResolutionError(
            f"expected exactly one label column among {label_candidates}; have {sorted(cols)}"
        )
    if len(found) > 1:
        raise ProfileResolutionError(
            f"ambiguous label columns {found!r}; disambiguate upstream or add a dedicated resolver"
        )
    label_col = found[0]
    mapping = (("image", "image"), ("label", label_col))
    rid = "hf_hub.classify.image_label_v1"
    return ContractDataset(
        source="hf_hub",
        identifier=profile.repo_id,
        revision=profile.revision_resolved,
        config_name=profile.config_name,
        split=profile.split,
        profile_id=rid,
        column_mapping=tuple(sorted(mapping, key=lambda x: x[0])),
    )


def _resolve_detect_objects_column_v1(task_id: str, profile: HubDatasetProfile) -> ContractDataset:
    spec = get_task(task_id)
    if spec.dataset_schema_kind != "detection":
        raise ProfileResolutionError(
            f"task {task_id!r} is not a detection schema task for this resolver"
        )
    _require_pinned_revision(profile)
    cols = _schema_columns(profile)
    if "image" not in cols:
        raise ProfileResolutionError(f"expected an 'image' column; have {sorted(cols)}")
    if "objects" not in cols:
        raise ProfileResolutionError(
            f"expected an 'objects' column for this resolver; have {sorted(cols)}"
        )
    rid = "hf_hub.detect.objects_column_v1"
    mapping = (("image", "image"), ("objects", "objects"))
    return ContractDataset(
        source="hf_hub",
        identifier=profile.repo_id,
        revision=profile.revision_resolved,
        config_name=profile.config_name,
        split=profile.split,
        profile_id=rid,
        column_mapping=tuple(sorted(mapping, key=lambda x: x[0])),
    )


def _register_builtin_resolvers() -> None:
    register_profile_resolver("hf_hub.classify.image_label_v1", _resolve_classify_image_label_v1)
    register_profile_resolver("hf_hub.detect.objects_column_v1", _resolve_detect_objects_column_v1)


_register_builtin_resolvers()
