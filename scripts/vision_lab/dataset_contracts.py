"""Dataset adapter contracts: schema kinds and canonical validation reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vision_lab.task_registry import TASK_BY_ID

# Primary HF-style schema kinds used by TaskSpec.dataset_schema_kind
KNOWN_SCHEMA_KINDS = frozenset({"detection", "classification", "segmentation"})

# Non-HF schema kinds adapters may emit (no registered task yet — preflight blocks mismatch).
EXTENDED_SCHEMA_KINDS = frozenset({"video", "ocr", "depth", "image_to_image"})

# Maps adapter id → dataset schema kind satisfied by validated layouts.
ADAPTER_SCHEMA_KIND: dict[str, str] = {
    "hf_hub": "dynamic",  # resolved per task / HF features
    "coco_json": "detection",
    "yolo_folder": "detection",
    "voc_xml": "detection",
    "semantic_masks": "segmentation",
    "sam_prompt_mask": "segmentation",
    "video_folder": "video",
    "ocr_table": "ocr",
    "depth_pairs": "depth",
    "image_pairs": "image_to_image",
}


def schema_kind_for_adapter(adapter_id: str) -> str | None:
    """Return fixed schema kind for a local adapter; ``hf_hub`` has no fixed kind."""
    if adapter_id == "hf_hub":
        return None
    return ADAPTER_SCHEMA_KIND.get(adapter_id)


def tasks_compatible_with_schema_kind(kind: str) -> list[str]:
    """Registered task ids whose ``dataset_schema_kind`` matches ``kind``."""
    return sorted(tid for tid, spec in TASK_BY_ID.items() if spec.dataset_schema_kind == kind)


def preflight_adapter_matches_task(adapter_id: str, task_id: str) -> tuple[bool, str | None]:
    """
    Return (ok, error_message) when ``dataset_adapter`` in config must agree with ``task_id``.

    ``hf_hub`` always matches any registered task (HF column contracts are per-task).
    """
    if task_id not in TASK_BY_ID:
        return False, f"unknown task: {task_id!r}"
    spec = TASK_BY_ID[task_id]
    if adapter_id in ("auto", "hf_hub"):
        return True, None
    fixed = schema_kind_for_adapter(adapter_id)
    if fixed is None:
        return False, f"unknown dataset adapter: {adapter_id!r}"
    if fixed not in KNOWN_SCHEMA_KINDS:
        return False, (
            f"adapter {adapter_id!r} targets schema {fixed!r}; "
            f"no registered task uses that schema yet (supported: {sorted(KNOWN_SCHEMA_KINDS)})"
        )
    if spec.dataset_schema_kind != fixed:
        return False, (
            f"task {task_id!r} expects dataset_schema_kind={spec.dataset_schema_kind!r} "
            f"but adapter {adapter_id!r} provides {fixed!r}"
        )
    return True, None


def compatible_tasks_for_schema_kind(kind: str) -> list[str]:
    if kind in EXTENDED_SCHEMA_KINDS:
        return []
    if kind in KNOWN_SCHEMA_KINDS:
        return tasks_compatible_with_schema_kind(kind)
    return []


@dataclass
class AdapterPartialReport:
    """Normalized adapter output before exporting the public validation dict."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    adapter_id: str = ""
    dataset_schema_kind: str = ""
    required_fields: list[str] = field(default_factory=list)
    detected_class_names: list[str] = field(default_factory=list)
    label_remapping: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    inspection: dict[str, Any] | None = None


def to_validation_report(
    partial: AdapterPartialReport,
    *,
    dataset_config: str | None = None,
    cache_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Single canonical shape for ``validate_dataset`` / CLI JSON."""
    deduped_errors = list(dict.fromkeys(partial.errors))
    return {
        "valid": len(deduped_errors) == 0,
        "errors": deduped_errors,
        "warnings": list(partial.warnings),
        "adapter_id": partial.adapter_id,
        "dataset_schema_kind": partial.dataset_schema_kind,
        "compatible_tasks": compatible_tasks_for_schema_kind(partial.dataset_schema_kind),
        "required_fields": list(partial.required_fields),
        "detected_class_names": list(partial.detected_class_names),
        "label_remapping": dict(partial.label_remapping),
        "splits": dict(partial.splits),
        "row_counts": dict(partial.row_counts),
        "columns": list(partial.columns),
        "dataset_config": dataset_config,
        "inspection": partial.inspection,
        "cache_manifest_path": cache_manifest_path,
    }
