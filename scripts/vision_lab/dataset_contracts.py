"""Dataset adapter contracts: schema kinds, compatibility, and report merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vision_lab.task_registry import TASK_BY_ID, all_task_ids

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


@dataclass
class AdapterPartialReport:
    """Normalized adapter output before merging into the CLI JSON payload."""

    valid: bool
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


def merge_hf_legacy_payload(partial: AdapterPartialReport, legacy: dict[str, Any]) -> dict[str, Any]:
    """Merge adapter partial with legacy HF-only keys for backward compatibility."""
    kind = partial.dataset_schema_kind
    if kind in EXTENDED_SCHEMA_KINDS:
        compatible: list[str] = []
    elif kind in KNOWN_SCHEMA_KINDS:
        compatible = tasks_compatible_with_schema_kind(kind)
    else:
        compatible = sorted(all_task_ids())
    row_counts = dict(partial.row_counts)
    num_rows = legacy.get("num_rows", -1)
    split = legacy.get("_split", "train")
    if num_rows >= 0 and split:
        row_counts.setdefault(split, num_rows)

    out: dict[str, Any] = {
        "valid": partial.valid and len(partial.errors) == 0,
        "errors": list(partial.errors) + [e for e in legacy.get("errors", []) if e],
        "warnings": list(partial.warnings),
        "adapter_id": partial.adapter_id,
        "dataset_schema_kind": partial.dataset_schema_kind,
        "compatible_tasks": compatible,
        "required_fields": list(partial.required_fields),
        "detected_class_names": list(partial.detected_class_names),
        "label_remapping": dict(partial.label_remapping),
        "splits": dict(partial.splits),
        "row_counts": row_counts,
        "columns": list(partial.columns),
        "num_rows": num_rows,
        "config": legacy.get("config"),
        "inspection": partial.inspection if partial.inspection is not None else legacy.get("inspection"),
        "cache_manifest_path": legacy.get("cache_manifest_path"),
    }
    # De-duplicate errors while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for e in out["errors"]:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    out["errors"] = deduped
    out["valid"] = len(out["errors"]) == 0
    return out


def finalize_local_report(partial: AdapterPartialReport, *, split: str = "train") -> dict[str, Any]:
    """Convert a local adapter partial into the public ``validate_dataset`` dict shape."""
    num_rows = partial.row_counts.get(split, -1)
    if num_rows < 0 and partial.row_counts:
        num_rows = next(iter(partial.row_counts.values()))
    legacy = {
        "errors": [],
        "num_rows": num_rows,
        "config": None,
        "_split": split,
        "inspection": partial.inspection,
        "columns": partial.columns,
    }
    return merge_hf_legacy_payload(partial, legacy)


def partial_from_legacy_hf(
    *,
    adapter_id: str,
    schema_kind: str,
    valid: bool,
    errors: list[str],
    columns: list[str],
    inspection: dict[str, Any] | None,
    config: str | None,
    num_rows: int,
    split: str,
) -> dict[str, Any]:
    """Build full validation dict for HF Hub path (legacy shape extended)."""
    p = AdapterPartialReport(
        valid=valid,
        errors=list(errors),
        adapter_id=adapter_id,
        dataset_schema_kind=schema_kind,
        required_fields=list(columns),
        columns=list(columns),
        inspection=inspection,
    )
    legacy = {
        "errors": [],
        "num_rows": num_rows,
        "config": config,
        "_split": split,
        "inspection": inspection,
        "columns": columns,
    }
    return merge_hf_legacy_payload(p, legacy)
