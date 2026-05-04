"""SAM-style promptable segmentation: images, masks, plus bbox/point prompts (columns or JSONL)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, finalize_local_report

from .semantic_masks import validate_semantic_masks  # reuse layout


def validate_sam_prompt_mask(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    prompt_sources: list[str] = []

    for name in ("prompts.jsonl", "annotations.jsonl", "instances.jsonl"):
        p = root / name
        if p.is_file():
            prompt_sources.append(name)

    manifest = root / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and (
                    "bbox" in first or "points" in first or "prompt" in first
                ):
                    prompt_sources.append("manifest.json")
            elif isinstance(data, dict) and data.get("samples"):
                prompt_sources.append("manifest.json(samples)")
        except (json.JSONDecodeError, OSError):
            warnings.append("manifest.json present but not valid JSON")

    bbox_cols = any((root / f).is_file() for f in ("bboxes.jsonl", "boxes.jsonl"))
    if bbox_cols:
        prompt_sources.append("bbox-jsonl")

    base = validate_semantic_masks(root)
    if not base.get("valid"):
        errors.extend(base.get("errors", []))

    if not prompt_sources:
        warnings.append(
            "No prompts manifest detected — SAM training expects bbox/point prompts "
            "(prompts.jsonl, manifest.json, or HF columns)."
        )

    inspection = base.get("inspection") or {}
    inspection["prompt_sources"] = prompt_sources
    inspection["warnings"] = list(dict.fromkeys(warnings + inspection.get("warnings", [])))

    # Merge warnings back via Adapter rebuild would be heavy — patch dict:
    out = dict(base)
    out["warnings"] = list(dict.fromkeys(out.get("warnings", []) + warnings))
    out["inspection"] = inspection
    if errors:
        out["errors"] = list(dict.fromkeys(out.get("errors", []) + errors))
        out["valid"] = len(out["errors"]) == 0
    out["adapter_id"] = "sam_prompt_mask"
    out["required_fields"] = list(
        dict.fromkeys((out.get("required_fields") or []) + ["prompts or bbox jsonl"])
    )
    return out
