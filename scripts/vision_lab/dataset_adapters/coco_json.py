"""COCO-format JSON annotations (instances / object detection)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_lab.dataset_contracts import AdapterPartialReport, to_validation_report


def find_coco_json(root: Path) -> Path | None:
    candidates = sorted(root.glob("**/instances*.json"))
    if candidates:
        return candidates[0]
    direct = root / "annotations.json"
    if direct.is_file():
        return direct
    for name in ("train.json", "val.json", "annotations_train.json"):
        p = root / name
        if p.is_file():
            return p
    return None


def validate_coco_json(root: Path, *, inspect_sample_images: int = 5) -> dict[str, Any]:
    root = root.resolve()
    json_path = root if root.is_file() else find_coco_json(root)
    errors: list[str] = []
    warnings: list[str] = []

    if json_path is None or not json_path.is_file():
        p = AdapterPartialReport(
            errors=[f"No COCO-style instances*.json or annotations.json under {root}"],
            adapter_id="coco_json",
            dataset_schema_kind="detection",
            required_fields=["images", "annotations", "categories"],
        )
        return to_validation_report(p)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        p = AdapterPartialReport(
            errors=[f"Failed to read COCO JSON {json_path}: {e}"],
            adapter_id="coco_json",
            dataset_schema_kind="detection",
        )
        return to_validation_report(p)

    if not isinstance(data, dict):
        p = AdapterPartialReport(
            errors=["COCO JSON root must be an object"],
            adapter_id="coco_json",
            dataset_schema_kind="detection",
        )
        return to_validation_report(p)

    req = ("images", "annotations", "categories")
    missing = [k for k in req if k not in data]
    if missing:
        errors.append(f"COCO JSON missing keys: {missing}")

    categories = data.get("categories", [])
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    names: list[str] = []
    label_remapping: dict[str, Any] = {}
    if isinstance(categories, list):
        for c in categories[:512]:
            if isinstance(c, dict) and "name" in c:
                cid = c.get("id")
                names.append(str(c["name"]))
                if cid is not None:
                    label_remapping[str(cid)] = c["name"]

    if not isinstance(images, list):
        errors.append("'images' must be a list")
    if not isinstance(annotations, list):
        errors.append("'annotations' must be a list")

    sample_dirs_checked = 0
    if isinstance(images, list) and errors == []:
        for im in images[:inspect_sample_images]:
            if not isinstance(im, dict) or "file_name" not in im:
                warnings.append("image entry missing file_name")
                continue
            fn = im["file_name"]
            found = False
            for base in (root, json_path.parent, root / "images", root / "train"):
                if isinstance(base, Path):
                    cand = base / fn if not Path(fn).is_absolute() else Path(fn)
                    if cand.is_file():
                        found = True
                        break
            sample_dirs_checked += 1
            if not found and sample_dirs_checked <= inspect_sample_images:
                warnings.append(
                    f"Could not resolve image file {fn!r} relative to dataset root (ok if hub-only paths)."
                )

    row_counts = {}
    if isinstance(images, list):
        row_counts["train"] = len(images)
    if isinstance(annotations, list):
        row_counts["annotations"] = len(annotations)

    inspection = {
        "coco_json_path": str(json_path),
        "num_images": len(images) if isinstance(images, list) else 0,
        "num_annotations": len(annotations) if isinstance(annotations, list) else 0,
        "categories_sample": names[:20],
        "warnings": warnings,
    }

    p = AdapterPartialReport(
        errors=errors,
        warnings=warnings,
        adapter_id="coco_json",
        dataset_schema_kind="detection",
        required_fields=list(req),
        detected_class_names=names,
        label_remapping=label_remapping,
        splits={"default": str(json_path.parent)},
        row_counts=row_counts,
        inspection=inspection,
    )
    return to_validation_report(p)
