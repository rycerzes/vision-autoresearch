#!/usr/bin/env python3
"""
Dataset validation and preparation for vision-autoresearch experiments.

Validates that a HF Hub dataset has the required schema for a given task type.
Performs both column-level and sample-level checks (bbox format, label types,
mask dimensions, prompt presence).

Read-only during experiments -- never modified by experiment workers.

Usage:
    python prepare.py --dataset cppe-5 --task detect --split train
    python prepare.py --dataset cppe-5 --task detect_yolo --split train
    python prepare.py --dataset food101 --task classify --split train
    python prepare.py --dataset <name> --task segment --split train
    python prepare.py --dataset cppe-5 --task detect --split train --inspect
    python prepare.py --dataset cppe-5 --task detect --split train --json
"""
from __future__ import annotations

import argparse
import json as json_mod
import math
import sys
from typing import Any

from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import dataset_info


NUM_INSPECT_SAMPLES = 5


def detect_bbox_format(bbox: list[float], image_w: int | None = None, image_h: int | None = None) -> str:
    """Detect bounding box format from a single 4-element bbox."""
    if len(bbox) != 4:
        return "unknown"
    a, b, c, d = bbox
    is_normalized = all(0 <= v <= 1 for v in bbox)
    if c < a or d < b:
        return "xywh_normalized" if is_normalized else "xywh"
    if image_w is not None and image_h is not None:
        xywh_exceeds = (a + c > image_w * 1.05) or (b + d > image_h * 1.05)
        xyxy_exceeds = (c > image_w * 1.05) or (d > image_h * 1.05)
        if xywh_exceeds and not xyxy_exceeds:
            return "xyxy"
        if xyxy_exceeds and not xywh_exceeds:
            return "xywh"
    if is_normalized:
        return "xyxy_normalized"
    return "xyxy"


def _get_nested_keys(feature) -> set[str]:
    """Extract inner keys from a Sequence/struct feature."""
    if hasattr(feature, "feature"):
        inner = feature.feature
        return set(inner.keys()) if hasattr(inner, "keys") else set()
    if hasattr(feature, "keys"):
        return set(feature.keys())
    return set()


def validate_detection_schema(features: dict, dataset_name: str) -> list[str]:
    """Validate detection dataset has bbox + category columns."""
    errors = []
    column_names = set(features.keys())

    if "image" not in column_names:
        errors.append(f"Missing 'image' column. Found: {sorted(column_names)}")
        return errors

    if "objects" in column_names:
        inner_keys = _get_nested_keys(features["objects"])
        has_bbox = "bbox" in inner_keys or "bboxes" in inner_keys
        has_cat = bool(inner_keys & {"category", "label", "categories"})
        if not has_bbox:
            errors.append(f"'objects' exists but missing bbox sub-field. Found: {sorted(inner_keys)}")
        if not has_cat:
            errors.append(f"'objects' exists but missing category sub-field. Found: {sorted(inner_keys)}")
        return errors

    has_bbox = any(c in column_names for c in ("bboxes", "bbox", "boxes"))
    has_cat = any(c in column_names for c in ("labels", "label", "categories", "category"))
    if not has_bbox:
        errors.append(f"No bbox column found. Expected 'objects.bbox', 'bboxes', or 'bbox'. Found: {sorted(column_names)}")
    if not has_cat:
        errors.append(f"No category column found. Expected 'objects.category', 'labels', or 'label'. Found: {sorted(column_names)}")
    return errors


def inspect_detection_samples(samples: list[dict], features: dict) -> dict[str, Any]:
    """Deeper sample-level checks for detection datasets."""
    info: dict[str, Any] = {
        "bbox_format": None,
        "num_classes": None,
        "avg_objects_per_image": None,
        "min_objects": None,
        "max_objects": None,
        "categories_sample": [],
        "warnings": [],
    }
    column_names = set(features.keys())
    all_cats: set = set()
    obj_counts: list[int] = []
    bbox_formats: list[str] = []

    for sample in samples:
        bboxes = []
        cats = []
        img_w = sample.get("width")
        img_h = sample.get("height")

        if "objects" in column_names:
            obj = sample.get("objects", {})
            if isinstance(obj, dict):
                bboxes = obj.get("bbox", obj.get("bboxes", []))
                cats = obj.get("category", obj.get("label", obj.get("categories", [])))
            elif isinstance(obj, list):
                bboxes = [o.get("bbox", o.get("bboxes")) for o in obj if isinstance(o, dict)]
                cats = [o.get("category", o.get("label")) for o in obj if isinstance(o, dict)]
        else:
            bboxes = sample.get("bboxes", sample.get("bbox", sample.get("boxes", [])))
            cats = sample.get("labels", sample.get("label", sample.get("categories", sample.get("category", []))))

        if not isinstance(bboxes, list):
            bboxes = [bboxes] if bboxes else []
        if not isinstance(cats, list):
            cats = [cats] if cats else []

        obj_counts.append(len(bboxes))
        for c in cats:
            if c is not None:
                all_cats.add(c)

        for bbox in bboxes:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    vals = [float(v) for v in bbox]
                    if all(math.isfinite(v) for v in vals):
                        fmt = detect_bbox_format(vals, img_w, img_h)
                        bbox_formats.append(fmt)
                    else:
                        info["warnings"].append("Non-finite bbox values detected")
                except (TypeError, ValueError):
                    info["warnings"].append("Non-numeric bbox values detected")

    if bbox_formats:
        from collections import Counter
        fmt_counts = Counter(bbox_formats)
        info["bbox_format"] = fmt_counts.most_common(1)[0][0]

    if all_cats:
        info["num_classes"] = len(all_cats)
        info["categories_sample"] = sorted(str(c) for c in list(all_cats)[:20])

    if obj_counts:
        info["avg_objects_per_image"] = round(sum(obj_counts) / len(obj_counts), 2)
        info["min_objects"] = min(obj_counts)
        info["max_objects"] = max(obj_counts)

    info["warnings"] = list(set(info["warnings"]))
    return info


def validate_classification_schema(features: dict, dataset_name: str) -> list[str]:
    """Validate classification dataset has image + label columns."""
    errors = []
    column_names = set(features.keys())
    if "image" not in column_names:
        errors.append(f"Missing 'image' column. Found: {sorted(column_names)}")
    if "label" not in column_names and "labels" not in column_names:
        errors.append(f"Missing 'label' column. Found: {sorted(column_names)}")
    return errors


def inspect_classification_samples(samples: list[dict], features: dict) -> dict[str, Any]:
    """Deeper sample-level checks for classification datasets."""
    info: dict[str, Any] = {
        "label_column": None,
        "label_type": None,
        "num_classes": None,
        "class_names": [],
        "warnings": [],
    }
    column_names = set(features.keys())
    label_col = "label" if "label" in column_names else "labels" if "labels" in column_names else None
    if not label_col:
        return info
    info["label_column"] = label_col

    feat = features.get(label_col)
    if feat is not None:
        type_name = type(feat).__name__
        if "ClassLabel" in type_name:
            info["label_type"] = "ClassLabel"
            if hasattr(feat, "names"):
                info["num_classes"] = len(feat.names)
                info["class_names"] = feat.names[:20]
        elif hasattr(feat, "dtype"):
            info["label_type"] = str(feat.dtype)

    if info["num_classes"] is None:
        unique_labels: set = set()
        for sample in samples:
            val = sample.get(label_col)
            if val is not None:
                unique_labels.add(val)
        info["num_classes"] = len(unique_labels)
        if not info["class_names"]:
            info["class_names"] = sorted(str(v) for v in list(unique_labels)[:20])

    return info


def validate_segmentation_schema(features: dict, dataset_name: str) -> list[str]:
    """Validate segmentation dataset has image + mask columns."""
    errors = []
    column_names = set(features.keys())
    if "image" not in column_names:
        errors.append(f"Missing 'image' column. Found: {sorted(column_names)}")
    mask_cols = {"mask", "label", "annotation", "segmentation_mask", "masks", "segmentation"}
    if not (column_names & mask_cols):
        errors.append(f"No mask column found. Expected one of {sorted(mask_cols)}. Found: {sorted(column_names)}")
    return errors


def inspect_segmentation_samples(samples: list[dict], features: dict) -> dict[str, Any]:
    """Deeper sample-level checks for segmentation datasets."""
    info: dict[str, Any] = {
        "mask_column": None,
        "has_prompt": False,
        "prompt_type": None,
        "prompt_source": None,
        "warnings": [],
    }
    column_names = set(features.keys())

    mask_options = ["mask", "label", "annotation", "segmentation_mask", "masks", "segmentation"]
    for col in mask_options:
        if col in column_names:
            info["mask_column"] = col
            break

    prompt_cols = [c for c in column_names if "prompt" in c.lower()]
    bbox_cols = [c for c in column_names if c in ("bbox", "bboxes", "box", "boxes")]
    point_cols = [c for c in column_names if c in ("point", "points", "input_point", "input_points")]

    for sample in samples:
        if prompt_cols:
            raw = sample.get(prompt_cols[0])
            parsed = raw if isinstance(raw, dict) else _try_json(raw)
            if isinstance(parsed, dict):
                if "bbox" in parsed or "box" in parsed:
                    info["has_prompt"] = True
                    info["prompt_type"] = "bbox"
                    info["prompt_source"] = f"JSON column '{prompt_cols[0]}'"
                    break
                if "point" in parsed or "points" in parsed:
                    info["has_prompt"] = True
                    info["prompt_type"] = "point"
                    info["prompt_source"] = f"JSON column '{prompt_cols[0]}'"
                    break

    if not info["has_prompt"] and bbox_cols:
        info["has_prompt"] = True
        info["prompt_type"] = "bbox"
        info["prompt_source"] = f"column '{bbox_cols[0]}'"
    if not info["has_prompt"] and point_cols:
        info["has_prompt"] = True
        info["prompt_type"] = "point"
        info["prompt_source"] = f"column '{point_cols[0]}'"

    if not info["has_prompt"]:
        info["warnings"].append("No prompt column detected. SAM training needs bbox or point prompts.")

    return info


def _try_json(value) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json_mod.loads(value)
    except (json_mod.JSONDecodeError, TypeError):
        return None


VALIDATORS = {
    "detect": validate_detection_schema,
    "detect_yolo": validate_detection_schema,
    "classify": validate_classification_schema,
    "segment": validate_segmentation_schema,
}

INSPECTORS = {
    "detect": inspect_detection_samples,
    "detect_yolo": inspect_detection_samples,
    "classify": inspect_classification_samples,
    "segment": inspect_segmentation_samples,
}


def validate_dataset(
    dataset_name: str,
    task_type: str,
    split: str = "train",
    config: str | None = None,
    inspect: bool = False,
    num_samples: int = NUM_INSPECT_SAMPLES,
) -> dict:
    """
    Validate a HF Hub dataset for a given task type.

    Returns dict with keys:
        valid: bool
        errors: list[str]
        columns: list[str]
        num_rows: int
        config: str | None
        inspection: dict | None  (only when inspect=True)
    """
    if task_type not in VALIDATORS:
        return {
            "valid": False,
            "errors": [f"Unknown task type: {task_type}"],
            "columns": [],
            "num_rows": 0,
            "config": config,
            "inspection": None,
        }

    if config is None:
        configs = get_dataset_config_names(dataset_name)
        if len(configs) == 1:
            config = configs[0]
        elif "default" in configs:
            config = "default"
        elif configs:
            config = configs[0]

    try:
        # Prefer a bounded non-streaming slice: streaming mode often leaves HF/fsspec
        # worker threads running; interpreter shutdown then blocks after validation
        # output is printed, so the CLI appears "stuck" on success.
        slice_split = f"{split}[:{num_samples}]"
        ds = load_dataset(dataset_name, config, split=slice_split, streaming=False)
        samples = [ds[i] for i in range(len(ds))]
        features = ds.features
    except Exception:
        try:
            ds = load_dataset(dataset_name, config, split=split, streaming=True)
            samples = []
            for i, sample in enumerate(ds):
                samples.append(sample)
                if i + 1 >= num_samples:
                    break
            features = ds.features
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Failed to load dataset: {e}"],
                "columns": [],
                "num_rows": 0,
                "config": config,
                "inspection": None,
            }

    column_names = list(features.keys())
    validator = VALIDATORS[task_type]
    errors = validator(features, dataset_name)

    try:
        info = dataset_info(dataset_name, config)
        num_rows = 0
        if info.splits:
            for s in info.splits.values():
                if s.name == split:
                    num_rows = s.num_examples
                    break
    except Exception:
        num_rows = -1

    inspection = None
    if inspect and samples:
        inspector = INSPECTORS.get(task_type)
        if inspector:
            inspection = inspector(samples, features)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "columns": column_names,
        "num_rows": num_rows,
        "config": config,
        "inspection": inspection,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a HF dataset for vision training")
    parser.add_argument("--dataset", required=True, help="HF Hub dataset name")
    parser.add_argument(
        "--task",
        required=True,
        choices=["detect", "detect_yolo", "classify", "segment"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None, help="Dataset config name")
    parser.add_argument("--inspect", action="store_true", help="Run deeper sample-level inspection")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print(f"Validating {args.dataset} for task={args.task}, split={args.split}...")
    result = validate_dataset(
        args.dataset, args.task, args.split, args.config, inspect=args.inspect
    )

    if args.json_output:
        print(json_mod.dumps(result, indent=2, default=str))
        if not result["valid"]:
            sys.exit(1)
        return

    print(f"  Config: {result['config']}")
    print(f"  Columns: {result['columns']}")
    print(f"  Rows: {result['num_rows']}")

    if result["valid"]:
        print("  [OK] Dataset schema is valid")
    else:
        print("  [FAIL] Validation errors:")
        for err in result["errors"]:
            print(f"    - {err}")

    if result["inspection"]:
        print("  Inspection details:")
        for key, val in result["inspection"].items():
            if val is not None and val != [] and val != {}:
                print(f"    {key}: {val}")

    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
