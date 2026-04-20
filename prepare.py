#!/usr/bin/env python3
"""
Dataset validation and preparation for vision-autoresearch experiments.

Validates that a HF Hub dataset has the required schema for a given task type.
Read-only during experiments — never modified by experiment workers.

Usage:
    python prepare.py --dataset cppe-5 --task detect --split train
    python prepare.py --dataset food101 --task classify --split train
    python prepare.py --dataset <name> --task segment --split train
"""
from __future__ import annotations

import argparse
import sys

from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import dataset_info


# Schema requirements per task type

TASK_SCHEMAS = {
    "detect": {
        "required_columns": {"image"},
        "bbox_columns": [
            # Common bbox column patterns (checked in order)
            ("objects", "bbox", "category"),       # CPPE-5 style
            ("objects", "bbox", "label"),           # alternative
            ("bboxes", None, "labels"),             # flat style
        ],
        "description": "Object detection requires 'image' + bounding boxes + categories",
    },
    "classify": {
        "required_columns": {"image", "label"},
        "description": "Classification requires 'image' + 'label' columns",
    },
    "segment": {
        "required_columns": {"image"},
        "mask_columns": ["mask", "label", "annotation", "segmentation_mask"],
        "description": "Segmentation requires 'image' + a mask/annotation column",
    },
}


def validate_detection_schema(features: dict, dataset_name: str) -> list[str]:
    """Validate detection dataset has bbox + category columns."""
    errors = []
    column_names = set(features.keys())

    if "image" not in column_names:
        errors.append(f"Missing 'image' column. Found: {sorted(column_names)}")
        return errors

    # Check for nested objects column (CPPE-5 style: objects.bbox, objects.category)
    if "objects" in column_names:
        obj_feature = features["objects"]
        # Handle Sequence of struct
        if hasattr(obj_feature, "feature"):
            inner = obj_feature.feature
            if hasattr(inner, "keys"):
                inner_keys = set(inner.keys())
            else:
                inner_keys = set()
        elif hasattr(obj_feature, "keys"):
            inner_keys = set(obj_feature.keys())
        else:
            inner_keys = set()

        has_bbox = "bbox" in inner_keys or "bboxes" in inner_keys
        has_cat = "category" in inner_keys or "label" in inner_keys or "categories" in inner_keys

        if not has_bbox:
            errors.append(f"'objects' column exists but missing bbox sub-field. Found: {sorted(inner_keys)}")
        if not has_cat:
            errors.append(f"'objects' column exists but missing category sub-field. Found: {sorted(inner_keys)}")
        return errors

    # Check for flat bbox columns
    has_bbox = any(c in column_names for c in ("bboxes", "bbox", "boxes"))
    has_cat = any(c in column_names for c in ("labels", "label", "categories", "category"))

    if not has_bbox:
        errors.append(f"No bbox column found. Expected 'objects.bbox', 'bboxes', or 'bbox'. Found: {sorted(column_names)}")
    if not has_cat:
        errors.append(f"No category column found. Expected 'objects.category', 'labels', or 'label'. Found: {sorted(column_names)}")

    return errors


def validate_classification_schema(features: dict, dataset_name: str) -> list[str]:
    """Validate classification dataset has image + label columns."""
    errors = []
    column_names = set(features.keys())

    if "image" not in column_names:
        errors.append(f"Missing 'image' column. Found: {sorted(column_names)}")
    if "label" not in column_names and "labels" not in column_names:
        errors.append(f"Missing 'label' column. Found: {sorted(column_names)}")

    return errors


def validate_segmentation_schema(features: dict, dataset_name: str) -> list[str]:
    """Validate segmentation dataset has image + mask columns."""
    errors = []
    column_names = set(features.keys())

    if "image" not in column_names:
        errors.append(f"Missing 'image' column. Found: {sorted(column_names)}")

    mask_cols = {"mask", "label", "annotation", "segmentation_mask", "masks", "segmentation"}
    found_mask = column_names & mask_cols
    if not found_mask:
        errors.append(
            f"No mask column found. Expected one of {sorted(mask_cols)}. Found: {sorted(column_names)}"
        )

    return errors


VALIDATORS = {
    "detect": validate_detection_schema,
    "classify": validate_classification_schema,
    "segment": validate_segmentation_schema,
}


def validate_dataset(dataset_name: str, task_type: str, split: str = "train", config: str | None = None) -> dict:
    """
    Validate a HF Hub dataset for a given task type.

    Returns dict with keys:
        valid: bool
        errors: list[str]
        columns: list[str]
        num_rows: int
        config: str | None
    """
    if task_type not in VALIDATORS:
        return {"valid": False, "errors": [f"Unknown task type: {task_type}"], "columns": [], "num_rows": 0, "config": config}

    # Resolve config if not provided
    if config is None:
        configs = get_dataset_config_names(dataset_name)
        if len(configs) == 1:
            config = configs[0]
        elif "default" in configs:
            config = "default"
        elif configs:
            config = configs[0]

    # Load a small sample to inspect schema
    try:
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
        # Take first sample to materialize features
        sample = next(iter(ds))
        features = ds.features
    except Exception as e:
        return {"valid": False, "errors": [f"Failed to load dataset: {e}"], "columns": [], "num_rows": 0, "config": config}

    column_names = list(features.keys())
    validator = VALIDATORS[task_type]
    errors = validator(features, dataset_name)

    # Get row count (non-streaming)
    try:
        info = dataset_info(dataset_name, config)
        num_rows = 0
        if info.splits and split in {s.name for s in info.splits.values() if hasattr(s, 'name')}:
            for s in info.splits.values():
                if s.name == split:
                    num_rows = s.num_examples
                    break
    except Exception:
        num_rows = -1

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "columns": column_names,
        "num_rows": num_rows,
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a HF dataset for vision training")
    parser.add_argument("--dataset", required=True, help="HF Hub dataset name")
    parser.add_argument("--task", required=True, choices=["detect", "classify", "segment"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None, help="Dataset config name")
    args = parser.parse_args()

    print(f"Validating {args.dataset} for task={args.task}, split={args.split}...")
    result = validate_dataset(args.dataset, args.task, args.split, args.config)

    print(f"  Config: {result['config']}")
    print(f"  Columns: {result['columns']}")
    print(f"  Rows: {result['num_rows']}")

    if result["valid"]:
        print("  [OK] Dataset schema is valid")
    else:
        print("  [FAIL] Validation errors:")
        for err in result["errors"]:
            print(f"    - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
