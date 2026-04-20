#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Dataset Format Inspector for Vision Autoresearch.

Uses the HF Datasets Server API for instant results -- no dataset download needed.
Checks compatibility with object detection, image classification, and SAM segmentation.

Usage:
    uv run scripts/dataset_inspector.py --dataset cppe-5 --split train
    uv run scripts/dataset_inspector.py --dataset food101 --split train --json-output
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.parse
from collections import Counter
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect dataset format for vision model training"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name on HF Hub"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Dataset config (default: default)",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of samples to fetch"
    )
    parser.add_argument("--json-output", action="store_true", help="Output as JSON")
    return parser.parse_args()


def api_request(url: str) -> dict | None:
    """Make API request to Datasets Server."""
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"API request failed: {e.code} {e.reason}")


def get_splits(dataset: str) -> dict | None:
    url = f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(dataset)}"
    return api_request(url)


def get_rows(
    dataset: str, config: str, split: str, offset: int = 0, length: int = 5
) -> dict | None:
    url = (
        f"https://datasets-server.huggingface.co/rows?"
        f"dataset={urllib.parse.quote(dataset)}&config={config}"
        f"&split={split}&offset={offset}&length={length}"
    )
    return api_request(url)


def find_columns(columns: list[str], patterns: list[str]) -> list[str]:
    return [c for c in columns if any(p in c.lower() for p in patterns)]


def detect_bbox_format(
    bbox: list[float], image_size: tuple[int, int] | None = None
) -> str:
    if len(bbox) != 4:
        return "unknown"
    a, b, c, d = bbox
    is_normalized = all(0 <= v <= 1 for v in bbox)
    if c < a or d < b:
        return "xywh_normalized" if is_normalized else "xywh"
    if image_size is not None:
        img_w, img_h = image_size
        xywh_exceeds = (a + c > img_w * 1.05) or (b + d > img_h * 1.05)
        xyxy_exceeds = (c > img_w * 1.05) or (d > img_h * 1.05)
        if xywh_exceeds and not xyxy_exceeds:
            return "xyxy"
        if xyxy_exceeds and not xywh_exceeds:
            return "xywh"
    return "xyxy_normalized" if is_normalized else "xyxy"


def _extract_image_size(row: dict) -> tuple[int, int] | None:
    for col in ("image", "img"):
        img = row.get(col)
        if isinstance(img, dict):
            w, h = img.get("width"), img.get("height")
            if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                return (int(w), int(h))
    return None


def analyze_annotations(
    sample_rows: list[dict], annotation_cols: list[str]
) -> dict[str, Any]:
    if not annotation_cols:
        return {"found": False}

    annotation_col = annotation_cols[0]
    info: dict[str, Any] = {
        "found": True,
        "column": annotation_col,
        "bbox_formats": [],
        "categories_found": [],
        "avg_objects_per_image": 0,
        "max_objects": 0,
        "min_objects": float("inf"),
    }
    total_objects = 0
    valid_samples = 0

    for row_data in sample_rows:
        row = row_data["row"]
        ann = row.get(annotation_col)
        if not ann:
            continue
        valid_samples += 1
        image_size = _extract_image_size(row)

        if isinstance(ann, dict):
            bbox_key = (
                "bbox" if "bbox" in ann else "bboxes" if "bboxes" in ann else None
            )
            if bbox_key:
                bboxes = ann[bbox_key]
                if isinstance(bboxes, list) and bboxes:
                    if isinstance(bboxes[0], list):
                        total_objects += len(bboxes)
                        info["max_objects"] = max(info["max_objects"], len(bboxes))
                        info["min_objects"] = min(info["min_objects"], len(bboxes))
                        info["bbox_formats"].append(
                            detect_bbox_format(bboxes[0], image_size)
                        )
                    else:
                        total_objects += 1
                        info["max_objects"] = max(info["max_objects"], 1)
                        info["min_objects"] = min(info["min_objects"], 1)
                        info["bbox_formats"].append(
                            detect_bbox_format(bboxes, image_size)
                        )

            for key in ("category", "categories", "label", "labels"):
                if key in ann:
                    cats = ann[key]
                    if isinstance(cats, list):
                        info["categories_found"].extend(str(c) for c in cats)
                    else:
                        info["categories_found"].append(str(cats))

        elif isinstance(ann, list) and ann and isinstance(ann[0], dict):
            total_objects += len(ann)
            info["max_objects"] = max(info["max_objects"], len(ann))
            info["min_objects"] = min(info["min_objects"], len(ann))
            first = ann[0]
            if "bbox" in first:
                info["bbox_formats"].append(
                    detect_bbox_format(first["bbox"], image_size)
                )
            for key in ("category", "label"):
                if key in first:
                    for item in ann:
                        if key in item:
                            info["categories_found"].append(str(item[key]))

    if valid_samples > 0:
        info["avg_objects_per_image"] = round(total_objects / valid_samples, 2)
    if info["min_objects"] == float("inf"):
        info["min_objects"] = 0

    info["categories_found"] = list(set(info["categories_found"]))
    info["num_classes"] = len(info["categories_found"])

    if info["bbox_formats"]:
        info["primary_bbox_format"] = Counter(info["bbox_formats"]).most_common(1)[0][0]

    return info


def check_classification(columns: list[str], features: list[dict]) -> dict[str, Any]:
    image_cols = find_columns(columns, ["image", "img"])
    label_cols = find_columns(
        columns, ["label", "labels", "class", "fine_label", "coarse_label"]
    )
    label_info: dict[str, Any] = {"found": bool(label_cols)}

    if label_cols:
        label_info["column"] = label_cols[0]
        for f in features:
            if f.get("name") == label_cols[0]:
                ftype = f.get("type", "")
                if isinstance(ftype, dict) and ftype.get("_type") == "ClassLabel":
                    label_info["type"] = "ClassLabel"
                    names = ftype.get("names", [])
                    label_info["num_classes"] = len(names)
                    label_info["class_names"] = names[:20]
                elif isinstance(ftype, dict) and ftype.get("dtype") in (
                    "int64",
                    "int32",
                ):
                    label_info["type"] = "int"
                elif isinstance(ftype, dict) and ftype.get("dtype") == "string":
                    label_info["type"] = "string"
                break

    ready = bool(image_cols) and bool(label_cols)
    return {
        "ready": ready,
        "image_columns": image_cols,
        "label_columns": label_cols,
        "label_info": label_info,
    }


def check_detection(columns: list[str], sample_rows: list[dict]) -> dict[str, Any]:
    image_cols = find_columns(columns, ["image", "img"])
    annotation_cols = find_columns(
        columns, ["objects", "annotations", "bbox", "bboxes", "detection"]
    )
    bbox_cols = find_columns(columns, ["bbox", "bboxes", "boxes"])
    category_cols = find_columns(
        columns, ["category", "label", "class", "categories", "labels"]
    )

    annotations_info = (
        analyze_annotations(sample_rows, annotation_cols)
        if annotation_cols
        else {"found": False}
    )
    ready = bool(image_cols) and (
        bool(annotation_cols) or (bool(bbox_cols) and bool(category_cols))
    )

    return {
        "ready": ready,
        "image_columns": image_cols,
        "annotation_columns": annotation_cols,
        "separate_bbox_columns": bbox_cols,
        "separate_category_columns": category_cols,
        "annotations_info": annotations_info,
    }


def check_segmentation(columns: list[str], sample_rows: list[dict]) -> dict[str, Any]:
    image_cols = find_columns(columns, ["image", "img"])
    mask_cols = find_columns(columns, ["mask", "segmentation", "alpha", "matte"])
    prompt_cols = find_columns(columns, ["prompt"])
    bbox_cols = [c for c in columns if c in ("bbox", "bboxes", "box", "boxes")]
    point_cols = [c for c in columns if c in ("point", "points", "input_point")]

    prompt_info: dict[str, Any] = {"has_prompt": False, "prompt_type": None}

    if prompt_cols:
        for row_data in sample_rows:
            raw = row_data["row"].get(prompt_cols[0])
            parsed = raw if isinstance(raw, dict) else _try_json(raw)
            if isinstance(parsed, dict):
                if "bbox" in parsed or "box" in parsed:
                    prompt_info = {
                        "has_prompt": True,
                        "prompt_type": "bbox",
                        "source": f"column '{prompt_cols[0]}'",
                    }
                    break
                if "point" in parsed or "points" in parsed:
                    prompt_info = {
                        "has_prompt": True,
                        "prompt_type": "point",
                        "source": f"column '{prompt_cols[0]}'",
                    }
                    break

    if not prompt_info["has_prompt"] and bbox_cols:
        prompt_info = {
            "has_prompt": True,
            "prompt_type": "bbox",
            "source": f"column '{bbox_cols[0]}'",
        }
    if not prompt_info["has_prompt"] and point_cols:
        prompt_info = {
            "has_prompt": True,
            "prompt_type": "point",
            "source": f"column '{point_cols[0]}'",
        }

    ready = bool(image_cols) and bool(mask_cols) and prompt_info["has_prompt"]
    return {
        "ready": ready,
        "image_columns": image_cols,
        "mask_columns": mask_cols,
        "prompt_info": prompt_info,
    }


def _try_json(value) -> Any:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def main():
    args = parse_args()
    print("Fetching dataset info via Datasets Server API...")

    splits_data = get_splits(args.dataset)
    if not splits_data or "splits" not in splits_data:
        print(f"ERROR: Could not fetch splits for '{args.dataset}'")
        sys.exit(1)

    available_configs: set[str] = set()
    split_found = False
    config_to_use = args.config

    for si in splits_data["splits"]:
        available_configs.add(si["config"])
        if si["config"] == args.config and si["split"] == args.split:
            split_found = True

    if not split_found and available_configs:
        config_to_use = sorted(available_configs)[0]
        print(f"Config '{args.config}' not found, trying '{config_to_use}'...")

    rows_data = get_rows(args.dataset, config_to_use, args.split, length=args.samples)
    if not rows_data or "rows" not in rows_data or not rows_data["rows"]:
        print(f"ERROR: No rows found for '{args.dataset}' split='{args.split}'")
        sys.exit(1)

    rows = rows_data["rows"]
    first_row = rows[0]["row"]
    columns = list(first_row.keys())
    features = rows_data.get("features", [])

    total_examples = "Unknown"
    for si in splits_data["splits"]:
        if si["config"] == config_to_use and si["split"] == args.split:
            n = si.get("num_examples")
            total_examples = f"{n:,}" if isinstance(n, int) else "Unknown"
            break

    od_info = check_detection(columns, rows)
    ic_info = check_classification(columns, features)
    sam_info = check_segmentation(columns, rows)

    if args.json_output:
        result = {
            "dataset": args.dataset,
            "config": config_to_use,
            "split": args.split,
            "total_examples": total_examples,
            "columns": columns,
            "object_detection": od_info,
            "image_classification": ic_info,
            "sam_segmentation": sam_info,
        }
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"\nDataset: {args.dataset}")
    print(f"Config: {config_to_use}")
    print(f"Split: {args.split}")
    print(f"Total examples: {total_examples}")
    print(f"Columns: {', '.join(columns)}")

    if features:
        print("\nFeature types:")
        for f in features:
            print(f"  {f['name']}: {f['type']}")

    print(
        f"\nImage Classification: {'[OK] READY' if ic_info['ready'] else '[FAIL] NOT COMPATIBLE'}"
    )
    if ic_info["ready"]:
        li = ic_info["label_info"]
        if li.get("num_classes"):
            print(f"  {li['num_classes']} classes")
        print(f"  image={ic_info['image_columns'][0]}, label={li.get('column', '?')}")

    print(
        f"\nObject Detection: {'[OK] READY' if od_info['ready'] else '[FAIL] NOT COMPATIBLE'}"
    )
    if od_info["ready"]:
        ann = od_info["annotations_info"]
        if ann.get("found"):
            print(
                f"  column={ann['column']}, format={ann.get('primary_bbox_format', '?')}"
            )
            print(
                f"  {ann.get('num_classes', '?')} classes, avg {ann.get('avg_objects_per_image', '?')} objects/image"
            )

    print(
        f"\nSAM Segmentation: {'[OK] READY' if sam_info['ready'] else '[FAIL] NOT COMPATIBLE'}"
    )
    if sam_info["ready"]:
        pi = sam_info["prompt_info"]
        print(f"  prompt_type={pi['prompt_type']}, mask={sam_info['mask_columns'][0]}")
    elif sam_info.get("mask_columns"):
        pi = sam_info["prompt_info"]
        if not pi["has_prompt"]:
            print("  Has mask column but no prompt (bbox/point) detected")


if __name__ == "__main__":
    main()
