#!/usr/bin/env python3
"""
Dataset validation and preparation for vision-autoresearch experiments.

Validates Hub datasets or local dataset layouts via ``vision_lab.dataset_validation``.
See ``scripts/vision_lab/dataset_adapters/`` for supported filesystem layouts.

Usage:
    python prepare.py --dataset cppe-5 --task detect --split train
    python prepare.py --dataset /data/coco_subset --task detect_yolo --adapter yolo_folder
    python prepare.py --dataset food101 --task classify --split train
    python prepare.py --dataset ./voc_dataset --task detect --adapter voc_xml
    python prepare.py --dataset cppe-5 --task detect --split train --inspect
    python prepare.py --dataset cppe-5 --task detect --split train --json
"""
from __future__ import annotations

import argparse
import json as json_mod
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vision_lab.dataset_validation import (
    NUM_INSPECT_SAMPLES_DEFAULT,
    all_adapter_ids_cli,
    validate_dataset,
)
from vision_lab.task_registry import all_task_ids

NUM_INSPECT_SAMPLES = NUM_INSPECT_SAMPLES_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Validate a HF or local dataset for vision training")
    parser.add_argument("--dataset", required=True, help="HF Hub dataset id or local path")
    parser.add_argument(
        "--task",
        required=True,
        choices=list(all_task_ids()),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None, help="HF dataset config name (Hub only)")
    parser.add_argument(
        "--adapter",
        default="auto",
        choices=list(all_adapter_ids_cli()),
        help="Dataset adapter (auto: Hub id vs local path inference)",
    )
    parser.add_argument("--inspect", action="store_true", help="Run deeper sample-level inspection (Hub only)")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--no-cache-manifest",
        action="store_true",
        help="Do not write .runtime/datasets manifest after successful local validation",
    )
    args = parser.parse_args()

    print(f"Validating {args.dataset} for task={args.task}, split={args.split}...")
    result = validate_dataset(
        args.dataset,
        args.task,
        args.split,
        args.config,
        inspect=args.inspect,
        num_samples=NUM_INSPECT_SAMPLES_DEFAULT,
        adapter_id=args.adapter,
        write_cache=not args.no_cache_manifest,
    )

    if args.json_output:
        print(json_mod.dumps(result, indent=2, default=str))
        if not result["valid"]:
            sys.exit(1)
        return

    print(f"  Adapter: {result.get('adapter_id', '?')}")
    print(f"  Schema kind: {result.get('dataset_schema_kind', '?')}")
    print(f"  Config: {result['config']}")
    print(f"  Columns: {result['columns']}")
    print(f"  Rows: {result['num_rows']}")
    if result.get("compatible_tasks"):
        print(f"  Compatible tasks: {result['compatible_tasks']}")

    if result["valid"]:
        print("  [OK] Dataset schema is valid")
    else:
        print("  [FAIL] Validation errors:")
        for err in result["errors"]:
            print(f"    - {err}")

    for w in result.get("warnings") or []:
        print(f"  [WARN] {w}")

    if result.get("cache_manifest_path"):
        print(f"  Cache manifest: {result['cache_manifest_path']}")

    if result["inspection"]:
        print("  Inspection details:")
        for key, val in result["inspection"].items():
            if val is not None and val != [] and val != {}:
                print(f"    {key}: {val}")

    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
