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
    python prepare.py --emit-contract /path/to/contract.yaml --experiment-config configs/base_classify.yaml
"""
from __future__ import annotations

import argparse
import json as json_mod
import sys
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vision_lab.compile_launch_contract import compile_config_file_to_path
from vision_lab.dataset_validation import (
    NUM_INSPECT_SAMPLES_DEFAULT,
    all_adapter_ids_cli,
    validate_dataset,
)
from vision_lab.task_registry import all_task_ids

NUM_INSPECT_SAMPLES = NUM_INSPECT_SAMPLES_DEFAULT


def _emit_contract_main(args: argparse.Namespace) -> None:
    cfg = args.experiment_config.expanduser().resolve()
    if not cfg.is_file():
        raise SystemExit(f"Experiment config not found: {cfg}")
    out = args.emit_contract.expanduser().resolve()
    raw = cfg.read_text(encoding="utf-8")
    if cfg.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(raw)
    elif cfg.suffix.lower() == ".json":
        data = json_mod.loads(raw)
    else:
        raise SystemExit("--experiment-config must be .yaml, .yml, or .json")
    if not isinstance(data, dict):
        raise SystemExit("Experiment config root must be a mapping")
    task = args.task or str(data.get("task_type", "")).strip()
    if not task:
        raise SystemExit("task is required (--task) or set task_type in the experiment config")
    if task not in all_task_ids():
        raise SystemExit(f"Unknown task {task!r}")
    compile_config_file_to_path(task_id=task, config_path=cfg, output_path=out)
    if args.json_output:
        print(json_mod.dumps({"contract_path": str(out), "task": task}, indent=2))
    else:
        print(f"Wrote resolved RunContract to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a HF or local dataset for vision training")
    parser.add_argument(
        "--emit-contract",
        type=Path,
        metavar="OUTPUT",
        default=None,
        help="Compile --experiment-config to a validated RunContract YAML at OUTPUT (Phase 5)",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        help="Experiment YAML/JSON path (required with --emit-contract)",
    )
    parser.add_argument("--dataset", default=None, help="HF Hub dataset id or local path")
    parser.add_argument(
        "--task",
        required=False,
        default=None,
        choices=list(all_task_ids()),
        help="Task type (for dataset validation, or optional override with --emit-contract)",
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
        help="Do not write dataset cache manifest after successful local validation",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Explicit directory for dataset cache manifests (overrides --run-output-dir)",
    )
    parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=None,
        help="Write cache manifest under <dir>/dataset (also via env VISION_RUN_OUTPUT_DIR)",
    )
    args = parser.parse_args()

    if args.emit_contract is not None:
        if args.experiment_config is None:
            raise SystemExit("--emit-contract requires --experiment-config")
        _emit_contract_main(args)
        return

    if not args.dataset:
        raise SystemExit("--dataset is required unless using --emit-contract")
    if not args.task:
        raise SystemExit("--task is required for dataset validation")

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
        run_output_dir=args.run_output_dir,
        cache_root=args.cache_root,
    )

    if args.json_output:
        print(json_mod.dumps(result, indent=2, default=str))
        if not result["valid"]:
            sys.exit(1)
        return

    print(f"  Adapter: {result.get('adapter_id', '?')}")
    print(f"  Schema kind: {result.get('dataset_schema_kind', '?')}")
    print(f"  Dataset config (HF subset): {result.get('dataset_config')}")
    print(f"  Columns: {result.get('columns')}")
    print(f"  Row counts: {result.get('row_counts')}")
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

    if result.get("inspection"):
        print("  Inspection details:")
        for key, val in result["inspection"].items():
            if val is not None and val != [] and val != {}:
                print(f"    {key}: {val}")

    if not result["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
