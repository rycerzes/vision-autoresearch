#!/usr/bin/env python3
"""Vision-autoresearch CLI entrypoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from vision_lab.dataset_validation import all_adapter_ids_cli, validate_dataset
from vision_lab.task_registry import all_task_ids


def main():
    parser = argparse.ArgumentParser(description="Vision Autoresearch")
    sub = parser.add_subparsers(dest="command")

    # validate command
    val_parser = sub.add_parser("validate", help="Validate a dataset for a task")
    val_parser.add_argument("--dataset", required=True)
    val_parser.add_argument(
        "--task",
        required=True,
        choices=list(all_task_ids()),
    )
    val_parser.add_argument("--split", default="train")
    val_parser.add_argument("--config", default=None)
    val_parser.add_argument(
        "--adapter",
        default="auto",
        choices=list(all_adapter_ids_cli()),
        help="Dataset adapter (see prepare.py)",
    )

    args = parser.parse_args()

    if args.command == "validate":
        result = validate_dataset(
            args.dataset,
            args.task,
            args.split,
            args.config,
            adapter_id=args.adapter,
            write_cache=False,
        )
        if result["valid"]:
            print(f"[OK] {args.dataset} is valid for {args.task}")
        else:
            print(f"[FAIL] {args.dataset} validation failed:")
            for err in result["errors"]:
                print(f"  - {err}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
