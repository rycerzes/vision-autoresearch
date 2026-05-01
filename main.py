#!/usr/bin/env python3
"""Vision-autoresearch CLI entrypoint."""
from __future__ import annotations

import argparse
import sys

from prepare import validate_dataset


def main():
    parser = argparse.ArgumentParser(description="Vision Autoresearch")
    sub = parser.add_subparsers(dest="command")

    # validate command
    val_parser = sub.add_parser("validate", help="Validate a dataset for a task")
    val_parser.add_argument("--dataset", required=True)
    val_parser.add_argument(
        "--task",
        required=True,
        choices=[
            "detect",
            "detect_yolo",
            "track_yolo",
            "segment_yolo",
            "classify_yolo",
            "pose_yolo",
            "obb_yolo",
            "classify",
            "segment",
        ],
    )
    val_parser.add_argument("--split", default="train")
    val_parser.add_argument("--config", default=None)

    args = parser.parse_args()

    if args.command == "validate":
        result = validate_dataset(args.dataset, args.task, args.split, args.config)
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
