#!/usr/bin/env python3
"""Restore config YAML from the current promoted local master."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_results import (
    CONFIGS_DIR,
    MASTER_DETAIL_PATH,
    MASTER_PATH,
    ROOT,
    current_promoted_row,
    ensure_results_ledger,
    load_json,
    write_json,
    build_master_snapshot,
)


def restore_config(task_type: str, force: bool = False) -> dict:
    """Restore the config YAML for task_type from the promoted master detail."""
    detail = load_json(MASTER_DETAIL_PATH)
    if not isinstance(detail, dict):
        raise RuntimeError(
            "No master_detail.json found. Run a first experiment and promote it."
        )

    stored_task = detail.get("task_type", "")
    if task_type and stored_task and stored_task != task_type:
        raise RuntimeError(
            f"Requested task={task_type} but master_detail has task={stored_task}"
        )
    task_type = task_type or stored_task

    config_content = detail.get("config_content")
    if not isinstance(config_content, str) or not config_content:
        raise RuntimeError("master_detail.json has no config_content")

    config_path = CONFIGS_DIR / f"base_{task_type}.yaml"
    if config_path.exists() and not force:
        existing = config_path.read_text(encoding="utf-8")
        if existing != config_content:
            raise RuntimeError(
                f"{config_path.relative_to(ROOT)} differs from promoted master. "
                "Use --force to overwrite."
            )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content, encoding="utf-8")

    rows = ensure_results_ledger(task_type)
    row = current_promoted_row(rows, task_type=task_type)
    if row:
        snapshot = build_master_snapshot(row)
        write_json(MASTER_PATH, snapshot)
    else:
        snapshot = {}

    return {
        "task_type": task_type,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_hash": detail.get("config_hash", ""),
        "snapshot": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore config YAML from the current promoted local master."
    )
    parser.add_argument(
        "--task",
        choices=[
            "detect",
            "classify",
            "segment",
            "detect_yolo",
            "track_yolo",
            "segment_yolo",
            "classify_yolo",
            "pose_yolo",
            "obb_yolo",
        ],
        help="Task type to restore (auto-detected from master_detail if omitted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite config YAML even if it differs from the promoted master",
    )
    args = parser.parse_args()

    try:
        result = restore_config(args.task or "", force=args.force)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    snapshot = result.get("snapshot", {})
    metric = snapshot.get("promotion_metric", "")
    value = snapshot.get("promotion_metric_value", "unknown")
    print(f"Restored {result['config_path']} from promoted master")
    print(f"  task: {result['task_type']}")
    print(f"  config_hash: {result['config_hash']}")
    if metric:
        print(f"  {metric}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
