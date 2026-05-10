#!/usr/bin/env python3
"""Compile an experiment YAML/JSON config into a validated RunContract file on disk.

Equivalent to ``prepare.py --emit-contract`` but focused on contract compilation only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from vision_lab.compile_launch_contract import compile_config_file_to_path
from vision_lab.preflight_report import resolve_task_from_config
from vision_lab.task_registry import all_task_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile experiment config to validated RunContract YAML."
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML/JSON path")
    parser.add_argument(
        "--task",
        choices=list(all_task_ids()),
        help="Task id (default: task_type from config)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output RunContract YAML path",
    )
    args = parser.parse_args()
    cfg = args.config.expanduser().resolve()
    if not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")
    task = args.task or resolve_task_from_config(cfg)
    if not task:
        raise SystemExit("Could not determine task; pass --task or set task_type in config")
    out = args.output.expanduser().resolve()
    compile_config_file_to_path(task_id=task, config_path=cfg, output_path=out)
    print(f"Wrote RunContract to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
