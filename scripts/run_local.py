#!/usr/bin/env python3
"""Run a vision autoresearch experiment locally (no HF Jobs).

Uses CUDA_VISIBLE_DEVICES=0 when unset so single-GPU workstations pick GPU 0 by default.
Export CUDA_VISIBLE_DEVICES yourself to override (other indices or multi-GPU policy).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vision_lab.preflight_report import (
    build_preflight_report,
    print_preflight_report,
    resolve_task_from_config,
)
from vision_lab.task_registry import all_task_ids, task_script_map

TASK_SCRIPTS = task_script_map()
_CLI_TASKS = list(all_task_ids())


def main() -> int:
    # Pin single-GPU workstations unless the caller already exported CUDA_VISIBLE_DEVICES.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description="Run a vision experiment locally.")
    parser.add_argument(
        "--task",
        choices=_CLI_TASKS,
        help="Task type",
    )
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--output", type=Path, help="Write log to this file")
    parser.add_argument(
        "--submit", action="store_true", help="Auto-submit result via submit_patch.py"
    )
    parser.add_argument(
        "--comment",
        default=None,
        help="Comment for submit_patch.py when using --submit",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip vision_lab preflight (task/promotion/model/dataset adapter checks)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    task = args.task or resolve_task_from_config(config_path)
    if not task:
        raise SystemExit("Could not determine task type; pass --task explicitly")
    if task not in TASK_SCRIPTS:
        raise SystemExit(f"Unknown task: {task}")

    train_script = ROOT / TASK_SCRIPTS[task]
    if not train_script.exists():
        raise SystemExit(f"Training script not found: {train_script}")

    if not args.skip_preflight:
        report = build_preflight_report(task, config_path)
        print_preflight_report(report)
        if report.get("errors"):
            raise SystemExit(
                "Preflight failed. Fix errors above or pass --skip-preflight."
            )

    log_path = args.output or (
        ROOT / ".runtime" / "local-logs" / f"local-{task}-{int(time.time())}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running {train_script.name} with {config_path.name}")
    print(f"Log: {log_path}")

    argv = [sys.executable, str(train_script), str(config_path)]
    start = time.time()

    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
    rc = proc.wait()
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s with exit code {rc}")

    if rc != 0:
        print(f"Training failed. Log: {log_path}")
        return rc

    sys.path.insert(0, str(ROOT / "scripts"))
    from parse_metric import parse_summary

    log_text = log_path.read_text(encoding="utf-8")
    metrics = parse_summary(log_text)
    if metrics:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print("Warning: no VISION AUTORESEARCH SUMMARY block found in output")

    if args.submit:
        comment = (args.comment or "").strip() or f"local {task} run"
        submit_argv = [
            sys.executable,
            str(ROOT / "scripts" / "submit_patch.py"),
            "--log",
            str(log_path),
            "--config",
            str(config_path),
            "--task",
            task,
            "--comment",
            comment,
        ]
        print(f"\nSubmitting: {' '.join(submit_argv)}")
        submit_rc = subprocess.run(submit_argv, cwd=str(ROOT)).returncode
        if submit_rc != 0:
            print("submit_patch.py failed")
            return submit_rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
