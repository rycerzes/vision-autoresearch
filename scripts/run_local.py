#!/usr/bin/env python3
"""Run a vision autoresearch experiment locally (no HF Jobs)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASK_SCRIPTS = {
    "detect": "train_detect.py",
    "classify": "train_classify.py",
    "segment": "train_segment.py",
}


def resolve_task_from_config(config_path: Path) -> str | None:
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data.get("task_type") if isinstance(data, dict) else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a vision experiment locally.")
    parser.add_argument(
        "--task", choices=["detect", "classify", "segment"], help="Task type"
    )
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--output", type=Path, help="Write log to this file")
    parser.add_argument(
        "--submit", action="store_true", help="Auto-submit result via submit_patch.py"
    )
    parser.add_argument(
        "--comment", help="Comment for submit_patch (required if --submit)"
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
        comment = args.comment or f"local {task} run"
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
