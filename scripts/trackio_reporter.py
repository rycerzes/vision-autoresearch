#!/usr/bin/env python3
"""Trackio reporter for vision autoresearch experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
STATE_PATH = RUNTIME_DIR / "trackio-reporter-state.json"
DEFAULT_PROJECT = os.environ.get("VISION_TRACKIO_PROJECT", "vision-autoresearch")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"runs": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"runs": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_step_metrics(text: str) -> list[dict[str, Any]]:
    """Parse HF Trainer log lines for step-level metrics."""
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"\{'loss':\s*(?P<loss>[0-9.]+),.*?'learning_rate':\s*(?P<lr>[0-9.e-]+),.*?'epoch':\s*(?P<epoch>[0-9.]+)"
    )
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    "loss": float(match.group("loss")),
                    "learning_rate": float(match.group("lr")),
                    "epoch": float(match.group("epoch")),
                }
            )
    return rows


def parse_eval_metrics(text: str) -> list[dict[str, Any]]:
    """Parse eval result lines."""
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"\{'eval_loss':\s*(?P<loss>[0-9.]+)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            row: dict[str, Any] = {"eval_loss": float(match.group("loss"))}
            for key in (
                "eval_mAP",
                "eval_mAP_50",
                "eval_accuracy",
                "eval_iou",
                "eval_dice",
            ):
                km = re.search(rf"'{key}':\s*([0-9.]+)", line)
                if km:
                    row[key] = float(km.group(1))
            rows.append(row)
    return rows


def report_to_trackio(log_path: Path, run_name: str, project: str) -> int:
    try:
        import trackio
    except ImportError:
        print("trackio not installed; skipping report", file=sys.stderr)
        return 1

    text = log_path.read_text(encoding="utf-8")
    step_metrics = parse_step_metrics(text)
    eval_metrics = parse_eval_metrics(text)

    if not step_metrics and not eval_metrics:
        print("No metrics found in log to report")
        return 1

    run = trackio.Run(project=project, name=run_name)
    for i, row in enumerate(step_metrics):
        run.log(row, step=i)
    for i, row in enumerate(eval_metrics):
        run.log(row, step=len(step_metrics) + i)
    run.finish()

    print(
        f"Reported {len(step_metrics)} train steps + {len(eval_metrics)} eval results to {project}/{run_name}"
    )
    return 0


def summary_command(args: argparse.Namespace) -> int:
    """Print a summary of recent runs from the results ledger."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from local_results import load_results_rows, parse_float, truthy

    rows = load_results_rows()
    if not rows:
        print("No runs recorded yet.")
        return 0

    print(f"{'run_id':<25} {'task':<10} {'metric':<10} {'value':<10} {'promoted'}")
    print("-" * 70)
    for row in rows[-20:]:
        metric = row.get("promotion_metric", "")
        value = row.get("promotion_metric_value", "")
        promoted = "[PROMOTED]" if truthy(row.get("promoted")) else ""
        print(
            f"{row.get('run_id', ''):<25} {row.get('task_type', ''):<10} {metric:<10} {value:<10} {promoted}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vision autoresearch Trackio reporter."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_p = subparsers.add_parser("report", help="Report a log to Trackio")
    report_p.add_argument("--log", type=Path, required=True, help="Training log path")
    report_p.add_argument("--name", required=True, help="Run name")
    report_p.add_argument("--project", default=DEFAULT_PROJECT, help="Trackio project")

    subparsers.add_parser("summary", help="Print summary of recent runs")

    args = parser.parse_args()
    if args.command == "report":
        return report_to_trackio(args.log, args.name, args.project)
    if args.command == "summary":
        return summary_command(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
