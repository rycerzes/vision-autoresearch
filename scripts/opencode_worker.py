#!/usr/bin/env python3
"""Create, run, and clean isolated OpenCode vision autoresearch experiment workers."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker_common import (
    build_worker_contract,
    cleanup_worktree,
    create_worker_state,
    load_state,
    require_tool,
    worker_env,
)


def create_command(args: argparse.Namespace) -> int:
    state, state_path = create_worker_state(
        experiment_id=args.experiment_id,
        campaign=args.campaign,
        hypothesis=args.hypothesis,
        task=args.task,
        config=args.config,
        worker_id=args.worker_id,
    )
    run_cmd = ["uv", "run", "scripts/opencode_worker.py", "run", str(state["experiment_id"])]
    print(json.dumps(state, indent=2, sort_keys=True))
    print(f"state: {state_path}")
    print(f"run: {' '.join(shlex.quote(part) for part in run_cmd)}")
    return 0


def build_prompt(state: dict) -> str:
    return build_worker_contract(state)


def run_command_for_worker(args: argparse.Namespace) -> int:
    state = load_state(args.experiment_id)
    opencode_bin = (
        args.opencode_bin
        or os.environ.get("VISION_OPENCODE_BIN")
        or require_tool("opencode")
    )
    worktree = Path(str(state["worktree_path"]))
    if not worktree.exists():
        raise SystemExit(f"Missing worktree: {worktree}")

    prompt = build_prompt(state)
    env = os.environ.copy()
    env.update(worker_env(state))

    argv = [opencode_bin, "run", "--agent", "experiment-worker", prompt]
    if args.dry_run:
        print("cwd:", worktree)
        print("command:", " ".join(shlex.quote(part) for part in argv))
        for key in sorted(env):
            if key.startswith("VISION_"):
                print(f"{key}={env[key]}")
        return 0

    result = subprocess.run(argv, cwd=worktree, env=env, check=False)
    return result.returncode


def cleanup_command(args: argparse.Namespace) -> int:
    cleanup_worktree(args.experiment_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create, run, and clean isolated OpenCode vision autoresearch experiment workers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create", help="Create an isolated worktree and state file"
    )
    create.add_argument(
        "experiment_id",
        help="Stable experiment identifier used for worktree and state",
    )
    create.add_argument(
        "--campaign", required=True, help="Campaign name for this experiment"
    )
    create.add_argument(
        "--hypothesis", required=True, help="One-sentence experiment hypothesis"
    )
    create.add_argument(
        "--task",
        required=True,
        choices=["detect", "classify", "segment", "detect_yolo"],
        help="Vision task type",
    )
    create.add_argument("--config", help="Config YAML path (defaults to base config for task)")
    create.add_argument("--worker-id", help="Logical worker id; defaults to w-<experiment_id>")

    run_worker = subparsers.add_parser(
        "run", help="Run the isolated experiment worker through OpenCode"
    )
    run_worker.add_argument(
        "experiment_id", help="Experiment id created by the `create` command"
    )
    run_worker.add_argument("--opencode-bin", help="Override the OpenCode executable")
    run_worker.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact command and environment without running OpenCode",
    )

    cleanup = subparsers.add_parser(
        "cleanup", help="Remove a finished worktree and its local worker state"
    )
    cleanup.add_argument(
        "experiment_id", help="Experiment id created by the `create` command"
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "create":
        return create_command(args)
    if args.command == "run":
        return run_command_for_worker(args)
    if args.command == "cleanup":
        return cleanup_command(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
