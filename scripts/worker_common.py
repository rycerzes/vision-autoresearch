#!/usr/bin/env python3
"""Worktree isolation for vision autoresearch experiment workers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".runtime"
WORKTREE_ROOT = RUNTIME_DIR / "worktrees"
STATE_DIR = RUNTIME_DIR / "vision-workers"
EXPERIMENT_DIR = ROOT / "research" / "experiments"
LIVE_DIR = ROOT / "research" / "live"
MASTER_PATH = LIVE_DIR / "master.json"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_id(name: str, value: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise SystemExit(f"{name} must match {ID_PATTERN.pattern!r}: {value!r}")
    return value


def run(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=cwd or ROOT, text=True, capture_output=True, check=False
    )


def load_master_snapshot() -> dict:
    if not MASTER_PATH.exists():
        return {}
    try:
        return json.loads(MASTER_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def worker_state_path(experiment_id: str) -> Path:
    return STATE_DIR / f"{experiment_id}.json"


def worktree_path(experiment_id: str) -> Path:
    return WORKTREE_ROOT / experiment_id


def write_state(state: dict) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = worker_state_path(str(state["experiment_id"]))
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_state(experiment_id: str) -> dict:
    path = worker_state_path(experiment_id)
    if not path.exists():
        raise SystemExit(f"Missing worker state: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse worker state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Unexpected worker state in {path}")
    return payload


def ensure_worktree(target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    result = run(["git", "worktree", "add", "--detach", str(target)])
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "git worktree add failed")


def worker_env(state: dict) -> dict[str, str]:
    return {
        "VISION_CAMPAIGN": str(state.get("campaign", "")),
        "VISION_EXPERIMENT_ID": str(state["experiment_id"]),
        "VISION_WORKER_ID": str(state.get("worker_id", "")),
        "VISION_HYPOTHESIS": str(state.get("hypothesis", "")),
    }


def create_worker_state(
    experiment_id: str,
    campaign: str,
    hypothesis: str,
    task: str,
    config: str | None = None,
    worker_id: str | None = None,
) -> tuple[dict, Path]:
    experiment_id = ensure_id("experiment_id", experiment_id)
    campaign = ensure_id("campaign", campaign)
    worker_id = worker_id or f"w-{experiment_id}"

    master = load_master_snapshot()
    wt = worktree_path(experiment_id)
    ensure_worktree(wt)

    config_path = config or f"configs/base_{task}.yaml"

    state = {
        "experiment_id": experiment_id,
        "campaign": campaign,
        "hypothesis": hypothesis,
        "task": task,
        "config": config_path,
        "worker_id": worker_id,
        "master_hash": master.get("hash", ""),
        "worktree_path": str(wt),
        "created_at": utc_now(),
        "status": "created",
    }
    path = write_state(state)

    src_config = ROOT / config_path
    if src_config.exists():
        dst_config = wt / config_path
        dst_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_config, dst_config)

    return state, path


def list_workers() -> list[dict]:
    if not STATE_DIR.exists():
        return []
    workers = []
    for path in sorted(STATE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                workers.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return workers


def cleanup_worktree(experiment_id: str) -> None:
    wt = worktree_path(experiment_id)
    if wt.exists():
        run(["git", "worktree", "remove", "--force", str(wt)])
    state_path = worker_state_path(experiment_id)
    if state_path.exists():
        state_path.unlink()
