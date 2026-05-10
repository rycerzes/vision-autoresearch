#!/usr/bin/env python3
"""Manage vision autoresearch benchmark jobs on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vision_lab.preflight_report import (
    build_preflight_report,
    print_preflight_report,
    resolve_task_from_config,
)
from vision_lab.task_registry import all_task_ids, task_script_map

RUNTIME_DIR = ROOT / ".runtime"
LAST_JOB_PATH = RUNTIME_DIR / "hf-job-last.json"
HF_JOB_STATE_DIR = RUNTIME_DIR / "hf-jobs"
HF_JOB_LOG_DIR = RUNTIME_DIR / "hf-logs"
TERMINAL_JOB_STAGES = {
    "COMPLETED",
    "CANCELED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "ERROR",
}
DEFAULT_NAMESPACE = os.environ.get("VISION_HF_NAMESPACE")

TASK_SCRIPTS = task_script_map()
_CLI_TASKS = list(all_task_ids())


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def resolve_hf_cli() -> str:
    explicit = os.environ.get("VISION_HF_CLI")
    if explicit:
        return explicit
    fallback = shutil.which("hf")
    if fallback:
        return fallback
    raise SystemExit("could not find `hf`; install the Hugging Face CLI first")


def git_output(*argv: str) -> str | None:
    result = subprocess.run(
        list(argv), cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def run_command(
    argv: list[str], capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_EXPERIMENTAL_WARNING", "1")
    return subprocess.run(
        argv, text=True, capture_output=capture_output, check=False, env=env
    )


def parse_job_id(text: str) -> str | None:
    matches = re.findall(r"\b[0-9a-f]{24}\b", text)
    return matches[-1] if matches else None


def parse_metrics(text: str) -> dict | None:
    from parse_metric import parse_summary

    parsed = parse_summary(text)
    return parsed if parsed else None


def persist_job_state(state: dict) -> None:
    job_id = state.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return
    HF_JOB_STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = HF_JOB_STATE_DIR / f"{job_id}.json"
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def env_context() -> dict[str, str]:
    context: dict[str, str] = {}
    for env_name, key in (
        ("VISION_CAMPAIGN", "campaign"),
        ("VISION_EXPERIMENT_ID", "experiment_id"),
        ("VISION_WORKER_ID", "worker_id"),
        ("VISION_HYPOTHESIS", "hypothesis"),
    ):
        value = os.environ.get(env_name)
        if isinstance(value, str) and value.strip():
            context[key] = value.strip()
    return context


def collect_launch_context(task: str, config_path: Path) -> dict:
    context: dict = {
        "workspace": str(ROOT),
        "launched_at": now_utc_iso(),
        "task": task,
        "config": str(config_path),
    }
    git_commit = git_output("git", "rev-parse", "HEAD")
    if git_commit:
        context["git_commit"] = git_commit
    branch = git_output("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        context["branch"] = branch
    context.update(env_context())

    master_data = load_json_file(ROOT / "research" / "live" / "master.json")
    if master_data:
        master_hash = master_data.get("hash")
        if isinstance(master_hash, str) and master_hash:
            context["master_hash"] = master_hash
    return context


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_text(encoding="utf-8").encode("utf-8")).decode(
        "ascii"
    )


def encode_vision_lab_tree_b64() -> str:
    """Zip ``scripts/vision_lab`` so HF Jobs workdir can run ``compile_launch_contract``."""
    buf = io.BytesIO()
    scripts_root = ROOT / "scripts"
    base = scripts_root / "vision_lab"
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            arc = path.relative_to(scripts_root)
            zf.write(path, arc.as_posix())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_bundle_script(task: str, config_path: Path) -> str:
    """Build a self-contained script that HF Jobs can run via `uv run`."""
    train_script_path = ROOT / TASK_SCRIPTS[task]
    train_payload = encode_file(train_script_path)
    config_payload = encode_file(config_path)
    config_name = config_path.name
    train_name = TASK_SCRIPTS[task]
    vision_lab_zip_b64 = encode_vision_lab_tree_b64()
    task_json = json.dumps(task)

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found]

    with (ROOT / "pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    deps = pyproject.get("project", {}).get("dependencies", [])
    requires_python = pyproject.get("project", {}).get("requires-python", ">=3.10")

    dep_lines = "\n".join(f'#   "{d}",' for d in deps)
    header = f"""# /// script
# requires-python = "{requires_python}"
# dependencies = [
{dep_lines}
# ]
# ///"""

    return f'''{header}
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

TRAIN_PAYLOAD = "{train_payload}"
CONFIG_PAYLOAD = "{config_payload}"
TRAIN_NAME = "{train_name}"
CONFIG_NAME = "{config_name}"
VISION_LAB_ZIP = "{vision_lab_zip_b64}"
TASK_ID = {task_json}


def main() -> int:
    workdir = Path(os.environ.get("VISION_JOB_WORKDIR", "/tmp/vision-autoresearch"))
    workdir.mkdir(parents=True, exist_ok=True)

    train_path = workdir / TRAIN_NAME
    config_path = workdir / CONFIG_NAME
    train_path.write_text(base64.b64decode(TRAIN_PAYLOAD).decode("utf-8"))
    config_path.write_text(base64.b64decode(CONFIG_PAYLOAD).decode("utf-8"))

    deps = workdir / "_vision_lab_scripts"
    deps.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(VISION_LAB_ZIP))).extractall(deps)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(deps)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    contract_path = workdir / "run_contract.resolved.yaml"
    compile_rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vision_lab.compile_launch_contract",
            TASK_ID,
            str(config_path),
            str(contract_path),
        ],
        cwd=str(workdir),
        env=env,
    )
    if compile_rc.returncode != 0:
        return compile_rc.returncode

    log_path = workdir / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            [sys.executable, str(train_path), str(contract_path)],
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
'''


def default_flavor(task: str) -> str:
    env_val = os.environ.get("VISION_HF_FLAVOR")
    if env_val:
        return env_val
    return "l4"


def default_timeout(task: str) -> str:
    env_val = os.environ.get("VISION_HF_TIMEOUT")
    if env_val:
        return env_val
    return "2h"


def slugify(value: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:max_len].rstrip("_")


def build_job_labels(task: str, context: dict) -> list[str]:
    labels = ["vision-autoresearch", f"task={task}", "launcher=hf-job-py"]
    master_hash = context.get("master_hash")
    if isinstance(master_hash, str) and master_hash:
        labels.append(f"master={master_hash[:12]}")
    for ctx_key, label_key in (
        ("campaign", "campaign"),
        ("experiment_id", "experiment"),
        ("hypothesis", "hypothesis"),
    ):
        value = context.get(ctx_key)
        if isinstance(value, str) and value:
            labels.append(f"{label_key}={slugify(value)}")
    return labels


def preflight_command(args: argparse.Namespace) -> int:
    config_path = (
        Path(args.config)
        if args.config
        else ROOT / "configs" / f"base_{args.task}.yaml"
    )
    report = build_preflight_report(args.task, config_path)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_preflight_report(report)
    return 2 if report.get("errors") else 0


def launch_job(args: argparse.Namespace) -> int:
    config_path = (
        Path(args.config)
        if args.config
        else ROOT / "configs" / f"base_{args.task}.yaml"
    )
    task = args.task
    if not task:
        task = resolve_task_from_config(config_path)
    if not task:
        raise SystemExit("Could not determine task; pass --task explicitly")
    if task not in TASK_SCRIPTS:
        raise SystemExit(f"Unknown task: {task}")

    context = collect_launch_context(task, config_path)

    report = build_preflight_report(task, config_path)
    print_preflight_report(report)
    if report.get("errors"):
        raise SystemExit("Preflight failed. Fix errors or pass --allow-preflight-fail.")

    bundle_path = RUNTIME_DIR / "vision-hf-job.py"
    bundle_text = build_bundle_script(task, config_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(bundle_text, encoding="utf-8")

    flavor = args.flavor or default_flavor(task)
    timeout = args.timeout or default_timeout(task)
    hf_cli = resolve_hf_cli()

    command = [hf_cli, "jobs", "uv", "run", "--flavor", flavor, "--timeout", timeout]
    if args.namespace:
        command.extend(["--namespace", args.namespace])
    command.append("--detach")

    for label in build_job_labels(task, context):
        command.extend(["--label", label])
    for env_entry in args.env:
        command.extend(["--env", env_entry])
    command.extend(["--secrets", "HF_TOKEN"])
    command.append(str(bundle_path))

    print("\nLaunching HF Job:")
    print("  " + " ".join(shlex.quote(part) for part in command))
    result = run_command(command, capture_output=True)
    combined = (result.stdout or "") + (result.stderr or "")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    state = {
        "task": task,
        "config": str(config_path),
        "bundle_path": str(bundle_path),
        "flavor": flavor,
        "timeout": timeout,
        "command": command,
    }
    state.update(context)
    job_id = parse_job_id(combined)
    if job_id:
        state["job_id"] = job_id
    if args.namespace:
        state["namespace"] = args.namespace
    write_json_file(LAST_JOB_PATH, state)
    persist_job_state(state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def resolve_job_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    if LAST_JOB_PATH.exists():
        data = json.loads(LAST_JOB_PATH.read_text(encoding="utf-8"))
        job_id = data.get("job_id")
        if isinstance(job_id, str) and job_id:
            return job_id
    raise SystemExit("Job ID required; pass one explicitly or launch a job first")


def stream_logs(args: argparse.Namespace) -> int:
    job_id = resolve_job_id(args.job_id)
    argv = [resolve_hf_cli(), "jobs", "logs"]
    if args.follow:
        argv.append("--follow")
    if args.namespace:
        argv.extend(["--namespace", args.namespace])
    argv.append(job_id)

    local_log_path = HF_JOB_LOG_DIR / f"{job_id}.log"
    local_log_path.parent.mkdir(parents=True, exist_ok=True)
    output_handles = [local_log_path.open("w", encoding="utf-8")]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.resolve() != local_log_path.resolve():
            output_handles.append(args.output.open("w", encoding="utf-8"))

    collected: list[str] = []
    try:
        proc = subprocess.Popen(
            argv,
            env={**os.environ, "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            collected.append(line)
            for h in output_handles:
                h.write(line)
        rc = proc.wait()
    finally:
        for h in output_handles:
            h.close()

    metrics = parse_metrics("".join(collected))
    state = load_json_file(HF_JOB_STATE_DIR / f"{job_id}.json") or {"job_id": job_id}
    state["cached_log_path"] = str(local_log_path)
    if args.output:
        state["output_log_path"] = str(args.output)
    if metrics:
        state["metrics"] = metrics
    persist_job_state(state)

    last = load_json_file(LAST_JOB_PATH)
    if isinstance(last, dict) and last.get("job_id") == job_id:
        last["cached_log_path"] = str(local_log_path)
        if metrics:
            last["metrics"] = metrics
        write_json_file(LAST_JOB_PATH, last)

    if metrics:
        print(
            json.dumps({"job_id": job_id, "metrics": metrics}, indent=2, sort_keys=True)
        )
    return rc


def inspect_job(args: argparse.Namespace) -> int:
    job_id = resolve_job_id(args.job_id)
    argv = [resolve_hf_cli(), "jobs", "inspect"]
    if args.namespace:
        argv.extend(["--namespace", args.namespace])
    argv.append(job_id)
    return run_command(argv).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage vision autoresearch benchmark jobs on HF Jobs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_p = subparsers.add_parser(
        "preflight", help="Audit config before launching"
    )
    preflight_p.add_argument(
        "--task",
        required=True,
        choices=_CLI_TASKS,
    )
    preflight_p.add_argument(
        "--config", help="Config YAML path (defaults to configs/base_<task>.yaml)"
    )
    preflight_p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    preflight_p.add_argument("--json", action="store_true")

    launch_p = subparsers.add_parser("launch", help="Bundle and submit an HF Job")
    launch_p.add_argument(
        "--task",
        required=True,
        choices=_CLI_TASKS,
    )
    launch_p.add_argument(
        "--config", help="Config YAML path (defaults to configs/base_<task>.yaml)"
    )
    launch_p.add_argument("--flavor", help="HF Jobs flavor (default: l4)")
    launch_p.add_argument("--timeout", help="HF Jobs timeout (default: 2h)")
    launch_p.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    launch_p.add_argument(
        "--env", action="append", default=[], help="Extra --env entries"
    )

    logs_p = subparsers.add_parser("logs", help="Stream or fetch HF Jobs logs")
    logs_p.add_argument("job_id", nargs="?")
    logs_p.add_argument("--follow", action="store_true")
    logs_p.add_argument("--output", type=Path, help="Write logs to this file")
    logs_p.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    inspect_p = subparsers.add_parser("inspect", help="Inspect HF Job status")
    inspect_p.add_argument("job_id", nargs="?")
    inspect_p.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        return preflight_command(args)
    if args.command == "launch":
        return launch_job(args)
    if args.command == "logs":
        return stream_logs(args)
    if args.command == "inspect":
        return inspect_job(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
