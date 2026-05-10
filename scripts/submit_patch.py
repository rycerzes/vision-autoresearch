#!/usr/bin/env python3
"""Record a completed run and promote if it beats the current local master."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml

from local_results import (
    CONFIGS_DIR,
    MASTER_DETAIL_PATH,
    ROOT,
    append_result_row,
    config_hash,
    current_master_snapshot,
    current_promoted_row,
    ensure_results_ledger,
    load_json,
    now_utc_iso,
    parse_float,
    rebuild_live_state,
    stringify_field,
    truthy,
    write_json,
    write_run_contract_artifact,
    write_run_metrics_artifact,
)
from parse_metric import parse_summary
from vision_lab.promotion import (
    assert_summary_eligible_for_recording,
    evaluate_promotion,
    evaluation_to_jsonable,
    load_promotion_policy,
    policy_to_jsonable,
)
from vision_lab.task_registry import all_task_ids

RUNTIME_DIR = ROOT / ".runtime"
LAST_JOB_PATH = RUNTIME_DIR / "hf-job-last.json"

_SUBMIT_TASK_CHOICES = list(all_task_ids())


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


def load_last_job() -> dict | None:
    if not LAST_JOB_PATH.exists():
        return None
    data = load_json(LAST_JOB_PATH)
    return data if isinstance(data, dict) else None


def resolve_config_path(task_type: str, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return CONFIGS_DIR / f"base_{task_type}.yaml"


def build_run_id(
    existing_rows: list[dict[str, str]], job_id: str | None, c_hash: str
) -> str:
    if job_id:
        base = f"job-{job_id}"
    else:
        base = f"run-{c_hash[:12]}"
    existing_ids = {row.get("run_id", "") for row in existing_rows}
    if base not in existing_ids:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing_ids:
        suffix += 1
    return f"{base}-{suffix}"


def write_master_config(config_path: Path, task_type: str) -> None:
    """Copy the promoted config into master_detail so refresh_master can restore it."""
    config_path = config_path.resolve()
    if not config_path.exists():
        return
    config_text = config_path.read_text(encoding="utf-8")
    detail = {
        "task_type": task_type,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_content": config_text,
        "config_hash": config_hash(config_path),
    }
    write_json(MASTER_DETAIL_PATH, detail)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a vision autoresearch run and promote if it beats the local master."
    )
    parser.add_argument(
        "--comment", required=True, help="One-sentence hypothesis summary"
    )
    parser.add_argument("--log", type=Path, help="Parse metrics from this log file")
    parser.add_argument(
        "--metrics-json", type=Path, help="Load metrics from a JSON file"
    )
    parser.add_argument("--config", type=Path, help="Config YAML used for this run")
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Resolved RunContract YAML/JSON written alongside metrics as contract.json",
    )
    parser.add_argument(
        "--task", choices=_SUBMIT_TASK_CHOICES, help="Task type"
    )
    parser.add_argument("--job-id", help="HF Job ID for this run")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    args = parser.parse_args()

    if args.metrics_json:
        metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    elif args.log:
        log_text = args.log.read_text(encoding="utf-8")
        metrics = parse_summary(log_text)
        if not metrics:
            raise SystemExit(f"No VISION AUTORESEARCH SUMMARY found in {args.log}")
    else:
        last_job = load_last_job()
        log_key = next(
            (k for k in ("cached_log_path", "output_log_path", "log_path") if k in (last_job or {})),
            None,
        )
        if last_job and log_key:
            log_path = Path(last_job[log_key])
            if log_path.exists():
                metrics = parse_summary(log_path.read_text(encoding="utf-8"))
            else:
                raise SystemExit(f"Last job log not found: {log_path}")
        else:
            raise SystemExit("Provide --log or --metrics-json, or run a job first")

    task_type = args.task or str(metrics.get("task_type", ""))
    if not task_type:
        raise SystemExit("Could not determine task_type; pass --task explicitly")
    if task_type not in _SUBMIT_TASK_CHOICES:
        raise SystemExit(f"Unknown task type: {task_type!r}")

    config_path = resolve_config_path(
        task_type, str(args.config) if args.config else None
    )
    c_hash = config_hash(config_path) if config_path.exists() else ""

    config_data = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )

    try:
        promotion_policy = load_promotion_policy(
            config_data if isinstance(config_data, dict) else {},
            task_id=task_type,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        assert_summary_eligible_for_recording(
            task_id=task_type,
            policy=promotion_policy,
            summary_metrics=metrics,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    context = env_context()
    existing_rows = ensure_results_ledger(task_type)
    current_master = current_master_snapshot(existing_rows, task_type=task_type)
    baseline_row = current_promoted_row(existing_rows, task_type=task_type)

    promotion_eval = evaluate_promotion(
        policy=promotion_policy,
        candidate_metrics=metrics,
        baseline_row=baseline_row,
    )
    promoted = promotion_eval.promoted
    promotion_metric = promotion_eval.primary

    candidate_value = parse_float(metrics.get(promotion_metric))
    status = "completed"

    master_value = promotion_eval.baseline_value

    job_id = args.job_id or ""
    row = {
        "run_id": build_run_id(existing_rows, job_id or None, c_hash),
        "created_at": now_utc_iso(),
        "status": status,
        "job_id": job_id,
        "task_type": task_type,
        "backend": config_data.get("backend", "transformers"),
        "campaign": context.get("campaign", ""),
        "experiment_id": context.get("experiment_id", ""),
        "worker_id": context.get("worker_id", ""),
        "hypothesis": context.get("hypothesis", args.comment),
        "model_name": (
            config_data.get("model_name") or config_data.get("model_name_or_path", "")
        ),
        "dataset_name": config_data.get("dataset_name", ""),
        "config_hash": c_hash,
        "parent_hash": current_master.get("hash", "") if current_master else "",
        "candidate_hash": c_hash,
        "promotion_metric": promotion_metric,
        "promotion_metric_value": metrics.get(promotion_metric, ""),
        "promotion_baseline_value": stringify_field(promotion_eval.baseline_value),
        "promotion_delta": stringify_field(promotion_eval.delta),
        "promotion_relative_delta": stringify_field(promotion_eval.relative_delta),
        "promotion_min_delta_met": stringify_field(promotion_eval.min_delta_met),
        "promotion_gates_met": stringify_field(promotion_eval.gates_met),
        "promotion_rerun_recommended": stringify_field(promotion_eval.rerun_recommended),
        "mAP": metrics.get("mAP", ""),
        "mAP_50": metrics.get("mAP_50", ""),
        "mask_map": metrics.get("mask_map", ""),
        "accuracy": metrics.get("accuracy", ""),
        "mIoU": metrics.get("mIoU", ""),
        "dice": metrics.get("dice", ""),
        "training_seconds": metrics.get("training_seconds", ""),
        "total_seconds": metrics.get("training_seconds", ""),
        "peak_vram_mb": metrics.get("peak_vram_mb", ""),
        "promoted": promoted,
        "comment": args.comment,
    }

    preview = {
        "row": {key: stringify_field(value) for key, value in row.items()},
        "metrics": metrics,
        "current_master": {
            "hash": current_master.get("hash", "") if current_master else "",
            "promotion_metric": promotion_metric,
            "promotion_metric_value": stringify_field(master_value),
        },
        "promotion": {
            "promoted": promoted,
            "reason": promotion_eval.reason,
            "evaluation": evaluation_to_jsonable(promotion_eval),
            "policy": policy_to_jsonable(promotion_policy),
        },
    }

    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    rebuild_live_state(existing_rows)
    appended = append_result_row(row)

    metrics_payload = {
        "run_id": appended["run_id"],
        "task_type": task_type,
        "metrics": {k: metrics[k] for k in sorted(metrics.keys())},
        "promotion_policy": policy_to_jsonable(promotion_policy),
        "promotion_evaluation": evaluation_to_jsonable(promotion_eval),
    }
    write_run_metrics_artifact(appended["run_id"], metrics_payload)
    if args.contract is not None:
        write_run_contract_artifact(appended["run_id"], args.contract)

    if truthy(appended["promoted"]):
        updated_rows = [*existing_rows, appended]
        rebuild_live_state(updated_rows)
        write_master_config(config_path, task_type)
        print(f"PROMOTED: {promotion_metric}={candidate_value} ({promotion_eval.reason})")
    else:
        print(
            f"NOT promoted: {promotion_metric}={candidate_value} ({promotion_eval.reason})"
        )

    print(json.dumps({**preview, "recorded": True}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
