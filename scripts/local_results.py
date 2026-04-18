#!/usr/bin/env python3
"""Local results ledger and master snapshot management for vision-autoresearch."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "research"
LIVE_DIR = RESEARCH_DIR / "live"
REFERENCE_DIR = RESEARCH_DIR / "reference"
RESULTS_PATH = RESEARCH_DIR / "results.tsv"
CONFIGS_DIR = ROOT / "configs"
MASTER_PATH = LIVE_DIR / "master.json"
MASTER_DETAIL_PATH = LIVE_DIR / "master_detail.json"
DAG_PATH = LIVE_DIR / "dag.json"
MASTER_SEED_PATH = REFERENCE_DIR / "master.seed.json"
MASTER_DETAIL_SEED_PATH = REFERENCE_DIR / "master_detail.seed.json"

RESULTS_COLUMNS = [
    "run_id",
    "created_at",
    "status",
    "job_id",
    "task_type",
    "backend",
    "campaign",
    "experiment_id",
    "worker_id",
    "hypothesis",
    "model_name",
    "dataset_name",
    "config_hash",
    "parent_hash",
    "candidate_hash",
    "promotion_metric",
    "promotion_metric_value",
    "mAP",
    "mAP_50",
    "accuracy",
    "iou",
    "dice",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "promoted",
    "comment",
]

# All vision metrics are higher-is-better
HIGHER_IS_BETTER_METRICS = {"mAP", "mAP_50", "accuracy", "iou", "dice"}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def config_hash(config_path: Path) -> str:
    """Hash a YAML config file for change tracking."""
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stringify_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def normalize_row(row: dict[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for column in RESULTS_COLUMNS:
        normalized[column] = stringify_field(row.get(column, ""))
    return normalized


def load_results_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != RESULTS_COLUMNS:
            raise RuntimeError(
                f"{RESULTS_PATH.relative_to(ROOT)} has unexpected columns; "
                f"expected {RESULTS_COLUMNS}, got {reader.fieldnames}"
            )
        return [normalize_row(row) for row in reader]


def write_results_rows(rows: list[dict[str, object]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULTS_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row))


def append_result_row(row: dict[str, object]) -> dict[str, str]:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0
    normalized = normalize_row(row)
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULTS_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized)
    return normalized


def promoted_rows(rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    resolved_rows = rows if rows is not None else load_results_rows()
    return [row for row in resolved_rows if truthy(row.get("promoted"))]


def current_promoted_row(rows: list[dict[str, str]] | None = None) -> dict[str, str] | None:
    promoted = promoted_rows(rows)
    if not promoted:
        return None
    return promoted[-1]


def current_master_hash(rows: list[dict[str, str]] | None = None) -> str | None:
    row = current_promoted_row(rows)
    return row["candidate_hash"] if row else None


def is_improvement(new_value: float, old_value: float | None, metric: str) -> bool:
    """Check if new_value beats old_value. All vision metrics are higher-is-better."""
    if old_value is None:
        return True
    if metric not in HIGHER_IS_BETTER_METRICS:
        raise ValueError(f"Unknown metric: {metric}. Expected one of {HIGHER_IS_BETTER_METRICS}")
    return new_value > old_value


def build_master_snapshot(row: dict[str, str]) -> dict[str, Any]:
    return {
        "task_type": row.get("task_type", ""),
        "hash": row.get("candidate_hash", ""),
        "parent_hash": row.get("parent_hash", ""),
        "model_name": row.get("model_name", ""),
        "dataset_name": row.get("dataset_name", ""),
        "promotion_metric": row.get("promotion_metric", ""),
        "promotion_metric_value": parse_float(row.get("promotion_metric_value")),
        "mAP": parse_float(row.get("mAP")),
        "mAP_50": parse_float(row.get("mAP_50")),
        "accuracy": parse_float(row.get("accuracy")),
        "iou": parse_float(row.get("iou")),
        "dice": parse_float(row.get("dice")),
        "created_at": row.get("created_at", ""),
        "job_id": row.get("job_id", ""),
        "campaign": row.get("campaign", ""),
        "experiment_id": row.get("experiment_id", ""),
        "worker_id": row.get("worker_id", ""),
        "hypothesis": row.get("hypothesis", ""),
        "status": row.get("status", ""),
        "comment": row.get("comment", ""),
        "promoted": truthy(row.get("promoted")),
    }


def current_master_snapshot(rows: list[dict[str, str]] | None = None) -> dict[str, Any] | None:
    row = current_promoted_row(rows)
    if row is None:
        return None
    return build_master_snapshot(row)
