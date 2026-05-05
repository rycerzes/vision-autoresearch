#!/usr/bin/env python3
"""Local results ledger and master snapshot management for vision-autoresearch."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from vision_lab.task_registry import promotion_metric_for_task

ROOT = Path(__file__).resolve().parents[1]

RESEARCH_DIR = ROOT / "research"
RUNS_DIR = RESEARCH_DIR / "runs"
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
    "promotion_baseline_value",
    "promotion_delta",
    "promotion_relative_delta",
    "promotion_min_delta_met",
    "promotion_gates_met",
    "promotion_rerun_recommended",
    "mAP",
    "mAP_50",
    "mask_map",
    "accuracy",
    "mIoU",
    "dice",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "promoted",
    "comment",
]


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
    return config_hash_from_text(text)


def config_hash_from_text(text: str) -> str:
    """Hash config text for change tracking."""
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
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames == tuple(RESULTS_COLUMNS):
            return [normalize_row(cast(dict[str, object], row)) for row in reader]
        raise RuntimeError(
            f"{RESULTS_PATH.relative_to(ROOT)} has unexpected columns; "
            f"expected {RESULTS_COLUMNS}, got {reader.fieldnames}"
        )


def write_run_metrics_artifact(run_id: str, payload: dict[str, Any]) -> Path:
    """Persist structured metrics + promotion evaluation under ``research/runs/<run_id>/``."""
    out_path = RUNS_DIR / run_id / "metrics.json"
    write_json(out_path, payload)
    return out_path


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


def reference_master_metadata() -> dict[str, Any]:
    """Load the best available master metadata (live > seed)."""
    for path in (MASTER_PATH, MASTER_SEED_PATH):
        data = load_json(path)
        if isinstance(data, dict):
            return data
    return {}


def seed_config_content(task_type: str) -> str:
    """Return the base config content for seeding. Checks the config file first,
    then falls back to master_detail."""
    config_path = CONFIGS_DIR / f"base_{task_type}.yaml"
    if config_path.exists():
        return config_path.read_text(encoding="utf-8")
    detail = load_json(MASTER_DETAIL_PATH) or load_json(MASTER_DETAIL_SEED_PATH)
    if isinstance(detail, dict):
        content = detail.get("config_content")
        if isinstance(content, str) and content:
            return content
    raise RuntimeError(
        f"Could not determine seed config; configs/base_{task_type}.yaml "
        "and master_detail are both missing"
    )


def seed_row(task_type: str = "detect") -> dict[str, object]:
    """Build a synthetic seed row from the base config."""
    metadata = reference_master_metadata()
    seed_task = metadata.get("task_type") or task_type
    content = seed_config_content(seed_task)
    c_hash = config_hash_from_text(content)
    promo_raw = metadata.get("promotion_metric", "")
    promo = promo_raw.strip() if isinstance(promo_raw, str) else ""
    if not promo:
        try:
            promo = promotion_metric_for_task(str(seed_task))
        except KeyError:
            promo = ""
    return {
        "run_id": "seed",
        "created_at": metadata.get("created_at") or now_utc_iso(),
        "status": "seed",
        "job_id": metadata.get("job_id", ""),
        "task_type": seed_task,
        "backend": metadata.get("backend", "transformers"),
        "campaign": "baseline-seed",
        "experiment_id": "seed",
        "worker_id": "seed",
        "hypothesis": f"seed local master from base_{seed_task}.yaml",
        "model_name": metadata.get("model_name", ""),
        "dataset_name": metadata.get("dataset_name", ""),
        "config_hash": c_hash,
        "parent_hash": metadata.get("parent_hash", ""),
        "candidate_hash": c_hash,
        "promotion_metric": promo,
        "promotion_metric_value": metadata.get("promotion_metric_value", ""),
        "mAP": metadata.get("mAP", ""),
        "mAP_50": metadata.get("mAP_50", ""),
        "mask_map": metadata.get("mask_map", ""),
        "accuracy": metadata.get("accuracy", ""),
        "mIoU": metadata.get("mIoU", ""),
        "dice": metadata.get("dice", ""),
        "training_seconds": "",
        "total_seconds": "",
        "peak_vram_mb": "",
        "promoted": True,
        "comment": metadata.get("comment") or f"Seeded local master from base_{seed_task}.yaml",
    }


def ensure_results_ledger(task_type: str = "detect") -> list[dict[str, str]]:
    """Load results.tsv, auto-seeding if empty."""
    rows = load_results_rows()
    if rows:
        return rows
    seeded = seed_row(task_type)
    write_results_rows([seeded])
    return [normalize_row(seeded)]


def current_promoted_row(
    rows: list[dict[str, str]] | None = None,
    task_type: str | None = None,
) -> dict[str, str] | None:
    promoted = promoted_rows(rows)
    if task_type:
        promoted = [row for row in promoted if row.get("task_type") == task_type]
    if not promoted:
        return None
    return promoted[-1]


def current_master_hash(rows: list[dict[str, str]] | None = None) -> str | None:
    row = current_promoted_row(rows)
    return row["candidate_hash"] if row else None


def build_master_snapshot(row: dict[str, str]) -> dict[str, Any]:
    """Snapshot keys align with ``RESULTS_COLUMNS`` headline metrics + promotion fields."""
    return {
        "task_type": row.get("task_type", ""),
        "hash": row.get("candidate_hash", ""),
        "parent_hash": row.get("parent_hash", ""),
        "model_name": row.get("model_name", ""),
        "dataset_name": row.get("dataset_name", ""),
        "promotion_metric": row.get("promotion_metric", ""),
        "promotion_metric_value": parse_float(row.get("promotion_metric_value")),
        "promotion_baseline_value": parse_float(row.get("promotion_baseline_value")),
        "promotion_delta": parse_float(row.get("promotion_delta")),
        "promotion_relative_delta": parse_float(row.get("promotion_relative_delta")),
        "promotion_min_delta_met": truthy(row.get("promotion_min_delta_met"))
        if row.get("promotion_min_delta_met") not in (None, "")
        else None,
        "promotion_gates_met": truthy(row.get("promotion_gates_met"))
        if row.get("promotion_gates_met") not in (None, "")
        else None,
        "promotion_rerun_recommended": truthy(row.get("promotion_rerun_recommended"))
        if row.get("promotion_rerun_recommended") not in (None, "")
        else None,
        "mAP": parse_float(row.get("mAP")),
        "mAP_50": parse_float(row.get("mAP_50")),
        "mask_map": parse_float(row.get("mask_map")),
        "accuracy": parse_float(row.get("accuracy")),
        "mIoU": parse_float(row.get("mIoU")),
        "dice": parse_float(row.get("dice")),
        "training_seconds": parse_float(row.get("training_seconds")),
        "total_seconds": parse_float(row.get("total_seconds")),
        "peak_vram_mb": parse_float(row.get("peak_vram_mb")),
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


def current_master_snapshot(
    rows: list[dict[str, str]] | None = None,
    task_type: str | None = None,
) -> dict[str, Any] | None:
    row = current_promoted_row(rows, task_type=task_type)
    if row is None:
        return None
    return build_master_snapshot(row)


def build_dag(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Build a DAG of promoted runs showing the lineage of experiments."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for row in promoted_rows(rows):
        node_id = row.get("candidate_hash", "")
        parent_id = row.get("parent_hash", "")
        nodes.append({
            "id": node_id,
            "run_id": row.get("run_id", ""),
            "task_type": row.get("task_type", ""),
            "promotion_metric": row.get("promotion_metric", ""),
            "promotion_metric_value": parse_float(row.get("promotion_metric_value")),
            "hypothesis": row.get("hypothesis", ""),
            "created_at": row.get("created_at", ""),
        })
        if parent_id:
            edges.append({"from": parent_id, "to": node_id})
    return {"nodes": nodes, "edges": edges}


def rebuild_live_state(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    """Rebuild master.json and dag.json from the results ledger."""
    row = current_promoted_row(rows)
    if row is None:
        return None
    snapshot = build_master_snapshot(row)
    write_json(MASTER_PATH, snapshot)
    dag = build_dag(rows)
    write_json(DAG_PATH, dag)
    return snapshot
