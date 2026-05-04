"""Single source of truth for supported vision tasks and runner mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """One benchmark task the lab can run or validate (CLI / jobs / submit)."""

    id: str
    backend: str
    """High-level trainer family: ``transformers`` or ``ultralytics``."""
    train_script: str
    """Python entry script filename at repo root (e.g. ``train_detect.py``)."""
    default_promotion_metric: str
    """Metric name as emitted in ``VISION AUTORESEARCH SUMMARY`` / submit row."""
    dataset_schema_kind: str
    """Which ``prepare.py`` HF-dataset column contract applies: detection, segmentation, or classification."""


_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        id="detect",
        backend="transformers",
        train_script="train_detect.py",
        default_promotion_metric="mAP",
        dataset_schema_kind="detection",
    ),
    TaskSpec(
        id="classify",
        backend="transformers",
        train_script="train_classify.py",
        default_promotion_metric="accuracy",
        dataset_schema_kind="classification",
    ),
    TaskSpec(
        id="segment",
        backend="transformers",
        train_script="train_segment.py",
        default_promotion_metric="iou",
        dataset_schema_kind="segmentation",
    ),
    TaskSpec(
        id="detect_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        default_promotion_metric="mAP",
        dataset_schema_kind="detection",
    ),
    TaskSpec(
        id="track_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        default_promotion_metric="mAP",
        dataset_schema_kind="detection",
    ),
    TaskSpec(
        id="segment_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        default_promotion_metric="iou",
        dataset_schema_kind="segmentation",
    ),
    TaskSpec(
        id="classify_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        default_promotion_metric="accuracy",
        dataset_schema_kind="classification",
    ),
    TaskSpec(
        id="pose_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        default_promotion_metric="mAP",
        dataset_schema_kind="detection",
    ),
    TaskSpec(
        id="obb_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        default_promotion_metric="mAP",
        dataset_schema_kind="detection",
    ),
)

TASK_BY_ID: dict[str, TaskSpec] = {t.id: t for t in _TASKS}

# HF Jobs / local cost heuristics (minutes by dataset size bucket).
ESTIMATED_MINUTES_BY_TASK: dict[str, dict[str, int]] = {
    "detect": {"small": 15, "medium": 45, "large": 120},
    "detect_yolo": {"small": 15, "medium": 45, "large": 120},
    "track_yolo": {"small": 15, "medium": 45, "large": 120},
    "pose_yolo": {"small": 20, "medium": 50, "large": 130},
    "obb_yolo": {"small": 20, "medium": 50, "large": 130},
    "segment_yolo": {"small": 20, "medium": 55, "large": 150},
    "classify_yolo": {"small": 10, "medium": 30, "large": 90},
    "classify": {"small": 10, "medium": 30, "large": 90},
    "segment": {"small": 20, "medium": 60, "large": 180},
}


def all_task_ids() -> tuple[str, ...]:
    """Stable ordering for argparse ``choices``."""
    return tuple(t.id for t in _TASKS)


def task_script_map() -> dict[str, str]:
    return {t.id: t.train_script for t in _TASKS}


def promotion_metric_for_task(task_id: str) -> str:
    if task_id not in TASK_BY_ID:
        raise KeyError(f"Unknown task: {task_id!r}")
    return TASK_BY_ID[task_id].default_promotion_metric


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASK_BY_ID:
        raise KeyError(f"Unknown task: {task_id!r}")
    return TASK_BY_ID[task_id]


def assert_estimates_complete() -> None:
    """Fail fast if a registered task lacks cost-estimate buckets."""
    missing = [tid for tid in TASK_BY_ID if tid not in ESTIMATED_MINUTES_BY_TASK]
    if missing:
        raise RuntimeError(
            "ESTIMATED_MINUTES_BY_TASK missing entries for: " + ", ".join(sorted(missing))
        )


assert_estimates_complete()
