"""Single source of truth for supported vision tasks and runner mapping."""

from __future__ import annotations

from dataclasses import dataclass


def _m(*names: str) -> frozenset[str]:
    return frozenset(names)


@dataclass(frozen=True)
class TaskSpec:
    """One benchmark task the lab can run or validate (CLI / jobs / submit)."""

    id: str
    backend: str
    """High-level trainer family: ``transformers`` or ``ultralytics``."""
    train_script: str
    """Python entry script filename at repo root (e.g. ``train_detect.py``)."""
    dataset_schema_kind: str
    """HF dataset column contract (see ``vision_lab.dataset_validation``)."""
    primary_metric: str
    """Standard summary key used as default promotion primary."""
    allowed_promotion_metrics: frozenset[str]
    """Metrics allowed in ``promotion`` (primary, secondary, gates, tie_breakers)."""
    trainable: bool = True
    evaluable: bool = True


_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        id="detect",
        backend="transformers",
        train_script="train_detect.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        allowed_promotion_metrics=_m("mAP", "mAP_50"),
    ),
    TaskSpec(
        id="classify",
        backend="transformers",
        train_script="train_classify.py",
        dataset_schema_kind="classification",
        primary_metric="accuracy",
        allowed_promotion_metrics=_m("accuracy"),
    ),
    TaskSpec(
        id="segment",
        backend="transformers",
        train_script="train_segment.py",
        dataset_schema_kind="segmentation",
        primary_metric="mIoU",
        allowed_promotion_metrics=_m("mIoU"),
    ),
    TaskSpec(
        id="detect_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        allowed_promotion_metrics=_m("mAP", "mAP_50"),
    ),
    TaskSpec(
        id="track_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        allowed_promotion_metrics=_m("mAP", "mAP_50"),
    ),
    TaskSpec(
        id="segment_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="segmentation",
        primary_metric="mask_map",
        allowed_promotion_metrics=_m("mask_map"),
    ),
    TaskSpec(
        id="classify_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="classification",
        primary_metric="accuracy",
        allowed_promotion_metrics=_m("accuracy"),
    ),
    TaskSpec(
        id="pose_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        allowed_promotion_metrics=_m("mAP", "mAP_50"),
    ),
    TaskSpec(
        id="obb_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        allowed_promotion_metrics=_m("mAP", "mAP_50"),
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
    """Default promotion primary metric name for ``task_id`` (standard key)."""
    if task_id not in TASK_BY_ID:
        raise KeyError(f"Unknown task: {task_id!r}")
    return TASK_BY_ID[task_id].primary_metric


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
