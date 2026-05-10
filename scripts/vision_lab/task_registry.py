"""Single source of truth for supported vision tasks and runner mapping."""

from __future__ import annotations

from dataclasses import dataclass

from vision_lab.metrics import (
    MetricDirection,
    assert_standard_metric_name,
    direction_for_standard_metric,
)


def _m(*names: str) -> frozenset[str]:
    return frozenset(names)


@dataclass(frozen=True)
class TaskSpec:
    """One benchmark task the lab can run or validate (CLI / jobs / submit)."""

    id: str
    backend: str
    """High-level trainer family: ``transformers`` or ``ultralytics``."""
    train_script: str
    """Python entry script filename at repo root (e.g. ``train_hf_vision.py``)."""
    dataset_schema_kind: str
    """HF dataset column contract (see ``vision_lab.dataset_validation``)."""
    primary_metric: str
    """Standard summary key used as default promotion primary."""
    metric_direction: MetricDirection
    """Default direction for ``primary_metric`` (must match ``STANDARD_METRICS``)."""
    allowed_primary_metrics: frozenset[str]
    """Metrics allowed as ``promotion.primary``."""
    allowed_secondary_metrics: frozenset[str]
    """Metrics allowed as ``promotion.secondary`` (empty means secondary must be omitted)."""
    allowed_gate_metrics: frozenset[str]
    """Metrics allowed in ``promotion.gates[].metric``."""
    allowed_tie_breaker_metrics: frozenset[str]
    """Metrics allowed in ``promotion.tie_breakers``."""
    allowed_auxiliary_summary_keys: frozenset[str] = frozenset()
    """Non-standard summary keys permitted for this task (e.g. ``dice`` on ``segment``)."""
    trainable: bool = True
    evaluable: bool = True

    def promotion_metrics_union(self) -> frozenset[str]:
        """All standard metric names that may appear anywhere in the promotion block."""
        return (
            self.allowed_primary_metrics
            | self.allowed_secondary_metrics
            | self.allowed_gate_metrics
            | self.allowed_tie_breaker_metrics
        )


def _task(
    *,
    id: str,
    backend: str,
    train_script: str,
    dataset_schema_kind: str,
    primary_metric: str,
    metric_direction: MetricDirection,
    promotion_metrics: frozenset[str],
    allowed_auxiliary_summary_keys: frozenset[str] = frozenset(),
    trainable: bool = True,
    evaluable: bool = True,
) -> TaskSpec:
    """Register a task where the same metric set is valid for every promotion role."""
    return TaskSpec(
        id=id,
        backend=backend,
        train_script=train_script,
        dataset_schema_kind=dataset_schema_kind,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        allowed_primary_metrics=promotion_metrics,
        allowed_secondary_metrics=promotion_metrics,
        allowed_gate_metrics=promotion_metrics,
        allowed_tie_breaker_metrics=promotion_metrics,
        allowed_auxiliary_summary_keys=allowed_auxiliary_summary_keys,
        trainable=trainable,
        evaluable=evaluable,
    )


_TASKS: tuple[TaskSpec, ...] = (
    _task(
        id="detect",
        backend="transformers",
        train_script="train_hf_vision.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mAP", "mAP_50"),
        allowed_auxiliary_summary_keys=_m("mAR"),
    ),
    _task(
        id="classify",
        backend="transformers",
        train_script="train_hf_vision.py",
        dataset_schema_kind="classification",
        primary_metric="accuracy",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("accuracy"),
    ),
    _task(
        id="segment",
        backend="transformers",
        train_script="train_hf_vision.py",
        dataset_schema_kind="semantic_segmentation",
        primary_metric="mIoU",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mIoU"),
        allowed_auxiliary_summary_keys=_m("dice"),
    ),
    _task(
        id="semantic_segment",
        backend="transformers",
        train_script="train_hf_vision.py",
        dataset_schema_kind="semantic_segmentation",
        primary_metric="mIoU",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mIoU"),
    ),
    _task(
        id="instance_segment",
        backend="transformers",
        train_script="train_hf_vision.py",
        dataset_schema_kind="instance_segmentation",
        primary_metric="mask_map",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mask_map", "mAP", "mAP_50"),
    ),
    _task(
        id="universal_segment",
        backend="transformers",
        train_script="train_hf_vision.py",
        dataset_schema_kind="panoptic_segmentation",
        primary_metric="pq",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("pq", "sq", "rq"),
    ),
    _task(
        id="detect_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mAP", "mAP_50"),
        allowed_auxiliary_summary_keys=_m("mAR"),
    ),
    _task(
        id="track_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mAP", "mAP_50"),
        allowed_auxiliary_summary_keys=_m("mAR"),
    ),
    _task(
        id="segment_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="instance_segmentation",
        primary_metric="mask_map",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mask_map"),
    ),
    _task(
        id="classify_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="classification",
        primary_metric="accuracy",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("accuracy"),
    ),
    _task(
        id="pose_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mAP", "mAP_50"),
        allowed_auxiliary_summary_keys=_m("mAR"),
    ),
    _task(
        id="obb_yolo",
        backend="ultralytics",
        train_script="train_ultralytics.py",
        dataset_schema_kind="detection",
        primary_metric="mAP",
        metric_direction=MetricDirection.HIGHER,
        promotion_metrics=_m("mAP", "mAP_50"),
        allowed_auxiliary_summary_keys=_m("mAR"),
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
    "semantic_segment": {"small": 20, "medium": 60, "large": 180},
    "instance_segment": {"small": 25, "medium": 75, "large": 210},
    "universal_segment": {"small": 30, "medium": 90, "large": 240},
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


def _assert_task_metric_contracts() -> None:
    """Every registered task must reference only standard metrics with consistent defaults."""
    for t in _TASKS:
        assert_standard_metric_name(t.primary_metric)
        if t.primary_metric not in t.allowed_primary_metrics:
            raise RuntimeError(
                f"Task {t.id!r}: primary_metric {t.primary_metric!r} must be in allowed_primary_metrics."
            )
        if direction_for_standard_metric(t.primary_metric) != t.metric_direction:
            raise RuntimeError(
                f"Task {t.id!r}: metric_direction {t.metric_direction!r} disagrees with "
                f"STANDARD_METRICS for primary_metric {t.primary_metric!r}."
            )
        for m in t.promotion_metrics_union():
            assert_standard_metric_name(m)
        for m in t.allowed_auxiliary_summary_keys:
            if m in t.promotion_metrics_union():
                raise RuntimeError(
                    f"Task {t.id!r}: auxiliary summary key {m!r} duplicates a promotion metric."
                )


_assert_task_metric_contracts()
assert_estimates_complete()
