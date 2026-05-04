"""Canonical metric names and directions for promotion and log parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricDirection(str, Enum):
    """Whether a larger numeric value is better for promotion."""

    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True)
class MetricSpec:
    """One metric that may appear in run summaries or promotion logic."""

    direction: MetricDirection
    display_name: str | None = None
    """Human-readable label in dashboards; defaults to the metric key."""


# Metrics emitted in training logs / summaries (Ultralytics bridge included).
METRICS: dict[str, MetricSpec] = {
    "mAP": MetricSpec(MetricDirection.HIGHER, display_name="mAP"),
    "mAP_50": MetricSpec(MetricDirection.HIGHER, display_name="mAP@0.5"),
    "mAR": MetricSpec(MetricDirection.HIGHER, display_name="mAR"),
    "accuracy": MetricSpec(MetricDirection.HIGHER),
    "iou": MetricSpec(MetricDirection.HIGHER, display_name="IoU"),
    "dice": MetricSpec(MetricDirection.HIGHER, display_name="Dice"),
    "training_seconds": MetricSpec(MetricDirection.LOWER),
    "peak_vram_mb": MetricSpec(MetricDirection.LOWER),
    "train_loss": MetricSpec(MetricDirection.LOWER),
    "num_train_epochs": MetricSpec(MetricDirection.HIGHER),
}


def higher_is_better_metric_names() -> frozenset[str]:
    """Metric keys where a larger value is better (used by promotion today)."""
    return frozenset(
        name for name, spec in METRICS.items() if spec.direction == MetricDirection.HIGHER
    )


def lower_is_better_metric_names() -> frozenset[str]:
    return frozenset(
        name for name, spec in METRICS.items() if spec.direction == MetricDirection.LOWER
    )


# Metrics that may be used as the scalar promotion target with higher-is-better
# comparison (see `local_results.is_improvement`). Subset of METRICS with
# direction HIGHER; excludes auxiliary fields like `num_train_epochs`.
PROMOTION_METRICS_HIGHER: frozenset[str] = frozenset(
    {"mAP", "mAP_50", "accuracy", "iou", "dice"}
)
