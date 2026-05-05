"""Task-aware standard metric names and directions for promotion and log parsing.

The global catalog ``STANDARD_METRICS`` defines canonical names and higher/lower
semantics. Which of those names may headline promotion or appear in summaries for
a given benchmark is declared per task in ``vision_lab.task_registry.TaskSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricDirection(str, Enum):
    """Whether a larger numeric value is better for promotion."""

    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True)
class MetricSpec:
    """One standard benchmark metric that may headline a task."""

    direction: MetricDirection
    display_name: str | None = None
    """Human-readable label in dashboards; defaults to the metric key."""


# Standard headline metrics only (forward-only contract). No aliases or legacy keys.
STANDARD_METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec(MetricDirection.HIGHER),
    "mAP": MetricSpec(MetricDirection.HIGHER, display_name="mAP"),
    "mAP_50": MetricSpec(MetricDirection.HIGHER, display_name="mAP@0.5"),
    "mask_map": MetricSpec(MetricDirection.HIGHER, display_name="mask mAP"),
    "mIoU": MetricSpec(MetricDirection.HIGHER, display_name="mIoU"),
    "pq": MetricSpec(MetricDirection.HIGHER, display_name="PQ"),
    "abs_rel": MetricSpec(MetricDirection.LOWER, display_name="AbsRel"),
    "rmse": MetricSpec(MetricDirection.LOWER, display_name="RMSE"),
    "silog": MetricSpec(MetricDirection.LOWER, display_name="SILog"),
    "delta1": MetricSpec(MetricDirection.HIGHER, display_name="δ<1.25"),
    "cer": MetricSpec(MetricDirection.LOWER, display_name="CER"),
    "wer": MetricSpec(MetricDirection.LOWER, display_name="WER"),
    "psnr": MetricSpec(MetricDirection.HIGHER, display_name="PSNR"),
    "ssim": MetricSpec(MetricDirection.HIGHER, display_name="SSIM"),
    "reconstruction_loss": MetricSpec(MetricDirection.LOWER),
}

STANDARD_METRIC_NAMES: frozenset[str] = frozenset(STANDARD_METRICS.keys())

# Back-compat alias for trainers and tooling that import ``METRICS``.
METRICS = STANDARD_METRICS


def assert_standard_metric_name(metric: str) -> None:
    """Raise if ``metric`` is not a registered standard headline name."""
    if metric not in STANDARD_METRICS:
        raise ValueError(
            f"Metric {metric!r} is not a standard metric name "
            f"(allowed: {', '.join(sorted(STANDARD_METRICS))})."
        )


def higher_is_better_metric_names() -> frozenset[str]:
    """Metric keys where a larger value is better."""
    return frozenset(
        name
        for name, spec in STANDARD_METRICS.items()
        if spec.direction == MetricDirection.HIGHER
    )


def lower_is_better_metric_names() -> frozenset[str]:
    return frozenset(
        name
        for name, spec in STANDARD_METRICS.items()
        if spec.direction == MetricDirection.LOWER
    )


def direction_for_standard_metric(metric: str) -> MetricDirection:
    """Return promotion direction for a standard metric key."""
    assert_standard_metric_name(metric)
    return STANDARD_METRICS[metric].direction
