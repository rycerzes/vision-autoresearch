"""Metric derivation from head category and auto-extending metric registry.

Replaces the old ``vision_lab.metrics`` with a unified registry that works
for both backends and auto-accepts unknown metrics from agent code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MetricDirection(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True)
class MetricSpec:
    direction: MetricDirection
    display_name: str | None = None


# Known metrics (direction is authoritative)

METRICS: dict[str, MetricSpec] = {
    # Detection
    "mAP": MetricSpec(MetricDirection.HIGHER, "mAP"),
    "mAP_50": MetricSpec(MetricDirection.HIGHER, "mAP@0.5"),
    "mAR": MetricSpec(MetricDirection.HIGHER, "mAR"),
    "mask_map": MetricSpec(MetricDirection.HIGHER, "mask mAP"),
    # Classification
    "accuracy": MetricSpec(MetricDirection.HIGHER),
    # Segmentation
    "iou": MetricSpec(MetricDirection.HIGHER, "IoU"),
    "miou": MetricSpec(MetricDirection.HIGHER, "mIoU"),
    "dice": MetricSpec(MetricDirection.HIGHER, "Dice"),
    "pq": MetricSpec(MetricDirection.HIGHER, "PQ"),
    "sq": MetricSpec(MetricDirection.HIGHER, "SQ"),
    "rq": MetricSpec(MetricDirection.HIGHER, "RQ"),
    # Depth
    "abs_rel": MetricSpec(MetricDirection.LOWER, "AbsRel"),
    "rmse": MetricSpec(MetricDirection.LOWER, "RMSE"),
    "delta1": MetricSpec(MetricDirection.HIGHER, "δ<1.25"),
    "delta2": MetricSpec(MetricDirection.HIGHER, "δ<1.25²"),
    "silog": MetricSpec(MetricDirection.LOWER, "SILog"),
    # Image quality
    "psnr": MetricSpec(MetricDirection.HIGHER, "PSNR"),
    "ssim": MetricSpec(MetricDirection.HIGHER, "SSIM"),
    "lpips": MetricSpec(MetricDirection.LOWER, "LPIPS"),
    "fid": MetricSpec(MetricDirection.LOWER, "FID"),
    # OCR / sequence
    "cer": MetricSpec(MetricDirection.LOWER, "CER"),
    "wer": MetricSpec(MetricDirection.LOWER, "WER"),
    "normalized_edit_distance": MetricSpec(MetricDirection.LOWER, "normalized edit distance"),
    "teds": MetricSpec(MetricDirection.HIGHER, "TEDS"),
    # Keypoint
    "oks_map": MetricSpec(MetricDirection.HIGHER, "OKS mAP"),
    "pck": MetricSpec(MetricDirection.HIGHER, "PCK"),
    # Matching
    "match_precision": MetricSpec(MetricDirection.HIGHER, "Match Precision"),
    "match_recall": MetricSpec(MetricDirection.HIGHER, "Match Recall"),
    "auc": MetricSpec(MetricDirection.HIGHER, "AUC"),
    # Training
    "training_seconds": MetricSpec(MetricDirection.LOWER),
    "peak_vram_mb": MetricSpec(MetricDirection.LOWER),
    "train_loss": MetricSpec(MetricDirection.LOWER),
    "num_train_epochs": MetricSpec(MetricDirection.HIGHER),
    "inference_ms": MetricSpec(MetricDirection.LOWER),
    "throughput_img_per_sec": MetricSpec(MetricDirection.HIGHER),
}


# Head category → default metrics

HEAD_METRICS: dict[str, list[str]] = {
    "classification": ["accuracy"],
    "detection": ["mAP", "mAP_50"],
    "dense_classification": ["miou"],
    "dense_regression": ["abs_rel", "rmse", "delta1"],
    "sequence_generation": ["cer"],
    "contrastive": ["accuracy"],
    "image_reconstruction": ["psnr", "ssim"],
    "self_supervised": ["train_loss"],
    "structured_detection": ["oks_map"],
    "pair_matching": ["match_precision", "auc"],
    "prompted_segmentation": ["iou", "dice"],
}


def default_promotion_metric(head_category: str) -> tuple[str, MetricDirection]:
    """Return ``(metric_name, direction)`` for a head category."""
    metrics = HEAD_METRICS.get(head_category)
    if not metrics:
        return "train_loss", MetricDirection.LOWER
    primary = metrics[0]
    spec = METRICS.get(primary)
    direction = spec.direction if spec else MetricDirection.HIGHER
    return primary, direction


def get_direction(metric_name: str) -> MetricDirection:
    """Direction for a metric. Unknown metrics default to HIGHER with a warning."""
    spec = METRICS.get(metric_name)
    if spec is not None:
        return spec.direction

    # Heuristic: names containing loss/error/rel → lower is better
    lower_hints = ("loss", "error", "err", "rel", "rmse", "mae", "cer", "wer", "fid", "lpips")
    name_lower = metric_name.lower()
    if any(h in name_lower for h in lower_hints):
        logger.info("Unknown metric %r — inferred direction=lower from name", metric_name)
        return MetricDirection.LOWER

    logger.info("Unknown metric %r — defaulting to direction=higher", metric_name)
    return MetricDirection.HIGHER


def register_metric(name: str, direction: MetricDirection, display_name: str | None = None) -> None:
    """Dynamically register a metric (e.g., from agent code)."""
    if name in METRICS:
        return  # already known
    METRICS[name] = MetricSpec(direction, display_name)
    logger.info("Registered new metric: %s (direction=%s)", name, direction.value)
