"""Single implementation of the ``VISION AUTORESEARCH SUMMARY`` log block for HF vision tasks."""

from __future__ import annotations

from typing import Any


def print_vision_autoresearch_summary(
    task_type: str,
    eval_metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    training_seconds: float,
    peak_vram_mb: float,
) -> None:
    """Emit the summary region parsed by ``scripts/parse_metric.py`` (keys must stay stable)."""
    print("\n--- VISION AUTORESEARCH SUMMARY ---")
    print(f"task_type: {task_type}")
    if task_type == "classify":
        acc = eval_metrics.get("eval_accuracy", eval_metrics.get("test_accuracy", 0.0))
        print(f"accuracy: {acc}")
    elif task_type == "detect":
        print(f"mAP: {eval_metrics.get('map', eval_metrics.get('eval_map', 0.0))}")
        print(f"mAP_50: {eval_metrics.get('map_50', eval_metrics.get('eval_map_50', 0.0))}")
    elif task_type == "segment":
        print(f"mIoU: {eval_metrics.get('eval_mIoU', eval_metrics.get('mIoU', 0.0))}")
        print(f"dice: {eval_metrics.get('eval_dice', eval_metrics.get('dice', 0.0))}")
    else:
        raise ValueError(f"Unknown task_type for summary emission: {task_type!r}")

    print(f"training_seconds: {training_seconds:.1f}")
    print(f"peak_vram_mb: {peak_vram_mb:.0f}")
    print(f"train_loss: {train_metrics.get('train_loss', 0.0)}")
    print(f"num_train_epochs: {train_metrics.get('epoch', 0)}")
    print("--- END SUMMARY ---")
