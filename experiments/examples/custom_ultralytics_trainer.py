"""Example: Custom Ultralytics trainer with gradient clipping and per-layer LRs.

Demonstrates the custom_trainer_class() hook for Ultralytics models.
This creates a trainer subclass that overrides optimizer construction
and gradient clipping — useful for DETR-family models via Ultralytics
(RT-DETR) or when you want finer control over YOLO training.

Usage:
    cp experiments/examples/custom_ultralytics_trainer.py experiments/modification.py
    uv run train_vision.py configs/example_unified_yolo_detect.yaml
"""

from __future__ import annotations

from typing import Any


def custom_trainer_class() -> type:
    """Return a custom DetectionTrainer with gradient clipping."""
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer

    class GradClipTrainer(DetectionTrainer):
        """Trainer with aggressive gradient clipping (DETR-style)."""

        def optimizer_step(self) -> None:
            """Clip gradients to max_norm=0.1 before stepping."""
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=0.1
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

    return GradClipTrainer
