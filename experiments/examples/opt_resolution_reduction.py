"""Example: Resolution reduction optimization.

Reduces input resolution for faster inference. Trades accuracy for speed.
Works on both HF and Ultralytics models.

Usage:
    cp experiments/examples/opt_resolution_reduction.py experiments/optimization.py
    uv run optimize_vision.py configs/example_optimize.yaml
"""

from __future__ import annotations

from typing import Any

OPTIMIZATION_TYPE = "resolution_reduction"
DESCRIPTION = "Reduce input resolution from 640 to 320 (4x fewer pixels)"

# Target resolution (adjust as needed)
TARGET_SIZE = 320


def optimize(model: Any, config: dict[str, Any]) -> Any:
    """Reduce the model's expected input resolution.

    For HF models: override the processor's size config.
    For Ultralytics: set the imgsz override.
    """
    if model.backend == "hf":
        # Override processor size
        processor = model.processor
        if hasattr(processor, "image_processor"):
            ip = processor.image_processor
        else:
            ip = processor

        # Set new size
        if hasattr(ip, "size"):
            if isinstance(ip.size, dict):
                ip.size = {"height": TARGET_SIZE, "width": TARGET_SIZE}
            else:
                ip.size = TARGET_SIZE

        if hasattr(ip, "crop_size"):
            if isinstance(ip.crop_size, dict):
                ip.crop_size = {"height": TARGET_SIZE, "width": TARGET_SIZE}
            else:
                ip.crop_size = TARGET_SIZE

    else:
        # Ultralytics: set overrides for predict
        model._raw.overrides["imgsz"] = TARGET_SIZE

    return model


def evaluate_accuracy(
    model: Any, dataset: Any, sample_images: list[Any]
) -> dict[str, float]:
    """Accuracy check at reduced resolution.

    Resolution reduction typically causes 5-15% accuracy drop.
    """
    try:
        for img in sample_images[:5]:
            model.predict(img)
        return {}  # Let the harness decide via max_accuracy_drop
    except Exception:
        return {"accuracy": 0.0}
