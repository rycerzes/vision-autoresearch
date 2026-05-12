"""Example: Ultralytics TensorRT export optimization.

Exports the model to TensorRT engine format for maximum GPU inference speed.
Only works for Ultralytics backend models (YOLO, RT-DETR, etc.).

Requires: tensorrt, nvidia-tensorrt packages installed.

Usage:
    cp experiments/examples/opt_tensorrt_export.py experiments/optimization.py
    uv run optimize_vision.py configs/example_optimize.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

OPTIMIZATION_TYPE = "tensorrt_export"
DESCRIPTION = "Ultralytics TensorRT engine export (FP16)"
EXPORT_PATH = None  # Set after export


def optimize(model: Any, config: dict[str, Any]) -> Any:
    """Export Ultralytics model to TensorRT engine."""
    global EXPORT_PATH

    if model.backend != "ultralytics":
        raise RuntimeError(
            "TensorRT export via Ultralytics only works for Ultralytics models. "
            "For HF models, use ONNX export + trtexec instead."
        )

    output_dir = Path(config.get("output_dir", "./output/optimization"))
    imgsz = config.get("image_size", 640)

    # Export to TensorRT (FP16)
    export_path = model._raw.export(
        format="engine",
        imgsz=imgsz,
        half=True,  # FP16 for best speed
        device=0,
    )

    EXPORT_PATH = str(export_path)

    # Reload the exported engine model
    from ultralytics import YOLO
    engine_model = YOLO(export_path)

    # Wrap in a minimal adapter that provides benchmark_latency
    model._raw = engine_model
    return model
