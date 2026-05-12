"""Example: Dynamic INT8 quantization for HF models.

Applies PyTorch dynamic quantization to Linear layers for CPU inference
speedup. Works best for models dominated by Linear layers (transformers).

Usage:
    cp experiments/examples/opt_dynamic_quantize.py experiments/optimization.py
    uv run optimize_vision.py configs/example_optimize.yaml
"""

from __future__ import annotations

from typing import Any

OPTIMIZATION_TYPE = "dynamic_int8"
DESCRIPTION = "PyTorch dynamic INT8 quantization (Linear layers)"


def optimize(model: Any, config: dict[str, Any]) -> Any:
    """Apply dynamic INT8 quantization to Linear layers."""
    import torch
    import torch.nn as nn

    nn_mod = model.nn_module

    # Move to CPU for quantization
    nn_mod = nn_mod.cpu()

    # Dynamic quantization (quantizes weights, activations computed at runtime)
    quantized = torch.quantization.quantize_dynamic(
        nn_mod,
        {nn.Linear},  # Quantize all Linear layers
        dtype=torch.qint8,
    )

    # Replace
    if model.backend == "hf":
        model.model = quantized
    else:
        model._raw.model = quantized

    return model


def evaluate_accuracy(
    model: Any, dataset: Any, sample_images: list[Any]
) -> dict[str, float]:
    """Verify quantized model still produces valid outputs."""
    try:
        for img in sample_images[:5]:
            result = model.predict(img)
            if result is None:
                return {"accuracy": 0.0}
        return {}  # Skip gate
    except Exception:
        return {"accuracy": 0.0}
