"""Example: torch.compile() optimization.

Applies torch.compile with reduce-overhead mode for best latency.
Works on both HF and Ultralytics models (any nn.Module).

Usage:
    cp experiments/examples/opt_torch_compile.py experiments/optimization.py
    uv run optimize_vision.py configs/example_optimize.yaml
"""

from __future__ import annotations

from typing import Any

OPTIMIZATION_TYPE = "torch_compile"
DESCRIPTION = "torch.compile with reduce-overhead mode"


def optimize(model: Any, config: dict[str, Any]) -> Any:
    """Apply torch.compile to the model's nn.Module."""
    import torch

    nn_mod = model.nn_module

    # Compile with reduce-overhead for best inference latency
    # Other options: "default", "max-autotune", "max-autotune-no-cudagraphs"
    compiled = torch.compile(nn_mod, mode="reduce-overhead", fullgraph=False)

    # Replace the internal module
    if model.backend == "hf":
        model.model = compiled
    else:
        model._raw.model = compiled

    return model
