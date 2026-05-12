"""Inference optimization module — agent writes this per optimization iteration.

The optimize() function receives a loaded model and returns an optimized version.
The harness measures latency before and after, and applies an accuracy gate.

All constants (OPTIMIZATION_TYPE, DESCRIPTION) are optional metadata.

Works on BOTH HF Transformers and Ultralytics models via the UnifiedModel API.
"""

from __future__ import annotations

from typing import Any


OPTIMIZATION_TYPE = "none"  # e.g., "torch_compile", "quantization", "export_tensorrt"
DESCRIPTION = "No-op baseline — replace with actual optimization"


def optimize(model: Any, config: dict[str, Any]) -> Any:
    """Apply optimization to the model.

    Parameters
    ----------
    model:
        A ``UnifiedModel`` (``HFModel`` or ``UltralyticsModel``).
        Access the raw PyTorch module via ``model.nn_module``.
        Access Ultralytics raw model via ``model._raw`` (for export).
    config:
        Dict with:
        - ``head_category``: str
        - ``image_size``: int
        - ``output_dir``: str (for saving exported models)
        - ``sample_images``: list[PIL.Image] (for calibration)

    Returns
    -------
    The optimized model (same UnifiedModel interface), or None to use
    the model as-is.

    Examples
    --------
    # torch.compile (HF or Ultralytics)
    import torch
    model.model = torch.compile(model.nn_module, mode="reduce-overhead")
    return model

    # Ultralytics TensorRT export
    model._raw.export(format="engine", imgsz=config["image_size"])
    return model

    # Half precision
    model.nn_module.half()
    return model
    """
    return model  # no-op baseline


def evaluate_accuracy(
    model: Any, dataset: Any, sample_images: list[Any]
) -> dict[str, float]:
    """Evaluate accuracy of the optimized model.

    Parameters
    ----------
    model:
        The optimized UnifiedModel.
    dataset:
        The UnifiedDataset.
    sample_images:
        Subset of images for quick accuracy estimation.

    Returns
    -------
    Dict with metric values (e.g., {"mAP": 0.42, "accuracy": 0.95}).
    Return empty dict to skip accuracy gating.

    Notes
    -----
    For full accuracy evaluation, use model.evaluate(dataset).
    For quick estimation on a subset, implement custom logic here.
    """
    return {}  # Skip accuracy gate (assume no degradation)
