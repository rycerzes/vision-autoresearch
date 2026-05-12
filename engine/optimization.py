"""Built-in inference optimization strategies.

Each strategy is a self-contained function that:
1. Checks if its dependencies are available (try-import)
2. Applies the optimization to the model
3. Returns the optimized model or None if unavailable

Strategies are selected via YAML config, not agent-written code.
The engine probes availability at runtime — never crashes on missing deps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Result of applying one optimization strategy."""

    name: str
    available: bool = True
    applied: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    throughput: float = 0.0
    vram_mb: float = 0.0
    export_path: str | None = None


def probe_available_strategies() -> dict[str, bool]:
    """Detect which optimization backends are installed.

    Returns a dict of strategy_name → is_available.
    No crashes — just try-import each backend.
    """
    available: dict[str, bool] = {}

    # Always available (part of torch)
    available["torch_compile"] = True
    available["half_precision"] = True
    available["dynamic_int8"] = True

    # torch.compile with inductor (always available with torch >= 2.0)
    try:
        import torch
        available["torch_compile"] = hasattr(torch, "compile")
    except ImportError:
        available["torch_compile"] = False

    # TensorRT via torch_tensorrt
    try:
        import torch_tensorrt  # noqa: F401
        available["torch_tensorrt"] = True
    except ImportError:
        available["torch_tensorrt"] = False

    # ONNX Runtime
    try:
        import onnxruntime  # noqa: F401
        available["onnx"] = True
    except ImportError:
        available["onnx"] = False

    # OpenVINO
    try:
        import openvino  # noqa: F401
        available["openvino"] = True
    except ImportError:
        available["openvino"] = False

    # BitsAndBytes
    try:
        import bitsandbytes  # noqa: F401
        available["bitsandbytes"] = True
    except ImportError:
        available["bitsandbytes"] = False

    # Ultralytics TensorRT export (via ultralytics, which handles deps internally)
    try:
        from ultralytics import YOLO  # noqa: F401
        available["ultralytics_export"] = True
    except ImportError:
        available["ultralytics_export"] = False

    return available


def apply_strategy(
    model: Any,
    strategy: str,
    config: dict[str, Any],
) -> StrategyResult:
    """Apply a single optimization strategy to a model.

    Parameters
    ----------
    model:
        A UnifiedModel instance.
    strategy:
        Strategy name (e.g., "torch_compile", "half_precision").
    config:
        Strategy-specific options from YAML.

    Returns
    -------
    StrategyResult with status and any error info.
    """
    result = StrategyResult(name=strategy)

    dispatch = {
        "torch_compile": _apply_torch_compile,
        "half_precision": _apply_half_precision,
        "dynamic_int8": _apply_dynamic_int8,
        "torch_tensorrt": _apply_torch_tensorrt,
        "onnx": _apply_onnx_export,
        "openvino": _apply_openvino,
        "ultralytics_export": _apply_ultralytics_export,
        "ultralytics_benchmark": _apply_ultralytics_benchmark,
        "resolution_reduction": _apply_resolution_reduction,
    }

    fn = dispatch.get(strategy)
    if fn is None:
        result.available = False
        result.error = f"Unknown strategy: {strategy!r}"
        return result

    try:
        fn(model, config, result)
    except ImportError as e:
        result.available = False
        result.error = f"Missing dependency: {e}"
        logger.info("Strategy %s unavailable: %s", strategy, e)
    except Exception as e:
        result.applied = False
        result.error = str(e)
        logger.warning("Strategy %s failed: %s", strategy, e)

    return result


def auto_optimize(
    model: Any,
    config: dict[str, Any],
    sample_images: list[Any],
) -> tuple[Any, list[StrategyResult]]:
    """Try all available strategies and pick the best.

    Applies strategies in order, measures latency after each,
    keeps the one that gives the best speedup.
    """
    import torch

    available = probe_available_strategies()
    logger.info("Available optimization backends: %s",
                {k: v for k, v in available.items() if v})

    # Strategies to try, in priority order
    if model.backend == "ultralytics":
        strategy_order = [
            "half_precision",
            "ultralytics_export",
            "torch_compile",
        ]
    else:
        strategy_order = [
            "half_precision",
            "torch_compile",
            "dynamic_int8",
            "torch_tensorrt",
            "onnx",
        ]

    # Filter to available
    strategies = [s for s in strategy_order if available.get(s, False)]

    # Baseline
    baseline = model.benchmark_latency(sample_images, num_warmup=5, num_runs=20)
    baseline_ms = baseline["inference_ms"]
    logger.info("Baseline latency: %.2f ms", baseline_ms)

    results: list[StrategyResult] = []
    best_ms = baseline_ms
    best_model = model

    for strategy in strategies:
        # We can't easily undo optimizations, so we just measure what's possible
        result = apply_strategy(model, strategy, config)
        if result.applied:
            latency = model.benchmark_latency(sample_images, num_warmup=5, num_runs=20)
            result.latency_ms = latency["inference_ms"]
            result.throughput = latency["throughput_img_per_sec"]
            result.vram_mb = latency["peak_vram_mb"]

            if result.latency_ms < best_ms:
                best_ms = result.latency_ms
                best_model = model

        results.append(result)

    return best_model, results


def _apply_torch_compile(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """torch.compile with configurable mode."""
    import torch

    mode = config.get("mode", "reduce-overhead")
    backend = config.get("backend", "inductor")
    fullgraph = config.get("fullgraph", False)

    nn_mod = model.nn_module
    compiled = torch.compile(nn_mod, mode=mode, backend=backend, fullgraph=fullgraph)

    if model.backend == "hf":
        model.model = compiled
    else:
        model._raw.model = compiled

    result.applied = True
    logger.info("Applied torch.compile(mode=%r, backend=%r)", mode, backend)


def _apply_half_precision(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """Convert model to FP16."""
    import torch

    device = next(model.nn_module.parameters()).device
    if device.type == "cpu":
        result.applied = False
        result.error = "FP16 not beneficial on CPU"
        return

    model.nn_module.half()
    result.applied = True
    logger.info("Applied half precision (FP16)")


def _apply_dynamic_int8(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """PyTorch dynamic INT8 quantization on Linear layers."""
    import torch
    import torch.nn as nn

    nn_mod = model.nn_module.cpu()
    layers = config.get("layers", [nn.Linear])

    # Resolve layer types from strings
    layer_types = set()
    for layer in layers:
        if isinstance(layer, str):
            layer_types.add(getattr(nn, layer, nn.Linear))
        else:
            layer_types.add(layer)

    quantized = torch.quantization.quantize_dynamic(
        nn_mod, layer_types, dtype=torch.qint8
    )

    if model.backend == "hf":
        model.model = quantized
    else:
        model._raw.model = quantized

    result.applied = True
    logger.info("Applied dynamic INT8 quantization")


def _apply_torch_tensorrt(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """Compile with Torch-TensorRT backend."""
    import torch
    import torch_tensorrt  # noqa: F401

    nn_mod = model.nn_module
    imgsz = config.get("image_size", 640)

    compiled = torch.compile(
        nn_mod,
        backend="torch_tensorrt",
        dynamic=False,
        options={
            "min_block_size": config.get("min_block_size", 3),
            "optimization_level": config.get("optimization_level", 3),
        },
    )

    if model.backend == "hf":
        model.model = compiled
    else:
        model._raw.model = compiled

    result.applied = True
    logger.info("Applied Torch-TensorRT compilation")


def _apply_onnx_export(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """Export to ONNX and load with ONNX Runtime."""
    import onnxruntime  # noqa: F401

    # ONNX export is complex and model-specific — mark as available but not auto-applied
    # The user should use model.export("onnx", path) explicitly
    result.applied = False
    result.error = "ONNX export requires explicit path — use model.export('onnx', output_path)"


def _apply_openvino(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """Export to OpenVINO format."""
    import openvino  # noqa: F401

    result.applied = False
    result.error = "OpenVINO export requires explicit path — use model.export('openvino', output_path)"


def _apply_ultralytics_export(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """Ultralytics model export (TensorRT, ONNX, etc.)."""
    if model.backend != "ultralytics":
        result.applied = False
        result.error = "ultralytics_export only works for Ultralytics models"
        return

    fmt = config.get("format", "engine")
    imgsz = config.get("image_size", 640)
    half = config.get("half", True)

    try:
        export_path = model._raw.export(format=fmt, imgsz=imgsz, half=half)
        result.export_path = str(export_path)
        result.applied = True
        logger.info("Exported Ultralytics model to %s: %s", fmt, export_path)
    except Exception as e:
        result.applied = False
        result.error = f"Export to {fmt} failed: {e}"


def _apply_ultralytics_benchmark(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """Run Ultralytics benchmark across all export formats."""
    if model.backend != "ultralytics":
        result.applied = False
        result.error = "ultralytics_benchmark only works for Ultralytics models"
        return

    imgsz = config.get("image_size", 640)
    half = config.get("half", False)
    fmt = config.get("format", "")

    try:
        bench_results = model._raw.benchmark(imgsz=imgsz, half=half, format=fmt)
        result.applied = True
        logger.info("Ultralytics benchmark complete")
    except Exception as e:
        result.applied = False
        result.error = f"Benchmark failed: {e}"


def _apply_resolution_reduction(model: Any, config: dict[str, Any], result: StrategyResult) -> None:
    """Reduce input resolution for faster inference."""
    target_size = config.get("target_size", 320)

    if model.backend == "hf":
        processor = model.processor
        ip = getattr(processor, "image_processor", processor)

        if hasattr(ip, "size"):
            if isinstance(ip.size, dict):
                ip.size = {"height": target_size, "width": target_size}
            else:
                ip.size = target_size
        if hasattr(ip, "crop_size"):
            if isinstance(ip.crop_size, dict):
                ip.crop_size = {"height": target_size, "width": target_size}
            else:
                ip.crop_size = target_size
    else:
        model._raw.overrides["imgsz"] = target_size

    result.applied = True
    logger.info("Reduced resolution to %d", target_size)
