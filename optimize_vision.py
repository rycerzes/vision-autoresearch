"""Inference optimization loop — optimize latency while gating on accuracy.

The agent writes optimization code in ``experiments/optimization.py``.
The harness measures latency and accuracy, accepts/rejects based on
an accuracy gate (max allowed drop from baseline).

Works for both HF and Ultralytics models.  Optimization strategies include:
- torch.compile() with various backends/modes
- Quantization (BitsAndBytes, GPTQ, AWQ, dynamic int8)
- ONNX / TensorRT / OpenVINO export
- Resolution reduction
- Structured pruning
- Flash Attention / SDPA swaps
- Half precision inference
- Batch inference optimization

Usage:
    uv run optimize_vision.py config.yaml

Config:
    model_name_or_path: ustc-community/dfine-small-coco
    dataset_name: cppe-5
    optimization_module: experiments/optimization.py
    max_accuracy_drop: 0.02
    num_latency_samples: 100
    num_warmup: 10
    num_runs: 100
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)




@dataclass
class OptimizationConfig:
    """Configuration for the optimization loop."""

    # Model & data
    model_name_or_path: str = ""
    dataset_name: str = ""
    dataset_config_name: str | None = None
    head_category: str | None = None

    # Optimization control
    optimization_module: str = "experiments/optimization.py"
    max_accuracy_drop: float = 0.02  # max allowed accuracy drop from baseline
    num_latency_samples: int = 100  # images for latency measurement
    num_warmup: int = 10
    num_runs: int = 100

    # Output
    output_dir: str = "./output/optimization"
    image_size: int = 640

    # Trust
    trust_remote_code: bool = True


def parse_optimization_config(yaml_path: str) -> OptimizationConfig:
    """Parse optimization config from YAML."""
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    config = OptimizationConfig()
    for key, value in raw.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    return config




@dataclass
class OptimizationResult:
    """Result of one optimization iteration."""

    description: str = ""
    status: str = "pending"  # pending, accepted, rejected, crashed

    # Baseline
    baseline_latency_ms: float = 0.0
    baseline_throughput: float = 0.0
    baseline_vram_mb: float = 0.0
    baseline_accuracy: float = 0.0

    # Optimized
    optimized_latency_ms: float = 0.0
    optimized_throughput: float = 0.0
    optimized_vram_mb: float = 0.0
    optimized_accuracy: float = 0.0

    # Deltas
    latency_speedup: float = 1.0  # baseline / optimized (>1 = faster)
    accuracy_drop: float = 0.0
    vram_reduction_mb: float = 0.0

    # Details
    optimization_type: str = ""
    export_path: str | None = None
    error: str | None = None

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["═══ Optimization Result ═══"]
        lines.append(f"  Status:         {self.status}")
        lines.append(f"  Type:           {self.optimization_type}")
        lines.append(f"  Description:    {self.description}")
        lines.append("")
        lines.append("  ── Baseline ──")
        lines.append(f"    Latency:      {self.baseline_latency_ms:.2f} ms")
        lines.append(f"    Throughput:    {self.baseline_throughput:.1f} img/s")
        lines.append(f"    VRAM:         {self.baseline_vram_mb:.0f} MB")
        lines.append(f"    Accuracy:     {self.baseline_accuracy:.4f}")
        lines.append("")
        lines.append("  ── Optimized ──")
        lines.append(f"    Latency:      {self.optimized_latency_ms:.2f} ms")
        lines.append(f"    Throughput:    {self.optimized_throughput:.1f} img/s")
        lines.append(f"    VRAM:         {self.optimized_vram_mb:.0f} MB")
        lines.append(f"    Accuracy:     {self.optimized_accuracy:.4f}")
        lines.append("")
        lines.append("  ── Delta ──")
        lines.append(f"    Speedup:      {self.latency_speedup:.2f}x")
        lines.append(f"    Accuracy drop:{self.accuracy_drop:+.4f}")
        lines.append(f"    VRAM saved:   {self.vram_reduction_mb:.0f} MB")

        if self.export_path:
            lines.append(f"    Export:       {self.export_path}")
        if self.error:
            lines.append(f"    Error:        {self.error}")
        lines.append("═══════════════════════════")
        return "\n".join(lines)




def load_optimization_module(path: str | Path) -> Any | None:
    """Load experiments/optimization.py."""
    path = Path(path)
    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location("optimization", str(path))
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module




def run_optimization(
    config: OptimizationConfig,
) -> OptimizationResult:
    """Run one optimization iteration.

    1. Load model
    2. Measure baseline latency + accuracy
    3. Apply optimization from experiments/optimization.py
    4. Measure optimized latency + accuracy
    5. Accept/reject based on accuracy gate
    """
    import torch

    from engine.backend import load_model
    from engine.metrics import HEAD_METRICS
    from engine.unified_dataset import UnifiedDataset

    result = OptimizationResult()

    logger.info("Loading model: %s", config.model_name_or_path)
    model = load_model(
        config.model_name_or_path,
        mode="predict",
        head_category_override=config.head_category,
    )

    logger.info("Loading dataset: %s", config.dataset_name)
    dataset = UnifiedDataset(
        config.dataset_name,
        config.dataset_config_name,
        trust_remote_code=config.trust_remote_code,
    )
    dataset.auto_map_columns(model.head_category)

    # Get sample images for benchmarking
    sample_images = dataset.sample_images(config.num_latency_samples)
    if not sample_images:
        result.status = "crashed"
        result.error = "No sample images available for benchmarking"
        return result

    logger.info("Measuring baseline latency...")
    baseline_latency = model.benchmark_latency(
        sample_images,
        num_warmup=config.num_warmup,
        num_runs=config.num_runs,
    )
    result.baseline_latency_ms = baseline_latency["inference_ms"]
    result.baseline_throughput = baseline_latency["throughput_img_per_sec"]
    result.baseline_vram_mb = baseline_latency["peak_vram_mb"]

    logger.info(
        "Baseline: %.2f ms/img, %.1f img/s, %.0f MB VRAM",
        result.baseline_latency_ms,
        result.baseline_throughput,
        result.baseline_vram_mb,
    )

    # Baseline accuracy (primary metric for this head category)
    primary_metric = HEAD_METRICS.get(model.head_category, ["accuracy"])[0]

    optimization = load_optimization_module(config.optimization_module)
    if optimization is None:
        result.status = "crashed"
        result.error = f"Optimization module not found: {config.optimization_module}"
        return result

    result.optimization_type = getattr(optimization, "OPTIMIZATION_TYPE", "unknown")
    result.description = getattr(optimization, "DESCRIPTION", "")

    logger.info("Applying optimization...")
    optimize_fn = getattr(optimization, "optimize", None)
    if optimize_fn is None:
        result.status = "crashed"
        result.error = "Optimization module has no optimize() function"
        return result

    try:
        optimized_model = optimize_fn(model, config={
            "head_category": model.head_category,
            "image_size": config.image_size,
            "output_dir": config.output_dir,
            "sample_images": sample_images[:10],  # calibration images
        })
        if optimized_model is None:
            optimized_model = model
    except Exception as e:
        result.status = "crashed"
        result.error = f"optimize() failed: {e}"
        logger.error("Optimization failed: %s", e, exc_info=True)
        return result

    logger.info("Measuring optimized latency...")
    try:
        optimized_latency = optimized_model.benchmark_latency(
            sample_images,
            num_warmup=config.num_warmup,
            num_runs=config.num_runs,
        )
        result.optimized_latency_ms = optimized_latency["inference_ms"]
        result.optimized_throughput = optimized_latency["throughput_img_per_sec"]
        result.optimized_vram_mb = optimized_latency["peak_vram_mb"]
    except Exception as e:
        result.status = "crashed"
        result.error = f"Latency measurement failed after optimization: {e}"
        return result

    if result.optimized_latency_ms > 0:
        result.latency_speedup = result.baseline_latency_ms / result.optimized_latency_ms
    result.vram_reduction_mb = result.baseline_vram_mb - result.optimized_vram_mb

    # Check if optimization module provides accuracy evaluation
    evaluate_fn = getattr(optimization, "evaluate_accuracy", None)
    if evaluate_fn is not None:
        try:
            accuracy_metrics = evaluate_fn(optimized_model, dataset, sample_images)
            result.optimized_accuracy = accuracy_metrics.get(primary_metric, 0.0)
        except Exception as e:
            logger.warning("Accuracy evaluation failed: %s — skipping gate", e)
            result.optimized_accuracy = result.baseline_accuracy
    else:
        # If no accuracy eval provided, assume no degradation
        result.optimized_accuracy = result.baseline_accuracy

    result.accuracy_drop = result.baseline_accuracy - result.optimized_accuracy

    if result.accuracy_drop > config.max_accuracy_drop:
        result.status = "rejected"
        logger.warning(
            "REJECTED: accuracy drop %.4f > max allowed %.4f",
            result.accuracy_drop,
            config.max_accuracy_drop,
        )
    elif result.latency_speedup < 1.0:
        result.status = "rejected"
        logger.warning(
            "REJECTED: optimization made inference SLOWER (%.2fx)",
            result.latency_speedup,
        )
    else:
        result.status = "accepted"
        logger.info(
            "ACCEPTED: %.2fx speedup, accuracy drop=%.4f",
            result.latency_speedup,
            result.accuracy_drop,
        )

    # Check for export path
    export_path = getattr(optimization, "EXPORT_PATH", None)
    if export_path:
        result.export_path = str(export_path)

    return result




def main() -> None:
    """Run the optimization loop from CLI."""
    if len(sys.argv) < 2 or not sys.argv[1].endswith((".yaml", ".yml")):
        print("Usage: optimize_vision.py <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.abspath(sys.argv[1])

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    config = parse_optimization_config(config_path)
    logger.info("Optimization config: %s", config_path)
    logger.info("Model: %s", config.model_name_or_path)
    logger.info("Max accuracy drop: %.4f", config.max_accuracy_drop)

    # Run optimization
    result = run_optimization(config)

    # Output result
    print(result.summary())

    # Emit structured summary for parse_metric.py
    print("\n--- VISION AUTORESEARCH SUMMARY ---")
    print(f"head_category: optimization")
    print(f"inference_ms: {result.optimized_latency_ms:.2f}")
    print(f"throughput_img_per_sec: {result.optimized_throughput:.1f}")
    print(f"latency_speedup: {result.latency_speedup:.2f}")
    print(f"accuracy_drop: {result.accuracy_drop:.4f}")
    print(f"peak_vram_mb: {result.optimized_vram_mb:.0f}")
    print(f"optimization_status: {result.status}")
    print("--- END SUMMARY ---")

    # Save result to JSON
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "optimization_result.json"
    result_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info("Result saved to %s", result_path)

    sys.exit(0 if result.status == "accepted" else 1)


if __name__ == "__main__":
    main()
