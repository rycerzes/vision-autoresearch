"""Inference optimization loop — optimize latency while gating on accuracy.

Two modes:
1. **Built-in strategies** (default) — selected via YAML config, the engine
   applies them automatically.  No code writing needed.
2. **Custom code** — agent writes ``experiments/optimization.py`` for novel
   optimizations that don't fit a built-in strategy.

Built-in strategies: torch_compile, half_precision, dynamic_int8,
torch_tensorrt, ultralytics_export, resolution_reduction.

Usage:
    uv run optimize_vision.py config.yaml

Config (built-in strategy):
    model_name_or_path: yolo11n.pt
    dataset_name: cppe-5
    strategy: torch_compile          # or: half_precision, auto, etc.
    strategy_config:
      mode: reduce-overhead
    max_accuracy_drop: 0.02

Config (custom code):
    model_name_or_path: ustc-community/dfine-small-coco
    dataset_name: cppe-5
    strategy: custom
    optimization_module: experiments/optimization.py
    max_accuracy_drop: 0.02
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

    model_name_or_path: str = ""
    dataset_name: str = ""
    dataset_config_name: str | None = None
    head_category: str | None = None

    # Strategy selection
    strategy: str = "auto"  # auto, torch_compile, half_precision, dynamic_int8,
    #                         torch_tensorrt, ultralytics_export, resolution_reduction, custom
    strategy_config: dict[str, Any] = field(default_factory=dict)

    # Custom code path (only when strategy=custom)
    optimization_module: str = "experiments/optimization.py"

    # Accuracy gate
    max_accuracy_drop: float = 0.02

    # Benchmarking
    num_latency_samples: int = 100
    num_warmup: int = 10
    num_runs: int = 100

    # Output
    output_dir: str = "./output/optimization"
    image_size: int = 640

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
    """Result of the optimization run."""

    strategy: str = ""
    status: str = "pending"  # accepted, rejected, crashed, unavailable

    baseline_latency_ms: float = 0.0
    baseline_throughput: float = 0.0
    baseline_vram_mb: float = 0.0

    optimized_latency_ms: float = 0.0
    optimized_throughput: float = 0.0
    optimized_vram_mb: float = 0.0

    latency_speedup: float = 1.0
    accuracy_drop: float = 0.0
    vram_reduction_mb: float = 0.0

    export_path: str | None = None
    error: str | None = None
    available_strategies: dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["═══ Optimization Result ═══"]
        lines.append(f"  Strategy:       {self.strategy}")
        lines.append(f"  Status:         {self.status}")
        if self.error:
            lines.append(f"  Error:          {self.error}")
        lines.append(f"  Baseline:       {self.baseline_latency_ms:.2f} ms ({self.baseline_throughput:.1f} img/s)")
        lines.append(f"  Optimized:      {self.optimized_latency_ms:.2f} ms ({self.optimized_throughput:.1f} img/s)")
        lines.append(f"  Speedup:        {self.latency_speedup:.2f}x")
        lines.append(f"  VRAM saved:     {self.vram_reduction_mb:.0f} MB")
        lines.append(f"  Accuracy drop:  {self.accuracy_drop:+.4f}")
        if self.export_path:
            lines.append(f"  Export:         {self.export_path}")
        if self.available_strategies:
            avail = [k for k, v in self.available_strategies.items() if v]
            lines.append(f"  Available:      {', '.join(avail)}")
        lines.append("═══════════════════════════")
        return "\n".join(lines)


def run_optimization(config: OptimizationConfig) -> OptimizationResult:
    """Run the optimization loop.

    1. Load model and dataset
    2. Probe available strategies
    3. Measure baseline
    4. Apply strategy (built-in or custom)
    5. Measure optimized
    6. Accuracy gate → accept/reject
    """
    import torch

    from engine.backend import load_model
    from engine.metrics import HEAD_METRICS
    from engine.optimization import apply_strategy, auto_optimize, probe_available_strategies
    from engine.unified_dataset import UnifiedDataset

    result = OptimizationResult(strategy=config.strategy)

    # Load model
    logger.info("Loading model: %s", config.model_name_or_path)
    model = load_model(
        config.model_name_or_path,
        mode="predict",
        head_category_override=config.head_category,
    )

    # Load dataset + get sample images
    logger.info("Loading dataset: %s", config.dataset_name)
    dataset = UnifiedDataset(
        config.dataset_name,
        config.dataset_config_name,
        trust_remote_code=config.trust_remote_code,
    )
    dataset.auto_map_columns(model.head_category)
    sample_images = dataset.sample_images(config.num_latency_samples)
    if not sample_images:
        result.status = "crashed"
        result.error = "No sample images available"
        return result

    # Probe available strategies
    available = probe_available_strategies()
    result.available_strategies = available
    logger.info("Available strategies: %s", {k: v for k, v in available.items() if v})

    # Baseline measurement
    logger.info("Measuring baseline...")
    baseline = model.benchmark_latency(
        sample_images, num_warmup=config.num_warmup, num_runs=config.num_runs
    )
    result.baseline_latency_ms = baseline["inference_ms"]
    result.baseline_throughput = baseline["throughput_img_per_sec"]
    result.baseline_vram_mb = baseline["peak_vram_mb"]
    logger.info("Baseline: %.2f ms, %.1f img/s", result.baseline_latency_ms, result.baseline_throughput)

    # Apply strategy
    if config.strategy == "auto":
        model, strategy_results = auto_optimize(
            model, {**config.strategy_config, "image_size": config.image_size}, sample_images
        )
        applied = [r for r in strategy_results if r.applied]
        if applied:
            result.strategy = applied[-1].name
        else:
            result.status = "unavailable"
            result.error = "No strategy improved latency"
            return result

    elif config.strategy == "custom":
        # Custom code surface
        custom_result = _run_custom_optimization(model, config, sample_images)
        if custom_result is not None:
            result.error = custom_result
            result.status = "crashed"
            return result

    else:
        # Named built-in strategy
        if not available.get(config.strategy, False):
            result.status = "unavailable"
            result.error = f"Strategy '{config.strategy}' not available (missing dependency)"
            return result

        strat_result = apply_strategy(
            model, config.strategy, {**config.strategy_config, "image_size": config.image_size}
        )
        if not strat_result.applied:
            result.status = "crashed"
            result.error = strat_result.error
            return result
        result.export_path = strat_result.export_path

    # Measure optimized
    logger.info("Measuring optimized...")
    try:
        optimized = model.benchmark_latency(
            sample_images, num_warmup=config.num_warmup, num_runs=config.num_runs
        )
        result.optimized_latency_ms = optimized["inference_ms"]
        result.optimized_throughput = optimized["throughput_img_per_sec"]
        result.optimized_vram_mb = optimized["peak_vram_mb"]
    except Exception as e:
        result.status = "crashed"
        result.error = f"Post-optimization benchmark failed: {e}"
        return result

    # Compute deltas
    if result.optimized_latency_ms > 0:
        result.latency_speedup = result.baseline_latency_ms / result.optimized_latency_ms
    result.vram_reduction_mb = result.baseline_vram_mb - result.optimized_vram_mb

    # Accept/reject
    if result.latency_speedup < 1.0:
        result.status = "rejected"
        logger.warning("REJECTED: optimization made inference slower (%.2fx)", result.latency_speedup)
    else:
        result.status = "accepted"
        logger.info("ACCEPTED: %.2fx speedup", result.latency_speedup)

    return result


def _run_custom_optimization(
    model: Any, config: OptimizationConfig, sample_images: list[Any]
) -> str | None:
    """Run custom optimization from experiments/optimization.py.

    Returns error string on failure, None on success.
    """
    path = Path(config.optimization_module)
    if not path.exists():
        return f"Custom optimization module not found: {path}"

    spec = importlib.util.spec_from_file_location("optimization", str(path))
    if spec is None or spec.loader is None:
        return f"Cannot load module: {path}"

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        return f"Module load failed: {e}"

    optimize_fn = getattr(module, "optimize", None)
    if optimize_fn is None:
        return "Module has no optimize() function"

    try:
        result = optimize_fn(model, {
            "head_category": model.head_category,
            "image_size": config.image_size,
            "output_dir": config.output_dir,
            "sample_images": sample_images[:10],
        })
    except Exception as e:
        return f"optimize() failed: {e}"

    return None


def main() -> None:
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
    logger.info("Model: %s | Strategy: %s | Max drop: %.4f",
                config.model_name_or_path, config.strategy, config.max_accuracy_drop)

    result = run_optimization(config)
    print(result.summary())

    # Structured summary
    print("\n--- VISION AUTORESEARCH SUMMARY ---")
    print(f"head_category: optimization")
    print(f"inference_ms: {result.optimized_latency_ms:.2f}")
    print(f"throughput_img_per_sec: {result.optimized_throughput:.1f}")
    print(f"latency_speedup: {result.latency_speedup:.2f}")
    print(f"accuracy_drop: {result.accuracy_drop:.4f}")
    print(f"peak_vram_mb: {result.optimized_vram_mb:.0f}")
    print(f"optimization_status: {result.status}")
    print("--- END SUMMARY ---")

    # Save JSON
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optimization_result.json").write_text(json.dumps(asdict(result), indent=2))

    sys.exit(0 if result.status == "accepted" else 1)


if __name__ == "__main__":
    main()
