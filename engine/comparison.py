"""Model comparison engine — cross-backend leaderboard and Pareto analysis.

Compares any mix of HF Transformers and Ultralytics models on the same
dataset.  Three comparison modes:

- **zero_shot**: Evaluate pretrained models directly (no training).
- **finetuned**: Fine-tune each model with identical universal args, then evaluate.
- **pareto**: Measure accuracy + latency + VRAM for Pareto frontier analysis.

All models are evaluated with the same metric regardless of backend.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)



@dataclass
class ComparisonConfig:
    """Configuration for the model comparison loop."""

    # Models to compare (mixed HF + Ultralytics supported)
    models: list[str] = field(default_factory=list)

    # Dataset
    dataset_name: str = ""
    dataset_config_name: str | None = None

    # Comparison mode
    comparison_mode: str = "zero_shot"  # zero_shot | finetuned | pareto

    # Head category override (auto-inferred per model if None)
    head_category: str | None = None

    # Training args (used in finetuned mode)
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    image_size: int = 640
    fp16: bool = True
    seed: int = 42
    freeze_backbone: bool = False

    # HF / Ultralytics pass-through
    hf_train: dict[str, Any] = field(default_factory=dict)
    ultralytics_train: dict[str, Any] = field(default_factory=dict)

    # Latency benchmark settings (pareto + zero_shot)
    measure_latency: bool = True
    num_latency_samples: int = 100
    num_warmup: int = 10
    num_latency_runs: int = 100

    # Metric override (auto-derived from head_category if None)
    primary_metric: str | None = None

    # Output
    output_dir: str = "./output/comparison"

    # Limits
    max_eval_samples: int | None = None

    trust_remote_code: bool = True


def parse_comparison_config(yaml_path: str) -> ComparisonConfig:
    """Parse comparison config from YAML."""
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    config = ComparisonConfig()
    for key, value in raw.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    if not config.models:
        raise ValueError("No models specified in comparison config")
    if not config.dataset_name:
        raise ValueError("No dataset_name specified in comparison config")

    return config



@dataclass
class ModelResult:
    """Result for a single model in the comparison."""

    model_name: str = ""
    backend: str = ""
    head_category: str = ""
    num_parameters: int = 0
    status: str = "pending"  # completed | failed | skipped
    error: str | None = None

    # Eval metrics (standard keys: mAP, accuracy, miou, etc.)
    metrics: dict[str, float] = field(default_factory=dict)

    # Latency (optional, depends on config)
    inference_ms: float = 0.0
    throughput_img_per_sec: float = 0.0
    peak_vram_mb: float = 0.0

    # Training info (finetuned mode)
    training_seconds: float = 0.0
    train_loss: float = 0.0

    # Pareto
    is_pareto_optimal: bool = False


@dataclass
class ComparisonResult:
    """Aggregated comparison result with leaderboard and Pareto frontier."""

    config: ComparisonConfig = field(default_factory=ComparisonConfig)
    results: list[ModelResult] = field(default_factory=list)
    primary_metric: str = ""
    primary_direction: str = "higher"
    pareto_frontier: list[str] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)
    total_seconds: float = 0.0

    def get_result(self, model_name: str) -> ModelResult | None:
        for r in self.results:
            if r.model_name == model_name:
                return r
        return None



def run_comparison(config: ComparisonConfig) -> ComparisonResult:
    """Execute the full model comparison loop.

    1. Load each model via ``load_model()`` (auto-detects backend)
    2. Load dataset once via ``UnifiedDataset``
    3. For each model:
       a. Map columns for the model's head category
       b. Evaluate (zero-shot) or train-then-evaluate (finetuned)
       c. Optionally benchmark latency
    4. Build leaderboard (rank by primary metric)
    5. Compute Pareto frontier (accuracy vs latency vs params)
    """
    from engine.backend import detect_backend, load_model
    from engine.metrics import HEAD_METRICS, MetricDirection, default_promotion_metric, get_direction
    from engine.unified_dataset import UnifiedDataset

    overall_start = time.time()
    comparison = ComparisonResult(config=config)
    completed_results: list[ModelResult] = []

    # Load dataset once (shared across all models)
    logger.info("Loading dataset: %s", config.dataset_name)
    dataset = UnifiedDataset(
        config.dataset_name,
        config.dataset_config_name,
        trust_remote_code=config.trust_remote_code,
    )

    # Create train/val split once before the loop (not per-model)
    dataset.ensure_train_val_split(seed=config.seed)

    for model_name in config.models:
        logger.info("\n══════════════════════════════════════════")
        logger.info("  Evaluating: %s", model_name)
        logger.info("══════════════════════════════════════════")

        model_result = ModelResult(model_name=model_name)

        try:
            result = _evaluate_single_model(
                model_name, dataset, config, model_result
            )
            completed_results.append(result)
        except Exception as e:
            logger.error("Model %s FAILED: %s", model_name, e, exc_info=True)
            model_result.status = "failed"
            model_result.error = str(e)
            completed_results.append(model_result)
        finally:
            # Free GPU memory between models to avoid OOM
            _cleanup_gpu()

    comparison.results = completed_results

    # Determine primary metric
    primary_metric, direction = _resolve_primary_metric(config, completed_results)
    comparison.primary_metric = primary_metric
    comparison.primary_direction = direction.value if hasattr(direction, "value") else str(direction)

    # Build ranking
    comparison.ranking = _build_ranking(
        completed_results, primary_metric, direction
    )

    # Compute Pareto frontier (accuracy vs latency)
    if config.measure_latency:
        comparison.pareto_frontier = _compute_pareto_frontier(
            completed_results, primary_metric, direction
        )
        for r in completed_results:
            r.is_pareto_optimal = r.model_name in comparison.pareto_frontier

    comparison.total_seconds = time.time() - overall_start
    return comparison


def _evaluate_single_model(
    model_name: str,
    dataset: Any,
    config: ComparisonConfig,
    result: ModelResult,
) -> ModelResult:
    """Load, optionally train, evaluate, and benchmark one model."""
    from engine.backend import load_model
    from engine.pipeline import auto_infer_pipeline

    mode = "train" if config.comparison_mode == "finetuned" else "predict"
    model = load_model(
        model_name,
        mode=mode,
        head_category_override=config.head_category,
    )
    result.backend = model.backend
    result.head_category = model.head_category
    result.num_parameters = model.num_parameters

    logger.info(
        "  backend=%s  head_category=%s  params=%s",
        model.backend,
        model.head_category,
        f"{model.num_parameters:,}",
    )

    # Auto-infer pipeline (column mapping, preprocessing, etc.)
    pipeline_config = auto_infer_pipeline(
        model,
        dataset,
        image_size_override=config.image_size,
    )

    # Adapt num classes for HF models
    if pipeline_config.num_classes is not None:
        from engine.hf_backend import HFModel
        if isinstance(model, HFModel):
            model.adapt_num_classes(
                pipeline_config.num_classes,
                class_names=pipeline_config.class_names,
            )

    if config.comparison_mode == "finetuned":
        result = _train_and_evaluate(model, dataset, config, pipeline_config, result)
    else:
        # zero_shot or pareto — evaluate only
        result = _evaluate_only(model, dataset, config, pipeline_config, result)

    # Latency benchmark
    if config.measure_latency:
        result = _benchmark_latency(model, dataset, config, result)

    result.status = "completed"
    return result


def _evaluate_only(
    model: Any,
    dataset: Any,
    config: ComparisonConfig,
    pipeline_config: Any,
    result: ModelResult,
) -> ModelResult:
    """Evaluate a pretrained model without training (zero-shot)."""
    logger.info("  Mode: zero-shot evaluation")

    if model.backend == "hf":
        # Prepare ONLY eval dataset (skip train split for zero-shot — saves
        # significant time on large datasets)
        eval_ds = _prepare_hf_eval_only(
            model, dataset, config, pipeline_config
        )

        if config.max_eval_samples is not None:
            eval_ds = eval_ds.select(
                range(min(config.max_eval_samples, len(eval_ds)))
            )

        metrics = model.evaluate(eval_ds)
    else:
        # Ultralytics: need data.yaml for validation
        output_dir = Path(config.output_dir) / _safe_dirname(result.model_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        id2label = _resolve_id2label(pipeline_config, model, dataset)
        task = getattr(model._raw, "task", "detect")
        data_yaml = dataset.for_ultralytics(
            task=task,
            output_dir=output_dir / "dataset",
            id2label=id2label,
        )

        # Set classes for open-vocab models
        if pipeline_config.class_names:
            model.set_classes(pipeline_config.class_names)

        metrics = model.evaluate(dataset, data_yaml=data_yaml)

    # Clean metric keys (strip eval_ prefix from HF)
    clean_metrics: dict[str, float] = {}
    for k, v in metrics.items():
        key = k.replace("eval_", "")
        try:
            clean_metrics[key] = float(v)
        except (TypeError, ValueError):
            pass

    result.metrics = clean_metrics
    logger.info("  Metrics: %s", {k: f"{v:.4f}" for k, v in clean_metrics.items()})
    return result


def _train_and_evaluate(
    model: Any,
    dataset: Any,
    config: ComparisonConfig,
    pipeline_config: Any,
    result: ModelResult,
) -> ModelResult:
    """Train a model with universal args, then evaluate."""
    import torch

    from engine.training import (
        UniversalTrainingArgs,
        to_hf_training_args,
        to_ultralytics_train_kwargs,
    )

    logger.info("  Mode: fine-tuned evaluation")
    train_start = time.time()

    # Check if model supports training (YOLO-NAS, FastSAM, SAM, YOLOv4, YOLOv7 don't)
    if model.backend == "ultralytics" and not getattr(model, "can_train", True):
        result.status = "skipped"
        result.error = (
            f"{result.model_name} does not support training. "
            "Use zero_shot comparison mode instead."
        )
        logger.warning("  Skipped: %s", result.error)
        return result

    # Build universal args from comparison config
    args = UniversalTrainingArgs(
        model_name_or_path=result.model_name,
        dataset_name=config.dataset_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        image_size=config.image_size,
        fp16=config.fp16,
        seed=config.seed,
        output_dir=str(
            Path(config.output_dir) / _safe_dirname(result.model_name)
        ),
        hf_train=config.hf_train,
        ultralytics_train=config.ultralytics_train,
    )

    if model.backend == "hf":
        hf_datasets = dataset.for_hf(
            processor=model.processor,
            head_category=model.head_category,
            column_map=pipeline_config.column_map,
            train_augmentation=pipeline_config.train_augmentation,
            eval_augmentation=pipeline_config.eval_augmentation,
            image_size=pipeline_config.image_size,
        )
        train_ds = hf_datasets["train"]
        eval_ds = hf_datasets["eval"]

        hf_args = to_hf_training_args(args)

        all_metrics = model.train(
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=hf_args,
            pipeline_config=pipeline_config,
        )

        # Clean metrics
        for k, v in all_metrics.items():
            key = k.replace("eval_", "")
            if "train" in k or k == "epoch":
                continue
            try:
                result.metrics[key] = float(v)
            except (TypeError, ValueError):
                pass

        result.train_loss = float(all_metrics.get("train_loss", 0.0))

    else:
        # Ultralytics
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        id2label = _resolve_id2label(pipeline_config, model, dataset)
        task = getattr(model._raw, "task", "detect")
        data_yaml = dataset.for_ultralytics(
            task=task,
            output_dir=output_dir / "dataset",
            id2label=id2label,
        )

        train_kwargs = to_ultralytics_train_kwargs(args, data_yaml, output_dir)

        # Set classes for open-vocab models
        if pipeline_config.class_names:
            model.set_classes(pipeline_config.class_names)

        all_metrics = model.train(
            train_dataset=dataset,
            eval_dataset=None,
            args=train_kwargs,
            pipeline_config=pipeline_config,
        )

        for k, v in all_metrics.items():
            if k in ("train_loss", "epoch"):
                continue
            try:
                result.metrics[k] = float(v)
            except (TypeError, ValueError):
                pass

        result.train_loss = float(all_metrics.get("train_loss", 0.0))

    result.training_seconds = time.time() - train_start
    logger.info("  Training: %.1fs", result.training_seconds)
    logger.info("  Metrics: %s", {k: f"{v:.4f}" for k, v in result.metrics.items()})
    return result


def _prepare_hf_eval_only(
    model: Any,
    dataset: Any,
    config: ComparisonConfig,
    pipeline_config: Any,
) -> Any:
    """Prepare only the eval split for HF models (zero-shot).

    Unlike ``dataset.for_hf()`` which processes both train and eval splits
    through ``.map()``, this processes only the eval split — avoids wasting
    time on the train split when we just need zero-shot evaluation.
    """
    import numpy as np
    from PIL import Image as PILImage

    processor = model.processor
    head_category = model.head_category
    cmap = pipeline_config.column_map
    image_col = cmap["image"]
    target_col = cmap.get("target")
    subfield_map = cmap.get("target_subfields")
    eval_augmentation = pipeline_config.eval_augmentation

    eval_split = dataset.eval_split_name
    eval_raw = dataset.hf_dataset[eval_split]

    def transform_example(example: dict) -> dict:
        img = example[image_col]
        if hasattr(img, "convert"):
            img = img.convert("RGB")

        if eval_augmentation is not None:
            img_array = np.array(img)
            aug_result = eval_augmentation(image=img_array)
            img = PILImage.fromarray(aug_result["image"])

        processed = processor(images=img, return_tensors="pt")
        result = {}
        for k, v in processed.items():
            result[k] = v.squeeze(0) if hasattr(v, "squeeze") else v

        if target_col and target_col in example:
            from engine.unified_dataset import _format_labels
            result["labels"] = _format_labels(
                example[target_col], head_category, subfield_map
            )

        return result

    eval_ds = eval_raw.map(
        transform_example,
        remove_columns=eval_raw.column_names,
    )
    return eval_ds


def _benchmark_latency(
    model: Any,
    dataset: Any,
    config: ComparisonConfig,
    result: ModelResult,
) -> ModelResult:
    """Measure inference latency for a model."""
    logger.info("  Benchmarking latency...")

    sample_images = dataset.sample_images(config.num_latency_samples)
    if not sample_images:
        logger.warning("  No sample images available for latency benchmark")
        return result

    try:
        latency = model.benchmark_latency(
            sample_images,
            num_warmup=config.num_warmup,
            num_runs=config.num_latency_runs,
        )
        result.inference_ms = latency["inference_ms"]
        result.throughput_img_per_sec = latency["throughput_img_per_sec"]
        result.peak_vram_mb = latency["peak_vram_mb"]
        logger.info(
            "  Latency: %.2f ms  (%.1f img/s)  VRAM: %.0f MB",
            result.inference_ms,
            result.throughput_img_per_sec,
            result.peak_vram_mb,
        )
    except Exception as e:
        logger.warning("  Latency benchmark failed: %s", e)

    return result



def _resolve_primary_metric(
    config: ComparisonConfig,
    results: list[ModelResult],
) -> tuple[str, Any]:
    """Determine the primary metric for ranking.

    Uses config override, or auto-derives from the first successful model's
    head category.  Warns if models have mixed head categories.
    """
    from engine.metrics import MetricDirection, default_promotion_metric, get_direction

    if config.primary_metric:
        return config.primary_metric, get_direction(config.primary_metric)

    # Check for mixed head categories
    categories = set(
        r.head_category
        for r in results
        if r.status == "completed" and r.head_category
    )
    if len(categories) > 1:
        logger.warning(
            "Mixed head categories detected: %s. "
            "Models with different tasks use different metrics — "
            "set `primary_metric` in YAML to compare meaningfully.",
            categories,
        )

    # Auto-derive from first completed model
    for r in results:
        if r.status == "completed" and r.head_category:
            metric, direction = default_promotion_metric(r.head_category)
            return metric, direction

    return "mAP", MetricDirection.HIGHER


def _build_ranking(
    results: list[ModelResult],
    primary_metric: str,
    direction: Any,
) -> list[str]:
    """Rank models by primary metric value."""
    from engine.metrics import MetricDirection

    completed = [r for r in results if r.status == "completed"]
    if not completed:
        return []

    reverse = True  # higher is better by default
    if hasattr(direction, "value"):
        reverse = direction == MetricDirection.HIGHER
    elif isinstance(direction, str):
        reverse = direction.lower() == "higher"

    ranked = sorted(
        completed,
        key=lambda r: r.metrics.get(primary_metric, 0.0),
        reverse=reverse,
    )
    return [r.model_name for r in ranked]


def _compute_pareto_frontier(
    results: list[ModelResult],
    primary_metric: str,
    direction: Any,
) -> list[str]:
    """Compute the Pareto frontier: accuracy vs latency.

    A model is Pareto-optimal if no other model is both faster AND more
    accurate.  Direction of the primary metric determines "more accurate".
    """
    from engine.metrics import MetricDirection

    completed = [
        r for r in results
        if r.status == "completed" and r.inference_ms > 0
    ]
    if not completed:
        return []

    higher_is_better = True
    if hasattr(direction, "value"):
        higher_is_better = direction == MetricDirection.HIGHER
    elif isinstance(direction, str):
        higher_is_better = direction.lower() == "higher"

    pareto: list[ModelResult] = []
    for candidate in completed:
        c_metric = candidate.metrics.get(primary_metric, 0.0)
        c_latency = candidate.inference_ms

        dominated = False
        for other in completed:
            if other is candidate:
                continue
            o_metric = other.metrics.get(primary_metric, 0.0)
            o_latency = other.inference_ms

            if higher_is_better:
                # "other" dominates if it's at least as good on both and strictly better on one
                better_or_equal_metric = o_metric >= c_metric
                better_or_equal_latency = o_latency <= c_latency
                strictly_better = o_metric > c_metric or o_latency < c_latency
            else:
                better_or_equal_metric = o_metric <= c_metric
                better_or_equal_latency = o_latency <= c_latency
                strictly_better = o_metric < c_metric or o_latency < c_latency

            if better_or_equal_metric and better_or_equal_latency and strictly_better:
                dominated = True
                break

        if not dominated:
            pareto.append(candidate)

    return [r.model_name for r in pareto]



def format_leaderboard(
    comparison: ComparisonResult,
    *,
    show_latency: bool = True,
) -> str:
    """Format the comparison result as a human-readable leaderboard table."""
    primary = comparison.primary_metric
    results = comparison.results

    # Build rows ordered by ranking
    ordered: list[ModelResult] = []
    for name in comparison.ranking:
        r = comparison.get_result(name)
        if r:
            ordered.append(r)
    # Append failed models at the end
    for r in results:
        if r.model_name not in comparison.ranking:
            ordered.append(r)

    if not ordered:
        return "No models evaluated."

    # Column widths
    col_model = max(len("Model"), max(len(_short_name(r.model_name)) for r in ordered))
    col_backend = max(len("Backend"), 7)
    col_metric = max(len(primary), 8)
    col_params = max(len("Params"), 8)
    col_status = max(len("Status"), 8)

    # Header
    parts = [
        f"{'#':>3}",
        f"{'Model':<{col_model}}",
        f"{'Backend':<{col_backend}}",
        f"{primary:>{col_metric}}",
    ]
    if show_latency:
        parts.extend([
            f"{'Latency':>10}",
            f"{'Tput':>10}",
            f"{'VRAM':>8}",
        ])
    parts.extend([
        f"{'Params':>{col_params}}",
        f"{'Pareto':>7}",
        f"{'Status':<{col_status}}",
    ])
    header = "  ".join(parts)
    sep = "─" * len(header)

    lines = [
        "",
        f"═══ Model Comparison: {comparison.config.comparison_mode} ═══",
        f"Dataset: {comparison.config.dataset_name}  |  Primary: {primary} ({comparison.primary_direction})",
        f"Models: {len(results)}  |  Time: {comparison.total_seconds:.1f}s",
        "",
        sep,
        header,
        sep,
    ]

    for rank, r in enumerate(ordered, 1):
        metric_val = r.metrics.get(primary, 0.0)
        metric_str = f"{metric_val:.4f}" if r.status == "completed" else "—"
        pareto_str = "  ★" if r.is_pareto_optimal else ""
        params_str = _format_params(r.num_parameters) if r.num_parameters > 0 else "—"
        status_str = r.status
        if r.error:
            status_str = f"FAIL: {r.error[:30]}"

        parts = [
            f"{rank:>3}",
            f"{_short_name(r.model_name):<{col_model}}",
            f"{r.backend:<{col_backend}}",
            f"{metric_str:>{col_metric}}",
        ]
        if show_latency:
            lat_str = f"{r.inference_ms:.1f} ms" if r.inference_ms > 0 else "—"
            tput_str = f"{r.throughput_img_per_sec:.1f}/s" if r.throughput_img_per_sec > 0 else "—"
            vram_str = f"{r.peak_vram_mb:.0f} MB" if r.peak_vram_mb > 0 else "—"
            parts.extend([
                f"{lat_str:>10}",
                f"{tput_str:>10}",
                f"{vram_str:>8}",
            ])
        parts.extend([
            f"{params_str:>{col_params}}",
            f"{pareto_str:>7}",
            f"{status_str:<{col_status}}",
        ])
        lines.append("  ".join(parts))

    lines.append(sep)

    # Pareto summary
    if comparison.pareto_frontier:
        pareto_names = [_short_name(n) for n in comparison.pareto_frontier]
        lines.append(f"Pareto optimal: {', '.join(pareto_names)}")

    # Winner
    if comparison.ranking:
        winner = comparison.ranking[0]
        winner_r = comparison.get_result(winner)
        if winner_r:
            val = winner_r.metrics.get(primary, 0.0)
            lines.append(
                f"Best {primary}: {_short_name(winner)} ({val:.4f})"
            )

    lines.append("")
    return "\n".join(lines)


def format_summary_block(comparison: ComparisonResult) -> str:
    """Format structured summary block for parse_metric.py extraction."""
    primary = comparison.primary_metric

    lines = ["\n--- VISION AUTORESEARCH SUMMARY ---"]
    lines.append(f"head_category: comparison")
    lines.append(f"comparison_mode: {comparison.config.comparison_mode}")
    lines.append(f"primary_metric: {primary}")
    lines.append(f"num_models: {len(comparison.results)}")
    lines.append(f"num_completed: {sum(1 for r in comparison.results if r.status == 'completed')}")

    if comparison.ranking:
        winner = comparison.ranking[0]
        winner_r = comparison.get_result(winner)
        if winner_r:
            val = winner_r.metrics.get(primary, 0.0)
            lines.append(f"best_model: {winner}")
            lines.append(f"best_{primary}: {val:.4f}")
            lines.append(f"best_inference_ms: {winner_r.inference_ms:.2f}")
            lines.append(f"best_params: {winner_r.num_parameters}")

    if comparison.pareto_frontier:
        lines.append(f"pareto_models: {', '.join(comparison.pareto_frontier)}")

    lines.append(f"total_seconds: {comparison.total_seconds:.1f}")
    lines.append("--- END SUMMARY ---")
    return "\n".join(lines)



def _cleanup_gpu() -> None:
    """Release GPU memory between model evaluations.

    Critical for comparison loops: without cleanup, each model's weights
    accumulate in VRAM and later models OOM.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _short_name(model_name: str) -> str:
    """Shorten model name for display (drop org prefix if needed)."""
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def _format_params(n: int) -> str:
    """Format parameter count: 42100000 → '42.1M'."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _safe_dirname(model_name: str) -> str:
    """Convert model name to a safe directory name."""
    return model_name.replace("/", "__").replace(":", "_").replace(" ", "_")


def _resolve_id2label(
    pipeline_config: Any,
    model: Any,
    dataset: Any,
) -> dict[int, str]:
    """Resolve id2label mapping from pipeline config, model, or dataset."""
    if pipeline_config.class_names:
        return {i: name for i, name in enumerate(pipeline_config.class_names)}
    names = model.get_class_names()
    if names:
        return {i: name for i, name in enumerate(names)}
    ds_names = dataset.class_names
    if ds_names:
        return {i: name for i, name in enumerate(ds_names)}
    return {0: "object"}
