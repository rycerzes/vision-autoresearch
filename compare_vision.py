"""Model comparison entry point — cross-backend leaderboard.

Compare any mix of HF Transformers and Ultralytics models on the same
dataset with one command.  Supports zero-shot, fine-tuned, and Pareto
comparison modes.

Usage:
    uv run compare_vision.py configs/example_compare.yaml

    # Quick CLI (no YAML needed):
    uv run compare_vision.py --models yolo11n.pt ustc-community/dfine-small-coco \\
        --dataset cppe-5 --mode zero_shot

Config YAML:
    models:
      - facebook/detr-resnet-50
      - yolo11n.pt
      - yoloe-l.pt
    dataset_name: cppe-5
    comparison_mode: zero_shot
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare vision models across HF and Ultralytics backends",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to comparison YAML config (e.g. configs/example_compare.yaml)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names/paths to compare (alternative to YAML)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (alternative to YAML)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero_shot", "finetuned", "pareto"],
        default="zero_shot",
        help="Comparison mode (default: zero_shot)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/comparison",
        help="Output directory (default: ./output/comparison)",
    )
    parser.add_argument(
        "--no-latency",
        action="store_true",
        help="Skip latency benchmarking",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default=None,
        help="Override primary metric (e.g. mAP, accuracy)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image size for evaluation (default: 640)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit eval samples per model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for finetuned mode (default: 10)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    # HF login if token available
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("Logged in to Hugging Face Hub")

    # Build config
    if args.config and Path(args.config).exists():
        from engine.comparison import parse_comparison_config
        config = parse_comparison_config(os.path.abspath(args.config))
    elif args.models and args.dataset:
        from engine.comparison import ComparisonConfig
        config = ComparisonConfig(
            models=list(args.models),
            dataset_name=args.dataset,
            comparison_mode=args.mode,
            output_dir=args.output_dir,
            measure_latency=not args.no_latency,
            primary_metric=args.primary_metric,
            image_size=args.image_size,
            max_eval_samples=args.max_eval_samples,
            num_train_epochs=args.epochs,
        )
    else:
        parser.print_help()
        print(
            "\nError: Provide either a YAML config or --models + --dataset",
            file=sys.stderr,
        )
        sys.exit(1)

    # CLI overrides (applied even when using YAML)
    if args.no_latency:
        config.measure_latency = False
    if args.primary_metric:
        config.primary_metric = args.primary_metric
    if args.max_eval_samples is not None:
        config.max_eval_samples = args.max_eval_samples

    logger.info("═══ Model Comparison ═══")
    logger.info("  Mode:    %s", config.comparison_mode)
    logger.info("  Dataset: %s", config.dataset_name)
    logger.info("  Models:  %s", config.models)
    if config.comparison_mode == "finetuned":
        logger.info("  Epochs:  %d", config.num_train_epochs)
    logger.info("  Latency: %s", config.measure_latency)

    # Run comparison
    from engine.comparison import (
        format_leaderboard,
        format_summary_block,
        run_comparison,
    )

    comparison = run_comparison(config)

    # Print leaderboard
    leaderboard = format_leaderboard(
        comparison, show_latency=config.measure_latency
    )
    print(leaderboard)

    # Print structured summary
    summary = format_summary_block(comparison)
    print(summary)

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON results
    results_json = {
        "comparison_mode": config.comparison_mode,
        "dataset": config.dataset_name,
        "primary_metric": comparison.primary_metric,
        "primary_direction": comparison.primary_direction,
        "ranking": comparison.ranking,
        "pareto_frontier": comparison.pareto_frontier,
        "total_seconds": comparison.total_seconds,
        "models": [asdict(r) for r in comparison.results],
    }
    json_path = output_dir / "comparison_result.json"
    json_path.write_text(json.dumps(results_json, indent=2, default=str))
    logger.info("Results saved to: %s", json_path)

    # Leaderboard text
    txt_path = output_dir / "leaderboard.txt"
    txt_path.write_text(leaderboard)

    # Per-model metric files (for downstream consumption)
    for r in comparison.results:
        if r.status == "completed":
            model_dir = output_dir / r.model_name.replace("/", "__")
            model_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = model_dir / "metrics.json"
            metrics_path.write_text(json.dumps(
                {
                    "model": r.model_name,
                    "backend": r.backend,
                    "head_category": r.head_category,
                    "metrics": r.metrics,
                    "inference_ms": r.inference_ms,
                    "throughput_img_per_sec": r.throughput_img_per_sec,
                    "peak_vram_mb": r.peak_vram_mb,
                    "num_parameters": r.num_parameters,
                    "is_pareto_optimal": r.is_pareto_optimal,
                },
                indent=2,
            ))

    # Exit code: 0 if at least one model completed
    completed = sum(1 for r in comparison.results if r.status == "completed")
    if completed == 0:
        logger.error("All models failed!")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
