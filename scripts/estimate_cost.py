#!/usr/bin/env python3
"""Estimate HF Jobs cost for a vision autoresearch run based on config and task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vision_lab.task_registry import ESTIMATED_MINUTES_BY_TASK, all_task_ids

GPU_HOURLY_RATES = {
    "l4": 0.89,
    "a10g": 1.10,
    "a100": 3.72,
    "a100-80gb": 5.00,
    "t4": 0.60,
}

DATASET_SIZE_THRESHOLDS = {
    "small": 5_000,
    "medium": 50_000,
}


def estimate_dataset_size(dataset_name: str) -> str:
    """Heuristic: try to query HF Hub for row count, fall back to 'medium'."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info(dataset_name)
        card = info.card_data
        if card and hasattr(card, "size_categories") and card.size_categories:
            cat = card.size_categories[0] if isinstance(card.size_categories, list) else str(card.size_categories)
            if "1K" in cat or "n<" in cat:
                return "small"
            if "10K" in cat or "100K" in cat:
                return "medium"
            return "large"
    except Exception:
        pass
    return "medium"


def estimate(task: str, config_path: Path, flavor: str = "l4") -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    dataset_name = config.get("dataset_name", "unknown")
    epochs = int(config.get("num_train_epochs", 10))
    batch_size = int(config.get("per_device_train_batch_size", 8))
    grad_accum = int(config.get("gradient_accumulation_steps", 1))

    size_category = estimate_dataset_size(dataset_name)
    base_minutes = ESTIMATED_MINUTES_BY_TASK.get(task, ESTIMATED_MINUTES_BY_TASK["detect"]).get(
        size_category, 45
    )

    epoch_factor = epochs / 10.0
    batch_factor = 8.0 / (batch_size * grad_accum)
    estimated_minutes = base_minutes * epoch_factor * max(batch_factor, 0.5)
    estimated_minutes = max(5, min(estimated_minutes, 720))

    rate = GPU_HOURLY_RATES.get(flavor, GPU_HOURLY_RATES["l4"])
    estimated_cost = (estimated_minutes / 60.0) * rate

    return {
        "task": task,
        "config": str(config_path),
        "dataset": dataset_name,
        "dataset_size_category": size_category,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "flavor": flavor,
        "gpu_hourly_rate_usd": rate,
        "estimated_minutes": round(estimated_minutes, 1),
        "estimated_cost_usd": round(estimated_cost, 3),
        "note": "Rough estimate; actual cost depends on dataset size, model, and convergence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate HF Jobs cost for a run.")
    parser.add_argument(
        "--task",
        required=True,
        choices=list(all_task_ids()),
    )
    parser.add_argument("--config", type=Path, help="Config YAML path")
    parser.add_argument("--flavor", default="l4", choices=list(GPU_HOURLY_RATES.keys()))
    args = parser.parse_args()

    config_path = args.config or ROOT / "configs" / f"base_{args.task}.yaml"
    result = estimate(args.task, config_path, args.flavor)

    print(f"Task:      {result['task']}")
    print(f"Dataset:   {result['dataset']} ({result['dataset_size_category']})")
    print(f"Epochs:    {result['epochs']}")
    print(f"Batch:     {result['batch_size']} x {result['gradient_accumulation_steps']} accum")
    print(f"GPU:       {result['flavor']} (${result['gpu_hourly_rate_usd']}/hr)")
    print(f"Est. time: ~{result['estimated_minutes']} minutes")
    print(f"Est. cost: ~${result['estimated_cost_usd']}")
    print(f"Note:      {result['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
