"""Unified training entry point for ALL vision models.

Single script that handles both HF Transformers and Ultralytics models.
Replaces train_detect.py, train_classify.py, train_segment.py, and
train_ultralytics.py with one backend-agnostic entry point.

Usage:
    uv run train_vision.py config.yaml

The config YAML is the experiment surface.  Universal args are mapped
to backend-native equivalents automatically.  Backend-specific overrides
go in ``hf_train:`` or ``ultralytics_train:`` blocks.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def main() -> None:
    start_time = time.time()

    # ── Parse config ────────────────────────────────────────────
    if len(sys.argv) < 2 or not sys.argv[1].endswith((".yaml", ".yml")):
        print("Usage: train_vision.py <config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.abspath(sys.argv[1])

    from engine.training import UniversalTrainingArgs, parse_config_yaml

    args = parse_config_yaml(config_path)

    # ── Setup logging ───────────────────────────────────────────
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    logger.info("Config: %s", config_path)
    logger.info("Model: %s", args.model_name_or_path)
    logger.info("Dataset: %s", args.dataset_name)

    # ── Hub authentication ──────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("Logged in to Hugging Face Hub")

    # ── Trackio ─────────────────────────────────────────────────
    try:
        import trackio
        trackio.init(project=args.output_dir, name=args.run_name or "train_vision")
    except ImportError:
        trackio = None  # type: ignore[assignment]

    # ── Load model via unified API ──────────────────────────────
    from engine.backend import load_model

    model = load_model(
        args.model_name_or_path,
        mode="train",
        head_category_override=args.head_category,
    )
    logger.info(
        "Model loaded: backend=%s  head_category=%s  params=%s",
        model.backend,
        model.head_category,
        f"{model.num_parameters:,}",
    )

    # ── Load dataset via unified API ────────────────────────────
    from engine.unified_dataset import UnifiedDataset

    dataset = UnifiedDataset(
        args.dataset_name,
        args.dataset_config_name,
        trust_remote_code=args.trust_remote_code,
    )

    # ── Auto-infer pipeline ─────────────────────────────────────
    from engine.pipeline import auto_infer_pipeline, summarize_pipeline

    pipeline_config = auto_infer_pipeline(
        model,
        dataset,
        image_size_override=args.image_size,
        column_map_override=args.column_map,
        use_albumentations=args.use_albumentations,
        use_trivial_augment=args.use_trivial_augment,
    )
    logger.info("\n%s", summarize_pipeline(pipeline_config))

    # ── Apply research modifications (if present) ───────────────
    modification = _load_modification(args.modification_module)
    if modification is not None:
        model, dataset, pipeline_config = _apply_modification(
            modification, model, dataset, pipeline_config, args
        )

    # ── Freeze backbone if requested ────────────────────────────
    if args.freeze_backbone:
        _freeze_backbone(model)

    # ── Route to backend-specific training ──────────────────────
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if model.backend == "hf":
        eval_metrics, train_metrics = _train_hf(model, dataset, args, pipeline_config, modification)
    else:
        eval_metrics, train_metrics = _train_ultralytics(model, dataset, args, pipeline_config, modification)

    # ── Emit summary ────────────────────────────────────────────
    training_seconds = time.time() - start_time
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1e6
        if torch.cuda.is_available()
        else 0.0
    )

    from engine.training import emit_summary

    emit_summary(
        model.head_category,
        eval_metrics,
        train_metrics,
        training_seconds,
        peak_vram_mb,
    )

    # ── Trackio finish ──────────────────────────────────────────
    if trackio is not None:
        try:
            trackio.finish()
        except Exception:
            pass

    logger.info("Training complete in %.1fs", training_seconds)


# ══ HF Training ═════════════════════════════════════════════════


def _train_hf(
    model: Any,
    dataset: Any,
    args: Any,
    pipeline_config: Any,
    modification: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run training via HF Trainer."""
    from engine.training import to_hf_training_args

    # Prepare datasets for HF Trainer
    image_size = pipeline_config.image_size
    hf_datasets = dataset.for_hf(
        processor=model.processor,
        head_category=model.head_category,
        column_map=pipeline_config.column_map,
        train_augmentation=pipeline_config.train_augmentation,
        eval_augmentation=pipeline_config.eval_augmentation,
        image_size=image_size,
    )
    train_ds = hf_datasets["train"]
    eval_ds = hf_datasets["eval"]

    # Truncate if requested
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    # Build HF TrainingArguments
    hf_args = to_hf_training_args(args)

    # Override metric_for_best_model if promotion config provides it
    if args.promotion_metric and "metric_for_best_model" not in args.hf_train:
        hf_args["metric_for_best_model"] = args.promotion_metric

    # Custom loss from modification
    custom_loss_fn = None
    if modification is not None:
        modify_loss = getattr(modification, "modify_loss", None)
        if modify_loss is not None:
            custom_loss_fn = modify_loss

    # Custom compute_metrics from modification
    compute_metrics_fn = None
    if modification is not None:
        modify_metrics = getattr(modification, "modify_metrics", None)
        if modify_metrics is not None:
            compute_metrics_fn = modify_metrics

    # Train
    all_metrics = model.train(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=hf_args,
        pipeline_config=pipeline_config,
        compute_metrics_fn=compute_metrics_fn,
        custom_loss_fn=custom_loss_fn,
    )

    # Separate train and eval metrics
    train_metrics = {k: v for k, v in all_metrics.items() if "train" in k or k == "epoch"}
    eval_metrics = {k: v for k, v in all_metrics.items() if k not in train_metrics}

    return eval_metrics, train_metrics


# ══ Ultralytics Training ════════════════════════════════════════


def _train_ultralytics(
    model: Any,
    dataset: Any,
    args: Any,
    pipeline_config: Any,
    modification: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run training via Ultralytics .train() API."""
    from engine.training import (
        read_ultralytics_eval_metrics,
        read_ultralytics_train_metrics,
        resolve_ultralytics_trainer,
        to_ultralytics_train_kwargs,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve Ultralytics task from model
    task = getattr(model._raw, "task", "detect")

    # Export dataset to YOLO format
    id2label = {}
    if pipeline_config.class_names:
        id2label = {i: name for i, name in enumerate(pipeline_config.class_names)}
    elif model.get_class_names():
        names = model.get_class_names()
        id2label = {i: name for i, name in enumerate(names)}
    else:
        # Attempt from dataset
        ds_names = dataset.class_names
        if ds_names:
            id2label = {i: name for i, name in enumerate(ds_names)}
        else:
            id2label = {0: "object"}

    data_yaml = dataset.for_ultralytics(
        task=task,
        output_dir=output_dir / "dataset",
        id2label=id2label,
    )
    logger.info("Dataset exported to YOLO format: %s", data_yaml)

    # Build train kwargs
    train_kwargs = to_ultralytics_train_kwargs(args, data_yaml, output_dir)

    # Resolve trainer class
    trainer_cls = resolve_ultralytics_trainer(model, args)

    # Custom trainer from modification
    if modification is not None:
        custom_trainer_fn = getattr(modification, "custom_trainer_class", None)
        if custom_trainer_fn is not None:
            custom_cls = custom_trainer_fn()
            if custom_cls is not None:
                trainer_cls = custom_cls

    # Train
    all_metrics = model.train(
        train_dataset=dataset,
        eval_dataset=None,
        args=train_kwargs,
        pipeline_config=pipeline_config,
        trainer_cls=trainer_cls,
    )

    # Separate train/eval
    train_keys = {"train_loss", "epoch"}
    train_metrics = {k: v for k, v in all_metrics.items() if k in train_keys}
    eval_metrics = {k: v for k, v in all_metrics.items() if k not in train_keys}

    return eval_metrics, train_metrics


# ══ Research modifications ══════════════════════════════════════


def _load_modification(module_path: str | None) -> Any | None:
    """Load the modification module (experiments/modification.py or custom path)."""
    if module_path is None:
        # Check default location
        default_path = Path("experiments/modification.py")
        if not default_path.exists():
            return None
        module_path = str(default_path)

    path = Path(module_path)
    if not path.exists():
        logger.warning("Modification module not found: %s", module_path)
        return None

    spec = importlib.util.spec_from_file_location("modification", path)
    if spec is None or spec.loader is None:
        logger.warning("Cannot load modification module: %s", module_path)
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    logger.info("Loaded modification module: %s", module_path)
    return module


def _apply_modification(
    modification: Any,
    model: Any,
    dataset: Any,
    pipeline_config: Any,
    args: Any,
) -> tuple[Any, Any, Any]:
    """Apply all modification hooks."""
    config_dict = {
        "head_category": model.head_category,
        "image_size": pipeline_config.image_size,
        "num_classes": pipeline_config.num_classes,
        "class_names": pipeline_config.class_names,
    }

    # modify_model
    modify_model_fn = getattr(modification, "modify_model", None)
    if modify_model_fn is not None:
        logger.info("Applying modify_model()")
        model = modify_model_fn(model, config_dict) or model

    # modify_data
    modify_data_fn = getattr(modification, "modify_data", None)
    if modify_data_fn is not None:
        logger.info("Applying modify_data()")
        result = modify_data_fn(dataset, model)
        if result is not None:
            dataset = result

    # freeze_strategy
    freeze_fn = getattr(modification, "freeze_strategy", None)
    if freeze_fn is not None:
        logger.info("Applying freeze_strategy()")
        freeze_fn(model)

    return model, dataset, pipeline_config


def _freeze_backbone(model: Any) -> None:
    """Freeze backbone using module role detection."""
    try:
        backbone_path, _ = model.find_module_by_role("backbone")
        model.freeze_except([])  # freeze all first
        # Then unfreeze non-backbone (head) modules
        import torch.nn as nn

        backbone_params = set()
        for name, _ in model.nn_module.named_parameters():
            if name.startswith(backbone_path):
                backbone_params.add(name)

        frozen = trainable = 0
        for name, param in model.nn_module.named_parameters():
            if name in backbone_params:
                param.requires_grad_(False)
                frozen += 1
            else:
                param.requires_grad_(True)
                trainable += 1
        logger.info(
            "Backbone frozen: %d frozen, %d trainable", frozen, trainable
        )
    except ValueError:
        # Fallback: freeze first 70% of parameters by declaration order
        all_params = list(model.nn_module.named_parameters())
        cutoff = int(len(all_params) * 0.7)
        for _, param in all_params[:cutoff]:
            param.requires_grad_(False)
        logger.info(
            "Backbone frozen (heuristic): %d/%d params frozen",
            cutoff,
            len(all_params),
        )


if __name__ == "__main__":
    main()
