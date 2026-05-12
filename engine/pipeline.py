"""Auto-inference pipeline — assembles the full training/eval pipeline.

Given any model + dataset, this module:
1. Discovers preprocessing config (image size, normalization)
2. Maps dataset columns to model inputs (type-based)
3. Probes loss mode (builtin vs external)
4. Derives evaluation metrics from head category
5. Selects augmentation family
6. Builds collation function
7. Returns a fully-configured ``PipelineConfig`` ready for training

Works for both HF and Ultralytics backends.  Ultralytics skips steps that
it handles internally (augmentation, collation, loss).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Everything needed to train/eval a model on a dataset.

    This is the output of ``auto_infer_pipeline``.  No hardcoded values —
    everything is derived from model and dataset metadata at runtime.
    """

    backend: str  # "hf" or "ultralytics"
    head_category: str  # "detection", "classification", etc.
    model_name: str

    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    modalities: list[str] = field(default_factory=lambda: ["image"])
    has_tokenizer: bool = False

    column_map: dict[str, str] = field(default_factory=dict)

    loss_mode: str = "builtin"  # "builtin" or "external"
    external_loss_fn: Callable[..., torch.Tensor] | None = None

    default_metrics: list[str] = field(default_factory=list)
    promotion_metric: str = ""
    promotion_direction: str = "higher"

    augmentation_family: str = "image_only"
    train_augmentation: Callable[..., dict[str, Any]] | None = None
    eval_augmentation: Callable[..., dict[str, Any]] | None = None

    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None

    class_names: list[str] | None = None
    num_classes: int | None = None

    warnings: list[str] = field(default_factory=list)


def auto_infer_pipeline(
    model: Any,
    dataset: Any,
    *,
    image_size_override: int | tuple[int, int] | None = None,
    column_map_override: dict[str, str] | None = None,
    use_albumentations: bool = True,
    use_trivial_augment: bool = False,
    bbox_format: str = "coco",
) -> PipelineConfig:
    """Auto-infer the full training pipeline from model + dataset.

    Parameters
    ----------
    model:
        A ``UnifiedModel`` (``HFModel`` or ``UltralyticsModel``).
    dataset:
        A ``UnifiedDataset``.
    image_size_override:
        Force image size instead of reading from processor.
    column_map_override:
        Explicit column map from YAML config.
    use_albumentations:
        Whether to use albumentations (True) or torchvision (False).
    use_trivial_augment:
        Use TrivialAugment for classification.
    bbox_format:
        Bbox format for detection augmentation (``"coco"``, ``"pascal_voc"``, etc.).

    Returns
    -------
    PipelineConfig
        Fully populated config with all pipeline components.
    """
    config = PipelineConfig(
        backend=model.backend,
        head_category=model.head_category,
        model_name=getattr(model, "_model_name", "unknown"),
    )

    _discover_preprocessing(model, config, image_size_override)

    _resolve_column_map(model, dataset, config, column_map_override)

    _detect_loss(model, config)

    _derive_metrics(model, config)

    _build_augmentations(model, config, use_albumentations, use_trivial_augment, bbox_format)

    _build_collation(model, config)

    _resolve_class_names(model, dataset, config)

    logger.info(
        "Pipeline auto-inferred: backend=%s head=%s size=%s loss=%s metrics=%s "
        "aug=%s classes=%s",
        config.backend,
        config.head_category,
        config.image_size,
        config.loss_mode,
        config.default_metrics,
        config.augmentation_family,
        config.num_classes,
    )

    return config




def _discover_preprocessing(
    model: Any,
    config: PipelineConfig,
    image_size_override: int | tuple[int, int] | None,
) -> None:
    """Step 1: Extract preprocessing parameters from processor/model."""
    from engine.preprocessing import (
        discover_preprocessing,
        discover_ultralytics_preprocessing,
    )

    if model.backend == "hf":
        processor = getattr(model, "processor", None)
        if processor is not None:
            pre = discover_preprocessing(
                processor, image_size_override=image_size_override
            )
        else:
            pre = discover_preprocessing(
                None, image_size_override=image_size_override or 224
            )
            config.warnings.append("No processor found — using default preprocessing")
    else:
        # Ultralytics
        raw = getattr(model, "_raw", None)
        override_int = (
            image_size_override
            if isinstance(image_size_override, int)
            else (image_size_override[0] if image_size_override else None)
        )
        pre = discover_ultralytics_preprocessing(
            raw, image_size_override=override_int
        )

    config.image_size = pre.image_size
    config.image_mean = pre.image_mean
    config.image_std = pre.image_std
    config.modalities = pre.modalities
    config.has_tokenizer = pre.has_tokenizer


def _resolve_column_map(
    model: Any,
    dataset: Any,
    config: PipelineConfig,
    column_map_override: dict[str, str] | None,
) -> None:
    """Step 2: Map dataset columns to model inputs."""
    if column_map_override:
        dataset.set_column_map(column_map_override)
        config.column_map = dict(column_map_override)
    else:
        processor = getattr(model, "processor", None)
        mapping = dataset.auto_map_columns(
            model.head_category, processor=processor
        )
        config.column_map = dict(mapping)


def _detect_loss(model: Any, config: PipelineConfig) -> None:
    """Step 3: Determine if the model computes its own loss."""
    from engine.loss import get_external_loss, probe_loss_mode

    if model.backend == "ultralytics":
        # Ultralytics models ALWAYS compute loss internally during training
        config.loss_mode = "builtin"
        config.external_loss_fn = None
        return

    # HF: probe the model
    processor = getattr(model, "processor", None)
    try:
        mode = probe_loss_mode(model.nn_module, processor, model.head_category)
    except Exception as e:
        logger.warning("Loss probe failed: %s — assuming external", e)
        mode = "external"

    config.loss_mode = mode
    if mode == "external":
        config.external_loss_fn = get_external_loss(model.head_category)
        if config.external_loss_fn is None:
            config.warnings.append(
                f"No external loss function for head_category={model.head_category!r}. "
                "Model will need builtin loss or a custom loss via modification.py."
            )


def _derive_metrics(model: Any, config: PipelineConfig) -> None:
    """Step 4: Determine which metrics to evaluate."""
    from engine.metrics import HEAD_METRICS, default_promotion_metric

    config.default_metrics = list(HEAD_METRICS.get(model.head_category, ["train_loss"]))
    primary, direction = default_promotion_metric(model.head_category)
    config.promotion_metric = primary
    config.promotion_direction = direction.value


def _build_augmentations(
    model: Any,
    config: PipelineConfig,
    use_albumentations: bool,
    use_trivial_augment: bool,
    bbox_format: str,
) -> None:
    """Step 5: Build train and eval augmentation transforms."""
    from engine.augmentation import (
        build_eval_augmentation,
        build_train_augmentation,
        infer_augmentation_family,
    )

    if model.backend == "ultralytics":
        # Ultralytics has built-in augmentation — don't build external transforms
        family = infer_augmentation_family(model.head_category)
        config.augmentation_family = family.value
        config.train_augmentation = None
        config.eval_augmentation = None
        return

    family = infer_augmentation_family(model.head_category)
    config.augmentation_family = family.value

    try:
        config.train_augmentation = build_train_augmentation(
            model.head_category,
            config.image_size,
            use_albumentations=use_albumentations,
            use_trivial_augment=use_trivial_augment,
            bbox_format=bbox_format,
        )
    except ImportError as e:
        config.warnings.append(f"Augmentation build failed: {e}")
        config.train_augmentation = None

    try:
        config.eval_augmentation = build_eval_augmentation(
            model.head_category,
            config.image_size,
        )
    except ImportError as e:
        config.warnings.append(f"Eval augmentation build failed: {e}")
        config.eval_augmentation = None


def _build_collation(model: Any, config: PipelineConfig) -> None:
    """Step 6: Build the collation function."""
    from engine.collation import build_collate_fn

    if model.backend == "ultralytics":
        # Ultralytics has built-in data loading + collation
        config.collate_fn = None
        return

    processor = getattr(model, "processor", None)
    config.collate_fn = build_collate_fn(
        model.head_category,
        processor,
        column_map=config.column_map,
    )


def _resolve_class_names(
    model: Any,
    dataset: Any,
    config: PipelineConfig,
) -> None:
    """Step 7: Discover class names from model or dataset."""
    # Try model first (pretrained models know their classes)
    names = model.get_class_names()
    if names:
        config.class_names = names
        config.num_classes = len(names)
        return

    # Fall back to dataset
    ds_names = dataset.class_names
    if ds_names:
        config.class_names = ds_names
        config.num_classes = len(ds_names)
        return

    # Try to infer from model config
    model_config = getattr(model, "config", None)
    if model_config is not None:
        num_labels = getattr(model_config, "num_labels", None) or getattr(
            model_config, "num_classes", None
        )
        if num_labels:
            config.num_classes = num_labels
            return

    config.warnings.append("Could not determine class names or count")




def summarize_pipeline(config: PipelineConfig) -> str:
    """Human-readable summary of the inferred pipeline."""
    lines = [
        f"═══ Auto-Inferred Pipeline ═══",
        f"  Backend:            {config.backend}",
        f"  Head category:      {config.head_category}",
        f"  Model:              {config.model_name}",
        f"  Image size:         {config.image_size}",
        f"  Normalization:      mean={config.image_mean}  std={config.image_std}",
        f"  Modalities:         {config.modalities}",
        f"  Column map:         {config.column_map}",
        f"  Loss mode:          {config.loss_mode}",
        f"  Metrics:            {config.default_metrics}",
        f"  Promotion:          {config.promotion_metric} ({config.promotion_direction})",
        f"  Augmentation:       {config.augmentation_family}",
        f"  Classes:            {config.num_classes}",
    ]
    if config.warnings:
        lines.append(f"  ⚠ Warnings:")
        for w in config.warnings:
            lines.append(f"    - {w}")
    lines.append("═══════════════════════════════")
    return "\n".join(lines)
