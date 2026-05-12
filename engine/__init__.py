"""Autonomous Vision Research Lab — engine package.

The engine auto-infers training/eval pipelines from model metadata and
dataset features. Two backends: HF Transformers and Ultralytics.
"""

from __future__ import annotations

from engine.augmentation import (
    AugmentationConfig,
    AugmentationFamily,
    build_eval_augmentation,
    build_train_augmentation,
    infer_augmentation_family,
)
from engine.backend import detect_backend, load_model
from engine.collation import build_collate_fn
from engine.introspection import HEAD_CATEGORIES, head_category_from_arch
from engine.metrics import HEAD_METRICS, default_promotion_metric, get_direction
from engine.pipeline import PipelineConfig, auto_infer_pipeline, summarize_pipeline
from engine.preprocessing import (
    PreprocessingConfig,
    discover_preprocessing,
    discover_ultralytics_preprocessing,
)
from engine.training import (
    UniversalTrainingArgs,
    emit_summary,
    parse_config_yaml,
    to_hf_training_args,
    to_ultralytics_train_kwargs,
)
from engine.unified_model import ModuleInfo, UnifiedModel

# Lazy imports for research module (avoid torch import at package level
# when only training utilities are needed)
def _lazy_research():
    from engine import research as _r
    return _r

__all__ = [
    # Backend detection & loading
    "detect_backend",
    "load_model",
    # Introspection
    "HEAD_CATEGORIES",
    "head_category_from_arch",
    # Unified model protocol
    "UnifiedModel",
    "ModuleInfo",
    # Preprocessing
    "PreprocessingConfig",
    "discover_preprocessing",
    "discover_ultralytics_preprocessing",
    # Collation
    "build_collate_fn",
    # Augmentation
    "AugmentationConfig",
    "AugmentationFamily",
    "build_train_augmentation",
    "build_eval_augmentation",
    "infer_augmentation_family",
    # Metrics
    "HEAD_METRICS",
    "default_promotion_metric",
    "get_direction",
    # Pipeline
    "PipelineConfig",
    "auto_infer_pipeline",
    "summarize_pipeline",
    # Training
    "UniversalTrainingArgs",
    "parse_config_yaml",
    "to_hf_training_args",
    "to_ultralytics_train_kwargs",
    "emit_summary",
    # Research (use engine.research directly for full API)
]
