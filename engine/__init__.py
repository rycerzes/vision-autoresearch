"""Autonomous Vision Research Lab — engine package.

The engine auto-infers training/eval pipelines from model metadata and
dataset features. Two backends: HF Transformers and Ultralytics.
"""

from __future__ import annotations

from engine.backend import detect_backend, load_model
from engine.introspection import HEAD_CATEGORIES, head_category_from_arch
from engine.unified_model import UnifiedModel

__all__ = [
    "HEAD_CATEGORIES",
    "UnifiedModel",
    "detect_backend",
    "head_category_from_arch",
    "load_model",
]
