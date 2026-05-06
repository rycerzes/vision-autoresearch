"""Shared Hugging Face vision runner building blocks."""

from vision_lab.hf_vision.adaptation import apply_adaptation_mode
from vision_lab.hf_vision.constants import (
    ADAPTATION_MODE_CHOICES,
    HF_VISION_SUPPORTED_TASKS,
    MODEL_LOADER_CHOICES,
    MODEL_LOADER_CHOICES_BY_TASK,
    ROUTED_TASK_IDS,
    TASKS_USING_SHARED_MODEL_LOADER,
)
from vision_lab.hf_vision.loaders import load_hf_vision_model
from vision_lab.hf_vision.transforms import build_transforms

__all__ = [
    "ADAPTATION_MODE_CHOICES",
    "HF_VISION_SUPPORTED_TASKS",
    "MODEL_LOADER_CHOICES",
    "MODEL_LOADER_CHOICES_BY_TASK",
    "ROUTED_TASK_IDS",
    "TASKS_USING_SHARED_MODEL_LOADER",
    "apply_adaptation_mode",
    "build_transforms",
    "load_hf_vision_model",
]
