"""Allowed model loader ids and adaptation modes for ``train_hf_vision``."""

from __future__ import annotations

# Hugging Face vision runner: how weights are instantiated before adaptation.
MODEL_LOADER_CHOICES: frozenset[str] = frozenset(
    {
        "auto_task_head",
        "auto_model",
        "auto_backbone",
    }
)

# Post-load training / eval posture (orthogonal to ``TrainingArguments.do_train``).
ADAPTATION_MODE_CHOICES: frozenset[str] = frozenset(
    {
        "full_finetune",
        "freeze_backbone",
        "linear_probe",
        "feature_extract_eval",
        "zero_shot_eval",
        "prompt_or_class_adapter",
    }
)

# Tasks routed through ``train_hf_vision.py`` at the repo root.
ROUTED_TASK_IDS: frozenset[str] = frozenset({"classify", "detect", "segment"})

# Tasks whose weights are loaded through ``vision_lab.hf_vision.loaders`` (shared runner classify path).
HF_VISION_SUPPORTED_TASKS: frozenset[str] = frozenset({"classify"})
