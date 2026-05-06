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

# ``task_type`` values accepted by ``train_hf_vision.py``.
ROUTED_TASK_IDS: frozenset[str] = frozenset({"classify", "detect", "segment"})

# Tasks built via ``vision_lab.hf_vision.loaders.load_hf_vision_model``.
TASKS_USING_SHARED_MODEL_LOADER: frozenset[str] = ROUTED_TASK_IDS

# Loader choices are task-scoped. Representation loaders currently have a real
# probe head and standard metric only for classification.
MODEL_LOADER_CHOICES_BY_TASK: dict[str, frozenset[str]] = {
    "classify": MODEL_LOADER_CHOICES,
    "detect": frozenset({"auto_task_head"}),
    "segment": frozenset({"auto_task_head"}),
}

# Alias used by ``loaders.py`` and ``train_hf_vision.py``.
HF_VISION_SUPPORTED_TASKS = TASKS_USING_SHARED_MODEL_LOADER
