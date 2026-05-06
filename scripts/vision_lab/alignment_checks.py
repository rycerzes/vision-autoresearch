"""Training YAML alignment checks for preflight (task, promotion, model backend)."""

from __future__ import annotations

from typing import Any

from vision_lab.hf_vision.constants import ADAPTATION_MODE_CHOICES, MODEL_LOADER_CHOICES
from vision_lab.promotion import load_promotion_policy
from vision_lab.task_registry import TASK_BY_ID


def verify_task_type_field(cfg: dict[str, Any], task: str) -> list[str]:
    tt = cfg.get("task_type")
    if tt is None:
        return []
    if str(tt).strip() != task:
        return [f"config task_type={tt!r} disagrees with runner task={task!r}"]
    return []


def verify_promotion_block(cfg: dict[str, Any], task: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        load_promotion_policy(cfg, task_id=task)
    except ValueError as e:
        errors.append(f"promotion config: {e}")
    return errors, warnings


def verify_model_backend(cfg: dict[str, Any], task: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    spec = TASK_BY_ID.get(task)
    if not spec:
        return errors, warnings

    raw = cfg.get("model_name_or_path") or cfg.get("model") or ""
    if not isinstance(raw, str):
        return errors, warnings
    model = raw.strip()
    if not model:
        warnings.append("model_name_or_path is empty; training may fail at runtime.")
        return errors, warnings

    lower = model.lower()
    is_weight_file = lower.endswith(".pt") or lower.endswith(".pth") or lower.endswith(".onnx")

    if spec.backend == "ultralytics":
        if "/" in model and not is_weight_file:
            warnings.append(
                f"Ultralytics task {task}: model {model!r} looks like a Hugging Face-style repo id "
                "(ensure Ultralytics accepts this checkpoint)."
            )
    elif spec.backend == "transformers":
        if is_weight_file:
            errors.append(
                f"Transformers task {task}: model {model!r} looks like a raw weight file; "
                "use a Hugging Face model id or a Transformers saved-model directory."
            )
    return errors, warnings


def verify_hf_vision_yaml(cfg: dict[str, Any], task: str) -> tuple[list[str], list[str]]:
    """Validate ``model_loader`` / ``adaptation_mode`` for tasks routed to ``train_hf_vision.py``."""
    errors: list[str] = []
    warnings: list[str] = []
    spec = TASK_BY_ID.get(task)
    if not spec or spec.train_script != "train_hf_vision.py":
        return errors, warnings

    ml = cfg.get("model_loader", "auto_task_head")
    if isinstance(ml, str):
        ml = ml.strip()
        if ml and ml not in MODEL_LOADER_CHOICES:
            errors.append(
                f"model_loader={ml!r} is not supported (expected one of {sorted(MODEL_LOADER_CHOICES)})."
            )
    elif ml is not None:
        errors.append(f"model_loader must be a string, got {type(ml).__name__}.")

    mode = cfg.get("adaptation_mode", "full_finetune")
    if isinstance(mode, str):
        mode = mode.strip()
        if mode and mode not in ADAPTATION_MODE_CHOICES:
            errors.append(
                f"adaptation_mode={mode!r} is not supported "
                f"(expected one of {sorted(ADAPTATION_MODE_CHOICES)})."
            )
    elif mode is not None:
        errors.append(f"adaptation_mode must be a string, got {type(mode).__name__}.")

    return errors, warnings


def collect_alignment_issues(cfg: dict[str, Any], task: str) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for task ↔ YAML fields (excluding dataset paths)."""
    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(verify_task_type_field(cfg, task))

    e, w = verify_promotion_block(cfg, task)
    errors.extend(e)
    warnings.extend(w)

    e2, w2 = verify_model_backend(cfg, task)
    errors.extend(e2)
    warnings.extend(w2)

    e3, w3 = verify_hf_vision_yaml(cfg, task)
    errors.extend(e3)
    warnings.extend(w3)

    return errors, warnings
