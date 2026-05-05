"""Training YAML alignment checks for preflight (task, promotion, model backend)."""

from __future__ import annotations

from typing import Any

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

    return errors, warnings
