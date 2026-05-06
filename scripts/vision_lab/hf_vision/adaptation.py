"""Apply adaptation modes to a loaded HF vision model."""

from __future__ import annotations

import logging

import torch.nn as nn

from vision_lab.hf_vision.constants import ADAPTATION_MODE_CHOICES

logger = logging.getLogger(__name__)

_HEAD_NAME_HINTS = ("classifier", "head", "fc", "pre_logits", "lm_head")
_DETECT_HEAD_HINTS = ("class_embed", "bbox_embed", "input_proj", "decoder")


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def _apply_classify(model: nn.Module, mode: str) -> None:
    if mode == "full_finetune":
        _set_requires_grad(model, True)
        model.train()
        return

    if mode in ("freeze_backbone", "linear_probe"):
        for name, param in model.named_parameters():
            if any(h in name for h in _HEAD_NAME_HINTS) or name.endswith("classifier.weight") or name.endswith(
                "classifier.bias"
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        unfrozen = sum(1 for p in model.parameters() if p.requires_grad)
        if unfrozen == 0:
            logger.warning(
                "No parameters matched head hints for freeze_backbone/linear_probe; "
                "unfreezing final 'classifier' module if present."
            )
            if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
                _set_requires_grad(model.classifier, True)
        model.train()
        return

    if mode == "prompt_or_class_adapter":
        adapterish = 0
        for name, param in model.named_parameters():
            lower = name.lower()
            if any(
                k in lower
                for k in (
                    "adapter",
                    "prompt",
                    "prefix",
                    "lora",
                    "ia3",
                    "low_rank",
                )
            ):
                param.requires_grad = True
                adapterish += 1
            else:
                param.requires_grad = False
        if adapterish == 0:
            logger.warning(
                "prompt_or_class_adapter: no adapter/prompt-like params found; "
                "falling back to linear_probe-style head unfreezing."
            )
            _apply_classify(model, "linear_probe")
        else:
            model.train()
        return

    if mode in ("feature_extract_eval", "zero_shot_eval"):
        _set_requires_grad(model, False)
        model.eval()
        return

    raise RuntimeError(f"Unhandled adaptation_mode {mode!r} for classify")


def _apply_detect(model: nn.Module, mode: str) -> None:
    if mode == "full_finetune":
        _set_requires_grad(model, True)
        model.train()
        return

    if mode in ("freeze_backbone", "linear_probe"):
        for name, param in model.named_parameters():
            if any(h in name for h in _DETECT_HEAD_HINTS):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if sum(1 for p in model.parameters() if p.requires_grad) == 0:
            logger.warning("No detection head params matched; training all parameters.")
            _set_requires_grad(model, True)
        model.train()
        return

    if mode == "prompt_or_class_adapter":
        logger.info("prompt_or_class_adapter: using detect linear_probe-style unfreezing.")
        _apply_detect(model, "linear_probe")
        return

    if mode in ("feature_extract_eval", "zero_shot_eval"):
        _set_requires_grad(model, False)
        model.eval()
        return

    raise RuntimeError(f"Unhandled adaptation_mode {mode!r} for detect")


def _apply_segment(model: nn.Module, mode: str) -> None:
    """SAM / SAM2: ``linear_probe`` freezes vision + prompt encoders (mask decoder trains)."""

    if mode == "full_finetune":
        _set_requires_grad(model, True)
        model.train()
        return

    if mode in ("freeze_backbone", "linear_probe"):
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad = False
            else:
                param.requires_grad = True
        model.train()
        return

    if mode == "prompt_or_class_adapter":
        for name, param in model.named_parameters():
            lower = name.lower()
            if any(k in lower for k in ("adapter", "prompt", "prefix", "lora", "ia3", "low_rank")):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if sum(1 for p in model.parameters() if p.requires_grad) == 0:
            _apply_segment(model, "linear_probe")
        else:
            model.train()
        return

    if mode in ("feature_extract_eval", "zero_shot_eval"):
        _set_requires_grad(model, False)
        model.eval()
        return

    raise RuntimeError(f"Unhandled adaptation_mode {mode!r} for segment")


def apply_adaptation_mode(
    model: nn.Module,
    adaptation_mode: str,
    *,
    architecture: str = "classify",
) -> None:
    """
    Configure ``requires_grad`` / ``train()`` according to ``adaptation_mode``.

    ``architecture`` selects head naming: ``classify`` (ViT-style), ``detect`` (DETR-style),
    ``segment`` (SAM/SAM2 encoders vs mask path).
    """
    mode = adaptation_mode.strip()
    if mode not in ADAPTATION_MODE_CHOICES:
        raise ValueError(
            f"Unknown adaptation_mode {adaptation_mode!r}; expected one of "
            f"{sorted(ADAPTATION_MODE_CHOICES)}."
        )
    arch = architecture.strip().lower()
    if arch == "detect":
        _apply_detect(model, mode)
        return
    if arch in ("segment", "semantic_segment"):
        _apply_segment(model, mode)
        return
    if arch == "classify":
        _apply_classify(model, mode)
        return

    raise ValueError(
        f"Unknown architecture {architecture!r} (expected classify, detect, segment, semantic_segment)."
    )
