"""Surgery utilities for architecture modification on any nn.Module.

Works on both HF Transformers and Ultralytics models — they are all
``nn.Module`` underneath.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from engine.unified_model import ModuleInfo

logger = logging.getLogger(__name__)


# Module graph

def get_module_graph(model: nn.Module) -> dict[str, ModuleInfo]:
    """Return ``{dotted_path: ModuleInfo}`` for every sub-module."""
    return {
        name: ModuleInfo.from_module(mod)
        for name, mod in model.named_modules()
        if name  # skip root ""
    }


# Module replacement

def _resolve_parent_and_attr(
    root: nn.Module, path: str
) -> tuple[nn.Module, str]:
    """Walk *path* (dot-separated, integer indices for Sequential) and return
    ``(parent, last_attr_or_index)``."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def replace_module(model: nn.Module, path: str, new_module: nn.Module) -> None:
    """Swap the module at *path* inside *model*."""
    parent, attr = _resolve_parent_and_attr(model, path)
    if attr.isdigit():
        parent[int(attr)] = new_module  # type: ignore[index]
    else:
        setattr(parent, attr, new_module)
    logger.info("Replaced module at %s with %s", path, type(new_module).__name__)


# Freeze / unfreeze

def freeze_except(model: nn.Module, keep_paths: list[str]) -> tuple[int, int]:
    """Freeze all parameters except those whose name starts with any of
    *keep_paths*.  Returns ``(frozen_count, trainable_count)``."""
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in keep_paths):
            param.requires_grad_(True)
            trainable += 1
        else:
            param.requires_grad_(False)
            frozen += 1
    logger.info(
        "freeze_except: %d frozen, %d trainable (paths=%s)",
        frozen,
        trainable,
        keep_paths,
    )
    return frozen, trainable


def freeze_all(model: nn.Module) -> int:
    """Freeze every parameter.  Returns count."""
    n = 0
    for param in model.parameters():
        param.requires_grad_(False)
        n += 1
    return n


def unfreeze_all(model: nn.Module) -> int:
    """Unfreeze every parameter.  Returns count."""
    n = 0
    for param in model.parameters():
        param.requires_grad_(True)
        n += 1
    return n


# Module role detection (heuristic)

def find_module_by_role(
    model: nn.Module,
    role: str,
    *,
    num_labels: int | None = None,
    model_config: Any | None = None,
) -> tuple[str, nn.Module]:
    """Locate a module by its functional role using heuristics.

    Supported roles:
        ``classification_head`` – Linear whose ``out_features == num_labels``
        ``bbox_head``           – Linear whose ``out_features`` is 4 (or multiple)
        ``backbone``            – Largest sub-module by parameter count
        ``neck``                – Sub-module with 'neck' / 'fpn' / 'pan' in name
        ``detect_head``         – Last top-level module (YOLO convention)

    Raises ``ValueError`` when the role cannot be resolved.
    """
    if num_labels is None and model_config is not None:
        num_labels = getattr(model_config, "num_labels", None) or getattr(
            model_config, "num_classes", None
        )

    if role == "classification_head":
        return _find_classification_head(model, num_labels)
    if role == "bbox_head":
        return _find_bbox_head(model)
    if role == "backbone":
        return _find_backbone(model)
    if role == "neck":
        return _find_by_name_hint(model, ("neck", "fpn", "pan", "bifpn"))
    if role == "detect_head":
        return _find_detect_head(model)
    raise ValueError(f"Unknown module role: {role!r}")



def _find_classification_head(
    model: nn.Module, num_labels: int | None
) -> tuple[str, nn.Module]:
    # Strategy 1: Linear with out_features == num_labels
    if num_labels is not None:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.out_features == num_labels:
                return name, mod

    # Strategy 2: last module named *class* / *cls* / *head* / *classifier*
    hints = ("class_embed", "classifier", "cls_head", "class_head", "fc", "head")
    candidates: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        name_lower = name.lower()
        if any(h in name_lower for h in hints):
            candidates.append((name, mod))
    if candidates:
        return candidates[-1]

    # Strategy 3: last Linear layer overall
    last_linear: tuple[str, nn.Module] | None = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_linear = (name, mod)
    if last_linear is not None:
        return last_linear

    raise ValueError("Could not find classification_head")


def _find_bbox_head(model: nn.Module) -> tuple[str, nn.Module]:
    hints = ("bbox_embed", "bbox_head", "bbox_pred", "reg_head", "box_head")
    for name, mod in model.named_modules():
        name_lower = name.lower()
        if any(h in name_lower for h in hints):
            return name, mod
    # Fallback: Linear with out_features == 4
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features == 4:
            return name, mod
    raise ValueError("Could not find bbox_head")


def _find_backbone(model: nn.Module) -> tuple[str, nn.Module]:
    # Pick the top-level child with the most parameters.
    best_name = ""
    best_mod: nn.Module | None = None
    best_params = 0
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        if n > best_params:
            best_name, best_mod, best_params = name, mod, n
    if best_mod is not None:
        return best_name, best_mod
    raise ValueError("Could not find backbone (model has no children?)")


def _find_by_name_hint(
    model: nn.Module, hints: tuple[str, ...]
) -> tuple[str, nn.Module]:
    for name, mod in model.named_modules():
        name_lower = name.lower()
        if any(h in name_lower for h in hints):
            return name, mod
    raise ValueError(f"Could not find module matching hints: {hints}")


def _find_detect_head(model: nn.Module) -> tuple[str, nn.Module]:
    """YOLO convention: last element of ``model.model`` Sequential."""
    # Try Ultralytics layout: model.model is Sequential, last is Detect/Segment
    inner = getattr(model, "model", None)
    if inner is not None and isinstance(inner, nn.Sequential):
        last_idx = len(inner) - 1
        return f"model.{last_idx}", inner[last_idx]
    # Fallback: last named child
    children = list(model.named_children())
    if children:
        return children[-1]
    raise ValueError("Could not find detect_head")
