"""Example: Cosine template head swap for HF detection models (DETR, D-FINE, Relation DETR).

Replaces the standard classification head with cosine-similarity template
matching.  This is from Section 4.3 of the implementation plan.

Usage:
    cp experiments/examples/cosine_template_head.py experiments/modification.py
    uv run train_vision.py configs/example_unified_research.yaml
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class CosineTemplateHead(nn.Module):
    """Classification via cosine similarity to learned class templates.

    Instead of a linear projection ``Wx + b``, this computes:
        logits = cos(decoder_output, templates) / temperature

    Benefits:
    - Better calibrated confidence scores
    - More interpretable (templates ARE class prototypes)
    - Potential for open-vocabulary extension (replace templates with text embeddings)
    """

    def __init__(self, hidden_dim: int, num_classes: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.templates = nn.Parameter(torch.randn(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.templates)

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(decoder_output, dim=-1)
        normed_t = F.normalize(self.templates, dim=-1)
        return torch.matmul(normed, normed_t.T) / self.temperature.clamp(min=1e-4)


def modify_model(model: Any, config: dict[str, Any]) -> Any:
    """Replace classification head with CosineTemplateHead."""
    path, cls_head = model.find_module_by_role("classification_head")

    # Detect dimensions from the existing head
    if hasattr(cls_head, "in_features"):
        hidden_dim = cls_head.in_features
        num_classes = cls_head.out_features
    elif hasattr(cls_head, "in_channels"):
        hidden_dim = cls_head.in_channels
        num_classes = cls_head.out_channels
    else:
        raise ValueError(
            f"Cannot detect dimensions of classification head at '{path}' "
            f"(type={type(cls_head).__name__})"
        )

    # Use config num_classes if available (dataset may differ from pretrained)
    if config.get("num_classes") is not None:
        num_classes = config["num_classes"]

    new_head = CosineTemplateHead(hidden_dim, num_classes)
    model.replace_module(path, new_head)

    print(f"Replaced {type(cls_head).__name__} at '{path}' with CosineTemplateHead")
    print(f"  hidden_dim={hidden_dim}, num_classes={num_classes}")
    return model


def freeze_strategy(model: Any) -> None:
    """Only train the template embeddings — preserve spatial reasoning."""
    model.freeze_except(["templates", "temperature"])
