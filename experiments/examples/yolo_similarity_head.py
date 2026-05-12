"""Example: Similarity detect head for YOLO models (Ultralytics backend).

Replaces YOLO's classification branch with cosine-similarity matching
using learned prototypes.  This is from Section 4.4 of the implementation plan.

Usage:
    cp experiments/examples/yolo_similarity_head.py experiments/modification.py
    uv run train_vision.py configs/example_unified_yolo_detect.yaml
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class SimilarityDetectHead(nn.Module):
    """Replace YOLO's classification branch with cosine-similarity matching.

    YOLO Detect heads have separate regression (.cv2) and classification (.cv3)
    branches per scale.  This module replaces the final conv in .cv3 with
    prototype matching.
    """

    def __init__(self, in_channels: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_channels))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W) from preceding conv layers
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        normed_x = F.normalize(x_flat, dim=-1)
        normed_p = F.normalize(self.prototypes, dim=-1)
        cls_scores = torch.matmul(normed_x, normed_p.T) / self.temperature.clamp(min=1e-4)
        return cls_scores.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, num_classes, H, W)


def modify_model(model: Any, config: dict[str, Any]) -> Any:
    """Replace YOLO classification branches with SimilarityDetectHead."""
    # Ultralytics YOLO: model.nn_module.model[-1] is the Detect head
    # The Detect head has .cv3 (classification convolutions per scale)
    nn_mod = model.nn_module

    # Find the detect head (last module in the sequential model)
    inner = getattr(nn_mod, "model", None)
    if inner is None:
        raise ValueError("Cannot find YOLO model.model sequential")

    detect = inner[-1]
    if not hasattr(detect, "cv3"):
        raise ValueError(
            f"Last module ({type(detect).__name__}) has no .cv3 attribute — "
            "not a standard YOLO Detect head"
        )

    num_classes = detect.nc

    # Replace each scale's classification branch
    for i, cv3_branch in enumerate(detect.cv3):
        # cv3_branch is a Sequential of Conv layers
        # The last Conv does the final classification
        last_conv = cv3_branch[-1]
        if hasattr(last_conv, "conv"):
            in_channels = last_conv.conv.in_channels
        elif hasattr(last_conv, "in_channels"):
            in_channels = last_conv.in_channels
        else:
            raise ValueError(f"Cannot detect in_channels for cv3[{i}][-1]")

        detect.cv3[i][-1] = SimilarityDetectHead(in_channels, num_classes)
        print(f"Replaced cv3[{i}][-1] with SimilarityDetectHead (in={in_channels}, nc={num_classes})")

    return model


def freeze_strategy(model: Any) -> None:
    """Freeze backbone + neck + bbox regression.  Train only prototype heads."""
    for name, param in model.nn_module.named_parameters():
        if "prototypes" in name or "temperature" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
