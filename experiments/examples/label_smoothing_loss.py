"""Example: Label smoothing loss for classification models.

Works on any HF classification model (ViT, DeiT, Swin, ConvNeXt, etc.).

Usage:
    cp experiments/examples/label_smoothing_loss.py experiments/modification.py
    uv run train_vision.py configs/example_unified_hf_classify.yaml
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any


def modify_loss(outputs: Any, labels: Any) -> torch.Tensor:
    """Label smoothing cross-entropy loss.

    Replaces hard one-hot targets with smoothed distribution:
        target = (1 - smooth) * one_hot + smooth / num_classes
    """
    smooth = 0.1
    logits = outputs.logits
    num_classes = logits.size(-1)

    log_probs = F.log_softmax(logits.view(-1, num_classes), dim=-1)
    targets = labels.view(-1)

    nll_loss = F.nll_loss(log_probs, targets, reduction="mean")
    smooth_loss = -log_probs.mean(dim=-1).mean()

    return (1.0 - smooth) * nll_loss + smooth * smooth_loss
