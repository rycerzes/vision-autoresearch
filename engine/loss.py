"""Loss auto-detection and external loss functions.

Probes the model with a dummy forward pass to determine if it computes its
own loss.  When it doesn't, selects a loss function from the head category.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Loss mode detection

def probe_loss_mode(
    model: nn.Module,
    processor: Any,
    head_category: str,
) -> str:
    """Probe whether the model computes loss in its ``forward()``.

    Returns ``"builtin"`` or ``"external"``.
    Works on CPU — no GPU required.
    """
    model.eval()
    device = next(model.parameters()).device

    try:
        dummy_image = torch.rand(3, 224, 224)
        inputs = processor(images=dummy_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        dummy_labels = _make_dummy_labels(head_category, model, device)
        if dummy_labels is not None:
            inputs["labels"] = dummy_labels

        with torch.no_grad():
            outputs = model(**inputs)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            logger.info("Loss mode: builtin (model.forward returns loss)")
            return "builtin"
        else:
            logger.info("Loss mode: external (model.forward does not return loss)")
            return "external"

    except Exception as e:
        logger.warning("Loss probe failed (%s), defaulting to external", e)
        return "external"


def _make_dummy_labels(
    head_category: str, model: nn.Module, device: torch.device
) -> Any:
    """Create minimal dummy labels for probing."""
    config = getattr(model, "config", None)
    num_labels = getattr(config, "num_labels", 2) if config else 2

    if head_category == "classification":
        return torch.tensor([0], device=device)

    if head_category == "dense_classification":
        return torch.zeros(1, 224, 224, dtype=torch.long, device=device)

    if head_category == "dense_regression":
        return torch.zeros(1, 1, 224, 224, dtype=torch.float32, device=device)

    if head_category == "detection":
        return [
            {
                "class_labels": torch.tensor([0], device=device),
                "boxes": torch.tensor([[0.1, 0.1, 0.5, 0.5]], device=device),
            }
        ]

    if head_category in ("sequence_generation",):
        return torch.tensor([[0, 1, 2]], device=device)

    if head_category == "self_supervised":
        # MAE/BEiT: need bool_masked_pos, not labels
        return None

    # Contrastive, image_reconstruction, etc. — skip labels probe
    return None


# External loss functions (for models whose forward() doesn't return loss)

def cross_entropy_on_logits(outputs: Any, labels: torch.Tensor, **kw: Any) -> torch.Tensor:
    """Standard classification cross-entropy."""
    logits = outputs.logits
    if logits.dim() == 2:
        return F.cross_entropy(logits, labels)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


def pixel_cross_entropy(outputs: Any, labels: torch.Tensor, **kw: Any) -> torch.Tensor:
    """Per-pixel cross-entropy for semantic segmentation."""
    logits = outputs.logits
    # logits: (B, C, H, W), labels: (B, H, W)
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return F.cross_entropy(logits, labels, ignore_index=255)


def silog_loss(outputs: Any, labels: torch.Tensor, **kw: Any) -> torch.Tensor:
    """Scale-invariant log loss for depth estimation."""
    predicted = outputs.predicted_depth
    if predicted.shape != labels.shape:
        predicted = F.interpolate(
            predicted.unsqueeze(1), size=labels.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

    valid = labels > 0
    if not valid.any():
        return torch.tensor(0.0, device=predicted.device, requires_grad=True)

    log_pred = torch.log(predicted[valid].clamp(min=1e-8))
    log_gt = torch.log(labels[valid].clamp(min=1e-8))
    diff = log_pred - log_gt
    return torch.sqrt((diff ** 2).mean() - 0.5 * (diff.mean() ** 2))


def info_nce_loss(outputs: Any, labels: torch.Tensor | None = None, **kw: Any) -> torch.Tensor:
    """Contrastive InfoNCE loss for CLIP-style models."""
    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text
    batch_size = logits_per_image.shape[0]
    targets = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return (loss_i + loss_t) / 2


def l1_reconstruction_loss(outputs: Any, labels: torch.Tensor, **kw: Any) -> torch.Tensor:
    """L1 pixel loss for image-to-image models."""
    reconstruction = outputs.reconstruction
    return F.l1_loss(reconstruction, labels)


def mse_masked_loss(outputs: Any, labels: torch.Tensor | None = None, **kw: Any) -> torch.Tensor:
    """MSE on masked patches for MAE-style self-supervised models."""
    logits = outputs.logits
    # MAE models typically return loss when bool_masked_pos is passed
    # This is a fallback
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    return logits.mean() * 0  # placeholder


def dice_ce_loss(outputs: Any, labels: torch.Tensor, **kw: Any) -> torch.Tensor:
    """Dice + CE loss for prompted segmentation (SAM family)."""
    predicted = outputs.pred_masks
    if predicted.dim() == 4:
        predicted = predicted.squeeze(1)
    labels_float = labels.float()
    if labels_float.dim() == 4:
        labels_float = labels_float.squeeze(1)
    if predicted.shape[-2:] != labels_float.shape[-2:]:
        predicted = F.interpolate(predicted.unsqueeze(1), size=labels_float.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    bce = F.binary_cross_entropy_with_logits(predicted, labels_float)
    pred_sigmoid = torch.sigmoid(predicted)
    intersection = (pred_sigmoid * labels_float).sum(dim=(-2, -1))
    union = pred_sigmoid.sum(dim=(-2, -1)) + labels_float.sum(dim=(-2, -1))
    dice = 1 - (2 * intersection + 1e-8) / (union + 1e-8)
    return bce + dice.mean()


def ce_on_decoder_logits(outputs: Any, labels: torch.Tensor, **kw: Any) -> torch.Tensor:
    """Cross-entropy on decoder outputs for sequence generation (OCR, table)."""
    logits = outputs.logits
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


# Registry

EXTERNAL_LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "classification": cross_entropy_on_logits,
    "dense_classification": pixel_cross_entropy,
    "dense_regression": silog_loss,
    "contrastive": info_nce_loss,
    "image_reconstruction": l1_reconstruction_loss,
    "self_supervised": mse_masked_loss,
    "sequence_generation": ce_on_decoder_logits,
    "prompted_segmentation": dice_ce_loss,
}


def get_external_loss(head_category: str) -> Callable[..., torch.Tensor] | None:
    """Return the external loss function for a head category, or None."""
    return EXTERNAL_LOSSES.get(head_category)
