"""Dynamic collate_fn builder for HF Trainer.

Builds a collation function from the processor output keys and head category.
Handles variable-length targets (detection), fixed-shape tensors
(classification, segmentation), and mixed-type batches.

Ultralytics does NOT need this — it has built-in collation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def build_collate_fn(
    head_category: str,
    processor: Any,
    *,
    column_map: dict[str, str] | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Return a collate function tailored to the model's head category.

    The returned function accepts a list of per-example dicts (as produced
    by the dataset transform) and returns a batched dict ready for
    ``model.forward()``.

    Parameters
    ----------
    head_category:
        From ``model.head_category`` (e.g., ``"detection"``, ``"classification"``).
    processor:
        HF processor — used to determine expected input keys.
    column_map:
        Resolved column map (``{"image": "col", "target": "col"}``).
    """
    if head_category == "detection":
        return _collate_detection
    if head_category in ("classification", "contrastive"):
        return _collate_classification
    if head_category in ("dense_classification", "dense_regression"):
        return _collate_dense
    if head_category == "prompted_segmentation":
        return _collate_prompted_segmentation
    if head_category == "structured_detection":
        return _collate_detection  # same variable-length handling
    if head_category in ("sequence_generation",):
        return _collate_sequence
    if head_category == "self_supervised":
        return _collate_self_supervised
    if head_category in ("image_reconstruction", "pair_matching"):
        return _collate_paired_images

    # Generic fallback
    logger.warning(
        "No specialised collation for head_category=%r, using generic",
        head_category,
    )
    return _collate_generic


# ── Collation implementations ──────────────────────────────────


def _collate_classification(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack pixel_values + labels (both fixed-shape)."""
    result: dict[str, Any] = {}
    if not batch:
        return result

    # Stack all tensor keys
    for key in batch[0]:
        vals = [ex[key] for ex in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result


def _collate_detection(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack pixel_values, keep labels as a list of dicts (variable-length).

    Detection targets have variable numbers of boxes per image, so they
    can't be stacked into a single tensor.  HF DETR-family models accept
    ``labels`` as ``list[dict]``.

    The key ``"labels"`` is our internal convention set by
    ``UnifiedDataset.for_hf()`` — not a dataset assumption.
    """
    result: dict[str, Any] = {}
    if not batch:
        return result

    for key in batch[0]:
        vals = [ex[key] for ex in batch]

        # Labels: keep as list (variable-length per image)
        # This is the key set by our own for_hf() transform, not
        # a hardcoded dataset column name.
        if key == "labels":
            result[key] = vals
            continue

        if isinstance(vals[0], torch.Tensor):
            # Try stacking, fall back to list if shapes differ
            try:
                result[key] = torch.stack(vals)
            except RuntimeError:
                result[key] = vals
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result


def _collate_dense(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack pixel_values + label masks (both fixed-shape after resize)."""
    result: dict[str, Any] = {}
    if not batch:
        return result

    for key in batch[0]:
        vals = [ex[key] for ex in batch]
        if isinstance(vals[0], torch.Tensor):
            try:
                result[key] = torch.stack(vals)
            except RuntimeError:
                # Segmentation masks may have inconsistent sizes if transforms
                # didn't resize them — pad to max
                result[key] = _pad_and_stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result


def _collate_prompted_segmentation(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """SAM-family: pixel_values + input_points/input_boxes + ground_truth_masks."""
    result: dict[str, Any] = {}
    if not batch:
        return result

    for key in batch[0]:
        vals = [ex[key] for ex in batch]
        if isinstance(vals[0], torch.Tensor):
            try:
                result[key] = torch.stack(vals)
            except RuntimeError:
                result[key] = vals  # variable prompts
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result


def _collate_sequence(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pad token sequences to max length in batch."""
    result: dict[str, Any] = {}
    if not batch:
        return result

    for key in batch[0]:
        vals = [ex[key] for ex in batch]
        if isinstance(vals[0], torch.Tensor):
            if vals[0].dim() >= 1 and any(v.shape != vals[0].shape for v in vals):
                # Variable-length sequences → pad
                result[key] = _pad_sequences(vals)
            else:
                result[key] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result


def _collate_self_supervised(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stack pixel_values + bool_masked_pos (MAE/BEiT)."""
    return _collate_classification(batch)  # same fixed-shape stacking


def _collate_paired_images(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Image-to-image / pair matching: stack both input and target images."""
    return _collate_classification(batch)


def _collate_generic(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Best-effort: stack tensors, list everything else."""
    result: dict[str, Any] = {}
    if not batch:
        return result

    for key in batch[0]:
        vals = [ex[key] for ex in batch]
        if isinstance(vals[0], torch.Tensor):
            try:
                result[key] = torch.stack(vals)
            except RuntimeError:
                result[key] = vals
        elif isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals
    return result


# ── Helpers ─────────────────────────────────────────────────────


def _pad_and_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Pad 2D/3D tensors to the largest spatial dims, then stack."""
    if not tensors:
        return torch.tensor([])

    ndim = tensors[0].dim()
    if ndim < 2:
        return torch.stack(tensors)

    # Find max size per spatial dimension
    max_sizes = list(tensors[0].shape)
    for t in tensors[1:]:
        for i in range(len(max_sizes)):
            max_sizes[i] = max(max_sizes[i], t.shape[i])

    padded: list[torch.Tensor] = []
    for t in tensors:
        # Compute padding (F.pad takes reversed dim order)
        pad: list[int] = []
        for i in range(ndim - 1, -1, -1):
            pad.extend([0, max_sizes[i] - t.shape[i]])
        padded.append(torch.nn.functional.pad(t, pad, value=0))

    return torch.stack(padded)


def _pad_sequences(
    tensors: list[torch.Tensor],
    pad_value: int = 0,
) -> torch.Tensor:
    """Pad 1D sequences to max length, then stack."""
    max_len = max(t.shape[0] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[0] < max_len:
            padding = torch.full(
                (max_len - t.shape[0],) + t.shape[1:],
                pad_value,
                dtype=t.dtype,
                device=t.device,
            )
            padded.append(torch.cat([t, padding]))
        else:
            padded.append(t)
    return torch.stack(padded)
