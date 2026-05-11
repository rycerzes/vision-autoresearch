"""Processor / preprocessing discovery.

Auto-detects image size, normalization parameters, and input requirements
from HF processors and Ultralytics model configs.  No hardcoded values —
everything is read from the processor or model metadata at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Everything needed to preprocess inputs for a model."""

    # Image size: (height, width) or single int for square
    image_size: tuple[int, int] = (224, 224)
    # Normalization: mean/std per channel (RGB order)
    image_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    # Whether the processor handles resizing internally
    processor_handles_resize: bool = True
    # Whether the processor handles normalization internally
    processor_handles_normalize: bool = True
    # Required input modalities
    modalities: list[str] = field(default_factory=lambda: ["image"])
    # Whether a tokenizer is available (multimodal models)
    has_tokenizer: bool = False
    # Pixel value range the model expects (0-1 vs 0-255)
    rescale_factor: float | None = None
    # Whether padding is needed (variable-size inputs)
    do_pad: bool = False
    pad_size: tuple[int, int] | None = None


def discover_preprocessing(
    processor: Any,
    *,
    image_size_override: int | tuple[int, int] | None = None,
) -> PreprocessingConfig:
    """Extract preprocessing config from an HF processor.

    Reads all parameters from the processor object — no hardcoded defaults.
    Falls back to sensible values only when the processor lacks metadata.
    """
    config = PreprocessingConfig()

    # ── Image size ──────────────────────────────────────────────
    if image_size_override is not None:
        if isinstance(image_size_override, int):
            config.image_size = (image_size_override, image_size_override)
        else:
            config.image_size = tuple(image_size_override)  # type: ignore[arg-type]
    else:
        config.image_size = _extract_image_size(processor)

    # ── Normalization ───────────────────────────────────────────
    img_proc = _get_image_processor(processor)
    if img_proc is not None:
        mean = getattr(img_proc, "image_mean", None)
        std = getattr(img_proc, "image_std", None)
        if mean is not None:
            config.image_mean = tuple(mean)
        if std is not None:
            config.image_std = tuple(std)

        config.processor_handles_normalize = getattr(
            img_proc, "do_normalize", True
        )
        config.processor_handles_resize = getattr(
            img_proc, "do_resize", True
        )

        rescale = getattr(img_proc, "rescale_factor", None)
        if rescale is not None:
            config.rescale_factor = float(rescale)

        config.do_pad = getattr(img_proc, "do_pad", False)
        pad_size = getattr(img_proc, "pad_size", None)
        if pad_size is not None:
            if isinstance(pad_size, dict):
                config.pad_size = (
                    pad_size.get("height", config.image_size[0]),
                    pad_size.get("width", config.image_size[1]),
                )
            elif isinstance(pad_size, (list, tuple)) and len(pad_size) == 2:
                config.pad_size = tuple(pad_size)  # type: ignore[arg-type]

    # ── Modalities ──────────────────────────────────────────────
    config.modalities = _detect_modalities(processor)
    config.has_tokenizer = hasattr(processor, "tokenizer") and processor.tokenizer is not None

    logger.info(
        "Preprocessing: size=%s mean=%s std=%s modalities=%s tokenizer=%s",
        config.image_size,
        config.image_mean,
        config.image_std,
        config.modalities,
        config.has_tokenizer,
    )
    return config


def discover_ultralytics_preprocessing(
    model: Any,
    *,
    image_size_override: int | None = None,
) -> PreprocessingConfig:
    """Extract preprocessing config from an Ultralytics model.

    Ultralytics models always: normalize to 0-1 float32, use standard RGB,
    and handle preprocessing internally.  Image size comes from the model's
    ``overrides`` or defaults to 640.
    """
    # Resolve image size from model overrides
    overrides = getattr(model, "overrides", {}) or {}
    imgsz = image_size_override or overrides.get("imgsz", 640)
    if isinstance(imgsz, int):
        size = (imgsz, imgsz)
    elif isinstance(imgsz, (list, tuple)):
        size = (imgsz[0], imgsz[-1])
    else:
        size = (640, 640)

    return PreprocessingConfig(
        image_size=size,
        image_mean=(0.0, 0.0, 0.0),  # Ultralytics normalizes to 0-1 internally
        image_std=(1.0, 1.0, 1.0),
        processor_handles_resize=True,
        processor_handles_normalize=True,
        modalities=["image"],
        has_tokenizer=False,
        rescale_factor=1.0 / 255.0,
    )


# ── Internal helpers ────────────────────────────────────────────


def _get_image_processor(processor: Any) -> Any | None:
    """Unwrap an AutoProcessor to get the underlying image processor."""
    # AutoProcessor wraps image_processor + tokenizer
    if hasattr(processor, "image_processor"):
        return processor.image_processor
    # Bare image processor
    if hasattr(processor, "do_resize") or hasattr(processor, "image_mean"):
        return processor
    return None


def _extract_image_size(processor: Any) -> tuple[int, int]:
    """Extract target image size from a processor."""
    img_proc = _get_image_processor(processor)
    if img_proc is None:
        return (224, 224)

    # Try .size attribute (dict or int)
    size = getattr(img_proc, "size", None)
    if size is not None:
        if isinstance(size, dict):
            h = size.get("height") or size.get("shortest_edge") or size.get("longest_edge", 224)
            w = size.get("width") or size.get("shortest_edge") or size.get("longest_edge", 224)
            return (int(h), int(w))
        if isinstance(size, int):
            return (size, size)
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return (int(size[0]), int(size[1]))

    # Try .crop_size (some processors resize then crop)
    crop_size = getattr(img_proc, "crop_size", None)
    if crop_size is not None:
        if isinstance(crop_size, dict):
            h = crop_size.get("height", 224)
            w = crop_size.get("width", 224)
            return (int(h), int(w))
        if isinstance(crop_size, int):
            return (crop_size, crop_size)

    return (224, 224)


def _detect_modalities(processor: Any) -> list[str]:
    """Detect what modalities a processor supports."""
    modalities: list[str] = []

    img_proc = _get_image_processor(processor)
    if img_proc is not None:
        modalities.append("image")

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        modalities.append("text")

    # Feature extractor fallback (older HF API)
    if not modalities and hasattr(processor, "feature_extractor"):
        modalities.append("image")

    return modalities or ["image"]
