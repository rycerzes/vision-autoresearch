"""Augmentation families for HF Trainer pipelines.

Three families:
1. **Spatial-aligned** — for detection/segmentation/keypoint: transforms BOTH
   the image and its spatial annotations (boxes, masks, keypoints) together.
2. **Image-only** — for classification/contrastive: transforms only the image.
   Labels are scalar and don't need spatial adjustment.
3. **Pair-aligned** — for image-to-image/depth: transforms both input and
   target images identically (same crop, flip, etc.).

Ultralytics does NOT use this — its training pipeline has built-in augmentation
(mosaic, mixup, copy-paste, hsv, etc.) controlled via training args.

Uses albumentations when available (it supports bbox/mask/keypoint transforms).
Falls back to torchvision transforms otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class AugmentationFamily(str, Enum):
    SPATIAL_ALIGNED = "spatial_aligned"
    IMAGE_ONLY = "image_only"
    PAIR_ALIGNED = "pair_aligned"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""

    family: AugmentationFamily
    image_size: tuple[int, int] = (224, 224)
    # Strength controls (0.0 = disabled, 1.0 = full)
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.0
    color_jitter: bool = True
    random_crop: bool = False
    # Scale range for random resize crop
    scale_range: tuple[float, float] = (0.08, 1.0)
    # Use albumentations (True) or torchvision (False)
    use_albumentations: bool = True
    # TrivialAugment (classification only)
    use_trivial_augment: bool = False
    # Bbox format for detection (albumentations format string)
    bbox_format: str = "coco"  # [x_min, y_min, width, height]


def infer_augmentation_family(head_category: str) -> AugmentationFamily:
    """Pick the right augmentation family from the head category."""
    if head_category in ("detection", "structured_detection"):
        return AugmentationFamily.SPATIAL_ALIGNED
    if head_category in ("dense_classification", "prompted_segmentation"):
        return AugmentationFamily.SPATIAL_ALIGNED
    if head_category in ("image_reconstruction", "dense_regression"):
        return AugmentationFamily.PAIR_ALIGNED
    if head_category == "pair_matching":
        return AugmentationFamily.PAIR_ALIGNED
    # classification, contrastive, self_supervised, sequence_generation
    return AugmentationFamily.IMAGE_ONLY


def build_augmentation(
    config: AugmentationConfig,
) -> Callable[..., dict[str, Any]]:
    """Build an augmentation transform function.

    Returns a callable that takes a dict of inputs and returns a dict
    with augmented values. The exact signature depends on the family.
    """
    if config.use_albumentations:
        return _build_albumentations(config)
    return _build_torchvision(config)


def build_train_augmentation(
    head_category: str,
    image_size: tuple[int, int],
    *,
    use_albumentations: bool = True,
    use_trivial_augment: bool = False,
    bbox_format: str = "coco",
) -> Callable[..., dict[str, Any]]:
    """Convenience: build training augmentation from head category.

    This is the main entry point. It infers the family and builds
    the appropriate transform.
    """
    family = infer_augmentation_family(head_category)
    config = AugmentationConfig(
        family=family,
        image_size=image_size,
        use_albumentations=use_albumentations,
        use_trivial_augment=use_trivial_augment,
        bbox_format=bbox_format,
    )
    return build_augmentation(config)


def build_eval_augmentation(
    head_category: str,
    image_size: tuple[int, int],
) -> Callable[..., dict[str, Any]]:
    """Build eval-time transform (resize only, no augmentation)."""
    family = infer_augmentation_family(head_category)
    config = AugmentationConfig(
        family=family,
        image_size=image_size,
        horizontal_flip_p=0.0,
        vertical_flip_p=0.0,
        color_jitter=False,
        random_crop=False,
        use_albumentations=True,
        use_trivial_augment=False,
    )
    return build_augmentation(config)


# ── Albumentations builders ────────────────────────────────────


def _build_albumentations(
    config: AugmentationConfig,
) -> Callable[..., dict[str, Any]]:
    """Build augmentation using albumentations."""
    import albumentations as A

    h, w = config.image_size

    if config.family == AugmentationFamily.SPATIAL_ALIGNED:
        return _build_albu_spatial(config, h, w)
    elif config.family == AugmentationFamily.PAIR_ALIGNED:
        return _build_albu_pair(config, h, w)
    else:
        return _build_albu_image_only(config, h, w)


def _build_albu_spatial(
    config: AugmentationConfig, h: int, w: int,
) -> Callable[..., dict[str, Any]]:
    """Spatial-aligned augmentation for detection / segmentation / keypoint.

    Transforms image, bboxes, masks, and keypoints together.
    """
    import albumentations as A

    transforms: list[Any] = []

    # Resize
    transforms.append(A.Resize(height=h, width=w))

    # Spatial augmentations
    if config.horizontal_flip_p > 0:
        transforms.append(A.HorizontalFlip(p=config.horizontal_flip_p))
    if config.vertical_flip_p > 0:
        transforms.append(A.VerticalFlip(p=config.vertical_flip_p))

    # Color augmentations (safe for spatial targets — only affect pixels)
    if config.color_jitter:
        transforms.append(
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=1.0,
                    ),
                    A.RandomBrightnessContrast(p=1.0),
                ],
                p=0.5,
            )
        )
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=0.1))

    bbox_params = A.BboxParams(
        format=config.bbox_format,
        label_fields=["bbox_labels"],
        min_visibility=0.3,
    )

    pipeline = A.Compose(transforms, bbox_params=bbox_params)

    def transform(
        image: np.ndarray,
        bboxes: list[list[float]] | None = None,
        bbox_labels: list[int] | None = None,
        mask: np.ndarray | None = None,
        keypoints: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"image": image}
        if bboxes is not None:
            kwargs["bboxes"] = bboxes
            kwargs["bbox_labels"] = bbox_labels or [0] * len(bboxes)
        else:
            kwargs["bboxes"] = []
            kwargs["bbox_labels"] = []
        if mask is not None:
            kwargs["mask"] = mask
        result = pipeline(**kwargs)
        return result

    return transform


def _build_albu_image_only(
    config: AugmentationConfig, h: int, w: int,
) -> Callable[..., dict[str, Any]]:
    """Image-only augmentation for classification / contrastive."""
    import albumentations as A

    transforms: list[Any] = []

    if config.use_trivial_augment:
        # TrivialAugment-like: random single operation
        transforms.append(
            A.OneOf(
                [
                    A.RandomResizedCrop(
                        size=(h, w),
                        scale=config.scale_range,
                        p=1.0,
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.2,
                        rotate_limit=30,
                        p=1.0,
                    ),
                    A.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.2,
                        p=1.0,
                    ),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.Posterize(num_bits=4, p=1.0),
                    A.Equalize(p=1.0),
                    A.Solarize(p=1.0),
                ],
                p=0.8,
            )
        )
        transforms.append(A.Resize(height=h, width=w))
    elif config.random_crop:
        transforms.append(
            A.RandomResizedCrop(
                size=(h, w),
                scale=config.scale_range,
                p=1.0,
            )
        )
    else:
        transforms.append(A.Resize(height=h, width=w))

    if config.horizontal_flip_p > 0:
        transforms.append(A.HorizontalFlip(p=config.horizontal_flip_p))

    if config.color_jitter and not config.use_trivial_augment:
        transforms.append(
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            )
        )

    pipeline = A.Compose(transforms)

    def transform(
        image: np.ndarray,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return pipeline(image=image)

    return transform


def _build_albu_pair(
    config: AugmentationConfig, h: int, w: int,
) -> Callable[..., dict[str, Any]]:
    """Pair-aligned augmentation for image-to-image / depth estimation.

    Applies the same spatial transform to both input and target images.
    """
    import albumentations as A

    transforms: list[Any] = [A.Resize(height=h, width=w)]

    if config.horizontal_flip_p > 0:
        transforms.append(A.HorizontalFlip(p=config.horizontal_flip_p))
    if config.vertical_flip_p > 0:
        transforms.append(A.VerticalFlip(p=config.vertical_flip_p))

    # Compose with additional_targets for the paired image
    pipeline = A.Compose(
        transforms,
        additional_targets={"target_image": "image"},
    )

    def transform(
        image: np.ndarray,
        target_image: np.ndarray | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"image": image}
        if target_image is not None:
            kwargs["target_image"] = target_image
        return pipeline(**kwargs)

    return transform


# ── Torchvision fallback ────────────────────────────────────────


def _build_torchvision(
    config: AugmentationConfig,
) -> Callable[..., dict[str, Any]]:
    """Fallback augmentation using torchvision (image-only, no bbox/mask support)."""
    from torchvision import transforms as T

    h, w = config.image_size
    transform_list: list[Any] = []

    if config.use_trivial_augment:
        transform_list.append(T.TrivialAugmentWide())

    if config.random_crop:
        transform_list.append(T.RandomResizedCrop((h, w), scale=config.scale_range))
    else:
        transform_list.append(T.Resize((h, w)))

    if config.horizontal_flip_p > 0:
        transform_list.append(T.RandomHorizontalFlip(p=config.horizontal_flip_p))

    if config.color_jitter and not config.use_trivial_augment:
        transform_list.append(T.ColorJitter(0.2, 0.2, 0.2, 0.1))

    pipeline = T.Compose(transform_list)

    def transform(
        image: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        result_image = pipeline(image)
        return {"image": np.array(result_image)}

    return transform
