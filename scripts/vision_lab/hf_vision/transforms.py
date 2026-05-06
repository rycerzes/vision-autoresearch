"""Image transforms aligned with ``AutoImageProcessor`` (shared HF vision runner)."""

from __future__ import annotations

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)


def build_transforms(image_processor, is_training: bool):
    if hasattr(image_processor, "size"):
        size = image_processor.size
        if "shortest_edge" in size:
            img_size = size["shortest_edge"]
        elif "height" in size and "width" in size:
            img_size = (size["height"], size["width"])
        else:
            img_size = 224
    else:
        img_size = 224

    if hasattr(image_processor, "image_mean") and image_processor.image_mean:
        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    else:
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_training:
        return Compose(
            [
                RandomResizedCrop(img_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
    if isinstance(img_size, int):
        resize_size = int(img_size / 0.875)
    else:
        resize_size = tuple(int(s / 0.875) for s in img_size)
    return Compose(
        [
            Resize(resize_size),
            CenterCrop(img_size),
            ToTensor(),
            normalize,
        ]
    )
