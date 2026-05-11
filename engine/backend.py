"""Backend detection and model loading — the router."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.hf_backend import HFModel
    from engine.ultralytics_backend import UltralyticsModel

logger = logging.getLogger(__name__)


def detect_backend(model_name: str, *, mode: str = "train") -> str:
    """Determine which backend owns *model_name*.

    **HF is preferred** when first-class support exists.  Ultralytics is used
    only for models that lack HF training support (YOLO family, YOLOE,
    YOLO-World) or for predict-only convenience (SAM via Ultralytics CLI).

    Parameters
    ----------
    model_name:
        Pretrained model identifier — HF Hub id, local path, or ``.pt`` file.
    mode:
        ``"train"`` or ``"predict"``.  Affects SAM routing (HF for training,
        Ultralytics for predict-only).
    """
    name_lower = model_name.lower()

    if model_name.endswith((".pt", ".yaml", ".pth")):
        return "ultralytics"

    yolo_indicators = (
        "yolov",
        "yolo11",
        "yolo26",
        "yolo-nas",
        "yolonas",
        "yoloe",
        "yolo-world",
        "yoloworld",
    )
    if any(k in name_lower for k in yolo_indicators):
        return "ultralytics"

    if "fastsam" in name_lower or "mobilesam" in name_lower:
        return "ultralytics"

    if "sam3" in name_lower:
        return "ultralytics"

    if "sam" in name_lower:
        return "hf" if mode == "train" else "ultralytics"

    if any(k in name_lower for k in ("rtdetr", "rt-detr", "rt_detr")):
        if "/" in model_name:  # HF Hub repo id
            return "hf"
        return "ultralytics"  # local .pt

    if "/" in model_name and not Path(model_name).exists():
        return "hf"

    local_path = Path(model_name)
    if local_path.is_dir() and (local_path / "config.json").exists():
        return "hf"

    return "hf"


def load_model(
    model_name: str,
    *,
    mode: str = "train",
    head_category_override: str | None = None,
) -> "HFModel | UltralyticsModel":
    """Single entry point: detect backend → return the right wrapper.

    Parameters
    ----------
    model_name:
        Pretrained model identifier.
    mode:
        ``"train"`` or ``"predict"`` — affects SAM backend routing.
    head_category_override:
        Force a specific head category (for community models with missing
        ``config.architectures``).
    """
    backend = detect_backend(model_name, mode=mode)
    logger.info("Backend for %r: %s", model_name, backend)

    if backend == "hf":
        from engine.hf_backend import HFModel

        return HFModel(model_name, head_category_override=head_category_override)
    else:
        from engine.ultralytics_backend import UltralyticsModel

        return UltralyticsModel(model_name, head_category_override=head_category_override)
