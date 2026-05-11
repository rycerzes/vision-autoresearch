"""Ultralytics backend — wraps YOLO, YOLOE, YOLO-World, RT-DETR, SAM, etc."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from engine.introspection import head_category_from_ultralytics_task
from engine.surgery_utils import (
    find_module_by_role,
    freeze_except,
    get_module_graph,
    replace_module,
)
from engine.unified_model import ModuleInfo

logger = logging.getLogger(__name__)

# Models that have NO training support in Ultralytics.
_NO_TRAIN_MODELS = frozenset({"NAS", "SAM", "SAM2", "SAM3", "FastSAM", "MobileSAM"})
_INFERENCE_ONLY_CLASSES: set[str] = set()  # populated at import time below


def _model_class_name(obj: Any) -> str:
    return type(obj).__name__


class UltralyticsModel:
    """Unified wrapper for any Ultralytics vision model."""

    def __init__(
        self,
        model_name: str,
        *,
        head_category_override: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._raw = self._load_ultralytics(model_name)
        self._cls_name = _model_class_name(self._raw)

        if head_category_override:
            self._head_category = head_category_override
        else:
            self._head_category = self._infer_head_category()

        self._can_train = self._cls_name not in _NO_TRAIN_MODELS
        # YOLOv4 and YOLOv7 load via YOLO() but have no training support
        stem = Path(model_name).stem.lower() if not model_name.startswith("/") else model_name.lower()
        if "yolov4" in stem or "yolov7" in stem:
            self._can_train = False

        logger.info(
            "UltralyticsModel loaded: %s  class=%s  head_category=%s  trainable=%s",
            model_name,
            self._cls_name,
            self._head_category,
            self._can_train,
        )


    @staticmethod
    def _load_ultralytics(model_name: str) -> Any:
        """Instantiate the right Ultralytics model class."""
        from ultralytics import YOLO

        name_lower = model_name.lower()

        if "yoloe" in name_lower:
            from ultralytics import YOLOE
            return YOLOE(model_name)

        if "world" in name_lower:
            from ultralytics.models.yolo.model import YOLOWorld
            return YOLOWorld(model_name)

        if "rtdetr" in name_lower or "rt-detr" in name_lower:
            from ultralytics import RTDETR
            return RTDETR(model_name)

        if "nas" in name_lower:
            from ultralytics import NAS
            return NAS(model_name)

        if "sam3" in name_lower:
            try:
                from ultralytics import SAM3
                return SAM3(model_name)
            except ImportError:
                from ultralytics import SAM
                return SAM(model_name)

        if "sam2" in name_lower:
            try:
                from ultralytics import SAM2
                return SAM2(model_name)
            except ImportError:
                from ultralytics import SAM
                return SAM(model_name)

        if "fastsam" in name_lower:
            from ultralytics import FastSAM
            return FastSAM(model_name)

        if "mobilesam" in name_lower or ("sam" in name_lower and "sam2" not in name_lower and "sam3" not in name_lower):
            from ultralytics import SAM
            return SAM(model_name)

        return YOLO(model_name)

    def _infer_head_category(self) -> str:
        """Derive head category from model type and task."""
        cls_name = self._cls_name

        # Predict-only models
        if cls_name in ("SAM", "SAM2", "SAM3"):
            return "prompted_segmentation"
        if cls_name == "NAS":
            return "detection"
        if cls_name == "FastSAM":
            return "dense_classification"

        # Standard YOLO family: use .task property
        task = getattr(self._raw, "task", None)
        if task is not None:
            cat = head_category_from_ultralytics_task(task)
            if cat is not None:
                return cat

        # Fallback
        return "detection"


    @property
    def backend(self) -> str:
        return "ultralytics"

    @property
    def head_category(self) -> str:
        return self._head_category

    @property
    def nn_module(self) -> nn.Module:
        """The raw PyTorch ``nn.Module`` inside the Ultralytics wrapper."""
        return self._raw.model

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self._raw.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self._raw.model.parameters() if p.requires_grad)

    @property
    def can_train(self) -> bool:
        return self._can_train


    def train(self, train_dataset: Any, eval_dataset: Any, args: dict[str, Any]) -> dict[str, Any]:
        if not self._can_train:
            raise RuntimeError(
                f"{self._cls_name} ({self._model_name}) does not support training. "
                "Use research mode with modification.py for custom training."
            )
        raise NotImplementedError("UltralyticsModel.train() not yet implemented")

    def evaluate(self, dataset: Any) -> dict[str, Any]:
        raise NotImplementedError("UltralyticsModel.evaluate() not yet implemented")

    def predict(self, image: Any) -> Any:
        return self._raw.predict(image)


    @torch.no_grad()
    def benchmark_latency(
        self,
        sample_images: list[Any],
        *,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> dict[str, float]:
        images = sample_images[:num_runs] if len(sample_images) >= num_runs else sample_images
        n = len(images)

        # Warmup
        for img in images[: min(num_warmup, n)]:
            self._raw.predict(img, verbose=False)

        device = next(self._raw.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for img in images:
            self._raw.predict(img, verbose=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        peak_vram = (
            torch.cuda.max_memory_allocated(device) / 1e6
            if device.type == "cuda"
            else 0.0
        )

        return {
            "inference_ms": (elapsed / n) * 1000,
            "throughput_img_per_sec": n / elapsed,
            "peak_vram_mb": peak_vram,
        }


    def export(self, fmt: str, output_path: Path) -> Path:
        result = self._raw.export(format=fmt)
        return Path(result) if result else output_path


    def get_module_graph(self) -> dict[str, ModuleInfo]:
        return get_module_graph(self._raw.model)

    def find_module_by_role(self, role: str) -> tuple[str, nn.Module]:
        return find_module_by_role(self._raw.model, role)

    def replace_module(self, path: str, new_module: nn.Module) -> None:
        replace_module(self._raw.model, path, new_module)

    def freeze_except(self, module_paths: list[str]) -> None:
        freeze_except(self._raw.model, module_paths)

    def get_class_names(self) -> list[str] | None:
        names = getattr(self._raw, "names", None)
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        if isinstance(names, (list, tuple)):
            return list(names)
        return None


    def set_classes(self, class_names: list[str]) -> None:
        """Call ``set_classes`` on open-vocab models (YOLOE, YOLO-World)."""
        from ultralytics.models.yolo.model import YOLOWorld

        try:
            from ultralytics import YOLOE
            is_open_vocab = isinstance(self._raw, (YOLOWorld, YOLOE))
        except ImportError:
            is_open_vocab = isinstance(self._raw, YOLOWorld)

        if not is_open_vocab:
            logger.debug("set_classes skipped: %s is not an open-vocab model", self._cls_name)
            return

        if len(class_names) > 80:
            logger.warning(
                "YOLOE has an 80-class limit per text prompt set. "
                "Dataset has %d classes — batched prompt handling may be needed.",
                len(class_names),
            )

        self._raw.set_classes(class_names)
        logger.info("set_classes(%d names) on %s", len(class_names), self._cls_name)
