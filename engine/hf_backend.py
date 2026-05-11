"""HF Transformers backend — wraps any ``AutoModelFor*`` vision model."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from engine.introspection import HEAD_CATEGORIES, head_category_from_arch
from engine.surgery_utils import (
    find_module_by_role,
    freeze_except,
    get_module_graph,
    replace_module,
)
from engine.unified_model import ModuleInfo

logger = logging.getLogger(__name__)


class HFModel:
    """Unified wrapper for any HuggingFace Transformers vision model."""

    def __init__(
        self,
        model_name: str,
        *,
        head_category_override: str | None = None,
    ) -> None:
        from transformers import AutoConfig, AutoProcessor

        self._model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # resolve head category
        if head_category_override:
            self._head_category = head_category_override
        else:
            archs = getattr(self.config, "architectures", None) or []
            if not archs:
                raise ValueError(
                    f"Cannot auto-infer head category: model {model_name!r} has no "
                    "`config.architectures` field.  Set `head_category` in YAML to override."
                )
            cat = head_category_from_arch(archs[0])
            if cat is None:
                raise ValueError(
                    f"Architecture {archs[0]!r} has no known head suffix in HEAD_CATEGORIES. "
                    f"Known suffixes: {sorted(HEAD_CATEGORIES)}.  "
                    "Set `head_category` in YAML to override."
                )
            self._head_category = cat

        # load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception:
            from transformers import AutoImageProcessor

            self.processor = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )

        # load model
        self.model = self._load_model(model_name)
        logger.info(
            "HFModel loaded: %s  head_category=%s  params=%s",
            model_name,
            self._head_category,
            f"{self.num_parameters:,}",
        )


    def _load_model(self, model_name: str) -> nn.Module:
        """Load the model via the appropriate ``AutoModelFor*`` class."""
        from transformers import (
            AutoModelForDepthEstimation,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForImageToImage,
            AutoModelForInstanceSegmentation,
            AutoModelForKeypointDetection,
            AutoModelForMaskedImageModeling,
            AutoModelForObjectDetection,
            AutoModelForSemanticSegmentation,
            AutoModelForUniversalSegmentation,
            AutoModelForVideoClassification,
            AutoModelForZeroShotImageClassification,
            AutoModelForZeroShotObjectDetection,
        )

        # Try loading via architecture string first
        archs = getattr(self.config, "architectures", None) or []
        arch_name = archs[0] if archs else ""

        # Map head category → AutoModel class
        _CATEGORY_TO_AUTO: dict[str, type] = {
            "detection": AutoModelForObjectDetection,
            "classification": AutoModelForImageClassification,
            "dense_classification": AutoModelForSemanticSegmentation,
            "dense_regression": AutoModelForDepthEstimation,
            "self_supervised": AutoModelForMaskedImageModeling,
            "image_reconstruction": AutoModelForImageToImage,
            "contrastive": AutoModelForZeroShotImageClassification,
            "structured_detection": AutoModelForKeypointDetection,
        }

        # Refine: some categories have multiple AutoModel classes depending on arch
        if "ForZeroShotObjectDetection" in arch_name:
            auto_cls = AutoModelForZeroShotObjectDetection
        elif "ForInstanceSegmentation" in arch_name:
            auto_cls = AutoModelForInstanceSegmentation
        elif "ForUniversalSegmentation" in arch_name:
            auto_cls = AutoModelForUniversalSegmentation
        elif "ForImageSegmentation" in arch_name:
            auto_cls = AutoModelForImageSegmentation
        elif "ForVideoClassification" in arch_name:
            auto_cls = AutoModelForVideoClassification
        elif "ForTextRecognition" in arch_name:
            # AutoModelForTextRecognition may not exist in all transformers versions
            try:
                from transformers import AutoModelForTextRecognition
                auto_cls = AutoModelForTextRecognition
            except ImportError:
                from transformers import AutoModel
                auto_cls = AutoModel
        elif "ForTableRecognition" in arch_name:
            try:
                from transformers import AutoModelForTableRecognition
                auto_cls = AutoModelForTableRecognition
            except ImportError:
                from transformers import AutoModel
                auto_cls = AutoModel
        elif "ForKeypointMatching" in arch_name:
            try:
                from transformers import AutoModelForKeypointMatching
                auto_cls = AutoModelForKeypointMatching
            except ImportError:
                from transformers import AutoModel
                auto_cls = AutoModel
        elif "Sam2" in arch_name:
            from transformers import Sam2Model
            return Sam2Model.from_pretrained(model_name, trust_remote_code=True)
        elif "Sam" in arch_name and "Sam2" not in arch_name:
            from transformers import SamModel
            return SamModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            auto_cls = _CATEGORY_TO_AUTO.get(self._head_category)
            if auto_cls is None:
                from transformers import AutoModel
                auto_cls = AutoModel

        return auto_cls.from_pretrained(
            model_name,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )


    @property
    def backend(self) -> str:
        return "hf"

    @property
    def head_category(self) -> str:
        return self._head_category

    @property
    def nn_module(self) -> nn.Module:
        return self.model

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def train(self, train_dataset: Any, eval_dataset: Any, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("HFModel.train() not yet implemented")

    def evaluate(self, dataset: Any) -> dict[str, Any]:
        raise NotImplementedError("HFModel.evaluate() not yet implemented")

    def predict(self, image: Any) -> Any:
        raise NotImplementedError("HFModel.predict() not yet implemented")


    @torch.no_grad()
    def benchmark_latency(
        self,
        sample_images: list[Any],
        *,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> dict[str, float]:
        device = next(self.model.parameters()).device
        self.model.eval()

        images = sample_images[:num_runs] if len(sample_images) >= num_runs else sample_images
        n = len(images)

        # Warmup
        for img in images[: min(num_warmup, n)]:
            inputs = self.processor(images=img, return_tensors="pt").to(device)
            _ = self.model(**inputs)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for img in images:
            inputs = self.processor(images=img, return_tensors="pt").to(device)
            _ = self.model(**inputs)
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
        if fmt == "onnx":
            import torch

            output_path = output_path.with_suffix(".onnx")
            dummy = self.processor(
                images=torch.zeros(3, 224, 224), return_tensors="pt"
            )
            torch.onnx.export(
                self.model,
                tuple(dummy.values()),
                str(output_path),
                opset_version=17,
            )
            return output_path
        raise NotImplementedError(f"HF export format {fmt!r} not yet implemented")


    def get_module_graph(self) -> dict[str, ModuleInfo]:
        return get_module_graph(self.model)

    def find_module_by_role(self, role: str) -> tuple[str, nn.Module]:
        return find_module_by_role(
            self.model,
            role,
            model_config=self.config,
        )

    def replace_module(self, path: str, new_module: nn.Module) -> None:
        replace_module(self.model, path, new_module)

    def freeze_except(self, module_paths: list[str]) -> None:
        freeze_except(self.model, module_paths)

    def get_class_names(self) -> list[str] | None:
        id2label = getattr(self.config, "id2label", None)
        if id2label:
            return [id2label[i] for i in sorted(id2label)]
        return None
