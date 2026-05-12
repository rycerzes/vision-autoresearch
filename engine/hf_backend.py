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

    def adapt_num_classes(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
    ) -> None:
        """Adapt the model's output head to match the dataset's class count.

        Reloads the model with the correct ``num_labels``, ``id2label``, and
        ``label2id`` in the config.  Uses ``ignore_mismatched_sizes=True`` to
        handle weight shape differences gracefully.
        """
        current_num_labels = getattr(self.config, "num_labels", None)
        if current_num_labels == num_classes:
            return

        logger.info(
            "Adapting model from %s classes to %s classes",
            current_num_labels,
            num_classes,
        )

        id2label = (
            {i: name for i, name in enumerate(class_names)}
            if class_names
            else {i: f"class_{i}" for i in range(num_classes)}
        )
        label2id = {v: k for k, v in id2label.items()}

        self.config.num_labels = num_classes
        self.config.id2label = id2label
        self.config.label2id = label2id

        # Reload model with updated config
        self.model = self._load_model_with_config(self._model_name, self.config)

    def _load_model_with_config(self, model_name: str, config: Any) -> nn.Module:
        """Reload model from pretrained with an explicit config."""
        from transformers import (
            AutoModelForDepthEstimation,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForInstanceSegmentation,
            AutoModelForMaskedImageModeling,
            AutoModelForObjectDetection,
            AutoModelForSemanticSegmentation,
            AutoModelForUniversalSegmentation,
            AutoModelForZeroShotObjectDetection,
        )

        archs = getattr(config, "architectures", None) or []
        arch_name = archs[0] if archs else ""

        _CATEGORY_TO_AUTO: dict[str, type] = {
            "detection": AutoModelForObjectDetection,
            "classification": AutoModelForImageClassification,
            "dense_classification": AutoModelForSemanticSegmentation,
            "dense_regression": AutoModelForDepthEstimation,
            "self_supervised": AutoModelForMaskedImageModeling,
        }

        if "ForZeroShotObjectDetection" in arch_name:
            auto_cls = AutoModelForZeroShotObjectDetection
        elif "ForInstanceSegmentation" in arch_name:
            auto_cls = AutoModelForInstanceSegmentation
        elif "ForUniversalSegmentation" in arch_name:
            auto_cls = AutoModelForUniversalSegmentation
        elif "ForImageSegmentation" in arch_name:
            auto_cls = AutoModelForImageSegmentation
        else:
            auto_cls = _CATEGORY_TO_AUTO.get(self._head_category)
            if auto_cls is None:
                from transformers import AutoModel
                auto_cls = AutoModel

        return auto_cls.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )


    def train(
        self,
        train_dataset: Any,
        eval_dataset: Any,
        args: dict[str, Any],
        *,
        pipeline_config: Any | None = None,
        compute_metrics_fn: Any | None = None,
        custom_loss_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Train using HF Trainer.

        Parameters
        ----------
        train_dataset:
            HF Dataset (already transformed with pixel_values, labels).
        eval_dataset:
            HF Dataset for evaluation.
        args:
            Dict of HF TrainingArguments kwargs.
        pipeline_config:
            PipelineConfig from auto_infer_pipeline (provides collate_fn, metrics).
        compute_metrics_fn:
            Custom compute_metrics callable.  If None, auto-derived from head_category.
        custom_loss_fn:
            Custom loss function (for external loss models).
        """
        from transformers import Trainer, TrainingArguments

        # Build TrainingArguments
        training_args = TrainingArguments(**args)

        # Resolve collate_fn
        collate_fn = None
        if pipeline_config is not None:
            collate_fn = pipeline_config.collate_fn
        if collate_fn is None:
            from engine.collation import build_collate_fn
            collate_fn = build_collate_fn(self._head_category, self.processor)

        # Resolve compute_metrics
        if compute_metrics_fn is None:
            compute_metrics_fn = self._build_compute_metrics()

        # Build custom trainer if we need external loss
        trainer_cls = Trainer
        if custom_loss_fn is not None or (
            pipeline_config is not None and pipeline_config.loss_mode == "external"
        ):
            loss_fn = custom_loss_fn
            if loss_fn is None and pipeline_config is not None:
                loss_fn = pipeline_config.external_loss_fn
            if loss_fn is not None:
                trainer_cls = self._make_custom_loss_trainer(Trainer, loss_fn)

        trainer = trainer_cls(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processor,
            data_collator=collate_fn,
            compute_metrics=compute_metrics_fn,
        )

        train_result = trainer.train()
        train_metrics = train_result.metrics
        trainer.save_model()

        # Run final eval
        eval_metrics = trainer.evaluate()

        # Store trainer for later use
        self._trainer = trainer

        # Merge and return
        all_metrics = {**train_metrics, **eval_metrics}
        return all_metrics

    def evaluate(self, dataset: Any, *, compute_metrics_fn: Any | None = None) -> dict[str, Any]:
        """Run evaluation on a dataset.

        If train() was called previously, reuses the trainer.  Otherwise
        builds a fresh Trainer for eval-only.
        """
        from transformers import Trainer, TrainingArguments

        if compute_metrics_fn is None:
            compute_metrics_fn = self._build_compute_metrics()

        if hasattr(self, "_trainer") and self._trainer is not None:
            return self._trainer.evaluate(eval_dataset=dataset)

        # Build eval-only trainer
        from engine.collation import build_collate_fn
        collate_fn = build_collate_fn(self._head_category, self.processor)

        eval_args = TrainingArguments(
            output_dir="./eval_output",
            do_train=False,
            do_eval=True,
            per_device_eval_batch_size=16,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=dataset,
            processing_class=self.processor,
            data_collator=collate_fn,
            compute_metrics=compute_metrics_fn,
        )
        return trainer.evaluate()

    def predict(self, image: Any) -> Any:
        """Single-image inference."""
        device = next(self.model.parameters()).device
        self.model.eval()
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def _build_compute_metrics(self) -> Any | None:
        """Build compute_metrics function from head_category."""
        if self._head_category == "classification":
            return self._compute_metrics_classification
        if self._head_category in ("detection", "structured_detection"):
            return self._make_detection_compute_metrics()
        if self._head_category in ("dense_classification",):
            return self._compute_metrics_segmentation
        return None

    @staticmethod
    def _compute_metrics_classification(eval_pred: Any) -> dict[str, float]:
        """Accuracy metric for classification."""
        import numpy as np
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": float(accuracy)}

    def _make_detection_compute_metrics(self) -> Any:
        """Build a detection compute_metrics closure with access to processor and config."""
        processor = self.processor
        config = self.config
        id2label = getattr(config, "id2label", None)

        def compute_metrics_detection(eval_pred: Any) -> dict[str, float]:
            """mAP metric for detection via torchmetrics MeanAveragePrecision.

            Post-processes model outputs using the image processor (NMS, box rescaling),
            then computes COCO-style mAP.
            """
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
            from transformers.image_transforms import center_to_corners_format

            predictions, targets = eval_pred.predictions, eval_pred.label_ids

            post_processed_targets: list[dict[str, Any]] = []
            post_processed_predictions: list[dict[str, Any]] = []
            image_sizes: list[Any] = []

            for batch in targets:
                batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
                image_sizes.append(batch_image_sizes)
                for image_target in batch:
                    boxes = torch.tensor(image_target["boxes"])
                    # Convert from center format to corners (xyxy)
                    boxes = center_to_corners_format(boxes)
                    orig_size = image_target["orig_size"]
                    if isinstance(orig_size, (list, tuple)):
                        h, w = orig_size
                    else:
                        h, w = orig_size.tolist() if hasattr(orig_size, "tolist") else (orig_size, orig_size)
                    boxes = boxes * torch.tensor([[w, h, w, h]])
                    labels = torch.tensor(image_target["class_labels"])
                    post_processed_targets.append({"boxes": boxes, "labels": labels})

            for batch, target_sizes in zip(predictions, image_sizes):
                batch_logits, batch_boxes = batch[1], batch[2]

                class ModelOutput:
                    def __init__(self, logits, pred_boxes):
                        self.logits = logits
                        self.pred_boxes = pred_boxes

                output = ModelOutput(
                    logits=torch.tensor(batch_logits),
                    pred_boxes=torch.tensor(batch_boxes),
                )
                post_processed_output = processor.post_process_object_detection(
                    output, threshold=0.0, target_sizes=target_sizes
                )
                post_processed_predictions.extend(post_processed_output)

            metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            metric.update(post_processed_predictions, post_processed_targets)
            metrics = metric.compute()

            # Remove per-class tensors, extract scalars
            metrics.pop("classes", None)
            metrics.pop("map_per_class", None)
            metrics.pop("mar_100_per_class", None)

            result = {k: round(v.item(), 4) for k, v in metrics.items() if hasattr(v, "item")}
            # Map to standard keys
            return {
                "mAP": result.get("map", 0.0),
                "mAP_50": result.get("map_50", 0.0),
                "mAR": result.get("mar_100", 0.0),
            }

        return compute_metrics_detection

    @staticmethod
    def _compute_metrics_segmentation(eval_pred: Any) -> dict[str, float]:
        """Per-class mean IoU for segmentation."""
        import numpy as np
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=1)

        ignore_index = 255
        num_classes = logits.shape[1]

        iou_per_class: list[float] = []
        for cls_id in range(num_classes):
            pred_mask = predictions == cls_id
            true_mask = labels == cls_id
            valid = labels != ignore_index

            pred_mask = pred_mask & valid
            true_mask = true_mask & valid

            intersection = (pred_mask & true_mask).sum()
            union = (pred_mask | true_mask).sum()

            if union == 0:
                continue
            iou_per_class.append(float(intersection) / float(union))

        miou = float(np.mean(iou_per_class)) if iou_per_class else 0.0
        return {"miou": miou}

    @staticmethod
    def _make_custom_loss_trainer(base_cls: type, loss_fn: Any) -> type:
        """Create a Trainer subclass that uses a custom loss function."""

        class CustomLossTrainer(base_cls):  # type: ignore[valid-type]
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels", None)
                outputs = model(**inputs)
                if labels is not None:
                    loss = loss_fn(outputs, labels)
                elif hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    raise ValueError(
                        "Custom loss function returned None and model has no builtin loss"
                    )
                return (loss, outputs) if return_outputs else loss

        return CustomLossTrainer


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
