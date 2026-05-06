"""Load Hugging Face vision weights for ``train_hf_vision`` (task + loader dispatch)."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoBackbone,
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from vision_lab.hf_vision.constants import HF_VISION_SUPPORTED_TASKS, MODEL_LOADER_CHOICES

logger = logging.getLogger(__name__)


class PooledClassifier(nn.Module):
    """``AutoModel`` + mean-pool CLS token (or pooler) + linear head for image classification."""

    def __init__(self, base: nn.Module, num_labels: int) -> None:
        super().__init__()
        self.base = base
        self.config = getattr(base, "config", None)
        cfg = getattr(base, "config", None)
        hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
        if hidden is None:
            raise ValueError("AutoModel base has no config.hidden_size; cannot attach classifier head.")
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        out = self.base(pixel_values=pixel_values, **kwargs)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled: torch.Tensor = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            pooled = out.last_hidden_state[:, 0]
        else:
            raise ValueError("AutoModel output has neither pooler_output nor last_hidden_state.")
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(
            loss=cast(Any, loss),
            logits=logits,
        )


class BackboneClassifier(nn.Module):
    """``AutoBackbone`` + global average pool on last feature map + linear head."""

    def __init__(self, backbone: nn.Module, num_labels: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = getattr(backbone, "config", None)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            bo = self.backbone(pixel_values=dummy)
            fmaps = bo["feature_maps"]
            last = fmaps[-1]
            in_ch = last.shape[1]
        self.classifier = nn.Linear(in_ch, num_labels)

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        bo = self.backbone(pixel_values=pixel_values, **kwargs)
        fmaps = bo["feature_maps"]
        x = fmaps[-1]
        pooled = x.mean(dim=(2, 3))
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(
            loss=cast(Any, loss),
            logits=logits,
        )


def _common_pretrained_kwargs(
    *,
    cache_dir: str | None,
    revision: str,
    token: str | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    return {
        "cache_dir": cache_dir,
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
    }


def load_hf_vision_model(
    *,
    task_type: str,
    model_loader: str,
    model_name_or_path: str,
    config_name: str | None,
    num_labels: int,
    label2id: dict[str, int],
    id2label: dict[int, str],
    cache_dir: str | None,
    model_revision: str,
    token: str | None,
    trust_remote_code: bool,
    ignore_mismatched_sizes: bool,
    image_processor_name: str | None,
) -> tuple[nn.Module, Any]:
    """
    Return ``(model, image_processor)`` for ``task_type`` using the requested loader.

    ``auto_task_head`` maps to the task-specific ``AutoModelFor*`` head. Probe-style
    ``auto_model`` / ``auto_backbone`` loaders are classification-only until dense
    task probe heads have a standard evaluator contract.
    """
    if task_type not in HF_VISION_SUPPORTED_TASKS:
        raise ValueError(
            f"train_hf_vision does not support task_type={task_type!r} yet "
            f"(supported: {sorted(HF_VISION_SUPPORTED_TASKS)})."
        )
    ml = model_loader.strip()
    if ml not in MODEL_LOADER_CHOICES:
        raise ValueError(f"Unknown model_loader {model_loader!r}; expected one of {sorted(MODEL_LOADER_CHOICES)}.")

    common = _common_pretrained_kwargs(
        cache_dir=cache_dir,
        revision=model_revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    cfg_id = config_name or model_name_or_path
    proc_src = image_processor_name or model_name_or_path
    image_processor = AutoImageProcessor.from_pretrained(proc_src, **common)

    if task_type == "classify":
        if ml == "auto_task_head":
            config = AutoConfig.from_pretrained(
                cfg_id,
                num_labels=num_labels,
                label2id=label2id,
                id2label=id2label,
                **common,
            )
            model = AutoModelForImageClassification.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                **common,
            )
            return model, image_processor

        if ml == "auto_model":
            config = AutoConfig.from_pretrained(cfg_id, **common)
            base = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                **common,
            )
            model = PooledClassifier(base, num_labels)
            logger.info("Loaded AutoModel + pooled linear head (model_loader=auto_model).")
            return model, image_processor

        if ml == "auto_backbone":
            try:
                backbone = AutoBackbone.from_pretrained(model_name_or_path, **common)
            except ValueError as exc:
                raise ValueError(
                    "model_loader=auto_backbone requires a backbone checkpoint supported by "
                    "transformers.AutoBackbone (e.g. facebook/dinov2-small). ViT classification "
                    "checkpoints such as google/vit-base-patch16-224 are not valid here — use "
                    "auto_task_head instead."
                ) from exc
            model = BackboneClassifier(backbone, num_labels)
            logger.info("Loaded AutoBackbone + linear probe head (model_loader=auto_backbone).")
            return model, image_processor

    if task_type == "detect":
        if ml != "auto_task_head":
            raise ValueError(f"detect supports model_loader=auto_task_head only, not {ml!r}")
        config = AutoConfig.from_pretrained(
            cfg_id,
            label2id=label2id,
            id2label=id2label,
            **common,
        )
        model = AutoModelForObjectDetection.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **common,
        )
        return model, image_processor

    if task_type == "segment":
        if ml != "auto_task_head":
            raise ValueError(f"segment supports model_loader=auto_task_head only, not {ml!r}")
        model_id = model_name_or_path.lower()
        if "sam2" in model_id:
            from transformers import Sam2Model, Sam2Processor

            processor = Sam2Processor.from_pretrained(proc_src, **common)
            model = Sam2Model.from_pretrained(model_name_or_path, **common)
        else:
            from transformers import SamModel, SamProcessor

            processor = SamProcessor.from_pretrained(proc_src, **common)
            model = SamModel.from_pretrained(model_name_or_path, **common)
        return model, processor

    raise RuntimeError(f"Unhandled combination task_type={task_type!r}, model_loader={ml!r}")
