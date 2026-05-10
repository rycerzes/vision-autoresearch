"""Prompt segmentation slice for the shared HF vision runner.

This is stable infrastructure - do NOT edit during experiments.
Experiments modify config YAMLs only.

Adapted from huggingface/skills huggingface-vision-trainer.
"""

# pyright: reportPrivateImportUsage=false
# PyTorch typings mark many public APIs as private re-exports.

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from monai.losses.dice import DiceCELoss
from torch.utils.data import Dataset
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from vision_lab.hf_vision.adaptation import apply_adaptation_mode
from vision_lab.hf_vision.loaders import load_hf_vision_model
from vision_lab.hf_vision.runner_session import (
    finish_trackio_session,
    setup_hf_training_environment,
)
from vision_lab.hf_vision.summary_block import print_vision_autoresearch_summary

logger = logging.getLogger(__name__)


# Dataset wrapper

class SAMSegmentationDataset(Dataset):
    """Wraps an HF dataset for SAM/SAM2.

    Each sample must contain an image, a binary mask, and a prompt (bbox or point).
    """

    def __init__(self, dataset, processor, prompt_type: str,
                 image_col: str, mask_col: str, prompt_col: str | None,
                 bbox_col: str | None, point_col: str | None):
        self.dataset = dataset
        self.processor = processor
        self.prompt_type = prompt_type
        self.image_col = image_col
        self.mask_col = mask_col
        self.prompt_col = prompt_col
        self.bbox_col = bbox_col
        self.point_col = point_col

    def __len__(self):
        return len(self.dataset)

    def _extract_prompt(self, item):
        if self.prompt_col and self.prompt_col in item:
            raw = item[self.prompt_col]
            parsed = None
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
            elif isinstance(raw, dict):
                parsed = raw
            if isinstance(parsed, dict):
                if self.prompt_type == "bbox":
                    return parsed.get("bbox") or parsed.get("box")
                return parsed.get("point") or parsed.get("points")

        if self.prompt_type == "bbox" and self.bbox_col:
            return item.get(self.bbox_col)
        if self.prompt_type == "point" and self.point_col:
            return item.get(self.point_col)
        return None

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_col]
        prompt = self._extract_prompt(item)

        if self.prompt_type == "bbox":
            if prompt is None or (isinstance(prompt, (list, tuple)) and len(prompt) == 0):
                # Fallback to full-image box for sparse/noisy prompt columns.
                prompt = [0, 0, image.size[0] - 1, image.size[1] - 1]
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        else:
            if prompt is None:
                prompt = [[image.size[0] / 2.0, image.size[1] / 2.0]]
            elif isinstance(prompt, (list, tuple)) and prompt and isinstance(prompt[0], (int, float)):
                prompt = [prompt]
            inputs = self.processor(image, input_points=[[prompt]], return_tensors="pt")

        raw_mask = item.get(self.mask_col)
        if raw_mask is None:
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        else:
            mask = np.array(raw_mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        inputs["labels"] = (mask > 0).astype(np.float32)
        inputs["original_image_size"] = torch.tensor(image.size[::-1])
        return inputs


def collate_fn(batch):
    pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
    original_sizes = torch.stack([item["original_sizes"] for item in batch])
    original_image_size = torch.stack([item["original_image_size"] for item in batch])

    has_boxes = "input_boxes" in batch[0]
    has_points = "input_points" in batch[0]

    labels = torch.cat(
        [
            F.interpolate(
                torch.as_tensor(x["labels"]).unsqueeze(0).unsqueeze(0).float(),
                size=(256, 256),
                mode="nearest",
            )
            for x in batch
        ],
        dim=0,
    ).long()

    result = {
        "pixel_values": pixel_values,
        "original_sizes": original_sizes,
        "labels": labels,
        "original_image_size": original_image_size,
        "multimask_output": False,
    }

    if has_boxes:
        result["input_boxes"] = torch.cat([item["input_boxes"] for item in batch], dim=0)
    if has_points:
        result["input_points"] = torch.cat([item["input_points"] for item in batch], dim=0)
        if "input_labels" in batch[0]:
            result["input_labels"] = torch.cat([item["input_labels"] for item in batch], dim=0)

    return result


# Loss (SAM/SAM2 don't compute loss in forward())

seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")


def compute_loss(outputs, labels, num_items_in_batch=None):
    predicted_masks = outputs.pred_masks.squeeze(1)
    return seg_loss(predicted_masks, labels.float())


# IoU computation for eval

@torch.no_grad()
def compute_iou_metrics(eval_pred):
    """Compute IoU and Dice from Trainer's EvalPrediction."""
    predictions, label_ids = eval_pred.predictions, eval_pred.label_ids

    # predictions are raw logits -> sigmoid -> threshold
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = torch.as_tensor(predictions)
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    preds = (torch.sigmoid(preds) > 0.5).float()

    labels = torch.as_tensor(label_ids).float()
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    intersection = (preds * labels).sum(dim=(-2, -1))
    union = preds.sum(dim=(-2, -1)) + labels.sum(dim=(-2, -1)) - intersection
    iou = (intersection / (union + 1e-8)).mean().item()

    dice_num = 2 * intersection
    dice_den = preds.sum(dim=(-2, -1)) + labels.sum(dim=(-2, -1))
    dice = (dice_num / (dice_den + 1e-8)).mean().item()

    return {"mIoU": round(iou, 4), "dice": round(dice, 4)}


# CLI dataclasses

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="",
        metadata={"help": "Hub dataset ID."},
    )
    dataset_config_name: str | None = field(
        default=None,
        metadata={"help": "Dataset config name."},
    )
    dataset_revision: str | None = field(
        default=None,
        metadata={"help": "Optional Hub dataset revision (commit, tag, or branch)."},
    )
    train_val_split: float | None = field(
        default=0.1,
        metadata={"help": "Fraction to split off for validation."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Truncate training set."},
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "Truncate evaluation set."},
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "Column containing PIL images."},
    )
    mask_column_name: str = field(
        default="mask",
        metadata={"help": "Column containing ground-truth binary masks."},
    )
    prompt_column_name: str | None = field(
        default="prompt",
        metadata={"help": "Column with JSON-encoded prompt."},
    )
    bbox_column_name: str | None = field(
        default=None,
        metadata={"help": "Column with bbox prompt."},
    )
    point_column_name: str | None = field(
        default=None,
        metadata={"help": "Column with point prompt."},
    )
    prompt_type: str = field(
        default="bbox",
        metadata={"help": "Prompt type: 'bbox' or 'point'."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/sam2.1-hiera-small",
        metadata={"help": "Pretrained SAM/SAM2 model."},
    )
    cache_dir: str | None = field(default=None, metadata={"help": "Cache directory."})
    model_revision: str = field(default="main", metadata={"help": "Model revision."})
    token: str | None = field(default=None, metadata={"help": "Auth token."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code."})
    model_loader: str = field(
        default="auto_task_head",
        metadata={"help": "Weight graph: auto_task_head (SAM/SAM2 from Hub) only for segment."},
    )
    adaptation_mode: str = field(
        default="linear_probe",
        metadata={"help": "Training posture (see vision_lab.hf_vision.constants.ADAPTATION_MODE_CHOICES)."},
    )


# Main

def main():
    if len(sys.argv) != 2 or not sys.argv[1].endswith((".yaml", ".yml", ".json")):
        raise SystemExit(
            "segment_train is an internal runner slice; invoke train_hf_vision.py <run-contract.yaml|json>."
        )
    run_from_config(Path(os.path.abspath(sys.argv[1])))


def _parse_segment_config(
    config_path: Path,
) -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    parser = HfArgumentParser(cast(Any, (ModelArguments, DataTrainingArguments, TrainingArguments)))
    if config_path.suffix.lower() in (".yaml", ".yml"):
        return parser.parse_yaml_file(yaml_file=str(config_path), allow_extra_keys=True)
    if config_path.suffix.lower() == ".json":
        return parser.parse_json_file(json_file=str(config_path))
    raise SystemExit("Config must be .yaml, .yml, or .json")


def run_from_config(config_path: Path) -> None:
    run_segment_training(*_parse_segment_config(config_path))


def run_segment_training(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
) -> None:
    start_time = time.time()

    setup_hf_training_environment(training_args, logger=logger)

    # Ensure Trainer keeps/collects segmentation labels for loss + compute_metrics.
    training_args.label_names = ["labels"]

    logger.info(
        "HF vision runner (segment vertical): model_loader=%s adaptation_mode=%s",
        model_args.model_loader.strip(),
        model_args.adaptation_mode.strip(),
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    ld_kwargs: dict[str, Any] = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if data_args.dataset_revision:
        ld_kwargs["revision"] = data_args.dataset_revision
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        **ld_kwargs,
    )

    if "train" not in dataset:
        if len(dataset.keys()) == 1:
            only_split = list(dataset.keys())[0]
            dataset[only_split] = dataset[only_split].shuffle(seed=training_args.seed)
            split = dataset[only_split].train_test_split(test_size=data_args.train_val_split or 0.1)
            dataset = {"train": split["train"], "validation": split["test"]}
        else:
            raise ValueError(f"No 'train' split found. Available: {list(dataset.keys())}")
    elif "validation" not in dataset and "test" not in dataset:
        dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
        split = dataset["train"].train_test_split(
            test_size=data_args.train_val_split or 0.1, seed=training_args.seed
        )
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    if data_args.max_train_samples is not None:
        n = min(data_args.max_train_samples, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(n))
        logger.info(f"Truncated training set to {n} samples")
    eval_key = "validation" if "validation" in dataset else "test"
    if data_args.max_eval_samples is not None and eval_key in dataset:
        n = min(data_args.max_eval_samples, len(dataset[eval_key]))
        dataset[eval_key] = dataset[eval_key].select(range(n))
        logger.info(f"Truncated eval set to {n} samples")

    ml = model_args.model_loader.strip()
    model, processor = load_hf_vision_model(
        task_type="segment",
        model_loader=ml,
        model_name_or_path=model_args.model_name_or_path,
        config_name=None,
        num_labels=1,
        label2id={"mask": 0},
        id2label={0: "mask"},
        cache_dir=model_args.cache_dir,
        model_revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=False,
        image_processor_name=None,
    )
    apply_adaptation_mode(model, model_args.adaptation_mode, architecture="segment")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # Build datasets
    prompt_col = data_args.prompt_column_name if data_args.prompt_column_name else None

    train_dataset = SAMSegmentationDataset(
        dataset=dataset["train"],
        processor=processor,
        prompt_type=str(data_args.prompt_type),
        image_col=str(data_args.image_column_name),
        mask_col=str(data_args.mask_column_name),
        prompt_col=prompt_col,
        bbox_col=data_args.bbox_column_name,
        point_col=data_args.point_column_name,
    )
    eval_dataset = None
    if eval_key in dataset:
        eval_dataset = SAMSegmentationDataset(
            dataset=dataset[eval_key],
            processor=processor,
            prompt_type=str(data_args.prompt_type),
            image_col=str(data_args.image_column_name),
            mask_col=str(data_args.mask_column_name),
            prompt_col=prompt_col,
            bbox_col=data_args.bbox_column_name,
            point_col=data_args.point_column_name,
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_loss_func=compute_loss,
        compute_metrics=compute_iou_metrics,
    )

    # Train
    train_metrics = {}
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    # Evaluate
    eval_metrics = {}
    if training_args.do_eval and eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    training_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0

    finish_trackio_session()

    print_vision_autoresearch_summary(
        "segment", eval_metrics, train_metrics, training_seconds, peak_vram_mb
    )

    # Push to Hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["image-segmentation", "vision", "sam"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


